"""Qwen3-0.6B 单卡严格数值对照候选，复用已验证的适配写法。"""
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context


class Qwen3AttentionImpl(Attention):

    def forward_torch(self, q, k, v):
        ctx = get_context()
        if ctx.is_prefill:
            offsets = ctx.cu_seqlens_q_list
            ends = ctx.cu_seqlens_k_list
            lengths = [b - a for a, b in zip(ends, ends[1:])]
        else:
            lengths = ctx.context_lens_list
            offsets = list(range(len(lengths) + 1))
        outputs = []
        for i, length in enumerate(lengths):
            start, stop = offsets[i:i + 2]
            count = stop - start
            query = q[start:stop].transpose(0, 1).unsqueeze(0)
            if count == 1:
                query = q[start:stop].view(1, self.num_heads, 1, self.head_dim)
            if ctx.block_tables is None:
                key, value = k[start:stop], v[start:stop]
            else:
                size = self.k_cache.size(1)
                blocks = ctx.block_tables[i, :(length + size - 1) // size].long()
                key = self.k_cache[blocks].flatten(0, 1)[:length]
                value = self.v_cache[blocks].flatten(0, 1)[:length]
            key = key.transpose(0, 1).unsqueeze(0).contiguous()
            value = value.transpose(0, 1).unsqueeze(0).contiguous()
            mask = None
            if 1 < count < length:
                pos = torch.arange(count, device=q.device) + length - count
                mask = (torch.arange(length, device=q.device) <= pos[:, None])[None, None]
                repeats = self.num_heads // self.num_kv_heads
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
            y = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, scale=self.scale,
                is_causal=count > 1 and count == length, enable_gqa=mask is None,
            )
            outputs.append(y[0].transpose(0, 1))
        result = torch.cat(outputs)
        return result if ctx.is_prefill else result.unsqueeze(1)


class Qwen3RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states = hidden_states + residual
            new_residual = hidden_states
        else:
            new_residual = None

        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * x.to(input_dtype)

        if residual is not None:
            return out, new_residual
        return out


class Qwen3RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu")
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.attention_scaling = 1.0

    def forward(self, positions, dtype):
        device = positions.device
        inv_freq = self.inv_freq.to(device)
        freqs = (
            inv_freq.view(1, -1, 1) @ positions.view(1, 1, -1).float()
        )[0].T
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if rope_scaling:
            raise ValueError("该对照候选仅覆盖 Qwen3-0.6B 使用的默认 RoPE")
        self.rotary_emb = Qwen3RotaryEmbedding(self.head_dim, rope_theta)
        self.attn = Qwen3AttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        widths = (self.q_size, self.kv_size, self.kv_size)
        weights = self.qkv_proj.weight.split(widths)
        bias = self.qkv_proj.bias
        biases = bias.split(widths) if bias is not None else (None,) * 3
        q, k, v = (F.linear(hidden_states, w, b) for w, b in zip(weights, biases))
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rotary_emb(positions, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"

    def forward(self, x):
        gate, up = self.gate_up_proj.weight.chunk(2, dim=0)
        return self.down_proj(F.silu(F.linear(x, gate)) * F.linear(x, up))


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        if getattr(config, 'use_sliding_window', False):
            raise ValueError('该对照候选仅覆盖全注意力')
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
