import math
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import LlamaConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context


class LlamaRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            x = hidden_states
            input_dtype = x.dtype
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(-1, keepdim=True)
            x_norm = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
            x_norm = x_norm.to(input_dtype)
            return x_norm * self.weight
        else:
            x = hidden_states + residual
            input_dtype = x.dtype
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(-1, keepdim=True)
            x_norm = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
            x_norm = x_norm.to(input_dtype)
            return x_norm * self.weight, x


class LlamaRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        base: float = 500000.0,
        max_position: int = 131072,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu") / head_dim)
        )

        rope_type = "default"
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))

        if rope_type == "llama3":
            inv_freq = self._compute_llama3_inv_freq(inv_freq, rope_scaling)
        elif rope_type == "linear":
            factor = rope_scaling["factor"]
            inv_freq = inv_freq / factor

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _compute_llama3_inv_freq(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling["low_freq_factor"]
        high_freq_factor = rope_scaling["high_freq_factor"]
        old_context_len = rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

        return torch.stack(new_freqs)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(positions.device)
        freqs = (inv_freq.view(1, -1, 1) @ positions.float().view(1, 1, -1))[0].T
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(query.dtype).unsqueeze(1)
        sin = (emb.sin() * self.attention_scaling).to(query.dtype).unsqueeze(1)

        q_embed = (query * cos) + (self._rotate_half(query) * sin)
        k_embed = (key * cos) + (self._rotate_half(key) * sin)
        return q_embed, k_embed


class LlamaPagedAttention(Attention):

    def forward_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        if ctx.is_prefill:
            offsets = ctx.cu_seqlens_q.tolist()
            ends = ctx.cu_seqlens_k.tolist()
            lengths = [b - a for a, b in zip(ends, ends[1:])]
        else:
            lengths = ctx.context_lens.tolist()
            offsets = list(range(len(lengths) + 1))

        outputs = []
        for i, length in enumerate(lengths):
            start, stop = offsets[i : i + 2]
            count = stop - start
            query = q[start:stop].transpose(0, 1).unsqueeze(0)
            if count == 1:
                query = q[start:stop].view(1, self.num_heads, 1, self.head_dim)

            if ctx.block_tables is None:
                key, value = k[start:stop], v[start:stop]
            else:
                size = self.k_cache.size(1)
                blocks = ctx.block_tables[i, : (length + size - 1) // size].long()
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
                query,
                key,
                value,
                attn_mask=mask,
                scale=self.scale,
                is_causal=count > 1 and count == length,
                enable_gqa=mask is None,
            )
            outputs.append(y[0].transpose(0, 1))

        result = torch.cat(outputs)
        return result if ctx.is_prefill else result.unsqueeze(1)


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-05,
        qkv_bias: bool = False,
        rope_theta: float = 500000.0,
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
        self.qkv_bias = qkv_bias

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

        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            base=rope_theta,
            max_position=max_position,
            rope_scaling=rope_scaling,
        )
        self.attn = LlamaPagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        widths = (self.q_size, self.kv_size, self.kv_size)
        weights = self.qkv_proj.weight.split(widths)
        if self.qkv_proj.bias is not None:
            biases = self.qkv_proj.bias.split(widths)
        else:
            biases = (None, None, None)
        q, k, v = (F.linear(hidden_states, w, b) for w, b in zip(weights, biases))
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class LlamaMLP(nn.Module):

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_w, up_w = self.gate_up_proj.weight.chunk(2, dim=0)
        if self.gate_up_proj.bias is not None:
            gate_b, up_b = self.gate_up_proj.bias.chunk(2)
        else:
            gate_b, up_b = None, None
        gate = F.linear(x, gate_w, gate_b)
        up = F.linear(x, up_w, up_b)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return x


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 500000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class LlamaModel(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
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