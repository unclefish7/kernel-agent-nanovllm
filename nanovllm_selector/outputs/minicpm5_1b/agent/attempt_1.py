import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from transformers import LlamaConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context


class LlamaRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype) * self.weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(hidden_states)
        hidden_states = hidden_states + residual
        return self._norm(hidden_states), hidden_states


class LlamaRotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_position: int, base: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32) / head_dim)
        )
        self.inv_freq = inv_freq

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=positions.device, dtype=torch.float32)
        freqs = (inv_freq.view(1, -1, 1) @ positions.float().view(1, 1, -1))[0].T
        cos = freqs.cos()
        sin = freqs.sin()
        cos = torch.cat((cos, cos), dim=-1).to(query.dtype).unsqueeze(1)
        sin = torch.cat((sin, sin), dim=-1).to(query.dtype).unsqueeze(1)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        return query, key


class LlamaAttentionBackend(Attention):

    def forward_torch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
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
            start, stop = offsets[i], offsets[i + 1]
            count = stop - start
            if count == 1:
                query = q[start:stop].view(1, self.num_heads, 1, self.head_dim)
            else:
                query = q[start:stop].transpose(0, 1).unsqueeze(0)

            if ctx.block_tables is None:
                key, value = k[start:stop], v[start:stop]
            else:
                size = self.k_cache.size(1)
                blocks = ctx.block_tables[i, :(length + size - 1) // size].to(
                    device=q.device, dtype=torch.long
                )
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
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = LlamaAttentionBackend(
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
        bias = getattr(self.qkv_proj, "bias", None)
        biases = bias.split(widths) if bias is not None else (None,) * 3
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
        bias = getattr(self.gate_up_proj, "bias", None)
        biases = bias.chunk(2) if bias is not None else (None, None)
        gate = F.linear(x, gate_w, biases[0])
        up = F.linear(x, up_w, biases[1])
        return self.down_proj(F.silu(gate) * up)


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10000),
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

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
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

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.model = LlamaModel(config)
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