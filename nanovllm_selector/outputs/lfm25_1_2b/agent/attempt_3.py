import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

from transformers import Lfm2Config

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.request_state import (
    RequestStateSpec,
    read_request_state,
    write_request_state,
)
from nanovllm.utils.context import get_context


class Lfm2RMSNorm(nn.Module):

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


class Lfm2RotaryEmbedding(nn.Module):

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


class Lfm2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.q_layernorm = Lfm2RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.rotary_emb = Lfm2RotaryEmbedding(self.head_dim, base=rope_theta)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions, hidden_states):
        widths = (self.q_size, self.kv_size, self.kv_size)
        weights = self.qkv_proj.weight.split(widths, dim=0)
        q, k, v = (F.linear(hidden_states, w, None) for w in weights)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        cos, sin = self.rotary_emb(positions, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        o = self.attn(q, k, v)
        output = self.out_proj(o.flatten(1, -1))
        return output


class Lfm2ShortConv(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        conv_dim: int,
        l_cache: int,
    ) -> None:
        super().__init__()
        self.conv_dim = conv_dim
        self.l_cache = l_cache

        self.in_proj = nn.Linear(hidden_size, 3 * conv_dim, bias=False)
        self.conv = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=l_cache,
            groups=conv_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(conv_dim, hidden_size, bias=False)

        self.request_state_specs = {
            "conv_cache": RequestStateSpec((conv_dim, l_cache)),
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_context()
        bcx = self.in_proj(hidden_states)
        conv_cache = read_request_state(self, "conv_cache")

        if ctx.is_prefill:
            cu_q = getattr(ctx, "cu_seqlens_q_list", None)
            if cu_q is None:
                cu_q = ctx.cu_seqlens_q.tolist()
            cu_k = getattr(ctx, "cu_seqlens_k_list", None)
            if cu_k is None:
                cu_k = ctx.cu_seqlens_k.tolist()

            outputs = []
            new_states = []
            num_requests = len(cu_q) - 1
            for i in range(num_requests):
                start, end = cu_q[i], cu_q[i + 1]
                seq_len = end - start
                if seq_len == 0:
                    new_states.append(conv_cache[i])
                    continue

                bcx_i = bcx[start:end].transpose(0, 1)
                b_part, c_part, x_part = bcx_i.chunk(3, dim=0)
                bx = b_part * x_part

                key_len = cu_k[i + 1] - cu_k[i]
                if key_len == seq_len:
                    new_state = F.pad(bx, (self.l_cache - seq_len, 0))
                    conv_out = F.conv1d(
                        bx.unsqueeze(0),
                        self.conv.weight,
                        bias=None,
                        stride=1,
                        padding=self.l_cache - 1,
                        groups=self.conv_dim,
                    )[0, :, :seq_len]
                else:
                    old_state = conv_cache[i]
                    total = torch.cat([old_state, bx], dim=1)
                    conv_total = F.conv1d(
                        total.unsqueeze(0),
                        self.conv.weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        groups=self.conv_dim,
                    )[0]
                    conv_out = conv_total[:, -seq_len:]
                    new_state = total[:, -self.l_cache:]

                y = c_part * conv_out
                outputs.append(y.transpose(0, 1))
                new_states.append(new_state)

            y = torch.cat(outputs, dim=0)
            write_request_state(
                self, "conv_cache", torch.stack(new_states, dim=0)
            )
        else:
            batch = hidden_states.shape[0]
            b_part, c_part, x_part = bcx.chunk(3, dim=-1)
            bx = b_part * x_part

            pos = (ctx.context_lens - 1).clamp(0, self.l_cache - 1).long()
            new_state = torch.roll(conv_cache, shifts=-1, dims=-1)
            new_state.scatter_(
                2,
                pos.view(batch, 1, 1).expand(-1, self.conv_dim, 1),
                bx.unsqueeze(-1),
            )
            conv_out = (
                new_state * self.conv.weight[:, 0, :].unsqueeze(0)
            ).sum(dim=-1)
            y = c_part * conv_out
            write_request_state(self, "conv_cache", new_state)

        return self.out_proj(y)


class Lfm2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.w2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_w, up_w = self.gate_up_proj.weight.chunk(2, dim=0)
        gate = F.linear(x, gate_w, None)
        up = F.linear(x, up_w, None)
        x = F.silu(gate) * up
        x = self.w2(x)
        return x


class Lfm2DecoderLayer(nn.Module):

    def __init__(self, config: Lfm2Config, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        hidden_size = config.hidden_size
        norm_eps = config.norm_eps

        if layer_type == "conv":
            conv_dim = getattr(config, "conv_dim", hidden_size)
            l_cache = getattr(config, "conv_L_cache", 3)
            self.conv = Lfm2ShortConv(
                hidden_size=hidden_size,
                conv_dim=conv_dim,
                l_cache=l_cache,
            )
        else:
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = hidden_size // num_heads
            self.self_attn = Lfm2Attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=norm_eps,
                rope_theta=getattr(config, "rope_theta", 1000000.0),
            )

        self.operator_norm = Lfm2RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = Lfm2RMSNorm(hidden_size, eps=norm_eps)

        intermediate_size = _get_intermediate_size(config)
        self.feed_forward = Lfm2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(
                hidden_states, residual
            )

        if self.layer_type == "conv":
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class Lfm2Model(nn.Module):

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "conv" if i % 2 == 0 else "full_attention"
                for i in range(config.num_hidden_layers)
            ]
        assert len(layer_types) == config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(config, layer_types[i])
                for i in range(config.num_hidden_layers)
            ]
        )
        self.embedding_norm = Lfm2RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states


def _get_intermediate_size(config: Lfm2Config) -> int:
    hidden_size = config.hidden_size
    intermediate = getattr(config, "intermediate_size", None)
    if intermediate is None:
        intermediate = getattr(config, "block_ff_dim", hidden_size * 4)
    multiplier = getattr(config, "block_ffn_dim_multiplier", 1.0)
    intermediate = int(intermediate * multiplier)
    if getattr(config, "block_auto_adjust_ff_dim", False):
        intermediate = int(2 * intermediate / 3)
        multiple = getattr(config, "block_multiple_of", 256)
        if multiple > 0:
            intermediate = int(multiple * round(intermediate / multiple))
    return intermediate


class Lfm2ForCausalLM(nn.Module):

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "w1": ("gate_up_proj", 0),
        "w3": ("gate_up_proj", 1),
    }

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.model = Lfm2Model(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        if getattr(config, "tie_word_embeddings", True):
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