"""单卡 DeepSeek V3。展开低秩 K/V，并把跨步数据写回引擎的分页缓存。"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.utils.context import get_context


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * (xf * inv).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        settings = config.rope_scaling or {}
        kind = settings.get("rope_type", settings.get("type", "default"))
        freq, self.amplitude = ROPE_INIT_FUNCTIONS[kind](config, torch.device("cpu"))
        self.register_buffer("inv_freq", freq, persistent=False)

    def forward(self, positions, dtype):
        freq = self.inv_freq.to(positions.device).float().view(1, -1, 1)
        phase = (freq @ positions.float().view(1, 1, -1)).transpose(1, 2)
        phase = torch.cat((phase, phase), dim=-1).unsqueeze(1)
        return (phase.cos() * self.amplitude).to(dtype), (phase.sin() * self.amplitude).to(dtype)


def rotate(x, cos, sin, interleaved):
    if interleaved:
        x = x.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(x)
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class PagedAttention(nn.Module):
    def __init__(self, heads, key_width, value_width, scale):
        super().__init__()
        self.num_kv_heads = heads
        self.head_dim = max(key_width, value_width)
        self.key_width = key_width
        self.value_width = value_width
        self.scale = scale
        self.k_cache = self.v_cache = torch.empty(0)

    def forward(self, query, key, value):
        ctx = get_context()
        if self.k_cache.numel():
            key_rows = key[0].transpose(0, 1)
            value_rows = value[0].transpose(0, 1)
            key_rows = F.pad(key_rows, (0, self.head_dim - self.key_width)).contiguous()
            value_rows = F.pad(value_rows, (0, self.head_dim - self.value_width)).contiguous()
            store_kvcache(key_rows, value_rows, self.k_cache, self.v_cache, ctx.slot_mapping)

        if ctx.is_prefill:
            offsets = ctx.cu_seqlens_q_list
            ends = ctx.cu_seqlens_k_list
            if offsets is None:
                offsets = ctx.cu_seqlens_q.tolist()
            if ends is None:
                ends = ctx.cu_seqlens_k.tolist()
            lengths = [end - start for start, end in zip(ends, ends[1:])]
        else:
            lengths = ctx.context_lens_list
            if lengths is None:
                lengths = ctx.context_lens.tolist()
            offsets = range(len(lengths) + 1)

        outputs = []
        for i, length in enumerate(lengths):
            start, end = offsets[i:i + 2]
            count = end - start
            q = query[:, :, start:end]
            if count == length:
                k = key[:, :, start:end]
                v = value[:, :, start:end]
            else:
                page_size = self.k_cache.size(1)
                page_count = (length + page_size - 1) // page_size
                pages = ctx.block_tables[i, :page_count].long()
                k_rows = self.k_cache[pages].flatten(0, 1)[:length, :, :self.key_width]
                v_rows = self.v_cache[pages].flatten(0, 1)[:length, :, :self.value_width]
                k = k_rows.transpose(0, 1).unsqueeze(0).contiguous()
                v = v_rows.transpose(0, 1).unsqueeze(0).contiguous()
            mask = None
            if 1 < count < length:
                positions = torch.arange(count, device=q.device) + length - count
                mask = (torch.arange(length, device=q.device) <= positions[:, None])[None, None]
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=(count == length and count > 1), scale=self.scale,
            )
            outputs.append(attended.transpose(1, 2).contiguous().view(count, -1))
        return torch.cat(outputs, dim=0)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.plain = config.qk_nope_head_dim
        self.rope = config.qk_rope_head_dim
        self.value = config.v_head_dim
        self.rank = config.kv_lora_rank
        self.interleaved = config.rope_interleave
        self.q_rank = config.q_lora_rank
        width = config.hidden_size
        key_width = self.plain + self.rope
        if self.q_rank is None:
            self.q_proj = nn.Linear(width, self.heads * key_width, bias=False)
        else:
            self.q_a_proj = nn.Linear(width, self.q_rank, bias=config.attention_bias)
            self.q_a_layernorm = RMSNorm(self.q_rank)
            self.q_b_proj = nn.Linear(self.q_rank, self.heads * key_width, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(width, self.rank + self.rope, bias=config.attention_bias)
        self.kv_a_layernorm = RMSNorm(self.rank)
        self.kv_b_proj = nn.Linear(self.rank, self.heads * (self.plain + self.value), bias=False)
        self.o_proj = nn.Linear(self.heads * self.value, width, bias=config.attention_bias)
        scale = key_width ** -0.5
        settings = config.rope_scaling or {}
        factor = settings.get("factor", 1)
        if factor > 1 and settings.get("mscale_all_dim"):
            mult = 1 + 0.1 * settings["mscale_all_dim"] * math.log(factor)
            scale = scale * mult * mult
        self.attention = PagedAttention(self.heads, key_width, self.value, scale)

    def forward(self, x, cos, sin):
        token_count = x.size(0)
        if self.q_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(1, token_count, self.heads, -1).transpose(1, 2)
        q_plain, q_rotary = q.split((self.plain, self.rope), dim=-1)
        latent, k_rotary = self.kv_a_proj_with_mqa(x).split((self.rank, self.rope), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(latent))
        kv = kv.view(1, token_count, self.heads, -1).transpose(1, 2)
        k_plain, value = kv.split((self.plain, self.value), dim=-1)
        q_rotary = rotate(q_rotary, cos, sin, self.interleaved)
        k_rotary = rotate(k_rotary.view(1, 1, token_count, self.rope), cos, sin, self.interleaved)
        query = torch.cat((q_plain, q_rotary), dim=-1)
        key = torch.cat((k_plain, k_rotary.expand(-1, self.heads, -1, -1)), dim=-1)
        return self.o_proj(self.attention(query, key, value))


class GatedMLP(nn.Module):
    def __init__(self, config, intermediate):
        super().__init__()
        width = config.hidden_size
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)
        self.activate = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.activate(self.gate_proj(x)) * self.up_proj(x))


class GroupedRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))
        self.groups = config.n_group
        self.group_count = config.topk_group
        self.expert_count = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob
        self.scale = config.routed_scaling_factor

    def forward(self, x):
        scores = F.linear(x.float(), self.weight.float()).sigmoid()
        biased = (scores + self.e_score_correction_bias).unflatten(1, (self.groups, -1))
        ranked = biased.topk(2, dim=-1).values.sum(-1)
        chosen = ranked.topk(self.group_count, sorted=False).indices
        allowed = torch.zeros_like(ranked, dtype=torch.bool).scatter_(1, chosen, True)
        eligible = biased.masked_fill(~allowed.unsqueeze(-1), 0).flatten(1)
        ids = eligible.topk(self.expert_count, sorted=False).indices
        weights = scores.gather(1, ids)
        if self.normalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return ids, weights * self.scale


class ExpertMixture(nn.Module):
    def __init__(self, config):
        super().__init__()
        size = config.moe_intermediate_size
        self.experts = nn.ModuleList(GatedMLP(config, size) for _ in range(config.n_routed_experts))
        self.gate = GroupedRouter(config)
        self.shared_experts = GatedMLP(config, size * config.n_shared_experts)

    def forward(self, x):
        ids, weights = self.gate(x)
        result = torch.zeros_like(x, dtype=weights.dtype)
        # 升序逐专家累加，保持同一词元多个专家输出的浮点加法顺序。
        for expert in ids.unique(sorted=True).tolist():
            rows, slots = torch.where(ids == expert)
            weighted = self.experts[expert](x[rows]) * weights[rows, slots, None]
            result.index_add_(0, rows, weighted)
        return result.to(x.dtype) + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if index < config.first_k_dense_replace:
            self.mlp = GatedMLP(config, config.intermediate_size)
        else:
            self.mlp = ExpertMixture(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList(DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class DeepseekV3ForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config):
        super().__init__()
        self.model = Decoder(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
