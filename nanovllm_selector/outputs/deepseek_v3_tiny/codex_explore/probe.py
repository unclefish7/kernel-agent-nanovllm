import math
import torch
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

seen = {}

def setup(model):
    def capture(name):
        def hook(module, args):
            seen[name] = args[0].detach().clone()
        return hook
    model.model.layers[0].input_layernorm.register_forward_pre_hook(capture("first"))
    model.model.layers[1].mlp.register_forward_pre_hook(capture("mixture"))

def rms(x, layer):
    return layer.weight * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + layer.variance_epsilon)).to(x.dtype)

def dense(x, m):
    gated = F.silu(F.linear(x, m.gate_proj.weight)) * F.linear(x, m.up_proj.weight)
    return F.linear(gated, m.down_proj.weight)

def rotated(x, c, s, interleave):
    if interleave:
        x = x.unflatten(-1, (-1, 2)).transpose(-1, -2).reshape_as(x)
    a, b = x.chunk(2, dim=-1)
    return c * x + s * torch.cat((-b, a), dim=-1)

def probe(model, step, inputs, output):
    config = model.config
    block = model.model.layers[0]
    normed = rms(seen["first"], block.input_layernorm)
    a = block.self_attn
    batch, length = normed.shape[:2]
    q = F.linear(rms(F.linear(normed, a.q_a_proj.weight), a.q_a_layernorm), a.q_b_proj.weight)
    q = q.view(batch, length, a.num_heads, a.qk_head_dim).transpose(1, 2)
    qplain, qr = q.split((a.qk_nope_head_dim, a.qk_rope_head_dim), -1)
    kv, kr = F.linear(normed, a.kv_a_proj_with_mqa.weight).split((a.kv_lora_rank, a.qk_rope_head_dim), -1)
    kv = F.linear(rms(kv, a.kv_a_layernorm), a.kv_b_proj.weight)
    kp, value = kv.view(batch, length, a.num_heads, -1).transpose(1, 2).split((a.qk_nope_head_dim, a.v_head_dim), -1)
    rope = config.rope_scaling
    freq, amplitude = ROPE_INIT_FUNCTIONS[rope.get("rope_type", rope.get("type"))](config, torch.device("cpu"))
    prev = 0 if step == 0 else seen["keys"].size(2)
    positions = torch.arange(prev, prev + length, device=q.device).float()
    phase = (freq.to(q.device).float().view(1, -1, 1) @ positions.view(1, 1, -1)).transpose(1, 2)
    phase = torch.cat((phase, phase), dim=-1).unsqueeze(1)
    cos, sin = phase.cos() * amplitude, phase.sin() * amplitude
    query = torch.cat((qplain, rotated(qr, cos, sin, config.rope_interleave)), -1)
    key = rotated(kr.unsqueeze(1), cos, sin, config.rope_interleave).expand(-1, a.num_heads, -1, -1)
    key = torch.cat((kp, key), -1)
    if step:
        key = torch.cat((seen["keys"], key), 2)
        value = torch.cat((seen["values"], value), 2)
    seen["keys"], seen["values"] = key, value
    mult = 1 + 0.1 * rope["mscale_all_dim"] * math.log(rope["factor"])
    coeff = a.qk_head_dim ** -0.5 * mult * mult
    attended = F.scaled_dot_product_attention(query, key, value, scale=coeff, is_causal=length > 1)
    attn = F.linear(attended.transpose(1, 2).contiguous().flatten(2), a.o_proj.weight)

    moe = model.model.layers[1].mlp
    x = seen["mixture"].flatten(0, 1)
    gate = moe.gate
    scores = F.linear(x.float(), gate.weight.float()).sigmoid()
    biased = (scores + gate.e_score_correction_bias).unflatten(1, (gate.n_group, -1))
    ranked = biased.topk(2, dim=-1).values.sum(-1)
    groups = ranked.topk(gate.topk_group, sorted=False).indices
    mask = torch.zeros_like(ranked, dtype=torch.bool).scatter_(1, groups, True)
    ids = biased.masked_fill(~mask.unsqueeze(-1), 0).flatten(1).topk(gate.top_k, sorted=False).indices
    weights = scores.gather(1, ids) * gate.routed_scaling_factor
    merged = torch.zeros_like(x)
    for expert_id in ids.unique(sorted=True).tolist():
        rows, slots = torch.where(ids == expert_id)
        merged.index_add_(0, rows, dense(x[rows], moe.experts[expert_id]) * weights[rows, slots, None])
    mixture = merged.view_as(seen["mixture"]) + dense(seen["mixture"], moe.shared_experts)
    return {"rms": normed, "mla": attn, "experts": mixture, "selected": ids, "weights": weights}
