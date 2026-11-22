"""逐节点重放：用一份独立实现，把图里每个节点重新算一遍。

不变量（checks.py）只能证明"没漏东西"，证明不了"标签对"。这一层管的就是
语义：节点标成 `rmsnorm`，就真的按 RMSNorm 的公式复算一次，和原模型在
同一次前向里该节点的实际输出逐元素比对。

每个节点先按实测原语配方重算，使用隔离的边界输入和写前快照，比较每一步的
返回值及写入。已有高层公式再按实际输出槽对照。独立实现见 dispatch_program.py。

比较标准：fp32 下 `rtol=1e-5, atol=1e-6`，即相对误差 1e-5 量级。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .dispatch_program import call_arguments, replay_program

RTOL = 1e-5
ATOL = 1e-6


# ---- 每个 op 的独立实现 ------------------------------------------------------
# 签名统一为 (inputs, state, attrs, ctx) -> Tensor 或 list[Tensor]

def _matmul(inputs, state, attrs, ctx):
    w = state.get("weight")
    b = state.get("bias")
    return F.linear(inputs[0], w, b)


def _rmsnorm(inputs, state, attrs, ctx):
    x = inputs[0]
    w = state.get("weight")
    eps = state.get("variance_epsilon", state.get("eps", 1e-6))
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + float(eps))
    return (w * x.to(orig)) if w is not None else x.to(orig)


def _add(inputs, state, attrs, ctx):
    args, kwargs = ctx["arguments"]
    return args[0] + args[1] * kwargs.get("alpha", 1)


def _mul(inputs, state, attrs, ctx):
    args, _ = ctx["arguments"]
    return args[0] * args[1]


def _silu(inputs, state, attrs, ctx):
    return F.silu(inputs[0])


def _embedding(inputs, state, attrs, ctx):
    return F.embedding(inputs[0].long(), state["weight"])


def _arange(inputs, state, attrs, ctx):
    args, kwargs = ctx["arguments"]
    return torch.arange(*args, **kwargs)


def _rope_table(inputs, state, attrs, ctx):
    pos = inputs[0].float().reshape(1, -1)      # [1, 1, L] -> [1, L]
    dim = ctx["head_dim"]
    # 频率表以节点声明的 inv_freq 为准：llama3 型缩放、动态 NTK 等都会改它，
    # 从 rope_theta 硬推只对"无缩放"的模型成立。没有该 buffer 时才回退。
    inv_freq = state.get("inv_freq")
    if inv_freq is None:
        theta = float(ctx.get("rope_theta", 10000.0))
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    inv_freq = inv_freq.float()
    freqs = pos[..., None] * inv_freq           # [1, 1, L, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return [torch.cos(emb), torch.sin(emb)]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _attention(inputs, state, attrs, ctx):
    q, cos, sin, k = inputs[0], inputs[1], inputs[2], inputs[3]
    rest = [t for t in inputs[4:] if t.numel() > 0]
    past_k = past_v = None
    if len(rest) == 3:
        # decode：多出来的是历史 cache，按位置拼到当前 k/v 前面
        past_k, past_v, v = rest
    else:
        v = rest[-1]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    if past_k is not None:
        k = torch.cat([past_k, k], dim=-2)
        v = torch.cat([past_v, v], dim=-2)
    key_cache, value_cache = k, v
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    # 单 query 的 decode 没有下三角可言，直接看全部 k/v
    causal = q.shape[-2] > 1
    results = {"out": F.scaled_dot_product_attention(q, k, v, is_causal=causal),
               "key": key_cache, "value": value_cache}
    # 部分模型直接将投影层产出的 value 保存为状态；当前节点只交付它实际产生的端口。
    return {role: results[role] for role in ctx["output_roles"]}


IMPL = {
    "matmul": _matmul,
    "lm_head": _matmul,
    "rmsnorm": _rmsnorm,
    "add": _add,
    "mul": _mul,
    "silu": _silu,
    "embedding": _embedding,
    "arange": _arange,
    "rope_table": _rope_table,
    "attention": _attention,
}


# ---- 重放 -------------------------------------------------------------------

def _state_of(node, module):
    """把节点声明的 state 变成真实张量或标量。"""
    out = {}
    params = dict(module.named_parameters(recurse=False)) if module is not None else {}
    buffers = dict(module.named_buffers(recurse=False)) if module is not None else {}
    for s in node.state:
        name = s["name"] if isinstance(s, dict) else s.name
        kind = s["kind"] if isinstance(s, dict) else s.kind
        if kind == "parameter":
            out[name] = params.get(name)
        elif kind == "buffer":
            out[name] = buffers.get(name)
        else:
            out[name] = getattr(module, name, None) if module is not None else None
    return out


def replay_nodes(nodes, builder, config, input_ids, tensors=None) -> dict:
    """按 dispatch 顺序逐节点重算，和实际输出比对。"""
    tensors = tensors if tensors is not None else builder._tensors
    ctx = {
        "seq_len": int(input_ids.shape[-1]),
        "device": input_ids.device,
        "head_dim": getattr(config, "head_dim", None)
        or getattr(config, "hidden_size", 1) // getattr(config, "num_attention_heads", 1),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
    }

    checked = 0
    skipped = []
    failures = []
    for node in sorted(nodes, key=lambda n: n.start_seq):
        steps = builder.programs[node._program_cid]
        try:
            values, expected = replay_program(node, steps, builder, RTOL, ATOL)
            if ([r.tensor for r in node.outputs] != node.attrs.get("observed_outputs")
                    or [r.role for r in node.outputs] != node.attrs.get("observed_output_roles")):
                raise ValueError("输出槽位、顺序或角色与录制不符")
            for ref in node.outputs:
                if ref.tensor not in expected or ref.tensor not in values:
                    raise ValueError(f"输出 {ref.tensor} 无配方产生记录")
        except Exception as exc:
            node.attrs["program_replay"]["ok"] = False
            failures.append({"node": node.id, "op": node.op, "why": str(exc)})
            continue
        if node.op in {"unknown", "conv1d", "bmm", "view"}:
            checked += 1
            continue
        impl = IMPL.get(node.op)
        if impl is None:
            skipped.append(node.id)
            continue
        path = node.impl["module_path"]
        module = builder._module("" if path == "R" else path)
        boundary = {}
        for d in steps:
            for key, value in d.before.items():
                boundary.setdefault(key, value)
        inputs = []
        missing = False
        for t in node.inputs:
            meta = tensors.get(t.tensor)
            obj = meta.get("obj") if meta else None
            if obj is None:
                missing = True
                break
            inputs.append(boundary.get(id(obj), obj))
        if missing or not node.outputs:
            skipped.append(node.id)
            continue

        try:
            local = dict(ctx, output_roles=node.attrs["observed_output_roles"])
            if node.op in {"add", "mul", "arange"}:
                step = next(s for s in node.attrs["program"] if s["op"] in node.attrs["dispatch_ops"])
                local["arguments"] = call_arguments(step, values)
            got = impl(inputs, _state_of(node, module), node.attrs, local)
        except Exception as exc:  # noqa: BLE001 - 重放失败要留证据，不能中断整张图
            failures.append({"node": node.id, "op": node.op, "why": f"{type(exc).__name__}: {exc}"})
            continue
        if isinstance(got, dict):
            roles = [ref.role for ref in node.outputs]
            if len(roles) != len(got) or set(roles) != set(got):
                failures.append({"node": node.id, "op": node.op, "why": "输出角色与录制不符"})
                continue
            got = [got[role] for role in roles]
        elif not isinstance(got, (list, tuple)):
            got = [got]
        expected_ids = node.attrs.get("observed_outputs")
        if ([ref.tensor for ref in node.outputs] != expected_ids
                or len(got) != len(node.outputs)):
            failures.append({"node": node.id, "op": node.op, "why": "输出数量或顺序与录制不符"})
            continue
        for index, (ref, result) in enumerate(zip(node.outputs, got)):
            actual = expected[ref.tensor]
            if (result is None or actual.shape != result.shape or result.dtype != actual.dtype
                    or not torch.allclose(result, actual, rtol=RTOL, atol=ATOL)):
                failures.append({"node": node.id, "op": node.op, "slot": index,
                                 "tensor": ref.tensor, "why": f"out[{index}] 形状、类型或数值不符"})
                break
        else:
            checked += 1

    return {
        # 跳过也是失败：未登记的 op 或拿不到输入的节点，不能算"通过"。
        "ok": not failures and not skipped,
        "checked": checked,
        "skipped": sorted(set(skipped)),
        "failures": failures,
        "stats": {"checked": checked, "skipped": len(set(skipped)), "failed": len(failures)},
    }
