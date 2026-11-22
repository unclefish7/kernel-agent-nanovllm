"""对账：用"模块树的已知数字"去校验"执行轨迹重建出来的图"。

这是整条流程里唯一能自动发现"漏了/连错了"的环节：
如果图里 matmul 的个数和模块树里 Linear 的个数对不上，说明分解或建边有问题。
"""

from __future__ import annotations

import torch.nn as nn

from .dispatch_program import PRIMITIVES
from .schema import Edge, Graph, Node

PLAN_EXPECT_QWEN3 = {
    "modules": 427,
    "params": 310,
    "matmul": 196,  # 28 层 × 7 个 Linear；第 197 个 matmul 是 lm_head，单列
    "rmsnorm": 113,
    "silu": 28,
    "add": 56,  # 每层两次残差加
    "embedding": 1,
    "lm_head": 1,
}


def _count_modules(model) -> int:
    return sum(1 for _ in model.modules())


def _count_params(model) -> int:
    return sum(1 for _ in model.named_parameters())


def _count_class(model, pred) -> int:
    return sum(1 for _, m in model.named_modules() if pred(m))


def reconcile(
    instance_nodes: list[Node],
    all_nodes: list[Node],
    edges: list[Edge],
    model,
    config,
    rec=None,
) -> dict:
    counts: dict[str, int] = {}
    for n in instance_nodes:
        counts[n.op] = counts.get(n.op, 0) + 1

    # 每个阶段都是一次独立前向：模块树里的 Linear/RMSNorm 只对得上其中一次。
    # 两个阶段混在一起数会正好翻倍，所以按阶段分开对账。
    phases = sorted({n.phase for n in instance_nodes}) or ["prefill"]
    by_phase: dict[str, dict[str, int]] = {p: {} for p in phases}
    for n in instance_nodes:
        c = by_phase[n.phase]
        c[n.op] = c.get(n.op, 0) + 1

    checks: list[dict] = []

    def add(name, expected, actual, note=""):
        ok = expected == actual
        checks.append({"name": name, "expected": expected, "actual": actual, "ok": ok, "note": note})

    n_modules = _count_modules(model)
    n_params = _count_params(model)
    add("modules", n_modules, n_modules, "模块树规模（自洽检查）")

    is_qwen3 = getattr(config, "model_type", "") == "qwen3" and getattr(config, "num_hidden_layers", 0) == 28
    if is_qwen3:
        add("modules(plan)", PLAN_EXPECT_QWEN3["modules"], n_modules, "计划 §8.1")
        add("params(plan)", PLAN_EXPECT_QWEN3["params"], n_params, "tied 权重被 dedupe")

    linear = _count_class(model, lambda m: isinstance(m, nn.Linear))
    embedding = _count_class(model, lambda m: isinstance(m, nn.Embedding))
    rmsnorm = _count_class(model, lambda m: type(m).__name__.endswith("RMSNorm"))

    multi = len(phases) > 1
    for phase in phases:
        c = by_phase[phase]
        tag = f"[{phase}]" if multi else ""
        add(f"matmul{tag}", linear, c.get("matmul", 0) + c.get("lm_head", 0),
            "模块树 Linear 数 vs 图上 matmul+lm_head（单次前向）")
        add(f"embedding{tag}", embedding, c.get("embedding", 0), "模块树 Embedding 数")
        add(f"rmsnorm{tag}", rmsnorm, c.get("rmsnorm", 0), "模块树 RMSNorm 数")
        members = [n for n in instance_nodes if n.phase == phase]
        recorded = next((n._recorded_activations for n in members if hasattr(n, "_recorded_activations")), {})
        for activation in ("silu", "gelu"):
            op = f"aten.{activation}.default"
            actual = sum(sum(step["op"] == op for step in n.attrs.get("program", ())) for n in members)
            add(f"{activation}{tag}", recorded.get(op, 0), actual, "原始 dispatch 次数与节点配方出现次数")

    if is_qwen3:
        pre = by_phase.get("prefill", counts)
        for op in ("matmul", "rmsnorm", "silu", "add", "embedding", "lm_head"):
            add(f"{op}(plan)", PLAN_EXPECT_QWEN3[op], pre.get(op, 0), "计划 §8.1")

    node_ids = {n.id for n in all_nodes}
    bad_edges = [e for e in edges if e.from_node not in node_ids or e.to_node not in node_ids]
    add("edges_endpoints", 0, len(bad_edges), "每条边两端都必须存在")
    no_evidence = [n.id for n in all_nodes if not n.evidence]
    add("evidence_coverage", 0, len(no_evidence), "每个节点至少 1 条证据")
    unsupported = [dict(node=n.id, seq=s["seq"], op=s["op"]) for n in instance_nodes
                   for s in n.attrs.get("program", ()) if s["op"] not in PRIMITIVES]
    add("primitive_coverage", 0, len(unsupported),
        ", ".join(sorted({s["op"] for s in unsupported})) or "实测原语均有独立重放规则")

    unknown = []
    verified_unknown = []
    for n in instance_nodes:
        if n.op == "unknown":
            status = n.attrs.get("program_replay", {})
            size = len(n.attrs.get("program", []))
            verified = size and status.get("ok") is True and status.get("total") == status.get("checked") == size
            (verified_unknown if verified else unknown).append(n.id)
    checks.append(
        {
            "name": "unknown_ops",
            "expected": 0,
            "actual": len(unknown),
            "ok": len(unknown) == 0,
            "note": ",".join(unknown[:5]) if unknown else "",
        }
    )

    stats = {
        # 人看的概览按第一个阶段（prefill）统计，别把 decode 的数量混进来。
        "op_counts": dict(sorted(by_phase[phases[0]].items(), key=lambda kv: (-kv[1], kv[0]))),
        "op_counts_by_phase": {
            p: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
            for p, c in by_phase.items()
        },
        "nodes": len(instance_nodes),
        "verified_unknown": verified_unknown,
        "unsupported_dispatches": unsupported,
        "edges": len(edges),
        "graph_inputs": [n.id for n in all_nodes if n.origin == "input"],
        "module_calls": len(rec.calls) if rec is not None else None,
        "dispatch_count": len(rec.dispatches) if rec is not None else None,
    }

    return {"ok": all(c["ok"] for c in checks), "checks": checks, "stats": stats}
