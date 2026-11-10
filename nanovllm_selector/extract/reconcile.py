"""对账：用"模块树的已知数字"去校验"执行轨迹重建出来的图"。

这是整条流程里唯一能自动发现"漏了/连错了"的环节：
如果图里 matmul 的个数和模块树里 Linear 的个数对不上，说明分解或建边有问题。
"""

from __future__ import annotations

import torch.nn as nn

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
    act = _count_class(model, lambda m: type(m).__name__ in ("SiLUActivation", "GELUActivation", "NewGELUActivation"))

    add("matmul", linear, counts.get("matmul", 0) + counts.get("lm_head", 0), "模块树 Linear 数 vs 图上 matmul+lm_head")
    add("embedding", embedding, counts.get("embedding", 0), "模块树 Embedding 数")
    add("rmsnorm", rmsnorm, counts.get("rmsnorm", 0), "模块树 RMSNorm 数")
    add("silu", act, counts.get("silu", 0) + counts.get("gelu", 0), "激活模块数")

    if is_qwen3:
        for op in ("matmul", "rmsnorm", "silu", "add", "embedding", "lm_head"):
            add(f"{op}(plan)", PLAN_EXPECT_QWEN3[op], counts.get(op, 0), "计划 §8.1")

    node_ids = {n.id for n in all_nodes}
    bad_edges = [e for e in edges if e.from_node not in node_ids or e.to_node not in node_ids]
    add("edges_endpoints", 0, len(bad_edges), "每条边两端都必须存在")
    no_evidence = [n.id for n in all_nodes if not n.evidence]
    add("evidence_coverage", 0, len(no_evidence), "每个节点至少 1 条证据")

    unknown = [n.id for n in instance_nodes if n.op == "unknown"]
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
        "op_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "nodes": len(instance_nodes),
        "edges": len(edges),
        "graph_inputs": [n.id for n in all_nodes if n.origin == "input"],
        "module_calls": len(rec.calls) if rec is not None else None,
        "dispatch_count": len(rec.dispatches) if rec is not None else None,
    }

    return {"ok": all(c["ok"] for c in checks), "checks": checks, "stats": stats}
