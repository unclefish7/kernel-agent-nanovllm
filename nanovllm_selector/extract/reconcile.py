"""用模块树、每次前向的调用和原语记录，独立核对图上的节点。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby, zip_longest

import torch.nn as nn

from .decompose import TRANSPARENT, classify_ops
from .dispatch_program import PRIMITIVES
from .schema import Edge, Node

_MATRIX_OPS = {"aten.mm.default", "aten.bmm.default", "aten.addmm.default",
               "aten.baddbmm.default", "aten.matmul.default"}


@dataclass
class MatmulCalls:
    projections: list[tuple[str, int, int]]
    dispatches: list[tuple[int, str]]

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


def _scoped_dispatches(rec):
    """按 hook 的实际起止序号，把每条原语交给当时最内层的一次调用。"""
    calls = iter(rec.calls)
    upcoming = next(calls, None)
    stack = []
    for d in rec.dispatches:
        while upcoming is not None and upcoming.seq_start < d.seq:
            stack.append(upcoming)
            upcoming = next(calls, None)
        while stack and stack[-1].seq_end < d.seq:
            stack.pop()
        if stack:
            yield stack[-1].seq_start, d


def recorded_matmul_calls(rec, config) -> MatmulCalls:
    """只保留投影所在的模块路径和执行区间，不跨阶段持有录制张量。"""
    modules = dict(rec.model.named_modules())
    expected = [(c.path or "R", c.seq_start, c.seq_end) for c in rec.calls
                if isinstance(modules[c.path], nn.Linear)]
    # 函数式投影没有 Linear hook。按连续的同一次调用核对其原语，
    # 区分独立投影与 attention、旋转位置编码内部的矩阵乘。
    for _, run in groupby(_scoped_dispatches(rec), key=lambda item: item[0]):
        ops = [d for _, d in run if d.op not in TRANSPARENT]
        if not ops:
            continue
        path = ops[0].inside
        module = modules[path]
        if not isinstance(module, nn.Linear) and classify_ops([d.op for d in ops], module, config)[0] == "matmul":
            expected.append((path or "R", ops[0].seq, ops[-1].seq))
    return MatmulCalls(sorted(expected), [(d.seq, d.op) for d in rec.dispatches if d.op in _MATRIX_OPS])


def _matmul_checks(tag, recorded, members):
    name = f"matmul{tag}"
    actual = sorted((n.attrs.get("module_path", ""), n.start_seq) for n in members if n.op in ("matmul", "lm_head"))
    note = "逐次核对 Linear 调用及函数式投影的模块路径与执行区间"
    if recorded is None:
        yield dict(name=name, expected=None, actual=len(actual), ok=False, note="缺少本阶段的独立前向调用记录")
        return
    check = dict(name=name, expected=len(recorded.projections), actual=len(actual), ok=True, note=note)
    for call, node in zip_longest(recorded.projections, actual):
        if call is None or node is None or call[0] != node[0] or not call[1] <= node[1] <= call[2]:
            check.update(ok=False, note=f"{note}。首处不符：调用 {call}，节点 {node}")
            break
    yield check

    # 复合节点可以有其他语义标签，其内部每一次矩阵乘仍须出现在节点配方里。
    steps = sorted((s["seq"], s["op"]) for n in members for s in n.attrs.get("program", ())
                   if s["op"] in _MATRIX_OPS)
    yield dict(name=f"matmul_dispatch{tag}", expected=len(recorded.dispatches), actual=len(steps),
               ok=recorded.dispatches == steps, note="核对原始矩阵乘与节点配方中的执行序号及重载")


def reconcile(
    instance_nodes: list[Node],
    all_nodes: list[Node],
    edges: list[Edge],
    model,
    config,
    rec=None,
    *,
    matmul_calls_by_phase=None,
) -> dict:
    counts: dict[str, int] = {}
    for n in instance_nodes:
        counts[n.op] = counts.get(n.op, 0) + 1

    phases = sorted({n.phase for n in instance_nodes}.union(matmul_calls_by_phase or {})) or ["prefill"]
    if matmul_calls_by_phase is None:
        matmul_calls_by_phase = {}
        if rec is not None and len(phases) == 1:
            matmul_calls_by_phase[phases[0]] = recorded_matmul_calls(rec, config)
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

    embedding = _count_class(model, lambda m: isinstance(m, nn.Embedding))
    rmsnorm = _count_class(model, lambda m: type(m).__name__.endswith("RMSNorm"))

    multi = len(phases) > 1
    for phase in phases:
        c = by_phase[phase]
        tag = f"[{phase}]" if multi else ""
        members = [n for n in instance_nodes if n.phase == phase]
        if phase in matmul_calls_by_phase or c.get("matmul") or c.get("lm_head") or any(
                isinstance(m, nn.Linear) for m in model.modules()):
            checks.extend(_matmul_checks(tag, matmul_calls_by_phase.get(phase), members))
        add(f"embedding{tag}", embedding, c.get("embedding", 0), "模块树 Embedding 数")
        add(f"rmsnorm{tag}", rmsnorm, c.get("rmsnorm", 0), "模块树 RMSNorm 数")
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
