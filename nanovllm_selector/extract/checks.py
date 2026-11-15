"""图的三条机械不变量。

这里不验证"op 标签对不对"（那要靠逐节点重放），只验证**没有东西被悄悄丢掉**。
每条不变量都是一个可数的等式，不是"看起来没问题"。

I1 dispatch 全覆盖：每个 aten 算子要么属于某个节点（簇），要么在透明算子白名单里。
I2 张量有主：每个节点输入要么有生产者节点，要么是一个图输入；边必须落在具体张量上。
I3 入参有槽：每个节点输入的 index / path / arg_name 都不为空。
"""

from __future__ import annotations

from .decompose import TRANSPARENT


def _check(cid: str, name: str, expected, actual, note: str = "") -> dict:
    return {
        "name": f"{cid} {name}",
        "expected": expected,
        "actual": actual,
        "ok": expected == actual,
        "note": note,
    }


def check_invariants(builder, rec, nodes, edges) -> dict:
    """对一张已经建好的图跑三条不变量，返回 {ok, checks, stats, problems}。"""
    problems: list[dict] = []

    # ---- I1 dispatch 全覆盖 -------------------------------------------------
    assigned = set()
    for cl in builder._clusters.values():
        assigned.update(cl.op_seqs)
    total = len(rec.dispatches)
    transparent_seqs = {d.seq for d in rec.dispatches if d.op in TRANSPARENT}
    covered = assigned | transparent_seqs
    unassigned = [d for d in rec.dispatches if d.seq not in covered]
    for d in unassigned[:20]:
        problems.append({
            "invariant": "I1",
            "detail": f"aten.{d.op} (seq={d.seq}) 不属于任何节点，也不在透明白名单",
        })

    # ---- I2 张量有主 --------------------------------------------------------
    id_of_cid = getattr(builder, "id_of_cid", {})
    edge_keys = {(e.to_node, e.tensor) for e in edges}
    missing = []
    for n in nodes:
        empty = set(n.attrs.get("empty_inputs") or [])
        for tid, src in getattr(n, "_in_srcs", {}).items():
            if tid in empty:
                continue
            if src[0] == "node" and src[1] not in id_of_cid:
                missing.append({"node": n.id, "tensor": tid, "why": "生产者簇没有节点 id"})
                continue
            if (n.id, tid) not in edge_keys:
                missing.append({"node": n.id, "tensor": tid, "why": "没有对应的边"})
    for m in missing[:20]:
        problems.append({"invariant": "I2", "detail": m})

    # ---- I3 入参有槽 --------------------------------------------------------
    no_slot = []
    for n in nodes:
        for inp in n.inputs:
            if inp.index is None or inp.path is None or inp.arg_name is None:
                no_slot.append({
                    "node": n.id,
                    "tensor": inp.tensor,
                    "index": inp.index,
                    "path": inp.path,
                    "arg_name": inp.arg_name,
                })
    for m in no_slot[:20]:
        problems.append({"invariant": "I3", "detail": m})

    only_assigned = len(assigned - transparent_seqs)
    both = len(assigned & transparent_seqs)
    only_transparent = len(transparent_seqs - assigned)
    checks = [
        _check("I1", "dispatch 全覆盖", 0, len(unassigned),
               f"{total} 个 aten 算子：{only_assigned} 个进节点，"
               f"{only_transparent} 个只属透明算子，{both} 个两者都算；"
               f"三部分相加 = {only_assigned + only_transparent + both}"),
        _check("I2", "张量有主", 0, len(missing), "每个输入都有边连到它的来源"),
        _check("I3", "入参有槽", 0, len(no_slot), "每个输入都记着参数位置"),
    ]
    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "stats": {
            "dispatch_total": total,
            "dispatch_in_nodes": only_assigned,
            "dispatch_transparent_only": only_transparent,
            "dispatch_both": both,
            "dispatch_unassigned": len(unassigned),
        },
        "problems": problems,
    }
