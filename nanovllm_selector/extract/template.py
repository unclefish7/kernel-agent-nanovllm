"""模板折叠：把 28 个结构相同的层折成 1 个模板 + instances。

折叠只在**表示层**做，语义上仍然是 28 份。折叠必须满足同构：
op 相同、输入/输出角色序列相同、state key 集合相同、去掉层号后的形状相同。
不满足就报错，而不是悄悄合并。
"""

from __future__ import annotations

import re
from dataclasses import replace

from .schema import Edge, Node

_LAYER = re.compile(r"^L(\d+)")


def layer_index(node_id: str) -> int | None:
    m = _LAYER.match(node_id)
    return int(m.group(1)) if m else None


def template_id_of(node_id: str) -> str:
    return re.sub(r"^L\d+", "L*", node_id)


def _signature(n: Node) -> tuple:
    return (
        n.op,
        tuple(t.role for t in n.inputs),
        tuple(t.role for t in n.outputs),
        tuple(sorted(s.name for s in n.state)),
        tuple(tuple(s.shape or ()) for s in n.state),
    )


def fold(nodes: list[Node], edges: list[Edge]) -> tuple[list[Node], list[Edge], dict]:
    groups: dict[str, list[Node]] = {}
    for n in nodes:
        groups.setdefault(template_id_of(n.id), []).append(n)

    out_nodes: list[Node] = []
    id_map: dict[str, str] = {}
    mismatches: list[dict] = []

    for tid, members in groups.items():
        members.sort(key=lambda n: layer_index(n.id) if layer_index(n.id) is not None else -1)
        base = members[0]
        sig = _signature(base)
        for m in members[1:]:
            if _signature(m) != sig:
                mismatches.append({"template": tid, "node": m.id, "reason": "结构不同构"})
        instances = [layer_index(m.id) for m in members]
        new = replace(base, id=tid, template_id=tid)
        new.instances = [i for i in instances if i is not None]
        out_nodes.append(new)
        for m in members:
            id_map[m.id] = tid

    seen: set[tuple[str, str, str]] = set()
    out_edges: list[Edge] = []
    parallel: dict[tuple[str, str], int] = {}
    self_loops = 0
    # 一条折叠后的边，可能既来自"层内"（第 k 层里的 a→b），也来自"层间"
    # （第 k 层的 a → 第 k+1 层的 b）。只有纯层间的那条要标成 layer，
    # 否则折叠图上会出现假环（第 0 层的特例会让整张图绕回来）。
    flags: dict[tuple[str, str, str], dict[str, bool]] = {}
    for e in edges:
        a = id_map.get(e.from_node, e.from_node)
        b = id_map.get(e.to_node, e.to_node)
        if a == b:
            self_loops += 1
        key = (a, b, e.kind)
        parallel[key] = parallel.get(key, 0) + 1
        li, lj = layer_index(e.from_node), layer_index(e.to_node)
        cross = li is not None and lj is not None and li != lj
        f = flags.setdefault(key, {"intra": False, "cross": False})
        f["cross" if cross else "intra"] = True
        dedupe_key = (a, b, e.tensor)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out_edges.append(
            replace(e, from_node=a, to_node=b, tensor=e.tensor if parallel[key] == 1 else "aggregated")
        )

    # 同一 (from,to,kind) 的平行边（28 层各一条）合并成一条
    collapsed: dict[tuple[str, str, str], Edge] = {}
    for e in out_edges:
        key = (e.from_node, e.to_node, e.kind)
        if key in collapsed:
            continue
        f = flags.get(key, {})
        kind = "layer" if (f.get("cross") and not f.get("intra")) else e.kind
        collapsed[key] = replace(e, kind=kind)
    final_edges = list(collapsed.values())
    layer_edges = sum(1 for e in final_edges if e.kind == "layer")

    report = {
        "templates": len(out_nodes),
        "instances_total": len(nodes),
        "self_loops": self_loops,
        "mismatches": mismatches,
        "collapsed_edges": len(final_edges),
        "layer_edges": layer_edges,
    }
    return out_nodes, final_edges, report
