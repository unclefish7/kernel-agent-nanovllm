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

    # 同一层里，同一个生产者给同一个消费者喂两个不同张量（cos 与 sin、q 与 k）
    # 是"并行的多个输入"，不是重复的边。跨层要折叠、层内不能折叠，所以去重键
    # 要带上"这是该节点的第几个输入"。
    slot_of: dict[tuple[str, str], int] = {}
    for n in nodes:
        for ref in n.inputs:
            slot_of[(n.id, ref.tensor)] = ref.index if ref.index is not None else -1

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
        li, lj = layer_index(e.from_node), layer_index(e.to_node)
        cross = li is not None and lj is not None and li != lj
        f = flags.setdefault(key, {"intra": False, "cross": False})
        f["cross" if cross else "intra"] = True

    collapsed: dict[tuple, Edge] = {}
    for e in edges:
        a = id_map.get(e.from_node, e.from_node)
        b = id_map.get(e.to_node, e.to_node)
        slot = slot_of.get((e.to_node, e.tensor), -1)
        key = (a, b, e.kind, slot)
        if key in collapsed:
            continue
        f = flags.get((a, b, e.kind), {})
        kind = "layer" if (f.get("cross") and not f.get("intra")) else e.kind
        collapsed[key] = replace(e, from_node=a, to_node=b, kind=kind)
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
