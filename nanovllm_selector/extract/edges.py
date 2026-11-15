"""建边：把"每个节点的输入张量是谁产生的"变成图的边。

三级匹配（实测命中数）：

1. 对象身份 `id(tensor)`：254 条；
2. 同一个 storage（视图跨模块，如 `q_proj → view → transpose → q_norm`）：+57 条；
3. 字节区间包含：+0 条。

第 2 级是必需的。只用第 1 级会漏掉所有"中间隔着一个 view/transpose"的边。
"""

from __future__ import annotations

from .schema import Edge, Evidence, Node, TensorMeta, TensorRef
from .decompose import template_of

STATE_ROLES = ("kv_cache", "attention_mask")


def _layer_of(node_id: str) -> int | None:
    if not node_id.startswith("L"):
        return None
    head = node_id.split(".", 1)[0][1:]
    return int(head) if head.isdigit() else None


def _is_state_edge(producer: Node, consumer: Node) -> bool:
    """跨层的 attention → attention 边，语义上是 KV cache 的读写，不是数据流。"""
    if producer.op != "attention" or consumer.op != "attention":
        return False
    a, b = _layer_of(producer.id), _layer_of(consumer.id)
    return a is not None and b is not None and a != b


def _external_role(tid: str, consumer: Node, meta: dict) -> str:
    shape = meta.get(tid, {}).get("shape") or []
    dtype = meta.get(tid, {}).get("dtype")
    if consumer.op == "embedding":
        return "input_ids"
    if consumer.op == "attention":
        if len(shape) == 4:
            return "kv_cache"
        if dtype == "bool" or len(shape) == 2:
            return "attention_mask"
        return "attention_aux"
    return f"ext_{tid}"


def build_edges(nodes: list[Node], builder, id_prefix: str = "") -> tuple[list[Edge], list[Node], dict]:
    id_of_cid = getattr(builder, "id_of_cid", {})
    meta = builder.tensor_meta()
    node_by_id = {n.id: n for n in nodes}

    edges: list[Edge] = []
    input_nodes: dict[str, Node] = {}
    unresolved: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()

    def add(edge: Edge) -> None:
        key = (edge.from_node, edge.to_node, edge.tensor)
        if key in seen:
            return
        seen.add(key)
        edges.append(edge)

    for n in nodes:
        for tid, src in getattr(n, "_in_srcs", {}).items():
            if src[0] == "node":
                pid = id_of_cid.get(src[1])
                if pid is not None:
                    pid = id_prefix + pid
                if pid is None:
                    unresolved[f"cluster:{src[1]}"] = unresolved.get(f"cluster:{src[1]}", 0) + 1
                    continue
                if pid == n.id:
                    continue
                producer = node_by_id.get(pid)
                if producer is None:
                    unresolved[f"node:{pid}"] = unresolved.get(f"node:{pid}", 0) + 1
                    continue
                add(
                    Edge(
                        from_node=pid,
                        to_node=n.id,
                        tensor=tid,
                        role="input",
                        kind="state" if _is_state_edge(producer, n) else "data",
                    )
                )
            else:
                m = meta.get(tid, {})
                if int(m.get("numel", 0) or 0) == 0:
                    # 0 元素的张量是"可选参数没传"的占位（比如 prefill 时
                    # attention_mask 为空），把它当成图输入会凭空多出一个节点。
                    n.attrs.setdefault("empty_inputs", []).append(tid)
                    continue
                role = _external_role(tid, n, meta)
                iid = f"state.{role}" if role in STATE_ROLES else f"input.{role}"
                iid = id_prefix + iid
                if iid not in input_nodes:
                    input_nodes[iid] = _make_input_node(iid, tid, meta, role)
                add(
                    Edge(
                        from_node=iid,
                        to_node=n.id,
                        tensor=tid,
                        role="input",
                        kind="state" if role in STATE_ROLES else "data",
                    )
                )

    for iid, node in input_nodes.items():
        node_by_id[iid] = node

    all_nodes = list(nodes) + list(input_nodes.values())
    return edges, all_nodes, {"unresolved": unresolved, "inputs": len(input_nodes)}


def _make_input_node(node_id: str, tid: str, meta: dict, role: str) -> Node:
    m = meta.get(tid, {})
    shape = m.get("shape")
    dtype = m.get("dtype")
    kind = "state" if role in STATE_ROLES else "input"
    return Node(
        id=node_id,
        op="input",
        origin="input",
        op_confidence="exact",
        inputs=[],
        outputs=[TensorRef(tensor=tid, role="out")],
        attrs={"role": role, "kind": kind},
        impl={"class": None, "module_path": None, "source": None},
        semantics={"formula": f"图输入 / 状态：{role}"},
        evidence=[
            Evidence("observed", f"{role} shape={shape} dtype={dtype}，在本次执行中没有任何节点产生它"),
            Evidence("rule", "R-INPUT-1：外部张量按角色归为图输入或运行时状态"),
        ],
    )


def to_tensor_meta(builder) -> dict[str, TensorMeta]:
    out = {}
    for tid, m in builder.tensor_meta().items():
        out[tid] = TensorMeta(
            id=tid,
            shape=list(m["shape"]),
            dtype=m["dtype"],
            storage=m["storage"],
            offset=int(m["offset"]),
            numel=int(m["numel"]),
        )
    return out


__all__ = ["build_edges", "to_tensor_meta", "template_of"]
