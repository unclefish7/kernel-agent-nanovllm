"""从原始调用中定位每个输入的生产者、输出序号和跨步缓存身份。"""

from __future__ import annotations

from .schema import Edge, Evidence, Node, StateRef, TensorMeta, TensorRef
from .decompose import template_of


def tensor_view(tensor):
    return (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype)


def build_edges(nodes, builder, observation, history, external):
    origins, consumed, slots, _ = observation
    metadata = builder.tensor_meta()
    inputs = {}
    named_inputs = {builder.node_prefix + "input." + name: name for name in external}
    edges = []
    reads = set()
    for node in nodes:
        observed_inputs = []
        connected = set()
        for index, ref in enumerate(node.inputs):
            tid = ref.tensor
            if tid not in metadata:
                raise ValueError(f"{node.id} in[{index}] 未登记张量 {tid}")
            tensor = builder._tensors[tid]["obj"]
            origin = origins.get(id(tensor))
            role, kind = "input", "data"
            if origin and origin[0] != node.id:
                producer, source_slot = origin[0], slots[origin]
                connected.add(origin)
                if producer in named_inputs and producer not in inputs:
                    inputs[producer] = _make_input_node(producer, origin[1], metadata, named_inputs[producer])
                    inputs[producer].phase = node.phase
            elif tensor_view(tensor) in history:
                key, producer, source_slot = history[tensor_view(tensor)]
                reads.add(key)
                role, kind = key[1], "state"
                ref.role = "state"
                node.state.append(StateRef(role, "runtime_tensor", "read_write",
                                           list(tensor.shape), metadata[tid]["dtype"]))
            elif tensor.numel() == 0:
                node.attrs.setdefault("empty_inputs", []).append(tid)
                observed_inputs.append(None)
                continue
            else:
                name = f"ext_{tid}"
                producer = builder.node_prefix + "input." + name
                if producer not in inputs:
                    inputs[producer] = _make_input_node(producer, tid, metadata, name)
                    inputs[producer].phase = node.phase
                source_slot = 0
            edges.append(Edge(producer, node.id, tid, role, kind, source_slot, index))
            observed_inputs.append([producer, source_slot, role])
        missing = set(consumed[node.id].values()) - connected
        if missing:
            raise ValueError(f"{node.id} 缺少轨迹实测的输入来源 {sorted(missing)}")
        node.attrs["observed_inputs"] = observed_inputs
    unread = {value[0] for value in history.values()} - reads
    if unread:
        raise ValueError(f"decode 缺少逐层缓存读取 {sorted(unread)}")
    return edges, list(nodes) + list(inputs.values())


def _make_input_node(node_id: str, tid: str, meta: dict, role: str) -> Node:
    m = meta.get(tid, {})
    shape = m.get("shape")
    dtype = m.get("dtype")
    kind = "input"
    return Node(
        id=node_id,
        op="input",
        origin="input",
        op_confidence="exact",
        inputs=[],
        outputs=[TensorRef(tensor=tid, role="out", index=0)],
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
            stride=list(m["stride"]),
        )
    return out


__all__ = ["build_edges", "to_tensor_meta", "template_of"]
