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
    for node in nodes:
        observed_inputs = []
        connected = set()
        supplied = {consumed[node.id].get(ref.tensor, origins.get(id(builder._tensors[ref.tensor]["obj"])))
                    for ref in node.inputs}
        for tid, source in consumed[node.id].items():
            if source not in supplied:
                index = len(node.inputs)
                node.inputs.append(TensorRef(tid, role="input", index=index,
                                             path=f"dispatch[{index}]", arg_name="tensor"))
                supplied.add(source)
        incoming_state = builder.state_reads.get(node.id, {})
        bound_states = {tensor_view(builder._tensors[ref.tensor]["obj"]) for ref in node.inputs}
        for tid, state in incoming_state.items():
            view = tensor_view(builder._tensors[tid]["obj"])
            if view not in bound_states:
                node.inputs.append(TensorRef(tid, role="state", index=len(node.inputs),
                                             path=state.key.path, arg_name=state.key.role))
                bound_states.add(view)
        for index, ref in enumerate(node.inputs):
            tid = ref.tensor
            if tid not in metadata:
                raise ValueError(f"{node.id} in[{index}] 未登记张量 {tid}")
            tensor = builder._tensors[tid]["obj"]
            binding = builder.port_witnesses[node.id].get(tensor_view(tensor))
            if binding and ref.path is None:
                ref.path = binding.get("path")
                ref.arg_name = binding.get("arg_name")
                ref.arg_source = "dispatch"
            origin = consumed[node.id].get(tid, origins.get(id(tensor)))
            prior = incoming_state.get(tid)
            if prior is None and tensor_view(tensor) in history:
                prior = next((s for s in incoming_state.values()
                              if s is history[tensor_view(tensor)]), None)
            role, kind = "input", "data"
            if prior is not None:
                producer, source_slot = prior.producer, prior.slot
                role, kind = prior.key.role, "state"
                ref.role = "state"
                node.state.append(StateRef(role, "runtime_tensor", "read_write",
                                           list(tensor.shape), metadata[tid]["dtype"]))
            elif origin and origin[0] != node.id and origin in slots:
                producer, source_slot = origin[0], slots[origin]
                connected.add(origin)
                if producer in named_inputs and producer not in inputs:
                    inputs[producer] = _make_input_node(producer, origin[1], metadata, named_inputs[producer])
                    inputs[producer].phase = node.phase
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
