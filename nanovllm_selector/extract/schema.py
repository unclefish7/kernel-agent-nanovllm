"""计算图的数据契约。

设计原则：
1. 只用标准库 + dataclass，方便 JSON 往返；
2. 每个字段都能回答"这个结论是怎么来的"，证据放在 evidence；
3. 模板折叠（28 层 → 1 个模板 + instances）是序列化层面的折叠，不是语义上的合并。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SCHEMA_VERSION = "graph.v1"

OP_FAMILIES = {
    "embedding": "memory",
    "matmul": "compute",
    "lm_head": "compute",
    "rmsnorm": "normalization",
    "silu": "activation",
    "gelu": "activation",
    "mul": "elementwise",
    "add": "elementwise",
    "attention": "attention",
    "rope": "attention",
    "rope_table": "attention",
    "kv_write": "memory",
    "arange": "shape",
    "identity": "shape",
    "unknown": "unknown",
}


def _as_dict(value):
    """序列化时容忍两种写法：dataclass 实例，或已经转好的 dict。"""
    if isinstance(value, dict):
        return value
    return value.to_dict()


@dataclass
class Evidence:
    kind: str  # observed | source | param | dispatch | loader | rule
    detail: str
    ref: str | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "ref": self.ref}


@dataclass
class TensorRef:
    """节点的一个输入/输出。

    只记 `tensor` 是"吃了哪个张量"，不够生成代码用：`SiluAndMul` 要求
    打包张量的前一半是 gate、后一半是 up；`store_kvcache(key, value, ...)`
    的 key/value 同形同源，传反了不报错、结果全错。所以要额外记清
    "这个张量摆在这次调用的第几个参数位"。

    - `index`：在节点 inputs/outputs 列表里的下标（0 起）。
    - `path`：在原始调用里的位置表达式，如 `args[0]`、`args[1][2]`、
      `kwargs['hidden_states']`；来自 aten 算子的合成输入记成
      `aten.mul.Tensor#1` 这种形式（`#` 后是该算子的参数下标）。
    - `arg_name`：该位置对应的形参名。模块调用从 `forward` 签名取，
      aten 算子从 schema 取。取不到时为 None。
    """

    tensor: str
    role: str | None = None
    index: int | None = None
    path: str | None = None
    arg_name: str | None = None
    # 这个名字是从哪来的：`module`（模块 forward 的形参）还是 `aten`
    # （簇内某个算子的参数名）。两者含义不同——`self` / `other` /
    # `tensors` / `mat2` 是 aten 的通用参数名，不是模型里有人这样命名。
    arg_source: str | None = None

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor,
            "role": self.role,
            "index": self.index,
            "path": self.path,
            "arg_name": self.arg_name,
            "arg_source": self.arg_source,
        }


@dataclass
class StateRef:
    """节点的状态依赖：权重 / buffer / 运行时张量 / 标量。"""

    name: str
    kind: str  # parameter | buffer | runtime_tensor | scalar
    role: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "shape": self.shape,
            "dtype": self.dtype,
        }


@dataclass
class Node:
    id: str
    op: str
    inputs: list[TensorRef] = field(default_factory=list)
    outputs: list[TensorRef] = field(default_factory=list)
    state: list[StateRef] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    impl: dict = field(default_factory=dict)
    semantics: dict = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    op_confidence: str = "exact"  # exact | inferred | unknown
    origin: str = "module"  # module | synthetic | input
    phase: str = "prefill"
    template_id: str | None = None
    instances: list[int] = field(default_factory=list)
    # 仅内存态，不序列化
    start_seq: int = -1
    module_path: str | None = None

    @property
    def family(self) -> str:
        return OP_FAMILIES.get(self.op, "unknown")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "op": self.op,
            "op_confidence": self.op_confidence,
            "origin": self.origin,
            "phase": self.phase,
            "template_id": self.template_id,
            "instances": self.instances,
            "impl": self.impl,
            "semantics": self.semantics,
            "inputs": [_as_dict(t) for t in self.inputs],
            "outputs": [_as_dict(t) for t in self.outputs],
            "state": [_as_dict(s) for s in self.state],
            "attrs": self.attrs,
            "evidence": [_as_dict(e) for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=d["id"],
            op=d["op"],
            op_confidence=d.get("op_confidence", "exact"),
            origin=d.get("origin", "module"),
            phase=d.get("phase", "prefill"),
            template_id=d.get("template_id"),
            instances=list(d.get("instances") or []),
            impl=d.get("impl") or {},
            semantics=d.get("semantics") or {},
            inputs=[TensorRef(**t) for t in d.get("inputs") or []],
            outputs=[TensorRef(**t) for t in d.get("outputs") or []],
            state=[StateRef(**s) for s in d.get("state") or []],
            attrs=d.get("attrs") or {},
            evidence=[Evidence(**e) for e in d.get("evidence") or []],
        )


@dataclass
class Edge:
    from_node: str
    to_node: str
    tensor: str
    role: str | None = "input"
    kind: str = "data"  # data | state
    source_slot: int | None = None
    target_slot: int | None = None

    def to_dict(self) -> dict:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "tensor": self.tensor,
            "role": self.role,
            "kind": self.kind,
            "source_slot": self.source_slot,
            "target_slot": self.target_slot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)


@dataclass
class TensorMeta:
    id: str
    shape: list[int]
    dtype: str
    storage: str | None = None
    offset: int = 0
    numel: int = 0
    stride: list[int] | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "shape": self.shape,
            "dtype": self.dtype,
            "storage": self.storage,
            "offset": self.offset,
            "numel": self.numel,
            "stride": self.stride,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TensorMeta":
        return cls(**d)


@dataclass
class Graph:
    model: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    phases: list[str] = field(default_factory=lambda: ["prefill"])
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    tensors: dict[str, TensorMeta] = field(default_factory=dict)
    params: list[dict] = field(default_factory=list)
    graph_inputs: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    reconcile: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    graph_outputs: dict[str, list[dict]] = field(default_factory=dict)

    # ---- 查询 -------------------------------------------------------
    def node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def nodes_by_op(self, op: str) -> list[Node]:
        return [n for n in self.nodes if n.op == op]

    def op_counts(self, expand: bool = False) -> dict:
        """算子计数。

        expand=False：按图上实际存储的节点数统计（模板折叠后是 22 个节点，
        所以 matmul 只有 7）；expand=True：把每个模板节点按 instances 展开，
        得到"这一个模型里到底跑了几次"，与对账表的口径一致。
        """
        counts: dict[str, int] = {}
        for n in self.nodes:
            weight = len(n.instances) if (expand and n.instances) else 1
            counts[n.op] = counts.get(n.op, 0) + weight
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def template_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.instances]

    # ---- 校验 -------------------------------------------------------
    def validate(self) -> list[str]:
        problems: list[str] = []
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            seen, dup = set(), set()
            for i in ids:
                (dup if i in seen else seen).add(i)
            problems.append(f"节点 id 重复: {sorted(dup)[:5]}")
        id_set = set(ids)
        by_id = {node.id: node for node in self.nodes}
        for phase, outputs in self.graph_outputs.items():
            for ref in outputs:
                node = by_id.get(ref["node"])
                if (not node or node.phase != phase or not 0 <= ref["slot"] < len(node.outputs)
                        or ref["tensor"] not in self.tensors):
                    problems.append(f"{phase} 模型返回的张量缺少来源 {ref}")
        connections = set()
        for e in self.edges:
            if e.from_node not in id_set:
                problems.append(f"边指向不存在的源节点: {e.from_node}")
            if e.to_node not in id_set:
                problems.append(f"边指向不存在的目标节点: {e.to_node}")
            if e.tensor not in self.tensors:
                problems.append(f"边引用未登记张量 {e.tensor}")
            if (e.to_node in by_id and "observed_inputs" in by_id[e.to_node].attrs
                    and (e.source_slot is None or e.target_slot is None)):
                problems.append(f"{e.from_node} → {e.to_node} 缺少实测槽位")
            if e.source_slot is not None and e.from_node in by_id and e.to_node in by_id:
                source, target = by_id[e.from_node], by_id[e.to_node]
                if not 0 <= e.source_slot < len(source.outputs):
                    problems.append(f"{source.id} 缺少 out[{e.source_slot}]")
                if e.target_slot is None or not 0 <= e.target_slot < len(target.inputs):
                    problems.append(f"{target.id} 输入槽位无效 {e.target_slot}")
                elif e.kind != "layer" and "observed_inputs" in target.attrs:
                    observed = target.attrs["observed_inputs"]
                    if (e.target_slot >= len(observed)
                            or observed[e.target_slot] != [source.id, e.source_slot, e.role]
                            or target.inputs[e.target_slot].tensor != e.tensor):
                        problems.append(f"{target.id} in[{e.target_slot}] 有未经实测的来源 {source.id}")
                connections.add((target.id, e.target_slot, e.from_node, e.source_slot, e.role,
                                 e.tensor, e.kind))
        for n in self.nodes:
            if "observed_outputs" in n.attrs:
                if ([ref.tensor for ref in n.outputs] != n.attrs["observed_outputs"]
                        or [ref.role for ref in n.outputs] != n.attrs.get("observed_output_roles")
                        or [ref.index for ref in n.outputs] != list(range(len(n.outputs)))):
                    problems.append(f"{n.id} 输出槽位与实测数量、顺序或角色不符")
            if "observed_inputs" in n.attrs:
                observed = n.attrs["observed_inputs"]
                if len(observed) != len(n.inputs):
                    problems.append(f"{n.id} 输入数量与录制不符")
                for index, (ref, witness) in enumerate(zip(n.inputs, observed)):
                    if ref.index != index:
                        problems.append(f"{n.id} in[{index}] 输入序号与录制不符")
                    if witness is None:
                        if ref.tensor not in self.tensors or self.tensors[ref.tensor].numel != 0:
                            problems.append(f"{n.id} 非空输入被跳过: {ref.tensor}")
                        continue
                    source, slot, role = witness
                    kind = "state" if ref.role == "state" else "data"
                    if (n.id, index, source, slot, role, ref.tensor, kind) not in connections:
                        problems.append(f"{n.id} in[{index}] {role} 缺少实测来源 {source} out[{slot}]")
            if not n.evidence:
                problems.append(f"节点 {n.id} 没有任何 evidence")
            for t in list(n.inputs) + list(n.outputs):
                if t.tensor not in self.tensors:
                    problems.append(f"节点 {n.id} 引用了未登记的张量 {t.tensor}")
        previous = {}
        state_edges = {(e.from_node, e.source_slot, e.to_node, e.role)
                       for e in self.edges if e.kind == "state"}
        for slot in self.env.get("state_slots", ()):
            name = slot["path"]
            producer = by_id.get(slot["producer"])
            source = slot["source_slot"]
            tensor = self.tensors.get(slot["tensor"])
            if (not producer or not 0 <= source < len(producer.outputs)
                    or producer.outputs[source].role != slot["role"] or tensor is None
                    or tensor.shape != slot["shape"] or tensor.dtype != slot["dtype"]):
                problems.append(f"{slot['phase']} {name} 的状态输出与图不符")
            prior = previous.get(name)
            for reader in slot["readers"]:
                if not prior or (prior["producer"], prior["source_slot"], reader, prior["role"]) not in state_edges:
                    problems.append(f"{slot['phase']} {name} 缺少通往 {reader} 的历史边")
            previous[name] = slot
        if self.env.get("shape_probes"):
            from .shapes import validate_profiles

            problems.extend(validate_profiles(self))
        return problems

    # ---- 序列化 -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "env": self.env,
            "phases": self.phases,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "params": self.params,
            "graph_inputs": self.graph_inputs,
            "graph_outputs": self.graph_outputs,
            "conflicts": self.conflicts,
            "reconcile": self.reconcile,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            model=d.get("model") or {},
            env=d.get("env") or {},
            phases=d.get("phases") or ["prefill"],
            nodes=[Node.from_dict(x) for x in d.get("nodes") or []],
            edges=[Edge.from_dict(x) for x in d.get("edges") or []],
            tensors={k: TensorMeta.from_dict(v) for k, v in (d.get("tensors") or {}).items()},
            params=d.get("params") or [],
            graph_inputs=d.get("graph_inputs") or [],
            graph_outputs=d.get("graph_outputs") or {},
            conflicts=d.get("conflicts") or [],
            reconcile=d.get("reconcile") or {},
        )

    @classmethod
    def from_json(cls, text: str) -> "Graph":
        return cls.from_dict(json.loads(text))
