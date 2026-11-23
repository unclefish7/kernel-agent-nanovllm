"""从录制结果重建节点。

核心想法：**节点 = 一次模块调用里"连在一起"的那组非透明 aten 算子。**

为什么这样切：

* `nn.Linear` 里是 `t → mm → view` 四个 aten 算子，语义上只有一个 matmul；
* `Qwen3DecoderLayer` 自己不调用子模块，只做两次残差加法，应该出两个 add 节点；
* `Qwen3Attention` 内部有 rope、写 KV cache、sdpa，它们通过张量连成一串，
  在 M1 里合成一个 attention 节点（`attrs.sub_ops` 保留完整清单）。

透明算子（view/transpose/slice/...）不建节点，只做张量别名传播。这样既能
"一个节点对应一次有意义的计算"，又不会丢掉连接关系。
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .schema import Evidence, Node, StateRef, TensorRef
from .source import source_ref
from .dispatch_program import instruction

# 只改变张量视图/布局、不产生新数值的算子：不建节点，只传播别名
TRANSPARENT = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.reshape.default",
    "aten.transpose.int",
    "aten.t.default",
    "aten.permute.default",
    "aten.squeeze.dim",
    "aten.squeeze.default",
    "aten.squeeze.dims",
    "aten.unsqueeze.default",
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.narrow.default",
    "aten.expand.default",
    "aten.expand_as.default",
    "aten.contiguous.default",
    "aten.clone.default",
    "aten.detach.default",
    "aten.alias.default",
    "aten.lift_fresh.default",
    "aten._to_copy.default",
    "aten.as_strided.default",
    "aten.split_with_sizes.default",
    "aten.split.Tensor",
    "aten.unbind.int",
    "aten.chunk.default",
}

FORMULAS = {
    "embedding": "y = W_embed[vocab_id]",
    "matmul": "y = x @ W.T + b",
    "lm_head": "logits = hidden @ W_lm.T",
    "rmsnorm": "y = x / sqrt(mean(x^2) + eps) * w",
    "silu": "y = x * sigmoid(x)",
    "gelu": "y = x * Phi(x)",
    "rope": "(q, k) -> (q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin)",
    "rope_table": "cos, sin = cos_sin(position_ids; inv_freq)",
    "attention": "y = softmax(scale * (q @ k^T) + mask) @ v; scale from dispatch",
    "safe_softmax": "沿指定轴归一化；整行均为负无穷时输出零",
    "topk_routing": "选择索引 = topk(选择分数).indices；按索引从路由分数取权重，见有序配方",
    "indexed_dispatch": "由掩码的非零坐标索引输入行及其权重，索引保留在配方中",
    "indexed_accumulate": "将加权行按索引累加回目标；后续选择按配方顺序执行",
    "mul": "y = a * b",
    "add": "y = a + alpha * b",
    "arange": "y = arange(start, end, step)",
    "conv1d": "y = conv1d(x, weight, bias, stride, padding, dilation, groups)",
    "bmm": "y[b] = a[b] @ b_matrix[b]",
    "view": "y = view(x; shape, stride, offset)",
    "unknown": "",
}

LAYER_PATH = re.compile(r"^(?:model\.)?layers\.(\d+)(?:\.(.*))?$")


def short_path(path: str) -> str:
    if path == "":
        return "R"
    m = LAYER_PATH.match(path)
    if m:
        idx, rest = m.group(1), m.group(2)
        return f"L{idx}" + (f".{rest}" if rest else "")
    if path.startswith("model.layers."):
        rest = path[len("model.layers.") :]
        idx, _, tail = rest.partition(".")
        return f"L{idx}" + (f".{tail}" if tail else "")
    if path.startswith("model."):
        return path[len("model.") :]
    return path


def template_of(short: str) -> str:
    return re.sub(r"^L\d+", "L*", short)


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def add(self, x: int) -> int:
        self.parent.setdefault(x, x)
        return x

    def find(self, x: int) -> int:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra


@dataclass
class _Cluster:
    """一个节点在聚类阶段的样子：只存事实，不做取舍。

    `consumed` / `produced` 是这个簇里**所有**算子（含透明算子）吃进和吐出的
    张量全集。节点的输入不是"扫一遍觉得该留的"，而是这两个集合的差——
    见 `build()` 里的 `_bucket`。
    """

    cid: int
    path: str
    call_seq: int
    ops: list = field(default_factory=list)          # 非透明算子（用于判定 op）
    op_seqs: list = field(default_factory=list)      # 归属本簇的算子序号（闭世界核对用）
    consumed: dict = field(default_factory=dict)     # tid -> slot；所有算子的操作数
    consumed_at: dict = field(default_factory=dict)  # tid -> 第一次被消费的 seq
    produced: set = field(default_factory=set)       # 所有算子产出过的 tid（含视图）
    produced_at: dict = field(default_factory=dict)  # tid -> 第一次被产出的 seq
    min_seq: int = 1 << 60
    max_seq: int = -1
    call: object = None  # 产生这个簇的模块调用（decompose 用）

    def merge(self, other: "_Cluster") -> None:
        self.ops.extend(other.ops)
        self.op_seqs.extend(other.op_seqs)
        for tid, slot in other.consumed.items():
            self.consumed.setdefault(tid, slot)
        for tid, seq in other.consumed_at.items():
            if tid not in self.consumed_at or seq < self.consumed_at[tid]:
                self.consumed_at[tid] = seq
        self.produced |= other.produced
        for tid, seq in other.produced_at.items():
            if tid not in self.produced_at or seq < self.produced_at[tid]:
                self.produced_at[tid] = seq
        self.min_seq = min(self.min_seq, other.min_seq)
        self.max_seq = max(self.max_seq, other.max_seq)


class GraphBuilder:
    def __init__(self, model: nn.Module, recorder, config, dtype: str = "float32",
                 tensor_prefix: str = "", node_prefix: str = ""):
        self.model = model
        self.rec = recorder
        self.config = config
        self.dtype = dtype
        self.tensor_prefix = tensor_prefix
        self.node_prefix = node_prefix
        self._tensor_ids: dict[int, str] = {}
        self._tensors: dict[str, dict] = {}
        self._uf = _UnionFind()
        self._clusters: dict[int, _Cluster] = {}
        self._src: dict[int, tuple] = {}
        self._next_cid = 0
        self._id_counter = 0
        self.conflicts: list[dict] = []
        self.unresolved: dict[str, int] = {}
        # storage -> ("node", cid)：按存储块找生产者，视图/别名也认。
        self._storage_src: dict[str, tuple] = {}
        self._state_by_storage = {}
        for kind, items in [("parameter", model.named_parameters()), ("buffer", model.named_buffers())]:
            for name, tensor in items:
                self._state_by_storage[tensor.untyped_storage()._cdata] = (name, kind, tensor)

    # ---- 张量登记 ---------------------------------------------------
    def tid(self, t: torch.Tensor) -> str:
        key = id(t)
        if key not in self._tensor_ids:
            self._id_counter += 1
            tid = f"{self.tensor_prefix}t{self._id_counter:04d}"
            self._tensor_ids[key] = tid
            self._tensors[tid] = {
                "shape": list(t.shape),
                "dtype": _dtype_str(t),
                "storage": hex(t.untyped_storage().data_ptr()),
                "offset": int(t.storage_offset()),
                "stride": list(t.stride()),
                "numel": int(t.numel()),
                "obj": t,
            }
        return self._tensor_ids[key]

    # ---- 第一遍：聚类 -----------------------------------------------
    #
    # 规则：把"同一次模块调用里连续执行的一串非透明算子"并成一个节点。
    #
    #   - Linear 里的 mm 前后夹着 view/t（透明），所以一个 Linear = 一个节点；
    #   - Qwen3Attention 内部 rope + 写 cache + sdpa 是一口气算完的，= 一个节点；
    #   - Qwen3DecoderLayer 的两次残差 add 中间隔着 norm/mlp 的调用，会被打断，
    #     所以 = 两个节点（这正是我们想要的）。
    #
    # 关键：跨模块不能并。一并对，整张图会沿数据依赖缩成一个点。
    def _cluster_dispatches(self):
        current: _Cluster | None = None
        current_call = None
        broken = True

        for d in self.rec.dispatches:
            call = self._call_for(d)
            if call is None:
                continue

            if d.op in TRANSPARENT:
                # 透明算子只传播别名；但如果它属于别的模块，说明中间被插入了
                # 别的计算，当前"连续串"到此为止。
                if current_call is not None and call is not current_call:
                    broken = True
                src = None
                for t in d.inputs:
                    s = self._src.get(id(t))
                    if s is not None:
                        src = s
                        break
                for t in d.outputs:
                    if src is None:
                        self._src[id(t)] = ("ext", self.tid(t))
                    else:
                        self._src[id(t)] = src
                # 簇内部自己的 view/transpose 也是这个节点的一部分：把它们的
                # 产出记进 produced，否则"同一张量换个形状再吃一次"会被算成
                # 第二次从外面拿进来的输入。
                if current is not None and call is current_call and not broken:
                    current.op_seqs.append(d.seq)
                    for t in d.inputs:
                        slot = _aten_slot(d, t)
                        self._note_consumed(current, self.tid(t),
                                            dict(slot, source="aten") if slot else None, d.seq)
                    for t in d.outputs:
                        tid = self.tid(t)
                        current.produced.add(tid)
                        current.produced_at.setdefault(tid, d.seq)
                        self._storage_src[self._tensors[tid]["storage"]] = (
                            "node", current.cid,
                        )
                continue

            if current is not None and call is current_call and not broken:
                cluster = current
            else:
                self._next_cid += 1
                cluster = _Cluster(cid=self._next_cid, path=call.path, call_seq=call.seq_start,
                                   call=call)
                self._clusters[cluster.cid] = cluster
                self._uf.add(cluster.cid)
                current, current_call, broken = cluster, call, False

            cluster.min_seq = min(cluster.min_seq, d.seq)
            cluster.max_seq = max(cluster.max_seq, d.seq)
            cluster.ops.append(d.op)
            cluster.op_seqs.append(d.seq)

            for t in d.inputs:
                tid_in = self.tid(t)
                # 每个操作数有它自己的参数位置，不能拿第一个对上的套给全组。
                slot = None
                if call is not None:
                    slot = call.input_slots.get(id(t))
                    if slot is None:
                        # 模块内部先 view 再算（`F.linear` 会把 3D 输入摊成 2D），
                        # 对象换了但 storage 没换，靠 storage 把参数位找回来。
                        slot = call.input_storage_slots.get(
                            hex(t.untyped_storage().data_ptr())
                        )
                    if slot is not None:
                        slot = dict(slot, source="module")
                if slot is None:
                    slot = _aten_slot(d, t)
                    if slot is not None:
                        slot = dict(slot, source="aten")
                self._note_consumed(cluster, tid_in, slot, d.seq)
            for t in d.outputs:
                tid = self.tid(t)
                cluster.produced.add(tid)
                cluster.produced_at.setdefault(tid, d.seq)
                self._src[id(t)] = ("node", cluster.cid)
                self._storage_src[self._tensors[tid]["storage"]] = ("node", cluster.cid)

    def _note_consumed(self, cluster: "_Cluster", tid: str, slot, seq: int) -> None:
        """记一次"被消费"。槽位信息只补不覆盖：先到的可能是 aten 参数名，
        后到的才是真正的模块形参名，谁的更有用留谁。"""
        if tid not in cluster.consumed or (slot and not cluster.consumed[tid]):
            cluster.consumed[tid] = slot
        if tid not in cluster.consumed_at or seq < cluster.consumed_at[tid]:
            cluster.consumed_at[tid] = seq

    def _call_for(self, d):
        best = None
        for c in self.rec.calls:
            if c.path == d.inside and c.seq_start <= d.seq <= c.seq_end:
                if best is None or c.seq_start > best.seq_start:
                    best = c
        return best

    # ---- 第二遍：合并同根簇 -----------------------------------------
    def _merged_clusters(self) -> dict[int, _Cluster]:
        merged: dict[int, _Cluster] = {}
        for cid, cl in self._clusters.items():
            root = self._uf.find(cid)
            if root not in merged:
                merged[root] = _Cluster(cid=root, path=cl.path, call_seq=cl.call_seq,
                                        min_seq=cl.min_seq, max_seq=cl.max_seq,
                                        call=cl.call)
            merged[root].merge(cl)
        return merged

    # ---- 节点命名 ---------------------------------------------------
    def _assign_ids(self, clusters: list[_Cluster]) -> list[tuple[str, _Cluster]]:
        by_path: dict[str, list[_Cluster]] = {}
        for cl in clusters:
            by_path.setdefault(cl.path, []).append(cl)
        out = []
        for path, cls in by_path.items():
            cls.sort(key=lambda c: c.min_seq)
            base = short_path(path)
            module = self._module(path)
            for i, cl in enumerate(cls):
                op = classify_ops(cl.ops, module, self.config)[0]
                bare = op in ("mul", "add", "arange", "unknown")
                container = module is not None and any(module.children())
                if len(cls) == 1 and not (bare and container):
                    out.append((base, cl))
                elif len(cls) == 1:
                    # 容器模块（如 Qwen3MLP）里只有一个逐元素算子时，用
                    # `L*.mlp.mul` 这种名字，而不是把整个容器叫成 mul。
                    out.append((f"{base}.{_op_name(op)}", cl))
                else:
                    out.append((f"{base}.{_op_name(op)}@{i}", cl))
        return out

    def _module(self, path: str) -> nn.Module | None:
        for p, m in self.model.named_modules():
            if p == path:
                return m
        return None

    def _own_state_storages(self, module: nn.Module | None) -> set[str]:
        """该模块自己持有的权重/buffer 的 storage —— 它们不是数据流输入。

        比张量身份更宽一层：`weight.t()` / `expand` 之类是视图，张量对象变了，
        但 storage 不变。权重常常以视图形式进入算子（mm 吃的是转置后的 weight），
        只按对象身份过滤会漏掉全部权重。
        """
        if module is None:
            return set()
        storages = set()
        for _, p in module.named_parameters(recurse=False):
            storages.add(hex(p.untyped_storage().data_ptr()))
        for _, b in module.named_buffers(recurse=False):
            storages.add(hex(b.untyped_storage().data_ptr()))
        return storages

    def _collect_programs(self):
        # 每条透明变换都保留在配方中，包括首个数值算子之前的布局变换。
        if not self._clusters and self.rec.dispatches:
            steps = self.rec.dispatches
            call = self._call_for(steps[0])
            if call is not None and all(d.op in TRANSPARENT for d in steps):
                cl = _Cluster(cid=0, path=call.path, call_seq=call.seq_start, call=call,
                              min_seq=steps[0].seq, max_seq=steps[-1].seq)
                self._clusters[0] = cl
                for d in steps:
                    cl.op_seqs.append(d.seq)
                    for t in d.inputs:
                        slot = _aten_slot(d, t)
                        self._note_consumed(cl, self.tid(t), dict(slot, source="aten") if slot else None, d.seq)
                    for t in d.outputs:
                        tid = self.tid(t)
                        cl.produced.add(tid)
                        cl.produced_at.setdefault(tid, d.seq)
        owners = {s: cl.cid for cl in self._clusters.values() for s in cl.op_seqs}
        last_owner = None
        pending = []
        self.programs = {cid: [] for cid in self._clusters}
        for d in self.rec.dispatches:
            pending.append(d)
            if d.seq in owners:
                last_owner = owners[d.seq]
                self.programs[last_owner].extend(pending)
                pending.clear()
        if last_owner is not None:
            self.programs[last_owner].extend(pending)

    # ---- 主流程 -----------------------------------------------------
    def build(self, phase: str = "prefill") -> list[Node]:
        self._cluster_dispatches()
        self._collect_programs()
        activation_counts = Counter(d.op for d in self.rec.dispatches
                                    if d.op in {"aten.silu.default", "aten.gelu.default"})
        merged = self._merged_clusters()

        # 合并后重新解析 src 指向
        def resolve_src(s):
            if s[0] == "node":
                return ("node", self._uf.find(s[1]))
            return s

        cluster_list = list(merged.values())
        named = [(self.node_prefix + name, cluster)
                 for name, cluster in self._assign_ids(cluster_list)]
        self.id_of_cid = {cl.cid: nid for nid, cl in named}

        nodes: list[Node] = []
        for nid, cl in named:
            module = self._module(cl.path)
            op, confidence = classify_ops(cl.ops, module, self.config)
            state_storages = self._own_state_storages(module)
            in_refs = self._node_inputs(cl, state_storages, resolve_src)
            node = self._make_node(nid, cl, op, confidence, module, in_refs, phase)
            node._recorded_activations = activation_counts
            node._program_cid = cl.cid
            nodes.append(node)
        self._bind_program_inputs(nodes)
        return nodes

    def _bind_program_inputs(self, nodes):
        """以执行时序解析配方自由变量，包括槽位登记前已经变形的张量。"""
        owners = {d.seq: n for n in nodes for d in self.programs[n._program_cid]}
        previous = {}
        input_views = {}
        defined = {}
        for n in nodes:
            n.attrs["program_inputs"] = {}
            defined[n.id] = set()
            input_views[n.id] = {self._tensors[r.tensor]["obj"].untyped_storage()._cdata: r.tensor
                                 for r in reversed(n.inputs)}
        root_slots = self.rec.calls[0].input_slots if self.rec.calls else {}
        for d in self.rec.dispatches:
            node = owners.get(d.seq)
            if node is None:
                continue
            bindings = node.attrs["program_inputs"]
            for t in d.inputs:
                tid = self.tid(t)
                if tid in defined[node.id] or tid in bindings:
                    continue
                name = node.attrs["program_state"].get(tid)
                if name is not None:
                    binding = {"kind": "state", "name": name}
                elif tid in previous:
                    producer, seq = previous[tid]
                    binding = {"kind": "producer", "node": producer, "seq": seq, "tensor": tid}
                elif d.op == "aten.lift_fresh.default":
                    binding = {"kind": "literal", "value": d.before[id(t)].tolist()}
                else:
                    view = next((r.tensor for r in node.inputs if r.tensor == tid),
                                input_views[node.id].get(t.untyped_storage()._cdata))
                    if view:
                        binding = {"kind": "input", "tensor": view}
                    else:
                        binding = dict(kind="external", tensor=tid, **root_slots.get(id(t), {}))
                bindings[tid] = binding
            for t in d.outputs:
                tid = self.tid(t)
                defined[node.id].add(tid)
                previous[tid] = node.id, d.seq
        self.program_input_witnesses = {n.id: deepcopy(n.attrs["program_inputs"]) for n in nodes}

    # ---- 输入/输出：集合差，不是边扫边丢 -----------------------------
    def _node_inputs(self, cl: "_Cluster", state_storages: set[str], resolve_src):
        """节点的输入 = 簇内算子的 `consumed` − `produced` − `state`。

        闭世界：每个被消费的张量必须恰好落进三个桶之一——
          1. internal：本簇自己算出来的（按对象身份，或按存储块 + 生产早于消费）
          2. state：模块自己的权重/buffer（按存储块）
          3. input：剩下的
        没有第四个桶。所以这里不存在"判断不要就 `continue`"的位置：
        被丢掉的东西只可能是"明确归了类"，而不是"忘了留"。
        """
        # 同一存储块的最早生产时间：视图/别名的归属靠它判断先后。
        produced_storage_seq: dict[str, int] = {}
        for tid, seq in cl.produced_at.items():
            st = self._tensors[tid]["storage"]
            cur = produced_storage_seq.get(st)
            produced_storage_seq[st] = seq if cur is None else min(cur, seq)

        internal: set[str] = set()
        state: set[str] = set()
        external: dict[str, dict] = {}
        for tid, slot in cl.consumed.items():
            m = self._tensors[tid]
            if tid in cl.produced and cl.produced_at[tid] < cl.consumed_at[tid]:
                internal.add(tid)
                continue
            if (m["storage"] in state_storages
                    or m["obj"].untyped_storage()._cdata in self._state_by_storage):
                state.add(tid)
                continue
            seq = produced_storage_seq.get(m["storage"])
            if seq is not None and seq < cl.consumed_at.get(tid, 1 << 60):
                # 同一块存储上更早时刻产出的东西：视图、切片、原地写。
                internal.add(tid)
                continue
            external[tid] = slot or {}

        # 只合并指向完全相同视图的别名；同一块内存的不同切片可占不同参数位。
        groups: dict[tuple, list[str]] = {}
        for tid in external:
            m = self._tensors[tid]
            slot = external[tid]
            module_path = slot.get("path") if slot.get("source") == "module" else None
            key = (m["storage"], m["offset"], tuple(m["shape"]), tuple(m["stride"]), m["dtype"], module_path)
            groups.setdefault(key, []).append(tid)

        in_refs = []
        for tids in groups.values():
            # 有模块形参名的那一个当代表；否则用最早出现的。
            rep = next(
                (t for t in tids if (external[t] or {}).get("source") == "module"),
                tids[0],
            )
            slot = external[rep] or {}
            in_refs.append(
                (rep, self._producer_src(rep, resolve_src), "input", slot)
            )

        # 闭世界核对：三类桶必须正好覆盖 `consumed`，不多不少。
        covered = len(internal) + len(state) + len(external)
        if covered != len(cl.consumed):
            self.conflicts.append({
                "kind": "input_classification",
                "path": cl.path,
                "detail": f"consumed={len(cl.consumed)} internal={len(internal)} "
                          f"state={len(state)} external={len(external)}",
            })

        return in_refs

    def _producer_src(self, tid: str, resolve_src):
        """这个张量是谁产的：先按对象身份，再按存储块（视图跨模块时用）。"""
        obj = self._tensors[tid]["obj"]
        src = self._src.get(id(obj))
        if src is None:
            src = self._storage_src.get(self._tensors[tid]["storage"], ("ext", tid))
        return resolve_src(src)

    def tensor_meta(self) -> dict:
        return {tid: {k: v for k, v in m.items() if k != "obj"} for tid, m in self._tensors.items()}

    def _make_node(self, nid, cl, op, confidence, module, in_refs, phase) -> Node:
        attrs = {
            "dispatch_ops": _op_histogram(cl.ops),
            "dispatch_count": len(cl.ops),
            "module_path": cl.path or "R",
            "module_class": type(module).__name__ if module is not None else None,
            "seq": [cl.min_seq, cl.max_seq],
            "program": [instruction(d, self.tid) for d in self.programs[cl.cid]],
            "program_state": {},
        }
        state: list[StateRef] = []
        for d in self.programs[cl.cid]:
            for t in d.inputs:
                binding = self._state_by_storage.get(t.untyped_storage()._cdata)
                if binding is None:
                    continue
                name, kind, base = binding
                attrs["program_state"][self.tid(t)] = name
                parent, _, local = name.rpartition(".")
                if parent != cl.path and not any(s.name == name for s in state):
                    state.append(StateRef(name=name, kind=kind, role=_state_role(local, op),
                                          shape=list(base.shape), dtype=_dtype_str(base)))
        if module is not None:
            for name, p in module.named_parameters(recurse=False):
                state.append(StateRef(name=name, kind="parameter", role=_state_role(name, op),
                                      shape=list(p.shape), dtype=_dtype_str(p)))
            for name, b in module.named_buffers(recurse=False):
                state.append(StateRef(name=name, kind="buffer", role=_state_role(name, op),
                                      shape=list(b.shape), dtype=_dtype_str(b)))
            # 标量也算 state：eps / variance_epsilon 这类归一化参数不写在 config 里
            # 就得到别处去找，模型文件里还得再用一次。
            for attr in ("eps", "variance_epsilon", "scale", "negative_slope"):
                if hasattr(module, attr) and isinstance(getattr(module, attr), (int, float)):
                    state.append(StateRef(name=attr, kind="scalar", role=attr,
                                          shape=None, dtype=type(getattr(module, attr)).__name__))
        evidence = [
            Evidence("dispatch", f"{len(cl.ops)} 个 aten 算子: " + ", ".join(sorted(set(cl.ops))[:6]),
                     ref="seq:%d-%d" % (cl.min_seq, cl.max_seq)),
        ]
        if module is not None:
            src = source_ref(module)
            if src["file"]:
                evidence.append(Evidence("source", f"{src['qualname']} @ {src['file']}:{src['class_line']}",
                                         ref=src["sha256"]))
            else:
                evidence.append(Evidence("rule", "builtin: 没有 Python 源码可定位", ref=None))
        for name, s in [(s.name, s) for s in state]:
            if s.kind == "parameter":
                evidence.append(Evidence("param", f"{name} shape={s.shape}", ref=nid))

        inputs = []
        for i, (tid, _, _, slot) in enumerate(in_refs):
            slot = slot or {}
            inputs.append(
                TensorRef(
                    tensor=tid,
                    role="in",
                    index=i,
                    path=slot.get("path"),
                    arg_name=slot.get("arg_name"),
                    arg_source=slot.get("source"),
                )
            )
        node = Node(
            id=nid,
            op=op,
            op_confidence=confidence,
            origin="module",
            phase=phase,
            inputs=inputs,
            state=state,
            attrs=attrs,
            impl={
                "class": type(module).__name__ if module is not None else None,
                "module_path": cl.path or "R",
                "source": source_ref(module) if module is not None else None,
            },
            semantics={
                "formula": FORMULAS.get(op, ""),
                "op": op,
                "bindings": _bindings(op, inputs),
            },
            evidence=evidence,
        )
        node._in_srcs = {tid: s for tid, s, _, _ in in_refs}  # type: ignore[attr-defined]
        node.start_seq = cl.min_seq
        return node


# 公式变量名 -> 输入位置。只给"单个算子、且输入顺序就是数学式里的顺序"的类型
# 写死名字。复合节点（attention 是 rope + 写 cache + sdpa 合出来的）输入里
# 还有 cos/sin 和 cache，叫 q/k/v 就是撒谎，一律退化成 in0/in1。
BINDING_NAMES = {
    "mul": ("a", "b"),
    "add": ("a", "b"),
    "matmul": ("x",),
    "lm_head": ("x",),
    "rmsnorm": ("x",),
    "silu": ("x",),
    "embedding": ("input_ids",),
}


def _bindings(op: str, inputs: list) -> dict:
    """公式变量 -> `in[i]`。配合 TensorRef 的 path/arg_name 用。"""
    names = BINDING_NAMES.get(op, ())
    out = {}
    for i, ref in enumerate(inputs):
        key = names[i] if i < len(names) else f"in{i}"
        out[key] = f"in[{i}]"
    return out


def _aten_slot(d, t) -> dict | None:
    """合成输入的参数位：来自哪个 aten 算子的第几个参数。

    形如 `aten.cat.default` 的 `tensors` 列表会被展开成 `args[0][1]`，
    所以位置是真实的参数树位置，不会把列表下标误当成参数下标。
    """
    slot = getattr(d, "input_slots", {}).get(id(t))
    if slot is not None:
        return slot
    return None


def _op_name(op: str) -> str:
    return {"add": "residual_add"}.get(op, op)


def _state_role(name: str, op: str) -> str:
    if op == "rmsnorm" and name == "weight":
        return "weight"
    if op == "rmsnorm" and name == "eps":
        return "eps"
    if op in ("matmul", "lm_head", "embedding") and name == "weight":
        return "weight"
    if name == "bias":
        return "bias"
    return "other"


def _op_histogram(ops: list[str]) -> dict:
    h: dict[str, int] = {}
    for o in ops:
        h[o] = h.get(o, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))





def classify_ops(ops: list[str], module: nn.Module | None, config) -> tuple[str, str]:
    """op 标签由"实测到的 aten 算子"决定，不看模块名。

    名字只用于最后一道兜底（并且要求同时有形状/属性证据）。
    """
    o = set(ops)
    if not ops:
        return "view", "exact"
    if {"aten.topk.default", "aten.gather.default"} <= o:
        allowed = {"aten.mm.default", "aten.addmm.default", "aten.sigmoid.default", "aten.add.Tensor",
                   "aten.topk.default", "aten.gather.default", "aten.sum.dim_IntList", "aten.zeros_like.default",
                   "aten.scatter_.value", "aten.bitwise_not.default", "aten.masked_fill.Scalar",
                   "aten.mul.Tensor", "aten.div_.Tensor"}
        return ("topk_routing", "inferred") if o <= allowed else ("unknown", "unknown")
    if {"aten.index_add_.default", "aten.mul.Tensor"} <= o:
        allowed = {"aten.mul.Tensor", "aten.index_add_.default", "aten.nonzero.default", "aten.index.Tensor"}
        return ("indexed_accumulate", "inferred") if o <= allowed else ("unknown", "unknown")
    if {"aten.scatter_.value", "aten.nonzero.default", "aten.index.Tensor"} <= o:
        allowed = {"aten.scatter_.value", "aten.nonzero.default", "aten.index.Tensor", "aten.zeros_like.default",
                   "aten.zeros.default", "aten.min.default", "aten.max.default", "aten._local_scalar_dense.default"}
        return ("indexed_dispatch", "inferred") if o <= allowed else ("unknown", "unknown")
    if ops == ["aten._safe_softmax.default"]:
        return "safe_softmax", "exact"
    if any("scaled_dot_product" in x for x in o):
        return "attention", "exact"
    if (ops.count("aten.bmm.default") == 2 and ops.count("aten._safe_softmax.default") == 1
            and ops.index("aten.bmm.default") < ops.index("aten._safe_softmax.default")
            and "aten.bmm.default" in ops[ops.index("aten._safe_softmax.default") + 1:]):
        return "attention", "inferred"
    if "aten.embedding.default" in o:
        return "embedding", "exact"
    if o == {"aten.convolution.default"} and isinstance(module, nn.Conv1d):
        return "conv1d", "exact"
    if o == {"aten.bmm.default"}:
        if isinstance(module, nn.Linear):
            return ("lm_head" if module.out_features == getattr(config, "vocab_size", None)
                    else "matmul"), "exact"
        return "bmm", "exact"
    if o & {"aten.mm.default", "aten.addmm.default", "aten.baddbmm.default", "aten.matmul.default"}:
        vocab = getattr(config, "vocab_size", None)
        if isinstance(module, nn.Linear) and vocab and module.out_features == vocab:
            return "lm_head", "exact"
        return "matmul", "exact"
    if "aten.rsqrt.default" in o and ("aten.mean.dim" in o or "aten.mean.default" in o):
        return "rmsnorm", "exact"
    if ops == ["aten.silu.default"]:
        return "silu", "exact"
    if ops == ["aten.gelu.default"]:
        return "gelu", "exact"
    if "aten.cos.default" in o and "aten.sin.default" in o:
        # 这一簇只算 cos/sin 表（Qwen3RotaryEmbedding），不旋转任何张量。
        # 旋转（q*cos + rotate_half(q)*sin）发生在 attention 簇内部。
        return "rope_table", "exact"
    if "aten.mul.Tensor" in o and "aten.add.Tensor" in o and "aten.cat.default" in o:
        # rope 展开后的乘法/加法/拼接群
        return "rope", "inferred"
    if ops == ["aten.mul.Tensor"]:
        return "mul", "exact"
    if ops == ["aten.add.Tensor"]:
        return "add", "exact"
    if len(ops) == 1 and "arange" in ops[0]:
        return "arange", "exact"
    return "unknown", "unknown"
