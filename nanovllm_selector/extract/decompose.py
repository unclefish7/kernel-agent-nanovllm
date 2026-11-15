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
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .schema import Evidence, Node, StateRef, TensorRef
from .source import source_ref

# 只改变张量视图/布局、不产生新数值的算子：不建节点，只传播别名
TRANSPARENT = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.reshape.default",
    "aten.transpose.int",
    "aten.t.default",
    "aten.permute.default",
    "aten.squeeze.dim",
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
    "attention": "y = softmax(q @ k^T / sqrt(d) + mask) @ v",
    "mul": "y = a * b",
    "add": "y = a + b",
    "arange": "y = arange(start, end)",
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
    out_tensors: list = field(default_factory=list)  # 非透明算子的产出（节点的输出）
    internal_out: set = field(default_factory=set)   # 被本簇非透明算子继续消费的产出（不算节点输出）
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
        self.out_tensors.extend(other.out_tensors)
        self.internal_out |= other.internal_out
        self.min_seq = min(self.min_seq, other.min_seq)
        self.max_seq = max(self.max_seq, other.max_seq)


class GraphBuilder:
    def __init__(self, model: nn.Module, recorder, config, dtype: str = "float32"):
        self.model = model
        self.rec = recorder
        self.config = config
        self.dtype = dtype
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

    # ---- 张量登记 ---------------------------------------------------
    def tid(self, t: torch.Tensor) -> str:
        key = id(t)
        if key not in self._tensor_ids:
            self._id_counter += 1
            tid = f"t{self._id_counter:04d}"
            self._tensor_ids[key] = tid
            self._tensors[tid] = {
                "shape": list(t.shape),
                "dtype": _dtype_str(t),
                "storage": hex(t.untyped_storage().data_ptr()),
                "offset": int(t.storage_offset()),
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
                        self._note_consumed(current, self.tid(t), None, d.seq)
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
                if tid_in in cluster.produced:
                    # 本簇自己的产出又被本簇的下一个算子吃掉：它还是中间量。
                    cluster.internal_out.add(tid_in)
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
                cluster.out_tensors.append(tid)
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

    # ---- 主流程 -----------------------------------------------------
    def build(self, phase: str = "prefill") -> list[Node]:
        self._cluster_dispatches()
        merged = self._merged_clusters()

        # 合并后重新解析 src 指向
        def resolve_src(s):
            if s[0] == "node":
                return ("node", self._uf.find(s[1]))
            return s

        cluster_list = list(merged.values())
        named = self._assign_ids(cluster_list)
        self.id_of_cid = {cl.cid: nid for nid, cl in named}

        nodes: list[Node] = []
        for nid, cl in named:
            module = self._module(cl.path)
            op, confidence = classify_ops(cl.ops, module, self.config)
            state_storages = self._own_state_storages(module)
            in_refs, outs = self._node_io(cl, state_storages, resolve_src)
            nodes.append(
                self._make_node(nid, cl, op, confidence, module, in_refs, outs, phase)
            )
        return nodes

    # ---- 输入/输出：集合差，不是边扫边丢 -----------------------------
    def _node_io(self, cl: "_Cluster", state_storages: set[str], resolve_src):
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
            if tid in cl.produced:
                internal.add(tid)
                continue
            if m["storage"] in state_storages:
                state.add(tid)
                continue
            seq = produced_storage_seq.get(m["storage"])
            if seq is not None and seq < cl.consumed_at.get(tid, 1 << 60):
                # 同一块存储上更早时刻产出的东西：视图、切片、原地写。
                internal.add(tid)
                continue
            external[tid] = slot or {}

        # 同一块存储的多个别名只算一个输入（展示层合并，不影响上面的分类）。
        groups: dict[str, list[str]] = {}
        for tid in external:
            groups.setdefault(self._tensors[tid]["storage"], []).append(tid)

        in_refs = []
        for storage, tids in groups.items():
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

        seen_out: set[tuple] = set()
        outs: list[str] = []
        for tid in dict.fromkeys(cl.out_tensors):
            if tid in cl.internal_out:
                continue
            m = self._tensors[tid]
            key = (m["storage"], m["offset"], m["numel"])
            if key in seen_out:
                continue
            seen_out.add(key)
            outs.append(tid)
        return in_refs, outs

    def _producer_src(self, tid: str, resolve_src):
        """这个张量是谁产的：先按对象身份，再按存储块（视图跨模块时用）。"""
        obj = self._tensors[tid]["obj"]
        src = self._src.get(id(obj))
        if src is None:
            src = self._storage_src.get(self._tensors[tid]["storage"], ("ext", tid))
        return resolve_src(src)

    def tensor_meta(self) -> dict:
        return {tid: {k: v for k, v in m.items() if k != "obj"} for tid, m in self._tensors.items()}

    def _make_node(self, nid, cl, op, confidence, module, in_refs, outs, phase) -> Node:
        attrs = {
            "dispatch_ops": _op_histogram(cl.ops),
            "dispatch_count": len(cl.ops),
            "module_path": cl.path or "R",
            "module_class": type(module).__name__ if module is not None else None,
            "seq": [cl.min_seq, cl.max_seq],
        }
        state: list[StateRef] = []
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
            Evidence("observed", _shape_summary(self._tensors, in_refs, outs)),
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
                )
            )
        outputs = [TensorRef(tensor=tid, role="out", index=i) for i, tid in enumerate(outs)]

        node = Node(
            id=nid,
            op=op,
            op_confidence=confidence,
            origin="module",
            phase=phase,
            inputs=inputs,
            outputs=outputs,
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
        node._start_seq = cl.min_seq  # type: ignore[attr-defined]
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


def _shape_summary(tensors: dict, in_refs, outs) -> str:
    def fmt(tid):
        meta = tensors.get(tid)
        if not meta:
            return "?"
        return "x".join(str(x) for x in meta["shape"]) + f" {meta['dtype']}"

    ins = ", ".join(fmt(tid) for tid, _, _, _ in in_refs[:4])
    os_ = ", ".join(fmt(tid) for tid in outs[:4])
    return f"输入 [{ins}] → 输出 [{os_}]"


def classify_ops(ops: list[str], module: nn.Module | None, config) -> tuple[str, str]:
    """op 标签由"实测到的 aten 算子"决定，不看模块名。

    名字只用于最后一道兜底（并且要求同时有形状/属性证据）。
    """
    o = set(ops)
    if any("scaled_dot_product" in x or "softmax" in x for x in o):
        return "attention", "exact"
    if "aten.embedding.default" in o:
        return "embedding", "exact"
    if o & {"aten.mm.default", "aten.addmm.default", "aten.baddbmm.default", "aten.matmul.default"}:
        vocab = getattr(config, "vocab_size", None)
        if isinstance(module, nn.Linear) and vocab and module.out_features == vocab:
            return "lm_head", "exact"
        return "matmul", "exact"
    if "aten.rsqrt.default" in o and ("aten.mean.dim" in o or "aten.mean.default" in o):
        return "rmsnorm", "exact"
    if "aten.silu.default" in o:
        return "silu", "exact"
    if any("gelu" in x for x in o):
        return "gelu", "exact"
    if "aten.cos.default" in o and "aten.sin.default" in o:
        # 这一簇只算 cos/sin 表（Qwen3RotaryEmbedding），不旋转任何张量。
        # 旋转（q*cos + rotate_half(q)*sin）发生在 attention 簇内部。
        return "rope_table", "exact"
    if "aten.mul.Tensor" in o and "aten.add.Tensor" in o and "aten.cat.default" in o:
        # rope 展开后的乘法/加法/拼接群
        return "rope", "inferred"
    if "aten.mul.Tensor" in o and len(o) == 1:
        return "mul", "exact"
    if "aten.add.Tensor" in o and len(o) == 1:
        return "add", "exact"
    if any("arange" in x for x in o):
        return "arange", "exact"
    return "unknown", "unknown"
