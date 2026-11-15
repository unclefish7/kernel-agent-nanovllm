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
    cid: int
    path: str
    call_seq: int
    ops: list = field(default_factory=list)
    in_tensors: list = field(default_factory=list)  # 去重后的 tensor id
    out_tensors: list = field(default_factory=list)
    in_srcs: dict = field(default_factory=dict)  # tid -> src tuple
    # tid -> {"path": "args[0]", "arg_name": "hidden_states"}：输入摆在参数的第几位
    in_slots: dict = field(default_factory=dict)
    internal: set = field(default_factory=set)  # 簇内产生又被簇内消费的中间张量
    min_seq: int = 1 << 60
    max_seq: int = -1
    call: object = None  # 产生这个簇的模块调用（decompose 用）

    def merge(self, other: "_Cluster") -> None:
        self.ops.extend(other.ops)
        self.in_tensors.extend(other.in_tensors)
        self.out_tensors.extend(other.out_tensors)
        self.in_srcs.update(other.in_srcs)
        self.in_slots.update(other.in_slots)
        self.internal |= other.internal
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
            for t in d.inputs:
                tid = self.tid(t)
                if tid in cluster.in_tensors:
                    continue
                src = self._src.get(id(t), ("ext", tid))
                if src == ("node", cluster.cid):
                    # 这个中间张量是本簇自己刚算出来的：既不是节点的输入，
                    # 也不该算节点的输出（它没有离开这个节点）。
                    cluster.internal.add(tid)
                    continue
                cluster.in_tensors.append(tid)
                cluster.in_srcs[tid] = src
                slot = None
                if call is not None:
                    slot = call.input_slots.get(id(t))
                    if slot is None:
                        # 模块内部先 view 再算（`F.linear` 会把 3D 输入摊成 2D），
                        # 对象换了但 storage 没换，靠 storage 把参数位找回来。
                        slot = call.input_storage_slots.get(
                            hex(t.untyped_storage().data_ptr())
                        )
                cluster.in_slots[tid] = slot or _aten_slot(d, t)
            for t in d.outputs:
                tid = self.tid(t)
                cluster.out_tensors.append(tid)
                self._src[id(t)] = ("node", cluster.cid)

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
            in_refs, outs = [], []
            state_storages = self._own_state_storages(module)
            seen_in: set[tuple] = set()
            for tid in cl.in_tensors:
                # 权重/buffer 不是数据流输入，它们已经记在 node.state 里。
                # 不排掉的话，图上会多出几百个"输入节点"，每个权重都冒充一次图输入。
                src = resolve_src(cl.in_srcs.get(tid, ("ext", tid)))
                if src[0] == "ext" and self._tensors[tid]["storage"] in state_storages:
                    continue
                # 同一个"生产者 + 同一块存储"的别名（view / transpose / slice）只算
                # 一个输入：rotate_half(q) 会把 q 切两半，那不 q 又吃了两次。
                # 但 cos 与 sin 虽然同一个生产者，存储不同，必须各算一个——
                # 只按生产者去重会把 sin 整条边吞掉。
                m_in = self._tensors[tid]
                key = (src, m_in["storage"])
                if key in seen_in:
                    continue
                seen_in.add(key)
                in_refs.append((tid, src, "input"))
            seen_out: set[tuple] = set()
            for tid in dict.fromkeys(cl.out_tensors):
                if tid in cl.internal:
                    continue
                m = self._tensors[tid]
                key = (m["storage"], m["offset"], m["numel"])
                if key in seen_out:
                    continue
                seen_out.add(key)
                outs.append(tid)
            nodes.append(
                self._make_node(nid, cl, op, confidence, module, in_refs, outs, phase)
            )
        return nodes

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
        for i, (tid, _, _) in enumerate(in_refs):
            slot = cl.in_slots.get(tid) or {}
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
        node._in_srcs = {tid: s for tid, s, _ in in_refs}  # type: ignore[attr-defined]
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

    ins = ", ".join(fmt(tid) for tid, _, _ in in_refs[:4])
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
