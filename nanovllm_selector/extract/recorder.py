"""探针：模块 hook 定边界，TorchDispatchMode 定连接。

为什么要两个一起用：

* 只挂模块 hook，能知道"有哪些模块被调用了"，但不知道模块之间隔着什么运算。
  实测 Qwen3 会因此漏掉 86/397 条边，漏的正是 `o_proj`、`post_attention_layernorm`、
  `down_proj` 这些"输入被父模块里的张量运算改过"的模块。
* 只用 TorchDispatchMode，能知道"有哪些 aten 算子"，但 2664 个算子没有边界，
  没法对应回"这是第几层的哪个模块"。

两者用同一个单调递增的 seq 串起来，才能既知道边界、又知道连接。
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import torch
from dataclasses import dataclass, field
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass
class Call:
    seq_start: int
    seq_end: int
    path: str
    cls: str
    qualname: str
    depth: int
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    # id(tensor) -> {"path": "args[0]", "arg_name": "hidden_states"}
    # 记"这个张量摆在这次调用的第几个参数位"，见 schema.TensorRef 的说明。
    input_slots: dict = field(default_factory=dict)
    # 按 storage 索引的同一份信息：模块内部做了 view/reshape 之后，张量对象变了
    # 但仍是同一个参数的内容（`F.linear` 会先把 3D 输入 view 成 2D），靠它补回位置。
    input_storage_slots: dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return False  # 由 Recorder 事后填充


@dataclass
class Dispatch:
    seq: int
    op: str
    inputs: list
    outputs: list
    inside: str
    # aten 算子本身，留着取参数名（`_schema.arguments[i].name`）
    func: object = None
    # id(tensor) -> {"path": "args[0][1]", "arg_name": "tensors"}，口径同 Call.input_slots
    input_slots: dict = field(default_factory=dict)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    result: object = None
    writes: list = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


class _Snapshots:
    """同一 storage 的各种视图共用一份快照，原地写后建立下一个版本。"""

    def __init__(self):
        self.storage = {}

    def capture(self, tensors):
        result = {}
        for t in tensors:
            storage = t.untyped_storage()
            key = storage._cdata
            if key not in self.storage:
                self.storage[key] = storage.clone()
            result[id(t)] = torch.empty(0, dtype=t.dtype, device=t.device).set_(
                self.storage[key], t.storage_offset(), t.shape, t.stride())
        return result

    def invalidate(self, tensors):
        for t in tensors:
            self.storage.pop(t.untyped_storage()._cdata, None)


def flatten_tensors(obj, out: list | None = None) -> list:
    """把 forward 的输入/输出（可能嵌套）拍平成张量列表。"""
    if out is None:
        out = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            flatten_tensors(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            flatten_tensors(x, out)
    return out


def flatten_with_paths(obj, path: str, out: list | None = None) -> list:
    """拍平张量，同时保留位置表达式，如 `args[1][2]`、`kwargs['hidden_states']`。

    返回 `[(tensor, path), ...]`，顺序与递归顺序一致。
    """
    if out is None:
        out = []
    if isinstance(obj, torch.Tensor):
        out.append((obj, path))
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            flatten_with_paths(x, f"{path}[{i}]", out)
    elif isinstance(obj, dict):
        for k, x in obj.items():
            flatten_with_paths(x, f"{path}[{k!r}]", out)
    return out


@lru_cache(maxsize=512)
def _forward_param_names(cls: type) -> tuple[str, ...]:
    """模块 forward 的形参名（去掉 self，只保留可接收位置参数的）。"""
    try:
        params = inspect.signature(cls.forward).parameters.values()
    except (ValueError, TypeError):
        return ()
    names = []
    for p in params:
        if p.name == "self":
            continue
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
    return tuple(names)


def _slots(names, args, kwargs) -> dict:
    """按形参名把 `args[i]` / `kwargs['name']` 里的张量标上位置。"""
    slots: dict[int, dict] = {}
    for i, a in enumerate(args):
        arg_name = names[i] if i < len(names) else None
        for t, p in flatten_with_paths(a, f"args[{i}]"):
            slots.setdefault(id(t), {"path": p, "arg_name": arg_name})
    for k, v in (kwargs or {}).items():
        for t, p in flatten_with_paths(v, f"kwargs[{k!r}]"):
            slots.setdefault(id(t), {"path": p, "arg_name": k})
    return slots


def call_input_slots(module, args, kwargs) -> tuple[dict, dict]:
    """记录一次模块调用里每个张量出现在哪个参数位。

    返回 `(按对象索引, 按 storage 索引)`。位置优先取最外层的
    `args[i]` / `kwargs['name']`，嵌套结构继续往下标；形参名取最外层的那个
    （`args[1][0]` 的 arg_name 仍是 `args` 的第 1 个形参）。
    """
    slots = _slots(_forward_param_names(type(module)), args, kwargs)
    by_storage = {}
    for t, p in flatten_with_paths(args, "args"):
        by_storage.setdefault(hex(t.untyped_storage().data_ptr()),
                              slots.get(id(t), {"path": p, "arg_name": None}))
    for k, v in (kwargs or {}).items():
        for t, p in flatten_with_paths(v, f"kwargs[{k!r}]"):
            by_storage.setdefault(hex(t.untyped_storage().data_ptr()),
                                  slots.get(id(t), {"path": p, "arg_name": k}))
    return slots, by_storage


def _aten_param_names(func) -> tuple[str, ...]:
    try:
        return tuple(a.name for a in func._schema.arguments)
    except Exception:
        return ()


def dispatch_input_slots(func, args, kwargs) -> dict:
    return _slots(_aten_param_names(func), args, kwargs)


class _RecorderMode(TorchDispatchMode):
    def __init__(self, rec: "Recorder"):
        super().__init__()
        self.rec = rec

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        slots = dispatch_input_slots(func, args, kwargs)
        ins = flatten_tensors((args, kwargs))
        writes = []
        for i, param in enumerate(func._schema.arguments):
            if param.alias_info is not None and param.alias_info.is_write:
                flatten_tensors(args[i] if i < len(args) else kwargs.get(param.name), writes)
        before = self.rec._snapshots.capture(ins)
        out = func(*args, **kwargs)
        outs = flatten_tensors(out)
        rec = self.rec
        rec._snapshots.invalidate(writes)
        after = rec._snapshots.capture(outs + writes)
        rec._keep.extend(ins)
        rec._keep.extend(outs)
        rec.dispatches.append(
            Dispatch(
                seq=rec._next_seq(),
                op=str(func),
                inputs=ins,
                outputs=outs,
                inside=rec._stack[-1] if rec._stack else "",
                func=func,
                input_slots=slots,
                args=args, kwargs=kwargs, result=out, writes=writes, before=before, after=after,
            )
        )
        return out


class Recorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.calls: list[Call] = []
        self.dispatches: list[Dispatch] = []
        self._seq = 0
        self._stack: list[str] = []
        self._handles: list = []
        self._keep: list = []
        self._open: list[Call] = []
        self._mode = _RecorderMode(self)
        self._snapshots = _Snapshots()
        self._leaf_paths: set[str] = set()

    # ---- 序号 -------------------------------------------------------
    def shift(self, offset: int) -> None:
        """把这次 trace 的序号整体后移，接在上一次后面。"""
        for c in self.calls:
            c.seq_start += offset
            c.seq_end += offset
        for d in self.dispatches:
            d.seq += offset
        self._seq += offset

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ---- hooks ------------------------------------------------------
    def _make_pre(self, path: str, cls: str, qualname: str, depth: int):
        def pre(module, args, kwargs):
            seq = self._next_seq()
            ins = flatten_tensors((args, kwargs))
            self._keep.extend(ins)
            by_obj, by_storage = call_input_slots(module, args, kwargs)
            call = Call(
                seq_start=seq,
                seq_end=-1,
                path=path,
                cls=cls,
                qualname=qualname,
                depth=depth,
                inputs=ins,
                input_slots=by_obj,
                input_storage_slots=by_storage,
            )
            self.calls.append(call)
            self._open.append(call)
            self._stack.append(path)
            return None

        return pre

    def _make_post(self, path: str):
        def post(module, args, kwargs, output):
            # 顺序很重要：先弹栈，再记结束序号
            if self._stack:
                self._stack.pop()
            if self._open:
                call = self._open.pop()
                call.seq_end = self._next_seq()
                outs = flatten_tensors(output)
                self._keep.extend(outs)
                call.outputs = outs
            return None

        return post

    # ---- 生命周期 ---------------------------------------------------
    def __enter__(self) -> "Recorder":
        depth_of = {}
        paths = {}
        for path, mod in self.model.named_modules():
            paths[id(mod)] = path
        # 计算每个模块的树深度
        for path in paths.values():
            depth_of[path] = 0 if path == "" else path.count(".") + 1
        children: dict[str, int] = {}
        for path in paths.values():
            if path:
                parent = path.rsplit(".", 1)[0] if "." in path else ""
                children[parent] = children.get(parent, 0) + 1

        for path, mod in self.model.named_modules():
            cls = type(mod)
            depth = depth_of.get(path, 0)
            self._handles.append(
                mod.register_forward_pre_hook(
                    self._make_pre(path, cls.__name__, f"{cls.__module__}.{cls.__qualname__}", depth),
                    with_kwargs=True,
                )
            )
            self._handles.append(mod.register_forward_hook(self._make_post(path), with_kwargs=True))
        # 记录哪些路径是叶子（没有子模块）
        all_paths = sorted(p for p in paths.values() if p)
        for p in all_paths:
            self._leaf_paths.add(p)
        for p in list(all_paths):
            if "." in p:
                parent = p.rsplit(".", 1)[0]
                self._leaf_paths.discard(parent)
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._mode.__exit__(exc_type, exc, tb)
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
        return False

    # ---- 统计 -------------------------------------------------------
    def leaf_paths(self) -> set[str]:
        return set(self._leaf_paths)

    def calls_by_path(self) -> dict[str, list[Call]]:
        out: dict[str, list[Call]] = {}
        for c in self.calls:
            out.setdefault(c.path, []).append(c)
        return out

    def dispatches_inside(self, path: str) -> list[Dispatch]:
        return [d for d in self.dispatches if d.inside == path]


def record_forward(
    model,
    input_ids: torch.Tensor,
    past_key_values=None,
    cache_position: torch.Tensor | None = None,
    seq_offset: int = 0,
) -> tuple[Recorder, object]:
    """跑一次 forward，返回 recorder 和模型输出。

    `past_key_values` 非空时走的就是 decode 分支；`seq_offset` 把这一次的
    算子序号接在上一次后面，两次 trace 才能放进同一个时间轴。
    """
    rec = Recorder(model)
    kw = {"input_ids": input_ids, "use_cache": True}
    if past_key_values is not None:
        kw["past_key_values"] = past_key_values
    if cache_position is not None:
        params = inspect.signature(model.forward).parameters.values()
        if any(p.kind == p.VAR_KEYWORD or (p.name == "cache_position" and
               p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)) for p in params):
            kw["cache_position"] = cache_position
    with torch.no_grad():
        with rec:
            out = model(**kw)
    if seq_offset:
        rec.shift(seq_offset)
    return rec, out
