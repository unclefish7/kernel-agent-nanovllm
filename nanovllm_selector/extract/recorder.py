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


class _RecorderMode(TorchDispatchMode):
    def __init__(self, rec: "Recorder"):
        super().__init__()
        self.rec = rec

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins = flatten_tensors((args, kwargs))
        outs = flatten_tensors(out)
        rec = self.rec
        rec._keep.extend(ins)
        rec._keep.extend(outs)
        rec.dispatches.append(
            Dispatch(
                seq=rec._next_seq(),
                op=str(func),
                inputs=ins,
                outputs=outs,
                inside=rec._stack[-1] if rec._stack else "",
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
        self._leaf_paths: set[str] = set()

    # ---- 序号 -------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ---- hooks ------------------------------------------------------
    def _make_pre(self, path: str, cls: str, qualname: str, depth: int):
        def pre(module, args, kwargs):
            seq = self._next_seq()
            ins = flatten_tensors((args, kwargs))
            self._keep.extend(ins)
            call = Call(
                seq_start=seq,
                seq_end=-1,
                path=path,
                cls=cls,
                qualname=qualname,
                depth=depth,
                inputs=ins,
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


def record_forward(model, input_ids: torch.Tensor) -> tuple[Recorder, object]:
    """跑一次 forward，返回 recorder 和模型输出。"""
    rec = Recorder(model)
    with torch.no_grad():
        with rec:
            out = model(input_ids=input_ids, use_cache=True)
    return rec, out
