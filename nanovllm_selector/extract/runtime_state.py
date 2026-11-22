"""通过实际容器中的张量和前向轨迹记录跨步状态。"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from types import MemberDescriptorType
from typing import NamedTuple

import torch

from .edges import tensor_view


class StateKey(NamedTuple):
    layer: int | None
    role: str
    path: str


def _leaves(value, path=(), parents=()):
    if isinstance(value, torch.Tensor):
        if value.numel():
            yield path, value
        return
    if id(value) in parents or isinstance(value, (type, torch.nn.Module)):
        return
    if isinstance(value, Mapping):
        children = value.items()
    elif isinstance(value, (list, tuple)):
        children = enumerate(value)
    elif callable(value):
        return
    else:
        fields = dict(getattr(value, "__dict__", {}))
        for cls in type(value).__mro__:
            for key, descriptor in vars(cls).items():
                if isinstance(descriptor, MemberDescriptorType) and key not in fields and hasattr(value, key):
                    fields[key] = getattr(value, key)
        children = fields.items()
    for key, child in children:
        yield from _leaves(child, (*path, key), (*parents, id(value)))


def _identity(path):
    layer = next((v for v in path if isinstance(v, int)), None)
    fields = [str(v) for v in path if not isinstance(v, int)]
    field = fields[-1] if fields else "state"
    role = {"keys": "key", "values": "value", "key_cache": "key", "value_cache": "value"}.get(field, field)
    display = "".join(f"[{v}]" if isinstance(v, int) else "." + str(v) for v in path)
    return StateKey(layer, role, display.lstrip("."))


class StateTensors(dict):
    def __init__(self, cache, buffers=()):
        super().__init__()
        self.aliases = {}
        seen = {}
        for path, tensor in chain(_leaves(cache), buffers):
            identity = tensor_view(tensor)
            key = _identity(path)
            if identity in seen:
                self.aliases[seen[identity]].append(key.path)
            else:
                self[key] = tensor
                self.aliases[key] = []
                seen[identity] = key


@contextmanager
def restore_buffers(model):
    """每组输入从相同的模型状态出发，包括被重新赋值或原地修改的 buffer。"""
    bindings = [(module, dict(module._buffers)) for module in model.modules()]
    initial = [(t, t.detach().clone()) for t in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for tensor, original in initial:
                if not torch.equal(tensor, original):
                    if tensor.shape != original.shape:
                        tensor.resize_(original.shape)
                    tensor.copy_(original)
            for module, old in bindings:
                module._buffers.update(old)
                for key in tuple(module._buffers):
                    if key not in old:
                        del module._buffers[key]


def written_buffers(model, recording, old, active):
    # 持久缓冲区也可以跨步携带状态。仅纳入本次执行真正写入的缓冲区。
    modified = set()
    for op in recording.dispatches:
        if op.func and op.func._schema.is_mutable:
            modified.update(t.untyped_storage().data_ptr() for t in op.outputs)
    for name, value in model.named_buffers():
        if value is not old.get(name) or value.untyped_storage().data_ptr() in modified:
            active.add(name)
        if value.numel() and name in active:
            yield ("buffers", *(int(x) if x.isdigit() else x for x in name.split("."))), value


@dataclass
class PreviousState:
    key: StateKey
    producer: str
    slot: int
    tensor: torch.Tensor
    before: torch.Tensor

    def update(self, new):
        before = self.before
        if new.shape != before.shape:
            return {"kind": "resized", "previous_shape": list(before.shape)}
        if torch.equal(before, new):
            return {"kind": "unchanged"}
        for axis, size in enumerate(new.shape):
            if size > 1 and torch.equal(before.narrow(axis, 1, size - 1), new.narrow(axis, 0, size - 1)):
                return {"kind": "shift", "axis": axis, "retained_columns": size - 1}
        return {"kind": "updated"}
