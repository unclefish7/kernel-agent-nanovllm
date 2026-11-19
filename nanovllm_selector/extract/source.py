"""源码对应：把每个模块类映射回"文件 + 行号 + 文件 sha256"。

这是"不能只基于名称"的证据链里最基础的一环：
看到 `Qwen3MLP` 这个名字不算证据，知道它定义在哪个文件的第几行、
而且那个文件的哈希是多少，才算可复查。
"""

from __future__ import annotations

import hashlib
import inspect
import os
from functools import lru_cache

import torch.nn as nn


@lru_cache(maxsize=512)
def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def source_ref(module: nn.Module) -> dict:
    """返回 {class, qualname, file, class_line, sha256}。

    `file` 可能为 None（C 扩展、动态生成的类），这时调用方应补一条
    `{"kind": "rule", "detail": "builtin"}` 证据。
    """
    return _class_source(type(module))


@lru_cache(maxsize=512)
def _class_source(cls: type) -> dict:
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:
        path = None
    line = None
    if path is not None:
        try:
            line = inspect.getsourcelines(cls)[1]
        except (OSError, TypeError):
            line = None
    return {
        "class": cls.__name__,
        "qualname": f"{cls.__module__}.{cls.__qualname__}",
        "file": os.path.realpath(path) if path else None,
        "class_line": line,
        "sha256": sha256_file(os.path.realpath(path)) if path else None,
    }
