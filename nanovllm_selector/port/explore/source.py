"""从本地模型配置、checkpoint 元数据和已安装的 Python 源码回答 read_code 查询。

ref 是模块实例路径或包内全限定名。tree 与 weights 的 items 按条目分页，source
按原文件的行分页。offset 从零开始，line、start_line 和 end_line 从一开始。
source_name 对应解析后的符号，sha256 对应完整文件。树中的参数来自 meta 实例；
weights 优先报告 checkpoint 原始键和 dtype，只有配置时才返回推导出的参数。
只支持 Transformers 原生架构，模型目录里的自定义 Python 不参与符号解析。
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import keyword
import sys
import tokenize
from dataclasses import dataclass
from functools import lru_cache, partial
from importlib import import_module
from importlib.util import find_spec, resolve_name
from pathlib import Path


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITIONS = (*_FUNCTIONS, ast.ClassDef)


def _page(values, offset, limit):
    end = min(offset + limit, len(values))
    info = {"offset": offset, "total": len(values), "next_offset": end if end < len(values) else None}
    return values[offset:end], info


def _in_scope(name, prefixes):
    return any(not prefix or name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _model_file(root, name, suffix):
    relative = Path(name)
    if (relative.is_absolute() or not relative.parts or not name.endswith(suffix)
            or any(part.startswith(".") for part in relative.parts)):
        raise ValueError(f"模型元数据路径无效：{name!r}")
    path = (root / relative).resolve(strict=True)
    if path.is_relative_to(root) and path.name.endswith(suffix) and path.is_file():
        if not any(part.startswith(".") for part in path.relative_to(root).parts):
            return path
    # Hugging Face 本地快照使用指向同一仓库 blobs/<摘要> 的链接。
    if (root.parent.name == "snapshots" and path.parent == root.parent.parent / "blobs"
            and len(path.name) in (40, 64) and all(c in "0123456789abcdef" for c in path.name)
            and path.is_file()):
        return path
    raise ValueError(f"模型元数据链接目标无效：{name!r}")


def _config(root):
    path = _model_file(root, "config.json", ".json")
    raw = path.read_bytes()
    values = json.loads(raw)
    if values.get("configuration_files"):
        from transformers.configuration_utils import get_configuration_file

        name = get_configuration_file(values["configuration_files"])
        path = _model_file(root, name, ".json")
        raw = path.read_bytes()
        values = json.loads(raw)
    return path, values, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def _meta_model(config_file, config_hash):
    """同一配置在连续翻页时共用只含元数据的实例；文件内容变化时重新解析。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    raw = json.loads(config_file.read_bytes())
    remote = raw.get("auto_map") or {}
    try:
        config = AutoConfig.from_pretrained(str(config_file), local_files_only=True, trust_remote_code=False)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except (ValueError, KeyError) as exc:
        if remote:
            raise ValueError(f"无法用已安装的原生实现解析 {raw.get('model_type')!r}；"
                             f"不支持 trust_remote_code=True：{exc}") from exc
        raise
    custom = remote.get("AutoModelForCausalLM")
    if custom and type(model).__name__ != custom.rsplit(".", 1)[-1]:
        raise ValueError(f"配置指定了自定义实现 {custom}，原生类为 {type(model).__name__}；"
                         "不支持 trust_remote_code=True")
    return model.eval()


def _model(root):
    path, _, digest = _config(root)
    return _meta_model(path, digest)


def _checkpoint(root):
    """采用 Transformers 的默认单文件、分片索引优先级。额外格式只作最后的回退。"""
    for format_name, filename in (("safetensors", "model.safetensors"), ("pytorch", "pytorch_model.bin")):
        if (root / filename).is_file():
            return format_name, [filename], None
        index_name = filename + ".index.json"
        if (root / index_name).is_file():
            index = json.loads(_model_file(root, index_name, ".json").read_bytes())["weight_map"]
            if not isinstance(index, dict):
                raise ValueError(f"{index_name} 的 weight_map 无效")
            return format_name, [], index
    return "safetensors", sorted(path.name for path in root.glob("*.safetensors")), None


def _parameter(key, tensor):
    return {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def _shard_rows(root, shard, format_name, wanted, prefixes):
    if format_name == "safetensors":
        from safetensors import safe_open

        path = _model_file(root, shard, ".safetensors")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = handle.keys() if wanted is None else wanted
            for key in keys:
                if _in_scope(key, prefixes):
                    tensor = handle.get_slice(key)
                    yield {"key": key, "shape": tensor.get_shape(), "dtype": tensor.get_dtype(), "file": shard}
    else:
        import torch

        path = _model_file(root, shard, ".bin")
        state = torch.load(path, weights_only=True, map_location="meta", mmap=True)
        keys = state.keys() if wanted is None else wanted
        for key in keys:
            if _in_scope(key, prefixes):
                yield {**_parameter(key, state[key]), "file": shard}


def _weights(root, prefixes, offset, limit):
    """有索引时只打开当前页涉及的分片。两种格式都无需物化张量值。"""
    format_name, files, index = _checkpoint(root)
    if index is None and not files:
        return None
    shards = dict.fromkeys(files)
    if index is not None:
        keys = sorted(key for key in index if _in_scope(key, prefixes))
        selected, info = _page(keys, offset, limit)
        for key in selected:
            shards.setdefault(index[key], []).append(key)

    rows = {}
    for shard, wanted in shards.items():
        for row in _shard_rows(root, shard, format_name, wanted, prefixes):
            key = row["key"]
            if key in rows:
                raise ValueError(f"多个 checkpoint 文件含有同名权重 {key!r}")
            rows[key] = row
    if index is None:
        selected, info = _page(sorted(rows), offset, limit)
    return {"provenance": format_name, "items": [rows[key] for key in selected], **info}


def _qualified(obj):
    return obj.__module__ + "." + obj.__qualname__


def _package_roots(target):
    names = ("transformers", "torch") if target == "hf" else ("nanovllm",)
    roots = {}
    for name in names:
        spec = find_spec(name)
        if spec and spec.submodule_search_locations:
            roots[name] = Path(next(iter(spec.submodule_search_locations))).resolve()
        elif name == "nanovllm":
            checkout = Path(__file__).resolve().parents[3] / "nano-vllm" / name
            if checkout.is_dir():
                roots[name] = checkout.resolve()
    return roots


def _identifier(name):
    return name.isidentifier() and not keyword.iskeyword(name)


def _trusted_path(path, root):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.suffix != ".py" or not resolved.is_file():
        raise ValueError("只能读取所选 Python 包内的 .py 源码")
    return resolved


@lru_cache(maxsize=16)
def _parse(path, mtime_ns, size):
    data = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    lines = data.decode(encoding).splitlines(keepends=True)
    return ast.parse(data, str(path)), lines, hashlib.sha256(data).hexdigest()


def _document(path):
    stat = path.stat()
    return _parse(path, stat.st_mtime_ns, stat.st_size)


@dataclass
class _Symbol:
    name: str
    module: str
    path: Path
    node: ast.AST | None  # 命名空间包没有 __init__.py
    runtime: bool = False


def _members(symbol, roots):
    """枚举静态声明和允许继续追踪的导入，保留局部名字与原始全限定名。"""
    package = symbol.module if symbol.path.name == "__init__.py" else symbol.module.rpartition(".")[0]
    for node in getattr(symbol.node, "body", []):
        if isinstance(node, _DEFINITIONS):
            yield node.name, symbol.name + "." + node.name, node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    yield target.id, symbol.name + "." + target.id, node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            base = ""
            if isinstance(node, ast.ImportFrom):
                base = resolve_name("." * node.level + (node.module or ""), package) + "."
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                ref = base + alias.name if base or alias.asname else name
                if name != "*" and ref.split(".")[0] in roots:
                    yield name, ref, None


def _module_symbol(ref, roots):
    parts = ref.split(".")
    root = roots.get(parts[0])
    if root is None or not all(_identifier(part) for part in parts):
        raise ValueError(f"无效的包内 Python 引用：{ref!r}")
    for count in range(len(parts), 0, -1):
        path = root.joinpath(*parts[1:count])
        module = ".".join(parts[:count])
        if path.is_dir():
            if not path.resolve().is_relative_to(root):
                raise ValueError("包目录超出源码范围")
            if not (path / "__init__.py").is_file():
                return _Symbol(module, module, path, None), parts[count:]
            path = path / "__init__.py"
        elif count > 1 and path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        else:
            continue
        path = _trusted_path(path, root)
        return _Symbol(module, module, path, _document(path)[0]), parts[count:]
    raise ValueError(f"找不到安装包源码：{ref}")


def _loaded_object(ref, roots):
    """只访问已导入模块的名字表和类的静态描述符，避免按用户输入执行 import。"""
    parts = ref.split(".")
    for count in range(len(parts), 0, -1):
        obj = sys.modules.get(".".join(parts[:count]))
        if obj is None:
            continue
        for part in parts[count:]:
            if inspect.ismodule(obj) and obj.__name__.split(".")[0] in roots:
                obj = vars(obj).get(part)
            elif inspect.isclass(obj) and obj.__module__.split(".")[0] in roots:
                obj = inspect.getattr_static(obj, part, None)
            else:
                return None
        return obj
    return None


def _first_line(node):
    return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])


def _object_symbol(obj, roots):
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        obj = obj.fget
    if isinstance(obj, partial):
        obj = obj.func
    obj = inspect.unwrap(obj)
    if inspect.ismodule(obj):
        return _module_symbol(obj.__name__, roots)[0]
    if not (inspect.isroutine(obj) or inspect.isclass(obj)):
        raise ValueError("该属性不是可读取源码的 Python 类、函数或方法")
    name = _qualified(obj)
    root = roots.get(name.split(".")[0])
    if root is None:
        raise ValueError(f"符号位于目标包之外：{name}")
    filename = inspect.getsourcefile(obj)
    if not filename:
        raise ValueError(f"{name} 没有可读取的 Python 源码")
    path = _trusted_path(Path(filename), root)
    line = inspect.getsourcelines(obj)[1]
    for node in ast.walk(_document(path)[0]):
        if isinstance(node, _DEFINITIONS) and node.name == obj.__name__ and _first_line(node) == line:
            return _Symbol(name, obj.__module__, path, node, runtime=True)
    raise ValueError(f"无法定位运行时定义 {name} 的源码行")


def _resolve(ref, roots, seen=frozenset()):
    if ref in seen:
        raise ValueError(f"循环的 Python 引用：{ref}")
    module, parts = _module_symbol(ref, roots)
    package_name = ref.split(".")[0]
    if package_name in ("transformers", "torch"):
        # 只初始化固定的原生包；具体子模块由源码解析，不按用户给出的名字导入。
        package = import_module(package_name)
        if parts and package_name == "transformers":
            exported = vars(package).get("_class_to_module", {}).get(parts[0])
            if exported:
                defining_module = package_name + "." + exported
                if defining_module.startswith(module.name + "."):
                    return _resolve(".".join([defining_module, *parts]), roots, seen | {ref})
    obj = _loaded_object(ref, roots)
    if inspect.isroutine(obj) or inspect.isclass(obj) or isinstance(obj, (property, staticmethod, classmethod)):
        return _object_symbol(obj, roots)
    if inspect.ismodule(obj) and obj.__name__ != ref:
        return _module_symbol(obj.__name__, roots)[0]
    symbol = module
    for index, part in enumerate(parts):
        members = {name: (name_ref, node) for name, name_ref, node in _members(symbol, roots)}
        tail = ".".join(parts[index:])
        if part not in members:
            if isinstance(symbol.node, ast.ClassDef):
                visible = {name: name_ref for name, name_ref, _ in _members(module, roots)}
                for base in symbol.node.bases:
                    parent, dot, attr = ast.unparse(base).partition(".")
                    if parent in visible:
                        try:
                            return _resolve(visible[parent] + dot + attr + "." + tail, roots, seen | {ref})
                        except LookupError:
                            continue
            raise LookupError(f"{symbol.name} 没有可读取的符号 {part!r}")
        name_ref, node = members[part]
        if node is None:
            return _resolve(".".join([name_ref, *parts[index + 1:]]), roots, seen | {ref})
        symbol = _Symbol(name_ref, module.name, module.path, node)
    return symbol


def _entry(name, ref, node):
    result = {"name": name, "ref": ref}
    if isinstance(node, ast.ClassDef):
        result["type"] = "class"
        node = next((n for n in node.body if isinstance(n, _FUNCTIONS) and n.name == "__init__"), None)
    elif isinstance(node, _FUNCTIONS):
        result["type"] = "function"
    else:
        result["type"] = "import" if node is None else "attribute"
    if isinstance(node, _FUNCTIONS):
        result["signature"] = "(" + ast.unparse(node.args) + ")"
        if node.returns:
            result["signature"] += " -> " + ast.unparse(node.returns)
    return result


def _symbol_view(symbol, roots, view, offset, limit):
    provenance = "installed_source"
    if "nanovllm" in roots and find_spec("nanovllm") is None:
        provenance = "checkout_source"
    location = {"provenance": provenance, "source_name": symbol.name,
                "resolution": "runtime_binding" if symbol.runtime else "static_definition"}
    if view == "tree":
        items = []
        if symbol.node is None or symbol.path.name == "__init__.py" and isinstance(symbol.node, ast.Module):
            directory = symbol.path if symbol.node is None else symbol.path.parent
            for path in sorted(directory.rglob("*.py")):
                parts = path.relative_to(directory).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts and all(_identifier(part) for part in parts) and path.resolve().is_relative_to(directory):
                    items.append({"ref": ".".join((symbol.name, *parts)), "type": "module"})
        items.extend(_entry(*member) for member in _members(symbol, roots))
        entries, info = _page(items, offset, limit)
        return {**location, "items": entries, **info}

    if symbol.node is None:
        raise ValueError(f"{symbol.name} 是命名空间包，请用 tree 查询其中的模块")
    tree, lines, digest = _document(symbol.path)
    node = symbol.node
    first, last = (1, len(lines)) if isinstance(node, ast.Module) else (_first_line(node), node.end_lineno)
    visible, info = _page(lines[first - 1:last], offset, limit)
    module = _Symbol(symbol.module, symbol.module, symbol.path, tree)
    names = (n for n in ast.walk(node) if isinstance(n, ast.Name))
    if isinstance(node, ast.Module):
        names = (n for n in names if first + offset <= n.lineno < first + offset + len(visible))
    used = {n.id for n in names}
    related = [_entry(*member) for member in _members(module, roots) if member[0] in used]
    definition = _entry(symbol.name.rsplit(".", 1)[-1], symbol.name, node)
    return {**location, "module": symbol.module,
            "signature": definition.get("signature"),
            "file": str(symbol.path), "sha256": digest, "line": first,
            "start_line": first + offset if visible else None,
            "end_line": first + offset + len(visible) - 1 if visible else None,
            "source": "".join(visible), "related": related, **info}


def _query(root, target, ref, view, method, offset, limit):
    roots = _package_roots(target)
    qualified = ref.split(".")[0] in roots
    if view == "config":
        if ref:
            raise ValueError("config 视图使用根引用")
        if target == "engine":
            symbol = _resolve("nanovllm.config.Config", roots)
            return _symbol_view(symbol, roots, "source", offset, limit)
        path, config, digest = _config(root)
        return {"provenance": "config.json", "file": str(path), "sha256": digest, "config": config}

    if view == "weights":
        prefixes = [ref]
        if qualified:
            if target == "engine":
                raise ValueError("checkpoint 使用参数名或模块实例路径")
            name = _resolve(ref, roots).name
            modules = _model(root).named_modules(remove_duplicate=False)
            prefixes = [path for path, obj in modules if _qualified(type(obj)) == name]
            if not prefixes:
                raise ValueError(f"模型中没有 {name} 的实例")
        result = _weights(root, prefixes, offset, limit)
        if result is not None:
            return result
        if target == "engine":
            raise ValueError("模型目录中没有可读的 checkpoint 元数据")
        values = [(key, p) for key, p in _model(root).named_parameters(remove_duplicate=False)
                  if _in_scope(key, prefixes)]
        page, info = _page(values, offset, limit)
        rows = [_parameter(key, p) for key, p in page]
        return {"provenance": "meta_from_config", "items": rows, **info}

    if target == "engine" or qualified:
        if target == "engine" and not qualified:
            ref = "nanovllm." + ref if ref else "nanovllm"
        symbol = _resolve(ref + "." + method if method else ref, roots)
        return _symbol_view(symbol, roots, view, offset, limit)

    model = _model(root)
    obj = model
    if ref:
        for part in ref.split("."):
            obj = obj._modules.get(part)
            if obj is None:
                raise LookupError(f"模型没有已注册的模块 {ref!r}")
    model_info = {"model_class": _qualified(type(model)), "device": "meta", "trust_remote_code": False}
    if view == "source":
        bound = inspect.getattr_static(obj, method) if method else type(obj)
        return {**_symbol_view(_object_symbol(bound, roots), roots, view, offset, limit), **model_info}
    modules = list(obj.named_modules(prefix=ref, remove_duplicate=False))
    page, info = _page(modules, offset, limit)
    items = []
    for path, child in page:
        params = child.named_parameters(prefix=path, recurse=False, remove_duplicate=False)
        items.append({"ref": path, "type": _qualified(type(child)),
                      "parameters": [_parameter(key, p) for key, p in params]})
    return {"provenance": "meta_from_config", "items": items, **model_info, **info}


def read_code(model_dir: Path, arguments: dict) -> dict:
    """读取已选模型和引擎的证据。参数或视图不可用时返回 ok=False 和原因。"""
    try:
        extra = arguments.keys() - {"target", "ref", "view", "method", "offset", "limit"}
        if extra:
            raise ValueError(f"未知参数：{sorted(extra)}")
        target, view = arguments.get("target", "hf"), arguments.get("view", "tree")
        ref, method = arguments.get("ref", ""), arguments.get("method")
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 80)
        if target not in ("hf", "engine") or view not in ("tree", "source", "config", "weights"):
            raise ValueError("target 应为 hf 或 engine，view 应为 tree、source、config 或 weights")
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("offset 应为非负整数，limit 应为正整数")
        if not isinstance(ref, str):
            raise ValueError("ref 应为点分的模块实例路径或 Python 符号")
        parts = ref.split(".") if ref else []
        if any(not part or not all(c.isalnum() or c in "_-" for c in part) for part in parts):
            raise ValueError("ref 应为点分的模块实例路径或 Python 符号")
        if any(part.startswith("__") for part in parts[:-1]):
            raise ValueError("ref 不能遍历 Python 的内部属性")
        if method is not None and (view != "source" or not isinstance(method, str) or not _identifier(method)):
            raise ValueError("method 应为 source 视图中的方法名")
        result = _query(Path(model_dir).resolve(), target, ref, view, method, offset, limit)
        return {"ok": True, "target": target, "ref": ref, "view": view, **result}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
