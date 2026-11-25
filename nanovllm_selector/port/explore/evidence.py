"""前向边界上的快照、只读运行目录和完整数值比较。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from types import MemberDescriptorType

import torch
from safetensors import safe_open
from safetensors.torch import load, save


MAX_TENSOR_BYTES = 64 << 20
MAX_RUN_BYTES = 256 << 20
MAX_MANIFEST_BYTES = 8 << 20
MAX_ITEMS = 16384
MAX_STEPS = 65
MAX_TOKENS = 2048
MAX_WATCH = 64
MAX_PAGE = 256
SELECTOR_PAGE_SIZE = 8
INDEX_PAGE_BYTES = 8192
KEY = ("step", "module", "call", "port")
FORMAT = "nanovllm-explore-v1"
DIGEST = re.compile(r"[0-9a-f]{64}")
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def integer(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def name(value, maximum=1024):
    return isinstance(value, str) and 0 < len(value) <= maximum and value.isprintable()


def module_name(value):
    return value == "" or (name(value, 256) and "/" not in value and "\\" not in value
                           and all(value.split(".")))


def check_ids(prompt, decode):
    if (not isinstance(prompt, list) or not prompt or not isinstance(decode, list)
            or len(prompt) + len(decode) > MAX_TOKENS or len(decode) >= MAX_STEPS
            or any(not integer(token, 2**63 - 1) for token in (*prompt, *decode))):
        raise ValueError(f"需要非负整数词元、非空提示、最多 {MAX_TOKENS} 个词元及 {MAX_STEPS - 1} 步续写")


def check_watch(watch):
    if (not isinstance(watch, list) or len(watch) > MAX_WATCH
            or any(not module_name(path) or path == "@probe" for path in watch)
            or len(set(watch)) != len(watch)):
        raise ValueError(f"watch 需要最多 {MAX_WATCH} 个互异的原生模块路径")


def alias_index(groups, watch):
    """校验注册路径的分组，并建立从任一别名到规范路径的索引。"""
    if not isinstance(groups, dict) or len(groups) > len(watch):
        raise ValueError("模块别名表无效")
    index = {}
    watched = set(watch)
    for canonical, paths in groups.items():
        if not isinstance(paths, list) or not paths or paths[0] != canonical or len(index) + len(paths) > MAX_ITEMS:
            raise ValueError("模块别名组无效或过大")
        for path in paths:
            if not module_name(path) or path == "@probe" or path in index:
                raise ValueError("模块别名重复或无效")
            index[path] = canonical
        if watched.isdisjoint(paths):
            raise ValueError("模块别名组缺少被监视的注册路径")
    if watched - index.keys():
        raise ValueError("模块别名表缺少被监视的路径")
    return index


def validate_selector(selector, *, partial=False):
    if not isinstance(selector, dict) or set(selector) - {"run_id", *KEY}:
        raise ValueError("证据键含有未知字段")
    run_id = selector.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", run_id):
        raise ValueError("无效的 run_id")
    if not partial and any(field not in selector for field in KEY):
        raise ValueError("请提供完整的 step、module、call 和 port")
    for field in ("step", "call"):
        if field in selector and not integer(selector[field], 2**31 - 1):
            raise ValueError(f"无效的 {field}")
    if "module" in selector and not module_name(selector["module"]):
        raise ValueError("无效的模块路径")
    if "port" in selector and not name(selector["port"]):
        raise ValueError("无效的端口名")


def json_data(raw):
    def fields(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("JSON 字段重复")
            result[k] = v
        return result

    def invalid(value):
        raise ValueError(f"无效的 JSON 常量 {value}")

    return json.loads(raw, object_pairs_hook=fields, parse_constant=invalid)


def json_bytes(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode()


def display(value):
    if isinstance(value, complex):
        return {"real": display(value.real), "imag": display(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def segment(key):
    if isinstance(key, str):
        return key.replace("~", "~0").replace(".", "~1").replace("/", "~2") or "~e"
    if type(key) in (int, float, bool, type(None)):
        return "~" + type(key).__name__ + str(key)
    raise ValueError("容器键的类型无法编码")


def walk(value, path, parents=()):
    """逐层保留端口。只读实例字段和槽，避免调用模型属性或迭代未知对象。"""
    typ = type(value)
    label = f"{typ.__module__}.{typ.__qualname__}"
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized or value.is_meta:
            yield path, "unknown", {"type": label, "reason": f"无法物化张量 {value.layout}/{value.device}"}
        else:
            yield path, "tensor", value
        return
    if value is None or typ in (bool, int, float, str, torch.dtype, torch.device):
        scalar = str(value) if typ in (torch.dtype, torch.device) else display(value)
        yield path, "scalar", {"type": label, "value": scalar}
        return
    if id(value) in parents or len(parents) >= 32:
        yield path, "unknown", {"type": label, "reason": "循环引用或层级超限"}
        return
    if isinstance(value, Mapping):
        children = ((segment(k), v) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        children = ((str(i), v) for i, v in enumerate(value))
    elif isinstance(value, (type, torch.nn.Module)) or callable(value):
        yield path, "unknown", {"type": label, "reason": "对象未公开可读取的状态"}
        return
    else:
        fields = dict(getattr(value, "__dict__", {}))
        for base in typ.__mro__:
            for key, descriptor in vars(base).items():
                if isinstance(descriptor, MemberDescriptorType) and key not in fields:
                    try:
                        fields[key] = descriptor.__get__(value, typ)
                    except AttributeError:
                        continue
        if not fields:
            yield path, "unknown", {"type": label, "reason": "对象未公开可读取的字段"}
            return
        children = ((segment(k), v) for k, v in fields.items())
    yield path, "container", {"type": label}
    for part, child in children:
        yield from walk(child, f"{path}.{part}", (*parents, id(value)))


def snapshot(tensor):
    size = tensor.numel() * tensor.element_size()
    if size > MAX_TENSOR_BYTES:
        raise ValueError(f"张量大小 {size} 超过 {MAX_TENSOR_BYTES} 字节，请缩短输入或缩小 watch")
    return tensor.detach().resolve_conj().resolve_neg().to("cpu", copy=True, memory_format=torch.contiguous_format)


@contextmanager
def run_directory(runs, run_id):
    validate_selector({"run_id": run_id}, partial=True)
    root = os.open(runs, DIR_FLAGS)
    try:
        fd = os.open(run_id, DIR_FLAGS, dir_fd=root)
        try:
            yield fd
        finally:
            os.close(fd)
    finally:
        os.close(root)


def write_new(fd, filename, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(filename, flags, 0o644, dir_fd=fd), "wb") as stream:
        stream.write(data)


def checked_size(info, limit):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        raise ValueError("证据必须是大小受限、独立的常规文件")
    return info.st_size


@contextmanager
def checked_stream(fd, filename, limit):
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        with os.fdopen(os.open(filename, flags, dir_fd=fd), "rb") as stream:
            size = checked_size(os.fstat(stream.fileno()), limit)
            yield stream, size
    except OSError as exc:
        raise ValueError(f"运行证据不完整或文件不可读：{filename}") from exc


def read_file(fd, filename, limit):
    with checked_stream(fd, filename, limit) as (stream, size):
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError("证据文件大小已发生变化")
    return data


class Writer:
    def __init__(self, fd):
        self.fd = fd
        with os.scandir(fd) as entries:
            if next(entries, None) is not None:
                raise ValueError("运行目录已使用，需要宿主分配新的空目录")
        write_new(fd, ".started", b"")
        self.items = []
        self.size = 0

    def record(self, step, module, call, port, value):
        for path, kind, data in walk(value, port):
            if len(self.items) >= MAX_ITEMS or not name(path):
                raise ValueError("证据端口数或路径长度超限，请缩小 watch")
            item = dict(step=step, module=module, call=call, port=path, kind=kind)
            if kind == "tensor":
                nbytes = data.numel() * data.element_size()
                self.reserve(nbytes + 4096)
                tensor = snapshot(data)
                if tensor.is_complex():
                    # safetensors 的复数类型支持随版本变化，实部和虚部可无损地共用同一存储。
                    tensor = torch.view_as_real(tensor)
                    item["encoding"] = "complex_parts"
                raw = save({"value": tensor})
                filename = f"tensor-{len(self.items):05}.safetensors"
                write_new(self.fd, filename, raw)
                item.update(file=filename, shape=list(data.shape), dtype=str(data.dtype),
                            nbytes=nbytes, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                self.size += len(raw)
            else:
                if len(json_bytes(data)) > 4096:
                    raise ValueError("端口元数据超限，请缩小 watch")
                item.update(data)
            self.items.append(item)

    def reserve(self, amount):
        fs = os.fstatvfs(self.fd)
        if self.size + amount > MAX_RUN_BYTES or fs.f_bavail * fs.f_frsize < amount:
            raise ValueError(f"运行存储或可用磁盘不足，单次上限 {MAX_RUN_BYTES} 字节，请缩短输入或缩小 watch")

    def finish(self, run_id, provenance, calibration):
        manifest = dict(format=FORMAT, run_id=run_id, provenance=provenance,
                        calibration=calibration, items=self.items)
        raw = json_bytes(manifest)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("运行清单超限，请缩小 watch")
        self.reserve(len(raw) + 64)
        write_new(self.fd, "manifest.json", raw)
        # 最后发布摘要。进程中途退出时，没有完整发布标记的目录不能查询或复用。
        write_new(self.fd, ".complete", hashlib.sha256(raw).hexdigest().encode())


def validate_item(item, index, run_id, provenance):
    if not isinstance(item, dict) or not {*KEY, "kind"} <= item.keys():
        raise ValueError("端口清单缺少字段")
    selector = {k: item[k] for k in KEY}
    validate_selector(dict(run_id=run_id, **selector))
    if item["step"] > len(provenance["decode_ids"]):
        raise ValueError("清单中的 step 超过实际步数")
    if item["module"] == "@probe":
        if provenance["target"] != "probe" or item["call"] != 0:
            raise ValueError("探针端口来源无效")
    elif item["module"] not in provenance.get("module_aliases", provenance["watch"]):
        raise ValueError("模块未在 watch 中声明")
    kind = item["kind"]
    fields = {"tensor": {"file", "size", "sha256", "shape", "dtype", "nbytes"},
              "scalar": {"type", "value"}, "container": {"type"}, "unknown": {"type", "reason"}}
    if kind == "tensor" and item.get("encoding") == "complex_parts":
        fields["tensor"].add("encoding")
    if not isinstance(kind, str) or kind not in fields or set(item) != {*KEY, "kind", *fields[kind]}:
        raise ValueError("无效的端口元数据")
    if kind == "tensor":
        shape = item["shape"]
        if (item["file"] != f"tensor-{index:05}.safetensors" or not name(item["dtype"], 64)
                or not isinstance(shape, list) or len(shape) > 64
                or any(not integer(n, 2**63 - 1) for n in shape) or math.prod(shape) > MAX_TENSOR_BYTES
                or not integer(item["nbytes"], MAX_TENSOR_BYTES)
                or not integer(item["size"], item["nbytes"] + 4096) or item["size"] <= item["nbytes"]
                or not isinstance(item["sha256"], str) or not DIGEST.fullmatch(item["sha256"])):
            raise ValueError("张量文件引用或元数据无效")
    elif (not name(item["type"], 512) or (kind == "unknown" and not name(item["reason"]))
          or (kind == "scalar" and type(item["value"]) not in (type(None), bool, int, float, str))):
        raise ValueError("标量或对象元数据无效")


def logical_tensor(value, item):
    if item.get("encoding") == "complex_parts":
        if value.dtype not in (torch.float16, torch.float32, torch.float64) or value.shape[-1:] != (2,):
            raise ValueError("无效的复数张量编码")
        value = torch.view_as_complex(value)
    if (list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]
            or value.numel() * value.element_size() != item["nbytes"]):
        raise ValueError("张量与清单标注不符")
    return value


def verify_header(fd, item):
    with checked_stream(fd, item["file"], item["size"]) as (stream, size):
        header = int.from_bytes(stream.read(8), "little")
        if not 0 < header <= 4088 or size != 8 + header + item["nbytes"]:
            raise ValueError("无效的 safetensors 文件头长度")
        # Linux worker 以已核验的 fd 打开同一个常规文件；CPU 映射可读取元数据而不复制张量。
        try:
            with safe_open(f"/proc/self/fd/{stream.fileno()}", framework="pt", device="cpu") as handle:
                if handle.keys() != ["value"]:
                    raise ValueError("张量文件的端口无效")
                logical_tensor(handle.get_tensor("value"), item)
        except Exception as exc:
            raise ValueError("张量文件头与证据清单不符") from exc


def validate_provenance(run_id, p, calibration):
    if not isinstance(p, dict) or p.get("target") not in ("hf", "probe"):
        raise ValueError("证据来源无效")
    issued = re.fullmatch(r"(hf|probe)-[a-f0-9]{32}", run_id)
    if issued and issued[1] != p["target"]:
        raise ValueError("证据来源与宿主分配的运行身份不符")
    check_ids(p.get("prompt_ids"), p.get("decode_ids"))
    check_watch(p.get("watch"))
    for field in ("model_dir", "model_class", "model_type", "dtype", "device", "device_backend",
                  "torch_version", "transformers_version"):
        if not name(p.get(field)):
            raise ValueError(f"缺少运行来源字段 {field}")
    files = p.get("checkpoint_files")
    if (not isinstance(files, dict) or not 1 <= len(files) <= 1030 or "config.json" not in files
            or any(not name(k, 255) or "/" in k or "\\" in k or not k.endswith((".json", ".safetensors"))
                   or not isinstance(v, str) or not DIGEST.fullmatch(v) for k, v in files.items())):
        raise ValueError("checkpoint 来源清单无效")
    if not isinstance(calibration, dict) or "hf_reliable" not in p:
        raise ValueError("缺少校准结果")
    status = calibration.get("status")
    probe_hash = p.get("probe_sha256")
    if p["target"] == "probe":
        if (status != "probe_derived" or p["hf_reliable"] is not None
                or not isinstance(probe_hash, str) or not DIGEST.fullmatch(probe_hash)):
            raise ValueError("探针证据来源无效")
        return
    steps = calibration.get("steps")
    if (probe_hash is not None or not isinstance(steps, list) or len(steps) != len(p["decode_ids"]) + 1
            or status not in ("match", "mismatch") or p["hf_reliable"] is not (status == "match")):
        raise ValueError("HF 证据校准状态无效")
    reliable = True
    for index, row in enumerate(steps):
        if (not isinstance(row, dict) or row.get("step") != index
                or any(type(row.get(k)) is not bool for k in ("exact", "finite", "shape_match", "dtype_match"))
                or not isinstance(row.get("state"), dict) or row["state"].get("status") not in ("match", "mismatch")
                or (row["exact"] and not (row["finite"] and row["shape_match"] and row["dtype_match"]))):
            raise ValueError("逐步校准记录无效")
        reliable = reliable and row["exact"] and row["state"]["status"] == "match"
    if reliable != p["hf_reliable"]:
        raise ValueError("HF 可靠性与逐步校准结果不符")


class Archive:
    def __init__(self, fd, run_id):
        self.fd = fd
        self.run_id = run_id
        complete = read_file(fd, ".complete", 64)
        raw = read_file(fd, "manifest.json", MAX_MANIFEST_BYTES)
        if hashlib.sha256(raw).hexdigest().encode() != complete:
            raise ValueError("清单与发布摘要不符")
        data = json_data(raw)
        if (not isinstance(data, dict) or set(data) != {"format", "run_id", "provenance", "calibration", "items"}
                or data["format"] != FORMAT or data["run_id"] != run_id):
            raise ValueError("运行清单版本或身份无效")
        self.provenance = data["provenance"]
        self.calibration = data["calibration"]
        validate_provenance(run_id, self.provenance, self.calibration)
        groups = self.provenance.get("module_aliases", {})
        self.registered = alias_index(groups, self.provenance["watch"]) if "module_aliases" in self.provenance else {}
        items = data["items"]
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            raise ValueError("端口清单超限")
        self.items = {}
        files = {".started": 0, ".complete": 64, "manifest.json": len(raw)}
        for index, item in enumerate(items):
            validate_item(item, index, run_id, self.provenance)
            key = tuple(item[k] for k in KEY)
            if key in self.items:
                raise ValueError("清单存在重复端口")
            self.items[key] = item
            if item["kind"] == "tensor":
                files[item["file"]] = item["size"]
        if sum(files.values()) > MAX_RUN_BYTES:
            raise ValueError("运行文件总大小超限")
        with os.scandir(fd) as entries:
            for entry in entries:
                size = files.pop(entry.name, None)
                if size is None or checked_size(entry.stat(follow_symlinks=False), size) != size:
                    raise ValueError("运行目录含有未声明或不完整的文件")
        if files:
            raise ValueError("运行目录缺少清单中的文件")
        for item in self.items.values():
            if item["kind"] == "tensor":
                verify_header(fd, item)

    def metadata(self, item):
        public = {k: v for k, v in item.items() if k not in ("file", "size", "sha256", "value")}
        module = item["module"]
        if module == "@probe":
            public["origin"] = "probe_callback"
        else:
            # hook 收到模块对象。共享模块本次调用使用的注册别名无法从对象推断。
            aliases = self.provenance.get("module_aliases", {}).get(module)
            public.update(origin="forward_hook", canonical_module=self.registered.get(module),
                          aliases=aliases, caller_path=None, caller_status="unknown")
            if aliases and (len(aliases) > 32 or len(json_bytes(aliases)) > 2048):
                public.update(aliases=aliases[:4], alias_count=len(aliases),
                              aliases_observe=dict(run_id=self.run_id, view="aliases", module=module))
        return dict(public, run_id=self.run_id, selector=item_selector(self.run_id, item),
                    probe_derived=self.provenance["target"] == "probe",
                    hf_reliable=self.provenance["hf_reliable"])

    def resolve(self, selector):
        if "module" in selector:
            return dict(selector, module=self.registered.get(selector["module"], selector["module"]))
        return selector

    def tensor(self, item):
        raw = read_file(self.fd, item["file"], item["size"])
        if hashlib.sha256(raw).hexdigest() != item["sha256"] or int.from_bytes(raw[:8], "little") > 4088:
            raise ValueError("张量文件校验失败")
        try:
            values = load(raw)
        except Exception as exc:
            raise ValueError("无效的 safetensors 文件") from exc
        if set(values) != {"value"}:
            raise ValueError("张量文件的端口无效")
        return logical_tensor(values["value"], item)


def open_archive(stack, runs, run_id):
    try:
        fd = stack.enter_context(run_directory(runs, run_id))
    except FileNotFoundError:
        return None
    return Archive(fd, run_id)


def item_selector(run_id, item):
    return {"run_id": run_id, **{field: item[field] for field in KEY}}


def ranked_tensors(items):
    """首页给出具名探针、公开 logits 和各模块输出的样例，后续页覆盖所有张量。"""
    probe, logits, outputs, other = [], [], [], []
    first_probes, first_outputs = {}, {}
    for item in items:
        if item["kind"] != "tensor":
            continue
        module, port = item["module"], item["port"]
        if module == "@probe":
            probe.append(item)
            if len(first_probes) < 4:
                first_probes.setdefault(port, item)
        elif module == "" and port == "output.logits":
            logits.append(item)
        elif port == "output" or port.startswith("output."):
            outputs.append(item)
            first_outputs.setdefault(module, item)
        else:
            other.append(item)
    if logits:
        first_outputs.pop("", None)
    featured = [*first_probes.values(), *logits[:1], *first_outputs.values()]
    seen = {id(item) for item in featured}
    return featured + [item for group in (probe, logits, outputs, other) for item in group if id(item) not in seen]


def index_page(rows, render, query, offset, limit):
    page = []
    size = 0
    for row in rows[offset:offset + limit]:
        value = render(row)
        size += len(json_bytes(value)) + 1
        if page and size > INDEX_PAGE_BYTES:
            break
        page.append(value)
    end = offset + len(page)
    next_observe = None
    if page and end < len(rows):
        next_observe = dict(query, offset=end, limit=limit)
    return page, dict(offset=offset, total=len(rows), next_observe=next_observe)


def selector_provenance(provenance):
    return dict(target=provenance["target"], probe_derived=provenance["target"] == "probe",
                hf_reliable=provenance["hf_reliable"], probe_sha256=provenance.get("probe_sha256"),
                module_paths="canonical_registration" if "module_aliases" in provenance else "as_recorded",
                caller_status="unknown")


def selector_page(items, query, provenance, offset=0, limit=SELECTOR_PAGE_SIZE):
    keys, info = index_page(ranked_tensors(items), lambda item: item_selector(query["run_id"], item),
                            dict(query, view="selectors"), offset, limit)
    return dict(selectors=keys, selector_page=dict(info, provenance=selector_provenance(provenance)))


def alias_page(archive, selector, offset, limit):
    if set(selector) - {"module"}:
        raise ValueError("注册路径只按 module 筛选")
    if "module_aliases" not in archive.provenance:
        return dict(status="not_captured", run_id=archive.run_id, reason="此运行没有记录模块别名")
    rows = [(canonical, path) for path, canonical in archive.registered.items()
            if "module" not in selector or selector["module"] == canonical]
    page, info = index_page(rows, lambda row: dict(canonical_module=row[0], alias=row[1]),
                            dict(run_id=archive.run_id, view="aliases", **selector), offset, limit)
    return dict(status="ok", run_id=archive.run_id, aliases=page,
                alias_page=dict(info, caller_path=None, caller_status="unknown"))


def observe(runs, arguments):
    selector = dict(arguments)
    view = selector.pop("view", None)
    if view not in (None, "selectors", "aliases"):
        raise ValueError("observe 的可选 view 为 selectors 或 aliases")
    offset = selector.pop("offset", 0)
    limit = selector.pop("limit", SELECTOR_PAGE_SIZE if view == "selectors" else 48)
    if not integer(offset, 2**63 - 1) or not integer(limit, MAX_PAGE):
        raise ValueError(f"offset 需要非负整数，limit 范围为 0 到 {MAX_PAGE}")
    validate_selector(selector, partial=True)
    run_id = selector.pop("run_id")
    with ExitStack() as stack:
        archive = open_archive(stack, runs, run_id)
        if archive is None:
            return {"status": "not_captured", "run_id": run_id}
        selector = archive.resolve(selector)
        if view == "aliases":
            return alias_page(archive, selector, offset, limit)
        found = [item for item in archive.items.values() if all(item[k] == v for k, v in selector.items())]
        if view == "selectors":
            page = selector_page(found, dict(run_id=run_id, **selector), archive.provenance, offset, limit)
            return dict(status="ok", run_id=run_id, **page)
        qualified = {"step", "module", "port"} <= selector.keys()
        if qualified and not found:
            return {"status": "not_captured", "run_id": run_id, **selector}
        if qualified and len(found) == 1 and ("call" in selector or found[0]["kind"] == "tensor"):
            item = found[0]
            result = {"status": "ok", "item": archive.metadata(item), "offset": offset}
            if item["kind"] == "tensor":
                flat = archive.tensor(item).reshape(-1)
                result.update(total=flat.numel(), values=[display(v) for v in flat[offset:offset + limit].tolist()])
            elif item["kind"] == "scalar":
                result.update(total=1, values=[item["value"]][offset:offset + limit])
            elif item["kind"] == "unknown":
                result["status"] = "not_captured"
            else:
                scope = (item["step"], item["module"], item["call"])
                children = [child for k, child in archive.items.items()
                            if k[:3] == scope and k[3].rpartition(".")[0] == item["port"]]
                result.update(total=len(children), items=[archive.metadata(i) for i in children[offset:offset + limit]])
            return result
        if qualified and len(found) > 1:
            found.sort(key=lambda item: item["call"])
            choices, info = index_page(found, archive.metadata, dict(run_id=run_id, **selector), offset, limit)
            return dict(status="ambiguous", run_id=run_id, items=choices, **info,
                        provenance=selector_provenance(archive.provenance))
        return dict(status="ok", run_id=run_id, total=len(found), offset=offset,
                    items=[archive.metadata(i) for i in found[offset:offset + limit]],
                    provenance=archive.provenance, calibration=archive.calibration)


def coordinates(index, shape):
    result = []
    for width in reversed(shape):
        index, remainder = divmod(index, width)
        result.append(remainder)
    return result[::-1]


def tensor_comparison(reference, actual):
    """逐块按原精度比较所有元素，仅限制返回的差异样例数。"""
    if reference.dtype != actual.dtype:
        return dict(status="mismatch", reason="dtype")
    if reference.shape != actual.shape:
        return dict(status="mismatch", reason="shape")
    left, right = reference.reshape(-1), actual.reshape(-1)
    mismatches = []
    count = 0
    finite = True
    for start in range(0, left.numel(), 65536):
        a, b = left[start:start + 65536], right[start:start + 65536]
        # 一些后端未实现 float8 的比较；到 float32 的扩展可精确表示其全部数值。
        if a.is_floating_point() and a.element_size() == 1:
            a, b = a.float(), b.float()
        valid = torch.isfinite(a) & torch.isfinite(b)
        finite = finite and bool(valid.all())
        bad = (a != b) | ~valid
        count += int(bad.sum())
        if len(mismatches) < 8:
            for ix in bad.nonzero().flatten()[:8 - len(mismatches)].tolist():
                mismatches.append(dict(index=coordinates(start + ix, reference.shape),
                                       reference=display(a[ix].item()), actual=display(b[ix].item())))
    return dict(status="mismatch" if count else "match", reason="nonfinite" if not finite else
                ("values" if count else "equal"), finite=finite, mismatch_count=count, mismatches=mismatches)


def compare(runs, arguments):
    if set(arguments) != {"reference", "actual"}:
        raise ValueError("compare 需要 reference 和 actual 两个完整键")
    with ExitStack() as stack:
        opened = {}
        values = []
        details = {}
        missing = []
        for side in ("reference", "actual"):
            selector = arguments[side]
            validate_selector(selector)
            run_id = selector["run_id"]
            if run_id not in opened:
                opened[run_id] = open_archive(stack, runs, run_id)
            archive = opened[run_id]
            item = None
            if archive is not None:
                resolved = archive.resolve(selector)
                item = archive.items.get(tuple(resolved[k] for k in KEY))
            details[side] = archive.metadata(item) if item else selector
            if item is None or item["kind"] not in ("scalar", "tensor"):
                missing.append(side)
            else:
                values.append(archive.tensor(item) if item["kind"] == "tensor" else item["value"])
        if missing:
            return dict(status="not_captured", missing=missing, **details)
        a, b = values
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return dict(tensor_comparison(a, b), **details)
        if details["reference"].get("type") != details["actual"].get("type"):
            return dict(status="mismatch", reason="dtype", **details)
        finite = details["reference"].get("type") != "builtins.float" or not any(isinstance(v, str) for v in values)
        if finite and a == b:
            return dict(status="match", **details)
        return dict(status="mismatch", reason="values" if finite else "nonfinite", mismatch_count=1,
                    mismatches=[dict(index=[], reference=a, actual=b)], **details)
