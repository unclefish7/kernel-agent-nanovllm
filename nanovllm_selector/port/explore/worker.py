"""隔离容器内的模型探查入口。每个进程处理一个 JSON 请求。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from pathlib import Path


def checkpoint_files(model_dir):
    """记录本次实际选用的本地配置和 safetensors 权重，可复核分片来源。"""
    from .evidence import json_data

    root = model_dir.resolve(strict=True)

    def local(filename):
        if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
            raise ValueError("checkpoint 中存在无效的本地文件引用")
        path = (root / filename).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("checkpoint 文件超出模型目录")
        return path

    names = ["config.json"]
    if (root / "model.safetensors").is_file():
        names.append("model.safetensors")
    else:
        index_name = "model.safetensors.index.json"
        index = local(index_name)
        if index.stat().st_size > 64 << 20:
            raise ValueError("权重索引过大")
        weight_map = json_data(index.read_bytes()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("缺少本地 safetensors 权重")
        shards = set(weight_map.values())
        if len(shards) > 1024 or any(not isinstance(s, str) or not s.endswith(".safetensors") for s in shards):
            raise ValueError("无效的 safetensors 分片索引")
        names.extend([index_name, *sorted(shards)])
    if (root / "generation_config.json").is_file():
        names.append("generation_config.json")
    result = {}
    for filename in names:
        digest = hashlib.sha256()
        with local(filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        result[filename] = digest.hexdigest()
    return result


def direct_state(module):
    import torch

    from .evidence import segment

    for key, value in module.named_buffers(recurse=False):
        yield "buffers." + segment(key), value
    for key, value in vars(module).items():
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
            yield "attributes." + segment(key), value


class Capture:
    def __init__(self, writer):
        self.writer = writer
        self.step = None
        self.calls = Counter()
        self.active = defaultdict(list)

    def begin(self, step):
        self.step = step
        self.calls.clear()
        self.active.clear()

    def instrument(self, stack, model, watch):
        from .evidence import alias_index

        modules = dict(model.named_modules(remove_duplicate=False))
        by_object = {}
        for path in watch:
            if path not in modules:
                raise ValueError(f"watch 找不到模块 {path!r}，请用 read_code 查询模块树")
            by_object[id(modules[path])] = []
        # 规范名取注册树中第一次出现的路径，独立于 watch 的顺序。
        for path, module in modules.items():
            if id(module) in by_object:
                by_object[id(module)].append(path)
        aliases = {paths[0]: paths for paths in by_object.values()}
        alias_index(aliases, watch)
        for canonical in aliases:
            self.attach(stack, canonical, modules[canonical])
        return aliases

    def attach(self, stack, path, module):
        def record_inputs(call, prefix, args, kwargs):
            self.writer.record(self.step, path, call, prefix + ".args", args)
            self.writer.record(self.step, path, call, prefix + ".kwargs", kwargs)

        def record_state(call, prefix):
            for field, value in direct_state(module):
                self.writer.record(self.step, path, call, f"state.{prefix}.{field}", value)

        def before(_module, args, kwargs):
            if self.step is None:
                return
            call = self.calls[path]
            self.calls[path] += 1
            self.active[path].append(call)
            record_inputs(call, "input", args, kwargs)
            record_state(call, "before")

        def after(_module, args, kwargs, output):
            if self.step is None:
                return
            call = self.active[path].pop()
            self.writer.record(self.step, path, call, "output", output)
            record_inputs(call, "post_input", args, kwargs)
            record_state(call, "after")

        stack.callback(module.register_forward_pre_hook(before, with_kwargs=True).remove)
        stack.callback(module.register_forward_hook(after, with_kwargs=True).remove)


def visible_state(model, output):
    """校准返回的非 logits 状态及模型上可读取的 buffer 和直接张量属性。"""
    import torch

    from .evidence import MAX_ITEMS, MAX_RUN_BYTES, MAX_TENSOR_BYTES, segment, snapshot, walk

    def roots():
        for key, value in output.items():
            if key != "logits":
                yield "output." + segment(key), value
        for path, module in model.named_modules():
            for field, value in direct_state(module):
                yield f"module[{path}].{field}", value

    fields = {}
    unknown = {}
    tensors = 0
    size = 0
    for root, value in roots():
        for path, kind, leaf in walk(value, root):
            if len(fields) + len(unknown) >= MAX_ITEMS:
                unknown[path] = "已达到校准状态端口上限"
                return fields, unknown, tensors
            if kind == "tensor":
                nbytes = leaf.numel() * leaf.element_size()
                if nbytes > MAX_TENSOR_BYTES or size + nbytes > MAX_RUN_BYTES:
                    unknown[path] = "状态张量超出校准容量"
                    continue
                flat = snapshot(leaf).reshape(-1).view(torch.uint8).numpy()
                fields[path] = (str(leaf.dtype), tuple(leaf.shape), hashlib.sha256(flat).hexdigest())
                tensors += 1
                size += nbytes
            elif kind == "unknown":
                unknown[path] = leaf["reason"]
            else:
                fields[path] = leaf
    return fields, unknown, tensors


class Calibration:
    def __init__(self):
        self.baseline = []
        self.size = 0
        self.steps = []

    def sample(self, model, step, output, *, baseline):
        import torch

        from .evidence import MAX_RUN_BYTES, snapshot, tensor_comparison

        logits = output.logits
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or not logits.numel():
            raise ValueError("校准需要模型各位置返回的完整 logits")
        current = snapshot(logits)
        fields, unknown, tensors = visible_state(model, output)
        if baseline:
            self.size += current.numel() * current.element_size()
            if self.size > MAX_RUN_BYTES:
                raise ValueError("用于校准的完整 logits 总量超限，请缩短输入")
            self.baseline.append((current, fields, unknown))
            return
        expected, old_fields, old_unknown = self.baseline[step]
        comparison = tensor_comparison(expected, current)
        changed = sorted(path for path in old_fields.keys() | fields.keys()
                         if old_fields.get(path) != fields.get(path))
        # 公开字段变化本身也是差异；不可见的状态单独列出。
        changed.extend(sorted(old_unknown.keys() ^ unknown.keys()))
        state = dict(status="mismatch" if changed else "match", tensors=tensors,
                     fields=len(fields), changed_count=len(changed), changed_paths=changed[:8],
                     unknown_count=len(unknown), unknown_paths=sorted(unknown)[:8])
        finite = comparison.get("finite")
        if finite is None:
            finite = bool(torch.isfinite(expected).all() and torch.isfinite(current).all())
        self.steps.append(dict(step=step, finite=finite,
                               dtype_match=expected.dtype == current.dtype, shape_match=expected.shape == current.shape,
                               exact=comparison["status"] == "match", logits=comparison, state=state))
        self.baseline[step] = None

    def result(self):
        reliable = all(row["exact"] and row["state"]["status"] == "match" for row in self.steps)
        return dict(status="match" if reliable else "mismatch", steps=self.steps,
                    state_scope="返回的非 logits 字段、已注册 buffer 和模块直接张量属性；其他内部状态未知")


def prepare_probe(source, model):
    import torch

    scope = {"__name__": "__nanovllm_probe__", "torch": torch}
    exec(compile(source, "<agent-probe>", "exec"), scope)
    callback = scope.get("probe")
    if not callable(callback):
        raise ValueError("探针需要定义 probe(model, step, inputs, output)")
    if "setup" in scope:
        if not callable(scope["setup"]):
            raise ValueError("setup 需要是可调用对象")
        scope["setup"](model)
    return callback


def forwards(model, prompt, decode, writer, watch, callback, calibration):
    import torch
    from transformers.generation.configuration_utils import GenerationMode
    from transformers.generation.utils import ALL_CACHE_NAMES

    from .evidence import MAX_ITEMS, name

    sequence = torch.tensor([prompt + decode], device=model.device, dtype=torch.long)
    length = len(prompt)
    config = deepcopy(model.generation_config)
    config.use_cache = True
    config.num_beams = config.num_return_sequences = 1
    kwargs = {"attention_mask": torch.ones_like(sequence[:, :length]), "use_cache": True}
    model._prepare_cache_for_generation(config, kwargs, GenerationMode.GREEDY_SEARCH, 1, sequence.shape[1])
    kwargs = model._get_initial_cache_position(length, sequence.device, kwargs)
    capture = Capture(writer) if writer is not None else None
    with ExitStack() as stack:
        aliases = capture.instrument(stack, model, watch) if capture is not None else {}
        for step in range(len(decode) + 1):
            inputs = model.prepare_inputs_for_generation(sequence[:, :length + step], **kwargs)
            inputs["return_dict"] = True
            if capture is not None:
                capture.begin(step)
            output = model(**inputs)
            if capture is not None:
                capture.step = None
            if calibration is not None:
                calibration.sample(model, step, output, baseline=writer is None)
            if callback is not None:
                values = callback(model, step, inputs, output)
                if (not isinstance(values, dict) or len(values) > MAX_ITEMS or
                        any(not name(k, 192) or not isinstance(v, torch.Tensor) for k, v in values.items())):
                    raise ValueError("probe 需要返回名称映射到张量的字典")
                writer.record(step, "@probe", 0, "output", values)
            if step < len(decode):
                kwargs = model._update_model_kwargs_for_generation(output, kwargs, is_encoder_decoder=False)
                carried = [v for k, v in output.items() if k != "logits" and v is not None]
                if (not any(output.get(k) is not None for k in ALL_CACHE_NAMES)
                        and not any(v is state for v in kwargs.values() for state in carried)):
                    raise ValueError("模型未向下一步传递可确认的返回状态")
    return aliases


def execute(model_dir, arguments, device, writer, calibration):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    # 两次加载的初始随机状态一致，兼容从配置重建的非持久 buffer。
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto", local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True)
    model = model.to(device).eval()
    actual_device = model.device
    backend = actual_device.type
    if backend == "cuda" and torch.version.hip:
        backend = "rocm"
    info = dict(model_class=f"{type(model).__module__}.{type(model).__qualname__}",
                model_type=model.config.model_type, dtype=str(model.dtype), device=str(actual_device),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                device_backend=backend, attn_implementation=model.config._attn_implementation,
                torch_version=str(torch.__version__), transformers_version=transformers.__version__)
    if actual_device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(actual_device)
    with torch.inference_mode():
        probe = prepare_probe(arguments["probe"], model) if arguments["target"] == "probe" else None
        info["module_aliases"] = forwards(model, arguments["prompt_ids"], arguments.get("decode_ids", []),
                                          writer, arguments.get("watch", [""]), probe, calibration)
    return info


def compact_run_reply(reply):
    """聊天工具在 30000 字符处折叠结果。优先保留可直接调用的键和逐步校准结论。"""
    if len(json.dumps(reply, ensure_ascii=False)) <= 28000:
        return reply
    run_id = reply["run_id"]
    full = reply["provenance"]
    provenance = {k: v for k, v in full.items() if len(json.dumps(v, ensure_ascii=False)) <= 1024}
    provenance["omitted_fields"] = sorted(full.keys() - provenance.keys())
    if "module_aliases" not in provenance:
        provenance["aliases_observe"] = dict(run_id=run_id, view="aliases")
    calibration = reply["calibration"]
    if "steps" in calibration:
        steps = []
        for row in calibration["steps"]:
            facts = {k: row[k] for k in ("step", "finite", "dtype_match", "shape_match", "exact")}
            facts["state"] = {k: row["state"][k] for k in ("status", "tensors", "changed_count", "unknown_count")}
            steps.append(facts)
        calibration = dict(calibration, steps=steps)
    return dict(reply, provenance=provenance, calibration=calibration,
                details_observe=dict(run_id=run_id, limit=0))


def run_model(model_dir, runs, arguments):
    import torch

    from .evidence import MAX_RUN_BYTES, MAX_TENSOR_BYTES, Writer, check_ids, check_watch, run_directory, selector_page

    allowed = {"run_id", "target", "prompt_ids", "decode_ids", "watch", "probe"}
    if set(arguments) - allowed or not {"run_id", "target", "prompt_ids"} <= arguments.keys():
        raise ValueError("run_model 参数无效")
    target = arguments["target"]
    if target not in ("hf", "probe"):
        raise ValueError("target 需要为 hf 或 probe")
    prompt, decode, watch = arguments["prompt_ids"], arguments.get("decode_ids", []), arguments.get("watch", [""])
    check_ids(prompt, decode)
    check_watch(watch)
    source = arguments.get("probe")
    if target == "probe" and (not isinstance(source, str) or not 0 < len(source.encode()) <= 65536):
        raise ValueError("probe 模式需要最多 65536 字节的 Python 源码")
    if target == "hf" and "probe" in arguments:
        raise ValueError("仅 probe 模式接收探针源码")
    run_id = arguments["run_id"]
    with run_directory(runs, run_id) as fd:
        writer = Writer(fd)
        files = checkpoint_files(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        calibration = Calibration() if target == "hf" else None
        if calibration is not None:
            execute(model_dir, arguments, device, None, calibration)
        info = execute(model_dir, arguments, device, writer, calibration)
        result = calibration.result() if calibration is not None else {"status": "probe_derived"}
        reliable = result["status"] == "match" if calibration is not None else None
        provenance = dict(target=target, model_dir=str(model_dir), checkpoint_files=files,
                          prompt_ids=prompt, decode_ids=decode, watch=watch, hf_reliable=reliable,
                          seed=0, state_scope="input 与 post_input 为调用前后实参；state 为直接 buffer 和张量属性；未公开的内部状态未知",
                          probe_sha256=hashlib.sha256(source.encode()).hexdigest() if source else None,
                          tensor_limit_bytes=MAX_TENSOR_BYTES, run_limit_bytes=MAX_RUN_BYTES, **info)
        writer.finish(run_id, provenance, result)
        reply = dict(ok=True, status="unreliable" if reliable is False else "ok", hf_reliable=reliable,
                     run_id=run_id, steps=len(decode) + 1, item_count=len(writer.items),
                     **selector_page(writer.items, {"run_id": run_id}, provenance),
                     provenance=provenance, calibration=result)
        return compact_run_reply(reply)


def dispatch(model_dir: Path, runs: Path, request: dict) -> dict:
    if not isinstance(request, dict) or set(request) != {"tool", "arguments"} or not isinstance(request["arguments"], dict):
        raise ValueError("需要含 tool 和 arguments 的 JSON 对象")
    tool, arguments = request["tool"], request["arguments"]
    if tool == "read_code":
        from .source import read_code

        return read_code(model_dir, arguments)
    if tool == "run_model":
        return run_model(model_dir, runs, arguments)
    if tool == "observe":
        from .evidence import observe

        return observe(runs, arguments)
    if tool == "compare":
        from .evidence import compare

        return compare(runs, arguments)
    raise ValueError(f"未知工具 {tool!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    options = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(262145)
        if len(raw) > 262144:
            raise ValueError("工具请求超过 256 KiB")
        with redirect_stdout(sys.stderr):
            result = dispatch(options.model, options.runs, json.loads(raw))
        line = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        line = json.dumps({"ok": False, "status": "error", "error": str(exc)[:2000]}, ensure_ascii=False)
    print(line, flush=True)


if __name__ == "__main__":
    main()
