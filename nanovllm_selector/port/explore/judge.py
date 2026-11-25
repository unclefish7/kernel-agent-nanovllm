"""单卡正式验收。engine 运行候选，reference 只读取交接文件并从本地 HF 权重重算。"""

import argparse
import ast
from contextlib import redirect_stdout
from functools import cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import NamedTemporaryFile
import traceback

DECODE_STEPS = 8
BLOCK_SIZE = 256  # 正式验收使用 nano-vLLM Config 的默认页大小。
HOLDOUT_TEXT = "".join(
    f"Station {i} recorded the river level before sunrise. 午班核对水位计并抄录当日的读数。\n"
    for i in range(64)
)
META_LIMIT = 64 * 1024
HEADER_LIMIT = 16 * 1024
TENSOR_LIMIT = 64 * 1024 * 1024


def _json_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _checkpoint_path(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("无效的 checkpoint 文件名")
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or any(p.startswith(".") for p in relative.parts):
        raise ValueError(f"无效的 checkpoint 相对路径 {name!r}")
    path = (root / relative).resolve(strict=True)
    inside = (path.is_relative_to(root) and path.suffix == relative.suffix
              and not any(p.startswith(".") for p in path.relative_to(root).parts))
    cached = (root.parent.name == "snapshots" and path.parent == root.parent.parent / "blobs"
              and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", path.name))
    if not (inside or cached) or not path.is_file():
        raise ValueError(f"checkpoint 文件不在模型目录或对应的 HF 缓存内：{name}")
    return path


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@cache
def _file_sha256(path, stamp):
    """宿主可跨多个候选复用未变分片的散列，两阶段的独立进程各自核实。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != stamp:
            raise ValueError(f"计算摘要前 {path.name} 已变化")
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        if _stamp(os.fstat(stream.fileno())) != stamp:
            raise ValueError(f"计算摘要期间 {path.name} 已变化")
    return digest.hexdigest()


def _indexed_weights(root, index_name, suffix):
    index = json.loads(_read_bounded(_checkpoint_path(root, index_name), 64 << 20))
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("checkpoint 缺少有效的 weight_map")
    if any(not isinstance(shard, str) or not shard.endswith(suffix) for shard in mapping.values()):
        raise ValueError(f"{index_name} 中的权重分片无效")
    return {index_name, *mapping.values()}


def _checkpoint_files(model_dir):
    root = Path(model_dir).resolve(strict=True)
    # 引擎实际遍历顶层所有 safetensors，HF 按单文件或索引选择；覆盖两端读取的权重。
    weights = {p.name for p in root.glob("*.safetensors") if not p.name.startswith(".")}
    names = {"config.json", *(p.name for p in root.glob("config.*.json"))}
    if (root / "generation_config.json").is_file():
        names.add("generation_config.json")
    if "model.safetensors" not in weights:
        safe_index = "model.safetensors.index.json"
        bin_index = "pytorch_model.bin.index.json"
        if (root / safe_index).is_file():
            names.update(_indexed_weights(root, safe_index, ".safetensors"))
        elif (root / "pytorch_model.bin").is_file():
            names.add("pytorch_model.bin")
        elif (root / bin_index).is_file():
            names.update(_indexed_weights(root, bin_index, ".bin"))
    if not weights:
        raise ValueError("没有可供引擎和参照核对的 safetensors 权重")
    files = {}
    for name in sorted(names | weights):
        path = _checkpoint_path(root, name)
        files[name] = (path, _stamp(path.stat()))
    return files


def _checkpoint_digest(files):
    return _json_digest({name: _file_sha256(path, stamp) for name, (path, stamp) in files.items()})


def checkpoint_sha256(model_dir: Path) -> str:
    """按相对文件名和内容标识同一 checkpoint，宿主调用此函数只需 Python 标准库。"""
    return _checkpoint_digest(_checkpoint_files(model_dir))


def _check_snapshot(model_dir, files):
    # 独立 CLI 的模型目录可以变化。交接和打分前复核文件身份，无需再次读取权重。
    if _checkpoint_files(model_dir) != files:
        raise _BindingError("验收期间 checkpoint_sha256 对应的文件发生变化")


def _inputs(model, text, lengths):
    from transformers import AutoConfig, AutoTokenizer

    if not model.is_dir():
        raise ValueError("需要本地模型目录")
    options = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(model, **options)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, **options)
    tokens = tokenizer(text).input_ids[:max(lengths)]
    if len(tokens) != max(lengths) or any(type(t) is not int or not 0 <= t < config.vocab_size for t in tokens):
        raise ValueError("本地分词器未产生足量的合法提示词元")
    return config, [tokens[:n] for n in lengths]


def _row_names(lengths):
    return [*(f"prefill_{i}" for i in range(len(lengths))), "generation_prefill",
            *(f"decode_{i}" for i in range(1, DECODE_STEPS + 1))]


def _check_rows(rows, names, vocab_size):
    import torch

    if set(rows) != set(names):
        raise ValueError("全词表分数的请求或步数不符")
    for name in names:
        row = rows[name]
        if row.shape != (vocab_size,) or row.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"{name} 需要长度为 {vocab_size} 的浮点词表，实际为 {tuple(row.shape)} {row.dtype}")


def _tensor_limit(names, vocab_size):
    return min(TENSOR_LIMIT, 8 + HEADER_LIMIT + len(names) * vocab_size * 4)


def _read_bounded(path, limit):
    # 只打开交接目录内的普通文件。先核对同一文件描述符的实际大小，再分配缓冲区。
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError(f"{path.name} 不是大小在 1 至 {limit} 字节内的普通文件")
        data = stream.read(info.st_size + 1)
        if len(data) != info.st_size:
            raise ValueError(f"读取期间 {path.name} 大小发生变化")
        return data


def _replace(path, data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def _candidate_class(path, source, architectures):
    """从已经计算摘要的同一份字节装载候选，仅选择配置声明的本地类。"""
    tree = ast.parse(source, filename=str(path))
    declared = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    spec = importlib.util.spec_from_file_location("nanovllm_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法装载候选 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(tree, str(path), "exec"), vars(module))
    for arch in architectures or ():
        cls = vars(module).get(arch)
        if (arch in declared and isinstance(cls, type)
                and cls.__module__ == module.__name__ and cls.__name__ == cls.__qualname__ == arch):
            return cls
    raise ValueError(f"候选没有在本文件定义 checkpoint.architectures={architectures!r} 中声明的模型类")


def _capture(args, config, sequences, lengths, identity, source, files):
    from safetensors.torch import save
    from nanovllm_selector.port import evaluate as ev

    cls = _candidate_class(args.candidate, source, config.architectures)
    arch = cls.__name__
    eng = ev.engine_run(cls, str(args.model), sequences, arch, max_model_len=max(lengths) + DECODE_STEPS + 64,
                        decode_steps=DECODE_STEPS, tp=1)
    generation = eng["decode"]
    names = _row_names(lengths)
    rows = dict(zip(names, [*eng["prefill"], generation["prefill"], *generation["logits"]], strict=True))
    _check_rows(rows, names, config.vocab_size)
    prompts = [*sequences, sequences[-1]]
    if len(eng["prefill"]) != len(lengths) or generation["ids"] != prompts[-1] + generation["generated"]:
        raise ValueError("引擎返回的 prefill 数量或生成历史不符")
    sampled = [[t] for t in eng["prefill_generated"]] + [generation["generated"]]
    requests = [{"prompt_ids": prompt, "cached_prefix_tokens": hit, "generated_ids": ids}
                for prompt, hit, ids in zip(prompts, eng["prefix_lengths"], sampled, strict=True)]
    limit = _tensor_limit(names, config.vocab_size)
    if sum(t.numel() * t.element_size() for t in rows.values()) > limit:
        raise ValueError("引擎词表分数超出交接文件上限")
    data = save(rows, metadata={**identity, "architecture": arch})
    meta = {"version": 2, **identity, "architecture": arch, "holdout": args.holdout,
            "lengths": lengths, "decode_steps": DECODE_STEPS,
            "requests": requests, "prefix_reuse": eng["prefix_reuse"],
            "logits_sha256": hashlib.sha256(data).hexdigest()}
    metadata = json.dumps(meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(metadata) > META_LIMIT or len(data) > limit:
        raise ValueError("引擎交接文件超出大小上限")
    _check_snapshot(args.model, files)
    args.trial.mkdir(parents=True, exist_ok=True)
    _replace(args.trial / "engine.safetensors", data)
    _replace(args.trial / "engine.json", metadata)
    return {"engine_json_sha256": hashlib.sha256(metadata).hexdigest(), "logits_sha256": meta["logits_sha256"],
            "architecture": arch}


def _ints(items, n):
    return isinstance(items, list) and len(items) == n and all(type(x) is int for x in items)


class _BindingError(ValueError):
    """交接身份不一致，该试验不能归属于待验候选。"""


def _require_fields(actual, expected):
    if not isinstance(actual, dict):
        raise _BindingError("缺少交接身份字段")
    for key, value in expected.items():
        field = actual.get(key)
        if type(field) is not type(value) or field != value:
            raise _BindingError(f"交接字段 {key} 与可信输入不符")


def _read_trial(args, sequences, lengths, config, identity):
    import torch
    from safetensors.torch import load
    from nanovllm_selector.port import evaluate as ev
    from .evidence import json_data

    vocab_size = config.vocab_size
    metadata = _read_bounded(args.trial / "engine.json", META_LIMIT)
    engine_json_sha = hashlib.sha256(metadata).hexdigest()
    meta = json_data(metadata)
    _require_fields(meta, {"version": 2, **identity, "holdout": args.holdout})
    arch = meta.get("architecture")
    if not isinstance(arch, str) or arch not in (config.architectures or ()):
        raise _BindingError("交接架构与 checkpoint.architectures 不符")
    authority = None
    if args.binding is not None:
        authority = json_data(_read_bounded(args.binding, META_LIMIT))
        trusted = {"version": 1, "candidate_sha256": identity["candidate_sha256"],
                   "checkpoint_sha256": identity["checkpoint_sha256"], "holdout": args.holdout,
                   "engine_json_sha256": engine_json_sha}
        _require_fields(authority, trusted)
        if set(authority) != {*trusted, "logits_sha256"}:
            raise _BindingError("宿主 binding.json 的字段集合不符")
    if (not _ints(meta["lengths"], len(lengths)) or meta["lengths"] != lengths
            or type(meta["decode_steps"]) is not int or meta["decode_steps"] != DECODE_STEPS):
        raise ValueError("试验版本、长度或生成步数与正式用例不符")
    prompts = [*sequences, sequences[-1]]
    requests = meta["requests"]
    if not isinstance(requests, list) or len(requests) != len(prompts):
        raise ValueError("引擎请求数量不符")
    policy = meta["prefix_reuse"]
    if (type(policy["enabled"]) is not bool or type(policy["block_size"]) is not int
            or policy["block_size"] != BLOCK_SIZE):
        raise ValueError("引擎前缀复用设置或页大小与正式引擎不符")
    policy = {"enabled": policy["enabled"], "block_size": BLOCK_SIZE, "reason": str(policy.get("reason", ""))[:240]}
    hits, sampled = [], []
    for i, (prompt, req) in enumerate(zip(prompts, requests, strict=True)):
        if not _ints(req["prompt_ids"], len(prompt)) or req["prompt_ids"] != prompt:
            raise ValueError(f"第 {i + 1} 个请求与本地分词器重建的提示不符")
        hit = req["cached_prefix_tokens"]
        if type(hit) is not int:
            raise ValueError("命中词元数应为整数")
        hits.append(hit)
        ids = req["generated_ids"]
        if not _ints(ids, 1 if i < len(lengths) else DECODE_STEPS + 1) or any(t < 0 or t >= vocab_size for t in ids):
            raise ValueError(f"第 {i + 1} 个请求采样词元数或取值不合法")
        sampled.extend(ids)
    ev.check_prefix_hits(prompts, hits, policy)
    names = _row_names(lengths)
    data = _read_bounded(args.trial / "engine.safetensors", _tensor_limit(names, vocab_size))
    size = int.from_bytes(data[:8], "little")
    if not 0 < size <= min(HEADER_LIMIT, len(data) - 8):
        raise ValueError("safetensors 头部长度不合法")
    artifacts = {"logits_sha256": hashlib.sha256(data).hexdigest(), "engine_json_sha256": engine_json_sha}
    _require_fields(meta, {"logits_sha256": artifacts["logits_sha256"]})
    if authority is not None:
        _require_fields(authority, artifacts)
    _require_fields(json_data(data[8:8 + size]).get("__metadata__"), {**identity, "architecture": arch})
    rows = load(data)
    _check_rows(rows, names, vocab_size)
    for name, token in zip(names, sampled, strict=True):
        row = rows[name]
        if not torch.isfinite(row).all() or row[token] != row.max():
            raise ValueError(f"{name} 包含非有限分数或实际采样词元 {token} 不是最高分词元")
    return rows, hits, policy, requests[-1]["generated_ids"], {**artifacts, "architecture": arch}


def _reference(args, config, sequences, lengths, identity, files, report):
    from nanovllm_selector.port import evaluate as ev

    report.update(stage="trial", outcome="candidate_failed")
    rows, hits, policy, generated, artifacts = _read_trial(args, sequences, lengths, config, identity)
    report.update(stage="reference", outcome="reference_unreliable", cached_prefix_tokens=hits,
                  prefix_reuse=policy, generated_ids=generated, reference={}, **artifacts)
    prefill, decode = ev.reference_steps(str(args.model), sequences, hits, generated, DECODE_STEPS,
                                       diagnostics=report["reference"])
    _check_snapshot(args.model, files)
    if len(prefill) != len(lengths) + 1 or len(decode) != DECODE_STEPS:
        raise ValueError("HF 全词表分数的请求或步数不符")
    names = _row_names(lengths)
    expected = dict(zip(names, [*prefill, *decode], strict=True))
    _check_rows(expected, names, config.vocab_size)
    report.update(stage="compare", outcome="candidate_failed")
    if report["reference"]["prefix_strategy"] == "full_kv_snapshot" and not policy["enabled"]:
        ev.check_prefix_hits([*sequences, sequences[-1]], hits, policy, require_kv_hits=True)
    items = [ev.compare_logits(rows[name], expected[name], exact=True) for name in names]
    comparisons = items[:len(lengths)]
    for item, length in zip(comparisons, lengths, strict=True):
        item["length"] = length
    steps = items[len(lengths) + 1:]
    for i, item in enumerate(steps):
        item.update(step=i + 1, position=max(lengths) + i, input_token=generated[i])
    matched = sum(item["ok"] for item in items)
    report.update(comparisons=comparisons, generation_prefill=items[len(lengths)], matched=matched,
                  decode={"steps": len(steps), "matched": sum(item["ok"] for item in steps),
                          "expected_steps": DECODE_STEPS, "items": steps, "ok": all(item["ok"] for item in steps)})
    report["ok"] = matched == report["total"]
    if report["ok"]:
        report["outcome"] = "passed"
    else:
        failed = [name for name, item in zip(names, items, strict=True) if not item["ok"]]
        report["error"] = "全词表分数不一致：" + "、".join(failed)


def _candidate_input(args):
    digest = args.candidate_sha256
    if digest is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("宿主提供的候选 SHA256 格式不合法")
        digest = digest.lower()
    source = None
    if args.candidate is not None:
        source = _read_bounded(args.candidate, 2 << 20)
        actual = hashlib.sha256(source).hexdigest()
        if digest is not None and digest != actual:
            raise _BindingError("宿主 candidate_sha256 与候选文件不符")
        digest = actual
    if args.phase == "engine" and source is None:
        raise ValueError("engine 阶段需要 --candidate")
    if digest is None:
        raise _BindingError("reference 阶段需要可信的 --candidate 或 --candidate-sha256")
    return digest, source


def _engine_error(exc, candidate):
    trace = traceback.TracebackException.from_exception(exc)
    tail = "".join(trace.format()).rstrip()[-1800:]
    if candidate is not None:
        for frame in reversed(trace.stack):
            if Path(frame.filename).absolute() == candidate.absolute():
                location = "".join(traceback.format_list([frame])).rstrip()
                if location not in tail:
                    prefix = location[:500] + "\n…\n"
                    tail = prefix + tail[-(1800 - len(prefix)):]
                break
    return tail


def main(argv=None):
    import torch
    from nanovllm_selector.port import evaluate as ev

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("engine", "reference"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--candidate", type=Path, help="引擎装载的候选；参照仅计算此文件的摘要")
    parser.add_argument("--candidate-sha256", help="宿主从冻结候选计算的 SHA256")
    parser.add_argument("--binding", type=Path, help="参照阶段由宿主只读提供的交接摘要文件")
    parser.add_argument("--holdout", action="store_true", help="改用独立文本和 255 词元提示")
    args = parser.parse_args(argv)
    lengths = [255] if args.holdout else list(ev.DEFAULT_LENGTHS)
    text = HOLDOUT_TEXT if args.holdout else ev.FILLER_TEXT
    report = {"ok": False, "stage": "input", "outcome": "evaluation_error"}
    if args.phase == "reference":
        report.update(criterion="exact", tensor_parallel_size=1, lengths=lengths, holdout=args.holdout,
                      matched=0, total=len(lengths) + 1 + DECODE_STEPS,
                      prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    previous = torch.get_deterministic_debug_mode()
    with redirect_stdout(sys.stderr):
        try:
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            os.environ.pop("NANOVLLM_MODEL_FILE", None)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(0)
            digest, source = _candidate_input(args)
            report.update(candidate_sha256=digest, holdout=args.holdout)
            if args.phase == "engine" and args.binding is not None:
                raise ValueError("--binding 只适用于 reference 阶段")
            files = _checkpoint_files(args.model)
            checkpoint_digest = _checkpoint_digest(files)
            config, sequences = _inputs(args.model, text, lengths)
            # 请求包含正式模式、完整词元、每个请求实际生成的数量和固定推理设置。
            request = {"holdout": args.holdout, "text": text, "prefill_ids": sequences,
                       "generation_prompt_ids": sequences[-1], "decode_steps": DECODE_STEPS, "lengths": lengths,
                       "prefill_max_tokens": 1, "seed": 0, "tensor_parallel_size": 1, "block_size": BLOCK_SIZE}
            identity = {"candidate_sha256": digest, "checkpoint_sha256": checkpoint_digest,
                        "request_sha256": _json_digest(request)}
            report.update(identity)
            if args.phase == "engine":
                os.environ.update(GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo")
                report.update(stage="engine", outcome="candidate_failed")
                artifacts = _capture(args, config, sequences, lengths, identity, source, files)
                report.update(ok=True, outcome="captured", **artifacts)
            else:
                report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
                _reference(args, config, sequences, lengths, identity, files, report)
        except Exception as exc:
            if isinstance(exc, _BindingError):
                report["outcome"] = "evaluation_error"
            report["error"] = (_engine_error(exc, args.candidate) if args.phase == "engine"
                               else f"{type(exc).__name__}: {exc}"[:2000])
        finally:
            torch.set_deterministic_debug_mode(previous)
    print(json.dumps(report, separators=(",", ":"), ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
