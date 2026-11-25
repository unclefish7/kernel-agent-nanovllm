"""单卡正式验收。engine 运行候选，reference 只读取交接文件并从本地 HF 权重重算。"""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import NamedTemporaryFile
import traceback

import torch
from safetensors.torch import load, save

from nanovllm_selector.port import evaluate as ev


DECODE_STEPS = 8
BLOCK_SIZE = 256  # 正式验收使用 nano-vLLM Config 的默认页大小。
HOLDOUT_TEXT = "".join(
    f"Station {i} recorded the river level before sunrise. 午班核对水位计并抄录当日的读数。\n"
    for i in range(64)
)
META_LIMIT = 64 * 1024
HEADER_LIMIT = 16 * 1024
TENSOR_LIMIT = 64 * 1024 * 1024
LOGIT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


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
    if set(rows) != set(names):
        raise ValueError("全词表分数的请求或步数不符")
    for name in names:
        row = rows[name]
        if row.shape != (vocab_size,) or row.dtype not in LOGIT_DTYPES:
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


def _capture(args, config, sequences, lengths):
    arch = config.architectures[0]
    if args.candidate is None:
        raise ValueError("engine 阶段需要 --candidate")
    cls = ev.pick_model_class(ev.load_candidate(args.candidate), arch)
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
    data = save(rows)
    meta = {"version": 1, "lengths": lengths, "decode_steps": DECODE_STEPS,
            "requests": requests, "prefix_reuse": eng["prefix_reuse"],
            "logits_sha256": hashlib.sha256(data).hexdigest()}
    metadata = json.dumps(meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(metadata) > META_LIMIT or len(data) > limit:
        raise ValueError("引擎交接文件超出大小上限")
    args.trial.mkdir(parents=True, exist_ok=True)
    _replace(args.trial / "engine.safetensors", data)
    _replace(args.trial / "engine.json", metadata)


def _ints(items, n):
    return isinstance(items, list) and len(items) == n and all(type(x) is int for x in items)


def _read_trial(trial, sequences, lengths, vocab_size):
    meta = json.loads(_read_bounded(trial / "engine.json", META_LIMIT))
    if (type(meta["version"]) is not int or meta["version"] != 1
            or not _ints(meta["lengths"], len(lengths)) or meta["lengths"] != lengths
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
    data = _read_bounded(trial / "engine.safetensors", _tensor_limit(names, vocab_size))
    if len(data) < 8 or int.from_bytes(data[:8], "little") > HEADER_LIMIT:
        raise ValueError("safetensors 头部长度不合法")
    if hashlib.sha256(data).hexdigest() != meta["logits_sha256"]:
        raise ValueError("元数据与词表分数文件的摘要不符")
    rows = load(data)
    _check_rows(rows, names, vocab_size)
    for name, token in zip(names, sampled, strict=True):
        row = rows[name]
        if not torch.isfinite(row).all() or row[token] != row.max():
            raise ValueError(f"{name} 包含非有限分数或实际采样词元 {token} 不是最高分词元")
    return rows, hits, policy, requests[-1]["generated_ids"]


def _reference(args, config, sequences, lengths, report):
    report.update(stage="trial", outcome="candidate_failed")
    rows, hits, policy, generated = _read_trial(args.trial, sequences, lengths, config.vocab_size)
    report.update(stage="reference", outcome="reference_unreliable", cached_prefix_tokens=hits,
                  prefix_reuse=policy, generated_ids=generated, reference={})
    prefill, decode = ev.reference_steps(str(args.model), sequences, hits, generated, DECODE_STEPS,
                                       diagnostics=report["reference"])
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


def _candidate_digest(args):
    digest = args.candidate_sha256
    if digest is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("宿主提供的候选 SHA256 格式不合法")
        digest = digest.lower()
    if args.candidate is not None:
        sha = hashlib.sha256()
        with args.candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                sha.update(chunk)
        if digest is not None and digest != sha.hexdigest():
            raise ValueError("宿主提供的摘要与候选文件不符")
        digest = sha.hexdigest()
    return digest


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("engine", "reference"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--candidate", type=Path, help="引擎装载的候选；参照仅计算此文件的摘要")
    parser.add_argument("--candidate-sha256", help="宿主从冻结候选计算的 SHA256")
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
            digest = _candidate_digest(args)
            if digest is not None:
                report["candidate_sha256"] = digest
            config, sequences = _inputs(args.model, text, lengths)
            if args.phase == "engine":
                os.environ.update(GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo")
                report.update(stage="engine", outcome="candidate_failed")
                _capture(args, config, sequences, lengths)
                report.update(ok=True, outcome="captured")
            else:
                report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
                _reference(args, config, sequences, lengths, report)
        except Exception as exc:
            report["error"] = (_engine_error(exc, args.candidate) if args.phase == "engine"
                               else f"{type(exc).__name__}: {exc}"[:2000])
        finally:
            torch.set_deterministic_debug_mode(previous)
    print(json.dumps(report, separators=(",", ":"), ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
