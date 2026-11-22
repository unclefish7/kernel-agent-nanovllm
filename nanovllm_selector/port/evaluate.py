"""在引擎里跑一遍候选模型文件，与 transformers 的 logits 对比，输出 JSON 报告。

调用方（iterate.py）用子进程运行本模块：一次进程只评测一个候选文件。
这样候选代码里的语法错误、导入错误、崩溃、显存泄漏都不会污染上一轮的状态，
调用方只需要读报告，不需要理解 torch。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

DEFAULT_LENGTHS = (128, 256, 768)
FILLER_TEXT = "The quick brown fox jumps over the lazy dog. " * 400
# 引擎的 SamplingParams 有 assert temperature > 1e-10（不允许贪心），这个值数值上等价于贪心
GREEDY_TEMPERATURE = 1e-8


def load_candidate(path: Path):
    """按文件路径导入候选模块（不要求它在 sys.path 上）。"""
    spec = importlib.util.spec_from_file_location("nanovllm_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法按路径导入 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nanovllm_candidate"] = module
    spec.loader.exec_module(module)
    return module


def pick_model_class(module, arch: str | None = None) -> type:
    """在候选模块里找模型类：优先本文件定义的 `*ForCausalLM`。"""
    candidates = []
    for name, value in vars(module).items():
        if isinstance(value, type) and name.endswith("ForCausalLM"):
            candidates.append((getattr(value, "__module__", "") == module.__name__, name, value))
    if arch:
        exact = [c for c in candidates if c[1] == arch]
        if exact:
            return exact[0][2]
    if not candidates:
        raise LookupError(
            f"模块里没有以 ForCausalLM 结尾的类；顶层有 {sorted(vars(module))[:20]}"
        )
    pool = [c for c in candidates if c[0]] or candidates
    # 去掉"只是别的候选的父类"的那些（基类通常也叫 *ForCausalLM）
    pool = [
        c for c in pool
        if not any(issubclass(o[2], c[2]) and o[2] is not c[2] for o in pool)
    ]
    pool.sort(key=lambda c: c[1])
    return pool[0][2]


def compare_logits(actual: torch.Tensor, expected: torch.Tensor,
                   max_abs_tol: float = 2.0, min_cosine: float = 0.9990, *, exact: bool = False) -> dict:
    """两个 [vocab] 的 next-token logits 对比。"""
    a = actual.detach().float()
    b = expected.detach().float()
    if a.shape != b.shape or a.ndim != 1 or a.numel() == 0:
        return {"ok": False, "reason": f"形状不一致：{tuple(a.shape)} vs {tuple(b.shape)}"}
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"ok": False, "exact": False, "reason": "词表分数含 NaN 或无穷大"}
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    argmax_match = int(a.argmax()) == int(b.argmax())
    report = {
        "argmax_actual": int(a.argmax()),
        "argmax_expected": int(b.argmax()),
        "argmax_match": argmax_match,
        "max_abs_diff": max_abs,
        "cosine": cosine,
        "vocab_size": a.numel(),
        "exact": torch.equal(a, b),
        "top5_actual": a.topk(min(5, a.numel())).indices.tolist(),
        "top5_expected": b.topk(min(5, b.numel())).indices.tolist(),
    }
    if exact:
        report["ok"] = report["exact"]
    else:
        report["ok"] = bool(argmax_match and max_abs <= max_abs_tol and cosine >= min_cosine)
    return report


def reference_logits(model_dir: str, ids: list[int],
                     positions: list[int]) -> list[torch.Tensor]:
    """一次前向取多个位置的 logits（因果掩码保证：位置 p 的 logits 只取决于它之前的内容）。"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model = model.cuda().eval()
    try:
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device="cuda")).logits[0]
        return [logits[p].detach().float().cpu().clone() for p in positions]
    finally:
        del model
        torch.cuda.empty_cache()


def reference_logits_multi(model_dir: str, sequences: list[list[int]]) -> list[torch.Tensor]:
    """只有同一提示的真实前缀才能共享一次前向。"""
    if not sequences or any(not seq for seq in sequences):
        raise ValueError("HF 参照需要非空提示")
    longest = max(sequences, key=len)
    if any(seq != longest[:len(seq)] for seq in sequences):
        return [reference_logits(model_dir, seq, [len(seq) - 1])[0] for seq in sequences]
    return reference_logits(model_dir, longest, [len(s) - 1 for s in sequences])


def _hf_consistency(actual, expected, dtype):
    """不同输入形状的 HF 自检允许舍入误差。此容差只用于参照内部。"""
    tolerance = max(1e-4, 4 * torch.finfo(dtype).eps)
    if actual.shape != expected.shape:
        return {"ok": False, "reason": f"HF 输出形状不同 {actual.shape} / {expected.shape}"}
    a, b = actual.float(), expected.float()
    delta = (a - b).abs().max().item()
    limit = tolerance * max(b.abs().max().item(), 1e-6)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"ok": finite and delta <= limit, "max_abs_diff": delta if finite else None,
            "limit": limit if finite else None}


def _full_kv(cache):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    return (type(cache) is DynamicCache and bool(cache.layers)
            and vars(cache).keys() <= {"layers", "layer_class_to_replicate", "offloading"}
            and all(type(layer) is DynamicLayer and vars(layer).keys() <= {
                "keys", "values", "is_initialized", "dtype", "device"
            } for layer in cache.layers))


def _kv_snapshot(prior, history_length, prefix):
    """保留原来分段计算的逐位置 K/V；严格限制为没有额外状态的完整 KV 层。"""
    if not _full_kv(prior):
        raise ValueError("HF 命中来源含有无法准确快照的状态")
    snapshot = copy.deepcopy(prior)
    for layer in snapshot.layers:
        k, v = layer.keys, layer.values
        if (not layer.is_initialized or k.ndim != 4 or v.ndim != 4
                or k.shape[:3] != v.shape[:3] or k.size(-2) != history_length):
            raise ValueError("HF 完整 KV 页长度或形状与来源历史不符")
        layer.keys = k[..., :prefix, :]
        layer.values = v[..., :prefix, :]
    return snapshot


def _probe_reuse(call, tokens, info, model_dtype):
    """在独立短前缀上一次只改变一个条件。探针缓存与验收缓存互不共享。"""
    n = len(tokens) - 2
    if n < 1:
        for name in ("crop", "multi_token"):
            info[name] = {"ok": None, "reason": "探针至少需要三个词元"}
        return
    prefix, suffix = tokens[:n], tokens[n:]
    def clean_first():
        return call(suffix[:1], call(prefix).past_key_values)

    try:
        cropped = call(tokens).past_key_values
        cropped.crop(n)
        resumed = call(suffix[:1], cropped)
        clean = clean_first().logits[0, -1]
        info["crop"] = _hf_consistency(resumed.logits[0, -1], clean, model_dtype or clean.dtype)
    except Exception as exc:
        info["crop"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        first = clean_first()
        serial = [first.logits[0, -1].clone()]
        second = call(suffix[1:], first.past_key_values)
        serial.append(second.logits[0, -1])
        together = call(suffix, call(prefix).past_key_values, keep=0)
        info["multi_token"] = _hf_consistency(together.logits[0], torch.stack(serial), model_dtype or serial[0].dtype)
    except Exception as exc:
        info["multi_token"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    info["probe_prefix_tokens"] = n


@torch.inference_mode()
def reference_trace(model, prompts, prefix_lengths, continuation, diagnostics):
    """按请求顺序计算各提示末位、最后一个提示的逐词续写。仅接受词元与命中长度。"""
    if not prompts or len(prefix_lengths) != len(prompts):
        raise ValueError("前缀长度记录与实际请求数不符")
    supports_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    model_dtype = getattr(model, "dtype", None)

    def call(ids, past=None, *, keep=1, use_cache=True):
        options = {"logits_to_keep": keep} if supports_keep else {}
        return model(input_ids=torch.tensor([ids], device=model.device), past_key_values=past,
                     use_cache=use_cache, **options)

    def score(output):
        row = output.logits[0, -1]
        if row.ndim != 1 or not row.numel() or not torch.isfinite(row).all():
            raise ValueError("HF 未返回有限的完整词表分数")
        return row.detach().float().cpu().clone()

    def check_history(out, history):
        full = call(history, use_cache=False)
        check = _hf_consistency(out.logits[0, -1], full.logits[0, -1], model_dtype or out.logits.dtype)
        if not check["ok"]:
            raise ValueError(f"HF 逐词缓存与从头计算在长度 {len(history)} 处不一致：{check}")

    sample = max(prompts, key=len)[:8]
    cache = call(sample).past_key_values
    if cache is None and (continuation or any(prefix_lengths)):
        raise ValueError("HF 未返回续写所需的历史缓存")
    full_kv = _full_kv(cache)
    diagnostics.update(cache_class=type(cache).__name__,
                       prefix_strategy="full_kv_snapshot" if full_kv else "rebuild_and_step",
                       reason="逐层完整 DynamicLayer 键值历史" if full_kv else "缓存含其他或未知状态，独立重建目标前缀")
    _probe_reuse(call, sample, diagnostics, model_dtype)
    if full_kv and any(diagnostics[k]["ok"] is False for k in ("crop", "multi_token")):
        diagnostics["reason"] = ("跨形状探针分数不一致。按已验证的完整 KV 结构复制原分段产生的逐位置键值，"
                                 "保持实际命中的输入形状进行 HF 评分")
    saved = []
    prefill, decode = [], []
    for seq, prefix in zip(prompts, prefix_lengths):
        if not 0 <= prefix < len(seq):
            raise ValueError(f"非法前缀长度 {prefix}，本次提示长度 {len(seq)}")
        if not prefix:
            out = call(seq)
        elif full_kv:
            # 快照保留产生这些页时的实际分段、矩阵形状和舍入。
            prior = next(((tokens, past) for tokens, past in reversed(saved) if tokens[:prefix] == seq[:prefix]), None)
            if prior is None:
                raise ValueError(f"HF 缺少匹配前 {prefix} 个词元的完整 KV 快照")
            tokens, past = prior
            out = call(seq[prefix:], _kv_snapshot(past, len(tokens), prefix))
        else:
            out = call(seq[:prefix])
            for token in seq[prefix:]:
                out = call([token], out.past_key_values)
        if full_kv:
            saved.append((seq, out.past_key_values))
        else:
            check_history(out, seq)
        prefill.append(score(out))
    history = list(prompts[-1])
    for token in continuation:
        history.append(token)
        out = call([token], out.past_key_values)
        if not full_kv:
            check_history(out, history)
        decode.append(score(out))
    return prefill, decode


def reference_steps(model_dir, sequences, prefix_lengths, generated, decode_steps, *, diagnostics=None):
    from transformers import AutoModelForCausalLM

    if len(generated) < decode_steps:
        raise ValueError("生成历史短于要求验证的步数")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    prompts = [*sequences, max(sequences, key=len)] if decode_steps else sequences
    return reference_trace(model, prompts, prefix_lengths, generated[:decode_steps],
                           {} if diagnostics is None else diagnostics)


def engine_prefix_reuse(scheduler):
    manager = scheduler.block_manager
    enabled = manager is not None and getattr(manager, "reuse_prefix", True)
    reason = getattr(scheduler, "prefix_cache_disabled_reason", None)
    return {"enabled": enabled, "block_size": scheduler.block_size,
            "reason": reason or ("完整 KV 页可按前缀复用" if enabled else "无可复用的 KV 页")}


def check_prefix_hits(prompts, lengths, policy, *, require_kv_hits=False):
    """检查串行请求的观测值。上一请求的共同完整页仍应可用。"""
    if len(lengths) != len(prompts):
        raise ValueError("引擎记录的前缀长度数量与请求数不同")
    for i, (tokens, hit) in enumerate(zip(prompts, lengths)):
        if not 0 <= hit < len(tokens):
            raise ValueError(f"引擎报告非法命中长度 {hit}，提示长度 {len(tokens)}")
        if not policy["enabled"]:
            if hit:
                raise ValueError(f"引擎已禁用前缀复用但报告命中 {hit} 个词元")
            if not require_kv_hits:
                continue
        block = policy["block_size"]
        if hit % block or (hit and not any(p[:hit] == tokens[:hit] for p in prompts[:i])):
            raise ValueError(f"引擎报告的 {hit} 个已缓存词元没有匹配的完整页")
        if i and hit + block < len(tokens):
            end = hit + block
            if prompts[i - 1][:end] == tokens[:end]:
                raise ValueError(f"共同 KV 页仅命中 {hit} 个词元，至少应命中 {end}")


def engine_run(cls, model_dir: str, sequences: list[list[int]], arch: str,
               max_model_len: int = 1024, decode_steps: int = 0,
               tp: int = 1) -> dict:
    """把候选类挂进注册表，走真实引擎。

    - 每个长度跑一次 prefill，抓那一份 logits（`prefill`）；
    - 再对最后一条（最长的）输入跑一次生成，抓 prefill 之后的每一步 logits（`decode`），
      也就是真正走 KV cache 读写的那些步。
    """
    import atexit

    import nanovllm.models as models
    from nanovllm import LLM
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import get_context

    models.ARCH[arch] = cls
    captured: list[torch.Tensor] = []
    prefix_lengths = []
    original = cls.compute_logits

    def patched(self, hidden_states):
        out = original(self, hidden_states)
        if out.ndim != 2 or out.size(0) != 1:
            raise ValueError(f"单条请求应返回 [1, 词表大小] 的分数，实际为 {tuple(out.shape)}")
        ctx = get_context()
        if ctx.is_prefill:
            prefix_lengths.append(ctx.cu_seqlens_k_list[-1] - ctx.cu_seqlens_q_list[-1])
        captured.append(out.detach().float().cpu().clone())
        return out

    cls.compute_logits = patched
    try:
        llm = LLM(
            model_dir,
            enforce_eager=True,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            if type(llm.model_runner.model) is not cls:
                raise RuntimeError("引擎实际装载的模型类与指定的候选不同")
            prefill: list[torch.Tensor] = []
            prefix_lengths.clear()
            for ids in sequences:
                # 构造时的 warmup、上一条请求都会调 compute_logits，先清空；
                # 清空后取到的第一份，就是这条输入 prefill 阶段的输出。
                captured.clear()
                llm.generate(
                    [ids],
                    SamplingParams(temperature=GREEDY_TEMPERATURE, max_tokens=1),
                    use_tqdm=False,
                )
                if len(captured) != 1:
                    raise RuntimeError(f"长度 {len(ids)} 的 prefill 捕获了 {len(captured)} 份 logits")
                prefill.append(captured[0][0])

            decode: dict[str, Any] = {"logits": [], "ids": []}
            if decode_steps > 0:
                seq = max(sequences, key=len)
                captured.clear()
                out = llm.generate(
                    [seq],
                    # 生成 N 个 token 需要 N 份分布：1 份来自 prefill，N-1 份来自 decode
                    SamplingParams(
                        temperature=GREEDY_TEMPERATURE, max_tokens=decode_steps + 1, ignore_eos=True
                    ),
                    use_tqdm=False,
                )
                if len(captured) != decode_steps + 1:
                    raise RuntimeError(
                        f"期望抓 {decode_steps + 1} 份 logits，实际 {len(captured)} 份"
                    )
                generated = list(out[0]["token_ids"])
                if len(generated) != len(captured):
                    raise RuntimeError("生成的词元数量与捕获的分数数量不同")
                decode["prefill"] = captured[0][0]
                # generate() 返回的 token_ids 只有补全部分，补回前缀才是完整序列
                decode["ids"] = list(seq) + generated
                decode["generated"] = generated
                decode["logits"] = [item[0] for item in captured[1:decode_steps + 1]]
            return {"prefill": prefill, "decode": decode, "prefix_lengths": prefix_lengths,
                    "prefix_reuse": engine_prefix_reuse(llm.scheduler)}
        finally:
            atexit.unregister(llm.exit)
            llm.exit()
    finally:
        cls.compute_logits = original


def engine_logits(cls, model_dir: str, sequences: list[list[int]], arch: str,
                  max_model_len: int = 1024, tp: int = 1) -> list[torch.Tensor]:
    """只要 prefill 的 logits（demo 的逐位置比对仍然用它）。"""
    return engine_run(cls, model_dir, sequences, arch, max_model_len, tp=tp)["prefill"]


def evaluate(model_dir: str, candidate: Path, arch: str | None = None,
             lengths: tuple[int, ...] = DEFAULT_LENGTHS, decode_steps: int = 8,
             tp: int = 1, exact: bool = True, logits_out: Path | None = None,
             prompt_file: Path | None = None, **kwargs: Any) -> dict:
    """完整评测一个候选文件；任何阶段失败都折算成报告，不抛异常。"""
    started = time.time()
    report: dict[str, Any] = {
        "ok": False, "stage": "load", "candidate": str(candidate),
        "lengths": list(lengths), "tensor_parallel_size": tp,
        "criterion": "exact" if exact else "argmax + max_abs <= 2.0 + cosine >= 0.999",
    }

    try:
        report["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        module = load_candidate(candidate)
    except Exception:
        report["error"] = traceback.format_exc()[-4000:]
        return _finish(report, started)

    report["stage"] = "class"
    try:
        cls = pick_model_class(module, arch)
    except Exception:
        report["error"] = traceback.format_exc()[-4000:]
        return _finish(report, started)

    # tp>1 时引擎用 spawn 起其他 rank，子进程会重新 import nanovllm.models；
    # 它们读这个环境变量重放“候选登记”，否则只认 ARCH 里内置的 Qwen3。
    os.environ["NANOVLLM_MODEL_FILE"] = str(candidate)

    report["stage"] = "input"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        text = prompt_file.read_text(encoding="utf-8") if prompt_file is not None else FILLER_TEXT
        report["prompt_file"] = str(prompt_file) if prompt_file is not None else None
        report["prompt_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        base = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        longest = max(lengths)
        if min(lengths) < 1 or decode_steps < 0:
            raise ValueError("输入长度必须为正，decode_steps 必须非负")
        if len(base) < longest:
            raise ValueError(f"输入文本只有 {len(base)} 个词元，最少需要 {longest} 个")
        sequences = [base[:length] for length in lengths]
        # 先跑引擎（要占显存），再加载 HF 取参照：两份模型不同时驻留
        previous_mode = torch.get_deterministic_debug_mode()
        try:
            if exact:
                torch.use_deterministic_algorithms(True)
            report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
            report["stage"] = "run"
            eng = engine_run(cls, model_dir, sequences, arch or cls.__name__,
                             max_model_len=longest + decode_steps + 64, decode_steps=decode_steps, tp=tp)
            report["prefix_reuse"] = eng["prefix_reuse"]
            report["cached_prefix_tokens"] = eng["prefix_lengths"]
            prompts = sequences + ([max(sequences, key=len)] if decode_steps else [])
            check_prefix_hits(prompts, eng["prefix_lengths"], eng["prefix_reuse"])
            decode_logits = eng["decode"]["logits"]
            if len(eng["prefill"]) != len(lengths) or len(decode_logits) != decode_steps:
                raise ValueError("引擎捕获的分数数量与请求的步数不同")
            report["stage"] = "reference"
            report["reference"] = {}
            expected_prefill, expected_decode = reference_steps(
                model_dir, sequences, eng["prefix_lengths"], eng["decode"].get("generated", []), decode_steps,
                diagnostics=report["reference"],
            )
        finally:
            torch.set_deterministic_debug_mode(previous_mode)
        actual_ids = eng["decode"]["ids"] or max(sequences, key=len)
        start = longest
        generation_reference = expected_prefill.pop() if decode_steps else None
        if len(expected_prefill) != len(lengths) or len(expected_decode) != decode_steps:
            raise ValueError("参照分数数量与请求的步数不同")
        if report["reference"].get("prefix_strategy") == "full_kv_snapshot" and not eng["prefix_reuse"]["enabled"]:
            report["stage"] = "run"
            check_prefix_hits(prompts, eng["prefix_lengths"], eng["prefix_reuse"], require_kv_hits=True)
    except Exception:
        report["error"] = traceback.format_exc()[-6000:]
        return _finish(report, started)

    report["stage"] = "compare"
    comparisons = []
    for length, got, want in zip(lengths, eng["prefill"], expected_prefill):
        item = compare_logits(got, want, exact=exact)
        item["length"] = length
        comparisons.append(item)
    report["comparisons"] = comparisons
    if decode_steps:
        report["generation_prefill"] = compare_logits(
            eng["decode"]["prefill"], generation_reference, exact=exact
        )

    decode_items = []
    for step, (got, want) in enumerate(zip(decode_logits, expected_decode), start=1):
        item = compare_logits(got, want, exact=exact)
        item["step"] = step
        item["position"] = start + step - 1
        item["input_token"] = actual_ids[start + step - 1]
        decode_items.append(item)
    report["decode"] = {
        "steps": len(decode_items),
        "matched": sum(1 for item in decode_items if item.get("ok")),
        "items": decode_items,
        "expected_steps": decode_steps,
        "ok": all(item.get("ok") for item in decode_items),
    }

    failed = [item for item in comparisons if not item.get("ok")]
    failed_decode = [item for item in decode_items if not item.get("ok")]
    report["ok"] = not failed and report["decode"]["ok"] and report.get("generation_prefill", {}).get("ok", True)
    report["comparison"] = failed[0] if failed else comparisons[0]
    if logits_out is not None:
        pairs = [{"phase": "prefill", "length": length, "engine": got, "hf": want}
                 for length, got, want in zip(lengths, eng["prefill"], expected_prefill)]
        if decode_steps:
            pairs.append({"phase": "generation_prefill", "length": longest,
                          "engine": eng["decode"]["prefill"], "hf": generation_reference})
        for i, (got, want) in enumerate(zip(decode_logits, expected_decode), start=1):
            pairs.append({"phase": "decode", "step": i, "input_token": actual_ids[start+i-1],
                          "engine": got, "hf": want})
        path = Path(logits_out)
        pending = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                pending = Path(stream.name)
                torch.save(pairs, stream)
            pending.replace(path)
            report["logits_file"] = str(path)
        except (OSError, RuntimeError) as exc:
            report.update(ok=False, stage="save_logits", error=str(exc))
            return _finish(report, started)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    if failed:
        report["error"] = "；".join(
            f"长度 {item['length']}：argmax 相同={item.get('argmax_match')}，"
            f"最大绝对误差={item.get('max_abs_diff', float('nan')):.4f}，"
            f"余弦相似度={item.get('cosine', float('nan')):.6f}"
            for item in failed
        )
    elif not report.get("generation_prefill", {}).get("ok", True):
        report["error"] = "连续生成的首步分数与参照不符"
    elif failed_decode:
        report["error"] = "decode 第 " + "、".join(
            str(item["step"]) for item in failed_decode[:3]
        ) + " 步不一致：" + "；".join(
            f"第 {item['step']} 步 argmax 相同={item.get('argmax_match')}，"
            f"最大绝对误差={item.get('max_abs_diff', float('nan')):.4f}，"
            f"余弦相似度={item.get('cosine', float('nan')):.6f}"
            for item in failed_decode[:3]
        )
    return _finish(report, started)


def _finish(report: dict, started: float) -> dict:
    report["elapsed"] = round(time.time() - started, 2)
    if report["ok"]:
        report["outcome"] = "passed"
    elif report["stage"] == "reference":
        report["outcome"] = "reference_unreliable"
    elif report["stage"] in ("input", "save_logits"):
        report["outcome"] = "evaluation_error"
    else:
        report["outcome"] = "candidate_failed"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在引擎里评测候选模型文件")
    parser.add_argument("--model", required=True, help="HF 模型目录（含权重）")
    parser.add_argument("--candidate", required=True, type=Path, help="候选 .py 文件")
    parser.add_argument("--arch", default=None, help="期望的架构名（architectures[0]）")
    parser.add_argument("--lengths", default="128,256,768", help="要对比多少个 token 位置，逗号分隔")
    parser.add_argument("--decode-steps", type=int, default=8,
                        help="逐 token 走 KV cache 的步数，0 表示只验 prefill")
    parser.add_argument("--report", type=Path, help="报告路径，默认在项目 outputs 下以候选文件名保存")
    parser.add_argument("--logits-out", type=Path, help="额外保存每一步引擎和 HF 的完整词表分数（.pt）")
    parser.add_argument("--prompt-file", type=Path, help="从指定 UTF-8 文本取输入，用于独立复验")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size，默认 1")
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True,
                        help="默认要求每一步整个词表逐项相等，--no-exact 使用原有容差")
    args = parser.parse_args(argv)

    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    report = evaluate(args.model, args.candidate, args.arch, lengths,
                      decode_steps=args.decode_steps, tp=args.tp, exact=args.exact, logits_out=args.logits_out,
                      prompt_file=args.prompt_file)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    report_path = args.report
    if report_path is None:
        candidate = args.candidate.resolve()
        outputs = Path(__file__).resolve().parents[1] / "outputs"
        directory = candidate.parent if candidate.is_relative_to(outputs) else outputs
        report_path = directory / (candidate.stem + ".eval.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
