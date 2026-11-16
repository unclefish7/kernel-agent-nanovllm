"""在引擎里跑一遍候选模型文件，与 transformers 的 logits 对比，输出 JSON 报告。

调用方（iterate.py）用子进程运行本模块：一次进程只评测一个候选文件。
这样候选代码里的语法错误、导入错误、崩溃、显存泄漏都不会污染上一轮的状态，
调用方只需要读报告，不需要理解 torch。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

DEFAULT_LENGTHS = (128, 256, 768)
FILLER_TEXT = "The quick brown fox jumps over the lazy dog. " * 400


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
                   max_abs_tol: float = 2.0, min_cosine: float = 0.9990) -> dict:
    """两个 [vocab] 的 next-token logits 对比。"""
    a = actual.detach().float().flatten()
    b = expected.detach().float().flatten()
    if a.shape != b.shape:
        return {"ok": False, "reason": f"形状不一致：{tuple(a.shape)} vs {tuple(b.shape)}"}
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    argmax_match = int(a.argmax()) == int(b.argmax())
    report = {
        "argmax_actual": int(a.argmax()),
        "argmax_expected": int(b.argmax()),
        "argmax_match": argmax_match,
        "max_abs_diff": max_abs,
        "cosine": cosine,
        "top5_actual": a.topk(5).indices.tolist(),
        "top5_expected": b.topk(5).indices.tolist(),
    }
    report["ok"] = bool(argmax_match and max_abs <= max_abs_tol and cosine >= min_cosine)
    return report


def reference_logits_multi(model_dir: str, sequences: list[list[int]]) -> list[torch.Tensor]:
    """一次前向取多个位置的 logits（因果掩码保证：前缀的末位 logits 与单独跑前缀相同）。"""
    from transformers import AutoModelForCausalLM

    longest = max(len(s) for s in sequences)
    ids = [[s for s in sequences if len(s) == longest][0]]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model = model.cuda().eval()
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor(ids, device="cuda")).logits[0]
    out = [logits[len(s) - 1].detach().float().cpu().clone() for s in sequences]
    del model, logits
    torch.cuda.empty_cache()
    return out


def engine_logits(cls, model_dir: str, sequences: list[list[int]], arch: str,
                  max_model_len: int = 1024) -> list[torch.Tensor]:
    """把候选类挂进注册表，走真实引擎；每个长度跑一次 prefill，抓那一次的第一份 logits。"""
    import atexit

    import nanovllm.models as models
    from nanovllm import LLM
    from nanovllm.sampling_params import SamplingParams

    models.ARCH[arch] = cls
    captured: list[torch.Tensor] = []
    original = cls.compute_logits

    def patched(self, hidden_states):
        out = original(self, hidden_states)
        if not captured:
            captured.append(out.detach().float().cpu().clone())
        return out

    cls.compute_logits = patched
    try:
        llm = LLM(
            model_dir,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=0.35,
        )
        try:
            results = []
            for ids in sequences:
                # 构造时的 warmup 和上一条都会调 compute_logits，先清空；
                # 清空后取到的第一份，就是这条输入 prefill 阶段的输出。
                captured.clear()
                llm.generate(
                    [ids],
                    SamplingParams(temperature=1.0, max_tokens=1),
                    use_tqdm=False,
                )
                if not captured:
                    raise RuntimeError(f"长度 {len(ids)} 的 prefill 没有捕获到 logits")
                results.append(captured[0][0])
        finally:
            atexit.unregister(llm.exit)
            llm.exit()
    finally:
        cls.compute_logits = original
    return results


def evaluate(model_dir: str, candidate: Path, arch: str | None = None,
             lengths: tuple[int, ...] = DEFAULT_LENGTHS, **kwargs: Any) -> dict:
    """完整评测一个候选文件；任何阶段失败都折算成报告，不抛异常。"""
    started = time.time()
    report: dict[str, Any] = {
        "ok": False, "stage": "load", "candidate": str(candidate),
        "lengths": list(lengths),
    }

    try:
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

    report["stage"] = "run"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        base = tokenizer(FILLER_TEXT, return_tensors="pt").input_ids[0].tolist()
        longest = max(lengths)
        if len(base) < longest:
            raise RuntimeError(f"填充文本只有 {len(base)} 个 token，不够 {longest}")
        sequences = [base[:length] for length in lengths]
        expected = reference_logits_multi(model_dir, sequences)
        actual = engine_logits(
            cls, model_dir, sequences, arch or cls.__name__, max_model_len=longest + 64
        )
    except Exception:
        report["error"] = traceback.format_exc()[-6000:]
        return _finish(report, started)

    report["stage"] = "compare"
    comparisons = []
    for length, got, want in zip(lengths, actual, expected):
        item = compare_logits(got, want)
        item["length"] = length
        comparisons.append(item)
    report["comparisons"] = comparisons
    report["ok"] = all(item.get("ok") for item in comparisons)
    failed = [item for item in comparisons if not item.get("ok")]
    report["comparison"] = failed[0] if failed else comparisons[0]
    if failed:
        report["error"] = "；".join(
            f"长度 {item['length']}：argmax 相同={item.get('argmax_match')}，"
            f"最大绝对误差={item.get('max_abs_diff', float('nan')):.4f}，"
            f"余弦相似度={item.get('cosine', float('nan')):.6f}"
            for item in failed
        )
    return _finish(report, started)


def _finish(report: dict, started: float) -> dict:
    report["elapsed"] = round(time.time() - started, 2)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在引擎里评测候选模型文件")
    parser.add_argument("--model", required=True, help="HF 模型目录（含权重）")
    parser.add_argument("--candidate", required=True, type=Path, help="候选 .py 文件")
    parser.add_argument("--arch", default=None, help="期望的架构名（architectures[0]）")
    parser.add_argument("--lengths", default="128,256,768", help="要对比多少个 token 位置，逗号分隔")
    parser.add_argument("--report", type=Path, default=None, help="把 JSON 报告写到这个文件")
    args = parser.parse_args(argv)

    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    report = evaluate(args.model, args.candidate, args.arch, lengths)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
