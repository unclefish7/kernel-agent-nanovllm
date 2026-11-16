"""在引擎里跑一遍候选模型文件，与 transformers 的 logits 对比，输出 JSON 报告。

调用方（iterate.py）用子进程运行本模块：一次进程只评测一个候选文件。
这样候选代码里的语法错误、导入错误、崩溃、显存泄漏都不会污染上一轮的状态，
调用方只需要读报告，不需要理解 torch。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
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
    """每个序列一条 prompt，取各自最后一位的 logits；拼成最长的那个再按位置取。"""
    longest = max(sequences, key=len)
    return reference_logits(model_dir, longest, [len(s) - 1 for s in sequences])


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

    models.ARCH[arch] = cls
    captured: list[torch.Tensor] = []
    original = cls.compute_logits

    def patched(self, hidden_states):
        out = original(self, hidden_states)
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
            prefill: list[torch.Tensor] = []
            for ids in sequences:
                # 构造时的 warmup、上一条请求都会调 compute_logits，先清空；
                # 清空后取到的第一份，就是这条输入 prefill 阶段的输出。
                captured.clear()
                llm.generate(
                    [ids],
                    SamplingParams(temperature=GREEDY_TEMPERATURE, max_tokens=1),
                    use_tqdm=False,
                )
                if not captured:
                    raise RuntimeError(f"长度 {len(ids)} 的 prefill 没有捕获到 logits")
                prefill.append(captured[0][0])

            decode: dict[str, Any] = {"logits": [], "ids": []}
            if decode_steps > 0:
                seq = sequences[-1]
                captured.clear()
                out = llm.generate(
                    [seq],
                    # 生成 N 个 token 需要 N 份分布：1 份来自 prefill，N-1 份来自 decode
                    SamplingParams(
                        temperature=GREEDY_TEMPERATURE, max_tokens=decode_steps + 1
                    ),
                    use_tqdm=False,
                )
                if len(captured) < decode_steps + 1:
                    raise RuntimeError(
                        f"期望抓 {decode_steps + 1} 份 logits，实际 {len(captured)} 份"
                    )
                generated = list(out[0]["token_ids"])
                # generate() 返回的 token_ids 只有补全部分，补回前缀才是完整序列
                decode["ids"] = list(seq) + generated
                decode["generated"] = generated
                decode["logits"] = [item[0] for item in captured[1:decode_steps + 1]]
            return {"prefill": prefill, "decode": decode}
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
             tp: int = 1, **kwargs: Any) -> dict:
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

    # tp>1 时引擎用 spawn 起其他 rank，子进程会重新 import nanovllm.models；
    # 它们读这个环境变量重放“候选登记”，否则只认 ARCH 里内置的 Qwen3。
    os.environ["NANOVLLM_MODEL_FILE"] = str(candidate)

    report["stage"] = "run"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        base = tokenizer(FILLER_TEXT, return_tensors="pt").input_ids[0].tolist()
        longest = max(lengths)
        need = longest + decode_steps + 1
        if len(base) < need:
            raise RuntimeError(f"填充文本只有 {len(base)} 个 token，不够 {need}")
        sequences = [base[:length] for length in lengths]
        # 先跑引擎（要占显存），再加载 HF 取参照：两份模型不同时驻留
        eng = engine_run(
            cls, model_dir, sequences, arch or cls.__name__,
            max_model_len=longest + decode_steps + 64, decode_steps=decode_steps,
            tp=tp,
        )
        decode_logits = eng["decode"]["logits"]
        actual_ids = eng["decode"]["ids"] or sequences[-1]
        # 引擎第 j 步的 logits 对应"到第 start-1+j 个 token 为止"的上下文
        start = len(sequences[-1])
        positions = [length - 1 for length in lengths]
        positions += [start - 1 + step for step in range(1, len(decode_logits) + 1)]
        expected = reference_logits(model_dir, actual_ids, positions)
        expected_prefill = expected[:len(lengths)]
        expected_decode = expected[len(lengths):]
    except Exception:
        report["error"] = traceback.format_exc()[-6000:]
        return _finish(report, started)

    report["stage"] = "compare"
    comparisons = []
    for length, got, want in zip(lengths, eng["prefill"], expected_prefill):
        item = compare_logits(got, want)
        item["length"] = length
        comparisons.append(item)
    report["comparisons"] = comparisons

    decode_items = []
    for step, (got, want) in enumerate(zip(decode_logits, expected_decode), start=1):
        item = compare_logits(got, want)
        item["step"] = step
        decode_items.append(item)
    report["decode"] = {
        "steps": len(decode_items),
        "matched": sum(1 for item in decode_items if item.get("ok")),
        "items": decode_items,
        "ok": bool(decode_items) and all(item.get("ok") for item in decode_items),
    }

    failed = [item for item in comparisons if not item.get("ok")]
    failed_decode = [item for item in decode_items if not item.get("ok")]
    report["ok"] = not failed and report["decode"]["ok"]
    report["comparison"] = failed[0] if failed else comparisons[0]
    if failed:
        report["error"] = "；".join(
            f"长度 {item['length']}：argmax 相同={item.get('argmax_match')}，"
            f"最大绝对误差={item.get('max_abs_diff', float('nan')):.4f}，"
            f"余弦相似度={item.get('cosine', float('nan')):.6f}"
            for item in failed
        )
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
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在引擎里评测候选模型文件")
    parser.add_argument("--model", required=True, help="HF 模型目录（含权重）")
    parser.add_argument("--candidate", required=True, type=Path, help="候选 .py 文件")
    parser.add_argument("--arch", default=None, help="期望的架构名（architectures[0]）")
    parser.add_argument("--lengths", default="128,256,768", help="要对比多少个 token 位置，逗号分隔")
    parser.add_argument("--decode-steps", type=int, default=8,
                        help="逐 token 走 KV cache 的步数，0 表示只验 prefill")
    parser.add_argument("--report", type=Path, default=None, help="把 JSON 报告写到这个文件")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size，默认 1")
    args = parser.parse_args(argv)

    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    report = evaluate(args.model, args.candidate, args.arch, lengths,
                      decode_steps=args.decode_steps, tp=args.tp)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
