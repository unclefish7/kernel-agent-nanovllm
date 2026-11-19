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
        report["ok"] = report["exact"] and bool(torch.isfinite(a).all())
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
    """每个序列一条 prompt，取各自最后一位的 logits；拼成最长的那个再按位置取。"""
    longest = max(sequences, key=len)
    return reference_logits(model_dir, longest, [len(s) - 1 for s in sequences])


@torch.inference_mode()
def reference_steps(model_dir, sequences, prefix_lengths, generated, decode_steps):
    """HF 按引擎实际复用的前缀长度独立计算缓存，不读取引擎的缓存数值。"""
    import copy
    import inspect
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    output_options = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    prompts = [*sequences, max(sequences, key=len)] if decode_steps else sequences
    saved = []
    prefill, decode = [], []
    if len(prefix_lengths) != len(prompts):
        raise ValueError("前缀长度记录与实际请求数不符")
    for seq, prefix in zip(prompts, prefix_lengths):
        if not 0 <= prefix < len(seq):
            raise ValueError(f"非法前缀长度 {prefix}，本次提示长度 {len(seq)}")
        past = None
        if prefix:
            prior = next((cache for tokens, cache in reversed(saved)
                          if len(tokens) >= prefix and tokens[:prefix] == seq[:prefix]), None)
            if prior is None:
                raise ValueError(f"HF 没有该请求前 {prefix} 个词元的独立缓存")
            past = copy.deepcopy(prior)
            past.crop(prefix)
        out = model(input_ids=torch.tensor([seq[prefix:]], device="cuda"),
                    past_key_values=past, use_cache=True, **output_options)
        prefill.append(out.logits[0, -1].float().cpu())
        saved.append((seq, out.past_key_values))
    for token in generated[:decode_steps]:
        out = model(input_ids=torch.tensor([[token]], device="cuda"),
                    past_key_values=out.past_key_values, use_cache=True, **output_options)
        decode.append(out.logits[0, -1].float().cpu())
    return prefill, decode


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
            return {"prefill": prefill, "decode": decode, "prefix_lengths": prefix_lengths}
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
             tp: int = 1, exact: bool = True, logits_out: Path | None = None, **kwargs: Any) -> dict:
    """完整评测一个候选文件；任何阶段失败都折算成报告，不抛异常。"""
    started = time.time()
    report: dict[str, Any] = {
        "ok": False, "stage": "load", "candidate": str(candidate),
        "lengths": list(lengths), "tensor_parallel_size": tp,
        "criterion": "exact" if exact else "argmax + max_abs <= 2.0 + cosine >= 0.999",
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
        if min(lengths) < 1 or decode_steps < 0:
            raise ValueError("输入长度必须为正，decode_steps 必须非负")
        need = longest + decode_steps + 1
        if len(base) < need:
            raise RuntimeError(f"填充文本只有 {len(base)} 个 token，不够 {need}")
        sequences = [base[:length] for length in lengths]
        # 先跑引擎（要占显存），再加载 HF 取参照：两份模型不同时驻留
        previous_mode = torch.get_deterministic_debug_mode()
        try:
            if exact:
                torch.use_deterministic_algorithms(True)
            report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
            eng = engine_run(cls, model_dir, sequences, arch or cls.__name__,
                             max_model_len=longest + decode_steps + 64, decode_steps=decode_steps, tp=tp)
            expected_prefill, expected_decode = reference_steps(
                model_dir, sequences, eng["prefix_lengths"], eng["decode"].get("generated", []), decode_steps
            )
        finally:
            torch.set_deterministic_debug_mode(previous_mode)
        decode_logits = eng["decode"]["logits"]
        actual_ids = eng["decode"]["ids"] or max(sequences, key=len)
        start = longest
        report["cached_prefix_tokens"] = eng["prefix_lengths"]
        generation_reference = expected_prefill.pop() if decode_steps else None
        if (len(eng["prefill"]) != len(lengths) or len(expected_prefill) != len(lengths)
                or len(decode_logits) != decode_steps or len(expected_decode) != decode_steps):
            raise ValueError("捕获或参照的分数数量与请求的步数不同")
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
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size，默认 1")
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True,
                        help="默认要求每一步整个词表逐项相等，--no-exact 使用原有容差")
    args = parser.parse_args(argv)

    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    report = evaluate(args.model, args.candidate, args.arch, lengths,
                      decode_steps=args.decode_steps, tp=args.tp, exact=args.exact, logits_out=args.logits_out)
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
