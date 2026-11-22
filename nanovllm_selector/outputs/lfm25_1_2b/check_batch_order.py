"""单卡并发验收。用 --model 本地权重目录 --candidate 候选.py 调用，默认严格逐项比较。"""

import argparse
import atexit
import hashlib
import json
from pathlib import Path
import traceback
from unittest.mock import patch

import torch

from nanovllm_selector.port.evaluate import (
    GREEDY_TEMPERATURE, compare_logits, engine_prefix_reuse, load_candidate,
    pick_model_class, reference_trace,
)


TOKEN_COUNTS = {"A": 3, "B": 5}


def capture_round(llm, prompts, order):
    """通过正常的 step 入口运行，按调度器交给 runner 的真实请求顺序关联 logits。"""
    from nanovllm.sampling_params import SamplingParams

    if not llm.is_finished() or set(order) != set(TOKEN_COUNTS) or len(order) != 2:
        raise ValueError("需要空闲引擎和两个不同的请求 A、B")
    traces = {}
    labels = {}
    for name in order:
        llm.add_request(prompts[name], SamplingParams(temperature=GREEDY_TEMPERATURE,
                                                     max_tokens=TOKEN_COUNTS[name], ignore_eos=True))
        seq = llm.scheduler.waiting[-1]
        labels[seq.seq_id] = name
        traces[name] = {"prompt": list(seq.prompt_token_ids), "generated": [], "scores": []}

    runner = llm.model_runner
    compute_logits = runner.model.compute_logits
    run = runner.run
    scores, batches = [], []

    def capture(hidden):
        logits = compute_logits(hidden)
        scores.append(logits.detach().float().cpu().clone())
        return logits

    def record(seqs, is_prefill):
        scores.clear()
        tokens = run(seqs, is_prefill)
        if len(scores) != 1 or scores[0].ndim != 2 or scores[0].size(0) != len(seqs):
            raise ValueError("候选分数的批次维度与实际调度不同")
        names = [labels[s.seq_id] for s in seqs]
        batches.append({"phase": "prefill" if is_prefill else "decode", "requests": names})
        for seq, name, row, token in zip(seqs, names, scores[0], tokens, strict=True):
            trace = traces[name]
            if not 0 <= token < row.numel():
                raise ValueError(f"请求 {name} 的生成词元越出词表范围：{token}")
            if is_prefill:
                if seq.num_cached_tokens or seq.num_scheduled_tokens != len(trace["prompt"]):
                    raise ValueError("短提示应完整 prefill，不能跨请求复用")
            trace["scores"].append(row)
            trace["generated"].append(token)
        return tokens

    completed = {}
    with patch.object(runner.model, "compute_logits", capture), patch.object(runner, "run", record):
        while not llm.is_finished():
            finished, _ = llm.step()
            completed.update((labels[seq_id], tokens) for seq_id, tokens in finished)
            if len(batches) == 2:
                # 两条请求仍存活，下一步强制将它们放在与上一轮相反的行。
                llm.scheduler.running.reverse()
            if len(batches) > sum(TOKEN_COUNTS.values()):
                raise ValueError("调度步数超出验收用例范围")
    expected = [list(order), list(order), list(reversed(order)), ["B"], ["B"]]
    if [b["requests"] for b in batches] != expected or [b["phase"] for b in batches] != ["prefill"] + ["decode"] * 4:
        raise ValueError(f"未观测到预期的并发、换序和单请求继续生成：{batches}")
    for name, trace in traces.items():
        if completed.get(name) != trace["generated"] or len(trace["scores"]) != TOKEN_COUNTS[name]:
            raise ValueError(f"请求 {name} 的完成词元与逐步捕获不一致")
    return {"order": list(order), "batches": batches, "traces": traces}


def compare_round(model, captured, *, exact):
    """每个请求分别从自己的原始提示重算 HF，逐步喂入实际采样的词元。"""
    report = {"order": captured["order"], "batches": captured["batches"], "requests": {}, "ok": True}
    for name, trace in captured["traces"].items():
        info = {}
        generated = trace["generated"]
        prefill, decode = reference_trace(model, [trace["prompt"]], [0], generated[:-1], info)
        steps = []
        for i, (got, want) in enumerate(zip(trace["scores"], prefill + decode, strict=True)):
            item = compare_logits(got, want, exact=exact)
            item["step"] = i
            item["greedy_token_match"] = bool(want[generated[i]] == want.max()) if generated[i] < len(want) else False
            item["ok"] = item["ok"] and item["greedy_token_match"]
            steps.append(item)
        ok = all(item["ok"] for item in steps)
        report["requests"][name] = {"steps": steps, "ok": ok, "reference": info,
                                    "prompt": trace["prompt"], "generated": generated}
        report["ok"] &= ok
    return report


def verify(model_dir, candidate, exact=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from nanovllm import LLM
    from nanovllm.models import ARCH

    candidate = Path(candidate)
    report = {"ok": False, "candidate": str(candidate), "model": str(model_dir),
              "exact": exact, "tensor_parallel_size": 1, "rounds": [], "outcome": "evaluation_error"}
    previous_mode = torch.get_deterministic_debug_mode()
    try:
        if exact:
            torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        prompts = {"A": tokenizer.encode("The capital of France is"),
                   "B": tokenizer.encode("A short Python program can count the letters in this sentence by")}
        if prompts["A"] == prompts["B"] or len(prompts["A"]) == len(prompts["B"]):
            raise ValueError("两条提示必须具有不同内容和长度")
        arch = AutoConfig.from_pretrained(model_dir, local_files_only=True).architectures[0]
        report["outcome"] = "candidate_failed"
        report["candidate_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        cls = pick_model_class(load_candidate(candidate), arch)
        with patch.dict(ARCH, {arch: cls}):
            engine = LLM(str(model_dir), tensor_parallel_size=1, enforce_eager=True, max_num_seqs=2,
                         max_model_len=max(map(len, prompts.values())) + 8, gpu_memory_utilization=0.35)
            try:
                if type(engine.model_runner.model) is not cls:
                    raise ValueError("实际装载的模型与候选不同")
                report["prefix_reuse"] = engine_prefix_reuse(engine.scheduler)
                captures = [capture_round(engine, prompts, order) for order in (["A", "B"], ["B", "A"])]
            finally:
                atexit.unregister(engine.exit)
                engine.exit()
        del engine
        torch.cuda.empty_cache()
        report["outcome"] = "reference_unreliable"
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        for captured in captures:
            report["rounds"].append(compare_round(model, captured, exact=exact))
        report["ok"] = all(r["ok"] for r in report["rounds"])
        report["outcome"] = "passed" if report["ok"] else "candidate_failed"
    except Exception:
        report["error"] = traceback.format_exc()[-4000:]
    finally:
        torch.set_deterministic_debug_mode(previous_mode)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--exact", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", type=Path, help="可选的 JSON 报告路径，默认只打印")
    args = parser.parse_args()
    report = verify(args.model, args.candidate, args.exact)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
