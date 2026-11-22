"""单卡、单请求 prefill 的模块边界诊断。输出首个可对齐的分歧，不参与验收。"""

import argparse
import atexit
from collections import Counter
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch

from nanovllm_selector.port.evaluate import (
    FILLER_TEXT, GREEDY_TEMPERATURE, load_candidate, pick_model_class,
)


def _shapes(value):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shapes(x) for x in value]
    if hasattr(value, "last_hidden_state"):
        return {"last_hidden_state": _shapes(value.last_hidden_state)}
    return None


@torch.inference_mode()
def trace(model, run, side, sink):
    """只在实际执行时发出边界。sink 同步消费张量，hook 不保存设备张量。"""
    counts = Counter()

    def emit(path, phase, module, tensor, raw, rule):
        key = (path, phase, counts[path, phase])
        counts[path, phase] += 1
        sink(key, tensor, {"class": type(module).__name__, "raw_shapes": raw, "view": rule})

    def install(stack, path, module):
        signature = inspect.signature(module.forward)
        norm = "norm" in path.rsplit(".", 1)[-1].lower()
        block = path.split(".")[-2:-1] == ["layers"] and path.split(".")[-1].isdigit()
        fused = side == "engine" and (block or norm) and "residual" in signature.parameters

        def before(mod, args, kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            raw = {k: _shapes(v) for k, v in bound.items() if v is not None}
            x = next((bound[k] for k in ("hidden_states", "input_ids", "x", "input")
                      if torch.is_tensor(bound.get(k))), None)
            if x is None:
                tensors = [v for v in bound.values() if torch.is_tensor(v)]
                x = tensors[0] if len(tensors) == 1 else None
            rule = "主输入"
            if fused and x is not None and torch.is_tensor(bound.get("residual")):
                x = x + bound["residual"]
                rule = "分支 + residual（输入类型舍入）"
            if path == "lm_head" and x is not None:
                x = x[..., -1:, :]
                rule = "单请求最后一个词元"
            emit(path, "input", mod, x, raw, rule)

        def after(mod, args, output):
            raw = _shapes(output)
            x = output
            rule = "原始输出"
            if fused and isinstance(x, tuple) and len(x) == 2 and all(torch.is_tensor(v) for v in x):
                if block:
                    x = x[0] + x[1]
                    rule = "层输出 = 分支 + 延迟相加的残差"
                else:
                    emit(path, "residual", mod, x[1], raw, "返回的 residual 对照 HF 归一化之前的输入")
                    x = x[0]
                    rule = "tuple[0] 归一化输出"
            elif isinstance(x, tuple) and path.endswith(".self_attn"):
                x = x[0]
                rule = "注意力输出 tuple[0]"
            elif hasattr(x, "last_hidden_state"):
                x = x.last_hidden_state
                rule = "last_hidden_state"
            if path == "lm_head" and torch.is_tensor(x) and x.ndim >= 2:
                x = x[..., -1:, :]
                rule = "最后一个词元的词表投影"
            emit(path, "output", mod, x if torch.is_tensor(x) else None, raw, rule)

        stack.callback(module.register_forward_pre_hook(before, with_kwargs=True).remove)
        stack.callback(module.register_forward_hook(after).remove)

    with ExitStack() as stack:
        for path, module in model.named_modules():
            if path:
                install(stack, path, module)
        logits = run()
        emit("@logits", "output", model, logits, _shapes(logits), "实际返回的分数")


def _fingerprint(x):
    # 原精度全量摘要。规范化正负零，不把符号位差异判为数值误差。
    cpu = x.detach().to(device="cpu", copy=True).contiguous()
    if cpu.is_floating_point():
        cpu.masked_fill_(cpu == 0, 0)
    data = cpu.reshape(-1).view(torch.uint8).numpy()
    return cpu, hashlib.blake2b(data, digest_size=32).digest()


def _aligned(a, b):
    return a == b or (b[:1] == [1] and a == b[1:])


def _delta(a, b):
    if not _aligned(list(a.shape), list(b.shape)):
        return {"kind": "shape_mismatch", "reason": "除 HF 的单请求批次轴外，形状仍不同"}
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    different = (a != b) | ~finite
    stats = {"different_elements": int(different.sum()), "elements": a.numel(),
             "max_abs_diff": float((a - b).abs().max()) if a.numel() and finite.all() else None}
    if different.any():
        i = int(different.to(torch.uint8).argmax())
        stats.update(first_flat_index=i, engine_value=float(a[i]) if torch.isfinite(a[i]) else None,
                     hf_value=float(b[i]) if torch.isfinite(b[i]) else None)
    return stats


def diagnose_runs(run_hf, run_engine, packed_mapping):
    """两方各前向一次；出现分歧时仅为该边界另做一次 HF 前向，计算精确误差。"""
    reference, seen, unmatched = {}, set(), set()
    report = {"compared": 0, "first_difference": None}
    first = None
    last_match = None
    hf_logits = None

    def remember(key, x, meta):
        nonlocal hf_logits
        if x is None:
            return
        cpu, digest = _fingerprint(x)
        reference[key] = (list(x.shape), x.dtype, digest, meta)
        if key[0] == "@logits":
            hf_logits = cpu

    run_hf(remember)
    groups = {}
    for src, (dst, slot) in packed_mapping.items():
        rank = {"q": 0, "k": 1, "v": 2}.get(slot, slot)
        groups.setdefault(dst, []).append((rank, src))

    def compare(key, target, x, meta):
        nonlocal first, last_match
        seen.add(target)
        shape, dtype, expected_hash, hf_meta = reference[target]
        cpu, digest = _fingerprint(x)
        report["compared"] += 1
        same = _aligned(list(x.shape), shape) and x.dtype == dtype and digest == expected_hash
        if key[0] == "@logits":
            report["logits"] = _delta(cpu, hf_logits)
        if first is not None:
            return
        if same and torch.isfinite(cpu).all():
            last_match = [key[0], key[1], target[0]]
            return
        first = (target, cpu)
        report["first_difference"] = {
            "engine_path": key[0], "engine_phase": key[1], "hf_path": target[0], "hf_phase": target[1],
            "call_index": key[2], "engine_shape": list(x.shape), "hf_shape": shape,
            "engine_dtype": str(x.dtype), "hf_dtype": str(dtype), "engine": meta, "hf": hf_meta,
            "previous_match": last_match,
        }

    def observe(key, x, meta):
        path, phase, index = key
        target = (path, "input" if phase == "residual" else phase, index)
        if x is not None and target in reference:
            compare(key, target, x, meta)
            return
        parent, _, leaf = path.rpartition(".")
        entries = sorted(groups.get(leaf, []))
        targets = [(".".join(filter(None, (parent, name))), phase, index) for _, name in entries]
        if x is not None and targets and all(t in reference for t in targets):
            if phase == "input":
                for t in targets:
                    compare(key, t, x, meta)
                return
            widths = [reference[t][0][-1] for t in targets]
            if sum(widths) == x.shape[-1]:
                start = 0
                for t, part in zip(targets, x.split(widths, dim=-1)):
                    compare(key, t, part, {**meta, "view": f"融合投影最后一维 [{start}:{start + part.size(-1)}]"})
                    start += part.size(-1)
                return
        unmatched.add(path)

    run_engine(observe)
    if first is not None:
        target, actual = first

        def detail(key, x, meta):
            if key == target and x is not None:
                cpu, digest = _fingerprint(x)
                if digest == reference[key][2]:
                    report["first_difference"].update(_delta(actual, cpu))
                else:
                    report["first_difference"]["reason"] = "重跑 HF 边界数值发生变化，无法确认精确误差"

        run_hf(detail)
    missing = sorted({p for p, phase, i in reference.keys() - seen})
    report["unpaired"] = {"engine_paths": len(unmatched), "hf_paths": len(missing),
                          "engine_examples": sorted(unmatched)[:6], "hf_examples": missing[:6]}
    return report


def diagnose(model_dir, candidate, length=128, prompt_file=None, *, arch=None):
    from nanovllm import LLM
    from nanovllm.models import ARCH
    from nanovllm.sampling_params import SamplingParams
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    text = Path(prompt_file).read_text(encoding="utf-8") if prompt_file else FILLER_TEXT
    ids = tokenizer(text).input_ids[:length]
    if length < 1 or len(ids) != length:
        raise ValueError("提示词元不足或长度非正")
    arch = arch or AutoConfig.from_pretrained(model_dir, local_files_only=True).architectures[0]
    cls = pick_model_class(load_candidate(Path(candidate)), arch)

    def hf(sink):
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        tokens = torch.tensor([ids], device=model.device)
        options = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
        trace(model, lambda: model(input_ids=tokens, use_cache=True, **options).logits[:, -1:], "hf", sink)

    def engine(sink):
        torch.cuda.empty_cache()
        with patch.dict(ARCH, {arch: cls}):
            llm = LLM(str(model_dir), enforce_eager=True, tensor_parallel_size=1, max_num_seqs=1,
                      max_model_len=length + 64, gpu_memory_utilization=0.35)
            try:
                model = llm.model_runner.model
                if type(model) is not cls:
                    raise ValueError("实际装载的模型类与候选不同")
                scores = []
                compute = model.compute_logits

                def capture(hidden):
                    out = compute(hidden)
                    scores.append(out.detach().cpu().clone())
                    return out

                def run():
                    llm.generate([ids], SamplingParams(temperature=GREEDY_TEMPERATURE, max_tokens=1), use_tqdm=False)
                    if len(scores) != 1:
                        raise ValueError("期望单次完整 prefill")
                    return scores[0]

                with patch.object(model, "compute_logits", capture):
                    trace(model, run, "engine", sink)
            finally:
                atexit.unregister(llm.exit)
                llm.exit()

    previous = torch.get_deterministic_debug_mode()
    try:
        torch.use_deterministic_algorithms(True)
        report = diagnose_runs(hf, engine, getattr(cls, "packed_modules_mapping", {}))
    finally:
        torch.set_deterministic_debug_mode(previous)
    return {"purpose": "prefill_diagnostic", "candidate": str(candidate),
            "candidate_sha256": hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
            "length": length, "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--arch", help="默认取模型配置中的架构名")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--report", type=Path, help="可选的 JSON 文件，默认只打印")
    args = parser.parse_args()
    text = json.dumps(diagnose(args.model, args.candidate, args.length, args.prompt_file, arch=args.arch),
                      ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
