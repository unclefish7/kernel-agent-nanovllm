"""从仓库根目录运行；--model 指向本地单文件 safetensors 模型，--out 指定 JSON。

docker exec -w /workspace -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 nanovllm-selector-rocm-dev python -B \
  nanovllm_selector/outputs/deepseek_v3_tiny/reference.py --model 模型目录 --out 报告路径
"""

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE


PROMPT = [11, 23, 37, 41, 53]
DECODE = [67, 79]


def audit(model, archive, info):
    state = model.state_dict()
    keys = set(archive.keys())
    missing, extra = sorted(state.keys() - keys), sorted(keys - state.keys())
    mismatches, nonfinite = [], {}
    for name in sorted(keys & state.keys()):
        src, dst = archive.get_tensor(name), state[name]
        if not src.isfinite().all():
            nonfinite[name] = {"shape": list(src.shape), "nan": src.isnan().sum().item(),
                               "inf": src.isinf().sum().item(), "numel": src.numel()}
        # 先转为实际加载精度，再比位模式；保留原始 NaN 也算正确加载。
        if src.shape != dst.shape or not torch.equal(src.to(dst.dtype).contiguous().view(torch.uint8),
                                                     dst.contiguous().view(torch.uint8)):
            mismatches.append(name)
    return {"complete_and_matches_file": not any((*info.values(), missing, extra, mismatches)),
            "loading_info": info, "file_keys": len(keys), "state_dict_keys": len(state),
            "state_dict_minus_file": missing, "file_minus_state_dict": extra,
            "value_or_shape_mismatch": mismatches, "nonfinite_file_tensors": nonfinite}


def cache_probe(model, atol, rtol):
    cfg = model.config
    prefix, steps, cache = [], [], None
    for ids in (PROMPT, *([t] for t in DECODE)):
        prefix.extend(ids)
        out = model(torch.tensor([ids]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        actual = out.logits.float()
        full = model(torch.tensor([prefix]), use_cache=False).logits[:, -len(ids):].float()
        top, ref = actual.argmax(-1), full.argmax(-1)
        layers = [{"k": list(l.keys.shape), "v": list(l.values.shape), "dtype": str(l.keys.dtype),
                   "length": cache.get_seq_length(i)} for i, l in enumerate(cache.layers)]
        shape = [1, cfg.num_attention_heads, len(prefix)]
        shapes_ok = len(layers) == cfg.num_hidden_layers and all(
            l["k"] == shape + [cfg.qk_nope_head_dim + cfg.qk_rope_head_dim]
            and l["v"] == shape + [cfg.v_head_dim] and l["length"] == len(prefix) for l in layers)
        steps.append({"ids": ids, "prefix_length": len(prefix), "cache": type(cache).__name__, "layers": layers,
                      "shapes_ok": shapes_ok, "logits_shape": list(actual.shape),
                      "logits_finite": bool(actual.isfinite().all() and full.isfinite().all()),
                      "max_abs_diff": (actual - full).abs().max().item(), "actual_top1": top.tolist(),
                      "full_prefix_top1": ref.tolist(), "allclose": torch.allclose(actual, full, atol=atol, rtol=rtol)})
    return {"atol": atol, "rtol": rtol, "steps": steps,
            "ok": all(s["logits_finite"] and s["shapes_ok"] and s["allclose"] and
                      s["actual_top1"] == s["full_prefix_top1"] for s in steps)}


def routing_probe(model):
    # 两个 tiny 配置都只有一个 MoE 层。首尾行是重复对照，中间行只替换末词元。
    (name, moe), = [(n, m) for n, m in model.named_modules() if isinstance(m, DeepseekV3MoE)]
    ids = torch.tensor([PROMPT, [*PROMPT[:-1], 97], PROMPT])
    output, calls = [], Counter()

    def count_calls(expert):
        def record(module, args):
            calls[expert] += args[0].size(0)
        return record

    with ExitStack() as hooks:
        topk = hooks.enter_context(patch.object(moe.gate, "get_topk_indices", wraps=moe.gate.get_topk_indices))
        hooks.enter_context(moe.gate.register_forward_hook(lambda m, a, result: output.append(result)))
        for i, expert in enumerate(moe.experts):
            hooks.enter_context(expert.register_forward_pre_hook(count_calls(i)))
        logits = model(ids, use_cache=False, logits_to_keep=1).logits
    scores, = topk.call_args.args  # 原生 sigmoid 输出，尚未加校正偏置。
    (indices, weights), = output
    bias = moe.gate.e_score_correction_bias
    corrected = scores + bias
    expected = scores.gather(-1, indices)
    if moe.gate.norm_topk_prob:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    mixture_matches = torch.equal(weights, expected * moe.gate.routed_scaling_factor)
    dispatched = calls == Counter(indices.flatten().tolist())
    aligned = torch.zeros_like(scores).scatter_(-1, indices, weights).reshape(*ids.shape, -1)
    indices = indices.reshape(*ids.shape, -1)
    last = indices[:2, -1].tolist()
    repeat_ok = torch.equal(indices[0], indices[2]) and torch.allclose(aligned[0], aligned[2], atol=1e-6, rtol=0)
    return {"module": name, "input_ids": ids.tolist(), "routed_experts": len(moe.experts), "k": moe.gate.top_k,
            "bias_numel": bias.numel(), "bias_nan": bias.isnan().sum().item(), "bias_zero": (bias == 0).sum().item(),
            "sigmoid_finite": bool(scores.isfinite().all()), "corrected_scores_numel": corrected.numel(),
            "corrected_scores_nan": corrected.isnan().sum().item(), "last_token_expert_ids": last,
            "changed_experts_symmetric_difference": sorted(set(last[0]) ^ set(last[1])),
            "last_token_aligned_weight_max_abs_diff": (aligned[0, -1] - aligned[1, -1]).abs().max().item(),
            "mix_weights_finite": bool(weights.isfinite().all()), "mix_matches_uncorrected_scores": mixture_matches,
            "distinct_experts_executed": len(calls), "actual_dispatch_matches_gate_counts": dispatched,
            "identical_input_control_ok": repeat_ok, "logits_finite": bool(logits.isfinite().all())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "4.57.3":
        raise RuntimeError(f"需要 Transformers 4.57.3，实际为 {transformers.__version__}")
    torch.manual_seed(0)
    path = args.model / "model.safetensors"
    report = {"model_dir": str(args.model), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "transformers": transformers.__version__, "torch": str(torch.__version__), "device": "cpu",
              "attention": "eager", "threads": torch.get_num_threads(), "runs": {}}
    with safe_open(path, framework="pt", device="cpu") as archive:
        dtypes = Counter(archive.get_slice(k).get_dtype() for k in archive.keys())
        report["file_dtype_counts"] = dict(dtypes)
        for tag, dtype, atol, rtol in [("F32", torch.float32, 1e-5, 1e-5), ("BF16", torch.bfloat16, 2e-3, 1e-2)]:
            model, info = AutoModelForCausalLM.from_pretrained(
                args.model, local_files_only=True, trust_remote_code=False, dtype=dtype,
                attn_implementation="eager", output_loading_info=True)
            model.eval()
            report["architecture"] = f"{type(model).__module__}.{type(model).__name__}"
            run = report["runs"][tag] = {"precision": "original" if set(dtypes) == {tag} else "converted",
                                         "model_dtype": str(model.dtype), "audit": audit(model, archive, info)}
            if not run["audit"]["complete_and_matches_file"]:
                args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                raise RuntimeError(f"权重加载不完整，见 {args.out}")
            run["cache"] = cache_probe(model, atol, rtol)
            run["routing"] = routing_probe(model)
    runs = report["runs"].values()
    report["cached_logits_match_full_prefix"] = all(run["cache"]["ok"] for run in runs)
    report["routing_probe_consistent"] = all(run["routing"][key] for run in runs for key in
        ("mix_matches_uncorrected_scores", "identical_input_control_ok", "actual_dispatch_matches_gate_counts",
         "logits_finite", "sigmoid_finite", "mix_weights_finite"))
    report["suitable_as_finite_routing_reference"] = report["routing_probe_consistent"] and all(
        not run["audit"]["nonfinite_file_tensors"] and bool(run["routing"]["changed_experts_symmetric_difference"])
        for run in runs)
    if any(run["routing"]["bias_nan"] for run in runs):
        report["interpretation"] = ("原始专家校正偏置含 NaN，专家排序分数受污染。"
                                    "混合权重取自未校正的 sigmoid 分数，本次最终 logits 有限。")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not report["cached_logits_match_full_prefix"] or not report["routing_probe_consistent"]:
        raise SystemExit(f"缓存或路由对照未通过，见 {args.out}")


if __name__ == "__main__":
    main()
