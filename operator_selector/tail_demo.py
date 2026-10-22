"""逐步运行真实 Norm + MLP + 残差片段；只在 agent 模式请求 API。"""

import argparse
import json
from pathlib import Path
import statistics

import torch

from agent import select_plan, validate_plan
from candidates import load_swiglu
from evaluation import measure
from tail_candidates import catalog, functions, manual_plan, prepare, SelectedTail, load_vendor
from tail_task import make_tail_task

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_candidates(reference, cases, operators):
    norm = reference.layer.post_attention_layernorm
    checked = {}
    for x, residual in cases:
        summed = x + residual
        normalized = norm(summed)
        checks = {"torch_add": (operators["torch_add"](x, residual), summed)}
        for name in ("rms_reference", "rms_row", "rms_loop"):
            checks[name] = (operators[name](summed), normalized)
        actual_norm, actual_sum = operators["add_rms"](x, residual)
        checks["add_rms.normalized"] = (actual_norm, normalized)
        checks["add_rms.residual"] = (actual_sum, summed)
        mlp = reference.layer.mlp
        if hasattr(mlp, "gate_up_proj"):
            packed = mlp.gate_up_proj(normalized)
            gate, up = packed.chunk(2, dim=-1)
            split = operators["split_halves"](packed)
            checks["split_halves.gate"] = (split[0], gate)
            checks["split_halves.up"] = (split[1], up)
            tiled = operators["swiglu_tiles"](packed)
        else:
            gate, up = mlp.gate_proj(normalized), mlp.up_proj(normalized)
            tiled = operators["swiglu_tiles"](gate, up)
        activated = torch.nn.functional.silu(gate)
        expected = activated * up
        checks.update({"torch_silu": (operators["torch_silu"](gate), activated),
                       "torch_mul": (operators["torch_mul"](activated, up), expected),
                       "swiglu_tiles": (tiled, expected),
                       "swiglu_row": (operators["swiglu_row"](gate, up), expected)})
        for name, (actual, wanted) in checks.items():
            torch.testing.assert_close(actual, wanted, rtol=1e-4, atol=1e-5)
            checked.setdefault(name, []).append(list(x.shape))
    return checked


def evaluate(reference, cases, operators, plans):
    candidates = catalog(reference)
    models = {}
    for name, plan in plans.items():
        validate_plan(plan, candidates, ("x", "residual"))
        expected_linears = {item["id"] for item in candidates if item["id"].startswith("mlp.")}
        used = [step["op"] for step in plan["steps"] if step["op"].startswith("mlp.")]
        if set(used) != expected_linears or len(used) != len(expected_linears):
            raise ValueError("必须恰好保留每一个原始 Linear")
        models[name] = SelectedTail(reference, plan, operators)
    result = {"correctness": {}, "benchmark_shapes": [list(x.shape) for x in cases[-1]],
              "timing_unit": "ms", "rounds": []}
    for name, selected in models.items():
        for inputs in cases:
            before = [value.clone() for value in inputs]
            expected = reference(*inputs)
            actual = selected(*inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            # 被测算子不得改写共享输入，否则后续方案计时失去可比性。
            for value, original in zip(inputs, before):
                torch.testing.assert_close(value, original, rtol=0, atol=0)
        result["correctness"][name] = "通过：两种行数、完整片段输出、输入未改写"
    for index in range(3):
        order = ["reference", *models]
        if index % 2:
            order.reverse()
        operations = {"reference": reference, **models}
        stats = {name: measure(operations[name], list(cases[-1])) for name in order}
        for name in models:
            stats[name]["speedup_median"] = stats["reference"]["median"] / stats[name]["median"]
        result["rounds"].append({"round": index + 1, "order": order, "stats": stats})
    medians = {name: statistics.median(row["stats"][name]["median"] for row in result["rounds"])
               for name in ("reference", *models)}
    result["summary"] = {name: {"median_ms": value, "speedup": medians["reference"] / value}
                         for name, value in medians.items()}
    manual = [name for name in models if name != "agent" and name != "selected"]
    if manual:
        best = min(manual, key=medians.get)
        result["best_of_12"] = {"name": best, "median_ms": medians[best],
                                "scope": "预设的 4 种归一化路径 × 3 种激活路径，不声称任意计算图全局最优"}
        if "agent" in models:
            result["agent_slowdown_vs_best"] = medians["agent"] / medians[best]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "candidates", "manual", "agent", "compare"])
    parser.add_argument("--model", choices=["qwen3", "phi3"], default="qwen3")
    parser.add_argument("--tokens", type=int, choices=[1, 128, 512], default=128)
    args = parser.parse_args()
    base = ROOT / "outputs/tail" / args.model / str(args.tokens)
    out = base / args.mode
    out.mkdir(parents=True, exist_ok=True)
    reference, cases, context = make_tail_task(args.model, args.tokens)
    save(out / "task.json", context)
    with torch.no_grad():
        for inputs in cases:
            reference.check_original(*inputs)
        save(out / "reference_check.json", {"original_decoder_forward": "通过：只替换 Attention 输出，尾部与真实 Decoder 一致",
                                           "tested_shapes": [list(inputs[0].shape) for inputs in cases]})
        if args.mode == "reference":
            print("原 Decoder 与源码提取片段对照通过", flush=True)
            return
        operators = functions(reference)
        tuning = prepare(reference, cases[-1], operators)
        save(out / "tuning.json", tuning)
        save(out / "catalog.json", catalog(reference))
        save(out / "candidate_check.json", check_candidates(reference, cases, operators))
        if args.mode == "candidates":
            print("全部候选独立正确性检查通过", flush=True)
            return
        if args.mode == "agent":
            # 不提供人工组合、计时结果或哪种方案更快的提示。
            plans = {"selected": select_plan(context, catalog(reference), out)}
        else:
            plans = {f"{norm}__{activation}": manual_plan(reference, norm, activation)
                     for norm in ("reference", "row", "loop", "add_fused")
                     for activation in ("torch", "tiles", "row")}
            if args.mode == "compare":
                plans["agent"] = json.loads((base / "agent/selected_plan.json").read_text())
        for name, plan in plans.items():
            save(out / f"{name}_plan.json", plan)
        print(f"开始共同评测：{args.model}, tokens={args.tokens}, {list(plans)}", flush=True)
        result = evaluate(reference, cases, operators, plans)
        result["tuning"] = tuning
        result["model"] = args.model
        result["candidate_manifest"] = json.loads((ROOT / "vendor/manifest.json").read_text())
        save(out / "results.json", result)
        for name, stats in result["summary"].items():
            print(f"{name}: {stats['median_ms'] * 1000:.2f} us, {stats['speedup']:.3f}x")
        if "best_of_12" in result:
            print("人工组合中最快：", result["best_of_12"])
        if "agent_slowdown_vs_best" in result:
            print(f"Agent 耗时 / 最快组合耗时：{result['agent_slowdown_vs_best']:.3f}")


if __name__ == "__main__":
    main()
