"""manual 验证候选；agent 请求新方案；compare 无 API 对照已保存的方案。"""

import argparse
import json
from pathlib import Path

from candidates import candidate_catalog, manual_plan
from evaluation import evaluate
from task import make_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["manual", "agent", "compare"])
    parser.add_argument("--model", choices=["qwen3", "phi3"], default="qwen3")
    args = parser.parse_args()
    model_out = Path(__file__).resolve().parent / "outputs" / args.model
    out = model_out / args.mode
    out.mkdir(parents=True, exist_ok=True)
    reference, inputs, context = make_task(args.model)
    (out / "task.json").write_text(json.dumps(context, ensure_ascii=False, indent=2))
    if args.mode in ("manual", "compare"):
        plans = {"separate": manual_plan(reference, False), "fused": manual_plan(reference, True)}
        if args.mode == "compare":
            plans["agent"] = json.loads((model_out / "agent/selected_plan.json").read_text())
    else:
        from agent import select_plan
        plans = {"selected": select_plan(context, candidate_catalog(reference), out)}
    for name, plan in plans.items():
        (out / f"{name}_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"共同验证方案：{list(plans)}", flush=True)
    results = evaluate(reference, inputs, plans)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("融合配置：", results["fusion"], flush=True)
    for measured in results["rounds"]:
        print(f"第 {measured['round']} 轮顺序：{measured['order']}")
        for name in plans:
            for scope in ("activation", "mlp"):
                stats = measured[scope][name]
                print(f"{name} {scope}: 中位数 {stats['median'] * 1000:.2f} us，"
                      f"加速比 {stats['speedup_median']:.3f}x")


if __name__ == "__main__":
    main()
