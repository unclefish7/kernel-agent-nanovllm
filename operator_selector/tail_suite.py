"""顺序运行六个小案例；--generate 仅为缺失的 Agent 方案调用 API。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="允许为尚无方案的案例请求模型 API")
    args = parser.parse_args()
    rows = []
    for model in ("qwen3", "phi3"):
        for tokens in (1, 128, 512):
            base = ROOT / "outputs/tail" / model / str(tokens)
            base.mkdir(parents=True, exist_ok=True)

            def run(mode):
                print(f"运行 {model}, tokens={tokens}, mode={mode}", flush=True)
                with (base / f"{mode}.log").open("w") as log:
                    subprocess.run([sys.executable, "-B", str(ROOT / "tail_demo.py"), mode,
                                    "--model", model, "--tokens", str(tokens)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

            # 无 API 的原始对照、独立候选和 12 种组合先通过，再请求模型。
            for mode in ("reference", "candidates", "manual"):
                run(mode)
            if not (base / "agent/selected_plan.json").exists():
                if not args.generate:
                    raise SystemExit(f"{model}/{tokens} 缺少 Agent 方案；允许调用 API 时加 --generate")
                run("agent")
            run("compare")
            result = json.loads((base / "compare/results.json").read_text())
            response = json.loads((base / "agent/response.json").read_text())
            rows.append({"model": model, "tokens": tokens, "correctness": result["correctness"],
                         "summary": result["summary"], "best_of_12": result["best_of_12"],
                         "agent_slowdown_vs_best": result["agent_slowdown_vs_best"],
                         "usage": response.get("usage"),
                         "selected_plan": json.loads((base / "agent/selected_plan.json").read_text()),
                         "results_path": str((base / "compare/results.json").relative_to(ROOT))})
            (ROOT / "outputs/tail/suite.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            print(f"完成：{model}/{tokens}, Agent {result['summary']['agent']['speedup']:.3f}x, "
                  f"相对最快组合耗时 {result['agent_slowdown_vs_best']:.3f} 倍", flush=True)


if __name__ == "__main__":
    main()
