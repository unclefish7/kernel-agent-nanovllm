"""迭代式移植：生成 → 在引擎里跑 → 与 transformers 对比 → 回灌真实报错 → 再来。

循环的停止条件由代码判定，不靠模型自觉：

- 引擎里的 logits 与 transformers 对齐  → 成功；
- 轮数用完                              → 失败，留下每一轮的文件与报告。

模型的输出只被当作候选；是不是"对"，永远由 `evaluate.py` 在真实引擎里跑一遍说了算。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .generate import make_llm, strip_code_fence
from .prompt import build_messages, load_graph

REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV = REPO_ROOT / "nanovllm_selector" / ".env"

STAGE_NAMES = {
    "load": "按路径导入候选文件",
    "class": "找出 ForCausalLM 类",
    "run": "在引擎里构造、装权重、跑前向",
    "compare": "与 transformers 的 logits 对比",
}


def load_dotenv(path: Path = DOTENV) -> list[str]:
    """把一个 .env 文件读进 os.environ（已存在的变量不覆盖），返回读到的键。"""
    loaded: list[str] = []
    if not Path(path).exists():
        return loaded
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def build_feedback(report: dict) -> str:
    """把一份评测报告改写成给模型看的"哪里错了"。"""
    stage = report.get("stage", "?")
    lines = [f"上一次生成的文件没有通过，失败在「{STAGE_NAMES.get(stage, stage)}」。"]
    error = str(report.get("error") or "").strip()
    if error:
        lines += ["", "真实报错：", "```", error[-3000:], "```"]
    rows = report.get("comparisons") or []
    if rows:
        lines += ["", "数值对比（不同长度各跑一次 prefill，与 transformers 对齐）："]
        for item in rows:
            lines.append(
                f"- 长度 {item.get('length')}: top-1 {item.get('argmax_actual')} vs "
                f"{item.get('argmax_expected')}，最大绝对误差 "
                f"{item.get('max_abs_diff'):.4f}，余弦 {item.get('cosine'):.6f}"
                f"{'' if item.get('ok') else '  ← 未通过'}"
            )
    lines += ["", "请输出修正后的完整文件：只输出 Python 代码，不要解释、不要补丁。"]
    return "\n".join(lines)


def run_loop(
    *,
    messages: list[dict],
    call_llm: Callable[[list[dict]], str],
    evaluate: Callable[[Path], dict],
    out_dir: Path,
    max_iters: int = 3,
    log: Callable[[str], None] | None = None,
) -> dict:
    """跑最多 `max_iters` 轮；返回 {ok, attempt, path, report, history}。"""
    if max_iters < 1:
        raise ValueError("max_iters 至少是 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = log or (lambda text: None)
    history: list[dict] = []
    prompt = list(messages)

    for attempt in range(1, max_iters + 1):
        started = time.time()
        text = call_llm(prompt)
        code = strip_code_fence(text)
        path = out_dir / f"attempt_{attempt}.py"
        path.write_text(code, encoding="utf-8")
        report = evaluate(path)
        history.append(
            {
                "attempt": attempt,
                "path": str(path),
                "ok": bool(report.get("ok")),
                "stage": report.get("stage"),
                "elapsed": round(time.time() - started, 2),
                "error": (str(report.get("error") or ""))[-1500:] or None,
            }
        )
        say(f"[{attempt}/{max_iters}] {'通过' if report.get('ok') else '未通过'}"
            f"（{STAGE_NAMES.get(report.get('stage'), report.get('stage'))}）")
        if report.get("ok"):
            return {
                "ok": True,
                "attempt": attempt,
                "path": str(path),
                "report": report,
                "history": history,
            }
        prompt = prompt + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": build_feedback(report)},
        ]

    return {
        "ok": False,
        "attempt": max_iters,
        "path": history[-1]["path"] if history else None,
        "report": None,
        "history": history,
    }


def subprocess_evaluate(
    *,
    model_dir: str,
    arch: str | None,
    work_dir: Path = REPO_ROOT,
    timeout: float = 1200.0,
) -> Callable[[Path], dict]:
    """返回一个评测函数：每个候选文件起一个子进程，读它写出的 JSON 报告。"""

    def evaluate(candidate: Path) -> dict:
        report_path = Path(candidate).with_suffix(".report.json")
        if report_path.exists():
            report_path.unlink()
        cmd = [
            sys.executable, "-m", "nanovllm_selector.port.evaluate",
            "--model", str(model_dir),
            "--candidate", str(candidate),
            "--report", str(report_path),
        ]
        if arch:
            cmd += ["--arch", arch]
        try:
            proc = subprocess.run(
                cmd, cwd=str(work_dir), capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "run", "error": f"评测超时（{timeout:.0f} 秒）"}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report_path.unlink()
            return report
        tail = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-4000:]
        return {"ok": False, "stage": "run", "error": f"子进程退出码 {proc.returncode}\n{tail}"}

    return evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="迭代式：生成模型文件并在引擎里验证")
    parser.add_argument("--model", required=True, help="HF 模型目录（必须有权重）")
    parser.add_argument("--graph", type=Path, required=True, help="参考图 JSON")
    parser.add_argument("--spec", type=Path, required=True, help="引擎规约 spec.yaml")
    parser.add_argument("--kernels", type=Path, required=True, help="算子库索引 index.md")
    parser.add_argument("--template", type=Path, required=True, help="同构模板 qwen3.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="每轮生成的文件放这里")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--arch", default=None, help="架构名，默认取图的 architectures[0]")
    parser.add_argument("--no-toolbox", action="store_true", help="prompt 里省略 toolbox 表")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--timeout", type=float, default=180.0, help="单次 API 请求超时秒数")
    args = parser.parse_args(argv)

    loaded = load_dotenv()
    if loaded:
        print(f"从 .env 读了 {len(loaded)} 个键：{', '.join(loaded)}")

    graph = load_graph(args.graph)
    arch = args.arch or (graph.get("model", {}).get("architectures") or [None])[0]
    messages = build_messages(
        args.graph, args.spec, args.kernels, args.template,
        include_toolbox=not args.no_toolbox,
    )
    call_llm = make_llm(
        api_key_env=args.api_key_env,
        api_base=args.api_base,
        model=args.llm_model,
        timeout=args.timeout,
    )
    evaluate = subprocess_evaluate(model_dir=args.model, arch=arch)

    result = run_loop(
        messages=messages,
        call_llm=call_llm,
        evaluate=evaluate,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        log=print,
    )
    trace = args.out_dir / "trace.json"
    trace.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果：{'成功' if result['ok'] else '失败'}；明细见 {trace}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
