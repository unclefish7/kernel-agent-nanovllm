"""命令行入口。

    python -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --check
    python -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --summary
    python -m nanovllm_selector.extract.cli --model tmp/models/Qwen3-0.6B --outdir outputs

退出码：0 通过 / 1 对账未通过 / 2 运行出错。
"""

from __future__ import annotations

import argparse
import sys
import traceback

from .api import extract_graph
from .render import render_markdown, render_mermaid, render_text, write_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nanovllm_selector.extract.cli",
        description="从 HuggingFace 模型目录重建计算图（M1：prefill / CPU / fp32）",
    )
    p.add_argument("--model", required=True, help="本地模型目录，例如 tmp/models/Qwen3-0.6B")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--weights", action="store_true", help="加载真实权重（默认按 config 随机初始化）")
    p.add_argument("--no-collapse", action="store_true", help="不做层模板折叠，输出 28 层全部实例")
    p.add_argument("--out", default=None, help="只写 JSON 到这个路径")
    p.add_argument("--outdir", default=None, help="写全套视图（json/md/mmd/dot/svg/html）到目录")
    p.add_argument("--stem", default="graph", help="--outdir 的文件名前缀")
    p.add_argument("--text", action="store_true", help="打印终端可读的流水线视图")
    p.add_argument("--summary", action="store_true", help="打印 Markdown 概览与节点表")
    p.add_argument("--mermaid", action="store_true", help="打印 Mermaid 源码")
    p.add_argument("--json", dest="print_json", action="store_true", help="把 JSON 打到 stdout")
    p.add_argument("--check", action="store_true", help="运行对账，未通过返回 1")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        graph = extract_graph(
            args.model,
            dtype=args.dtype,
            device=args.device,
            seq_len=args.seq_len,
            collapse_template=not args.no_collapse,
            weights=args.weights,
        )
    except NotImplementedError as exc:
        print(f"[不支持] {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        print("[失败] 抽图过程中出错，见上方堆栈", file=sys.stderr)
        return 2

    checks = graph.reconcile.get("checks", [])
    failed = [c for c in checks if not c["ok"]]
    problems = graph.reconcile.get("validate_problems") or []

    if args.print_json:
        print(graph.to_json())
    elif args.mermaid:
        print(render_mermaid(graph))
    elif args.summary:
        print(render_markdown(graph))
    else:
        print(render_text(graph))

    if args.check:
        print()
        print(f"{'检查项':<32}{'期望':>10}{'实际':>10}  说明")
        print("-" * 88)
        for c in checks:
            mark = "✅" if c["ok"] else "❌"
            print(f"{mark} {c['name']:<30}{str(c['expected']):>10}{str(c['actual']):>10}  {c.get('note') or ''}")
        if problems:
            print("-" * 88)
            for pr in problems:
                print(f"❌ 结构自检：{pr}")
        print("-" * 88)
        print(f"结论：{'通过' if not failed and not problems else '未通过'}"
              f"（{len(checks) - len(failed)}/{len(checks)} 项对账通过，{len(problems)} 处结构问题）")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(graph.to_json())
        print(f"\nJSON 已写入 {args.out}")

    if args.outdir:
        paths = write_all(graph, args.outdir, stem=args.stem)
        print("\n已写入：")
        for kind, path in paths.items():
            print(f"  {kind:<5} {path}")

    if args.check and (failed or problems):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
