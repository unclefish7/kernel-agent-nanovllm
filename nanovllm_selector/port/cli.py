"""命令行入口：组装 prompt 或调用模型生成 `nanovllm/models/<model>.py`。

    python -m nanovllm_selector.port.cli \
        --graph nanovllm_selector/outputs/graph_qwen3/graph.json [--dry-run]

`--dry-run` 只拼 prompt、打印统计，不发网络请求，也不需要任何 key。
真正调用时从环境变量取 `DEEPSEEK_API_KEY`，没有配置时打印提示并以退出码 2 结束。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import (
    LLMNotConfigured,
    PortError,
    ValidationError,
    check_importable,
    generate_model,
    make_llm,
)
from .prompt import build_sections, prompt_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "nanovllm_selector" / "outputs"
DEFAULT_GRAPH = OUTPUTS / "graph_qwen3" / "graph.json"
DEFAULT_SPEC = REPO_ROOT / "contracts" / "nanovllm" / "spec.yaml"
DEFAULT_KERNELS = REPO_ROOT / "kernels" / "index.md"
DEFAULT_TEMPLATE = REPO_ROOT / "nano-vllm" / "nanovllm" / "models" / "qwen3.py"
DEFAULT_OUT = OUTPUTS / "generated.py"
DEFAULT_ENGINE_ROOT = REPO_ROOT / "nano-vllm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanovllm_selector.port.cli",
        description="把参考图 + 引擎规约 + 算子库索引 + 同构模板交给 LLM，生成模型文件。",
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="参考图 JSON")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="引擎规约 spec.yaml")
    parser.add_argument("--kernels", type=Path, default=DEFAULT_KERNELS, help="算子库索引 index.md")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="同构模板 qwen3.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="生成文件写到哪")
    parser.add_argument("--dry-run", action="store_true", help="只拼 prompt、打印统计，不调用模型")
    parser.add_argument("--no-toolbox", action="store_true", help="prompt 里省略 toolbox 表")
    parser.add_argument(
        "--check-import",
        action="store_true",
        help="写盘后在子进程里真正 import 一次（需要 torch，较慢）",
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=DEFAULT_ENGINE_ROOT,
        help="含 nanovllm 包的目录，供 --check-import 设置 PYTHONPATH",
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="存放 API key 的环境变量名")
    parser.add_argument("--api-base", default=None, help="OpenAI 兼容接口根地址")
    parser.add_argument("--model", default=None, help="模型名，默认读 DEEPSEEK_MODEL")
    parser.add_argument("--timeout", type=float, default=180.0, help="单次请求超时秒数")
    return parser


def _fallback(path: Path, default: Path, relative: Path) -> Path:
    """默认路径不存在时退到 tmp/port-deps（容器里 contracts/、kernels/ 未挂载）。"""
    if path == default and not path.is_file():
        candidate = REPO_ROOT / "tmp" / "port-deps" / relative
        if candidate.is_file():
            return candidate
    return path


def _resolve_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.spec = _fallback(
        args.spec, DEFAULT_SPEC, Path("contracts") / "nanovllm" / "spec.yaml"
    )
    args.kernels = _fallback(args.kernels, DEFAULT_KERNELS, Path("kernels") / "index.md")
    args.template = _fallback(
        args.template, DEFAULT_TEMPLATE, Path("nano-vllm") / "nanovllm" / "models" / "qwen3.py"
    )
    for label, path in (
        ("参考图", args.graph),
        ("引擎规约", args.spec),
        ("算子库索引", args.kernels),
        ("同构模板", args.template),
    ):
        if not path.is_file():
            parser.error(f"找不到{label}：{path}")


def _print_stats(stats: dict) -> None:
    print("prompt 统计（未调用模型）")
    print(f"  chars: {stats['chars']}")
    print(f"  nodes: {stats['nodes']}（模板）/ {stats['instance_nodes']}（实例）")
    print(f"  edges: {stats['edges']}")
    print(f"  template_nodes: {stats['template_nodes']}")
    if "must" in stats:
        print(
            f"  spec: must {stats['must']} / toolbox {stats['toolbox']} / "
            f"context {stats['context_fields']}"
        )
    print("  sections:")
    for name, size in stats["sections"].items():
        print(f"    {name}: {size}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_inputs(parser, args)

    sections = build_sections(
        args.graph,
        args.spec,
        args.kernels,
        args.template,
        include_toolbox=not args.no_toolbox,
    )
    stats = prompt_stats(sections, args.graph, args.spec)

    if args.dry_run:
        _print_stats(stats)
        return 0

    try:
        llm = make_llm(
            api_key_env=args.api_key_env,
            api_base=args.api_base,
            model=args.model,
            timeout=args.timeout,
        )
    except LLMNotConfigured as exc:
        print(f"未调用模型：{exc}", file=sys.stderr)
        return 2

    print(f"prompt 字符数：{stats['chars']}；节点：{stats['nodes']}/{stats['instance_nodes']}")
    try:
        code = generate_model(
            args.graph,
            args.spec,
            args.kernels,
            args.template,
            llm,
            include_toolbox=not args.no_toolbox,
        )
    except ValidationError as exc:
        print("生成物未通过校验：", file=sys.stderr)
        for reason in exc.reasons:
            print(f"- {reason}", file=sys.stderr)
        return 1
    except PortError as exc:
        print(f"调用模型失败：{exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(code + "\n", encoding="utf-8")
    print(f"已写入 {args.out}（{len(code)} 字符，通过结构校验）")

    if args.check_import:
        reasons = check_importable(code, args.engine_root)
        if reasons:
            print("导入检查未通过：", file=sys.stderr)
            for reason in reasons:
                print(f"- {reason}", file=sys.stderr)
            return 1
        print("导入检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
