"""将同一套模型上下文和四个探索工具提供给外部编码 agent。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import sys

from .runtime import DockerRuntime, ROOT
from .session import initial_messages, TOOLS, TOOL_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="0", help="物理 GPU 编号，或 cpu")
    parser.add_argument("--worker-image", default="nanovllm-explore-worker:local")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--context", action="store_true", help="输出初始消息和工具定义")
    action.add_argument("--tool", choices=sorted(TOOL_NAMES), help="从标准输入读取该工具的一组 JSON 参数")
    args = parser.parse_args(argv)
    try:
        if args.context:
            messages = initial_messages(args.model, ROOT / "contracts/nanovllm/spec.yaml",
                                        ROOT / "kernels/index.md", ROOT / "nano-vllm/nanovllm/models/qwen3.py")
            result = {"messages": messages, "tools": TOOLS}
        else:
            raw = sys.stdin.read((1 << 20) + 1)
            if len(raw.encode()) > 1 << 20:
                raise ValueError("工具参数超过 1 MiB")
            arguments = json.loads(raw)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
            runtime = DockerRuntime(args.model, args.out_dir, image=args.worker_image, device=args.device)
            path = runtime.out_dir / "external_tools.jsonl"
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
            with os.fdopen(os.open(path, flags, 0o600), "a", encoding="utf-8") as log:
                info = os.fstat(log.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("工具记录必须是本目录的独立常规文件")
                result = runtime.call(args.tool, arguments)
                event = {"tool": args.tool, "arguments": arguments, "result": result}
                fcntl.flock(log, fcntl.LOCK_EX)
                log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return int(result.get("ok") is False or result.get("status") == "error")


if __name__ == "__main__":
    raise SystemExit(main())
