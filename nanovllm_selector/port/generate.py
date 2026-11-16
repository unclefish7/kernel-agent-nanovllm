"""调用可注入的 LLM 生成模型文件，并对产物做结构与可导入性检查。

生成函数本身不绑定任何服务：调用方传入 `llm(messages) -> str` 即可，测试里传假函数，
命令行里默认用 `make_llm()` 从环境变量装配（没有配置时给出提示，不抛栈）。
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .prompt import build_messages

LLMFn = Callable[[list[dict]], str]

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)
_CONTEXT_IMPORT = "from nanovllm.utils.context import get_context"


class PortError(RuntimeError):
    """下游移植流程的基类错误。"""


class ValidationError(PortError):
    """生成物没有通过结构校验；`reasons` 逐条给出原因。"""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("生成物未通过校验：\n- " + "\n- ".join(self.reasons))


class LLMNotConfigured(PortError):
    """没有配置调用远端模型所需的环境变量。"""


# ---------------------------------------------------------------------------
# 代码围栏
# ---------------------------------------------------------------------------


def strip_code_fence(text: str) -> str:
    """剥掉 markdown 围栏，只留代码；没有围栏时原样返回去空白后的文本。"""
    if not isinstance(text, str):
        raise TypeError(f"llm 必须返回字符串，收到 {type(text).__name__}")
    if "```" not in text:
        return text.strip()

    blocks = [(lang.lower(), body) for lang, body in _FENCE_RE.findall(text)]
    if blocks:
        python_blocks = [body for lang, body in blocks if lang in ("python", "py")]
        if python_blocks:
            chosen = max(python_blocks, key=len)
        else:
            chosen = max((body for _, body in blocks), key=len)
        return chosen.strip()

    # 只有一行围栏，例如 “```Qwen3ForCausalLM```”，去掉围栏标记本身。
    return text.replace("```", "").strip()


# ---------------------------------------------------------------------------
# 结构校验
# ---------------------------------------------------------------------------


def _for_causal_lm_class(tree: ast.Module) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("ForCausalLM"):
            return node
    return None


def _defines_packed_modules_mapping(class_node: ast.ClassDef) -> bool:
    for node in class_node.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "packed_modules_mapping":
                return True
    return False


def _uses_get_context(tree: ast.Module) -> bool:
    """代码里是否真的调用了 `get_context`。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "get_context":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "get_context":
            return True
    return False


def _imports_get_context(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "nanovllm.utils.context":
            if any(alias.name == "get_context" for alias in node.names):
                return True
    return False


def validate_model_code(code: str) -> list[str]:
    """返回全部不满足项；空列表表示通过。"""
    reasons: list[str] = []
    if not code.strip():
        return ["生成物为空"]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        line = exc.lineno if exc.lineno is not None else "?"
        return [f"不是合法 Python：第 {line} 行 {exc.msg}（offset={exc.offset}）"]

    model_class = _for_causal_lm_class(tree)
    if model_class is None:
        reasons.append("没有定义顶层类名以 `ForCausalLM` 结尾的模型类")
    elif not _defines_packed_modules_mapping(model_class):
        reasons.append(
            f"类 `{model_class.name}` 内没有 `packed_modules_mapping` 赋值"
            "（没有需要打包的权重也要写 `packed_modules_mapping = {}`）"
        )

    # 只有真用到 context 才要求导入：模型文件本身可以完全不碰 Context
    #（`Attention` 在引擎内部自己取），硬要求会逼出用不上的 import。
    if _uses_get_context(tree) and not _imports_get_context(tree):
        reasons.append(f"用了 `get_context` 但缺少导入：`{_CONTEXT_IMPORT}`")

    return reasons


def assert_valid_model_code(code: str) -> None:
    reasons = validate_model_code(code)
    if reasons:
        raise ValidationError(reasons)


def check_importable(
    code: str,
    engine_root: str | Path | None = None,
    *,
    timeout: int = 120,
) -> list[str]:
    """在子进程里真正 import 一次生成代码；需要装了 torch 的环境。

    返回不满足项；空列表表示通过。导入失败时只保留 stderr 末尾几行，避免刷屏。
    """
    reasons = validate_model_code(code)
    if reasons:
        return reasons

    engine_path = Path(engine_root) if engine_root is not None else None
    with tempfile.TemporaryDirectory(prefix="nanovllm_port_") as workdir:
        module_path = Path(workdir) / "generated_model.py"
        module_path.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        if engine_path is not None:
            env["PYTHONPATH"] = str(engine_path) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util, sys;"
                f"spec = importlib.util.spec_from_file_location('generated_model', {str(module_path)!r});"
                "module = importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(module)",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [f"导入超时（>{timeout}s）"]
    if process.returncode == 0:
        return []
    detail = "\n".join(process.stderr.strip().splitlines()[-6:]) or "子进程无输出"
    return [f"导入失败（returncode={process.returncode}）：\n{detail}"]


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


def generate_model(
    graph: Any,
    spec_path: Any,
    kernels_index: Any,
    template_path: Any,
    llm: LLMFn,
    *,
    include_toolbox: bool = True,
) -> str:
    """组装 prompt、调用 llm、剥围栏、校验；返回可直接写盘的代码。"""
    messages = build_messages(
        graph,
        spec_path,
        kernels_index,
        template_path,
        include_toolbox=include_toolbox,
    )
    raw = llm(messages)
    code = strip_code_fence(raw)
    if not code.strip():
        raise ValidationError(["llm 返回为空或围栏里没有代码"])
    assert_valid_model_code(code)
    return code


# ---------------------------------------------------------------------------
# 命令行用的默认 llm（OpenAI 兼容接口）
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_KEY_ENV = "DEEPSEEK_API_KEY"


def make_llm(
    *,
    api_key_env: str = DEFAULT_KEY_ENV,
    api_base: str | None = None,
    model: str | None = None,
    timeout: float = 180.0,
) -> LLMFn:
    """从环境变量装配一个 chat completions 调用函数；缺 key 时抛 LLMNotConfigured。"""
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise LLMNotConfigured(
            f"环境变量 {api_key_env} 未配置；先 export {api_key_env}=... 再运行，"
            "或加 --dry-run 只看 prompt。"
        )
    base = (api_base or os.environ.get("DEEPSEEK_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    model_id = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
    url = base + "/chat/completions"

    def call(messages: list[dict]) -> str:
        payload = json.dumps(
            {"model": model_id, "messages": messages, "temperature": 0.0, "stream": False}
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise PortError(f"{url} 返回 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise PortError(f"请求 {url} 失败：{exc.reason}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise PortError(f"响应里没有 choices：{json.dumps(body, ensure_ascii=False)[:500]}")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish not in (None, "stop"):
            raise PortError(f"生成被截断（finish_reason={finish}）；本次输出不可用")
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise PortError("响应里没有 message.content 文本")
        return content

    return call


__all__ = [
    "LLMFn",
    "LLMNotConfigured",
    "PortError",
    "ValidationError",
    "assert_valid_model_code",
    "check_importable",
    "generate_model",
    "make_llm",
    "strip_code_fence",
    "validate_model_code",
]
