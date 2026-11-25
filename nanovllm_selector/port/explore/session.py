"""宿主机上的工具调用循环。只有独立运行的数值验收能判定候选通过。"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from ..generate import PortError, strip_code_fence, validate_model_code


def _params(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _tool(name, description, parameters):
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": parameters,
    }}


STR = {"type": "string"}
INT = {"type": "integer", "minimum": 0}
IDS = {"type": "array", "items": INT}
SELECTOR = {"run_id": STR, "step": INT, "module": STR, "call": INT, "port": STR}
ITEM = _params(SELECTOR, SELECTOR)

TOOLS = [
    _tool("read_code", "查实际安装的原生模型或引擎。tree 发现模块和符号，source 取源码，"
          "可跟进源码里的类、函数和方法。config 取模型配置，weights 查真实权重目录。",
          _params({"target": {"enum": ["hf", "engine"], "type": "string"},
                   "ref": STR, "view": {"enum": ["tree", "source", "config", "weights"], "type": "string"},
                   "method": STR, "offset": INT, "limit": INT}, ["target"])),
    _tool("run_model", "运行真实模型，prompt_ids 一次预填充，decode_ids 逐个作为下一步的输入，"
          "每步沿用返回的状态。watch 精确填写 tree 给出的模块路径，根模块是空字符串。"
          "probe 模式可提交 Python，定义 probe(model, step, inputs, output) 返回字典中的张量，"
          "可选 setup(model) 在运行前安装观测。结果属于该次实验。",
          _params({"target": {"type": "string", "enum": ["hf", "probe"]}, "prompt_ids": IDS,
                   "decode_ids": IDS, "watch": {"type": "array", "items": STR}, "probe": STR},
                  ["target", "prompt_ids"])),
    _tool("observe", "按 run_id 列举实际捕获的记录。填具体 step、module、call、port 可读到数值窗口。"
          "第 0 步是预填充，call 从 0 计数，先列目录获取准确的 port。", _params(
              {**SELECTOR, "offset": INT, "limit": INT}, ["run_id"])),
    _tool("compare", "在本地比较两个已保存的完整张量。先从 observe 获取两边的完整地址。"
          "检查形状、数据类型、有限性和数值，给出差异位置。",
          _params({"reference": ITEM, "actual": ITEM}, ["reference", "actual"])),
]
TOOL_NAMES = {item["function"]["name"] for item in TOOLS}


def initial_messages(model: Path, spec: Path, kernels: Path, template: Path,
                     graph: Path | None = None) -> list[dict]:
    from ..prompt import _check_agent_graph, _render_graph, load_graph

    config_bytes = (model / "config.json").read_bytes()
    config = json.loads(config_bytes)
    task = """在单卡 nano-vLLM 中实现此 checkpoint 对应的完整 Python 模型。请主动查实际原生
模型的源码、它调用的函数、参数名、激活值和状态，用短序列和至少两步 decode 查证数据依赖。
按运行证据决定需要读取的模块和工具。通过工具查看实现时可用多个调用查询不同模块。
你可以自己实现新算子，使用原生 PyTorch 运算。模型及其特有计算在候选或引擎内执行，
可复用 Transformers 的通用配置、激活与位置编码初始化工具。实际引擎负责权重加载、
请求管理、计算 logits、分页缓存和生成。保持单卡各运算的顺序、形状、dtype、后端和舍入。
不要从候选调用 Hugging Face 原模型或更改引擎的评测环境。先查观察依据，准备好后
直接回复完整 Python 文件。宿主机会把失败报告送回本对话，供下一轮定位。"""
    parts = [task, "本地 checkpoint 配置\n" + json.dumps(config, ensure_ascii=False),
             "引擎规约\n" + spec.read_text(), "可参考的算子目录\n" + kernels.read_text(),
             "已注册模型的接口示例\n" + template.read_text(),
             "已有模型上的严格数值经验，是否适用请以当前模型的实际源码和运行结果判断\n" +
             (Path(__file__).resolve().parents[1] / "exact_reference.md").read_text()]
    if graph is not None:
        data = load_graph(graph)
        try:
            identity = data.get("model") or {}
            if identity.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest():
                raise ValueError("参考图的配置摘要与当前 checkpoint 无法对应")
            if identity.get("weights_sha256"):
                with (model / "model.safetensors").open("rb") as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != identity["weights_sha256"]:
                    raise ValueError("参考图的权重摘要与当前 checkpoint 不同")
            _check_agent_graph(data)
        except (OSError, ValueError, RuntimeError) as exc:
            parts.append(f"该图的验证状态有缺口，请用工具重新取证。验证报告摘要：{str(exc)[:1200]}")
        else:
            parts.append("参考图摘要\n" + _render_graph(data)[:24000])
    return [{"role": "system", "content": "你负责把新模型主动移植到 nano-vLLM。工具结果是证据，最终由独立进程验收。"},
            {"role": "user", "content": "\n\n".join(parts)}]


def _write(path: Path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _candidate_errors(code):
    errors = validate_model_code(code)
    if errors:
        return errors
    for node in ast.walk(ast.parse(code)):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if node.module == "transformers":
                modules += [alias.name for alias in node.names]
        if any((name.startswith("transformers.models.") and ".modeling_" in name)
               or name.startswith("AutoModel") or name.endswith("ForCausalLM") for name in modules):
            errors.append("模型主体需在候选文件或 nano-vLLM 中执行，请根据已查的原生实现写出对应的运算。")
            break
    return errors


def run_session(*, messages, chat, runtime, out_dir: Path, max_attempts: int, max_turns: int,
                log=lambda *_: None) -> dict:
    """将工具响应按原始调用 ID 送回模型。首次通过公开用例后冻结该文件再验独立输入。"""
    if min(max_attempts, max_turns) < 1:
        raise ValueError("调用轮数和候选次数必须为正数")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "prompt.json", messages)
    messages = list(messages)
    history = []
    summary = {"ok": False, "history": history, "tool_calls": 0, "turns": 0}
    with (out_dir / "tools.jsonl").open("x", encoding="utf-8", buffering=1) as ledger:
        for turn in range(1, max_turns + 1):
            summary["turns"] = turn
            log(f"请求 agent，第 {turn}/{max_turns} 轮")
            try:
                response = chat(messages, TOOLS)
            except (OSError, PortError) as exc:
                summary.update(stage="api", error=str(exc))
                break
            messages.append(response)
            calls = response.get("tool_calls") or []
            if calls:
                for call in calls:
                    fn = call["function"]
                    name = fn["name"]
                    raw = fn.get("arguments")
                    args = None
                    try:
                        if name not in TOOL_NAMES:
                            raise ValueError(f"没有工具 {name}")
                        args = json.loads(raw)
                        if not isinstance(args, dict):
                            raise ValueError("参数必须是 JSON 对象")
                        log(f"  调用 {name}")
                        result = runtime.call(name, args)
                    except (OSError, TypeError, ValueError, RuntimeError) as exc:
                        result = {"ok": False, "error": str(exc)[:2000]}
                    event = {"turn": turn, "id": call["id"], "tool": name, "arguments": args,
                             "result": result}
                    ledger.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    summary["tool_calls"] += 1
                    content = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    if len(content) > 30000:
                        content = json.dumps({"truncated": True, "preview": content[:28500],
                                              "hint": "请用 offset、limit 或更精确的 ref 查询后续部分。"},
                                             ensure_ascii=False)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
                continue

            code = strip_code_fence(response.get("content") or "") + "\n"
            attempt = len(history) + 1
            candidate = out_dir / f"attempt_{attempt}.py"
            with candidate.open("x", encoding="utf-8") as stream:
                stream.write(code)
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            log(f"  验收候选 {candidate.name}")
            errors = _candidate_errors(code)
            report = ({"ok": False, "stage": "class", "error": "\n".join(errors)}
                      if errors else runtime.evaluate(candidate))
            _write(candidate.with_suffix(".report.json"), report)
            history.append({"path": candidate.name, "sha256": digest, "stage": report.get("stage"),
                            "ok": report.get("ok") is True})
            summary.update(path=candidate.name, stage=report.get("stage"), candidate_sha256=digest)
            if report.get("ok") is True:
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
                    raise RuntimeError("验收后候选文件摘要发生变化")
                holdout = runtime.evaluate(candidate, holdout=True)
                _write(candidate.with_suffix(".holdout.json"), holdout)
                summary.update(ok=holdout.get("ok") is True, stage="holdout")
                break
            if len(history) == max_attempts or report.get("outcome") in ("reference_unreliable", "evaluation_error"):
                break
            messages.append({"role": "user", "content": "候选验收失败。可继续调用工具，定位后回复完整修订文件。\n"
                             + json.dumps(report, ensure_ascii=False)[:18000]})
        else:
            summary["stage"] = "turn_limit"
    _write(out_dir / "trace.json", summary)
    return summary
