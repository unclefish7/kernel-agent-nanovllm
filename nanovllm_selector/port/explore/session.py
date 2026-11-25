"""宿主机上的工具调用循环。只有独立运行的数值验收能判定候选通过。"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    _tool("read_code", "查实际安装的原生模型或引擎。tree 发现模块和符号，source 取源码。"
          "ref 取 tree 返回的实例路径（如 model.layers.0）或完整点分符号名"
          "（如 nanovllm.layers.attention.Attention）。method 可选具体方法。"
          "config 取模型配置，weights 查真实权重目录。",
          _params({"target": {"enum": ["hf", "engine"], "type": "string"},
                   "ref": STR, "view": {"enum": ["tree", "source", "config", "weights"], "type": "string"},
                   "method": STR, "offset": INT, "limit": INT}, ["target"])),
    _tool("run_model", "运行真实模型，prompt_ids 一次预填充，decode_ids 逐个作为下一步的输入，"
          "每步沿用返回的状态。watch 精确填写 tree 给出的模块路径，根模块是空字符串。"
          "probe 模式可提交 Python，定义 probe(model, step, inputs, output) 返回字典中的张量，"
          "可选 setup(model) 在运行前安装观测。probe 返回的 NAME 位于 module='@probe'、call=0、"
          "port='output.NAME'。可把返回的张量地址直接用于 observe 和 compare。",
          _params({"target": {"type": "string", "enum": ["hf", "probe"]}, "prompt_ids": IDS,
                   "decode_ids": IDS, "watch": {"type": "array", "items": STR}, "probe": STR},
                  ["target", "prompt_ids"])),
    _tool("observe", "按 run_id 查看真实运行记录。将 run_model 返回的某个 selector 原样传入，"
          "读取该张量的形状、类型和数值窗口。view='selectors' 可分页找更多张量地址，"
          "view='aliases' 查询同一模块的其他注册名。"
          "第 0 步是预填充，call 是从 0 开始的调用序号。", _params(
              {**SELECTOR, "view": {"type": "string", "enum": ["selectors", "aliases"]},
               "offset": INT, "limit": INT}, ["run_id"])),
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
先从模块树找到相关边界，用短前向确认后端、形状和状态。从返回的 selectors 中选取相关张量，
原样传给 observe 查看实际数值。遇到源码中难以确认的计算，可以编写 probe 并用 compare 验证。
read_code 可以读取实例路径、类或所调用的完整函数名。检查由配置或运行时后端选择的实际实现。
同一轮可提交多个独立查询。确认关键计算后尽早提交候选，后续可根据首个实测分歧继续探查。
你可以自己实现新算子，使用原生 PyTorch 运算。模型及其特有计算在候选或引擎内执行，
可复用 Transformers 的通用配置、激活与位置编码初始化工具。实际引擎负责权重加载、
请求管理、计算 logits、分页缓存和生成。引擎将不同请求的词元拼在第一维，传入各层的
hidden_states 为 [T, hidden]。将原生模型的维度适配到这个入口，保持单卡各运算的
顺序、数值类型、后端与舍入位置。
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
            if identity.get("weights_loaded") is not True:
                provenance = "此次抽图使用配置初始化的权重。它提供结构信息；当前 checkpoint 的实际分支须用 run_model 取证。"
            elif identity.get("weights_sha256"):
                provenance = "抽图时已加载权重，并已核对本次 checkpoint 的 SHA256。"
            else:
                provenance = "抽图时已加载权重，图未记录可用于核对本次 checkpoint 的权重摘要。"
            parts.append("参考图摘要\n" + provenance + "\n" + _render_graph(data)[:24000])
    return [{"role": "system", "content": "你负责把新模型主动移植到 nano-vLLM。工具结果是证据，最终由独立进程验收。"},
            {"role": "user", "content": "\n\n".join(parts)}]


def _publish(path, text, *, replace=False):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(text.encode("utf-8") if isinstance(text, str) else text)
            stream.flush()
            if replace:
                pending.replace(path)
            else:
                os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def _write(path, data, *, replace=False):
    _publish(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", replace=replace)


def _candidate_errors(code):
    errors = validate_model_code(code, require_template=False)
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


def _tool_reply(event):
    content = json.dumps(event["result"], ensure_ascii=False, allow_nan=False)
    if len(content) > 30000:
        preview = {"truncated": True, "preview": content[:28500],
                   "hint": "请用 offset、limit 或更精确的 ref 查询后续部分。"}
        if isinstance(event["result"], dict) and isinstance(event["result"].get("run_id"), str):
            preview["run_id"] = event["result"]["run_id"][:256]
        content = json.dumps(preview, ensure_ascii=False)
        if len(content) > 30000:
            preview["preview"] = preview["preview"][:14000]
            content = json.dumps(preview, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": event["id"], "content": content}


def _replay_tools(messages, events):
    calls = []
    for event in events:
        raw = event.get("raw_arguments", json.dumps(event["arguments"], ensure_ascii=False))
        calls.append({"id": event["id"], "type": "function", "function": {"name": event["tool"], "arguments": raw}})
    messages.append({"role": "assistant", "content": events[0].get("assistant_content"), "tool_calls": calls})
    messages.extend(_tool_reply(event) for event in events)


def _timeline(out_dir, ledger, previous, *, seeded=False):
    """旧记录中每次完整候选占用工具轮之间的一个空轮；新记录还保存候选轮号。"""
    timeline = {}
    ledger.seek(0)
    while True:
        start = ledger.tell()
        line = ledger.readline()
        if not line:
            break
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if line.endswith(b"\n"):
                raise
            ledger.seek(start)
            ledger.truncate()  # 中断写入的最后半行没有完整的工具结果。
            break
        if not line.endswith(b"\n"):
            ledger.write(b"\n")
        timeline.setdefault(event["turn"], []).append(event)
    candidates = sorted(out_dir.glob("attempt_*.py"), key=lambda p: int(p.stem.split("_")[-1]))
    known = {item["path"]: item for item in previous.get("history", [])}
    if known.keys() - {p.name for p in candidates}:
        raise ValueError("已验收的候选文件缺失")
    if previous.get("pending"):
        known[previous["pending"]["path"]] = previous["pending"]
    turn = 0
    for index, path in enumerate(candidates, 1):
        if path.name != f"attempt_{index}.py":
            raise ValueError("恢复时发现缺失的候选文件")
        record = known.get(path.name, {})
        if record.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"{path.name} 的摘要与已记录候选不同")
        if index == 1 and seeded:
            if record.get("turn", 0) != 0 or 0 in timeline:
                raise ValueError("初始候选与历史轮号冲突")
            turn = 0
        elif "turn" in record:
            next_turn = record["turn"]
            if next_turn <= turn or next_turn in timeline:
                raise ValueError("候选与工具记录的轮号冲突")
            turn = next_turn
        else:
            turn += 1
            while turn in timeline:
                turn += 1
        timeline[turn] = path
    for suffix in ("report", "holdout", "diagnosis"):
        for report in out_dir.glob(f"attempt_*.{suffix}.json"):
            if report.with_suffix("").with_suffix(".py") not in candidates:
                raise ValueError(f"{report.name} 缺少原始候选文件")
    return timeline


def _invalid_report(reason):
    return {"ok": False, "stage": "report", "outcome": "evaluation_error", "error": reason}


def _diagnostic_length(report):
    """优先重现失败的预填充；只有生成失败时用其原始提示的长度。"""
    if report.get("stage") != "compare" or report.get("ok") is not False:
        return None
    prefills = report.get("comparisons") or []
    for row in prefills:
        length = row.get("length")
        if row.get("ok") is False and type(length) is int and length > 0:
            return length
    decode = report.get("decode") or {}
    rows = [*prefills, report.get("comparison") or {}, report.get("generation_prefill") or {},
            decode, *decode.get("items", [])]
    if any(row.get("ok") is False or row.get("exact") is False for row in rows):
        lengths = report.get("lengths") or [row.get("length") for row in prefills]
        return max((n for n in lengths if type(n) is int and n > 0), default=128)
    return None


def _diagnose(candidate, digest, length, runtime, run_missing):
    path = candidate.with_suffix(".diagnosis.json")
    saved = path.exists()
    diagnose = getattr(runtime, "diagnose", None)
    if not saved and not (run_missing and callable(diagnose)):
        return None
    try:
        data = json.loads(path.read_bytes()) if saved else diagnose(candidate, length)
        if (not isinstance(data, dict) or data.get("candidate_sha256", digest) != digest
                or data.get("length", length) != length):
            raise ValueError("附加诊断与本次候选或提示长度不符")
        if not saved:
            _write(path, data)
    except Exception as exc:
        data = {"purpose": "prefill_diagnostic", "length": length, "candidate_sha256": digest,
                "error": f"{type(exc).__name__}: {exc}"[:2000]}
        if not saved and not path.exists():
            _write(path, data)
    return {key: data[key] for key in ("length", "first_difference", "logits", "error") if key in data}


def _judge_candidate(candidate, turn, runtime, summary, messages, log, *, diagnose_missing=True):
    import io
    import tokenize

    original = candidate.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    code = original.decode(tokenize.detect_encoding(io.BytesIO(original).readline)[0])
    messages.append({"role": "assistant", "content": code})

    def report_for(holdout):
        path = candidate.with_suffix(".holdout.json" if holdout else ".report.json")
        saved = path.exists()
        if saved:
            report = json.loads(path.read_bytes())
        elif candidate.read_bytes() != original:
            report = _invalid_report(f"{candidate.name} 已发生变化")
        else:
            errors = [] if holdout else _candidate_errors(code)
            log(f"  验收候选 {candidate.name}" + ("（独立输入）" if holdout else ""))
            report = ({"ok": False, "stage": "class", "error": "\n".join(errors)} if errors
                      else runtime.evaluate(candidate, holdout=holdout))
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            report = _invalid_report(f"验收期间 {candidate.name} 的摘要发生变化")
        elif report.get("candidate_sha256", digest) != digest:
            report = _invalid_report(f"{path.name} 的候选摘要与 {candidate.name} 不符")
        if not saved:
            _write(path, report)
        return report

    report = report_for(False)
    passed = report.get("ok") is True
    summary["history"].append({"path": candidate.name, "turn": turn, "sha256": digest,
                               "stage": report.get("stage"), "ok": passed})
    summary.update(path=candidate.name, stage=report.get("stage"), candidate_sha256=digest)
    if passed:
        holdout = report_for(True)
        summary.update(ok=holdout.get("ok") is True, stage="holdout")
        if holdout.get("outcome") == "evaluation_error":
            summary.update(outcome="evaluation_error", error=holdout.get("error"))
        return True
    if report.get("outcome") in ("reference_unreliable", "evaluation_error"):
        summary.update(outcome=report["outcome"], error=report.get("error"))
        return True
    if candidate.with_suffix(".holdout.json").exists():
        raise ValueError("公开验收未通过的候选已有独立验收记录")
    feedback = "候选验收失败。可继续调用工具，定位后回复完整修订文件。\n" + json.dumps(report, ensure_ascii=False)[:18000]
    length = _diagnostic_length(report)
    if length is not None:
        diagnosis = _diagnose(candidate, digest, length, runtime, diagnose_missing)
        if diagnosis is not None:
            if candidate.read_bytes() != original:
                summary.update(_invalid_report("诊断期间候选文件发生变化"))
                return True
            summary["history"][-1]["diagnosis_file"] = candidate.with_suffix(".diagnosis.json").name
            feedback += "\n附加的单请求预填充诊断：\n" + json.dumps(diagnosis, ensure_ascii=False)[:6000]
    messages.append({"role": "user", "content": feedback})
    return False


def run_session(*, messages, chat, runtime, out_dir: Path, max_attempts: int, max_turns: int,
                resume=False, from_candidate: Path | None = None, log=lambda *_: None) -> dict:
    """从公开工具结果与候选原文恢复对话。轮数、候选次数均为整个会话的累计上限。"""
    if min(max_attempts, max_turns) < 1:
        raise ValueError("调用轮数和候选次数必须为正数")
    prompt = out_dir / "prompt.json"
    seed = out_dir / "seed.py"
    supplied = Path(from_candidate).read_bytes() if from_candidate is not None else None
    if not resume:
        if supplied is not None and max_attempts < 2:
            raise ValueError("种子占用首个候选，--max-iters 至少为 2，以便失败后修订")
        out_dir.mkdir(parents=True, exist_ok=True)
        if supplied is not None:
            if any((out_dir / name).exists() for name in ("prompt.json", "tools.jsonl", "trace.json", "seed.py")
                   ) or any(out_dir.glob("attempt_*")):
                raise FileExistsError("新种子会话的输出目录已有会话文件")
            _publish(seed, supplied)
        elif seed.exists():
            raise FileExistsError("输出目录已有种子文件")
        _write(prompt, messages)
    trace = out_dir / "trace.json"
    with prompt.open(encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.load(lock) != messages:
            raise ValueError("本次初始提示与已有 prompt.json 不同")
        if seed.exists():
            frozen = seed.read_bytes()
            if supplied is not None and supplied != frozen:
                raise ValueError("本次初始候选与会话冻结的种子不同")
            first = out_dir / "attempt_1.py"
            if not first.exists():
                _publish(first, frozen)
            elif first.read_bytes() != frozen:
                return _invalid_report("首个候选与已冻结种子不同")
        elif supplied is not None:
            raise ValueError("已有会话未使用初始候选，无法追加种子")
        previous = json.loads(trace.read_bytes()) if resume and trace.exists() else {}
        with (out_dir / "tools.jsonl").open("a+b" if resume else "x+b", buffering=0) as ledger:
            try:
                timeline = _timeline(out_dir, ledger, previous, seeded=seed.exists()) if resume or seed.exists() else {}
            except ValueError as exc:
                return {**previous, **_invalid_report(str(exc))}
            return _continue(list(messages), chat, runtime, out_dir, max_attempts, max_turns, log,
                             ledger, timeline, previous)


def _continue(messages, chat, runtime, out_dir, max_attempts, max_turns, log, ledger, timeline, previous):
    trace = out_dir / "trace.json"
    turns = max(previous.get("turns", 0), max(timeline, default=0))
    attempts = sum(isinstance(item, Path) for item in timeline.values())
    if turns > max_turns or attempts > max_attempts:
        return {**previous, "ok": False, "stage": "budget", "turns": turns,
                "error": f"总预算低于已有记录（{turns} 轮，{attempts} 个候选）"}
    summary = {"ok": False, "history": [], "tool_calls": 0, "turns": turns}
    terminal = False
    last_candidate = max((turn for turn, item in timeline.items() if isinstance(item, Path)), default=None)
    for turn, item in sorted(timeline.items()):
        if isinstance(item, list):
            _replay_tools(messages, item)
            summary["tool_calls"] += len(item)
        else:
            terminal = _judge_candidate(item, turn, runtime, summary, messages, log,
                                        diagnose_missing=turn == last_candidate)
            if terminal:
                break
    if not terminal and len(summary["history"]) < max_attempts:
        pending = previous.get("pending", {}).get("path")
        pending_done = any(item["path"] == pending for item in summary["history"])
        force_next = previous.get("stage") == "turn_limit" or (previous.get("synthesis_due") and not pending_done)
        for turn in range(turns + 1, max_turns + 1):
            # 每次提交失败后，为重新查证和下一次提交各预留一轮。
            remaining_attempts = max_attempts - len(summary["history"])
            deadline = max(2, max_turns - 2 * (remaining_attempts - 1))
            code_only = force_next or turn >= deadline or turn == max_turns
            force_next = False
            summary.update(turns=turn, stage="chat", synthesis_due=bool(code_only))
            summary.pop("error", None)
            _write(trace, summary, replace=True)
            request = messages
            if code_only:
                request = [*messages, {"role": "user", "content": "本轮请根据已有证据直接提交完整 Python 候选文件。"}]
            elif turn == 1 or turn == deadline - 1:
                request = [*messages, {"role": "user", "content": f"请最迟在第 {deadline} 轮提交下一份完整 Python 候选。"
                           "若验收发现差异，之后仍可使用工具查证。"}]
            log(f"请求 agent，第 {turn}/{max_turns} 轮")
            try:
                response = chat(request, [] if code_only else TOOLS)
            except (OSError, PortError) as exc:
                summary.update(stage="api", error=str(exc))
                break
            calls = response.get("tool_calls") or []
            if calls:
                if code_only:
                    summary.update(stage="synthesis", error="本轮要求完整候选，模型仍请求了工具")
                    force_next = True
                    continue
                messages.append(response)
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
                    if raw != json.dumps(args, ensure_ascii=False):
                        event["raw_arguments"] = raw
                    if call is calls[0] and response.get("content"):
                        event["assistant_content"] = response["content"]
                    ledger.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode())
                    summary["tool_calls"] += 1
                    messages.append(_tool_reply(event))
                continue

            code = strip_code_fence(response.get("content") or "") + "\n"
            attempt = len(summary["history"]) + 1
            candidate = out_dir / f"attempt_{attempt}.py"
            _write(trace, {**summary, "pending": {"path": candidate.name, "turn": turn,
                   "sha256": hashlib.sha256(code.encode()).hexdigest()}}, replace=True)
            _publish(candidate, code)
            terminal = _judge_candidate(candidate, turn, runtime, summary, messages, log)
            summary.pop("synthesis_due", None)
            if terminal or attempt >= max_attempts:
                break
        else:
            summary["stage"] = "turn_limit"
    _write(trace, summary, replace=True)
    return summary
