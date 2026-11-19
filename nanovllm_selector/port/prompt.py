"""把四样材料组装成给 LLM agent 的一次输入。

四样材料：

1. 参考图（`Graph` 或 `graph.json`）：模板折叠后的节点、边、形状、公式；
2. 引擎规约（`contracts/nanovllm/spec.yaml`）：must / toolbox / context；
3. 算子库索引（`kernels/index.md`）：只供查看写法，不供调用；
4. 同构模板（`nano-vllm/nanovllm/models/qwen3.py`）：类划分与写法基准。

本模块只做文本组装，不调用模型、不依赖 torch。图按模板折叠后的节点展开，
`instances` 只显示条数，不逐个列出实例。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SYSTEM_PROMPT = """你是把 HuggingFace 模型移植到 nano-vLLM 的工程师。
只输出一个完整的 Python 模型文件，放在 ```python 代码围栏里；围栏外不要写解释、测试或命令。"""

SECTION_ORDER = ("task", "graph", "spec", "kernels", "template")

_SECTION_TITLES = {
    "task": "任务",
    "graph": "参考图",
    "spec": "引擎规约",
    "kernels": "算子库索引",
    "template": "同构模板",
}


# ---------------------------------------------------------------------------
# 读入材料
# ---------------------------------------------------------------------------


def _read_text(value: Any, label: str) -> str:
    """接受路径或现成文本；路径读不到就报错，不把路径当正文。"""
    if isinstance(value, Path):
        return Path(value).read_text(encoding="utf-8")
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if "\n" in value:
            return value
        raise FileNotFoundError(f"找不到{label}：{value}")
    raise TypeError(f"{label} 必须是路径或文本，收到 {type(value).__name__}")


def load_graph(graph: Any) -> dict:
    """把 Graph 实例 / dict / 路径 / JSON 文本统一成 dict。"""
    if isinstance(graph, dict):
        return graph
    if hasattr(graph, "to_dict"):
        return graph.to_dict()
    if isinstance(graph, Path):
        text = graph.read_text(encoding="utf-8")
    elif isinstance(graph, str):
        path = Path(graph)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        elif graph.lstrip().startswith("{"):
            text = graph
        else:
            raise FileNotFoundError(f"找不到参考图：{graph}")
    else:
        raise TypeError(f"graph 必须是 Graph、dict 或路径，收到 {type(graph).__name__}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"参考图不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("参考图 JSON 的顶层必须是对象")
    return data


def load_spec(spec: Any) -> dict:
    """读入 spec.yaml；也接受已经解析好的 dict。"""
    if isinstance(spec, dict):
        return spec
    text = _read_text(spec, "引擎规约")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"引擎规约不是合法 YAML：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("引擎规约的顶层必须是 YAML 映射")
    return data


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    """表格单元格：压掉换行，转义竖线。"""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _fmt_shape(shape: Any) -> str:
    if shape is None:
        return "?"
    if isinstance(shape, (list, tuple)):
        if not shape:
            return "[]"
        return "[" + ", ".join(str(x) for x in shape) + "]"
    return str(shape)


def _fmt_dtype(dtype: Any) -> str:
    return str(dtype or "?").replace("torch.", "")


def _fmt_tensor(ref: dict, tensors: dict, *, output: bool = False, profile=None) -> str:
    meta = tensors.get(ref.get("tensor")) or {}
    index = ref.get("index")
    slot = "out" if output else "in"
    head = f"{slot}[{index}]" if index is not None else slot
    pieces = [f"`{head}`"]
    arg = ref.get("arg_name")
    if arg:
        pieces.append(f"arg=`{arg}`")
    path = ref.get("path")
    if path:
        pieces.append(f"path=`{path}`")
    pieces.append(_fmt_shape(meta.get("shape")))
    pieces.append(_fmt_dtype(meta.get("dtype")))
    if meta.get("stride") is not None:
        pieces.append("步长=" + _fmt_shape(meta["stride"]))
    if meta.get("offset"):
        pieces.append(f"偏移={meta['offset']}")
    if ref.get("role") in ("key", "value"):
        pieces.append(ref["role"])
    if profile:
        pieces.append("轴=" + _fmt_shape(profile["axes"]))
        pieces.append("样本=" + "/".join(_fmt_shape(shape) for shape in profile["samples"]))
    return " ".join(pieces)


def _model_names(graph: dict) -> tuple[str, str, str]:
    model = graph.get("model") or {}
    raw = str(model.get("model_type") or "model")
    filename = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_") or "model"
    prefix = "".join(part[:1].upper() + part[1:] for part in filename.split("_") if part)
    return raw, filename, prefix or "Model"


def _fold_layer_index(name: str) -> str:
    return re.sub(r"\.layers\.\d+\.", ".layers.*.", name)


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------


def _render_task(graph: dict) -> str:
    model_type, filename, prefix = _model_names(graph)
    return f"""# 一、任务

把第 2 节的参考图移植成一份能在 nano-vLLM 里跑的模型文件。

- 输出文件：`nanovllm/models/{filename}.py`
- 主类：`{prefix}ForCausalLM`
- 只输出一个 Python 文件，放在 ```python 围栏里；围栏外不要有解释、测试或命令行代码。

## 1.1 硬性要求

1. 第 3 节的 must 全部成立；第 5 节的 `qwen3.py` 是写法基准，类划分、Context 用法、`packed_modules_mapping` 写法都要同构同式。
2. 文件顶层必须有一个类名以 `ForCausalLM` 结尾的模型类，并且类体内有 `packed_modules_mapping` 类属性；没有需要打包的权重也写 `{{}}`。
3. 文件里必须出现 `from nanovllm.utils.context import get_context`。
4. 只使用标准库、`torch`、`transformers` 和 `nanovllm` 已有模块；不要新增依赖，不要修改引擎。
5. 第 4 节的 `kernels/` 只看写法，不得 import 或调用；算子最终由你自己写。
6. 权重名按第 2.3 节和模板的 `packed_modules_mapping` 映射；需要按 tp 切分的参数使用模板里的并行层。
7. 除 KV cache 外不要保存跨 step 状态；KV cache 只通过 `Attention` 读写。

模型类型是 `{model_type}`。形状来自实测；有多个样本时按样本解释各轴。`×N` 表示对应计算在 N 个层实例中重复。
"""


# ---------------------------------------------------------------------------
# 参考图
# ---------------------------------------------------------------------------


def _render_graph(graph: dict) -> str:
    model = graph.get("model") or {}
    env = graph.get("env") or {}
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    tensors = graph.get("tensors") or {}
    stats = (graph.get("reconcile") or {}).get("stats") or {}
    model_type, _, _ = _model_names(graph)

    instance_nodes = stats.get("instance_nodes") or sum(
        len(n.get("instances") or []) or 1 for n in nodes
    )
    op_counts = stats.get("op_counts") or {}
    lines = ["# 二、参考图", ""]
    lines.append(
        "> 这张图从模型实际执行中抽取；`×N` 表示该模板在模型里重复 N 次。"
        "图只描述本模型算什么，不规定在 nano-vLLM 里怎么实现。"
    )
    lines.append(
        "> `in[i]` 是输入槽位；`arg=`/`path=` 只记录录制时的调用参数位置，不保证可读。"
        "形状为 `[0]` 的输入表示录制时该张量为空（例如还没写入的 KV cache）。"
    )
    lines += ["", "## 2.1 概览", ""]
    architectures = model.get("architectures") or []
    lines.append(f"- 模型：`{model_type}`（architectures：{', '.join(architectures) or '未记录'}）")
    if model.get("num_layers") is not None:
        lines.append(f"- 层数：{model['num_layers']}")
    lines.append(
        f"- 参考执行：{_fmt_dtype(env.get('dtype'))} / {env.get('device', '未记录')} / "
        f"phases：{', '.join(graph.get('phases') or []) or '未记录'}"
    )
    lines.append(
        f"- 节点：{len(nodes)}（模板）/ {instance_nodes}（实例）；边：{len(edges)}"
    )
    if op_counts:
        lines.append(
            "- 算子计数（按实例展开）："
            + "、".join(f"{name} {count}" for name, count in op_counts.items())
        )
    inputs = graph.get("graph_inputs") or []
    if inputs:
        lines.append("- 图输入：" + "、".join(f"`{x}`" for x in inputs))
    for phase, results in (graph.get("graph_outputs") or {}).items():
        for i, ref in enumerate(results):
            shape = _fmt_shape(tensors[ref["tensor"]]["shape"])
            lines.append(f"- {phase} 模型返回[{i}]：{ref['node']}.out[{ref['slot']}]，返回形状 {shape}")
    probes = env.get("shape_probes")
    if probes:
        lines.append(f"- 尺寸样本按（批大小，提示长度）排列：{probes['fit']} 用于推断，"
                     f"{probes['holdout']} 独立检验。B=批大小，S=本次输入长度，"
                     "P=本步开始前已保存的位置数。数字表示样本中固定，? 表示含义未确定。")

    config = model.get("config")
    if config:
        lines += ["", "## 2.2 模型 config（写代码时直接取值）", "", "```json"]
        lines.append(json.dumps(config, ensure_ascii=False, indent=2))
        lines += ["```"]

    params = graph.get("params") or []
    if params:
        folded: dict[tuple[str, tuple], int] = {}
        for param in params:
            name = _fold_layer_index(str(param.get("name") or ""))
            shape = tuple(param.get("shape") or [])
            key = (name, shape)
            folded[key] = folded.get(key, 0) + 1
        lines += ["", "## 2.3 权重名（按层号折叠去重）", ""]
        for (name, shape), count in folded.items():
            suffix = f" ×{count}" if count > 1 else ""
            lines.append(f"- `{name}` {_fmt_shape(list(shape))}{suffix}")

    lines += ["", "## 2.4 逐节点", ""]
    for number, node in enumerate(nodes, 1):
        node_id = node.get("id", f"node{number}")
        op = node.get("op", "unknown")
        instances = node.get("instances") or []
        repeat = f"，×{len(instances)}" if instances else "，×1"
        lines.append(f"### {number}. `{node_id}` — {op}{repeat}")
        impl = node.get("impl") or {}
        impl_bits = []
        if impl.get("class"):
            impl_bits.append(str(impl["class"]))
        if impl.get("module_path"):
            impl_bits.append(f"`{impl['module_path']}`")
        if impl_bits:
            lines.append("- 实现：" + "，".join(impl_bits))
        formula = (node.get("semantics") or {}).get("formula")
        if formula:
            lines.append(f"- 公式：{formula}")

        node_inputs = node.get("inputs") or []
        if node_inputs:
            lines.append("- 输入：")
            profiles = node.get("attrs", {}).get("shape_profile", {}).get("inputs")
            for i, ref in enumerate(node_inputs):
                lines.append("  - " + _fmt_tensor(ref, tensors, profile=profiles[i] if profiles else None))
        else:
            lines.append("- 输入：无")
        node_outputs = node.get("outputs") or []
        if node_outputs:
            lines.append("- 输出：")
            profiles = node.get("attrs", {}).get("shape_profile", {}).get("outputs")
            for i, ref in enumerate(node_outputs):
                lines.append("  - " + _fmt_tensor(ref, tensors, output=True, profile=profiles[i] if profiles else None))
        else:
            lines.append("- 输出：无")
        state = node.get("state") or []
        if state:
            lines.append("- 状态：")
            for item in state:
                detail = (
                    f"`{item.get('name')}`（{item.get('kind', '?')}，{item.get('role') or '?'}）"
                    f" {_fmt_shape(item.get('shape'))} {_fmt_dtype(item.get('dtype'))}"
                )
                lines.append("  - " + detail)
        else:
            lines.append("- 状态：无")
        lines.append("")

    node_by_id = {n.get("id"): n for n in nodes}
    lines += ["## 2.5 边", ""]
    if not edges:
        lines.append("- 无")
    for edge in edges:
        source = edge.get("from_node", "?")
        target = edge.get("to_node", "?")
        target_node = node_by_id.get(target) or {}
        slot = ""
        for ref in target_node.get("inputs") or []:
            if ref.get("tensor") == edge.get("tensor"):
                index = ref.get("index")
                arg = ref.get("arg_name")
                bits = [f"in[{index}]" if index is not None else "in"]
                if arg:
                    bits.append(f"arg=`{arg}`")
                slot = "（" + " ".join(bits) + "）"
                break
        kind = edge.get("kind") or "data"
        note = "（层间回边，不参与拓扑排序）" if kind == "layer" else ""
        if edge.get("source_slot") is not None:
            source += f".out[{edge['source_slot']}]"
            slot = f".in[{edge['target_slot']}]"
        if kind == "state":
            note = f"（跨步读取上一步本层写入的 {edge.get('role')}）"
        lines.append(f"- `{source}` → `{target}`{slot}{note}")

    order, cyclic = _topological_order(graph)
    lines += ["", "## 2.6 按拓扑序的摘要", ""]
    if cyclic:
        lines.append("> 图里剩下无法排序的环，按原顺序补在末尾；层间回边已排除。")
        lines.append("")
    for rank, node in enumerate(order, 1):
        instances = node.get("instances") or []
        repeat = f"，×{len(instances)}" if instances else "，×1"
        outputs = node.get("outputs") or []
        shapes = []
        for ref in outputs[:2]:
            meta = tensors.get(ref.get("tensor")) or {}
            shapes.append(_fmt_shape(meta.get("shape")))
        tail = f" → {', '.join(shapes)}" if shapes else ""
        lines.append(f"{rank}. `{node.get('id')}`（{node.get('op', 'unknown')}{repeat}）{tail}")

    return "\n".join(lines).strip()


def _topological_order(graph: dict) -> tuple[list[dict], bool]:
    """按数据边排序；`kind == "layer"` 的层间回边表示层循环，排除在外。"""
    nodes = list(graph.get("nodes") or [])
    ids = [n.get("id") for n in nodes]
    rank = {node_id: i for i, node_id in enumerate(ids)}
    indegree = {node_id: 0 for node_id in ids}
    adjacent: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get("edges") or []:
        if (edge.get("kind") or "data") == "layer":
            continue
        source = edge.get("from_node")
        target = edge.get("to_node")
        if source not in indegree or target not in indegree or source == target:
            continue
        adjacent[source].append(target)
        indegree[target] += 1

    ready = sorted((rank[node_id], node_id) for node_id, degree in indegree.items() if degree == 0)
    order_ids: list[str] = []
    while ready:
        _, node_id = ready.pop(0)
        order_ids.append(node_id)
        for target in adjacent.get(node_id, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append((rank[target], target))
        ready.sort()

    cyclic = len(order_ids) < len(nodes)
    if cyclic:
        done = set(order_ids)
        order_ids.extend(node_id for node_id in ids if node_id not in done)
    return [nodes[rank[node_id]] for node_id in order_ids], cyclic


# ---------------------------------------------------------------------------
# 引擎规约
# ---------------------------------------------------------------------------


def _render_spec(spec: dict, include_toolbox: bool = True) -> str:
    meta = spec.get("meta") or {}
    must = spec.get("must") or []
    toolbox = spec.get("toolbox") or []
    context = spec.get("context") or {}
    fields = context.get("fields") or []

    lines = ["# 三、引擎规约", ""]
    lines.append(
        "> 来源：`contracts/nanovllm/spec.yaml`"
        f"（引擎 {meta.get('engine', 'nano-vLLM')}"
        + (f"，提交 `{meta['commit'][:7]}`" if meta.get("commit") else "")
        + "）。must 必须成立；toolbox 可用可不用；context 是前向时能读到的批信息。"
    )
    lines += ["", f"## 3.1 must（{len(must)} 条，必须成立）", ""]
    if not must:
        lines.append("- 无")
    for item in must:
        title = item.get("title")
        heading = f"### {item.get('id', '?')}" + (f" {title}" if title else "")
        lines += [heading, ""]
        statement = str(item.get("statement") or "").strip()
        if statement:
            lines += [statement, ""]
        evidence = item.get("evidence") or []
        if evidence:
            lines.append("- 证据：")
            for ref in evidence:
                lines.append(f"  - `{str(ref).strip()}`")
        lines.append("")

    if include_toolbox:
        lines += [f"## 3.2 toolbox（{len(toolbox)} 条，可用可不用）", ""]
        if toolbox:
            lines.append("| 名称 | 类别 | 签名 | 源码 | 给什么 | 坑 |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in toolbox:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{_cell(item.get('name'))}`",
                            _cell(item.get("kind")),
                            _cell(item.get("signature")),
                            f"`{_cell(item.get('source'))}`",
                            _cell(item.get("gives")),
                            _cell(item.get("notes")),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("- 无")
        lines.append("")

    lines += [f"## 3.3 context（{len(fields)} 个字段）", ""]
    access = context.get("access")
    if access:
        lines.append(f"- 取法：`{access}`")
    built_by = context.get("built_by") or []
    if built_by:
        lines.append("- 写入：" + "、".join(f"`{str(x).strip()}`" for x in built_by))
    if fields:
        lines.append("")
        lines.append("| 字段 | 类型 / shape | 何时有值 | 含义 |")
        lines.append("| --- | --- | --- | --- |")
        for field in fields:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_cell(field.get('name'))}`",
                        _cell(field.get("type")),
                        _cell(field.get("when")),
                        _cell(field.get("meaning")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- 无字段")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 算子库与模板
# ---------------------------------------------------------------------------


def _render_kernels(text: str) -> str:
    body = text.strip()
    if body.startswith("# 算子库索引"):
        body = body.split("\n", 1)[1].lstrip() if "\n" in body else ""
    return f"""# 四、算子库索引

> 只能看写法：不要 import，不要调用，也不要把库里的接口当成约定；库里没有的算子由你自己写。

{body}
"""


def _render_template(text: str) -> str:
    return f"""# 五、同构模板：`qwen3.py`

> 生成物必须与它同构同式：同样的类划分（Attention / MLP / DecoderLayer / Model / ForCausalLM）、同样的 Context 用法、同样的 `packed_modules_mapping` 写法。

```python
{text.strip()}
```
"""


# ---------------------------------------------------------------------------
# 对外组装
# ---------------------------------------------------------------------------


def build_sections(
    graph: Any,
    spec_path: Any,
    kernels_index: Any,
    template_path: Any,
    *,
    include_toolbox: bool = True,
    exact: bool = True,
) -> dict[str, str]:
    """生成五个小节，键与 `SECTION_ORDER` 一致。"""
    graph_data = load_graph(graph)
    if graph_data.get("reconcile", {}).get("ok") is False:
        raise ValueError("参考图未通过抽图检查")
    if any("observed_outputs" in node.get("attrs", {}) for node in graph_data.get("nodes", [])):
        from nanovllm_selector.extract.schema import Graph

        problems = Graph.from_dict(graph_data).validate()
        if problems:
            raise ValueError("参考图与实测记录不符：" + "；".join(problems[:5]))
    spec_data = load_spec(spec_path)
    kernels_text = _read_text(kernels_index, "算子库索引")
    template_text = _read_text(template_path, "模型模板")
    engine_context = _render_spec(spec_data, include_toolbox=include_toolbox)
    template = _render_template(template_text)
    if exact:
        engine_context += "\n\n" + Path(__file__).with_name("exact_reference.md").read_text(encoding="utf-8")
        template += ("\n\n本次使用逐项相等验收。第 3 节的单卡实现依据给出了经过验证的回退代码。"
                     "遇到相应运算时，请把这些计算写进候选文件；模板的类划分和权重装载接口继续沿用。")
    return {
        "task": _render_task(graph_data),
        "graph": _render_graph(graph_data),
        "spec": engine_context,
        "kernels": _render_kernels(kernels_text),
        "template": template,
    }


def build_prompt(
    graph: Any,
    spec_path: Any,
    kernels_index: Any,
    template_path: Any,
    *,
    include_toolbox: bool = True,
    exact: bool = True,
) -> str:
    """把四样材料拼成一个完整 prompt 字符串。"""
    sections = build_sections(
        graph,
        spec_path,
        kernels_index,
        template_path,
        include_toolbox=include_toolbox, exact=exact,
    )
    return "\n\n".join(sections[name] for name in SECTION_ORDER)


def build_messages(
    graph: Any,
    spec_path: Any,
    kernels_index: Any,
    template_path: Any,
    *,
    include_toolbox: bool = True,
    exact: bool = True,
) -> list[dict]:
    """转成 chat 接口的 messages；system 只放输出约束。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_prompt(
                graph,
                spec_path,
                kernels_index,
                template_path,
                include_toolbox=include_toolbox, exact=exact,
            ),
        },
    ]


def prompt_stats(
    sections: dict[str, str],
    graph: Any,
    spec: Any = None,
) -> dict:
    """给 CLI 的 dry-run 统计：字符数、节点数、边数。"""
    text = "\n\n".join(sections[name] for name in SECTION_ORDER if name in sections)
    graph_data = load_graph(graph)
    nodes = list(graph_data.get("nodes") or [])
    stats = (graph_data.get("reconcile") or {}).get("stats") or {}
    result = {
        "chars": len(text),
        "sections": {name: len(value) for name, value in sections.items()},
        "nodes": len(nodes),
        "template_nodes": sum(1 for node in nodes if node.get("instances")),
        "instance_nodes": stats.get("instance_nodes")
        or sum(len(node.get("instances") or []) or 1 for node in nodes),
        "edges": len(graph_data.get("edges") or []),
    }
    if spec is not None:
        spec_data = load_spec(spec)
        result["must"] = len(spec_data.get("must") or [])
        result["toolbox"] = len(spec_data.get("toolbox") or [])
        result["context_fields"] = len((spec_data.get("context") or {}).get("fields") or [])
    return result


__all__ = [
    "SECTION_ORDER",
    "SYSTEM_PROMPT",
    "build_messages",
    "build_prompt",
    "build_sections",
    "load_graph",
    "load_spec",
    "prompt_stats",
]
