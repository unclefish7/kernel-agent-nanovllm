"""把四样材料组装成给 LLM agent 的一次输入。

四样材料：

1. 参考图（`Graph` 或 `graph.json`）：模板折叠后的节点、边、形状、公式；
2. 引擎规约（`contracts/nanovllm/spec.yaml`）：must / toolbox / context；
3. 算子库索引（`kernels/index.md`）：只供查看写法，不供调用；
4. 同构模板（`nano-vllm/nanovllm/models/qwen3.py`）：类划分与写法基准。

本模块在生成前检查图的验证证据，再按实测阶段和层顺序组装文本。
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from bisect import bisect_left
from collections import defaultdict
from importlib.util import find_spec
from itertools import groupby
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
        if "\n" in value:
            return value
        path = Path(value)
        if path.is_file():
            return path.read_text(encoding="utf-8")
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
        if graph.lstrip().startswith("{"):
            text = graph
        else:
            text = Path(graph).read_text(encoding="utf-8")
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
    if profile and "alternatives" in profile:
        pieces.append("条件形状=" + _json(profile))
    elif profile:
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


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class _Catalog:
    """内容完全相同才共用编号。编号按首次出现的顺序分配。"""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.entries: dict[str, str] = {}

    def intern(self, body: str) -> str:
        return self.entries.setdefault(body, f"{self.prefix}{len(self.entries) + 1}")

    def lines(self) -> list[str]:
        return [f"{name} {body}" for body, name in self.entries.items()]


_NODE_ADDRESS = re.compile(r"^(?P<phase>D\d*\.)?(?:L(?P<layer>\d+|\*)\.)?(?P<name>.+)$")


def _node_address(node_id: str, phase_prefix: str, layer: int | None) -> str:
    match = _NODE_ADDRESS.fullmatch(node_id)
    prefix, index, name = match.groups()
    phase = "" if (prefix or "") == phase_prefix else (prefix or "prefill.")
    if index is None:
        return phase + name
    if index == "*" or layer is None:
        return f"{phase}L{index}.{name}"
    delta = int(index) - layer
    return phase + ("L." if delta == 0 else f"L[{delta:+d}].") + name


def _layer_path(name: str, layer: int | None) -> str:
    return re.sub(rf"(?<=\.layers\.){layer}(?=\.|$)", "*", name) if layer is not None else name


def _tensor_ids(value):
    if isinstance(value, dict):
        if value.keys() == {"tensor"}:
            yield value["tensor"]
        else:
            for item in value.values():
                yield from _tensor_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _tensor_ids(item)


def _program_text(node: dict, layer: int | None, layouts: _Catalog, tensors: dict,
                  sources: dict | None) -> str:
    """只重命名配方内的张量变量。常量、参数、容器位置和写入顺序均参与比较。"""
    attrs = node["attrs"]
    prefix = _NODE_ADDRESS.fullmatch(node["id"])["phase"] or ""
    inputs = {ref["tensor"]: f"in{i}" for i, ref in enumerate(node.get("inputs", []))}
    weights = attrs.get("program_state") or {}
    free = attrs.get("program_inputs") or {}
    names, bindings = {}, []
    temp = 0

    def operand(value, *, result=False):
        nonlocal temp
        if isinstance(value, dict):
            if value.keys() == {"tensor"}:
                tid = value["tensor"]
                if tid not in names:
                    if result:
                        temp += 1
                        names[tid] = f"v{temp}"
                    elif (tid in inputs and free.get(tid, {}).get("kind") not in ("literal", "state")
                          and (sources is not None or free[tid]["kind"] == "input")):
                        names[tid] = inputs[tid]
                    else:
                        binding = free.get(tid, {})
                        if binding.get("kind") == "literal":
                            source = _json(binding["value"])
                        elif binding.get("kind") == "input":
                            source = inputs[binding["tensor"]]
                        elif binding.get("kind") == "state":
                            source = _layer_path(binding["name"], layer)
                        elif binding.get("kind") == "producer" and sources is None:
                            owner = _node_address(binding["node"], prefix, layer)
                            output = (f"step{binding['step']}[{binding['result_index']}]"
                                      if "step" in binding else f"out{binding['output_slot']}")
                            source = f"{owner}.{output}"
                        elif binding.get("kind") == "external" and sources is None:
                            source = "external=" + _json(binding)
                        elif tid in weights:
                            source = _layer_path(weights[tid], layer)
                        else:
                            source = tid
                            previous = sources.get(tid, []) if sources is not None else []
                            before = bisect_left(previous, (step["seq"],))
                            if before:
                                _, owner, output = previous[before - 1]
                                source = _node_address(owner, prefix, layer) + "." + output
                        names[tid] = f"x{len(bindings)}"
                        layout = layouts.intern(_json(_layout(tensors.get(tid, {}))))
                        bindings.append(f"{names[tid]}={source}/{layout}")
                return "$" + names[tid]
            return {k: operand(v, result=result) for k, v in value.items()}
        if isinstance(value, list):
            return [operand(v, result=result) for v in value]
        return value

    lines = []
    for step in attrs["program"]:
        args = _json(operand(step["args"]))
        kwargs = _json(operand(step["kwargs"])) if step["kwargs"] else ""
        target = _json(operand(step["result"], result=True))
        writes = [names.get(tid, tid) for tid in step["writes"]]
        mutation = " 写入 " + ",".join(writes) if writes else ""
        lines.append(f"  {target} = {step['op']} {args}{kwargs}{mutation}")
    outputs = [operand({"tensor": ref["tensor"]}) for ref in node.get("outputs", [])]
    lines.append("  out=" + _json(outputs))
    if bindings:
        lines.insert(0, "  来源/布局 " + " ".join(bindings))
    return "\n".join(lines)


def _layout(meta: dict) -> dict:
    return {k: meta[k] for k in ("dtype", "shape", "stride", "offset") if k in meta}


def _variant_text(node: dict, layer: int | None, layouts: _Catalog, programs: _Catalog) -> str:
    rows = []
    for variant in node["attrs"]["program_variants"]:
        tensors = variant["tensors"]
        branch = {"id": node["id"], "attrs": variant,
                  "inputs": variant["inputs"], "outputs": variant["outputs"]}
        recipe = programs.intern(_program_text(branch, layer, layouts, tensors, sources=None))
        ports = []
        for direction in ("inputs", "outputs"):
            refs = [layouts.intern(_json(_layout(tensors[r["tensor"]]))) for r in variant[direction]]
            ports.append(("in=" if direction == "inputs" else "out=") + _json(refs))
        rows.append(f"  样本={_json(variant['cases'])} 代表={_json(variant['representative_case'])} "
                    f"{recipe} {' '.join(ports)}")
    return "\n" + "\n".join(rows)


def _state_contracts(records: list[dict]) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        layer = record["layer"]
        prefix = _NODE_ADDRESS.fullmatch(record["producer"])["phase"] or ""

        def normalize_path(path):
            # 抽取器将路径中的首个整型索引记为层号。其余索引属于槽内部。
            return re.sub(r"\[(\d+)\]", lambda m: "[L]" if int(m[1]) == layer else m[0],
                          path, count=1)

        def reader(item):
            if isinstance(item, str):
                return _node_address(item, prefix, layer)
            return {**item, "node": _node_address(item["node"], prefix, layer)}

        body = {k: v for k, v in record.items() if k not in ("phase", "layer", "tensor")}
        body.update(path=normalize_path(record["path"]),
                    aliases=[normalize_path(p) for p in record["aliases"]],
                    producer=_node_address(record["producer"], prefix, layer),
                    readers=[reader(r) for r in record["readers"]])
        # 未改写的旧生产者也可能来自更早的阶段，必须保留其阶段。
        producer_phase = "prefill" if not prefix else (
            "decode" if prefix == "D." else "decode_" + prefix[1:-1])
        if producer_phase != record["phase"]:
            body["producer_phase"] = producer_phase
        groups[_json(body)].append(f"{record['phase']}/L{layer}={record['tensor']}")
    return [f"{' '.join(locations)} {body}" for body, locations in groups.items()]


def _reference_methods(nodes: list[dict]) -> list[str]:
    """从录制定位且文件哈希一致的本地实现摘取计算方法，不执行 HF 代码。"""
    classes = defaultdict(set)
    for node in nodes:
        source = node.get("impl", {}).get("source") or {}
        qualname = source.get("qualname", "")
        if qualname.startswith("transformers.models.") and source.get("sha256"):
            key = qualname, source["sha256"]
            classes[key].add(node["op"])
            for step in node["attrs"].get("program", []):
                classes[key].add(step["op"].removeprefix("aten.").split(".")[0])
    package = find_spec("transformers") if classes else None
    root = Path(package.origin).parent if package and package.origin else None
    files, shown, lines = {}, set(), []
    for (qualname, digest), ops in classes.items():
        if not ops.intersection({"rmsnorm", "attention", "rope_table", "rsqrt", "roll", "sum"}):
            continue
        parts = qualname.split(".")
        if not all(p.isidentifier() for p in parts):
            continue
        path = Path(*parts[1:-1]).with_suffix(".py")
        if (path, digest) not in files:
            data = (root / path).read_bytes() if root and (root / path).is_file() else b""
            files[path, digest] = ({n.name: n for n in ast.parse(data).body
                                    if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
                                   if hashlib.sha256(data).hexdigest() == digest else {})
        definitions = files[path, digest]
        cls = definitions.get(parts[-1])
        if not isinstance(cls, ast.ClassDef):
            lines.append(f"- {qualname} sha256={digest}，当前环境缺少匹配源码。")
            continue
        methods = {fn.name: fn for fn in cls.body if isinstance(fn, ast.FunctionDef)}
        if "forward" not in methods:
            continue
        pending = [(parts[-1] + ".forward", methods["forward"])]
        while pending:
            name, fn = pending.pop(0)
            key = (path, digest, fn.lineno)
            if key in shown:
                continue
            shown.add(key)
            # 保留分支条件和实际算式；略去装饰器、注解、文档字符串。
            body = [stmt for stmt in fn.body if not (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
            code = "\n".join(ast.unparse(stmt) for stmt in body)
            lines += [f"`{path}:{fn.lineno} {name}` (sha256 {digest[:12]})",
                      "```python", code, "```"]
            for call in (n.func for n in ast.walk(fn) if isinstance(n, ast.Call)):
                if isinstance(call, ast.Name) and isinstance(definitions.get(call.id), ast.FunctionDef):
                    pending.append((call.id, definitions[call.id]))
                if (isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name)
                        and call.value.id == "self" and call.attr in methods):
                    helper = methods[call.attr]
                    evidence = {n.attr for n in ast.walk(helper) if isinstance(n, ast.Attribute)}
                    if evidence & ops:
                        pending.append((parts[-1] + "." + call.attr, helper))
    return lines


def _render_program_graph(graph: dict) -> list[str]:
    """分别对张量布局、原语和整层接口去重，保留每次执行指向的契约。"""
    tensors = graph.get("tensors", {})
    layouts, programs, contracts, blocks = _Catalog("T"), _Catalog("P"), _Catalog("N"), _Catalog("C")
    variants = _Catalog("V")
    sources = defaultdict(list)
    for node in graph.get("nodes", []):
        for i, step in enumerate(node.get("attrs", {}).get("program", [])):
            for j, tid in enumerate(_tensor_ids(step["result"])):
                sources[tid].append((step["seq"], node["id"], f"step{i}[{j}]"))
        if node.get("origin") == "input":
            for ref in node.get("outputs", []):
                sources[ref["tensor"]].append((-1, node["id"], f"out{ref['index']}"))
    for history in sources.values():
        history.sort()
    incoming = defaultdict(list)
    for edge in graph.get("edges", []):
        incoming[edge["to_node"]].append(edge)

    # 每段保留录制顺序；全局节点可同时出现在第一层之前和最后一层之后。
    events = defaultdict(list)
    for node in graph.get("nodes", []):
        node_id = node["id"]
        match = _NODE_ADDRESS.fullmatch(node_id)
        prefix = match["phase"] or ""
        indices = tuple(node.get("instances") or ())
        if match["layer"] and match["layer"] != "*":
            indices = (int(match["layer"]),)
        layer = indices[0] if indices else None
        attrs = node.get("attrs", {})
        row = [_node_address(node_id, prefix, layer), node["op"]]
        impl = node.get("impl") or {}
        if impl:
            row.extend([str(impl.get("class") or ""), _layer_path(impl.get("module_path") or "", layer)])
        if node.get("semantics", {}).get("formula"):
            row.append(node["semantics"]["formula"])
        for direction in ("inputs", "outputs"):
            ports = []
            profiles = attrs.get("shape_profile", {}).get(direction, [])
            for index, ref in enumerate(node.get(direction, [])):
                meta = _layout(tensors.get(ref["tensor"], {}))
                if profiles:
                    meta["profile"] = profiles[index]
                fields = [layouts.intern(_json(meta)), ref.get("role"), ref.get("path"), ref.get("arg_name")]
                ports.append(fields)
            row.append(("in=" if direction == "inputs" else "out=") + _json(ports))
        for edge in incoming[node_id]:
            source = _node_address(edge["from_node"], prefix, layer)
            row.append(f"{edge.get('kind')}:{source}.out{edge.get('source_slot')}"
                       f"→in{edge.get('target_slot')}({edge.get('role')})")
        if node.get("state"):
            state = [{**s, "name": _layer_path(s["name"], layer)} for s in node["state"]]
            row.append("state=" + _json(state))
        if attrs.get("program"):
            row.append(programs.intern(_program_text(node, layer, layouts, tensors, sources)))
        if "program_variants" in attrs:
            row.append(variants.intern(_variant_text(node, layer, layouts, programs)))
        phase = node.get("phase", "prefill")
        contract = contracts.intern(" ".join(row))
        program = attrs.get("program") or []
        if not program:
            events[phase].append((-1, indices, contract, 0, 1))
        for i, step in enumerate(program):
            events[phase].append((step["seq"], indices, contract, i, len(program)))

    layer_types = (graph.get("model", {}).get("config") or {}).get("layer_types") or []
    lines = ["## 2.4 各阶段的真实执行顺序", "",
             "L 表示当前层，L[-1] 表示前一层。输入输出按槽号排列；端口依次给出布局、角色、路径和形参。",
             "C 按执行顺序列出节点 N，T 为布局，P 为有序原语配方。$ 标记张量引用。"
             "stepi[j] 表示第 i 条原语的第 j 个张量返回，均从 0 起算。N[a:b] 只执行该节点的原语切片。"]
    for phase, steps in events.items():
        order = []
        for indices, run in groupby(sorted(steps), key=lambda e: e[1]):
            rows = []
            for contract, segment in groupby(run, key=lambda e: e[2]):
                ops = list(segment)
                first, last, size = ops[0][3], ops[-1][3] + 1, ops[-1][4]
                rows.append(contract if (first, last) == (0, size) else f"{contract}[{first}:{last}]")
            block = blocks.intern(" ".join(rows))
            labels = [f"L{i}({layer_types[i]})" if i < len(layer_types) else f"L{i}" for i in indices]
            order.append(f"{','.join(labels) or '全局'}={block}")
        lines.append(f"- {phase} " + " → ".join(order))
    lines += ["", "## 2.5 节点与依赖", *blocks.lines(), *contracts.lines()]
    if variants.entries:
        lines += ["", "### 按实测样本选择原语配方",
                  "V 按（批大小 B，提示长度 S）列出观测样本、代表样本的配方与端口布局。"
                  "匹配样本时按该分支的 P 执行。节点切片和全局顺序来自基础录制。"
                  "样本分组的适用范围仅限表内观测。", *variants.lines()]
    lines += ["", "## 2.6 原语配方（从上往下执行）", *programs.lines(),
              "", "## 2.7 张量布局与维度样本", *layouts.lines(),
              "", "## 2.8 跨步状态（阶段/层=本步张量）",
              *_state_contracts(graph.get("env", {}).get("state_slots", []))]
    if graph.get("env", {}).get("allocated_only"):
        lines.append("仅观察到分配的槽 " + _json(graph["env"]["allocated_only"]))
    source = _reference_methods(graph.get("nodes", []))
    if source:
        lines += ["", "## 2.9 实测模块的本地源码与舍入位置",
                  "FP32 抽图时，同类型转换可能消失。用以下源码确认 BF16 运行时的升精度、回转、归约和乘法顺序。"
                  "源码列出了可能分支，使用对应阶段配方确认实际路径。Q/K 的归一化位置、旋转维度、配对及顺序均以目标图和源码为准。",
                  *source]
    return lines


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

1. 遵守第 3 节全部 must。引擎调用与装载接口同模板同构同式；模型内部按图的真实层序、每层权重及分阶段计算实现。
2. 文件顶层必须有一个类名以 `ForCausalLM` 结尾的模型类，并且类体内有 `packed_modules_mapping` 类属性；没有需要打包的权重也写 `{{}}`。
3. 文件里必须出现 `from nanovllm.utils.context import get_context`。
4. 只使用标准库、`torch`、`transformers` 和 `nanovllm` 已有模块；不要新增依赖，不要修改引擎。
5. 第 4 节的 `kernels/` 只看写法，不得 import 或调用；算子最终由你自己写。
6. 权重名按第 2.3 节和模板的 `packed_modules_mapping` 映射；需要按 tp 切分的参数使用模板里的并行层。
7. 按图的状态槽逐项实现初始化、读取、更新和写回。K/V 使用引擎分页缓存；其他状态按第 3 节已支持的接口按请求和层隔离。遇到能力缺口，遵守规约指定的运行限制，不能用模块私有缓存伪造支持。
8. 按有序原语配方解析输入依赖、常量、权重视图、分支结果及原地写入。高层算子名未知时直接实现经重放的配方。尺寸样本中的数值用第 2 节的轴关系、配置及当前请求长度解释。

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
        folded = defaultdict(list)
        for param in params:
            original = param["name"]
            name = _fold_layer_index(original)
            shape = tuple(param.get("shape") or [])
            index = re.search(r"\.layers\.(\d+)\.", original)
            folded[name, shape, param.get("dtype")].append(int(index[1]) if index else None)
        lines += ["", "## 2.3 权重名（按层号折叠去重）", ""]
        for (name, shape, dtype), indices in folded.items():
            scope = " 层=" + ",".join(map(str, indices)) if indices[0] is not None else ""
            lines.append(f"- `{name}` {_fmt_shape(shape)} {_fmt_dtype(dtype)}{scope}")

    if "state_slots" in env or any("program" in n.get("attrs", {}) for n in nodes):
        return "\n".join([*lines, "", *_render_program_graph(graph)])

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
    lines += ["", "## 3.4 引擎能力与缺口", ""]
    if not spec.get("capabilities"):
        lines.append("当前规约没有单列能力清单。仅使用上文有依据的接口。")
    for capability in spec.get("capabilities") or []:
        label = "支持" if capability["supported"] is True else "缺口"
        lines.append(f"- {label} `{capability['id']}` {capability['detail']}")
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

> 参照模板的 Context、权重装载及模型入口。按目标图组装层类型，并保留各阶段独有的计算和状态更新。

```python
{text.strip()}
```
"""


# ---------------------------------------------------------------------------
# 对外组装
# ---------------------------------------------------------------------------


def _check_agent_graph(graph: dict) -> None:
    """核对实际传给 agent 的图及其重放覆盖，兼容早期已核验的 graph.v1。"""
    from nanovllm_selector.extract.schema import Graph, SCHEMA_VERSION

    report = graph.get("reconcile") or {}
    checks = report.get("checks") or []
    for check in checks:
        if check.get("ok") is not True:
            raise ValueError(f"{check.get('name')}，期望 {check.get('expected')}，实测 {check.get('actual')} "
                             f"{check.get('note', '')}")
    if report.get("ok") is not True or not checks:
        raise ValueError("缺少通过的 reconcile 对账记录")
    for issues in (graph.get("conflicts"), report.get("invariants"), report.get("validate_problems")):
        if issues:
            raise ValueError(str(issues[0]))
    version = graph.get("schema_version")
    if version not in {"graph.v1", SCHEMA_VERSION}:
        raise ValueError(f"无法验证 schema_version={version}")
    nodes = graph.get("nodes") or []
    phases = graph.get("phases") or []
    if not nodes or not phases or not graph.get("graph_inputs"):
        raise ValueError("缺少节点、图输入或阶段记录")
    if not graph.get("model", {}).get("config") or not graph.get("params"):
        raise ValueError("缺少模型配置或权重名表")
    check_ids = {check["name"].split()[0] for check in checks}
    missing = {"I1", "I2", "I3", "R1"} - check_ids
    if missing:
        raise ValueError("缺少抽取或重放检查 " + ", ".join(sorted(missing)))
    replay = report.get("replay") or {}
    if replay.get("ok") is not True or replay.get("skipped") or replay.get("failures"):
        raise ValueError("节点重放不完整 " + str(replay.get("failures") or replay.get("skipped") or ""))
    compute = [n for n in nodes if n.get("origin") != "input"]
    total = sum(len(n.get("instances") or []) or 1 for n in compute)
    if not total or replay.get("checked") != total:
        raise ValueError(f"重放覆盖 {replay.get('checked')} 个节点，图中有 {total} 个计算实例")
    env = graph.get("env") or {}
    if not env.get("shape_probes"):
        for node in nodes:
            if "program_variants" in node.get("attrs", {}):
                raise ValueError(f"{node['id']} 的 program_variants 缺少逐样本校验所需的 shape_probes")
    problems = Graph.from_dict(graph).validate()
    if problems:
        raise ValueError("；".join(problems[:5]))

    modern = (version != "graph.v1" or "state_slots" in env
              or any("program" in n.get("attrs", {}) for n in compute))
    if not modern:
        if any(n.get("op") in (None, "unknown") for n in compute):
            raise ValueError("旧图含高层名称未知且缺少重放配方的节点")
        return
    if phases[:3] != ["prefill", "decode", "decode_2"]:
        raise ValueError("缺少 prefill、decode 和连续第二步 decode_2 的执行证据")
    if not env.get("shape_probes") or "S1" not in check_ids:
        raise ValueError("缺少独立样本检验的 shape_probes")
    if not all(graph.get("graph_outputs", {}).get(p) for p in phases):
        raise ValueError("缺少各阶段的模型返回槽位")
    if not isinstance(env.get("state_slots"), list):
        raise ValueError("缺少逐阶段 state_slots")
    state_phases = {s["phase"] for s in env["state_slots"]}
    if (state_phases or any(e.get("kind") == "state" for e in graph.get("edges", []))) and (
            state_phases != set(phases)):
        raise ValueError("state_slots 未覆盖图中的跨步状态阶段")
    previous, witnessed = {}, set()
    edges = {(e["from_node"], e["source_slot"], e["to_node"], e["role"])
             for e in graph["edges"] if e.get("kind") == "state"}
    for slot in env["state_slots"]:
        path = slot["path"]
        update = slot.get("update") or {}
        if update.get("kind") not in {"initial", "shift", "resized", "updated", "unchanged"}:
            raise ValueError(f"{slot['phase']} {path} 缺少可解释的 update")
        if update["kind"] == "shift" and not {"axis", "retained_columns"} <= update.keys():
            raise ValueError(f"{slot['phase']} {path} 缺少移窗轴或保留列数")
        if path in previous:
            old = previous[path]
            witnessed.update((old["producer"], old["source_slot"], reader, old["role"])
                             for reader in slot["readers"])
        previous[path] = slot
    if edges != witnessed:
        raise ValueError(f"state_slots.readers 与跨步边不符 {next(iter(edges ^ witnessed))}")
    external = {r["tensor"] for n in nodes if n.get("origin") == "input" for r in n["outputs"]}
    produced = set()
    folded = {}
    for node in compute:
        owner = node["id"]
        indices = node.get("instances") or []
        if indices:
            for index in indices[1:]:
                folded[owner.replace("L*.", f"L{index}.", 1)] = owner
            owner = owner.replace("L*.", f"L{indices[0]}.", 1)
        for step in node.get("attrs", {}).get("program") or []:
            produced.update((owner, step["seq"], tid) for tid in _tensor_ids(step["result"]))
    for node in compute:
        attrs = node.get("attrs") or {}
        steps = attrs.get("program")
        evidence = attrs.get("program_replay") or {}
        if (not isinstance(steps, list) or not steps or evidence.get("ok") is not True
                or evidence.get("checked") != len(steps) or evidence.get("total") != len(steps)):
            raise ValueError(f"{node['id']} 缺少覆盖全部原语的 program_replay")
        if "observed_inputs" not in attrs or "observed_outputs" not in attrs:
            raise ValueError(f"{node['id']} 缺少实测输入输出槽")
        seqs = [s["seq"] for s in steps]
        if any(a >= b for a, b in zip(seqs, seqs[1:])):
            raise ValueError(f"{node['id']} 的原语没有按实测顺序排列")
        bound = attrs.get("program_inputs")
        if not isinstance(bound, dict):
            raise ValueError(f"{node['id']} 缺少 program_inputs 来源表")
        defined, free = set(), {}
        for step in steps:
            if not {"op", "args", "kwargs", "result", "writes"} <= step.keys():
                raise ValueError(f"{node['id']} 第 {step['seq']} 条原语缺少参数、结果或写入记录")
            operands = set(_tensor_ids([step["args"], step["kwargs"]]))
            for tid in operands - defined:
                free.setdefault(tid, step["seq"])
            defined.update(_tensor_ids(step["result"]))
            if not set(step["writes"]) <= operands:
                raise ValueError(f"{node['id']} 第 {step['seq']} 条原语写入目标未绑定")
        if bound.keys() != free.keys():
            raise ValueError(f"{node['id']} program_inputs 与原语自由张量不符 {sorted(bound.keys() ^ free.keys())}")
        input_ids = {r["tensor"] for r in node["inputs"]}
        for tid, binding in bound.items():
            kind = binding["kind"]
            if kind == "producer":
                owner, seq = binding["node"], binding["seq"]
                matches = (owner, seq, tid) in produced
                if not matches and owner in folded:
                    # 折叠后仅保留代表层的原语。其他实例必须仍能追溯到已验证的输入边。
                    storage = graph["tensors"][tid]["storage"]
                    matches = any(ref and ref[0] == folded[owner]
                                  and graph["tensors"][port["tensor"]]["storage"] == storage
                                  for port, ref in zip(node["inputs"], attrs["observed_inputs"]))
                if binding["tensor"] != tid or seq >= free[tid] or not matches:
                    raise ValueError(f"{node['id']} {tid} 的生产者 {owner} seq={seq} 无对应实测结果；"
                                     "保留逐层实例可使用 --no-collapse")
            valid = (kind == "literal" and "value" in binding
                     or kind == "state" and binding.get("name")
                     or kind == "input" and binding.get("tensor") in input_ids
                     or kind == "producer" and {"node", "seq", "tensor"} <= binding.keys()
                     or kind == "external" and (tid in external or tid in input_ids))
            if not valid:
                raise ValueError(f"{node['id']} {tid} 缺少可用的参数、输入、常量或生产者来源")


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
    try:
        _check_agent_graph(graph_data)
        graph_text = _render_graph(graph_data)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(f"参考图与实测记录不符：{exc}。请修复该项后重新抽取："
                         "python -m nanovllm_selector.extract.cli --model <本地模型目录> "
                         "--phases prefill,decode --decode-steps 2 --shape-probes --check "
                         "--out <graph.json>") from exc
    spec_data = load_spec(spec_path)
    kernels_text = _read_text(kernels_index, "算子库索引")
    template_text = _read_text(template_path, "模型模板")
    engine_context = _render_spec(spec_data, include_toolbox=include_toolbox)
    template = _render_template(template_text)
    if exact:
        engine_context += "\n\n" + Path(__file__).with_name("exact_reference.md").read_text(encoding="utf-8")
        template += ("\n\n本次使用逐项相等验收。第 3 节的单卡实现依据给出了经过验证的回退代码。"
                     "仅在目标图出现相同运算、类型与顺序时采用相应实现。")
    return {
        "task": _render_task(graph_data),
        "graph": graph_text,
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
