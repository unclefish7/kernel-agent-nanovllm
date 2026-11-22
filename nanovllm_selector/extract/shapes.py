"""用独立改变批大小、提示长度的执行样本解释张量维度。"""

from __future__ import annotations

import re


def _variables(batch, length, phase):
    if phase == "prefill":
        return {"B": batch, "S": length, "B*S": batch * length}
    step = 1 if phase == "decode" else int(phase.removeprefix("decode_"))
    past = length + step - 1
    return {"B": batch, "P": past, "P+1": past + 1, "B*P": batch * past,
            "B*(P+1)": batch * (past + 1)}


def _axis_value(label, variables):
    if label.isdecimal():
        return int(label)
    if label in variables:
        return variables[label]
    offset = re.fullmatch(r"(B|S|P)([+-]\d+)", label)
    if offset and offset[1] in variables:
        return variables[offset[1]] + int(offset[2])
    return None


def _signature(node):
    return (node.op, node.phase, node.instances, node.impl.get("module_path"),
            [(ref.role, ref.index, ref.arg_name, ref.path) for ref in node.inputs],
            node.attrs.get("observed_inputs"), [ref.role for ref in node.outputs])


def _config_axes(config, node, ref, direction, rank):
    # 用缓存身份、来源模块与注意力计算所用的布局约束命名，避免仅凭数值猜轴。
    source = ""
    if direction == "inputs":
        observed = node.attrs.get("observed_inputs", [])
        witness = observed[ref.index] if isinstance(ref.index, int) and 0 <= ref.index < len(observed) else None
        role = witness[2] if witness else None
        source = witness[0] if witness else ""
    else:
        role = ref.role
    if rank != 4:
        return {}
    if role in ("key", "value"):
        heads = "num_key_value_heads"
    elif node.op == "attention" and source.endswith((".q_proj", ".q_norm")):
        heads = "num_attention_heads"
    elif node.op == "attention" and source.endswith((".k_proj", ".k_norm", ".v_proj")):
        heads = "num_key_value_heads"
    else:
        return {}
    head_dim = config.get("head_dim")
    field = "head_dim"
    if not head_dim and config.get("num_attention_heads"):
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        field = "hidden_size/num_attention_heads"
    return {1: (heads, config.get(heads)), 3: (field, head_dim)}


def _layouts(cases, shapes, phase):
    """按实际批大小保留布局；尺寸规律用同批大小的保留样本检验。"""
    by_batch = {}
    for case, shape in zip(cases, shapes):
        group = by_batch.setdefault(case[0], {"cases": [], "shapes": []})
        group["cases"].append(list(case))
        group["shapes"].append(shape)
    for group in by_batch.values():
        found = group["shapes"]
        if len({len(shape) for shape in found}) != 1:
            raise ValueError("同批大小内张量秩也变化")
        variables = [_variables(*case, phase) for case in group["cases"]]
        axes = []
        for sizes in zip(*found):
            if len(set(sizes)) == 1:
                axes.append(str(sizes[0]))
            else:
                matching = [name for name in variables[0]
                            if all(v[name] == size for v, size in zip(variables, sizes))]
                if not matching:
                    raise ValueError("保留样本不能验证条件维度")
                axes.append(matching[0])
        group["axes"] = axes
    return list(by_batch.values())


def validate_profiles(graph):
    probes = graph.env.get("shape_probes")
    if not probes:
        return []
    cases = [*probes["fit"], probes["holdout"], probes["base"]]
    problems = []
    for node in graph.nodes:
        values = [_variables(*case, node.phase) for case in cases]
        for direction in ("inputs", "outputs"):
            refs = getattr(node, direction)
            profiles = node.attrs.get("shape_profile", {}).get(direction, [])
            if len(profiles) != len(refs):
                problems.append(f"{node.id} {direction} 尺寸样本数量与槽位不符")
            for ref, profile in zip(refs, profiles):
                if ref.tensor not in graph.tensors:
                    continue
                shapes = [*profile["samples"], graph.tensors[ref.tensor].shape]
                if "alternatives" in profile:
                    try:
                        expected = _layouts(cases[:-1], shapes[:-1], node.phase)
                        if expected != profile["alternatives"]:
                            raise ValueError("形状记录与样本不符")
                        layout = next(g for g in expected if g["cases"][0][0] == cases[-1][0])
                        sizes = [int(a) if a.isdecimal() else values[-1][a] for a in layout["axes"]]
                        if sizes != shapes[-1]:
                            raise ValueError("基础形状与条件维度不符")
                    except (ValueError, KeyError, StopIteration) as exc:
                        problems.append(f"{node.id} {direction}[{ref.index}] 条件形状无法验证：{exc}")
                    continue
                axes = profile["axes"]
                if len(cases) != len(shapes) or any(len(shape) != len(axes) for shape in shapes):
                    problems.append(f"{node.id} {direction}[{ref.index}] 尺寸记录不完整")
                    continue
                fixed = _config_axes(graph.model.get("config") or {}, node, ref, direction, len(axes))
                for i, axis in enumerate(axes):
                    if axis == "?":
                        continue
                    name, size = fixed.get(i, (None, None))
                    for shape, variables in zip(shapes, values):
                        expected = size if axis == name else _axis_value(axis, variables)
                        if shape[i] != expected:
                            problems.append(f"{node.id} {direction}[{ref.index}] 轴 {i} 的 {axis} 与实测不符")
                            break
    return problems


def profile_shapes(graph, batch, length, trace):
    cases = [(1, 5), (1, 17), (2, 5), (2, 11)]
    samples = [graph if case == (batch, length) else trace(*case) for case in cases]
    indexes = [{node.id: node for node in sample.nodes} for sample in samples]
    problems = []
    base_nodes = {node.id for node in graph.nodes}
    for case, sample, index in zip(cases, samples, indexes):
        if not sample.reconcile["ok"]:
            problems.append(f"{case} 样本的图检查失败")
        if set(index) != base_nodes:
            problems.append(f"{case} 路径不同，节点差异 {sorted(base_nodes ^ set(index))[:8]}")

    config = graph.model.get("config") or {}
    unknown = 0
    for node in graph.nodes:
        aligned = [index.get(node.id) for index in indexes]
        if not all(other and _signature(other) == _signature(node) for other in aligned):
            problems.append(f"{node.id} 不同样本的输入输出无法对齐")
            continue
        values = [_variables(b, s, node.phase) for b, s in cases]
        base_values = _variables(batch, length, node.phase)
        profiles = {}
        for direction in ("inputs", "outputs"):
            profiles[direction] = []
            for i, ref in enumerate(getattr(node, direction)):
                shapes = [sample.tensors[getattr(other, direction)[i].tensor].shape
                          for sample, other in zip(samples, aligned)]
                base = graph.tensors[ref.tensor].shape
                if any(len(shape) != len(base) for shape in shapes):
                    try:
                        variants = _layouts(cases, shapes, node.phase)
                        profiles[direction].append({"axes": ["?"] * len(base), "samples": shapes,
                                                    "alternatives": variants})
                    except ValueError as exc:
                        problems.append(f"{node.id} {direction}[{i}] 条件形状不明：{exc}")
                    continue
                axes = []
                config_axes = _config_axes(config, node, ref, direction, len(base))
                for axis, sizes in enumerate(zip(*shapes)):
                    fitted = sizes[:-1]
                    matches = [name for name in values[0]
                               if all(v[name] == size for v, size in zip(values[:-1], fitted))]
                    if len(set(fitted)) == 1:
                        label = str(fitted[0])
                        name, size = config_axes.get(axis, (None, None))
                        if name and size == fitted[0]:
                            label = name
                        prediction = base_prediction = fitted[0]
                    elif len(matches) == 1:
                        label = matches[0]
                        prediction, base_prediction = values[-1][label], base_values[label]
                    else:
                        offsets = []
                        for name in ("B", "S", "P"):
                            if name in values[0]:
                                shifts = {size - v[name] for size, v in zip(fitted, values[:-1])}
                                if len(shifts) == 1:
                                    offsets.append(f"{name}{shifts.pop():+d}")
                        if len(offsets) == 1:
                            label = offsets[0]
                            prediction = _axis_value(label, values[-1])
                            base_prediction = _axis_value(label, base_values)
                        else:
                            label = "?"
                            unknown += 1
                            prediction, base_prediction = sizes[-1], base[axis]
                    if prediction != sizes[-1] or base_prediction != base[axis]:
                        problems.append(f"{node.id} {direction}[{i}] 轴 {axis} 的 {label} 与实测不符")
                    axes.append(label)
                profiles[direction].append({"axes": axes, "samples": shapes})
        node.attrs["shape_profile"] = profiles
    graph.env["shape_probes"] = {"fit": cases[:-1], "holdout": cases[-1], "base": (batch, length),
                                 "unknown_axes": unknown, "problems": problems}
    graph.reconcile["checks"].append({"name": "S1 独立样本验证维度关系", "expected": 0,
                                      "actual": len(problems), "ok": not problems,
                                      "note": f"保留 {unknown} 个未确定的轴"})
    graph.reconcile["validate_problems"] = graph.validate()
    graph.reconcile["ok"] = graph.reconcile["ok"] and not problems and not graph.reconcile["validate_problems"]
