"""用独立改变批大小、提示长度的执行样本解释张量维度。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from copy import deepcopy


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


def _topology(graph):
    """按实际节点和槽位比较连线；录制内的 tensor id 不能跨样本比较。"""
    incoming = defaultdict(set)
    for edge in graph.edges:
        incoming[edge.to_node].add((edge.target_slot, edge.from_node, edge.source_slot, edge.kind, edge.role))
    returns = defaultdict(list)
    for phase, outputs in graph.graph_outputs.items():
        for index, ref in enumerate(outputs):
            returns[ref["node"]].append((phase, index, ref["slot"]))
    inputs = set(graph.graph_inputs)
    return {n.id: (_signature(n), incoming[n.id], returns[n.id], n.id in inputs) for n in graph.nodes}


def _topology_problem(change):
    return (f"{tuple(change['case'])} 拓扑与基础样本不同，新增 {len(change['added'])}、"
            f"缺少 {len(change['missing'])}、节点契约改变 {len(change['changed'])} 个")


def _unobserved_routed_members(graph):
    """用分发模块的参数清单核对本次配方实际读取的编号子模块。"""
    scopes = set()
    reads = defaultdict(set)
    for node in graph.nodes:
        reads[node.phase].update(node.attrs.get("program_state", {}).values())
        # 旧图可能没有 indexed_dispatch 这个高层名称，直接识别记录的分发原语。
        ops = {s["op"] for s in node.attrs.get("program", ())}
        if {"aten.scatter_.value", "aten.nonzero.default", "aten.index.Tensor"} <= ops:
            scopes.add((node.phase, node.impl.get("module_path") or ""))

    problems = []
    for phase, owner in sorted(scopes):
        prefix = owner + "." if owner else ""
        declared, observed = defaultdict(set), defaultdict(set)
        for param in graph.params:
            name = param["name"]
            if not name.startswith(prefix):
                continue
            member = re.match(r"((?:[^.]+\.)*?)(\d+)\.", name[len(prefix):])
            if member:
                bank = (prefix + member[1]).rstrip(".")
                declared[bank].add(member[2])
                if name in reads[phase]:
                    observed[bank].add(member[2])
        for bank, used in observed.items():
            if used != declared[bank]:
                problems.append(f"{phase} {bank} 的分发轨迹只读取 {len(used)}/{len(declared[bank])} "
                                "个编号子模块，当前拓扑无法验证其余分支")
    return problems


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


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _map_tensors(value, visit):
    if isinstance(value, dict):
        if value.keys() == {"tensor"}:
            return {"tensor": visit(value["tensor"])}
        return {key: _map_tensors(item, visit) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, visit) for item in value]
    return value


def _tensor_ids(value):
    ids = []
    _map_tensors(value, ids.append)
    return ids


def _producer_index(graph):
    """将本次录制的全局 seq 换成来源节点内部的返回位置。"""
    sources = {}
    for node in graph.nodes:
        owner = node.id
        if node.instances:
            owner = owner.replace("L*.", f"L{node.instances[0]}.", 1)
        ports = {r.tensor: r.index for r in node.outputs}
        for step, instruction in enumerate(node.attrs.get("program", [])):
            for index, tid in enumerate(_tensor_ids(instruction["result"])):
                location = {"step": step, "result_index": index}
                if tid in ports:
                    location["output_slot"] = ports[tid]
                sources[owner, instruction["seq"], tid] = location
    return sources


def _sample_recipe(node, graph, sources):
    attrs = node.attrs
    program = attrs["program"]
    bindings = deepcopy(attrs["program_inputs"])
    for tid, binding in bindings.items():
        if binding["kind"] == "producer":
            owner = binding["node"]
            location = sources.get((owner, binding["seq"], tid))
            if location is None:
                # 非代表层的指令已折叠，只能引用有实际边和同一存储支持的输出槽。
                template = re.sub(r"^((?:D\d*\.)?L)\d+(?=\.)", r"\1*", owner)
                slots = {edge[1] for ref, edge in zip(node.inputs, attrs["observed_inputs"])
                         if edge and edge[0] == template
                         and graph.tensors[ref.tensor].storage == graph.tensors[tid].storage}
                if len(slots) == 1:
                    location = {"output_slot": slots.pop()}
            binding.update(location or {})
    tids = set(bindings) | set(attrs["program_state"])
    tids.update(ref.tensor for ref in [*node.inputs, *node.outputs])
    for step in program:
        tids.update(_tensor_ids([step["args"], step["kwargs"], step["result"]]))
        tids.update(step["writes"])
    return dict(op=node.op, program=program, program_inputs=bindings, program_state=attrs["program_state"],
                inputs=[r.to_dict() for r in node.inputs], outputs=[r.to_dict() for r in node.outputs],
                observed_inputs=attrs["observed_inputs"], tensors={t: graph.tensors[t].to_dict() for t in tids},
                program_replay=attrs.get("program_replay"))


# 只有尺寸类实参可按已观测到的 B/S 规律归并。维号、步幅参数 step、缩放系数、
# 写入方向、分支布尔值等仍按字面比较。完整实参始终保存在代表样本中。
_SIZE_ARGS = {
    "aten.view.default": (1,), "aten._unsafe_view.default": (1,), "aten.reshape.default": (1,),
    "aten.expand.default": (1,), "aten.as_strided.default": (1, 2),
    "aten.zeros.default": (0,), "aten.ones.default": (0,), "aten.full.default": (0,),
    "aten.arange.default": (0,), "aten.arange.start": (0, 1), "aten.arange.start_step": (0, 1),
    "aten.slice.Tensor": (2, 3), "aten.narrow.default": (2, 3), "aten.constant_pad_nd.default": (1,),
}


def _recipe_structure(recipe):
    names, sizes = {}, []

    def local(tid):
        return names.setdefault(tid, len(names))

    def size(value):
        if isinstance(value, list):
            return [size(item) for item in value]
        if type(value) is int:
            sizes.append(value)
            return {"size": len(sizes) - 1}
        return value

    steps = []
    for step in recipe["program"]:
        op = step["op"]
        args = _map_tensors(step["args"], local)
        for i in _SIZE_ARGS.get(op, ()):
            if i < len(args):
                args[i] = size(args[i])
        steps.append([op, args, _map_tensors(step["kwargs"], local), _map_tensors(step["result"], local),
                      [local(tid) for tid in step["writes"]]])
    ports = [[local(r["tensor"]) for r in recipe[key]] for key in ("inputs", "outputs")]
    origins = {}
    for tid, binding in recipe["program_inputs"].items():
        if binding["kind"] == "producer":
            place = (["port", binding["output_slot"]] if "output_slot" in binding else
                     ["step", binding.get("step"), binding.get("result_index")])
            source = ["producer", binding["node"], place]
        else:
            source = dict(binding)
            if "tensor" in source:
                source["tensor"] = local(source["tensor"])
        origins[local(tid)] = source
    return _digest([recipe["op"], steps, ports, origins]), sizes


def _branch_keys(recipes, phase):
    """只归并张量连线、指令和非尺寸实参相同，且尺寸变化与采样变量相符的配方。"""
    groups = defaultdict(list)
    keys = {}
    for case, recipe in recipes.items():
        structure, sizes = _recipe_structure(recipe)
        groups[structure].append((case, sizes))
    for structure, items in groups.items():
        variables = [_variables(*case, phase) for case, _ in items]
        keep = []
        for index, lengths in enumerate(zip(*(sizes for _, sizes in items))):
            # 常量不随样本变化；只有同一变量加偏移（含反向 padding）才能解释动态列。
            dynamic = len(set(lengths)) > 1 and any(
                len({n - sign * v[key] for n, v in zip(lengths, variables)}) == 1
                for key in variables[0] for sign in (1, -1))
            if not dynamic:
                keep.append(index)
        for case, sizes in items:
            keys[case] = _digest([structure, [sizes[i] for i in keep]])
    return keys


def _profile_program(node, samples, observations):
    recipes = {case: _sample_recipe(other, graph, sources)
               for case, other, graph, sources in samples}
    keys = _branch_keys(recipes, node.phase)
    groups = defaultdict(list)
    for case, key in keys.items():
        groups[key].append(case)
    node.attrs.pop("program_variants", None)
    if len(groups) == 1:
        return
    witnesses, variants = [], []
    for cases in groups.values():
        for case in cases:
            recipe = recipes[case]
            n = len(recipe["program"])
            evidence = recipe["program_replay"] or {}
            if not n or evidence.get("ok") is not True or evidence.get("checked") != n or evidence.get("total") != n:
                raise ValueError(f"{case} 缺少全部 {n} 条指令的成功重放")
            witnesses.append(dict(case=list(case), branch=keys[case], sha256=_digest(recipe), checked=n))
        representative = cases[0]
        variants.append(dict(cases=[list(case) for case in cases], representative_case=list(representative),
                             **deepcopy(recipes[representative])))
    node.attrs["program_variants"] = variants
    observations[node.id] = witnesses


def _check_recipe(recipe):
    defined, free = set(), set()
    last = -1
    for step in recipe["program"]:
        operands = set(_tensor_ids([step["args"], step["kwargs"]]))
        if step["seq"] <= last or not set(step["writes"]) <= operands:
            raise ValueError("指令顺序或写入目标无效")
        free.update(operands - defined)
        defined.update(_tensor_ids(step["result"]))
        last = step["seq"]
    bindings = recipe["program_inputs"]
    inputs = {r["tensor"] for r in recipe["inputs"]}
    outputs = {r["tensor"] for r in recipe["outputs"]}
    if set(bindings) != free or not outputs <= defined:
        raise ValueError("自由张量或输出与指令不符")
    tids = inputs | defined | free | set(recipe["program_state"])
    if tids != recipe["tensors"].keys() or any(recipe["tensors"][tid]["id"] != tid for tid in tids):
        raise ValueError("缺少本分支的张量元数据")
    for tid, binding in bindings.items():
        kind = binding["kind"]
        if kind == "producer":
            located = ("step" in binding and "result_index" in binding) or "output_slot" in binding
            valid = located and binding.get("node") and binding["tensor"] == tid
        elif kind == "input":
            valid = binding["tensor"] in inputs
        elif kind == "state":
            valid = binding.get("name") and binding["name"] == recipe["program_state"].get(tid)
        elif kind == "external":
            valid = binding["tensor"] in recipe["tensors"]
        else:
            valid = kind == "literal" and "value" in binding
        if not valid:
            raise ValueError(f"{tid} 的 {kind} 来源不可定位")


def _check_variants(node, variants, witnesses, cases, base):
    by_case = {tuple(w["case"]): w for w in witnesses}
    if set(by_case) != cases or len(by_case) != len(witnesses):
        raise ValueError("实测重放记录未覆盖每个采样输入")
    expected = defaultdict(set)
    for case, witness in by_case.items():
        expected[witness["branch"]].add(case)
    if len(expected) < 2 or len(variants) != len(expected):
        raise ValueError("分支数量与独立样本记录不符")
    for variant in variants:
        rep = tuple(variant["representative_case"])
        key = by_case[rep]["branch"]
        members = {tuple(case) for case in variant["cases"]}
        if rep not in members or len(members) != len(variant["cases"]) or members != expected.pop(key):
            raise ValueError("分支的样本归属与实测不符")
        recipe = {k: v for k, v in variant.items() if k not in ("representative_case", "cases")}
        _check_recipe(recipe)
        count = len(recipe["program"])
        replay = recipe["program_replay"]
        if (not count or replay["ok"] is not True or replay["checked"] != count or replay["total"] != count
                or any(by_case[c]["checked"] != count for c in members)):
            raise ValueError("分支的原语重放覆盖不完整")
        if _digest(recipe) != by_case[rep]["sha256"]:
            raise ValueError("配方、来源或张量元数据与代表样本的重放记录不符")
        if base in members and (base != rep or recipe["program"] != node.attrs["program"]):
            raise ValueError("基础样本与节点原配方不一致")


def validate_profiles(graph):
    probes = graph.env.get("shape_probes")
    if not probes:
        return []
    if probes.get("topology_differences"):
        return [_topology_problem(change) for change in probes["topology_differences"]]
    cases = [*probes["fit"], probes["holdout"], probes["base"]]
    problems = _unobserved_routed_members(graph)
    observations = probes.get("program_observations", {})
    missing = set(observations) - {n.id for n in graph.nodes}
    problems.extend(f"{name} 的程序分支缺少节点" for name in sorted(missing))
    for node in graph.nodes:
        if node.id in observations or "program_variants" in node.attrs:
            try:
                _check_variants(node, node.attrs["program_variants"], observations[node.id],
                                set(map(tuple, cases)), tuple(probes["base"]))
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                problems.append(f"{node.id} 程序分支无法验证：{exc}")
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
    # 基础样本可能位于 fit/holdout 之外。代表配方优先取基础样本，重复 case 只算一次。
    profiled = {(batch, length): (graph, {n.id: n for n in graph.nodes})}
    profiled.update(zip(cases, zip(samples, indexes)))
    observations = {}
    replayed = "replay" in graph.reconcile
    problems = []
    differences = []
    base_topology = _topology(graph)
    for case, (sample, _) in profiled.items():
        if not sample.reconcile["ok"]:
            problems.append(f"{case} 样本的图检查失败")
        if sample is graph:
            continue
        other = _topology(sample)
        if other != base_topology:
            change = dict(case=list(case), added=sorted(other.keys() - base_topology.keys()),
                          missing=sorted(base_topology.keys() - other.keys()),
                          changed=sorted(n for n in base_topology.keys() & other.keys()
                                         if base_topology[n] != other[n]))
            differences.append(change)
            problems.append(_topology_problem(change))
    if not differences:
        problems.extend(_unobserved_routed_members(graph))

    # 拓扑改变时保留基础轨迹和每次差异。共同节点也可能属于不同的动态调用，不能硬配。
    if differences:
        for node in graph.nodes:
            node.attrs.pop("shape_profile", None)
            node.attrs.pop("program_variants", None)
    program_samples = [(case, sample, index, _producer_index(sample))
                       for case, (sample, index) in profiled.items()] if replayed and not differences else []
    config = graph.model.get("config") or {}
    unknown = 0
    for node in ([] if differences else graph.nodes):
        aligned = [index[node.id] for index in indexes]
        if replayed and node.origin != "input":
            try:
                _profile_program(node, [(case, index[node.id], sample, sources)
                                        for case, sample, index, sources in program_samples], observations)
            except (ValueError, KeyError) as exc:
                problems.append(f"{node.id} 程序分支缺少实测证据：{exc}")
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
                                 "unknown_axes": unknown, "problems": problems,
                                 "topology_scope": "observed_inputs_only", "topology_differences": differences}
    if replayed:
        graph.env["shape_probes"]["program_observations"] = observations
    note = f"仅适用于实测输入，保留 {unknown} 个未确定的轴"
    if differences:
        note = "采样拓扑不同，未合并尺寸与程序配方"
    elif problems:
        note = problems[0]
    graph.reconcile["checks"].append({"name": "S1 独立样本验证维度关系", "expected": 0,
                                      "actual": len(problems), "ok": not problems, "note": note})
    graph.reconcile["validate_problems"] = graph.validate()
    graph.reconcile["ok"] = graph.reconcile["ok"] and not problems and not graph.reconcile["validate_problems"]
