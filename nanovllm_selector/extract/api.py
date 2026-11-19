"""对外 API：`extract_graph(model_path) -> Graph`。

这一层只做编排，所有判断都在下面各模块里，方便单独测试。
"""

from __future__ import annotations

import platform
import sys

import torch
import transformers

from .decompose import GraphBuilder
from .edges import build_edges, tensor_view, to_tensor_meta
from .loader import load_model
from .trace_io import cache_tensors, observe_ports
from .recorder import record_forward
from .checks import check_invariants
from .replay import replay_nodes
from .reconcile import reconcile
from .schema import Graph
from .template import fold, template_id_of

__all__ = ["extract_graph"]


def extract_graph(
    model_path: str,
    dtype: str = "float32",
    device: str = "cpu",
    seq_len: int = 8,
    phases=("prefill",),
    collapse_template: bool = True,
    weights: bool = False,
    trust_remote_code: bool = False,
    batch_size: int = 1,
    reconcile_graph: bool = True,
    replay_graph: bool = True,
    decode_steps: int = 1,
) -> Graph:
    """从一个 HuggingFace 模型目录重建计算图。

    参数
    ----
    model_path : 本地模型目录（只需 config.json；weights=False 时不读权重）
    dtype : float32 / float16 / bfloat16
    device : cpu / cuda
    seq_len : 造多长的输入序列
    phases : prefill 或 prefill、decode。decode_steps 指定连续读取缓存的次数。
    collapse_template : 是否把结构相同的层折成模板 + instances
    """
    phases = tuple(phases)
    if phases not in (("prefill",), ("prefill", "decode")):
        raise NotImplementedError(f"不支持的 phases={phases}；只接受 prefill 或 prefill+decode")

    model, config = load_model(
        model_path,
        weights=weights,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    options = dict(dtype=dtype, device=device, phases=phases, collapse_template=collapse_template,
                   reconcile_graph=reconcile_graph, replay_graph=replay_graph, decode_steps=decode_steps)
    graph = _trace(model, config, batch_size=batch_size, seq_len=seq_len, **options)
    return graph


def _trace(model, config, *, dtype, device, batch_size, seq_len, phases, collapse_template,
           reconcile_graph, replay_graph, decode_steps):
    if min(batch_size, seq_len, decode_steps) < 1:
        raise ValueError("batch_size、seq_len 和 decode_steps 都必须为正数")
    generator = torch.Generator().manual_seed(0)
    vocab = int(config.vocab_size)
    counts = decode_steps + 1 if "decode" in phases else 1
    tokens = torch.randint(vocab, (batch_size, seq_len + counts - 1), generator=generator).to(device)
    instances, all_nodes, edges = [], [], []
    tensors, history, graph_outputs = {}, {}, {}
    phase_checks, phase_stats, problems, failures, skipped = [], {}, [], [], []
    replay_checked = dispatch_count = module_calls = offset = 0
    past = None
    for step in range(counts):
        phase = "prefill" if step == 0 else ("decode" if step == 1 else f"decode_{step}")
        suffix = "" if step < 2 else str(step)
        prefix = f"D{suffix}." if step else ""
        position = seq_len + step - 1
        cache_position = torch.tensor([position], device=device) if step else None
        ids = tokens[:, position:position + 1].contiguous() if step else tokens[:, :seq_len].contiguous()
        recording, output = record_forward(model, ids, past_key_values=past,
                                           cache_position=cache_position, seq_offset=offset)
        offset = recording._seq
        builder = GraphBuilder(model, recording, config, dtype, prefix.lower().rstrip("."), prefix)
        nodes = builder.build(phase)
        past = getattr(output, "past_key_values", None)
        if counts > 1 and past is None:
            raise ValueError(f"{phase} 未返回 decode 所需的历史缓存")
        cache = cache_tensors(past)
        external = {"input_ids": ids}
        if step:
            external["cache_position"] = cache_position
        observation = observe_ports(builder, nodes, recording.calls[0].outputs, cache, external)
        local_edges, local_nodes = build_edges(nodes, builder, observation, history, external)
        origins, _, slots, writes = observation
        graph_outputs[phase] = []
        for tensor in recording.calls[0].outputs:
            origin = origins[id(tensor)]
            producer = template_id_of(origin[0]) if collapse_template else origin[0]
            graph_outputs[phase].append(dict(node=producer, slot=slots[origin], tensor=builder.tid(tensor)))
        history = {tensor_view(cache[key]): (key, origin[0], slots[origin])
                   for key, origin in writes.items()}
        if len(history) != len(cache):
            raise ValueError(f"{phase} 多个缓存槽共享相同张量，归属不唯一")
        instances.extend(nodes)
        all_nodes.extend(local_nodes)
        edges.extend(local_edges)
        tensors.update(to_tensor_meta(builder))
        if replay_graph:
            replay = replay_nodes(nodes, builder, config, ids)
            replay_checked += replay["checked"]
            failures.extend(replay["failures"])
            skipped.extend(replay["skipped"])
        if reconcile_graph:
            invariant = check_invariants(builder, recording, nodes, local_edges)
            if counts > 1:
                for item in invariant["checks"]:
                    item["name"] += f"[{phase}]"
            phase_checks.extend(invariant["checks"])
            phase_stats[phase] = invariant["stats"]
            problems.extend(invariant["problems"])
        problems.extend(builder.conflicts)
        dispatch_count += len(recording.dispatches)
        module_calls += len(recording.calls)

    report = (reconcile(instances, all_nodes, edges, model, config) if reconcile_graph
              else {"ok": True, "checks": [], "stats": {}})
    if collapse_template:
        graph_nodes, graph_edges, fold_report = fold(all_nodes, edges)
    else:
        graph_nodes, graph_edges, fold_report = all_nodes, edges, {}
    report["checks"].extend(phase_checks)
    report["invariants"] = problems
    if replay_graph:
        stats = dict(checked=replay_checked, skipped=len(skipped), failed=len(failures))
        report["replay"] = dict(ok=not (failures or skipped), stats=stats, checked=replay_checked,
                                skipped=skipped, failures=failures)
        report["checks"].append(dict(name="R1 逐节点各输出重放", expected=0,
                                     actual=len(failures) + len(skipped), ok=not (failures or skipped),
                                     note=f"{replay_checked} 个节点通过"))
    report["stats"].update({key: sum(item[key] for item in phase_stats.values())
                            for key in next(iter(phase_stats.values()), {})})
    mismatches = fold_report.get("mismatches", [])
    report["stats"].update(instance_nodes=len(instances), templates=len(graph_nodes),
                           replay_checked=replay_checked, replay_failed=len(failures),
                           dispatch_count=dispatch_count, module_calls=module_calls,
                           dispatch_by_phase=phase_stats, template_mismatches=mismatches,
                           graph_inputs=sum(node.origin == "input" for node in all_nodes),
                           edges_unresolved={})
    meta = getattr(model, "_extract_meta", {})
    graph = Graph(
        model={key: meta.get(key) for key in ("model_path", "model_type", "architectures", "num_layers",
                                               "config_sha256", "weights_sha256", "weights_loaded")},
        env=dict(dtype=dtype, device=device, torch=torch.__version__, transformers=transformers.__version__,
                 python=sys.version.split()[0], platform=platform.platform()),
        phases=list(dict.fromkeys(node.phase for node in instances)), nodes=graph_nodes, edges=graph_edges,
        tensors=tensors, params=_collect_params(model),
        graph_inputs=[node.id for node in graph_nodes if node.origin == "input"],
        graph_outputs=graph_outputs,
        conflicts=mismatches + problems, reconcile=report,
    )
    graph.model["config"] = meta.get("config_json")
    report["validate_problems"] = graph.validate()
    report["ok"] = all(item["ok"] for item in report["checks"]) and not (
        graph.conflicts or report["validate_problems"])
    return graph


def _collect_params(model) -> list[dict]:
    out = []
    for name, p in model.named_parameters():
        out.append(
            {
                "name": name,
                "shape": list(p.shape),
                "dtype": str(p.dtype).replace("torch.", ""),
                "numel": int(p.numel()),
                "storage": hex(p.untyped_storage().data_ptr()),
            }
        )
    return out
