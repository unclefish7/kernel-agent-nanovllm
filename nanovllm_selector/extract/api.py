"""对外 API：`extract_graph(model_path) -> Graph`。

这一层只做编排，所有判断都在下面各模块里，方便单独测试。
"""

from __future__ import annotations

import platform
import sys

import torch
import transformers

from .decompose import GraphBuilder
from .edges import build_edges, to_tensor_meta
from .loader import load_model
from .schema import StateRef
from .recorder import record_forward
from .checks import check_invariants
from .replay import replay_nodes
from .reconcile import reconcile
from .schema import Graph
from .template import fold

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
) -> Graph:
    """从一个 HuggingFace 模型目录重建计算图。

    参数
    ----
    model_path : 本地模型目录（只需 config.json；weights=False 时不读权重）
    dtype : float32 / float16 / bfloat16
    device : cpu / cuda
    seq_len : 造多长的输入序列
    phases : M1 只支持 ("prefill",)
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

    vocab = int(getattr(config, "vocab_size", 1000))
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (batch_size, seq_len), generator=generator).to(device)

    rec, out = record_forward(model, input_ids)

    builder = GraphBuilder(model, rec, config, dtype=dtype)
    instance_nodes = builder.build(phase="prefill")
    edges, all_nodes, edge_info = build_edges(instance_nodes, builder)
    tensors = to_tensor_meta(builder)

    decode_tensors: dict = {}
    if "decode" in phases:
        past = getattr(out, "past_key_values", None)
        if past is not None:
            step = torch.tensor([input_ids.shape[-1]], device=input_ids.device)
            rec2, _ = record_forward(
                model, input_ids[:, -1:], past_key_values=past,
                cache_position=step, seq_offset=rec._seq,
            )
            builder2 = GraphBuilder(model, rec2, config, dtype=dtype)
            dec_nodes = _prefixed(builder2.build(phase="decode"))
            dec_edges, _, _ = build_edges(dec_nodes, builder2, id_prefix="D.")
            instance_nodes = instance_nodes + dec_nodes
            edges = edges + dec_edges
            all_nodes = all_nodes + dec_nodes
            decode_tensors.update({f"d{k}": v for k, v in builder2._tensors.items()})
            _mark_cross_step_state(dec_nodes, builder2, builder, model)
            tensors.update({f"d{k}": v for k, v in to_tensor_meta(builder2).items()})

    if collapse_template:
        graph_nodes, graph_edges, fold_report = fold(all_nodes, edges)
    else:
        graph_nodes, graph_edges, fold_report = all_nodes, edges, {"templates": len(all_nodes)}

    report = (
        reconcile(instance_nodes, all_nodes, edges, model, config, rec)
        if reconcile_graph
        else {"ok": True, "checks": [], "stats": {}}
    )
    if replay_graph:
        rep = replay_nodes(
            instance_nodes, builder, config, input_ids,
            tensors={**builder._tensors, **decode_tensors},
        )
        report["replay"] = rep
        report["ok"] = report["ok"] and rep["ok"]
        report["stats"]["replay_checked"] = rep["stats"]["checked"]
        report["stats"]["replay_failed"] = rep["stats"]["failed"]
        report["checks"].append({
            "name": "R1 逐节点重放",
            "expected": 0,
            "actual": rep["stats"]["failed"] + rep["stats"]["skipped"],
            "ok": rep["ok"],
            "note": f"{rep['stats']['checked']} 个实例节点按图重算，"
                    f"{rep['stats']['failed']} 个失败、{rep['stats']['skipped']} 个未覆盖"
                    f"（未覆盖也计为不通过）",
        })
    if reconcile_graph:
        inv = check_invariants(builder, rec, instance_nodes, edges)
        report["checks"].extend(inv["checks"])
        report["ok"] = report["ok"] and inv["ok"]
        report["stats"].update(inv["stats"])
        report["invariants"] = inv["problems"]
    report["stats"]["instance_nodes"] = len(instance_nodes)
    report["stats"]["templates"] = fold_report.get("templates")
    report["stats"]["template_mismatches"] = fold_report.get("mismatches", [])
    report["stats"]["edges_unresolved"] = edge_info["unresolved"]
    report["stats"]["graph_inputs"] = edge_info["inputs"]

    meta = getattr(model, "_extract_meta", {})
    graph = Graph(
        model={
            "model_path": meta.get("model_path"),
            "model_type": meta.get("model_type"),
            "architectures": meta.get("architectures"),
            "num_layers": meta.get("num_layers"),
            "config_sha256": meta.get("config_sha256"),
            "weights_sha256": meta.get("weights_sha256"),
            "weights_loaded": meta.get("weights_loaded"),
            # 完整 config 进图：eps、rope_theta、tie_word_embeddings 这些值
            # 在写模型文件时要用，而它们不在任何一个节点的 state 里。
            "config": meta.get("config_json"),
        },
        env={
            "dtype": dtype,
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        phases=list(phases),
        nodes=graph_nodes,
        edges=graph_edges,
        tensors=tensors,
        params=_collect_params(model),
        graph_inputs=[n.id for n in graph_nodes if n.origin == "input"],
        conflicts=report["stats"].get("template_mismatches", []),
        reconcile=report,
    )
    problems = graph.validate()
    report["validate_problems"] = problems
    report["ok"] = bool(report.get("ok")) and not problems
    return graph


def _mark_cross_step_state(nodes, builder2, builder_pre, model):
    """decode 里"不是这一步造的、但在 prefill 里出现过"的张量 = 跨 step 状态。

    典型就是 KV cache：prefill 写入，decode 读出来拼到当前 k/v 前面。
    这不是普通数据输入，它是节点自己的状态，要单独标出来。
    """
    pre_storages = {m["storage"] for m in builder_pre._tensors.values()}
    param_storages = {hex(p.untyped_storage().data_ptr()) for p in model.parameters()}
    for n in nodes:
        if n.op != "attention":
            continue
        for t in n.inputs:
            m = builder2._tensors.get(t.tensor.lstrip("d"))
            if m is None or m["storage"] in param_storages:
                continue
            if m["storage"] in pre_storages and len(m["shape"]) == 4:
                t.role = "state"
                n.state.append(StateRef(
                    name=f"kv_cache_{len(n.state)}",
                    kind="runtime_tensor",
                    role="read_write",
                    shape=list(m["shape"]),
                    dtype=m["dtype"],
                ))


def _prefixed(nodes):
    """decode 的节点 id 与张量 id 都加前缀，避免和 prefill 撞名。

    两个 builder 各自从 t0000 开始编号，不区分就会把 decode 的张量
    元数据覆盖到 prefill 上。
    """
    for n in nodes:
        n.id = "D." + n.id
        if n.template_id:
            n.template_id = "D." + n.template_id
        for t in n.inputs:
            t.tensor = "d" + t.tensor
        for t in n.outputs:
            t.tensor = "d" + t.tensor
        if getattr(n, "_in_srcs", None):
            n._in_srcs = {f"d{k}": v for k, v in n._in_srcs.items()}
        empty = n.attrs.get("empty_inputs")
        if empty:
            n.attrs["empty_inputs"] = [f"d{tid}" for tid in empty]
    return nodes


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
