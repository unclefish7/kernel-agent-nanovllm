"""在 CPU 上隔离检查 LFM2.5 的抽取、历史状态、引擎缓存分配和参照评测。"""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import transformers
from torch import nn

from nanovllm.engine.model_runner import ModelRunner
from nanovllm_selector.extract.decompose import GraphBuilder
from nanovllm_selector.extract.loader import load_model
from nanovllm_selector.extract.recorder import record_forward
from nanovllm_selector.extract.reconcile import reconcile
from nanovllm_selector.extract.replay import IMPL
from nanovllm_selector.extract.trace_io import cache_tensors

MODEL = Path("tmp/models/LFM2.5-1.2B-Base")


def error_from(func, *args):
    try:
        func(*args)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def check_engine(config):
    # 在 CPU 上执行真实分配方法。按官方层表提供缓存接口，仅替代显存计量。
    layers = nn.ModuleList([nn.Module() for _ in config.layer_types])
    for kind, layer in zip(config.layer_types, layers):
        if kind == "full_attention":
            layer.k_cache = layer.v_cache = None
        else:
            layer.conv_cache = None
    capacity = 64 * 1024 * 1024
    runner = SimpleNamespace(
        config=SimpleNamespace(hf_config=config, gpu_memory_utilization=0.5),
        world_size=1, block_size=256, model=layers,
    )
    with patch.object(torch.cuda, "mem_get_info", return_value=(capacity, capacity)), patch.object(
        torch.cuda, "memory_stats", return_value={"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0}
    ):
        failure = error_from(ModelRunner.allocate_kv_cache, runner)
    return {"cpu_interface_probe": True, "error": failure,
            "assigned_kv_modules": sum(getattr(layer, "k_cache", None) is not None for layer in layers),
            "assigned_conv_modules": sum(getattr(layer, "conv_cache", None) is not None for layer in layers)}


@torch.no_grad()
def main():
    torch.manual_seed(17)
    model, config = load_model(str(MODEL))
    seq = torch.tensor([[11, 23, 37, 41, 53, 67, 79]])
    result = dict(model_type=config.model_type, layer_types=config.layer_types, real_weights=False,
                  seed=17, device="cpu", torch=torch.__version__, transformers=transformers.__version__,
                  config_sha256=hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest())
    result["engine"] = check_engine(config)
    conv = [i for i, k in enumerate(config.layer_types) if k == "conv"]
    attn = [i for i, k in enumerate(config.layer_types) if k == "full_attention"]
    # 用原生完整前向作参照，每个位置只能读取它之前的信息。
    full = model(input_ids=seq, use_cache=False).logits
    past = old_windows = None
    phases = []
    for begin, end in [(0, 5), (5, 6), (6, 7)]:
        phase = ("prefill" if begin == 0 else f"decode_{begin-4}")
        rec, out = record_forward(model, seq[:, begin:end], past_key_values=past,
                                  cache_position=torch.tensor([begin]) if begin else None)
        past = out.past_key_values
        builder = GraphBuilder(model, rec, config, "float32", "", "")
        nodes = builder.build(phase)
        conv_nodes = [n for n in nodes if n.attrs["module_class"] == "Conv1d"]
        # 只保留第一个卷积层的详细节点，以便检查逐词路径的底层操作如何命名。
        first = [{"module": n.attrs["module_path"], "label": n.op,
                  "operations": list(n.attrs["dispatch_ops"])} for n in nodes
                 if n.attrs["module_path"].startswith("model.layers.0.conv")]
        windows = [past.conv_cache[i].clone() for i in conv]
        keys = [past.key_cache[i] for i in attn]
        vals = [past.value_cache[i] for i in attn]
        diff = (out.logits[:, -1] - full[:, end - 1]).abs()
        entry = dict(phase=phase, cached_length=past.get_seq_length(),
                     finite=bool(torch.isfinite(out.logits).all()),
                     max_abs_diff_to_full=float(diff.max()),
                     top1_matches_full=bool(out.logits[:, -1].argmax() == full[:, end-1].argmax()),
                     conv_shapes=dict(Counter(str(list(w.shape)) for w in windows)),
                     key_shapes=dict(Counter(str(list(w.shape)) for w in keys)),
                     value_shapes=dict(Counter(str(list(w.shape)) for w in vals)),
                     conv1d_labels=dict(Counter(n.op for n in conv_nodes)),
                     first_conv_nodes=first,
                     unknown_labels=sum(n.op == "unknown" for n in nodes))
        if old_windows is not None:
            entry["layers_with_preserved_overlap"] = sum(
                torch.equal(old[..., 1:], new[..., :-1]) for old, new in zip(old_windows, windows))
            entry["layers_with_changed_last_slot"] = sum(
                not torch.equal(old[..., -1], new[..., -1]) for old, new in zip(old_windows, windows))
        if begin == 0:
            result["cache_class"] = type(past).__name__
            result["extract_cache_error"] = error_from(cache_tensors, past)
            cropped = copy.deepcopy(past)
            cropped.crop(3)
            clean = model(input_ids=seq[:, :3], use_cache=True).past_key_values
            prefix = {
                "attention_length_after_crop": cropped.get_seq_length(),
                "conv_windows_unchanged_from_length_5": sum(
                    torch.equal(cropped.conv_cache[i], past.conv_cache[i]) for i in conv),
                "conv_windows_different_from_clean_length_3": sum(
                    not torch.equal(cropped.conv_cache[i], clean.conv_cache[i]) for i in conv),
            }
            def resume(cache, ids):
                return model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True).logits
            single_clean = resume(clean, seq[:, 3:4])
            single_cropped = resume(cropped, seq[:, 3:4])
            multi_cached = resume(clean, seq[:, 3:5])
            one_at_a_time = model(input_ids=seq[:, 3:4], past_key_values=copy.deepcopy(clean), use_cache=True)
            two_serial = resume(one_at_a_time.past_key_values, seq[:, 4:5])
            prefix.update(
                cropped_vs_clean_next_max_abs=float((single_clean - single_cropped).abs().max()),
                clean_multitoken_vs_serial_last_max_abs=float((multi_cached[:, -1] - two_serial[:, -1]).abs().max()),
                clean_serial_vs_full_last_max_abs=float((two_serial[:, -1] - full[:, 4]).abs().max()),
            )
            result["reference_prefix"] = prefix
            counts = reconcile(nodes, nodes, [], model, config)["checks"]
            result["activation_count_check"] = next(c for c in counts if c["name"] == "silu")
            result["has_conv_replay_rule"] = "unknown" in IMPL or any("conv" in op for op in IMPL)
            lost = copy.deepcopy(past)
            lost.conv_cache[conv[0]].zero_()
            with_history = model(input_ids=seq[:, 5:6], past_key_values=copy.deepcopy(past), use_cache=True).logits
            lost_history = model(input_ids=seq[:, 5:6], past_key_values=lost, use_cache=True).logits
            result["zero_first_conv_window_next_logits"] = dict(
                different_entries=int((with_history != lost_history).sum()),
                max_abs_diff=float((with_history-lost_history).abs().max()), entries=with_history.numel())
        phases.append(entry)
        old_windows = windows
    result["phases"] = phases
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
