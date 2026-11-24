"""在同一份旧候选上逐组替换实现，每个对照组使用独立进程。"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from nanovllm_selector.port.evaluate import evaluate
from .model_qwen3 import Qwen3AttentionImpl, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb


class ReferenceRotary(Qwen3RotaryEmbedding):
    def forward(self, positions, query, key):
        cos, sin = super().forward(positions, query.dtype)
        return apply_rotary_pos_emb(query, key, cos, sin)


def get_rotary(head_dim, *, rotary_dim, max_position, base):
    assert head_dim == rotary_dim
    return ReferenceRotary(head_dim, base)


def split_projections(layer, x):
    if isinstance(layer, QKVParallelLinear):
        widths = [layer.head_size * h for h in (layer.num_heads, layer.num_kv_heads, layer.num_kv_heads)]
    else:
        widths = [w // layer.tp_size for w in layer.output_sizes]
    weights = layer.weight.split(widths)
    biases = layer.bias.split(widths) if layer.bias is not None else [None] * len(widths)
    return torch.cat([F.linear(x, w, b) for w, b in zip(weights, biases)], dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    groups = ("norm", "norm_rope", "norm_rope_attention", "norm_rope_attention_activation", "all")
    parser.add_argument("group", choices=groups)
    args = parser.parse_args()
    directory = Path("nanovllm_selector/outputs")
    candidate = directory / "iter_qwen3/attempt_1.py"
    with ExitStack() as stack:
        stack.enter_context(patch("nanovllm.layers.layernorm.RMSNorm", Qwen3RMSNorm))
        if args.group in groups[1:]:
            stack.enter_context(patch("nanovllm.layers.rotary_embedding.get_rope", get_rotary))
        if args.group in groups[2:]:
            stack.enter_context(patch("nanovllm.layers.attention.Attention", Qwen3AttentionImpl))
        if args.group in groups[3:]:
            stack.enter_context(patch.object(SiluAndMul, "forward", SiluAndMul.forward.__wrapped__))
        if args.group == "all":
            for layer in (QKVParallelLinear, MergedColumnParallelLinear):
                stack.enter_context(patch.object(layer, "forward", split_projections))
        result = evaluate("tmp/models/Qwen3-0.6B", candidate, exact=True)
    result["control_group"] = args.group
    path = directory / "qwen3_exact" / f"control_{args.group}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if result["stage"] != "compare" and not result["ok"]:
        raise RuntimeError(result.get("error", result["stage"]))


if __name__ == "__main__":
    main()
