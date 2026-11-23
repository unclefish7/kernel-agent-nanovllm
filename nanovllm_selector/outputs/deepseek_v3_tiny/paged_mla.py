"""用 Transformers 实际产生的 Q/K/V 对照 nano-vLLM 分页注意力。"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn.functional import pad
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v3 import modeling_deepseek_v3 as native

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


@torch.inference_mode()
def compare(model_dir, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, trust_remote_code=False, local_files_only=True,
        attn_implementation="eager",
    ).cuda().eval()
    reference_attention = native.eager_attention_forward
    engines, observations = {}, []
    # 255 词提示，再逐词跨过引擎默认的 256 词页边界。逻辑页和物理页反序。
    pages = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    block_size = 256

    def compare_layer(module, query, key, value, *args, **kwargs):
        expected, weights = reference_attention(module, query, key, value, *args, **kwargs)
        _, heads, count, width = query.shape
        length, value_width = key.size(2), value.size(-1)
        layer = module.layer_idx
        if layer not in engines:
            engine = Attention(heads, width, module.scaling, key.size(1))
            engine.k_cache = key.new_zeros(2, block_size, key.size(1), width)
            engine.v_cache = torch.zeros_like(engine.k_cache)
            engines[layer] = engine
        engine = engines[layer]
        positions = torch.arange(length - count, length, device=key.device)
        slots = pages[0, positions // block_size] * block_size + positions % block_size
        prefill = count == length
        set_context(prefill, slot_mapping=slots, block_tables=None if prefill else pages,
                    cu_seqlens_q_list=[0, count], cu_seqlens_k_list=[0, length],
                    context_lens_list=[length])
        q = query[0].transpose(0, 1).contiguous()
        k = key[0, :, -count:].transpose(0, 1).contiguous()
        v = value[0, :, -count:].transpose(0, 1).contiguous()
        result = engine(q, k, pad(v, (0, width - value_width))).reshape(count, heads, width)
        actual = result[..., :value_width]
        expected_row = expected[0]
        # 检查所有逻辑位置，包含前一页中的历史。
        cached_k = engine.k_cache[pages[0].long()].flatten(0, 1)[:length]
        cached_v = engine.v_cache[pages[0].long()].flatten(0, 1)[:length]
        cache_exact = (torch.equal(cached_k, key[0].transpose(0, 1))
                       and torch.equal(cached_v[..., :value_width], value[0].transpose(0, 1))
                       and not cached_v[..., value_width:].count_nonzero().item())
        atol = 2e-6 if dtype == torch.float32 else 2e-3
        observations.append(dict(layer=layer, tokens=count, visible=length, qk_dim=width,
                                 v_dim=value_width, cache_exact=cache_exact,
                                 output_exact=torch.equal(actual, expected_row),
                                 max_abs=(actual.float() - expected_row.float()).abs().max().item(),
                                 atol=atol, within_atol=torch.allclose(
                                     actual, expected_row, rtol=0, atol=atol)))
        # 下一层的输入仍由完整原生模型产生，各层误差独立测量。
        return expected, weights

    try:
        with patch.object(native, "eager_attention_forward", compare_layer):
            history = None
            for ids in (torch.arange(1, 256), torch.tensor([256]), torch.tensor([257])):
                out = model(ids.cuda()[None, :], past_key_values=history, use_cache=True,
                            logits_to_keep=1)
                history = out.past_key_values
    finally:
        reset_context()
    ok = (len(observations) == 3 * model.config.num_hidden_layers
          and all(row["cache_exact"] and row["within_atol"] for row in observations))
    return dict(model=str(model_dir), dtype=str(dtype), observations=observations, ok=ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"device": str(torch.cuda.get_device_properties(0)),
              "runs": [compare(args.model, dtype) for dtype in (torch.float32, torch.bfloat16)]}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(0 if all(run["ok"] for run in report["runs"]) else 1)
