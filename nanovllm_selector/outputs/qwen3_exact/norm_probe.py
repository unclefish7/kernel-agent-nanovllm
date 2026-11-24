"""在同一份 Qwen3 第一层输入上对照逐步计算、引擎编译路径与延迟舍入。"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm.layers.layernorm import RMSNorm
from nanovllm_selector.port.evaluate import FILLER_TEXT


@torch.inference_mode()
def main():
    torch.use_deterministic_algorithms(True)
    directory = "tmp/models/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16,
                                                local_files_only=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    ids = tokenizer(FILLER_TEXT, return_tensors="pt").input_ids[:, :128].cuda()
    layer = model.model.layers[0].input_layernorm
    x = model.model.embed_tokens(ids)
    ref = layer(x).flatten(0, 1)
    x = x.flatten(0, 1)
    engine = RMSNorm(x.size(-1), eps=layer.variance_epsilon)
    engine.weight = layer.weight
    original = engine(x)
    eager = RMSNorm.rms_forward.__wrapped__(engine, x)
    fp = x.float()
    scaled = fp * torch.rsqrt(fp.pow(2).mean(-1, keepdim=True) + layer.variance_epsilon)
    late = (scaled * layer.weight.float()).to(x.dtype)

    def diff(a, b):
        return {"different_elements": (a != b).sum().item(),
                "max_abs_diff": (a.float() - b.float()).abs().max().item()}

    print(json.dumps({"shape": list(x.shape), "device": str(torch.cuda.get_device_properties(0)),
                      "compiled_vs_reference": diff(original, ref), "eager_vs_reference": diff(eager, ref),
                      "late_rounding_vs_reference": diff(late, ref), "compiled_vs_late_rounding": diff(original, late)},
                     indent=2))


if __name__ == "__main__":
    main()
