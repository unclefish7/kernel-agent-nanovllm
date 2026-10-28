"""MLP 激活候选及安装器；不持有 Decoder，不更改权重。"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from nanovllm.utils.context import get_context
from vendor.swiglu_fwd import _swiglu_fwd


def catalog():
    manifest = json.loads((Path(__file__).parent / "vendor/manifest.json").read_text())
    source = manifest["swiglu_fwd.py"]
    return [
        {"id": "native", "operation": "SiLU(gate) * up",
         "input": "packed gate/up: [..., 2 * intermediate_size]",
         "implementation": "nano-vLLM SiluAndMul，保留原对象作为回退",
         "support": "原生支持范围；未知规格时优先保留原生路径",
         "performance": "未做性能比较，不保证更快"},
        {"id": "geak_packed_swiglu", "operation": "SiLU(gate) * up",
         "input": "[rows, 2 * intermediate_size]；前半 gate，后半 up",
         "output": "[rows, intermediate_size]，相同 dtype/device，新分配输出，不改输入",
         "support": "本实验限 ROCm gfx908，二维非空连续张量，float16/bfloat16/float32",
         "implementation": "现有 GEAK-eval _swiglu_fwd；不包含 Linear 或权重",
         "rounding": "内部 FP32 计算后写回输入 dtype；可能与原生分步舍入不同，尚未数值验收",
         "performance": "未做性能比较；上游包含 autotune，不保证更快",
         "source": source},
    ]


def catalog_hash():
    return hashlib.sha256(json.dumps(catalog(), sort_keys=True).encode()).hexdigest()


def unsupported_reason(x):
    if x.device.type != "cuda" or torch.version.hip is None:
        return "需要 ROCm GPU"
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx908":
        return "本轮只开放 gfx908"
    if x.ndim != 2 or x.numel() == 0 or x.shape[-1] % 2:
        return "需要非空二维 packed gate/up，末维为偶数"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "不支持该 dtype"
    if not x.is_contiguous():
        return "本轮不适配非连续输入"
    return None


def validate_plan(plan, binding):
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "binding", "choice"} or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("方案结构或版本不合法")
    if plan["binding"] != binding:
        raise ValueError("方案与当前模型配置、硬件或候选版本不匹配，需要重新 prepare/select")
    choice = plan["choice"]
    if not isinstance(choice, dict) or set(choice) != {"target", "candidate_id", "reason"}:
        raise ValueError("choice 只能包含 target、candidate_id、reason")
    if choice["target"] != "mlp_activation":
        raise ValueError("本轮只允许替换 mlp_activation")
    if not isinstance(choice["candidate_id"], str) or choice["candidate_id"] not in {item["id"] for item in catalog()}:
        raise ValueError("Agent 返回了未登记的候选 ID")
    if not isinstance(choice["reason"], str) or not choice["reason"].strip():
        raise ValueError("方案必须附带选择理由")


class SelectedActivation(nn.Module):
    def __init__(self, original, candidate_id):
        super().__init__()
        self.original = original
        self.candidate_id = candidate_id
        self.warming = True
        self.calls = Counter()
        self.fallbacks = Counter()

    def forward(self, x):
        phase = "warmup" if self.warming else ("prefill" if get_context().is_prefill else "decode")
        reason = unsupported_reason(x) if self.candidate_id != "native" else None
        if reason:
            self.fallbacks[f"{phase}: {reason}"] += 1
            self.calls[f"{phase}/fallback_native"] += 1
            return self.original(x)
        if self.candidate_id == "native":
            output = self.original(x)
        else:
            # 不拦截编译或运行异常，避免静默回退后伪称候选已生效。
            output = _swiglu_fwd(x)
        if output.shape != (*x.shape[:-1], x.shape[-1] // 2) or output.dtype != x.dtype or output.device != x.device:
            raise RuntimeError("候选返回规格不符合 packed 激活接口")
        self.calls[f"{phase}/{self.candidate_id}"] += 1
        return output


def install(model, plan, binding):
    validate_plan(plan, binding)
    targets = [(f"model.layers.{i}.mlp.act_fn", layer.mlp)
               for i, layer in enumerate(model.model.layers)]
    if not targets or any(isinstance(mlp.act_fn, SelectedActivation) for _, mlp in targets):
        raise ValueError("没有目标层或已安装方案")
    installed = []
    for path, mlp in targets:
        wrapper = SelectedActivation(mlp.act_fn, plan["choice"]["candidate_id"])
        mlp.act_fn = wrapper
        installed.append((path, mlp, wrapper))
    return installed


def restore(installed):
    for _, mlp, wrapper in installed:
        mlp.act_fn = wrapper.original


def call_report(installed):
    return {path: {"candidate_id": wrapper.candidate_id, "calls": dict(wrapper.calls),
                   "fallbacks": dict(wrapper.fallbacks)} for path, _, wrapper in installed}
