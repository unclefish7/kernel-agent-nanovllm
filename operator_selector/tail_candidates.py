"""Norm、MLP、残差候选；这里只做已有实现的接口适配。"""

import importlib.util
from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from candidates import execute, load_swiglu
from task import linear_layers

VENDOR = Path(__file__).resolve().parent / "vendor"


@cache
def load_vendor(name):
    spec = importlib.util.spec_from_file_location("selector_" + name, VENDOR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Triton 需要能通过模块名和真实文件读取 kernel 定义。
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def catalog(reference):
    norm = reference.layer.post_attention_layernorm
    common = {"requires": "二维 FP32 MI100 GPU Tensor，最后一维 stride=1；归一化宽度 256，SwiGLU 中间宽度 1024；需要连续化时由标注的适配器完成"}
    rows = [
        {"id": "torch_add", "parameters": ["a", "b"], "computation": "a+b", "returns": 1},
        {"id": "rms_reference", "parameters": ["input"], "computation": "真实 post_attention_layernorm(input)", "returns": 1},
        {"id": "rms_row", "parameters": ["input"], "computation": "input*rsqrt(mean(input^2)+eps)*weight",
         "source": "vendor/rmsnorm_fused.py", "returns": 1},
        {"id": "rms_loop", "parameters": ["input"], "computation": "input/sqrt(mean(input^2)+eps)*weight",
         "source": "vendor/rmsnorm_implementation.py", "adapter": "增加长度为 1 的 batch 维再移除，仅 view", "returns": 1},
        {"id": "add_rms", "parameters": ["a", "b"], "computation": "s=a+b; normalized=s*rsqrt(mean(s^2)+eps)*weight",
         "output_order": ["normalized", "s"], "returns": 2, "source": "vendor/layer_norm_ops.py",
         "adapter": "原 _layer_norm_fwd 设置 is_rms_norm=True、bias=None，返回归一化结果和相加结果；不修改输入"},
        {"id": "torch_silu", "parameters": ["gate"], "computation": "silu(gate)", "returns": 1},
        {"id": "torch_mul", "parameters": ["a", "b"], "computation": "a*b", "returns": 1},
        {"id": "swiglu_row", "parameters": ["gate", "up"], "computation": "silu(gate)*up", "returns": 1,
         "source": "vendor/swiglu_triton.py", "adapter": "连续化两个输入后调用原 swiglu_forward，取第三个返回值；复制开销计入耗时"},
    ]
    packed = hasattr(reference.layer.mlp, "gate_up_proj")
    rows.append({"id": "swiglu_tiles", "parameters": ["packed"] if packed else ["gate", "up"], "returns": 1,
                 "computation": "对 packed 前后半部分计算 silu(gate)*up" if packed else "silu(gate)*up",
                 "source": "GEAK-eval/swiglu_fwd.py",
                 "adapter": "直接用原 packed wrapper" if packed else "直接启动原 kernel，传入两个 Tensor 的 stride，不做拼接"})
    if packed:
        rows.append({"id": "split_halves", "parameters": ["packed"], "computation": "packed.chunk(2,dim=-1)",
                     "returns": 2, "output_order": ["前半部分", "后半部分"],
                     "layout": "返回两个 view，行 stride 是打包张量的宽度，不保证连续"})
    for row in rows:
        row.update(common)
        if row["id"].startswith("rms_") or row["id"] == "add_rms":
            row["bound_parameters"] = {"weight": "原 post_attention_layernorm.weight，不改变权重",
                                       "eps": norm.variance_epsilon}
    rows += [{"id": "mlp." + name, "parameters": ["input"], "returns": 1,
              "computation": "原 Linear，与 reference 共用对象及权重",
              "in_features": layer.in_features, "out_features": layer.out_features}
             for name, layer in linear_layers(reference.layer.mlp).items()]
    return rows


def tiled_separate(gate, up):
    out = torch.empty_like(gate)
    kernel = load_swiglu().__globals__["_swiglu_fwd_kernel"]
    kernel[lambda meta: (gate.size(0), triton.cdiv(gate.size(1), meta["BLOCK_N"]))](
        gate, up, out, gate.stride(0), up.stride(0), out.stride(0), gate.size(1))
    return out


def functions(reference):
    norm = reference.layer.post_attention_layernorm
    fused_norm = load_vendor("rmsnorm_fused").TritonLlamaRMSNorm(norm.weight, norm.variance_epsilon)
    norm_loop = load_vendor("rmsnorm_implementation").rmsnorm_wrapper
    norm_add = load_vendor("layer_norm_ops")._layer_norm_fwd

    def add_rms(a, b):
        normalized, _, _, summed = norm_add(a, norm.weight, None, norm.variance_epsilon,
                                           residual=b, is_rms_norm=True)
        return normalized, summed

    def swiglu_row(gate, up):
        # 上游 wrapper 假设两个输入与输出行 stride 相同，因此先连续化。
        return load_vendor("swiglu_triton").swiglu_forward(gate.contiguous(), up.contiguous())[2]

    return {"torch_add": torch.add, "rms_reference": norm, "rms_row": fused_norm,
            "rms_loop": lambda x: norm_loop(x.unsqueeze(0), norm.weight, norm.variance_epsilon).squeeze(0),
            "add_rms": add_rms, "torch_silu": F.silu, "torch_mul": torch.mul,
            "split_halves": lambda x: x.chunk(2, dim=-1), "swiglu_row": swiglu_row,
            "swiglu_tiles": load_swiglu() if hasattr(reference.layer.mlp, "gate_up_proj") else tiled_separate,
            **{"mlp." + name: layer for name, layer in linear_layers(reference.layer.mlp).items()}}


def manual_plan(reference, norm="reference", activation="torch"):
    steps = []

    def add(op, inputs, output):
        steps.append({"op": op, "inputs": inputs, "output": output})

    if norm == "add_fused":
        add("add_rms", ["x", "residual"], ["normalized", "summed"])
    else:
        add("torch_add", ["residual", "x"], "summed")
        add("rms_" + norm, ["summed"], "normalized")
    if hasattr(reference.layer.mlp, "gate_up_proj"):
        add("mlp.gate_up_proj", ["normalized"], "packed")
        if activation == "tiles":
            add("swiglu_tiles", ["packed"], "hidden")
        else:
            add("split_halves", ["packed"], ["gate", "up"])
    else:
        add("mlp.gate_proj", ["normalized"], "gate")
        add("mlp.up_proj", ["normalized"], "up")
        if activation == "tiles":
            add("swiglu_tiles", ["gate", "up"], "hidden")
    if activation == "torch":
        add("torch_silu", ["gate"], "activated")
        add("torch_mul", ["activated", "up"], "hidden")
    elif activation == "row":
        add("swiglu_row", ["gate", "up"], "hidden")
    add("mlp.down_proj", ["hidden"], "mlp_out")
    add("torch_add", ["summed", "mlp_out"], "output")
    return {"steps": steps, "output": "output", "reason": "人工合法组合，用于候选对照，不提供给 Agent"}


class SelectedTail(torch.nn.Module):
    def __init__(self, reference, plan, operators):
        super().__init__()
        self.reference, self.plan, self.operators = reference, plan, operators

    def forward(self, x, residual):
        return execute(self.plan["steps"], self.operators, {"x": x, "residual": residual})[self.plan["output"]]


def prepare(reference, inputs, operators):
    # 每个输入规模重新调优；完成后固定配置，所有方案共同复用。
    x, residual = inputs
    summed = x + residual
    projected = reference.layer.mlp.gate_up_proj(reference.layer.post_attention_layernorm(summed)) \
        if hasattr(reference.layer.mlp, "gate_up_proj") else None
    normalized = reference.layer.post_attention_layernorm(summed)
    args = [projected] if projected is not None else [reference.layer.mlp.gate_proj(normalized),
                                                   reference.layer.mlp.up_proj(normalized)]
    kernels = [("swiglu_tiles", load_swiglu().__globals__["_swiglu_fwd_kernel"], args),
               ("add_rms", load_vendor("layer_norm_ops")._layer_norm_fwd_1pass_kernel, [x, residual])]
    result = {}
    for name, kernel, values in kernels:
        if not hasattr(kernel, "selector_original_configs"):
            kernel.selector_original_configs = list(kernel.configs)
        kernel.configs = kernel.selector_original_configs
        kernel.cache.clear()
        operators[name](*values)
        chosen = kernel.best_config
        kernel.configs = [chosen]
        result[name] = {"config": {**chosen.kwargs, "num_warps": chosen.num_warps,
                                  "num_stages": chosen.num_stages, "num_ctas": chosen.num_ctas},
                        "input_shapes": [list(value.shape) for value in values]}
    return result
