"""候选目录和调用适配器；不生成或改写 kernel。"""

import os
from functools import cache
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F

from task import linear_layers

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.getenv("CANDIDATE_SOURCE", str(
    ROOT.parent / "references/reps/GEAK-eval/geak_eval/data/TritonBench/data/TritonBench_G_v1/swiglu_fwd.py"
)))

KERNEL_CATALOG = [
    {"id": "torch_silu", "computation": "silu(gate)", "parameters": ["gate"],
     "output": "同 shape/dtype，PyTorch 单独激活"},
    {"id": "torch_mul", "computation": "a * b", "parameters": ["a", "b"],
     "output": "同 shape/dtype，逐元素乘法"},
    {"id": "triton_swiglu", "computation": "gate * sigmoid(gate) * up", "parameters": ["gate", "up"],
     "requires": "本 demo 已验证的连续二维 FP32 GPU Tensor；gate/up 同形状，列数为 intermediate_size",
     "adapter": "沿最后一维 cat([gate, up]) 后调用已有 _swiglu_fwd；拼接也计入耗时",
     "source": str(SOURCE)},
    {"id": "split_halves", "computation": "packed.chunk(2, dim=-1)", "parameters": ["packed"],
     "output": "两个 Tensor，依次为前半部分和后半部分；output 必须写两个名字的数组"},
    {"id": "triton_swiglu_packed", "computation": "对 packed 的前半部分 g 和后半部分 u 计算 silu(g)*u",
     "parameters": ["packed"], "requires": "连续二维 FP32 GPU Tensor，宽度为 2*intermediate_size",
     "adapter": "直接调用已有 _swiglu_fwd，不再拼接；输出宽度是输入的一半", "source": str(SOURCE)},
]


def candidate_catalog(reference):
    linears = [{"id": name, "computation": f"原模块 {name}(tensor)", "parameters": ["tensor"],
                "in_features": layer.in_features, "out_features": layer.out_features,
                "implementation": "保留原 Linear 和权重"} for name, layer in linear_layers(reference).items()]
    return linears + KERNEL_CATALOG


@cache
def load_swiglu():
    module = types.ModuleType("selector_existing_swiglu")
    module.__file__ = str(SOURCE)
    sys.modules[module.__name__] = module
    # 原文件底部会直接运行测试；只执行原始算子定义，保留真实文件位置供 Triton JIT 读取。
    definitions = SOURCE.read_text().split("################################################################")[0]
    exec(compile(definitions, str(SOURCE), "exec"), module.__dict__)
    return module._swiglu_fwd


def prepare_swiglu(projected):
    swiglu = load_swiglu()
    if "gate_up_proj" in projected:
        packed = projected["gate_up_proj"]
    else:
        packed = torch.cat([projected["gate_proj"], projected["up_proj"]], dim=-1)
    swiglu(packed)
    kernel = swiglu.__globals__["_swiglu_fwd_kernel"]
    config = kernel.best_config
    # 先用计时负载选配置，再固定它，避免正确性输入或后续方案重新调优。
    kernel.configs = [config]
    return {"source": str(SOURCE), "tuning_shape": [packed.size(0), packed.size(1) // 2],
            "packed_shape": list(packed.shape), "dtype": str(packed.dtype),
            "config": {**config.kwargs, "num_warps": config.num_warps,
                       "num_stages": config.num_stages, "num_ctas": config.num_ctas}}


def functions(reference):
    swiglu = load_swiglu()
    return {**linear_layers(reference), "torch_silu": F.silu, "torch_mul": torch.mul,
            "triton_swiglu": lambda gate, up: swiglu(torch.cat([gate, up], dim=-1)),
            "split_halves": lambda packed: packed.chunk(2, dim=-1), "triton_swiglu_packed": swiglu}


def manual_plan(reference, fused):
    if "gate_up_proj" in linear_layers(reference):
        steps = [{"op": "gate_up_proj", "inputs": ["x"], "output": "packed"}]
        if fused:
            steps.append({"op": "triton_swiglu_packed", "inputs": ["packed"], "output": "hidden"})
        else:
            steps.append({"op": "split_halves", "inputs": ["packed"], "output": ["gate", "up"]})
    else:
        steps = [{"op": "gate_proj", "inputs": ["x"], "output": "gate"},
                 {"op": "up_proj", "inputs": ["x"], "output": "up"}]
    if fused and "gate_up_proj" not in linear_layers(reference):
        steps.append({"op": "triton_swiglu", "inputs": ["gate", "up"], "output": "hidden"})
    if not fused:
        steps += [{"op": "torch_silu", "inputs": ["gate"], "output": "activated"},
                  {"op": "torch_mul", "inputs": ["activated", "up"], "output": "hidden"}]
    steps.append({"op": "down_proj", "inputs": ["hidden"], "output": "output"})
    return {"steps": steps, "output": "output", "reason": "人工方案，用于先验证候选和评测器"}


def execute(steps, operators, values):
    for step in steps:
        result = operators[step["op"]](*(values[name] for name in step["inputs"]))
        if isinstance(step["output"], list):
            if not isinstance(result, (tuple, list)) or len(result) != len(step["output"]):
                raise ValueError(f"{step['op']} 的实际返回值与 output 数量不匹配")
            values.update(zip(step["output"], result))
        else:
            values[step["output"]] = result
    return values


class SelectedMLP(torch.nn.Module):
    def __init__(self, reference, plan):
        super().__init__()
        self.reference = reference
        self.plan = plan
        self.operators = functions(reference)
        linears = linear_layers(reference)
        self.projection_steps = [step for step in plan["steps"] if step["op"] in linears and step["op"] != "down_proj"]
        self.activation_steps = [step for step in plan["steps"] if step["op"] not in linears]
        self.activation_output = next(step["inputs"][0] for step in plan["steps"] if step["op"] == "down_proj")

    def forward(self, x):
        return execute(self.plan["steps"], self.operators, {"x": x})[self.plan["output"]]

    def activation(self, projected):
        values = {step["output"]: projected[step["op"]] for step in self.projection_steps}
        return execute(self.activation_steps, self.operators, values)[self.activation_output]
