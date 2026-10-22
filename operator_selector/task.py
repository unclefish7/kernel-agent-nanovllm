"""从真实 Transformers 模块获取源码、配置和测试输入。"""

import inspect

import torch
from transformers import Phi3Config, Qwen3Config
from transformers.models.phi3.modeling_phi3 import Phi3MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


MODELS = {"qwen3": (Qwen3Config, Qwen3MLP), "phi3": (Phi3Config, Phi3MLP)}


def linear_layers(reference):
    return {name: layer for name, layer in reference.named_children() if isinstance(layer, torch.nn.Linear)}


def activation_module(reference):
    return next(layer for layer in reference.children() if not isinstance(layer, torch.nn.Linear))


def project(reference, x):
    return {name: layer(x) for name, layer in linear_layers(reference).items() if name != "down_proj"}


def original_activation(reference, projected):
    if "gate_up_proj" in projected:
        gate, up = projected["gate_up_proj"].chunk(2, dim=-1)
    else:
        gate, up = projected["gate_proj"], projected["up_proj"]
    return activation_module(reference)(gate) * up


def make_task(model="qwen3"):
    torch.manual_seed(42)
    config_cls, module_cls = MODELS[model]
    config = config_cls(hidden_size=256, intermediate_size=1024, hidden_act="silu")
    reference = module_cls(config).to(device="cuda:0", dtype=torch.float32).eval()
    inputs = [torch.randn(tokens, config.hidden_size, device="cuda:0") * 2 for tokens in (8, 128)]
    context = {
        "module": f"{module_cls.__module__}.{module_cls.__name__}",
        "source_file": inspect.getfile(module_cls),
        "source": inspect.getsource(module_cls),
        "activation_source": inspect.getsource(type(activation_module(reference)).forward),
        "linear_layers": {name: {"in_features": layer.in_features, "out_features": layer.out_features,
                                 "bias": layer.bias is not None} for name, layer in linear_layers(reference).items()},
        "config": {"hidden_size": config.hidden_size, "intermediate_size": config.intermediate_size,
                   "hidden_act": config.hidden_act},
        "input": {"shape": list(inputs[-1].shape), "dtype": str(inputs[-1].dtype),
                  "stride": list(inputs[-1].stride())},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "architecture": "gfx908",
                     "torch": torch.__version__, "hip": torch.version.hip},
    }
    return reference, inputs, context
