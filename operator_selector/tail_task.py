"""从真实 Decoder forward 提取 Attention 后的尾部，不运行 Attention。"""

import ast
import inspect
import textwrap

import torch
from transformers import Phi3Config, Qwen3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


MODELS = {"qwen3": (Qwen3Config, Qwen3DecoderLayer), "phi3": (Phi3Config, Phi3DecoderLayer)}


def extract_tail(layer_cls):
    original = textwrap.dedent(inspect.getsource(layer_cls.forward))
    function = ast.parse(original).body[0]
    # 边界限定为第一次 Attention 输出之后的残差相加；不是任意模型提取器。
    start = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                 and isinstance(node.value.op, ast.Add)
                 and isinstance(node.value.left, ast.Name) and node.value.left.id == "residual")
    function.body = function.body[start:]
    function.name = "extracted_forward"
    function.decorator_list = []
    function.returns = None
    function.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
                                  ("self", "hidden_states", "residual")], kwonlyargs=[],
                                  kw_defaults=[], defaults=[])
    ast.fix_missing_locations(function)
    source = ast.unparse(function)
    namespace = {}
    exec(compile(source, inspect.getfile(layer_cls), "exec"), namespace)
    return namespace[function.name], source


class TailReference(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.extracted, self.source = extract_tail(type(layer))

    def forward(self, x, residual):
        return self.extracted(self.layer, x, residual)

    def check_original(self, x, residual):
        class FixedAttention(torch.nn.Module):
            def forward(self, **kwargs):
                return x, None

        original = self.layer.self_attn
        self.layer.self_attn = FixedAttention()
        try:
            # 保留真实完整 Decoder forward；只将 Attention 替换为指定边界输入。
            expected = self.layer(residual)
            torch.testing.assert_close(self(x, residual), expected, rtol=1e-4, atol=1e-5)
        finally:
            self.layer.self_attn = original


def make_tail_task(model="qwen3", tokens=128):
    torch.manual_seed(42)
    config_cls, layer_cls = MODELS[model]
    config = config_cls(hidden_size=256, intermediate_size=1024, hidden_act="silu",
                        num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=4)
    layer = layer_cls(config, 0).to(device="cuda:0", dtype=torch.float32).eval()
    with torch.no_grad():
        layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    reference = TailReference(layer).eval()
    cases = [tuple(torch.randn(rows, 256, device="cuda:0") * 2 for _ in range(2))
             for rows in (3, tokens)]
    norm = layer.post_attention_layernorm
    context = {
        "module": f"{layer_cls.__module__}.{layer_cls.__name__}",
        "source": inspect.getsource(layer_cls),
        "extracted_source": reference.source,
        "mlp_source": inspect.getsource(type(layer.mlp)),
        "norm_source": inspect.getsource(type(norm)),
        "activation_source": inspect.getsource(type(next(child for child in layer.mlp.children()
                                                       if not isinstance(child, torch.nn.Linear))).forward),
        "external_inputs": ["x", "residual"],
        "boundary": "x 是 Self Attention 的输出，residual 是该 Attention 前的原始残差；执行 extracted_source 中的原始语句。",
        "execution": "eval 模式、FP32、无梯度；所有 Dropout 在此模式下为恒等映射。",
        "config": {"hidden_size": 256, "intermediate_size": 1024, "hidden_act": "silu",
                   "rms_norm_eps": norm.variance_epsilon},
        "inputs": {name: {"shape": list(value.shape), "stride": list(value.stride()),
                          "dtype": str(value.dtype)} for name, value in zip(("x", "residual"), cases[-1])},
        "hardware": {"gpu": torch.cuda.get_device_name(0), "architecture": "gfx908",
                     "torch": torch.__version__, "hip": torch.version.hip},
    }
    return reference, cases, context
