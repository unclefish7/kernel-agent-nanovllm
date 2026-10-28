"""接口与安装测试，不代表完整模型的数值或性能验收。"""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F

from agent import select
from candidates import SelectedActivation, install, restore, validate_plan
from vendor.swiglu_fwd import _swiglu_fwd


def plan(candidate="native"):
    return {"schema_version": 1, "binding": {},
            "choice": {"target": "mlp_activation", "candidate_id": candidate,
                       "reason": "测试接口接入"}}


class PlainActivation(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


def tiny_model():
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.proj = nn.Linear(4, 8)
        layer.mlp.act_fn = PlainActivation()
        root.model.layers.append(layer)
    return root


class SelectorTest(unittest.TestCase):
    def test_reject_invalid_decision(self):
        for change in [{"candidate_id": "arbitrary_code"}, {"candidate_id": []}, {"target": "attention"},
                       {"reason": ""}, {"code": "print('not executed')"}]:
            value = plan()
            value["choice"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_plan(value, {})
        value = plan()
        value["binding"] = {"different_model": True}
        with self.assertRaises(ValueError):
            validate_plan(value, {})
        for value in [None, [], {}, {"schema_version": True}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_plan(value, {})

    def test_install_preserves_weights_and_restore_identity(self):
        model = tiny_model()
        originals = [layer.mlp.act_fn for layer in model.model.layers]
        before = [(name, id(p)) for name, p in model.named_parameters()]
        keys = list(model.state_dict())
        installed = install(model, plan(), {})
        self.assertEqual(before, [(name, id(p)) for name, p in model.named_parameters()])
        self.assertEqual(keys, list(model.state_dict()))
        x = torch.randn(3, 8)
        torch.testing.assert_close(model.model.layers[0].mlp.act_fn(x), originals[0](x))
        self.assertEqual(installed[0][2].calls["warmup/native"], 1)
        with self.assertRaises(ValueError):
            install(model, plan(), {})
        restore(installed)
        self.assertTrue(all(layer.mlp.act_fn is original
                            for layer, original in zip(model.model.layers, originals)))

    def test_unsupported_input_records_fallback(self):
        wrapper = SelectedActivation(PlainActivation(), "geak_packed_swiglu")
        x = torch.randn(2, 8)
        torch.testing.assert_close(wrapper(x), wrapper.original(x))
        self.assertEqual(wrapper.calls["warmup/fallback_native"], 1)
        self.assertEqual(sum(wrapper.fallbacks.values()), 1)
        self.assertNotIn("warmup/geak_packed_swiglu", wrapper.calls)

    def test_agent_parses_json_without_recording_secret(self):
        Path("cache/tmp").mkdir(parents=True, exist_ok=True)
        response = Mock()
        response.json.return_value = {"choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps(plan()["choice"])}}]}
        with tempfile.TemporaryDirectory(dir="cache/tmp") as folder, patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "unit-test-not-a-real-key"}
        ), patch("agent.requests.post", return_value=response) as post:
            output_dir = Path(folder)
            actual = select({"binding": {}, "candidates": []}, output_dir)
            self.assertEqual(actual, plan())
            self.assertEqual(post.call_count, 1)
            self.assertNotIn("unit-test-not-a-real-key", (output_dir / "request.json").read_text())
            self.assertEqual(json.loads((output_dir / "plan.json").read_text()), actual)
            bad = copy.deepcopy(response.json.return_value)
            bad["choices"][0]["finish_reason"] = "length"
            response.json.return_value = bad
            with self.assertRaises(ValueError):
                select({"binding": {}}, output_dir)

    @unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "需要 ROCm")
    def test_gpu_interface_and_current_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.ones(2, 6144, device="cuda", dtype=torch.bfloat16)
            original = x.clone()
            output = _swiglu_fwd(x)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertEqual(output.shape, (2, 3072))
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.equal(x, original))
        self.assertNotEqual(output.data_ptr(), x.data_ptr())
        expected = F.silu(x[:, :3072]) * x[:, 3072:]
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
