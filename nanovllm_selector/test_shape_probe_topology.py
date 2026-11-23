"""用可控 token 驱动真实 CPU 轨迹，验证跨样本拓扑的适用范围。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from nanovllm_selector.extract.api import _trace
from nanovllm_selector.extract.schema import Graph
from nanovllm_selector.extract.shapes import profile_shapes


class Routed(torch.nn.Module):
    def __init__(self, both=False):
        super().__init__()
        self.experts = torch.nn.ModuleList([torch.nn.SiLU(), torch.nn.SiLU()])
        self.both = both

    def forward(self, input_ids, use_cache=True):
        selected = input_ids[0, 0].item()
        x = input_ids.float()
        if self.both:
            values = [expert(x) for expert in self.experts]
            logits = (values[selected], values[1 - selected])
        else:
            logits = self.experts[selected](x)
        return CausalLMOutputWithPast(logits=logits)


def trace(model, batch, length, token):
    # 仅替换抽取器的输入生成器；选择、模块调用、输出槽与配方全部实际录制。
    ids = torch.full((batch, length), token, dtype=torch.long)
    with patch("nanovllm_selector.extract.api.torch.randint", return_value=ids):
        return _trace(model, SimpleNamespace(vocab_size=2, num_hidden_layers=1),
                      dtype="float32", device="cpu", batch_size=batch, seq_len=length,
                      phases=("prefill",), collapse_template=False, reconcile_graph=False,
                      replay_graph=False, decode_steps=1)


class TestShapeProbeTopology(unittest.TestCase):
    def test_constant_router_does_not_certify_unexecuted_experts(self):
        cfg = DeepseekV3Config(vocab_size=23, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                               kv_lora_rank=4, q_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4,
                               v_head_dim=4, first_k_dense_replace=0, n_routed_experts=8,
                               num_experts_per_tok=2, n_group=2, topk_group=1,
                               moe_intermediate_size=8, n_shared_experts=1)
        model = DeepseekV3ForCausalLM(cfg).eval().requires_grad_(False)
        router = model.model.layers[0].mlp.gate
        router.weight.zero_()
        router.e_score_correction_bias.copy_(torch.arange(8))

        def record(b, s):
            return _trace(model, cfg, dtype="float32", device="cpu", batch_size=b, seq_len=s,
                          phases=("prefill",), collapse_template=False, reconcile_graph=False,
                          replay_graph=False, decode_steps=1)

        base = record(1, 5)
        self.assertTrue(base.reconcile["ok"], base.reconcile)
        profile_shapes(base, 1, 5, record)
        self.assertFalse(base.reconcile["ok"])
        self.assertFalse(base.env["shape_probes"]["topology_differences"])
        self.assertTrue(any("model.layers.0.mlp.experts" in p and "2/8" in p for p in base.validate()))

    def test_changed_expert_set_keeps_each_observation_separate(self):
        model = Routed()
        base = trace(model, 1, 5, 0)
        self.assertIsNotNone(base.node("experts.0"))
        self.assertIsNotNone(trace(model, 1, 5, 1).node("experts.1"))
        profile_shapes(base, 1, 5, lambda b, s: trace(model, b, s, int((b, s) == (2, 11))))
        self.assertFalse(base.reconcile["ok"])
        self.assertFalse(any("shape_profile" in n.attrs or "program_variants" in n.attrs for n in base.nodes))
        changes, = base.env["shape_probes"]["topology_differences"]
        self.assertEqual(changes["case"], [2, 11])
        self.assertIn("experts.1", changes["added"])
        self.assertIn("experts.0", changes["missing"])
        self.assertTrue(any("拓扑" in p for p in Graph.from_json(base.to_json()).validate()))

    def test_changed_return_order_fails_even_with_identical_executed_nodes(self):
        model = Routed(both=True)
        base = trace(model, 1, 5, 0)
        other = trace(model, 1, 5, 1)
        self.assertEqual([n.id for n in base.nodes], [n.id for n in other.nodes])
        self.assertEqual([r["node"] for r in base.graph_outputs["prefill"]], ["experts.0", "experts.1"])
        self.assertEqual([r["node"] for r in other.graph_outputs["prefill"]], ["experts.1", "experts.0"])
        profile_shapes(base, 1, 5, lambda b, s: trace(model, b, s, int((b, s) == (2, 11))))
        self.assertFalse(base.reconcile["ok"])
        changes, = base.env["shape_probes"]["topology_differences"]
        self.assertEqual(changes["changed"], ["experts.0", "experts.1"])

    def test_same_selection_across_shapes_is_scoped_to_observed_inputs(self):
        model = Routed()
        base = trace(model, 1, 5, 0)
        profile_shapes(base, 1, 5, lambda b, s: trace(model, b, s, 0))
        self.assertTrue(base.reconcile["ok"], base.reconcile)
        self.assertEqual(base.env["shape_probes"].get("topology_scope"), "observed_inputs_only")
        self.assertIsNotNone(trace(model, 1, 5, 1).node("experts.1"))
