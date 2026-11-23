"""CPU 回归：分组路由、动态索引和数学注意力的重放。"""

import unittest
from unittest.mock import patch

import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

from nanovllm_selector.extract.decompose import GraphBuilder, classify_ops
from nanovllm_selector.extract.dispatch_program import PRIMITIVES
from nanovllm_selector.extract.recorder import record_forward
from nanovllm_selector.extract.replay import replay_nodes
from nanovllm_selector.extract.trace_io import cache_tensors, observe_ports


class TestRoutingReplay(unittest.TestCase):
    def test_softmax_alone_is_not_an_attention_and_uncovered_routing_ops_remain_unknown(self):
        self.assertEqual(classify_ops(["aten._safe_softmax.default"], None, None)[0], "safe_softmax")
        ops = ["aten.topk.default", "aten.gather.default", "aten.unimplemented.default"]
        self.assertEqual(classify_ops(ops, None, None)[0], "unknown")

    def test_safe_softmax_handles_only_fully_masked_rows_and_requested_dtype(self):
        self.assertIn("aten._safe_softmax.default", PRIMITIVES)
        x = torch.tensor([[-torch.inf, -torch.inf], [0., -torch.inf], [torch.nan, 1.]])
        out = PRIMITIVES["aten._safe_softmax.default"](x, -1, torch.float64)
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(out[:2].tolist(), [[0., 0.], [1., 0.]])
        self.assertTrue(out[2].isnan().all())

    def test_advanced_index_preserves_full_axes_and_duplicate_choices(self):
        self.assertIn("aten.index.Tensor", PRIMITIVES)
        x = torch.arange(24).reshape(2, 3, 4)
        indices = [torch.tensor([1, 0, 1]), None, torch.tensor([2, 1, 2])]
        self.assertEqual(PRIMITIVES["aten.index.Tensor"](x, indices).tolist(),
                         [[14, 18, 22], [1, 5, 9], [14, 18, 22]])

    def test_prefill_and_decode_replay_all_routing_operands_and_cache_outputs(self):
        config = DeepseekV3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                                  moe_intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                                  qk_rope_head_dim=4, v_head_dim=4, first_k_dense_replace=0, n_routed_experts=8,
                                  num_experts_per_tok=2, n_group=2, topk_group=1, n_shared_experts=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = DeepseekV3ForCausalLM(config).eval()
        past = None
        for ids in (torch.tensor([[1, 2, 3]]), torch.tensor([[4]])):
            recording, out = record_forward(model, ids, past_key_values=past)
            past = out.past_key_values
            builder = GraphBuilder(model, recording, config)
            nodes = builder.build()
            observe_ports(builder, nodes, [out.logits], cache_tensors(past), {"input_ids": ids})
            report = replay_nodes(nodes, builder, config, ids)
            self.assertTrue(report["ok"], report["failures"])
            kinds = {n.op for n in nodes}
            self.assertTrue({"topk_routing", "indexed_dispatch", "indexed_accumulate", "attention"} <= kinds, kinds)
            self.assertNotIn("unknown", kinds)
            router = next(n for n in nodes if n.op == "topk_routing")
            self.assertEqual(len(router.outputs), 2)
            dispatch = next(n for n in nodes if n.op == "indexed_dispatch")
            sources = dispatch.attrs["program_inputs"].values()
            self.assertTrue({r.tensor for r in router.outputs} <=
                            {b["tensor"] for b in sources if b.get("node") == router.id})
            # 值正确但索引错误时，不能通过重复使用已录制的路由结果来过检。
            topk = PRIMITIVES["aten.topk.default"]
            def wrong_indices(*args, **kwargs):
                values, indices = topk(*args, **kwargs)
                return values, indices + 1
            with patch.dict(PRIMITIVES, {"aten.topk.default": wrong_indices}):
                failed = replay_nodes([router], builder, config, ids)
            self.assertFalse(failed["ok"])
            self.assertIn("aten.topk.default", failed["failures"][0]["why"])
            # 即使同时改了导出槽标签和其副本，key/value 也得走入 QK 和 PV 的相应支路。
            attention = next(n for n in nodes if n.op == "attention")
            for ref in attention.outputs:
                ref.role = {"key": "value", "value": "key"}.get(ref.role, ref.role)
            attention.attrs["observed_output_roles"] = [r.role for r in attention.outputs]
            self.assertFalse(replay_nodes([attention], builder, config, ids)["ok"])
