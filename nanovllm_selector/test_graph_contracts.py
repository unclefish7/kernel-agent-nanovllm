"""用独立录制的小 Llama 检查静默错误，故障只注入内存副本。"""

import copy
import unittest
from unittest.mock import patch

from transformers import LlamaConfig, LlamaForCausalLM

from nanovllm_selector.extract.api import _trace
from nanovllm_selector.extract.replay import IMPL
from nanovllm_selector.extract.schema import Graph
from nanovllm_selector.extract.shapes import profile_shapes


class TestGraphContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = LlamaConfig(vocab_size=71, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
        cls.model = LlamaForCausalLM(cls.config).eval().requires_grad_(False)
        cls.model._extract_meta = {"config_json": cls.config.to_dict()}
        cls.graph = cls.trace(1, 5)
        profile_shapes(cls.graph, 1, 5, cls.trace)

    @classmethod
    def trace(cls, batch, length):
        return _trace(cls.model, cls.config, dtype="float32", device="cpu", batch_size=batch,
                      seq_len=length, phases=("prefill", "decode"), collapse_template=True,
                      reconcile_graph=True, replay_graph=True, decode_steps=2)

    def test_all_cache_reads_and_positions_across_two_steps(self):
        graph = Graph.from_json(self.graph.to_json())
        self.assertTrue(graph.reconcile["ok"], graph.reconcile)
        self.assertEqual(graph.validate(), [])
        for phase in graph.phases:
            returned, = graph.graph_outputs[phase]
            self.assertEqual(graph.node(returned["node"]).op, "lm_head")
            self.assertEqual(graph.tensors[returned["tensor"]].shape[-1], 71)
        for start, end in [("", "D."), ("D.", "D2.")]:
            reads = [e for e in graph.edges if e.to_node == end + "L*.self_attn" and e.kind == "state"]
            self.assertEqual({e.role for e in reads}, {"key", "value"})
            self.assertEqual(len(reads), 2)
            for edge in reads:
                producer = graph.node(edge.from_node)
                self.assertEqual(producer.id, start + "L*.self_attn")
                self.assertEqual(producer.instances, [0, 1])
                self.assertEqual(producer.outputs[edge.source_slot].role, edge.role)
            self.assertIn(end + "input.cache_position", graph.graph_inputs)
            self.assertIn(end + "input.input_ids", graph.graph_inputs)

    def test_second_output_values_are_independently_compared(self):
        original = IMPL["rope_table"]

        def broken(*args):
            cos, sin = original(*args)
            return cos, sin * 0

        with patch.dict(IMPL, rope_table=broken):
            graph = self.trace(1, 5)
        self.assertFalse(graph.reconcile["ok"])
        failures = graph.reconcile["replay"]["failures"]
        self.assertTrue(any(f["node"] == "rotary_emb" and f["slot"] == 1 for f in failures), failures)

    def test_missing_or_swapped_outputs_fail_even_with_equal_shapes(self):
        for change in ("drop", "swap"):
            with self.subTest(change=change):
                graph = copy.deepcopy(self.graph)
                node = graph.node("rotary_emb")
                if change == "drop":
                    node.outputs.pop()
                else:
                    node.outputs.reverse()
                self.assertTrue(any("输出槽位" in p for p in graph.validate()))

    def test_missing_or_swapped_cache_edges_fail(self):
        for change in ("drop", "swap", "missing_metadata"):
            with self.subTest(change=change):
                graph = copy.deepcopy(self.graph)
                k, v = sorted((e for e in graph.edges if e.kind == "state" and
                               e.to_node == "D2.L*.self_attn"), key=lambda e: e.role)
                if change == "drop":
                    graph.edges.remove(k)
                elif change == "swap":
                    k.source_slot, v.source_slot = v.source_slot, k.source_slot
                else:
                    del graph.tensors[k.tensor]
                self.assertTrue(graph.validate())

    def test_history_axis_and_holdout(self):
        graph = copy.deepcopy(self.graph)
        node = graph.node("D2.L*.self_attn")
        key = next(e for e in graph.edges if e.to_node == node.id and e.role == "key")
        profile = node.attrs["shape_profile"]["inputs"][key.target_slot]
        self.assertEqual(profile["axes"], ["B", "num_key_value_heads", "P", "head_dim"])
        self.assertEqual(profile["samples"][-1], [2, 2, 12, 8])
        profile["samples"][-1][2] = 11
        self.assertTrue(any("轴 2 的 P 与实测不符" in p for p in graph.validate()))

    def test_independent_sample_can_reject_fitted_rule(self):
        graph = self.trace(1, 5)

        def corrupt_holdout(batch, length):
            sample = self.trace(batch, length)
            if (batch, length) == (2, 11):
                ref = sample.node("D2.L*.self_attn").outputs[0]
                sample.tensors[ref.tensor].shape[2] += 1
            return sample

        profile_shapes(graph, 1, 5, corrupt_holdout)
        self.assertFalse(graph.reconcile["ok"])
        self.assertTrue(any("P+1" in problem for problem in graph.env["shape_probes"]["problems"]))

    def test_unimplemented_operation_does_not_count_as_passing(self):
        with patch.dict(IMPL):
            del IMPL["rope_table"]
            graph = self.trace(1, 5)
        self.assertFalse(graph.reconcile["ok"])
        self.assertIn("D.rotary_emb", graph.reconcile["replay"]["skipped"])
