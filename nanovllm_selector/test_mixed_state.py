"""用混合层和改名的缓存容器验证跨步状态发现。"""
import unittest
from types import SimpleNamespace

import torch
from transformers import Lfm2Config, Lfm2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from nanovllm_selector.extract.api import _trace
from nanovllm_selector.extract.trace_io import cache_tensors
from nanovllm_selector.extract.shapes import profile_shapes


def tiny_mixed_graph(batch=1, length=5, *, check=True):
    config = Lfm2Config(vocab_size=67, hidden_size=32, intermediate_size=64,
                        block_multiple_of=8, num_hidden_layers=3, conv_L_cache=4,
                        num_attention_heads=4, num_key_value_heads=2,
                        layer_types=["conv", "full_attention", "conv"])
    model = Lfm2ForCausalLM(config).eval().requires_grad_(False)
    model._extract_meta = {"config_json": config.to_dict()}
    return _trace(model, config, dtype="float32", device="cpu", batch_size=batch, seq_len=length,
                  phases=("prefill", "decode"), collapse_template=True,
                  reconcile_graph=True, replay_graph=check, decode_steps=2)


class TestMixedState(unittest.TestCase):
    def test_batch_dependent_projection_layout(self):
        graph = tiny_mixed_graph(check=False)
        profile_shapes(graph, 1, 5, lambda b, s: tiny_mixed_graph(b, s, check=False))
        item = graph.node("D.L0.conv.out_proj").attrs["shape_profile"]["inputs"][0]
        self.assertEqual({len(v["shapes"][0]) for v in item["alternatives"]}, {2, 3})
        self.assertEqual(graph.node("L0.conv.conv").attrs["shape_profile"]["outputs"][0]["axes"][-1],
                         "S+3")
        self.assertFalse(any("维数变化" in p for p in graph.env["shape_probes"]["problems"]))
        item["alternatives"][0]["shapes"][0] = [19]
        self.assertTrue(any("条件形状" in p for p in graph.validate()))

    def test_named_leaves_and_shared_views(self):
        window = torch.randn(1, 7, 4)
        cache = SimpleNamespace(custom=[{"recent": window, "same": window.view_as(window),
                                         "accumulator": torch.zeros(1, 7)}])
        slots = cache_tensors(cache)
        self.assertEqual(len(slots), 2)
        self.assertEqual({v.numel() for v in slots.values()}, {7, 28})

    def test_cache_can_mix_instance_fields_and_inherited_slots(self):
        class Base:
            __slots__ = ("previous",)

        class Extended(Base):
            pass

        cache = Extended()
        cache.previous = torch.ones(2)
        cache.current = torch.zeros(2)
        cache.parent = cache
        self.assertEqual({key.path for key in cache_tensors(cache)}, {"previous", "current"})

    def test_two_distinct_cached_slices_of_one_operation(self):
        class Slices(torch.nn.Module):
            def forward(self, input_ids, past_key_values=None, use_cache=True, cache_position=None):
                values = input_ids.float()
                if past_key_values:
                    values = values + past_key_values.left.sum() + past_key_values.right.sum()
                packed = torch.stack([values[:, -1], values[:, 0]], -1)
                cache = SimpleNamespace(left=packed[:, :1], right=packed[:, 1:])
                return CausalLMOutputWithPast(logits=packed, past_key_values=cache)

        config = SimpleNamespace(vocab_size=17, num_hidden_layers=1)
        graph = _trace(Slices(), config, dtype="float32", device="cpu", batch_size=2, seq_len=4,
                       phases=("prefill", "decode"), collapse_template=False, decode_steps=2,
                       reconcile_graph=False, replay_graph=False)
        self.assertFalse(graph.conflicts, graph.conflicts)
        self.assertEqual(graph.validate(), [])
        slots = [s for s in graph.env["state_slots"] if s["phase"] == "decode"]
        self.assertEqual(sorted(s["shape"] for s in slots), [[2, 1], [2, 1]])
        self.assertEqual(len({s["source_slot"] for s in slots}), 2)

    def test_model_owned_history_is_restored_between_input_probes(self):
        class Running(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("accumulated", torch.zeros(1))
                self.register_buffer("previous", torch.zeros(1))

            def forward(self, input_ids, use_cache=True, cache_position=None):
                value = self.previous + self.accumulated + input_ids.float().sum()
                self.accumulated.copy_(value)
                self.previous = value + 1
                return CausalLMOutputWithPast(logits=value.reshape(1, 1, 1))

        model = Running()
        config = SimpleNamespace(vocab_size=23, num_hidden_layers=1)
        for _ in range(2):
            graph = _trace(model, config, dtype="float32", device="cpu", batch_size=1, seq_len=5,
                           phases=("prefill", "decode"), collapse_template=False, decode_steps=2,
                           reconcile_graph=False, replay_graph=False)
            self.assertEqual({s["path"] for s in graph.env["state_slots"]},
                             {"buffers.accumulated", "buffers.previous"})
            self.assertEqual(model.accumulated.item(), 0)
            self.assertEqual(model.previous.item(), 0)

    def test_only_active_layer_state_flows_across_steps(self):
        graph = tiny_mixed_graph(check=False)
        self.assertEqual(graph.validate(), [])
        for phase in ("decode", "decode_2"):
            slots = [s for s in graph.env["state_slots"] if s["phase"] == phase]
            self.assertEqual(len(slots), 4)
            windows = [s for s in slots if s["shape"] == [1, 32, 4]]
            self.assertEqual({s["layer"] for s in windows}, {0, 2})
            self.assertTrue(all(s["update"]["retained_columns"] == 3 for s in windows))
            self.assertTrue(all(s["readers"] for s in slots))

    def test_two_renamed_states_in_one_layer_and_missing_edge(self):
        class Running(torch.nn.Module):
            def forward(self, input_ids, use_cache=True, past_key_values=None, cache_position=None):
                x = input_ids.float().unsqueeze(-1)
                data = past_key_values
                if data is None:
                    data = SimpleNamespace(bank=[{"recent": x.new_zeros(x.shape[0], 3, 1),
                                                  "total": x.new_zeros(x.shape[0], 1)}])
                layer = data.bank[0]
                recent = torch.cat([layer["recent"], x], dim=1)[:, -3:]
                total = layer["total"] + x.sum(1)
                layer["total"].copy_(total)
                layer["recent"] = recent
                return CausalLMOutputWithPast(logits=recent + total[:, None], past_key_values=data)

        config = SimpleNamespace(vocab_size=19, hidden_size=1, num_hidden_layers=1)
        graph = _trace(Running(), config, dtype="float32", device="cpu", batch_size=2,
                       seq_len=5, phases=("prefill", "decode"), collapse_template=False,
                       reconcile_graph=False, replay_graph=False, decode_steps=2)
        self.assertEqual(graph.validate(), [])
        records = graph.env["state_slots"]
        self.assertEqual({s["path"] for s in records}, {"bank[0].total", "bank[0].recent"})
        self.assertTrue(all(s["readers"] for s in records if s["phase"] != "prefill"))
        # 单独改动状态证据也必须能由实际连线检出。
        next(s for s in records if s["phase"] == "decode")["readers"] = ["nonexistent"]
        self.assertTrue(graph.validate())
