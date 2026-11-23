"""按真实前向调用核对稀疏路由、共享投影及函数式矩阵乘。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from nanovllm_selector.extract import api
from nanovllm_selector.extract.decompose import GraphBuilder
from nanovllm_selector.extract.reconcile import reconcile
from nanovllm_selector.extract.recorder import Recorder


class Projection(nn.Linear):
    pass


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))

    def forward(self, x):
        return F.linear(x, self.weight)


class RoutedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 4)
        self.experts = nn.ModuleList(Projection(4, 4) for _ in range(6))
        self.gate = Router()
        self.head = nn.Linear(4, 11)
        self.register_buffer("last", torch.zeros(1, 1, 4))

    def forward(self, input_ids, cache_position=None, use_cache=True):
        repetitions = 1 if cache_position is None else int(cache_position[0]) - 1
        x = self.embed(input_ids) + self.last
        for _ in range(repetitions):
            x = self.experts[0](x)
        y = self.gate(x)
        self.last.copy_(y[:, -1:])
        return CausalLMOutputWithPast(logits=self.head(y))


def trace():
    return api._trace(RoutedModel(), SimpleNamespace(vocab_size=11), dtype="float32", device="cpu",
                      batch_size=1, seq_len=3, phases=("prefill", "decode"), collapse_template=False,
                      reconcile_graph=True, replay_graph=False, decode_steps=2)


def matmul_checks(graph):
    return {c["name"]: (c["expected"], c["actual"], c["ok"]) for c in graph.reconcile["checks"]
            if c["name"].startswith("matmul[")}


class TestReconcileCalls(unittest.TestCase):
    def test_projection_inside_composite_still_has_an_independent_witness(self):
        class Scores(Router):
            def forward(self, x):
                return torch.ops.aten._safe_softmax.default(super().forward(x), -1)

        model = Scores()
        config = SimpleNamespace()
        x = torch.randn(1, 4)
        with torch.no_grad(), Recorder(model) as rec:
            model(x)
        nodes = GraphBuilder(model, rec, config).build()
        self.assertTrue(reconcile(nodes, nodes, [], model, config, rec=rec)["ok"])
        report = reconcile([], [], [], model, config, rec=rec)
        self.assertFalse(report["ok"], report)

    def test_unused_experts_and_repeated_calls_are_counted_per_phase(self):
        self.assertEqual(matmul_checks(trace()), {
            "matmul[prefill]": (3, 3, True), "matmul[decode]": (4, 4, True),
            "matmul[decode_2]": (5, 5, True),
        })

    def test_missing_functional_projection_and_entire_phase_still_fail(self):
        def omit(nodes, *args, **kwargs):
            kept = [n for n in nodes if n.phase != "decode_2" and
                    not (n.phase == "decode" and n.attrs["module_path"] == "gate")]
            return reconcile(kept, *args, **kwargs)

        with patch.object(api, "reconcile", omit):
            checks = matmul_checks(trace())
        self.assertEqual(checks["matmul[decode]"], (4, 3, False))
        self.assertEqual(checks["matmul[decode_2]"], (5, 0, False))

    def test_duplicate_call_cannot_mask_missing_call_to_same_linear(self):
        def substitute(nodes, *args, **kwargs):
            first, second = [n for n in nodes if n.phase == "decode" and n.attrs["module_path"] == "experts.0"]
            return reconcile([first if n is second else n for n in nodes], *args, **kwargs)

        with patch.object(api, "reconcile", substitute):
            self.assertEqual(matmul_checks(trace())["matmul[decode]"], (4, 4, False))
