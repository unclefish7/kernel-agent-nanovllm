"""CPU 上用可手算的小张量核对执行配方、原地写和分支证据。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Lfm2Config, Lfm2ForCausalLM

from nanovllm_selector.extract.decompose import GraphBuilder
from nanovllm_selector.extract.dispatch_program import PRIMITIVES
from nanovllm_selector.extract.recorder import Recorder, record_forward
from nanovllm_selector.extract.reconcile import reconcile
from nanovllm_selector.extract.replay import IMPL, replay_nodes
from nanovllm_selector.extract.trace_io import cache_tensors, observe_ports


CONFIG = SimpleNamespace(hidden_size=4, num_attention_heads=1)


def trace(module, *args):
    with torch.no_grad(), Recorder(module) as recording:
        out = module(*args)
    builder = GraphBuilder(module, recording, CONFIG)
    nodes = builder.build()
    outputs = list(out) if isinstance(out, tuple) else [out]
    observe_ports(builder, nodes, outputs, {}, {f"x{i}": a for i, a in enumerate(args)})
    return builder, nodes


class Window(nn.Module):
    def forward(self, window, position, value):
        if position[0] > 0:
            rolled = window.roll(-1, -1)
            rolled[:, :, position] = value
            window.copy_(rolled)
        return window, window.sum(dim=-1, keepdim=True)


class TestDispatchProgram(unittest.TestCase):
    def test_only_views_preserve_input_and_every_dispatch(self):
        class OnlyViews(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).unbind(0)

        x = torch.arange(6.).reshape(2, 3)
        builder, nodes = trace(OnlyViews(), x)
        node, = nodes
        self.assertEqual([r.tensor for r in node.inputs], [builder.tid(x)])
        self.assertEqual(node.op, "view")
        self.assertEqual(len(node.attrs["program"]), len(builder.rec.dispatches))
        self.assertTrue(replay_nodes(nodes, builder, CONFIG, x)["ok"])

    def test_scalar_repeated_and_nondefault_arange_operands(self):
        class Add(nn.Module):
            def forward(self, x):
                return torch.add(x, 3, alpha=2)

        class Square(nn.Module):
            def forward(self, x):
                return x * x

        class Range(nn.Module):
            def forward(self, x):
                return torch.arange(3, 9, 2, dtype=torch.float32)

        x = torch.tensor([1., 2.])
        for module in (Add(), Square(), Range()):
            with self.subTest(type=type(module).__name__):
                builder, nodes = trace(module, x)
                report = replay_nodes(nodes, builder, CONFIG, x)
                self.assertTrue(report["ok"], report)

    def test_buffer_only_forward_accepts_optional_cache_position(self):
        class Buffered(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cursor", torch.tensor(-1))

            def forward(self, input_ids, use_cache=True, *, cache_position=None):
                if cache_position is not None:
                    self.cursor.copy_(cache_position[0])
                return self.cursor.clone()

        class NoPosition(nn.Module):
            def forward(self, input_ids, use_cache=True, past_key_values=None):
                return input_ids

        ids, position = torch.tensor([[2]]), torch.tensor([7])
        _, out = record_forward(Buffered(), ids, cache_position=position)
        self.assertEqual(out.item(), 7)
        _, out = record_forward(NoPosition(), ids, past_key_values=(), cache_position=position)
        self.assertTrue(torch.equal(out, ids))

    def test_same_values_do_not_hide_broken_view_alias(self):
        class Example(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).sum(dim=1)

        x = torch.arange(6.).reshape(2, 3)
        builder, nodes = trace(Example(), x)
        with patch.dict(PRIMITIVES, {"aten.transpose.int": lambda t, a, b: t.transpose(a, b).clone()}):
            report = replay_nodes(nodes, builder, CONFIG, x)
        self.assertFalse(report["ok"])
        self.assertIn("aten.transpose.int", str(report["failures"]))

    def test_integer_results_are_exact(self):
        class Example(nn.Module):
            def forward(self, x):
                return x.sum(dim=0)

        x = torch.tensor([500000, 700000])
        builder, nodes = trace(Example(), x)
        with patch.dict(PRIMITIVES, {"aten.sum.dim_IntList": lambda x, dim: x.sum(dim) + 1}):
            report = replay_nodes(nodes, builder, CONFIG, x)
        self.assertFalse(report["ok"])

    def test_program_boundary_identifies_prior_producer_constant_and_weight_view(self):
        class Example(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                projected = self.proj(x)
                return projected + torch.tensor([2., 3.])

        x = torch.arange(4.).view(1, 2, 2)
        builder, nodes = trace(Example(), x)
        proj = next(n for n in nodes if n.id == "proj")
        add = next(n for n in nodes if n is not proj)
        self.assertIn("program_inputs", proj.attrs)
        self.assertIn({"kind": "state", "name": "proj.weight"}, proj.attrs["program_inputs"].values())
        sources = list(add.attrs["program_inputs"].values())
        self.assertIn({"kind": "literal", "value": [2., 3.]}, sources)
        w = proj.attrs["program_inputs"][builder.tid(x)]
        self.assertIn(w["tensor"], {r.tensor for r in proj.inputs})
        self.assertEqual(w["kind"], "input")
        self.assertTrue(any(s["kind"] == "input" or s.get("node") == proj.id for s in sources))
        next(s for s in sources if s["kind"] == "literal")["value"][0] = 9.
        report = replay_nodes(nodes, builder, CONFIG, x)
        self.assertFalse(report["ok"])
        self.assertIn(add.id, [f["node"] for f in report["failures"]])

    def test_lfm_attention_validates_each_owned_output(self):
        config = Lfm2Config(vocab_size=53, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, layer_types=["conv", "full_attention"],
                            block_auto_adjust_ff_dim=False)
        model = Lfm2ForCausalLM(config).eval()
        ids = torch.tensor([[2, 5, 7]])
        recording, out = record_forward(model, ids)
        builder = GraphBuilder(model, recording, config)
        nodes = builder.build()
        observe_ports(builder, nodes, [out.logits], cache_tensors(out.past_key_values), {"input_ids": ids})
        attention = next(n for n in nodes if n.op == "attention")
        self.assertEqual({r.role for r in attention.outputs}, {"key", "out"})
        report = replay_nodes(nodes, builder, config, ids)
        self.assertTrue(report["ok"], report)
        original = IMPL["attention"]

        def wrong_key(*args):
            result = original(*args)
            result["key"] = result["key"] + 1
            return result

        with patch.dict(IMPL, attention=wrong_key):
            failure, = replay_nodes([attention], builder, config, ids)["failures"]
        self.assertEqual(attention.outputs[failure["slot"]].role, "key")

    def test_write_before_image_and_executable_branch(self):
        window = torch.tensor([[[3., 5., 7.]]])
        builder, nodes = trace(Window(), window, torch.tensor([2]), torch.tensor([[[11.]]]))
        node, = nodes
        self.assertIn(builder.tid(window), [r.tensor for r in node.inputs])
        self.assertIn("program", node.attrs)
        program = json.loads(json.dumps(node.attrs["program"]))
        by_op = {s["op"]: s for s in program}
        self.assertIs(by_op["aten._local_scalar_dense.default"]["result"], True)
        self.assertEqual(by_op["aten.roll.default"]["args"][1:], [[-1], [-1]])
        self.assertEqual(by_op["aten.copy_.default"]["writes"], [builder.tid(window)])
        copy = next(d for d in builder.rec.dispatches if d.op == "aten.copy_.default")
        self.assertEqual(copy.before[id(window)].tolist(), [[[3., 5., 7.]]])
        window.zero_()
        self.assertEqual(copy.after[id(window)].tolist(), [[[5., 7., 11.]]])
        report = replay_nodes(nodes, builder, CONFIG, window)
        self.assertTrue(report["ok"], report)
        self.assertEqual(node.attrs["program_replay"], dict(ok=True, total=len(program), checked=len(program)))
        self.assertEqual(node.op, "unknown")
        self.assertTrue(reconcile(nodes, nodes, [], builder.model, CONFIG)["ok"])

    def test_corrupt_scalar_or_result_cannot_pass(self):
        def run():
            window = torch.tensor([[[3., 5., 7.]]])
            return trace(Window(), window, torch.tensor([2]), torch.tensor([[[11.]]]))

        for damage in ("branch", "shift", "remove", "output"):
            with self.subTest(damage=damage):
                builder, nodes = run()
                node, = nodes
                self.assertIn("program", node.attrs)
                program = node.attrs["program"]
                if damage == "branch":
                    next(s for s in program if s["op"].startswith("aten._local_scalar"))["result"] = False
                elif damage == "shift":
                    next(s for s in program if s["op"].startswith("aten.roll"))["args"][1] = [1]
                elif damage == "remove":
                    program.pop()
                else:
                    d = next(d for d in builder.rec.dispatches if d.op.startswith("aten.sum"))
                    d.after[id(d.outputs[0])].fill_(19.)
                report = replay_nodes(nodes, builder, CONFIG, torch.ones(1))
                self.assertFalse(report["ok"])
                self.assertTrue(report["failures"])
                self.assertFalse(reconcile(nodes, nodes, [], builder.model, CONFIG)["ok"])

    def test_unsupported_real_op_remains_a_failure_inside_known_cluster(self):
        class Example(nn.Embedding):
            def forward(self, x):
                torch.special.digamma(self.weight)
                return super().forward(x)

        builder, nodes = trace(Example(5, 2), torch.tensor([1, 3]))
        self.assertEqual(nodes[0].op, "embedding")
        coverage = reconcile(nodes, nodes, [], builder.model, CONFIG)
        self.assertFalse(coverage["ok"], coverage)
        self.assertIn("aten.digamma.default", str(coverage["checks"]))
        report = replay_nodes(nodes, builder, CONFIG, torch.ones(1))
        self.assertFalse(report["ok"], report)
        self.assertIn("aten.digamma.default", str(report["failures"]))

    def test_functional_activations_count_occurrences(self):
        class Example(nn.Module):
            def forward(self, x):
                return F.silu(F.silu(x))

        builder, nodes = trace(Example(), torch.tensor([1., 3.]))
        report = reconcile(nodes, nodes, [], builder.model, CONFIG)
        check = next(c for c in report["checks"] if c["name"] == "silu")
        self.assertEqual((check["expected"], check["actual"]), (2, 2))

    def test_convolution_parameters_and_transparent_split(self):
        class Example(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(2, 2, 3, padding=2, dilation=2, groups=2, bias=True)

            def forward(self, x):
                left, right = self.conv(x).chunk(2, dim=1)
                return left + right

        x = torch.arange(18.).reshape(1, 2, 9)
        builder, nodes = trace(Example(), x)
        conv = next(n for n in nodes if n.id == "conv")
        self.assertEqual(conv.op, "conv1d")
        self.assertEqual(conv.attrs["program"][0]["args"][3:], [[1], [2], [2], False, [0], 2])
        self.assertEqual(nodes[-1].op, "add")
        report = replay_nodes(nodes, builder, CONFIG, x)
        self.assertTrue(report["ok"], report)
