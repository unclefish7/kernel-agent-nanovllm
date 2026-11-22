"""CPU 回归，检查不同输入下的原语轨迹及序列化后的分支证据。"""

import copy
import unittest

from nanovllm_selector.extract.schema import Graph
from nanovllm_selector.extract.shapes import profile_shapes
from nanovllm_selector.test_mixed_state import tiny_mixed_graph


class TestProgramVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = {case: tiny_mixed_graph(*case) for case in [(1, 5), (1, 17), (2, 5), (2, 11)]}
        cls.graph = copy.deepcopy(cls.samples[1, 5])
        profile_shapes(cls.graph, 1, 5, lambda b, s: cls.samples[b, s])
        cls.name = "D.L0.conv.out_proj"

    def test_each_projection_matches_its_own_recording(self):
        self.assertTrue(self.graph.reconcile["ok"], self.graph.reconcile["validate_problems"])
        node = self.graph.node(self.name)
        self.assertIn("program_variants", node.attrs)
        branches = node.attrs["program_variants"]
        self.assertEqual(len(branches), 2)
        for branch, cases, op in zip(branches, ([[1, 5], [1, 17]], [[2, 5], [2, 11]]),
                                     ("aten.bmm.default", "aten.mm.default"), strict=True):
            self.assertEqual(branch["cases"], cases)
            sample = self.samples[tuple(branch["representative_case"])]
            original = sample.node(self.name)
            self.assertEqual(branch["program"], original.attrs["program"])
            self.assertEqual(branch["outputs"], [r.to_dict() for r in original.outputs])
            self.assertIn(op, [step["op"] for step in branch["program"]])
            self.assertEqual(branch["program_replay"]["checked"], len(branch["program"]))
            for tid, meta in branch["tensors"].items():
                self.assertEqual(meta, sample.tensors[tid].to_dict())
            for tid, binding in branch["program_inputs"].items():
                if binding["kind"] == "producer":
                    source = sample.node(binding["node"]).attrs["program"][binding["step"]]
                    self.assertEqual((source["seq"], source["result"]), (binding["seq"], {"tensor": tid}))
        # 不同 B、S 使 view 的尺寸与全局 seq/tid 变化，但没有改变这两处的计算。
        for name in ("L0.conv.in_proj", "L0.conv.conv"):
            self.assertNotIn("program_variants", self.graph.node(name).attrs)
        self.assertEqual(Graph.from_dict(self.graph.to_dict()).validate(), [])

    def test_damaged_branch_cannot_keep_a_green_graph(self):
        def remove_branch(b):
            b.pop()

        def swap_cases(b):
            b[0]["cases"], b[1]["cases"] = b[1]["cases"], b[0]["cases"]

        def change_op(b):
            b[1]["program"][2]["op"] = "aten.add.Tensor"

        def change_input(b):
            b[1]["program_inputs"].clear()

        def change_layout(b):
            next(iter(b[1]["tensors"].values()))["shape"] = [999]

        def drop_replay(b):
            b[1]["program_replay"]["checked"] -= 1

        for damage in (remove_branch, swap_cases, change_op, change_input, change_layout, drop_replay):
            with self.subTest(damage=damage.__name__):
                graph = Graph.from_dict(copy.deepcopy(self.graph.to_dict()))
                self.assertIn("program_variants", graph.node(self.name).attrs)
                damage(graph.node(self.name).attrs["program_variants"])
                self.assertTrue(any(self.name in error for error in graph.validate()))
