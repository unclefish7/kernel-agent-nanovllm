"""单请求逐模块诊断的 CPU 测试。"""

import unittest

import torch
from torch import nn
from nanovllm_selector.port import diagnose as d


class Norm(nn.Module):
    def __init__(self, damage=False):
        super().__init__()
        self.damage = damage

    def forward(self, x, residual=None):
        if residual is None:
            return x * 2
        combined = x + residual
        return combined * 2, combined + int(self.damage)


class Block(nn.Module):
    def __init__(self, damage):
        super().__init__()
        self.ffn_norm = Norm(damage)

    def forward(self, hidden_states, residual=None):
        if residual is None:
            return self.ffn_norm(hidden_states) + hidden_states
        return self.ffn_norm(hidden_states, residual)


class Toy(nn.Module):
    def __init__(self, engine=False, damage=False):
        super().__init__()
        self.layers = nn.ModuleList([Block(damage)])
        self.engine = engine
        if engine:
            self.packed = nn.Linear(4, 5, bias=False)
        else:
            self.left = nn.Linear(4, 2, bias=False)
            self.right = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        if self.engine:
            delta, residual = self.layers[0](x - 1, torch.ones_like(x))
            return self.packed(delta + residual)
        x = self.layers[0](x)
        return torch.cat([self.left(x), self.right(x)], dim=-1)


class TestDiagnosis(unittest.TestCase):
    def test_head_projects_all_hf_positions_and_only_last_engine_position(self):
        class LastTokenHead(nn.Module):
            def forward(self, x):
                return x[-1:]

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        ref, engine = nn.Module(), nn.Module()
        ref.lm_head, engine.lm_head = nn.Identity(), LastTokenHead()
        report = d.diagnose_runs(lambda sink: d.trace(ref, lambda: ref.lm_head(x[None])[:, -1:], "hf", sink),
                                 lambda sink: d.trace(engine, lambda: engine.lm_head(x), "engine", sink), {})
        self.assertIsNone(report["first_difference"])
        self.assertEqual(report["logits"]["different_elements"], 0)

    def test_actual_execution_paths_packed_slices_and_fused_residual(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        for damage in (None, "residual", "packed"):
            with self.subTest(damage=damage):
                ref, engine = Toy(), Toy(True, damage == "residual")
                with torch.no_grad():
                    ref.left.weight.fill_(1)
                    ref.right.weight.fill_(2)
                    engine.packed.weight.copy_(torch.cat([ref.left.weight, ref.right.weight]))
                    if damage == "packed":
                        engine.packed.weight[-1, 0] += 1
                report = d.diagnose_runs(lambda sink: d.trace(ref, lambda: ref(x[None]), "hf", sink),
                                         lambda sink: d.trace(engine, lambda: engine(x), "engine", sink),
                                         {"left": ("packed", 0), "right": ("packed", 1)})
                self.assertGreater(report["compared"], 5)
                bad = report["first_difference"]
                if damage is None:
                    self.assertIsNone(bad)
                elif damage == "residual":
                    self.assertEqual((bad["engine_path"], bad["hf_path"]), ("layers.0.ffn_norm", "layers.0.ffn_norm"))
                    self.assertEqual((bad["engine_phase"], bad["hf_phase"]), ("residual", "input"))
                    self.assertEqual(bad["max_abs_diff"], 1)
                    self.assertEqual(bad["engine_shape"], [3, 4])
                    self.assertEqual(bad["hf_shape"], [1, 3, 4])
                else:
                    self.assertEqual((bad["engine_path"], bad["hf_path"]), ("packed", "right"))
                    self.assertEqual(bad["engine_phase"], "output")
                    self.assertEqual(bad["max_abs_diff"], 24)
