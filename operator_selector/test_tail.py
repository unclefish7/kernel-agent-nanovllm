"""在现有 ROCm 容器执行：python -B -m unittest test_tail -v。"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

import torch

from agent import validate_plan
from candidates import execute, candidate_catalog, manual_plan as mlp_plan, SelectedMLP
from tail_candidates import catalog, functions, manual_plan, SelectedTail
from tail_task import make_tail_task
from task import make_task, project, original_activation


class TailTests(unittest.TestCase):
    def test_vendor_hashes(self):
        root = Path(__file__).resolve().parent / "vendor"
        for name, record in json.loads((root / "manifest.json").read_text()).items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), record["vendored_sha256"])

    def test_reference_and_renamed_multi_outputs(self):
        with torch.no_grad():
            for model in ("qwen3", "phi3"):
                reference, cases, context = make_tail_task(model, 128)
                self.assertFalse(reference.layer.training)
                operators = functions(reference)
                for norm, activation in (("reference", "torch"), ("add_fused", "row")):
                    with self.subTest(model=model, norm=norm):
                        plan = manual_plan(reference, norm, activation)
                        mapping = {"x": "x", "residual": "residual"}
                        for i, step in enumerate(plan["steps"]):
                            step["inputs"] = [mapping[name] for name in step["inputs"]]
                            outputs = step["output"] if isinstance(step["output"], list) else [step["output"]]
                            renamed = [f"tensor_{i}_{j}" for j in range(len(outputs))]
                            mapping.update(zip(outputs, renamed))
                            step["output"] = renamed if isinstance(step["output"], list) else renamed[0]
                        plan["output"] = mapping[plan["output"]]
                        validate_plan(plan, catalog(reference), context["external_inputs"])
                        selected = SelectedTail(reference, plan, operators)
                        for x, residual in cases:
                            reference.check_original(x, residual)
                            torch.testing.assert_close(selected(x, residual), reference(x, residual), rtol=1e-4, atol=1e-5)
                        for name, layer in reference.layer.mlp.named_children():
                            if isinstance(layer, torch.nn.Linear):
                                self.assertIs(operators["mlp." + name], layer)

    def test_reject_bad_plan(self):
        catalog = [{"id": "two", "parameters": ["a", "b"], "returns": 2}]
        valid = {"steps": [{"op": "two", "inputs": ["x", "residual"], "output": ["normalized", "summed"]}],
                 "output": "normalized"}
        validate_plan(valid, catalog, ("x", "residual"))
        for field, value in (("op", "unknown"), ("inputs", ["missing", "x"]),
                             ("inputs", ["x"]), ("output", "only_one"),
                             ("output", ["x", "summed"]), ("output", ["same", "same"])):
            invalid = copy.deepcopy(valid)
            invalid["steps"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_plan(invalid, catalog, ("x", "residual"))

    def test_execute_rejects_wrong_return_count(self):
        with self.assertRaises(ValueError):
            execute([{"op": "wrong", "inputs": [], "output": ["a", "b"]}],
                    {"wrong": lambda: (1,)}, {})

    def test_original_mlp_regression(self):
        with torch.no_grad():
            for model in ("qwen3", "phi3"):
                reference, inputs, _ = make_task(model)
                for fused in (False, True):
                    with self.subTest(model=model, fused=fused):
                        plan = mlp_plan(reference, fused)
                        validate_plan(plan, candidate_catalog(reference))
                        selected = SelectedMLP(reference, plan)
                        for x in inputs:
                            torch.testing.assert_close(selected(x), reference(x), rtol=1e-4, atol=1e-5)
                            projected = project(reference, x)
                            torch.testing.assert_close(selected.activation(projected),
                                                       original_activation(reference, projected), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
