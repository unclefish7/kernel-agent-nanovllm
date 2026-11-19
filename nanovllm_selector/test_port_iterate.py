import json
import os
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from nanovllm_selector.port.evaluate import compare_logits, pick_model_class
from nanovllm_selector.port.iterate import build_feedback, load_dotenv, run_loop


class TestLoop(unittest.TestCase):
    def test_retry_until_success(self):
        calls = {"n": 0}

        def fake_llm(messages):
            calls["n"] += 1
            return f"```python\n# 第 {calls['n']} 版\nx = {calls['n']}\n```"

        def fake_eval(path):
            if calls["n"] < 3:
                return {"ok": False, "stage": "compare", "error": "logits 不一致"}
            return {"ok": True, "stage": "compare", "comparison": {"cosine": 1.0}}

        with TemporaryDirectory() as tmp:
            result = run_loop(
                messages=[{"role": "user", "content": "写代码"}],
                call_llm=fake_llm,
                evaluate=fake_eval,
                out_dir=Path(tmp),
                max_iters=5,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["attempt"], 3)
            self.assertEqual(len(result["history"]), 3)
            self.assertEqual(Path(result["path"]).name, "attempt_3.py")
            text = Path(result["path"]).read_text()
            self.assertNotIn("```", text)
            self.assertIn("x = 3", text)

    def test_exhausts_and_keeps_every_attempt(self):
        def fake_llm(messages):
            return "y = 1"

        def fake_eval(path):
            return {"ok": False, "stage": "run", "error": "CUDA out of memory"}

        with TemporaryDirectory() as tmp:
            result = run_loop(
                messages=[], call_llm=fake_llm, evaluate=fake_eval,
                out_dir=Path(tmp), max_iters=2,
            )
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["history"]), 2)
            self.assertTrue((Path(tmp) / "attempt_1.py").exists())
            self.assertTrue((Path(tmp) / "attempt_2.py").exists())

    def test_messages_grow_with_feedback(self):
        seen = []

        def fake_llm(messages):
            seen.append(len(messages))
            return "z = 1"

        with TemporaryDirectory() as tmp:
            run_loop(
                messages=[{"role": "user", "content": "写"}],
                call_llm=fake_llm,
                evaluate=lambda p: {"ok": False, "stage": "class", "error": "没有模型类"},
                out_dir=Path(tmp), max_iters=3,
            )
        self.assertEqual(seen, [1, 3, 5])

    def test_rejects_bad_max_iters(self):
        with self.assertRaises(ValueError):
            run_loop(messages=[], call_llm=lambda m: "", evaluate=lambda p: {},
                     out_dir=Path("/tmp"), max_iters=0)


class TestFeedback(unittest.TestCase):
    def test_mentions_stage_and_error(self):
        text = build_feedback({"stage": "run", "error": "KeyError: 'q_proj.weight'"})
        self.assertIn("在引擎里构造", text)
        self.assertIn("q_proj.weight", text)
        self.assertIn("完整文件", text)

    def test_includes_numbers_when_compare_failed(self):
        text = build_feedback({
            "stage": "compare",
            "comparison": {"argmax_actual": 1, "argmax_expected": 2,
                           "max_abs_diff": 3.5, "cosine": 0.5},
        })
        self.assertIn("最大绝对误差 = 3.5000", text)
        self.assertIn("余弦相似度 = 0.500000", text)

    def test_includes_decode_steps(self):
        text = build_feedback({
            "stage": "compare",
            "comparisons": [{"length": 128, "argmax_actual": 1, "argmax_expected": 1,
                             "max_abs_diff": 0.1, "cosine": 0.9999, "ok": True}],
            "decode": {
                "steps": 2, "matched": 1, "ok": False,
                "items": [
                    {"step": 1, "argmax_actual": 5, "argmax_expected": 5,
                     "max_abs_diff": 0.2, "cosine": 0.9999, "ok": True},
                    {"step": 2, "argmax_actual": 7, "argmax_expected": 8,
                     "max_abs_diff": 1.5, "cosine": 0.9, "ok": False},
                ],
            },
        })
        self.assertIn("逐 token 生成", text)
        self.assertIn("1/2 步通过", text)
        self.assertIn("第 2 步", text)
        self.assertIn("余弦 0.900000", text)


class TestDotenv(unittest.TestCase):
    def test_sets_missing_and_keeps_existing(self):
        with TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text('NEW_KEY_ABC=hello\nQUOTED_KEY="v1"\n# 注释\n')
            os.environ.pop("NEW_KEY_ABC", None)
            os.environ["QUOTED_KEY"] = "already"
            loaded = load_dotenv(env)
            self.assertEqual(os.environ["NEW_KEY_ABC"], "hello")
            self.assertEqual(os.environ["QUOTED_KEY"], "already")
            self.assertIn("NEW_KEY_ABC", loaded)
            self.assertNotIn("QUOTED_KEY", loaded)


class TestEvaluateHelpers(unittest.TestCase):
    def test_exact_detects_one_changed_score_with_unchanged_top_token(self):
        scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        changed = scores.clone()
        changed[0] = torch.nextafter(scores[0], scores[1])
        self.assertTrue(compare_logits(scores, changed)["ok"])
        self.assertFalse(compare_logits(scores, changed, exact=True)["ok"])
        self.assertTrue(compare_logits(scores, scores, exact=True)["ok"])
        self.assertFalse(compare_logits(scores[None, :], scores, exact=True)["ok"])

    def test_compare_passes_on_identical(self):
        x = torch.randn(100)
        report = compare_logits(x, x.clone())
        self.assertTrue(report["ok"])
        self.assertTrue(report["argmax_match"])
        self.assertAlmostEqual(report["cosine"], 1.0, places=5)

    def test_compare_fails_on_argmax_mismatch(self):
        a = torch.zeros(100)
        b = torch.zeros(100)
        a[3] = 10.0
        b[7] = 10.0
        report = compare_logits(a, b)
        self.assertFalse(report["ok"])
        self.assertFalse(report["argmax_match"])

    def test_compare_reports_shape_mismatch(self):
        report = compare_logits(torch.randn(10), torch.randn(11))
        self.assertFalse(report["ok"])
        self.assertIn("形状不一致", report["reason"])

    def test_pick_model_class_prefers_local(self):
        module = types.ModuleType("fake_candidate")
        exec(
            "from types import SimpleNamespace\n"
            "class BaseForCausalLM: pass\n"
            "class FakeForCausalLM(BaseForCausalLM): pass\n",
            module.__dict__,
        )
        chosen = pick_model_class(module)
        self.assertEqual(chosen.__name__, "FakeForCausalLM")

    def test_pick_model_class_respects_arch(self):
        module = types.ModuleType("fake2")
        exec("class AlphaForCausalLM: pass\nclass BetaForCausalLM: pass\n", module.__dict__)
        self.assertEqual(pick_model_class(module, "BetaForCausalLM").__name__,
                         "BetaForCausalLM")

    def test_pick_model_class_raises_when_absent(self):
        module = types.ModuleType("fake3")
        with self.assertRaises(LookupError):
            pick_model_class(module)


if __name__ == "__main__":
    unittest.main()
