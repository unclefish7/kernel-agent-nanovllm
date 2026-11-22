import json
import os
import types
import unittest
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from nanovllm_selector.port.evaluate import compare_logits, pick_model_class
from nanovllm_selector.port import iterate
from nanovllm_selector.port.iterate import build_feedback, load_dotenv, run_loop, subprocess_evaluate


class TestLoop(unittest.TestCase):
    def test_stops_on_evaluator_or_reference_failure_with_original_report(self):
        for outcome in ("reference_unreliable", "evaluation_error"):
            report = {"ok": False, "stage": "reference", "outcome": outcome, "error": "HF"}
            with TemporaryDirectory() as tmp, self.subTest(outcome=outcome):
                result = run_loop(messages=[], call_llm=lambda m: "x = 1", evaluate=lambda p: report,
                                  out_dir=Path(tmp), max_iters=3)
                self.assertEqual(result["attempt"], 1)
                self.assertEqual(result["report"], report)
                self.assertEqual(result["history"][0]["outcome"], outcome)
                self.assertFalse((Path(tmp) / "attempt_2.py").exists())

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
    def test_diagnostic_reports_verified_first_boundary_and_final_scores(self):
        diagnosis = {"length": 128, "first_difference": {
            "engine_path": "model.layers.0.norm", "engine_phase": "residual",
            "hf_path": "model.layers.0.norm", "hf_phase": "input", "max_abs_diff": 0.015625,
            "different_elements": 19, "elements": 32, "engine_shape": [4, 8], "hf_shape": [1, 4, 8],
            "engine": {"view": "返回残差"}, "hf": {"view": "主输入"}},
            "logits": {"different_elements": 31, "elements": 32, "max_abs_diff": 0.1875}}
        report = {"ok": False, "stage": "compare", "criterion": "exact", "diagnosis": diagnosis}
        feedback = build_feedback(report)
        for detail in ("128", "model.layers.0.norm", "residual", "input", "返回残差", "0.015625", "0.1875"):
            self.assertIn(detail, feedback)
        diagnosis["first_difference"] = {"engine_path": "unconfirmed", "reason": "重跑分数不同"}
        self.assertNotIn("unconfirmed", build_feedback(report))

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


class TestDiagnosticSubprocess(unittest.TestCase):
    def exercise(self, tmp, official, *, length=128, exact=True, failure=None):
        candidate = Path(tmp) / "attempt_1.py"
        candidate.write_text("x = 1")
        stages = []
        def fake_run(cmd, **kwargs):
            module = cmd[cmd.index("-m") + 1]
            stages.append(module.rsplit(".", 1)[-1])
            path = Path(cmd[cmd.index("--report") + 1])
            if stages[-1] == "evaluate":
                path.write_text(json.dumps(official))
                return subprocess.CompletedProcess(cmd, int(not official["ok"]), "", "")
            self.assertEqual(module, "nanovllm_selector.port.diagnose")
            self.assertEqual(cmd[cmd.index("--length") + 1], str(length))
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd, 12)
            if failure == "malformed":
                path.write_text("{")
            elif failure == "invalid_metrics":
                path.write_text(json.dumps({"purpose": "prefill_diagnostic", "length": length,
                                            "logits": {"max_abs_diff": "invalid"}}))
            elif failure == "crash":
                return subprocess.CompletedProcess(cmd, 1, "", "diagnostic failed")
            else:
                path.write_text(json.dumps({"purpose": "prefill_diagnostic", "length": length,
                                            "first_difference": None, "logits": {"max_abs_diff": 0.5}}))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with patch("nanovllm_selector.port.iterate.subprocess.run", side_effect=fake_run):
            report = subprocess_evaluate(model_dir="local", arch="Fake", exact=exact,
                                         diagnose_length=length, work_dir=Path(tmp))(candidate)
        self.assertEqual(json.loads(candidate.with_suffix(".report.json").read_text()), official)
        return report, stages

    def test_separate_diagnosis_keeps_official_verdict_even_if_diagnosis_fails(self):
        official = {"ok": False, "stage": "compare", "criterion": "exact",
                    "outcome": "candidate_failed", "error": "official mismatch"}
        for failure in (None, "timeout", "malformed", "crash", "invalid_metrics"):
            with self.subTest(failure=failure), TemporaryDirectory() as tmp:
                report, stages = self.exercise(tmp, official, failure=failure)
                self.assertEqual(stages, ["evaluate", "diagnose"])
                self.assertEqual({k: report[k] for k in official}, official)
                data = json.loads(Path(report["diagnosis_file"]).read_text())
                self.assertEqual(data, report["diagnosis"])
                self.assertEqual("error" in data, bool(failure))

    def test_only_opted_in_strict_numeric_mismatch_runs_diagnosis(self):
        for ok, stage, outcome, exact, length in [
            (True, "compare", "passed", True, 128),
            (False, "compare", "candidate_failed", False, 128),
            (False, "compare", "candidate_failed", True, None),
            (False, "run", "candidate_failed", True, 128),
            (False, "reference", "reference_unreliable", True, 128),
            (False, "compare", "evaluation_error", True, 128),
        ]:
            with self.subTest(stage=stage, outcome=outcome, length=length), TemporaryDirectory() as tmp:
                report, stages = self.exercise(tmp, {"ok": ok, "stage": stage, "outcome": outcome,
                                                     "criterion": "exact" if exact else "approximate"},
                                               exact=exact, length=length)
                self.assertEqual(stages, ["evaluate"])
                self.assertNotIn("diagnosis", report)

    def test_cli_delivers_diagnostic_feedback_and_retries_candidate_once(self):
        prompts = []
        def llm(messages):
            prompts.append(messages)
            return "x = 1"

        def child(cmd, **kwargs):
            candidate = Path(cmd[cmd.index("--candidate") + 1])
            if cmd[2].endswith(".diagnose"):
                self.assertEqual(cmd[cmd.index("--length") + 1], "7")
                data = {"length": 7, "purpose": "prefill_diagnostic", "logits": {"max_abs_diff": 0.1875},
                        "first_difference": {"engine_path": "model.norm", "engine_phase": "output",
                                             "hf_path": "model.norm", "hf_phase": "output", "max_abs_diff": 0.01,
                                             "different_elements": 1, "elements": 7}}
            else:
                ok = candidate.stem == "attempt_2"
                data = {"ok": ok, "stage": "compare", "criterion": "exact",
                        "outcome": "passed" if ok else "candidate_failed"}
            Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(data))
            return subprocess.CompletedProcess(cmd, int(data.get("ok") is False), "", "")

        with TemporaryDirectory() as tmp, patch.object(iterate, "load_graph", return_value={}), patch.object(
            iterate, "build_messages", return_value=[]
        ), patch.object(iterate, "load_dotenv", return_value=[]), patch.object(iterate, "make_llm", return_value=llm), patch(
            "nanovllm_selector.port.iterate.subprocess.run", side_effect=child
        ):
            args = ["--model", "local", "--graph", "g", "--spec", "s", "--template", "t", "--kernels", "k",
                    "--out-dir", tmp, "--diagnose-length", "7", "--max-iters", "2"]
            self.assertEqual(iterate.main(args), 0)
            feedback = prompts[1][-1]["content"]
            self.assertIn("model.norm", feedback)
            self.assertIn("0.1875", feedback)
            result = json.loads((Path(tmp) / "trace.json").read_text())
            self.assertEqual(result["attempt"], 2)
            self.assertTrue(Path(result["history"][0]["diagnosis_file"]).is_file())
            self.assertNotIn("diagnosis_file", result["history"][1])


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
