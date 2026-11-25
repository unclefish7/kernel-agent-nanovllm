import io
import hashlib
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nanovllm_selector.port.generate import make_chat
from nanovllm_selector.port.explore.session import initial_messages, run_session


CODE = "class ExampleForCausalLM:\n    packed_modules_mapping = {}\n"


def call_message(name, arguments, ident="a"):
    return {"id": ident, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class Runtime:
    def __init__(self, reports):
        self.calls = []
        self.judged = []
        self.reports = iter(reports)

    def call(self, name, args):
        self.calls.append((name, args))
        return {"ok": True, "ref": "model.layers"}

    def evaluate(self, candidate, *, holdout=False):
        self.judged.append((candidate.read_bytes(), holdout))
        return next(self.reports)


class SessionTests(unittest.TestCase):
    def test_public_numeric_mismatch_writes_focused_diagnosis_and_reuses_it_on_resume(self):
        for prefill_error, length in ((True, 256), (False, 768)):
            with self.subTest(prefill=prefill_error), TemporaryDirectory() as root:
                out = Path(root) / "session"
                seed = Path(root) / "previous.py"
                seed.write_text(CODE)
                report = {"stage": "compare", "ok": False, "criterion": "exact", "lengths": [128, 256, 768],
                          "comparisons": [{"length": 128, "ok": True}, {"length": 256, "ok": not prefill_error},
                                          {"length": 768, "ok": not prefill_error}],
                          "decode": {"ok": False, "items": [{"ok": False}]}}
                seen = []
                class Diagnostics(Runtime):
                    def diagnose(self, candidate, length):
                        seen.append((candidate.name, length))
                        return {"purpose": "prefill_diagnostic", "length": length,
                                "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                                "first_difference": {"engine_path": "model.layers.0.test_boundary"},
                                "logits": {"max_abs_diff": 0.125}}
                runtime = Diagnostics([report])
                def interrupted(messages, tools):
                    self.assertTrue(any("model.layers.0.test_boundary" in m["content"] for m in messages))
                    raise OSError("暂停后从文件重放")
                args = dict(messages=[], out_dir=out, max_turns=6, max_attempts=2)
                run_session(chat=interrupted, runtime=runtime, from_candidate=seed, **args)
                self.assertEqual(seen, [("attempt_1.py", length)])
                sidecar = out / "attempt_1.diagnosis.json"
                frozen = sidecar.read_bytes()
                self.assertEqual(json.loads(frozen)["length"], length)
                self.assertEqual(json.loads((out / "attempt_1.report.json").read_bytes()), report)
                def repair(messages, tools):
                    self.assertTrue(any("model.layers.0.test_boundary" in m["content"] for m in messages))
                    return {"role": "assistant", "content": CODE + "# 第二次"}
                result = run_session(chat=repair, runtime=Runtime([{"ok": True}, {"ok": True}]), resume=True, **args)
                self.assertTrue(result["ok"])
                self.assertEqual(sidecar.read_bytes(), frozen)
                self.assertEqual(seen, [("attempt_1.py", length)])

    def test_only_public_numeric_failures_run_diagnosis_and_errors_keep_official_report(self):
        numeric = {"ok": False, "stage": "compare", "generation_prefill": {"ok": False}}
        for reports, count in (([{"ok": True}, numeric], 0),
                               ([{"ok": False, "stage": "engine"}, {"ok": True}, {"ok": True}], 0),
                               ([{"ok": False, "stage": "compare", "error": "前缀计数不符"},
                                 {"ok": True}, {"ok": True}], 0),
                               ([numeric, {"ok": True}, {"ok": True}], 1)):
            with self.subTest(reports=reports), TemporaryDirectory() as root:
                out = Path(root) / "session"
                seed = Path(root) / "old.py"
                seed.write_text(CODE)
                calls = []
                class Unavailable(Runtime):
                    def diagnose(self, candidate, length):
                        calls.append(length)
                        raise OSError("观测失败")
                result = run_session(messages=[], chat=lambda *_: {"role": "assistant", "content": CODE},
                                     runtime=Unavailable(reports), out_dir=out, from_candidate=seed,
                                     max_turns=2, max_attempts=2)
                self.assertEqual(result["ok"], len(reports) == 3)
                self.assertEqual(calls, [128] * count)
                self.assertEqual(json.loads((out / "attempt_1.report.json").read_bytes()), reports[0])
                if count:
                    self.assertIn("观测失败", (out / "attempt_1.diagnosis.json").read_text())
                else:
                    self.assertFalse(list(out.glob("*.diagnosis.json")))

    def test_seed_is_first_frozen_verdict_then_repair_uses_its_actual_failure(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            seed, out = folder / "previous.py", folder / "new"
            original = b"\xef\xbb\xbf" + CODE.replace("\n", "\r\n").encode() + "# 旧候选".encode()
            seed.write_bytes(original)
            runtime = Runtime([{"ok": False, "stage": "engine", "error": "新引擎的实际错误"},
                               {"ok": True}, {"ok": True}])
            def chat(messages, tools):
                self.assertEqual(runtime.judged, [(original, False)])
                self.assertTrue(any("新引擎的实际错误" in (m.get("content") or "") for m in messages))
                return {"role": "assistant", "content": CODE}
            result = run_session(messages=[], chat=chat, runtime=runtime, out_dir=out,
                                 max_attempts=2, max_turns=1, from_candidate=seed)
            self.assertTrue(result["ok"])
            self.assertEqual([item["turn"] for item in result["history"]], [0, 1])
            self.assertEqual((out / "attempt_1.py").read_bytes(), original)
            self.assertEqual((out / "attempt_2.py").read_bytes(), CODE.encode())
            self.assertEqual(runtime.judged, [(original, False), (CODE.encode(), False), (CODE.encode(), True)])
            self.assertEqual(seed.read_bytes(), original)

    def test_cli_seed_passes_both_cases_without_chat_and_resume_verifies_actual_bytes(self):
        from nanovllm_selector.port import iterate

        with TemporaryDirectory() as root:
            seed = Path(root) / "candidate.py"
            seed.write_bytes(CODE.encode())
            out = Path(root) / "session"
            runtime = Runtime([{"ok": True}, {"ok": True}])
            argv = ["--explore", "--model", root, "--out-dir", str(out), "--max-iters", "2"]
            with patch.object(iterate, "load_dotenv", return_value=[]), patch(
                "nanovllm_selector.port.explore.session.initial_messages", return_value=[]
            ), patch("nanovllm_selector.port.explore.runtime.DockerRuntime", return_value=runtime), patch(
                "nanovllm_selector.port.generate.make_chat", side_effect=AssertionError("已验证种子无需请求模型")
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(iterate.main([*argv, "--from-candidate", str(seed)]), 0)
                reports = {p: p.read_bytes() for p in out.glob("attempt_1.*")}
                self.assertEqual(iterate.main([*argv, "--resume"]), 0)
                other = seed.with_name("same.py")
                other.write_bytes(seed.read_bytes())
                self.assertEqual(iterate.main([*argv, "--resume", "--from-candidate", str(other)]), 0)
                other.write_bytes(b"# changed")
                with self.assertRaisesRegex(ValueError, "种子|初始候选"):
                    iterate.main([*argv, "--resume", "--from-candidate", str(other)])
                self.assertEqual(runtime.judged, [(CODE.encode(), False), (CODE.encode(), True)])
                self.assertEqual({p: p.read_bytes() for p in reports}, reports)
                self.assertEqual(json.loads((out / "trace.json").read_bytes())["turns"], 0)

    def test_new_seed_rejects_a_stale_report_and_requires_room_for_revision(self):
        with TemporaryDirectory() as root:
            seed = Path(root) / "p.py"
            seed.write_text(CODE)
            out = Path(root) / "session"
            args = dict(messages=[], chat=None, runtime=Runtime([]), out_dir=out, max_turns=2, from_candidate=seed)
            with self.assertRaisesRegex(ValueError, "2|修订"):
                run_session(max_attempts=1, **args)
            out.mkdir(exist_ok=True)
            stale = out / "attempt_1.report.json"
            stale.write_bytes(b'{"ok":true}')
            with self.assertRaises((FileExistsError, ValueError)):
                run_session(max_attempts=2, **args)
            self.assertEqual(stale.read_bytes(), b'{"ok":true}')
            self.assertFalse((out / "attempt_1.py").exists())

    def test_resume_legacy_history_in_turn_order_and_synthesize_after_old_cap(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            prompt = [{"role": "user", "content": "原始任务"}]
            (out / "prompt.json").write_text(json.dumps(prompt))
            old_code = CODE + "# 原始候选\n"
            (out / "attempt_1.py").write_text(old_code)
            report = {"ok": False, "stage": "compare", "error": "首轮误差"}
            old_report = json.dumps(report)
            (out / "attempt_1.report.json").write_text(old_report)
            events = [{"turn": t, "id": ident, "tool": name, "arguments": args, "result": result}
                      for t, ident, name, args, result in [
                          (1, "real-1", "run_model", {"target": "hf"}, {"run_id": "hf-real"}),
                          (1, "real-2", "read_code", {}, {"source": "x" * 40000, "run_id": "saved-id"}),
                          (3, "real-3", "observe", {"run_id": "hf-real"}, {"value": 42})]]
            original = "".join(json.dumps(e) + "\n" for e in events)
            (out / "tools.jsonl").write_text(original)
            (out / "trace.json").write_text(json.dumps({"stage": "turn_limit", "turns": 3, "history": []}))
            runtime = Runtime([{"ok": False, "stage": "compare"}, {"ok": True}, {"ok": True}])
            responses = iter([{"role": "assistant", "content": CODE + "# 续写"},
                              {"role": "assistant", "tool_calls": [call_message("observe", {"run_id": "hf-real"}, "new")]},
                              {"role": "assistant", "content": CODE + "# 修订"}])
            calls = []
            def chat(messages, tools):
                calls.append((list(messages), tools))
                return next(responses)
            result = run_session(messages=prompt, chat=chat, runtime=runtime, out_dir=out,
                                 max_attempts=4, max_turns=12, resume=True)
            self.assertTrue(result["ok"])
            self.assertEqual((result["turns"], result["tool_calls"], len(result["history"])), (6, 4, 3))
            replay, tools = calls[0]
            self.assertEqual(tools, [])
            self.assertEqual([c["id"] for c in replay[1]["tool_calls"]], ["real-1", "real-2"])
            self.assertEqual([m["tool_call_id"] for m in replay if m["role"] == "tool"], ["real-1", "real-2", "real-3"])
            self.assertTrue(json.loads(replay[3]["content"])["truncated"])
            self.assertEqual(json.loads(replay[3]["content"])["run_id"], "saved-id")
            self.assertLess(len(replay[3]["content"]), 30000)
            self.assertEqual(replay[4]["content"], old_code)
            self.assertIn("首轮误差", replay[5]["content"])
            self.assertEqual(json.loads(replay[6]["tool_calls"][0]["function"]["arguments"]), {"run_id": "hf-real"})
            self.assertEqual(runtime.calls, [("observe", {"run_id": "hf-real"})])
            ledger = (out / "tools.jsonl").read_text()
            self.assertTrue(ledger.startswith(original))
            self.assertEqual(json.loads(ledger.splitlines()[-1])["turn"], 5)
            self.assertEqual((out / "attempt_1.report.json").read_text(), old_report)

    def test_resume_pending_candidate_and_terminal_holdout_without_another_chat(self):
        for existing in ("code", "public", "holdout"):
            with self.subTest(existing=existing), TemporaryDirectory() as directory:
                out = Path(directory)
                code = CODE + "# 已冻结\n"
                (out / "prompt.json").write_text("[]")
                (out / "tools.jsonl").write_text("")
                (out / "attempt_1.py").write_text(code)
                public = '{"ok":true}'
                if existing != "code":
                    (out / "attempt_1.report.json").write_text(public)
                if existing == "holdout":
                    (out / "attempt_1.holdout.json").write_text('{"ok":false,"error":"独立集"}')
                runtime = Runtime(([{"ok": True}] if existing == "code" else []) + [{"ok": False}])
                result = run_session(messages=[], chat=lambda *_: self.fail("已有候选应先验收"),
                                     runtime=runtime, out_dir=out, max_attempts=1, max_turns=1, resume=True)
                self.assertEqual((result["ok"], result["stage"], result["turns"]), (False, "holdout", 1))
                phases = {"code": [False, True], "public": [True], "holdout": []}[existing]
                self.assertEqual(runtime.judged, [(code.encode(), h) for h in phases])
                if existing != "code":
                    self.assertEqual((out / "attempt_1.report.json").read_text(), public)
                result = run_session(messages=[], chat=lambda *_: self.fail("禁止针对独立集修订"),
                                     runtime=Runtime([]), out_dir=out, max_attempts=9, max_turns=20, resume=True)
                self.assertEqual(result["stage"], "holdout")

    def test_resume_checks_prompt_and_digest_before_using_previous_report(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            args = dict(chat=lambda *_: {"role": "assistant", "content": CODE}, runtime=Runtime([{"ok": True}, {"ok": True}]),
                        out_dir=out, max_attempts=2, max_turns=4)
            run_session(messages=[], **args)
            with self.assertRaisesRegex(ValueError, "prompt|提示"):
                run_session(messages=[{"role": "user", "content": "不同"}], resume=True, **args)
            (out / "attempt_1.py").write_text(CODE + "# 已变动\n")
            self.assertEqual(run_session(messages=[], resume=True, **args)["outcome"], "evaluation_error")

    def test_public_and_holdout_check_report_digest_and_candidate_again_after_judging(self):
        for phase in (False, True):
            for damage in ("摘要错配", "验收期间文件改变"):
                with self.subTest(holdout=phase, damage=damage), TemporaryDirectory() as directory:
                    out = Path(directory)
                    runs = []
                    class Judge:
                        def evaluate(self, candidate, *, holdout=False):
                            runs.append(holdout)
                            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                            if holdout == phase:
                                if damage == "摘要错配":
                                    digest = "0" * 64
                                else:
                                    candidate.write_bytes(candidate.read_bytes() + b"# changed\n")
                            return {"ok": True, "candidate_sha256": digest}
                    result = run_session(messages=[], chat=lambda *_: {"role": "assistant", "content": CODE},
                                         runtime=Judge(), out_dir=out, max_attempts=3, max_turns=6)
                    self.assertFalse(result["ok"])
                    self.assertEqual(result["outcome"], "evaluation_error")
                    self.assertEqual(runs, [False, True] if phase else [False])
                    self.assertEqual(len(list(out.glob("attempt_*.py"))), 1)

    def test_resume_rejects_mispaired_public_or_holdout_without_rewriting_reports(self):
        for phase in ("report", "holdout"):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                out = Path(directory)
                (out / "prompt.json").write_text("[]")
                (out / "attempt_1.py").write_text(CODE)
                digest = hashlib.sha256(CODE.encode()).hexdigest()
                saved = {}
                for name in ("report", "holdout"):
                    path = out / f"attempt_1.{name}.json"
                    saved[path] = json.dumps({"ok": True, "candidate_sha256": "0" * 64 if name == phase else digest})
                    path.write_text(saved[path])
                result = run_session(messages=[], chat=lambda *_: self.fail("摘要不符应终止"), runtime=Runtime([]),
                                     out_dir=out, max_attempts=4, max_turns=8, resume=True)
                self.assertFalse(result["ok"])
                self.assertEqual(result["outcome"], "evaluation_error")
                self.assertEqual({path: path.read_text() for path in saved}, saved)

    def test_lower_total_ceiling_cannot_launch_an_already_saved_out_of_budget_attempt(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "prompt.json").write_text("[]")
            (out / "attempt_1.py").write_text(CODE)
            (out / "attempt_1.report.json").write_text('{"ok":false}')
            (out / "attempt_2.py").write_text(CODE + "# 第二轮\n")
            for attempts, turns in ((1, 4), (2, 1)):
                result = run_session(messages=[], chat=lambda *_: self.fail("超过总预算"), runtime=Runtime([]),
                                     out_dir=out, max_attempts=attempts, max_turns=turns, resume=True)
                self.assertEqual(result["stage"], "budget")
                self.assertFalse((out / "attempt_2.report.json").exists())

    def test_reserves_code_and_feedback_turns_warns_before_code_and_enforces_totals(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            seen = []
            def chat(messages, tools):
                seen.append((messages[-1].get("content", ""), bool(tools)))
                if tools:
                    return {"role": "assistant", "tool_calls": [call_message("read_code", {}, str(len(seen)))]}
                return {"role": "assistant", "content": CODE}
            runtime = Runtime([{"ok": False, "stage": "compare"}] * 4)
            result = run_session(messages=[], chat=chat, runtime=runtime, out_dir=out, max_attempts=4, max_turns=9)
            self.assertEqual(len(runtime.judged), 4)
            first_code = next(i for i, (_, tools) in enumerate(seen) if not tools)
            self.assertLessEqual(first_code + 1, 6)
            self.assertIn("完整", seen[first_code - 1][0])
            self.assertTrue(seen[first_code + 1][1], "收到数值失败后必须能继续查询差异来源")
            before = len(seen)
            again = run_session(messages=[], chat=chat, runtime=runtime, out_dir=out,
                                max_attempts=4, max_turns=20, resume=True)
            self.assertEqual((len(seen), again["turns"]), (before, result["turns"]))

    def test_last_turn_cannot_dispatch_tools_even_if_provider_returns_them(self):
        with TemporaryDirectory() as directory:
            runtime = Runtime([])
            def chat(messages, tools):
                self.assertEqual(tools, [])
                return {"role": "assistant", "tool_calls": [call_message("read_code", {})]}
            result = run_session(messages=[], chat=chat, runtime=runtime, out_dir=Path(directory),
                                 max_attempts=1, max_turns=1)
            self.assertFalse(result["ok"])
            self.assertEqual(runtime.calls, [])

    def test_interrupted_synthesis_retries_without_redoing_a_torn_tool_event(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "prompt.json").write_text("[]")
            line = json.dumps({"turn": 1, "id": "a", "tool": "observe", "arguments": {"run_id": "r"},
                               "result": {"ok": True}}).encode() + b"\n"
            (out / "tools.jsonl").write_bytes(line + b'{"partial":"\xe4\xb8')
            (out / "trace.json").write_text('{"turns":2,"stage":"turn_limit"}')
            def interrupted(messages, tools):
                self.assertFalse(tools)
                raise OSError("连接中断")
            args = dict(messages=[], runtime=Runtime([{"ok": True}, {"ok": True}]), out_dir=out,
                        max_attempts=2, max_turns=14, resume=True)
            self.assertEqual(run_session(chat=interrupted, **args)["turns"], 3)
            def retry(messages, tools):
                self.assertFalse(tools)
                return {"role": "assistant", "content": CODE}
            self.assertTrue(run_session(chat=retry, **args)["ok"])
            self.assertEqual((out / "tools.jsonl").read_bytes(), line)

    def test_cli_completed_resume_needs_no_chat_credentials(self):
        from nanovllm_selector.port import iterate

        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "prompt.json").write_text("[]")
            (out / "attempt_1.py").write_text(CODE)
            (out / "attempt_1.report.json").write_text('{"ok":true}')
            (out / "attempt_1.holdout.json").write_text('{"ok":true}')
            with patch.object(iterate, "load_dotenv", return_value=[]), patch(
                "nanovllm_selector.port.explore.session.initial_messages", return_value=[]
            ), patch("nanovllm_selector.port.explore.runtime.DockerRuntime", return_value=Runtime([])), patch(
                "nanovllm_selector.port.generate.make_chat", side_effect=AssertionError("不应初始化 API")
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(iterate.main(["--explore", "--resume", "--model", directory, "--out-dir", directory]), 0)

    def test_a_valid_graph_from_another_checkpoint_is_not_given_as_evidence(self):
        graph = {"model": {"config_sha256": "0" * 64}}
        with TemporaryDirectory() as directory, patch("nanovllm_selector.port.prompt._check_agent_graph"), patch(
            "nanovllm_selector.port.prompt._render_graph", return_value="UNRELATED_GRAPH"
        ):
            model = Path(directory)
            config = model / "config.json"
            config.write_text("{}")
            conversation = initial_messages(model, config, config, config, graph)
        self.assertNotIn("UNRELATED_GRAPH", conversation[-1]["content"])

    def test_tool_call_ids_repair_and_frozen_holdout(self):
        sequence = iter([
            {"role": "assistant", "tool_calls": [call_message("read_code", {"target": "hf"}),
                                                  call_message("shell", {}, "b")]},
            {"role": "assistant", "content": CODE},
            {"role": "assistant", "content": CODE + "\n# v2"},
        ])
        histories = []

        def chat(messages, tools):
            histories.append(list(messages))
            return next(sequence)

        runtime = Runtime([{"ok": False, "stage": "compare", "error": "差异"},
                           {"ok": True}, {"ok": True}])
        with TemporaryDirectory() as directory:
            out = Path(directory)
            result = run_session(messages=[], chat=chat, runtime=runtime, out_dir=out,
                                 max_attempts=2, max_turns=5)
            self.assertTrue(result["ok"])
            self.assertEqual([holdout for _, holdout in runtime.judged], [False, False, True])
            self.assertEqual(runtime.judged[1][0], runtime.judged[2][0])
            self.assertEqual(runtime.calls, [("read_code", {"target": "hf"})])
            replies = [m for m in histories[1] if m["role"] == "tool"]
            self.assertEqual([m["tool_call_id"] for m in replies], ["a", "b"])
            self.assertFalse(json.loads(replies[1]["content"])["ok"])
            self.assertEqual(len(list(out.glob("attempt_*.py"))), 2)
            self.assertTrue((out / "tools.jsonl").is_file())

    def test_failed_holdout_is_reported_without_tuning_on_it(self):
        runtime = Runtime([{"ok": True}, {"ok": False, "stage": "compare"}])
        with TemporaryDirectory() as directory:
            result = run_session(messages=[], chat=lambda *_: {"role": "assistant", "content": CODE},
                                 runtime=runtime, out_dir=Path(directory), max_attempts=3, max_turns=6)
            self.assertFalse(result["ok"])
            self.assertEqual(len(runtime.judged), 2)

    def test_chat_keeps_native_calls_and_excludes_private_reasoning(self):
        msg = {"role": "assistant", "content": None, "tool_calls": [call_message("observe", {})],
               "reasoning_content": "private"}
        response = io.BytesIO(json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls"}]}).encode())
        with patch.dict(os.environ, {"TEST_API_CREDENTIAL": "unused"}), patch(
            "urllib.request.urlopen", return_value=response
        ) as request:
            result = make_chat(api_key_env="TEST_API_CREDENTIAL")([], [{"type": "function"}])
        self.assertEqual(result["tool_calls"], msg["tool_calls"])
        self.assertNotIn("reasoning_content", result)
        self.assertIn("tools", json.loads(request.call_args.args[0].data))
