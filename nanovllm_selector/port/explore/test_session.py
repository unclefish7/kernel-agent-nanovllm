import io
import json
import os
import unittest
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
    def test_a_valid_graph_from_another_checkpoint_is_not_given_as_evidence(self):
        root = Path(__file__).resolve().parents[3]
        model = root / "nanovllm_selector/fixtures/llama-3.2-1b"
        graph = {"model": {"config_sha256": "0" * 64}}
        with patch("nanovllm_selector.port.prompt._check_agent_graph"), patch(
            "nanovllm_selector.port.prompt._render_graph", return_value="UNRELATED_GRAPH"
        ):
            conversation = initial_messages(model, root / "contracts/nanovllm/spec.yaml",
                                            root / "kernels/index.md", root / "nano-vllm/nanovllm/models/qwen3.py", graph)
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
