import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class ExternalAgentCliTests(unittest.TestCase):
    def test_context_and_one_tool_share_the_production_interfaces(self):
        from . import __main__ as cli
        from .runtime import ROOT

        with TemporaryDirectory(dir=ROOT / "tmp") as temp:
            out = Path(temp) / "out"
            model = Path(temp) / "model"
            model.mkdir()
            (model / "config.json").write_text('{"architectures":["FixtureLMHeadModel"]}')
            args = ["--model", str(model), "--out-dir", str(out)]
            with redirect_stdout(io.StringIO()) as response:
                self.assertEqual(cli.main([*args, "--context"]), 0)
            context = json.loads(response.getvalue())
            self.assertIn("FixtureLMHeadModel", context["messages"][1]["content"])
            self.assertEqual({t["function"]["name"] for t in context["tools"]},
                             {"read_code", "run_model", "observe", "compare"})
            result = {"status": "ok", "values": [3, 7]}
            with patch.object(cli.DockerRuntime, "call", return_value=result) as call, \
                    patch("sys.stdin", io.StringIO('{"run_id":"hf-abc"}')), redirect_stdout(io.StringIO()) as response:
                self.assertEqual(cli.main([*args, "--tool", "observe"]), 0)
            call.assert_called_once_with("observe", {"run_id": "hf-abc"})
            self.assertEqual(json.loads(response.getvalue()), result)
            event = json.loads((out / "external_tools.jsonl").read_text())
            self.assertEqual(event["result"], result)
            self.assertEqual(event["arguments"], {"run_id": "hf-abc"})
