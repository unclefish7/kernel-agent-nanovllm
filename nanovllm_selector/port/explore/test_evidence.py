"""在 CPU 和本地微型模型上验证探查 worker 的 JSON 协议与落盘证据。"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM


WORKER = "nanovllm_selector.port.explore.worker"
PROBE = """
calls = 0

class Pair(torch.nn.Module):
    def forward(self, x, *, extra):
        x.add_(1)
        extra["state"][0].mul_(2)
        return x, extra["state"][0]

def setup(model):
    global calls
    calls += 1
    model.pair = Pair()

    def before(module, args, kwargs):
        global first
        first = kwargs["input_ids"].clone()
        x = torch.tensor([1, 2])
        extra = {"state": [torch.tensor([3])], "opaque": object()}
        module.pair(x, extra=extra)
        module.pair(x, extra=extra)
        x.zero_()
        extra["state"][0].zero_()

    model.register_forward_pre_hook(before, with_kwargs=True)

def probe(model, step, inputs, output):
    base = torch.arange(300, dtype=torch.int64).reshape(3, 100) + 2**54
    changed = base.clone()
    changed[2, 99] += 1
    return {"ids": first, "calls": torch.tensor(calls), "logits": output.logits,
            "base": base, "changed": changed, "float": base.double(),
            "bad": torch.tensor([float("inf")]), "position": inputs["cache_position"]}
"""


def key(run_id, step, module, port, call=0):
    return dict(run_id=run_id, step=step, module=module, call=call, port=port)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = Path(__file__).resolve().parents[3] / "tmp" / "explore-evidence-cpu"
        parent.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=parent, prefix="fixture-")
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cls.model = cls.root / "model"
        cls.runs = cls.root / "runs"
        cls.runs.mkdir()
        config = LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            LlamaForCausalLM(config).save_pretrained(cls.model, safe_serialization=True)

    def request(self, tool, **arguments):
        from nanovllm_selector.port.explore.worker import dispatch

        return dispatch(self.model, self.runs, {"tool": tool, "arguments": arguments})

    def cli(self, tool, **arguments):
        process = subprocess.run(
            [sys.executable, "-B", "-m", WORKER, "--model", str(self.model), "--runs", str(self.runs)],
            input=json.dumps({"tool": tool, "arguments": arguments}), text=True,
            capture_output=True, timeout=90, check=True,
        )
        return json.loads(process.stdout.splitlines()[-1])

    def capture(self, run_id="probe"):
        directory = self.runs / run_id
        if directory.exists():
            return
        directory.mkdir()
        result = self.request("run_model", target="probe", run_id=run_id, prompt_ids=[1, 4, 7],
                              decode_ids=[2, 6], watch=["", "pair"], probe=PROBE)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["steps"], 3)

    def test_boundary_snapshots_keep_positional_keyword_tuple_and_repeated_calls(self):
        self.capture()
        for call, port, values in [
            (0, "input.args.0", [1, 2]), (0, "input.kwargs.extra.state.0", [3]),
            (0, "output.0", [2, 3]), (0, "output.1", [6]),
            (1, "input.args.0", [2, 3]), (1, "output.1", [12]),
            (0, "post_input.kwargs.extra.state.0", [6]),
        ]:
            with self.subTest(call=call, port=port):
                result = self.request("observe", **key("probe", 0, "pair", port, call))
                self.assertEqual(result["values"], values)
        opaque = self.request("observe", **key("probe", 0, "pair", "input.kwargs.extra.opaque"))
        self.assertEqual(opaque["status"], "not_captured")
        self.assertTrue(opaque["item"]["reason"])

    def test_forced_decode_uses_positions_returned_cache_and_exact_setup_inputs(self):
        self.capture()
        for step, inputs, positions in [(0, [1, 4, 7], [0, 1, 2]), (1, [2], [3]), (2, [6], [4])]:
            for port, values in [("output.ids", inputs), ("output.position", positions), ("output.calls", [1])]:
                item = self.request("observe", **key("probe", step, "@probe", port))
                self.assertEqual(item["values"], values)
                self.assertIs(item["item"]["probe_derived"], True)
        path = "kwargs.past_key_values.layers.0.keys"
        before = self.request("observe", **key("probe", 1, "", "input." + path))
        after = self.request("observe", **key("probe", 1, "", "post_input." + path))
        self.assertEqual(before["item"]["shape"], [1, 1, 3, 8])
        self.assertEqual(after["item"]["shape"], [1, 1, 4, 8])
        previous = key("probe", 0, "", "output.past_key_values.layers.0.keys")
        self.assertEqual(self.request("compare", reference=previous,
                                      actual=key("probe", 1, "", "input." + path))["status"], "match")

    def test_saved_hf_run_lists_keys_and_compares_with_probe_in_a_fresh_worker(self):
        self.capture()
        (self.runs / "native").mkdir()
        native = self.cli("run_model", target="hf", run_id="native", prompt_ids=[1, 4, 7],
                          decode_ids=[2, 6], watch=["", "lm_head"])
        self.assertEqual(native["status"], "ok", native)
        self.assertEqual(native["provenance"]["dtype"], "torch.float32")
        self.assertTrue(native["provenance"]["attn_implementation"])
        self.assertEqual(native["calibration"]["status"], "match")
        self.assertEqual([row["step"] for row in native["calibration"]["steps"]], [0, 1, 2])
        self.assertTrue(all(row["exact"] and row["finite"] for row in native["calibration"]["steps"]))
        self.assertGreater(native["calibration"]["steps"][0]["state"]["tensors"], 0)
        listing = self.cli("observe", run_id="native", step=1, module="lm_head")
        ports = {item["port"] for item in listing["items"]}
        self.assertTrue({"input.args.0", "output"} <= ports)
        self.assertEqual(listing["provenance"]["prompt_ids"], [1, 4, 7])
        self.assertEqual(listing["provenance"]["target"], "hf")
        self.assertFalse(listing["items"][0]["probe_derived"])
        result = self.cli("compare", reference=key("native", 2, "", "output.logits"),
                          actual=key("probe", 2, "@probe", "output.logits"))
        self.assertEqual(result["status"], "match")

    def test_hook_sensitive_forward_marks_saved_hf_evidence_unreliable(self):
        original = LlamaForCausalLM.forward

        def sensitive(model, *args, **kwargs):
            output = original(model, *args, **kwargs)
            if model._forward_hooks:
                output.logits.add_(1)
            return output

        (self.runs / "sensitive").mkdir()
        with patch.object(LlamaForCausalLM, "forward", sensitive):
            result = self.request("run_model", run_id="sensitive", target="hf", prompt_ids=[1], watch=[""])
        self.assertEqual(result["status"], "unreliable")
        self.assertTrue(result["ok"], "宿主按此字段收集已完成的运行，包括校准失配的诊断证据")
        self.assertEqual(result["calibration"]["status"], "mismatch")
        self.assertFalse(self.request("observe", run_id="sensitive")["provenance"]["hf_reliable"])

    def test_state_drift_with_equal_logits_is_visible_in_calibration(self):
        original = LlamaForCausalLM.forward

        def sensitive(model, *args, **kwargs):
            output = original(model, *args, **kwargs)
            model.cursor = torch.tensor(int(bool(model._forward_hooks)))
            return output

        (self.runs / "state-drift").mkdir()
        with patch.object(LlamaForCausalLM, "forward", sensitive):
            result = self.request("run_model", run_id="state-drift", target="hf", prompt_ids=[1], watch=[""])
        row = result["calibration"]["steps"][0]
        self.assertTrue(row["exact"])
        self.assertIn("module[].attributes.cursor", row["state"]["changed_paths"])
        self.assertEqual(result["status"], "unreliable")

    def test_compare_scans_all_coordinates_checks_dtype_shape_finiteness_and_missing_step(self):
        self.capture()
        base = key("probe", 0, "@probe", "output.base")
        different = self.request("compare", reference=base, actual=dict(base, port="output.changed"))
        self.assertEqual(different["status"], "mismatch")
        self.assertEqual(different["mismatch_count"], 1)
        self.assertEqual(different["mismatches"][0]["index"], [2, 99])
        self.assertEqual(different["mismatches"][0]["actual"], 2**54 + 300)
        for a, b, reason in [
            (base, dict(base, port="output.float"), "dtype"),
            (key("probe", 0, "", "output.logits"), key("probe", 1, "", "output.logits"), "shape"),
            (dict(base, port="output.bad"), dict(base, port="output.bad"), "nonfinite"),
        ]:
            result = self.request("compare", reference=a, actual=b)
            self.assertEqual((result["status"], result["reason"]), ("mismatch", reason))
        self.assertEqual(self.request("compare", reference=base, actual=dict(base, step=9))["status"],
                         "not_captured")
        page = self.request("observe", **base, offset=298, limit=1)
        self.assertEqual(page["values"], [2**54 + 298])
        self.assertEqual(page["total"], 300)

    def test_refuses_run_reuse_incomplete_data_and_unsafe_file_references(self):
        self.capture()
        run = dict(target="hf", prompt_ids=[1], decode_ids=[], watch=[""], run_id="probe")
        with self.assertRaises((ValueError, FileExistsError)):
            self.request("run_model", **run)
        with self.assertRaises(ValueError):
            self.request("observe", run_id="../model")
        (self.runs / "pending").mkdir()
        with self.assertRaises(ValueError):
            self.request("observe", run_id="pending")
        self.capture("tampered")
        path = self.runs / "tampered" / "manifest.json"
        manifest = json.loads(path.read_text())
        item = next(row for row in manifest["items"] if row["kind"] == "tensor")
        artifact = path.parent / item["file"]
        payload = artifact.read_bytes()
        artifact.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaises(ValueError):
            self.request("observe", **key("tampered", item["step"], item["module"], item["port"], item["call"]))
        artifact.write_bytes(payload)
        item["file"] = "../model.safetensors"
        self.seal(path, manifest)
        with self.assertRaises(ValueError):
            self.request("observe", run_id="tampered")

    def seal(self, path, manifest):
        raw = json.dumps(manifest).encode()
        path.write_bytes(raw)
        path.with_name(".complete").write_text(hashlib.sha256(raw).hexdigest())

    def test_reader_checks_file_headers_and_host_assigned_source_before_listing(self):
        run_id = "probe-" + "a" * 32
        self.capture(run_id)
        path = self.runs / run_id / "manifest.json"
        manifest = json.loads(path.read_bytes())
        item = next(row for row in manifest["items"] if row["kind"] == "tensor" and row["shape"] == [2])
        item["shape"] = [1, 2]
        self.seal(path, manifest)
        with self.assertRaises(ValueError):
            self.request("observe", run_id=run_id)
        item["shape"] = [2]
        manifest["provenance"].update(target="hf", probe_sha256=None, hf_reliable=True)
        manifest["items"] = [row for row in manifest["items"] if row["module"] != "@probe"]
        manifest["calibration"] = {"status": "match"}
        self.seal(path, manifest)
        with self.assertRaises(ValueError):
            self.request("observe", run_id=run_id)

    def test_cap_fails_before_an_oversized_probe_tensor_is_written(self):
        from nanovllm_selector.port.explore import evidence

        directory = self.runs / "capacity"
        directory.mkdir()
        with patch.object(evidence, "MAX_TENSOR_BYTES", 4), self.assertRaisesRegex(ValueError, "张量"):
            self.request("run_model", run_id="capacity", target="probe", prompt_ids=[1], watch=[],
                         probe="def probe(model, step, inputs, output): return {'large': torch.zeros(10)}")
        self.assertEqual([p.name for p in directory.iterdir()], [".started"])

    def test_probe_dictionary_exposes_escaped_names_empty_and_complex_tensors(self):
        (self.runs / "numeric").mkdir()
        source = """
def probe(model, step, inputs, output):
    return {"phase.real": torch.tensor([2+3j], dtype=torch.complex128), "empty": torch.empty(0, 2)}
"""
        self.request("run_model", run_id="numeric", target="probe", prompt_ids=[1], watch=[], probe=source)
        parent = key("numeric", 0, "@probe", "output")
        directory = self.request("observe", **parent)
        self.assertEqual({i["port"] for i in directory["items"]}, {"output.phase~1real", "output.empty"})
        phase = dict(parent, port="output.phase~1real")
        observation = self.request("observe", **phase)
        self.assertEqual(observation["item"]["dtype"], "torch.complex128")
        self.assertEqual(observation["values"], [{"real": 2.0, "imag": 3.0}])
        self.assertEqual(self.request("compare", reference=phase, actual=phase)["status"], "match")
        empty = self.request("observe", **dict(parent, port="output.empty"))
        self.assertEqual((empty["item"]["shape"], empty["values"]), ([0, 2], []))

    def test_failed_probe_leaves_an_unusable_run_and_cannot_overwrite_it(self):
        (self.runs / "failed").mkdir()
        args = dict(run_id="failed", target="probe", prompt_ids=[1], watch=[],
                    probe="def probe(model, step, inputs, output): return {'bad': 2}")
        with self.assertRaises(ValueError):
            self.request("run_model", **args)
        with self.assertRaises(ValueError):
            self.request("observe", run_id="failed")
        with self.assertRaises((ValueError, FileExistsError)):
            self.request("run_model", **args)


if __name__ == "__main__":
    unittest.main()
