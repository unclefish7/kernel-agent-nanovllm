"""离线检查宿主与 Docker 的交接。测试临时目录始终位于仓库中。"""

import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch


MODULE = "nanovllm_selector.port.explore.runtime"


def capture_report(model, candidate, trial, holdout):
    from nanovllm_selector.port.explore.judge import checkpoint_sha256

    trial.mkdir()
    metadata, logits = b'{"version":2}', b"tensor bytes"
    (trial / "engine.json").write_bytes(metadata)
    (trial / "engine.safetensors").write_bytes(logits)
    request = b"holdout" if holdout else b"public"
    return {
        "ok": True, "stage": "engine", "outcome": "captured", "holdout": holdout,
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256(model),
        "engine_json_sha256": hashlib.sha256(metadata).hexdigest(),
        "logits_sha256": hashlib.sha256(logits).hexdigest(),
        "request_sha256": hashlib.sha256(request).hexdigest(),
    }


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec(MODULE), "缺少宿主 Docker 运行时")
        from nanovllm_selector.port.explore import runtime

        self.module = runtime
        root = Path(__file__).resolve().parents[3] / "tmp"
        with_dir = tempfile.TemporaryDirectory(prefix="explore-runtime-test-", dir=root)
        self.addCleanup(with_dir.cleanup)
        self.root = Path(with_dir.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        self.out = self.root / "session"
        self.runtime = runtime.DockerRuntime(self.model, self.out, image="local-worker", device="cpu")

    def test_archive_accepts_only_bounded_regular_files_under_fresh_root(self):
        def archive(members):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as writer:
                for name, kind in members:
                    item = tarfile.TarInfo(name)
                    item.type = kind
                    item.size = 3 if item.isfile() else 0
                    writer.addfile(item, io.BytesIO(b"abc"))
            buf.seek(0)
            return buf

        dest = self.root / "received"
        self.module._receive_files(archive([("./a", tarfile.DIRTYPE),
                                            ("./a/b", tarfile.REGTYPE)]), dest)
        self.assertEqual((dest / "a/b").read_bytes(), b"abc")
        for i, members in enumerate([
            [("../other", tarfile.REGTYPE)], [("/other", tarfile.REGTYPE)],
            [("linked", tarfile.SYMTYPE)], [("linked", tarfile.LNKTYPE)],
            [("pipe", tarfile.FIFOTYPE)], [("a/" * 32 + "b", tarfile.REGTYPE)],
            [("file", tarfile.REGTYPE), ("file", tarfile.REGTYPE)],
        ]):
            with self.subTest(members=members), self.assertRaises(ValueError):
                self.module._receive_files(archive(members), self.root / f"bad-{i}")
        with patch.object(self.module, "MAX_ARTIFACT_BYTES", 2), self.assertRaises(ValueError):
            self.module._receive_files(archive([("large", tarfile.REGTYPE)]), self.root / "large")
        with patch.object(self.module, "MAX_ENTRIES", 1), self.assertRaises(ValueError):
            self.module._receive_files(archive([("a", tarfile.REGTYPE), ("b", tarfile.REGTYPE)]),
                                       self.root / "many")

    def test_run_ids_and_read_only_history_survive_runtime_restart(self):
        requests = []

        def invoke(job, command, *, request=None, mounts=(), scratch=None, collect=None):
            requests.append((request, dict((dst, src) for src, dst in mounts), scratch))
            if collect:
                collect.mkdir()
                args = request["arguments"]
                (collect / "manifest.json").write_text(json.dumps(
                    {"run_id": args["run_id"], "provenance": {"target": args["target"]}}))
            return {"ok": True, "status": "ok"}

        with patch.object(self.module.DockerRuntime, "_invoke", side_effect=invoke):
            args = {"target": "hf", "prompt_ids": [1]}
            first = self.runtime.call("run_model", args)
            another = self.module.DockerRuntime(self.model, self.out, image="local-worker", device="cpu")
            second = another.call("run_model", {"target": "probe", "prompt_ids": [2], "probe": "pass"})
            observed = another.call("observe", {"run_id": first["run_id"]})
        self.assertTrue(observed["ok"])
        self.assertNotIn("run_id", args)
        self.assertNotEqual(first["run_id"], second["run_id"])
        for (payload, mounts, scratch), reply in zip(requests, (first, second)):
            self.assertEqual(payload["arguments"]["run_id"], reply["run_id"])
            self.assertEqual(scratch, "/runs/" + reply["run_id"])
            self.assertEqual(mounts["/runs"], self.out / "runs")
        self.assertIsNone(requests[-1][2])
        self.assertTrue((self.out / "runs" / first["run_id"] / "manifest.json").is_file())

    def test_probe_cannot_publish_with_hf_provenance_or_overwrite_an_existing_id(self):
        def mislabel(job, command, *, request, mounts, scratch, collect):
            collect.mkdir()
            (collect / "manifest.json").write_text(json.dumps(
                {"run_id": request["arguments"]["run_id"], "provenance": {"target": "hf"}}))
            return {"ok": True}

        with patch.object(self.runtime, "_invoke", side_effect=mislabel):
            result = self.runtime.call("run_model", {"target": "probe", "run_id": "hf-existing"})
        self.assertFalse(result["ok"])
        self.assertEqual(list((self.out / "runs").iterdir()), [])

    def test_completed_hf_calibration_mismatch_can_still_be_observed(self):
        def unreliable(job, command, *, request, mounts, scratch, collect):
            collect.mkdir()
            run_id = request["arguments"]["run_id"]
            (collect / "manifest.json").write_text(json.dumps(
                {"run_id": run_id, "provenance": {"target": "hf"}}))
            return {"ok": False, "status": "unreliable", "calibration": {"status": "mismatch"}}

        with patch.object(self.runtime, "_invoke", side_effect=unreliable):
            result = self.runtime.call("run_model", {"target": "hf", "prompt_ids": [1]})
        self.assertFalse(result["ok"])
        self.assertTrue((self.out / "runs" / result["run_id"] / "manifest.json").is_file())

    def test_run_mount_can_be_traversed_with_private_host_umask(self):
        def check(job, command, *, request, mounts):
            root = dict((dst, src) for src, dst in mounts)["/runs"]
            self.assertEqual(root.stat().st_mode & 0o005, 0o005)
            return {"status": "not_captured"}

        previous = os.umask(0o077)
        try:
            with patch.object(self.runtime, "_invoke", side_effect=check):
                self.runtime.call("observe", {"run_id": "absent"})
        finally:
            os.umask(previous)

    def test_candidate_is_snapshotted_and_only_engine_data_reaches_reference(self):
        candidate = self.root / "candidate.py"
        original = b"class TestForCausalLM: pass\n"
        candidate.write_bytes(original)
        (self.model / "model.safetensors").write_bytes(b"weights")
        digest = hashlib.sha256(original).hexdigest()
        commands = []
        capture = {}

        def invoke(job, command, *, request=None, mounts=(), scratch=None, collect=None):
            commands.append(command)
            mounts = {dst: src for src, dst in mounts}
            self.assertIn("--holdout", command)
            if "engine" in command:
                self.assertEqual(mounts["/candidate.py"].read_bytes(), original)
                self.assertEqual(scratch, "/trial")
                candidate.write_text("# 改动晚于快照\n")
                capture.update(capture_report(self.model, mounts["/candidate.py"], collect, True))
                return capture
            self.assertEqual(set(mounts), {"/trial"})
            trial = mounts["/trial"]
            self.assertEqual({p.name for p in trial.iterdir()}, {"engine.json", "engine.safetensors", "binding.json"})
            self.assertEqual(command[command.index("--binding") + 1], "/trial/binding.json")
            self.assertEqual(command[command.index("--candidate-sha256") + 1], digest)
            self.assertIsNone(scratch)
            binding = json.loads((trial / "binding.json").read_bytes())
            self.assertEqual(binding, {"version": 1, **{key: capture[key] for key in
                             ("candidate_sha256", "checkpoint_sha256", "holdout", "engine_json_sha256",
                              "logits_sha256")}})
            return {**capture, "ok": False, "stage": "compare", "outcome": "candidate_failed"}

        with patch.object(self.runtime, "_invoke", side_effect=invoke):
            report = self.runtime.evaluate(candidate, holdout=True)
        self.assertEqual(len(commands), 2)
        self.assertEqual(report["candidate_sha256"], digest)
        self.assertEqual(report["stage"], "compare")
        self.assertFalse(report["ok"])

    def test_mismatched_engine_or_reference_is_an_evaluation_error(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("pass")
        (self.model / "model.safetensors").write_bytes(b"weights")
        for phase, field in (
            ("engine", "checkpoint_sha256"), ("engine", "engine_json_sha256"), ("engine", "logits_sha256"),
            ("engine", "holdout"), ("reference", "checkpoint_sha256"), ("reference", "request_sha256"),
            ("reference", "candidate_sha256"), ("reference", "missing_candidate"),
        ):
            calls = []
            capture = {}

            def invoke(job, command, *, mounts, scratch=None, collect=None):
                role = "engine" if collect is not None else "reference"
                calls.append(role)
                if role == "engine":
                    capture.update(capture_report(self.model, mounts[0][0], collect, False))
                    reply = dict(capture)
                else:
                    reply = {**capture, "ok": True, "stage": "compare", "outcome": "passed"}
                if role == phase:
                    if field == "missing_candidate":
                        del reply["candidate_sha256"]
                    else:
                        reply[field] = True if field == "holdout" else "a" * 64
                return reply

            with self.subTest(phase=phase, field=field), patch.object(self.runtime, "_invoke", side_effect=invoke):
                report = self.runtime.evaluate(candidate)
                self.assertEqual((report["ok"], report["outcome"]), (False, "evaluation_error"))
                self.assertEqual(calls, ["engine"] if phase == "engine" else ["engine", "reference"])
                if phase == "reference" and field == "candidate_sha256":
                    self.assertEqual(report["candidate_sha256"], "a" * 64)
                elif field == "missing_candidate":
                    self.assertNotIn("candidate_sha256", report)

    def test_concurrent_judges_keep_separate_candidate_and_holdout_bindings(self):
        (self.model / "model.safetensors").write_bytes(b"weights")
        both_engines = Barrier(2, timeout=10)
        candidates = [self.root / "a.py", self.root / "b.py"]
        for path in candidates:
            path.write_text(path.stem)

        def invoke(job, command, *, mounts, scratch=None, collect=None):
            holdout = "--holdout" in command
            if collect is not None:
                report = capture_report(self.model, mounts[0][0], collect, holdout)
                both_engines.wait()
                return report
            binding = json.loads((mounts[0][0] / "binding.json").read_bytes())
            request = b"holdout" if holdout else b"public"
            return {**binding, "ok": True, "stage": "compare", "outcome": "passed",
                    "request_sha256": hashlib.sha256(request).hexdigest()}

        with patch.object(self.runtime, "_invoke", side_effect=invoke), ThreadPoolExecutor(2) as pool:
            results = [pool.submit(self.runtime.evaluate, path, holdout=bool(i)) for i, path in enumerate(candidates)]
            for i, (future, path) in enumerate(zip(results, candidates)):
                report = future.result()
                self.assertTrue(report["ok"], report)
                self.assertIs(report["holdout"], bool(i))
                self.assertEqual(report["candidate_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(list(self.out.iterdir()), [])

    def test_engine_must_identify_the_frozen_candidate_before_reference_runs(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("pass")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        for mode in ("missing", "other", "changed_snapshot", "engine_binding_file"):
            def capture(job, command, *, mounts, scratch=None, collect=None):
                self.assertIn("engine", command)
                collect.mkdir()
                (collect / "engine.json").write_bytes(b"{}")
                (collect / "engine.safetensors").write_bytes(b"rows")
                result = {"ok": True, "stage": "engine", "candidate_sha256": digest}
                if mode == "missing":
                    del result["candidate_sha256"]
                elif mode == "other":
                    result["candidate_sha256"] = "f" * 64
                elif mode == "changed_snapshot":
                    snapshot = dict((dst, src) for src, dst in mounts)["/candidate.py"]
                    snapshot.chmod(0o600)
                    snapshot.write_text("pass # changed")
                else:
                    (collect / "binding.json").write_text("{}")
                return result

            with self.subTest(mode=mode), patch.object(self.runtime, "_invoke", side_effect=capture):
                result = self.runtime.evaluate(candidate)
                self.assertEqual((result["ok"], result["outcome"]), (False, "evaluation_error"))

    def test_host_rejects_linked_candidate_and_output_paths(self):
        source = self.root / "original.py"
        source.write_text("pass")
        candidate = self.root / "candidate.py"
        candidate.symlink_to(source)
        self.assertFalse(self.runtime.evaluate(candidate)["ok"])
        candidate.unlink()
        os.link(source, candidate)
        self.assertFalse(self.runtime.evaluate(candidate)["ok"])
        redirected = self.root / "redirected"
        redirected.symlink_to(self.out, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.module.DockerRuntime(self.model, redirected / "x", image="local-worker")
        with self.assertRaises(ValueError):
            self.module.DockerRuntime(self.model, self.model / "output", image="local-worker")
        self.assertFalse((self.model / "output").exists())

    def test_engine_input_errors_and_oversized_trial_do_not_reach_reference(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("pass")
        failure = {"ok": False, "outcome": "evaluation_error", "stage": "input", "error": "无词表"}
        with patch.object(self.runtime, "_invoke", return_value=failure):
            result = self.runtime.evaluate(candidate)
        self.assertEqual((result["stage"], result["outcome"]), ("input", "evaluation_error"))

        def oversized(job, command, *, mounts, scratch, collect):
            collect.mkdir()
            (collect / "engine.json").write_bytes(b"0" * 65537)
            (collect / "engine.safetensors").write_bytes(b"data")
            return {"ok": True, "stage": "engine", "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}

        with patch.object(self.runtime, "_invoke", side_effect=oversized):
            result = self.runtime.evaluate(candidate)
        self.assertFalse(result["ok"])
        self.assertIn("engine.json", result["error"])
        self.assertEqual(result.get("candidate_sha256"), hashlib.sha256(candidate.read_bytes()).hexdigest())

    def test_only_inductor_loading_triton_hip_driver_is_an_environment_error(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("pass")
        installed = "hip_utils.cpython-312-x86_64-linux-gnu.so"
        driver = f"/tmp/triton/23TEXOBCVW353YGPBQ2CSVMWKSIYWL3HKWKGKUARCDWLFNLK6BAA/{installed}"
        observed = f"InductorError: ImportError: {driver}: failed to map segment from shared object"
        for error, outcome in (
            (observed + "\n\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace", "evaluation_error"),
            ("Traceback (most recent call last):\n" + "torch._inductor.exc." + observed, "evaluation_error"),
            (observed.replace(driver, f"/candidate/{installed}"), "candidate_failed"),
            (observed.replace(installed, "candidate.so"), "candidate_failed"),
            (observed.replace("failed to map segment from shared object", "undefined symbol: op"),
             "candidate_failed"),
            ("InductorError: RuntimeError: shape mismatch", "candidate_failed"),
        ):
            with self.subTest(error=error), patch.object(self.runtime, "_invoke", return_value={
                    "ok": False, "stage": "engine", "error": error}):
                report = self.runtime.evaluate(candidate)
                self.assertEqual((report["ok"], report["outcome"]), (False, outcome))
                self.assertEqual(report["error"], error)

    def test_diagnosis_uses_one_isolated_frozen_candidate_and_returns_only_diagnostics(self):
        candidate = self.root / "layer.py"
        candidate.write_bytes(b"# input")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        first = {"engine_path": "decoder.layers.3.mlp", "max_abs_diff": 0.125}

        def diagnose(job, command, *, mounts, reply_limit):
            self.assertEqual(len(mounts), 1)
            snapshot, destination = mounts[0]
            self.assertEqual((destination, command[-3:]), ("/candidate.py", ["/model", "/candidate.py", "76"]))
            self.assertEqual(reply_limit, 64 << 10)
            candidate.write_bytes(b"# later")
            self.assertEqual(snapshot.read_bytes(), b"# input")
            self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
            return {"purpose": "prefill_diagnostic", "candidate_sha256": digest, "length": 76,
                    "compared": 27, "first_difference": first, "logits": {"different_elements": 2},
                    "outcome": "passed", "ok": True}

        with patch.object(self.runtime, "_invoke", side_effect=diagnose) as worker:
            report = self.runtime.diagnose(candidate, length=76)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(report["candidate_sha256"], digest)
        self.assertEqual(report["first_difference"], first)
        self.assertFalse(report["trusted_for_acceptance"])
        self.assertNotIn("ok", report)
        self.assertNotIn("outcome", report)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_diagnosis_rejects_excess_length_and_a_report_from_another_candidate(self):
        candidate = self.root / "layer.py"
        candidate.write_text("pass")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        with patch.object(self.runtime, "_invoke") as worker:
            for length in (0, -1, 1025, True, "128"):
                with self.subTest(length=length):
                    report = self.runtime.diagnose(candidate, length)
                    self.assertIn("error", report)
                    self.assertEqual(report["candidate_sha256"], digest)
            worker.assert_not_called()
        for field, value in (("candidate_sha256", "b" * 64), ("length", 2), ("purpose", "verdict")):
            reply = {"purpose": "prefill_diagnostic", "length": 1, "candidate_sha256": digest, "logits": {}}
            reply[field] = value
            with self.subTest(field=field), patch.object(self.runtime, "_invoke", return_value=reply):
                report = self.runtime.diagnose(candidate, 1)
                self.assertIn("error", report)
                self.assertNotIn("logits", report)
                self.assertNotIn("outcome", report)
                self.assertEqual(report["candidate_sha256"], digest)

    def test_worker_failure_preserves_last_json_and_docker_diagnostics(self):
        mismatch = {"ok": False, "stage": "compare", "error": "逐项不相等"}
        out = ("preparing\n" + json.dumps(mismatch) + "\n").encode()
        self.assertEqual(self.module._last_json(1, out, b"warning"), mismatch)
        for rc, stdout, stderr, expected in [
            (125, b"", b"no such image", "no such image"),
            (1, b"loading\ninvalid", b"missing model", "missing model"),
            (1, b'{"ok":true}', b"killed", "killed"),
        ]:
            with self.subTest(rc=rc), self.assertRaisesRegex(RuntimeError, expected):
                self.module._last_json(rc, stdout, stderr)

    @unittest.skipUnless(os.environ.get("NANOVLLM_TEST_WORKER_IMAGE"), "需要本机 Docker worker 镜像")
    def test_private_compiler_caches_can_load_shared_libraries(self):
        self.runtime.image = os.environ["NANOVLLM_TEST_WORKER_IMAGE"]
        script = """
import ctypes, json, os, pathlib, shutil
assert os.statvfs("/tmp").f_flag & os.ST_NOEXEC
for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
    directory = pathlib.Path(os.environ[key])
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "library.so"
    shutil.copyfile(next(pathlib.Path("/usr/lib").glob("*/libm.so.6")), library)
    ctypes.CDLL(str(library))
print(json.dumps({"ok": True}))
"""
        with tempfile.TemporaryDirectory(dir=self.root) as job:
            result = self.runtime._invoke(Path(job), [self.module.PYTHON, "-B", "-c", script])
        self.assertTrue(result["ok"])

    @unittest.skipUnless(os.environ.get("NANOVLLM_TEST_WORKER_IMAGE"), "需要本机 Docker worker 镜像")
    def test_model_exposes_only_data_inside_the_private_container(self):
        self.runtime.image = os.environ["NANOVLLM_TEST_WORKER_IMAGE"]
        (self.model / ".env").write_text("fixture")
        (self.model / "config.json").chmod(0o600)
        (self.model / "candidate.py").write_text("# 测试模型目录里混放的候选\n")
        script = """
import json, os, pathlib, resource, socket, sys
p = pathlib.Path
assert (p("/model/config.json")).read_text() == "{}"
assert sorted(entry.name for entry in p("/model").iterdir()) == ["config.json"]
assert os.getuid() != 0 and set(n for _, n in socket.if_nameindex()) == {"lo"}
assert os.stat("/proc/self/ns/ipc").st_ino != int(sys.argv[1])
status = dict(line.split(":", 1) for line in p("/proc/self/status").read_text().splitlines() if ":" in line)
assert int(status["CapEff"], 16) == 0 and int(status["NoNewPrivs"]) == 1
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
assert not any(p(n).exists() for n in ("/var/run/docker.sock", "/workspace/nanovllm_selector/.env", "/dev/kfd"))
for path in ("/model/config.json", "/workspace/nanovllm_selector/extract", "/workspace/nano-vllm/nanovllm"):
    assert os.statvfs(path).f_flag & os.ST_RDONLY
assert not list(p("/dev/dri").glob("renderD*"))
print(json.dumps({"ok": True}))
"""
        with socket.socket(socket.AF_UNIX) as sock, tempfile.TemporaryDirectory(dir=self.root) as job:
            sock.bind(str(self.model / "local.sock"))
            inode = str(os.stat("/proc/self/ns/ipc").st_ino)
            # 双阶段评测会从同一宿主任务目录启动第二个容器。
            for _ in range(2):
                result = self.runtime._invoke(Path(job), [self.module.PYTHON, "-B", "-c", script, inode])
                self.assertTrue(result["ok"])

    @unittest.skipUnless(os.environ.get("NANOVLLM_TEST_WORKER_IMAGE"), "需要本机 Docker worker 镜像")
    def test_timeout_removes_only_the_allocated_container(self):
        self.runtime.image = os.environ["NANOVLLM_TEST_WORKER_IMAGE"]
        self.runtime.timeout = 1.5
        invoke = self.module._bounded_command
        names = []

        def record(command, *args, **kwargs):
            if command[:2] == ["docker", "run"]:
                names.extend(arg.removeprefix("--name=") for arg in command if arg.startswith("--name="))
            return invoke(command, *args, **kwargs)

        with patch.object(self.module, "_bounded_command", side_effect=record):
            with tempfile.TemporaryDirectory(dir=self.root) as job, self.assertRaises(TimeoutError):
                self.runtime._invoke(Path(job), ["/bin/sleep", "15"])
        name, = names
        listing = subprocess.check_output(["docker", "ps", "-a", "--filter", f"name=^/{name}$", "-q"])
        self.assertEqual(listing.strip(), b"")


if __name__ == "__main__":
    unittest.main()
