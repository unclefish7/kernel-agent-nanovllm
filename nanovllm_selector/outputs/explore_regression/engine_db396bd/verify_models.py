"""从仓库根目录运行，使用已安装的生产 judge 对固定引擎提交做单卡严格回归。"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd().resolve()
ENGINE = ROOT / "nano-vllm"
UUID = "GPU-6fb1bbd498dc7bd1"
PCI = "0000:27:00.0"
RENDER = "/dev/dri/renderD131"
IMAGE = "nanovllm-explore-worker:local"
MODELS = {
    "qwen3": ("Qwen3-0.6B", "qwen3_exact/model_qwen3.py"),
    "llama": ("llama-3.2-1B", "iter_llama4/attempt_4.py"),
    "minicpm": ("MiniCPM5-1B", "minicpm5_1b/agent/attempt_1.py"),
    "lfm": ("LFM2.5-1.2B-Base", "lfm25_1_2b/agent_repair/attempt_1.py"),
}
IDENTITY = ("candidate_sha256", "checkpoint_sha256", "request_sha256", "engine_json_sha256", "logits_sha256")
PUBLIC_TEXT = "The quick brown fox jumps over the lazy dog. " * 400


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def write(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def revision(workdir, *args):
    return subprocess.check_output(["git", "-C", workdir, *args], text=True).strip()


def materialize(repo, commit):
    engine = repo / "nano-vllm"
    engine.mkdir()
    with tempfile.TemporaryFile(dir=repo) as archive:
        subprocess.run(["git", "-C", ENGINE, "archive", commit, "nanovllm", "tests"],
                       stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(engine, filter="data")
    for line in revision(ENGINE, "ls-tree", "-r", commit, "nanovllm", "tests").splitlines():
        spec, name = line.split("\t", 1)
        contents = (engine / name).read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(contents) + contents).hexdigest() == spec.split()[2]
    seen = {}
    for sub in ("extract", "port", "port/explore"):
        package = Path("nanovllm_selector") / sub
        (repo / package).mkdir(parents=True, exist_ok=True)
        for source in sorted((ROOT / package).glob("*.py")):
            if source.name.startswith("test_") or source.is_symlink():
                continue
            rel = source.relative_to(ROOT)
            shutil.copyfile(source, repo / rel)
            seen[str(rel)] = digest(repo / rel)
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="引擎 Git 提交")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, action="append")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update(ROCR_VISIBLE_DEVICES=UUID, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / "nanovllm_selector/outputs/explore_regression")
    out.mkdir(parents=True, exist_ok=True)
    commit = revision(ENGINE, "rev-parse", "--verify", args.engine + "^{commit}")
    gpu = Path("/sys/class/drm/renderD131/device")
    assert gpu.resolve().name == PCI and "GPU-" + (gpu / "unique_id").read_text().strip() == UUID
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="db396bd-score-") as temp:
        repo = Path(temp)
        sources = materialize(repo, commit)
        sys.path.insert(0, str(repo))
        from nanovllm_selector.port.explore import judge, runtime
        assert Path(runtime.__file__).is_relative_to(repo)
        host_env = {"PATH": os.environ["PATH"], "DOCKER_CONFIG": str(repo / "docker"), "TMPDIR": temp}
        image_id = subprocess.check_output(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE],
                                           env=host_env, text=True).strip()
        manifest = {"utc": datetime.now(timezone.utc).isoformat(), "parent": revision(ROOT, "rev-parse", "HEAD"),
                    "parent_gitlink": revision(ROOT, "rev-parse", "HEAD:nano-vllm"), "engine": commit,
                    "engine_tree": revision(ENGINE, "rev-parse", commit + ":nanovllm"), "image": image_id,
                    "registry_sha256": digest(repo / "nano-vllm/nanovllm/models/__init__.py"),
                    "loader_sha256": digest(repo / "nano-vllm/nanovllm/utils/loader.py"),
                    "package_manifest_sha256": json_sha(sources),
                    "critical_sha256": {p: value for p, value in sources.items() if p.endswith(
                        ("/runtime.py", "/judge.py", "/evaluate.py", "/evidence.py"))},
                    "package_source_sha256": sources, "workspace_tmp": str(repo.relative_to(ROOT))}
        prefix = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
        write(out / (prefix + ".json"), manifest)
        print(prefix, "engine", commit, "image", image_id, flush=True)

        class PinnedGPU(runtime.DockerRuntime):
            # 生产入口先不加可见性掩码校验唯一 render 节点和 PCI。只给实际 judge 进程补上 UUID。
            def _invoke(self, job, command, **options):
                phase = command[command.index(runtime.PACKAGE + ".judge") + 1]
                result = super()._invoke(job, ["/usr/bin/env", "ROCR_VISIBLE_DEVICES=" + UUID, *command], **options)
                if phase == "engine" and result.get("ok"):
                    self.frozen = digest(job / "candidate.py")
                if phase == "reference" and result.get("ok"):
                    trial, = [p for p, dest in options["mounts"] if dest == "/trial"]
                    meta = json.loads((trial / "engine.json").read_bytes())
                    binding = json.loads((trial / "binding.json").read_bytes())
                    with (trial / "engine.safetensors").open("rb") as stream:
                        hsize = int.from_bytes(stream.read(8), "little")
                        header = json.loads(stream.read(hsize))["__metadata__"]
                    req = {"holdout": meta["holdout"], "text": judge.HOLDOUT_TEXT if meta["holdout"] else PUBLIC_TEXT,
                           "prefill_ids": [r["prompt_ids"] for r in meta["requests"][:-1]],
                           "generation_prompt_ids": meta["requests"][-1]["prompt_ids"], "decode_steps": 8,
                           "lengths": meta["lengths"], "prefill_max_tokens": 1, "seed": 0,
                           "tensor_parallel_size": 1, "block_size": 256}
                    self.audit = {"candidate_sha256": self.frozen, "checkpoint_sha256": self.checkpoint,
                                  "request_sha256": json_sha(req), "engine_json_sha256": digest(trial / "engine.json"),
                                  "logits_sha256": digest(trial / "engine.safetensors")}
                    self.crosschecked = (all(meta[k] == header[k] == self.audit[k] for k in IDENTITY[:3])
                                         and all(binding[k] == self.audit[k] for k in IDENTITY if k != "request_sha256")
                                         and binding["holdout"] == meta["holdout"]
                                         and ("request_sha256" not in binding or binding["request_sha256"] == self.audit["request_sha256"])
                                         and result["architecture"] == header["architecture"] == meta["architecture"])
                return result

        for key in args.model or MODELS:
            name, rel = MODELS[key]
            model = ROOT / "tmp/models" / name
            source = ROOT / "nanovllm_selector/outputs" / rel
            assert source == source.resolve() and source.stat().st_nlink == 1
            assert model == model.resolve() and not any(p.is_symlink() for p in model.rglob("*"))
            candidate_sha = digest(source)
            files = sorted({*model.glob("*.safetensors"), *model.glob("*.safetensors.index.json"),
                            model / "config.json", model / "generation_config.json"})
            inputs = {p.name: digest(p) for p in files if p.is_file()}
            ck = json_sha(inputs)
            run = PinnedGPU(model, repo / "jobs" / key, image=IMAGE, device="1")
            run.checkpoint = ck
            from time import monotonic
            run._device_options(host_env, monotonic() + 30)
            assert run._gpu[:2] == (RENDER, PCI)
            for holdout in (False, True):
                case = key + (".holdout" if holdout else ".public")
                raw = out / (case + ".json")
                if raw.exists():
                    raw = out / (case + "." + prefix + ".json")
                run.crosschecked = False
                run.audit = {}
                result = run.evaluate(source, holdout=holdout)
                write(raw, result)
                info = {"run": prefix, "engine": commit, "model": str(model.relative_to(ROOT)),
                        "candidate": str(source.relative_to(ROOT)), "input_files": inputs,
                        "candidate_sha256": candidate_sha, "checkpoint_sha256": ck, "independent_checks": run.audit,
                        "crosschecked": run.crosschecked, "raw": raw.name, "report_sha256": digest(raw),
                        "render": RENDER, "pci": PCI, "rocr_uuid": UUID}
                write(raw.with_suffix(".audit.json"), info)
                print(raw.name, result.get("stage"), result.get("matched"), result.get("total"),
                      result.get("error", ""), flush=True)
                if not result.get("ok"):
                    return 1
                assert result["stage"] == "compare" and result["criterion"] == "exact"
                assert run.crosschecked and all(result[k] == run.audit[k] for k in IDENTITY)
                assert candidate_sha == run.audit["candidate_sha256"] == digest(source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
