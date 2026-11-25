"""探查与评测的宿主入口。容器只读挂载输入，产物经有界复制后在宿主发布。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import selectors
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "/opt/pytorch-venv/bin/python"
PACKAGE = "nanovllm_selector.port.explore"
MAX_ARTIFACT_BYTES = 256 << 20
MAX_ENTRIES = 16400
MAX_REPLY = 4 << 20
MAX_DIAGNOSE_LENGTH = 1024
MAX_DIAGNOSE_REPLY = 64 << 10
TRIAL_LIMITS = {"engine.json": 64 << 10, "engine.safetensors": 64 << 20}
HIP_DRIVER_MAP_ERROR = re.compile(
    r"^(?:torch\._inductor\.exc\.)?InductorError: ImportError: /(?:tmp|jit)/triton/[\w-]+/"
    r"hip_utils\.cpython-[0-9]+-x86_64-linux-gnu\.so: failed to map segment from shared object$",
    re.MULTILINE,
)
MODEL_SUFFIXES = {".json", ".safetensors", ".txt", ".model", ".spm", ".bpe", ".vocab",
                  ".tiktoken", ".jinja", ".jinja2"}

# 在能访问全部卡的宿主上运行。部分调用环境只有 Docker 客户端，可用既有的可信容器查询。
DISCOVER_GPU = """
import json, os, pathlib, stat, subprocess, sys
cards = json.loads(subprocess.check_output(["rocm-smi", "--showbus", "--json"], timeout=12))
pci = cards["card" + sys.argv[1]]["PCI Bus"].lower()
matches = [p for p in pathlib.Path("/sys/class/drm").glob("renderD*")
           if (p / "device").resolve().name == pci and (p / "device/vendor").read_text().strip() == "0x1002"]
if len(matches) != 1:
    raise ValueError(f"PCI {pci} 对应 {len(matches)} 个 AMD render 节点")
node = matches[0]
render = pathlib.Path("/dev/dri") / node.name
devices = [os.stat("/dev/kfd"), render.stat()]
if not all(stat.S_ISCHR(s.st_mode) for s in devices):
    raise ValueError("需要 KFD 和 DRM 字符设备")
major, minor = map(int, (node / "dev").read_text().split(":"))
if devices[1].st_rdev != os.makedev(major, minor):
    raise ValueError("DRM 设备号与 sysfs 不符")
print(json.dumps([str(render), pci, sorted({s.st_gid for s in devices})]))
"""

# 在同一容器中先核实真实可见设备，再运行 worker。可见性环境变量会掩盖额外设备。
CHECK_GPU = """
import pathlib, sys
render, pci = sys.argv[1:]
nodes = list(pathlib.Path("/dev/dri").glob("renderD*"))
if [str(p) for p in nodes] != [render] or (pathlib.Path("/sys/class/drm") / nodes[0].name / "device").resolve().name != pci:
    raise RuntimeError("容器 DRM 节点与所选 PCI 不符")
import torch
if torch.cuda.device_count() != 1:
    raise RuntimeError("容器必须恰好可用一张 GPU")
props = torch.cuda.get_device_properties(0)
actual = (props.pci_domain_id, props.pci_bus_id, props.pci_device_id)
if actual != tuple(int(part, 16) for part in pci.split(".")[0].split(":")):
    raise RuntimeError("PyTorch 与宿主选择的 PCI 不符")
"""

# 导出前结束同一私有 PID 命名空间中的后台任务。保留作为 PID 1 的 sleep 和它的 tmpfs。
STOP_CHILDREN = """
import os, pathlib, signal
if pathlib.Path("/proc/1/comm").read_text().strip() != "sleep":
    raise RuntimeError("未找到容器的 sleep 主进程")
try:
    os.kill(-1, signal.SIGKILL)
except ProcessLookupError:
    pass
"""

# 原有诊断在同一进程挂 hook 比较层边界，进度发往 stderr，协议只输出一行 JSON。
DIAGNOSE = """
import json, os, sys
from contextlib import redirect_stdout
from pathlib import Path
os.environ.update(GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo")
with redirect_stdout(sys.stderr):
    from nanovllm_selector.port.diagnose import diagnose
    report = diagnose(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
"""


def _bounded_command(command, env, deadline, *, stdin=None, output=None, limit=MAX_REPLY):
    """并行排空两个管道；只留有限的输出及最后的错误。容器寿命由调用方另外管理。"""
    stdout = io.BytesIO() if output is None else output
    stderr = bytearray()
    size = 0
    with subprocess.Popen(command, stdin=stdin if stdin else subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as proc:
        try:
            with selectors.DefaultSelector() as streams:
                for stream in (proc.stdout, proc.stderr):
                    streams.register(stream, selectors.EVENT_READ)
                while streams.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"命令超时：{command[0]} {command[1]}")
                    for key, _ in streams.select(remaining):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            streams.unregister(key.fileobj)
                        elif key.fileobj is proc.stderr:
                            stderr.extend(chunk)
                            del stderr[:-8192]
                        else:
                            size += len(chunk)
                            if size > limit:
                                raise ValueError(f"输出超过 {limit} 字节")
                            stdout.write(chunk)
            rc = proc.wait(timeout=max(0, deadline - time.monotonic()))
        finally:
            if proc.poll() is None:
                proc.kill()
    return rc, stdout.getvalue() if output is None else b"", bytes(stderr)


def _last_json(rc, stdout, stderr):
    lines = stdout.rstrip().rsplit(b"\n", 1)
    try:
        result = json.loads(lines[-1])
        if isinstance(result, dict) and (rc == 0 or result.get("ok") is False):
            json.dumps(result, allow_nan=False)
            return result
    except (ValueError, RecursionError):
        pass
    detail = (stderr.strip() or stdout.strip())[-2000:].decode(errors="replace")
    raise RuntimeError(f"工作进程退出码 {rc}，没有有效结果。{detail}")


def _has_artifacts(result):
    return result.get("ok") is True or result.get("status") == "unreliable"


def _receive_files(stream, dest):
    """在全新的宿主目录下只重建常规文件；忽略归档的属主与权限。"""
    dest.mkdir()
    dest.chmod(0o755)
    seen = set()
    size = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if (name.is_absolute() or ".." in name.parts or len(name.parts) > 8
                    or len(item.name) > 512 or not item.name.isprintable()
                    or name in seen or len(seen) >= MAX_ENTRIES):
                raise ValueError("归档路径越界、重复或数量超限")
            seen.add(name)
            path = dest / name
            if item.type == tarfile.DIRTYPE:
                path.mkdir(parents=True, exist_ok=True)
                path.chmod(0o755)
                continue
            size += item.size
            if (not name.parts or item.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or item.size < 0 or size > MAX_ARTIFACT_BYTES):
                raise ValueError("产物必须是大小受限的独立常规文件")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            path.chmod(0o444)


def _bind(source, destination):
    # Docker 把 --mount 解析为 CSV，路径中的逗号也必须是普通字符。
    value = io.StringIO()
    csv.writer(value, lineterminator="").writerow(
        ["type=bind", f"src={source}", f"dst={destination}", "readonly", "bind-recursive=disabled"])
    return "--mount=" + value.getvalue()


def _package_sources(job):
    # 工作树中的源码可能是 0600。只暂存 Python 源码，供容器的非特权 UID 读取。
    source = ROOT / "nanovllm_selector"
    dest = job / "nanovllm_selector"
    for package in ("extract", "port", "port/explore"):
        directory = dest / package
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o755)
        for path in (source / package).glob("*.py"):
            if not path.name.startswith("test_") and not path.is_symlink() and path.is_file():
                target = directory / path.name
                shutil.copyfile(path, target)
                target.chmod(0o444)
    dest.chmod(0o755)
    return dest


def _model_mounts(model, job):
    """只暴露本地配置、分词器及权重。HF 缓存快照的链接按实际 blob 挂载。"""
    mounts = []
    blobs = model.parent.parent / "blobs"
    for directory, dirs, files in os.walk(model):
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (Path(directory) / d).is_symlink()]
        for filename in sorted(files):
            path = Path(directory) / filename
            if filename.startswith(".") or path.suffix not in MODEL_SUFFIXES:
                continue
            src = path.resolve(strict=True)
            hf_blob = (model.parent.name == "snapshots" and src.parent == blobs
                       and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", src.name))
            if not hf_blob and (not src.is_relative_to(model) or src.suffix not in MODEL_SUFFIXES
                                or any(p.startswith(".") for p in src.relative_to(model).parts)):
                raise ValueError(f"模型数据链接越界：{filename}")
            info = src.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"模型数据必须是独立常规文件：{filename}")
            relative = path.relative_to(model)
            if not info.st_mode & stat.S_IROTH and info.st_size <= 32 << 20:
                readable = job / "model-metadata" / relative
                if not readable.exists():
                    readable.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, readable)
                    readable.chmod(0o444)
                src = readable
            mounts.append(_bind(src, "/model/" + relative.as_posix()))
    return mounts


def _safe_dir(path, *, worker_reads=False):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("输出目录必须在仓库内且路径不能经过链接")
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        if not path.is_dir():
            raise
    else:
        if worker_reads:
            path.chmod(0o755)
    return path


def _candidate_bytes(candidate):
    path = Path(candidate).absolute()
    if path.suffix != ".py" or path != path.resolve(strict=True):
        raise ValueError("候选必须是独立的 .py 文件")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 2 << 20:
            raise ValueError("候选必须是最多 2 MiB 的独立常规文件")
        data = file.read((2 << 20) + 1)
        if len(data) != info.st_size:
            raise ValueError("读取候选时文件大小变化")
        return data


def _write_snapshot(job, source):
    snapshot = job / "candidate.py"
    snapshot.write_bytes(source)
    snapshot.chmod(0o444)
    return snapshot


def _transfer_digest(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 < before.st_size <= TRIAL_LIMITS[path.name]):
            raise ValueError(f"无效或过大的引擎输出 {path.name}")
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            if size > before.st_size:
                raise ValueError(f"校验期间 {path.name} 增长")
            digest.update(chunk)
        if size != before.st_size or os.fstat(stream.fileno()).st_ctime_ns != before.st_ctime_ns:
            raise ValueError(f"校验期间 {path.name} 发生变化")
        return digest.hexdigest()


def _check_identity(phase, report, expected, *, partial=False):
    for key, value in expected.items():
        if partial and key not in report:
            continue
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"{phase} 返回的 {key} 与本次评测不符")


def _attest_trial(trial, snapshot, model, digest, holdout, capture):
    """引擎退出后，宿主核对冻结候选、本地模型和两份引擎输出，在本次交接目录写入摘要。"""
    if hashlib.sha256(_candidate_bytes(snapshot)).hexdigest() != digest:
        raise ValueError("候选快照的 candidate_sha256 在引擎运行期间发生变化")
    _check_identity("engine", capture, {"candidate_sha256": digest})
    if trial.is_symlink() or {path.name for path in trial.iterdir()} != TRIAL_LIMITS.keys():
        raise ValueError("引擎必须仅提交 engine.json 和 engine.safetensors")
    metadata = _transfer_digest(trial / "engine.json")
    logits = _transfer_digest(trial / "engine.safetensors")
    from .judge import checkpoint_sha256

    binding = {"candidate_sha256": digest, "checkpoint_sha256": checkpoint_sha256(model),
               "holdout": holdout, "engine_json_sha256": metadata, "logits_sha256": logits}
    _check_identity("engine", capture, binding)
    request = capture.get("request_sha256")
    if not isinstance(request, str) or not re.fullmatch(r"[0-9a-f]{64}", request):
        raise ValueError("engine 返回的 request_sha256 缺失或无效")
    path = trial / "binding.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, **binding}, stream, separators=(",", ":"))
    path.chmod(0o444)
    return {**binding, "request_sha256": request}


class DockerRuntime:
    """每次调用新建容器。device 为 rocm-smi 物理卡号，默认 0；cpu 仅供 CPU 工具使用。"""

    def __init__(self, model_dir: Path, out_dir: Path, *, image: str,
                 device: str | None = None, timeout: float = 600):
        self.model_dir = Path(model_dir).resolve(strict=True)
        if not (self.model_dir / "config.json").is_file() or not self.model_dir.is_dir():
            raise ValueError("模型目录必须含有 config.json")
        if not image or image.startswith("-") or any(c.isspace() for c in image):
            raise ValueError("无效的本地镜像名")
        self.device = "0" if device is None else str(device)
        if self.device != "cpu" and not re.fullmatch(r"0|[1-9][0-9]*", self.device):
            raise ValueError("device 需要 rocm-smi 物理卡号或 cpu")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("超时必须为正数")
        out_dir = Path(out_dir).absolute()
        if out_dir.is_relative_to(self.model_dir) or ROOT.is_relative_to(self.model_dir):
            raise ValueError("模型挂载不能包含宿主输出或仓库根目录")
        self.out_dir = _safe_dir(out_dir)
        self.image = image
        self.timeout = timeout
        self.last_error = None
        self._gpu = None

    def _device_options(self, env, deadline):
        if self.device == "cpu":
            return []
        if self._gpu is None:
            if Path("/dev/kfd").is_char_device():
                command = [sys.executable]
            else:
                helper = os.environ.get("NANOVLLM_ROCM_DISCOVERY_CONTAINER", "nanovllm-selector-rocm-dev")
                command = ["docker", "exec", "-e", "PYTHONDONTWRITEBYTECODE=1", helper, PYTHON]
            rc, out, err = _bounded_command(command + ["-IB", "-c", DISCOVER_GPU, self.device],
                                            env, min(deadline, time.monotonic() + 20))
            if rc:
                raise RuntimeError("无法按 PCI 确定物理卡：" + err.decode(errors="replace")[-1500:])
            render, pci, groups = json.loads(out)
            if (not re.fullmatch(r"/dev/dri/renderD[0-9]+", render)
                    or not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci)
                    or not all(type(g) is int and g >= 0 for g in groups)):
                raise ValueError("无效的 ROCm 设备映射")
            self._gpu = render, pci, groups
        render, _, groups = self._gpu
        return ["--device=/dev/kfd:/dev/kfd:rw", f"--device={render}:{render}:rw",
                *(f"--group-add={gid}" for gid in groups)]

    def _invoke(self, job, command, *, request=None, mounts=(), scratch=None, collect=None, reply_limit=MAX_REPLY):
        deadline = time.monotonic() + self.timeout
        env = {"PATH": os.environ["PATH"], "HOME": str(job), "DOCKER_CONFIG": str(job / "docker"),
               "TMPDIR": str(job), "PYTHONDONTWRITEBYTECODE": "1"}
        if host := os.environ.get("DOCKER_HOST"):
            if not host.startswith("unix://"):
                raise ValueError("工作目录挂载需要本机 Docker Unix socket")
            env["DOCKER_HOST"] = host
        name = "nanovllm-explore-" + uuid.uuid4().hex
        device_options = self._device_options(env, deadline)
        sources = job / "nanovllm_selector"
        if not sources.exists():
            sources = _package_sources(job)
        options = ["docker", "run", "--detach", "--rm", "--pull=never", "--name=" + name,
                   "--network=none", "--ipc=private", "--shm-size=512m", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
                   "--no-healthcheck", "--log-driver=none", "--pids-limit=512", "--memory=16g",
                   "--memory-swap=16g", "--ulimit=core=0:0", "--workdir=/workspace",
                   "--tmpfs=/tmp:rw,noexec,nosuid,nodev,mode=1777,size=2g,nr_inodes=65536",
                   "--tmpfs=/jit:rw,exec,nosuid,nodev,mode=0700,uid=65534,gid=65534,size=1g,nr_inodes=32768",
                   *device_options, *_model_mounts(self.model_dir, job),
                   _bind(sources, "/workspace/nanovllm_selector"),
                   _bind(ROOT / "nano-vllm/nanovllm", "/workspace/nano-vllm/nanovllm")]
        options.extend(_bind(src, dst) for src, dst in mounts)
        if scratch:
            quota = "size=66m,nr_inodes=64" if scratch == "/trial" else "size=320m,nr_inodes=16512"
            options.append(f"--tmpfs={scratch}:rw,nosuid,nodev,noexec,uid=65534,gid=65534,"
                           f"mode=0700,{quota}")
        for setting in ("HOME=/tmp", "TMPDIR=/tmp", "XDG_CACHE_HOME=/tmp/cache",
                        "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
                        "PYTHONPATH=/workspace:/workspace/nano-vllm", "HF_HOME=/tmp/huggingface",
                        "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
                        "TOKENIZERS_PARALLELISM=false", "TRITON_CACHE_DIR=/jit/triton",
                        "TORCHINDUCTOR_CACHE_DIR=/jit/inductor", "TORCH_EXTENSIONS_DIR=/jit/extensions",
                        "AMD_COMGR_CACHE_DIR=/tmp/comgr",
                        "OMP_NUM_THREADS=1", "CUBLAS_WORKSPACE_CONFIG=:4096:8"):
            options.append("--env=" + setting)

        def checked(argv, **kwargs):
            rc, _, err = _bounded_command(argv, env, deadline, **kwargs)
            if rc:
                raise RuntimeError(f"{argv[0]} {argv[1]} 退出码 {rc}。{err.decode(errors='replace')[-2000:]}")

        try:
            checked(options + ["--entrypoint=/bin/sleep", self.image, str(math.ceil(self.timeout) + 30)])
            if self._gpu:
                checked(["docker", "exec", name, PYTHON, "-IB", "-c", CHECK_GPU, *self._gpu[:2]])
            with tempfile.TemporaryFile(dir=job) as payload:
                if request is not None:
                    raw = json.dumps(request, allow_nan=False).encode()
                    if len(raw) > 1 << 20:
                        raise ValueError("工具请求超过 1 MiB")
                    payload.write(raw)
                    payload.seek(0)
                result = _last_json(*_bounded_command(["docker", "exec", "-i", name, *command],
                                                       env, deadline, stdin=payload, limit=reply_limit))
            if collect is not None and _has_artifacts(result):
                checked(["docker", "exec", name, PYTHON, "-IB", "-c", STOP_CHILDREN])
                archive_limit = MAX_ARTIFACT_BYTES + MAX_ENTRIES * 1024 + 10240
                if scratch == "/trial":
                    archive_limit = sum(TRIAL_LIMITS.values()) + 65536
                with tempfile.TemporaryFile(dir=job) as archive:
                    checked(["docker", "exec", name, "/bin/tar", "--format=ustar", "--one-file-system",
                             "-C", scratch, "-cf", "-", "."], output=archive, limit=archive_limit)
                    archive.seek(0)
                    _receive_files(archive, collect)
            return result
        finally:
            rc, _, err = _bounded_command(["docker", "rm", "--force", "--volumes", name],
                                          env, time.monotonic() + 12)
            if rc and b"No such container" not in err:
                raise RuntimeError(f"回收容器 {name} 失败。{err.decode(errors='replace')[-1000:]}")

    def _failure(self, exc):
        self.last_error = str(exc)[-2000:]
        return {"ok": False, "stage": "runtime", "outcome": "evaluation_error", "error": self.last_error}

    def call(self, name: str, args: dict) -> dict:
        self.last_error = None
        try:
            if name not in ("read_code", "run_model", "observe", "compare") or not isinstance(args, dict):
                raise ValueError("无效的 worker 工具名或参数")
            runs = _safe_dir(self.out_dir / "runs", worker_reads=True)
            command = [PYTHON, "-B", "-m", PACKAGE + ".worker", "--model", "/model", "--runs", "/runs"]
            arguments = dict(args)
            with tempfile.TemporaryDirectory(prefix=".worker-", dir=self.out_dir) as temp:
                job = Path(temp)
                mounts = [(runs, "/runs")]
                request = {"tool": name, "arguments": arguments}
                if name != "run_model":
                    result = self._invoke(job, command, request=request, mounts=mounts)
                else:
                    target = arguments.get("target")
                    if target not in ("hf", "probe") or (target == "hf" and "probe" in arguments):
                        raise ValueError("探针只能在 probe 运行中执行")
                    run_id = f"{target}-{uuid.uuid4().hex}"
                    arguments["run_id"] = run_id
                    destination = runs / run_id
                    destination.mkdir()
                    published = False
                    try:
                        collected = job / "received"
                        result = self._invoke(job, command, request=request, mounts=mounts,
                                              scratch="/runs/" + run_id, collect=collected)
                        if not isinstance(result.get("ok"), bool):
                            raise ValueError("run_model 缺少运行状态")
                        if _has_artifacts(result):
                            manifest = collected / "manifest.json"
                            if manifest.stat().st_size > 8 << 20:
                                raise ValueError("运行清单超过 8 MiB")
                            data = json.loads(manifest.read_bytes())
                            if data["run_id"] != run_id or data["provenance"]["target"] != target:
                                raise ValueError("运行清单与宿主分配的来源不符")
                            collected.replace(destination)
                            published = True
                        result.update(run_id=run_id, target=target)
                    finally:
                        if not published:
                            destination.rmdir()
                if result.get("ok") is False:
                    self.last_error = str(result.get("error", result.get("stage", "worker")))
                return result
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, tarfile.TarError,
                subprocess.SubprocessError) as exc:
            return self._failure(exc)

    def diagnose(self, candidate: Path, length: int = 128) -> dict:
        """在单独的候选进程中比较一次 prefill 的层边界，诊断信息供数值排查使用。"""
        self.last_error = None
        header = {"purpose": "prefill_diagnostic", "length": length, "trusted_for_acceptance": False}
        try:
            source = _candidate_bytes(candidate)
            header["candidate_sha256"] = hashlib.sha256(source).hexdigest()
            if type(length) is not int or not 1 <= length <= MAX_DIAGNOSE_LENGTH:
                raise ValueError(f"诊断长度必须是 1 到 {MAX_DIAGNOSE_LENGTH} 之间的整数")
            with tempfile.TemporaryDirectory(prefix=".diagnose-", dir=_safe_dir(self.out_dir)) as temp:
                job = Path(temp)
                snapshot = _write_snapshot(job, source)
                report = self._invoke(job, [PYTHON, "-B", "-c", DIAGNOSE, "/model", "/candidate.py", str(length)],
                                      mounts=[(snapshot, "/candidate.py")], reply_limit=MAX_DIAGNOSE_REPLY)
            if "error" in report:
                raise ValueError(str(report["error"])[-2000:])
            _check_identity("diagnose", report, {k: header[k] for k in ("purpose", "length", "candidate_sha256")})
            if not isinstance(report.get("logits"), dict):
                raise ValueError("诊断报告缺少输出 logits 的比较信息")
            fields = ("candidate", "token_ids_sha256", "compared", "first_difference", "unpaired", "logits")
            return {**header, **{k: report[k] for k in fields if k in report}}
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            self.last_error = str(exc)[-2000:]
            return {**header, "error": self.last_error}

    def evaluate(self, candidate: Path, *, holdout: bool = False) -> dict:
        self.last_error = None
        digest = None
        try:
            model = self.model_dir
            source = _candidate_bytes(candidate)
            digest = hashlib.sha256(source).hexdigest()
            with tempfile.TemporaryDirectory(prefix=".judge-", dir=_safe_dir(self.out_dir)) as temp:
                job = Path(temp)
                snapshot = _write_snapshot(job, source)
                trial = job / "trial"
                judge = [PYTHON, "-B", "-m", PACKAGE + ".judge"]
                options = ["--model", "/model", "--trial", "/trial", "--candidate-sha256", digest]
                if holdout:
                    options.append("--holdout")
                result = self._invoke(job, [*judge, "engine", *options, "--candidate", "/candidate.py"],
                                      mounts=[(snapshot, "/candidate.py")], scratch="/trial", collect=trial)
                if result.get("ok") is True and result.get("stage") == "engine":
                    expected = _attest_trial(trial, snapshot, model, digest, holdout, result)
                    result = self._invoke(job, [*judge, "reference", *options, "--binding", "/trial/binding.json"],
                                          mounts=[(trial, "/trial")])
                    # 参照在早期失败时仅返回已验证的身份；接受数值判断前要求全部字段一致。
                    early_error = result.get("ok") is False and result.get("stage") != "compare"
                    try:
                        _check_identity("reference", result, expected, partial=early_error)
                    except ValueError as exc:
                        failure = self._failure(exc)
                        failure["expected_candidate_sha256"] = digest
                        if "candidate_sha256" in result:
                            failure["candidate_sha256"] = result["candidate_sha256"]
                        return failure
                else:
                    input_error = result.get("ok") is False and result.get("stage") == "input"
                    error = str(result.get("error", "引擎未完成"))
                    driver_error = (result.get("ok") is False and result.get("stage") == "engine"
                                    and HIP_DRIVER_MAP_ERROR.search(error))
                    result = {"ok": False, "stage": "input" if input_error else "engine",
                              "outcome": "evaluation_error" if input_error or driver_error else "candidate_failed",
                              "candidate_sha256": digest, "error": error[-2000:]}
                if result.get("ok") is not True:
                    self.last_error = str(result.get("error", result.get("stage", "reference")))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, tarfile.TarError,
                subprocess.SubprocessError) as exc:
            result = self._failure(exc)
            if digest is not None:
                result["candidate_sha256"] = digest
        return result
