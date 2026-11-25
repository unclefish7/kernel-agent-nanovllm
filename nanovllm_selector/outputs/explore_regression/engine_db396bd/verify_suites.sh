#!/usr/bin/env bash
# 从仓库根目录运行。只读挂载当前规约、算子、引擎与已有 Qwen 权重。
set -euo pipefail
ulimit -c 0
root=$(pwd -P)
mkdir -p "${1:?需要给出本轮新的日志目录}"
result_dir=$(realpath "${1:?需要给出本轮新的日志目录}")
engine_sha=$(git -C "$root/nano-vllm" rev-parse --verify "${2:?需要引擎 Git 提交}^{commit}")
scratch=$(mktemp -d "$root/tmp/db396bd-suite-XXXXXXXX")
trap 'rm -r -- "$scratch"' EXIT
mkdir -p "$scratch/work/cache/tmp" "$scratch/work/models/Qwen3-0.6B" "$scratch/engine"
git -C "$root/nano-vllm" archive "$engine_sha" nanovllm tests | tar -xC "$scratch/engine"
image=nanovllm-explore-worker:local
gpu_uuid=GPU-6fb1bbd498dc7bd1
common=(--rm --pull=never --network=none --ipc=private --shm-size=512m
  --ulimit core=0:0 --cap-drop=ALL --security-opt=no-new-privileges --read-only --user=1000:1000
  --tmpfs=/tmp:rw,nosuid,nodev,noexec,size=128m --workdir=/workspace
  --mount "type=bind,src=$scratch/engine,dst=/workspace/nano-vllm,readonly"
  --mount "type=bind,src=$root/nanovllm_selector,dst=/workspace/nanovllm_selector,readonly"
  --mount "type=bind,src=$root/contracts,dst=/workspace/contracts,readonly"
  --mount "type=bind,src=$root/kernels,dst=/workspace/kernels,readonly"
  --mount "type=bind,src=$scratch/work,dst=/workspace/tmp"
  --mount "type=bind,src=$root/tmp/models/Qwen3-0.6B,dst=/workspace/tmp/models/Qwen3-0.6B,readonly"
  --mount "type=bind,src=$result_dir,dst=/result"
  -e PYTHONPATH=/workspace:/workspace/nano-vllm:/workspace/nanovllm_selector
  -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4
  -e TMPDIR=/workspace/tmp/cache/tmp -e XDG_CACHE_HOME=/workspace/tmp/cache
  -e TORCH_HOME=/workspace/tmp/cache/torch -e HF_HOME=/workspace/tmp/cache/huggingface
  -e TORCH_EXTENSIONS_DIR=/workspace/tmp/cache/extensions -e TORCHINDUCTOR_CACHE_DIR=/workspace/tmp/cache/inductor
  -e TRITON_CACHE_DIR=/workspace/tmp/cache/triton -e AMD_COMGR_CACHE_DIR=/workspace/tmp/cache/comgr
  -e MIOPEN_USER_DB_PATH=/workspace/tmp/cache/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/workspace/tmp/cache/miopen)
for name in gpu-preflight engine selector project; do test ! -e "$result_dir/$name.log"; done
test ! -e "$result_dir/suite_sources.log"
{
  git -C "$root" rev-parse HEAD
  echo "$engine_sha"
  (cd "$scratch/engine" && sha256sum nanovllm/models/__init__.py nanovllm/utils/loader.py tests/test_model_registry.py)
  (cd "$root" && sha256sum contracts/nanovllm/spec.yaml kernels/index.md)
} > "$result_dir/suite_sources.log"
printf 'scratch=%s\n' "$scratch"
docker run "${common[@]}" --device=/dev/kfd:/dev/kfd:rw --device=/dev/dri/renderD131:/dev/dri/renderD131:rw \
  --group-add=109 -e "ROCR_VISIBLE_DEVICES=$gpu_uuid" "$image" bash -ec '
    ulimit -c 0
    python -B -c "import os,resource,torch; from pathlib import Path; d=Path(\"/sys/class/drm/renderD131/device\"); u=\"GPU-\"+(d/\"unique_id\").read_text().strip(); assert os.environ[\"ROCR_VISIBLE_DEVICES\"]==u and torch.cuda.device_count()==1; p=torch.cuda.get_device_properties(0); pci=\"%04x:%02x:%02x.0\"%(p.pci_domain_id,p.pci_bus_id,p.pci_device_id); assert d.resolve().name==pci==\"0000:27:00.0\"; assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0); print(pci,u,p.name)" > /result/gpu-preflight.log 2>&1
    python -B -m unittest discover -s nano-vllm/tests -v > /result/engine.log 2>&1
    cd /workspace/tmp
    python -B -m unittest -v nanovllm_selector.test_selector > /result/selector.log 2>&1'
docker run "${common[@]}" "$image" bash -ec '
    ulimit -c 0
    python -B -c "import resource,torch; assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0) and torch.cuda.device_count()==0"
    python -B -m unittest -v nanovllm_selector.test_dispatch_program nanovllm_selector.test_extract nanovllm_selector.test_graph_contracts nanovllm_selector.test_mixed_state nanovllm_selector.test_port_iterate nanovllm_selector.test_program_variants nanovllm_selector.test_reconcile_calls nanovllm_selector.test_routing_replay nanovllm_selector.test_shape_probe_topology nanovllm_selector.port.test_diagnose nanovllm_selector.port.test_evaluate_batch_order nanovllm_selector.port.test_evaluate_reference nanovllm_selector.port.test_port nanovllm_selector.port.test_prompt_contracts > /result/project.log 2>&1'
