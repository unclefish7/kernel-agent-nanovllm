# 基础镜像 ID 为 sha256:1a5af9603e9dc4db8a18bd87180760678b654f5260fcc63f1bda8304953e03d5。
# 在仓库根目录准备仅含锁文件和 wheel 的构建上下文。可直接复用散列匹配的缓存 wheel。
#   mkdir -p tmp/explore-worker-build tmp/explore-worker-client/tmp
#   cp nanovllm_selector/port/explore/requirements.lock tmp/explore-worker-build/
# 若需下载，使用 Linux x86_64 的 CPython 3.12 执行：
#   TMPDIR="$PWD/tmp/explore-worker-client/tmp" python -B -m pip --no-cache-dir \
#     --disable-pip-version-check download --only-binary=:all: --no-deps --require-hashes \
#     -r tmp/explore-worker-build/requirements.lock -d tmp/explore-worker-build
#   DOCKER_CONFIG="$PWD/tmp/explore-worker-client" BUILDX_CONFIG="$PWD/tmp/explore-worker-client/buildx" \
#     TMPDIR="$PWD/tmp/explore-worker-client/tmp" docker build --pull=false --network=none \
#     --provenance=false -f nanovllm_selector/port/explore/Dockerfile \
#     -t nanovllm-explore-worker:local tmp/explore-worker-build
FROM jerryjwc/rocm-dev:10.0@sha256:1a5af9603e9dc4db8a18bd87180760678b654f5260fcc63f1bda8304953e03d5

ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=bind,target=/wheelhouse,readonly \
    /opt/pytorch-venv/bin/python -B -m pip install --no-index --no-cache-dir \
        --no-compile --no-deps --only-binary=:all: --require-hashes \
        --find-links=/wheelhouse -r /wheelhouse/requirements.lock

USER 65534:65534
