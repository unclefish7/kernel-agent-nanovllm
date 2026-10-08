# 算子实验

本目录用于存放彼此独立的小实验。每个子目录只回答一个明确问题，公共的
Docker Compose 配置放在本目录根部。

当前实验：

- `00_environment_check/`：检查容器能否访问 AMD GPU，以及 ROCm、PyTorch
  和最基本的 GPU 张量计算是否正常。
- `01_hip_vector_add/`：用 `hipcc` 编译并运行第一个手写 HIP kernel。
- `02_hip_block_reduction/`：使用 LDS 和同步完成 Block 内合作归约。
- `03_hip_multistage_reduction/`：通过多个顺序执行的 kernel 完成全 GPU 归约。
- `04_pytorch_hip_vector_add/`：将手写 HIP kernel 包装为 PyTorch 自定义扩展。
- `05_triton_block_sum/`：使用 Triton Program 和 `tl.sum` 完成块内归约。
- `06_triton_tiled_softmax/`：用二维 Grid、二维 Tile 和两次归约实现 Softmax。
- `07_pytorch_hip_tiled_softmax/`：用 HIP Block、LDS 和两次归约实现相同 Softmax。
- `08_softmax_benchmark/`：比较 PyTorch、Triton 和 HIP Softmax 的 GPU 延迟。

## 本地镜像要求

Compose 只使用本地镜像 `jerryjwc/rocm-dev:10.0`，并设置了
`pull_policy: never`，不会从网络拉取镜像。该镜像应由现有的
`rocm-dev-template` 构建，并且需要包含 ROCm 版 PyTorch 才能通过全部检查。

## 启动交互式开发容器

在本目录执行：

```bash
docker compose up -d
```

宿主机上的整个 `experiments/` 目录会以 bind mount 的方式挂载到容器的
`/workspace`。修改宿主机代码后，不需要重新构建镜像。

进入正在运行的容器：

```bash
docker compose exec rocm-dev /usr/bin/zsh
```

进入后默认位于 `/workspace/00_environment_check`，手动运行第一个实验：

```bash
python3 check_environment.py
```

暂时结束实验时停止容器：

```bash
docker compose stop
```

停止的容器可以再次启动：

```bash
docker compose start
```

确定不再使用该容器时删除容器和 Compose 网络：

```bash
docker compose down
```

这些命令不会删除宿主机的 `experiments/` 源代码。实验脚本不会主动下载
模型、Python 包或其他网络资源。容器保留默认网络能力，后续如需下载任何
内容，应先明确下载路径和预计影响。
