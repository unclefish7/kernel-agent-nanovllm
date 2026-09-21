# 实验 00：环境检查

本实验不做性能优化，只检查后续算子实验所需的最小环境：

1. 容器内能看到 `/dev/kfd` 和 `/dev/dri/render*`；
2. `rocminfo`、`rocm-smi` 和 `hipcc` 可执行；
3. ROCm 版 PyTorch 已安装，并且能识别 GPU；
4. PyTorch 能在 GPU 上完成矩阵乘法；
5. `F.silu(gate) * up` 能执行，并与等价公式结果一致。

从上一级 `experiments/` 目录启动并进入开发容器：

```bash
docker compose up -d
docker compose exec rocm-dev /usr/bin/zsh
```

进入容器后手动运行：

```bash
python3 check_environment.py
```

脚本成功时退出码为 `0`；任何必要检查失败时退出码为 `1`。脚本只读取环境
信息并执行小规模 GPU 计算，不进行网络访问，也不下载任何文件。

MI100 的目标架构通常显示为 `gfx908`。脚本会打印实际设备名称和架构信息，
但不会仅因设备不是 MI100 而直接失败，以便在其他 AMD GPU 上也能进行诊断。
