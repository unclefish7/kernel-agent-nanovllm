# 实验 08：Softmax Latency Benchmark

本实验比较同一个 `float32` Softmax 的三种实现：

- `torch.softmax`；
- 实验 06 的 Triton 二维 Tile 实现；
- 实验 07 的 PyTorch HIP 扩展。

计时前先进行 warmup，随后使用 `torch.cuda.Event` 测量 GPU stream 上 200 次
执行的平均延迟。ROCm 版 PyTorch 继续沿用 `torch.cuda` 命名。

先确保实验 07 的扩展已经构建，然后运行：

```bash
cd /workspace/08_softmax_benchmark
TRITON_CACHE_DIR=/workspace/08_softmax_benchmark/.triton_cache \
python benchmark.py
```

第一次调用 Triton 时可能触发 JIT，但它发生在 correctness 和 warmup 阶段，
不计入正式计时。结果只用于当前 MI100、当前软件版本和当前 shape 的比较，不能
直接推广到其他硬件或输入。

Triton JIT 缓存可以这样清理：

```bash
rm -rf /workspace/08_softmax_benchmark/.triton_cache
```

## 当前结果

测试环境为 AMD Instinct MI100（`gfx908`）、PyTorch `2.12.0+rocm10.0.0`、
Triton `3.8.0`。每项 warmup 20 次，正式测量 200 次平均延迟：

| shape | PyTorch | Triton | HIP |
|---|---:|---:|---:|
| `(3, 17, 200)` | 6.86 us | 32.87 us | 10.78 us |
| `(16, 128, 200)` | 7.38 us | 32.51 us | 12.34 us |
| `(32, 256, 256)` | 21.29 us | 40.78 us | 67.25 us |

这些是教学实现的首次结果。Triton 与 HIP kernel 尚未调优，重复运行时也可能因
GPU 状态产生小幅波动。
