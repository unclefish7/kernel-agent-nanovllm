# 实验 07：HIP 二维 Tile Softmax

本实验使用与 `06_triton_tiled_softmax` 相同的输入和逻辑 Tile，通过 PyTorch
C++ 扩展调用手写 HIP kernel。

```text
输入：[batch, rows, cols] = [3, 17, 200]
Grid：[batch, row_tiles] = [3, 5]
Block：256 threads
逻辑 Tile：[4, 256]
```

一个 HIP Block 定位一个 `(batch_id, row_tile_id)`，每个线程负责一列并依次
处理 Tile 中的 4 行。每一行显式执行两次 LDS 树形归约：先求最大值，再求
指数之和，最后完成归一化。

在容器中编译并验证：

```bash
cd /workspace/07_pytorch_hip_tiled_softmax
PYTORCH_ROCM_ARCH=gfx908 python setup.py build_ext --inplace
python test_tiled_softmax.py
```

当前教学实现只支持连续的三维 `float32` GPU Tensor，并要求列数不超过 256。
构建产物可以这样清理：

```bash
rm -rf /workspace/07_pytorch_hip_tiled_softmax/build
rm /workspace/07_pytorch_hip_tiled_softmax/hip_tiled_softmax*.so
rm /workspace/07_pytorch_hip_tiled_softmax/tiled_softmax_kernel.hip
```
