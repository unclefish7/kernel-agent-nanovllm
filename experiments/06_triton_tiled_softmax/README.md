# 实验 06：二维 Tile Softmax

本实验使用二维 Grid 和二维 Tile 计算三维 Tensor 最后一维上的 Softmax。

```text
输入：[batch, rows, cols] = [3, 17, 200]
Grid：[batch, row_tiles] = [3, 5]
Tile：[BLOCK_M, BLOCK_N] = [4, 256]
```

每个 Program 通过 `(batch_id, row_tile_id)` 定位一个二维 Tile，并沿列方向
连续完成两次 Reduction：

```text
row_max = max(values, axis=1)
exp_values = exp(values - row_max)
row_sum = sum(exp_values, axis=1)
output = exp_values / row_sum
```

`BLOCK_N=256` 覆盖实际的 200 列，多余位置通过 mask 填入负无穷，尾部不足
4 行的 Tile 也使用同一个二维 mask 防止越界。

进入容器后运行：

```bash
cd /workspace/06_triton_tiled_softmax
TRITON_CACHE_DIR=/workspace/06_triton_tiled_softmax/.triton_cache \
python tiled_softmax.py
```

第一次运行生成的 JIT 缓存可以删除后重新生成：

```bash
rm -rf /workspace/06_triton_tiled_softmax/.triton_cache
```
