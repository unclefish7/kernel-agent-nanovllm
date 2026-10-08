# 实验 05：Triton Block Sum

本实验与 `02_hip_block_reduction` 对照：每个 Triton Program 读取一个包含
256 个元素的 Tile，并输出一个局部和。

HIP 版本需要显式编写 LDS、树形归约和 Block 同步；Triton 版本只描述块级
数据和归约操作：

```python
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
values = tl.load(input_ptr + offsets, mask=offsets < n, other=0.0)
block_sum = tl.sum(values, axis=0)
tl.store(output_ptr + program_id, block_sum)
```

进入容器后运行：

```bash
cd /workspace/05_triton_block_sum
TRITON_CACHE_DIR=/workspace/05_triton_block_sum/.triton_cache python block_sum.py
```

第一次运行会 JIT 编译 kernel，并在 `.triton_cache/` 保存本地编译缓存；不会
下载依赖。删除缓存可以强制重新编译：

```bash
rm -rf /workspace/05_triton_block_sum/.triton_cache
```
