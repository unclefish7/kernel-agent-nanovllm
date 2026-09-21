# 实验 02：Block 内合作归约

本实验让每个 Block 使用 LDS 和同步操作，共同计算 256 个输入元素的局部和。
不同 Block 相互独立，GPU 计算出所有局部和后，再由 CPU 合并为最终结果。

执行流程：

```text
全局显存中的输入
        ↓ 每个线程读取一个元素
Block 的 LDS
        ↓ 256 → 128 → 64 → ... → 1
每个 Block 输出一个局部和
        ↓
CPU 合并所有局部和
```

进入开发容器后编译并运行：

```bash
cd /workspace/02_hip_block_reduction
hipcc block_reduction.hip -O2 -o block_reduction
./block_reduction
```

预期最后输出 `检查通过`。编译产物可以删除后重新生成：

```bash
rm /workspace/02_hip_block_reduction/block_reduction
```

这个实现以教学为目的，重点是理解 `__shared__`、`__syncthreads()` 和树形
归约，还没有使用 wavefront shuffle 或在 GPU 上完成跨 Block 的最终归约。
