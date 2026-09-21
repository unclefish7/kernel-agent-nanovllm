# 实验 03：多阶段 GPU Reduction

本实验重复启动同一个 Block 归约 kernel，将上一个阶段的 GPU 输出作为下一个
阶段的输入，直到只剩一个结果。CPU 最后只从 GPU 复制一个 `float`。

对于 `1,048,576` 个输入元素，归约过程是：

```text
1,048,576 → 4,096 → 16 → 1
```

两个中间 buffer 交替作为输入和输出，避免一个 kernel 在读取输入的同时覆盖它：

```text
d_input → d_buffer_a → d_buffer_b → d_buffer_a
```

进入开发容器后编译并运行：

```bash
cd /workspace/03_hip_multistage_reduction
hipcc multistage_reduction.hip -O2 -o multistage_reduction
./multistage_reduction
```

预期最后输出 `检查通过`。编译产物可以删除后重新生成：

```bash
rm /workspace/03_hip_multistage_reduction/multistage_reduction
```

同一默认 stream 中的 kernel 按提交顺序执行，因此下一阶段会在上一阶段完成后读取
中间结果，不需要在每两个 kernel 之间调用 `hipDeviceSynchronize()`。
