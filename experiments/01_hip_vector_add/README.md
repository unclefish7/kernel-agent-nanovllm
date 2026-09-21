# 实验 01：手写 HIP 向量加法

本实验用一个最小程序认识 HIP kernel 的完整执行过程：CPU 准备数据并分配
GPU 显存，将输入复制到 GPU，启动并行 kernel，最后把结果复制回 CPU 验证。

核心计算只有一行：

```cpp
c[i] = a[i] + b[i];
```

进入开发容器后运行：

```bash
cd /workspace/01_hip_vector_add
hipcc vector_add.hip -O2 -o vector_add
./vector_add
```

预期最后输出 `检查通过`。编译产生的 `vector_add` 是本地实验产物，可以用
下面的命令删除：

```bash
rm /workspace/01_hip_vector_add/vector_add
```

其中 `threadIdx.x` 是线程在线程块内的编号，`blockIdx.x` 是线程块编号，
`blockDim.x` 是每个线程块的线程数。三者共同计算当前线程负责的数组下标。
