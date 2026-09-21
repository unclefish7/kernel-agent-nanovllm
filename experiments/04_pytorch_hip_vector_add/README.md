# 实验 04：PyTorch 调用自定义 HIP Kernel

本实验将 HIP 向量加法包装为 PyTorch C++ 扩展，打通以下调用链：

```text
Python Tensor
    → pybind11 C++ 绑定
    → PyTorch Tensor 检查和输出分配
    → 当前 PyTorch HIP stream 上启动 kernel
    → 返回 Python Tensor
```

`setup.py` 使用 PyTorch 的 `CUDAExtension`。PyTorch 在 ROCm 环境中沿用
`CUDAExtension`、`torch.cuda` 等兼容名称，但实际编译器和运行时是 HIP/ROCm。
由于当前镜像没有安装 `ninja`，这里明确使用 setuptools 后端，不需要额外下载。

在容器中编译并验证：

```bash
cd /workspace/04_pytorch_hip_vector_add
PYTORCH_ROCM_ARCH=gfx908 python setup.py build_ext --inplace
python test_vector_add.py
```

测试将自定义算子与 `a + b` 比较。第一版只支持位于同一 GPU 上、形状相同、
内存连续的 `float32` Tensor，并且只实现前向计算。

编译会生成 `build/`、`hip_vector_add*.so`，以及由 PyTorch 自动转换得到的
`vector_add_kernel.hip`，可以在容器中清理：

```bash
rm -rf /workspace/04_pytorch_hip_vector_add/build
rm /workspace/04_pytorch_hip_vector_add/hip_vector_add*.so
rm /workspace/04_pytorch_hip_vector_add/vector_add_kernel.hip
```
