#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* output,
                                  int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = a[i] + b[i];
    }
}

at::Tensor vector_add_hip(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "输入必须是 GPU Tensor");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "输入必须是 float32 Tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "两个输入的形状必须相同");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "输入必须连续");
    TORCH_CHECK(a.device() == b.device(), "两个输入必须位于同一 GPU");

    c10::cuda::CUDAGuard device_guard(a.device());
    at::Tensor output = at::empty_like(a);
    int64_t n = a.numel();
    if (n == 0) {
        return output;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipStream_t stream = c10::hip::getCurrentHIPStream(a.get_device()).stream();
    vector_add_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
