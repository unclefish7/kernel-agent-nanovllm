#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cmath>

constexpr int kBlockM = 4;
constexpr int kThreads = 256;

__global__ void tiled_softmax_kernel(const float* input, float* output,
                                     int64_t row_count, int64_t col_count) {
    __shared__ float shared_data[kThreads];

    int64_t batch_id = blockIdx.x;
    int64_t row_tile_id = blockIdx.y;
    int tid = threadIdx.x;
    int64_t first_row = row_tile_id * kBlockM;
    int64_t batch_offset = batch_id * row_count * col_count;

    for (int local_row = 0; local_row < kBlockM; ++local_row) {
        int64_t row = first_row + local_row;
        if (row >= row_count) {
            continue;
        }

        int64_t offset = batch_offset + row * col_count + tid;
        float value = tid < col_count ? input[offset] : -INFINITY;
        shared_data[tid] = value;
        __syncthreads();

        for (int stride = kThreads / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_data[tid] =
                    fmaxf(shared_data[tid], shared_data[tid + stride]);
            }
            __syncthreads();
        }

        float row_max = shared_data[0];
        __syncthreads();

        float numerator = tid < col_count ? expf(value - row_max) : 0.0f;
        shared_data[tid] = numerator;
        __syncthreads();

        for (int stride = kThreads / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }

        float row_sum = shared_data[0];
        __syncthreads();

        if (tid < col_count) {
            output[offset] = numerator / row_sum;
        }
    }
}

at::Tensor tiled_softmax_hip(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "输入必须是 GPU Tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "输入必须是 float32 Tensor");
    TORCH_CHECK(input.is_contiguous(), "输入必须连续");
    TORCH_CHECK(input.dim() == 3, "输入形状必须是 [batch, rows, cols]");
    TORCH_CHECK(input.size(1) > 0 && input.size(2) > 0,
                "rows 和 cols 必须大于 0");
    TORCH_CHECK(input.size(2) <= kThreads, "当前实验最多支持 256 列");

    c10::cuda::CUDAGuard device_guard(input.device());
    at::Tensor output = at::empty_like(input);

    int64_t batch_size = input.size(0);
    int64_t row_count = input.size(1);
    int64_t col_count = input.size(2);
    dim3 grid(static_cast<unsigned int>(batch_size),
              static_cast<unsigned int>((row_count + kBlockM - 1) / kBlockM));
    dim3 block(kThreads);
    hipStream_t stream =
        c10::hip::getCurrentHIPStream(input.get_device()).stream();

    tiled_softmax_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), row_count, col_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
