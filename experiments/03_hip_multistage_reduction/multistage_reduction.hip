#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t error = call;                                               \
        if (error != hipSuccess) {                                             \
            std::cerr << hipGetErrorString(error) << '\n';                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void block_sum(const float* input, float* output, int n) {
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared_data[tid] = i < n ? input[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const int first_stage_blocks = (n + threads - 1) / threads;

    std::vector<float> input(n, 1.0f);

    float *d_input, *d_buffer_a, *d_buffer_b;
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_buffer_a, first_stage_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_buffer_b, first_stage_blocks * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), n * sizeof(float),
                        hipMemcpyHostToDevice));

    int current_size = n;
    int stage = 1;
    const float* current_input = d_input;
    float* current_output = d_buffer_a;

    while (current_size > 1) {
        int blocks = (current_size + threads - 1) / threads;
        std::cout << "阶段 " << stage << ": " << current_size << " -> "
                  << blocks << '\n';

        block_sum<<<blocks, threads>>>(current_input, current_output,
                                        current_size);
        HIP_CHECK(hipGetLastError());

        current_size = blocks;
        current_input = current_output;
        current_output = current_output == d_buffer_a ? d_buffer_b : d_buffer_a;
        ++stage;
    }

    HIP_CHECK(hipDeviceSynchronize());

    float gpu_sum = 0.0f;
    HIP_CHECK(hipMemcpy(&gpu_sum, current_input, sizeof(float),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_buffer_a));
    HIP_CHECK(hipFree(d_buffer_b));

    const float expected = static_cast<float>(n);
    const float error = std::abs(gpu_sum - expected);
    std::cout << "GPU 求和结果: " << gpu_sum << '\n';
    std::cout << "预期结果: " << expected << '\n';
    std::cout << (error == 0.0f ? "检查通过\n" : "检查失败\n");
    return error == 0.0f ? 0 : 1;
}
