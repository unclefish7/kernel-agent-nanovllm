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

__global__ void block_sum(const float* input, float* partial_sums, int n) {
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
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> input(n, 1.0f);
    std::vector<float> partial_sums(blocks);

    float *d_input, *d_partial_sums;
    HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial_sums, blocks * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), n * sizeof(float),
                        hipMemcpyHostToDevice));

    block_sum<<<blocks, threads>>>(d_input, d_partial_sums, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(partial_sums.data(), d_partial_sums,
                        blocks * sizeof(float), hipMemcpyDeviceToHost));

    float gpu_sum = 0.0f;
    for (float value : partial_sums) {
        gpu_sum += value;
    }

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_partial_sums));

    const float expected = static_cast<float>(n);
    const float error = std::abs(gpu_sum - expected);
    std::cout << "元素数量: " << n << '\n';
    std::cout << "Block 数量: " << blocks << '\n';
    std::cout << "每个 Block 的线程数: " << threads << '\n';
    std::cout << "GPU 求和结果: " << gpu_sum << '\n';
    std::cout << "预期结果: " << expected << '\n';
    std::cout << (error == 0.0f ? "检查通过\n" : "检查失败\n");
    return error == 0.0f ? 0 : 1;
}
