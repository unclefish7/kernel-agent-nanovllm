#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const size_t bytes = n * sizeof(float);

    std::vector<float> a(n, 1.5f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), bytes, hipMemcpyHostToDevice);

    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    hipDeviceSynchronize();

    hipMemcpy(c.data(), d_c, bytes, hipMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::abs(c[i] - (a[i] + b[i])));
    }

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    std::cout << "元素数量: " << n << '\n';
    std::cout << "启动配置: " << blocks << " blocks x " << threads
              << " threads\n";
    std::cout << "最大误差: " << max_error << '\n';
    std::cout << (max_error == 0.0f ? "检查通过\n" : "检查失败\n");
    return max_error == 0.0f ? 0 : 1;
}
