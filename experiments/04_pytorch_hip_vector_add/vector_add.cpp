#include <torch/extension.h>

at::Tensor vector_add_hip(const at::Tensor& a, const at::Tensor& b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_add", &vector_add_hip, "HIP vector addition");
}
