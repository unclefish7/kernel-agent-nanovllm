#include <torch/extension.h>

at::Tensor tiled_softmax_hip(const at::Tensor& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("tiled_softmax", &tiled_softmax_hip, "HIP tiled softmax");
}
