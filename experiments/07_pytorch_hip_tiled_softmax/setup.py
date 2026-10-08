from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="hip_tiled_softmax",
    ext_modules=[
        CUDAExtension(
            name="hip_tiled_softmax",
            sources=["tiled_softmax.cpp", "tiled_softmax_kernel.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
