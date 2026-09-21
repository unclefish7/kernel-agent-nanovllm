from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="hip_vector_add",
    ext_modules=[
        CUDAExtension(
            name="hip_vector_add",
            sources=["vector_add.cpp", "vector_add_kernel.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
