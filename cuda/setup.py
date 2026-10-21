"""Build the explicit cu128 companion without changing the root CPU package."""
import os
import sys

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
    raise RuntimeError("ragged-pool-cuda must be built with torch==2.9.1+cu128")
if os.environ.get("TORCH_CUDA_ARCH_LIST") != "8.9":
    raise RuntimeError("Set TORCH_CUDA_ARCH_LIST=8.9 explicitly for this bounded CUDA build")

cxx = ["/O2", "/std:c++17", "/fp:precise"] if sys.platform == "win32" else ["-O2", "-std=c++17"]
nvcc = ["-O2", "-std=c++17", "-lineinfo", "--threads", "1", "--fmad=false"]
if sys.platform == "win32":
    nvcc.extend(["-Xcompiler=/O2", "-Xcompiler=/fp:precise"])
link = ["/CGTHREADS:2"] if sys.platform == "win32" else []

setup(
    ext_modules=[CUDAExtension("ragged_pool_cuda._C", ["csrc/pool.cu"],
                               extra_compile_args={"cxx": cxx, "nvcc": nvcc},
                               extra_link_args=link)],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
