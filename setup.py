"""Ahead of time build for the pinned ATen API, without a stable ABI claim."""
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

flags = ["/O2", "/std:c++17", "/fp:precise"] if sys.platform == "win32" else ["-O2", "-std=c++17"]
link_flags = ["/CGTHREADS:2"] if sys.platform == "win32" else []
setup(
    ext_modules=[CppExtension("ragged_pool._C", ["csrc/pool.cpp"], extra_compile_args=flags,
                              extra_link_args=link_flags)],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
