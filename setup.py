"""Ahead of time build for the pinned ATen API, without a stable ABI claim."""
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

flags = ["/O2", "/std:c++17", "/fp:precise"] if sys.platform == "win32" else ["-O2", "-std=c++17"]
link_flags = ["/CGTHREADS:2"] if sys.platform == "win32" else []
if os.environ.get("RAGGED_POOL_ASAN") == "1":
    if sys.platform != "linux":
        raise RuntimeError("The declared AddressSanitizer build requires Linux")
    flags = ["-O1", "-g", "-std=c++17", "-fno-omit-frame-pointer", "-fsanitize=address"]
    link_flags = ["-fsanitize=address"]
setup(
    ext_modules=[CppExtension("ragged_pool._C", ["csrc/pool.cpp"], extra_compile_args=flags,
                              extra_link_args=link_flags)],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
