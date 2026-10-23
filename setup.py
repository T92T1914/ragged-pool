"""Ahead of time build for the pinned ATen API, without a stable ABI claim."""
import os
import sys
import importlib.util
import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel
import torch


_source = Path(__file__).resolve().parent / "src" / "ragged_pool" / "_build.py"
_specification = importlib.util.spec_from_file_location("ragged_pool_build_contract", _source)
_contract = importlib.util.module_from_spec(_specification)
_specification.loader.exec_module(_contract)
_config = _contract.build_config(os.environ.get("RAGGED_POOL_TORCH_VARIANT", "cpu"))
_contract.validate_runtime(_config, str(torch.__version__), torch.version.cuda)

# Importing the extension builder follows the exact framework-build check.
from torch.utils.cpp_extension import BuildExtension, CppExtension


class BuildPython(build_py):
    """Generate wheel metadata in the build directory, never in the source."""

    def run(self):
        super().run()
        destination = Path(self.build_lib) / "ragged_pool" / _contract.CONFIG_FILENAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_config, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8", newline="\n")


class BuildWheel(bdist_wheel):
    """Keep the cu128 artifact distinct from the default CPU wheel."""

    def finalize_options(self):
        super().finalize_options()
        expected = _config["wheel_build_tag"]
        if self.build_number not in (None, "", expected):
            raise RuntimeError("The ragged-pool wheel build tag is fixed by its Torch variant")
        self.build_number = expected or None


class BuildNative(BuildExtension):
    """Do not reuse objects from another Torch flavor's build directory."""

    def finalize_options(self):
        super().finalize_options()
        # Set force before build_ext creates its compiler, so both extension
        # and object timestamp caches are bypassed when the flavor changes.
        self.force = True


flags = ["/O2", "/std:c++17", "/fp:precise"] if sys.platform == "win32" else ["-O2", "-std=c++17"]
link_flags = ["/CGTHREADS:2"] if sys.platform == "win32" else []
if os.environ.get("RAGGED_POOL_ASAN") == "1":
    if sys.platform != "linux":
        raise RuntimeError("The declared AddressSanitizer build requires Linux")
    flags = ["-O1", "-g", "-std=c++17", "-fno-omit-frame-pointer", "-fsanitize=address"]
    link_flags = ["-fsanitize=address"]
setup(
    install_requires=["torch==" + _config["torch_version"]],
    ext_modules=[CppExtension("ragged_pool._C", ["csrc/pool.cpp"], extra_compile_args=flags,
                              extra_link_args=link_flags)],
    cmdclass={"build_ext": BuildNative.with_options(use_ninja=False),
              "build_py": BuildPython, "bdist_wheel": BuildWheel},
)
