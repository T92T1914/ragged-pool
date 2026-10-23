"""Check explicit Torch flavors without compiling or loading a native library."""
import builtins
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "ragged_pool"
SPECIFICATION = importlib.util.spec_from_file_location("build_variant_contract", SOURCE / "_build.py")
CONTRACT = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(CONTRACT)


@pytest.mark.parametrize(("variant", "torch_version", "cuda_version", "build_tag"), [
    ("cpu", "2.9.1+cpu", None, ""),
    ("cu128", "2.9.1+cu128", "12.8", "1cu128"),
])
def test_explicit_build_configuration_round_trips(variant, torch_version, cuda_version, build_tag):
    expected = {"schema": 1, "variant": variant, "torch_version": torch_version,
                "cuda_version": cuda_version, "wheel_build_tag": build_tag}
    assert CONTRACT.build_config(variant) == expected
    assert CONTRACT.validate_config(json.loads(json.dumps(expected))) == expected
    assert CONTRACT.validate_runtime(expected, torch_version, cuda_version) == expected


@pytest.mark.parametrize("variant", [None, "", "cuda", "cu129", "CPU", "cpu ", 1, True, [], {}])
def test_unknown_variant_is_not_inferred_from_the_environment(variant):
    with pytest.raises(RuntimeError, match="cpu or cu128"):
        CONTRACT.build_config(variant)


@pytest.mark.parametrize("fault", ["not_dict", "missing_schema", "bool_schema", "float_schema",
                                  "wrong_schema", "missing_variant", "unknown_variant",
                                  "missing_torch", "wrong_torch", "missing_cuda", "wrong_cuda",
                                  "missing_tag", "wrong_tag", "extra_key"])
@pytest.mark.parametrize("variant", ["cpu", "cu128"])
def test_invalid_generated_configuration_is_refused(variant, fault):
    config = CONTRACT.build_config(variant)
    if fault == "not_dict":
        config = list(config.items())
    elif fault.startswith("missing_"):
        key = {"missing_schema": "schema", "missing_variant": "variant", "missing_torch": "torch_version",
               "missing_cuda": "cuda_version", "missing_tag": "wheel_build_tag"}[fault]
        config.pop(key)
    else:
        key, value = {"bool_schema": ("schema", True), "float_schema": ("schema", 1.0),
                      "wrong_schema": ("schema", 2), "unknown_variant": ("variant", "cu129"),
                      "wrong_torch": ("torch_version", "2.9.1"), "wrong_cuda": ("cuda_version", "12.9"),
                      "wrong_tag": ("wheel_build_tag", "2cu128"), "extra_key": ("unreviewed", True)}[fault]
        config[key] = value
    with pytest.raises(RuntimeError):
        CONTRACT.validate_config(config)


@pytest.mark.parametrize(("variant", "torch_version", "cuda_version"), [
    ("cpu", "2.9.1+cu128", "12.8"),
    ("cu128", "2.9.1+cpu", None),
    ("cpu", "2.9.1", None),
    ("cu128", "2.9.1", "12.8"),
    ("cpu", "2.9.2+cpu", None),
    ("cu128", "2.9.2+cu128", "12.8"),
    ("cpu", "2.9.1+cpu", "12.8"),
    ("cu128", "2.9.1+cu128", None),
    ("cu128", "2.9.1+cu128", "12.9"),
])
def test_exact_framework_and_runtime_pin_is_required(variant, torch_version, cuda_version):
    with pytest.raises(RuntimeError, match="requires torch=="):
        CONTRACT.validate_runtime(CONTRACT.build_config(variant), torch_version, cuda_version)


class NativeLoadReached(Exception):
    pass


def import_fixture(tmp_path, monkeypatch, config, torch_version, cuda_version):
    package_root = tmp_path / "isolated_package"
    package_root.mkdir()
    for name in ("__init__.py", "_build.py"):
        (package_root / name).write_bytes((SOURCE / name).read_bytes())
    if config is not None:
        data = config if isinstance(config, bytes) else json.dumps(config).encode()
        (package_root / CONTRACT.CONFIG_FILENAME).write_bytes(data)
    (package_root / "_C.fixture.so").write_bytes(b"not executable")
    calls = []

    def native_load(path):
        calls.append(path)
        raise NativeLoadReached

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__=torch_version, version=SimpleNamespace(cuda=cuda_version),
        ops=SimpleNamespace(load_library=native_load)))
    name = "ragged_pool_import_fixture"
    specification = importlib.util.spec_from_file_location(
        name, package_root / "__init__.py", submodule_search_locations=[str(package_root)])
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, name, module)
    # The package's relative helper import is isolated from earlier test cases.
    monkeypatch.delitem(sys.modules, name + "._build", raising=False)
    return specification, module, calls


@pytest.mark.parametrize(("config", "torch_version", "cuda_version", "error"), [
    (None, "2.9.1+cpu", None, ImportError),
    (b"{broken", "2.9.1+cpu", None, ImportError),
    (b"\xff", "2.9.1+cpu", None, ImportError),
    ({}, "2.9.1+cpu", None, RuntimeError),
    (CONTRACT.build_config("cpu"), "2.9.1+cu128", "12.8", RuntimeError),
    (CONTRACT.build_config("cu128"), "2.9.1+cpu", None, RuntimeError),
    (CONTRACT.build_config("cu128"), "2.9.2+cu128", "12.8", RuntimeError),
    (CONTRACT.build_config("cu128"), "2.9.1+cu128", "12.9", RuntimeError),
])
def test_import_refuses_invalid_identity_before_native_load(tmp_path, monkeypatch, config,
                                                           torch_version, cuda_version, error):
    specification, module, calls = import_fixture(tmp_path, monkeypatch, config, torch_version, cuda_version)
    with pytest.raises(error):
        specification.loader.exec_module(module)
    assert calls == []


@pytest.mark.parametrize(("variant", "torch_version", "cuda_version"), [
    ("cpu", "2.9.1+cpu", None), ("cu128", "2.9.1+cu128", "12.8"),
])
def test_matching_import_reaches_exactly_one_native_load(tmp_path, monkeypatch, variant,
                                                        torch_version, cuda_version):
    specification, module, calls = import_fixture(
        tmp_path, monkeypatch, CONTRACT.build_config(variant), torch_version, cuda_version)
    with pytest.raises(NativeLoadReached):
        specification.loader.exec_module(module)
    assert len(calls) == 1
    assert Path(calls[0]).name == "_C.fixture.so"


def setup_fixture(monkeypatch, variant, torch_version, cuda_version):
    captured, extension_imports, extension_calls = [], [], []
    setuptools = ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.append(kwargs)
    build_py = ModuleType("setuptools.command.build_py")
    bdist_wheel = ModuleType("setuptools.command.bdist_wheel")

    class BuildPy:
        def run(self):
            pass

    class BuildWheel:
        def finalize_options(self):
            pass

    class BuildExtension:
        def finalize_options(self):
            pass

        @classmethod
        def with_options(cls, **kwargs):
            cls.test_options = kwargs
            return cls

    build_py.build_py = BuildPy
    bdist_wheel.bdist_wheel = BuildWheel
    torch = ModuleType("torch")
    torch.__version__ = torch_version
    torch.version = SimpleNamespace(cuda=cuda_version)
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = BuildExtension

    def extension(*args, **kwargs):
        extension_calls.append((args, kwargs))
        return SimpleNamespace(args=args, kwargs=kwargs)

    cpp_extension.CppExtension = extension
    for name, module in (("setuptools", setuptools), ("setuptools.command.build_py", build_py),
                         ("setuptools.command.bdist_wheel", bdist_wheel), ("torch", torch),
                         ("torch.utils.cpp_extension", cpp_extension)):
        monkeypatch.setitem(sys.modules, name, module)
    original_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        if name == "torch.utils.cpp_extension":
            extension_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.delenv("RAGGED_POOL_ASAN", raising=False)
    if variant is None:
        monkeypatch.delenv("RAGGED_POOL_TORCH_VARIANT", raising=False)
    else:
        monkeypatch.setenv("RAGGED_POOL_TORCH_VARIANT", variant)
    return captured, extension_imports, extension_calls


@pytest.mark.parametrize(("variant", "torch_version", "cuda_version"), [
    (None, "2.9.1+cu128", "12.8"), ("cpu", "2.9.1+cu128", "12.8"),
    ("cu128", "2.9.1+cpu", None), ("cu128", "2.9.1+cu128", "12.9"),
    ("cu129", "2.9.1+cu128", "12.8"),
])
def test_setup_refuses_wrong_flavor_before_importing_extension_builder(monkeypatch, variant,
                                                                    torch_version, cuda_version):
    captured, extension_imports, extension_calls = setup_fixture(monkeypatch, variant, torch_version, cuda_version)
    with pytest.raises(RuntimeError):
        runpy.run_path(str(ROOT / "setup.py"))
    assert captured == extension_imports == extension_calls == []


@pytest.mark.parametrize(("variant", "torch_version", "cuda_version", "build_tag"), [
    (None, "2.9.1+cpu", None, ""), ("cpu", "2.9.1+cpu", None, ""),
    ("cu128", "2.9.1+cu128", "12.8", "1cu128"),
])
def test_setup_keeps_variant_metadata_in_build_output(tmp_path, monkeypatch, variant,
                                                     torch_version, cuda_version, build_tag):
    captured, extension_imports, extension_calls = setup_fixture(monkeypatch, variant, torch_version, cuda_version)
    runpy.run_path(str(ROOT / "setup.py"))
    assert len(captured) == len(extension_imports) == len(extension_calls) == 1
    kwargs = captured[0]
    assert kwargs["install_requires"] == ["torch==" + torch_version]
    assert extension_calls[0][0] == ("ragged_pool._C", ["csrc/pool.cpp"])
    assert set(kwargs["cmdclass"]) == {"build_py", "bdist_wheel", "build_ext"}
    assert kwargs["cmdclass"]["build_ext"].test_options == {"use_ninja": False}
    native_command = kwargs["cmdclass"]["build_ext"]()
    native_command.force = False
    native_command.finalize_options()
    assert native_command.force is True
    command = kwargs["cmdclass"]["build_py"]()
    command.build_lib = str(tmp_path / "built")
    source_config = SOURCE / CONTRACT.CONFIG_FILENAME
    assert not source_config.exists()
    command.run()
    config = json.loads((tmp_path / "built" / "ragged_pool" / CONTRACT.CONFIG_FILENAME).read_text())
    assert config == CONTRACT.build_config(variant or "cpu")
    assert not source_config.exists()
    wheel = kwargs["cmdclass"]["bdist_wheel"]()
    wheel.build_number = None
    wheel.finalize_options()
    assert (wheel.build_number or "") == build_tag
    wheel.build_number = "2unreviewed"
    with pytest.raises(RuntimeError, match="build tag"):
        wheel.finalize_options()
