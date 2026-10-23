"""Train one fixed model through co-installed CPU and CUDA pooling wheels.

Each route performs two real SGD updates from identical CPU parameters and
fixtures. Device transfers are explicit caller operations. This tiny consumer
checks interoperability and numerical agreement, not model quality or speed.
Run a copied script outside the checkout in an ordinary wheel installation.
"""
import argparse
import copy
from email.parser import BytesParser
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import torch
from torch import nn

import ragged_pool as cpu_package
import ragged_pool_cuda as cuda_package


SEED = 1811
BOUNDARIES = ((0, 2, 2, 5, 7, 9), (0, 3, 5, 5, 8, 10, 12))
DTYPES = {"float32": torch.float32, "float64": torch.float64}
ROUTES = ("cpu_native", "cuda_native", "cpu_primitive", "cuda_primitive")


def primitive_pool(values, logits, boundaries):
    """Use independent double-precision PyTorch arithmetic at known boundaries."""
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        selected_values = values[start:stop].to(torch.float64)
        selected_logits = logits[start:stop].to(torch.float64)
        if start == stop:
            row = selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0
        else:
            weights = torch.softmax(selected_logits, dim=0)
            row = (weights[:, None] * selected_values).sum(dim=0)
        rows.append(row.to(values.dtype))
    return torch.stack(rows)


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A shared scalar score bias would cancel in each group's softmax.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, boundaries, route):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        if route == "cpu_native":
            pooled = cpu_package.ragged_pool(values, logits, offsets)
        elif route == "cuda_native":
            pooled = cuda_package.ragged_pool(values, logits, offsets)
        elif route in ("cpu_primitive", "cuda_primitive"):
            pooled = primitive_pool(values, logits, boundaries)
        else:
            raise ValueError("Unknown training route")
        return self.head(pooled).squeeze(-1), pooled


def snapshot(tensors):
    return {name: {"bytes": value.detach().clone(), "version": value._version,
                   "pointer": value.data_ptr(), "device": value.device,
                   "shape": value.shape, "stride": value.stride()}
            for name, value in tensors.items()}


def unchanged(tensors, before):
    for name, value in tensors.items():
        saved = before[name]
        torch.testing.assert_close(value, saved["bytes"], rtol=0, atol=0)
        if (value._version != saved["version"] or value.data_ptr() != saved["pointer"]
                or value.device != saved["device"] or value.shape != saved["shape"]
                or value.stride() != saved["stride"]):
            raise AssertionError(f"Training changed saved input ownership or contents: {name}")


def host_copy(tensors):
    # Comparisons explicitly copy detached results to CPU after their calculation.
    return {name: value.detach().to(device="cpu").clone() for name, value in tensors.items()}


def difference(first, second):
    return float((first - second).abs().max()) if first.numel() else 0.0


def compare(first, second, tolerance):
    if first.keys() != second.keys():
        raise AssertionError("Comparison keys differ")
    errors = {}
    for name in first:
        if not bool(torch.isfinite(first[name]).all()) or not bool(torch.isfinite(second[name]).all()):
            raise AssertionError("Compared results must be finite")
        torch.testing.assert_close(first[name], second[name], **tolerance)
        errors[name] = difference(first[name], second[name])
    return errors


def installed_identity(name, package, wheel, expected_build):
    installed = distribution(name)
    direct = installed.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise AssertionError("Use ordinary installed wheels")
    module = package.__name__
    prefix = Path(sys.prefix).resolve()
    imported = Path(package.__file__).resolve(strict=True)
    if (imported != Path(installed.locate_file(module + "/__init__.py")).resolve(strict=True)
            or not imported.is_relative_to(prefix) or installed.version != package.__version__):
        raise AssertionError("The import must match this environment's distribution")
    wheel = wheel.resolve(strict=True)
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise AssertionError("A wheel cannot contain duplicate members")
        metadata_paths = [path for path in names if path.endswith(".dist-info/METADATA")]
        wheel_paths = [path for path in names if path.endswith(".dist-info/WHEEL")]
        if len(metadata_paths) != 1 or len(wheel_paths) != 1:
            raise AssertionError("Expected one wheel metadata pair")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_paths[0]))
        requirements = metadata.get_all("Requires-Dist", [])
        if (metadata["Name"] != name or metadata["Version"] != installed.version
                or (wheel_metadata["Build"] or "") != expected_build
                or requirements != ["torch==2.9.1+cu128"]):
            raise AssertionError("Wheel metadata does not match the declared cu128 pair")
        for entry in (metadata_paths[0], wheel_paths[0]):
            actual = Path(installed.locate_file(entry)).resolve(strict=True)
            if not actual.is_relative_to(prefix) or actual.read_bytes() != archive.read(entry):
                raise AssertionError("Installed distribution metadata differs from its wheel")
        entries = [path for path in names if path.startswith(module + "/")
                   and path.endswith((".py", ".pyd", ".so", ".json"))]
        for entry in entries:
            actual = Path(installed.locate_file(entry)).resolve(strict=True)
            if not actual.is_relative_to(prefix) or actual.read_bytes() != archive.read(entry):
                raise AssertionError(f"Installed package bytes differ from its wheel: {entry}")
        native_entries = [path for path in entries if path.startswith(module + "/_C")
                          and path.endswith((".pyd", ".so"))]
        if len(native_entries) != 1:
            raise AssertionError("Expected one installed native library per package")
        native = Path(installed.locate_file(native_entries[0])).resolve(strict=True)
        loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
        recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
        if native not in loaded or native not in recorded:
            raise AssertionError("Expected the recorded installed native library to be loaded")
    result = {"distribution_name": name, "distribution_version": installed.version,
              "package_version": package.__version__, "package_path": str(imported),
              "native_library_path": str(native), "native_library_loaded": True,
              "native_library_sha256": sha256(native.read_bytes()).hexdigest(),
              "metadata_path": str(Path(installed.locate_file(metadata_paths[0])).resolve()),
              "requires_dist": requirements, "wheel_build_tag": expected_build,
              "wheel_path": str(wheel), "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
              "wheel_filename": wheel.name}
    result["metadata_sha256"] = sha256(Path(result["metadata_path"]).read_bytes()).hexdigest()
    if name == "ragged-pool":
        config_path = imported.parent / "_build_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        from ragged_pool._build import validate_runtime
        validate_runtime(config, str(torch.__version__), torch.version.cuda)
        if config["variant"] != "cu128" or config["wheel_build_tag"] != "1cu128":
            raise AssertionError("The root CPU operator wheel must be the explicit cu128 build")
        result.update({"build_config": config, "build_config_path": str(config_path),
                       "build_config_sha256": sha256(config_path.read_bytes()).hexdigest()})
    return result


def fixture(dtype_name):
    dtype = DTYPES[dtype_name]
    torch.manual_seed(SEED)
    initial = SetRegressor().to(dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 100)
    batches, digest = [], sha256()
    for boundaries in BOUNDARIES:
        groups = len(boundaries) - 1
        raw = 0.25 * torch.randn(boundaries[-1], 3, dtype=dtype, generator=generator)
        targets = (0.4 * torch.randn(groups, dtype=dtype, generator=generator)
                   + torch.linspace(-0.3, 0.5, groups, dtype=dtype))
        batch = {"raw": raw, "targets": targets,
                 "offsets": torch.tensor(boundaries, dtype=torch.int64), "boundaries": boundaries}
        for name in ("raw", "targets", "offsets"):
            tensor = batch[name]
            digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
            digest.update(tensor.numpy().tobytes())
        batches.append(batch)
    return initial, batches, digest.hexdigest()


def evaluate(model, raw, batch, route):
    inputs = {"raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(inputs)
    predictions, pooled = model(raw, batch["offsets"], batch["boundaries"], route)
    loss = (predictions - batch["targets"]).square().mean()
    loss.backward()
    if raw.grad is None or any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("All declared inputs and parameter blocks need first gradients")
    gradients = host_copy({name: value.grad for name, value in model.named_parameters()})
    if any(float(value.abs().max()) <= 0 for value in gradients.values()) or float(raw.grad.abs().max()) <= 0:
        raise AssertionError("The synthetic fixture must give informative gradients")
    empty = [row for row, (start, stop) in enumerate(zip(batch["boundaries"], batch["boundaries"][1:]))
             if start == stop]
    torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
    unchanged(inputs, before)
    return host_copy({"predictions": predictions, "pooled": pooled,
                      "loss": loss, "input_gradient": raw.grad}), gradients


def run(dtype_name, device_index, cpu_wheel, cuda_wheel, source_revision=None, source_tree=None):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if (sys.version_info[:3] != (3, 11, 8) or str(torch.__version__) != "2.9.1+cu128"
            or torch.version.cuda != "12.8"):
        raise AssertionError("This consumer requires CPython 3.11.8 and PyTorch 2.9.1+cu128")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA device is unavailable")
    script = Path(__file__).resolve()
    if any((parent / "pyproject.toml").is_file() for parent in script.parents):
        raise AssertionError("Copy the consumer outside every source checkout")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    identities = {"cpu": installed_identity("ragged-pool", cpu_package, cpu_wheel, "1cu128"),
                  "cuda": installed_identity("ragged-pool-cuda", cuda_package, cuda_wheel, "")}
    initial, source_batches, fixture_hash = fixture(dtype_name)
    initial_before = snapshot(dict(initial.named_parameters()))
    fixture_before = [snapshot({name: batch[name] for name in ("raw", "offsets", "targets")})
                      for batch in source_batches]
    devices = {name: device if name.startswith("cuda") else torch.device("cpu") for name in ROUTES}
    # Both native operators receive already placed tensors. Neither chooses or
    # transfers a device on the caller's behalf.
    models = {name: copy.deepcopy(initial).to(device=devices[name]) for name in ROUTES}
    for model in models.values():
        compare(host_copy(dict(model.named_parameters())), host_copy(dict(initial.named_parameters())),
                {"rtol": 0, "atol": 0})
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03, foreach=False, fused=False)
                  for name, model in models.items()}
    ownership = {name: {"parameters": tuple(id(value) for value in model.parameters()),
                        "storage": tuple(value.data_ptr() for value in model.parameters()),
                        "optimizer": id(optimizers[name])} for name, model in models.items()}
    for device_routes in (("cpu_native", "cpu_primitive"), ("cuda_native", "cuda_primitive")):
        first, second = device_routes
        if set(ownership[first]["parameters"]) & set(ownership[second]["parameters"]) or (
                set(ownership[first]["storage"]) & set(ownership[second]["storage"])):
            raise AssertionError("Comparison routes must own separate parameters and storage")
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    steps = []
    for index, source_batch in enumerate(source_batches, 1):
        before = {name: host_copy(dict(model.named_parameters())) for name, model in models.items()}
        batches = {name: {key: source_batch[key].to(device=devices[name]).clone()
                          for key in ("raw", "offsets", "targets")} for name in ROUTES}
        for name, batch in batches.items():
            batch["boundaries"] = source_batch["boundaries"]
            batch["raw"].requires_grad_()
            if any(batch[key].device != devices[name] for key in ("raw", "offsets", "targets")):
                raise AssertionError("Each input must already reside on its selected route's device")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        results = {name: evaluate(models[name], batches[name]["raw"], batches[name], name)
                   for name in ROUTES}
        pairs = (("cpu_native", "cpu_primitive"), ("cuda_native", "cuda_primitive"),
                 ("cuda_native", "cpu_native"), ("cuda_primitive", "cpu_primitive"))
        output_errors = {left + "__" + right: compare(results[left][0], results[right][0], tolerance)
                         for left, right in pairs}
        gradient_errors = {left + "__" + right: compare(results[left][1], results[right][1], tolerance)
                           for left, right in pairs}
        for optimizer in optimizers.values():
            optimizer.step()
        states = {name: host_copy(dict(model.named_parameters())) for name, model in models.items()}
        update_errors = {left + "__" + right: compare(states[left], states[right], tolerance)
                         for left, right in pairs}
        changes = {name: {key: difference(value, before[name][key]) for key, value in state.items()}
                   for name, state in states.items()}
        if any(change <= 0 for route in changes.values() for change in route.values()):
            raise AssertionError("Each actual update must change every parameter block")
        for name, model in models.items():
            owned = ownership[name]
            if (tuple(id(value) for value in model.parameters()) != owned["parameters"]
                    or tuple(value.data_ptr() for value in model.parameters()) != owned["storage"]
                    or id(optimizers[name]) != owned["optimizer"]
                    or tuple(id(value) for group in optimizers[name].param_groups for value in group["params"])
                    != owned["parameters"]):
                raise AssertionError("Training changed model or optimizer ownership")
        unchanged({key: source_batch[key] for key in ("raw", "offsets", "targets")}, fixture_before[index - 1])
        steps.append({"update": index, "N": source_batch["boundaries"][-1],
                      "B": len(source_batch["boundaries"]) - 1, "offsets": list(source_batch["boundaries"]),
                      "losses": {name: float(result[0]["loss"]) for name, result in results.items()},
                      "output_max_abs_errors": output_errors,
                      "parameter_gradient_max_abs_errors": gradient_errors,
                      "updated_parameter_max_abs_errors": update_errors, "parameter_update_max_abs": changes,
                      "inputs_unchanged": True, "all_parameter_blocks_changed": True,
                      "same_parameter_and_optimizer_objects": True})
    unchanged(dict(initial.named_parameters()), initial_before)
    for batch, before in zip(source_batches, fixture_before):
        unchanged({key: batch[key] for key in ("raw", "offsets", "targets")}, before)
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "eager.coinstalled_cpu_cuda_training", "dtype": dtype_name,
            "scope": "Two synthetic first-order updates on one pinned installed CPU/CUDA wheel pair",
            "seed": SEED, "routes": list(ROUTES), "cuda_device": str(device), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda, "python_executable": sys.executable,
            "environment_prefix": str(Path(sys.prefix).resolve()), "working_directory": str(Path.cwd()),
            "script_path": str(script), "script_sha256": sha256(script.read_bytes()).hexdigest(),
            "source_revision": source_revision, "source_tree": source_tree,
            "source_identity_basis": "Caller labels plus hashed script, wheels and loaded native bytes",
            "installed_packages": identities, "root_cpu_operator_build_variant": "cu128",
            "explicit_device_transfers": True, "comparison_copies_to_cpu": True,
            "fixture_sha256": fixture_hash, "optimizer": "SGD", "learning_rate": 0.03,
            "optimizer_steps_per_route": 2, "parameter_tensors": 5, "scalar_parameters": 25,
            "comparison_tolerance": tolerance, "steps": steps, "inputs_unchanged": True,
            "initial_parameters_unchanged": True, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", type=int, default=0, help="Selected CUDA device ordinal")
    parser.add_argument("--cpu-wheel", type=Path, required=True, help="Installed root CPU operator cu128 wheel")
    parser.add_argument("--cuda-wheel", type=Path, required=True, help="Installed CUDA companion wheel")
    parser.add_argument("--source-revision", help="Optional independently established source revision")
    parser.add_argument("--source-tree", help="Optional independently established source tree")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    arguments = parser.parse_args()
    for label in (arguments.source_revision, arguments.source_tree):
        if label is not None and not re.fullmatch(r"[0-9a-f]{40}", label):
            parser.error("Source labels must be complete lowercase SHA1 identifiers")
    if arguments.device < 0:
        parser.error("The CUDA device ordinal must be nonnegative")
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    result = run(arguments.dtype, arguments.device, arguments.cpu_wheel, arguments.cuda_wheel,
                 arguments.source_revision, arguments.source_tree)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
