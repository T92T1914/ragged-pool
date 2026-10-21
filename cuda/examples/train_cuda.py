"""Compare two installed packed CUDA training updates with primitive GPU math.

This synthetic eager first-order consumer establishes functional behavior only
when executed. Native diagnostic qualification remains a separate requirement.
It makes no quality, timing, memory, portability or security claim.
"""
import argparse
import copy
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import re
import sys

import torch
from torch import nn

import ragged_pool_cuda as package
from ragged_pool_cuda import ragged_pool


SEED = 1811
BOUNDARIES = ((0, 2, 2, 5, 7, 9), (0, 3, 5, 5, 8, 10, 12))
DTYPES = {"float32": torch.float32, "float64": torch.float64}


def primitive_pool(values, logits, boundaries):
    """Differentiate independent double-precision math at known boundaries."""
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
        # A common scalar score bias cancels in each group's softmax.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, boundaries=None):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = (ragged_pool(values, logits, offsets) if boundaries is None
                  else primitive_pool(values, logits, boundaries))
        return self.head(pooled).squeeze(-1), pooled


def difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def compare(first, second, tolerance):
    if first.keys() != second.keys():
        raise AssertionError("Comparison keys differ")
    errors = {}
    for name in first:
        torch.testing.assert_close(first[name], second[name], **tolerance)
        if not bool(torch.isfinite(first[name]).all()) or not bool(torch.isfinite(second[name]).all()):
            raise AssertionError("Compared results must be finite")
        errors[name] = difference(first[name], second[name])
    return errors


def snapshot(inputs):
    return {name: (value.detach().clone(), value._version) for name, value in inputs.items()}


def unchanged(inputs, before):
    for name, value in inputs.items():
        torch.testing.assert_close(value, before[name][0], rtol=0, atol=0)
        if value._version != before[name][1]:
            raise AssertionError(f"Training changed saved input {name}")


def installed_identity():
    installed = distribution("ragged-pool-cuda")
    direct = installed.read_text("direct_url.json")
    prefix = Path(sys.prefix).resolve()
    imported = Path(package.__file__).resolve(strict=True)
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise AssertionError("Use an ordinary installed CUDA wheel")
    if (imported != Path(installed.locate_file("ragged_pool_cuda/__init__.py")).resolve(strict=True)
            or not imported.is_relative_to(prefix) or installed.version != package.__version__):
        raise AssertionError("The CUDA import must match this environment's distribution")
    libraries = list(imported.parent.glob("_C*.pyd")) + list(imported.parent.glob("_C*.so"))
    recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    if len(libraries) != 1 or libraries[0].resolve() not in recorded or libraries[0].resolve() not in loaded:
        raise AssertionError("Expected one recorded and loaded installed CUDA native library")
    if any(name == "ragged_pool" or name.startswith("ragged_pool.") for name in sys.modules):
        raise AssertionError("The CUDA consumer must not import the CPU package")
    script = Path(__file__).resolve()
    if any((parent / "pyproject.toml").is_file() for parent in script.parents):
        raise AssertionError("Copy the consumer outside every source checkout")
    return {"installed_package": True, "package_path": str(imported),
            "package_version": package.__version__, "environment_prefix": str(prefix),
            "native_library_loaded": True, "native_library_path": str(libraries[0].resolve()),
            "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest(),
            "script_sha256": sha256(script.read_bytes()).hexdigest(), "cpu_package_imported": False}


def fixture(dtype_name, device):
    dtype = DTYPES[dtype_name]
    torch.manual_seed(SEED)
    initial = SetRegressor().to(device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 100)
    batches, digest = [], sha256()
    for boundaries in BOUNDARIES:
        groups = len(boundaries) - 1
        raw = 0.25 * torch.randn(boundaries[-1], 3, dtype=dtype, generator=generator)
        targets = (0.4 * torch.randn(groups, dtype=dtype, generator=generator)
                   + torch.linspace(-0.3, 0.5, groups, dtype=dtype))
        offsets = torch.tensor(boundaries, dtype=torch.int64)
        for name, tensor in (("raw", raw), ("targets", targets), ("offsets", offsets)):
            digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
            digest.update(tensor.numpy().tobytes())
        batches.append({"raw": raw.to(device), "targets": targets.to(device),
                        "offsets": offsets.to(device), "boundaries": boundaries})
    return initial, batches, digest.hexdigest()


def evaluate(model, raw, batch, reference):
    inputs = {"raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(inputs)
    predictions, pooled = model(raw, batch["offsets"], batch["boundaries"] if reference else None)
    loss = (predictions - batch["targets"]).square().mean()
    loss.backward()
    if raw.grad is None or any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("All declared inputs and parameter blocks need first gradients")
    gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()}
    if any(float(value.abs().max()) <= 0 for value in gradients.values()) or float(raw.grad.abs().max()) <= 0:
        raise AssertionError("The synthetic fixture must give informative gradients")
    empty = [row for row, (start, stop) in enumerate(zip(batch["boundaries"], batch["boundaries"][1:]))
             if start == stop]
    torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
    unchanged(inputs, before)
    return {"predictions": predictions.detach(), "pooled": pooled.detach(),
            "loss": loss.detach(), "input_gradient": raw.grad.detach().clone()}, gradients


def run(dtype_name, device_index, source_revision=None, source_tree=None):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise AssertionError("This candidate requires the declared PyTorch 2.9.1 cu128 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA device is unavailable")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    identity = installed_identity()
    initial, batches, fixture_hash = fixture(dtype_name, device)
    initial_state = {name: value.detach().clone() for name, value in initial.named_parameters()}
    initial_versions = {name: value._version for name, value in initial.named_parameters()}
    models = {name: copy.deepcopy(initial) for name in ("native", "primitive")}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03, foreach=False, fused=False)
                  for name, model in models.items()}
    model_ids = {name: tuple(id(value) for value in model.parameters()) for name, model in models.items()}
    optimizer_ids = {name: id(value) for name, value in optimizers.items()}
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    steps = []
    for index, batch in enumerate(batches, 1):
        before = {name: {key: value.detach().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}
        raw = {name: batch["raw"].detach().clone().requires_grad_() for name in models}
        source_inputs = {name: batch[name] for name in ("raw", "offsets", "targets")}
        source_before = snapshot(source_inputs)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        results = {name: evaluate(model, raw[name], batch, reference=name == "primitive")
                   for name, model in models.items()}
        output_errors = compare(results["native"][0], results["primitive"][0], tolerance)
        gradient_errors = compare(results["native"][1], results["primitive"][1], tolerance)
        for optimizer in optimizers.values():
            optimizer.step()
        states = {name: {key: value.detach().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}
        update_errors = compare(states["native"], states["primitive"], tolerance)
        changes = {name: {key: difference(value, before[name][key]) for key, value in state.items()}
                   for name, state in states.items()}
        if any(change <= 0 for route in changes.values() for change in route.values()):
            raise AssertionError("Every actual update must change every parameter block")
        for name, model in models.items():
            if (tuple(id(value) for value in model.parameters()) != model_ids[name]
                    or id(optimizers[name]) != optimizer_ids[name]
                    or tuple(id(value) for group in optimizers[name].param_groups for value in group["params"])
                    != model_ids[name]):
                raise AssertionError("Training changed model/optimizer ownership")
        unchanged(source_inputs, source_before)
        steps.append({"update": index, "N": batch["boundaries"][-1], "B": len(batch["boundaries"]) - 1,
                      "offsets": list(batch["boundaries"]), "native_loss": float(results["native"][0]["loss"]),
                      "primitive_loss": float(results["primitive"][0]["loss"]), "output_max_abs_errors": output_errors,
                      "parameter_gradient_max_abs_errors": gradient_errors,
                      "updated_parameter_max_abs_errors": update_errors, "parameter_update_max_abs": changes,
                      "inputs_unchanged": True, "all_parameter_blocks_changed": True,
                      "same_parameter_and_optimizer_objects": True})
    for name, value in initial.named_parameters():
        torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
        if value._version != initial_versions[name]:
            raise AssertionError("Training changed the initial model")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "eager.packed_cuda_training", "dtype": dtype_name,
            "scope": "Functional synthetic first-order checks. Native diagnostics remain a separate gate.",
            "seed": SEED, "device": str(device), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda, "python_executable": sys.executable,
            "working_directory": str(Path.cwd()), "source_revision": source_revision,
            "source_tree": source_tree, "source_identity_basis": "Caller labels plus hashed script/native bytes",
            "fixture_sha256": fixture_hash, "optimizer": "SGD", "learning_rate": 0.03,
            "optimizer_steps": 2, "reference_optimizer_steps": 2, "parameter_tensors": 5,
            "scalar_parameters": 25, "comparison_tolerance": tolerance, "steps": steps,
            "inputs_unchanged": True, "initial_parameters_unchanged": True,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", type=int, default=0, help="Selected CUDA device ordinal")
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
    result = run(arguments.dtype, arguments.device, arguments.source_revision, arguments.source_tree)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
