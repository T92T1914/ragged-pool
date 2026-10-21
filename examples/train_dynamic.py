"""Train one compiled packed model through five changing batch shapes.

The same model, parameters and SGD optimizer survive all five updates. A
single public torch.compile call uses aot_eager, fullgraph=True and
dynamic=True. Matched eager native and primitive PyTorch models check every
output, first gradient and update. This is framework integration, with no
graph-count, optimized-code, performance or model-quality claim.
"""
import argparse
import copy
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

import torch
from torch import nn

import ragged_pool as package
from ragged_pool import ragged_pool


SEED = 1811
DTYPES = {"float32": torch.float32, "float64": torch.float64}
OFFSETS = ((0, 2, 2, 5, 7, 9),
           (0, 3, 5, 5, 8, 10, 12),
           (0, 1, 5, 7, 11),
           (0, 3, 3, 8, 11),
           (0, 1, 4, 6, 6, 9))


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A shared scalar score bias cancels inside every group's softmax.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, pool=ragged_pool):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = pool(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def primitive_pool(boundaries):
    """Capture a batch's ordinary Python boundaries outside compilation."""
    def pool(values, logits, offsets):
        rows = []
        for start, stop in zip(boundaries, boundaries[1:]):
            selected_values, selected_logits = values[start:stop], logits[start:stop]
            if start == stop:
                rows.append(selected_values.sum(dim=0) * 0 + selected_logits.sum() * 0)
            else:
                weights = torch.softmax(selected_logits, dim=0)
                rows.append((weights[:, None] * selected_values).sum(dim=0))
        return torch.stack(rows)
    return pool


def difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def snapshot(tensors):
    return {name: (tensor.detach().clone(), tensor._version) for name, tensor in tensors.items()}


def unchanged(tensors, snapshots):
    versions = {}
    for name, tensor in tensors.items():
        before, version = snapshots[name]
        torch.testing.assert_close(tensor, before, rtol=0, atol=0)
        if tensor._version != version:
            raise AssertionError(f"The consumer mutated {name}")
        versions[name] = {"before": version, "after": tensor._version}
    return versions


def compare(first, second, tolerance):
    if first.keys() != second.keys():
        raise AssertionError("Comparison keys differ")
    errors = {}
    for name in first:
        torch.testing.assert_close(first[name], second[name], **tolerance)
        errors[name] = difference(first[name], second[name])
    return errors


def finite(tensors):
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise AssertionError("Outputs, losses, inputs, gradients and parameters must remain finite")


def installed_identity():
    installed = distribution("ragged-pool")
    direct = installed.read_text("direct_url.json")
    prefix, imported = Path(sys.prefix).resolve(), Path(package.__file__).resolve(strict=True)
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise AssertionError("Use an ordinary installed wheel")
    if (imported != Path(installed.locate_file("ragged_pool/__init__.py")).resolve(strict=True)
            or not imported.is_relative_to(prefix) or installed.version != package.__version__):
        raise AssertionError("The import must match this environment's installed distribution")
    source = Path(__file__).resolve().parents[1]
    if (source / "pyproject.toml").is_file() and prefix.is_relative_to(source):
        raise AssertionError("Use an environment outside the checkout")
    libraries = list(imported.parent.glob("_C*.pyd")) + list(imported.parent.glob("_C*.so"))
    recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    if len(libraries) != 1 or libraries[0].resolve() not in recorded or libraries[0].resolve() not in loaded:
        raise AssertionError("Expected one recorded and loaded installed native library")
    return {"package_path": str(imported), "package_version": package.__version__,
            "environment_prefix": str(prefix), "native_library_loaded": True,
            "native_library_path": str(libraries[0].resolve()),
            "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest(),
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def fixture(dtype_name):
    """Return one seeded initial model and the five fixed packed batches."""
    dtype = DTYPES[dtype_name]
    torch.manual_seed(SEED)
    initial = SetRegressor().to(device="cpu", dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 100)
    batches = []
    for boundaries in OFFSETS:
        n, b = boundaries[-1], len(boundaries) - 1
        raw = 0.25 * torch.randn(n, 3, dtype=dtype, device="cpu", generator=generator)
        targets = (0.4 * torch.randn(b, dtype=dtype, device="cpu", generator=generator)
                   + torch.linspace(-0.3, 0.5, b, dtype=dtype, device="cpu"))
        offsets = torch.tensor(boundaries, dtype=torch.int64, device="cpu")
        finite((raw, targets))
        if float(raw.abs().max()) > 6 or float(targets.abs().max()) > 6:
            raise AssertionError("The fixed consumer data must remain within its declared bounded envelope")
        batches.append({"raw": raw, "offsets": offsets, "targets": targets,
                        "python_boundaries": boundaries})
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    return initial, batches, tolerance


def forward_backward(call, model, raw, offsets, targets):
    inputs = {"raw": raw, "offsets": offsets, "targets": targets}
    before = snapshot(inputs)
    predictions, pooled = call(raw, offsets)
    loss = (predictions - targets).square().mean()
    loss.backward()
    if raw.grad is None or any(value.grad is None for value in model.parameters()):
        raise AssertionError("Each input and parameter block must receive a first gradient")
    gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()}
    maxima = {name: float(value.abs().max()) for name, value in gradients.items()}
    if any(value <= 0 for value in maxima.values()) or float(raw.grad.abs().max()) <= 0:
        raise AssertionError("Each declared training step needs informative input and parameter gradients")
    result = {"predictions": predictions.detach(), "pooled": pooled.detach(), "loss": loss.detach(),
              "input_gradient": raw.grad.detach().clone()}
    finite((*result.values(), *gradients.values()))
    return result, gradients, maxima, unchanged(inputs, before)


def train(initial, batches, tolerance):
    """Compile once, then retain all three model/optimizer trajectories."""
    models = {name: copy.deepcopy(initial) for name in ("compiled", "eager_native", "reference")}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03) for name, model in models.items()}
    parameter_ids = {name: tuple(id(value) for value in model.parameters()) for name, model in models.items()}
    optimizer_ids = {name: id(value) for name, value in optimizers.items()}
    compiled = torch.compile(models["compiled"], backend="aot_eager", fullgraph=True, dynamic=True)
    steps = []
    for index, batch in enumerate(batches, start=1):
        before = {name: {key: value.detach().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}
        parameter_versions = {name: {key: value._version for key, value in model.named_parameters()}
                              for name, model in models.items()}
        raw = {name: batch["raw"].detach().clone().requires_grad_() for name in models}
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        # Only the original native default is passed through the compiled model.
        # The independent reference closure remains outside the compiled region.
        reference = primitive_pool(batch["python_boundaries"])
        calls = {"compiled": compiled, "eager_native": models["eager_native"],
                 "reference": lambda features, offsets: models["reference"](features, offsets, reference)}
        evaluations = {name: forward_backward(calls[name], models[name], raw[name],
                                              batch["offsets"], batch["targets"])
                       for name in models}
        output_errors = {name: compare(evaluations["compiled"][0], evaluations[name][0], tolerance)
                         for name in ("eager_native", "reference")}
        gradient_errors = {name: compare(evaluations["compiled"][1], evaluations[name][1], tolerance)
                           for name in ("eager_native", "reference")}
        for optimizer in optimizers.values():
            optimizer.step()
        states = {name: {key: value.detach().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}
        updated_errors = {name: compare(states["compiled"], states[name], tolerance)
                          for name in ("eager_native", "reference")}
        changes = {name: {key: difference(value, before[name][key]) for key, value in state.items()}
                   for name, state in states.items()}
        if any(value <= 0 for route in changes.values() for value in route.values()):
            raise AssertionError("All five actual steps must change every parameter block in every route")
        finite(value for state in states.values() for value in state.values())
        if any(float(value.abs().max()) > 6 for state in states.values() for value in state.values()):
            raise AssertionError("The fixed model parameters must remain bounded")
        if any(tuple(id(value) for value in model.parameters()) != parameter_ids[name]
               for name, model in models.items()):
            raise AssertionError("Training must retain the original parameter objects")
        if any(id(value) != optimizer_ids[name] for name, value in optimizers.items()):
            raise AssertionError("Training must retain each original optimizer")
        for name, optimizer in optimizers.items():
            if tuple(id(value) for group in optimizer.param_groups for value in group["params"]) != parameter_ids[name]:
                raise AssertionError("Each optimizer must still own its model's original parameters")
        boundaries = batch["python_boundaries"]
        empty = [row for row, (start, stop) in enumerate(zip(boundaries, boundaries[1:])) if start == stop]
        for result, _, _, _ in evaluations.values():
            torch.testing.assert_close(result["pooled"][empty], torch.zeros_like(result["pooled"][empty]),
                                       rtol=0, atol=0)
        versions = {name: {key: {"before": parameter_versions[name][key], "after": value._version}
                           for key, value in model.named_parameters()} for name, model in models.items()}
        if any(value["after"] <= value["before"] for route in versions.values() for value in route.values()):
            raise AssertionError("Each SGD step must advance every parameter version")
        steps.append({"step": index, "N": boundaries[-1], "B": len(boundaries) - 1,
                      "offsets": list(boundaries),
                      "group_lengths": [stop - start for start, stop in zip(boundaries, boundaries[1:])],
                      "losses": {name: float(result[0]["loss"]) for name, result in evaluations.items()},
                      "output_and_input_gradient_errors": output_errors,
                      "parameter_gradient_errors": gradient_errors,
                      "updated_parameter_errors": updated_errors,
                      "parameter_gradient_max_abs": {name: result[2] for name, result in evaluations.items()},
                      "parameter_update_max_abs": changes,
                      "parameter_versions": versions,
                      "all_routes_all_parameter_blocks_changed": True,
                      "empty_group_indices": empty, "empty_group_pooled_zero": True,
                      "same_parameter_objects": True, "same_optimizer_objects": True,
                      "inputs_unchanged": True,
                      "input_versions": {name: result[3] for name, result in evaluations.items()}})
    return steps


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = installed_identity()
    initial, batches, tolerance = fixture(dtype_name)
    tensors = {f"batch{index}:{name}": batch[name] for index, batch in enumerate(batches)
               for name in ("raw", "offsets", "targets")}
    tensors.update({f"initial_parameter:{name}": value for name, value in initial.named_parameters()})
    before = snapshot(tensors)
    steps = train(initial, batches, tolerance)
    versions = unchanged(tensors, before)
    if len(steps) != 5 or tuple(tuple(row["offsets"]) for row in steps) != OFFSETS:
        raise AssertionError("The consumer must execute the complete frozen shape/partition sequence")
    if (steps[2]["N"], steps[2]["B"]) != (steps[3]["N"], steps[3]["B"]) or steps[2]["offsets"] == steps[3]["offsets"]:
        raise AssertionError("The unchanged-metadata partition control must remain present")
    if (steps[0]["N"], steps[0]["B"]) != (steps[4]["N"], steps[4]["B"]) or steps[0]["offsets"] == steps[4]["offsets"]:
        raise AssertionError("The return to an earlier shape must use a different partition")
    return {"status": "passed", "method": "torch.compile.aot_eager.dynamic.packed_training",
            "purpose": "First-order installed packed training across changing N, B and offset partitions",
            "scope": "Pinned CPU graph/autograd integration. No graph-count, optimized-code, general-shape, performance or model-quality claim.",
            "dtype": dtype_name, "device": "cpu", "seed": SEED,
            "compile": {"backend": "aot_eager", "fullgraph": True, "dynamic": True,
                        "compile_calls": 1, "graph_count_observed": False,
                        "optimizer_inside_compiled_region": False},
            "raw_feature_width": 3, "hidden_width": 4, "parameter_tensors": len(tuple(initial.parameters())),
            "scalar_parameters": sum(value.numel() for value in initial.parameters()),
            "architecture": "encoder 3->4, tanh, scorer 4->1 without bias, pool, head 4->1",
            "optimizer": "SGD", "learning_rate": 0.03,
            "optimizer_steps": 5, "compiled_optimizer_steps": 5,
            "native_optimizer_steps": 5, "original_native_optimizer_steps": 5, "reference_optimizer_steps": 5,
            "same_model_parameter_and_optimizer_objects": True,
            "comparison_tolerance": tolerance, "steps": steps,
            "changed_partition_at_same_metadata": True, "returned_shape_with_changed_partition": True,
            "inputs_unchanged": True, "initial_parameters_unchanged": True, "input_versions": versions,
            "gradients_finite": True, "python_version": sys.version.split()[0], "torch_version": torch.__version__,
            "python_executable": sys.executable, "working_directory": str(Path.cwd()),
            "installed_package": True, "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), **identity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt file")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    text = json.dumps(run(arguments.dtype), indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
