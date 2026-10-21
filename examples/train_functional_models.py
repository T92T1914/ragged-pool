"""Train three complete packed models using functional parameter batches.

Stacked leaf parameter dictionaries, functional_call and vmap(grad(loss)) drive
two SGD updates. Every model is checked against serial original-native and
independent PyTorch pooling. The trained models also check summed and mean
gradients of their shared raw input. These lanes are models sharing a complete
packed batch, not individual variable-length bags. This is integration, with no
model-quality, privacy, robustness, parallel-execution or performance result.
"""
import argparse
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

import torch
from torch import nn

import ragged_pool as package
from ragged_pool import ragged_pool, ragged_pool_forward_ad


SEED = 1507
MODELS = 3
DTYPES = {"float32": torch.float32, "float64": torch.float64}


class SetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        # A shared scalar score bias cancels inside each group.
        self.scorer = nn.Linear(4, 1, bias=False)
        self.head = nn.Linear(4, 1)

    def forward(self, raw, offsets, pool):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = pool(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def primitive_pool(boundaries):
    """Capture Python boundaries before either functional transformation."""
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
        raise AssertionError("Every compared output, loss, gradient and parameter must be finite")


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


def lane_parameters(parameters, lane):
    return {name: value[lane] for name, value in parameters.items()}


def evaluate(template, parameters, raw, offsets, targets, pool):
    predictions, pooled = torch.func.functional_call(template, parameters, (raw, offsets, pool))
    loss = (predictions - targets).square().mean()
    return {"predictions": predictions, "pooled": pooled, "loss": loss}


def mapped_evaluation(template, parameters, raw, offsets, targets):
    def one_model(params):
        return evaluate(template, params, raw, offsets, targets, ragged_pool_forward_ad)
    return torch.vmap(one_model)(parameters)


def train(template, initial, raw, offsets, targets, reference_pool, tolerance):
    routes = {"mapped": ragged_pool_forward_ad, "original_native": ragged_pool,
              "reference": reference_pool}
    states = {route: {name: value.detach().clone().requires_grad_() for name, value in initial.items()}
              for route in routes}
    if not all(value.is_leaf and value.requires_grad and value.grad_fn is None
               and value.shape[0] == MODELS for state in states.values() for value in state.values()):
        raise AssertionError("Each optimizer must own leaf stacked parameter tensors")
    optimizers = {route: torch.optim.SGD(list(state.values()), lr=0.03)
                  for route, state in states.items()}
    steps = []
    for step in (1, 2):
        before = {route: {name: value.detach().clone() for name, value in state.items()}
                  for route, state in states.items()}
        inputs = {"raw": raw, "offsets": offsets, "targets": targets}
        input_before = snapshot(inputs)
        evaluations = {"mapped": mapped_evaluation(template, states["mapped"], raw, offsets, targets)}
        def one_loss(params):
            return evaluate(template, params, raw, offsets, targets, ragged_pool_forward_ad)["loss"]
        gradients = {"mapped": torch.vmap(torch.func.grad(one_loss))(states["mapped"])}
        for route in ("original_native", "reference"):
            rows, gradient_rows = [], []
            for lane in range(MODELS):
                params = lane_parameters(states[route], lane)
                result = evaluate(template, params, raw, offsets, targets, routes[route])
                derivatives = torch.autograd.grad(result["loss"], tuple(params.values()))
                rows.append(result)
                gradient_rows.append(dict(zip(params, derivatives)))
            evaluations[route] = {name: torch.stack([row[name] for row in rows]) for name in rows[0]}
            gradients[route] = {name: torch.stack([row[name] for row in gradient_rows])
                                for name in states[route]}
        finite(value for result in evaluations.values() for value in result.values())
        finite(value for result in gradients.values() for value in result.values())
        output_errors, gradient_errors = {}, {}
        for route in ("original_native", "reference"):
            output_errors[route], gradient_errors[route] = [], []
            for lane in range(MODELS):
                output_errors[route].append(compare(
                    {name: value[lane] for name, value in evaluations["mapped"].items()},
                    {name: value[lane] for name, value in evaluations[route].items()}, tolerance))
                gradient_errors[route].append(compare(lane_parameters(gradients["mapped"], lane),
                                                       lane_parameters(gradients[route], lane), tolerance))
        for route, optimizer in optimizers.items():
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in states[route].items():
                parameter.grad = gradients[route][name].detach().clone()
            optimizer.step()
        finite(value for state in states.values() for value in state.values())
        updated_errors = {route: [compare(lane_parameters(states["mapped"], lane),
                                          lane_parameters(states[route], lane), tolerance)
                                  for lane in range(MODELS)]
                          for route in ("original_native", "reference")}
        changes = {route: [{name: difference(value[lane], before[route][name][lane])
                            for name, value in state.items()} for lane in range(MODELS)]
                   for route, state in states.items()}
        if any(change == 0 for route in changes.values() for lane in route for change in lane.values()):
            raise AssertionError("Each actual update must change every parameter block of every model and route")
        versions = unchanged(inputs, input_before)
        # The second group's pooled vector is zero, including inside every map.
        for result in evaluations.values():
            torch.testing.assert_close(result["pooled"][:, 1], torch.zeros_like(result["pooled"][:, 1]),
                                       rtol=0, atol=0)
        steps.append({"step": step,
                      "losses": {route: [float(loss.detach()) for loss in result["loss"]]
                                 for route, result in evaluations.items()},
                      "per_model_output_and_loss_errors": output_errors,
                      "per_model_parameter_gradient_errors": gradient_errors,
                      "per_model_updated_parameter_errors": updated_errors,
                      "parameter_update_max_abs": changes,
                      "all_models_all_parameter_blocks_changed": True,
                      "gradient_shapes": {name: list(value.shape) for name, value in gradients["mapped"].items()},
                      "empty_group_pooled_zero": True, "inputs_unchanged": True, "input_versions": versions})
    return states["mapped"], steps


def shared_input_gradients(template, trained, raw, offsets, targets, reference_pool, tolerance):
    """Accumulate three distinct model contributions to one shared raw input."""
    parameters = {name: value.detach().clone() for name, value in trained.items()}
    tensors = {"raw": raw, "offsets": offsets, "targets": targets,
               **{f"parameter:{name}": value for name, value in parameters.items()}}
    before = snapshot(tensors)
    def ensemble_sum(features):
        return mapped_evaluation(template, parameters, features, offsets, targets)["loss"].sum()
    def ensemble_mean(features):
        return mapped_evaluation(template, parameters, features, offsets, targets)["loss"].mean()
    actual_sum = torch.func.grad(ensemble_sum)(raw)
    actual_mean = torch.func.grad(ensemble_mean)(raw)
    individual = {}
    for route, pool in (("original_native", ragged_pool), ("reference", reference_pool)):
        contributions = []
        for lane in range(MODELS):
            features = raw.detach().clone().requires_grad_()
            result = evaluate(template, lane_parameters(parameters, lane), features, offsets, targets, pool)
            contributions.append(torch.autograd.grad(result["loss"], features)[0].detach())
        individual[route] = torch.stack(contributions)
    torch.testing.assert_close(individual["original_native"], individual["reference"], **tolerance)
    sum_errors, mean_errors = {}, {}
    for route, contributions in individual.items():
        expected_sum, expected_mean = contributions.sum(dim=0), contributions.mean(dim=0)
        torch.testing.assert_close(actual_sum, expected_sum, **tolerance)
        torch.testing.assert_close(actual_mean, expected_mean, **tolerance)
        sum_errors[route] = difference(actual_sum, expected_sum)
        mean_errors[route] = difference(actual_mean, expected_mean)
    torch.testing.assert_close(actual_sum, MODELS * actual_mean, **tolerance)
    if torch.allclose(actual_sum, actual_mean, **tolerance):
        raise AssertionError("The sum-versus-mean control must distinguish the two objectives")
    unequal = [difference(individual["original_native"][first], individual["original_native"][second])
               for first in range(MODELS) for second in range(first + 1, MODELS)]
    if any(error <= tolerance["atol"] for error in unequal):
        raise AssertionError("The shared-input check needs distinct model contributions")
    finite((actual_sum, actual_mean, *individual.values()))
    versions = unchanged(tensors, before)
    return {"status": "passed", "models": MODELS, "parameters_fixed": True, "optimizer_steps": 0,
            "objectives": ["sum of three complete-model MSE losses", "mean of three complete-model MSE losses"],
            "sum_max_abs_error": max(sum_errors.values()), "mean_max_abs_error": max(mean_errors.values()),
            "sum_errors": sum_errors, "mean_errors": mean_errors,
            "sum_differs_from_mean": True, "sum_vs_mean_max_abs_difference": difference(actual_sum, actual_mean),
            "unequal_model_contributions": True, "pairwise_contribution_max_abs_differences": unequal,
            "serial_native_vs_reference_contribution_error": difference(individual["original_native"],
                                                                         individual["reference"]),
            "per_model_input_gradient_max_abs": [float(value.abs().max())
                                                  for value in individual["original_native"]],
            "input_gradient_shape": list(actual_sum.shape), "inputs_unchanged": True, "input_versions": versions}


def run(dtype_name):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    identity = installed_identity()
    dtype = DTYPES[dtype_name]
    tolerance = ({"rtol": 4e-5, "atol": 4e-6} if dtype_name == "float32"
                 else {"rtol": 1e-9, "atol": 1e-11})
    models = []
    for seed in range(SEED, SEED + MODELS):
        torch.manual_seed(seed)
        models.append(SetRegressor().to(device="cpu", dtype=dtype))
    template = models[0]
    initial = {name: torch.stack([dict(model.named_parameters())[name].detach() for model in models])
               for name, _ in template.named_parameters()}
    if any(torch.equal(initial["encoder.weight"][first], initial["encoder.weight"][second])
           for first in range(MODELS) for second in range(first + 1, MODELS)):
        raise AssertionError("The seeded model parameter states must differ")
    generator = torch.Generator(device="cpu").manual_seed(SEED + 17)
    raw = 0.5 * torch.randn(6, 3, generator=generator, dtype=dtype, device="cpu")
    offsets = torch.tensor([0, 2, 2, 3, 6], dtype=torch.int64, device="cpu")
    targets = torch.tensor([-0.7, 0.2, 0.9, -0.1], dtype=dtype, device="cpu")
    reference_pool = primitive_pool(tuple(offsets.tolist()))
    tensors = {"raw": raw, "offsets": offsets, "targets": targets,
               **{f"template:{name}": value for name, value in template.named_parameters()}}
    before = snapshot(tensors)
    trained, steps = train(template, initial, raw, offsets, targets, reference_pool, tolerance)
    shared = shared_input_gradients(template, trained, raw, offsets, targets, reference_pool, tolerance)
    versions = unchanged(tensors, before)
    return {"status": "passed", "purpose": "Synthetic complete-model functional training integration",
            "scope": "Three models share one complete packed batch. No per-bag gradient, general nested transform, quality, privacy, robustness or performance result.",
            "method": "torch.func.vmap.grad.functional_call", "pool_api": "ragged_pool_forward_ad",
            "dtype": dtype_name, "device": "cpu", "seed": SEED, "model_seeds": list(range(SEED, SEED + MODELS)),
            "models": MODELS, "model_states_distinct": True, "raw_shape": list(raw.shape),
            "group_lengths": [2, 0, 1, 3], "shared_raw_input": True, "shared_offsets": True,
            "mapped_axis": "leading parameter dictionary axis", "parameters_stacked_leaf": True,
            "stacked_parameter_shapes": {name: list(value.shape) for name, value in trained.items()},
            "architecture": "encoder 3->4, tanh, scorer 4->1 without bias, pool, head 4->1",
            "parameter_tensors_per_model": len(tuple(template.parameters())),
            "scalar_parameters_per_model": sum(parameter.numel() for parameter in template.parameters()),
            "optimizer": "SGD", "learning_rate": 0.03, "objective": "Independent complete-model MSE",
            "optimizer_steps": 2, "original_native_optimizer_steps": 2, "reference_optimizer_steps": 2,
            "comparison_tolerance": tolerance, "steps": steps, "shared_input_gradients": shared,
            "inputs_unchanged": True, "template_parameters_unchanged": True, "input_versions": versions,
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
