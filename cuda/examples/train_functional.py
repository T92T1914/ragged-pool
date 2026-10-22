"""Check installed CUDA functional gradients and two actual SGD updates.

The functional model is compared with ordinary native autograd and independent
per-row GPU primitives. This is synthetic integration evidence when executed,
with no model-quality, performance, portability or security result.
The explicit three-model mode adds bounded vmap(grad) and shared-input checks.
"""
import argparse
import copy
from hashlib import sha256
from importlib.metadata import distribution
import importlib.util
import json
from pathlib import Path
import re
import sys

import torch

import ragged_pool_cuda as package
import ragged_pool_cuda.functional as functional_module
from ragged_pool_cuda.functional import ragged_pool_functional


_HELPER_PATH = Path(__file__).resolve().with_name("train_cuda.py")
_SPEC = importlib.util.spec_from_file_location("_functional_cuda_eager_helper", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("The adjacent train_cuda.py helper is required")
eager = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eager)
DTYPES = eager.DTYPES
PYTHON_PAYLOAD = ("__init__.py", "_registration.py", "_validation.py", "padded.py", "reference.py")
ROUTES = ("functional", "native", "primitive")


class SetRegressor(eager.SetRegressor):
    def forward(self, raw, offsets):
        values = self.encoder(raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = ragged_pool_functional(values, logits, offsets)
        return self.head(pooled).squeeze(-1), pooled


def _bytes(value):
    return value.detach().resolve_neg().resolve_conj().contiguous().reshape(-1).view(torch.uint8).cpu().clone()


def _stamp(value):
    return (id(value), value._version, value.untyped_storage().data_ptr(), value.storage_offset(),
            tuple(value.shape), tuple(value.stride()), value.dtype, value.device, value.requires_grad,
            value.is_neg(), value.is_conj(), _bytes(value))


def snapshot(tensors, *, gradients=False):
    result = {}
    for name, value in tensors.items():
        gradient = value.grad if gradients and value.is_leaf else None
        result[name] = (_stamp(value), None if gradient is None else _stamp(gradient))
    return result


def unchanged(tensors, before, *, gradients=False):
    if tensors.keys() != before.keys():
        raise AssertionError("The tensor collection changed")
    for name, value in tensors.items():
        current = snapshot({name: value}, gradients=gradients)[name]
        for actual, expected in zip(current, before[name]):
            if actual is None or expected is None:
                if actual is not expected:
                    raise AssertionError(f"A pure call changed .grad for {name}")
            elif actual[:-1] != expected[:-1] or not torch.equal(actual[-1], expected[-1]):
                raise AssertionError(f"A pure call changed bytes, versions or ownership for {name}")


def installed_identity():
    identity = eager.installed_identity()
    helper_hash = identity.pop("script_sha256")
    root = Path(package.__file__).resolve(strict=True).parent
    module_path = Path(functional_module.__file__).resolve(strict=True)
    installed = distribution("ragged-pool-cuda")
    recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
    if (module_path != root / "functional.py" or module_path not in recorded
            or not module_path.is_relative_to(Path(sys.prefix).resolve())):
        raise AssertionError("The functional bridge must come from the ordinary installed package")
    return {**identity, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_script_sha256": helper_hash, "functional_module_sha256": sha256(module_path.read_bytes()).hexdigest(),
            "package_python_sha256": {name: sha256((root / name).read_bytes()).hexdigest() for name in PYTHON_PAYLOAD}}


def _parameters(model):
    return dict(model.named_parameters())


def _clones(tensors):
    return {name: value.detach().clone() for name, value in tensors.items()}


def _check_result(result, gradients, batch):
    if any(not bool(torch.isfinite(value).all()) for value in (*result.values(), *gradients.values())):
        raise AssertionError("Every output and first gradient must be finite")
    if any(float(value.abs().max()) <= 0 for value in (*gradients.values(), result["input_gradient"])):
        raise AssertionError("Every parameter block and the raw input need informative gradients")
    empty = [row for row, (start, stop) in enumerate(zip(batch["boundaries"], batch["boundaries"][1:])) if start == stop]
    if not torch.equal(result["pooled"][empty], torch.zeros_like(result["pooled"][empty])):
        raise AssertionError("Empty groups must have zero pooled vectors")


def functional_grad(model, raw, batch):
    parameters = _parameters(model)
    tensors = {**parameters, "raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(tensors, gradients=True)

    def loss(parameters, features):
        predictions, pooled = torch.func.functional_call(model, parameters, (features, batch["offsets"]), strict=True)
        value = (predictions - batch["targets"]).square().mean()
        return value, {"loss": value, "predictions": predictions, "pooled": pooled}

    (parameter_gradients, input_gradient), auxiliary = torch.func.grad(loss, argnums=(0, 1), has_aux=True)(parameters, raw)
    unchanged(tensors, before, gradients=True)
    result = {name: value.detach().clone() for name, value in auxiliary.items()}
    result["input_gradient"] = input_gradient.detach().clone()
    gradients = _clones(parameter_gradients)
    _check_result(result, gradients, batch)
    return result, gradients


def ordinary_grad(model, raw, batch, reference):
    tensors = {**_parameters(model), "raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(tensors)
    result, gradients = eager.evaluate(model, raw, batch, reference)
    unchanged(tensors, before)
    _check_result(result, gradients, batch)
    return result, gradients


def _model_vjps(trained, batch, tolerance):
    models = {name: copy.deepcopy(trained) if name == "functional" else eager.SetRegressor().to(
              device=batch["raw"].device, dtype=batch["raw"].dtype) for name in ROUTES}
    state = trained.state_dict()
    for model in models.values():
        model.load_state_dict(state)
        model.zero_grad(set_to_none=True)
    raw = {name: batch["raw"].detach().clone().requires_grad_() for name in ROUTES}
    parameters = _parameters(models["functional"])
    tensors = {**{f"{route}:{key}": value for route, model in models.items() for key, value in model.named_parameters()},
               **{f"raw:{route}": value for route, value in raw.items()}, "offsets": batch["offsets"]}
    before = snapshot(tensors, gradients=True)

    def outputs(parameters, features):
        return torch.func.functional_call(models["functional"], parameters, (features, batch["offsets"]), strict=True)

    predictions, pullback = torch.func.vjp(outputs, parameters, raw["functional"])
    native = models["native"](raw["native"], batch["offsets"])
    primitive = models["primitive"](raw["primitive"], batch["offsets"], batch["boundaries"])
    output_errors = {route: eager.compare(dict(zip(("predictions", "pooled"), predictions)),
                    dict(zip(("predictions", "pooled"), values)), tolerance)
                     for route, values in (("native", native), ("primitive", primitive))}
    count = predictions[0].numel()
    first = (torch.linspace(-0.7, 0.9, count, device=raw["functional"].device, dtype=raw["functional"].dtype),
             torch.linspace(0.6, -0.4, count * 4, device=raw["functional"].device,
                            dtype=raw["functional"].dtype).reshape(count, 4))
    second = (first[0].flip(0) * 0.3 - 0.2, first[1].flip(0) * -0.4 + 0.1)
    cases, previous = [], None
    for index, cotangents in enumerate((first, second), 1):
        cotangent_tensors = {f"cotangent:{i}": value for i, value in enumerate(cotangents)}
        cotangent_before = snapshot(cotangent_tensors)
        parameter_vjp, input_vjp = pullback(cotangents)
        actual = {**parameter_vjp, "raw": input_vjp}
        errors = {}
        for route, values in (("native", native), ("primitive", primitive)):
            gradients = torch.autograd.grad(values, (*models[route].parameters(), raw[route]),
                                            grad_outputs=cotangents, retain_graph=True)
            expected = dict(zip((*_parameters(models[route]), "raw"), gradients))
            errors[route] = eager.compare(actual, expected, tolerance)
        if previous is not None and all(torch.equal(actual[key], previous[key]) for key in actual):
            raise AssertionError("Distinct cotangents must change this model's VJP")
        if not bool(torch.isfinite(input_vjp).all()) or float(input_vjp.abs().max()) <= 0:
            raise AssertionError("The model VJP must have an informative raw gradient")
        previous = _clones(actual)
        unchanged(cotangent_tensors, cotangent_before)
        unchanged(tensors, before, gradients=True)
        cases.append({"case": index, "cotangent_shapes": [list(value.shape) for value in cotangents],
                      "complete_vjp_max_abs_errors": errors, "raw_vjp_max_abs": float(input_vjp.abs().max()),
                      "inputs_parameters_and_grad_fields_unchanged": True})
    return {"status": "passed", "outputs": ["predictions", "pooled"], "cotangent_cases": cases,
            "output_max_abs_errors": output_errors, "same_pullback_reused": True,
            "distinct_cotangents_change_vjp": True, "optimizer_steps": 0}


def _lane(parameters, index):
    return {name: value[index] for name, value in parameters.items()}


def _mapped_grad(template, parameters, raw, batch):
    tensors = {**parameters, "raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(tensors, gradients=True)

    def loss(params, features):
        predictions, pooled = torch.func.functional_call(
            template, params, (features, batch["offsets"]), strict=True)
        value = (predictions - batch["targets"]).square().mean()
        return value, {"loss": value, "predictions": predictions, "pooled": pooled}

    (gradients, raw_gradients), auxiliary = torch.vmap(
        torch.func.grad(loss, argnums=(0, 1), has_aux=True), in_dims=(0, None))(parameters, raw)
    unchanged(tensors, before, gradients=True)
    result = _clones(auxiliary)
    result["input_gradient"] = raw_gradients.detach().clone()
    gradients = _clones(gradients)
    for index in range(3):
        _check_result(_lane(result, index), _lane(gradients, index), batch)
    return result, gradients


def _serial_grad(template, parameters, raw, batch, reference):
    tensors = {**parameters, "raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(tensors, gradients=True)
    results, gradient_rows = [], []
    for index in range(3):
        params = _lane(parameters, index)
        features = raw.detach().clone().requires_grad_()
        predictions, pooled = torch.func.functional_call(
            template, params, (features, batch["offsets"], batch["boundaries"] if reference else None),
            strict=True)
        loss = (predictions - batch["targets"]).square().mean()
        derivatives = torch.autograd.grad(loss, (*params.values(), features))
        result = {"loss": loss.detach().clone(), "predictions": predictions.detach().clone(),
                  "pooled": pooled.detach().clone(), "input_gradient": derivatives[-1].detach().clone()}
        gradients = {name: value.detach().clone() for name, value in zip(params, derivatives[:-1])}
        _check_result(result, gradients, batch)
        results.append(result)
        gradient_rows.append(gradients)
    unchanged(tensors, before, gradients=True)
    return ({name: torch.stack([result[name] for result in results]) for name in results[0]},
            {name: torch.stack([row[name] for row in gradient_rows]) for name in parameters})


def _shared_gradients(template, templates, trained, raw, batch, tolerance):
    parameters = _clones(trained)
    tensors = {**parameters, "raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(tensors, gradients=True)

    def objectives(features):
        def one_model(params):
            predictions, _ = torch.func.functional_call(
                template, params, (features, batch["offsets"]), strict=True)
            return (predictions - batch["targets"]).square().mean()
        return torch.vmap(one_model)(parameters)

    actual_sum = torch.func.grad(lambda features: objectives(features).sum())(raw)
    actual_mean = torch.func.grad(lambda features: objectives(features).mean())(raw)
    contributions = {}
    for route in ("native", "primitive"):
        rows = []
        for index in range(3):
            features = raw.detach().clone().requires_grad_()
            predictions, _ = torch.func.functional_call(
                templates[route], _lane(parameters, index),
                (features, batch["offsets"], batch["boundaries"] if route == "primitive" else None),
                strict=True)
            loss = (predictions - batch["targets"]).square().mean()
            rows.append(torch.autograd.grad(loss, features)[0].detach().clone())
        contributions[route] = torch.stack(rows)
    contribution_errors = eager.compare({"contributions": contributions["native"]},
                                        {"contributions": contributions["primitive"]}, tolerance)
    errors = {route: eager.compare({"sum": actual_sum, "mean": actual_mean},
                                   {"sum": value.sum(0), "mean": value.mean(0)}, tolerance)
              for route, value in contributions.items()}
    torch.testing.assert_close(actual_sum, 3 * actual_mean, **tolerance)
    if torch.allclose(actual_sum, actual_mean, **tolerance):
        raise AssertionError("The shared-input sum and mean must distinguish the objectives")
    pairwise = [eager.difference(contributions["native"][first], contributions["native"][second])
                for first in range(3) for second in range(first + 1, 3)]
    if any(value <= tolerance["atol"] for value in pairwise):
        raise AssertionError("Shared-input gradients need unequal model contributions")
    unchanged(tensors, before, gradients=True)
    return {"status": "passed", "models": 3, "optimizer_steps": 0, "parameters_fixed": True,
            "objectives": ["sum of three complete-model MSE losses", "mean of three complete-model MSE losses"],
            "complete_gradient_max_abs_errors": errors,
            "serial_native_vs_primitive_contribution_errors": contribution_errors,
            "sum_vs_mean_max_abs_difference": eager.difference(actual_sum, actual_mean),
            "sum_differs_from_mean": True, "unequal_model_contributions": True,
            "pairwise_contribution_max_abs_differences": pairwise,
            "per_model_input_gradient_max_abs": [float(value.abs().max()) for value in contributions["native"]],
            "input_gradient_shape": list(actual_sum.shape), "inputs_parameters_and_grad_fields_unchanged": True}


def _run_models(dtype_name, source_revision, source_tree, identity, initial, batches, fixture_hash):
    device, dtype = batches[0]["raw"].device, DTYPES[dtype_name]
    tolerance = {"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 2e-10, "atol": 2e-12}
    seeds = list(range(eager.SEED, eager.SEED + 3))
    originals = []
    for seed in seeds:
        torch.manual_seed(seed)
        originals.append(eager.SetRegressor().to(device=device, dtype=dtype))
    stacked = {name: torch.stack([_parameters(model)[name].detach() for model in originals])
               for name in _parameters(initial)}
    if any(torch.equal(stacked["encoder.weight"][first], stacked["encoder.weight"][second])
           for first in range(3) for second in range(first + 1, 3)):
        raise AssertionError("The seeded complete-model states must be distinct")
    templates = {"functional": SetRegressor().to(device=device, dtype=dtype),
                 "native": copy.deepcopy(initial), "primitive": copy.deepcopy(initial)}
    states = {route: {name: value.detach().clone().requires_grad_() for name, value in stacked.items()}
              for route in ROUTES}
    if not all(value.is_leaf and value.requires_grad and value.grad_fn is None and value.shape[0] == 3
               for state in states.values() for value in state.values()):
        raise AssertionError("Optimizers must own new stacked parameter leaves")
    optimizers = {route: torch.optim.SGD(list(state.values()), lr=0.03, momentum=0, foreach=False, fused=False)
                  for route, state in states.items()}

    def ownership(route):
        return (id(templates[route]), id(optimizers[route]),
                tuple((name, id(value), value.untyped_storage().data_ptr()) for name, value in states[route].items()))

    owners = {route: ownership(route) for route in ROUTES}
    if len({value.untyped_storage().data_ptr() for state in states.values() for value in state.values()}) != 15:
        raise AssertionError("Training routes must own independent stacked parameter storage")
    fixed = {**{f"source:{index}:{name}": batch[name] for index, batch in enumerate(batches)
                for name in ("raw", "offsets", "targets")},
             **{f"template:{route}:{name}": value for route, model in templates.items()
                for name, value in model.named_parameters()},
             **{f"initial:{name}": value for name, value in initial.named_parameters()},
             **{f"original:{index}:{name}": value for index, model in enumerate(originals)
                for name, value in model.named_parameters()},
             **{f"stacked:{name}": value for name, value in stacked.items()}}
    fixed_before = snapshot(fixed, gradients=True)
    steps = []
    for index, batch in enumerate(batches, 1):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        before = {route: _clones(state) for route, state in states.items()}
        raw = {route: batch["raw"].detach().clone().requires_grad_() for route in ROUTES}
        results = {"functional": _mapped_grad(templates["functional"], states["functional"], raw["functional"], batch)}
        results.update({route: _serial_grad(templates[route], states[route], raw[route], batch, route == "primitive")
                        for route in ("native", "primitive")})
        output_errors = {route: eager.compare(results["functional"][0], results[route][0], tolerance)
                         for route in ("native", "primitive")}
        gradient_errors = {route: eager.compare(results["functional"][1], results[route][1], tolerance)
                           for route in ("native", "primitive")}
        for route, state in states.items():
            if raw[route].grad is not None or any(value.grad is not None for value in state.values()):
                raise AssertionError("Pure mapped and serial derivative calls must preserve .grad fields")
            for name, parameter in state.items():
                parameter.grad = results[route][1][name].detach().clone()
            optimizers[route].step()
        update_errors = {route: eager.compare(states["functional"], states[route], tolerance)
                         for route in ("native", "primitive")}
        changes = {route: [{name: eager.difference(value[lane], before[route][name][lane])
                            for name, value in state.items()} for lane in range(3)]
                   for route, state in states.items()}
        if any(value <= 0 for route in changes.values() for lane in route for value in lane.values()):
            raise AssertionError("Every model and route must change every parameter block at each update")
        for route in ROUTES:
            if (ownership(route) != owners[route]
                    or tuple(id(value) for group in optimizers[route].param_groups for value in group["params"])
                    != tuple(id(value) for value in states[route].values())):
                raise AssertionError("Training changed model, stacked parameter or optimizer ownership")
        unchanged(fixed, fixed_before, gradients=True)
        steps.append({"update": index, "N": batch["boundaries"][-1], "B": len(batch["boundaries"]) - 1,
                      "offsets": list(batch["boundaries"]),
                      "losses": {route: [float(value) for value in result[0]["loss"]] for route, result in results.items()},
                      "complete_output_loss_and_raw_gradient_max_abs_errors": output_errors,
                      "complete_parameter_gradient_max_abs_errors": gradient_errors,
                      "complete_updated_parameter_max_abs_errors": update_errors,
                      "per_model_parameter_update_max_abs": changes,
                      "all_models_all_parameter_blocks_changed": True,
                      "pure_calls_preserve_grad_fields": True,
                      "inputs_and_templates_unchanged": True,
                      "same_stacked_parameter_optimizer_objects_and_storage": True})
    shared = _shared_gradients(templates["functional"], templates, states["functional"],
                               batches[-1]["raw"], batches[-1], tolerance)
    unchanged(fixed, fixed_before, gradients=True)
    if installed_identity() != identity:
        raise AssertionError("Installed code or consumer identity changed during execution")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "eager.cuda.vmap.grad.functional_call.training",
            "scope": "Three complete models share packed batches. Serial native lanes, without per-bag, parallel-kernel, speed, quality, security or general transform claims.",
            "dtype": dtype_name, "device": str(device), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor], "models": 3, "model_seeds": seeds,
            "model_states_distinct": True, "parameters_stacked_leaf": True,
            "mapped_axis": "leading parameter dictionary axis", "shared_raw_input": True, "shared_offsets": True,
            "stacked_parameter_shapes": {name: list(value.shape) for name, value in states["functional"].items()},
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda,
            "source_revision": source_revision, "source_tree": source_tree,
            "source_identity_basis": "Caller source labels plus installed module, helper, consumer and native byte hashes",
            "fixture_sha256": fixture_hash, "optimizer": "SGD", "learning_rate": 0.03, "momentum": 0,
            "foreach": False, "fused": False, "optimizer_steps": 2, "native_optimizer_steps": 2,
            "reference_optimizer_steps": 2, "parameter_tensors_per_model": 5, "scalar_parameters_per_model": 25,
            "comparison_tolerance": tolerance, "steps": steps, "shared_input_gradients": shared,
            "inputs_unchanged": True, "initial_parameters_unchanged": True, "template_parameters_unchanged": True,
            "package_identity_unchanged": True, "python_executable": sys.executable, "working_directory": str(Path.cwd()),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), **identity}


def run(dtype_name, device_index=0, source_revision=None, source_tree=None, models=1):
    for label in (source_revision, source_tree):
        if label is not None and not re.fullmatch(r"[0-9a-f]{40}", label):
            raise ValueError("Source labels must be complete lowercase SHA1 identifiers")
    if dtype_name not in DTYPES or device_index != 0:
        raise ValueError("This consumer supports float32/float64 on CUDA device zero only")
    if type(models) is not int or models not in (1, 3):
        raise ValueError("The functional consumer supports one model or the explicit three-model mode")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if sys.version.split()[0] != "3.11.8" or str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise AssertionError("Use the declared Python 3.11.8 and PyTorch 2.9.1 cu128 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA device is unavailable")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    identity = installed_identity()
    initial, batches, fixture_hash = eager.fixture(dtype_name, device)
    if models == 3:
        return _run_models(dtype_name, source_revision, source_tree, identity, initial, batches, fixture_hash)
    initial_before = snapshot(_parameters(initial), gradients=True)
    source_inputs = {f"{index}:{key}": batch[key] for index, batch in enumerate(batches)
                     for key in ("raw", "offsets", "targets")}
    source_before = snapshot(source_inputs, gradients=True)
    models = {name: copy.deepcopy(initial) for name in ("native", "primitive")}
    models["functional"] = SetRegressor().to(device=device, dtype=DTYPES[dtype_name])
    models["functional"].load_state_dict(initial.state_dict())
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03, momentum=0, foreach=False, fused=False)
                  for name, model in models.items()}
    ownership = {name: (id(model), id(optimizers[name]), tuple(id(value) for value in model.parameters()),
                       tuple(value.untyped_storage().data_ptr() for value in model.parameters()))
                 for name, model in models.items()}
    if len({pointer for entry in ownership.values() for pointer in entry[3]}) != 15:
        raise AssertionError("The three training routes must own independent parameter storage")
    tolerance = {"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 2e-10, "atol": 2e-12}
    steps = []
    for index, batch in enumerate(batches, 1):
        before = {name: _clones(_parameters(model)) for name, model in models.items()}
        raw = {name: batch["raw"].detach().clone().requires_grad_() for name in ROUTES}
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        results = {"functional": functional_grad(models["functional"], raw["functional"], batch)}
        results.update({name: ordinary_grad(models[name], raw[name], batch, name == "primitive")
                        for name in ("native", "primitive")})
        output_errors = {name: eager.compare(results["functional"][0], results[name][0], tolerance)
                         for name in ("native", "primitive")}
        gradient_errors = {name: eager.compare(results["functional"][1], results[name][1], tolerance)
                           for name in ("native", "primitive")}
        if raw["functional"].grad is not None or any(value.grad is not None for value in models["functional"].parameters()):
            raise AssertionError("Pure functional gradients must leave existing .grad fields untouched")
        for key, parameter in models["functional"].named_parameters():
            parameter.grad = results["functional"][1][key].detach().clone()
        for optimizer in optimizers.values():
            optimizer.step()
        states = {name: _clones(_parameters(model)) for name, model in models.items()}
        update_errors = {name: eager.compare(states["functional"], states[name], tolerance)
                         for name in ("native", "primitive")}
        changes = {name: {key: eager.difference(value, before[name][key]) for key, value in state.items()}
                   for name, state in states.items()}
        if any(value <= 0 for route in changes.values() for value in route.values()):
            raise AssertionError("Every actual update must change all five parameter blocks")
        for name, model in models.items():
            actual = (id(model), id(optimizers[name]), tuple(id(value) for value in model.parameters()),
                      tuple(value.untyped_storage().data_ptr() for value in model.parameters()))
            if actual != ownership[name] or tuple(id(value) for group in optimizers[name].param_groups
                                                   for value in group["params"]) != ownership[name][2]:
                raise AssertionError("Training changed model or optimizer ownership")
        unchanged(source_inputs, source_before, gradients=True)
        steps.append({"update": index, "N": batch["boundaries"][-1], "B": len(batch["boundaries"]) - 1,
                      "offsets": list(batch["boundaries"]), "losses": {name: float(result[0]["loss"]) for name, result in results.items()},
                      "output_max_abs_errors": output_errors, "parameter_gradient_max_abs_errors": gradient_errors,
                      "updated_parameter_max_abs_errors": update_errors, "parameter_update_max_abs": changes,
                      "pure_call_preserves_grad_fields": True, "inputs_unchanged": True,
                      "all_parameter_blocks_changed": True, "same_model_parameter_optimizer_objects_and_storage": True})
    vjp = _model_vjps(models["functional"], batches[-1], tolerance)
    unchanged(_parameters(initial), initial_before, gradients=True)
    unchanged(source_inputs, source_before, gradients=True)
    final_identity = installed_identity()
    if final_identity != identity:
        raise AssertionError("Installed code or consumer identity changed during execution")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "eager.torch.func.grad.functional_call.cuda_training",
            "scope": "First-order packed grad and model VJP integration only. No nested transforms, jvp, vmap, compile or export qualification.",
            "dtype": dtype_name, "device": str(device), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor], "seed": eager.SEED,
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda,
            "source_revision": source_revision, "source_tree": source_tree,
            "source_identity_basis": "Caller source labels plus installed module, helper, consumer and native byte hashes",
            "fixture_sha256": fixture_hash, "optimizer": "SGD", "learning_rate": 0.03, "momentum": 0,
            "foreach": False, "fused": False, "optimizer_steps": 2, "native_optimizer_steps": 2,
            "reference_optimizer_steps": 2, "parameter_tensors": 5, "scalar_parameters": 25,
            "comparison_tolerance": tolerance, "steps": steps, "model_vjp": vjp,
            "inputs_unchanged": True, "initial_parameters_unchanged": True, "package_identity_unchanged": True,
            "python_executable": sys.executable, "working_directory": str(Path.cwd()),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--models", type=int, choices=(1, 3), default=1,
                        help="Use 3 for bounded vmap(grad) training with distinct stacked models")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    arguments = parser.parse_args()
    if arguments.output is not None and (arguments.output.exists() or arguments.output.is_symlink()):
        parser.error("The output receipt must be new")
    result = run(arguments.dtype, arguments.device, arguments.source_revision, arguments.source_tree, arguments.models)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
