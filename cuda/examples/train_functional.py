"""Check installed CUDA functional gradients, two SGD updates and model VJPs.

The functional model is compared with ordinary native autograd and independent
per-row GPU primitives. This is synthetic integration evidence when executed,
with no model-quality, performance, portability or security result.
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


def run(dtype_name, device_index=0, source_revision=None, source_tree=None):
    for label in (source_revision, source_tree):
        if label is not None and not re.fullmatch(r"[0-9a-f]{40}", label):
            raise ValueError("Source labels must be complete lowercase SHA1 identifiers")
    if dtype_name not in DTYPES or device_index != 0:
        raise ValueError("This consumer supports float32/float64 on CUDA device zero only")
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
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt path")
    arguments = parser.parse_args()
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
