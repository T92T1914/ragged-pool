"""Compare installed compiled CUDA training with eager native and primitive math.

Copy this file and its sibling train_cuda.py outside the checkout before running.
The existing eager consumer supplies the model, independent primitive reference,
comparison helpers and installed-package checks. This route captures first-order
forward and backward graphs with aot_eager. It makes no optimized-code, speed,
model-quality, portability or security claim.
"""
import argparse
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys

import torch


SCRIPT = Path(__file__).resolve(strict=True)
HELPER = SCRIPT.with_name("train_cuda.py").resolve(strict=True)
_spec = importlib.util.spec_from_file_location("_ragged_cuda_eager_consumer", HELPER)
if _spec is None or _spec.loader is None:
    raise ImportError("Copy train_cuda.py beside this installed consumer")
eager = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eager)

BOUNDARIES = ((0, 2, 2, 5, 7, 9), (0, 1, 4, 4, 7, 9))
COMPILE_OPTIONS = {"backend": "aot_eager", "fullgraph": True, "dynamic": False}
ROUTES = ("compiled", "eager", "primitive")
PAIRS = (("compiled", "eager"), ("compiled", "primitive"), ("eager", "primitive"))


def tensor_hash(value):
    """Hash tensor metadata and logical bytes through an explicit consumer copy."""
    copied = value.detach().cpu().contiguous()
    digest = sha256(json.dumps([list(copied.shape), str(copied.dtype)]).encode())
    digest.update(copied.numpy().tobytes())
    return digest.hexdigest()


def hashes(tensors):
    return {name: tensor_hash(value) for name, value in tensors.items()}


def snapshot(tensors):
    return {name: (tensor_hash(value), value._version, id(value), value.data_ptr(),
                   value.storage_offset(), tuple(value.stride()), value.device)
            for name, value in tensors.items()}


def unchanged(tensors, before):
    if tensors.keys() != before.keys() or snapshot(tensors) != before:
        raise AssertionError("Training changed source bytes, versions or tensor ownership")


def parameter_ownership(models, optimizers, expected):
    for route, model in models.items():
        parameters = tuple(model.parameters())
        actual = (id(model), id(optimizers[route]), tuple(id(value) for value in parameters),
                  tuple(value.data_ptr() for value in parameters),
                  tuple(id(value) for group in optimizers[route].param_groups
                        for value in group["params"]))
        if actual != expected[route]:
            raise AssertionError("Training changed model, parameter or optimizer ownership")


def fixture(dtype_name, device):
    dtype = eager.DTYPES[dtype_name]
    torch.manual_seed(eager.SEED)
    initial = eager.SetRegressor().to(device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(eager.SEED + 100)
    batches = []
    for boundaries in BOUNDARIES:
        raw = 0.25 * torch.randn(9, 3, dtype=dtype, generator=generator)
        targets = (0.4 * torch.randn(5, dtype=dtype, generator=generator)
                   + torch.linspace(-0.3, 0.5, 5, dtype=dtype))
        batches.append({"raw": raw.to(device), "targets": targets.to(device),
                        "offsets": torch.tensor(boundaries, dtype=torch.int64, device=device),
                        "boundaries": boundaries})
    return initial, batches


def evaluate(call, model, raw, batch, reference, compiled=False):
    inputs = {"raw": raw, "offsets": batch["offsets"], "targets": batch["targets"]}
    before = snapshot(inputs)
    predictions, pooled = call(raw, batch["offsets"], batch["boundaries"] if reference else None)
    # A pinned runtime witness prevents a disabled compiler from passing as eager.
    graph_nodes = {"predictions": type(predictions.grad_fn).__name__,
                   "pooled": type(pooled.grad_fn).__name__}
    if compiled and any(name != "CompiledFunctionBackward" for name in graph_nodes.values()):
        raise AssertionError("Expected actual AOTAutograd outputs from the compiled model")
    loss = (predictions - batch["targets"]).square().mean()
    loss.backward()
    if raw.grad is None or any(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("All raw inputs and parameter blocks need first gradients")
    gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()}
    informative = tuple(gradients.values()) + (raw.grad,)
    if any(not bool(torch.isfinite(value).all()) or float(value.abs().max()) <= 0
           for value in informative):
        raise AssertionError("Every gradient block must be finite and informative")
    empty = [row for row, (start, stop) in enumerate(zip(batch["boundaries"], batch["boundaries"][1:]))
             if start == stop]
    torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
    unchanged(inputs, before)
    return {"predictions": predictions.detach(), "pooled": pooled.detach(),
            "loss": loss.detach(), "input_gradient": raw.grad.detach().clone()}, gradients, graph_nodes


def compare_routes(results, tolerance):
    return {f"{first}_vs_{second}": eager.compare(results[first], results[second], tolerance)
            for first, second in PAIRS}


def run(dtype_name, device_index, source_revision=None, source_tree=None):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise AssertionError("Use the declared PyTorch 2.9.1 cu128 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA device is unavailable")
    if any((parent / "pyproject.toml").is_file() for parent in SCRIPT.parents):
        raise AssertionError("Copy both consumers outside every source checkout")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    identity = eager.installed_identity()
    helper_hash = sha256(HELPER.read_bytes()).hexdigest()
    script_hash = sha256(SCRIPT.read_bytes()).hexdigest()
    identity.update({"script_sha256": script_hash, "helper_script_sha256": helper_hash,
                     "helper_script_name": HELPER.name})
    initial, batches = fixture(dtype_name, device)
    initial_parameters = dict(initial.named_parameters())
    initial_before = snapshot(initial_parameters)
    initial_hashes = hashes(initial_parameters)
    source_inputs = {f"batch{index}.{name}": batch[name]
                     for index, batch in enumerate(batches, 1) for name in ("raw", "targets", "offsets")}
    source_before = snapshot(source_inputs)
    input_hashes = hashes(source_inputs)
    if (input_hashes["batch1.raw"] == input_hashes["batch2.raw"]
            or input_hashes["batch1.targets"] == input_hashes["batch2.targets"]
            or input_hashes["batch1.offsets"] == input_hashes["batch2.offsets"]):
        raise AssertionError("The second fixed-shape batch must change data, targets and partitions")
    models = {route: copy.deepcopy(initial) for route in ROUTES}
    optimizers = {route: torch.optim.SGD(model.parameters(), lr=0.03, foreach=False, fused=False)
                  for route, model in models.items()}
    ownership = {route: (id(model), id(optimizers[route]), tuple(id(value) for value in model.parameters()),
                         tuple(value.data_ptr() for value in model.parameters()),
                         tuple(id(value) for group in optimizers[route].param_groups for value in group["params"]))
                 for route, model in models.items()}
    all_parameters = tuple(initial.parameters()) + tuple(value for model in models.values() for value in model.parameters())
    if len({value.data_ptr() for value in all_parameters}) != len(all_parameters):
        raise AssertionError("Initial and training models must own separate parameter storage")
    calls = dict(models)
    calls["compiled"] = torch.compile(models["compiled"], backend="aot_eager", fullgraph=True, dynamic=False)
    if tuple(id(value) for value in calls["compiled"].parameters()) != ownership["compiled"][2]:
        raise AssertionError("The compiled wrapper must use the original model parameters")
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    steps = []
    for index, batch in enumerate(batches, 1):
        before = {route: {name: value.detach().clone() for name, value in model.named_parameters()}
                  for route, model in models.items()}
        raw = {route: batch["raw"].detach().clone().requires_grad_() for route in ROUTES}
        if len({value.data_ptr() for value in raw.values()} | {batch["raw"].data_ptr()}) != 4:
            raise AssertionError("Training routes must own separate raw-input storage")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        results = {route: evaluate(calls[route], models[route], raw[route], batch, route == "primitive",
                                  compiled=route == "compiled")
                   for route in ROUTES}
        output_errors = compare_routes({route: result[0] for route, result in results.items()}, tolerance)
        gradient_errors = compare_routes({route: result[1] for route, result in results.items()}, tolerance)
        for optimizer in optimizers.values():
            optimizer.step()
        states = {route: {name: value.detach().clone() for name, value in model.named_parameters()}
                  for route, model in models.items()}
        update_errors = compare_routes(states, tolerance)
        changes = {route: {name: eager.difference(value, before[route][name]) for name, value in state.items()}
                   for route, state in states.items()}
        if any(change <= 0 for route in changes.values() for change in route.values()):
            raise AssertionError("Each real SGD step must change every parameter block in every route")
        parameter_ownership(models, optimizers, ownership)
        if tuple(id(value) for value in calls["compiled"].parameters()) != ownership["compiled"][2]:
            raise AssertionError("The compiled wrapper changed parameter ownership")
        unchanged(source_inputs, source_before)
        unchanged(initial_parameters, initial_before)
        steps.append({"update": index, "N": 9, "B": 5, "raw_width": 3, "pooled_width": 4,
                      "offsets": list(batch["boundaries"]),
                      "output_gradient_nodes": {route: result[2] for route, result in results.items()},
                      "losses": {route: float(result[0]["loss"]) for route, result in results.items()},
                      "output_and_input_gradient_max_abs_errors": output_errors,
                      "parameter_gradient_max_abs_errors": gradient_errors,
                      "updated_parameter_max_abs_errors": update_errors,
                      "parameter_update_max_abs": changes,
                      "updated_parameter_sha256": {route: hashes(state) for route, state in states.items()},
                      "inputs_unchanged": True, "initial_parameters_unchanged": True,
                      "all_parameter_blocks_changed": True, "same_parameter_and_optimizer_objects": True})
    if sha256(HELPER.read_bytes()).hexdigest() != helper_hash or sha256(SCRIPT.read_bytes()).hexdigest() != script_hash:
        raise AssertionError("Consumer source changed during execution")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "aot_eager.packed_cuda_training", "dtype": dtype_name,
            "scope": "Synthetic fixed-shape first-order graph integration, without optimized-code or native-diagnostic claims.",
            "seed": eager.SEED, "device": str(device), "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda, "python_executable": sys.executable,
            "working_directory": str(Path.cwd()), "source_revision": source_revision,
            "source_tree": source_tree, "source_identity_basis": "Caller labels plus hashed consumer/helper/native bytes",
            "compile_options": COMPILE_OPTIONS, "compile_api_calls": 1,
            "graph_count_claimed": False, "optimizer": "SGD", "learning_rate": 0.03,
            "optimizer_steps": {route: 2 for route in ROUTES}, "parameter_tensors": 5,
            "scalar_parameters": 25, "comparison_tolerance": tolerance, "steps": steps,
            "initial_parameter_sha256": initial_hashes, "input_sha256": input_hashes,
            "fixture_sha256": sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest(),
            "inputs_unchanged": True, "initial_parameters_unchanged": True,
            "separate_model_and_parameter_storage": True, "same_parameter_and_optimizer_objects": True,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(eager.DTYPES), default="float32")
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
