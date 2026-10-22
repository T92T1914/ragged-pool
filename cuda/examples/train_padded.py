"""Compare two padded CUDA training steps through three persistent routes.

Copy this file and train_cuda.py outside the checkout. The compiled native,
eager native and independent primitive models share an initial state, then own
separate SGD optimizers. Ignored raw NaN/Inf padding is removed before the
encoder. This synthetic fixture establishes integration, not model quality,
optimized code, performance, portability or a new native diagnostic result.
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
_spec = importlib.util.spec_from_file_location("_ragged_padded_cuda_helper", HELPER)
if _spec is None or _spec.loader is None:
    raise ImportError("Copy train_cuda.py beside this consumer")
eager = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eager)

from ragged_pool_cuda import ragged_pool_padded  # noqa: E402


ROUTES = ("compiled", "eager", "primitive")
PAIRS = (("compiled", "eager"), ("compiled", "primitive"), ("eager", "primitive"))
COMPILE_OPTIONS = {"backend": "aot_eager", "fullgraph": True, "dynamic": False}
MASKS = (((1, 0, 1, 1, 0, 0), (0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0),
          (1, 1, 0, 1, 0, 1), (0, 1, 1, 0, 0, 1)),
         ((0, 1, 0, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0, 0, 0, 0),
          (0, 0, 0, 1, 0, 0), (1, 0, 1, 1, 0, 1)))


def primitive_pool(values, logits, mask):
    """Differentiate per-row arithmetic without native calls or packed offsets."""
    rows = []
    for row in range(values.shape[0]):
        features, scores = values[row][mask[row]].double(), logits[row][mask[row]].double()
        if scores.numel() == 0:
            pooled = features.sum(0) * 0 + scores.sum() * 0
        else:
            pooled = (torch.softmax(scores, dim=0)[:, None] * features).sum(0)
        rows.append(pooled.to(values.dtype))
    return torch.stack(rows)


class PaddedRegressor(eager.SetRegressor):
    def forward(self, raw, mask, primitive=False):
        # Masking only after linear arithmetic can leave 0*NaN weight gradients.
        safe_raw = torch.where(mask.unsqueeze(-1), raw, torch.zeros_like(raw))
        values = self.encoder(safe_raw).tanh().contiguous()
        logits = self.scorer(values).squeeze(-1).contiguous()
        pooled = (primitive_pool(values, logits, mask) if primitive
                  else ragged_pool_padded(values, logits, mask))
        return self.head(pooled).squeeze(-1), pooled


def _metadata(value):
    return (value._version, id(value), value.data_ptr(), value.storage_offset(),
            tuple(value.stride()), tuple(value.shape), value.dtype, value.device)


def snapshot(tensors):
    return {name: (value.detach().view(torch.uint8).clone(), _metadata(value))
            for name, value in tensors.items()}


def unchanged(tensors, before):
    if tensors.keys() != before.keys():
        raise AssertionError("Source tensor keys changed")
    for name, value in tensors.items():
        data, metadata = before[name]
        if _metadata(value) != metadata or not torch.equal(value.detach().view(torch.uint8), data):
            raise AssertionError(f"Training changed {name} bytes, versions or ownership")


def hashes(tensors):
    """Use an explicit consumer copy for reproducible fixture and state hashes."""
    result = {}
    for name, value in tensors.items():
        copied = value.detach().cpu().contiguous()
        digest = sha256(json.dumps([list(copied.shape), str(copied.dtype)]).encode())
        digest.update(copied.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def fixture(dtype_name, device):
    dtype = eager.DTYPES[dtype_name]
    torch.manual_seed(eager.SEED)
    initial = PaddedRegressor().to(device=device, dtype=dtype)
    generator = torch.Generator(device="cpu").manual_seed(eager.SEED + 200)
    batches = []
    for specification in MASKS:
        mask = torch.tensor(specification, dtype=torch.bool)
        raw = 0.25 * torch.randn(5, 6, 3, dtype=dtype, generator=generator)
        raw[~mask] = torch.tensor([float("nan"), float("inf"), -float("inf")], dtype=dtype)
        targets = 0.4 * torch.randn(5, dtype=dtype, generator=generator) + torch.linspace(-0.3, 0.5, 5, dtype=dtype)
        batches.append({"raw": raw.to(device), "mask": mask.to(device), "targets": targets.to(device)})
    return initial, batches


def evaluate(call, model, raw, batch, primitive=False, compiled=False):
    inputs = {"raw": raw, "mask": batch["mask"], "targets": batch["targets"]}
    before = snapshot(inputs)
    predictions, pooled = call(raw, batch["mask"], primitive)
    nodes = {"predictions": type(predictions.grad_fn).__name__, "pooled": type(pooled.grad_fn).__name__}
    if compiled and any(name != "CompiledFunctionBackward" for name in nodes.values()):
        raise AssertionError("The compiled model must execute AOTAutograd forward and backward")
    loss = (predictions - batch["targets"]).square().mean()
    loss.backward()
    if raw.grad is None or any(value.grad is None for value in model.parameters()):
        raise AssertionError("Raw features and every parameter block need first gradients")
    gradients = {name: value.grad.detach().clone() for name, value in model.named_parameters()}
    for value in (*gradients.values(), raw.grad):
        if not bool(torch.isfinite(value).all()) or float(value.abs().max()) <= 0:
            raise AssertionError("Every gradient block must be finite and informative")
    if bool((raw.grad[~batch["mask"]] != 0).any()):
        raise AssertionError("Ignored raw padding must have exactly zero gradients")
    empty = ~batch["mask"].any(dim=1)
    torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
    unchanged(inputs, before)
    return {"predictions": predictions.detach(), "pooled": pooled.detach(), "loss": loss.detach(),
            "input_gradient": raw.grad.detach().clone()}, gradients, nodes


def _ownership(model, optimizer):
    parameters = tuple(model.parameters())
    return (id(model), id(optimizer), tuple(id(value) for value in parameters),
            tuple(value.data_ptr() for value in parameters),
            tuple(id(value) for group in optimizer.param_groups for value in group["params"]))


def _comparisons(results, tolerance):
    return {f"{first}_vs_{second}": eager.compare(results[first], results[second], tolerance)
            for first, second in PAIRS}


def run(dtype_name, device_index=0, source_revision=None, source_tree=None):
    if dtype_name not in eager.DTYPES or type(device_index) is not int or device_index < 0:
        raise ValueError("Choose a declared dtype and nonnegative CUDA device ordinal")
    for label in (source_revision, source_tree):
        if label is not None and (type(label) is not str or not re.fullmatch(r"[0-9a-f]{40}", label)):
            raise ValueError("Optional source labels must be complete lowercase SHA1 identifiers")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if sys.version.split()[0] != "3.11.8" or str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("Use the declared Python3.11.8/PyTorch2.9.1+cu128 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared CUDA device is unavailable")
    if any((parent / "pyproject.toml").is_file() for parent in SCRIPT.parents):
        raise AssertionError("Copy both consumers outside every source checkout")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    identity = eager.installed_identity()
    helper_hash, script_hash = sha256(HELPER.read_bytes()).hexdigest(), sha256(SCRIPT.read_bytes()).hexdigest()
    if identity.pop("script_sha256") != helper_hash:
        raise AssertionError("The installed-identity helper must identify the adjacent training script")
    package_directory = Path(identity["package_path"]).parent
    package_hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(package_directory.glob("*.py"))}
    initial, batches = fixture(dtype_name, device)
    initial_parameters = dict(initial.named_parameters())
    if len(initial_parameters) != 5 or sum(value.numel() for value in initial_parameters.values()) != 25:
        raise AssertionError("The declared model must retain five parameter tensors and 25 scalar parameters")
    initial_before, initial_hashes = snapshot(initial_parameters), hashes(initial_parameters)
    source_inputs = {f"batch{index}.{name}": batch[name] for index, batch in enumerate(batches, 1)
                     for name in ("raw", "targets", "mask")}
    source_before, input_hashes = snapshot(source_inputs), hashes(source_inputs)
    if any(input_hashes[f"batch1.{name}"] == input_hashes[f"batch2.{name}"] for name in ("raw", "targets", "mask")):
        raise AssertionError("The second batch must change raw contents, targets and mask")
    models = {route: copy.deepcopy(initial) for route in ROUTES}
    optimizers = {route: torch.optim.SGD(model.parameters(), lr=0.03, foreach=False, fused=False)
                  for route, model in models.items()}
    ownership = {route: _ownership(model, optimizers[route]) for route, model in models.items()}
    parameters = tuple(initial.parameters()) + tuple(value for model in models.values() for value in model.parameters())
    if len({value.data_ptr() for value in parameters}) != len(parameters):
        raise AssertionError("The initial model and all training routes must own separate parameter storage")
    calls = dict(models)
    calls["compiled"] = torch.compile(models["compiled"], **COMPILE_OPTIONS)
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 2e-10, "atol": 2e-12})
    steps = []
    for index, batch in enumerate(batches, 1):
        before = {route: {name: value.detach().clone() for name, value in model.named_parameters()}
                  for route, model in models.items()}
        raw = {route: batch["raw"].detach().clone().requires_grad_() for route in ROUTES}
        if len({value.data_ptr() for value in raw.values()} | {batch["raw"].data_ptr()}) != 4:
            raise AssertionError("Every route must own its raw-input storage")
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        results = {route: evaluate(calls[route], models[route], raw[route], batch, route == "primitive", route == "compiled")
                   for route in ROUTES}
        output_errors = _comparisons({route: result[0] for route, result in results.items()}, tolerance)
        gradient_errors = _comparisons({route: result[1] for route, result in results.items()}, tolerance)
        for optimizer in optimizers.values():
            optimizer.step()
        states = {route: {name: value.detach().clone() for name, value in model.named_parameters()}
                  for route, model in models.items()}
        update_errors = _comparisons(states, tolerance)
        changes = {route: {name: eager.difference(value, before[route][name]) for name, value in state.items()}
                   for route, state in states.items()}
        if any(change <= 0 for values in changes.values() for change in values.values()):
            raise AssertionError("Every real SGD step must change all five parameter blocks in every route")
        if any(_ownership(model, optimizers[route]) != ownership[route] for route, model in models.items()):
            raise AssertionError("Training changed model, parameter or optimizer ownership")
        if tuple(id(value) for value in calls["compiled"].parameters()) != ownership["compiled"][2]:
            raise AssertionError("The compiled wrapper must retain the original model parameters")
        unchanged(source_inputs, source_before)
        unchanged(initial_parameters, initial_before)
        steps.append({"update": index, "B": 5, "T": 6, "raw_width": 3, "pooled_width": 4,
            "selected_per_row": batch["mask"].sum(1).cpu().tolist(), "selected_items": int(batch["mask"].sum()),
            "losses": {route: float(result[0]["loss"]) for route, result in results.items()},
            "output_gradient_nodes": {route: result[2] for route, result in results.items()},
            "output_and_input_gradient_max_abs_errors": output_errors, "parameter_gradient_max_abs_errors": gradient_errors,
            "updated_parameter_max_abs_errors": update_errors, "parameter_update_max_abs": changes,
            "updated_parameter_sha256": {route: hashes(state) for route, state in states.items()},
            "ignored_raw_gradients_zero": True, "empty_pooled_rows_zero": True,
            "inputs_unchanged": True, "initial_parameters_unchanged": True, "all_parameter_blocks_changed": True,
            "same_parameter_and_optimizer_objects": True})
    if sha256(HELPER.read_bytes()).hexdigest() != helper_hash or sha256(SCRIPT.read_bytes()).hexdigest() != script_hash:
        raise AssertionError("The consumer or adjacent helper changed during execution")
    if {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(package_directory.glob("*.py"))} != package_hashes:
        raise AssertionError("Installed Python package source changed during execution")
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {"status": "passed", "method": "aot_eager.padded_cuda_training", "dtype": dtype_name,
        "scope": "Synthetic fixed-shape first-order training with poisoned padding, without optimized-code or new native-diagnostic claims.",
        "seed": eager.SEED, "device": str(device), "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor], "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__), "cuda_runtime_version": torch.version.cuda,
        "source_revision": source_revision, "source_tree": source_tree,
        "source_identity_basis": "Caller labels plus hashed consumer/helper/package/native bytes",
        "script_sha256": script_hash, "helper_script_sha256": helper_hash, "helper_script_name": HELPER.name,
        "package_python_sha256": package_hashes, "compile_options": COMPILE_OPTIONS, "compile_api_calls": 1,
        "graph_count_claimed": False, "optimizer": "SGD", "learning_rate": 0.03,
        "optimizer_steps": {route: 2 for route in ROUTES}, "parameter_tensors": 5, "scalar_parameters": 25,
        "comparison_tolerance": tolerance, "steps": steps, "initial_parameter_sha256": initial_hashes,
        "input_sha256": input_hashes, "fixture_sha256": sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest(),
        "inputs_unchanged": True, "initial_parameters_unchanged": True, "separate_model_and_parameter_storage": True,
        "same_parameter_and_optimizer_objects": True, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), **identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(eager.DTYPES), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON receipt outside the source checkout")
    parser.add_argument("--source-revision")
    parser.add_argument("--source-tree")
    args = parser.parse_args()
    for label in (args.source_revision, args.source_tree):
        if label is not None and not re.fullmatch(r"[0-9a-f]{40}", label):
            parser.error("Source labels must be complete lowercase SHA1 identifiers")
    if any((parent / "pyproject.toml").is_file() for parent in args.output.absolute().parents):
        parser.error("Keep the output receipt outside every source checkout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    receipt = {"status": "running", "dtype": args.dtype, "source_revision": args.source_revision,
               "source_tree": args.source_tree, "script_sha256": sha256(SCRIPT.read_bytes()).hexdigest()}
    try:
        receipt = run(args.dtype, source_revision=args.source_revision, source_tree=args.source_tree)
    except BaseException as error:
        receipt.update({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
