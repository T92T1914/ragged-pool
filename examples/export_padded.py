"""Train two CPU updates, then save and verify a strict padded-model export.

The three-file artifact is intended for the adjacent fresh-process consumer.
Its manifest hash belongs in a trusted producer receipt. These checks identify
owned bytes and model behavior, not publisher authenticity or model quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import torch
from torch import nn

from ragged_pool import ragged_pool_padded, ragged_pool_padded_reference


def _adjacent(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("n8_producer_" + name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load adjacent example {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


train = _adjacent("train_padded")
predict = _adjacent("predict_padded")


def difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def train_model(dtype_name):
    """Train the existing padded model against an independent identical start."""
    dtype, tolerance = predict.DTYPES[dtype_name], predict.tolerance(dtype_name)
    torch.manual_seed(20261020)
    raw = torch.randn(5, 6, 3, dtype=dtype, device="cpu")
    mask = torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1]], dtype=torch.bool, device="cpu")
    raw[~mask] = float("nan")
    raw[0, 1, 0], raw[1, 0, 1] = float("inf"), -float("inf")
    targets = torch.tensor([-0.3, 0.8, -0.5, 0.25, -0.1], dtype=dtype, device="cpu")
    native = train.PaddedRegressor().to(device="cpu", dtype=dtype)
    reference = copy.deepcopy(native)
    values = raw.clone().requires_grad_()
    reference_values = raw.clone().requires_grad_()
    predict.validate_inputs(values, mask, dtype_name)
    snapshots = (predict._snapshot(values, mask), predict._snapshot(reference_values, mask))
    optimizer = torch.optim.SGD(native.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    reference_parameters = dict(reference.named_parameters())
    steps = []
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        values.grad = reference_values.grad = None
        before = {name: parameter.detach().clone() for name, parameter in native.named_parameters()}
        reference_before = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
        predictions, pooled = native(values, mask, ragged_pool_padded)
        reference_predictions, reference_pooled = reference(reference_values, mask, ragged_pool_padded_reference)
        loss = nn.functional.mse_loss(predictions, targets)
        reference_loss = nn.functional.mse_loss(reference_predictions, targets)
        for actual, expected in ((predictions, reference_predictions), (pooled, reference_pooled), (loss, reference_loss)):
            if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
                raise AssertionError("Padded training outputs and losses must remain finite")
            torch.testing.assert_close(actual, expected, **tolerance)
        empty = ~mask.any(dim=1)
        torch.testing.assert_close(pooled[empty], torch.zeros_like(pooled[empty]), rtol=0, atol=0)
        loss.backward()
        reference_loss.backward()
        for value in (values, reference_values):
            if value.grad is None or not bool(torch.isfinite(value.grad).all()):
                raise AssertionError("Padded raw gradients must remain finite")
            if bool(torch.count_nonzero(value.grad[~mask])):
                raise AssertionError("Ignored raw padding must receive exactly zero gradients")
        torch.testing.assert_close(values.grad, reference_values.grad, **tolerance)
        gradient_errors = {}
        for name, parameter in native.named_parameters():
            other = reference_parameters[name]
            if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in (parameter, other)):
                raise AssertionError(f"Missing or nonfinite parameter gradient for {name}")
            torch.testing.assert_close(parameter.grad, other.grad, **tolerance)
            gradient_errors[name] = difference(parameter.grad, other.grad)
        optimizer.step()
        reference_optimizer.step()
        updates, reference_updates, update_errors = {}, {}, {}
        for name, parameter in native.named_parameters():
            other = reference_parameters[name]
            if not bool(torch.isfinite(parameter).all()) or not bool(torch.isfinite(other).all()):
                raise AssertionError("Updated padded parameters must remain finite")
            torch.testing.assert_close(parameter, other, **tolerance)
            updates[name], reference_updates[name] = difference(parameter, before[name]), difference(other, reference_before[name])
            update_errors[name] = difference(parameter, other)
            if updates[name] == 0 or reference_updates[name] == 0:
                raise AssertionError(f"Both SGD models must update {name} at step {step}")
        predict._check_unchanged(values, mask, snapshots[0])
        predict._check_unchanged(reference_values, mask, snapshots[1])
        steps.append({"step": step, "native_loss": float(loss.detach()), "reference_loss": float(reference_loss.detach()),
            "loss_max_abs_difference": difference(loss, reference_loss),
            "pooled_max_abs_difference": difference(pooled, reference_pooled),
            "prediction_max_abs_difference": difference(predictions, reference_predictions),
            "raw_gradient_max_abs_difference": difference(values.grad, reference_values.grad),
            "parameter_gradient_max_abs_differences": gradient_errors,
            "native_parameter_update_max_abs": updates, "reference_parameter_update_max_abs": reference_updates,
            "updated_parameter_max_abs_differences": update_errors, "inputs_unchanged": True})
    return native, {"status": "passed", "seed": 20261020, "dtype": dtype_name,
        "optimizer": "SGD", "learning_rate": 0.05, "optimizer_steps": 2, "steps": steps,
        "padding_raw_gradients_zero": True, "gradients_finite": True, "inputs_unchanged": True}


class InferenceModel(nn.Module):
    """Retain trained layer tensors and fix the pooling function in the graph."""
    def __init__(self, trained):
        super().__init__()
        self.encoder, self.scorer, self.head = trained.encoder, trained.scorer, trained.head

    def forward(self, raw, mask):
        clean = torch.where(mask[..., None], raw, 0)
        encoded = torch.tanh(self.encoder(clean))
        logits = self.scorer(encoded).squeeze(-1).contiguous()
        pooled = ragged_pool_padded(encoded, logits, mask)
        return self.head(pooled).squeeze(-1), pooled


def export_model(model, dtype_name):
    model.eval()
    example = predict.make_cases(dtype_name)[0]
    batches = torch.export.Dim("batches", min=2, max=32)
    steps = torch.export.Dim("steps", min=2, max=128)
    with torch.no_grad():
        exported = torch.export.export(model, (example["raw"], example["mask"]),
            dynamic_shapes=({0: batches, 1: steps}, {0: batches, 1: steps}), strict=True)
    predict.graph_metadata(exported)
    return exported


def _file_metadata(path):
    return {"sha256": predict.file_sha256(path), "size_bytes": path.stat().st_size}


def _verified_bytes(path, metadata):
    payload = predict.helpers._read_bounded(path, predict.FILE_LIMITS[path.name])
    if len(payload) != metadata["size_bytes"] or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError(f"Owned artifact changed before deserialization: {path.name}")
    return payload


def _guard_witnesses(predictor, dtype_name):
    witnesses = []
    for dimension, shape in (("B", (1, 6, 3)), ("T", (5, 1, 3))):
        raw = torch.zeros(shape, dtype=predict.DTYPES[dtype_name], device="cpu")
        mask = torch.ones(shape[:2], dtype=torch.bool, device="cpu")
        witness = {"dimension": dimension, "shape": list(shape)}
        try:
            with torch.inference_mode():
                result = predictor.module(raw, mask)
            witness.update({"direct_module": "accepted", "output_shapes": [list(value.shape) for value in result]})
        except Exception as error:
            witness.update({"direct_module": "rejected", "error_type": type(error).__name__, "error": str(error)})
        try:
            predictor(raw, mask)
        except ValueError as error:
            witness.update({"consumer": "rejected_before_graph", "consumer_error": str(error)})
        else:
            raise AssertionError("The explicit padded consumer must reject size-one dimensions")
        witnesses.append(witness)
    return witnesses


def create_artifact(model, directory, *, dtype_name, source_revision, wheel_sha256, training):
    if type(source_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source_revision must be a complete Git identifier")
    predict.helpers._sha(wheel_sha256, "Wheel hash")
    predict.contract(dtype_name)
    predict.validate_training(training, dtype_name)
    environment = predict.environment_identity()
    environment["wheel_sha256"] = wheel_sha256
    state = {name: value.detach().clone().contiguous() for name, value in model.state_dict().items()}
    predict.validate_state_dict(state, dtype_name)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "weights.pt"
    torch.save(state, weights)
    files = {"weights.pt": _file_metadata(weights)}
    reloaded = predict.read_state(_verified_bytes(weights, files["weights.pt"]), dtype_name)
    parameter_errors = {}
    for name, expected in state.items():
        torch.testing.assert_close(reloaded[name], expected, rtol=0, atol=0)
        parameter_errors[name] = difference(reloaded[name], expected)
    reloaded_model = train.PaddedRegressor().to(device="cpu", dtype=predict.DTYPES[dtype_name])
    reloaded_model.load_state_dict(reloaded, strict=True)
    original, native = InferenceModel(model).eval(), InferenceModel(reloaded_model).eval()
    exported = export_model(native, dtype_name)
    graph = predict.graph_metadata(exported)
    program = directory / "model.pt2"
    torch.export.save(exported, program)
    files["model.pt2"] = _file_metadata(program)
    cases, comparisons, expected_outputs = [], [], {}
    # These inputs need version counters for the nonmutation check. Creating
    # them inside inference_mode would remove those counters.
    with torch.no_grad():
        for case in predict.make_cases(dtype_name):
            raw, mask = case["raw"], case["mask"]
            predict.validate_inputs(raw, mask, dtype_name)
            snapshot = predict._snapshot(raw, mask)
            expected = original(raw, mask)
            variants = {"reloaded_native": native(raw, mask),
                "reference": reloaded_model(raw, mask, ragged_pool_padded_reference),
                "functional_reference": predict.functional_reference(raw, mask, reloaded)}
            errors = {"name": case["name"]}
            for label, output in variants.items():
                for value, target in zip(output, expected):
                    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(target).all()):
                        raise AssertionError("Padded artifact predictions and pools must remain finite")
                    torch.testing.assert_close(value, target, **predict.tolerance(dtype_name))
                errors[label + "_prediction_max_abs"] = difference(output[0], expected[0])
                errors[label + "_pooled_max_abs"] = difference(output[1], expected[1])
            torch.testing.assert_close(expected[1][~mask.any(dim=1)],
                torch.zeros_like(expected[1][~mask.any(dim=1)]), rtol=0, atol=0)
            predict._check_unchanged(raw, mask, snapshot)
            cases.append({"name": case["name"], "expected_predictions": expected[0].tolist(),
                          "expected_pooled": expected[1].tolist()})
            expected_outputs[case["name"]] = expected
            comparisons.append(errors)
    manifest = {"schema_version": 1, "source_revision": source_revision, "dtype": dtype_name,
        "environment": environment, "contract": predict.contract(dtype_name), "graph": graph,
        "training": training, "files": files, "cases": cases}
    predict.validate_manifest(manifest)
    manifest_path = directory / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest_hash = predict.file_sha256(manifest_path)
    predictor = predict.load_artifact(directory, manifest_hash)
    loaded_outputs = {}
    for case, errors in zip(predict.make_cases(dtype_name), comparisons):
        raw, mask = case["raw"], case["mask"]
        snapshot = predict._snapshot(raw, mask)
        actual, expected = predictor(raw, mask), expected_outputs[case["name"]]
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, **predict.tolerance(dtype_name))
        errors["exported_prediction_max_abs"] = difference(actual[0], expected[0])
        errors["exported_pooled_max_abs"] = difference(actual[1], expected[1])
        errors["inputs_unchanged"] = True
        errors["input_versions"] = predict._check_unchanged(raw, mask, snapshot)
        loaded_outputs[case["name"]] = actual
    metamorphic = []
    for name in predict.EQUIVALENT_CASES[1:]:
        actual, expected = loaded_outputs[name], loaded_outputs[predict.EQUIVALENT_CASES[0]]
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, **predict.tolerance(dtype_name))
        metamorphic.append({"case": name, "baseline": predict.EQUIVALENT_CASES[0],
            "prediction_max_abs_difference": difference(actual[0], expected[0]),
            "pooled_max_abs_difference": difference(actual[1], expected[1])})
    return {"manifest_sha256": manifest_hash, "files": files, "graph": graph,
        "environment": environment, "training": training,
        "state_parameter_reload_max_abs": parameter_errors, "case_comparisons": comparisons,
        "metamorphic_checks": metamorphic, "direct_guard_witnesses": _guard_witnesses(predictor, dtype_name)}


def run(arguments):
    directory, output = Path(arguments.directory).resolve(), Path(arguments.output).resolve()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("The padded artifact directory must be new")
    if output == directory or output.is_relative_to(directory):
        raise ValueError("Keep the producer receipt outside the artifact directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    receipt = {"status": "running", "dtype": arguments.dtype, "source_revision": arguments.source_revision,
               "script_sha256": predict.file_sha256(__file__)}
    try:
        environment = predict.environment_identity()
        wheel = Path(arguments.wheel).resolve(strict=True)
        with ZipFile(wheel) as archive:
            native = [name for name in archive.namelist() if name.startswith("ragged_pool/_C") and name.endswith((".pyd", ".so"))]
            if len(native) != 1 or hashlib.sha256(archive.read(native[0])).hexdigest() != environment["native_library_sha256"]:
                raise ValueError("The declared wheel differs from the installed native library")
        wheel_hash = predict.file_sha256(wheel)
        model, training = train_model(arguments.dtype)
        receipt["training"] = training
        receipt["artifact"] = create_artifact(model, directory, dtype_name=arguments.dtype,
            source_revision=arguments.source_revision, wheel_sha256=wheel_hash, training=training)
        receipt["manifest_sha256"] = receipt["artifact"]["manifest_sha256"]
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = run(arguments)
    print(json.dumps({"status": receipt["status"], "manifest_sha256": receipt["artifact"]["manifest_sha256"]}))


if __name__ == "__main__":
    main()
