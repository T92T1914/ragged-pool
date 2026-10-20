"""Load an owned padded-model export and compare independent CPU predictions.

Use only trusted local artifacts. An expected manifest hash identifies bytes
relative to the producer receipt. It does not authenticate a publisher or make
unknown serialization safe. This consumer imports no producer or model class.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys

import torch
from torch.nn import functional as F

import ragged_pool as package


def _adjacent(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("n8_helpers_" + name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load adjacent example {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


# Reuse the accepted file and installed-identity checks without its packed schema.
helpers = _adjacent("predict_sets")
file_sha256 = helpers.file_sha256
FILE_LIMITS = dict(helpers.FILE_LIMITS)
DTYPES = {"float32": torch.float32, "float64": torch.float64}
STATE_SHAPES = {"encoder.weight": (4, 3), "encoder.bias": (4,),
                "scorer.weight": (1, 4), "head.weight": (1, 4), "head.bias": (1,)}
GRAPH = {"pool_operator": "ragged_pool::padded", "pool_count": 1, "backward_count": 0}
EQUIVALENT_CASES = ("fixed_holes", "moved_holes", "extra_padding", "changed_padding")


def tolerance(dtype_name):
    return {"rtol": 2e-5, "atol": 2e-6} if dtype_name == "float32" else {"rtol": 1e-10, "atol": 1e-12}


def contract(dtype_name):
    if dtype_name not in DTYPES:
        raise ValueError("Padded artifacts support float32 or float64")
    return {"device": "cpu", "dtype": dtype_name, "features": 3,
            "b_min": 2, "b_max": 32, "t_min": 2, "t_max": 128,
            "slots_max": 4096, "contiguous": True, "mask_dtype": "bool",
            "selected_raw_min": -6.0, "selected_raw_max": 6.0,
            "ignored_nonfinite_padding": True, "gradients": "inference_only",
            "outputs": ["predictions[B]", "pooled[B,4]"]}


def environment_identity():
    identity = helpers.environment_identity()
    if not Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Use the wheel installed inside this Python environment")
    return identity


def validate_inputs(raw, mask, dtype_name):
    """Admit original tensors before encoder arithmetic or graph invocation."""
    if dtype_name not in DTYPES:
        raise ValueError("Unsupported padded artifact dtype")
    for value in (raw, mask):
        if not isinstance(value, torch.Tensor):
            raise ValueError("Padded predictor inputs must be tensors")
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Padded predictor inputs must use CPU strided tensors")
        if not value.is_contiguous():
            raise ValueError("Padded predictor inputs must be contiguous")
    if raw.dtype != DTYPES[dtype_name] or raw.dim() != 3 or raw.shape[2] != 3:
        raise ValueError("Raw input must have the artifact dtype and shape [B,T,3]")
    if mask.dtype != torch.bool or mask.dim() != 2 or mask.shape != raw.shape[:2]:
        raise ValueError("Mask must have matching bool shape [B,T]")
    batches, steps, _ = raw.shape
    if not 2 <= batches <= 32 or not 2 <= steps <= 128 or batches * steps > 4096:
        raise ValueError("Padded predictor requires 2 <= B <= 32 and 2 <= T <= 128")
    # Select first. Ignored NaN/Inf padding never reaches these numeric checks.
    selected = raw[mask]
    if not bool((torch.isfinite(selected) & (selected.abs() <= 6)).all()):
        raise ValueError("Selected raw features must be finite and within [-6,6]")


def validate_state_dict(state, dtype_name):
    if dtype_name not in DTYPES or type(state) is not dict or set(state) != set(STATE_SHAPES):
        raise ValueError("Weights must contain exactly the padded model's tensor keys")
    for name, shape in STATE_SHAPES.items():
        value = state[name]
        if type(value) is not torch.Tensor:
            raise ValueError(f"State entry {name} must be an ordinary tensor")
        if (value.device.type != "cpu" or value.layout != torch.strided
                or value.dtype != DTYPES[dtype_name] or tuple(value.shape) != shape
                or not value.is_contiguous() or value.is_conj() or value.is_neg()
                or value.requires_grad):
            raise ValueError(f"State entry {name} has an unsupported representation")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"State entry {name} must be finite")


def read_state(payload, dtype_name):
    state = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    validate_state_dict(state, dtype_name)
    return state


def functional_reference(raw, mask, state):
    """Evaluate selected items per row without a custom op or nn.Module class."""
    dtype_name = next((name for name, dtype in DTYPES.items()
                       if isinstance(raw, torch.Tensor) and raw.dtype == dtype), None)
    validate_inputs(raw, mask, dtype_name)
    validate_state_dict(state, dtype_name)
    rows = []
    for row in range(raw.shape[0]):
        selected = raw[row][mask[row]]
        if selected.shape[0] == 0:
            pooled = torch.zeros(4, dtype=raw.dtype, device="cpu")
        else:
            encoded = torch.tanh(F.linear(selected, state["encoder.weight"], state["encoder.bias"]))
            logits = F.linear(encoded, state["scorer.weight"]).squeeze(-1)
            weights = torch.softmax(logits, dim=0)
            pooled = (weights[:, None] * encoded).sum(dim=0)
        rows.append(pooled)
    pooled = torch.stack(rows)
    predictions = F.linear(pooled, state["head.weight"], state["head.bias"]).squeeze(-1)
    return predictions, pooled


def make_cases(dtype_name):
    """Include shape corners and equivalent selected data with different padding."""
    dtype = DTYPES[dtype_name]

    def finite(batches, steps):
        return torch.linspace(-1.3, 1.1, batches * steps * 3, dtype=dtype, device="cpu").reshape(batches, steps, 3)

    def case(name, values, mask, changed=False):
        values, mask = values.clone().contiguous(), mask.clone().contiguous()
        values[~mask] = 1e30 if changed else float("nan")
        padding = (~mask).nonzero()
        if padding.shape[0]:
            first, last = padding[0].tolist(), padding[-1].tolist()
            values[first[0], first[1], 0] = -float("inf") if changed else float("inf")
            values[last[0], last[1], 2] = float("nan") if changed else -float("inf")
        return {"name": name, "raw": values, "mask": mask}

    dense = finite(5, 6)
    mask = torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1]], dtype=torch.bool, device="cpu")
    moved_mask = torch.roll(mask, 1, dims=1).contiguous()
    moved = torch.zeros_like(dense)
    for row in range(5):
        moved[row][moved_mask[row]] = dense[row][mask[row]]
    extra = torch.zeros(5, 9, 3, dtype=dtype, device="cpu")
    extra_mask = torch.zeros(5, 9, dtype=torch.bool, device="cpu")
    extra[:, :6], extra_mask[:, :6] = dense, mask
    return [case("fixed_holes", dense, mask), case("moved_holes", moved, moved_mask),
            case("extra_padding", extra, extra_mask), case("changed_padding", dense, mask, True),
            case("changed_mask", dense, (torch.arange(30, device="cpu").reshape(5, 6) % 2 == 0)),
            case("changed_raw", dense * -0.5 + 0.25, mask),
            case("all_false", finite(3, 4), torch.zeros(3, 4, dtype=torch.bool, device="cpu")),
            case("min_rectangle_singletons", finite(2, 2), torch.eye(2, dtype=torch.bool, device="cpu")),
            case("max_rectangle_all_true", finite(32, 128), torch.ones(32, 128, dtype=torch.bool, device="cpu")),
            case("min_b_max_t", finite(2, 128), (torch.arange(256, device="cpu").reshape(2, 128) % 3 != 0)),
            case("max_b_min_t", finite(32, 2), (torch.arange(64, device="cpu").reshape(32, 2) % 3 == 0))]


def _snapshot(raw, mask):
    return [(value.detach().view(torch.uint8).clone(), value._version) for value in (raw, mask)]


def _check_unchanged(raw, mask, snapshot):
    versions = {}
    for name, value, (original, version) in zip(("raw", "mask"), (raw, mask), snapshot):
        if not torch.equal(value.detach().view(torch.uint8), original) or value._version != version:
            raise AssertionError(f"Padded inference changed {name} bytes or its version counter")
        versions[name] = {"before": version, "after": value._version}
    return versions


def graph_metadata(exported):
    targets = [node.target for node in exported.graph_module.graph.nodes if node.op == "call_function"]
    custom = [str(target) for target in targets if str(target).startswith("ragged_pool.")]
    if (custom != ["ragged_pool.padded.default"]
            or sum(target == torch.ops.ragged_pool.padded.default for target in targets) != 1
            or any("backward" in str(target).lower() for target in targets)):
        raise ValueError("Export must contain one padded forward and no other custom or backward operation")
    return dict(GRAPH)


def validate_training(training, dtype_name):
    helpers._keys(training, ("status", "seed", "dtype", "optimizer", "learning_rate", "optimizer_steps",
                            "steps", "padding_raw_gradients_zero", "gradients_finite", "inputs_unchanged"), "Training")
    expected = {"status": "passed", "seed": 20261020, "dtype": dtype_name, "optimizer": "SGD",
                "learning_rate": 0.05, "optimizer_steps": 2, "padding_raw_gradients_zero": True,
                "gradients_finite": True, "inputs_unchanged": True}
    if any(training[key] != value or type(training[key]) is not type(value) for key, value in expected.items()):
        raise ValueError("Padded artifact requires the declared two-update training route")
    if type(training["steps"]) is not list or len(training["steps"]) != 2:
        raise ValueError("Padded training must retain two update receipts")
    scalar_keys = ("native_loss", "reference_loss", "loss_max_abs_difference", "pooled_max_abs_difference",
                   "prediction_max_abs_difference", "raw_gradient_max_abs_difference")
    tensor_keys = ("parameter_gradient_max_abs_differences", "native_parameter_update_max_abs",
                   "reference_parameter_update_max_abs", "updated_parameter_max_abs_differences")
    for number, step in enumerate(training["steps"], 1):
        helpers._keys(step, ("step", *scalar_keys, *tensor_keys, "inputs_unchanged"), "Training step")
        if type(step["step"]) is not int or step["step"] != number or step["inputs_unchanged"] is not True:
            raise ValueError("Padded training step identity differs")
        for key in scalar_keys:
            if type(step[key]) is not float or not math.isfinite(step[key]) or step[key] < 0:
                raise ValueError("Padded training values must be finite nonnegative floats")
        for key in tensor_keys:
            helpers._keys(step[key], STATE_SHAPES, "Parameter measurements")
            for value in step[key].values():
                if type(value) is not float or not math.isfinite(value) or value < 0:
                    raise ValueError("Parameter measurements must be finite nonnegative floats")
                if key in ("native_parameter_update_max_abs", "reference_parameter_update_max_abs") and value == 0:
                    raise ValueError("Both trained models must update every parameter")
    helpers._bounded_json(training)
    if len(json.dumps(training, allow_nan=False).encode()) > 32768:
        raise ValueError("Padded training metadata exceeds its byte limit")


def validate_manifest(manifest):
    helpers._keys(manifest, ("schema_version", "source_revision", "dtype", "environment",
                             "contract", "graph", "training", "files", "cases"), "Padded manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported padded artifact schema")
    if type(manifest["source_revision"]) is not str or re.fullmatch(r"[0-9a-f]{40}", manifest["source_revision"]) is None:
        raise ValueError("Source revision must be a complete Git identifier")
    dtype_name = manifest["dtype"]
    if type(dtype_name) is not str or dtype_name not in DTYPES:
        raise ValueError("Unsupported padded artifact dtype")
    expected_contract = contract(dtype_name)
    helpers._keys(manifest["contract"], expected_contract, "Padded contract")
    if (manifest["contract"] != expected_contract
            or any(type(manifest["contract"].get(key)) is not type(value) for key, value in expected_contract.items())):
        raise ValueError("Padded artifact contract differs from this consumer")
    helpers._keys(manifest["graph"], GRAPH, "Padded graph")
    if manifest["graph"] != GRAPH or any(type(manifest["graph"].get(key)) is not type(value) for key, value in GRAPH.items()):
        raise ValueError("Padded graph declaration differs from this consumer")
    environment = manifest["environment"]
    helpers._keys(environment, (*helpers.PINNED, "native_library_sha256", "wheel_sha256"), "Environment")
    if any(environment[key] != value for key, value in helpers.PINNED.items()):
        raise ValueError("Padded artifact environment differs from the pinned versions")
    helpers._sha(environment["native_library_sha256"], "Native library hash")
    helpers._sha(environment["wheel_sha256"], "Wheel hash")
    validate_training(manifest["training"], dtype_name)
    helpers._keys(manifest["files"], ("weights.pt", "model.pt2"), "Files")
    for name, entry in manifest["files"].items():
        helpers._keys(entry, ("sha256", "size_bytes"), "File metadata")
        helpers._sha(entry["sha256"], "File hash")
        if type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= FILE_LIMITS[name]:
            raise ValueError("Padded artifact file exceeds its byte limit")
    specs = make_cases(dtype_name)
    if type(manifest["cases"]) is not list or len(manifest["cases"]) != len(specs):
        raise ValueError("Padded inference cases differ from this route")
    for entry, spec in zip(manifest["cases"], specs):
        helpers._keys(entry, ("name", "expected_predictions", "expected_pooled"), "Case")
        if entry["name"] != spec["name"]:
            raise ValueError("Padded inference case names or order differ")
        batches = spec["raw"].shape[0]
        predictions, pooled = entry["expected_predictions"], entry["expected_pooled"]
        if (type(predictions) is not list or len(predictions) != batches
                or type(pooled) is not list or len(pooled) != batches
                or any(type(row) is not list or len(row) != 4 for row in pooled)):
            raise ValueError("Expected padded outputs have unsupported shapes")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in predictions + [value for row in pooled for value in row]):
            raise ValueError("Expected padded outputs must contain finite JSON numbers")


class Predictor:
    def __init__(self, exported_program, manifest, state):
        self.exported_program, self.manifest, self.state = exported_program, manifest, state
        # The original model was set to eval before export. Saved modules have no eval().
        self.module = exported_program.module()

    def __call__(self, raw, mask):
        dtype_name = self.manifest["dtype"]
        validate_inputs(raw, mask, dtype_name)
        with torch.inference_mode():
            output = self.module(raw, mask)
        if type(output) is not tuple or len(output) != 2:
            raise RuntimeError("Padded export must return predictions and pooled values")
        predictions, pooled = output
        for value, shape in ((predictions, (raw.shape[0],)), (pooled, (raw.shape[0], 4))):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.device.type != "cpu" or value.dtype != DTYPES[dtype_name]
                    or not bool(torch.isfinite(value).all())):
                raise RuntimeError("Padded export violates its output contract")
        if bool(torch.count_nonzero(pooled[~mask.any(dim=1)])):
            raise RuntimeError("Pooled empty rows must remain zero")
        return predictions, pooled


def load_artifact(directory, manifest_sha256):
    """Verify every bounded file and installed identity before deserialization."""
    helpers._sha(manifest_sha256, "Expected manifest hash")
    directory = Path(directory).resolve()
    payload = helpers._read_bounded(directory / "manifest.json", FILE_LIMITS["manifest.json"])
    if hashlib.sha256(payload).hexdigest() != manifest_sha256:
        raise ValueError("Manifest SHA-256 differs from the expected producer receipt")
    manifest = json.loads(payload.decode("utf-8"), object_pairs_hook=helpers._object,
                          parse_constant=helpers._invalid_constant)
    validate_manifest(manifest)
    if {path.name for path in directory.iterdir()} != set(FILE_LIMITS):
        raise ValueError("Padded artifact must contain exactly its three declared files")
    identity = environment_identity()
    if any(manifest["environment"][key] != value for key, value in identity.items()):
        raise ValueError("Padded artifact differs from the installed native environment")
    verified = {}
    for name, entry in manifest["files"].items():
        data = helpers._read_bounded(directory / name, FILE_LIMITS[name])
        if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Padded artifact hash or size differs: {name}")
        verified[name] = data
    # Both files have passed admission before either framework loader is called.
    state = read_state(verified["weights.pt"], manifest["dtype"])
    exported = torch.export.load(io.BytesIO(verified["model.pt2"]))
    graph_metadata(exported)
    if set(exported.state_dict) != set(state):
        raise ValueError("Saved graph parameter keys differ from the verified state")
    for name, expected in state.items():
        actual = exported.state_dict[name]
        if (not isinstance(actual, torch.Tensor) or actual.device.type != "cpu"
                or actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.detach(), expected)):
            raise ValueError(f"Saved graph parameter differs from the verified state: {name}")
    return Predictor(exported, manifest, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    directory, output = arguments.directory.resolve(), arguments.output.resolve()
    if output == directory or output.is_relative_to(directory):
        parser.error("Keep the consumer receipt outside the artifact directory")
    if output.exists() or output.is_symlink():
        parser.error("The consumer receipt must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "started", "pid": os.getpid(), "python_executable": sys.executable,
               "working_directory": os.getcwd(), "script_sha256": file_sha256(__file__),
               "expected_manifest_sha256": arguments.manifest_sha256, "cases": []}
    status = 1
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        training_names = ("export_padded", "train_padded", "export_sets", "learn_sets")
        receipt["training_modules_loaded"] = [name for name in sys.modules
            if any(name == item or name.endswith("_" + item) for item in training_names)]
        if receipt["training_modules_loaded"]:
            raise RuntimeError("Fresh padded prediction unexpectedly imported training code")
        predictor = load_artifact(directory, arguments.manifest_sha256)
        receipt["training_modules_loaded"] = [name for name in sys.modules
            if any(name == item or name.endswith("_" + item) for item in training_names)]
        if receipt["training_modules_loaded"]:
            raise RuntimeError("Loading the padded artifact imported an original training module")
        dtype_name, comparisons = predictor.manifest["dtype"], {}
        receipt.update({"dtype": dtype_name, "source_revision": predictor.manifest["source_revision"],
                        "environment": environment_identity(), "graph": predictor.manifest["graph"],
                        "package_import_path": str(Path(package.__file__).resolve()),
                        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()})
        for case, canary in zip(make_cases(dtype_name), predictor.manifest["cases"]):
            raw, mask = case["raw"], case["mask"]
            snapshot = _snapshot(raw, mask)
            actual = predictor(raw, mask)
            reference = functional_reference(raw, mask, predictor.state)
            expected = (torch.tensor(canary["expected_predictions"], dtype=DTYPES[dtype_name], device="cpu"),
                        torch.tensor(canary["expected_pooled"], dtype=DTYPES[dtype_name], device="cpu"))
            for value, oracle, saved in zip(actual, reference, expected):
                torch.testing.assert_close(value, oracle, **tolerance(dtype_name))
                torch.testing.assert_close(value, saved, **tolerance(dtype_name))
            versions = _check_unchanged(raw, mask, snapshot)
            comparisons[case["name"]] = actual
            receipt["cases"].append({"name": case["name"], "b": raw.shape[0], "t": raw.shape[1],
                "selected_items": int(mask.sum()), "inputs_unchanged": True, "input_versions": versions,
                "reference_prediction_max_abs_difference": float((actual[0] - reference[0]).abs().max()),
                "reference_pooled_max_abs_difference": float((actual[1] - reference[1]).abs().max()),
                "canary_prediction_max_abs_difference": float((actual[0] - expected[0]).abs().max()),
                "canary_pooled_max_abs_difference": float((actual[1] - expected[1]).abs().max())})
        for name in EQUIVALENT_CASES[1:]:
            for actual, expected in zip(comparisons[name], comparisons[EQUIVALENT_CASES[0]]):
                torch.testing.assert_close(actual, expected, **tolerance(dtype_name))
        receipt["metamorphic_cases_agree"] = list(EQUIVALENT_CASES)
        receipt["status"], status = "passed", 0
    except Exception as error:
        receipt.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "cases": len(receipt["cases"]), "receipt": str(output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
