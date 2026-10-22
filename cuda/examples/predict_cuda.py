"""Run an owned CUDA export without its producer or original model class.

The expected manifest digest must arrive through a trusted route. These checks
identify a local artifact, not a safe format for hostile programs.
"""
import argparse
from hashlib import sha256
from importlib.metadata import distribution
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

import torch
from torch.nn import functional as F

import ragged_pool_cuda as package


DTYPES = {"float32": torch.float32, "float64": torch.float64}
STATE_SHAPES = {"encoder.weight": (4, 3), "encoder.bias": (4,),
                "scorer.weight": (1, 4), "head.weight": (1, 4), "head.bias": (1,)}
CASE_NAMES = ("mixed_empty", "changed_raw", "changed_partition", "singletons",
              "one_nonempty", "within_group_permutation")
MANIFEST_LIMIT, PROGRAM_LIMIT = 64 * 1024, 2 * 1024 * 1024
TRAINING = {"optimizer": "SGD", "learning_rate": 0.03, "optimizer_steps": 2,
            "reference_optimizer_steps": 2, "all_parameter_blocks_changed": True}


def _dtype(dtype):
    if isinstance(dtype, str) and dtype in DTYPES:
        return DTYPES[dtype]
    if dtype in DTYPES.values():
        return dtype
    raise ValueError("dtype must be float32 or float64")


def tolerance(dtype):
    return ({"rtol": 2e-4, "atol": 2e-6} if _dtype(dtype) == torch.float32
            else {"rtol": 2e-10, "atol": 2e-12})


def contract(dtype):
    return {"device": "cuda:0", "dtype": str(_dtype(dtype)).removeprefix("torch."),
            "raw_shape": [9, 3], "offsets_shape": [6], "offsets_dtype": "int64",
            "raw_absolute_limit": 6.0, "contiguous": True, "inference_only": True}


def environment_identity():
    """Identify this ordinary installed package without publishing local paths."""
    installed = distribution("ragged-pool-cuda")
    directory = Path(package.__file__).resolve().parent
    direct = installed.read_text("direct_url.json")
    libraries = list(directory.glob("_C*.pyd")) + list(directory.glob("_C*.so"))
    if (not directory.is_relative_to(Path(sys.prefix).resolve())
            or installed.version != package.__version__ or len(libraries) != 1
            or (direct and json.loads(direct).get("dir_info", {}).get("editable"))):
        raise RuntimeError("An ordinary CUDA package installation is required")
    if str(libraries[0].resolve()) not in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}:
        raise RuntimeError("The installed CUDA native library is not loaded")
    result = {"python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
              "cuda_runtime_version": torch.version.cuda, "package_version": installed.version,
              "native_library_sha256": sha256(libraries[0].read_bytes()).hexdigest(),
              "package_python_sha256": {name: sha256((directory / name).read_bytes()).hexdigest()
                                        for name in ("__init__.py", "_registration.py", "_validation.py", "reference.py", "padded.py")}}
    if tuple(result[k] for k in ("python_version", "torch_version", "cuda_runtime_version", "package_version")) != (
            "3.11.8", "2.9.1+cu128", "12.8", "0.1.0"):
        raise RuntimeError("This consumer requires the declared pinned CUDA environment")
    if "ragged_pool" in sys.modules:
        raise RuntimeError("The CUDA consumer must not import the CPU package")
    return result


def validate_inputs(raw, offsets, dtype=None):
    for tensor, shape, expected_dtype in ((raw, (9, 3), None), (offsets, (6,), torch.int64)):
        if (type(tensor) is not torch.Tensor or tensor.device != torch.device("cuda:0")
                or tensor.layout != torch.strided or tensor.shape != shape
                or not tensor.is_contiguous() or tensor.is_conj() or tensor.is_neg()
                or (expected_dtype is not None and tensor.dtype != expected_dtype)):
            raise ValueError("Inputs must retain the declared shape, CUDA device and representation")
    _dtype(raw.dtype)
    if dtype is not None and raw.dtype != _dtype(dtype):
        raise ValueError("Input dtype must match this artifact")
    if not bool(torch.isfinite(raw).all()) or bool((raw.abs() > 6).any()):
        raise ValueError("Raw features must be finite in [-6,6]")
    # Offset content stays under native admission, including for direct graph calls.


def _numbers(value, shape, label):
    if shape:
        if type(value) is not list or len(value) != shape[0]:
            raise ValueError(f"{label} has an unexpected shape")
        for item in value:
            _numbers(item, shape[1:], label)
    elif type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must contain finite numbers")


def validate_state(state, dtype):
    _dtype(dtype)
    if type(state) is not dict or set(state) != set(STATE_SHAPES):
        raise ValueError("Weights must contain exactly the five model parameters")
    for name, shape in STATE_SHAPES.items():
        _numbers(state[name], shape, name)


def state_tensors(state, dtype):
    validate_state(state, dtype)
    tensors = {name: torch.tensor(value, dtype=_dtype(dtype), device="cuda:0")
               for name, value in state.items()}
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("Weights must remain finite in their declared dtype")
    return tensors


def reference(raw, boundaries, state):
    """Use independent GPU primitives and producer-independent Python boundaries."""
    values = F.linear(raw, state["encoder.weight"], state["encoder.bias"]).tanh()
    logits = F.linear(values, state["scorer.weight"]).squeeze(-1)
    rows = []
    for first, last in zip(boundaries, boundaries[1:]):
        if first == last:
            rows.append(values.new_zeros(4))
        else:
            weights = torch.softmax(logits[first:last].double(), dim=0)
            rows.append((weights[:, None] * values[first:last].double()).sum(0).to(raw.dtype))
    pooled = torch.stack(rows)
    return F.linear(pooled, state["head.weight"], state["head.bias"]).squeeze(-1), pooled


def make_cases(dtype):
    raw = torch.linspace(-1.5, 1.5, 27, dtype=_dtype(dtype), device="cuda:0").reshape(9, 3)
    base = (0, 2, 2, 5, 7, 9)
    cases = ((raw, base), (raw * -0.4 + 0.25, base),
             (raw, (0, 1, 4, 4, 8, 9)), (raw, (0, 1, 2, 3, 4, 9)),
             (raw, (0, 0, 0, 9, 9, 9)), (raw[[1, 0, 4, 2, 3, 6, 5, 8, 7]], base))
    return [{"name": name, "raw": features.clone().contiguous(),
             "offsets": torch.tensor(bounds, dtype=torch.int64, device="cuda:0"),
             "boundaries": bounds} for name, (features, bounds) in zip(CASE_NAMES, cases)]


def graph_metadata(exported):
    operations = [node.target for node in exported.graph_module.graph.nodes if node.op == "call_function"]
    custom = [str(op) for op in operations if hasattr(op, "_schema")
              and op._schema.name.split("::")[0] not in ("aten", "prims")]
    backward = sum("backward" in str(op).lower() for op in operations)
    if custom != ["ragged_pool_cuda.pool.default"] or backward:
        raise ValueError("Export must retain one native CUDA pooling forward and no backward or other custom operation")
    return {"pool_operator": "ragged_pool_cuda::pool", "pool_count": 1, "backward_count": 0}


def snapshot(tensors):
    return {name: (value.detach().view(torch.uint8).clone(), value._version)
            for name, value in tensors.items()}


def unchanged(tensors, before):
    for name, value in tensors.items():
        data, version = before[name]
        if not torch.equal(value.detach().view(torch.uint8), data) or value._version != version:
            raise AssertionError(f"Inference changed {name} bytes or its version counter")


def _exact(actual, expected, label):
    if type(actual) is not type(expected):
        raise ValueError(f"{label} has an unexpected type")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label} has unexpected keys")
        for key in expected:
            _exact(actual[key], expected[key], label)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} has an unexpected length")
        for a, e in zip(actual, expected):
            _exact(a, e, label)
    elif actual != expected:
        raise ValueError(f"{label} differs from the declared contract")


def _digest(value, length, label):
    if type(value) is not str or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a complete lowercase digest")


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON keys are not admitted")
        result[name] = value
    return result


def _nonfinite_json(value):
    raise ValueError(f"Nonfinite JSON constant {value} is not admitted")


def _read(path, limit):
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or path.is_symlink()
            or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or metadata.st_size > limit):
        raise ValueError("Artifact entries must be bounded ordinary files")
    with path.open("rb") as stream:
        contents = stream.read(limit + 1)
    if len(contents) > limit:
        raise ValueError("Artifact entry exceeded its read limit")
    return contents


def _manifest(manifest):
    keys = {"schema_version", "dtype", "source_revision", "source_tree", "producer_pid",
            "producer_script_sha256", "environment", "contract", "graph", "training", "weights", "files", "expected"}
    if type(manifest) is not dict or set(manifest) != keys:
        raise ValueError("The manifest has unexpected keys")
    _exact(manifest["schema_version"], 1, "schema_version")
    _dtype(manifest["dtype"])
    if type(manifest["dtype"]) is not str:
        raise ValueError("The manifest dtype must be a name")
    for name in ("source_revision", "source_tree"):
        _digest(manifest[name], 40, name)
    _digest(manifest["producer_script_sha256"], 64, "producer_script_sha256")
    if type(manifest["producer_pid"]) is not int or manifest["producer_pid"] <= 0:
        raise ValueError("The producer PID must be a positive integer")
    _exact(manifest["environment"], environment_identity(), "environment")
    _exact(manifest["contract"], contract(manifest["dtype"]), "contract")
    _exact(manifest["graph"], {"pool_operator": "ragged_pool_cuda::pool", "pool_count": 1,
                              "backward_count": 0}, "graph")
    training = manifest["training"]
    if type(training) is not dict or set(training) != set(TRAINING) | {"steps"}:
        raise ValueError("Training metadata has unexpected fields")
    _exact({key: training[key] for key in TRAINING}, TRAINING, "training")
    steps = training["steps"]
    fields = {"step", "native_loss", "reference_loss", "output_max_abs_difference",
              "parameter_gradient_max_abs_difference", "native_parameter_update_max_abs",
              "reference_parameter_update_max_abs", "updated_parameter_max_abs_difference"}
    if type(steps) is not list or len(steps) != 2:
        raise ValueError("Training metadata requires exactly two steps")
    for number, step in enumerate(steps, 1):
        if type(step) is not dict or set(step) != fields:
            raise ValueError("Training step metadata has unexpected fields")
        _exact(step["step"], number, "training step")
        for name in ("native_loss", "reference_loss"):
            _numbers(step[name], (), name)
        for name in fields - {"step", "native_loss", "reference_loss"}:
            values = step[name]
            if type(values) is not dict or not values:
                raise ValueError("Training comparisons must be nonempty mappings")
            if name != "output_max_abs_difference" and set(values) != set(STATE_SHAPES):
                raise ValueError("Training comparisons must cover every model parameter")
            for key, value in values.items():
                _numbers(value, (), name)
                if type(key) is not str or value < 0 or ("update_max_abs" in name and value == 0):
                    raise ValueError("Training errors must be nonnegative and all parameter updates positive")
    validate_state(manifest["weights"], manifest["dtype"])
    files = manifest["files"]
    if type(files) is not dict or set(files) != {"model.pt2"}:
        raise ValueError("Only model.pt2 belongs in files")
    entry = files["model.pt2"]
    if type(entry) is not dict or set(entry) != {"sha256", "size_bytes"}:
        raise ValueError("Program metadata has unexpected keys")
    _digest(entry["sha256"], 64, "program sha256")
    if type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= PROGRAM_LIMIT:
        raise ValueError("Program size is outside the declared limit")
    expected = manifest["expected"]
    if type(expected) is not list or len(expected) != len(CASE_NAMES):
        raise ValueError("Expected outputs must cover the six declared cases")
    for name, case in zip(CASE_NAMES, expected):
        if type(case) is not dict or set(case) != {"name", "predictions", "pooled"} or case["name"] != name:
            raise ValueError("Expected case names or fields differ")
        _numbers(case["predictions"], (5,), name)
        _numbers(case["pooled"], (5, 4), name)


class Predictor:
    def __init__(self, manifest, exported):
        self.manifest, self.exported_program = manifest, exported
        self.module = exported.module()
        self.state = state_tensors(manifest["weights"], manifest["dtype"])
        if set(exported.state_dict) != set(STATE_SHAPES):
            raise ValueError("The graph state differs from the declared parameters")
        for name, expected in self.state.items():
            value = exported.state_dict[name]
            if (value.shape != expected.shape or value.dtype != expected.dtype
                    or value.device != torch.device("cuda:0") or not bool(torch.isfinite(value).all())):
                raise ValueError("The graph state has invalid metadata or values")
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def __call__(self, raw, offsets):
        validate_inputs(raw, offsets, self.manifest["dtype"])
        with torch.inference_mode():
            result = self.module(raw, offsets)
            if type(result) not in (tuple, list) or len(result) != 2:
                raise ValueError("The exported model must return predictions and pooled values")
            for output, shape in zip(result, ((5,), (5, 4))):
                if (type(output) is not torch.Tensor or output.shape != shape or output.dtype != raw.dtype
                        or output.device != raw.device or not bool(torch.isfinite(output).all())):
                    raise ValueError("The exported model returned invalid outputs")
            if bool((result[1][offsets[1:] == offsets[:-1]] != 0).any()):
                raise AssertionError("Empty groups must retain zero pooled vectors")
            return result


def load_artifact(directory, expected_manifest_sha256):
    _digest(expected_manifest_sha256, 64, "expected manifest sha256")
    directory = Path(directory).absolute()
    metadata = directory.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink()
            or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or {p.name for p in directory.iterdir()} != {"manifest.json", "model.pt2"}):
        raise ValueError("The artifact must be an ordinary directory containing exactly two files")
    data = _read(directory / "manifest.json", MANIFEST_LIMIT)
    if sha256(data).hexdigest() != expected_manifest_sha256:
        raise ValueError("Manifest digest differs from the trusted expected value")
    manifest = json.loads(data.decode("utf-8"), object_pairs_hook=_object, parse_constant=_nonfinite_json)
    _manifest(manifest)
    program = _read(directory / "model.pt2", PROGRAM_LIMIT)
    if (len(program) != manifest["files"]["model.pt2"]["size_bytes"]
            or sha256(program).hexdigest() != manifest["files"]["model.pt2"]["sha256"]):
        raise ValueError("Program bytes differ from the verified manifest")
    exported = torch.export.load(io.BytesIO(program))
    _exact(graph_metadata(exported), manifest["graph"], "loaded graph")
    return Predictor(manifest, exported)


def exercise(predictor):
    """Compare independent math, saved canaries and unchanged inference state."""
    errors, previous = [], None
    state_groups = {"loaded": predictor.module.state_dict(), "reference": predictor.state}
    state = {group: snapshot(values) for group, values in state_groups.items()}
    cases = make_cases(predictor.manifest["dtype"])
    with torch.inference_mode():
        for case, canary in zip(cases, predictor.manifest["expected"]):
            raw, offsets = case["raw"], case["offsets"]
            inputs = {"raw": raw, "offsets": offsets}
            before = snapshot(inputs)
            result = predictor(raw, offsets)
            oracle = reference(raw, case["boundaries"], predictor.state)
            expected = tuple(torch.tensor(canary[name], dtype=raw.dtype, device=raw.device)
                             for name in ("predictions", "pooled"))
            errors.append({"name": case["name"]})
            for name, actual, primitive, saved in zip(("predictions", "pooled"), result, oracle, expected):
                torch.testing.assert_close(actual, primitive, **tolerance(raw.dtype))
                torch.testing.assert_close(actual, saved, **tolerance(raw.dtype))
                errors[-1][name + "_reference_max_abs"] = float((actual - primitive).abs().max())
                errors[-1][name + "_canary_max_abs"] = float((actual - saved).abs().max())
                occupied = [raw, offsets, *predictor.module.state_dict().values(), *predictor.state.values()]
                if any(actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in occupied):
                    raise AssertionError("Inference output aliases inputs or model state")
            if previous is not None and any(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
                                            for a in result for b in previous):
                raise AssertionError("Successive inference outputs must own fresh storage")
            previous = result
            unchanged(inputs, before)
        for group, parameters in state.items():
            unchanged(state_groups[group], parameters)
        valid = make_cases(predictor.manifest["dtype"])[0]
        for bounds in ((0, 2, 1, 5, 7, 9), (0, 2, 2, 5, 7, 8)):
            try:
                predictor(valid["raw"], torch.tensor(bounds, dtype=torch.int64, device="cuda:0"))
            except RuntimeError:
                pass
            else:
                raise AssertionError("Malformed offsets were admitted")
            recovery = predictor(valid["raw"], valid["offsets"])
            for actual, expected in zip(recovery, reference(valid["raw"], valid["boundaries"], predictor.state)):
                torch.testing.assert_close(actual, expected, **tolerance(valid["raw"].dtype))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    if output.is_relative_to(Path(args.directory).absolute()):
        raise ValueError("The receipt must be outside the two-file artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    receipt = {"status": "running", "consumer_pid": os.getpid(),
               "consumer_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        predictor = load_artifact(args.directory, args.manifest_sha256)
        producer_imports = sorted(name for name, module in sys.modules.items()
                                  if getattr(module, "__file__", None)
                                  and Path(module.__file__).name in ("train_cuda.py", "train_compiled.py", "export_cuda.py"))
        if producer_imports or predictor.manifest["producer_pid"] == os.getpid():
            raise AssertionError("Inference requires a fresh process without producer or training imports")
        receipt.update({"environment": environment_identity(), "producer_imports": producer_imports,
                        "producer_pid": predictor.manifest["producer_pid"],
                        "source_revision": predictor.manifest["source_revision"],
                        "source_tree": predictor.manifest["source_tree"],
                        "producer_script_sha256": predictor.manifest["producer_script_sha256"],
                        "manifest_sha256": args.manifest_sha256, "comparisons": exercise(predictor),
                        "graph": graph_metadata(predictor.exported_program), "status": "passed"})
    except BaseException as error:
        receipt.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "consumer_pid": os.getpid()}, indent=2))


if __name__ == "__main__":
    main()
