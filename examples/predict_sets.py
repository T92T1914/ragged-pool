"""Load an owned, trusted export through an ordinary ragged-pool installation.

The expected manifest digest detects changes relative to the producer's receipt.
It does not authenticate an author or make torch.export.load safe for hostile
files. Only use artifacts produced locally from code and dependencies you trust.
This consumer imports no training script or original model class.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

import torch

# Importing the installed package registers the native operation before loading.
import ragged_pool as package


CONTRACT = {
    "device": "cpu", "dtype": "float32", "features": 4,
    "n_min": 0, "n_max": 4096, "b_min": 2, "b_max": 128,
    "values_min": -6.0, "values_max": 6.0, "contiguous": True,
    "offsets_dtype": "int64", "offsets_start": 0, "offsets_end": "N",
    "offsets_nondecreasing": True, "output": "logits[B]",
    "gradients": "inference_only",
}
GRAPH = {"pool_operator": "ragged_pool::pool", "pool_count": 1, "backward_count": 0}
PINNED = {"python_version": "3.11.8", "torch_version": "2.9.1+cpu",
          "package_version": "0.1.0"}
FILE_LIMITS = {"manifest.json": 256 * 1024, "model.pt2": 16 * 1024**2,
               "weights.pt": 1024**2}
TOLERANCE = {"rtol": 2e-5, "atol": 2e-6}


def file_sha256(path):
    """Hash a file without loading its contents into a framework deserializer."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_identity():
    """Verify ordinary installed imports and return path-free pinned identity."""
    distribution = importlib.metadata.distribution("ragged-pool")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise RuntimeError("Use an ordinary installed wheel, not an editable installation")
    imported = Path(package.__file__).resolve()
    installed = Path(distribution.locate_file("ragged_pool/__init__.py")).resolve()
    if imported != installed:
        raise RuntimeError("Imported ragged_pool differs from its installed distribution")
    if distribution.version != package.__version__:
        raise RuntimeError("Package and distribution versions differ")
    libraries = list(imported.parent.glob("_C*.pyd")) + list(imported.parent.glob("_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError("Expected exactly one installed native library")
    recorded = {Path(distribution.locate_file(item)).resolve()
                for item in (distribution.files or ())}
    if libraries[0].resolve() not in recorded:
        raise RuntimeError("Native library is outside the installed distribution file list")
    identity = {"python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "package_version": package.__version__,
                "native_library_sha256": file_sha256(libraries[0])}
    if any(identity[key] != value for key, value in PINNED.items()):
        raise RuntimeError("This artifact route requires the declared pinned environment")
    return identity


def make_cases():
    """Return deterministic inference inputs, independent of training data."""
    generator = torch.Generator(device="cpu").manual_seed(93001)

    def case(name, lengths, values=None):
        count = sum(lengths)
        if values is None:
            values = torch.randn(count, 4, generator=generator).clamp(-6, 6).contiguous()
        offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
        return {"name": name, "values": values, "offsets": offsets}

    fixed = torch.linspace(-2, 2, 28, dtype=torch.float32).reshape(7, 4).contiguous()
    return [case("fixed", [2, 0, 1, 4], fixed),
            case("changed_offsets", [1, 2, 0, 4], fixed.clone()),
            case("empty_values", [0, 0]), case("singleton", [0, 1]),
            case("two_items", [1, 1]), case("mixed_empty", [0, 2, 0, 1, 3, 0]),
            case("max_groups", [1] * 128), case("max_items", [32] * 128)]


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{label} has unsupported or missing fields")


def _sha(value, label):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _bounded_json(value, depth=0, count=None):
    if count is None:
        count = [0]
    count[0] += 1
    if depth > 8 or count[0] > 4096:
        raise ValueError("Training metadata exceeds its small JSON contract")
    if value is None or type(value) in (bool, int, str):
        if type(value) is str and len(value) > 4096:
            raise ValueError("Training metadata string is too long")
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _bounded_json(item, depth + 1, count)
        return
    if type(value) is list:
        for item in value:
            _bounded_json(item, depth + 1, count)
        return
    raise ValueError("Training metadata must contain finite JSON values")


def validate_manifest(manifest):
    """Check the small owned schema without deserializing either tensor file."""
    _keys(manifest, ("schema_version", "source_revision", "environment", "contract",
                     "graph", "training", "files", "cases"), "Manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema version")
    revision = manifest["source_revision"]
    if type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Source revision must be a complete Git commit identifier")
    _keys(manifest["contract"], CONTRACT, "Contract")
    if (manifest["contract"] != CONTRACT
            or any(type(manifest["contract"][key]) is not type(value)
                   for key, value in CONTRACT.items())):
        raise ValueError("Artifact input contract differs from this consumer")
    _keys(manifest["graph"], GRAPH, "Graph declaration")
    if (manifest["graph"] != GRAPH
            or any(type(manifest["graph"][key]) is not type(value)
                   for key, value in GRAPH.items())):
        raise ValueError("Artifact graph declaration differs from this consumer")
    environment = manifest["environment"]
    _keys(environment, (*PINNED, "native_library_sha256", "wheel_sha256"), "Environment")
    if any(environment[key] != value for key, value in PINNED.items()):
        raise ValueError("Artifact environment differs from the pinned versions")
    _sha(environment["native_library_sha256"], "Native library hash")
    _sha(environment["wheel_sha256"], "Wheel hash")
    training = manifest["training"]
    _keys(training, ("mode", "seed", "updates", "selected_update", "data", "validation"),
          "Training metadata")
    if training["mode"] not in ("full", "smoke", "test_fixture"):
        raise ValueError("Unsupported training mode")
    if type(training["seed"]) is not int or training["seed"] != 11:
        raise ValueError("Training seed differs from the declared route")
    updates, selected = training["updates"], training["selected_update"]
    if type(updates) is not int or type(selected) is not int or not 0 <= selected <= updates <= 200:
        raise ValueError("Training update counts are invalid")
    if training["mode"] in ("full", "smoke"):
        expected_updates = 200 if training["mode"] == "full" else 20
        if updates != expected_updates or selected < 1:
            raise ValueError("Training count differs from the declared mode")
    if type(training["data"]) is not dict or type(training["validation"]) is not dict:
        raise ValueError("Training data and validation metadata must be objects")
    _bounded_json(training)
    if len(json.dumps(training, allow_nan=False).encode()) > 32768:
        raise ValueError("Training metadata exceeds its byte limit")
    files = manifest["files"]
    _keys(files, ("model.pt2", "weights.pt"), "Artifact files")
    for name, entry in files.items():
        _keys(entry, ("sha256", "size_bytes"), f"File {name}")
        _sha(entry["sha256"], f"File {name} hash")
        if type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= FILE_LIMITS[name]:
            raise ValueError(f"File {name} exceeds its byte contract")
    specs = make_cases()
    if type(manifest["cases"]) is not list or len(manifest["cases"]) != len(specs):
        raise ValueError("Artifact inference case list differs from the declared route")
    for entry, spec in zip(manifest["cases"], specs):
        _keys(entry, ("name", "expected_logits"), "Inference case")
        if entry["name"] != spec["name"]:
            raise ValueError("Inference cases must use the declared names and order")
        logits = entry["expected_logits"]
        if type(logits) is not list or len(logits) != spec["offsets"].numel() - 1:
            raise ValueError("Expected logits differ from the case group count")
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in logits):
            raise ValueError("Expected logits must be finite JSON numbers")


def validate_inputs(values, offsets):
    """Admit original tensors without copying, converting or changing inputs."""
    for tensor in (values, offsets):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Predictor inputs must be tensors")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError("Predictor inputs must use CPU strided tensors")
        if not tensor.is_contiguous():
            raise ValueError("Predictor inputs must be contiguous")
    if values.dtype != torch.float32 or values.dim() != 2 or values.shape[1] != 4:
        raise ValueError("Values must have float32 shape [N,4]")
    if offsets.dtype != torch.int64 or offsets.dim() != 1:
        raise ValueError("Offsets must have int64 shape [B+1]")
    if not 0 <= values.shape[0] <= 4096 or not 3 <= offsets.numel() <= 129:
        raise ValueError("Predictor requires 0 <= N <= 4096 and 2 <= B <= 128")
    if not bool((torch.isfinite(values) & (values.abs() <= 6)).all()):
        raise ValueError("Values must be finite and within [-6,6]")
    boundaries = offsets.tolist()
    if boundaries[0] != 0 or boundaries[-1] != values.shape[0]:
        raise ValueError("Offsets must start at zero and end at N")
    if any(first > second for first, second in zip(boundaries, boundaries[1:])):
        raise ValueError("Offsets must be nondecreasing")


def _read_bounded(path, limit):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an ordinary artifact file: {path.name}")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if not payload or len(payload) > limit:
        raise ValueError(f"Artifact file exceeds its byte contract: {path.name}")
    return payload


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite manifest JSON value: {value}")


class Predictor:
    """Apply admission before the exported module's own runtime guards."""

    def __init__(self, exported_program, manifest):
        self.exported_program = exported_program
        self.manifest = manifest
        # Exported modules retain their saved mode and do not support .eval().
        self.module = exported_program.module()

    def __call__(self, values, offsets):
        validate_inputs(values, offsets)
        with torch.inference_mode():
            output = self.module(values, offsets)
        if (not isinstance(output, torch.Tensor) or output.shape != (offsets.numel() - 1,)
                or output.device.type != "cpu" or output.dtype != torch.float32
                or not bool(torch.isfinite(output).all())):
            raise RuntimeError("Exported prediction violates its output contract")
        return output


def load_artifact(directory, manifest_sha256):
    """Load only a trusted owned artifact matching an external expected digest."""
    _sha(manifest_sha256, "Expected manifest hash")
    directory = Path(directory).resolve()
    payload = _read_bounded(directory / "manifest.json", FILE_LIMITS["manifest.json"])
    if hashlib.sha256(payload).hexdigest() != manifest_sha256:
        raise ValueError("Manifest SHA-256 differs from the expected producer receipt")
    manifest = json.loads(payload.decode("utf-8"), object_pairs_hook=_object,
                          parse_constant=_invalid_constant)
    validate_manifest(manifest)
    if {path.name for path in directory.iterdir()} != set(FILE_LIMITS):
        raise ValueError("The artifact directory must contain exactly its three declared files")
    identity = environment_identity()
    if any(manifest["environment"][key] != value for key, value in identity.items()):
        raise ValueError("Artifact differs from the installed native environment")
    verified = {}
    for name, expected in manifest["files"].items():
        data = _read_bounded(directory / name, FILE_LIMITS[name])
        if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Artifact hash or size differs from its manifest: {name}")
        verified[name] = data
    # Load exactly the verified graph bytes. The separately hashed weights file
    # supports producer recovery checks and is never loaded by this consumer.
    exported = torch.export.load(io.BytesIO(verified["model.pt2"]))
    targets = [node.target for node in exported.graph_module.graph.nodes if node.op == "call_function"]
    if sum(target == torch.ops.ragged_pool.pool.default for target in targets) != 1:
        raise ValueError("Expected exactly one native pooling operation in the graph")
    if any("backward" in str(target).lower() for target in targets):
        raise ValueError("An inference artifact must not contain backward operations")
    custom = [str(target) for target in targets if str(target).startswith("ragged_pool.")]
    if custom != ["ragged_pool.pool.default"]:
        raise ValueError("The graph contains unsupported ragged-pool operations")
    return Predictor(exported, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    directory, output = arguments.directory.resolve(), arguments.output.resolve()
    if output == directory or output.is_relative_to(directory):
        parser.error("Keep the consumer receipt outside the artifact directory")
    if output.exists():
        parser.error("The consumer receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "started", "pid": os.getpid(), "working_directory": os.getcwd(),
               "python_executable": sys.executable, "script_sha256": file_sha256(__file__),
               "expected_manifest_sha256": arguments.manifest_sha256, "cases": []}
    result = 1
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        receipt["environment"] = environment_identity()
        receipt["package_import_path"] = str(Path(package.__file__).resolve())
        receipt["training_modules_loaded"] = [name for name in sys.modules
            if name.rsplit(".", 1)[-1] in ("learn_sets", "export_sets")
            or name.endswith(("_learn_sets", "_export_sets"))]
        if receipt["training_modules_loaded"]:
            raise RuntimeError("Fresh-process prediction unexpectedly imported training code")
        predictor = load_artifact(directory, arguments.manifest_sha256)
        receipt["source_revision"] = predictor.manifest["source_revision"]
        receipt["graph"] = predictor.manifest["graph"]
        receipt["artifact_files"] = predictor.manifest["files"]
        for inputs, expected in zip(make_cases(), predictor.manifest["cases"]):
            original_values, original_offsets = inputs["values"].clone(), inputs["offsets"].clone()
            actual = predictor(inputs["values"], inputs["offsets"])
            target = torch.tensor(expected["expected_logits"], dtype=torch.float32)
            torch.testing.assert_close(actual, target, **TOLERANCE)
            torch.testing.assert_close(inputs["values"], original_values, rtol=0, atol=0)
            torch.testing.assert_close(inputs["offsets"], original_offsets, rtol=0, atol=0)
            receipt["cases"].append({"name": inputs["name"], "n": inputs["values"].shape[0],
                                     "b": inputs["offsets"].numel() - 1,
                                     "max_abs_difference": float((actual - target).abs().max()),
                                     "logits": actual.tolist(), "inputs_unchanged": True})
        receipt["status"] = "passed"
        result = 0
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "cases": len(receipt["cases"]),
                      "receipt": str(output)}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
