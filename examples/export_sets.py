"""Train, save and export the bounded CPU set classifier from an installed wheel.

The bundle contains a tensor state dictionary, a strict exported program and a
manifest. The manifest's digest must reach the consumer through a trusted route.
Hash agreement identifies bytes and does not make an unknown archive safe.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

import torch


def _adjacent(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("n3_" + name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load adjacent example {path.name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


learn = _adjacent("learn_sets")
predict = _adjacent("predict_sets")


def validate_state_dict(state, expected_state):
    """Admit only the exact finite float32 tensors needed by this model."""
    if type(state) is not dict or set(state) != set(expected_state):
        raise ValueError("The state dictionary must contain exactly the model's parameter keys")
    for name, expected in expected_state.items():
        value = state[name]
        if type(value) is not torch.Tensor:
            raise ValueError(f"State entry {name} must be an ordinary tensor")
        if (value.device.type != "cpu" or value.layout != torch.strided
                or value.dtype != torch.float32 or value.shape != expected.shape
                or not value.is_contiguous() or value.is_conj() or value.is_neg()):
            raise ValueError(f"State entry {name} has an unsupported shape, dtype or representation")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"State entry {name} must be finite")


def read_weights(path, expected_state):
    """Load the producer's tensor archive once with the restricted loader."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    validate_state_dict(state, expected_state)
    return state


def graph_metadata(exported):
    """Require one opaque forward pooling node and no backward operation."""
    operations = [node.target for node in exported.graph_module.graph.nodes
                  if node.op == "call_function"]
    pool_count = sum(target == torch.ops.ragged_pool.pool.default for target in operations)
    backward_count = sum("backward" in str(target).lower() for target in operations)
    custom = [str(target) for target in operations if str(target).startswith("ragged_pool.")]
    if pool_count != 1 or backward_count or custom != ["ragged_pool.pool.default"]:
        raise AssertionError("Export must retain exactly one pooling forward and no backward node")
    return {"pool_operator": "ragged_pool::pool", "pool_count": pool_count,
            "backward_count": backward_count}


def export_model(model):
    """Export this model's declared inference subset with strict tracing."""
    if getattr(model, "pooling", None) != "native":
        raise ValueError("Export requires the native attention classifier")
    model.eval()
    values = torch.linspace(-1, 1, 32, dtype=torch.float32).reshape(8, 4).contiguous()
    offsets = torch.tensor([0, 3, 3, 8], dtype=torch.int64)
    items = torch.export.Dim("items", min=0, max=4096)
    offset_count = torch.export.Dim("offset_count", min=3, max=129)
    with torch.no_grad():
        exported = torch.export.export(
            model, (values, offsets),
            dynamic_shapes=({0: items}, {0: offset_count}), strict=True)
    graph_metadata(exported)
    return exported


def _save_receipt(path, receipt):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _difference(actual, expected):
    return float((actual.detach() - expected.detach()).abs().max()) if actual.numel() else 0.0


def _check(budget, location):
    if budget is not None:
        budget.check(location)


def create_artifact(model, directory, *, source_revision, wheel_sha256, training, budget=None):
    """Write a new trusted bundle through the same path used by the CLI and tests."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be a complete lowercase Git commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", wheel_sha256):
        raise ValueError("wheel_sha256 must be a lowercase SHA256 digest")
    if getattr(model, "pooling", None) != "native":
        raise ValueError("The producer expects the selected native attention model")
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    _check(budget, "before saving selected weights")
    state = {name: value.detach().clone().contiguous() for name, value in model.state_dict().items()}
    expected = learn.make_model(11, "native").state_dict()
    validate_state_dict(state, expected)
    weights_path = directory / "weights.pt"
    torch.save(state, weights_path)
    reloaded_state = read_weights(weights_path, expected)
    native = learn.make_model(11, "native")
    reference = learn.make_model(11, "reference")
    native.load_state_dict(reloaded_state, strict=True)
    reference.load_state_dict(reloaded_state, strict=True)
    parameter_errors = {}
    for name in state:
        torch.testing.assert_close(reloaded_state[name], state[name], rtol=0, atol=0)
        torch.testing.assert_close(native.state_dict()[name], state[name], rtol=0, atol=0)
        torch.testing.assert_close(reference.state_dict()[name], state[name], rtol=0, atol=0)
        parameter_errors[name] = _difference(reloaded_state[name], state[name])
    model.eval()
    native.eval()
    reference.eval()
    _check(budget, "after weights-only reload and strict parameter check")
    exported = export_model(native)
    graph = graph_metadata(exported)
    program_path = directory / "model.pt2"
    torch.export.save(exported, program_path)
    loaded = torch.export.load(program_path)
    if graph_metadata(loaded) != graph:
        raise AssertionError("Saved and loaded graph metadata differ")
    # Exported modules do not support eval(). The original model was put in
    # evaluation mode before export, and inference_mode controls these calls.
    exported_model = loaded.module()
    _check(budget, "after strict export and exported-program reload")
    cases, errors = [], []
    with torch.inference_mode():
        for case in predict.make_cases():
            values, offsets = case["values"], case["offsets"]
            predict.validate_inputs(values, offsets)
            trained_logits = model(values, offsets)
            native_logits = native(values, offsets)
            reference_logits = reference(values, offsets)
            exported_logits = exported_model(values, offsets)
            for actual in (native_logits, reference_logits, exported_logits):
                torch.testing.assert_close(actual, trained_logits, **learn.TOLERANCE)
                if actual.shape != (offsets.numel() - 1,) or not bool(torch.isfinite(actual).all()):
                    raise AssertionError("A producer case returned invalid logits")
            cases.append({"name": case["name"], "expected_logits": trained_logits.tolist()})
            errors.append({"name": case["name"],
                           "reloaded_native_max_abs": _difference(native_logits, trained_logits),
                           "reference_max_abs": _difference(reference_logits, trained_logits),
                           "exported_max_abs": _difference(exported_logits, trained_logits)})
            _check(budget, "checked export case " + case["name"])
    environment = predict.environment_identity()
    environment["wheel_sha256"] = wheel_sha256
    files = {path.name: {"sha256": predict.file_sha256(path), "size_bytes": path.stat().st_size}
             for path in (weights_path, program_path)}
    manifest = {"schema_version": 1, "source_revision": source_revision,
                "environment": environment, "contract": predict.CONTRACT,
                "graph": graph, "training": training, "files": files, "cases": cases}
    predict.validate_manifest(manifest)
    manifest_path = directory / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if set(path.name for path in directory.iterdir()) != {"weights.pt", "model.pt2", "manifest.json"}:
        raise AssertionError("The artifact directory contains unexpected files")
    return {"manifest_sha256": predict.file_sha256(manifest_path), "files": files,
            "graph": graph, "state_parameter_reload_max_abs": parameter_errors,
            "case_comparisons": errors, "environment": environment}


class Budget:
    def __init__(self, receipt):
        self.start = time.perf_counter()
        self.receipt = receipt

    def check(self, location):
        elapsed = time.perf_counter() - self.start
        rss, method = learn._rss_bytes()
        self.receipt["resources"]["samples"].append(
            {"location": location, "seconds": elapsed, "rss_bytes": rss, "method": method})
        if rss > 4 * 1024**3:
            raise learn.ResourceStop("Sampled RSS exceeded 4 GiB")
        if elapsed > 180:
            raise learn.ResourceStop("The 180-second producer budget expired")


def run(arguments):
    path = Path(arguments.output).resolve()
    directory = Path(arguments.directory).resolve()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("The artifact directory must be new")
    if path == directory or path.is_relative_to(directory):
        raise ValueError("The producer receipt must be outside the three-file artifact directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    receipt = {"status": "running", "mode": "smoke" if arguments.smoke else "full",
               "source_revision": arguments.source_revision,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_files": {name: predict.file_sha256(Path(__file__).resolve().with_name(name))
                                for name in ("export_sets.py", "predict_sets.py", "learn_sets.py")},
               "resources": {"budget_seconds": 180, "sampled_rss_stop_bytes": 4 * 1024**3,
                             "sampling_is_not_hard_os_limit": True, "samples": []}}
    budget = Budget(receipt)
    _save_receipt(path, receipt)
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", arguments.source_revision):
            raise ValueError("source_revision must be a complete lowercase Git commit SHA")
        identity = learn._identity(arguments.source_revision, arguments.wheel)
        receipt["identity"] = identity
        actual_environment = predict.environment_identity()
        required = {"python_version": "3.11.8", "torch_version": "2.9.1+cpu", "package_version": "0.1.0"}
        if any(actual_environment[key] != value for key, value in required.items()):
            raise RuntimeError("The producer requires the declared Python 3.11.8 / PyTorch 2.9.1+cpu environment")
        splits = {name: learn.make_split(*learn.SPLIT_SPECS[name]) for name in ("train", "validation")}
        data = learn._split_metadata(splits)
        receipt["data"] = data
        budget.check("generated training and validation only")
        updates = 20 if arguments.smoke else 200
        receipt["training"] = {}
        model, _, progress = learn._fit(
            splits["train"], splits["validation"], 11, "native", updates, budget,
            progress=receipt["training"], checkpoint=lambda: _save_receipt(path, receipt))
        training = {"mode": receipt["mode"], "seed": 11, "updates": updates,
                    "selected_update": progress["selected_update"], "data": data,
                    "validation": progress["validation"]}
        receipt["artifact"] = create_artifact(
            model, directory, source_revision=arguments.source_revision,
            wheel_sha256=identity["wheel_sha256"], training=training, budget=budget)
        budget.check("finished artifact")
        receipt["status"] = "passed_smoke" if arguments.smoke else "passed"
    except BaseException as error:
        receipt["status"] = "partial_resource_stop" if isinstance(error, learn.ResourceStop) else (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - budget.start
        _save_receipt(path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="New three-file artifact directory")
    parser.add_argument("--output", required=True, help="New producer JSON receipt outside the artifact")
    parser.add_argument("--source-revision", required=True, help="Complete Git commit SHA for these examples")
    parser.add_argument("--wheel", required=True, help="Installed wheel file for SHA256 identity")
    parser.add_argument("--smoke", action="store_true", help="20 updates instead of the full 200")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run(arguments)
    print(json.dumps({"status": result["status"], "seconds": result["seconds"],
                      "manifest_sha256": result["artifact"]["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
