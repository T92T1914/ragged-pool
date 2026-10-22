"""Train two CUDA updates and export one owned, fixed-shape inference graph.

Use only trusted local artifacts. The adjacent predictor does not import this
producer or its training model class. This is a same-environment handoff, with
no optimization, portable executable or hostile-model safety claim.
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import torch


def _adjacent(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("cuda_export_" + name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load adjacent example {path.name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


train = _adjacent("train_cuda")
predict = _adjacent("predict_cuda")


def train_model(dtype_name):
    """Retain complete two-update comparisons with independently derived math."""
    initial, batches, _ = train.fixture(dtype_name, torch.device("cuda", 0))
    models = {name: copy.deepcopy(initial) for name in ("native", "primitive")}
    initial_before = predict.snapshot(dict(initial.named_parameters()))
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.03, foreach=False, fused=False)
                  for name, model in models.items()}
    steps = []
    for index, batch in enumerate(batches, 1):
        inputs = {key: batch[key] for key in ("raw", "targets", "offsets")}
        inputs_before = predict.snapshot(inputs)
        before = {name: {key: value.detach().clone() for key, value in model.named_parameters()}
                  for name, model in models.items()}
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        results = {}
        for name, model in models.items():
            raw = batch["raw"].detach().clone().requires_grad_()
            results[name] = train.evaluate(model, raw, batch, name == "primitive")
        tolerance = predict.tolerance(dtype_name)
        output_errors = train.compare(results["native"][0], results["primitive"][0], tolerance)
        gradient_errors = train.compare(results["native"][1], results["primitive"][1], tolerance)
        for optimizer in optimizers.values():
            optimizer.step()
        updated = {name: dict(model.named_parameters()) for name, model in models.items()}
        parameter_errors = train.compare(updated["native"], updated["primitive"], tolerance)
        changes = {name: {key: train.difference(value, before[name][key])
                          for key, value in parameters.items()} for name, parameters in updated.items()}
        if any(value <= 0 for route in changes.values() for value in route.values()):
            raise AssertionError("Both routes must update every parameter block at each step")
        predict.unchanged(inputs, inputs_before)
        steps.append({"step": index, "native_loss": float(results["native"][0]["loss"]),
            "reference_loss": float(results["primitive"][0]["loss"]),
            "output_max_abs_difference": output_errors,
            "parameter_gradient_max_abs_difference": gradient_errors,
            "native_parameter_update_max_abs": changes["native"],
            "reference_parameter_update_max_abs": changes["primitive"],
            "updated_parameter_max_abs_difference": parameter_errors})
    predict.unchanged(dict(initial.named_parameters()), initial_before)
    return models["native"], {"optimizer": "SGD", "learning_rate": 0.03,
        "optimizer_steps": 2, "reference_optimizer_steps": 2,
        "all_parameter_blocks_changed": True, "steps": steps}


def create_artifact(model, directory, *, dtype_name, source_revision, source_tree, training):
    """Save only the selected state and a small trusted-local identity manifest."""
    for identifier in (source_revision, source_tree):
        if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-f]{40}", identifier) is None:
            raise ValueError("Source revision/tree must be complete lowercase Git identifiers")
    identity = predict.environment_identity()
    weights = {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}
    predict.validate_state(weights, dtype_name)
    native_before = predict.snapshot(dict(model.named_parameters()))
    frozen = copy.deepcopy(model).eval()
    cases = predict.make_cases(dtype_name)
    capture = cases[0]
    with torch.no_grad():
        exported = torch.export.export(frozen, (capture["raw"], capture["offsets"]), strict=True)
    graph = predict.graph_metadata(exported)
    expected, comparisons = [], []
    functional_state = predict.state_tensors(weights, dtype_name)
    with torch.no_grad():
        for case in cases:
            raw, offsets = case["raw"], case["offsets"]
            predict.validate_inputs(raw, offsets, dtype_name)
            inputs = {"raw": raw, "offsets": offsets}
            before = predict.snapshot(inputs)
            eager = model(raw, offsets)
            reference = predict.reference(raw, case["boundaries"], functional_state)
            captured = exported.module()(raw, offsets)
            errors = {"name": case["name"]}
            for name, output in (("reference", reference), ("captured", captured)):
                for actual, target in zip(output, eager):
                    torch.testing.assert_close(actual, target, **predict.tolerance(dtype_name))
                    if not bool(torch.isfinite(actual).all()):
                        raise AssertionError("Inference must produce finite outputs")
                errors[name + "_prediction_max_abs"] = train.difference(output[0], eager[0])
                errors[name + "_pooled_max_abs"] = train.difference(output[1], eager[1])
            empty = [i for i, (a, b) in enumerate(zip(case["boundaries"], case["boundaries"][1:])) if a == b]
            torch.testing.assert_close(eager[1][empty], torch.zeros_like(eager[1][empty]), rtol=0, atol=0)
            predict.unchanged(inputs, before)
            expected.append({"name": case["name"], "predictions": eager[0].tolist(), "pooled": eager[1].tolist()})
            comparisons.append(errors)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    program = directory / "model.pt2"
    torch.export.save(exported, program)
    manifest = {"schema_version": 1, "dtype": dtype_name, "source_revision": source_revision,
        "source_tree": source_tree, "producer_pid": os.getpid(),
        "producer_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": identity, "contract": predict.contract(dtype_name), "graph": graph,
        "training": training, "weights": weights,
        "files": {"model.pt2": {"sha256": sha256(program.read_bytes()).hexdigest(), "size_bytes": program.stat().st_size}},
        "expected": expected}
    path = directory / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    digest = sha256(path.read_bytes()).hexdigest()
    # The producer checks the same ordinary loader. Final acceptance also runs
    # that loader in a separate process with no producer/model imports.
    predictor = predict.load_artifact(directory, digest)
    for case, errors in zip(cases, comparisons):
        with torch.no_grad():
            actual, eager = predictor(case["raw"], case["offsets"]), model(case["raw"], case["offsets"])
        for value, target in zip(actual, eager):
            torch.testing.assert_close(value, target, **predict.tolerance(dtype_name))
        errors["reloaded_prediction_max_abs"] = train.difference(actual[0], eager[0])
        errors["reloaded_pooled_max_abs"] = train.difference(actual[1], eager[1])
    predict.unchanged(dict(model.named_parameters()), native_before)
    torch.cuda.synchronize(0)
    return {"manifest_sha256": digest, "files": manifest["files"], "graph": graph,
        "case_comparisons": comparisons, "parameters_unchanged_by_export": True,
        "training": training, "environment": identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=tuple(predict.DTYPES), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output, directory = arguments.output.resolve(), arguments.directory.resolve()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("Use a new artifact directory")
    if output == directory or output.is_relative_to(directory):
        raise ValueError("Keep the receipt outside the artifact directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    receipt = {"status": "running", "method": "strict.fixed_cuda_export", "dtype": arguments.dtype,
        "producer_pid": os.getpid(), "source_revision": arguments.source_revision, "source_tree": arguments.source_tree,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_script_sha256": sha256(Path(train.__file__).read_bytes()).hexdigest(),
        "predictor_script_sha256": sha256(Path(predict.__file__).read_bytes()).hexdigest()}
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        installed = train.installed_identity()
        with ZipFile(arguments.wheel) as archive:
            from importlib.metadata import distribution
            distribution_ = distribution("ragged-pool-cuda")
            for name in archive.namelist():
                if name.startswith("ragged_pool_cuda/") and not name.endswith("/"):
                    if Path(distribution_.locate_file(name)).read_bytes() != archive.read(name):
                        raise ValueError("Declared wheel payload differs from the ordinary installed package")
        if installed.pop("script_sha256") != receipt["training_script_sha256"]:
            raise AssertionError("Installed-identity helper must identify the adjacent training script")
        receipt.update(installed)
        receipt["wheel_sha256"] = sha256(arguments.wheel.read_bytes()).hexdigest()
        model, training = train_model(arguments.dtype)
        receipt["artifact"] = create_artifact(model, directory, dtype_name=arguments.dtype,
            source_revision=arguments.source_revision, source_tree=arguments.source_tree, training=training)
        properties = torch.cuda.get_device_properties(0)
        receipt.update({"status": "passed", "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()})
    except BaseException as error:
        receipt.update({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "manifest_sha256": receipt["artifact"]["manifest_sha256"]}))


if __name__ == "__main__":
    main()
