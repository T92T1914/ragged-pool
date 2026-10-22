"""Check the bounded installed stochastic-resume receipt, without loading tensors.

Numerical assertions belong to the executed consumer. This checks its declared
state progression and source identity, not arbitrary checkpoint safety.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


CONTROLS = ("reset_dropout_rng", "omit_dropout_rng", "reset_order_generator", "reset_optimizer",
            "repeat_cursor", "skip_cursor")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check(receipt, dtype, script_sha256, native_sha256):
    require(type(receipt) is dict and dtype in ("float32", "float64"), "Invalid receipt or dtype")
    literals = {"status": "passed", "method": "state_dict.stochastic_packed_training_resume",
                "dtype": dtype, "device": "cpu", "dropout_probability": 0.25,
                "training_mode": True, "optimizer_steps": 7, "producer_steps": 2,
                "resumed_steps": 5, "checkpoint_cursor": 2, "checkpoint_epoch": 0,
                "order_regenerations": 1, "distinct_processes": True,
                "same_native_exact": True, "primitive_comparisons_passed": True,
                "independent_momentum_recurrence": True, "checkpoint_unchanged": True,
                "input_sources_unchanged": True, "initial_parameters_unchanged": True,
                "loaded_state_storage_independent": True, "installed_package": True,
                "native_library_loaded": True, "threads": 1, "interop_threads": 1,
                "python_version": "3.11.8", "torch_version": "2.9.1+cpu"}
    for key, value in literals.items():
        actual = receipt.get(key)
        require(type(actual) is type(value) and actual == value, f"Unexpected {key}")
    require(number(receipt.get("same_native_max_abs_error"))
            and receipt["same_native_max_abs_error"] == 0, "Native replay was not exact")
    require(number(receipt.get("primitive_max_abs_error"))
            and receipt["primitive_max_abs_error"] >= 0, "Invalid primitive error")
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    require(receipt.get("comparison_tolerance") == tolerance, "Changed comparison tolerance")
    require(type(receipt.get("checkpoint_bytes")) is int
            and 0 < receipt["checkpoint_bytes"] <= 128 * 1024
            and receipt.get("checkpoint_byte_limit") == 128 * 1024, "Checkpoint byte bound failed")
    require(digest(receipt.get("checkpoint_sha256")), "Invalid checkpoint digest")
    require(digest(script_sha256) and digest(native_sha256), "Missing external identity pins")
    require(receipt.get("script_sha256") == script_sha256, "Consumer source identity differs")
    require(receipt.get("native_library_sha256") == native_sha256, "Native identity differs")
    pids = receipt.get("process_ids")
    require(type(pids) is dict and set(pids) == {"parent", "producer", "resumer"}
            and all(type(value) is int and value > 0 for value in pids.values())
            and len(set(pids.values())) == 3, "Processes were not distinct")
    children = receipt.get("child_installed_identities")
    require(type(children) is dict and set(children) == {"producer", "resumer"}, "Missing child identities")
    for identity in children.values():
        require(type(identity) is dict and identity.get("native_library_loaded") is True,
                "Child native library not loaded")
        for key in ("package_path", "package_version", "environment_prefix", "native_library_path",
                    "native_library_sha256", "script_sha256"):
            require(key in receipt and identity.get(key) == receipt[key], f"Child {key} differs")
    controls = receipt.get("controls")
    require(type(controls) is dict and set(controls) == set(CONTROLS), "Incomplete recovery controls")
    threshold = 1e-7 if dtype == "float32" else 1e-10
    for name, control in controls.items():
        require(type(control) is dict and control.get("diverged") is True
                and number(control.get("divergence_threshold"))
                and control["divergence_threshold"] == threshold
                and number(control.get("final_parameter_max_abs_error"))
                and control["final_parameter_max_abs_error"] > threshold,
                f"Control did not distinguish incomplete recovery: {name}")
    steps = receipt.get("steps")
    require(type(steps) is list and len(steps) == 7, "Missing completed updates")
    for index, step in enumerate(steps):
        require(type(step) is dict, "Invalid step")
        for key, value in {"update": index + 1, "epoch": index // 5,
                           "cursor_before": 5 if index == 5 else index % 5,
                           "cursor_after": index % 5 + 1}.items():
            require(type(step.get(key)) is int and step[key] == value, f"Unexpected step {key}")
        require(type(step.get("batch_id")) is int and 0 <= step["batch_id"] < 5, "Invalid batch identity")
        for key in ("default_rng_before_sha256", "default_rng_after_sha256", "order_rng_sha256"):
            require(digest(step.get(key)), f"Missing step random state: {key}")
        require(step["default_rng_before_sha256"] != step["default_rng_after_sha256"],
                "Dropout did not advance random state")
        if index:
            require(step["default_rng_before_sha256"] == steps[index - 1]["default_rng_after_sha256"],
                    "Random progression is discontinuous")
            require((step["order_rng_sha256"] != steps[index - 1]["order_rng_sha256"]) == (index == 5),
                    "Shuffle generator progression differs from the epoch boundary")
        offsets = step.get("offsets")
        require(type(offsets) is list and 2 <= len(offsets) <= 129
                and all(type(value) is int for value in offsets)
                and offsets[0] == 0 and all(a <= b for a, b in zip(offsets, offsets[1:])),
                "Invalid packed boundaries")
        require(type(step.get("N")) is int and offsets[-1] == step["N"] <= 4096
                and type(step.get("B")) is int and step["B"] == len(offsets) - 1,
                "Packed metadata differs from boundaries")
        require(number(step.get("loss")) and step["loss"] >= 0, "Invalid loss")
    require(len({step["batch_id"] for step in steps[:5]}) == 5
            and len({step["batch_id"] for step in steps[5:]}) == 2,
            "The recorded orders are not permutation prefixes")
    return {"status": "passed", "dtype": dtype, "updates": 7,
            "fresh_processes": 3, "epoch_boundary_exercised": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("dtype", choices=("float32", "float64"))
    args = parser.parse_args()
    payload = args.receipt.read_bytes()
    require(0 < len(payload) <= 512 * 1024, "Receipt size exceeds the declared bound")
    receipt = json.loads(payload)
    source = Path(__file__).resolve().parents[1] / "examples" / "resume_stochastic.py"
    native = Path(receipt["native_library_path"])
    result = check(receipt, args.dtype, sha256(source.read_bytes()).hexdigest(),
                   sha256(native.read_bytes()).hexdigest())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
