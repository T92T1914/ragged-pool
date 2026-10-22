"""Validate the bounded activation-recomputation receipt without loading tensors.

The executed consumer owns numerical assertions. This gate checks declared
state, original-forward save events and independently supplied byte identities.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def saved_events(record):
    require(type(record) is dict and type(record.get("events")) is list, "Missing save events")
    total = 0
    sizes = {"torch.float32": 4, "torch.float64": 8, "torch.int64": 8}
    for event in record["events"]:
        require(type(event) is dict and type(event.get("shape")) is list, "Invalid saved shape")
        shape = event["shape"]
        require(all(type(value) is int and value >= 0 for value in shape), "Invalid saved dimension")
        numel = math.prod(shape)
        dtype = event.get("dtype")
        require(dtype in sizes and type(event.get("numel")) is int and event["numel"] == numel,
                "Saved element count or dtype differs")
        require(type(event.get("element_size")) is int and event["element_size"] == sizes[dtype],
                "Saved element size differs")
        require(type(event.get("logical_bytes")) is int
                and event["logical_bytes"] == numel * sizes[dtype], "Saved event byte count differs")
        total += event["logical_bytes"]
    require(type(record.get("count")) is int and record["count"] == len(record["events"]),
            "Saved event count differs")
    require(type(record.get("logical_bytes")) is int and record["logical_bytes"] == total,
            "Saved payload sum differs")
    return total


def check(receipt, dtype, script_sha256, native_sha256, helper_sha256):
    require(type(receipt) is dict and dtype in ("float32", "float64"), "Invalid receipt or dtype")
    literals = {"status": "passed", "method": "checkpoint.non_reentrant.packed_training",
                "dtype": dtype, "device": "cpu", "use_reentrant": False,
                "preserve_rng_state": True, "early_stop": False, "determinism_check": "default",
                "dropout_probability": 0.25, "dropout_seed": 2211, "optimizer": "SGD",
                "learning_rate": 0.03, "momentum": 0.8, "optimizer_steps": 2,
                "scalar_parameters": 25, "parameter_tensors": 5,
                "independent_momentum_recurrence": True, "same_native_exact": True,
                "primitive_comparisons_passed": True, "inputs_unchanged": True,
                "parameter_ownership_unchanged": True, "installed_package": True,
                "native_library_loaded": True, "threads": 1, "interop_threads": 1,
                "python_version": "3.11.8", "torch_version": "2.9.1+cpu",
                "saved_accounting": "original_forward_pack_hook_events",
                "memory_performance_claim": False}
    for key, expected in literals.items():
        actual = receipt.get(key)
        require(type(actual) is type(expected) and actual == expected, f"Unexpected {key}")
    tolerance = ({"rtol": 2e-4, "atol": 2e-6} if dtype == "float32"
                 else {"rtol": 2e-10, "atol": 2e-12})
    require(receipt.get("comparison_tolerance") == tolerance, "Changed numerical tolerance")
    require(number(receipt.get("same_native_max_abs_error"))
            and receipt["same_native_max_abs_error"] == 0, "Native recomputation differed")
    require(number(receipt.get("primitive_max_abs_error")), "Invalid primitive comparison error")
    for key, expected in (("script_sha256", script_sha256),
                          ("native_library_sha256", native_sha256),
                          ("train_dynamic_sha256", helper_sha256)):
        require(digest(expected) and receipt.get(key) == expected, f"Changed external {key}")
    steps = receipt.get("steps")
    require(type(steps) is list and len(steps) == 2, "Missing completed updates")
    boundaries = ([0, 2, 2, 5, 7, 9], [0, 3, 5, 5, 8, 10, 12])
    for index, step in enumerate(steps):
        require(type(step) is dict and type(step.get("update")) is int
                and step["update"] == index + 1, "Wrong update sequence")
        require(step.get("offsets") == boundaries[index]
                and all(type(value) is int for value in step["offsets"])
                and type(step.get("N")) is int and step["N"] == boundaries[index][-1]
                and type(step.get("B")) is int and step["B"] == len(boundaries[index]) - 1,
                "Changed fixture identity")
        require(number(step.get("loss")) and number(step.get("native_exact_max_abs_error"))
                and step["native_exact_max_abs_error"] == 0
                and number(step.get("primitive_max_abs_error")), "Invalid update comparisons")
        require(step.get("checkpoint_scope_events") ==
                ["forward_enter", "forward_exit", "recompute_enter", "recompute_exit"],
                "Incomplete explicit full recomputation witness")
        saved = step.get("saved_forward")
        require(type(saved) is dict and set(saved) == {"direct", "checkpointed"},
                "Missing original-forward save accounting")
        direct = saved_events(saved["direct"])
        checkpointed = saved_events(saved["checkpointed"])
        require(0 <= checkpointed < direct, "The fixed forward save-event payload did not decrease")
        for key in ("default_rng_before_sha256", "default_rng_after_forward_sha256",
                    "default_rng_after_backward_sha256"):
            require(digest(step.get(key)), f"Missing {key}")
        require(step["default_rng_after_backward_sha256"] == step["default_rng_after_forward_sha256"]
                and step["default_rng_before_sha256"] != step["default_rng_after_forward_sha256"],
                "Proper recomputation changed the post-forward RNG")
        if index:
            require(step["default_rng_before_sha256"] == steps[index - 1]["default_rng_after_backward_sha256"],
                    "Discontinuous default RNG progression")
    controls = receipt.get("controls")
    require(type(controls) is dict and set(controls) == {"without_rng_preservation", "dropout_disabled"},
            "Missing controls")
    wrong = controls["without_rng_preservation"]
    threshold = 1e-7 if dtype == "float32" else 1e-10
    require(type(wrong) is dict and wrong.get("first_forward_exact") is True
            and wrong.get("diverged") is True and wrong.get("rng_diverged") is True
            and number(wrong.get("divergence_threshold")) and wrong["divergence_threshold"] == threshold
            and number(wrong.get("gradient_max_abs_error")) and wrong["gradient_max_abs_error"] > threshold
            and number(wrong.get("final_parameter_max_abs_error"))
            and wrong["final_parameter_max_abs_error"] > threshold, "RNG control was not deciding")
    plain = controls["dropout_disabled"]
    require(type(plain) is dict and type(plain.get("dropout_probability")) in (int, float)
            and plain["dropout_probability"] == 0 and plain.get("same_native_exact") is True
            and number(plain.get("max_abs_error")) and plain["max_abs_error"] == 0
            and plain.get("rng_unchanged") is True, "Dropout-disabled positive control failed")
    return {"status": "passed", "dtype": dtype, "updates": 2,
            "actual_recomputation": True, "saved_accounting": literals["saved_accounting"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("dtype", choices=("float32", "float64"))
    args = parser.parse_args()
    payload = args.receipt.read_bytes()
    require(0 < len(payload) <= 256 * 1024, "Receipt exceeds its byte bound")
    receipt = json.loads(payload)
    examples = Path(__file__).resolve().parents[1] / "examples"
    result = check(receipt, args.dtype,
                   sha256((examples / "train_checkpointed.py").read_bytes()).hexdigest(),
                   sha256(Path(receipt["native_library_path"]).read_bytes()).hexdigest(),
                   sha256((examples / "train_dynamic.py").read_bytes()).hexdigest())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
