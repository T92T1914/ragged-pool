"""Time the adaptive forward kernel against the existing kernel and the eager composition.

FORWARD_KERNEL.md fixed the protocol, the shapes and the predictions before this
script existed. It measures one card in one machine with one driver, one toolkit
and the bounded sizes of the companion, and it makes no general speed claim.
Every candidate is checked against float64 math on the CPU before any timing,
and so is the output of every timed block. A failed check stops the run.
"""
import argparse
import ctypes
import gc
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

import torch

import ragged_pool_cuda as package
from ragged_pool_cuda._validation import check_metadata


SEED = 72001
EXISTING, ADAPTIVE = 0, 1
DEFAULT_SWITCH, WARP_EVERYWHERE, BLOCK_EVERYWHERE = 32, 4096, 0
FULL = {"warmup": 20, "rounds": 5, "samples": 50}
SMOKE = {"warmup": 2, "rounds": 2, "samples": 5}
BAND = 0.10
TOLERANCES = {
    "float32": {"rtol": 2e-4, "atol": 2e-6},
    "float64": {"rtol": 2e-10, "atol": 2e-12},
}

# The CPU study's eight cases, in its order, so they draw the same values and logits.
CPU_CASES = (
    ("empty_batch", [], 8, "float32"),
    ("empty_groups", [0, 0, 0, 0], 8, "float64"),
    ("singletons", [1] * 128, 8, "float32"),
    ("many_short", [4] * 128, 16, "float32"),
    ("balanced", [32] * 32, 32, "float32"),
    ("skewed", [0] + [1] * 14 + [1000], 32, "float32"),
    ("one_long", [4096], 16, "float64"),
    ("wide", [64] * 8, 128, "float64"),
)
# data_sha256 of each case under receipts/operator_timing/cases in docs/results/cpu-study.json.
CPU_STUDY_DATA_SHA256 = {
    "empty_batch": "d8dfcf9f929e439e025ca8d62576f713008f9a9fcf548b522964bacfe9b851ba",
    "empty_groups": "ef7fb6c0e88ec1c81f74b018a7be8462ae83908c6574a8a1d53710a21b360256",
    "singletons": "17fd4ff15dcf5785ea37fa82ddde4ef04c01b07b0129b35819279ce744ef849e",
    "many_short": "1bbcfd5275b85606b68734082ed428ec2ba149fc03f55f2ee5f8ee6a4db23db2",
    "balanced": "62943104666b82e7c65df3cb307985202b1442a051f7d45b16cd009a22b97582",
    "skewed": "7f5bf84e361c4c9df7b52c3d16b93f2a4f5a2698bdce30dbcba850c23fb1888d",
    "one_long": "6adc27ccc331d58fe8ce27504ca14032183621fcd743ddf5584592d0379a60cf",
    "wide": "022b847d6eee2924524c746b5289480f466145eb95eab19e2fa083ab74d017e1",
}
BOUND_CASES = (
    ("full_batch_float32", [32] * 128, 128, "float32"),
    ("full_batch_float64", [32] * 128, 128, "float64"),
    ("one_full_group", [4096], 128, "float64"),
    ("narrowest_full_batch", [32] * 128, 1, "float32"),
    ("skewed_full_batch", [1] * 127 + [3969], 128, "float32"),
    ("widest_singletons", [1] * 128, 128, "float32"),
)
SWEEP_CASES = tuple(
    (f"sweep_l{length}_d{width}", [length] * min(128, 4096 // length), width, "float32")
    for length in (8, 16, 32, 64, 128, 256) for width in (8, 128)
)

# The Predictions section of FORWARD_KERNEL.md at 95b00b8, written before any timing.
# "<" means the left median is more than 10 percent below the right one.
# "~" stands for the note's approximately equal sign: the two medians are within
# 10 percent of the smaller one. None means no kernel runs, so nothing is predicted.
PREDICTIONS = {
    "empty_batch": (None, "E ~ N < C"),
    "empty_groups": ("N ~ E", "E ~ N < C"),
    "singletons": ("N ~ E", "E ~ N < C"),
    "many_short": ("N < E", "N ~ E < C"),
    "balanced": ("N < E", "C < N < E"),
    "skewed": ("N < E", "C < N < E"),
    "one_long": ("N < E", "C < N < E"),
    "wide": ("N < E", "C < N < E"),
    "full_batch_float32": ("N < E", "C < N ~ E"),
    "full_batch_float64": ("N < E", "C < N ~ E"),
    "one_full_group": ("N < E", "C < N < E"),
    "narrowest_full_batch": ("N < E", "N < E < C"),
    "skewed_full_batch": ("N < E", "C < N < E"),
    "widest_singletons": ("N < E", "N ~ E ~ C"),
}
SWEEP_PREDICTIONS = {
    "sweep_l8_d8": "W ~ K", "sweep_l8_d128": "W < K",
    "sweep_l16_d8": "W ~ K", "sweep_l16_d128": "W ~ K",
    "sweep_l32_d8": "W ~ K", "sweep_l32_d128": "W ~ K",
    "sweep_l64_d8": "W ~ K", "sweep_l64_d128": "K < W",
    "sweep_l128_d8": "K < W", "sweep_l128_d128": "K < W",
    "sweep_l256_d8": "K < W", "sweep_l256_d128": "K < W",
}
NATIVE = {
    "E": (EXISTING, DEFAULT_SWITCH),
    "N": (ADAPTIVE, DEFAULT_SWITCH),
    "W": (ADAPTIVE, WARP_EVERYWHERE),
    "K": (ADAPTIVE, BLOCK_EVERYWHERE),
}


def tensor_bytes(tensor):
    """The bytes numpy's tobytes gives for a contiguous CPU tensor, without numpy."""
    tensor = tensor.detach().contiguous()
    if tensor.numel() == 0:
        return b""
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def digest_tensors(*tensors):
    """The digest examples/cpu_cost.py records for each case."""
    digest = sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def make_cases(definitions, device):
    """Draw every case from one generator in a fixed order, then copy it to the card once."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    cases = []
    for group, entries in definitions:
        for name, lengths, width, dtype_name in entries:
            dtype = getattr(torch, dtype_name)
            offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
            values = torch.randn(sum(lengths), width, dtype=dtype, generator=generator).clamp_(-6, 6)
            logits = torch.randn(sum(lengths), dtype=dtype, generator=generator).clamp_(-6, 6)
            # The CPU study also draws an upstream gradient. Drawing it keeps the
            # generator in step with that study. Nothing here uses it.
            upstream = torch.randn(len(lengths), width, dtype=dtype, generator=generator)
            cases.append({
                "name": name, "group": group, "lengths": lengths, "width": width, "dtype": dtype_name,
                "data_sha256": digest_tensors(values, logits, offsets, upstream),
                "cpu": (values, logits, offsets),
                "cuda": tuple(item.to(device) for item in (values, logits, offsets)),
            })
    return cases


def oracle(values, logits, offsets):
    """Float64 softmax and weighted sums on the CPU, one group at a time."""
    values64, logits64 = values.to(torch.float64), logits.to(torch.float64)
    boundaries = offsets.tolist()
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        if start == stop:
            rows.append(torch.zeros(values64.shape[1], dtype=torch.float64))
        else:
            weights = torch.softmax(logits64[start:stop], dim=0)
            rows.append((weights[:, None] * values64[start:stop]).sum(dim=0))
    return torch.stack(rows) if rows else torch.zeros(0, values64.shape[1], dtype=torch.float64)


def native(name):
    kernel, switch_length = NATIVE[name]

    def call(values, logits, offsets):
        return torch.ops.ragged_pool_cuda._forward_study.default(values, logits, offsets, kernel, switch_length, 0)[0]
    return call


def composition_known_valid(values, logits, offsets):
    """The segmented stable softmax of examples/pool_baselines.py, for admitted inputs."""
    groups = offsets.shape[0] - 1
    if values.shape[0] == 0:
        zero = values.sum(dim=0) * 0 + logits.sum() * 0
        return zero.unsqueeze(0).expand(groups, values.shape[1]).clone()
    lengths = offsets[1:] - offsets[:-1]
    indices = torch.repeat_interleave(
        torch.arange(groups, device=offsets.device), lengths, output_size=values.shape[0])
    maxima = torch.segment_reduce(logits.detach(), "max", offsets=offsets)
    unnormalized = (logits - maxima[indices]).exp()
    totals = torch.segment_reduce(unnormalized, "sum", offsets=offsets)
    weights = unnormalized / totals[indices]
    return torch.segment_reduce(values * weights[:, None], "sum", offsets=offsets)


def composition(values, logits, offsets):
    """Check the native admission contract with tensor operations and one host read, then pool."""
    check_metadata(values, logits, offsets)
    count = values.shape[0]
    invalid = (offsets[0] != 0) | (offsets[-1] != count) | (offsets < 0).any() | (offsets > count).any()
    invalid = invalid | (offsets[1:] < offsets[:-1]).any()
    for tensor in (values, logits):
        invalid = invalid | ~torch.isfinite(tensor).all() | (tensor.abs() > 1000).any()
    if invalid.item():
        raise RuntimeError("offsets, values or logits fail the native admission contract")
    return composition_known_valid(values, logits, offsets)


COMPLETE = {"E": native("E"), "N": native("N"), "C": composition}


def ulp_distance(first, second):
    """The largest difference in units of the last place of the larger magnitude."""
    if first.numel() == 0:
        return 0.0
    finfo = torch.finfo(first.dtype)
    a, b = first.cpu().to(torch.float64), second.cpu().to(torch.float64)
    _, exponent = torch.frexp(torch.maximum(a.abs(), b.abs()))
    spacing = torch.clamp(finfo.eps * torch.pow(2.0, (exponent - 1).to(torch.float64)),
                          min=finfo.smallest_normal * finfo.eps)
    return float(((a - b).abs() / spacing).max())


def same_bytes(first, second):
    integer = torch.int32 if first.dtype == torch.float32 else torch.int64
    return first.shape == second.shape and torch.equal(first.contiguous().view(integer), second.contiguous().view(integer))


def largest_difference(first, second):
    return float((first.cpu().to(torch.float64) - second.cpu().to(torch.float64)).abs().max()) if first.numel() else 0.0


def oracle_check(case, name, actual, expected):
    """Stop the run when an output leaves the documented tolerance of float64 math."""
    values = case["cuda"][0]
    if actual.dtype != values.dtype or actual.device != values.device:
        raise AssertionError(f"{case['name']} {name} returned the wrong dtype or device")
    torch.testing.assert_close(actual.cpu(), expected.to(values.dtype), **TOLERANCES[case["dtype"]],
                               msg=lambda text: f"{case['name']} {name}: {text}")


def check_case(case):
    """Compare every candidate with float64 math. Return the record and the reference outputs."""
    values, logits, offsets = case["cuda"]
    expected = oracle(*case["cpu"])
    names = ("W", "K") if case["group"] == "sweep" else ("E", "N", "C")
    outputs, record = {}, {}
    for name in names:
        actual = (COMPLETE.get(name) or native(name))(values, logits, offsets)
        torch.cuda.synchronize()
        oracle_check(case, name, actual, expected)
        outputs[name] = actual
        record[f"{name}_largest_difference_from_float64"] = largest_difference(actual, expected)
    adaptive = "W" if case["group"] == "sweep" else "N"
    # A different result here is a missed prediction, not a stop, as long as it stays in tolerance.
    again = native(adaptive)(values, logits, offsets)
    oracle_check(case, adaptive, again, expected)
    record[f"{adaptive}_repeat_identical_bytes"] = same_bytes(outputs[adaptive], again)
    if "E" in outputs:
        record["E_N_largest_difference"] = largest_difference(outputs["E"], outputs["N"])
        record["E_N_largest_ulps"] = ulp_distance(outputs["E"], outputs["N"])
        record["E_N_identical_bytes"] = same_bytes(outputs["E"], outputs["N"])
    return record, outputs, expected


def quantile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def series_record(microseconds):
    """Round to a tenth of a microsecond first, so the summary can be recomputed from the record."""
    stored = [round(item, 1) for item in microseconds]
    ordered = sorted(stored)
    return {"summary": {"count": len(ordered), "median_us": statistics.median(ordered),
                        "p5_us": quantile(ordered, 0.05), "p95_us": quantile(ordered, 0.95)},
            "samples_us": stored}


def rotated(names, round_index):
    shift = round_index % len(names)
    return names[shift:] + names[:shift]


class Quiet:
    """Inference mode and no garbage collection while timing, as the CPU study's timer had."""

    def __enter__(self):
        self.mode = torch.inference_mode()
        self.mode.__enter__()
        self.collecting = gc.isenabled()
        gc.disable()

    def __exit__(self, *exception):
        if self.collecting:
            gc.enable()
        return self.mode.__exit__(*exception)


def timed_output_matches(case, name, output, reference, expected):
    """True when a timed output has the reference bytes. Outside tolerance stops the run."""
    if same_bytes(output, reference):
        return True
    oracle_check(case, name, output, expected)
    return False


def time_complete(case, names, plan, references, expected):
    """Synchronize, read the clock, make one call, synchronize, read the clock."""
    values, logits, offsets = case["cuda"]
    samples = {name: [] for name in names}
    identical = {name: True for name in names}
    with Quiet():
        for name in names:
            for _ in range(plan["warmup"]):
                COMPLETE[name](values, logits, offsets)
        torch.cuda.synchronize()
        for round_index in range(plan["rounds"]):
            for name in rotated(list(names), round_index):
                call = COMPLETE[name]
                for _ in range(plan["samples"]):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    output = call(values, logits, offsets)
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter_ns() - start) / 1000.0)
                identical[name] &= timed_output_matches(case, name, output, references[name], expected)
                del output
    return {name: {**series_record(series), "last_output_of_each_round_identical_bytes": identical[name]}
            for name, series in samples.items()}


def time_kernel(case, names, plan, references, expected):
    """Event pairs inside _forward_study, one block of launches per candidate and round."""
    values, logits, offsets = case["cuda"]
    samples = {name: [] for name in names}
    blocks = {name: [] for name in names}
    identical = {name: True for name in names}
    with Quiet():
        for name in names:
            kernel, switch_length = NATIVE[name]
            torch.ops.ragged_pool_cuda._forward_study.default(values, logits, offsets, kernel, switch_length, plan["warmup"])
        for round_index in range(plan["rounds"]):
            for name in rotated(list(names), round_index):
                kernel, switch_length = NATIVE[name]
                output, timing = torch.ops.ragged_pool_cuda._forward_study.default(
                    values, logits, offsets, kernel, switch_length, plan["samples"])
                pairs = [item * 1000.0 for item in timing[:-1].tolist()]
                samples[name].extend(pairs)
                blocks[name].append({"span_us": round(float(timing[-1]) * 1000.0, 3),
                                     "sum_of_pairs_us": round(sum(pairs), 3)})
                identical[name] &= timed_output_matches(case, name, output, references[name], expected)
    result = {name: {**series_record(series), "blocks": blocks[name],
                     "output_of_each_block_identical_bytes": identical[name]}
              for name, series in samples.items()}
    if not case["lengths"]:
        for entry in result.values():
            entry["note"] = "No kernel launches for an empty batch. The event pairs enclose no work."
    return result


def judge(chain, medians):
    """Score every pair a chain such as "C < N ~ E" implies.

    Each name in the chain is compared with every name to its right. The pair
    must satisfy "<" when any link between them is "<", and "~" when every
    link between them is "~". So "C < N ~ E" needs C below N and below E, and
    N ~ E ~ C needs all three within 10 percent of each other.
    """
    tokens = chain.split()
    names, links = tokens[0::2], tokens[1::2]
    if any(link not in ("<", "~") for link in links):
        raise ValueError(f"unknown relation in {chain}")
    relations = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            relation = "<" if "<" in links[i:j] else "~"
            a, b = medians[left], medians[right]
            held = a < (1.0 - BAND) * b if relation == "<" else abs(a - b) <= BAND * min(a, b)
            relations.append({"relation": f"{left} {relation} {right}",
                              "ratio": a / b if b else None, "held": held})
    return {"predicted": chain, "relations": relations, "held": all(item["held"] for item in relations)}


def numeric_predictions(entries):
    """The three predictions about the numbers in the Predictions section of FORWARD_KERNEL.md."""
    timed = [entry for entry in entries if entry["group"] != "sweep"]
    repeats = all(entry["correctness"]["N_repeat_identical_bytes"] for entry in timed)
    float32 = max((entry["correctness"]["E_N_largest_ulps"] for entry in timed if entry["dtype"] == "float32"), default=0.0)
    float64 = max((entry["correctness"]["E_N_largest_difference"] for entry in timed if entry["dtype"] == "float64"), default=0.0)
    return [
        {"predicted": "Two runs of N on the same inputs give identical bytes", "measured": repeats, "held": repeats},
        {"predicted": "The largest E and N difference in float32 is at most one unit in the last place",
         "measured_ulps": float32, "held": float32 <= 1.0},
        {"predicted": "The largest E and N difference in float64 stays below 1e-13",
         "measured": float64, "held": float64 < 1e-13},
    ]


def nvidia_smi(fields):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "unavailable: nvidia-smi is not on PATH"
    try:
        completed = subprocess.run([executable, f"--query-gpu={fields}", "--format=csv,noheader"],
                                   capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {type(error).__name__}"
    lines = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not lines:
        return f"unavailable: nvidia-smi exited {completed.returncode} with {len(lines)} lines"
    return lines[0]


DEVICE_STATE = "clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw,pstate"


def utilization_readings():
    """Read the card's utilization five times, one second apart, before any timing."""
    readings = []
    for _ in range(5):
        text = nvidia_smi("utilization.gpu")
        match = re.match(r"(\d+)", text)
        if match is None:
            raise SystemExit(f"nvidia-smi gave no utilization, so the card cannot be shown idle: {text}")
        readings.append(int(match.group(1)))
        time.sleep(1.0)
    return readings


def declared_environment(properties):
    """The limits in the record name this environment, so a run anywhere else stops."""
    found = (sys.platform, torch.__version__, torch.version.cuda, torch.cuda.device_count(),
             (properties.major, properties.minor), properties.name)
    if (sys.platform != "win32" or str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8"
            or torch.cuda.device_count() != 1 or (properties.major, properties.minor) != (8, 9)
            or "RTX 4090" not in properties.name):
        raise AssertionError(f"The study's limits describe one RTX 4090 on Windows with torch 2.9.1+cu128, not {found}")


def installed_identity(script, wheel):
    installed = distribution("ragged-pool-cuda")
    direct = installed.read_text("direct_url.json")
    prefix = Path(sys.prefix).resolve()
    imported = Path(package.__file__).resolve(strict=True)
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise AssertionError("Use an ordinary installed CUDA wheel")
    if (imported != Path(installed.locate_file("ragged_pool_cuda/__init__.py")).resolve(strict=True)
            or not imported.is_relative_to(prefix) or installed.version != package.__version__):
        raise AssertionError("The CUDA import must match this environment's distribution")
    libraries = list(imported.parent.glob("_C*.pyd")) + list(imported.parent.glob("_C*.so"))
    recorded = {Path(installed.locate_file(path)).resolve() for path in installed.files or ()}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    if len(libraries) != 1 or libraries[0].resolve() not in recorded or libraries[0].resolve() not in loaded:
        raise AssertionError("Expected one recorded and loaded installed CUDA native library")
    library_sha256 = sha256(libraries[0].read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if re.fullmatch(r"ragged_pool_cuda/_C[^/]*\.(pyd|so)", name)]
        if len(members) != 1 or sha256(archive.read(members[0])).hexdigest() != library_sha256:
            raise AssertionError("The --wheel file does not hold the installed native library")
    if any((parent / "pyproject.toml").is_file() for parent in script.parents):
        raise AssertionError("Copy the study outside every source checkout")
    # Local paths stay out of the record. The checks above ran against them.
    return {"installed_wheel_not_editable": True, "import_matches_distribution": True,
            "native_library_recorded_and_loaded": True, "wheel_holds_installed_library": True,
            "package_version": package.__version__, "native_library": libraries[0].name,
            "native_library_sha256": library_sha256,
            "script_sha256": sha256(script.read_bytes()).hexdigest()}


def run(arguments):
    started = time.perf_counter()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    plan = SMOKE if arguments.smoke else FULL
    script = Path(__file__).resolve()
    identity = installed_identity(script, arguments.wheel)
    if not torch.cuda.is_available():
        raise AssertionError("An unavailable CUDA device is not a skipped pass")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    declared_environment(properties)
    readings = utilization_readings()
    if max(readings) > arguments.busy_limit:
        raise SystemExit(f"The card is busy: utilization readings {readings} reach above {arguments.busy_limit} percent")
    record = {
        "schema": "ragged-pool-cuda forward kernel study 1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_timezone": "UTC",
        "description": "Kernel only and complete call times of the existing and adaptive forward kernels and the "
                       "eager composition on one card. One machine, one driver, one toolkit, bounded sizes. "
                       "No general speed claim.",
        "limits": "One RTX 4090 with unlocked clocks in one Windows machine, the CUDA 12.8.1 toolkit and the sizes "
                  "N<=4096, 1<=D<=128 and 0<=B<=128. Backward kernels are unchanged and untimed. Times are from "
                  "event pairs and from the host clock as the protocol in FORWARD_KERNEL.md describes.",
        "smoke": bool(arguments.smoke),
        "source_revision": arguments.source_revision,
        "wheel": {"name": arguments.wheel.name, "sha256": sha256(arguments.wheel.read_bytes()).hexdigest()},
        "identity": identity,
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda, "device_count": torch.cuda.device_count(),
            "device_name": properties.name, "capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count, "total_memory_bytes": properties.total_memory,
            "driver": nvidia_smi("name,driver_version"),
            "utilization_before_start_percent": readings, "busy_limit_percent": arguments.busy_limit,
            "nsight_compute": ("available, but this script runs no profile" if shutil.which("ncu")
                               else "unavailable: ncu is not on PATH"),
        },
        "protocol": {
            "seed": SEED, "draw": "values, logits and upstream for every case with randn and clamp to [-6,6] "
                                  "where the study clamps, in the order cpu study, bound, sweep. upstream is unused.",
            **plan, "band": BAND, "tolerances": TOLERANCES, "default_switch": DEFAULT_SWITCH,
            "complete_call": "torch.cuda.synchronize, perf_counter_ns, one call, torch.cuda.synchronize, perf_counter_ns",
            "kernel_only": "CUDA event pairs inside _forward_study behind a stream hold, one block per candidate and round",
            "while_timing": "torch.inference_mode, garbage collection off, one intraop and one interop thread",
            "candidates": {"E": "kernel 0, switch 32", "N": "kernel 1, switch 32", "C": "eager composition",
                           "W": "kernel 1, switch 4096", "K": "kernel 1, switch 0"},
            "relations": {"<": "left median more than 10 percent below the right",
                          "~": "medians within 10 percent of the smaller",
                          "chains": "every pair a chain implies is judged, see judge in the script"},
        },
        "cases": [],
    }
    cases = make_cases((("cpu study", CPU_CASES), ("bound", BOUND_CASES), ("sweep", SWEEP_CASES)), device)
    for case in cases:
        if case["group"] == "cpu study" and case["data_sha256"] != CPU_STUDY_DATA_SHA256[case["name"]]:
            raise AssertionError(f"{case['name']} does not draw the CPU study's data")
    # Correctness first, for every case, before any timing.
    with Quiet():
        checks = {case["name"]: check_case(case) for case in cases}
    for case in cases:
        correctness, references, expected = checks[case["name"]]
        entry = {"name": case["name"], "group": case["group"], "B": len(case["lengths"]),
                 "N": sum(case["lengths"]), "D": case["width"], "dtype": case["dtype"],
                 "longest_group": max(case["lengths"], default=0), "lengths": case["lengths"],
                 "data_sha256": case["data_sha256"], "correctness": correctness}
        if case["group"] == "cpu study":
            entry["matches_cpu_study_data"] = True
        entry["device_before"] = nvidia_smi(DEVICE_STATE)
        if case["group"] == "sweep":
            kernel = time_kernel(case, ["W", "K"], plan, references, expected)
            entry["kernel_only"] = kernel
            entry["prediction"] = {"kernel_only": judge(SWEEP_PREDICTIONS[case["name"]],
                                                       {n: s["summary"]["median_us"] for n, s in kernel.items()})}
        else:
            kernel = time_kernel(case, ["E", "N"], plan, references, expected)
            complete = time_complete(case, ["E", "N", "C"], plan, references, expected)
            entry["kernel_only"], entry["complete_call"] = kernel, complete
            kernel_chain, complete_chain = PREDICTIONS[case["name"]]
            entry["prediction"] = {
                "kernel_only": None if kernel_chain is None else judge(
                    kernel_chain, {n: s["summary"]["median_us"] for n, s in kernel.items()}),
                "complete_call": judge(complete_chain, {n: s["summary"]["median_us"] for n, s in complete.items()}),
            }
        entry["device_after"] = nvidia_smi(DEVICE_STATE)
        # The card's copies of the inputs must still be the drawn data.
        entry["inputs_unchanged_after_timing"] = all(
            torch.equal(on_card.cpu(), drawn) for on_card, drawn in zip(case["cuda"], case["cpu"]))
        if not entry["inputs_unchanged_after_timing"]:
            raise AssertionError(f"{case['name']} inputs changed on the card during timing")
        record["cases"].append(entry)
    record["numeric_predictions"] = numeric_predictions(record["cases"])
    judged = [p for case in record["cases"] for p in case["prediction"].values() if p is not None]
    judged += record["numeric_predictions"]
    record["predictions_held"] = sum(p["held"] for p in judged)
    record["predictions_total"] = len(judged)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return record


def main():
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON record path")
    parser.add_argument("--source-revision", required=True, help="Commit the wheel was built from")
    parser.add_argument("--wheel", type=Path, required=True, help="The installed wheel file, checked against the library")
    parser.add_argument("--smoke", action="store_true", help="Few samples, to check the script, not a study record")
    parser.add_argument("--busy-limit", type=int, default=30,
                        help="Refuse to time when any of five utilization readings exceeds this percent")
    arguments = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.source_revision):
        parser.error("The source revision must be a complete lowercase SHA1 identifier")
    if arguments.output.exists() or arguments.output.is_symlink():
        parser.error("The output record must be new")
    if not arguments.wheel.is_file():
        parser.error("The wheel must be an existing file")
    record = run(arguments)
    text = json.dumps(record, indent=1, allow_nan=False) + "\n"
    # Each list of numbers goes on one line, so the samples do not take one line each.
    # Only a list that json.dumps opened with a line break matches, never text inside a string.
    text = re.sub(r"\[\n\s*(-?[0-9][0-9.eE+-]*(?:,\n\s*-?[0-9][0-9.eE+-]*)*)\n\s*\]",
                  lambda match: "[" + ", ".join(part.strip() for part in match.group(1).split(",")) + "]", text)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(f"STUDY {'SMOKE ' if arguments.smoke else ''}PASSED, {len(record['cases'])} cases, "
          f"{record['predictions_held']} of {record['predictions_total']} predictions held, wrote {arguments.output}")


if __name__ == "__main__":
    main()
