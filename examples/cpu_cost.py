"""Measure one fixed, bounded CPU pooling workload using an installed wheel.

Run timing, CPU profiling, and optional compiler probes separately. Results keep
raw samples, scope, failures, package identity, and partial progress in JSON.
"""
import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback

# A compiler child waits before importing torch. Its parent first assigns the
# owned Windows job or Linux process group, so no compiler can escape that scope
# by starting before process ownership is established.
_start_gate = os.environ.get("RAGGED_POOL_COMPILER_START_GATE")
if _start_gate:
    _gate_deadline = time.perf_counter() + 10
    while not Path(_start_gate).is_file():
        if time.perf_counter() > _gate_deadline:
            raise TimeoutError("Compiler parent did not establish its owned process scope")
        time.sleep(0.01)

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch.utils.benchmark import Timer

import ragged_pool as package
from ragged_pool import ragged_pool, ragged_pool_reference


def adjacent_module(name):
    """Load a copied example helper without adding the checkout to sys.path."""
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("ragged_study_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = adjacent_module("pool_baselines")
CANDIDATES = {
    "native": ragged_pool,
    "reference": ragged_pool_reference,
    "builtin_checked": baseline.ragged_pool_builtin,
}
CASES = (
    ("empty_batch", [], 8, "float32"),
    ("empty_groups", [0, 0, 0, 0], 8, "float64"),
    ("singletons", [1] * 128, 8, "float32"),
    ("many_short", [4] * 128, 16, "float32"),
    ("balanced", [32] * 32, 32, "float32"),
    ("skewed", [0] + [1] * 14 + [1000], 32, "float32"),
    ("one_long", [4096], 16, "float64"),
    ("wide", [64] * 8, 128, "float64"),
)
TOLERANCES = {
    "float32": {"rtol": 2e-5, "atol": 2e-6},
    "float64": {"rtol": 1e-10, "atol": 1e-12},
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_tensors(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def current_rss():
    """Sample this worker's RSS, not an OS-enforced memory or descendant cap."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    if sys.platform.startswith("linux"):
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    raise RuntimeError("RSS sampling is supported only on the declared Windows/Linux paths")


def checkpoint(path, receipt):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def guard(receipt, started, seconds):
    sample = {"elapsed_seconds": time.perf_counter() - started, "rss_bytes": current_rss()}
    receipt.setdefault("resource_samples", []).append(sample)
    if sample["rss_bytes"] > 4 * 1024 ** 3:
        raise RuntimeError("Sampled worker RSS exceeded the 4-GiB stopping threshold")
    if sample["elapsed_seconds"] > seconds:
        raise TimeoutError(f"Declared {seconds}-second worker budget exceeded")


def environment(arguments):
    libraries = list(Path(package.__file__).parent.glob("_C*.pyd")) + list(
        Path(package.__file__).parent.glob("_C*.so")
    )
    if len(libraries) != 1:
        raise RuntimeError("Expected exactly one installed native library")
    wheel = Path(arguments.wheel).resolve() if arguments.wheel else None
    return {
        "source_revision": arguments.source_revision,
        "script_sha256": sha256_file(__file__),
        "baseline_script_sha256": sha256_file(Path(__file__).with_name("pool_baselines.py")),
        "consumer_script_sha256": sha256_file(Path(__file__).with_name("learn_sets.py")),
        "wheel_path": str(wheel) if wheel else None,
        "wheel_sha256": sha256_file(wheel) if wheel else None,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_config": torch.__config__.show(),
        "package_version": package.__version__,
        "package_path": package.__file__,
        "native_library_path": str(libraries[0]),
        "native_library_sha256": sha256_file(libraries[0]),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "perf_counter_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        "memory_limit_scope": "Sampled worker RSS at bounded checkpoints, not a hard OS limit",
    }


def make_cases():
    generator = torch.Generator(device="cpu").manual_seed(72001)
    cases = []
    for name, lengths, width, dtype_name in CASES:
        dtype = getattr(torch, dtype_name)
        offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
        values = torch.randn(sum(lengths), width, dtype=dtype, generator=generator).clamp_(-6, 6)
        logits = torch.randn(sum(lengths), dtype=dtype, generator=generator).clamp_(-6, 6)
        upstream = torch.randn(len(lengths), width, dtype=dtype, generator=generator)
        cases.append({
            "name": name, "lengths": lengths, "width": width, "dtype": dtype_name,
            "values": values, "logits": logits, "offsets": offsets, "upstream": upstream,
            "data_sha256": digest_tensors(values, logits, offsets, upstream),
        })
    return cases


def delta(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def evaluate(pool, case):
    values = case["values"].clone().requires_grad_()
    logits = case["logits"].clone().requires_grad_()
    output = pool(values, logits, case["offsets"])
    gradients = torch.autograd.grad(output, (values, logits), case["upstream"])
    return output.detach(), *(item.detach() for item in gradients)


def check_case(case):
    reference = evaluate(ragged_pool_reference, case)
    records = {}
    for name, pool in CANDIDATES.items():
        result = evaluate(pool, case)
        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual, expected, **TOLERANCES[case["dtype"]])
        records[name] = dict(zip(
            ("output_max_abs_difference", "values_gradient_max_abs_difference", "logits_gradient_max_abs_difference"),
            (delta(a, b) for a, b in zip(result, reference)),
        ))
    return records


def summary(seconds):
    ordered = sorted(seconds)

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "sample_count": len(seconds), "median_seconds": statistics.median(seconds),
        "iqr_seconds": quantile(0.75) - quantile(0.25),
    }


def time_call(call):
    for _ in range(3):
        call()
    measurement = Timer(stmt="call()", globals={"call": call}, num_threads=1).blocked_autorange(min_run_time=0.2)
    return {
        "raw_block_seconds": list(measurement.raw_times),
        "calls_per_block": measurement.number_per_run,
        "seconds_per_call": list(measurement.times),
    }


def run_timing(arguments, receipt, output, started):
    receipt["scope"] = {
        "primary": "Complete checked pooling call, including every admission and temporary construction",
        "arithmetic": "Builtin float32 uses float32 accumulation, native float32 uses double accumulation",
        "seed": 72001, "rounds": 3, "timer": "torch.utils.benchmark.Timer, num_threads=1",
        "replicate_window_seconds": 0.2, "forward_mode": "torch.inference_mode",
        "vjp": "Fresh forward graph and autograd.grad per call, fixed leaf inputs and upstream, no .grad accumulation",
        "noise": "Shared-machine measurements with rotated order, nested blocks are not independent experiments",
    }
    receipt["cases"] = []
    for case in make_cases():
        guard(receipt, started, arguments.budget_seconds)
        record = {key: value for key, value in case.items() if not isinstance(value, torch.Tensor)}
        record["correctness"] = check_case(case)
        record["measurements"] = {mode: {name: [] for name in CANDIDATES} for mode in ("inference", "forward_vjp")}
        receipt["cases"].append(record)
        values = case["values"].clone().requires_grad_()
        logits = case["logits"].clone().requires_grad_()
        for mode in record["measurements"]:
            for round_index in range(3):
                names = list(CANDIDATES)
                order = names[round_index:] + names[:round_index]
                for name in order:
                    guard(receipt, started, arguments.budget_seconds)
                    pool = CANDIDATES[name]
                    if mode == "inference":
                        def call(pool=pool):
                            with torch.inference_mode():
                                return pool(case["values"], case["logits"], case["offsets"])
                    else:
                        def call(pool=pool):
                            result = pool(values, logits, case["offsets"])
                            return torch.autograd.grad(result, (values, logits), case["upstream"])
                    sample = time_call(call)
                    sample.update(round=round_index, candidate_order=order)
                    record["measurements"][mode][name].append(sample)
                    checkpoint(output, receipt)
            for name, rounds in record["measurements"][mode].items():
                record.setdefault("summary", {}).setdefault(mode, {})[name] = summary(
                    [sample for window in rounds for sample in window["seconds_per_call"]]
                )
        checkpoint(output, receipt)
    run_full_step(arguments, receipt, output, started)


def run_full_step(arguments, receipt, output, started):
    study = adjacent_module("learn_sets")
    split = study.make_split(71001, 512, 1, 16)
    order = study.batch_order(11, 512)
    values, offsets, labels = study.pack_batch(split, order[:16])
    canonical = study.make_model(11, "reference")
    canonical_optimizer = torch.optim.Adam(canonical.parameters(), lr=0.003)
    canonical_optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(canonical(values, offsets), labels)
    loss.backward()
    canonical_optimizer.step()
    model_state = copy.deepcopy(canonical.state_dict())
    optimizer_state = copy.deepcopy(canonical_optimizer.state_dict())
    models = {name: study.make_model(11, "reference") for name in CANDIDATES}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=0.003) for name, model in models.items()}

    def restore(name):
        models[name].load_state_dict(model_state)
        optimizers[name].load_state_dict(copy.deepcopy(optimizer_state))

    def step(name, capture=False):
        optimizer = optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        predictions = models[name](values, offsets, pool_override=CANDIDATES[name])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels)
        loss.backward()
        optimizer.step()
        if capture:
            return predictions.detach(), loss.detach()

    parity = {}
    saved = {}
    for name in ("reference", "native", "builtin_checked"):
        restore(name)
        predictions, loss = step(name, capture=True)
        saved[name] = (
            predictions, loss,
            {key: parameter.grad.detach().clone() for key, parameter in models[name].named_parameters()},
            copy.deepcopy(models[name].state_dict()),
        )
    for name, actual in saved.items():
        expected = saved["reference"]
        torch.testing.assert_close(actual[0], expected[0], **TOLERANCES["float32"])
        torch.testing.assert_close(actual[1], expected[1], **TOLERANCES["float32"])
        for collection in (2, 3):
            for key in actual[collection]:
                torch.testing.assert_close(actual[collection][key], expected[collection][key], **TOLERANCES["float32"])
        parity[name] = {
            "predictions_max_abs_difference": delta(actual[0], expected[0]),
            "parameter_gradients_max_abs_difference": max(delta(actual[2][key], expected[2][key]) for key in actual[2]),
            "post_step_parameters_max_abs_difference": max(delta(actual[3][key], expected[3][key]) for key in actual[3]),
        }
    record = {
        "seed": 11, "data_sha256": digest_tensors(values, offsets, labels),
        "batch_indices": [int(index) for index in order[:16]],
        "scope": "zero_grad(set_to_none=True), encoder/scorer/pool/head, BCE, backward, Adam.step",
        "excluded": "Generation, packing, canonical reference warmup, model and Adam state restoration, compiler setup",
        "state": "One canonical reference warmup step, identical model and initialized Adam snapshot restored before every timed step",
        "correctness": parity, "rounds": [],
    }
    receipt["full_step"] = record
    for round_index in range(3):
        names = list(CANDIDATES)
        order = names[round_index:] + names[:round_index]
        for name in order:
            times = []
            for _ in range(10):
                guard(receipt, started, arguments.budget_seconds)
                restore(name)
                before = time.perf_counter()
                step(name)
                times.append(time.perf_counter() - before)
            record["rounds"].append({"round": round_index, "candidate": name, "candidate_order": order, "raw_step_seconds": times})
            checkpoint(output, receipt)
    record["summary"] = {
        name: summary([sample for row in record["rounds"] if row["candidate"] == name for sample in row["raw_step_seconds"]])
        for name in CANDIDATES
    }


def run_profile(arguments, receipt, output, started):
    case = next(case for case in make_cases() if case["name"] == "balanced")
    receipt["workload_sha256"] = case["data_sha256"]
    receipt["phase"] = "candidate_correctness"
    receipt["correctness"] = check_case(case)
    receipt["scope"] = "One separate CPU-only profile of checked forward plus VJP per candidate, excluded from latency tables"
    receipt["limitation"] = "An opaque custom-op event does not locate an internal C++ allocation, loop, or exponentiation bottleneck"
    receipt["profiles"] = {}
    # No tensorboard handler, trace viewer, stack capture, or visible application.
    for name, pool in CANDIDATES.items():
        guard(receipt, started, arguments.budget_seconds)
        for _ in range(3):
            evaluate(pool, case)
        values = case["values"].clone().requires_grad_()
        logits = case["logits"].clone().requires_grad_()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False, with_stack=False) as profile:
            with torch.profiler.record_function("checked_pool_forward_vjp"):
                result = pool(values, logits, case["offsets"])
                torch.autograd.grad(result, (values, logits), case["upstream"])
        receipt["profiles"][name] = [
            {"key": event.key, "count": event.count, "self_cpu_time_us": event.self_cpu_time_total, "cpu_time_us": event.cpu_time_total}
            for event in sorted(profile.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)[:25]
        ]
        checkpoint(output, receipt)


def compile_child(arguments, receipt, output, started):
    case = next(case for case in make_cases() if case["name"] == "balanced")
    pool = ragged_pool if arguments.compile_child == "native" else baseline.ragged_pool_builtin_known_valid
    receipt["candidate"] = arguments.compile_child
    receipt["scope"] = "Native opaque operation includes native admission" if arguments.compile_child == "native" else "Already-admitted numerical composition, excludes Python admission"
    receipt["workload_sha256"] = case["data_sha256"]
    receipt["phase"] = "candidate_correctness"
    receipt["correctness"] = check_case(case)
    values = case["values"].clone().requires_grad_()
    logits = case["logits"].clone().requires_grad_()
    expected = evaluate(pool, case)
    guard(receipt, started, arguments.budget_seconds)
    receipt["phase"] = "compile_forward_and_backward"
    checkpoint(output, receipt)
    before = time.perf_counter()
    compiled = torch.compile(pool, backend="inductor", fullgraph=True, dynamic=False)
    result = compiled(values, logits, case["offsets"])
    gradients = torch.autograd.grad(result, (values, logits), case["upstream"])
    receipt["first_forward_backward_seconds"] = time.perf_counter() - before
    for actual, reference in zip((result, *gradients), expected):
        torch.testing.assert_close(actual, reference, **TOLERANCES["float32"])
    receipt["compiled_differences"] = [delta(a, b) for a, b in zip((result, *gradients), expected)]
    guard(receipt, started, arguments.budget_seconds)
    receipt["phase"] = "warm_forward_and_backward"

    def call():
        result = compiled(values, logits, case["offsets"])
        return torch.autograd.grad(result, (values, logits), case["upstream"])

    receipt["warm_measurement"] = time_call(call)
    receipt["warm_summary"] = summary(receipt["warm_measurement"]["seconds_per_call"])
    guard(receipt, started, arguments.budget_seconds)


def stop_child(child):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=15)


def windows_compiler_job(child):
    """Keep compiler descendants in an owned job that dies when its handle closes."""
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_ulonglong * 6)] + [
            (name, ctypes.c_size_t) for name in (
                "ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    # CPython's Windows Popen handle belongs to the worker we just launched.
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(child._handle))):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    return lambda: kernel.CloseHandle(job)


def run_compile(arguments, receipt, output, started):
    receipt["scope"] = "At most two isolated fullgraph Inductor forward/backward attempts, dynamic=False, one compilation worker, 120 seconds each"
    receipt["attempts"] = []
    for candidate in ("native", "builtin_known_valid"):
        guard(receipt, started, arguments.budget_seconds)
        child_output = output.with_name(output.stem + "-" + candidate + ".json")
        stdout_path = output.with_name(output.stem + "-" + candidate + ".stdout.log")
        stderr_path = output.with_name(output.stem + "-" + candidate + ".stderr.log")
        cache = output.with_name(output.stem + "-" + candidate + "-cache")
        gate = output.with_name(output.stem + "-" + candidate + ".ready")
        if any(path.exists() for path in (child_output, stdout_path, stderr_path, cache, gate)):
            raise FileExistsError("Compiler evidence/cache destination already exists")
        environment_variables = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cache), TORCHINDUCTOR_COMPILE_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAGGED_POOL_COMPILER_START_GATE=str(gate))
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "--compile-child", candidate, "--output", str(child_output), "--source-revision", arguments.source_revision, "--budget-seconds", "120"]
        if arguments.wheel:
            command += ["--wheel", arguments.wheel]
        lifecycle = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
        row = {"candidate": candidate, "receipt": str(child_output), "stdout": str(stdout_path), "stderr": str(stderr_path), "status": "running", "timeout_seconds": 120}
        receipt["attempts"].append(row)
        checkpoint(output, receipt)
        with stdout_path.open("x", encoding="utf-8") as standard_output, stderr_path.open("x", encoding="utf-8") as standard_error:
            child = subprocess.Popen(command, stdout=standard_output, stderr=standard_error, env=environment_variables, **lifecycle)
            close_job = None
            try:
                if sys.platform == "win32":
                    close_job = windows_compiler_job(child)
                with gate.open("x", encoding="utf-8") as stream:
                    stream.write("Owned process scope established\n")
                row["returncode"] = child.wait(timeout=min(120, max(1, arguments.budget_seconds - (time.perf_counter() - started))))
                row["status"] = "completed" if row["returncode"] == 0 else "failed"
            except subprocess.TimeoutExpired:
                stop_child(child)
                row.update(status="timed_out", returncode=child.returncode)
            except BaseException:
                if child.poll() is None:
                    stop_child(child)
                raise
            finally:
                if close_job is not None:
                    close_job()
                    row["descendant_cleanup"] = "Closed owned Windows job with KILL_ON_JOB_CLOSE, on normal exit or timeout"
                elif sys.platform != "win32":
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    row["descendant_cleanup"] = "Terminated any remaining members of the owned process group, on normal exit or timeout"
        if child_output.exists():
            row["result"] = json.loads(child_output.read_text(encoding="utf-8"))
        checkpoint(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--timing", action="store_true")
    modes.add_argument("--profile", action="store_true")
    modes.add_argument("--compile", action="store_true")
    modes.add_argument("--compile-child", choices=("native", "builtin_known_valid"), help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--budget-seconds", type=float, default=300)
    arguments = parser.parse_args()
    if not 0 < arguments.budget_seconds <= 300:
        parser.error("--budget-seconds must be in (0,300]")
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError("Create the evidence directory before running this example")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    receipt = {
        "schema": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(arguments), "budget_seconds": arguments.budget_seconds,
    }
    checkpoint(output, receipt)
    try:
        if arguments.timing:
            run_timing(arguments, receipt, output, started)
        elif arguments.profile:
            run_profile(arguments, receipt, output, started)
        elif arguments.compile_child:
            compile_child(arguments, receipt, output, started)
        else:
            run_compile(arguments, receipt, output, started)
        guard(receipt, started, arguments.budget_seconds)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="partial" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        checkpoint(output, receipt)


if __name__ == "__main__":
    main()
