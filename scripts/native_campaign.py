"""Run bounded generated cases against an installed, instrumented CPU extension."""
import argparse
from collections import Counter
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch
import ragged_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--memory-mib", type=int, default=4096)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600 or not 512 <= args.memory_mib <= 4096:
        parser.error("seconds must be in [1,600] and memory-mib in [512,4096]")
    if args.output.exists():
        parser.error("refusing to overwrite a campaign receipt")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    module_path = Path(ragged_pool.__file__).resolve()
    libraries = list(module_path.parent.glob("_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError("Expected one installed Linux extension")
    library = libraries[0].resolve()
    if str(library) not in {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}:
        raise RuntimeError("The identified native library was not loaded by the dispatcher")
    symbols = subprocess.check_output(["nm", "-D", str(library)], text=True)
    if "__asan_init" not in symbols or "__asan_report_load" not in symbols:
        raise RuntimeError("The loaded extension has no expected ASan instrumentation symbols")
    # The preload must provide the active runtime in this very Python process.
    is_poisoned = ctypes.CDLL(None).__asan_address_is_poisoned
    is_poisoned.argtypes = [ctypes.c_void_p]
    is_poisoned.restype = ctypes.c_int
    probe = ctypes.create_string_buffer(1)
    if is_poisoned(ctypes.addressof(probe)) != 0:
        raise RuntimeError("ASan runtime activation probe failed")
    spec = importlib.util.spec_from_file_location("generated_cases", args.cases.resolve())
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    counters = Counter()
    started = time.monotonic()
    receipt = {
        "status": "running", "seed": args.seed, "requested_seconds": args.seconds,
        "python": sys.version, "torch": torch.__version__, "package": str(module_path),
        "native_library": str(library),
        "native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_revision": os.environ.get("RAGGED_REVISION", "unrecorded"),
        "asan_runtime": os.environ.get("ASAN_RUNTIME", "unrecorded"),
        "asan_options": os.environ.get("ASAN_OPTIONS", "unrecorded"),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "instrumentation": "ASan symbols in loaded extension and runtime activation probe",
        "limits": {"workers": 1, "torch_threads": 1, "rss_mib": args.memory_mib,
                   "rss_check": "sampled between bounded cases; not a hard OS memory cap"},
        "coverage": "generated property/admission categories; no edge coverage measurement",
        "iterations": 0, "counters": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["counters"] = dict(counters)
        receipt["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    try:
        while time.monotonic() - started < args.seconds:
            seed = args.seed + receipt["iterations"]
            receipt["current_seed"] = seed
            # Preserve the exact in-flight seed even if ASan terminates Python.
            args.output.with_suffix(".seed").write_text(str(seed) + "\n", encoding="ascii")
            result = cases.run_case(seed)
            counters.update(result["counters"])
            receipt["iterations"] += 1
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > args.memory_mib * 1024:
                raise RuntimeError("Campaign exceeded the sampled resident-memory budget")
            if receipt["iterations"] % 100 == 0:
                checkpoint()
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        checkpoint()
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
