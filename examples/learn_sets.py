"""Run the fixed synthetic presence study through an installed ragged-pool wheel.

This task checks model consumption and simple alternatives. It is not a paper
reproduction or evidence of real-world accuracy. Run outside the source checkout
for installed-wheel acceptance. The JSON receipt is never silently overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn

import ragged_pool as package
from ragged_pool import ragged_pool, ragged_pool_reference


SPLIT_SPECS = {
    "train": (71001, 512, 1, 16),
    "validation": (71002, 128, 1, 16),
    "test": (71003, 256, 1, 16),
    "longer_test": (71004, 256, 17, 32),
}
MODEL_SEEDS = (11, 29, 47)
POOLINGS = ("native", "reference", "mean", "max", "sum")
TOLERANCE = {"rtol": 2e-5, "atol": 2e-6}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_record():
    """Return the fixed protocol without generating data or running a model."""
    return {
        "protocol": "ragged-pool-n2-presence-v1",
        "task": "Balanced latent presence, one signal in each positive bag",
        "features": 4,
        "nuisance": "independent standard normal, clipped to [-6,6]",
        "signal": "normal mean [3,0,0,0], unit covariance, clipped to [-6,6]",
        "construction": "one positive and negative per sampled length, independent contents, shuffled bags",
        "splits": {name: dict(zip(("seed", "bags", "min_length", "max_length"), spec))
                   for name, spec in SPLIT_SPECS.items()},
        "model_seeds": list(MODEL_SEEDS),
        "poolings": list(POOLINGS),
        "architecture": "Linear(4,16), tanh, pool, Linear(16,1)",
        "attention_scorer": "Linear(16,1,bias=False)",
        "shared_initialization": "encoder then head then optional scorer, same seed",
        "batch_order": "CPU torch.Generator seed 81000+model_seed, randperm each epoch",
        "dtype": "float32",
        "device": "cpu",
        "optimizer": {"name": "Adam", "learning_rate": 0.003, "default_other_options": True},
        "batch_size": 16,
        "updates": 200,
        "validation_interval": 20,
        "checkpoint": "minimum validation BCE, earliest update breaks ties",
        "metrics": ["binary_cross_entropy_with_logits", "accuracy at probability >=0.5"],
        "permutation": "reverse each held-out bag, preserve bag order",
        "full_run_budget_seconds": 900,
        "sampled_rss_stop_bytes": 4 * 1024**3,
        "threads": 1,
        "limitations": [
            "Three initialization seeds support descriptive comparisons only",
            "Longer bags also dilute a single signal among more nuisance items",
            "Attention coefficients are not causal explanations",
        ],
    }


def _tensor_bytes(tensor):
    # CPU contiguous numpy conversion preserves the actual generated float32 bytes.
    return tensor.detach().contiguous().numpy().tobytes()


def make_split(seed, count, min_length, max_length):
    """Generate label-paired lengths and independently sampled bag contents."""
    if count <= 0 or count % 2 or min_length < 1 or max_length < min_length:
        raise ValueError("Use an even positive count and nonempty length range")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bags, labels, lengths = [], [], []
    for _ in range(count // 2):
        length = int(torch.randint(min_length, max_length + 1, (), generator=generator))
        for label in (0, 1):
            bag = torch.randn(length, 4, generator=generator, dtype=torch.float32)
            if label:
                position = int(torch.randint(length, (), generator=generator))
                signal = torch.randn(4, generator=generator, dtype=torch.float32)
                signal[0] += 3.0
                bag[position] = signal
            bags.append(bag.clamp(-6, 6).contiguous())
            labels.append(float(label))
            lengths.append(length)
    order = torch.randperm(count, generator=generator).tolist()
    bags = [bags[index] for index in order]
    labels = torch.tensor([labels[index] for index in order], dtype=torch.float32)
    lengths = [lengths[index] for index in order]
    bag_ids = [hashlib.sha256(_tensor_bytes(bag)).hexdigest() for bag in bags]
    instance_ids = {hashlib.sha256(_tensor_bytes(row)).hexdigest()
                    for bag in bags for row in bag}
    histogram = {}
    for label in (0, 1):
        histogram[str(label)] = {
            str(length): sum(actual == length and int(target) == label
                             for actual, target in zip(lengths, labels.tolist()))
            for length in range(min_length, max_length + 1)
        }
    if histogram["0"] != histogram["1"]:
        raise AssertionError("The two classes must have identical length histograms")
    digest = hashlib.sha256()
    for bag, label in zip(bags, labels.tolist()):
        digest.update(len(bag).to_bytes(8, "little"))
        digest.update(bytes((int(label),)))
        digest.update(_tensor_bytes(bag))
    return {"bags": bags, "labels": labels, "lengths": lengths,
            "bag_ids": bag_ids, "instance_ids": instance_ids,
            "digest": digest.hexdigest(), "seed": seed,
            "histograms": histogram}


def pack_batch(split, indices):
    """Pack selected bags in their requested order, before timing or training."""
    indices = list(indices)
    lengths = [split["lengths"][index] for index in indices]
    offsets = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
    values = torch.cat([split["bags"][index] for index in indices], dim=0)
    return values.contiguous(), offsets.contiguous(), split["labels"][indices].contiguous()


def batch_order(seed, count=512, batch_size=16):
    """Return the first epoch permutation used by every pooling variant."""
    if count % batch_size:
        raise ValueError("This fixed study uses complete minibatches")
    generator = torch.Generator(device="cpu").manual_seed(81000 + seed)
    return torch.randperm(count, generator=generator).tolist()


def training_batches(seed, count=512, updates=200, batch_size=16):
    if count % batch_size:
        raise ValueError("This fixed study uses complete minibatches")
    generator = torch.Generator(device="cpu").manual_seed(81000 + seed)
    result = []
    while len(result) < updates:
        order = torch.randperm(count, generator=generator).tolist()
        result.extend(order[start:start + batch_size] for start in range(0, count, batch_size))
    return result[:updates]


class SetClassifier(nn.Module):
    def __init__(self, pooling):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"Unknown pooling {pooling}")
        self.pooling = pooling
        # Construct common parameters first so optional attention has no effect
        # on the shared encoder/head initialization.
        self.encoder = nn.Linear(4, 16)
        self.head = nn.Linear(16, 1)
        if pooling in ("native", "reference"):
            self.scorer = nn.Linear(16, 1, bias=False)

    def forward(self, values, offsets, pool_override=None):
        encoded = torch.tanh(self.encoder(values)).contiguous()
        if self.pooling in ("native", "reference"):
            logits = self.scorer(encoded).squeeze(-1).contiguous()
            pool = pool_override or (ragged_pool if self.pooling == "native" else ragged_pool_reference)
            pooled = pool(encoded, logits, offsets)
        else:
            boundaries = offsets.tolist()
            groups = [encoded[start:end] for start, end in zip(boundaries, boundaries[1:])]
            if any(len(group) == 0 for group in groups):
                raise ValueError("This classification task excludes empty bags")
            if self.pooling == "max":
                pooled = torch.stack([group.max(0).values for group in groups])
            elif self.pooling == "mean":
                pooled = torch.stack([group.mean(0) for group in groups])
            else:
                pooled = torch.stack([group.sum(0) for group in groups])
        return self.head(pooled).squeeze(-1)


def make_model(seed, pooling):
    # Helpers can be imported by tests without changing the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SetClassifier(pooling)


def maximum_difference(first, second):
    return float((first.detach() - second.detach()).abs().max()) if first.numel() else 0.0


def _state_difference(first, second):
    return {name: maximum_difference(first[name], second[name]) for name in first}


def _copy_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def same_start_parity(split, seed=11):
    """Check the operator's two derivatives and one full Adam update separately."""
    values, offsets, labels = pack_batch(split, training_batches(seed)[0])
    native, reference = make_model(seed, "native"), make_model(seed, "reference")
    torch.testing.assert_close(native.state_dict(), reference.state_dict(), rtol=0, atol=0)
    with torch.no_grad():
        encoded = torch.tanh(native.encoder(values)).contiguous()
        logits = native.scorer(encoded).squeeze(-1).contiguous()
    nvalues, rvalues = encoded.clone().requires_grad_(), encoded.clone().requires_grad_()
    nlogits, rlogits = logits.clone().requires_grad_(), logits.clone().requires_grad_()
    nout, rout = ragged_pool(nvalues, nlogits, offsets), ragged_pool_reference(rvalues, rlogits, offsets)
    upstream = torch.linspace(-0.7, 0.9, nout.numel(), dtype=torch.float32).reshape_as(nout)
    ngrads = torch.autograd.grad(nout, (nvalues, nlogits), upstream)
    rgrads = torch.autograd.grad(rout, (rvalues, rlogits), upstream)
    torch.testing.assert_close(nout, rout, **TOLERANCE)
    for actual, expected in zip(ngrads, rgrads):
        torch.testing.assert_close(actual, expected, **TOLERANCE)
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.003) for model in (native, reference)]
    inputs = [values.clone().requires_grad_(), values.clone().requires_grad_()]
    predictions = [model(data, offsets) for model, data in zip((native, reference), inputs)]
    losses = [nn.functional.binary_cross_entropy_with_logits(prediction, labels)
              for prediction in predictions]
    for loss in losses:
        loss.backward()
    torch.testing.assert_close(predictions[0], predictions[1], **TOLERANCE)
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad, **TOLERANCE)
    gradient_differences = {}
    for (name, first), (_, second) in zip(native.named_parameters(), reference.named_parameters()):
        if first.grad is None or second.grad is None:
            raise AssertionError(f"Missing gradient for {name}")
        torch.testing.assert_close(first.grad, second.grad, **TOLERANCE)
        gradient_differences[name] = maximum_difference(first.grad, second.grad)
    before = _copy_state(native)
    for optimizer in optimizers:
        optimizer.step()
    for first, second in zip(native.parameters(), reference.parameters()):
        torch.testing.assert_close(first, second, **TOLERANCE)
    changes = _state_difference(native.state_dict(), before)
    if changes["scorer.weight"] == 0 or changes["head.weight"] == 0:
        raise AssertionError("Adam did not update the scorer and head")
    return {
        "seed": seed, "tolerance": TOLERANCE,
        "operator_output_max_abs": maximum_difference(nout, rout),
        "operator_values_gradient_max_abs": maximum_difference(ngrads[0], rgrads[0]),
        "operator_logits_gradient_max_abs": maximum_difference(ngrads[1], rgrads[1]),
        "model_prediction_max_abs": maximum_difference(*predictions),
        "model_input_gradient_max_abs": maximum_difference(inputs[0].grad, inputs[1].grad),
        "parameter_gradient_max_abs": gradient_differences,
        "adam_parameter_max_abs": _state_difference(native.state_dict(), reference.state_dict()),
        "native_update_max_abs": changes,
    }


def _evaluate(model, split, reverse=False):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(split["bags"]), 16):
            indices = list(range(start, min(start + 16, len(split["bags"]))))
            values, offsets, _ = pack_batch(split, indices)
            if reverse:
                boundaries = offsets.tolist()
                values = torch.cat([values[left:right].flip(0)
                                    for left, right in zip(boundaries, boundaries[1:])]).contiguous()
            predictions.append(model(values, offsets))
    predictions = torch.cat(predictions)
    loss = float(nn.functional.binary_cross_entropy_with_logits(predictions, split["labels"]))
    accuracy = float(((predictions >= 0) == split["labels"].bool()).float().mean())
    if not math.isfinite(loss) or not torch.isfinite(predictions).all():
        raise AssertionError("Nonfinite evaluation")
    return {"bce": loss, "accuracy": accuracy, "bags": len(predictions)}, predictions


class ResourceStop(RuntimeError):
    pass


def _rss_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize), "Windows WorkingSetSize"
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"), "/proc/self/statm resident"
    import resource
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(size if sys.platform == "darwin" else size * 1024), "getrusage peak RSS"


class Budget:
    def __init__(self, receipt):
        self.start = time.perf_counter()
        self.receipt = receipt

    def check(self, location):
        elapsed = time.perf_counter() - self.start
        rss, method = _rss_bytes()
        self.receipt["resources"]["samples"].append(
            {"location": location, "seconds": elapsed, "rss_bytes": rss, "method": method})
        if rss > 4 * 1024**3:
            raise ResourceStop("Sampled RSS exceeded 4 GiB")
        if elapsed > 900:
            raise ResourceStop("The 900-second consumer budget expired")


def _fit(split, validation, seed, pooling, updates, budget, progress=None, checkpoint=None):
    model = make_model(seed, pooling)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    batches = training_batches(seed, count=len(split["bags"]), updates=updates)
    best_loss, best_state, best_update = math.inf, None, None
    history, losses = [], []
    start = time.perf_counter()
    progress = {} if progress is None else progress
    progress.update({"seed": seed, "pooling": pooling, "updates_completed": 0,
                     "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                     "batch_order_sha256": hashlib.sha256(_json_bytes(batches)).hexdigest(),
                     "validation_history": history, "training_losses": losses})
    for update, indices in enumerate(batches, start=1):
        model.train()
        values, offsets, labels = pack_batch(split, indices)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(values, offsets)
        loss = nn.functional.binary_cross_entropy_with_logits(predictions, labels)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite training loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing or nonfinite training gradient for {name}")
        optimizer.step()
        losses.append(float(loss.detach()))
        progress["updates_completed"] = update
        if update % 20 == 0:
            metrics, _ = _evaluate(model, validation)
            history.append({"update": update, **metrics})
            if metrics["bce"] < best_loss:
                best_loss, best_state, best_update = metrics["bce"], _copy_state(model), update
            progress["selected_update"] = best_update
            progress["seconds"] = time.perf_counter() - start
            if checkpoint is not None:
                checkpoint()
            budget.check(f"seed={seed},pooling={pooling},update={update}")
    if best_state is None:
        raise AssertionError("No declared validation checkpoint completed")
    final_state = _copy_state(model)
    model.load_state_dict(best_state)
    selected_metrics, _ = _evaluate(model, validation)
    progress.update({"selected_update": best_update, "validation": selected_metrics,
                     "seconds": time.perf_counter() - start})
    return model, final_state, progress


def _split_metadata(splits):
    metadata = {}
    for name, split in splits.items():
        metadata[name] = {
            "seed": split["seed"], "bags": len(split["bags"]),
            "instances": sum(split["lengths"]), "data_sha256": split["digest"],
            "bag_ids_sha256": hashlib.sha256(_json_bytes(split["bag_ids"])).hexdigest(),
            "unique_bag_ids": len(set(split["bag_ids"])),
            "unique_instance_ids": len(split["instance_ids"]),
            "class_counts": {str(label): int((split["labels"] == label).sum()) for label in (0, 1)},
            "class_length_histograms": split["histograms"],
        }
    intersections = []
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            bags = set(splits[left]["bag_ids"]) & set(splits[right]["bag_ids"])
            instances = splits[left]["instance_ids"] & splits[right]["instance_ids"]
            intersections.append({"left": left, "right": right,
                                  "shared_bags": len(bags), "shared_instances": len(instances)})
            if bags or instances:
                raise AssertionError(f"Identical generated data across {left} and {right}")
    return {"splits": metadata, "cross_split_intersections": intersections}


def _summarize(results):
    summary = {}
    for pooling in POOLINGS:
        rows = [row for row in results if row["pooling"] == pooling]
        if not rows:
            continue
        summary[pooling] = {}
        for split in ("validation", "test", "longer_test"):
            if not all(split in row for row in rows):
                continue
            summary[pooling][split] = {}
            for metric in ("bce", "accuracy"):
                values = [row[split][metric] for row in rows]
                summary[pooling][split][metric] = {
                    "mean": sum(values) / len(values), "min": min(values), "max": max(values),
                    "seeds": [row["seed"] for row in rows]}
    return summary


def _identity(source_revision, wheel):
    distribution = importlib.metadata.distribution("ragged-pool")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise RuntimeError("Use an ordinary installed wheel, not an editable installation")
    package_root = Path(package.__file__).resolve().parent
    install_root = Path(distribution.locate_file("")).resolve()
    if not package_root.is_relative_to(install_root):
        raise RuntimeError("Imported ragged_pool is outside its installed distribution")
    libraries = list(package_root.glob("_C*.pyd")) + list(package_root.glob("_C*.so"))
    if len(libraries) != 1:
        raise RuntimeError("Expected exactly one installed native library")
    return {
        "source_revision": source_revision, "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(__file__), "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "torch_version": torch.__version__, "torch_build": torch.__config__.show(),
        "package_version": package.__version__, "distribution_version": distribution.version,
        "package_path": str(package_root), "native_library_path": str(libraries[0]),
        "native_library_sha256": sha256_file(libraries[0]),
        "wheel_path": str(Path(wheel).resolve()) if wheel else None,
        "wheel_sha256": sha256_file(wheel) if wheel else os.environ.get("RAGGED_WHEEL_SHA256"),
        "working_directory": os.getcwd(), "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }


def _save(path, receipt):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(arguments):
    path = Path(arguments.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a completed or adverse run.
    with path.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    protocol = protocol_record()
    receipt = {"status": "running", "mode": "smoke" if arguments.smoke else
               "train_only" if arguments.train_only else "full",
               "protocol": protocol, "protocol_sha256": hashlib.sha256(_json_bytes(protocol)).hexdigest(),
               "source_revision": arguments.source_revision,
               "script_sha256": sha256_file(__file__),
               "resources": {"budget_seconds": 900, "sampled_rss_stop_bytes": 4 * 1024**3,
                             "sampling_is_not_hard_os_limit": True, "samples": []},
               "results": [], "native_reference_trajectory_differences": []}
    budget = Budget(receipt)
    _save(path, receipt)
    try:
        if arguments.protocol_only:
            receipt["status"] = "protocol_frozen"
            _save(path, receipt)
            return receipt
        receipt["identity"] = _identity(arguments.source_revision, arguments.wheel)
        heldout = not (arguments.smoke or arguments.train_only)
        split_names = list(SPLIT_SPECS) if heldout else ["train", "validation"]
        splits = {name: make_split(*SPLIT_SPECS[name]) for name in split_names}
        receipt["data"] = _split_metadata(splits)
        receipt["same_start_parity"] = same_start_parity(splits["train"])
        budget.check("data generation and same-start parity")
        _save(path, receipt)
        seeds, poolings, updates = ((11,), ("native", "reference"), 20) if arguments.smoke else (
            MODEL_SEEDS, POOLINGS, 200)
        for seed in seeds:
            paired = {}
            for pooling in poolings:
                budget.check(f"before seed={seed},pooling={pooling}")
                receipt["active_run"] = {"seed": seed, "pooling": pooling, "updates_completed": 0}
                _save(path, receipt)
                model, final_state, result = _fit(
                    splits["train"], splits["validation"], seed, pooling, updates, budget,
                    progress=receipt["active_run"], checkpoint=lambda: _save(path, receipt))
                predictions = {}
                if heldout:
                    for name in ("test", "longer_test"):
                        result[name], predictions[name] = _evaluate(model, splits[name])
                        _, permuted = _evaluate(model, splits[name], reverse=True)
                        torch.testing.assert_close(permuted, predictions[name], **TOLERANCE)
                        result[name]["permutation_max_abs_logit_difference"] = maximum_difference(
                            permuted, predictions[name])
                receipt["results"].append(result)
                receipt.pop("active_run")
                if pooling in ("native", "reference"):
                    paired[pooling] = {"selected": _copy_state(model), "final": final_state,
                                       "predictions": predictions, "result": result}
                budget.check(f"finished seed={seed},pooling={pooling}")
                _save(path, receipt)
            first, second = paired["native"], paired["reference"]
            receipt["native_reference_trajectory_differences"].append({
                "seed": seed, "selected_parameter_max_abs": _state_difference(first["selected"], second["selected"]),
                "final_parameter_max_abs": _state_difference(first["final"], second["final"]),
                "selected_updates": {name: paired[name]["result"]["selected_update"] for name in paired},
                "training_loss_max_abs": max(abs(a - b) for a, b in zip(
                    first["result"]["training_losses"], second["result"]["training_losses"])),
                "heldout_logit_max_abs": {name: maximum_difference(first["predictions"][name], second["predictions"][name])
                                          for name in first["predictions"]}})
            _save(path, receipt)
        receipt["summary"] = _summarize(receipt["results"])
        receipt["status"] = "passed_smoke" if arguments.smoke else "passed_train_only" if arguments.train_only else "passed"
    except BaseException as error:
        receipt["status"] = "partial_resource_stop" if isinstance(error, ResourceStop) else (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - budget.start
        _save(path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON receipt path, never overwrite a run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--protocol-only", action="store_true", help="Freeze protocol without generating data")
    modes.add_argument("--smoke", action="store_true", help="Seed 11, attention pair, 20 updates, no held-out data")
    modes.add_argument("--train-only", action="store_true", help="Full fixed training and validation, no held-out data")
    parser.add_argument("--source-revision", default=os.environ.get("RAGGED_REVISION", "unrecorded"))
    parser.add_argument("--wheel", help="Installed wheel file to identify with SHA256")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run(arguments)
    print(json.dumps({key: result[key] for key in ("status", "mode", "protocol_sha256", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
