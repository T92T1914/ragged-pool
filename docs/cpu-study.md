# One synthetic set task and CPU cost study

This study asks two questions about the existing CPU layer: how learned normalized pooling behaves in one small bag-level classifier, and what its checked calls cost compared with equivalent PyTorch composition. It does not introduce a new pooling algorithm. The existing one-step example remains the installation smoke test. Neither a modeling advantage nor a speed advantage is required.

## Protocol declared before evaluation

The protocol below was fixed before held-out evaluation on October 20, 2026. The scripts and initial protocol digest are retained with the execution record. A resource stop or exception is a partial result. A later run may not silently replace an adverse result.

Bag labels are balanced. A negative bag contains four-dimensional standard-normal nuisance vectors. A positive bag contains exactly one randomly positioned signal vector drawn from a unit-covariance normal with mean `[3,0,0,0]`, plus nuisance vectors. Features are clipped to `[-6,6]`. Positive and negative bags are generated in same-length pairs with independent contents, then shuffled, so their length histograms match. The visible distributions overlap. The generative label therefore does not imply that every individual bag is perfectly classifiable from its noisy features.

| Split | Generator seed | Bags | Lengths |
| --- | --- | --- | --- |
| Train | 71001 | 512 | Uniform integers 1 through 16 |
| Validation | 71002 | 128 | Uniform integers 1 through 16 |
| In-range test | 71003 | 256 | Uniform integers 1 through 16 |
| Longer-bag test | 71004 | 256 | Uniform integers 17 through 32 |

Data digests, actual class/length histograms and disjoint bag-content identities are recorded. Empty bags are excluded from this task, while operator checks retain their accepted empty behavior. Longer positive bags still have one signal, so the longer-bag result also changes signal dilution and difficulty. It is a narrow shift test.

Every model uses a shared `Linear(4,16)` encoder, tanh and `Linear(16,1)` head. The attention models add a `Linear(16,1,bias=False)` scalar scorer. A shared scalar bias in scores would cancel under softmax. Native attention, reference attention, mean, max and sum variants share corresponding encoder/head initialization and batches per seed. The attention variants share their scorer. Parameter-count differences are reported, without dummy parameters for cosmetic parity.

Model seeds are 11, 29 and 47. Each completed run performs 200 Adam updates at learning rate 0.003, batch size 16, float32. Validation is evaluated every 20 updates. The minimum validation binary-cross-entropy checkpoint is selected, with earliest checkpoint breaking ties. Test data is not used for tuning or checkpoint selection. Report every seed's binary cross-entropy and accuracy at threshold 0.5, plus descriptive mean/range. Three seeds do not establish significance or a stable ranking. Check prediction invariance to a deterministic within-bag permutation. Same-start native/reference outputs, both input gradients, parameter gradients and an optimizer update must agree before longer training. Longer trajectories are reported rather than assumed bitwise equal.

Attention weights are aggregation coefficients. This synthetic task supplies no medical, cyber, defense or causal-explanation result. Bag-level learning and sum pooling have established precedents. [Attention-based MIL, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html), [Deep Sets, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).

## Cost protocol

Use an ordinary optimized wheel installed outside the checkout, not an instrumented build. Data seed is 72001. These eight configurations are the entire set, rather than axes for a larger sweep.

| Case | Group lengths | D | dtype |
| --- | --- | --- | --- |
| Empty batch | `[]` | 8 | float32 |
| Empty groups | `[0,0,0,0]` | 8 | float64 |
| Singletons | 128 groups of 1 | 8 | float32 |
| Many short | 128 groups of 4 | 16 | float32 |
| Balanced | 32 groups of 32 | 32 | float32 |
| Skewed | `[0] + [1]*14 + [1000]` | 32 | float32 |
| One long | `[4096]` | 16 | float64 |
| Wide | 8 groups of 64 | 128 | float64 |

Compare the native public call, readable reference and one segmented stable-softmax composition. Check output and both gradients first. The checked built-in wrapper enforces bounded metadata/content/offset admission before numerical execution. Never run malformed offsets through its raw numerical path. PyTorch v2.9.1's offsets reduction has incomplete `unsafe=False` checks. Include lengths/group-index/normalizer construction in the call. No precomputed helper cache is used. [Versioned SegmentReduce source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/SegmentReduce.cpp).

Native float32 uses double accumulation, while ordinary float32 PyTorch reductions perform different arithmetic. Retain and disclose that difference. Float64 rows provide the cleaner arithmetic comparison. Modeling mean/max/sum variants produce different functions and are not equivalent-operation speed baselines.

Use one PyTorch thread and interop thread. Measure inference under consistent grad mode and fresh-graph forward plus a fixed upstream vector-Jacobian product separately. Use Timer warmup and three rotated candidate rounds with at least 0.2-second measurement windows where practical, retaining raw blocks, block sizes, median and IQR. Shared-PC load can make a result noisy. Nested repetitions are not independent hardware trials. [Versioned Timer source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/benchmark/utils/timer.py).

Full optimizer-step timing uses the first shuffled training minibatch for seed 11 and the same attention architecture for native/reference/built-in. Make a canonical reference warmup step to create Adam state. Restore its model/optimizer snapshot outside each interval. Time `zero_grad(set_to_none=True)`, encoder/scorer/pooling/head, binary-cross-entropy loss, backward and `step()`. Collect 30 outer repetitions per candidate in three rotated rounds using a monotonic high-resolution timer. Generation, packing, state restoration and cold compilation are excluded. This measures one fixed training state, not data-loader throughput. Do not advance optimizer state inside Timer's automatic inner repetition.

Profile the balanced float32 case with CPU activity, separately from latency timing. Whole custom-op totals do not identify internal C++ exponentiation, vector allocation or loop-line costs. Shape/stack recording can add overhead and retain references. At most one production optimization may follow a concrete hypothesis supported by source and measurements. Retaining the kernel is a valid decision. [Profiler documentation](https://docs.pytorch.org/docs/2.9/profiler.html).

Probe `torch.compile(backend="inductor",fullgraph=True,dynamic=False)` on balanced float32 native and known-valid built-in numerical forward/backward, at most two attempts of 120 seconds each. Native keeps opaque in-operation validation. The built-in compiled path excludes data-dependent Python admission, which is checked separately. Label that weaker scope. Record capture failures separately from missing toolchain or backend failures. Require real forward/backward agreement, keep cold cost separate and never call eager fallback success. Existing aot_eager remains its separate accepted integration path. No installation or upgrade is part of the probe.

The experiment uses one worker, one intra-op/interop thread, at most two build/test threads, a 15-minute consumer cap, a five-minute timing cap and five minutes combined for profiler/compiler probes. Sample resident memory between bounded runs against 4 GiB, without claiming a hard OS memory limit. Build, installation and required CI are separate. Stop rather than silently extending experiments to obtain a favorable answer.

## Results

The study completed on October 20, 2026 using the scripts at source revision [a0adbb0](https://github.com/T92T1914/ragged-pool/tree/a0adbb0d2cca9dce7438da7468ab1cd7fa8c67fa). The ordinary wheel was installed in a fresh environment outside the checkout. Execution used Windows x64 build 26300, CPython 3.11.8, PyTorch 2.9.1+cpu and the existing MSVC 19.44/C++17 extension build. The CPU identifies as AMD64 Family 25 Model 97 with 16 logical processors. Each experiment used one execution worker and one PyTorch intra-op/interop thread. Background load on this shared machine was not controlled.

The [machine-readable record](results/cpu-study.json) retains every model seed, validation checkpoint, data histogram/digest, candidate correctness check, raw timing block, profile event and compiler outcome. The measured wheel SHA256 is `1b6285d9547cc857ee8b7887ce7fb5688a2763ed2dfc7e7e64b29b363cda652b`, and the installed native library SHA256 is `1672d8ac80162330934e92c26078d324c386a64c6c78e206e49a96db26cafee4`. These identities describe the executed build, not the older v0.1.0 downloadable release assets. The frozen consumer protocol digest is `e6023a1ee1ed7175c500663b62bb8e9e9ec25ad878652c5def5e9724f99c7cb4`.

### Model behavior

All 15 runs completed 200 updates. The table reports descriptive means across seeds 11, 29 and 47, with accuracy ranges in brackets. Every seed's BCE and accuracy, including validation results, remain in the linked record. Attention has 113 parameters, while mean, max and sum have 97. The extra 16 scorer weights are part of the architecture comparison.

| Pooling | Parameters | In-range BCE | In-range accuracy [range] | Longer-bag BCE | Longer-bag accuracy [range] |
| --- | --- | --- | --- | --- | --- |
| Native attention | 113 | 0.3593 | 86.07% [84.77%, 87.50%] | 0.5196 | 73.31% [71.09%, 75.00%] |
| Reference attention | 113 | 0.3593 | 86.07% [84.77%, 87.50%] | 0.5196 | 73.31% [71.09%, 75.00%] |
| Mean | 97 | 0.6038 | 66.15% [64.45%, 67.58%] | 0.6856 | 54.04% [53.12%, 54.69%] |
| Max | 97 | 0.5278 | 83.85% [82.42%, 86.33%] | 0.6105 | 75.78% [75.39%, 76.56%] |
| Sum | 97 | 0.6159 | 68.36% [66.80%, 69.14%] | 0.8580 | 58.59% [56.25%, 61.72%] |

Attention had the lowest mean BCE in both test sets and the highest mean in-range accuracy in this fixed experiment. Its longer-bag accuracy fell to 73.31%, below max pooling's 75.78%. Accuracy and BCE therefore give different answers on that comparison. The longer test also places one signal among more nuisance instances. This result does not establish general size robustness or a stable ranking from three seeds.

Native and reference attention selected the same checkpoint and produced identical accuracy for every seed and split. Their largest held-out logit difference was `9.54e-7`. All deterministic within-bag reversals passed, with maximum logit difference `1.43e-6` across the five variants. Sum pooling for seed 47 selected update 180. All other models selected update 200 through validation BCE, with test data excluded from selection.

Before training, same-start pooling differed by at most `1.19e-7` in output, `1.49e-8` in the values gradient and `5.96e-8` in the logits gradient. The full model's largest parameter-gradient difference was `1.12e-8`, and its post-Adam parameter difference was `4.66e-10`. The generator recorded distinct bag and instance contents across all six split pairs, with balanced classes and exactly matched per-class length histograms. The fixed consumer run completed within its 900-second budget.

### Checked-call CPU costs

Each cell below is median [IQR] in microseconds over the three rotated rounds. Inference uses inference mode. Forward plus VJP creates a fresh graph for the same fixed upstream tensor. The checked reference and built-in calls include Python admission, while native admission runs inside the operation. Every built-in helper tensor and conversion remains in the interval. These results compare the complete admitted routes under this contract, rather than isolated numerical kernels.

| Case | Inference native | Inference reference | Inference built-in | Forward + VJP native | Forward + VJP reference | Forward + VJP built-in |
| --- | --- | --- | --- | --- | --- | --- |
| Empty batch | 7.45 [2.45] | 82.99 [50.09] | 93.27 [25.65] | 107.75 [31.75] | 237.79 [55.11] | 240.67 [55.69] |
| Empty groups | 7.17 [1.83] | 103.92 [26.83] | 99.12 [25.21] | 102.21 [29.43] | 233.61 [47.81] | 231.80 [73.40] |
| Singletons | 20.40 [6.45] | 1963.10 [494.20] | 193.38 [42.52] | 143.57 [39.21] | 9257.00 [1620.20] | 379.12 [82.55] |
| Many short | 64.52 [6.70] | 2049.40 [412.40] | 284.05 [81.90] | 265.29 [50.25] | 10100.20 [1813.80] | 536.00 [105.55] |
| Balanced | 166.25 [29.38] | 872.23 [99.82] | 503.37 [76.48] | 532.10 [81.62] | 3251.45 [741.70] | 844.86 [115.93] |
| Skewed | 169.68 [27.36] | 641.02 [101.56] | 500.75 [101.67] | 521.92 [56.84] | 1798.85 [395.60] | 801.67 [94.57] |
| One long | 342.27 [50.72] | 749.94 [106.53] | 920.00 [231.45] | 979.13 [215.99] | 1328.60 [394.30] | 1524.30 [376.72] |
| Wide | 341.65 [107.22] | 868.30 [250.30] | 868.40 [283.45] | 949.70 [220.96] | 1784.80 [485.75] | 1235.40 [334.40] |

Native had lower medians in all eight checked-call cases against these two eager baselines on this machine. The comparison retains native double accumulation for float32 and ordinary float32 arithmetic for the compositions. Candidate output/gradient differences were at most `5.96e-7` in float32 and `2.22e-15` in float64. The raw block times and large spreads remain part of the result. These measurements do not establish a universal advantage, independent hardware trials or stable tail latency.

The restored-state optimizer step gave the following medians and IQR over 30 outer repetitions per candidate. This includes the model, BCE, backward and Adam update, with the canonical model/optimizer snapshot restored outside each interval.

| Attention implementation | Full-step median, microseconds | IQR, microseconds |
| --- | --- | --- |
| Native | 796.90 | 413.90 |
| Reference | 2120.85 | 606.57 |
| Checked built-in | 1000.30 | 752.38 |

The native worst step was approximately 15 milliseconds, well above its 0.797-millisecond median. That outlier and the broad IQR make a tight latency promise inappropriate. The monotonic timer's reported resolution was 100 nanoseconds. Before collection, the representative native and built-in full steps agreed with reference predictions within `5.96e-8` and parameter gradients/updates within `1.49e-8`. The complete eager timing investigation took 38.06 seconds within its 300-second cap.

### Compiler, profile and implementation decision

The balanced float32 native path completed real forward and backward with `backend="inductor", fullgraph=True, dynamic=False`. Its outputs and both input gradients matched eager native exactly for this admitted case. First forward/backward, including cold compilation, took 4.377 seconds. The separate warm probe recorded median 770.90 microseconds and IQR 113.19 microseconds over 23 blocks. It was one availability probe, not the rotated eager timing protocol. The C++ operation remains opaque in the compiled graph, including its native admission. This result establishes that path in the executed environment without showing fusion inside the native body or an optimized speed advantage.

The second and final attempt, for already-admitted built-in numerical composition, failed with `InductorError` wrapping `CppCompileError`. MSVC 19.44 was present and invoked for generated C++20 code. Paths containing spaces were split in its command, producing warning D9024 and fatal error C1083. This was a generated-code build failure, rather than missing compiler tools or a Python admission graph break. The failed attempt is retained, with no eager fallback labeled compiled success and no toolchain installation or retry. The existing `aot_eager` acceptance remains separate. Profiling and both compiler attempts together stayed below the 300-second allowance.

The separate CPU profile contains native dispatcher and generated-autograd events, many per-group reference operations, and segmented built-in reductions. Its instrumented totals are excluded from the latency tables. An opaque native event does not attribute time to an internal exponentiation, temporary vector or loop line. Source inspection and these bounded measurements did not justify a production change, so the native implementation was retained. No numerical validation or precision was removed to obtain a faster result.

### Reproduction

Build an ordinary wheel with the pinned route in the [README](../README.md), then install that wheel and the same pinned dependencies in a fresh environment outside the checkout. Copy `examples/learn_sets.py`, `examples/pool_baselines.py` and `examples/cpu_cost.py` together into an external consumer directory. Run with isolated Python and new output filenames. Replace the example paths and use the actual revision of the source you built.

```powershell
$python = 'C:\work\ragged-accept\Scripts\python.exe'
$consumer = 'C:\work\ragged-consumer'
$wheel = 'C:\work\ragged-pool\dist\ragged_pool-0.1.0-cp311-cp311-win_amd64.whl'
$revision = git -C C:\work\ragged-pool rev-parse HEAD
& $python -I "$consumer\learn_sets.py" --output "$consumer\study.json" --source-revision $revision --wheel $wheel
& $python -I "$consumer\cpu_cost.py" --timing --output "$consumer\timing.json" --source-revision $revision --wheel $wheel --budget-seconds 300
& $python -I "$consumer\cpu_cost.py" --profile --output "$consumer\profile.json" --source-revision $revision --wheel $wheel --budget-seconds 30
& $python -I "$consumer\cpu_cost.py" --compile --output "$consumer\compile.json" --source-revision $revision --wheel $wheel --budget-seconds 270
```

For the Windows compiler probe, make the existing compiler available in the child process, using `call "<Visual Studio installation>\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44` before invoking the Python command from a noninteractive `.cmd` file. This is the same child-environment approach used by the README build route. The executed probe used that setup, with `TORCHINDUCTOR_COMPILE_THREADS=1`, and its generated built-in command still failed on paths containing spaces. No global compiler setting was changed.

The 30-second profile and 270-second compiler caps share the five-minute allowance. Compiler children are bounded and owned by their parent process. Failed or partial receipts are preserved rather than overwritten. The shorter `learn_sets.py --smoke` mode uses only seed 11, native/reference and 20 updates, without generating or evaluating held-out sets. Its `passed_smoke` result is distinct from a completed study. The original one-step dtype consumers remain separate installation checks.
