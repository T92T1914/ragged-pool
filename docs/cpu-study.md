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

Execution is pending. This section will be replaced with the observed results and linked source identity after the declared runs complete.
