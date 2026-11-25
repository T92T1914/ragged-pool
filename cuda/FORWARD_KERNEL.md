# A parallel forward kernel, designed before it is measured

This note fixes the design of a second CUDA forward kernel for the companion and writes down what I expect it to do before any timing run. It holds no measurement. When the kernel is built and measured, the results go under [Results](#results) at the end, and the commit that adds them comes after the commit that adds this note. The existing kernel stays in the code and can still be selected, so every comparison here can be run again.

## Why the existing kernel is slow by construction

`pool_forward` in [csrc/pool.cu](csrc/pool.cu) launches one block per group with one thread in it. That thread finds the maximum logit, sums the exponentials, then computes every weight a second time and adds weight times value into a 128 entry `double` array, one item and one column at a time. Nothing inside a group runs in parallel. A batch holds at most 128 groups, so the whole GPU never has more than 128 working threads, and each of them is one lane of a 32 lane warp.

Double precision makes that worse on this card. The [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) lists a 64 to 1 ratio between FP32 and FP64 throughput for compute capability 8.9. I assume a warp instruction in double precision costs about the same issue time with one active lane as with 32, so the existing kernel pays the full price of each double operation for one lane of work.

## The new kernel

`pool_forward_adaptive` keeps the grid of the existing kernel: one block per group, so `B` blocks, and every group stays on one SM. Each block has 256 threads, which is 8 warps. The block reads the length `L` of its own group and picks a mapping.

1. `L == 0`: the block returns. The host already allocated a zero filled output, as it does today.
2. `L <= 32`: warp 0 handles the group alone and the other 7 warps return at once. This is the warp path.
3. `L > 32`: all 8 warps handle the group together. This is the block path.

Both paths loop over items with a stride, so either one can take any length when a study run forces it. A lane or thread with no item of its own starts from negative infinity for the maximum and from 0 for a sum, so idle lanes change nothing.

### Warp path

1. Maximum. Lane `j` reads items `j`, `j + 32` and so on, converts each logit to `double` and keeps its own maximum. A butterfly reduction then leaves the group maximum in every lane: five rounds of `__shfl_xor_sync` in which each lane combines its value with a partner's.
2. Weights. Each lane computes `exp(l - m)` in `double` for its items, keeps them in a shared memory buffer and adds them up. A second butterfly gives every lane the sum `s`. Each lane then replaces its own entries with `exp(l - m) / s`, and `__syncwarp()` makes the weights visible to the whole warp.
3. Accumulation. Lanes map to columns so that consecutive lanes read consecutive values of one row. A row gets `R` lanes, the smallest power of two that is at least `min(D, 32)`, and the warp runs `32 / R` item slots side by side. A lane keeps up to four `double` partial sums, for columns `d`, `d + R`, `d + 2R` and `d + 3R`, over the items of its slot. A butterfly across slots combines the partial sums, and the lanes of slot 0 write the row, cast once to the input dtype.

With `L <= 32` every lane holds at most one item in the first two phases, so those phases have no serial loop at all.

### Block path

1. Maximum. Thread `t` reads items `t`, `t + 256` and so on. Each warp reduces with a butterfly, the 8 warp results go to shared memory, and after a barrier every thread takes the maximum of the 8 in warp order.
2. Weights. A barrier first makes sure every thread has read the 8 maxima before those slots are written again. Each thread then stores `exp(l - m)` for its items in the shared buffer and sums them. The sum goes through the same warp butterfly, shared memory slots and barrier, and the 8 warp sums are added in warp order. Each thread then divides its own entries by `s`, followed by a barrier.
3. Accumulation. The column mapping is the same as on the warp path, with the item slots spread over all 8 warps. Each warp combines its slots with a butterfly and writes its `D` partial sums to shared memory. After a barrier, thread `d` for each `d < D` adds the 8 warp partials in warp order and writes column `d`.

Shared memory per block holds the weights for `min(N, 4096)` items, 8 rows of `D` partial sums and 8 slots for warp results, all `double`. At the bounds that is 4096 plus 1024 plus 8 values, or 41024 bytes, which fits under the 48 KiB a launch may request without opting in to more.

## Why a warp, a block and a switch at 32

The mapping follows two papers on segmented reduction.

Larsen and Henriksen, [Strategies for regular segmented reductions on GPU](https://doi.org/10.1145/3122948.3122952) (FHPC 2017), define three strategies and pick one at run time. Their rule uses one thread per segment only when there are more than 2^16 segments. Otherwise a segment longer than half the workgroup goes to one or more whole workgroups, and shorter segments are packed several to a workgroup. They evaluate four microbenchmarks and two Rodinia applications on an NVIDIA K40. One of the microbenchmarks, Black-Scholes option pricing, is bound by computation rather than memory, and they report the same performance pattern for it as for the others. They also say the best choice depends on the hardware and on the operator.

Cordeiro and Zola, [Optimized Parallel Reduction for Regular and Irregular Segments on GPU](https://doi.org/10.1002/cpe.70715) (Concurrency and Computation: Practice and Experience, 2026), give a segment to a thread, a warp, a block, several blocks or a whole kernel, and choose by the average segment size at run time with thresholds tuned for each GPU. Their warp strategy combines the lanes' partial results with shuffles, as mine does. On an RTX A4500, for the maximum of 32 bit unsigned integers, they measured the warp strategy best from 63 to 254 elements and one block per segment best from 255 to 8105.

This operator never has more than 128 groups, which is far below 2^16, so Larsen and Henriksen's rule would never choose one thread per group here. The existing kernel sits in the regime their rule keeps for very many segments. Both papers give short segments to a warp or to a shared workgroup and long segments to one or more blocks, and that is the shape of the new kernel.

I depart from them in three places.

1. I choose per group, not per launch. A batch here can mix lengths, and the Skewed shape below holds 14 singletons beside one group of 1000. A rule on the average length would send all of them down one path.
2. I never pack several groups into one block. With at most 128 groups and the 128 SMs of this card, one group per block lets the block scheduler give each group its own SM, and with it that SM's FP64 units.
3. I never split one group across several blocks. That keeps the kernel to a single launch with no second reduction pass, and it keeps the order of each group's arithmetic fixed. The cost is plain: a batch with one long group runs that group on one SM while the rest of the card waits.

Larsen and Henriksen's rule with 256 thread blocks would switch at 128, and Cordeiro and Zola measured 255 on their card. Both thresholds were set with a fixed and large total amount of work spread over the whole GPU, so a segment's cost was its share of the card's throughput. Here the batch never holds more than 4096 items and each group has an SM to itself, so a group's cost is the length of its own critical path on one SM. Heavy work per item is not by itself a reason to move the switch, since Larsen and Henriksen saw the same pattern for a compute bound operator. The regime is the reason. A lone warp has the shortest critical path while no lane has to loop, and the block path's barriers start to pay once a lane of a lone warp would walk several items or rows. At 32 the warp path still gives every lane at most one item in the maximum and weight phases, so I put the switch there. My estimate is that the two paths stay within 10 percent of each other from about 16 to 32 items at every width, and up to 64 for narrow rows, so the exact switch should cost little. The sweep below tests that directly.

## What carries over and what changes

1. Admission is unchanged. The metadata checks, the single thread validation kernel, the four byte status transfer and the stream synchronization run before the new kernel exactly as they run before the old one.
2. Each group subtracts its maximum logit first. The maximum is exact in any order, so both kernels subtract the same value, the largest exponential is exactly 1 and the sum is at least 1. Logits in `[-1000,1000]` keep every exponent in `[-2000,0]`, so nothing overflows.
3. Every exponential, sum, weight and product stays in `double`, with one cast to the input dtype at the end. The build keeps the companion's flags, including `--fmad=false`, so a multiply followed by an add stays two rounded operations.
4. The order of the sums changes. The existing thread adds items in index order. The new kernel adds in index order inside a lane, then combines lanes and warps in a fixed tree. The two kernels can differ in the last bits of a `double` sum. After the cast to float32 I expect nearly every output to match the existing kernel exactly and the rest to differ by one unit in the last place. In float64 I expect differences of a few units in the last place for short groups, growing with the length of the group because the existing thread's rounding error grows along its long serial sum. Both stay inside the documented tolerances, float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`, apart from cancellation near magnitude 1000, where the existing tests already state their own absolute tolerance.
5. The kernel uses no atomics and every reduction tree is fixed, so the same inputs give the same bytes on every run.
6. Backward, JVP and double backward keep their one thread per group kernels.

## How it is selected

The public `pool` operation keeps the existing kernel by default. Setting `RAGGED_POOL_CUDA_FORWARD=adaptive` before the library loads switches `pool` to the new kernel for the whole process. Every route that calls `pool`, padded, functional, forward AD, second order, compiled and export included, then goes through the new kernel, so the existing suite runs against it unchanged.

The study and the new tests use one private operation:

```text
ragged_pool_cuda::_forward_study(Tensor values, Tensor logits, Tensor offsets,
                                 int kernel, int switch_length, int repeats) -> (Tensor, Tensor)
```

`kernel` is 0 for the existing kernel and 1 for the new one. Groups with `L <= switch_length` take the warp path. The public `pool` and the study's `N` use 32, 0 sends every nonempty group down the block path and 4096 sends every nonempty group down the warp path. The operation runs the same admission as `pool`. With `repeats` at 0 it launches the chosen kernel once and returns an empty timing tensor. Otherwise it launches the kernel `repeats` times with a pair of CUDA events around each launch and returns the output with a CPU float64 tensor. That tensor holds the milliseconds of each launch, then one last entry for the span from the first start to the last stop. It has no autograd formula and no fake rule, because it exists for the study and the tests only.

Whether the new kernel becomes the default for `pool` is a decision for after the study and after a Compute Sanitizer run of the new binary, which on this Windows machine needs the elevated helper the [README](README.md) describes. Until then the README's sentence about one thread per group stays true of the default.

## Study protocol

Everything in this section is fixed before the first timing run.

1. Environment. The companion's tested envelope: Windows x64, CPython 3.11.8, `torch==2.9.1+cu128`, CUDA 12.8.1 toolkit components with NVCC 12.8.93, MSVC 19.44, `sm_89` only and the build flags in [setup.py](setup.py). The device is the RTX 4090 in this machine. The record keeps the driver, the SM count the device reports and the `nvidia-smi` clocks and utilization before and after each run. Clocks are not locked, because that needs administrator rights.
2. Sharing the machine. The run is a heavy job. It takes the shared lock first and starts only when `nvidia-smi` shows no other work on the card.
3. Data. Seed 72001 with the generator, the draw order and the clamp to `[-6,6]` of the CPU study's `make_cases` in [examples/cpu_cost.py](../examples/cpu_cost.py). The eight CPU study shapes are drawn first and in that study's order, so they get the same values and logits it used, and the upstream tensors it also draws go unused here. The six bound shapes and then the sweep follow from the same generator. Everything is drawn on the CPU and copied to `cuda:0` once before any timing.
4. Correctness first. For every shape, every candidate's output is compared with a float64 CPU computation of the same formula inside the tolerances above, and the largest difference between the existing and the new kernel is recorded. A failed comparison stops the run.
5. Candidates.
   - `E`: the existing kernel, `_forward_study` with `kernel=0`.
   - `N`: the new kernel, `kernel=1` with the switch at 32.
   - `C`: the eager composition, the CPU study's segmented stable softmax from [examples/pool_baselines.py](../examples/pool_baselines.py) on the same device. Before it, tensor operations check the same contract as native admission (offsets start at 0, end at N and never decrease, values and logits are finite and within `[-1000,1000]`), fold the checks into one flag and read it with one `.item()`, so the composition pays one host synchronization like the native call. In PyTorch 2.9.1 the offsets form of [`segment_reduce`](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/SegmentReduce.cpp) runs no check that synchronizes. Its one dimensional reductions call CUB's `DeviceSegmentedReduce`, and its two dimensional reduction runs [one thread per output element](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/SegmentReduce.cu) over the whole segment. In float32 the composition accumulates in float32 while both kernels accumulate in `double`. The CPU study kept that difference in view and so does this one.
   - `W` and `K`, for the sweep only: the new kernel with the switch at 4096, which is the warp path for every group, and at 0, which is the block path for every nonempty group.
6. Complete call scope, for `E`, `N` and `C`. Each sample synchronizes the device, reads `time.perf_counter_ns`, makes one call, synchronizes again and reads the clock. It includes admission with its host synchronization, allocation and the launch.
7. Kernel only scope, for `E` and `N`, and for `W` and `K` in the sweep. The times come from the event pairs inside `_forward_study`, after one admission. Before each block of timed launches the operation enqueues a spin kernel that holds the stream while the host enqueues the whole block, so no event pair can contain time the GPU spent waiting for the host. If the spin has already ended when the host finishes, the operation spins longer and runs the block again. The record keeps the span of each block beside the sum of its pairs as a check on that.
8. Repeats. 20 warmup calls or launches per candidate and shape, then 5 rounds with the candidates in a rotated order and 50 samples per candidate in each round. That gives 250 samples per candidate, shape and scope, reported as the median with the 5th and 95th percentiles.
9. Profile. If Nsight Compute is installed and allowed to read the performance counters, one profile of the kernel behind the worst miss. Otherwise the record says it was unavailable.
10. Output. A script under `examples` that exits 0 and writes a new JSON file under `evidence`. No existing evidence file changes.

## Shapes

The eight shapes of the [CPU study](../docs/cpu-study.md) keep their lengths, widths and dtypes. Six more sit at the documented bounds of `N<=4096`, `1<=D<=128` and `0<=B<=128`.

| Case | Group lengths | D | dtype | New kernel path |
| --- | --- | --- | --- | --- |
| Empty batch | `[]` | 8 | float32 | no launch |
| Empty groups | `[0,0,0,0]` | 8 | float64 | 4 blocks return at once |
| Singletons | 128 groups of 1 | 8 | float32 | warp |
| Many short | 128 groups of 4 | 16 | float32 | warp |
| Balanced | 32 groups of 32 | 32 | float32 | warp |
| Skewed | `[0] + [1]*14 + [1000]` | 32 | float32 | warp for the singletons, block for 1000 |
| One long | `[4096]` | 16 | float64 | block |
| Wide | 8 groups of 64 | 128 | float64 | block |
| Full batch float32 | 128 groups of 32 | 128 | float32 | warp |
| Full batch float64 | 128 groups of 32 | 128 | float64 | warp |
| One full group | `[4096]` | 128 | float64 | block |
| Narrowest full batch | 128 groups of 32 | 1 | float32 | warp |
| Skewed full batch | `[1]*127 + [3969]` | 128 | float32 | warp for the singletons, block for 3969 |
| Widest singletons | 128 groups of 1 | 128 | float32 | warp |

The sweep uses float32 batches of equal groups, `L` in 8, 16, 32, 64, 128 and 256, with `B = min(128, 4096 / L)` groups, at `D` of 8 and of 128.

## Predictions

These rest on six assumptions, written down so that a miss can be traced to one of them.

1. Each group runs on its own SM, so a batch costs about as much as its slowest group.
2. Double arithmetic dominates at 1/64 of the float rate. A warp instruction costs about the same issue time with one active lane as with 32, and one warp can keep its SM's FP64 units busy.
3. A `double` exponential takes a few dozen double instructions, so computing the weights costs more than finding the maximum.
4. Every timed launch carries a fixed device cost that is the same for both kernels, which pushes small kernels toward the 10 percent band.
5. Admission reads all `N*D` values one at a time in one thread, with a double comparison for each. I expect that scan alone to cost about as much as the whole composition near sixteen thousand values, which is the size of Widest singletons, less below that and more above it.
6. Each eager operation from Python costs about the same host time on Windows, so the composition's complete call hardly depends on the shape until one of its kernels runs long.

`N < E` means the median of `N` is more than 10 percent below the median of `E`. `N ≈ E` means the two medians are within 10 percent of the smaller one. A prediction holds when every relation in it holds. Otherwise it is a miss, and the Results section names the relation that failed with a reason, or with the words not explained.

| Case | Kernel only | Complete call |
| --- | --- | --- |
| Empty batch | no kernel runs | `E ≈ N < C` |
| Empty groups | `N ≈ E` | `E ≈ N < C` |
| Singletons | `N ≈ E` | `E ≈ N < C` |
| Many short | `N < E` | `N ≈ E < C` |
| Balanced | `N < E` | `C < N < E` |
| Skewed | `N < E` | `C < N < E` |
| One long | `N < E` | `C < N < E` |
| Wide | `N < E` | `C < N < E` |
| Full batch float32 | `N < E` | `C < N ≈ E` |
| Full batch float64 | `N < E` | `C < N ≈ E` |
| One full group | `N < E` | `C < N < E` |
| Narrowest full batch | `N < E` | `N < E < C` |
| Skewed full batch | `N < E` | `C < N < E` |
| Widest singletons | `N < E` | `N ≈ E ≈ C` |

The reasons, case by case.

1. Empty batch. Neither kernel launches, so both native routes run the same admission and allocation. The composition still issues its admission and the operations of its empty branch.
2. Empty groups and Singletons. Four empty blocks cost only the launch. With one item per group the warp path leaves 31 lanes idle and its butterflies have nothing to combine, while the existing thread computes two exponentials and one row. I expect the fixed launch cost to hide the difference.
3. Many short, Balanced, Narrowest full batch and both Full batches. The warp path removes the serial loop over items and computes each exponential once, where the existing thread computes it twice and walks every column of every item alone.
4. Skewed, One long, One full group and Skewed full batch. One long group decides the existing kernel's time. The block path spreads that group over 256 threads, but the group still sits on one SM, which is the cost of not splitting it. Wide has eight equal groups of 64, each on the block path, while the existing thread walks 64 rows of 128 columns alone.
5. Widest singletons. One item per group is little work for either kernel, but each item has 128 columns, and the existing thread walks all of them alone. I expect that to be enough for `N < E` even after the launch cost.
6. Complete calls. Admission is identical for `E` and `N`, so `N` pulls ahead only where the existing kernel's own time is a visible share of the call. Well above sixteen thousand values the composition's admission and its small kernels finish first, because its admission runs across the whole card while the native scan runs in one thread. Narrowest full batch holds 4096 values and Many short 8192, so their native admission stays cheaper than the composition. Widest singletons holds 16384, right at the point I expect the two to meet, so I expect all three routes within 10 percent of each other.

The sweep, kernel only, `W` for the warp path on every group and `K` for the block path on every group:

| L | B | D of 8 | D of 128 |
| --- | --- | --- | --- |
| 8 | 128 | `W ≈ K` | `W < K` |
| 16 | 128 | `W ≈ K` | `W ≈ K` |
| 32 | 128 | `W ≈ K` | `W ≈ K` |
| 64 | 64 | `W ≈ K` | `K < W` |
| 128 | 32 | `K < W` | `K < W` |
| 256 | 16 | `K < W` | `K < W` |

With narrow rows the warp path is short of independent loads and the block path is short of work, so I expect them to stay close until the warp path has several items per lane. With 128 columns the block path's final step adds 8 warp partials for each of the 128 columns, which I expect to cost it the shortest groups, while its eight warps hide memory latency once each lane of a lone warp would walk 64 rows or more. If the last part of assumption 2 is wrong and one warp cannot reach all of an SM's FP64 throughput, the block path will win the 128 column side by more than this table expects, and the switch at 32 will look too high.

Two more predictions about the numbers themselves. Two runs of `N` on the same inputs give identical bytes. The largest difference between `E` and `N` is at most one unit in the last place of the float32 output. In float64 it grows with the length of the group, and I expect it to stay below `1e-13` on every shape here, far inside the documented tolerance.

## Limits

The study measures one card in one machine with one driver, one toolkit and the bounded sizes above, using the timing method above. It says nothing about other GPUs, other sizes or a general speed advantage, and the companion still claims none. The backward kernels stay as they are. The single thread admission scan stays as it is, and I expect the study to show it as the largest cost of a complete call on most shapes.

Implementation and verification use AI coding assistance, consistent with [NOTICE](NOTICE).

## Results

Not measured yet.
