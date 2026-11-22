# A parallel forward kernel, designed before it is measured

This note fixes the design of a second CUDA forward kernel for the companion and writes down what I expect it to do before any timing run. The design and the predictions above [Results](#results) were written before any measurement. The results sit at the end, in a later commit than the predictions. The existing kernel stays in the code and can still be selected, so every comparison here can be run again.

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

`kernel` is 0 for the existing kernel and 1 for the new one. Groups with `L <= switch_length` take the warp path. The study's `N` uses 32, and so did the public `pool` until [A switch that follows the width](#a-switch-that-follows-the-width) gave it a switch that depends on `D`. 0 sends every nonempty group down the block path and 4096 sends every nonempty group down the warp path. The operation runs the same admission as `pool`. With `repeats` at 0 it launches the chosen kernel once and returns an empty timing tensor. Otherwise it launches the kernel `repeats` times with a pair of CUDA events around each launch and returns the output with a CPU float64 tensor. That tensor holds the milliseconds of each launch, then one last entry for the span from the first start to the last stop. It has no autograd formula and no fake rule, because it exists for the study and the tests only.

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

The study ran once, on November 22, 2026 at 15:44 UTC, from `examples/forward_study.py` at revision 0c64e74, whose `cuda` tree differs from the kernel's revision a6a38e9 only by that script. Its record is [evidence/forward-kernel-study-windows-sm89-20261122.json](evidence/forward-kernel-study-windows-sm89-20261122.json). The predictions above were committed in 95b00b8, before any timing run. The record names the device, the RTX 4090 with driver 621.24 and 128 SMs, and its clocks, which stayed at 2865 MHz for the SMs and 10701 MHz for memory, in P0, before and after every case. Nsight Compute was not on PATH, so no profile ran. The build and test log of the same run, which the record does not hold, shows the rest: the wheel was built with NVCC 12.8.93 and the MSVC 14.44 toolset and installed into a fresh acceptance environment, where the whole CUDA suite passed twice before the study started, 975 passed and 4 skipped with no `RAGGED_POOL_CUDA_` variable, and the same with `RAGGED_POOL_CUDA_FORWARD=adaptive`.

The record counts 32 of the 42 predictions as held. Every kernel only prediction on the fourteen shapes held. Ten failed in the record's count: four complete call predictions and six in the sweep. The record judges only the bound of the float64 prediction about differences, and that prediction also said the difference grows with the length of the group, which did not happen. By the rule above that a prediction holds only when all of it holds, that makes 31 of 42. Each miss is named below.

### What the measurement stood on

One part of the protocol did not hold in full. Item 2 asks for a card with no other work on it. The script refuses to time above 30 percent, and its five readings before the start were 22, 10, 9, 12 and 10 percent, so other work was on the card when the run began. The record does not say what that work was or how long it lasted. The tails suggest it stayed. Most kernels with medians of 5 to 13 microseconds have 95th percentiles between 170 and 270 microseconds, while every kernel with a median above 19 microseconds keeps its 95th percentile within 20 percent of the median. That pattern fits short launches that sometimes wait behind the other work, though I have not shown that cause. I read the medians, which tails of that size move very little, and every number here was measured beside that background work.

Kernel only times in the record fall mostly on steps of about 1.02 microseconds. Below about 10 microseconds two medians one step apart already differ by more than 10 percent, so the band in the predictions cannot judge kernels that short.

### Kernel only

Microseconds, the median with the 5th and 95th percentiles in brackets.

| Case | E | N | Predicted | Result |
| --- | --- | --- | --- | --- |
| Empty batch | 2.0 (1.0 to 13.5) | 2.0 (1.0 to 11.5) | no kernel runs | |
| Empty groups | 4.1 (3.1 to 202.3) | 4.1 (3.1 to 202.3) | `N ≈ E` | held |
| Singletons | 6.1 (5.1 to 219.3) | 6.1 (5.1 to 221.3) | `N ≈ E` | held |
| Many short | 11.3 (10.2 to 140.8) | 6.1 (5.1 to 205.9) | `N < E` | held |
| Balanced | 80.9 (79.9 to 81.4) | 11.3 (11.3 to 228.4) | `N < E` | held |
| Skewed | 2336.8 (2335.7 to 2776.3) | 89.1 (88.1 to 89.1) | `N < E` | held |
| One long | 6196.2 (5983.2 to 7016.2) | 133.1 (132.1 to 134.1) | `N < E` | held |
| Wide | 437.2 (436.2 to 437.2) | 10.2 (9.2 to 224.3) | `N < E` | held |
| Full batch float32 | 242.7 (242.7 to 243.7) | 12.3 (11.3 to 227.3) | `N < E` | held |
| Full batch float64 | 221.2 (221.2 to 222.7) | 11.3 (11.3 to 261.3) | `N < E` | held |
| One full group | 28836.3 (27988.0 to 29860.7) | 245.8 (244.7 to 246.8) | `N < E` | held |
| Narrowest full batch | 28.7 (27.6 to 28.7) | 6.1 (5.1 to 179.7) | `N < E` | held |
| Skewed full batch | 29413.3 (28422.0 to 30354.3) | 340.0 (338.9 to 341.0) | `N < E` | held |
| Widest singletons | 13.3 (13.3 to 202.3) | 5.1 (5.1 to 193.0) | `N < E` | held |

The new kernel's median is lower on 11 of the 14 shapes. It ties on Empty batch, where no kernel runs, and on Empty groups and Singletons. The ratio runs from 1.9 on Many short to 117 on One full group, where one thread walked 4096 rows of 128 columns alone and 256 threads now share them.

### Complete call

| Case | E | N | C | Predicted | Result |
| --- | --- | --- | --- | --- | --- |
| Empty batch | 29.4 (28.2 to 148.6) | 28.9 (26.7 to 44.1) | 365.1 (334.2 to 696.6) | `E ≈ N < C` | held |
| Empty groups | 39.2 (36.6 to 85.0) | 43.7 (35.5 to 117.9) | 468.1 (352.7 to 862.8) | `E ≈ N < C` | missed `E ≈ N` |
| Singletons | 83.0 (78.8 to 284.3) | 82.8 (78.8 to 321.0) | 668.8 (589.9 to 1147.9) | `E ≈ N < C` | held |
| Many short | 378.4 (354.4 to 698.6) | 363.0 (350.1 to 658.4) | 644.9 (587.2 to 1089.5) | `N ≈ E < C` | held |
| Balanced | 1566.5 (1326.1 to 1863.5) | 1551.2 (1262.3 to 1757.7) | 701.5 (599.0 to 1169.3) | `C < N < E` | missed `N < E` |
| Skewed | 4055.0 (3883.0 to 4452.7) | 1625.8 (1334.8 to 1814.1) | 707.9 (608.9 to 1089.1) | `C < N < E` | held |
| One long | 9759.5 (9409.1 to 10276.6) | 3367.8 (3151.7 to 3717.4) | 870.5 (767.8 to 1389.9) | `C < N < E` | held |
| Wide | 3533.1 (3393.8 to 3927.4) | 3116.5 (2856.8 to 3461.7) | 658.0 (585.8 to 1074.9) | `C < N < E` | held |
| Full batch float32 | 20158.9 (19691.5 to 21026.2) | 19780.2 (19450.5 to 20571.6) | 747.8 (610.0 to 1311.1) | `C < N ≈ E` | held |
| Full batch float64 | 22317.8 (21852.5 to 23156.1) | 22093.2 (21660.1 to 23180.5) | 786.9 (616.4 to 1327.6) | `C < N ≈ E` | held |
| One full group | 51241.2 (50255.7 to 53637.4) | 22513.8 (21868.7 to 23635.6) | 960.0 (809.1 to 1510.2) | `C < N < E` | held |
| Narrowest full batch | 367.2 (352.7 to 731.4) | 353.5 (332.5 to 709.2) | 697.2 (583.2 to 1146.4) | `N < E < C` | missed `N < E` |
| Skewed full batch | 49231.6 (48566.9 to 50523.4) | 20329.4 (19882.6 to 21156.2) | 836.3 (708.5 to 1347.1) | `C < N < E` | held |
| Widest singletons | 691.3 (636.7 to 1035.3) | 791.0 (632.5 to 1054.9) | 690.2 (596.3 to 1149.8) | `N ≈ E ≈ C` | missed `N ≈ E` and `N ≈ C` |

The composition's complete call stays between about 0.4 and 1 millisecond on every shape, while the native call's time outside the kernel grows with the number of values, as assumption 5 expected of the single thread admission scan. The record does not time admission on its own, so that attribution rests on the assumption and on how the time scales. The two routes meet at Widest singletons, the 16384 value shape, where `E ≈ C` held. Above that the composition is faster on every shape. On Full batch float32 and Full batch float64 about 20 to 22 milliseconds of the native call go to everything but the kernel, against a new kernel of 11 to 12 microseconds.

### Sweep, kernel only

| L | D | W | K | Predicted | Result |
| --- | --- | --- | --- | --- | --- |
| 8 | 8 | 6.1 (5.1 to 173.1) | 7.2 (6.1 to 197.6) | `W ≈ K` | missed |
| 8 | 128 | 7.2 (6.1 to 180.6) | 7.2 (6.1 to 176.8) | `W < K` | missed |
| 16 | 8 | 6.1 (5.1 to 172.6) | 7.2 (6.1 to 174.1) | `W ≈ K` | missed |
| 16 | 128 | 9.2 (8.2 to 179.3) | 8.2 (7.2 to 195.2) | `W ≈ K` | missed |
| 32 | 8 | 7.2 (6.1 to 204.8) | 7.2 (7.2 to 221.7) | `W ≈ K` | held |
| 32 | 128 | 12.3 (11.3 to 269.2) | 9.2 (8.2 to 253.0) | `W ≈ K` | missed |
| 64 | 8 | 9.2 (8.2 to 209.9) | 8.2 (7.2 to 211.2) | `W ≈ K` | missed |
| 64 | 128 | 19.5 (18.4 to 20.5) | 11.3 (11.3 to 228.4) | `K < W` | held |
| 128 | 8 | 13.3 (12.3 to 219.5) | 9.2 (9.2 to 196.6) | `K < W` | held |
| 128 | 128 | 33.8 (32.8 to 33.8) | 16.4 (16.4 to 227.3) | `K < W` | held |
| 256 | 8 | 20.5 (19.5 to 21.5) | 13.3 (12.3 to 199.2) | `K < W` | held |
| 256 | 128 | 62.5 (61.4 to 62.5) | 27.6 (26.6 to 28.3) | `K < W` | held |

### The misses and their reasons

1. Empty groups, `E ≈ N` in the complete call. The medians are 39.2 and 43.7 microseconds, 11 percent apart, while both kernels take 4.1 microseconds on their own. Two of the five rounds of `N` came in at 52 and 53 microseconds and the other three near 39, and one round of `E` came in near 50. The relation failed inside that scatter between rounds. What caused the slow rounds is not explained.
2. Balanced, `N < E` in the complete call. The new kernel saves 70 microseconds, but the rest of the call, which assumption 5 puts mostly on admitting 32768 values, takes about 1.5 milliseconds, so the saving is under 5 percent of the call. I judged the existing kernel's share of the call as large enough to show, and it is not.
3. Narrowest full batch, `N < E` in the complete call. The same reason at a smaller size: the kernel saves about 23 microseconds of a call near 360, which is under the 10 percent the relation needs.
4. Widest singletons, `N ≈ E` and `N ≈ C` in the complete call. The median of `N` is 14 percent above `E`, although its own kernel is 8 microseconds faster and the two 5th percentiles are 632.5 and 636.7. The medians of single rounds run from 669 to 797 microseconds for `E` and from 661 to 815 for `N`, about 20 percent, twice the band. The relations failed inside that scatter, and its cause is not explained.
5. The sweep at 8 and 16 items with 8 columns, and at 16 items with 128 columns and 64 items with 8 columns, each `W ≈ K`. In every one the medians are one timer step apart, and with kernels of 6 to 9 microseconds one step is more than 10 percent. A gap of one step fails the band at this size, so the verdict turns on whether both medians land on the same step, as they did at 32 items with 8 columns, where `W ≈ K` held.
6. The sweep at 8 items with 128 columns, `W < K`. Both medians are 7.2 microseconds. I expected the block path's final step to cost it the shortest groups, and any such cost is smaller than one step of the clock.
7. The sweep at 32 items with 128 columns, `W ≈ K`. The block path takes 9.2 microseconds against 12.3 for the warp path, three steps apart and 25 percent faster. With 128 columns the block path is already ahead at 32 items, and one step ahead at 16, so the switch at 32 is too high for wide rows on this card. This matches the failure the sweep paragraph named in advance. That paragraph gives two mechanisms that fit it: one warp may not reach all of an SM's FP64 throughput at that width, and eight warps hide memory latency that a lone warp cannot. No profile ran, so the record cannot tell them apart. The sweep's row at 32 items and 128 columns has the same shape as Full batch float32, so on that shape the block path would likely take about 9.2 microseconds where the switch at 32 gives the warp path 12.3. Moving the switch, or letting it depend on the width, is a design change for a later item.
8. The float64 prediction about differences, its clause that the difference grows with the length of the group. The record scores only the bound, which held. The largest difference, 5.6e-16, is on Full batch float64 with groups of 32 and on Wide with groups of 64, not on the two shapes with 4096 items. I expected the existing thread's long serial sum to gather the most rounding error. The record holds only the largest difference for each shape, so why that did not happen is not explained.

### The numbers themselves

Two runs of `N` on the same inputs gave identical bytes on all 14 shapes. On every float32 shape among them the two kernels gave identical bytes, which is stronger than the one unit in the last place the prediction allowed. In float64 the largest difference between them is 5.6e-16, below the predicted 1e-13, and every candidate's float64 output stays within 6.7e-16 of the same computation in float64 on the CPU, far inside the documented absolute tolerance of 2e-12. Every output of the sweep was within the float32 tolerance of the float64 computation as well.

Two expectations in [What carries over](#what-carries-over-and-what-changes) did not hold as written. Counted in units of the last place of each output, the float64 differences reach 12288 on Full batch float64, because some outputs are smaller than 1e-3, so a difference of 5.6e-16 is many units of a small number. Outputs that small are weighted sums of values of both signs that nearly cancel, which is my reading rather than something the record shows. The second expectation is the growth with the length of the group, the eighth miss above.

### What this shows

On this card the new kernel's own time is lower than the existing kernel's on 11 of the 14 shapes, from 1.9 times on Many short to 117 times on One full group, and it changes no float32 output. A complete call is now mostly time outside the kernel, which the limits above expected the single thread admission scan to take, and above about 16384 values the eager composition still finishes first. Nothing here is a general speed claim. It is one card in one machine, one driver, one toolkit, the bounded sizes above and other work on the card during the run. Whether the new kernel becomes the default for `pool` still waits on a Compute Sanitizer run of this binary.

## A switch that follows the width

The sweep above put the block path ahead at 32 items with 128 columns, where the switch at 32 still sent each group down the warp path, and the shape of Full batch float32 sits right there. This section gives `pool` a switch that depends on the width and fixes how the next run measures it. Everything in it above its own Results was written and committed before that run.

### What the sweep showed

Count the rows one lane of the lone warp walks in the accumulation. The warp runs `32 / R` item slots side by side, where `R` is the number of lanes that share a row, so a group of `L` items gives each lane `ceil(L * R / 32)` rows. `R` is 8 with 8 columns and 32 with 128 columns. Set the two halves of the sweep side by side by that count, in microseconds from the record above:

| Rows per lane | 8 columns: L | W | K | 128 columns: L | W | K |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 | 6.1 | 7.2 | | | |
| 4 | 16 | 6.1 | 7.2 | | | |
| 8 | 32 | 7.2 | 7.2 | 8 | 7.2 | 7.2 |
| 16 | 64 | 9.2 | 8.2 | 16 | 9.2 | 8.2 |
| 32 | 128 | 13.3 | 9.2 | 32 | 12.3 | 9.2 |
| 64 | 256 | 20.5 | 13.3 | 64 | 19.5 | 11.3 |
| 128 | | | | 128 | 33.8 | 16.4 |
| 256 | | | | 256 | 62.5 | 27.6 |

At the same number of rows the warp path takes the same time at both widths, or one timer step more with 8 columns, although each lane holds one column there and four with 128 columns. Both widths tie at 8 rows and put the block path ahead from 16 rows on. Past that the warp path adds about 0.22 microseconds a row at both widths: 7.2 microseconds over the 32 rows from 128 to 256 items with 8 columns, and 28.7 over the 128 rows from 128 to 256 items with 128 columns. That is about 640 cycles at the recorded 2865 MHz. My reading is that a lone warp waits on the loads and the dependent double additions of each row, so the number of columns a lane holds matters little and the number of rows it walks decides. No profile ran, so that cause is not shown.

The same table says the maximum and the weights cost the warp path little at these sizes. At 64 rows the warp walks 8 items a lane in those phases with 8 columns and 2 with 128 columns, and the two times differ by one step.

### The rule

`pool` with `adaptive` now sends a group down the warp path while each lane walks at most 8 rows and the group holds at most 32 items:

```text
switch(D) = min(32, 8 * 32 / R)    with R = row_lanes(D)
```

That is 32 for `D <= 8`, 16 for `9 <= D <= 16` and 8 for `D >= 17`. The cap at 32 keeps the property the first design chose, that the maximum and the weights give each lane at most one item. The table above suggests those two phases cost the warp path little even when a lane walks several items, so the cap may be stricter than it needs to be. But no row in it has the warp path ahead while a lane walks more than one item in those phases, so I keep the cap and let the sweep test it. Nothing changes for `D <= 8`. `_forward_study` keeps its explicit `switch_length`, so the switch at 32 can still be run and every comparison above can be repeated.

Of the fourteen shapes, three change path: Balanced, with 32 columns and groups of 32, and both Full batches, with 128 columns and groups of 32, move to the block path. Every other shape keeps its paths.

The CUDA suite checks `pool` against this rule for every width from 1 to 128. `test_public_pool_takes_the_path_the_width_rule_names` builds one float64 group whose result shows which path ran: 1000, -1000 and `2**-50` placed so that one path adds the tiny value to 1000, where it rounds away, and the other lets 1000 and -1000 cancel first. It checks one item past the switch for every width and the switch itself for widths 2 to 16. For `D = 1` and for `D >= 17`, both paths add the items of a group at the switch in the same order, so the choice there cannot change any output and the test cannot see it.

### How the next run measures it

The same script and protocol as above: the same seed, warmups, rounds and samples, event pairs behind the stream hold, and the same refusal to time when any of five utilization readings before the start exceeds 30 percent. Three things are added.

1. A new candidate `A`, the new kernel with the switch `pool` now uses for the case's width, is timed kernel only beside `E` and `N` on the fourteen shapes. The rotation now runs over three candidates. The complete call still times `E`, `N` and `C`.
2. The sweep gains widths 1, 16, 32 and 64 at the same six lengths, so it has 36 rows. The new rows are drawn after every earlier case, so the twelve rows above get the same values and logits they had.
3. The record is a new file under `evidence` with schema 2. No existing evidence file changes. The predictions of 95b00b8 are judged again on the same cases and counted apart from the predictions below.

### Predictions for the switch

1. On every one of the 36 sweep rows, the path the rule picks is no slower than the other path: its median is at most the other median, judged in twentieths of a microsecond, so a tie holds. This is the check the queue item names.
2. Kernel only, `A < N` on Balanced, Full batch float32 and Full batch float64, the three shapes the rule moves to the block path. The sweep row at 32 items with 128 columns gave 9.2 against 12.3 microseconds.
3. Kernel only, `A ≈ N` on the ten other shapes where a kernel runs. There `A` and `N` launch the same kernel with the same paths, so these ten are a control that shows how far two identical launches land apart.

Where I expect a miss to come from, if one comes. The rows at the switch itself sit near a tie: 32 items with 1 and 8 columns, 16 items with 16 columns, and 8 items with 32, 64 and 128 columns. Moving one timer step there fails the first prediction, and the Results will say whether a step is all it was. The rows with one column are the other risk. A lone warp there walks only 2 rows at 64 items, so if rows were the whole story the warp path would win, and the cap sends those groups to the block path. If the warp path wins at 64 items or more with one column, the cap is too low for narrow rows, and lifting it is a later item. The control in prediction 3 has kernels of 4 to 6 microseconds on several shapes, where one step already breaks the 10 percent band, so a miss there would be the timer and not the kernel.

### Limits of the switch

The sweep covers float32 and the widths 1, 8, 16, 32, 64 and 128, so `R` of 2 and 4 is not measured, and neither is any width between the ones listed. The rule rests on one card and one run of the sweep, with the conditions the first run states. It changes only which path the new kernel takes. The existing kernel stays the default for `pool`, and the backward kernels stay as they are.

### Results of the switch

The run took place once, on November 22, 2026 at 23:28 UTC, from `examples/forward_study.py` at revision 14980dc, the commit that holds the rule and the predictions above. Its record is [evidence/forward-switch-study-windows-sm89-20261122.json](evidence/forward-switch-study-windows-sm89-20261122.json). The build and test log of the same run, which the record does not hold, shows the wheel built with NVCC 12.8.93 and the Visual Studio 2022 Build Tools, and the whole CUDA suite passed twice before the study started: 1103 passed and 4 skipped with no `RAGGED_POOL_CUDA_` variable, and the same with `RAGGED_POOL_CUDA_FORWARD=adaptive`. The 128 tests beyond the first run's 975 are the width rule test, one for each width. The five utilization readings before the start were 19, 15, 18, 16 and 20 percent, under the limit of 30, so desktop work was on the card again, as in the first run. All 26 cases of the first record drew the same data here, by their `data_sha256`. The record's `default_switch` field still reads 32. It names the switch of `N`, not the switch `pool` uses now.

46 of the 49 switch predictions held. All three `A < N` held, 9 of the 10 controls held, and the rule's check held on 34 of the 36 sweep rows.

Kernel only, microseconds, the median with the 5th and 95th percentiles in brackets.

| Case | N | A | Predicted | Result |
| --- | --- | --- | --- | --- |
| Empty batch | 2.0 (1.0 to 8.3) | 2.0 (1.0 to 9.3) | no kernel runs | |
| Empty groups | 4.1 (3.1 to 189.7) | 4.1 (3.1 to 202.8) | `A ≈ N` | held |
| Singletons | 6.1 (5.1 to 252.5) | 6.1 (5.1 to 207.4) | `A ≈ N` | held |
| Many short | 6.1 (5.1 to 204.8) | 6.1 (5.1 to 205.3) | `A ≈ N` | held |
| Balanced | 11.3 (11.3 to 210.2) | 9.2 (8.2 to 225.4) | `A < N` | held |
| Skewed | 89.1 (88.6 to 89.2) | 89.1 (88.1 to 89.1) | `A ≈ N` | held |
| One long | 133.1 (132.1 to 133.1) | 133.1 (132.1 to 133.1) | `A ≈ N` | held |
| Wide | 10.2 (9.2 to 298.1) | 10.2 (9.2 to 298.2) | `A ≈ N` | held |
| Full batch float32 | 12.3 (11.3 to 392.0) | 9.2 (8.2 to 557.5) | `A < N` | held |
| Full batch float64 | 11.3 (11.3 to 439.5) | 8.2 (8.2 to 573.4) | `A < N` | held |
| One full group | 245.8 (244.7 to 245.8) | 245.8 (244.7 to 246.8) | `A ≈ N` | held |
| Narrowest full batch | 6.1 (5.1 to 223.8) | 6.1 (5.1 to 202.5) | `A ≈ N` | held |
| Skewed full batch | 340.0 (338.9 to 341.5) | 340.0 (338.9 to 341.0) | `A ≈ N` | held |
| Widest singletons | 5.2 (5.1 to 207.1) | 6.1 (5.1 to 294.0) | `A ≈ N` | missed |

The sweep, kernel only, by width and then length. Rows a lane counts what one lane of the lone warp walks on the warp path, and the rule picks the warp path `W` when `L` is at most the switch.

| D | L | Rows a lane | Switch | W | K | Rule picks | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 1 | 32 | 6.1 (5.1 to 205.9) | 7.8 (7.2 to 183.0) | W | held |
| 1 | 16 | 1 | 32 | 6.1 (5.1 to 212.5) | 8.1 (7.2 to 212.6) | W | held |
| 1 | 32 | 1 | 32 | 6.1 (5.1 to 219.4) | 8.2 (7.2 to 205.8) | W | held |
| 1 | 64 | 2 | 32 | 6.1 (6.1 to 208.5) | 7.3 (7.2 to 212.5) | K | missed |
| 1 | 128 | 4 | 32 | 7.3 (7.2 to 225.8) | 8.2 (7.2 to 217.7) | K | missed |
| 1 | 256 | 8 | 32 | 10.2 (9.2 to 218.0) | 9.2 (8.2 to 207.4) | K | held |
| 8 | 8 | 2 | 32 | 6.1 (5.1 to 205.8) | 7.2 (6.1 to 209.1) | W | held |
| 8 | 16 | 4 | 32 | 6.1 (6.1 to 222.8) | 7.2 (6.1 to 262.0) | W | held |
| 8 | 32 | 8 | 32 | 7.2 (6.1 to 194.6) | 7.2 (7.2 to 214.8) | W | held |
| 8 | 64 | 16 | 32 | 9.2 (8.2 to 257.2) | 8.2 (7.2 to 298.2) | K | held |
| 8 | 128 | 32 | 32 | 22.5 (16.4 to 316.4) | 16.4 (11.3 to 540.6) | K | held |
| 8 | 256 | 64 | 32 | 20.5 (19.5 to 238.6) | 13.3 (12.3 to 301.2) | K | held |
| 16 | 8 | 4 | 16 | 6.1 (5.1 to 205.8) | 7.2 (6.1 to 194.1) | W | held |
| 16 | 16 | 8 | 16 | 7.2 (6.1 to 210.2) | 7.2 (6.1 to 222.2) | W | held |
| 16 | 32 | 16 | 16 | 9.2 (8.2 to 226.3) | 8.2 (7.2 to 264.3) | K | held |
| 16 | 64 | 32 | 16 | 12.3 (11.3 to 229.9) | 9.2 (8.2 to 206.1) | K | held |
| 16 | 128 | 64 | 16 | 18.5 (18.4 to 239.1) | 12.3 (11.3 to 229.4) | K | held |
| 16 | 256 | 128 | 16 | 32.8 (32.8 to 33.8) | 17.4 (17.4 to 234.5) | K | held |
| 32 | 8 | 8 | 8 | 7.2 (6.1 to 193.2) | 7.2 (6.1 to 206.5) | W | held |
| 32 | 16 | 16 | 8 | 8.2 (8.2 to 224.8) | 8.2 (7.2 to 224.8) | K | held |
| 32 | 32 | 32 | 8 | 11.3 (11.3 to 383.8) | 9.2 (8.2 to 291.2) | K | held |
| 32 | 64 | 64 | 8 | 18.4 (17.4 to 230.5) | 11.3 (10.2 to 277.0) | K | held |
| 32 | 128 | 128 | 8 | 31.7 (30.7 to 31.7) | 16.4 (15.4 to 239.4) | K | held |
| 32 | 256 | 256 | 8 | 58.4 (57.3 to 58.4) | 26.6 (26.6 to 27.6) | K | held |
| 64 | 8 | 8 | 8 | 7.2 (6.1 to 192.8) | 7.2 (6.1 to 223.2) | W | held |
| 64 | 16 | 16 | 8 | 9.2 (8.2 to 244.0) | 8.2 (7.2 to 224.4) | K | held |
| 64 | 32 | 32 | 8 | 12.3 (11.3 to 520.1) | 9.2 (8.2 to 413.2) | K | held |
| 64 | 64 | 64 | 8 | 19.5 (18.4 to 237.7) | 11.3 (11.3 to 291.7) | K | held |
| 64 | 128 | 128 | 8 | 32.8 (32.8 to 33.8) | 16.4 (16.4 to 261.3) | K | held |
| 64 | 256 | 256 | 8 | 61.4 (60.4 to 61.4) | 27.6 (26.6 to 27.6) | K | held |
| 128 | 8 | 8 | 8 | 7.2 (6.1 to 255.4) | 7.2 (6.2 to 261.5) | W | held |
| 128 | 16 | 16 | 8 | 9.2 (8.2 to 548.0) | 8.2 (7.2 to 292.5) | K | held |
| 128 | 32 | 32 | 8 | 12.3 (11.3 to 575.3) | 9.2 (8.2 to 519.5) | K | held |
| 128 | 64 | 64 | 8 | 19.5 (18.4 to 587.8) | 11.3 (11.3 to 568.1) | K | held |
| 128 | 128 | 128 | 8 | 33.8 (32.8 to 33.8) | 16.4 (16.4 to 261.3) | K | held |
| 128 | 256 | 256 | 8 | 62.5 (61.4 to 62.5) | 27.6 (26.6 to 28.7) | K | held |

On the twelve rows of the first sweep, the rule picked the faster or tied path every time, including the two rows where the switch at 32 had picked the slower one. Seven rows here are ones where the rule and the switch at 32 pick different paths: 32 items with 16 columns, and 16 and 32 items with 32, 64 and 128 columns. On six of them the block path the rule picks is one to three timer steps faster, and on the seventh, 16 items with 32 columns, the two tie.

Rows a lane held up as the axis for every width from 8 to 128. At 64 rows the warp path takes 18.4 to 20.5 microseconds at every one of those widths, and at each width that reaches them it takes 31.7 to 33.8 at 128 rows and 58.4 to 62.5 at 256 rows. One column is the exception, named in the first miss below.

### The misses of the switch and their reasons

1. One column at 64 and at 128 items, the rule's check. The rule picks the block path at 7.3 and 8.2 microseconds where the warp path took 6.1 and 7.3, about one timer step each time. At 256 items the block path leads again, 9.2 against 10.2. The rule keeps the first design's switch at 32 for one column because of its cap, so this is the miss the predictions named in advance: the cap is too low for one column. The same rows also show where rows alone stop explaining the warp path. At 8 rows the warp path takes 7.2 microseconds with 8, 16, 32, 64 and 128 columns, where each lane walks one item in the maximum and the weights, but 10.2 with one column at 256 items, where it walks 8. So those two phases cost the warp path about three timer steps once a lane walks 8 items in them. That corrects a sentence above, written before this run, which said they cost the warp path little. A switch that counts the items a lane walks in those phases as well as its rows would be a later item, with its own predictions and run.
2. Widest singletons, `A ≈ N`. `A` and `N` launch the same kernel with the same paths there, and their medians are 6.1 and 5.2, one timer step apart. The medians of single rounds run from 5.1 to 6.1 for both. This is the timer, as the control was meant to show, and not the kernel.

One row needs a word although its prediction held. At 128 items with 8 columns the medians are 22.5 and 16.4 microseconds, against 13.3 and 9.2 in the first record. The record shows why: the SM clock read 1365 MHz before that row and 2085 MHz after it, which was also the reading before the next row, while every other reading of the run was 2865 MHz. The next row, 128 items with 128 columns, repeated its first medians. The clocks are not locked, so this row ran slower than its neighbors, and the rule still picked the faster path there.

### The first predictions, judged again

The same run judged the 42 predictions of 95b00b8 again on the same inputs. 32 held, as in the first run, and the same ten cases missed. Widest singletons missed on `N ≈ C` and `E ≈ C` this time, where the first run missed on `N ≈ E` and `N ≈ C`, which again sits inside the scatter between rounds that its miss describes. Eleven of the twelve rows of the first sweep repeated both medians to the tenth of a microsecond. The twelfth is the row at 128 items with 8 columns above.

On every float32 shape `A` gave the same bytes as `N`, including Balanced and Full batch float32, where it takes the other path. On Full batch float64 the two differ by at most 5.6e-16, and `A` stays within 5.6e-16 of the same computation in float64 on the CPU, far inside the documented tolerance.

### What the switch shows

On this card, a switch that follows the width moved the three shapes with groups of 32 and 32 or more columns to the block path and cut their kernel times from 11.3 to 9.2, 12.3 to 9.2 and 11.3 to 8.2 microseconds, without changing a float32 output. Across the sweep the rule picked the faster or tied path on all 30 rows with 8 or more columns. With one column it gives up about one timer step at 64 and 128 items, which its cap causes. The run shares the first run's conditions: one card in one machine, desktop work on the card, unlocked clocks and one run. The lengths where the rule switches inside a width were not measured between the sweep's powers of two, so for 17 or more columns nothing here times 9 to 15 items, and for 9 to 16 columns nothing times 17 to 31. The existing kernel is still the default for `pool`, and the Compute Sanitizer run that decision waits on has not happened.
