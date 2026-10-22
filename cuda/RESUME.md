# Resume a small CUDA training run

`examples/resume_training.py` saves a known eager training run after its second update. The producer exits. A new process restores model parameters and SGD momentum, then completes updates three through five on CUDA0. The parent compares the complete continuation with uninterrupted native and independent primitive GPU trajectories. This demonstrates model and optimizer integration for one fixed fixture.

The example reuses the accepted CUDA wheel and native binary. It adds a consumer and tests, without changing the pooling operators, building another native payload or extending the CPU package's recovery contract.

## One finite trajectory

The seeded model encodes width3 features into width4, learns an importance score per item, pools each packed group and predicts one scalar. Its five parameter blocks contain25 scalars. The batches vary `(N,B)` across `(9,5)`, `(12,6)`, `(12,4)`, `(11,4)` and `(9,5)`, including empty and singleton groups. SGD uses learning rate0.03, momentum0.8, no dampening, weight decay or Nesterov update, and explicitly disables foreach and fused implementations.

Saving occurs after `optimizer.step()` and `zero_grad(set_to_none=True)`, with no live gradients. The next-update cursor is2 in zero-based indexing. The checkpoint contains all five model tensors and five momentum buffers. Each is an ordinary detached contiguous CPU tensor with independent storage. CPU staging is explicit consumer behavior. The native pooling API keeps its own device-transfer contract.

The producer exits before the resumer starts. Both children finish before the parent creates its CUDA fixture or uninterrupted baseline models. Child results use CPU tensors. This keeps the three GPU workloads sequential and makes the process boundary real, rather than reconstructing state inside a process that still owns the producer's GPU model.

## What is checked

Every update compares pooled vectors, predictions, loss, raw-input gradients, all parameter gradients, updated parameters and momentum. Same-native resumed and uninterrupted records must match literal tensor bytes. The independent primitive path uses float32 `rtol=2e-4, atol=2e-6` and float64 `rtol=2e-10, atol=2e-12`. Both paths retain their model and optimizer objects. Source inputs, initial parameters and checkpoint bytes, tensor versions and storage ownership remain checked.

The momentum recurrence is checked separately from optimizer output: the first buffer is the gradient, later buffers are `0.8 * previous_buffer + gradient`, then parameters subtract `0.03 * buffer`. A weights-only control restores the same model but resets momentum. Its next forward pass matches, while subsequent updates must diverge. Model weights alone therefore do not establish this continuation.

[PyTorch's optimizer documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html) explains that saved parameter IDs are associated with current parameters by order, without additional verification. The scorer and prediction-head weights both have shape `[1,4]`, so shape checks would miss a swap. This consumer admits exact named order, ordinal IDs, options, cursor, fixture and source identities before calling either load method. It also checks that the live optimizer owns those same parameter objects in that order.

Public `model.load_state_dict` and `optimizer.load_state_dict` perform restoration. The pinned [PyTorch optimizer source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/optimizer.py#L761-L792) casts floating optimizer state to each live parameter's device and dtype. The consumer verifies the resulting CUDA0 buffers, values and independent storage after loading. It preserves existing parameter objects rather than silently replacing the optimizer's targets.

## Run from an installed environment

Use the companion's qualified Windows x64, CPython3.11.8, `torch==2.9.1+cu128`, CUDA runtime12.8, RTX4090 route from [README](README.md). Install its ordinary wheel into a separate isolated environment. Copy `resume_training.py`, `train_compiled.py` and `train_cuda.py` from a checked source archive to the same external consumer directory. The first file loads the two helpers explicitly. The installed `ragged_pool_cuda` package supplies the native operation.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MAX_JOBS = '2'
& $acceptPython -I C:\work\cuda-consumer\resume_training.py --dtype float32 --directory C:\work\resume-f32 --output C:\work\resume-f32.json
& $acceptPython -I C:\work\cuda-consumer\resume_training.py --dtype float64 --directory C:\work\resume-f64 --output C:\work\resume-f64.json
```

Each artifact directory must be new. A receipt identifies the actual processes, installed files, source/helper hashes, checkpoint digest, fixture, comparisons and optimizer-reset control. Optional `--source-revision` and `--source-tree` labels accept complete Git SHA1 identifiers. Those caller labels are accompanied by actual file hashes; a label by itself is not proof of source identity. Each child has a120-second timeout and retained stdout/stderr logs. Tests run from the checked archive's external `tests` directory with the ordinary wheel installed.

## Trusted local state and limits

The example reads a bounded ordinary local file once, checks its expected SHA256 and passes those exact bytes to `torch.load(..., weights_only=True, map_location="cpu")`. It admits only its fixed schema, dtype, shape, independent storage and finite bounded tensors before applying state. Negative tests verify that malformed model, optimizer, cursor and storage associations stop before either load method, followed by successful restoration of the valid state.

The expected digest must come from the same trusted local workflow. It identifies bytes and does not authenticate a publisher. [PyTorch's serialization notes](https://docs.pytorch.org/docs/2.9/notes/serialization.html#torch-load-with-weights-only-true) explain that restricted loading reduces code-execution exposure while retaining denial-of-service and other risks. This consumer is not a sandbox for arbitrary checkpoints, an atomic persistence service or an operational security guarantee.

The fixture has no dropout, scheduler, shuffled loader or stochastic forward. It does not save or resume live RNG state. CPU stochastic recovery has a separate example and evidence. This CUDA route does not establish stochastic, compiled-cache, distributed or crash recovery, exact replay across devices or releases, cross-platform portability, model quality or performance. [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.9/notes/randomness.html) explicitly limits reproducibility across releases and platforms. Exact equality here is an executed result for the declared pin and fixture.
