# Continue training on the other device

`examples/resume_backends.py` is a bounded deterministic CPU/CUDA training handoff consumer for the existing coinstalled wheel pair. It saves the fixed model and SGD momentum after completed update two, exits its producer, then restores a new model and optimizer in a fresh target process and takes two further updates. It qualifies CPU-to-CUDA0 and CUDA0-to-CPU separately for float32 and float64. The two changing packed batches come from `train_backends.py`, which must stay beside it in an external consumer folder.

The intended route checks exact restored tensor values and target device/dtype/ownership separately from tolerant native and independent primitive numerical continuation. Device copies belong explicitly to this consumer. Pooling operations keep their original input/device contracts. This example adds no runtime API, native kernel or checkpoint service.

## Qualification pending

The candidate [result record](results/cross-device-resume.json) is pending. This checkpoint makes no successful new handoff claim. Its intended pin is the already qualified Windows x64 / CPython3.11.8 / Torch2.9.1+cu128 / RTX4090 environment and literal accepted CPU1cu128/CUDA wheels. No new native compile or instrumentation is planned.

Checkpoint admission concerns trusted local artifacts. A digest identifies bytes, rather than authenticating a publisher. `weights_only=True`, bounded files and schema checks do not make hostile checkpoints safe. Known validation refusals are checked before either model/optimizer load. Arbitrary exceptions during an applied load have no rollback guarantee. Stochastic RNG recovery, arbitrary optimizers/models, distributed training, AMP, broad hardware portability, security hardening, performance and model quality remain outside this route.
