# Install the CPU and CUDA packages together

The default CPU distribution requires `torch==2.9.1+cpu`. The CUDA companion requires `torch==2.9.1+cu128`. Those exact requirements cannot share an environment. Do not bypass the conflict with `--no-deps` when installing the two ordinary distributions.

An explicit CPU build variant uses `RAGGED_POOL_TORCH_VARIANT=cu128`, compiles the CPU extension against that exact framework, and produces `ragged_pool-0.1.0-1cu128-cp311-cp311-win_amd64.whl`. It still runs its pooling operations on CPU. The wheel build tag identifies its framework dependency separately from the project version. Both installed distributions then require the same cu128 framework. The CPU-only default and its wheel name remain unchanged.

The build checks the selected Torch version and CUDA runtime before importing the extension builder. Its generated `_build_config.json` lives in the wheel, rather than the source tree. Import checks that configuration against the active framework before loading the CPU native library. Native builds force recompilation so switching flavors cannot reuse an earlier object file. These checks are a compatibility boundary, not authentication of untrusted packages or a cross-version ABI guarantee.

## Qualification in progress

The candidate's [execution record](results/coinstallation-cu128.json) is pending. No successful new native build, co-installation or consumer result is claimed at this source checkpoint. The intended qualification is one Windows x64 / CPython 3.11.8 / Torch 2.9.1+cu128 / MSVC 14.44 route, with the unchanged accepted CUDA companion on the existing RTX 4090. CPU source compilation uses C++17 and the ATen/dispatcher API. No CUDA kernel change, new CUDA compiler result, performance result or broader platform compatibility is implied.

`examples/train_backends.py` supplies two actual SGD updates through CPU native, CUDA native, CPU primitive and CUDA primitive routes. It checks complete predictions, pools, losses, raw-input and parameter gradients, and updated parameters for float32 and float64. It uses explicit device copies for its fixtures and model parameters. The pooling APIs themselves do not move inputs between devices. The installed example requires both exact wheel paths to join the loaded packages to their artifact bytes.

Existing CPU exported and training-resume artifacts retain their original `+cpu` environment admission. This opt-in package variant does not broaden those saved-artifact contracts. No AMP, cross-device optimizer migration, distributed training, security hardening or model-quality result is added.
