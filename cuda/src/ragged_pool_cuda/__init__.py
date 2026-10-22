"""Bounded CUDA pooling with explicit first-order reverse autograd."""
from pathlib import Path

import torch


_libraries = list(Path(__file__).parent.glob("_C*.pyd")) + list(Path(__file__).parent.glob("_C*.so"))
if len(_libraries) != 1:
    raise ImportError("ragged-pool-cuda needs exactly one native library. Install its CUDA wheel first.")
torch.ops.load_library(str(_libraries[0]))

from . import _registration  # noqa: E402,F401
from .reference import ragged_pool_reference  # noqa: E402
from .padded import ragged_pool_padded, ragged_pool_padded_reference  # noqa: E402

__version__ = "0.1.0"
__all__ = ["ragged_pool", "ragged_pool_reference", "ragged_pool_padded", "ragged_pool_padded_reference"]


def ragged_pool(values, logits, offsets):
    """Pool contiguous same-device CUDA [N,D], [N] and int64 [B+1] tensors.

    Native content admission transfers one four-byte flag to the host and
    synchronizes the current stream before offset-indexed work. Native admission
    checks real inputs, and registered fake metadata checks tracing inputs.
    This route supports first-order reverse differentiation only.
    """
    return torch.ops.ragged_pool_cuda.pool.default(values, logits, offsets)
