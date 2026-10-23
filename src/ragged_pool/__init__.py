"""Stable weighted pooling for bounded packed CPU groups."""
from pathlib import Path
import json

import torch

from ._build import CONFIG_FILENAME, validate_runtime

# This metadata is generated into the wheel, rather than inferred from the
# current environment. A CPU-built wheel must not silently load under cu128.
try:
    _build_config = json.loads((Path(__file__).parent / CONFIG_FILENAME).read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ImportError("ragged-pool needs its generated build configuration. Install the wheel first.") from error
validate_runtime(_build_config, str(torch.__version__), torch.version.cuda)

# Load dispatcher registrations directly rather than invoking a pybind11 API.
_libraries = list(Path(__file__).parent.glob("_C*.pyd")) + list(Path(__file__).parent.glob("_C*.so"))
if len(_libraries) != 1:
    raise ImportError("ragged-pool needs exactly one built native library. Install the wheel first.")
torch.ops.load_library(str(_libraries[0]))

from . import _registration  # noqa: E402,F401
from .reference import ragged_pool_reference  # noqa: E402
from .padded import ragged_pool_padded, ragged_pool_padded_reference  # noqa: E402
from . import _padded_registration  # noqa: E402,F401
from .forward_ad import ragged_pool_forward_ad  # noqa: E402

__version__ = "0.1.0"
__all__ = ["ragged_pool", "ragged_pool_reference", "ragged_pool_padded", "ragged_pool_padded_reference",
           "ragged_pool_forward_ad"]


def ragged_pool(values, logits, offsets):
    """Pool contiguous CPU values [N,D] using logits [N] and int64 offsets [B+1]."""
    return torch.ops.ragged_pool.pool.default(values, logits, offsets)
