"""Describe the two explicit Torch builds admitted by the CPU extension."""

CONFIG_FILENAME = "_build_config.json"


def build_config(variant):
    """Return the wheel's declared framework pin, without importing Torch."""
    if type(variant) is not str or variant not in ("cpu", "cu128"):
        raise RuntimeError("RAGGED_POOL_TORCH_VARIANT must be cpu or cu128")
    return {
        "schema": 1,
        "variant": variant,
        "torch_version": "2.9.1+" + variant,
        "cuda_version": None if variant == "cpu" else "12.8",
        "wheel_build_tag": "" if variant == "cpu" else "1cu128",
    }


def validate_config(config):
    """Refuse missing, stale or extended generated build metadata."""
    if type(config) is not dict or type(config.get("schema")) is not int:
        raise RuntimeError("Invalid ragged-pool build configuration")
    expected = build_config(config.get("variant"))
    if config != expected:
        raise RuntimeError("Invalid ragged-pool build configuration")
    return expected


def validate_runtime(config, torch_version, cuda_version):
    """Reject a different framework build before loading native code."""
    expected = validate_config(config)
    if torch_version != expected["torch_version"] or cuda_version != expected["cuda_version"]:
        raise RuntimeError(
            "This ragged-pool wheel requires torch==" + expected["torch_version"]
            + " with CUDA runtime " + str(expected["cuda_version"])
            + ". Install the matching wheel and framework build."
        )
    return expected
