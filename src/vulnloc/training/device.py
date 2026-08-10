"""Device selection and accelerator reporting (``docs/Architecture.md`` §8).

`Rules.md` §2 bans hard-coded values, so no module may reference ``"cuda"`` directly:
the device is resolved once, from config, and passed down. ``auto`` picks CUDA when it
is genuinely usable and falls back to CPU otherwise.

The fallback is **loud**. A run that silently drops to CPU looks like a slow run rather
than a broken environment, and on this project's hardware that difference is hours.
Set ``device: cuda`` explicitly to make an unusable GPU a hard failure instead.
"""

from __future__ import annotations

from typing import Any

import torch

from vulnloc.utils.exceptions import ConfigError
from vulnloc.utils.logging import get_logger

__all__ = ["accelerator_info", "log_accelerator", "resolve_device"]

_LOGGER = get_logger(__name__)

_BYTES_PER_GIB = 1024**3
_VALID_KINDS = ("auto", "cuda", "cpu")


def _cuda_is_usable() -> bool:
    """Return True only if a CUDA device is present *and* a tensor can reach it.

    ``torch.cuda.is_available()`` returns True for a driver/runtime pair that then
    fails on first allocation (a CPU-only wheel is the common case, but so is a stale
    driver). Probing with a one-element tensor turns that into an answer we can trust.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False
    try:
        torch.zeros(1, device="cuda")
    except (RuntimeError, AssertionError) as exc:  # driver/runtime mismatch, OOM at init
        _LOGGER.warning("CUDA reported available but unusable", extra={"reason": str(exc)})
        return False
    return True


def resolve_device(kind: str = "auto", *, index: int = 0) -> torch.device:
    """Resolve a config device string into a concrete :class:`torch.device`.

    Args:
        kind: One of ``"auto"``, ``"cuda"``, ``"cpu"``.
        index: CUDA device ordinal, ignored for CPU.

    Returns:
        The device to place models and batches on.

    Raises:
        ConfigError: If ``kind`` is not recognized, if ``cuda`` was demanded but is
            unusable, or if ``index`` is out of range.
    """
    normalized = kind.strip().lower()
    if normalized not in _VALID_KINDS:
        raise ConfigError(
            f"Unknown device {kind!r}. Expected one of {list(_VALID_KINDS)}.", device=kind
        )

    if normalized == "cpu":
        return torch.device("cpu")

    usable = _cuda_is_usable()
    if normalized == "cuda" and not usable:
        raise ConfigError(
            "device=cuda was requested but no usable CUDA device is present. "
            "Install a CUDA build of torch (the default PyPI Windows wheel is CPU-only) "
            "or set device=auto to fall back to CPU.",
            torch_version=torch.__version__,
            cuda_build=torch.version.cuda,
        )
    if not usable:
        _LOGGER.warning(
            "No usable CUDA device; falling back to CPU. Training will be substantially "
            "slower. Set device=cuda to make this a hard failure.",
            extra={"torch_version": torch.__version__, "cuda_build": torch.version.cuda},
        )
        return torch.device("cpu")

    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ConfigError(
            f"CUDA device index {index} is out of range; {count} device(s) visible.",
            index=index,
            device_count=count,
        )
    return torch.device("cuda", index)


def accelerator_info(device: torch.device) -> dict[str, Any]:
    """Describe ``device`` for run logging and MLflow tags.

    Args:
        device: The resolved device.

    Returns:
        A mapping with ``device``, ``torch_version``, ``cuda_build`` and, for CUDA,
        the GPU name, total VRAM in GiB, and compute capability.
    """
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda or "cpu",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            gpu_name=properties.name,
            gpu_total_gib=round(properties.total_memory / _BYTES_PER_GIB, 2),
            gpu_capability=f"{properties.major}.{properties.minor}",
        )
    return info


def log_accelerator(device: torch.device) -> dict[str, Any]:
    """Log the accelerator description at INFO and return it.

    Args:
        device: The resolved device.

    Returns:
        The same mapping as :func:`accelerator_info`, for logging as MLflow tags.
    """
    info = accelerator_info(device)
    _LOGGER.info("Accelerator", extra=info)
    return info
