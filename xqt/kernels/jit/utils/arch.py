"""CUDA architecture helpers for the optional XQT JIT toolchain."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_ARCH_RE = re.compile(
    r"^(?:sm_|compute_)?(?P<major>\d+)[._]?(?P<minor>\d+)"
    r"(?P<variant>[af])?(?:\+ptx)?$"
)


def normalize_cuda_arch(value: str | int | tuple[int, int]) -> str:
    """Return a canonical ``sm_<major><minor>`` architecture string."""

    if isinstance(value, bool):
        raise ValueError("CUDA architecture must not be bool")
    if isinstance(value, tuple):
        if len(value) != 2 or any(
            isinstance(part, bool) or not isinstance(part, int) or part < 0
            for part in value
        ):
            raise ValueError("CUDA architecture tuple must be (major, minor)")
        major, minor = value
        if minor > 9:
            raise ValueError("CUDA architecture minor must be between 0 and 9")
        return f"sm_{major}{minor}"
    text = str(value).strip().lower()
    match = _ARCH_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid CUDA architecture: {value!r}")
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    if minor > 9:
        raise ValueError(f"invalid CUDA architecture: {value!r}")
    variant = match.group("variant") or ""
    return f"sm_{major}{minor}{variant}"


def make_jit_cuda_arch(value: str | int | tuple[int, int]) -> str:
    """Return the ``TORCH_CUDA_ARCH_LIST`` spelling for one architecture."""

    normalized = normalize_cuda_arch(value)
    body = normalized[3:]
    variant = body[-1] if body[-1] in "af" else ""
    digits = body[:-1] if variant else body
    return f"{digits[:-1]}.{digits[-1]}{variant}"


def get_jit_cuda_arch(default: str | int | tuple[int, int] | None = None) -> str | None:
    """Resolve an explicit architecture, environment value, or CUDA device."""

    if default is not None:
        return normalize_cuda_arch(default)
    configured = os.environ.get("XQT_JIT_CUDA_ARCH") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured:
        first = configured.replace(",", " ").replace(";", " ").split()[0]
        return normalize_cuda_arch(first)
    try:
        import torch

        if torch.cuda.is_available():
            return normalize_cuda_arch(torch.cuda.get_device_capability())
    except (ImportError, RuntimeError):
        return None
    return None


def cuda_target_arch(tensor: torch.Tensor) -> str | None:
    """Return the runtime ``sm_*`` architecture for a CUDA tensor."""

    if not tensor.is_cuda:
        return None
    import torch

    major, minor = torch.cuda.get_device_capability(tensor.device)
    return f"sm_{major}{minor}"


def target_arch_mismatch(
    requested: str | None,
    tensor: torch.Tensor,
) -> str | None:
    """Describe a requested/runtime SM mismatch, if one exists."""

    if not requested or not tensor.is_cuda:
        return None
    actual = cuda_target_arch(tensor)
    if actual is None or str(requested) == actual:
        return None
    return f"target_arch_mismatch:{requested}!={actual}"


def architecture_env(value: str | int | tuple[int, int] | None) -> dict[str, str]:
    """Build the isolated architecture environment for an extension build."""

    arch = get_jit_cuda_arch(value)
    return {} if arch is None else {"TORCH_CUDA_ARCH_LIST": make_jit_cuda_arch(arch)}


__all__ = [
    "architecture_env",
    "cuda_target_arch",
    "get_jit_cuda_arch",
    "make_jit_cuda_arch",
    "normalize_cuda_arch",
    "target_arch_mismatch",
]
