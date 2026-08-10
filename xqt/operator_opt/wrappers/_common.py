"""Minimal shared helpers used by 2+ wrapper modules.

Do not add dequant-only or build-only helpers here — those belong in their
respective modules.
"""

from __future__ import annotations

from typing import Any

import torch


def _matching_tensor_dtype_name(*tensors: torch.Tensor) -> str:
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) != 1:
        return "mixed"
    return str(next(iter(dtypes))).removeprefix("torch.")


def _resolved_target_arch(settings: dict[str, Any], x: torch.Tensor) -> str | None:
    target_arch = settings.get("target_arch")
    if isinstance(target_arch, str) and target_arch:
        return target_arch
    if x.is_cuda:
        major, minor = torch.cuda.get_device_capability(x.device)
        return f"sm_{major}{minor}"
    return None
