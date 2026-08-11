"""Minimal shared helpers used by 2+ wrapper modules.

Do not add dequant-only or build-only helpers here — those belong in their
respective modules.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError


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


def _scaled_dot_product_attention_with_causal_semantics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """Run SDPA with lower-right masking for non-square causal inputs."""

    if causal and q.size(-2) != k.size(-2):
        try:
            from torch.nn.attention.bias import causal_lower_right
        except Exception as exc:
            raise XQTBackendError(
                "non-square causal attention requires "
                "torch.nn.attention.bias.causal_lower_right"
            ) from exc
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_lower_right(q.size(-2), k.size(-2)),
            dropout_p=dropout_p,
        )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
    )
