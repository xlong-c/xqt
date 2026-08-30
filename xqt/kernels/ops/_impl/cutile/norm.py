"""CuTile Norm operator references and guarded entry points."""

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ._common import require_cuda_tensors, require_cutile, require_fp16_tensors


def layer_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference LayerNorm path used by the CuTile backend."""

    normalized_shape = (int(weight.shape[0]),)
    ref_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return F.layer_norm(
        x,
        normalized_shape,
        weight.to(dtype=x.dtype, device=x.device),
        ref_bias,
        eps=float(eps),
    )


def layer_norm_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
    threads: int = 64,
) -> torch.Tensor:
    """CUDA-only CuTile guarded half LayerNorm path."""

    del threads
    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    if x.shape[-1] != weight.shape[0]:
        raise XQTBackendError(
            "CuTile LayerNorm path requires x.shape[-1] == weight.shape[0]"
        )
    if bias is not None and bias.shape != weight.shape:
        raise XQTBackendError(
            "CuTile LayerNorm path requires bias and weight to share shape"
        )
    require_cutile()
    return layer_norm_reference(x, weight, bias, eps=eps)


CUTILE_NORM_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "norm": {
        "kernel_name": "layer_norm_half",
        "threads": 64,
        "baseline": "torch.nn.functional.layer_norm",
        "usage": "Standalone half LayerNorm path for direct CuTile operator planning.",
        "weight_encoding": "dense_fp16",
        "fusion_status": "cutile_layer_norm_reference_guarded",
        "normalized_last_dim_only": True,
        "epilogue_stage": None,
        "production_status": "reference_guarded",
    },
}

__all__ = [
    "CUTILE_NORM_KERNEL_METADATA",
    "layer_norm_cutile",
    "layer_norm_reference",
]
