"""CUTLASS Python operator optimization references and guarded entry points."""

from __future__ import annotations

from typing import Any

import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import dense_gemm_reference


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("CUTLASS kernels require CUDA tensors")


def _require_cutlass() -> object:
    try:
        import cutlass
    except ImportError as exc:
        raise XQTBackendError(
            "cutlass is required for CUTLASS Python operator kernels. Install the CUTLASS Python extras."
        ) from exc
    return cutlass


def gemm_epilogue_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference GEMM with optional bias and activation epilogue."""

    runtime_weight = weight.to(dtype=x.dtype, device=x.device)
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        runtime_weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


def gemm_epilogue_cutlass(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    tile_shape: tuple[int, int, int] = (128, 128, 64),
) -> torch.Tensor:
    """CUDA-only CUTLASS Python placeholder entry point for GEMM epilogue."""

    del tile_shape
    tensors = (x, weight) if bias is None else (x, weight, bias)
    _require_cuda_tensors(*tensors)
    _require_cutlass()
    return gemm_epilogue_reference(x, weight, bias, activation=activation)


CUTLASS_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "gemm_epilogue": {
        "kernel_name": "gemm_epilogue",
        "tile_shape": [128, 128, 64],
        "baseline": "torch.matmul + epilogue",
        "usage": "Linear-heavy path where CUTLASS Python can own GEMM scheduling and epilogue fusion.",
        "production_status": "reference_guarded",
    },
    "grouped_gemm": {
        "kernel_name": "grouped_gemm",
        "tile_shape": [128, 128, 64],
        "baseline": "loop(torch.matmul)",
        "usage": "Many small GEMMs with compatible dtype/layout, common in MoE or batched adapter paths.",
        "production_status": "metadata_only",
    },
}


__all__ = [
    "CUTLASS_KERNEL_METADATA",
    "gemm_epilogue_cutlass",
    "gemm_epilogue_reference",
]
