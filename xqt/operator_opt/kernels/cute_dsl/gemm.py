"""CuTe DSL operator optimization references and guarded entry points."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("CuTe DSL kernels require CUDA tensors")


def _require_cute_dsl() -> object:
    try:
        import cutlass.cute as cute
    except ImportError as exc:
        raise XQTBackendError(
            "cutlass.cute is required for CuTe DSL operator kernels. Install the CUTLASS Python DSL extras."
        ) from exc
    return cute


def gemm_epilogue_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference GEMM with optional bias and activation epilogue."""

    output = x.matmul(weight.t().to(dtype=x.dtype, device=x.device))
    if bias is not None:
        output = output + bias.to(dtype=output.dtype, device=output.device)
    if activation is None:
        return output
    if activation == "gelu":
        return F.gelu(output)
    if activation == "silu":
        return F.silu(output)
    if activation == "relu":
        return F.relu(output)
    raise ValueError(f"unsupported activation: {activation}")


def gemm_epilogue_cute_dsl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    tile_shape: tuple[int, int, int] = (128, 128, 64),
    cluster_shape: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """CUDA-only CuTe DSL placeholder entry point for GEMM epilogue."""

    del tile_shape, cluster_shape
    tensors = (x, weight) if bias is None else (x, weight, bias)
    _require_cuda_tensors(*tensors)
    _require_cute_dsl()
    return gemm_epilogue_reference(x, weight, bias, activation=activation)


CUTE_DSL_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "gemm_epilogue": {
        "kernel_name": "gemm_epilogue",
        "tile_shape": [128, 128, 64],
        "cluster_shape": None,
        "baseline": "torch.matmul + epilogue",
        "usage": "Linear-heavy path where CuTe DSL can own GEMM tiling and epilogue fusion.",
        "production_status": "reference_guarded",
    },
    "grouped_gemm": {
        "kernel_name": "grouped_gemm",
        "tile_shape": [128, 128, 64],
        "cluster_shape": None,
        "baseline": "loop(torch.matmul)",
        "usage": "Many small GEMMs with compatible dtype/layout, common in MoE or adapter paths.",
        "production_status": "metadata_only",
    },
}


__all__ = [
    "CUTE_DSL_KERNEL_METADATA",
    "gemm_epilogue_cute_dsl",
    "gemm_epilogue_reference",
]
