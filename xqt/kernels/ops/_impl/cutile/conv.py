"""CuTile Conv operator references and guarded entry points."""

from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ._common import require_cuda_tensors, require_cutile, require_fp16_tensors
from .linear import dense_linear_epilogue_cutile


def _effective_tile_block(configured: int | None, extent: int, name: str) -> int:
    if extent <= 0:
        raise XQTBackendError(f"{name} extent must be positive")
    if configured is None or int(configured) <= 0:
        configured = 64
    configured = int(configured)
    if extent % configured == 0:
        return configured
    limit = min(configured, extent)
    for candidate in range(limit, 0, -1):
        if extent % candidate == 0:
            return candidate
    raise XQTBackendError(f"could not resolve a valid block size for {name}")


def conv2d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Reference Conv2d path used by operator-family runtime splitting."""

    return F.conv2d(
        x,
        weight.to(dtype=x.dtype, device=x.device),
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv2d_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only CuTile guarded Conv2d lowering path."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    require_cutile()
    if groups != 1:
        raise XQTBackendError("CuTile Conv2d path currently supports groups=1 only")
    if x.ndim != 4 or weight.ndim != 4:
        raise XQTBackendError("CuTile Conv2d path expects NCHW input and OIHW weight")
    if x.shape[1] != weight.shape[1]:
        raise XQTBackendError(
            "CuTile Conv2d path requires x.shape[1] == weight.shape[1]"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise XQTBackendError("CuTile Conv2d path expects bias shaped [out_channels]")
    if (
        stride == (1, 1)
        and padding == (0, 0)
        and dilation == (1, 1)
        and weight.shape[2:] == (1, 1)
    ):
        flat_input = x.permute(0, 2, 3, 1).reshape(-1, int(x.shape[1]))
        flat_weight = weight.reshape(int(weight.shape[0]), int(weight.shape[1]))
        flat_output = dense_linear_epilogue_cutile(
            flat_input,
            flat_weight,
            bias,
            activation=None,
            block_m=_effective_tile_block(
                block_m, int(flat_input.shape[0]), "conv flattened rows"
            ),
            block_n=_effective_tile_block(
                block_n, int(flat_weight.shape[0]), "conv out_channels"
            ),
            block_k=_effective_tile_block(
                block_k, int(flat_input.shape[1]), "conv reduction"
            ),
            threads=threads,
            target_arch=target_arch,
        )
        return (
            flat_output.reshape(
                int(x.shape[0]), int(x.shape[2]), int(x.shape[3]), int(weight.shape[0])
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )
    return conv2d_reference(
        x,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


CUTILE_CONV_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "conv": {
        "kernel_name": "conv2d_cutile_half_gemm",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "baseline": "torch.nn.functional.conv2d",
        "usage": (
            "Standalone half Conv2d path aligned with TileLang coverage; 1x1 NCHW "
            "routes through dense half GEMM metadata and other shapes use eager reference fallback."
        ),
        "weight_encoding": "dense_fp16",
        "fusion_status": "cutile_conv_reference_guarded",
        "fastpath": "cutile_conv1x1_nchw_half_gemm_guard",
        "fallback": "torch_conv2d_reference",
        "supports_grouped_conv": False,
        "epilogue_stage": None,
        "production_status": "reference_guarded",
    },
}

__all__ = [
    "CUTILE_CONV_KERNEL_METADATA",
    "conv2d_cutile",
    "conv2d_reference",
]
