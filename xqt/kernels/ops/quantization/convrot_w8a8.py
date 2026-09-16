"""ConvRot W8A8 quantization ops aligned with sglang.kernels."""

from __future__ import annotations

from xqt.kernels.ops._impl.cute.convrot_w8a8_sm89 import (
    ConvRotW8A8Workspace,
    PackedConvRotW8A8Linear,
    allocate_convrot_w8a8_workspace,
    bind_convrot_w8a8_linear,
    convrot_w8a8_linear,
    fused_swiglu,
    gemm_convrot_w8a8_sm89,
    native_convrot_w8a8_available,
    native_convrot_w8a8_shape_supported,
    native_convrot_w8a8_version,
    pack_convrot_w8a8_linear,
    quantize_rotated_activation_sm89,
)

__all__ = [
    "ConvRotW8A8Workspace",
    "PackedConvRotW8A8Linear",
    "allocate_convrot_w8a8_workspace",
    "bind_convrot_w8a8_linear",
    "convrot_w8a8_linear",
    "fused_swiglu",
    "gemm_convrot_w8a8_sm89",
    "native_convrot_w8a8_available",
    "native_convrot_w8a8_shape_supported",
    "native_convrot_w8a8_version",
    "pack_convrot_w8a8_linear",
    "quantize_rotated_activation_sm89",
]
