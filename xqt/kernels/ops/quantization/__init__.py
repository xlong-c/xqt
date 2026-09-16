"""quantization kernels unified with sglang.kernels."""
from __future__ import annotations

from typing import Any

import torch

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import CapabilityRequirement, FormatSignature, KernelBackend, KernelSpec
from xqt.kernels.ops._legacy_api import load_legacy
from xqt.kernels.ops.quantization.nvfp4 import (
    expand_group_scale,
    normalize_group_scale,
    unpack_nvfp4e2m1,
)
from xqt.kernels.ops.quantization.convrot_w8a8 import (
    ConvRotW8A8Workspace,
    PackedConvRotW8A8Linear,
    allocate_convrot_w8a8_workspace,
    bind_convrot_w8a8_linear,
    convrot_w8a8_linear,
    fused_swiglu,
    native_convrot_w8a8_available,
    native_convrot_w8a8_shape_supported,
    native_convrot_w8a8_version,
    pack_convrot_w8a8_linear,
)
from xqt.kernels.ops.quantization.svdq_w8a8 import (
    PackedSVDQW8A8Linear,
    W8A8SVDQWorkspace,
    allocate_svdq_w8a8_workspace,
    bind_svdq_w8a8_linear,
    native_svdq_w8a8_available,
    native_svdq_w8a8_shape_supported,
    native_svdq_w8a8_version,
    pack_svdq_w8a8_linear,
    svdq_w8a8_linear,
    w8a8_linear,
)
from xqt.kernels.ops.quantization.awq_w4a16 import (
    awq_w4a16_decode,
    awq_w4a16_decode_bias,
    bind_awq_w4a16_decode,
    native_awq_w4a16_available,
    native_awq_w4a16_version,
    pack_awq_w4a16_interleaved,
)
from xqt.kernels.ops.quantization.convrot_w4a4_rowwise import (
    ConvRotW4A4RowwiseWorkspace,
    PackedConvRotW4A4Rowwise,
    allocate_convrot_w4a4_rowwise_workspace,
    bind_convrot_w4a4_rowwise_linear,
    bind_dynamic_convrot_w4a4_rowwise_linear,
    convrot_w4a4_rowwise_linear,
    native_rowwise_convrot_w4a4_available,
    native_rowwise_convrot_w4a4_shape_supported,
    native_rowwise_convrot_w4a4_version,
    pack_convrot_w4a4_rowwise_weight,
)
from xqt.kernels.ops.quantization.svdq_w4a4 import (
    PackedSVDQW4A4Linear,
    PackedW4A4Linear,
    SVDQW4A4GeluMLPWorkspace,
    W4A4Workspace,
    allocate_svdq_w4a4_gelu_mlp_workspace,
    allocate_w4a4_workspace,
    bind_convrot_w4a4_linear,
    bind_svdq_w4a4_linear,
    bind_svdq_w4a4_linear_norm,
    bind_svdq_w4a4_linear_smalln,
    bind_svdq_w4a4_linear_smalln_norm,
    bind_svdq_w4a4_qkv_rmsnorm_rope,
    convrot_w4a4_linear,
    native_convrot_w4a4_shape_supported,
    native_w4a4_available,
    native_w4a4_shape_supported,
    native_w4a4_smalln_available,
    native_w4a4_smalln_version,
    native_w4a4_version,
    pack_lowrank_weight,
    pack_scale,
    pack_svdq_w4a4_linear,
    pack_svdq_w4a4_linear_smalln,
    pack_svdq_w4a4_rotary_emb,
    pack_w4a4_linear,
    svdq_w4a4_gelu_mlp,
    svdq_w4a4_linear,
    svdq_w4a4_linear_norm,
    svdq_w4a4_linear_smalln,
    svdq_w4a4_linear_smalln_norm,
    w4a4_linear,
)


_CUDA = frozenset({CapabilityRequirement.CUDA})
_SM89 = frozenset({CapabilityRequirement.cuda(min_sm=(8, 9), max_sm=(8, 9))})

def _per_token_quant_fp8_torch(x):
    import torch
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 448.0
    q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16)
    return q, scale

register_kernel(KernelSpec(op="quantization.per_token_quant_fp8", backend=KernelBackend.TORCH, target="xqt.kernels.ops.quantization:_per_token_quant_fp8_torch", format_signature=FormatSignature(description="per token quant fp8")))
register_kernel(
    KernelSpec(
        op="quantization.svd_fused_dequant_gemm_low_rank",
        backend=KernelBackend.TILELANG,
        target="xqt.kernels.ops._impl.tilelang.svd_fused:svd_fused_dequant_gemm_low_rank_tilelang",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("float16",),
            description="SVDQuant fused dequant GEMM with low-rank branch",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="quantization.convrot_w8a8_linear",
        backend=KernelBackend.TVM_FFI,
        target="xqt.kernels.ops.quantization.convrot_w8a8:convrot_w8a8_linear",
        capabilities=_SM89,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16", "float16"),
            description="ConvRot W8A8 Linear via TVM FFI",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="quantization.svdq_w8a8_linear",
        backend=KernelBackend.TVM_FFI,
        target="xqt.kernels.ops.quantization.svdq_w8a8:svdq_w8a8_linear",
        capabilities=_SM89,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="SVDQ W8A8 Linear via TVM FFI",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="quantization.awq_w4a16_decode",
        backend=KernelBackend.TVM_FFI,
        target="xqt.kernels.ops.quantization.awq_w4a16:awq_w4a16_decode",
        capabilities=_SM89,
        format_signature=FormatSignature(
            supported_dtypes=("float16", "bfloat16"),
            description="AWQ W4A16 Decode GEMM via TVM FFI",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="quantization.convrot_w4a4_rowwise_linear",
        backend=KernelBackend.TVM_FFI,
        target="xqt.kernels.ops.quantization.convrot_w4a4_rowwise:convrot_w4a4_rowwise_linear",
        capabilities=_SM89,
        format_signature=FormatSignature(
            supported_dtypes=("float16", "bfloat16"),
            description="ConvRot W4A4 Rowwise Linear via TVM FFI",
        ),
    )
)
register_kernel(
    KernelSpec(
        op="quantization.svdq_w4a4_linear",
        backend=KernelBackend.TVM_FFI,
        target="xqt.kernels.ops.quantization.svdq_w4a4:svdq_w4a4_linear",
        capabilities=_SM89,
        format_signature=FormatSignature(
            supported_dtypes=("float16", "bfloat16"),
            description="SVDQ W4A4 Linear via TVM FFI",
        ),
    )
)

__all__ = [
    "_per_token_quant_fp8_torch",
    "expand_group_scale",
    "normalize_group_scale",
    "per_token_quant_fp8",
    "svd_fused_dequant_gemm_low_rank_tilelang",
    "unpack_nvfp4e2m1",
    # ConvRot W8A8
    "ConvRotW8A8Workspace",
    "PackedConvRotW8A8Linear",
    "allocate_convrot_w8a8_workspace",
    "bind_convrot_w8a8_linear",
    "convrot_w8a8_linear",
    "fused_swiglu",
    "native_convrot_w8a8_available",
    "native_convrot_w8a8_shape_supported",
    "native_convrot_w8a8_version",
    "pack_convrot_w8a8_linear",
    # SVDQ W8A8
    "PackedSVDQW8A8Linear",
    "W8A8SVDQWorkspace",
    "allocate_svdq_w8a8_workspace",
    "bind_svdq_w8a8_linear",
    "native_svdq_w8a8_available",
    "native_svdq_w8a8_shape_supported",
    "native_svdq_w8a8_version",
    "pack_svdq_w8a8_linear",
    "svdq_w8a8_linear",
    "w8a8_linear",
    # AWQ W4A16
    "awq_w4a16_decode",
    "awq_w4a16_decode_bias",
    "bind_awq_w4a16_decode",
    "native_awq_w4a16_available",
    "native_awq_w4a16_version",
    "pack_awq_w4a16_interleaved",
    # ConvRot W4A4 Rowwise
    "ConvRotW4A4RowwiseWorkspace",
    "PackedConvRotW4A4Rowwise",
    "allocate_convrot_w4a4_rowwise_workspace",
    "bind_convrot_w4a4_rowwise_linear",
    "bind_dynamic_convrot_w4a4_rowwise_linear",
    "convrot_w4a4_rowwise_linear",
    "native_rowwise_convrot_w4a4_available",
    "native_rowwise_convrot_w4a4_shape_supported",
    "native_rowwise_convrot_w4a4_version",
    "pack_convrot_w4a4_rowwise_weight",
    # SVDQ W4A4
    "PackedSVDQW4A4Linear",
    "PackedW4A4Linear",
    "SVDQW4A4GeluMLPWorkspace",
    "W4A4Workspace",
    "allocate_svdq_w4a4_gelu_mlp_workspace",
    "allocate_w4a4_workspace",
    "bind_convrot_w4a4_linear",
    "bind_svdq_w4a4_linear",
    "bind_svdq_w4a4_linear_norm",
    "bind_svdq_w4a4_linear_smalln",
    "bind_svdq_w4a4_linear_smalln_norm",
    "bind_svdq_w4a4_qkv_rmsnorm_rope",
    "convrot_w4a4_linear",
    "native_convrot_w4a4_shape_supported",
    "native_w4a4_available",
    "native_w4a4_shape_supported",
    "native_w4a4_smalln_available",
    "native_w4a4_smalln_version",
    "native_w4a4_version",
    "pack_lowrank_weight",
    "pack_scale",
    "pack_svdq_w4a4_linear",
    "pack_svdq_w4a4_linear_smalln",
    "pack_svdq_w4a4_rotary_emb",
    "pack_w4a4_linear",
    "svdq_w4a4_gelu_mlp",
    "svdq_w4a4_linear",
    "svdq_w4a4_linear_norm",
    "svdq_w4a4_linear_smalln",
    "svdq_w4a4_linear_smalln_norm",
    "w4a4_linear",
]


def __getattr__(name: str) -> Any:
    return load_legacy("quantization", name)

def per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from xqt.kernels.selector import get_kernel
    return get_kernel("quantization.per_token_quant_fp8", KernelBackend.TORCH)(x)


def svd_fused_dequant_gemm_low_rank_tilelang(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run the SVDQuant fused TileLang kernel through the unified registry."""

    from xqt.kernels.selector import get_kernel

    return get_kernel(
        "quantization.svd_fused_dequant_gemm_low_rank",
        KernelBackend.TILELANG,
    )(
        x,
        packed_weight,
        scale,
        down_weight,
        up_weight,
        bias,
        input_features=input_features,
        group_size=group_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )
