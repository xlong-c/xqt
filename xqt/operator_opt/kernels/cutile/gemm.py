"""CuTile dequantized GEMM operator references and guarded entry points."""

from typing import Any

import torch

from xqt.core.errors import XQTBackendError
from xqt.gemm import dense_gemm_reference

from ._common import require_cuda_tensors, require_cutile, require_fp16_tensors

_NVFP4_E2M1_CODEBOOK_VALUES: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _validate_dequant_gemm_inputs(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    activation: str | None,
) -> None:
    if x.ndim != 2 or qweight.ndim != 2:
        raise XQTBackendError(
            "CuTile dequant GEMM expects x and qweight to be 2D tensors"
        )
    if scale.ndim not in {1, 2}:
        raise XQTBackendError("CuTile dequant GEMM expects scale to be 1D or 2D")
    if x.shape[1] != qweight.shape[1]:
        raise XQTBackendError(
            "CuTile dequant GEMM requires x.shape[1] == qweight.shape[1]"
        )
    if scale.ndim == 1 and scale.shape[0] != qweight.shape[0]:
        raise XQTBackendError("1D scale must match qweight out_features")
    if scale.ndim == 2 and scale.shape != qweight.shape:
        raise XQTBackendError("2D scale must match qweight shape")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != qweight.shape[0]):
        raise XQTBackendError("bias must be 1D and match qweight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")


def dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference dequantized GEMM with optional bias and activation epilogue."""

    weight_scale = scale.to(dtype=x.dtype, device=x.device)
    if weight_scale.ndim == 1:
        weight_scale = weight_scale.unsqueeze(-1)
    weight = qweight.to(dtype=x.dtype, device=x.device) * weight_scale
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


def _unpack_low_high_nibbles(
    packed_weight: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    if packed_weight.dtype != torch.uint8:
        raise TypeError("packed_weight must be uint8")
    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(packed_weight.shape[0], -1)
    return unpacked[:, : int(input_features)]


def _decode_packed_signed_int4(
    packed_weight: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    unpacked = _unpack_low_high_nibbles(packed_weight, input_features=input_features)
    signed = torch.where(
        unpacked >= 8,
        unpacked.to(torch.int16) - 16,
        unpacked.to(torch.int16),
    )
    return signed.to(torch.float32)


def _nvfp4_e2m1_codebook(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        _NVFP4_E2M1_CODEBOOK_VALUES,
        dtype=torch.float32,
        device=device,
    )


def _decode_packed_nvfp4_e2m1(
    packed_weight: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    unpacked_codes = _unpack_low_high_nibbles(
        packed_weight,
        input_features=input_features,
    )
    return _nvfp4_e2m1_codebook(packed_weight.device)[unpacked_codes.long()]


def _validate_packed_fp4_inputs(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None,
) -> None:
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise XQTBackendError("packed FP4 CuTile path expects 2D x and packed_weight")
    if x.shape[1] != int(input_features):
        raise XQTBackendError(
            "packed FP4 CuTile path requires x.shape[1] == input_features"
        )
    if packed_weight.dtype != torch.uint8:
        raise XQTBackendError("packed FP4 CuTile path expects uint8 packed_weight")
    if (
        scale.ndim != 3
        or scale.shape[0] != packed_weight.shape[0]
        or scale.shape[2] != 1
    ):
        raise XQTBackendError(
            "packed FP4 CuTile path expects scale shaped [out_features, groups, 1]"
        )
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    if scale.shape[1] * int(group_size) != int(packed_weight.shape[1]) * 2:
        raise XQTBackendError(
            "scale groups must cover the packed padded input features"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != packed_weight.shape[0]):
        raise XQTBackendError("bias must be 1D and match packed_weight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")


def fp4_packed_dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference packed FP4 dequantized GEMM with optional epilogue."""

    padded_input_features = int(packed_weight.shape[1]) * 2
    qweight = _decode_packed_signed_int4(
        packed_weight.to(device=x.device),
        input_features=padded_input_features,
    ).to(dtype=x.dtype, device=x.device)
    weight_scale = scale.to(dtype=x.dtype, device=x.device)
    grouped = qweight.reshape(qweight.shape[0], -1, int(group_size))
    weight = (grouped * weight_scale).reshape(qweight.shape[0], padded_input_features)[
        :, : int(input_features)
    ]
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


def nvfp4_packed_dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference packed NVFP4 dequantized GEMM with optional epilogue."""

    padded_input_features = int(packed_weight.shape[1]) * 2
    qweight = _decode_packed_nvfp4_e2m1(
        packed_weight.to(device=x.device),
        input_features=padded_input_features,
    ).to(dtype=x.dtype, device=x.device)
    weight_scale = scale.to(dtype=x.dtype, device=x.device)
    grouped = qweight.reshape(qweight.shape[0], -1, int(group_size))
    weight = (grouped * weight_scale).reshape(qweight.shape[0], padded_input_features)[
        :, : int(input_features)
    ]
    if weight_global_scale is not None:
        weight = weight / weight_global_scale.to(dtype=x.dtype, device=x.device)
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


def dequant_gemm_epilogue_cutile(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only CuTile guarded dequant GEMM epilogue entry point."""

    del block_m, block_n, threads, target_arch
    tensors = (x, qweight, scale) if bias is None else (x, qweight, scale, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    _validate_dequant_gemm_inputs(
        x,
        qweight,
        scale,
        bias=bias,
        activation=activation,
    )
    require_cutile()
    return dequant_gemm_epilogue_reference(
        x, qweight, scale, bias, activation=activation
    )


def fp4_packed_dequant_gemm_epilogue_cutile(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only CuTile guarded packed FP4 dequant GEMM entry point."""

    del block_m, block_n, threads, target_arch
    tensors = (
        (x, packed_weight, scale) if bias is None else (x, packed_weight, scale, bias)
    )
    require_cuda_tensors(*tensors)
    if x.dtype != torch.float16 or scale.dtype != torch.float16:
        raise XQTBackendError(
            "packed FP4 CuTile path currently requires float16 x and scale"
        )
    if bias is not None and bias.dtype != torch.float16:
        raise XQTBackendError("packed FP4 CuTile path currently requires float16 bias")
    _validate_packed_fp4_inputs(
        x,
        packed_weight,
        scale,
        bias,
        input_features=input_features,
        group_size=group_size,
        activation=activation,
    )
    require_cutile()
    return fp4_packed_dequant_gemm_epilogue_reference(
        x,
        packed_weight,
        scale,
        bias,
        input_features=input_features,
        group_size=group_size,
        activation=activation,
    )


def nvfp4_packed_dequant_gemm_epilogue_cutile(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 16,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only CuTile guarded packed NVFP4 dequant GEMM entry point."""

    del block_m, block_n, threads, target_arch
    tensors = (
        (x, packed_weight, scale) if bias is None else (x, packed_weight, scale, bias)
    )
    require_cuda_tensors(*tensors)
    if x.dtype != torch.float16 or scale.dtype != torch.float16:
        raise XQTBackendError(
            "packed NVFP4 CuTile path currently requires float16 x and scale"
        )
    if bias is not None and bias.dtype != torch.float16:
        raise XQTBackendError(
            "packed NVFP4 CuTile path currently requires float16 bias"
        )
    _validate_packed_fp4_inputs(
        x,
        packed_weight,
        scale,
        bias,
        input_features=input_features,
        group_size=group_size,
        activation=activation,
    )
    require_cutile()
    return nvfp4_packed_dequant_gemm_epilogue_reference(
        x,
        packed_weight,
        scale,
        bias,
        input_features=input_features,
        group_size=group_size,
        weight_global_scale=weight_global_scale,
        activation=activation,
    )


CUTILE_DEQUANT_GEMM_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "dequant_gemm_epilogue": {
        "kernel_name": "dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "threads": 128,
        "baseline": "torch.matmul(dequantized_weight.T) + epilogue",
        "usage": "Reference-guarded dequant GEMM epilogue aligned with TileLang coverage.",
        "weight_encoding": "dense_int_or_fp_quantized",
        "fusion_status": "cutile_dequant_gemm_epilogue_reference_guarded",
        "epilogue_stage": "torch_bias_activation",
        "production_status": "reference_guarded",
    },
    "fp4_packed_dequant_gemm_epilogue": {
        "kernel_name": "fp4_packed_dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "threads": 128,
        "baseline": "torch unpack/dequant + matmul + epilogue",
        "usage": "Packed signed-int4 FP4 dequant GEMM planning path for CuTile.",
        "weight_encoding": "packed_signed_int4",
        "unpack_stage": "cutile_metadata_only",
        "fusion_status": "cutile_packed_fp4_reference_guarded",
        "epilogue_stage": "torch_bias_activation",
        "production_status": "reference_guarded",
    },
    "nvfp4_packed_dequant_gemm_epilogue": {
        "kernel_name": "nvfp4_packed_dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 16,
        "threads": 128,
        "baseline": "torch NVFP4 unpack/dequant + matmul + epilogue",
        "usage": "Packed NVFP4 E2M1 dequant GEMM planning path for CuTile.",
        "weight_encoding": "packed_nvfp4_e2m1",
        "unpack_stage": "cutile_metadata_only",
        "fusion_status": "cutile_packed_nvfp4_reference_guarded",
        "epilogue_stage": "torch_bias_activation",
        "production_status": "reference_guarded",
    },
}

__all__ = [
    "CUTILE_DEQUANT_GEMM_KERNEL_METADATA",
    "dequant_gemm_epilogue_cutile",
    "dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_cutile",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_cutile",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
]
