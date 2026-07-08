"""TileLang dequantized GEMM operator references and guarded entry points."""

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)
from xqt.operator_opt.kernels.tilelang.gemm_builder import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
    build_tilelang_gemm_kernel,
    build_tilelang_nvfp4_fused_dequant_gemm_kernel,
)

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
    block_m: int,
    block_n: int,
) -> None:
    if x.ndim != 2 or qweight.ndim != 2:
        raise XQTBackendError(
            "TileLang dequant GEMM expects x and qweight to be 2D tensors"
        )
    if scale.ndim not in {1, 2}:
        raise XQTBackendError("TileLang dequant GEMM expects scale to be 1D or 2D")
    if x.shape[1] != qweight.shape[1]:
        raise XQTBackendError(
            "TileLang dequant GEMM requires x.shape[1] == qweight.shape[1]"
        )
    if scale.ndim == 1 and scale.shape[0] != qweight.shape[0]:
        raise XQTBackendError("1D scale must match qweight out_features")
    if scale.ndim == 2 and scale.shape != qweight.shape:
        raise XQTBackendError("2D scale must match qweight shape")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != qweight.shape[0]):
        raise XQTBackendError("bias must be 1D and match qweight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[0] % block_m != 0 or qweight.shape[0] % block_n != 0:
        raise XQTBackendError(
            "minimal TileLang dequant GEMM currently requires batch and out_features to be multiples of block sizes"
        )


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
    output = x.matmul(weight.t())
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
    output = x.matmul(weight.t())
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


def dequant_gemm_epilogue_tilelang(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only TileLang dequant GEMM epilogue entry point."""

    tensors = (x, qweight, scale) if bias is None else (x, qweight, scale, bias)
    require_cuda_tensors(*tensors)
    require_fp16_tensors(*tensors)
    _validate_dequant_gemm_inputs(
        x,
        qweight,
        scale,
        bias=bias,
        activation=activation,
        block_m=int(block_m),
        block_n=int(block_n),
    )
    require_tilelang()
    dequantized = qweight * scale if scale.ndim == 2 else qweight * scale.unsqueeze(-1)
    kernel = build_tilelang_gemm_kernel(
        m=int(x.shape[0]),
        n=int(qweight.shape[0]),
        k=int(x.shape[1]),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(min(int(x.shape[1]), int(block_n))),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    output = kernel(x, dequantized)
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


def _normalize_group_scale_for_tilelang(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        return scale.unsqueeze(-1)
    return scale


def _resolve_cuda_target_arch(tensor: torch.Tensor, target_arch: str | None) -> str | None:
    if target_arch is not None:
        return str(target_arch)
    if not tensor.is_cuda:
        return None
    major, minor = torch.cuda.get_device_capability(tensor.device)
    return f"sm_{major}{minor}"


def fp4_packed_dequant_gemm_epilogue_tilelang(
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
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only packed FP4 entry using fused TileLang unpack/dequant GEMM."""

    tensors = (
        (x, packed_weight, scale) if bias is None else (x, packed_weight, scale, bias)
    )
    require_cuda_tensors(*tensors)
    if x.dtype != torch.float16 or scale.dtype != torch.float16:
        raise XQTBackendError(
            "packed FP4 TileLang path currently requires float16 x and scale"
        )
    if bias is not None and bias.dtype != torch.float16:
        raise XQTBackendError(
            "packed FP4 TileLang path currently requires float16 bias"
        )
    if packed_weight.dtype != torch.uint8:
        raise XQTBackendError("packed FP4 TileLang path expects uint8 packed_weight")
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise XQTBackendError("packed FP4 TileLang path expects 2D x and packed_weight")
    if x.shape[1] != int(input_features):
        raise XQTBackendError(
            "packed FP4 TileLang path requires x.shape[1] == input_features"
        )
    if (
        scale.ndim != 3
        or scale.shape[0] != packed_weight.shape[0]
        or scale.shape[2] != 1
    ):
        raise XQTBackendError(
            "packed FP4 TileLang path expects scale shaped [out_features, groups, 1]"
        )
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    padded_input_features = int(packed_weight.shape[1]) * 2
    if scale.shape[1] * int(group_size) != padded_input_features:
        raise XQTBackendError(
            "scale groups must cover the packed padded input features"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != packed_weight.shape[0]):
        raise XQTBackendError("bias must be 1D and match packed_weight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[0] % int(block_m) != 0 or packed_weight.shape[0] % int(block_n) != 0:
        raise XQTBackendError(
            "minimal packed FP4 TileLang path requires batch and out_features to be multiples of block sizes"
        )
    require_tilelang()
    kernel = build_tilelang_fp4_fused_dequant_gemm_kernel(
        m=int(x.shape[0]),
        n=int(packed_weight.shape[0]),
        input_features=int(input_features),
        group_size=int(group_size),
        block_m=int(block_m),
        block_n=int(block_n),
        threads=int(threads),
        target_arch=_resolve_cuda_target_arch(x, target_arch),
        has_bias=bias is not None,
        activation=activation,
    )
    if bias is not None:
        return kernel(x, packed_weight, scale, bias)
    return kernel(x, packed_weight, scale)


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
    """Reference packed NVFP4 E2M1 dequantized GEMM with optional epilogue."""

    padded_input_features = int(packed_weight.shape[1]) * 2
    qweight = _decode_packed_nvfp4_e2m1(
        packed_weight.to(device=x.device),
        input_features=padded_input_features,
    ).to(dtype=x.dtype, device=x.device)
    weight_scale = _normalize_group_scale_for_tilelang(scale).to(
        dtype=x.dtype, device=x.device
    )
    if weight_global_scale is not None:
        weight_scale = weight_scale / weight_global_scale.to(
            dtype=x.dtype, device=x.device
        ).reshape(1, 1, 1)
    grouped = qweight.reshape(qweight.shape[0], -1, int(group_size))
    weight = (grouped * weight_scale).reshape(qweight.shape[0], padded_input_features)[
        :, : int(input_features)
    ]
    output = x.matmul(weight.t())
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


def nvfp4_packed_dequant_gemm_epilogue_tilelang(
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
    block_k: int = 128,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only packed NVFP4 entry using fused TileLang unpack/dequant GEMM."""

    tensors = (
        (x, packed_weight, scale) if bias is None else (x, packed_weight, scale, bias)
    )
    require_cuda_tensors(*tensors)
    scale = _normalize_group_scale_for_tilelang(scale)
    if weight_global_scale is not None:
        require_cuda_tensors(weight_global_scale)
    if x.dtype != torch.float16 or scale.dtype != torch.float16:
        raise XQTBackendError(
            "packed NVFP4 TileLang path currently requires float16 x and scale"
        )
    if weight_global_scale is not None and weight_global_scale.dtype != torch.float16:
        raise XQTBackendError(
            "packed NVFP4 TileLang path currently requires float16 weight_global_scale"
        )
    if bias is not None and bias.dtype != torch.float16:
        raise XQTBackendError(
            "packed NVFP4 TileLang path currently requires float16 bias"
        )
    if packed_weight.dtype != torch.uint8:
        raise XQTBackendError("packed NVFP4 TileLang path expects uint8 packed_weight")
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise XQTBackendError(
            "packed NVFP4 TileLang path expects 2D x and packed_weight"
        )
    if x.shape[1] != int(input_features):
        raise XQTBackendError(
            "packed NVFP4 TileLang path requires x.shape[1] == input_features"
        )
    if (
        scale.ndim != 3
        or scale.shape[0] != packed_weight.shape[0]
        or scale.shape[2] != 1
    ):
        raise XQTBackendError(
            "packed NVFP4 TileLang path expects scale shaped [out_features, groups, 1]"
        )
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    padded_input_features = int(packed_weight.shape[1]) * 2
    if scale.shape[1] * int(group_size) != padded_input_features:
        raise XQTBackendError(
            "scale groups must cover the packed padded input features"
        )
    if weight_global_scale is not None and weight_global_scale.numel() != 1:
        raise XQTBackendError("weight_global_scale must be a scalar tensor")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != packed_weight.shape[0]):
        raise XQTBackendError("bias must be 1D and match packed_weight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[0] % int(block_m) != 0 or packed_weight.shape[0] % int(block_n) != 0:
        raise XQTBackendError(
            "minimal packed NVFP4 TileLang path requires batch and out_features to be multiples of block sizes"
        )
    require_tilelang()
    global_scale = (
        weight_global_scale
        if weight_global_scale is not None
        else torch.ones(1, device=x.device, dtype=x.dtype)
    )
    kernel = build_tilelang_nvfp4_fused_dequant_gemm_kernel(
        m=int(x.shape[0]),
        n=int(packed_weight.shape[0]),
        input_features=int(input_features),
        group_size=int(group_size),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
        has_bias=bias is not None,
        activation=activation,
    )
    if bias is not None:
        return kernel(x, packed_weight, scale, global_scale, bias)
    return kernel(x, packed_weight, scale, global_scale)


TILELANG_DEQUANT_GEMM_KERNEL_METADATA = {
    "dequant_gemm_epilogue": {
        "kernel_name": "dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.matmul + epilogue",
        "usage": "Quantized Linear path with dequantize + matmul + bias/activation epilogue.",
    },
    "fp4_packed_dequant_gemm_epilogue": {
        "kernel_name": "fp4_packed_dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "fused TileLang packed FP4 unpack/dequant GEMM + bias/activation epilogue",
        "usage": "FP4WeightOnlyLinear path that consumes packed uint8 weight and group-wise scale.",
        "unpack_stage": "tilelang_fused_gemm_kernel",
        "fusion_status": "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue",
        "epilogue_stage": "tilelang_fused_bias_activation",
    },
    "nvfp4_packed_dequant_gemm_epilogue": {
        "kernel_name": "nvfp4_packed_dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 16,
        "block_k": 128,
        "threads": 128,
        "num_stages": 2,
        "baseline": "fused TileLang packed NVFP4 unpack/dequant GEMM + bias/activation epilogue",
        "usage": "Compressed-tensors NVFP4 path that consumes packed uint8 E2M1 weight and group-wise scale.",
        "weight_encoding": "packed_nvfp4_e2m1",
        "unpack_stage": "tilelang_fused_gemm_kernel",
        "fusion_status": "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue",
        "epilogue_stage": "tilelang_fused_bias_activation",
    },
}

__all__ = [
    "TILELANG_DEQUANT_GEMM_KERNEL_METADATA",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_tilelang",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_tilelang",
]
