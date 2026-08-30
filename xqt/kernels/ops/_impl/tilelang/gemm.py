"""TileLang dequantized GEMM operator references and guarded entry points."""

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.kernels.ops.gemm.reference import dense_gemm_reference
from xqt.kernels.ops._impl.fp4_quant_common import dequantize_nvfp4_codes

from xqt.kernels.ops._impl.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)
from xqt.kernels.ops._impl.tilelang.gemm_builder import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
    build_tilelang_fp4_packed_activation_fused_gemm_kernel,
    build_tilelang_gemm_kernel,
    build_tilelang_nvfp4_unpack_dequant_kernel,
    build_tilelang_nvfp4_packed_activation_fused_gemm_kernel,
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
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


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
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


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


def _tilelang_unpack_dequant_fp4_codes(
    packed_codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    input_features: int,
    group_size: int,
    global_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
) -> torch.Tensor:
    require_cuda_tensors(packed_codes, scale)
    if packed_codes.dtype != torch.uint8:
        raise XQTBackendError("TileLang packed activation path expects uint8 packed codes")
    if packed_codes.ndim != 2:
        raise XQTBackendError("TileLang packed activation path expects 2D packed codes")
    if scale.ndim == 3 and scale.shape[2] == 1:
        scale_2d = scale.squeeze(-1)
    elif scale.ndim == 2:
        scale_2d = scale
    else:
        raise XQTBackendError("TileLang packed activation path expects scale shaped [rows, groups] or [rows, groups, 1]")
    rows = int(packed_codes.shape[0])
    groups = int(scale_2d.shape[1])
    if groups * int(group_size) < int(input_features):
        raise XQTBackendError("TileLang packed activation scale groups must cover input_features")
    target_dtype = torch.float16 if output_dtype == torch.float16 else torch.float32
    if target_dtype == torch.float32:
        raise XQTBackendError("TileLang packed activation path currently supports float16 output only")
    scale_fp16 = scale_2d.to(device=packed_codes.device, dtype=torch.float16).contiguous()
    resolved_target_arch = _resolve_cuda_target_arch(packed_codes, target_arch)
    if global_scale is not None:
        require_cuda_tensors(global_scale)
        if global_scale.numel() != 1:
            raise XQTBackendError("global_scale must be a scalar tensor")
        kernel = build_tilelang_nvfp4_unpack_dequant_kernel(
            rows,
            int(input_features),
            int(group_size),
            block_n=int(block_n),
            block_k=int(block_k),
            threads=int(threads),
            target_arch=resolved_target_arch,
            use_global_scale=True,
        )
        return kernel(
            packed_codes,
            scale_fp16,
            global_scale.to(device=packed_codes.device, dtype=torch.float16).reshape(1),
        )
    kernel = build_tilelang_nvfp4_unpack_dequant_kernel(
        rows,
        int(input_features),
        int(group_size),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        target_arch=resolved_target_arch,
        use_global_scale=False,
    )
    return kernel(
        packed_codes,
        scale_fp16,
    )


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


def mxfp4_packed_dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference MXFP4 packed dequantized GEMM with optional epilogue."""

    return fp4_packed_dequant_gemm_epilogue_reference(
        x,
        packed_weight,
        _normalize_group_scale_for_tilelang(scale),
        bias,
        input_features=input_features,
        group_size=group_size,
        activation=activation,
    )


def mxfp4_packed_dequant_gemm_epilogue_tilelang(
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
    """CUDA-only packed MXFP4 entry reusing the signed-int4 TileLang kernel."""

    return fp4_packed_dequant_gemm_epilogue_tilelang(
        x,
        packed_weight,
        _normalize_group_scale_for_tilelang(scale),
        bias,
        input_features=input_features,
        group_size=group_size,
        activation=activation,
        block_m=block_m,
        block_n=block_n,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )


def mxfp4_packed_activation_gemm_epilogue_reference(
    packed_activation: torch.Tensor,
    activation_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference MXFP4 GEMM that accepts packed activation and packed weight."""

    activation_dense = dequantize_nvfp4_codes(
        packed_activation,
        activation_scale,
        input_features=int(input_features),
        group_size=int(group_size),
        global_scale=None,
        output_dtype=output_dtype,
    )
    return mxfp4_packed_dequant_gemm_epilogue_reference(
        activation_dense,
        packed_weight,
        weight_scale,
        bias,
        input_features=int(input_features),
        group_size=int(group_size),
        activation=activation,
    )


def mxfp4_packed_activation_gemm_epilogue_tilelang(
    packed_activation: torch.Tensor,
    activation_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
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
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """TileLang MXFP4 GEMM that accepts packed activation and packed weight."""

    require_cuda_tensors(packed_activation, activation_scale, packed_weight, weight_scale)
    if packed_activation.dtype != torch.uint8 or packed_weight.dtype != torch.uint8:
        raise XQTBackendError("TileLang MXFP4 packed activation path expects uint8 packed tensors")
    if packed_activation.ndim != 2 or packed_weight.ndim != 2:
        raise XQTBackendError("TileLang MXFP4 packed activation path expects 2D packed tensors")
    activation_scale_2d = _normalize_group_scale_for_tilelang(activation_scale)
    tilelang_weight_scale = _normalize_group_scale_for_tilelang(weight_scale).to(
        device=packed_activation.device,
        dtype=torch.float16,
    )
    if (
        activation_scale_2d.ndim != 3
        or activation_scale_2d.shape[0] != packed_activation.shape[0]
        or activation_scale_2d.shape[2] != 1
    ):
        raise XQTBackendError(
            "TileLang MXFP4 packed activation path expects activation_scale shaped [rows, groups, 1]"
        )
    if (
        tilelang_weight_scale.ndim != 3
        or tilelang_weight_scale.shape[0] != packed_weight.shape[0]
        or tilelang_weight_scale.shape[2] != 1
    ):
        raise XQTBackendError(
            "TileLang MXFP4 packed activation path expects weight_scale shaped [out_features, groups, 1]"
        )
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    padded_input_features = int(packed_weight.shape[1]) * 2
    if activation_scale_2d.shape[1] * int(group_size) < int(input_features):
        raise XQTBackendError("activation scale groups must cover input_features")
    if tilelang_weight_scale.shape[1] * int(group_size) != padded_input_features:
        raise XQTBackendError("weight scale groups must cover the packed padded input features")
    resolved_target_arch = _resolve_cuda_target_arch(packed_activation, target_arch)
    tilelang_bias = (
        None
        if bias is None
        else bias.to(device=packed_activation.device, dtype=torch.float16)
    )
    if output_dtype != torch.float16:
        raise XQTBackendError("TileLang MXFP4 packed activation fused path currently supports float16 output only")
    kernel = build_tilelang_fp4_packed_activation_fused_gemm_kernel(
        m=int(packed_activation.shape[0]),
        n=int(packed_weight.shape[0]),
        input_features=int(input_features),
        group_size=int(group_size),
        block_m=int(block_m),
        block_n=int(block_n),
        threads=int(threads),
        target_arch=resolved_target_arch,
        has_bias=tilelang_bias is not None,
        activation=activation,
    )
    if tilelang_bias is not None:
        return kernel(
            packed_activation,
            activation_scale_2d.to(device=packed_activation.device, dtype=torch.float16).squeeze(-1).contiguous(),
            packed_weight,
            tilelang_weight_scale,
            tilelang_bias,
        )
    return kernel(
        packed_activation,
        activation_scale_2d.to(device=packed_activation.device, dtype=torch.float16).squeeze(-1).contiguous(),
        packed_weight,
        tilelang_weight_scale,
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
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device)
    return dense_gemm_reference(
        x,
        weight,
        runtime_bias,
        activation=activation,
        transpose_b=True,
    )


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


def nvfp4_packed_activation_gemm_epilogue_reference(
    packed_activation: torch.Tensor,
    activation_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation_global_scale: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference NVFP4 GEMM that accepts packed activation and packed weight."""

    activation_dense = dequantize_nvfp4_codes(
        packed_activation,
        activation_scale,
        input_features=int(input_features),
        group_size=int(group_size),
        global_scale=activation_global_scale,
        output_dtype=output_dtype,
    )
    return nvfp4_packed_dequant_gemm_epilogue_reference(
        activation_dense,
        packed_weight,
        weight_scale,
        bias,
        input_features=int(input_features),
        group_size=int(group_size),
        weight_global_scale=weight_global_scale,
        activation=activation,
    )


def nvfp4_packed_activation_gemm_epilogue_tilelang(
    packed_activation: torch.Tensor,
    activation_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation_global_scale: torch.Tensor | None = None,
    weight_global_scale: torch.Tensor | None = None,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 16,
    block_k: int = 128,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
    output_dtype: torch.dtype = torch.float16,
    weight_is_prepacked: bool = False,
) -> torch.Tensor:
    """TileLang NVFP4 GEMM with row-major or CTA-tiled packed weights."""

    tensors = (packed_activation, activation_scale, packed_weight, weight_scale)
    if bias is not None:
        tensors = (*tensors, bias)
    if activation_global_scale is not None:
        tensors = (*tensors, activation_global_scale)
    if weight_global_scale is not None:
        tensors = (*tensors, weight_global_scale)
    require_cuda_tensors(*tensors)
    if packed_activation.dtype != torch.uint8 or packed_weight.dtype != torch.uint8:
        raise XQTBackendError(
            "TileLang NVFP4 packed activation path expects uint8 packed tensors"
        )
    expected_weight_ndim = 4 if weight_is_prepacked else 2
    if packed_activation.ndim != 2 or packed_weight.ndim != expected_weight_ndim:
        raise XQTBackendError(
            "TileLang NVFP4 packed activation path received an invalid packed weight layout"
        )
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    if output_dtype != torch.float16:
        raise XQTBackendError(
            "TileLang NVFP4 packed activation fused path currently supports float16 output only"
        )
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")

    input_features = int(input_features)
    group_size = int(group_size)
    block_m = int(block_m)
    block_n = int(block_n)
    block_k = int(block_k)
    threads = int(threads)
    num_stages = int(num_stages)
    if min(input_features, block_m, block_n, block_k, threads, num_stages) <= 0:
        raise XQTBackendError("TileLang NVFP4 packed activation launch parameters must be positive")
    if packed_activation.shape[0] % block_m != 0:
        raise XQTBackendError(
            "minimal TileLang NVFP4 packed activation path requires batch to be a multiple of block_m"
        )
    if input_features % block_k != 0:
        raise XQTBackendError(
            "minimal TileLang NVFP4 packed activation path requires input_features to be a multiple of block_k"
        )
    if block_k % group_size != 0:
        raise XQTBackendError(
            "minimal TileLang NVFP4 packed activation path requires block_k divisible by group_size"
        )

    expected_padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    padded_input_features = int(packed_activation.shape[1]) * 2
    if padded_input_features != expected_padded_input_features:
        raise XQTBackendError(
            "packed NVFP4 tensors must exactly match the group-padded input_features extent"
        )
    if weight_is_prepacked:
        output_features = int(packed_weight.shape[0]) * block_n
        expected_weight_shape = (
            output_features // block_n,
            padded_input_features // block_k,
            block_n,
            block_k // 2,
        )
        if tuple(packed_weight.shape) != expected_weight_shape:
            raise XQTBackendError(
                "TileLang NVFP4 prepacked weight must use [N/block_n, K/block_k, block_n, block_k/2]"
            )
    else:
        output_features = int(packed_weight.shape[0])
        if int(packed_weight.shape[1]) * 2 != padded_input_features:
            raise XQTBackendError(
                "TileLang NVFP4 packed activation and weight must use the same packed K extent"
            )
    if output_features % block_n != 0:
        raise XQTBackendError(
            "minimal TileLang NVFP4 packed activation path requires out_features to be a multiple of block_n"
        )
    activation_scale_3d = _normalize_group_scale_for_tilelang(activation_scale)
    if (
        activation_scale_3d.ndim != 3
        or activation_scale_3d.shape[0] != packed_activation.shape[0]
        or activation_scale_3d.shape[2] != 1
    ):
        raise XQTBackendError(
            "TileLang NVFP4 packed activation path expects activation_scale shaped [rows, groups, 1]"
        )
    expected_groups = padded_input_features // group_size
    if padded_input_features % group_size != 0 or activation_scale_3d.shape[1] != expected_groups:
        raise XQTBackendError(
            "TileLang NVFP4 packed activation scale groups must exactly cover packed K"
        )
    if weight_is_prepacked:
        expected_scale_shape = (
            output_features // block_n,
            padded_input_features // block_k,
            block_n,
            block_k // group_size,
        )
        if tuple(weight_scale.shape) != expected_scale_shape:
            raise XQTBackendError(
                "TileLang NVFP4 prepacked scale must use [N/block_n, K/block_k, block_n, block_k/group_size]"
            )
        tilelang_weight_scale = weight_scale.to(
            device=packed_activation.device,
            dtype=torch.float16,
        ).contiguous()
    else:
        tilelang_weight_scale = _normalize_group_scale_for_tilelang(weight_scale).to(
            device=packed_activation.device,
            dtype=torch.float16,
        )
        if (
            tilelang_weight_scale.ndim != 3
            or tilelang_weight_scale.shape[0] != output_features
            or tilelang_weight_scale.shape[1] != expected_groups
            or tilelang_weight_scale.shape[2] != 1
        ):
            raise XQTBackendError(
                "TileLang NVFP4 packed activation path expects weight_scale shaped [out_features, groups, 1]"
            )
    if activation_global_scale is not None and activation_global_scale.numel() != 1:
        raise XQTBackendError("activation_global_scale must be a scalar tensor")
    if weight_global_scale is not None and weight_global_scale.numel() != 1:
        raise XQTBackendError("weight_global_scale must be a scalar tensor")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != output_features):
        raise XQTBackendError("bias must be 1D and match packed_weight out_features")

    activation_scale_2d = activation_scale_3d.to(
        device=packed_activation.device,
        dtype=torch.float16,
    ).squeeze(-1).contiguous()
    tilelang_bias = (
        None
        if bias is None
        else bias.to(device=packed_activation.device, dtype=torch.float16)
    )
    tilelang_activation_global_scale = (
        torch.ones(1, device=packed_activation.device, dtype=torch.float16)
        if activation_global_scale is None
        else activation_global_scale.to(
            device=packed_activation.device,
            dtype=torch.float16,
        ).reshape(1)
    )
    tilelang_weight_global_scale = (
        torch.ones(1, device=packed_activation.device, dtype=torch.float16)
        if weight_global_scale is None
        else weight_global_scale.to(
            device=packed_activation.device,
            dtype=torch.float16,
        ).reshape(1)
    )
    require_tilelang()
    resolved_target_arch = _resolve_cuda_target_arch(packed_activation, target_arch)
    kernel = build_tilelang_nvfp4_packed_activation_fused_gemm_kernel(
        m=int(packed_activation.shape[0]),
        n=output_features,
        input_features=input_features,
        group_size=group_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        threads=threads,
        num_stages=num_stages,
        target_arch=resolved_target_arch,
        has_bias=tilelang_bias is not None,
        activation=activation,
        weight_is_prepacked=weight_is_prepacked,
    )
    if tilelang_bias is not None:
        return kernel(
            packed_activation,
            activation_scale_2d,
            tilelang_activation_global_scale,
            packed_weight,
            tilelang_weight_scale,
            tilelang_weight_global_scale,
            tilelang_bias,
        )
    return kernel(
        packed_activation,
        activation_scale_2d,
        tilelang_activation_global_scale,
        packed_weight,
        tilelang_weight_scale,
        tilelang_weight_global_scale,
    )


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
    "mxfp4_packed_dequant_gemm_epilogue": {
        "kernel_name": "mxfp4_packed_dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "fused TileLang packed MXFP4 unpack/dequant GEMM + bias/activation epilogue",
        "usage": "MXFPWeightOnlyLinear MXFP4 path that consumes packed uint8 signed-int4 mantissas and block-wise scale.",
        "weight_encoding": "packed_mxfp4_signed_int4",
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
    "mxfp4_packed_activation_gemm_epilogue_reference",
    "mxfp4_packed_activation_gemm_epilogue_tilelang",
    "mxfp4_packed_dequant_gemm_epilogue_reference",
    "mxfp4_packed_dequant_gemm_epilogue_tilelang",
    "nvfp4_packed_activation_gemm_epilogue_reference",
    "nvfp4_packed_activation_gemm_epilogue_tilelang",
    "nvfp4_packed_dequant_gemm_epilogue_reference",
    "nvfp4_packed_dequant_gemm_epilogue_tilelang",
]
