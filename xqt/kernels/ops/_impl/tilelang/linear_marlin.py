"""Marlin-inspired multi-precision TileLang Linear kernels.

This module keeps the Marlin idea that matters for XQT integration:
store static weights in a compact inference format, decode the current
weight tile close to the GEMM, accumulate in FP32, and fuse the small
epilogue. It does not try to clone Marlin's hand-written PTX layout or
L2 cache hints; TileLang owns those lower-level scheduling choices here.
"""

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.tilelang._common import (
    require_cuda_tensors,
    require_tilelang,
)

_SUPPORTED_ACTIVATIONS = {None, "gelu", "silu", "relu"}
_SUPPORTED_PRECISIONS = {"auto", "fp16", "bf16", "int8", "int4"}
_DENSE_PRECISIONS = {"fp16", "bf16"}
_QUANT_PRECISIONS = {"int8", "int4"}


def _canonical_precision(precision: str, x_dtype: torch.dtype | None = None) -> str:
    normalized = str(precision).lower()
    if normalized not in _SUPPORTED_PRECISIONS:
        allowed = ", ".join(sorted(_SUPPORTED_PRECISIONS))
        raise XQTBackendError(f"unsupported Marlin Linear precision {precision!r}; expected one of: {allowed}")
    if normalized == "auto":
        return "bf16" if x_dtype == torch.bfloat16 else "fp16"
    return normalized


def _activation_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise XQTBackendError("Marlin Linear TileLang path requires float16 or bfloat16 activations")


def _apply_activation_reference(output: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return output
    if activation == "gelu":
        return F.gelu(output)
    if activation == "silu":
        return F.silu(output)
    if activation == "relu":
        return F.relu(output)
    raise ValueError(f"unsupported activation: {activation}")


def _scale_groups(input_features: int, group_size: int) -> int:
    if group_size <= 0:
        group_size = input_features
    if input_features % group_size != 0:
        raise XQTBackendError("Marlin Linear group_size must divide input_features")
    return input_features // group_size


def _normalize_group_scale(
    scale: torch.Tensor,
    *,
    out_features: int,
    input_features: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    groups = _scale_groups(input_features, group_size)
    if scale.ndim == 1:
        if groups != 1 or scale.shape[0] != out_features:
            raise XQTBackendError("1D scale is valid only for one group per output feature")
        normalized = scale.reshape(out_features, 1, 1)
    elif scale.ndim == 2:
        if scale.shape != (out_features, groups):
            raise XQTBackendError("2D scale must be shaped [out_features, groups]")
        normalized = scale.unsqueeze(-1)
    elif scale.ndim == 3:
        if scale.shape != (out_features, groups, 1):
            raise XQTBackendError("3D scale must be shaped [out_features, groups, 1]")
        normalized = scale
    else:
        raise XQTBackendError("scale must be 1D, 2D, or 3D")
    return normalized.to(dtype=dtype, device=device).contiguous()


def _validate_common(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    precision: str,
    input_features: int,
    out_features: int,
    activation: str | None,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    if x.ndim != 2:
        raise XQTBackendError("Marlin Linear TileLang path expects a 2D input tensor")
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[1] != input_features:
        raise XQTBackendError("Marlin Linear requires x.shape[1] == input_features")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != out_features):
        raise XQTBackendError("bias must be 1D and match out_features")
    if x.shape[0] % block_m != 0 or out_features % block_n != 0:
        raise XQTBackendError(
            "Marlin Linear TileLang path requires batch and out_features to be multiples of block sizes"
        )
    if input_features % block_k != 0:
        raise XQTBackendError("Marlin Linear TileLang path requires input_features to be a multiple of block_k")
    if precision in _DENSE_PRECISIONS and weight.shape != (out_features, input_features):
        raise XQTBackendError("dense Marlin Linear weight must be shaped [out_features, input_features]")


def quantize_int8_weight(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense weight matrix to signed INT8 plus per-group scales."""

    if weight.ndim != 2:
        raise XQTBackendError("INT8 Marlin quantization expects a 2D weight tensor")
    out_features, input_features = int(weight.shape[0]), int(weight.shape[1])
    groups = _scale_groups(input_features, int(group_size))
    grouped = weight.detach().float().reshape(out_features, groups, int(group_size))
    scale = grouped.abs().amax(dim=2, keepdim=True).clamp_min(float(eps)) / 127.0
    qweight = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8)
    return qweight.reshape(out_features, input_features).contiguous(), scale.to(dtype=weight.dtype).contiguous()


def quantize_int4_weight(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and pack signed INT4 weights in low-high nibble order."""

    if weight.ndim != 2:
        raise XQTBackendError("INT4 Marlin quantization expects a 2D weight tensor")
    out_features, input_features = int(weight.shape[0]), int(weight.shape[1])
    groups = _scale_groups(input_features, int(group_size))
    grouped = weight.detach().float().reshape(out_features, groups, int(group_size))
    # Symmetric INT4 uses codes [-8, 7]. The scale is based on 7 so positive
    # maxima round exactly while the extra negative code catches outliers.
    scale = grouped.abs().amax(dim=2, keepdim=True).clamp_min(float(eps)) / 7.0
    signed = torch.round(grouped / scale).clamp(-8, 7).to(torch.int16)
    signed = signed.reshape(out_features, input_features)
    codes = torch.where(signed < 0, signed + 16, signed).to(torch.uint8)
    if input_features % 2 != 0:
        pad = torch.zeros(out_features, 1, dtype=torch.uint8, device=weight.device)
        codes = torch.cat((codes, pad), dim=1)
    low = codes[:, 0::2]
    high = codes[:, 1::2] << 4
    packed = (low | high).contiguous()
    return packed, scale.to(dtype=weight.dtype).contiguous()


def _unpack_signed_int4(
    packed_weight: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    if packed_weight.dtype != torch.uint8:
        raise XQTBackendError("INT4 Marlin reference expects uint8 packed_weight")
    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(packed_weight.shape[0], -1)
    codes = codes[:, : int(input_features)]
    signed = torch.where(codes >= 8, codes.to(torch.int16) - 16, codes.to(torch.int16))
    return signed.to(torch.float32)


def _dequantize_reference(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    precision: str,
    input_features: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if precision in _DENSE_PRECISIONS:
        return weight.to(dtype=dtype, device=device)
    if scale is None:
        raise XQTBackendError("quantized Marlin Linear requires scale")
    out_features = int(weight.shape[0])
    normalized_scale = _normalize_group_scale(
        scale,
        out_features=out_features,
        input_features=input_features,
        group_size=group_size,
        dtype=dtype,
        device=device,
    )
    if precision == "int8":
        if weight.shape != (out_features, input_features):
            raise XQTBackendError("INT8 Marlin weight must be shaped [out_features, input_features]")
        qweight = weight.to(dtype=dtype, device=device)
    else:
        qweight = _unpack_signed_int4(weight.to(device=device), input_features=input_features).to(dtype=dtype)
    grouped = qweight.reshape(out_features, -1, int(group_size))
    return (grouped * normalized_scale).reshape(out_features, input_features)


def linear_marlin_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    precision: str = "auto",
    input_features: int | None = None,
    group_size: int = 128,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference Marlin-style Linear for dense, INT8, and packed INT4 weights."""

    resolved_precision = _canonical_precision(precision, x.dtype)
    resolved_input_features = int(input_features or x.shape[-1])
    dense_weight = _dequantize_reference(
        weight,
        scale,
        precision=resolved_precision,
        input_features=resolved_input_features,
        group_size=int(group_size) if resolved_precision in _QUANT_PRECISIONS else resolved_input_features,
        dtype=x.dtype,
        device=x.device,
    )
    output = F.linear(
        x,
        dense_weight,
        None if bias is None else bias.to(dtype=x.dtype, device=x.device),
    )
    return _apply_activation_reference(output, activation)


@lru_cache(maxsize=64)
def build_tilelang_marlin_linear_kernel(
    m: int,
    n: int,
    k: int,
    *,
    precision: str,
    activation_precision: str,
    group_size: int = 128,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
    has_bias: bool = False,
    activation: str | None = None,
) -> Any:
    """Build a TileLang Marlin-style Linear kernel for one static shape."""

    import tilelang
    import tilelang.language as T

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if precision not in _DENSE_PRECISIONS | _QUANT_PRECISIONS:
        raise ValueError(f"unsupported Marlin Linear precision: {precision}")
    if activation_precision not in _DENSE_PRECISIONS:
        raise ValueError("activation_precision must be fp16 or bf16")
    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise ValueError("Marlin Linear TileLang kernel requires m,n,k to align with block sizes")
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported activation: {activation}")
    if precision in _QUANT_PRECISIONS:
        _scale_groups(k, int(group_size))

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    dtype = T.float16 if activation_precision == "fp16" else T.bfloat16
    accum_dtype = T.float32
    a_shape = [m, k]
    dense_weight_shape = [n, k]
    int4_weight_shape = [n, (k + 1) // 2]
    scale_shape = [n, _scale_groups(k, int(group_size)) if precision in _QUANT_PRECISIONS else 1, 1]
    bias_shape = [n]
    c_shape = [m, n]

    def _apply_activation(value):
        if activation is None:
            return value
        if activation == "relu":
            return T.max(value, 0.0)
        if activation == "silu":
            return value * T.sigmoid(value)
        return 0.5 * value * (1.0 + T.erf(value / T.sqrt(2.0)))

    def _decode_int4(byte_value, feature):
        low = T.bitwise_and(byte_value, 15)
        high = T.bitwise_and(T.shift_right(byte_value, 4), 15)
        nibble = T.if_then_else(feature % 2 == 0, low, high)
        return T.if_then_else(
            nibble >= 8,
            nibble.astype(T.int16) - 16,
            nibble.astype(T.int16),
        )

    shape_suffix = (
        f"{precision}_{activation_precision}_m{m}_n{n}_k{k}_g{group_size}_"
        f"bm{block_m}_bn{block_n}_bk{block_k}_t{threads}_s{num_stages}_"
        f"{activation or 'none'}"
    )

    if precision in _DENSE_PRECISIONS:
        out_idx = [3] if has_bias else [2]

        def linear_marlin_dense_with_bias_main(
            a: T.Tensor(a_shape, dtype),
            weight: T.Tensor(dense_weight_shape, dtype),
            bias: T.Tensor(bias_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=threads) as (bx, by):
                a_shared = T.alloc_shared([block_m, block_k], dtype)
                b_shared = T.alloc_shared([block_n, block_k], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.fill(acc_o, 0)
                for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(
                        a[bx * block_m : (bx + 1) * block_m, k_tile * block_k : (k_tile + 1) * block_k],
                        a_shared,
                    )
                    T.copy(
                        weight[by * block_n : (by + 1) * block_n, k_tile * block_k : (k_tile + 1) * block_k],
                        b_shared,
                    )
                    T.gemm(a_shared, b_shared, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for row, col in T.Parallel(block_m, block_n):
                    out[bx * block_m + row, by * block_n + col] = _apply_activation(
                        acc_o[row, col] + bias[by * block_n + col]
                    )

        linear_marlin_dense_with_bias_main.__name__ = (
            f"linear_marlin_dense_with_bias_main_{shape_suffix}"
        )
        linear_marlin_dense_with_bias_prim = T.prim_func(linear_marlin_dense_with_bias_main)

        def linear_marlin_dense_without_bias_main(
            a: T.Tensor(a_shape, dtype),
            weight: T.Tensor(dense_weight_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=threads) as (bx, by):
                a_shared = T.alloc_shared([block_m, block_k], dtype)
                b_shared = T.alloc_shared([block_n, block_k], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.fill(acc_o, 0)
                for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(
                        a[bx * block_m : (bx + 1) * block_m, k_tile * block_k : (k_tile + 1) * block_k],
                        a_shared,
                    )
                    T.copy(
                        weight[by * block_n : (by + 1) * block_n, k_tile * block_k : (k_tile + 1) * block_k],
                        b_shared,
                    )
                    T.gemm(a_shared, b_shared, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for row, col in T.Parallel(block_m, block_n):
                    out[bx * block_m + row, by * block_n + col] = _apply_activation(acc_o[row, col])

        linear_marlin_dense_without_bias_main.__name__ = (
            f"linear_marlin_dense_without_bias_main_{shape_suffix}"
        )
        linear_marlin_dense_without_bias_prim = T.prim_func(linear_marlin_dense_without_bias_main)

        def dense_with_bias():
            return linear_marlin_dense_with_bias_prim
        dense_with_bias.__name__ = f"linear_marlin_dense_with_bias_builder_{shape_suffix}"
        dense_with_bias_jit = tilelang.jit(
            out_idx=out_idx,
            target=target,
            pass_configs=pass_configs,
        )(dense_with_bias)

        def dense_without_bias():
            return linear_marlin_dense_without_bias_prim
        dense_without_bias.__name__ = f"linear_marlin_dense_without_bias_builder_{shape_suffix}"
        dense_without_bias_jit = tilelang.jit(
            out_idx=out_idx,
            target=target,
            pass_configs=pass_configs,
        )(dense_without_bias)

        return dense_with_bias_jit() if has_bias else dense_without_bias_jit()

    out_idx = [4] if has_bias else [3]
    weight_dtype = T.int8 if precision == "int8" else T.uint8
    weight_shape = dense_weight_shape if precision == "int8" else int4_weight_shape

    def linear_marlin_quant_with_bias_main(
        a: T.Tensor(a_shape, dtype),
        weight: T.Tensor(weight_shape, weight_dtype),
        scale: T.Tensor(scale_shape, dtype),
        bias: T.Tensor(bias_shape, dtype),
        out: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=threads) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], dtype)
            b_shared = T.alloc_shared([block_n, block_k], dtype)
            acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

            T.fill(acc_o, 0)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(
                    a[bx * block_m : (bx + 1) * block_m, k_tile * block_k : (k_tile + 1) * block_k],
                    a_shared,
                )
                # Marlin-style weight-only path: decode just the weight tile
                # that will feed the next tensor-core GEMM tile.
                for row_offset, feature_offset in T.Parallel(block_n, block_k):
                    row = by * block_n + row_offset
                    feature = k_tile * block_k + feature_offset
                    if precision == "int8":
                        decoded = weight[row, feature].astype(accum_dtype)
                    else:
                        byte_value = weight[row, feature // 2].astype(T.uint16)
                        decoded = _decode_int4(byte_value, feature).astype(accum_dtype)
                    b_shared[row_offset, feature_offset] = (
                        decoded.astype(dtype) * scale[row, feature // group_size, 0]
                    )
                T.gemm(a_shared, b_shared, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for row, col in T.Parallel(block_m, block_n):
                out[bx * block_m + row, by * block_n + col] = _apply_activation(
                    acc_o[row, col] + bias[by * block_n + col]
                )

    linear_marlin_quant_with_bias_main.__name__ = (
        f"linear_marlin_quant_with_bias_main_{shape_suffix}"
    )
    linear_marlin_quant_with_bias_prim = T.prim_func(linear_marlin_quant_with_bias_main)

    def linear_marlin_quant_without_bias_main(
        a: T.Tensor(a_shape, dtype),
        weight: T.Tensor(weight_shape, weight_dtype),
        scale: T.Tensor(scale_shape, dtype),
        out: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(n, block_n), threads=threads) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], dtype)
            b_shared = T.alloc_shared([block_n, block_k], dtype)
            acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

            T.fill(acc_o, 0)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(
                    a[bx * block_m : (bx + 1) * block_m, k_tile * block_k : (k_tile + 1) * block_k],
                    a_shared,
                )
                # INT4 uses one byte per two weights. INT8 keeps one signed
                # byte per weight but shares the same group-scale path.
                for row_offset, feature_offset in T.Parallel(block_n, block_k):
                    row = by * block_n + row_offset
                    feature = k_tile * block_k + feature_offset
                    if precision == "int8":
                        decoded = weight[row, feature].astype(accum_dtype)
                    else:
                        byte_value = weight[row, feature // 2].astype(T.uint16)
                        decoded = _decode_int4(byte_value, feature).astype(accum_dtype)
                    b_shared[row_offset, feature_offset] = (
                        decoded.astype(dtype) * scale[row, feature // group_size, 0]
                    )
                T.gemm(a_shared, b_shared, acc_o, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for row, col in T.Parallel(block_m, block_n):
                out[bx * block_m + row, by * block_n + col] = _apply_activation(acc_o[row, col])

    linear_marlin_quant_without_bias_main.__name__ = (
        f"linear_marlin_quant_without_bias_main_{shape_suffix}"
    )
    linear_marlin_quant_without_bias_prim = T.prim_func(linear_marlin_quant_without_bias_main)

    def quant_with_bias():
        return linear_marlin_quant_with_bias_prim
    quant_with_bias.__name__ = f"linear_marlin_quant_with_bias_builder_{shape_suffix}"
    quant_with_bias_jit = tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )(quant_with_bias)

    def quant_without_bias():
        return linear_marlin_quant_without_bias_prim
    quant_without_bias.__name__ = f"linear_marlin_quant_without_bias_builder_{shape_suffix}"
    quant_without_bias_jit = tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )(quant_without_bias)

    return quant_with_bias_jit() if has_bias else quant_without_bias_jit()


def linear_marlin_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    precision: str = "auto",
    input_features: int | None = None,
    group_size: int = 128,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run a static-shape Marlin-style TileLang Linear kernel."""

    tensors = (x, weight) if bias is None else (x, weight, bias)
    if scale is not None:
        tensors = (*tensors, scale)
    require_cuda_tensors(*tensors)
    require_tilelang()
    activation_precision = _activation_dtype_name(x.dtype)
    resolved_precision = _canonical_precision(precision, x.dtype)
    resolved_input_features = int(input_features or x.shape[1])
    out_features = int(weight.shape[0])
    _validate_common(
        x,
        weight,
        bias,
        precision=resolved_precision,
        input_features=resolved_input_features,
        out_features=out_features,
        activation=activation,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )
    if resolved_precision == "fp16" and x.dtype != torch.float16:
        raise XQTBackendError("fp16 Marlin Linear requires float16 activations")
    if resolved_precision == "bf16" and x.dtype != torch.bfloat16:
        raise XQTBackendError("bf16 Marlin Linear requires bfloat16 activations")
    if resolved_precision in _DENSE_PRECISIONS and weight.dtype != x.dtype:
        raise XQTBackendError("dense Marlin Linear weight dtype must match activation dtype")
    if resolved_precision == "int8" and weight.dtype != torch.int8:
        raise XQTBackendError("INT8 Marlin Linear expects torch.int8 weight")
    if resolved_precision == "int4" and weight.dtype != torch.uint8:
        raise XQTBackendError("INT4 Marlin Linear expects uint8 packed weight")
    if resolved_precision == "int4" and weight.shape[1] < (resolved_input_features + 1) // 2:
        raise XQTBackendError("INT4 Marlin packed weight does not cover input_features")

    normalized_scale = None
    resolved_group_size = resolved_input_features
    if resolved_precision in _QUANT_PRECISIONS:
        if scale is None:
            raise XQTBackendError("quantized Marlin Linear requires scale")
        resolved_group_size = int(group_size)
        normalized_scale = _normalize_group_scale(
            scale,
            out_features=out_features,
            input_features=resolved_input_features,
            group_size=resolved_group_size,
            dtype=x.dtype,
            device=x.device,
        )
    runtime_bias = None if bias is None else bias.to(dtype=x.dtype, device=x.device).contiguous()
    kernel = build_tilelang_marlin_linear_kernel(
        m=int(x.shape[0]),
        n=out_features,
        k=resolved_input_features,
        precision=resolved_precision,
        activation_precision=activation_precision,
        group_size=resolved_group_size,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
        has_bias=runtime_bias is not None,
        activation=activation,
    )
    if resolved_precision in _DENSE_PRECISIONS:
        if runtime_bias is not None:
            return kernel(x, weight, runtime_bias)
        return kernel(x, weight)
    if runtime_bias is not None:
        return kernel(x, weight, normalized_scale, runtime_bias)
    return kernel(x, weight, normalized_scale)


TILELANG_MARLIN_LINEAR_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "linear_marlin": {
        "kernel_name": "linear_marlin",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.nn.functional.linear or eager dequant + linear",
        "usage": "Marlin-inspired TileLang Linear path for dense FP16/BF16 and weight-only INT8/INT4.",
        "supported_precisions": ["fp16", "bf16", "int8", "int4"],
        "weight_encoding": "dense_fp_or_signed_int8_or_packed_signed_int4",
        "unpack_stage": "tilelang_fused_weight_tile_decode",
        "fusion_status": "single_tilelang_kernel_for_dequant_gemm_epilogue",
        "epilogue_stage": "tilelang_fused_bias_activation",
    },
}

__all__ = [
    "TILELANG_MARLIN_LINEAR_KERNEL_METADATA",
    "build_tilelang_marlin_linear_kernel",
    "linear_marlin_reference",
    "linear_marlin_tilelang",
    "quantize_int4_weight",
    "quantize_int8_weight",
]
