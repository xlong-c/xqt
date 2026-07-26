"""TileLang true INT8 MMA kernels."""

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_tilelang,
)


def _validate_int8_mma_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise XQTBackendError("TileLang INT8 MMA expects 2D input tensors")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise XQTBackendError("TileLang INT8 MMA expects torch.int8 inputs")
    if a.shape[1] != b.shape[0]:
        raise XQTBackendError("TileLang INT8 MMA requires a.shape[1] == b.shape[0]")
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[1])
    if m <= 0 or n <= 0 or k <= 0:
        raise XQTBackendError("TileLang INT8 MMA dimensions must be positive")
    if m % int(block_m) != 0 or n % int(block_n) != 0 or k % int(block_k) != 0:
        raise XQTBackendError(
            "TileLang INT8 MMA requires m, n, and k to align with block sizes"
        )
    return m, n, k


def int8_mma_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference W8A8 GEMM that returns int32 accumulators."""

    if a.ndim != 2 or b.ndim != 2:
        raise XQTBackendError("INT8 MMA reference expects 2D input tensors")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise XQTBackendError("INT8 MMA reference expects torch.int8 inputs")
    if a.shape[1] != b.shape[0]:
        raise XQTBackendError("INT8 MMA reference requires a.shape[1] == b.shape[0]")
    int_mm = getattr(torch, "_int_mm", None)
    if a.is_cuda and b.is_cuda and callable(int_mm):
        return int_mm(a, b)
    return a.to(torch.int32) @ b.to(torch.int32)


@lru_cache(maxsize=32)
def build_tilelang_int8_mma_kernel(
    m: int,
    n: int,
    k: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> Any:
    """Build a static-shape TileLang W8A8 GEMM kernel with int32 accumulation."""

    tilelang = require_tilelang()
    import tilelang.language as T

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise ValueError("m, n, and k must be multiples of block_m, block_n, block_k")

    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    a_shape = [m, k]
    b_shape = [k, n]
    out_shape = [m, n]

    def int8_mma_main(
        a: T.Tensor(a_shape, T.int8),
        b: T.Tensor(b_shape, T.int8),
        out: T.Tensor(out_shape, T.int32),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], T.int8)
            b_shared = T.alloc_shared([block_k, block_n], T.int8)
            acc = T.alloc_fragment([block_m, block_n], T.int32)

            T.fill(acc, 0)
            T.annotate_consumer_reg_alloc(255)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(
                    a[
                        bx * block_m : (bx + 1) * block_m,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    a_shared,
                )
                T.copy(
                    b[
                        k_tile * block_k : (k_tile + 1) * block_k,
                        by * block_n : (by + 1) * block_n,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out[bx * block_m + row, by * block_n + col] = acc[row, col]

    int8_mma_main.__name__ = (
        f"int8_mma_main_m{m}_n{n}_k{k}_"
        f"bm{block_m}_bn{block_n}_bk{block_k}_t{threads}_s{num_stages}"
    )
    prim = T.prim_func(int8_mma_main)

    def builder():
        return prim

    builder.__name__ = f"int8_mma_builder_m{m}_n{n}_k{k}"
    return tilelang.jit(
        out_idx=[],
        target=target,
        pass_configs=pass_configs,
    )(builder)()



def int8_mma_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run true W8A8 INT8 MMA with int32 accumulation through TileLang."""

    require_cuda_tensors(a, b)
    m, n, k = _validate_int8_mma_inputs(
        a,
        b,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )
    kernel = build_tilelang_int8_mma_kernel(
        m,
        n,
        k,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    out = torch.empty((m, n), device=a.device, dtype=torch.int32)
    kernel(a.contiguous(), b.contiguous(), out)
    return out


def _activation_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise XQTBackendError(f"unsupported activation dtype for INT8 quantization: {dtype}")


@lru_cache(maxsize=32)
def build_tilelang_static_activation_quant_kernel(
    rows: int,
    features: int,
    *,
    input_dtype: str,
    block_size: int = 256,
    target_arch: str | None = None,
) -> Any:
    """Build a TileLang kernel that quantizes activations with a static scale."""

    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or features <= 0:
        raise ValueError("rows and features must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if input_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("input_dtype must be float16, bfloat16, or float32")

    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    x_shape = [rows, features]
    scale_shape = [1]
    out_shape = [rows, features]
    x_dtype = (
        T.float16
        if input_dtype == "float16"
        else T.bfloat16
        if input_dtype == "bfloat16"
        else T.float32
    )
    total = rows * features

    def static_activation_quant_main(
        x: T.Tensor(x_shape, x_dtype),
        scale: T.Tensor(scale_shape, T.float32),
        out: T.Tensor(out_shape, T.int8),
    ):
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as block:
            for offset in T.Parallel(block_size):
                index = block * block_size + offset
                if index < total:
                    row = index // features
                    col = index - row * features
                    scaled = T.round(x[row, col].astype(T.float32) / scale[0])
                    clipped = T.max(T.min(scaled, 127.0), -127.0)
                    out[row, col] = clipped.astype(T.int8)

    shape_suffix = (
        f"rows{rows}_features{features}_{input_dtype}_"
        f"block{block_size}"
    )
    static_activation_quant_main.__name__ = (
        f"static_activation_quant_main_{shape_suffix}"
    )
    prim = T.prim_func(static_activation_quant_main)

    def builder():
        return prim

    builder.__name__ = f"static_activation_quant_builder_{shape_suffix}"
    return tilelang.jit(
        out_idx=[],
        target=target,
        pass_configs=pass_configs,
    )(builder)()


def static_activation_quantize_tilelang(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 256,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Quantize dense activations to INT8 with one TileLang kernel."""

    require_cuda_tensors(inputs, scale)
    if inputs.ndim != 2:
        raise XQTBackendError("TileLang activation quantization expects a 2D tensor")
    if scale.numel() != 1:
        raise XQTBackendError("activation scale must contain one scalar")
    rows, features = int(inputs.shape[0]), int(inputs.shape[1])
    kernel = build_tilelang_static_activation_quant_kernel(
        rows,
        features,
        input_dtype=_activation_dtype_name(inputs.dtype),
        block_size=int(block_size),
        target_arch=target_arch,
    )
    out = torch.empty((rows, features), device=inputs.device, dtype=torch.int8)
    kernel(
        inputs.contiguous(),
        scale.reshape(1).to(device=inputs.device, dtype=torch.float32),
        out,
    )
    return out


@lru_cache(maxsize=32)
def build_tilelang_groupwise_hadamard_static_quant_kernel(
    rows: int,
    features: int,
    rot_size: int,
    *,
    input_dtype: str,
    rotation_dtype: str,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> Any:
    """Build fused groupwise Hadamard rotation + static INT8 quantization."""

    tilelang = require_tilelang()
    import tilelang.language as T

    if rows <= 0 or features <= 0 or rot_size <= 0:
        raise ValueError("rows, features, and rot_size must be positive")
    if features % rot_size != 0:
        raise ValueError("features must be divisible by rot_size")
    if rot_size % int(block_n) != 0 or rot_size % int(block_k) != 0:
        raise ValueError("rot_size must be divisible by block_n and block_k")
    if rows % int(block_m) != 0:
        raise ValueError("rows must be divisible by block_m")
    if input_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("input_dtype must be float16, bfloat16, or float32")
    if rotation_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("rotation_dtype must be float16, bfloat16, or float32")

    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    x_shape = [rows, features]
    rotation_shape = [rot_size, rot_size]
    scale_shape = [1]
    out_shape = [rows, features]
    x_dtype = (
        T.float16
        if input_dtype == "float16"
        else T.bfloat16
        if input_dtype == "bfloat16"
        else T.float32
    )
    rot_dtype = (
        T.float16
        if rotation_dtype == "float16"
        else T.bfloat16
        if rotation_dtype == "bfloat16"
        else T.float32
    )
    groups = features // rot_size
    n_tiles = rot_size // int(block_n)

    def groupwise_hadamard_static_quant_main(
        x: T.Tensor(x_shape, x_dtype),
        rotation: T.Tensor(rotation_shape, rot_dtype),
        scale: T.Tensor(scale_shape, T.float32),
        out: T.Tensor(out_shape, T.int8),
    ):
        with T.Kernel(
            T.ceildiv(rows, block_m),
            groups * n_tiles,
            threads=threads,
        ) as (bx, by):
            group = by // n_tiles
            n_tile = by - group * n_tiles
            x_shared = T.alloc_shared([block_m, block_k], x_dtype)
            rotation_shared = T.alloc_shared([block_k, block_n], rot_dtype)
            acc = T.alloc_fragment([block_m, block_n], T.float32)

            T.fill(acc, 0.0)
            T.annotate_consumer_reg_alloc(255)
            for k_tile in T.Pipelined(T.ceildiv(rot_size, block_k), num_stages=num_stages):
                T.copy(
                    x[
                        bx * block_m : (bx + 1) * block_m,
                        group * rot_size
                        + k_tile * block_k : group * rot_size
                        + (k_tile + 1) * block_k,
                    ],
                    x_shared,
                )
                T.copy(
                    rotation[
                        k_tile * block_k : (k_tile + 1) * block_k,
                        n_tile * block_n : (n_tile + 1) * block_n,
                    ],
                    rotation_shared,
                )
                T.gemm(
                    x_shared,
                    rotation_shared,
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                scaled = T.round(acc[row, col] / scale[0])
                clipped = T.max(T.min(scaled, 127.0), -127.0)
                out[
                    bx * block_m + row,
                    group * rot_size + n_tile * block_n + col,
                ] = clipped.astype(T.int8)

    shape_suffix = (
        f"rows{rows}_features{features}_rot{rot_size}_"
        f"{input_dtype}_{rotation_dtype}_bm{block_m}_bn{block_n}_bk{block_k}"
    )
    groupwise_hadamard_static_quant_main.__name__ = (
        f"groupwise_hadamard_static_quant_main_{shape_suffix}"
    )
    prim = T.prim_func(groupwise_hadamard_static_quant_main)

    def builder():
        return prim

    builder.__name__ = f"groupwise_hadamard_static_quant_builder_{shape_suffix}"
    return tilelang.jit(
        out_idx=[],
        target=target,
        pass_configs=pass_configs,
    )(builder)()


def groupwise_hadamard_static_quantize_tilelang(
    inputs: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    *,
    rot_size: int,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Rotate groupwise regular-Hadamard activations and quantize to INT8."""

    require_cuda_tensors(inputs, rotation, scale)
    if inputs.ndim != 2:
        raise XQTBackendError("TileLang Hadamard quantization expects a 2D tensor")
    if rotation.ndim != 2:
        raise XQTBackendError("rotation must be a 2D matrix")
    if scale.numel() != 1:
        raise XQTBackendError("activation scale must contain one scalar")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError("TileLang Hadamard quantization expects fp16, bf16, or fp32 inputs")
    if rotation.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError("rotation must be fp16, bf16, or fp32")
    rows, features = int(inputs.shape[0]), int(inputs.shape[1])
    normalized_rot = int(rot_size)
    if rotation.shape != (normalized_rot, normalized_rot):
        raise XQTBackendError("rotation shape must match rot_size")
    if features % normalized_rot != 0:
        raise XQTBackendError("features must be divisible by rot_size")
    if normalized_rot % int(block_n) != 0 or normalized_rot % int(block_k) != 0:
        raise XQTBackendError("rot_size must align with TileLang Hadamard block sizes")
    padded, original_rows = pad_rows_to_block(inputs, int(block_m))
    padded_rows = int(padded.shape[0])
    kernel = build_tilelang_groupwise_hadamard_static_quant_kernel(
        padded_rows,
        features,
        normalized_rot,
        input_dtype=_activation_dtype_name(inputs.dtype),
        rotation_dtype=_activation_dtype_name(rotation.dtype),
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    out = torch.empty((padded_rows, features), device=inputs.device, dtype=torch.int8)
    kernel(
        padded.contiguous(),
        rotation.contiguous(),
        scale.reshape(1).to(device=inputs.device, dtype=torch.float32),
        out,
    )
    return out[:original_rows], padded_rows


@lru_cache(maxsize=32)
def build_tilelang_int8_linear_kernel(
    m: int,
    n: int,
    k: int,
    *,
    output_dtype: str,
    has_bias: bool,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> Any:
    """Build static-shape TileLang INT8 MMA with fused dequant epilogue."""

    tilelang = require_tilelang()
    import tilelang.language as T

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise ValueError("m, n, and k must be multiples of block_m, block_n, block_k")
    if output_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("output_dtype must be float16, bfloat16, or float32")

    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    a_shape = [m, k]
    b_shape = [k, n]
    scale_shape = [n]
    scalar_shape = [1]
    out_shape = [m, n]
    out_dtype = (
        T.float16
        if output_dtype == "float16"
        else T.bfloat16
        if output_dtype == "bfloat16"
        else T.float32
    )

    def int8_linear_with_bias_main(
        a: T.Tensor(a_shape, T.int8),
        b: T.Tensor(b_shape, T.int8),
        activation_scale: T.Tensor(scalar_shape, T.float32),
        weight_scale: T.Tensor(scale_shape, T.float32),
        bias: T.Tensor(scale_shape, T.float32),
        out: T.Tensor(out_shape, out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], T.int8)
            b_shared = T.alloc_shared([block_k, block_n], T.int8)
            acc = T.alloc_fragment([block_m, block_n], T.int32)

            T.fill(acc, 0)
            T.annotate_consumer_reg_alloc(255)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(
                    a[
                        bx * block_m : (bx + 1) * block_m,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    a_shared,
                )
                T.copy(
                    b[
                        k_tile * block_k : (k_tile + 1) * block_k,
                        by * block_n : (by + 1) * block_n,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out_col = by * block_n + col
                value = (
                    acc[row, col].astype(T.float32)
                    * activation_scale[0]
                    * weight_scale[out_col]
                    + bias[out_col]
                )
                out[bx * block_m + row, out_col] = value

    def int8_linear_without_bias_main(
        a: T.Tensor(a_shape, T.int8),
        b: T.Tensor(b_shape, T.int8),
        activation_scale: T.Tensor(scalar_shape, T.float32),
        weight_scale: T.Tensor(scale_shape, T.float32),
        out: T.Tensor(out_shape, out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], T.int8)
            b_shared = T.alloc_shared([block_k, block_n], T.int8)
            acc = T.alloc_fragment([block_m, block_n], T.int32)

            T.fill(acc, 0)
            T.annotate_consumer_reg_alloc(255)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(
                    a[
                        bx * block_m : (bx + 1) * block_m,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    a_shared,
                )
                T.copy(
                    b[
                        k_tile * block_k : (k_tile + 1) * block_k,
                        by * block_n : (by + 1) * block_n,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out_col = by * block_n + col
                value = (
                    acc[row, col].astype(T.float32)
                    * activation_scale[0]
                    * weight_scale[out_col]
                )
                out[bx * block_m + row, out_col] = value

    shape_suffix = (
        f"m{m}_n{n}_k{k}_{output_dtype}_bias{int(has_bias)}_"
        f"bm{block_m}_bn{block_n}_bk{block_k}_t{threads}_s{num_stages}"
    )
    if has_bias:
        int8_linear_with_bias_main.__name__ = f"int8_linear_with_bias_main_{shape_suffix}"
        prim = T.prim_func(int8_linear_with_bias_main)
    else:
        int8_linear_without_bias_main.__name__ = f"int8_linear_without_bias_main_{shape_suffix}"
        prim = T.prim_func(int8_linear_without_bias_main)

    def builder():
        return prim

    builder.__name__ = f"int8_linear_builder_{shape_suffix}"
    return tilelang.jit(
        out_idx=[],
        target=target,
        pass_configs=pass_configs,
    )(builder)()


def _output_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise XQTBackendError(f"unsupported INT8 Linear output dtype: {dtype}")


@lru_cache(maxsize=32)
def build_tilelang_int8_linear_static_activation_kernel(
    m: int,
    n: int,
    k: int,
    *,
    input_dtype: str,
    output_dtype: str,
    has_bias: bool,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> Any:
    """Build TileLang INT8 MMA with fused static activation quantization."""

    tilelang = require_tilelang()
    import tilelang.language as T

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if m % block_m != 0 or n % block_n != 0 or k % block_k != 0:
        raise ValueError("m, n, and k must be multiples of block_m, block_n, block_k")
    if input_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("input_dtype must be float16, bfloat16, or float32")
    if output_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("output_dtype must be float16, bfloat16, or float32")

    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    x_shape = [m, k]
    b_shape = [k, n]
    scale_shape = [n]
    scalar_shape = [1]
    out_shape = [m, n]
    x_dtype = (
        T.float16
        if input_dtype == "float16"
        else T.bfloat16
        if input_dtype == "bfloat16"
        else T.float32
    )
    out_dtype = (
        T.float16
        if output_dtype == "float16"
        else T.bfloat16
        if output_dtype == "bfloat16"
        else T.float32
    )

    def _quantize(value, scale):
        scaled = T.round(value.astype(T.float32) / scale)
        clipped = T.max(T.min(scaled, 127.0), -127.0)
        return clipped.astype(T.int8)

    def int8_linear_static_activation_with_bias_main(
        x: T.Tensor(x_shape, x_dtype),
        b: T.Tensor(b_shape, T.int8),
        activation_scale: T.Tensor(scalar_shape, T.float32),
        weight_scale: T.Tensor(scale_shape, T.float32),
        bias: T.Tensor(scale_shape, T.float32),
        out: T.Tensor(out_shape, out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], T.int8)
            b_shared = T.alloc_shared([block_k, block_n], T.int8)
            acc = T.alloc_fragment([block_m, block_n], T.int32)

            T.fill(acc, 0)
            T.annotate_consumer_reg_alloc(255)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                for row, col in T.Parallel(block_m, block_k):
                    a_shared[row, col] = _quantize(
                        x[bx * block_m + row, k_tile * block_k + col],
                        activation_scale[0],
                    )
                T.copy(
                    b[
                        k_tile * block_k : (k_tile + 1) * block_k,
                        by * block_n : (by + 1) * block_n,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out_col = by * block_n + col
                value = (
                    acc[row, col].astype(T.float32)
                    * activation_scale[0]
                    * weight_scale[out_col]
                    + bias[out_col]
                )
                out[bx * block_m + row, out_col] = value

    def int8_linear_static_activation_without_bias_main(
        x: T.Tensor(x_shape, x_dtype),
        b: T.Tensor(b_shape, T.int8),
        activation_scale: T.Tensor(scalar_shape, T.float32),
        weight_scale: T.Tensor(scale_shape, T.float32),
        out: T.Tensor(out_shape, out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], T.int8)
            b_shared = T.alloc_shared([block_k, block_n], T.int8)
            acc = T.alloc_fragment([block_m, block_n], T.int32)

            T.fill(acc, 0)
            T.annotate_consumer_reg_alloc(255)
            for k_tile in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                for row, col in T.Parallel(block_m, block_k):
                    a_shared[row, col] = _quantize(
                        x[bx * block_m + row, k_tile * block_k + col],
                        activation_scale[0],
                    )
                T.copy(
                    b[
                        k_tile * block_k : (k_tile + 1) * block_k,
                        by * block_n : (by + 1) * block_n,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc,
                    transpose_B=False,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out_col = by * block_n + col
                value = (
                    acc[row, col].astype(T.float32)
                    * activation_scale[0]
                    * weight_scale[out_col]
                )
                out[bx * block_m + row, out_col] = value

    shape_suffix = (
        f"m{m}_n{n}_k{k}_{input_dtype}_{output_dtype}_bias{int(has_bias)}_"
        f"bm{block_m}_bn{block_n}_bk{block_k}_t{threads}_s{num_stages}"
    )
    if has_bias:
        int8_linear_static_activation_with_bias_main.__name__ = (
            f"int8_linear_static_activation_with_bias_main_{shape_suffix}"
        )
        prim = T.prim_func(int8_linear_static_activation_with_bias_main)
    else:
        int8_linear_static_activation_without_bias_main.__name__ = (
            f"int8_linear_static_activation_without_bias_main_{shape_suffix}"
        )
        prim = T.prim_func(int8_linear_static_activation_without_bias_main)

    def builder():
        return prim

    builder.__name__ = f"int8_linear_static_activation_builder_{shape_suffix}"
    return tilelang.jit(
        out_idx=[],
        target=target,
        pass_configs=pass_configs,
    )(builder)()


def int8_linear_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run TileLang INT8 MMA and directly emit dequantized Linear output."""

    tensors = (a, b, activation_scale, weight_scale) if bias is None else (
        a,
        b,
        activation_scale,
        weight_scale,
        bias,
    )
    require_cuda_tensors(*tensors)
    m, n, k = _validate_int8_mma_inputs(
        a,
        b,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )
    if activation_scale.numel() != 1:
        raise XQTBackendError("activation_scale must contain one scalar")
    if weight_scale.shape != (n,):
        raise XQTBackendError("weight_scale must be shaped [out_features]")
    if bias is not None and bias.shape != (n,):
        raise XQTBackendError("bias must be shaped [out_features]")
    dtype_name = _output_dtype_name(output_dtype)
    kernel = build_tilelang_int8_linear_kernel(
        m,
        n,
        k,
        output_dtype=dtype_name,
        has_bias=bias is not None,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    scale = activation_scale.reshape(1).to(device=a.device, dtype=torch.float32)
    wscale = weight_scale.to(device=a.device, dtype=torch.float32).contiguous()
    if bias is not None:
        kernel(
            a.contiguous(),
            b.contiguous(),
            scale,
            wscale,
            bias.to(device=a.device, dtype=torch.float32).contiguous(),
            out,
        )
        return out
    kernel(a.contiguous(), b.contiguous(), scale, wscale, out)
    return out


def int8_linear_static_activation_tilelang(
    inputs: torch.Tensor,
    b: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run INT8 MMA Linear with activation quantization fused into TileLang."""

    tensors = (inputs, b, activation_scale, weight_scale) if bias is None else (
        inputs,
        b,
        activation_scale,
        weight_scale,
        bias,
    )
    require_cuda_tensors(*tensors)
    if inputs.ndim != 2 or b.ndim != 2:
        raise XQTBackendError("TileLang fused INT8 Linear expects 2D input tensors")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError("TileLang fused INT8 Linear expects fp16, bf16, or fp32 activations")
    if b.dtype != torch.int8:
        raise XQTBackendError("TileLang fused INT8 Linear expects int8 weights")
    if inputs.shape[1] != b.shape[0]:
        raise XQTBackendError("TileLang fused INT8 Linear requires inputs.shape[1] == b.shape[0]")
    m, k = int(inputs.shape[0]), int(inputs.shape[1])
    n = int(b.shape[1])
    if m <= 0 or n <= 0 or k <= 0:
        raise XQTBackendError("TileLang fused INT8 Linear dimensions must be positive")
    if m % int(block_m) != 0 or n % int(block_n) != 0 or k % int(block_k) != 0:
        raise XQTBackendError(
            "TileLang fused INT8 Linear requires m, n, and k to align with block sizes"
        )
    if activation_scale.numel() != 1:
        raise XQTBackendError("activation_scale must contain one scalar")
    if weight_scale.shape != (n,):
        raise XQTBackendError("weight_scale must be shaped [out_features]")
    if bias is not None and bias.shape != (n,):
        raise XQTBackendError("bias must be shaped [out_features]")

    kernel = build_tilelang_int8_linear_static_activation_kernel(
        m,
        n,
        k,
        input_dtype=_activation_dtype_name(inputs.dtype),
        output_dtype=_output_dtype_name(output_dtype),
        has_bias=bias is not None,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
        threads=int(threads),
        num_stages=int(num_stages),
        target_arch=target_arch,
    )
    out = torch.empty((m, n), device=inputs.device, dtype=output_dtype)
    scale = activation_scale.reshape(1).to(device=inputs.device, dtype=torch.float32)
    wscale = weight_scale.to(device=inputs.device, dtype=torch.float32).contiguous()
    if bias is not None:
        kernel(
            inputs.contiguous(),
            b.contiguous(),
            scale,
            wscale,
            bias.to(device=inputs.device, dtype=torch.float32).contiguous(),
            out,
        )
        return out
    kernel(inputs.contiguous(), b.contiguous(), scale, wscale, out)
    return out


def int8_linear_mma_reference(
    x: torch.Tensor,
    qweight_t: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference dequant epilogue for an INT8 MMA Linear path."""

    acc = int8_mma_reference(x, qweight_t)
    output = acc.to(torch.float32) * (
        activation_scale.to(torch.float32) * weight_scale.to(torch.float32)
    )
    if bias is not None:
        output = output + bias.to(dtype=output.dtype, device=output.device)
    return output


def int8_linear_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference dequantized INT8 Linear output for pre-quantized activations."""

    return int8_linear_mma_reference(
        a,
        b,
        activation_scale,
        weight_scale,
        bias,
    ).to(output_dtype)


def int8_linear_static_activation_reference(
    inputs: torch.Tensor,
    b: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference fused activation-quant INT8 Linear output."""

    if activation_scale.numel() != 1:
        raise XQTBackendError("activation_scale must contain one scalar")
    quantized = torch.round(
        inputs.to(torch.float32)
        / activation_scale.reshape(1).to(device=inputs.device, dtype=torch.float32)
    ).clamp(-127, 127).to(torch.int8)
    return int8_linear_reference(
        quantized,
        b,
        activation_scale,
        weight_scale,
        bias,
        output_dtype=output_dtype,
    )


def pad_rows_to_block(x: torch.Tensor, block_m: int) -> tuple[torch.Tensor, int]:
    """Pad rows to a block multiple and return the original row count."""

    rows = int(x.shape[0])
    remainder = rows % int(block_m)
    if remainder == 0:
        return x, rows
    padded_rows = rows + int(block_m) - remainder
    return F.pad(x, (0, 0, 0, padded_rows - rows), value=0), rows


def int8_linear_static_activation_m1_tilelang(
    inputs: torch.Tensor,
    b: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype,
    block_n: int = 128,
    block_k: int = 128,
    threads: int = 256,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run no-pad M=1 static-activation INT8 Linear (decode GEMV).

    TileLang sm_89 MMA cannot legally use block_m=1. This path keeps exact W8A8
    quantize-matmul-dequant semantics without row padding, using float32 products
    of int8 codes (CUDA has no int32 GEMM for M=1).
    """

    del block_n, block_k, threads, target_arch
    tensors = (inputs, b, activation_scale, weight_scale) if bias is None else (
        inputs,
        b,
        activation_scale,
        weight_scale,
        bias,
    )
    require_cuda_tensors(*tensors)
    if inputs.ndim != 2 or b.ndim != 2:
        raise XQTBackendError("M=1 INT8 Linear expects 2D tensors")
    if int(inputs.shape[0]) != 1:
        raise XQTBackendError("M=1 INT8 Linear requires exactly one input row")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError("M=1 INT8 Linear expects fp16, bf16, or fp32 activations")
    if b.dtype != torch.int8:
        raise XQTBackendError("M=1 INT8 Linear expects int8 weights")
    if inputs.shape[1] != b.shape[0]:
        raise XQTBackendError("M=1 INT8 Linear requires inputs.shape[1] == b.shape[0]")
    n = int(b.shape[1])
    if activation_scale.numel() != 1:
        raise XQTBackendError("activation_scale must contain one scalar")
    if weight_scale.shape != (n,):
        raise XQTBackendError("weight_scale must be shaped [out_features]")
    if bias is not None and bias.shape != (n,):
        raise XQTBackendError("bias must be shaped [out_features]")

    act = activation_scale.reshape(()).to(device=inputs.device, dtype=torch.float32)
    q_act = torch.round(inputs.to(torch.float32) / act).clamp(-127, 127)
    acc = torch.mm(q_act, b.to(torch.float32))
    out = acc * act * weight_scale.to(device=inputs.device, dtype=torch.float32).reshape(
        1, -1
    )
    if bias is not None:
        out = out + bias.to(device=inputs.device, dtype=torch.float32).reshape(1, -1)
    return out.to(dtype=output_dtype)


TILELANG_INT8_MMA_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "int8_mma": {
        "kernel_name": "int8_mma",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch._int_mm",
        "usage": "True W8A8 INT8 MMA with int32 accumulation.",
        "supported_precisions": ["int8"],
        "activation_encoding": "signed_int8",
        "weight_encoding": "signed_int8_transposed",
        "accumulation": "int32",
        "mma_instruction": "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        "quantization_nature": "true",
    },
    "int8_linear": {
        "kernel_name": "int8_linear",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch._int_mm + dequant epilogue",
        "usage": "True W8A8 GEMM with dequantized linear output epilogue.",
        "supported_precisions": ["int8"],
        "activation_encoding": "signed_int8",
        "weight_encoding": "signed_int8_transposed",
        "accumulation": "int32",
        "quantization_nature": "true",
        "fusion_status": "tilelang_dequant_output_epilogue",
    },
    "int8_linear_static_activation": {
        "kernel_name": "int8_linear_static_activation",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "activation_quant + torch._int_mm + dequant epilogue",
        "usage": "INT8 GEMM with activation quantization fused into the TileLang kernel.",
        "supported_precisions": ["int8"],
        "activation_encoding": "static_scaled_fp16_bf16_fp32_to_int8",
        "weight_encoding": "signed_int8_transposed",
        "accumulation": "int32",
        "quantization_nature": "true_with_fused_activation_quant",
        "fusion_status": "tilelang_static_activation_quant_dequant_output_epilogue",
    },
    "int8_linear_static_activation_m1": {
        "kernel_name": "int8_linear_static_activation_m1",
        "block_m": 1,
        "block_n": 0,
        "block_k": 0,
        "threads": 0,
        "baseline": "padded INT8 MMA",
        "usage": "No-pad M=1 decode GEMV with static activation quant (float32 products of int8).",
        "supported_precisions": ["int8"],
        "activation_encoding": "static_scaled_fp16_bf16_fp32_to_int8",
        "weight_encoding": "signed_int8_transposed",
        "accumulation": "float32_int8_products",
        "quantization_nature": "true_with_fused_activation_quant",
        "fusion_status": "torch_m1_static_activation_quant_dequant",
    },
}


__all__ = [
    "TILELANG_INT8_MMA_KERNEL_METADATA",
    "build_tilelang_int8_linear_kernel",
    "build_tilelang_int8_linear_static_activation_kernel",
    "build_tilelang_int8_mma_kernel",
    "build_tilelang_groupwise_hadamard_static_quant_kernel",
    "build_tilelang_static_activation_quant_kernel",
    "groupwise_hadamard_static_quantize_tilelang",
    "int8_linear_reference",
    "int8_linear_static_activation_m1_tilelang",
    "int8_linear_static_activation_reference",
    "int8_linear_static_activation_tilelang",
    "int8_linear_tilelang",
    "int8_linear_mma_reference",
    "int8_mma_reference",
    "int8_mma_tilelang",
    "pad_rows_to_block",
    "static_activation_quantize_tilelang",
]
