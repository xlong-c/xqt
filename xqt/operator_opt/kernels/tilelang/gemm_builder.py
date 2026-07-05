"""Minimal TileLang GEMM builders used by XQT operator kernels."""

# TileLang DSL patterns (T.Tensor, T.ceildiv, .astype(), PassConfigKey enum keys
# in @tilelang.jit pass_configs) trigger Pylance false positives; these rules are
# not applicable inside DSL-decorated builder functions.
# pyright: reportInvalidTypeForm=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

from functools import lru_cache


@lru_cache(maxsize=32)
def build_tilelang_gemm_kernel(
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
    has_bias: bool = False,
    activation: str | None = None,
):
    import tilelang
    import tilelang.language as T

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, k must be positive")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            "minimal TileLang GEMM builder currently requires m,n to be multiples of block sizes"
        )
    if k % block_k != 0:
        raise ValueError(
            "minimal TileLang GEMM builder currently requires k to be a multiple of block_k"
        )
    if activation not in {None, "gelu", "silu", "relu"}:
        raise ValueError(f"unsupported activation: {activation}")

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    a_shape = [m, k]
    b_shape = [n, k]
    bias_shape = [n]
    c_shape = [m, n]
    dtype = T.float16
    accum_dtype = T.float32

    def _apply_activation(value):
        if activation is None:
            return value
        if activation == "relu":
            return T.max(value, 0.0)
        if activation == "silu":
            return value * T.sigmoid(value)
        return 0.5 * value * (1.0 + T.erf(value / T.sqrt(2.0)))

    if has_bias:
        out_idx = [3]
    else:
        out_idx = [2]
    shape_suffix = (
        f"m{m}_n{n}_k{k}_bm{block_m}_bn{block_n}_bk{block_k}_"
        f"t{threads}_s{num_stages}_{activation or 'none'}"
    )

    def tilelang_gemm_with_bias_main(
        a: T.Tensor(a_shape, dtype),
        b: T.Tensor(b_shape, dtype),
        bias: T.Tensor(bias_shape, dtype),
        out: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], dtype)
            b_shared = T.alloc_shared([block_n, block_k], dtype)
            acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

            T.fill(acc_o, 0)
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
                        by * block_n : (by + 1) * block_n,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out[bx * block_m + row, by * block_n + col] = _apply_activation(
                    acc_o[row, col] + bias[by * block_n + col]
                )

    tilelang_gemm_with_bias_main.__name__ = (
        f"tilelang_gemm_with_bias_main_{shape_suffix}"
    )
    tilelang_gemm_with_bias_prim = T.prim_func(tilelang_gemm_with_bias_main)

    def tilelang_gemm_without_bias_main(
        a: T.Tensor(a_shape, dtype),
        b: T.Tensor(b_shape, dtype),
        out: T.Tensor(c_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(m, block_m),
            T.ceildiv(n, block_n),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared([block_m, block_k], dtype)
            b_shared = T.alloc_shared([block_n, block_k], dtype)
            acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

            T.fill(acc_o, 0)
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
                        by * block_n : (by + 1) * block_n,
                        k_tile * block_k : (k_tile + 1) * block_k,
                    ],
                    b_shared,
                )
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for row, col in T.Parallel(block_m, block_n):
                out[bx * block_m + row, by * block_n + col] = _apply_activation(
                    acc_o[row, col]
                )

    tilelang_gemm_without_bias_main.__name__ = (
        f"tilelang_gemm_without_bias_main_{shape_suffix}"
    )
    tilelang_gemm_without_bias_prim = T.prim_func(tilelang_gemm_without_bias_main)

    def gemm_with_bias():
        return tilelang_gemm_with_bias_prim
    gemm_with_bias.__name__ = f"tilelang_gemm_with_bias_builder_{shape_suffix}"
    gemm_with_bias_jit = tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )(gemm_with_bias)

    def gemm_without_bias():
        return tilelang_gemm_without_bias_prim
    gemm_without_bias.__name__ = f"tilelang_gemm_without_bias_builder_{shape_suffix}"
    gemm_without_bias_jit = tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )(gemm_without_bias)

    if has_bias:
        return gemm_with_bias_jit()
    return gemm_without_bias_jit()


@lru_cache(maxsize=32)
def build_tilelang_fp4_unpack_dequant_kernel(
    n: int,
    input_features: int,
    group_size: int,
    *,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
):
    import tilelang
    import tilelang.language as T

    if n <= 0 or input_features <= 0 or group_size <= 0:
        raise ValueError("n, input_features, and group_size must be positive")
    if block_n <= 0 or block_k <= 0:
        raise ValueError("block_n and block_k must be positive")

    padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    packed_k = (padded_input_features + 1) // 2
    groups = padded_input_features // group_size
    packed_shape = [n, packed_k]
    scale_shape = [n, groups, 1]
    weight_shape = [n, input_features]
    dtype = T.float16
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None

    @tilelang.jit(out_idx=[2], target=target)
    def unpack_dequant():
        @T.prim_func
        def main(
            packed_weight: T.Tensor(packed_shape, T.uint8),
            scale: T.Tensor(scale_shape, dtype),
            out: T.Tensor(weight_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(input_features, block_k),
                threads=threads,
            ) as (by, bx):
                for row_offset, feature_offset in T.Parallel(block_n, block_k):
                    row = by * block_n + row_offset
                    feature = bx * block_k + feature_offset
                    if row < n:
                        if feature < input_features:
                            byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                            low = T.bitwise_and(byte_u16, 15)
                            high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                            nibble = T.if_then_else(feature % 2 == 0, low, high)
                            signed = T.if_then_else(
                                nibble >= 8,
                                nibble.astype(T.int16) - 16,
                                nibble.astype(T.int16),
                            )
                            out[row, feature] = (
                                signed.astype(dtype)
                                * scale[row, feature // group_size, 0]
                            )

        return main

    return unpack_dequant()


@lru_cache(maxsize=32)
def build_tilelang_fp4_fused_dequant_gemm_kernel(
    m: int,
    n: int,
    input_features: int,
    group_size: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    target_arch: str | None = None,
    has_bias: bool = False,
    activation: str | None = None,
):
    import tilelang
    import tilelang.language as T

    if m <= 0 or n <= 0 or input_features <= 0 or group_size <= 0:
        raise ValueError("m, n, input_features, and group_size must be positive")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            "minimal fused FP4 TileLang GEMM requires m,n to be multiples of block sizes"
        )
    if activation not in {None, "gelu", "silu", "relu"}:
        raise ValueError(f"unsupported activation: {activation}")

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    packed_k = (padded_input_features + 1) // 2
    groups = padded_input_features // group_size
    a_shape = [m, input_features]
    packed_shape = [n, packed_k]
    scale_shape = [n, groups, 1]
    bias_shape = [n]
    c_shape = [m, n]
    dtype = T.float16
    accum_dtype = T.float32

    def _apply_activation(value):
        if activation is None:
            return value
        if activation == "relu":
            return T.max(value, 0.0)
        if activation == "silu":
            return value * T.sigmoid(value)
        return 0.5 * value * (1.0 + T.erf(value / T.sqrt(2.0)))

    if has_bias:
        out_idx = [4]
    else:
        out_idx = [3]

    @tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )
    def fused_gemm_with_bias():
        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            packed_weight: T.Tensor(packed_shape, T.uint8),
            scale: T.Tensor(scale_shape, dtype),
            bias: T.Tensor(bias_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(m, block_m),
                T.ceildiv(n, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, input_features], dtype)
                b_shared = T.alloc_shared([block_n, input_features], dtype)
                o_shared = T.alloc_shared([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.copy(a[bx * block_m : (bx + 1) * block_m, :], a_shared)
                for row_offset, feature in T.Parallel(block_n, input_features):
                    row = by * block_n + row_offset
                    byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                    low = T.bitwise_and(byte_u16, 15)
                    high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                    nibble = T.if_then_else(feature % 2 == 0, low, high)
                    signed = T.if_then_else(
                        nibble >= 8,
                        nibble.astype(T.int16) - 16,
                        nibble.astype(T.int16),
                    )
                    b_shared[row_offset, feature] = (
                        signed.astype(dtype) * scale[row, feature // group_size, 0]
                    )
                T.fill(acc_o, 0)
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for row_offset, column_offset in T.Parallel(block_m, block_n):
                    value = acc_o[row_offset, column_offset] + bias[
                        by * block_n + column_offset
                    ].astype(accum_dtype)
                    acc_o[row_offset, column_offset] = _apply_activation(value)
                T.copy(acc_o, o_shared)
                T.copy(
                    o_shared,
                    out[
                        bx * block_m : (bx + 1) * block_m,
                        by * block_n : (by + 1) * block_n,
                    ],
                )

        return main

    @tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )
    def fused_gemm_without_bias():
        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            packed_weight: T.Tensor(packed_shape, T.uint8),
            scale: T.Tensor(scale_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(m, block_m),
                T.ceildiv(n, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, input_features], dtype)
                b_shared = T.alloc_shared([block_n, input_features], dtype)
                o_shared = T.alloc_shared([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.copy(a[bx * block_m : (bx + 1) * block_m, :], a_shared)
                for row_offset, feature in T.Parallel(block_n, input_features):
                    row = by * block_n + row_offset
                    byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                    low = T.bitwise_and(byte_u16, 15)
                    high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                    nibble = T.if_then_else(feature % 2 == 0, low, high)
                    signed = T.if_then_else(
                        nibble >= 8,
                        nibble.astype(T.int16) - 16,
                        nibble.astype(T.int16),
                    )
                    b_shared[row_offset, feature] = (
                        signed.astype(dtype) * scale[row, feature // group_size, 0]
                    )
                T.fill(acc_o, 0)
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for row_offset, column_offset in T.Parallel(block_m, block_n):
                    acc_o[row_offset, column_offset] = _apply_activation(
                        acc_o[row_offset, column_offset]
                    )
                T.copy(acc_o, o_shared)
                T.copy(
                    o_shared,
                    out[
                        bx * block_m : (bx + 1) * block_m,
                        by * block_n : (by + 1) * block_n,
                    ],
                )

        return main

    if has_bias:
        return fused_gemm_with_bias()
    return fused_gemm_without_bias()


@lru_cache(maxsize=32)
def build_tilelang_nvfp4_fused_dequant_gemm_kernel(
    m: int,
    n: int,
    input_features: int,
    group_size: int,
    *,
    block_m: int = 64,
    block_n: int = 16,
    block_k: int = 128,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
    has_bias: bool = False,
    activation: str | None = None,
):
    import tilelang
    import tilelang.language as T

    if m <= 0 or n <= 0 or input_features <= 0 or group_size <= 0:
        raise ValueError("m, n, input_features, and group_size must be positive")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            "minimal fused NVFP4 TileLang GEMM requires m,n to be multiples of block sizes"
        )
    if input_features % block_k != 0:
        raise ValueError(
            "minimal fused NVFP4 TileLang GEMM requires input_features to be a multiple of block_k"
        )
    if activation not in {None, "gelu", "silu", "relu"}:
        raise ValueError(f"unsupported activation: {activation}")

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = {"kind": "cuda", "arch": str(target_arch)} if target_arch else None
    padded_input_features = (
        (input_features + group_size - 1) // group_size
    ) * group_size
    packed_k = (padded_input_features + 1) // 2
    groups = padded_input_features // group_size
    a_shape = [m, input_features]
    packed_shape = [n, packed_k]
    scale_shape = [n, groups, 1]
    global_scale_shape = [1]
    bias_shape = [n]
    c_shape = [m, n]
    dtype = T.float16
    accum_dtype = T.float32

    def _apply_activation(value):
        if activation is None:
            return value
        if activation == "relu":
            return T.max(value, 0.0)
        if activation == "silu":
            return value * T.sigmoid(value)
        return 0.5 * value * (1.0 + T.erf(value / T.sqrt(2.0)))

    def _decode_nvfp4_e2m1(nibble):
        unsigned_nibble = nibble.astype(T.uint16)
        magnitude_code = T.bitwise_and(unsigned_nibble, 7)
        sign_bit = T.bitwise_and(unsigned_nibble, 8)
        magnitude = T.if_then_else(
            magnitude_code <= 4,
            magnitude_code.astype(accum_dtype) * 0.5,
            T.if_then_else(
                magnitude_code == 5,
                3.0,
                T.if_then_else(magnitude_code == 6, 4.0, 6.0),
            ),
        )
        return T.if_then_else(sign_bit == 0, magnitude, -magnitude)

    if has_bias:
        out_idx = [5]
    else:
        out_idx = [4]

    @tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )
    def fused_gemm_with_bias():
        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            packed_weight: T.Tensor(packed_shape, T.uint8),
            scale: T.Tensor(scale_shape, dtype),
            global_scale: T.Tensor(global_scale_shape, dtype),
            bias: T.Tensor(bias_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(m, block_m),
                T.ceildiv(n, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, block_k], dtype)
                b_shared = T.alloc_shared([block_n, block_k], dtype)
                o_shared = T.alloc_shared([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.fill(acc_o, 0)
                for k_tile in T.Pipelined(
                    T.ceildiv(input_features, block_k), num_stages=num_stages
                ):
                    T.copy(
                        a[
                            bx * block_m : (bx + 1) * block_m,
                            k_tile * block_k : (k_tile + 1) * block_k,
                        ],
                        a_shared,
                    )
                    for row_offset, feature_offset in T.Parallel(block_n, block_k):
                        row = by * block_n + row_offset
                        feature = k_tile * block_k + feature_offset
                        byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                        low = T.bitwise_and(byte_u16, 15)
                        high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                        nibble = T.if_then_else(feature % 2 == 0, low, high)
                        decoded = _decode_nvfp4_e2m1(nibble)
                        b_shared[row_offset, feature_offset] = (
                            decoded.astype(dtype)
                            * scale[row, feature // group_size, 0]
                            / global_scale[0]
                        )
                    T.gemm(
                        a_shared,
                        b_shared,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for row_offset, column_offset in T.Parallel(block_m, block_n):
                    value = acc_o[row_offset, column_offset] + bias[
                        by * block_n + column_offset
                    ].astype(accum_dtype)
                    acc_o[row_offset, column_offset] = _apply_activation(value)
                T.copy(acc_o, o_shared)
                T.copy(
                    o_shared,
                    out[
                        bx * block_m : (bx + 1) * block_m,
                        by * block_n : (by + 1) * block_n,
                    ],
                )

        return main

    @tilelang.jit(
        out_idx=out_idx,
        target=target,
        pass_configs=pass_configs,
    )
    def fused_gemm_without_bias():
        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            packed_weight: T.Tensor(packed_shape, T.uint8),
            scale: T.Tensor(scale_shape, dtype),
            global_scale: T.Tensor(global_scale_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(m, block_m),
                T.ceildiv(n, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, block_k], dtype)
                b_shared = T.alloc_shared([block_n, block_k], dtype)
                o_shared = T.alloc_shared([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.fill(acc_o, 0)
                for k_tile in T.Pipelined(
                    T.ceildiv(input_features, block_k), num_stages=num_stages
                ):
                    T.copy(
                        a[
                            bx * block_m : (bx + 1) * block_m,
                            k_tile * block_k : (k_tile + 1) * block_k,
                        ],
                        a_shared,
                    )
                    for row_offset, feature_offset in T.Parallel(block_n, block_k):
                        row = by * block_n + row_offset
                        feature = k_tile * block_k + feature_offset
                        byte_u16 = packed_weight[row, feature // 2].astype(T.uint16)
                        low = T.bitwise_and(byte_u16, 15)
                        high = T.bitwise_and(T.shift_right(byte_u16, 4), 15)
                        nibble = T.if_then_else(feature % 2 == 0, low, high)
                        decoded = _decode_nvfp4_e2m1(nibble)
                        b_shared[row_offset, feature_offset] = (
                            decoded.astype(dtype)
                            * scale[row, feature // group_size, 0]
                            / global_scale[0]
                        )
                    T.gemm(
                        a_shared,
                        b_shared,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for row_offset, column_offset in T.Parallel(block_m, block_n):
                    acc_o[row_offset, column_offset] = _apply_activation(
                        acc_o[row_offset, column_offset]
                    )
                T.copy(acc_o, o_shared)
                T.copy(
                    o_shared,
                    out[
                        bx * block_m : (bx + 1) * block_m,
                        by * block_n : (by + 1) * block_n,
                    ],
                )

        return main

    if has_bias:
        return fused_gemm_with_bias()
    return fused_gemm_without_bias()


__all__ = [
    "build_tilelang_fp4_fused_dequant_gemm_kernel",
    "build_tilelang_fp4_unpack_dequant_kernel",
    "build_tilelang_gemm_kernel",
    "build_tilelang_nvfp4_fused_dequant_gemm_kernel",
]
