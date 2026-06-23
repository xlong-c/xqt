"""Minimal TileLang GEMM builders used by XQT operator kernels."""

from functools import lru_cache

@lru_cache(maxsize=32)
def build_tilelang_gemm_kernel(
    m: int,
    n: int,
    k: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, k must be positive")
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError("minimal TileLang GEMM builder currently requires m,n to be multiples of block sizes")

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    a_shape = [m, k]
    b_shape = [n, k]
    c_shape = [m, n]
    dtype = T.float16
    accum_dtype = T.float32

    @tilelang.jit(
        out_idx=[2],
        pass_configs=pass_configs,
    )
    def gemm():
        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            b: T.Tensor(b_shape, dtype),
            out: T.Tensor(c_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(m, block_m),
                T.ceildiv(n, block_n),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared([block_m, k], dtype)
                b_shared = T.alloc_shared([block_n, k], dtype)
                o_shared = T.alloc_shared([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, block_n], accum_dtype)

                T.copy(a[bx * block_m : (bx + 1) * block_m, :], a_shared)
                T.copy(b[by * block_n : (by + 1) * block_n, :], b_shared)
                T.fill(acc_o, 0)
                T.gemm(
                    a_shared,
                    b_shared,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(acc_o, o_shared)
                T.copy(o_shared, out[bx * block_m : (bx + 1) * block_m, by * block_n : (by + 1) * block_n])

        return main

    return gemm()


__all__ = ["build_tilelang_gemm_kernel"]
