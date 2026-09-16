"""TileLang KV-int8 fused attention kernel.

该模块提供真正消费 int8 存储 K/V 的融合 attention CUDA kernel:
K/V 以 int8 tensor 加 per-tensor scale 传入, kernel 在把 tile 搬进
shared memory 后立刻做 int8 -> fp32 * scale -> fp16 的 dequant, 再进入
flash-attention 风格的 online softmax mainloop. dequant 发生在 kernel
内部, 不需要在 host 侧先物化 fp16 K/V.

量化语义与 `xqt.runtime.modules.kv_attention.KvScaleAttention` 的
`_quantize_kv` 完全一致: round(x / scale).clamp(-qmax, qmax).to(int8),
dequant 为 int8 * scale. 本模块只做 attention 本身, 不涉及 page table,
KV cache 管理或 serving 调度.

注意: 本文件不能使用 `from __future__ import annotations`, TileLang eager
builder 通过 get_type_hints 求值 prim_func 注解, 字符串化注解会丢失闭包内的
shape/dtype 变量.
"""

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.tilelang._common import (
    require_cuda_tensors,
    require_fp16_tensors,
    require_tilelang,
)

#: 已在本机 (sm_89) 验证的 fused kernel 名, report/contract 引用此常量.
KV_INT8_FUSED_KERNEL_NAME = "tilelang_kv_int8_fused_attention"
KV_INT8_PROJECTION_IO_KERNEL_NAME = (
    "tilelang_kv_int8_fused_attention_projection_io"
)
KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME = (
    "tilelang_kv_int8_fused_attention_packed_qkv_io"
)
KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME = (
    "tilelang_kv_int8_packed_qkv_quantize_layout"
)
KV_INT8_QUANTIZE_LAYOUT_KERNEL_NAME = "tilelang_kv_int8_quantize_layout"

_REFERENCE_KERNEL_NAME = "torch_sdpa_kv_int8_dequant_reference"


def _validate_kv_int8_inputs(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
) -> None:
    """校验 fused kernel 的输入形状与 dtype 契约."""

    if q.ndim != 4 or k_int8.ndim != 4 or v_int8.ndim != 4:
        raise XQTBackendError(
            "KV-int8 attention expects 4D tensors shaped [batch, heads, seq, head_dim]"
        )
    if k_int8.dtype != torch.int8 or v_int8.dtype != torch.int8:
        raise XQTBackendError("KV-int8 attention requires int8 K/V storage tensors")
    if q.shape[0] != k_int8.shape[0] or q.shape[0] != v_int8.shape[0]:
        raise XQTBackendError("KV-int8 attention requires matching batch size for q, k, v")
    if q.shape[1] != k_int8.shape[1] or q.shape[1] != v_int8.shape[1]:
        raise XQTBackendError("KV-int8 attention requires matching head count for q, k, v")
    if q.shape[3] != k_int8.shape[3] or q.shape[3] != v_int8.shape[3]:
        raise XQTBackendError("KV-int8 attention requires matching head_dim for q, k, v")
    if k_int8.shape[2] != v_int8.shape[2]:
        raise XQTBackendError("KV-int8 attention requires matching key/value sequence length")
    if k_int8.shape[2] < q.shape[2]:
        raise XQTBackendError("KV-int8 attention currently requires seq_kv >= seq_q")
    if q.shape[3] % 16 != 0:
        raise XQTBackendError(
            "KV-int8 attention requires head_dim to be a multiple of 16 for fp16 tensor-core GEMM"
        )


def _validate_scale_tensor(
    name: str,
    scale: torch.Tensor,
    *,
    expected_device: torch.device,
) -> None:
    """校验 device-resident per-tensor scale, 不读取其标量值."""

    if not isinstance(scale, torch.Tensor):
        raise XQTBackendError(
            f"KV-int8 attention {name} must be a CUDA float32 tensor"
        )
    if scale.dtype != torch.float32:
        raise XQTBackendError(
            f"KV-int8 attention {name} must have dtype torch.float32"
        )
    if scale.numel() != 1:
        raise XQTBackendError(
            f"KV-int8 attention {name} must contain exactly one element"
        )
    if scale.device != expected_device:
        raise XQTBackendError(
            f"KV-int8 attention {name} must be on {expected_device}; got {scale.device}"
        )


def _validate_kv_projection_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> tuple[int, int]:
    """校验量化前 K/V projection 的 BSI layout 契约."""

    require_cuda_tensors(k, v)
    require_fp16_tensors(k, v)
    if k.ndim != 3 or v.ndim != 3:
        raise XQTBackendError(
            "KV-int8 quantize-layout expects K/V shaped [batch, seq, inner_dim]"
        )
    if k.shape != v.shape:
        raise XQTBackendError("KV-int8 quantize-layout requires matching K/V shapes")
    if k.device != v.device:
        raise XQTBackendError("KV-int8 quantize-layout requires K/V on the same device")
    if not k.is_contiguous() or not v.is_contiguous():
        raise XQTBackendError(
            "KV-int8 quantize-layout requires contiguous K/V projections"
        )
    if int(heads) <= 0 or int(head_dim) <= 0:
        raise XQTBackendError("KV-int8 quantize-layout heads/head_dim must be positive")
    if int(k.shape[2]) != int(heads) * int(head_dim):
        raise XQTBackendError(
            "KV-int8 quantize-layout requires inner_dim == heads * head_dim"
        )
    return int(k.shape[0]), int(k.shape[1])


def _validate_packed_qkv_projection_input(
    qkv: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> tuple[int, int]:
    """校验等宽 packed QKV projection 的 BSI layout 契约."""

    require_cuda_tensors(qkv)
    require_fp16_tensors(qkv)
    if qkv.ndim != 3:
        raise XQTBackendError(
            "KV-int8 packed QKV expects [batch, seq, 3 * heads * head_dim]"
        )
    if not qkv.is_contiguous():
        raise XQTBackendError(
            "KV-int8 packed QKV requires a contiguous projection tensor"
        )
    if int(heads) <= 0 or int(head_dim) <= 0:
        raise XQTBackendError("KV-int8 packed QKV heads/head_dim must be positive")
    if int(qkv.shape[2]) != 3 * int(heads) * int(head_dim):
        raise XQTBackendError(
            "KV-int8 packed QKV requires packed_dim == 3 * heads * head_dim"
        )
    return int(qkv.shape[0]), int(qkv.shape[1])


def _validate_projection_attention_inputs(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    packed_qkv: bool = False,
) -> tuple[int, int, int]:
    """校验 Q/O projection-layout fused attention 输入契约."""

    require_cuda_tensors(q, k_int8, v_int8)
    require_fp16_tensors(q)
    if q.ndim != 3 or k_int8.ndim != 4 or v_int8.ndim != 4:
        raise XQTBackendError(
            "KV-int8 projection attention expects Q [batch, seq, inner_dim] "
            "and K/V [batch, heads, seq, head_dim]"
        )
    if k_int8.dtype != torch.int8 or v_int8.dtype != torch.int8:
        raise XQTBackendError(
            "KV-int8 projection attention requires int8 K/V storage tensors"
        )
    if q.device != k_int8.device or q.device != v_int8.device:
        raise XQTBackendError(
            "KV-int8 projection attention requires Q/K/V on the same device"
        )
    if not q.is_contiguous() or not k_int8.is_contiguous() or not v_int8.is_contiguous():
        raise XQTBackendError(
            "KV-int8 projection attention requires contiguous Q/K/V tensors"
        )
    if int(heads) <= 0 or int(head_dim) <= 0:
        raise XQTBackendError(
            "KV-int8 projection attention heads/head_dim must be positive"
        )
    expected_width = (3 if packed_qkv else 1) * int(heads) * int(head_dim)
    if int(q.shape[2]) != expected_width:
        raise XQTBackendError(
            "KV-int8 projection attention requires input width to match "
            "the selected Q or packed-QKV layout"
        )
    if int(q.shape[0]) != int(k_int8.shape[0]) or q.shape[0] != v_int8.shape[0]:
        raise XQTBackendError(
            "KV-int8 projection attention requires matching batch size for q, k, v"
        )
    if int(k_int8.shape[1]) != int(heads) or int(v_int8.shape[1]) != int(heads):
        raise XQTBackendError(
            "KV-int8 projection attention requires matching head count for q, k, v"
        )
    if int(k_int8.shape[3]) != int(head_dim) or int(v_int8.shape[3]) != int(head_dim):
        raise XQTBackendError(
            "KV-int8 projection attention requires matching head_dim for q, k, v"
        )
    if k_int8.shape[2] != v_int8.shape[2]:
        raise XQTBackendError(
            "KV-int8 projection attention requires matching key/value sequence length"
        )
    if int(k_int8.shape[2]) < int(q.shape[1]):
        raise XQTBackendError(
            "KV-int8 projection attention currently requires seq_kv >= seq_q"
        )
    if int(head_dim) % 16 != 0:
        raise XQTBackendError(
            "KV-int8 projection attention requires head_dim to be a multiple of 16 "
            "for fp16 tensor-core GEMM"
        )
    return int(q.shape[0]), int(q.shape[1]), int(k_int8.shape[2])


@lru_cache(maxsize=32)
def build_tilelang_kv_int8_quantize_layout_kernel(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    qmax: int = 127,
    block_size: int = 256,
) -> Any:
    """编译双输出 K/V static quantize + BSI -> BHSD layout kernel."""

    require_tilelang()
    import tilelang
    import tilelang.language as T

    if batch <= 0 or heads <= 0 or seq <= 0 or head_dim <= 0:
        raise ValueError("batch, heads, seq, and head_dim must be positive")
    if int(qmax) <= 0:
        raise ValueError("qmax must be positive")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")

    inner_dim = heads * head_dim
    projection_shape = [batch, seq, inner_dim]
    kv_shape = [batch, heads, seq, head_dim]
    scale_shape = [1]
    dtype = T.float16
    int8_dtype = "int8"
    total = batch * heads * seq * head_dim
    quant_max = float(qmax)
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

    @tilelang.jit(out_idx=[], pass_configs=pass_configs)
    def kernel() -> Any:
        @T.prim_func
        def tilelang_kv_int8_quantize_layout(
            k: T.Tensor(projection_shape, dtype),
            v: T.Tensor(projection_shape, dtype),
            k_scale: T.Tensor(scale_shape, T.float32),
            v_scale: T.Tensor(scale_shape, T.float32),
            k_int8: T.Tensor(kv_shape, int8_dtype),
            v_int8: T.Tensor(kv_shape, int8_dtype),
        ):
            with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as block:
                for offset in T.Parallel(block_size):
                    index = block * block_size + offset
                    if index < total:
                        dim_index = index % head_dim
                        seq_linear = index // head_dim
                        seq_index = seq_linear % seq
                        head_linear = seq_linear // seq
                        head_index = head_linear % heads
                        batch_index = head_linear // heads
                        feature_index = head_index * head_dim + dim_index

                        # PyTorch FP16 tensor / 0-d FP32 scale 使用 wrapped-scalar
                        # 语义;先把 scale cast 到 FP16 以保持相同舍入边界.
                        k_scaled = T.round(
                            T.cast(
                                k[batch_index, seq_index, feature_index]
                                / T.cast(k_scale[0], dtype),
                                T.float32,
                            )
                        )
                        v_scaled = T.round(
                            T.cast(
                                v[batch_index, seq_index, feature_index]
                                / T.cast(v_scale[0], dtype),
                                T.float32,
                            )
                        )
                        k_clipped = T.max(
                            T.min(k_scaled, quant_max),
                            -quant_max,
                        )
                        v_clipped = T.max(
                            T.min(v_scaled, quant_max),
                            -quant_max,
                        )
                        k_int8[
                            batch_index,
                            head_index,
                            seq_index,
                            dim_index,
                        ] = T.cast(k_clipped, int8_dtype)
                        v_int8[
                            batch_index,
                            head_index,
                            seq_index,
                            dim_index,
                        ] = T.cast(v_clipped, int8_dtype)

        return tilelang_kv_int8_quantize_layout

    return kernel()


def quantize_kv_int8_layout_tilelang(
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    qmax: int = 127,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次 TileLang launch 同时量化 K/V 并写成 BHSD INT8 layout."""

    batch, seq = _validate_kv_projection_inputs(
        k,
        v,
        heads=int(heads),
        head_dim=int(head_dim),
    )
    _validate_scale_tensor("k_scale", k_scale, expected_device=k.device)
    _validate_scale_tensor("v_scale", v_scale, expected_device=k.device)
    if int(qmax) <= 0:
        raise XQTBackendError("KV-int8 quantize-layout qmax must be positive")
    if int(block_size) <= 0:
        raise XQTBackendError("KV-int8 quantize-layout block_size must be positive")
    kernel = build_tilelang_kv_int8_quantize_layout_kernel(
        batch=batch,
        heads=int(heads),
        seq=seq,
        head_dim=int(head_dim),
        qmax=int(qmax),
        block_size=int(block_size),
    )
    output_shape = (batch, int(heads), seq, int(head_dim))
    k_int8 = torch.empty(output_shape, device=k.device, dtype=torch.int8)
    v_int8 = torch.empty(output_shape, device=v.device, dtype=torch.int8)
    kernel(
        k,
        v,
        k_scale.reshape(1),
        v_scale.reshape(1),
        k_int8,
        v_int8,
    )
    return k_int8, v_int8


@lru_cache(maxsize=32)
def build_tilelang_packed_qkv_int8_quantize_layout_kernel(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    qmax: int = 127,
    block_size: int = 256,
) -> Any:
    """编译 packed QKV -> 双输出 BHSD INT8 K/V kernel."""

    require_tilelang()
    import tilelang
    import tilelang.language as T

    if batch <= 0 or heads <= 0 or seq <= 0 or head_dim <= 0:
        raise ValueError("batch, heads, seq, and head_dim must be positive")
    if int(qmax) <= 0:
        raise ValueError("qmax must be positive")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")

    inner_dim = heads * head_dim
    projection_shape = [batch, seq, 3 * inner_dim]
    kv_shape = [batch, heads, seq, head_dim]
    scale_shape = [1]
    dtype = T.float16
    int8_dtype = "int8"
    total = batch * heads * seq * head_dim
    quant_max = float(qmax)
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

    @tilelang.jit(out_idx=[], pass_configs=pass_configs)
    def kernel() -> Any:
        @T.prim_func
        def tilelang_kv_int8_packed_qkv_quantize_layout(
            qkv: T.Tensor(projection_shape, dtype),
            k_scale: T.Tensor(scale_shape, T.float32),
            v_scale: T.Tensor(scale_shape, T.float32),
            k_int8: T.Tensor(kv_shape, int8_dtype),
            v_int8: T.Tensor(kv_shape, int8_dtype),
        ):
            with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as block:
                for offset in T.Parallel(block_size):
                    index = block * block_size + offset
                    if index < total:
                        dim_index = index % head_dim
                        seq_linear = index // head_dim
                        seq_index = seq_linear % seq
                        head_linear = seq_linear // seq
                        head_index = head_linear % heads
                        batch_index = head_linear // heads
                        feature_index = head_index * head_dim + dim_index
                        k_feature_index = inner_dim + feature_index
                        v_feature_index = 2 * inner_dim + feature_index

                        k_scaled = T.round(
                            T.cast(
                                qkv[batch_index, seq_index, k_feature_index]
                                / T.cast(k_scale[0], dtype),
                                T.float32,
                            )
                        )
                        v_scaled = T.round(
                            T.cast(
                                qkv[batch_index, seq_index, v_feature_index]
                                / T.cast(v_scale[0], dtype),
                                T.float32,
                            )
                        )
                        k_clipped = T.max(
                            T.min(k_scaled, quant_max),
                            -quant_max,
                        )
                        v_clipped = T.max(
                            T.min(v_scaled, quant_max),
                            -quant_max,
                        )
                        k_int8[
                            batch_index,
                            head_index,
                            seq_index,
                            dim_index,
                        ] = T.cast(k_clipped, int8_dtype)
                        v_int8[
                            batch_index,
                            head_index,
                            seq_index,
                            dim_index,
                        ] = T.cast(v_clipped, int8_dtype)

        return tilelang_kv_int8_packed_qkv_quantize_layout

    return kernel()


def quantize_packed_qkv_int8_layout_tilelang(
    qkv: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    qmax: int = 127,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次 launch 从 packed QKV 量化 K/V 并写成 BHSD INT8 layout."""

    batch, seq = _validate_packed_qkv_projection_input(
        qkv,
        heads=int(heads),
        head_dim=int(head_dim),
    )
    _validate_scale_tensor("k_scale", k_scale, expected_device=qkv.device)
    _validate_scale_tensor("v_scale", v_scale, expected_device=qkv.device)
    if int(qmax) <= 0:
        raise XQTBackendError("KV-int8 packed QKV qmax must be positive")
    if int(block_size) <= 0:
        raise XQTBackendError("KV-int8 packed QKV block_size must be positive")
    kernel = build_tilelang_packed_qkv_int8_quantize_layout_kernel(
        batch=batch,
        heads=int(heads),
        seq=seq,
        head_dim=int(head_dim),
        qmax=int(qmax),
        block_size=int(block_size),
    )
    output_shape = (batch, int(heads), seq, int(head_dim))
    k_int8 = torch.empty(output_shape, device=qkv.device, dtype=torch.int8)
    v_int8 = torch.empty(output_shape, device=qkv.device, dtype=torch.int8)
    kernel(
        qkv,
        k_scale.reshape(1),
        v_scale.reshape(1),
        k_int8,
        v_int8,
    )
    return k_int8, v_int8


@lru_cache(maxsize=32)
def build_tilelang_kv_int8_attention_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    projection_io: bool = False,
    packed_qkv_input: bool = False,
) -> Any:
    """编译 (并按形状缓存) KV-int8 fused attention TileLang kernel.

    mainloop 结构沿用 `xqt/kernels/ops/_impl/tilelang/flashatt_kernel.py` 的 flash-attention 设计,
    差异在于 K/V 输入为 int8, 每个 kv tile 先经 `T.copy` 进入 int8 shared,
    再在 kernel 内 dequant 为 fp16 shared 后参与 GEMM.
    """

    require_tilelang()
    import tilelang
    import tilelang.language as T

    scale = (1.0 / head_dim) ** 0.5 * 1.44269504
    if packed_qkv_input:
        q_shape = [batch, seq_q, 3 * heads * head_dim]
        out_shape = [batch, seq_q, heads * head_dim]
    elif projection_io:
        q_shape = [batch, seq_q, heads * head_dim]
        out_shape = q_shape
    else:
        q_shape = [batch, heads, seq_q, head_dim]
        out_shape = q_shape
    kv_shape = [batch, heads, seq_kv, head_dim]
    dtype = T.float16
    int8_dtype = "int8"
    accum_dtype = T.float32
    kv_scale_shape = [1]
    past_len = seq_kv - seq_q

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }

    @tilelang.jit(out_idx=[5], pass_configs=pass_configs)
    def kernel() -> Any:
        @T.prim_func
        def main(
            q: T.Tensor(q_shape, dtype),
            k_int8: T.Tensor(kv_shape, int8_dtype),
            v_int8: T.Tensor(kv_shape, int8_dtype),
            k_scale: T.Tensor(kv_scale_shape, T.float32),
            v_scale: T.Tensor(kv_scale_shape, T.float32),
            out: T.Tensor(out_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_q, block_m),
                heads,
                batch,
                threads=threads,
            ) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, head_dim], dtype)
                k_shared_int8 = T.alloc_shared([block_n, head_dim], int8_dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                v_shared_int8 = T.alloc_shared([block_n, head_dim], int8_dtype)
                v_shared = T.alloc_shared([block_n, head_dim], dtype)

                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, head_dim], accum_dtype)

                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                if projection_io or packed_qkv_input:
                    T.copy(
                        q[
                            bz,
                            bx * block_m : (bx + 1) * block_m,
                            by * head_dim : (by + 1) * head_dim,
                        ],
                        q_shared,
                    )
                else:
                    T.copy(
                        q[bz, by, bx * block_m : (bx + 1) * block_m, :],
                        q_shared,
                    )
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(
                        T.ceildiv(seq_kv, block_n),
                        T.ceildiv((bx + 1) * block_m + past_len, block_n),
                    )
                    if causal
                    else T.ceildiv(seq_kv, block_n)
                )

                for kv_tile in T.Pipelined(loop_range, num_stages=num_stages):
                    # int8 K tile 进 shared, kernel 内 dequant 为 fp16.
                    T.copy(
                        k_int8[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        k_shared_int8,
                    )
                    for i, j in T.Parallel(block_n, head_dim):
                        k_shared[i, j] = T.cast(
                            T.cast(k_shared_int8[i, j], accum_dtype) * k_scale[0],
                            dtype,
                        )

                    if causal:
                        for i, j in T.Parallel(block_m, block_n):
                            q_idx = bx * block_m + i + past_len
                            k_idx = kv_tile * block_n + j
                            acc_s[i, j] = T.if_then_else(
                                q_idx >= k_idx,
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                kv_tile * block_n + j >= seq_kv,
                                -T.infinity(acc_s.dtype),
                                0,
                            )

                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale
                        )

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_m, head_dim):
                        acc_o[i, j] *= scores_scale[i]

                    # int8 V tile 进 shared, kernel 内 dequant 为 fp16.
                    T.copy(
                        v_int8[bz, by, kv_tile * block_n : (kv_tile + 1) * block_n, :],
                        v_shared_int8,
                    )
                    for i, j in T.Parallel(block_n, head_dim):
                        v_shared[i, j] = T.cast(
                            T.cast(v_shared_int8[i, j], accum_dtype) * v_scale[0],
                            dtype,
                        )
                    T.gemm(
                        acc_s_cast,
                        v_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, head_dim):
                    acc_o[i, j] /= logsum[i]

                if projection_io or packed_qkv_input:
                    T.copy(
                        acc_o,
                        out[
                            bz,
                            bx * block_m : (bx + 1) * block_m,
                            by * head_dim : (by + 1) * head_dim,
                        ],
                    )
                else:
                    T.copy(
                        acc_o,
                        out[bz, by, bx * block_m : (bx + 1) * block_m, :],
                    )

        return main

    return kernel()


def kv_int8_attention_dequant_reference(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: float,
    v_scale: float,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """参考路径: int8 K/V 先 dequant 为 q 的 dtype, 再走 torch SDPA.

    dequant 语义为 int8 -> dtype * scale, 与 fused kernel 内的
    int8 -> fp32 * scale -> fp16 在 fp16 舍入级别一致.
    """

    k = k_int8.to(dtype=q.dtype) * float(k_scale)
    v = v_int8.to(dtype=q.dtype) * float(v_scale)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def fused_kv_int8_attention_forward_tilelang(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
) -> torch.Tensor:
    """CUDA-only KV-int8 fused attention 入口.

    输入契约: q 为 fp16 CUDA tensor [batch, heads, seq_q, head_dim];
    k_int8/v_int8 为 int8 CUDA tensor [batch, heads, seq_kv, head_dim];
    k_scale/v_scale 为与 q 同设备的单元素 fp32 tensor. dequant 在 kernel
    内完成, 不物化完整 fp16 K/V, 也不把 scale 同步回 host.
    """

    require_cuda_tensors(q, k_int8, v_int8)
    require_fp16_tensors(q)
    _validate_kv_int8_inputs(q, k_int8, v_int8)
    _validate_scale_tensor("k_scale", k_scale, expected_device=q.device)
    _validate_scale_tensor("v_scale", v_scale, expected_device=q.device)
    kernel = build_tilelang_kv_int8_attention_kernel(
        batch=int(q.shape[0]),
        heads=int(q.shape[1]),
        seq_q=int(q.shape[2]),
        seq_kv=int(k_int8.shape[2]),
        head_dim=int(q.shape[3]),
        causal=bool(causal),
        block_m=int(block_m),
        block_n=int(block_n),
        num_stages=int(num_stages),
        threads=int(threads),
    )
    return kernel(
        q,
        k_int8,
        v_int8,
        k_scale.reshape(1),
        v_scale.reshape(1),
    )


def fused_kv_int8_attention_projection_forward_tilelang(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
) -> torch.Tensor:
    """消费并产出 BSI projection layout 的 KV-int8 attention 入口."""

    batch, seq_q, seq_kv = _validate_projection_attention_inputs(
        q,
        k_int8,
        v_int8,
        heads=int(heads),
        head_dim=int(head_dim),
    )
    _validate_scale_tensor("k_scale", k_scale, expected_device=q.device)
    _validate_scale_tensor("v_scale", v_scale, expected_device=q.device)
    kernel = build_tilelang_kv_int8_attention_kernel(
        batch=batch,
        heads=int(heads),
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=int(head_dim),
        causal=bool(causal),
        block_m=int(block_m),
        block_n=int(block_n),
        num_stages=int(num_stages),
        threads=int(threads),
        projection_io=True,
    )
    return kernel(
        q,
        k_int8,
        v_int8,
        k_scale.reshape(1),
        v_scale.reshape(1),
    )


def fused_kv_int8_attention_packed_qkv_forward_tilelang(
    qkv: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
) -> torch.Tensor:
    """直接消费 packed QKV 并输出 BSI 的 KV-int8 attention 入口."""

    batch, seq_q, seq_kv = _validate_projection_attention_inputs(
        qkv,
        k_int8,
        v_int8,
        heads=int(heads),
        head_dim=int(head_dim),
        packed_qkv=True,
    )
    _validate_scale_tensor("k_scale", k_scale, expected_device=qkv.device)
    _validate_scale_tensor("v_scale", v_scale, expected_device=qkv.device)
    kernel = build_tilelang_kv_int8_attention_kernel(
        batch=batch,
        heads=int(heads),
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=int(head_dim),
        causal=bool(causal),
        block_m=int(block_m),
        block_n=int(block_n),
        num_stages=int(num_stages),
        threads=int(threads),
        packed_qkv_input=True,
    )
    return kernel(
        qkv,
        k_int8,
        v_int8,
        k_scale.reshape(1),
        v_scale.reshape(1),
    )


TILELANG_KV_INT8_ATTENTION_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "kv_int8_attention": {
        "kernel_name": KV_INT8_FUSED_KERNEL_NAME,
        "packed_qkv_attention_kernel": KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME,
        "packed_qkv_quantize_layout_kernel": (
            KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME
        ),
        "projection_io_kernel": KV_INT8_PROJECTION_IO_KERNEL_NAME,
        "quantize_layout_kernel": KV_INT8_QUANTIZE_LAYOUT_KERNEL_NAME,
        "block_m": 64,
        "block_n": 64,
        "quantize_block_size": 256,
        "threads": 128,
        "num_stages": 2,
        "baseline": _REFERENCE_KERNEL_NAME,
        "kv_storage": "int8 per-tensor scale, in-kernel dequant to fp16",
        "kv_scale_abi": "single-element CUDA fp32 tensors; no host scalar readback",
        "projection_io": "contiguous fp16 [batch, seq, heads * head_dim]",
        "packed_qkv_input": (
            "contiguous fp16 [batch, seq, 3 * heads * head_dim]"
        ),
        "source_mainloop": "xqt/kernels/ops/_impl/tilelang/flashatt_kernel.py",
    },
}

__all__ = [
    "KV_INT8_FUSED_KERNEL_NAME",
    "KV_INT8_PACKED_QKV_ATTENTION_KERNEL_NAME",
    "KV_INT8_PACKED_QKV_QUANTIZE_LAYOUT_KERNEL_NAME",
    "KV_INT8_PROJECTION_IO_KERNEL_NAME",
    "KV_INT8_QUANTIZE_LAYOUT_KERNEL_NAME",
    "TILELANG_KV_INT8_ATTENTION_KERNEL_METADATA",
    "build_tilelang_kv_int8_attention_kernel",
    "build_tilelang_kv_int8_quantize_layout_kernel",
    "build_tilelang_packed_qkv_int8_quantize_layout_kernel",
    "fused_kv_int8_attention_forward_tilelang",
    "fused_kv_int8_attention_packed_qkv_forward_tilelang",
    "fused_kv_int8_attention_projection_forward_tilelang",
    "kv_int8_attention_dequant_reference",
    "quantize_kv_int8_layout_tilelang",
    "quantize_packed_qkv_int8_layout_tilelang",
]
