"""Decode-time CUDA kernels for the CUDA-graph runtime.

Two kernels back :mod:`xqt.runtime.graph_decode`:

``decode_attention_forward_triton``
    Single-pass GQA decode attention over a static KV cache. The number of
    valid cache slots is read from a device tensor *inside* the kernel, so one
    captured CUDA graph serves every decode step of a generation; there is no
    padding correction and no per-length re-capture. K/V bytes are read once
    per layer.

``rope_write_qkv_triton``
    HF-Llama (split-half) rotary embedding for one decode row plus the KV
    cache scatter, so RoPE, the k/v writes collapse into one launch.

Both kernels are forward-only, inference-only, and CUDA-graph safe: they take
device pointers plus device-resident scalars and never synchronize.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from xqt.core.errors import XQTBackendError


@triton.jit
def _round_half_away(values):
    """Round to the nearest integer, half away from zero."""

    return tl.where(values >= 0.0, tl.floor(values + 0.5), tl.ceil(values - 0.5))


@triton.jit
def _decode_attn_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    len_ptr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    KSTRIDE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Per-split online-softmax attention over ``[0, valid_len)`` cache slots."""

    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // GROUP
    valid_len = tl.load(len_ptr)
    chunk = tl.cdiv(valid_len, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, valid_len)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + head * HEAD_DIM + offs_d).to(tl.float32)
    running_max = tl.full((), -1e30, tl.float32)
    running_sum = tl.zeros((), tl.float32)
    acc = tl.zeros([HEAD_DIM], tl.float32)
    for start_l in range(start, end, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)
        in_range = offs_l < end
        k = tl.load(
            k_ptr + kv_head * KSTRIDE + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * SCALE
        scores = tl.where(in_range, scores, -1e30)
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max)
        v = tl.load(
            v_ptr + kv_head * KSTRIDE + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * rescale + tl.sum(probs[:, None] * v, axis=0)
        running_sum = running_sum * rescale + tl.sum(probs, axis=0)
        running_max = new_max

    tl.store(pm_ptr + head * SPLITS + split, running_max)
    tl.store(pl_ptr + head * SPLITS + split, running_sum)
    tl.store(pacc_ptr + (head * SPLITS + split) * HEAD_DIM + offs_d, acc)


@triton.jit
def _decode_attn_merge_kernel(
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    out_ptr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    QUANT: tl.constexpr,
):
    """Merge split partials with a log-sum-exp rescale and normalize.

    With ``QUANT`` the stored row is rounded through a symmetric INT8
    quantize/dequantize step (per-head scale), i.e. the attention output is
    handed to ``o_proj`` in an INT8-quantized form.
    """

    head = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLITS)
    part_max = tl.load(pm_ptr + head * SPLITS + offs_s)
    part_sum = tl.load(pl_ptr + head * SPLITS + offs_s)
    part_acc = tl.load(
        pacc_ptr + (head * SPLITS + offs_s)[:, None] * HEAD_DIM + offs_d[None, :]
    )
    shared_max = tl.max(part_max, axis=0)
    weights = tl.exp(part_max - shared_max)
    total = tl.sum(weights * part_sum, axis=0)
    merged = tl.sum(weights[:, None] * part_acc, axis=0) / tl.where(
        total > 0.0, total, 1.0
    )
    if QUANT:
        scale = tl.max(tl.abs(merged), axis=0) / 127.0
        merged = tl.where(
            scale > 0.0,
            _round_half_away(merged / scale) * scale,
            merged,
        )
    tl.store(out_ptr + head * HEAD_DIM + offs_d, merged.to(OUT_DTYPE))


@triton.jit
def _decode_attn_partial_tc_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    len_ptr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    KSTRIDE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BM: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Tensor-core per-split attention: one program covers a whole GQA group.

    Same split geometry as ``_decode_attn_partial_kernel``, but the query tile
    spans all ``GROUP`` query heads of one KV head (padded to ``BM`` so
    ``tl.dot`` sees a legal tensor-core tile), which turns QK and PV into MMA
    work instead of per-element dot products. Rows >= ``GROUP`` stay masked out
    of both the loads and the partial stores.
    """

    kv_head = tl.program_id(0)
    split = tl.program_id(1)
    valid_len = tl.load(len_ptr)
    chunk = tl.cdiv(valid_len, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, valid_len)

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, HEAD_DIM)
    head_ids = kv_head * GROUP + offs_m
    row_valid = offs_m < GROUP
    q = tl.load(
        q_ptr + head_ids[:, None] * HEAD_DIM + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    running_max = tl.full([BM], -1e30, tl.float32)
    running_sum = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HEAD_DIM], tl.float32)
    for start_l in range(start, end, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)
        in_range = offs_l < end
        k = tl.load(
            k_ptr + kv_head * KSTRIDE + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * SCALE
        scores = tl.where(in_range[None, :], scores, -1e30)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        v = tl.load(
            v_ptr + kv_head * KSTRIDE + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        acc = acc * rescale[:, None] + tl.dot(probs.to(v.dtype), v)
        running_sum = running_sum * rescale + tl.sum(probs, axis=1)
        running_max = new_max

    tl.store(pm_ptr + head_ids * SPLITS + split, running_max, mask=row_valid)
    tl.store(pl_ptr + head_ids * SPLITS + split, running_sum, mask=row_valid)
    tl.store(
        pacc_ptr + (head_ids[:, None] * SPLITS + split) * HEAD_DIM + offs_d[None, :],
        acc,
        mask=row_valid[:, None],
    )


@triton.jit
def _decode_attn_partial_tc_int8_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_scale_ptr,
    v_scale_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    len_ptr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    KSTRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BM: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Tensor-core per-split attention over token-wise INT8 quantized K/V."""

    kv_head = tl.program_id(0)
    split = tl.program_id(1)
    valid_len = tl.load(len_ptr)
    chunk = tl.cdiv(valid_len, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, valid_len)

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, HEAD_DIM)
    head_ids = kv_head * GROUP + offs_m
    row_valid = offs_m < GROUP
    q = tl.load(
        q_ptr + head_ids[:, None] * HEAD_DIM + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    running_max = tl.full([BM], -1e30, tl.float32)
    running_sum = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HEAD_DIM], tl.float32)

    k_base = k_ptr + kv_head * KSTRIDE
    v_base = v_ptr + kv_head * KSTRIDE
    ks_base = k_scale_ptr + kv_head * SCALE_STRIDE
    vs_base = v_scale_ptr + kv_head * SCALE_STRIDE

    for start_l in range(start, end, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)
        in_range = offs_l < end

        # Load INT8 K (1 byte per element)
        k_int = tl.load(
            k_base + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0,
        )
        k_s = tl.load(ks_base + offs_l, mask=in_range, other=0.0)
        k_fp = k_int.to(tl.float32) * k_s[:, None]

        scores = tl.dot(q, tl.trans(k_fp.to(q.dtype))) * SCALE
        scores = tl.where(in_range[None, :], scores, -1e30)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])

        # Load INT8 V (1 byte per element)
        v_int = tl.load(
            v_base + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0,
        )
        v_s = tl.load(vs_base + offs_l, mask=in_range, other=0.0)
        v_fp = v_int.to(tl.float32) * v_s[:, None]

        acc = acc * rescale[:, None] + tl.dot(probs.to(q.dtype), v_fp.to(q.dtype))
        running_sum = running_sum * rescale + tl.sum(probs, axis=1)
        running_max = new_max

    tl.store(pm_ptr + head_ids * SPLITS + split, running_max, mask=row_valid)
    tl.store(pl_ptr + head_ids * SPLITS + split, running_sum, mask=row_valid)
    tl.store(
        pacc_ptr + (head_ids[:, None] * SPLITS + split) * HEAD_DIM + offs_d[None, :],
        acc,
        mask=row_valid[:, None],
    )


@triton.jit
def _decode_attn_partial_tc_int4_kernel(
    q_ptr,
    k_packed_ptr,
    v_packed_ptr,
    k_scale_ptr,
    v_scale_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    len_ptr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    K_PACKED_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BM: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Tensor-core per-split attention over token-wise INT4 packed K/V."""

    kv_head = tl.program_id(0)
    split = tl.program_id(1)
    valid_len = tl.load(len_ptr)
    chunk = tl.cdiv(valid_len, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, valid_len)

    offs_m = tl.arange(0, BM)
    offs_half = tl.arange(0, HALF_DIM)
    head_ids = kv_head * GROUP + offs_m
    row_valid = offs_m < GROUP

    q1 = tl.load(
        q_ptr + head_ids[:, None] * HEAD_DIM + offs_half[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    q2 = tl.load(
        q_ptr + head_ids[:, None] * HEAD_DIM + (HALF_DIM + offs_half)[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    running_max = tl.full([BM], -1e30, tl.float32)
    running_sum = tl.zeros([BM], tl.float32)
    acc1 = tl.zeros([BM, HALF_DIM], tl.float32)
    acc2 = tl.zeros([BM, HALF_DIM], tl.float32)

    k_base = k_packed_ptr + kv_head * K_PACKED_STRIDE
    v_base = v_packed_ptr + kv_head * K_PACKED_STRIDE
    ks_base = k_scale_ptr + kv_head * SCALE_STRIDE
    vs_base = v_scale_ptr + kv_head * SCALE_STRIDE

    for start_l in range(start, end, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)
        in_range = offs_l < end

        # 1. Load INT4 Packed K: shape [BLOCK_L, HALF_DIM]
        k_packed = tl.load(
            k_base + offs_l[:, None] * HALF_DIM + offs_half[None, :],
            mask=in_range[:, None],
            other=0,
        )
        k_s = tl.load(ks_base + offs_l, mask=in_range, other=0.0)

        low_k = k_packed & 0x0F
        k1_s = tl.where(low_k >= 8, low_k - 16, low_k)
        k2_s = k_packed >> 4

        k1_fp = (k1_s.to(tl.float32) * k_s[:, None]).to(q1.dtype)
        k2_fp = (k2_s.to(tl.float32) * k_s[:, None]).to(q2.dtype)

        scores = (tl.dot(q1, tl.trans(k1_fp)) + tl.dot(q2, tl.trans(k2_fp))) * SCALE
        scores = tl.where(in_range[None, :], scores, -1e30)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None]).to(q1.dtype)

        # 2. Load INT4 Packed V: shape [BLOCK_L, HALF_DIM]
        v_packed = tl.load(
            v_base + offs_l[:, None] * HALF_DIM + offs_half[None, :],
            mask=in_range[:, None],
            other=0,
        )
        v_s = tl.load(vs_base + offs_l, mask=in_range, other=0.0)

        low_v = v_packed & 0x0F
        v1_s = tl.where(low_v >= 8, low_v - 16, low_v)
        v2_s = v_packed >> 4

        v1_fp = (v1_s.to(tl.float32) * v_s[:, None]).to(q1.dtype)
        v2_fp = (v2_s.to(tl.float32) * v_s[:, None]).to(q2.dtype)

        acc1 = acc1 * rescale[:, None] + tl.dot(probs, v1_fp)
        acc2 = acc2 * rescale[:, None] + tl.dot(probs, v2_fp)
        running_sum = running_sum * rescale + tl.sum(probs, axis=1)
        running_max = new_max

    tl.store(pm_ptr + head_ids * SPLITS + split, running_max, mask=row_valid)
    tl.store(pl_ptr + head_ids * SPLITS + split, running_sum, mask=row_valid)

    tl.store(
        pacc_ptr + (head_ids[:, None] * SPLITS + split) * HEAD_DIM + offs_half[None, :],
        acc1,
        mask=row_valid[:, None],
    )
    tl.store(
        pacc_ptr
        + (head_ids[:, None] * SPLITS + split) * HEAD_DIM
        + (HALF_DIM + offs_half)[None, :],
        acc2,
        mask=row_valid[:, None],
    )


@triton.jit
def _rmsnorm_int8_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    eps,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """RMSNorm followed by a per-row symmetric INT8 quantize/dequantize step."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < COLS
    x = tl.load(x_ptr + row * COLS + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / COLS
    normed = x * tl.math.rsqrt(variance + eps) * weight
    scale = tl.max(tl.abs(normed), axis=0) / 127.0
    quantized = tl.where(
        scale > 0.0,
        _round_half_away(normed / scale) * scale,
        normed,
    )
    tl.store(
        out_ptr + row * COLS + offs,
        quantized.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _swiglu_int8_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row-wise SwiGLU followed by a symmetric INT8 quantize/dequantize step."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < COLS
    base = row * COLS
    gate = tl.load(gate_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    value = gate * sigmoid * up
    scale = tl.max(tl.abs(value), axis=0) / 127.0
    value = tl.where(scale > 0.0, _round_half_away(value / scale) * scale, value)
    tl.store(
        out_ptr + base + offs,
        value.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _rmsnorm_true_int8_kernel(
    x_ptr,
    weight_ptr,
    out_q_ptr,
    out_scale_ptr,
    eps,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """RMSNorm that directly stores a symmetric INT8 row and its float32 scale."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < COLS
    x = tl.load(x_ptr + row * COLS + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / COLS
    normed = x * tl.math.rsqrt(variance + eps) * weight
    scale = tl.maximum(tl.max(tl.abs(normed), axis=0) / 127.0, 1e-8)
    q = tl.clamp(_round_half_away(normed / scale), -128.0, 127.0).to(tl.int8)
    tl.store(out_q_ptr + row * COLS + offs, q, mask=mask)
    tl.store(out_scale_ptr + row, scale)


@triton.jit
def _swiglu_true_int8_kernel(
    gate_ptr,
    up_ptr,
    out_q_ptr,
    out_scale_ptr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row-wise SwiGLU that directly stores a symmetric INT8 row and its float32 scale."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < COLS
    base = row * COLS
    gate = tl.load(gate_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    value = gate * sigmoid * up
    scale = tl.maximum(tl.max(tl.abs(value), axis=0) / 127.0, 1e-8)
    q = tl.clamp(_round_half_away(value / scale), -128.0, 127.0).to(tl.int8)
    tl.store(out_q_ptr + base + offs, q, mask=mask)
    tl.store(out_scale_ptr + row, scale)


@triton.jit
def _quantize_row_true_int8_kernel(
    x_ptr,
    out_q_ptr,
    out_scale_ptr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row-wise per-token symmetric INT8 quantization storing INT8 row and float32 scale."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < COLS
    base = row * COLS
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1e-8)
    q = tl.clamp(_round_half_away(x / scale), -128.0, 127.0).to(tl.int8)
    tl.store(out_q_ptr + base + offs, q, mask=mask)
    tl.store(out_scale_ptr + row, scale)


@triton.jit
def _rope_decode_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    position_ptr,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """Apply RoPE to q and k for a single token without writing to KV cache."""

    row = tl.program_id(0)
    offs = tl.arange(0, HEAD_DIM)
    position = tl.load(position_ptr)
    cos = tl.load(cos_ptr + position * HEAD_DIM + offs)
    sin = tl.load(sin_ptr + position * HEAD_DIM + offs)
    half = offs < HALF_DIM
    partner = tl.where(half, offs + HALF_DIM, offs - HALF_DIM)
    if row < Q_HEADS:
        x = tl.load(q_ptr + row * HEAD_DIM + offs)
        other = tl.load(q_ptr + row * HEAD_DIM + partner)
        rotated = tl.where(half, -other, other)
        tl.store(q_out_ptr + row * HEAD_DIM + offs, x * cos + rotated * sin)
    else:
        kv_head = row - Q_HEADS
        x = tl.load(k_ptr + kv_head * HEAD_DIM + offs)
        other = tl.load(k_ptr + kv_head * HEAD_DIM + partner)
        rotated = tl.where(half, -other, other)
        tl.store(k_out_ptr + kv_head * HEAD_DIM + offs, x * cos + rotated * sin)


@triton.jit
def _rope_write_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    position_ptr,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    """RoPE q and k, write the roped k and raw v into the cache at ``position``."""

    row = tl.program_id(0)
    offs = tl.arange(0, HEAD_DIM)
    position = tl.load(position_ptr)
    cos = tl.load(cos_ptr + position * HEAD_DIM + offs)
    sin = tl.load(sin_ptr + position * HEAD_DIM + offs)
    half = offs < HALF_DIM
    partner = tl.where(half, offs + HALF_DIM, offs - HALF_DIM)
    if row < Q_HEADS:
        x = tl.load(q_ptr + row * HEAD_DIM + offs)
        other = tl.load(q_ptr + row * HEAD_DIM + partner)
        rotated = tl.where(half, -other, other)
        tl.store(q_out_ptr + row * HEAD_DIM + offs, x * cos + rotated * sin)
    else:
        kv_head = row - Q_HEADS
        x = tl.load(k_ptr + kv_head * HEAD_DIM + offs)
        other = tl.load(k_ptr + kv_head * HEAD_DIM + partner)
        rotated = tl.where(half, -other, other)
        tl.store(
            k_cache_ptr + kv_head * CACHE_STRIDE + position * HEAD_DIM + offs,
            x * cos + rotated * sin,
        )
        v = tl.load(v_ptr + kv_head * HEAD_DIM + offs)
        tl.store(
            v_cache_ptr + kv_head * CACHE_STRIDE + position * HEAD_DIM + offs,
            v,
        )


@triton.jit
def _rope_write_qkv_int8_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    position_ptr,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
):
    """RoPE q and k, apply OScaR Omni-Token Scaling and store INT8 K/V with scales."""

    row = tl.program_id(0)
    offs = tl.arange(0, HEAD_DIM)
    position = tl.load(position_ptr)
    cos = tl.load(cos_ptr + position * HEAD_DIM + offs)
    sin = tl.load(sin_ptr + position * HEAD_DIM + offs)
    half = offs < HALF_DIM
    partner = tl.where(half, offs + HALF_DIM, offs - HALF_DIM)
    if row < Q_HEADS:
        x = tl.load(q_ptr + row * HEAD_DIM + offs)
        other = tl.load(q_ptr + row * HEAD_DIM + partner)
        rotated = tl.where(half, -other, other)
        tl.store(q_out_ptr + row * HEAD_DIM + offs, x * cos + rotated * sin)
    else:
        kv_head = row - Q_HEADS
        x = tl.load(k_ptr + kv_head * HEAD_DIM + offs)
        other = tl.load(k_ptr + kv_head * HEAD_DIM + partner)
        rotated = tl.where(half, -other, other)
        k_val = x * cos + rotated * sin
        v_val = tl.load(v_ptr + kv_head * HEAD_DIM + offs)

        # OScaR Omni-Token Scaling for K
        k_fp = k_val.to(tl.float32)
        k_max = tl.max(tl.abs(k_fp), axis=0)
        k_scale = tl.maximum(k_max / 127.0, 1e-8)
        k_int8 = tl.clamp(_round_half_away(k_fp / k_scale), -128.0, 127.0)

        # OScaR Omni-Token Scaling for V
        v_fp = v_val.to(tl.float32)
        v_max = tl.max(tl.abs(v_fp), axis=0)
        v_scale = tl.maximum(v_max / 127.0, 1e-8)
        v_int8 = tl.clamp(_round_half_away(v_fp / v_scale), -128.0, 127.0)

        cache_base = kv_head * CACHE_STRIDE + position * HEAD_DIM
        tl.store(k_cache_ptr + cache_base + offs, k_int8.to(tl.int8))
        tl.store(v_cache_ptr + cache_base + offs, v_int8.to(tl.int8))

        tl.store(k_scale_ptr + kv_head * SCALE_STRIDE + position, k_scale)
        tl.store(v_scale_ptr + kv_head * SCALE_STRIDE + position, v_scale)


@triton.jit
def _pack_write_kv_int4_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    position_ptr,
    HALF_DIM: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
):
    """Pack and store token-wise INT4 K/V with per-token scales."""

    kv_h = tl.program_id(0)
    offs = tl.arange(0, HALF_DIM)
    position = tl.load(position_ptr)

    # 1. K
    k1 = tl.load(k_ptr + kv_h * (HALF_DIM * 2) + offs).to(tl.float32)
    k2 = tl.load(k_ptr + kv_h * (HALF_DIM * 2) + HALF_DIM + offs).to(tl.float32)
    max_k = tl.maximum(tl.max(tl.abs(k1), axis=0), tl.max(tl.abs(k2), axis=0))
    scale_k = tl.maximum(max_k / 7.0, 1e-8)
    q_k1 = tl.clamp(tl.math.floor(k1 / scale_k + 0.5), -8.0, 7.0).to(tl.int8)
    q_k2 = tl.clamp(tl.math.floor(k2 / scale_k + 0.5), -8.0, 7.0).to(tl.int8)
    packed_k = (q_k1 & 0x0F) | ((q_k2 & 0x0F) << 4)

    # 2. V
    v1 = tl.load(v_ptr + kv_h * (HALF_DIM * 2) + offs).to(tl.float32)
    v2 = tl.load(v_ptr + kv_h * (HALF_DIM * 2) + HALF_DIM + offs).to(tl.float32)
    max_v = tl.maximum(tl.max(tl.abs(v1), axis=0), tl.max(tl.abs(v2), axis=0))
    scale_v = tl.maximum(max_v / 7.0, 1e-8)
    q_v1 = tl.clamp(tl.math.floor(v1 / scale_v + 0.5), -8.0, 7.0).to(tl.int8)
    q_v2 = tl.clamp(tl.math.floor(v2 / scale_v + 0.5), -8.0, 7.0).to(tl.int8)
    packed_v = (q_v1 & 0x0F) | ((q_v2 & 0x0F) << 4)

    cache_base = kv_h * CACHE_STRIDE + position * HALF_DIM
    tl.store(k_cache_ptr + cache_base + offs, packed_k)
    tl.store(v_cache_ptr + cache_base + offs, packed_v)

    tl.store(k_scale_ptr + kv_h * SCALE_STRIDE + position, scale_k)
    tl.store(v_scale_ptr + kv_h * SCALE_STRIDE + position, scale_v)


def _decode_attention_geometry(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    valid_len: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the calling convention shared by both decode-attention variants.

    They take ``[1, heads, 1, head_dim]`` q and ``[1, kv_heads, max_len,
    head_dim]`` caches plus a device-resident int32 ``valid_len``. Returns
    ``(heads, kv_heads, head_dim)``.
    """

    if not q.is_cuda or not k_cache.is_cuda or not v_cache.is_cuda:
        raise XQTBackendError("decode attention requires CUDA tensors")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("decode attention requires float16 or bfloat16 q")
    if q.ndim != 4 or k_cache.ndim != 4 or q.shape[2] != 1:
        raise XQTBackendError(
            "decode attention expects [1, heads, 1, dim] q and 4D caches"
        )
    heads = int(q.shape[1])
    kv_heads = int(k_cache.shape[1])
    head_dim = int(q.shape[3])
    if heads % kv_heads != 0:
        raise XQTBackendError("decode attention requires heads divisible by kv_heads")
    if int(k_cache.shape[3]) not in {head_dim, head_dim // 2} or int(
        v_cache.shape[3]
    ) != int(k_cache.shape[3]):
        raise XQTBackendError("decode attention requires matching cache head_dim")
    if int(k_cache.shape[2]) != int(v_cache.shape[2]):
        raise XQTBackendError("decode attention requires equal cache lengths")
    if valid_len.dtype != torch.int32 or valid_len.numel() != 1:
        raise XQTBackendError("valid_len must be a one-element int32 tensor")
    return heads, kv_heads, head_dim


def decode_attention_forward_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    valid_len: torch.Tensor,
    *,
    splits: int = 16,
    block_l: int = 128,
    num_warps: int = 2,
    quantize_out: bool = False,
) -> torch.Tensor:
    """Run GQA decode attention for a single query row.

    ``q`` is ``[1, heads, 1, head_dim]`` and the caches are
    ``[1, kv_heads, max_len, head_dim]``; ``valid_len`` is a device-resident
    int32 scalar. Returns ``[1, heads, 1, head_dim]``. With ``quantize_out``
    the output rows pass through a per-head symmetric INT8
    quantize/dequantize step.
    """

    heads, kv_heads, head_dim = _decode_attention_geometry(
        q, k_cache, v_cache, valid_len
    )
    if splits < 1 or block_l < 1:
        raise XQTBackendError("splits and block_l must be positive")
    if block_l & (block_l - 1):
        raise XQTBackendError("block_l must be a power of two")

    q_flat = q.reshape(heads, head_dim).contiguous()
    max_len = int(k_cache.shape[2])
    k_flat = k_cache.reshape(kv_heads, max_len, head_dim)
    v_flat = v_cache.reshape(kv_heads, max_len, head_dim)
    part_max = torch.empty(heads, splits, dtype=torch.float32, device=q.device)
    part_sum = torch.empty(heads, splits, dtype=torch.float32, device=q.device)
    part_acc = torch.empty(
        heads, splits, head_dim, dtype=torch.float32, device=q.device
    )
    out = torch.empty(heads, head_dim, dtype=q.dtype, device=q.device)
    scale = 1.0 / float(head_dim) ** 0.5
    _decode_attn_partial_kernel[(heads, splits)](
        q_flat,
        k_flat,
        v_flat,
        part_max,
        part_sum,
        part_acc,
        valid_len,
        GROUP=heads // kv_heads,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        KSTRIDE=int(k_cache.shape[2]) * head_dim,
        BLOCK_L=block_l,
        SCALE=scale,
        num_warps=num_warps,
    )
    _decode_attn_merge_kernel[(heads,)](
        part_max,
        part_sum,
        part_acc,
        out,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        OUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        QUANT=bool(quantize_out),
        num_warps=1,
    )
    return out.reshape(1, heads, 1, head_dim)


def decode_attention_forward_triton_tc(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    valid_len: torch.Tensor,
    *,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    splits: int = 16,
    block_l: int = 64,
    num_warps: int = 4,
    quantize_out: bool = False,
) -> torch.Tensor:
    """Tensor-core variant of :func:`decode_attention_forward_triton`.

    One program covers a whole GQA group (``heads // kv_heads`` query heads
    that share a KV head); the padded group tile goes through ``tl.dot`` for
    both QK and PV, the KV length is split exactly like the SIMT kernel, and
    the same ``_decode_attn_merge_kernel`` merges the partials. Supports
    standard float16/bfloat16 caches as well as token-wise INT8 caches via
    ``k_scale`` and ``v_scale``.
    """

    heads, kv_heads, head_dim = _decode_attention_geometry(
        q, k_cache, v_cache, valid_len
    )
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise XQTBackendError(
            "tensor-core decode attention requires head_dim >= 16 and a power of two"
        )
    if splits < 1 or block_l < 1:
        raise XQTBackendError("splits and block_l must be positive")
    if block_l < 16 or block_l & (block_l - 1):
        raise XQTBackendError(
            "tensor-core decode attention requires block_l >= 16 and a power of two"
        )

    group = heads // kv_heads
    q_flat = q.reshape(heads, head_dim).contiguous()
    max_len = int(k_cache.shape[2])
    part_max = torch.empty(heads, splits, dtype=torch.float32, device=q.device)
    part_sum = torch.empty(heads, splits, dtype=torch.float32, device=q.device)
    part_acc = torch.empty(
        heads, splits, head_dim, dtype=torch.float32, device=q.device
    )
    out = torch.empty(heads, head_dim, dtype=q.dtype, device=q.device)
    scale = 1.0 / float(head_dim) ** 0.5
    if int(k_cache.shape[3]) == head_dim // 2:
        if k_scale is None or v_scale is None:
            raise XQTBackendError("INT4 decode attention requires k_scale and v_scale")
        half_dim = head_dim // 2
        _decode_attn_partial_tc_int4_kernel[(kv_heads, splits)](
            q_flat,
            k_cache.reshape(kv_heads, max_len, half_dim),
            v_cache.reshape(kv_heads, max_len, half_dim),
            k_scale.reshape(kv_heads, max_len),
            v_scale.reshape(kv_heads, max_len),
            part_max,
            part_sum,
            part_acc,
            valid_len,
            GROUP=group,
            HEAD_DIM=head_dim,
            HALF_DIM=half_dim,
            SPLITS=splits,
            K_PACKED_STRIDE=max_len * half_dim,
            SCALE_STRIDE=max_len,
            BLOCK_L=block_l,
            BM=max(16, triton.next_power_of_2(group)),
            SCALE=scale,
            num_warps=num_warps,
        )
    elif k_scale is not None and v_scale is not None:
        k_flat = k_cache.reshape(kv_heads, max_len, head_dim)
        v_flat = v_cache.reshape(kv_heads, max_len, head_dim)
        _decode_attn_partial_tc_int8_kernel[(kv_heads, splits)](
            q_flat,
            k_flat,
            v_flat,
            k_scale.reshape(kv_heads, max_len),
            v_scale.reshape(kv_heads, max_len),
            part_max,
            part_sum,
            part_acc,
            valid_len,
            GROUP=group,
            HEAD_DIM=head_dim,
            SPLITS=splits,
            KSTRIDE=max_len * head_dim,
            SCALE_STRIDE=max_len,
            BLOCK_L=block_l,
            BM=max(16, triton.next_power_of_2(group)),
            SCALE=scale,
            num_warps=num_warps,
        )
    else:
        k_flat = k_cache.reshape(kv_heads, max_len, head_dim)
        v_flat = v_cache.reshape(kv_heads, max_len, head_dim)
        _decode_attn_partial_tc_kernel[(kv_heads, splits)](
            q_flat,
            k_flat,
            v_flat,
            part_max,
            part_sum,
            part_acc,
            valid_len,
            GROUP=group,
            HEAD_DIM=head_dim,
            SPLITS=splits,
            KSTRIDE=max_len * head_dim,
            BLOCK_L=block_l,
            BM=max(16, triton.next_power_of_2(group)),
            SCALE=scale,
            num_warps=num_warps,
        )
    _decode_attn_merge_kernel[(heads,)](
        part_max,
        part_sum,
        part_acc,
        out,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        OUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        QUANT=bool(quantize_out),
        num_warps=1,
    )
    return out.reshape(1, heads, 1, head_dim)


def rmsnorm_int8_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    num_warps: int = 8,
) -> torch.Tensor:
    """RMSNorm with a per-row symmetric INT8 quantize/dequantize step."""

    if not x.is_cuda or x.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("rmsnorm_int8 requires CUDA float16/bfloat16 input")
    if x.shape[-1] != weight.numel():
        raise XQTBackendError("rmsnorm_int8 weight must match the last dimension")
    rows = x.numel() // int(x.shape[-1])
    cols = int(x.shape[-1])
    block = max(triton.next_power_of_2(cols), 16)
    out = torch.empty_like(x)
    _rmsnorm_int8_kernel[(rows,)](
        x.reshape(rows, cols),
        weight.reshape(-1),
        out.reshape(rows, cols),
        float(eps),
        COLS=cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def swiglu_int8_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    num_warps: int = 4,
) -> torch.Tensor:
    """SwiGLU with a per-row symmetric INT8 quantize/dequantize step."""

    if not gate.is_cuda or gate.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("swiglu_int8 requires CUDA float16/bfloat16 input")
    if gate.shape != up.shape:
        raise XQTBackendError("swiglu_int8 requires gate and up to share a shape")
    cols = int(gate.shape[-1])
    rows = gate.numel() // cols
    block = max(triton.next_power_of_2(cols), 16)
    out = torch.empty_like(gate)
    _swiglu_int8_kernel[(rows,)](
        gate.reshape(rows, cols),
        up.reshape(rows, cols),
        out.reshape(rows, cols),
        COLS=cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def rmsnorm_true_int8_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm writing directly into INT8 activation and float32 scale."""

    if not x.is_cuda or x.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("rmsnorm_true_int8 requires CUDA float16/bfloat16 input")
    if not weight.is_cuda or weight.dtype != x.dtype:
        raise XQTBackendError("rmsnorm_true_int8 requires weight to match input dtype")
    cols = int(x.shape[-1])
    rows = x.numel() // cols
    block = max(triton.next_power_of_2(cols), 16)
    if out_q is None:
        out_q = torch.empty((*x.shape[:-1], cols), dtype=torch.int8, device=x.device)
    if out_scale is None:
        out_scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _rmsnorm_true_int8_kernel[(rows,)](
        x.reshape(rows, cols),
        weight.reshape(-1),
        out_q.reshape(rows, cols),
        out_scale.reshape(rows),
        float(eps),
        COLS=cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out_q, out_scale


def swiglu_true_int8_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU writing directly into INT8 activation and float32 scale."""

    if not gate.is_cuda or gate.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("swiglu_true_int8 requires CUDA float16/bfloat16 input")
    if gate.shape != up.shape:
        raise XQTBackendError("swiglu_true_int8 requires gate and up to share a shape")
    cols = int(gate.shape[-1])
    rows = gate.numel() // cols
    block = max(triton.next_power_of_2(cols), 16)
    if out_q is None:
        out_q = torch.empty(
            (*gate.shape[:-1], cols), dtype=torch.int8, device=gate.device
        )
    if out_scale is None:
        out_scale = torch.empty((rows,), dtype=torch.float32, device=gate.device)
    _swiglu_true_int8_kernel[(rows,)](
        gate.reshape(rows, cols),
        up.reshape(rows, cols),
        out_q.reshape(rows, cols),
        out_scale.reshape(rows),
        COLS=cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out_q, out_scale


def quantize_row_true_int8_triton(
    x: torch.Tensor,
    *,
    out_q: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric INT8 quantization writing directly to INT8 tensor and float32 scale."""

    if not x.is_cuda or x.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError(
            "quantize_row_true_int8 requires CUDA float16/bfloat16 input"
        )
    cols = int(x.shape[-1])
    rows = x.numel() // cols
    block = max(triton.next_power_of_2(cols), 16)
    if out_q is None:
        out_q = torch.empty((*x.shape[:-1], cols), dtype=torch.int8, device=x.device)
    if out_scale is None:
        out_scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _quantize_row_true_int8_kernel[(rows,)](
        x.reshape(rows, cols),
        out_q.reshape(rows, cols),
        out_scale.reshape(rows),
        COLS=cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out_q, out_scale


def rope_write_qkv_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: torch.Tensor,
) -> None:
    """Apply HF-Llama RoPE to q/k and scatter k/v into the cache in one launch.

    ``cos_table``/``sin_table`` are ``[max_len, head_dim]`` HF rotary tables and
    ``position`` is a one-element device tensor selecting the row.
    """

    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise XQTBackendError("rope_write_qkv requires CUDA tensors")
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    head_dim = int(q.shape[3])
    if head_dim % 2 != 0:
        raise XQTBackendError("rope_write_qkv requires an even head_dim")
    if (
        cos_table.dim() != 2
        or sin_table.dim() != 2
        or int(cos_table.shape[1]) != head_dim
        or int(sin_table.shape[1]) != head_dim
        or cos_table.shape != sin_table.shape
    ):
        raise XQTBackendError(
            "rope_write_qkv expects [max_len, head_dim] cos/sin tables"
        )
    cache_len = int(k_cache.shape[2])
    if int(cos_table.shape[0]) < cache_len:
        raise XQTBackendError("rope_write_qkv tables must cover the whole cache")
    _rope_write_qkv_kernel[(heads + kv_heads,)](
        q.reshape(heads, head_dim),
        k.reshape(kv_heads, head_dim),
        v.reshape(kv_heads, head_dim),
        cos_table,
        sin_table,
        q_out.reshape(heads, head_dim),
        k_cache.reshape(kv_heads, cache_len, head_dim),
        v_cache.reshape(kv_heads, cache_len, head_dim),
        position,
        Q_HEADS=heads,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        CACHE_STRIDE=cache_len * head_dim,
        num_warps=1,
    )


def rope_write_qkv_int8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    position: torch.Tensor,
) -> None:
    """Apply RoPE and store token-wise INT8 K/V with per-token scales."""

    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise XQTBackendError("rope_write_qkv_int8 requires CUDA tensors")
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    head_dim = int(q.shape[3])
    if head_dim % 2 != 0:
        raise XQTBackendError("rope_write_qkv_int8 requires an even head_dim")
    cache_len = int(k_cache.shape[2])
    _rope_write_qkv_int8_kernel[(heads + kv_heads,)](
        q.reshape(heads, head_dim),
        k.reshape(kv_heads, head_dim),
        v.reshape(kv_heads, head_dim),
        cos_table,
        sin_table,
        q_out.reshape(heads, head_dim),
        k_cache.reshape(kv_heads, cache_len, head_dim),
        v_cache.reshape(kv_heads, cache_len, head_dim),
        k_scale.reshape(kv_heads, cache_len),
        v_scale.reshape(kv_heads, cache_len),
        position,
        Q_HEADS=heads,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        CACHE_STRIDE=cache_len * head_dim,
        SCALE_STRIDE=cache_len,
        num_warps=1,
    )


def hadamard_matrix(
    n: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate normalized Hadamard matrix H_n where H_n @ H_n^T = I."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard dimension must be a power of 2, got {n}")
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / math.sqrt(n)
    return h.to(dtype)


def pack_write_kv_int4_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    position: torch.Tensor,
) -> None:
    """Quantize K/V to token-wise 4-bit, pack into int8 cache and store per-token scales."""
    if not k.is_cuda or not v.is_cuda:
        raise XQTBackendError("pack_write_kv_int4 requires CUDA tensors")
    kv_heads = int(k.shape[1])
    head_dim = int(k.shape[3])
    if head_dim % 2 != 0:
        raise XQTBackendError("pack_write_kv_int4 requires an even head_dim")
    half_dim = head_dim // 2
    cache_len = int(k_cache.shape[2])
    _pack_write_kv_int4_kernel[(kv_heads,)](
        k.reshape(kv_heads, head_dim),
        v.reshape(kv_heads, head_dim),
        k_cache.reshape(kv_heads, cache_len, half_dim),
        v_cache.reshape(kv_heads, cache_len, half_dim),
        k_scale.reshape(kv_heads, cache_len),
        v_scale.reshape(kv_heads, cache_len),
        position,
        HALF_DIM=half_dim,
        CACHE_STRIDE=cache_len * half_dim,
        SCALE_STRIDE=cache_len,
        num_warps=1,
    )


def rope_decode_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    position: torch.Tensor,
) -> None:
    """Apply rotary embeddings to single-token q and k."""

    if not q.is_cuda or not k.is_cuda:
        raise XQTBackendError("rope_decode requires CUDA tensors")
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    head_dim = int(q.shape[3])
    if head_dim % 2 != 0:
        raise XQTBackendError("rope_decode requires an even head_dim")
    _rope_decode_kernel[(heads + kv_heads,)](
        q.reshape(heads, head_dim),
        k.reshape(kv_heads, head_dim),
        cos_table,
        sin_table,
        q_out.reshape(heads, head_dim),
        k_out.reshape(kv_heads, head_dim),
        position,
        Q_HEADS=heads,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        num_warps=1,
    )


__all__ = [
    "decode_attention_forward_triton",
    "decode_attention_forward_triton_tc",
    "hadamard_matrix",
    "pack_write_kv_int4_triton",
    "quantize_row_true_int8_triton",
    "rmsnorm_int8_triton",
    "rmsnorm_true_int8_triton",
    "rope_decode_triton",
    "rope_write_qkv_triton",
    "rope_write_qkv_int8_triton",
    "swiglu_int8_triton",
    "swiglu_true_int8_triton",
]
