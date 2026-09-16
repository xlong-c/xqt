"""Speculative decode kernels: multi-row verify attention and RoPE/KV scatter.

These back the W4A8 spec-decode route (R-055 ff). Unlike the single-row decode
kernels in the same module, every kernel here processes ``ROWS`` query rows in
one launch so a verification pass over k draft tokens pays one weight read and
one K/V sweep:

``decode_attention_rows_forward_triton``
    Split-K online-softmax attention for ``[1, heads, ROWS, head_dim]``
    queries over the same static KV cache. Each program still covers one
    (head, split); the ``ROWS`` queries share the K/V tiles as they stream, so
    the K/V bytes are read once per layer, not once per draft token.

``rope_write_qkv_rows_triton``
    HF split-half RoPE for ``ROWS`` q/k rows plus the KV-cache scatter at
    ``position + row`` (contiguous block starting at a device-resident
    position), so one launch covers all drafted tokens' rope and cache
    writes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xqt.core.errors import XQTBackendError


@triton.jit
def _decode_attn_rows_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    len_ptr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    KSTRIDE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Online-softmax attention for ROWS queries over ``[0, valid_len)`` slots.

    Query row ``r`` only attends slots ``<= len_ptr[r]``: row ``r`` is one
    draft position whose exclusive upper bound is the count of cache slots
    valid for that row (contiguous prefix includes the row's own token).
    """

    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // GROUP
    valid_lens = tl.load(len_ptr + tl.arange(0, ROWS))
    row_max = tl.max(valid_lens, axis=0)
    chunk = tl.cdiv(row_max, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, row_max)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_r = tl.arange(0, ROWS)
    # Scores go through tl.dot so the [ROWS, BLOCK_L, HEAD_DIM] product never
    # materializes as a register tensor (that form spills and makes a 4-row
    # verify step ~5x slower than four single-row steps).
    q = tl.load(q_ptr + (head * ROWS + offs_r[:, None]) * HEAD_DIM + offs_d[None, :])
    running_max = tl.full([ROWS], -1e30, tl.float32)
    running_sum = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, HEAD_DIM], tl.float32)
    for start_l in range(start, end, BLOCK_L):
        offs_l = start_l + tl.arange(0, BLOCK_L)
        in_range = offs_l < end
        k = tl.load(
            k_ptr + kv_head * KSTRIDE + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + kv_head * KSTRIDE + offs_l[:, None] * HEAD_DIM + offs_d[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * SCALE
        row_lens = valid_lens[:, None]
        scores = tl.where(
            (offs_l[None, :] < row_lens) & in_range[None, :],
            scores,
            -1e30,
        )
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        acc = acc * rescale[:, None] + tl.dot(
            probs.to(v.dtype), v, input_precision="ieee"
        ).to(tl.float32)
        running_sum = running_sum * rescale + tl.sum(probs, axis=1)
        running_max = new_max
    base = head * SPLITS + split
    tl.store(pm_ptr + base * ROWS + offs_r, running_max)
    tl.store(pl_ptr + base * ROWS + offs_r, running_sum)
    tl.store(
        pacc_ptr + (base * ROWS + offs_r[:, None]) * HEAD_DIM + offs_d[None, :],
        acc,
    )


@triton.jit
def _decode_attn_rows_merge_kernel(
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Log-sum-exp merge of split partials for ROWS query rows."""

    head = tl.program_id(0)
    row = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLITS)
    base = (head * SPLITS + offs_s) * ROWS + row
    part_max = tl.load(pm_ptr + base)
    part_sum = tl.load(pl_ptr + base)
    part_acc = tl.load(pacc_ptr + base[:, None] * HEAD_DIM + offs_d[None, :])
    shared_max = tl.max(part_max, axis=0)
    weights = tl.exp(part_max - shared_max)
    total = tl.sum(weights * part_sum, axis=0)
    merged = tl.sum(weights[:, None] * part_acc, axis=0) / tl.where(
        total > 0.0, total, 1.0
    )
    tl.store(out_ptr + (head * ROWS + row) * HEAD_DIM + offs_d, merged.to(OUT_DTYPE))


@triton.jit
def _rope_write_qkv_rows_kernel(
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
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    """RoPE ROWS q/k rows and scatter k/v into cache slots ``position + row``.

    Every row writes: padded rows (present because ``tl.arange`` needs a
    power-of-two extent) carry a repeat of the window's last token, so their
    slots hold a defined value instead of stale garbage that a later window
    could read.
    """

    row = tl.program_id(1)
    head = tl.program_id(0)
    offs = tl.arange(0, HEAD_DIM)
    position = tl.load(position_ptr) + row
    cos = tl.load(cos_ptr + position * HEAD_DIM + offs)
    sin = tl.load(sin_ptr + position * HEAD_DIM + offs)
    half = offs < HALF_DIM
    partner = tl.where(half, offs + HALF_DIM, offs - HALF_DIM)
    if head < Q_HEADS:
        src = (head * ROWS + row) * HEAD_DIM
        x = tl.load(q_ptr + src + offs)
        other = tl.load(q_ptr + src + partner)
        rotated = tl.where(half, -other, other)
        tl.store(q_out_ptr + src + offs, x * cos + rotated * sin)
    else:
        kv_head = head - Q_HEADS
        src = (kv_head * ROWS + row) * HEAD_DIM
        x = tl.load(k_ptr + src + offs)
        other = tl.load(k_ptr + src + partner)
        rotated = tl.where(half, -other, other)
        tl.store(
            k_cache_ptr + kv_head * CACHE_STRIDE + position * HEAD_DIM + offs,
            x * cos + rotated * sin,
        )
        v = tl.load(v_ptr + src + offs)
        tl.store(
            v_cache_ptr + kv_head * CACHE_STRIDE + position * HEAD_DIM + offs,
            v,
        )


def decode_attention_rows_forward_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    valid_lens: torch.Tensor,
    *,
    splits: int = 16,
    block_l: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Run GQA attention for ``[1, heads, ROWS, head_dim]`` query rows.

    ``valid_lens`` is a device int32 ``[ROWS]`` tensor: row ``r`` attends
    ``[0, valid_lens[r])`` cache slots. Returns ``[heads, ROWS, head_dim]``.
    """

    if not q.is_cuda or not k_cache.is_cuda or not v_cache.is_cuda:
        raise XQTBackendError("rows attention requires CUDA tensors")
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError("rows attention requires float16 or bfloat16 q")
    if q.ndim != 4 or int(q.shape[2]) < 1:
        raise XQTBackendError("rows attention expects [1, heads, ROWS, dim] q")
    heads = int(q.shape[1])
    rows = int(q.shape[2])
    head_dim = int(q.shape[3])
    kv_heads = int(k_cache.shape[1])
    if heads % kv_heads != 0:
        raise XQTBackendError("rows attention requires heads divisible by kv_heads")
    if int(k_cache.shape[3]) != head_dim or int(v_cache.shape[3]) != head_dim:
        raise XQTBackendError("rows attention requires matching head_dim")
    if int(k_cache.shape[2]) != int(v_cache.shape[2]):
        raise XQTBackendError("rows attention requires equal cache lengths")
    if valid_len_dtype_ok(valid_lens) is False or valid_lens.numel() != rows:
        raise XQTBackendError(
            "rows attention requires a [ROWS] int32 valid_lens tensor"
        )
    q_flat = q.permute(1, 2, 0, 3).reshape(heads, rows, head_dim).contiguous()
    max_len = int(k_cache.shape[2])
    k_flat = k_cache.reshape(kv_heads, max_len, head_dim)
    v_flat = v_cache.reshape(kv_heads, max_len, head_dim)
    part_max = torch.empty(heads, splits, rows, dtype=torch.float32, device=q.device)
    part_sum = torch.empty(heads, splits, rows, dtype=torch.float32, device=q.device)
    part_acc = torch.empty(
        heads, splits, rows, head_dim, dtype=torch.float32, device=q.device
    )
    out = torch.empty(heads, rows, head_dim, dtype=q.dtype, device=q.device)
    scale = 1.0 / float(head_dim) ** 0.5
    _decode_attn_rows_partial_kernel[(heads, splits)](
        q_flat,
        k_flat,
        v_flat,
        part_max,
        part_sum,
        part_acc,
        valid_lens,
        GROUP=heads // kv_heads,
        ROWS=rows,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        KSTRIDE=max_len * head_dim,
        BLOCK_L=block_l,
        SCALE=scale,
        num_stages=1,
        num_warps=num_warps,
    )
    _decode_attn_rows_merge_kernel[(heads, rows)](
        part_max,
        part_sum,
        part_acc,
        out,
        ROWS=rows,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        OUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        num_warps=1,
    )
    return out


def valid_len_dtype_ok(valid_lens: torch.Tensor) -> bool:
    return valid_lens.dtype == torch.int32 and valid_lens.ndim == 1


def rope_write_qkv_rows_triton(
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
    """Apply RoPE to ROWS q/k rows and scatter k/v at ``position + row``.

    ``q``/``k``/``v`` are ``[1, heads, ROWS, head_dim]``; ``q_out`` mirrors
    ``q``; ``position`` is a one-element device tensor holding the slot of
    row 0, and row ``r`` lands at ``position + r``.
    """

    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise XQTBackendError("rope_write_qkv_rows requires CUDA tensors")
    heads = int(q.shape[1])
    kv_heads = int(k.shape[1])
    rows = int(q.shape[2])
    head_dim = int(q.shape[3])
    if head_dim % 2 != 0:
        raise XQTBackendError("rope_write_qkv_rows requires an even head_dim")
    if int(k.shape[2]) != rows or int(v.shape[2]) != rows:
        raise XQTBackendError("rope_write_qkv_rows requires matching row counts")
    if (
        cos_table.dim() != 2
        or sin_table.dim() != 2
        or int(cos_table.shape[1]) != head_dim
        or int(sin_table.shape[1]) != head_dim
        or cos_table.shape != sin_table.shape
    ):
        raise XQTBackendError(
            "rope_write_qkv_rows expects [max_len, head_dim] cos/sin tables"
        )
    cache_len = int(k_cache.shape[2])
    if int(cos_table.shape[0]) < cache_len:
        raise XQTBackendError("rope_write_qkv_rows tables must cover the cache")
    _rope_write_qkv_rows_kernel[(heads + kv_heads, rows)](
        q.reshape(heads, rows, head_dim),
        k.reshape(kv_heads, rows, head_dim),
        v.reshape(kv_heads, rows, head_dim),
        cos_table,
        sin_table,
        q_out.reshape(heads, rows, head_dim),
        k_cache.reshape(kv_heads, cache_len, head_dim),
        v_cache.reshape(kv_heads, cache_len, head_dim),
        position,
        Q_HEADS=heads,
        ROWS=rows,
        HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        CACHE_STRIDE=cache_len * head_dim,
        num_warps=1,
    )
