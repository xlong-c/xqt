"""SageAttention-v1-style quantized forward attention in Triton.

This module implements a *SageAttention-v1-style* forward attention: the QK^T
product is quantized to INT8 with symmetric per-block scales and the PV product
runs in FP16/BF16 with an FP32 accumulator, all wrapped in an FA2-style online
softmax.

Honesty note
------------
SageAttention 2++ uses per-thread INT4 QK^T and FP8 PV with an FP16
accumulator. That full path is *not* reproduced here: this kernel is an
INT8 QK / FP16 PV / FP32 accumulate approximation in the spirit of
SageAttention v1. Nothing in this module is labeled "2++", "INT4" or "FP8", and
the metadata records the same limitation explicitly.

K smoothing (SageAttention v1)
------------------------------
When ``smooth_k=True`` the per-channel mean of K over the sequence dimension is
subtracted before K is quantized. For a fixed query row the subtracted term
``q @ mean`` is constant across all keys, so it cancels inside the softmax; the
mathematical result is unchanged, but the INT8 dynamic range of the K tile
narrows. The comparison gate against SDPA still has to be met with smoothing on,
which is verified separately from the unsmoothed path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from xqt.core.errors import XQTBackendError

from .decode_kernels import _round_half_away


_LOG2_E = 1.4426950408889634
_SUPPORTED_HEAD_DIMS = frozenset({32, 64, 128})
_SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16})


@dataclass(frozen=True)
class SageAttentionSchedule:
    """One immutable SageAttention-v1-style launch schedule."""

    block_m: int
    block_n: int
    block_s: int
    num_warps: int
    num_stages: int

    def to_dict(self) -> dict[str, int]:
        """Return JSON-friendly schedule metadata."""

        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "block_s": self.block_s,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


_SAGE_ATTENTION_DEFAULT_SCHEDULE = (64, 64, 128, 4, 2)


def _sage_schedule() -> SageAttentionSchedule:
    """Return the fixed tile schedule used for every head dimension."""

    block_m, block_n, block_s, num_warps, num_stages = _SAGE_ATTENTION_DEFAULT_SCHEDULE
    return SageAttentionSchedule(
        block_m=block_m,
        block_n=block_n,
        block_s=block_s,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _validate_sage_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Validate layout, device, dtype and GQA head mapping."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise XQTBackendError(
            "SageAttention expects q, k, v shaped [batch, heads, seq, head_dim]"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise XQTBackendError("SageAttention requires matching q, k, v batch size")
    if k.shape[1] != v.shape[1]:
        raise XQTBackendError("SageAttention requires matching k, v head count")
    if q.shape[1] % k.shape[1] != 0:
        raise XQTBackendError(
            "SageAttention GQA requires heads_q to be divisible by heads_kv"
        )
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise XQTBackendError("SageAttention requires matching q, k, v head_dim")
    if k.shape[2] != v.shape[2]:
        raise XQTBackendError(
            "SageAttention requires matching key/value sequence length"
        )
    if k.shape[2] < q.shape[2]:
        raise XQTBackendError("SageAttention currently requires seq_kv >= seq_q")
    if any(int(size) <= 0 for size in (*q.shape, int(k.shape[2]))):
        raise XQTBackendError("SageAttention dimensions must be positive")


@triton.jit
def _k_channel_mean_kernel(
    k_ptr,
    mean_ptr,
    SEQ_KV,
    HEAD_DIM,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-channel mean of K over the sequence dimension.

    One program handles one ``(batch, kv_head)`` slice and writes a
    ``[HEAD_DIM]`` mean vector. Accumulation is FP32.
    """

    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    base = pid * SEQ_KV * HEAD_DIM
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for start_s in range(0, SEQ_KV, BLOCK_S):
        offs_s = start_s + tl.arange(0, BLOCK_S)
        mask = (offs_s[:, None] < SEQ_KV) & (offs_d[None, :] < HEAD_DIM)
        tile = tl.load(
            k_ptr + base + offs_s[:, None] * HEAD_DIM + offs_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tile, axis=0)
    mean = acc / SEQ_KV
    tl.store(mean_ptr + pid * HEAD_DIM + offs_d, mean, mask=offs_d < HEAD_DIM)


@triton.jit
def _sage_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    smooth_ptr,
    out_ptr,
    sm_scale_log2,
    SEQ_Q,
    SEQ_KV,
    HEADS_Q: tl.constexpr,
    HEADS_KV: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    SMOOTH_K: tl.constexpr,
):
    """Online-softmax forward with INT8 QK^T and FP16 PV."""

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // HEADS_Q
    head_q = pid_bh % HEADS_Q
    kv_head = head_q // GROUP

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = pid_bh * SEQ_Q * HEAD_DIM
    kv_base = (batch_idx * HEADS_KV + kv_head) * SEQ_KV * HEAD_DIM

    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < SEQ_Q) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    mean = tl.zeros((BLOCK_D,), tl.float32)
    if SMOOTH_K:
        mean = tl.load(
            smooth_ptr + (batch_idx * HEADS_KV + kv_head) * HEAD_DIM + offs_d,
            mask=offs_d < HEAD_DIM,
            other=0.0,
        )

    # Per-row symmetric INT8 quantization of the Q tile.
    q_scale = tl.maximum(tl.max(tl.abs(q), axis=1) / 127.0, 1e-30)
    q_int = _round_half_away(q / q_scale[:, None])
    q_int = tl.maximum(tl.minimum(q_int, 127.0), -127.0).to(tl.int8)

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    loop_end = SEQ_KV
    if CAUSAL:
        loop_end = tl.minimum(
            (pid_m + 1) * BLOCK_M + (SEQ_KV - SEQ_Q),
            SEQ_KV,
        )
    for start_n in tl.range(0, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_offsets = start_n + offs_n
        kv_mask = (kv_offsets[:, None] < SEQ_KV) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(
            k_ptr + kv_base + kv_offsets[:, None] * HEAD_DIM + offs_d[None, :],
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        if SMOOTH_K:
            k = k - mean[None, :]
        v = tl.load(
            v_ptr + kv_base + kv_offsets[:, None] * HEAD_DIM + offs_d[None, :],
            mask=kv_mask,
            other=0.0,
        )

        # Per-row symmetric INT8 quantization of the K tile.
        k_scale = tl.maximum(tl.max(tl.abs(k), axis=1) / 127.0, 1e-30)
        k_int = _round_half_away(k / k_scale[:, None])
        k_int = tl.maximum(tl.minimum(k_int, 127.0), -127.0).to(tl.int8)

        # INT8 QK^T, dequantized by the outer product of the two scales.
        score_int = tl.dot(q_int, tl.trans(k_int), out_dtype=tl.int32)
        scores = score_int.to(tl.float32) * (q_scale[:, None] * k_scale[None, :])
        scores = scores * sm_scale_log2

        valid = (offs_m[:, None] < SEQ_Q) & (kv_offsets[None, :] < SEQ_KV)
        if CAUSAL:
            valid = valid & (offs_m[:, None] + (SEQ_KV - SEQ_Q) >= kv_offsets[None, :])
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, block_max)
        safe_new_row_max = tl.where(
            new_row_max > -float("inf"),
            new_row_max,
            0.0,
        )
        old_scale = tl.where(
            row_max > -float("inf"),
            tl.math.exp2(row_max - safe_new_row_max),
            0.0,
        )
        probs = tl.where(
            valid,
            tl.math.exp2(scores - safe_new_row_max[:, None]),
            0.0,
        )

        acc = acc * old_scale[:, None]
        acc += tl.dot(probs.to(INPUT_DTYPE), v.to(INPUT_DTYPE))
        row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
        row_max = new_row_max

    denominator = tl.where(row_sum == 0.0, 1.0, row_sum)
    output = acc / denominator[:, None]
    tl.store(
        out_ptr + q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < SEQ_Q) & (offs_d[None, :] < HEAD_DIM),
    )


def sage_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """Run SageAttention-v1-style INT8/FP16 forward attention in Triton.

    QK^T is computed in INT8 with symmetric per-block scales (per query row for
    Q, per key row for K) and dequantized by the outer product of the two
    scales; PV runs in the input FP16/BF16 dtype with an FP32 accumulator. With
    ``smooth_k=True`` the per-channel sequence mean of K is subtracted before K
    quantization, which narrows the INT8 dynamic range without changing the
    exact softmax result.

    Supported: ``[batch, heads, seq, head_dim]`` contiguous FP16/BF16, GQA
    (``heads_q % heads_kv == 0``), ``head_dim`` in ``{32, 64, 128}``, forward
    only. This is *not* SageAttention 2++ and does not use FP8.
    """

    _validate_sage_attention_inputs(q, k, v)
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise XQTBackendError("SageAttention requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise XQTBackendError("SageAttention requires one shared CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise XQTBackendError(
            "SageAttention requires matching float16 or bfloat16 tensors"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError(
            "SageAttention requires matching float16 or bfloat16 tensors"
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise XQTBackendError("SageAttention requires contiguous BHSD q, k, v tensors")
    head_dim = int(q.shape[3])
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        allowed = ", ".join(str(value) for value in sorted(_SUPPORTED_HEAD_DIMS))
        raise XQTBackendError(f"SageAttention head_dim must be one of {allowed}")
    if float(dropout_p) != 0.0:
        raise XQTBackendError("SageAttention does not support dropout_p != 0")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
        raise XQTBackendError("SageAttention is inference-only and has no backward")

    resolved_scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    if not math.isfinite(resolved_scale):
        raise XQTBackendError("SageAttention scale must be finite")

    heads_q = int(q.shape[1])
    heads_kv = int(k.shape[1])
    group = heads_q // heads_kv
    seq_q = int(q.shape[2])
    seq_kv = int(k.shape[2])
    schedule = _sage_schedule()
    block_d = triton.next_power_of_2(head_dim)

    smooth = None
    if smooth_k:
        smooth = torch.empty(
            (int(q.shape[0]) * heads_kv, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        _k_channel_mean_kernel[(int(q.shape[0]) * heads_kv,)](
            k,
            smooth,
            seq_kv,
            head_dim,
            BLOCK_S=schedule.block_s,
            BLOCK_D=block_d,
            num_warps=4,
        )

    output = torch.empty_like(q)
    grid = (
        triton.cdiv(seq_q, schedule.block_m),
        int(q.shape[0]) * heads_q,
    )
    _sage_attention_forward_kernel[grid](
        q,
        k,
        v,
        smooth if smooth is not None else k,
        output,
        resolved_scale * _LOG2_E,
        seq_q,
        seq_kv,
        HEADS_Q=heads_q,
        HEADS_KV=heads_kv,
        GROUP=group,
        HEAD_DIM=head_dim,
        BLOCK_M=schedule.block_m,
        BLOCK_N=schedule.block_n,
        BLOCK_D=block_d,
        INPUT_DTYPE=tl.float16 if q.dtype == torch.float16 else tl.bfloat16,
        CAUSAL=causal,
        SMOOTH_K=smooth_k,
        num_warps=schedule.num_warps,
        num_stages=schedule.num_stages,
    )
    return output


SAGE_ATTENTION_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "sage_attention": {
        "kernel_name": "sage_attention_forward_triton",
        "algorithm": (
            "FA2-style online-softmax forward attention with INT8 QK^T "
            "(symmetric per-block scales) and FP16/BF16 PV, FP32 accumulate"
        ),
        "attention_variant": "SageAttention-v1-style (INT8 QK, FP16 PV, FP32 acc)",
        "version_note": (
            "This is SageAttention-v1-style INT8/FP16, NOT SageAttention 2++ and "
            "NOT FP8: no per-thread INT4 QK^T, no FP8 PV, no FP16 accumulator."
        ),
        "qk_dtype": "int8 (symmetric per-block quantize, dequant by scale outer product)",
        "qk_scale": "per query row for Q, per key row for K, computed in FP32",
        "pv_dtype": "float16 or bfloat16, accumulated in float32",
        "k_smoothing": (
            "optional per-channel sequence mean subtraction before K quantization"
        ),
        "tensor_layout": "batch, heads, seq, head_dim",
        "supported_dtypes": ["float16", "bfloat16"],
        "supported_head_dims": sorted(_SUPPORTED_HEAD_DIMS),
        "supports_gqa": True,
        "gqa_requirement": "heads_q % heads_kv == 0",
        "supports_non_square": True,
        "causal_semantics": "lower-right when seq_kv >= seq_q",
        "supports_dropout": False,
        "supports_backward": False,
        "requires_contiguous": True,
        "block_m": _SAGE_ATTENTION_DEFAULT_SCHEDULE[0],
        "block_n": _SAGE_ATTENTION_DEFAULT_SCHEDULE[1],
        "block_s": _SAGE_ATTENTION_DEFAULT_SCHEDULE[2],
        "num_warps": _SAGE_ATTENTION_DEFAULT_SCHEDULE[3],
        "num_stages": _SAGE_ATTENTION_DEFAULT_SCHEDULE[4],
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "hardware": "CUDA SM80+; correctness evidence scoped to SM89",
        "source": (
            "SageAttention v1 (INT8 QK + smoothing) reimplemented on top of the "
            "XQT FA2-style Triton attention and decode INT8 quantize idioms"
        ),
    },
}


__all__ = [
    "SAGE_ATTENTION_KERNEL_METADATA",
    "SageAttentionSchedule",
    "sage_attention_forward_triton",
]
