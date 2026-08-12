"""Forward-only Triton Flash Attention kernels for XQT inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.runtime import target_arch_mismatch


_LOG2_E = 1.4426950408889634
_SUPPORTED_HEAD_DIMS = frozenset({16, 32, 64, 128})


@dataclass(frozen=True)
class TritonAttentionSchedule:
    """One immutable Triton forward-attention launch schedule."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    target_arch: str | None = None
    preset: str = "default"

    def to_dict(self) -> dict[str, int | str | None]:
        """Return JSON-friendly schedule metadata."""

        return {
            "block_m": self.block_m,
            "block_n": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "target_arch": self.target_arch,
            "preset": self.preset,
        }


_TRITON_ATTENTION_DEFAULT_SCHEDULE = (64, 64, 4, 2)
_TRITON_ATTENTION_SM89_PRESETS: dict[
    tuple[int, int, int, int, int, str, bool],
    tuple[str, tuple[int, int, int, int]],
] = {
    (1, 8, 1, 1024, 64, "float16", True): (
        "sm89_fp16_decode_q1_kv1024_d64",
        (16, 64, 4, 2),
    ),
    (1, 8, 1, 1024, 64, "bfloat16", True): (
        "sm89_bf16_decode_q1_kv1024_d64",
        (16, 128, 4, 2),
    ),
}


@lru_cache(maxsize=256)
def resolve_triton_attention_schedule(
    *,
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    input_dtype: str,
    causal: bool,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
) -> TritonAttentionSchedule:
    """Resolve exact-signature presets and explicit per-field overrides."""

    preset = "default"
    defaults = _TRITON_ATTENTION_DEFAULT_SCHEDULE
    if target_arch == "sm_89":
        resolved = _TRITON_ATTENTION_SM89_PRESETS.get(
            (
                int(batch),
                int(heads),
                int(seq_q),
                int(seq_kv),
                int(head_dim),
                str(input_dtype),
                bool(causal),
            )
        )
        if resolved is not None:
            preset, defaults = resolved

    return TritonAttentionSchedule(
        block_m=defaults[0] if block_m is None else int(block_m),
        block_n=defaults[1] if block_n is None else int(block_n),
        num_warps=defaults[2] if num_warps is None else int(num_warps),
        num_stages=defaults[3] if num_stages is None else int(num_stages),
        target_arch=target_arch,
        preset=preset,
    )


def _validate_attention_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise XQTBackendError(
            "Triton attention expects q, k, v shaped [batch, heads, seq, head_dim]"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise XQTBackendError("Triton attention requires matching q, k, v batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise XQTBackendError("Triton attention requires matching q, k, v head count")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise XQTBackendError("Triton attention requires matching q, k, v head_dim")
    if k.shape[2] != v.shape[2]:
        raise XQTBackendError(
            "Triton attention requires matching key/value sequence length"
        )
    if k.shape[2] < q.shape[2]:
        raise XQTBackendError("Triton attention currently requires seq_kv >= seq_q")
    if any(int(size) <= 0 for size in (*q.shape, int(k.shape[2]))):
        raise XQTBackendError("Triton attention dimensions must be positive")


def fused_attention_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """SDPA reference with lower-right causal semantics for non-square inputs."""

    _validate_attention_shapes(q, k, v)
    if q.device != k.device or q.device != v.device:
        raise XQTBackendError("Triton attention reference requires one shared device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise XQTBackendError("Triton attention reference requires matching q, k, v dtype")
    if causal and q.shape[2] != k.shape[2]:
        try:
            from torch.nn.attention.bias import causal_lower_right
        except Exception as exc:
            raise XQTBackendError(
                "non-square causal attention requires causal_lower_right support"
            ) from exc
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_lower_right(q.shape[2], k.shape[2]),
            dropout_p=dropout_p,
            scale=scale,
        )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_schedule(schedule: TritonAttentionSchedule) -> None:
    for name, value in (
        ("block_m", schedule.block_m),
        ("block_n", schedule.block_n),
    ):
        if value < 16 or not _is_power_of_two(value):
            raise XQTBackendError(
                f"Triton attention {name} must be a power of two and at least 16"
            )
    if schedule.num_warps not in {1, 2, 4, 8}:
        raise XQTBackendError(
            "Triton attention num_warps must be one of 1, 2, 4, or 8"
        )
    if schedule.num_stages <= 0:
        raise XQTBackendError("Triton attention num_stages must be positive")


@lru_cache(maxsize=16)
def _cuda_target_arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


@triton.jit
def _flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    sm_scale_log2,
    SEQ_Q: tl.constexpr,
    SEQ_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_base = pid_bh * SEQ_Q * HEAD_DIM
    kv_base = pid_bh * SEQ_KV * HEAD_DIM

    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < SEQ_Q) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

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
        k = tl.load(
            k_ptr + kv_base + kv_offsets[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(kv_offsets[:, None] < SEQ_KV) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_ptr + kv_base + kv_offsets[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(kv_offsets[:, None] < SEQ_KV) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        valid = (offs_m[:, None] < SEQ_Q) & (kv_offsets[None, :] < SEQ_KV)
        if CAUSAL:
            valid = valid & (
                offs_m[:, None] + (SEQ_KV - SEQ_Q) >= kv_offsets[None, :]
            )
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
        acc += tl.dot(probs.to(INPUT_DTYPE), v)
        row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
        row_max = new_row_max

    denominator = tl.where(row_sum == 0.0, 1.0, row_sum)
    output = acc / denominator[:, None]
    tl.store(
        out_ptr + q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        output.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < SEQ_Q) & (offs_d[None, :] < HEAD_DIM),
    )


def fused_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run constrained forward-only FP16/BF16 Flash Attention in Triton."""

    _validate_attention_shapes(q, k, v)
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise XQTBackendError("Triton attention requires CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise XQTBackendError("Triton attention requires one shared CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise XQTBackendError(
            "Triton attention requires matching float16 or bfloat16 tensors"
        )
    if q.dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError(
            "Triton attention requires matching float16 or bfloat16 tensors"
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise XQTBackendError(
            "Triton attention requires contiguous BHSD q, k, v tensors"
        )
    if int(q.shape[3]) not in _SUPPORTED_HEAD_DIMS:
        allowed = ", ".join(str(value) for value in sorted(_SUPPORTED_HEAD_DIMS))
        raise XQTBackendError(f"Triton attention head_dim must be one of {allowed}")
    if float(dropout_p) != 0.0:
        raise XQTBackendError("Triton attention does not support dropout_p != 0")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v)):
        raise XQTBackendError("Triton attention is inference-only and has no backward")

    resolved_scale = 1.0 / math.sqrt(int(q.shape[3])) if scale is None else float(scale)
    if not math.isfinite(resolved_scale):
        raise XQTBackendError("Triton attention scale must be finite")

    resolved_target_arch = target_arch or _cuda_target_arch(q.device)
    mismatch = target_arch_mismatch(target_arch, q)
    if mismatch is not None:
        raise XQTBackendError(
            f"Triton attention target architecture is not executable: {mismatch}"
        )
    input_dtype = "float16" if q.dtype == torch.float16 else "bfloat16"
    schedule = resolve_triton_attention_schedule(
        batch=int(q.shape[0]),
        heads=int(q.shape[1]),
        seq_q=int(q.shape[2]),
        seq_kv=int(k.shape[2]),
        head_dim=int(q.shape[3]),
        input_dtype=input_dtype,
        causal=causal,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        target_arch=resolved_target_arch,
    )
    _validate_schedule(schedule)

    block_d = triton.next_power_of_2(int(q.shape[3]))
    output = torch.empty_like(q)
    grid = (
        triton.cdiv(int(q.shape[2]), schedule.block_m),
        int(q.shape[0]) * int(q.shape[1]),
    )
    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        resolved_scale * _LOG2_E,
        SEQ_Q=int(q.shape[2]),
        SEQ_KV=int(k.shape[2]),
        HEAD_DIM=int(q.shape[3]),
        BLOCK_M=schedule.block_m,
        BLOCK_N=schedule.block_n,
        BLOCK_D=block_d,
        INPUT_DTYPE=tl.float16 if q.dtype == torch.float16 else tl.bfloat16,
        CAUSAL=causal,
        num_warps=schedule.num_warps,
        num_stages=schedule.num_stages,
    )
    return output


TRITON_ATTENTION_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "attention": {
        "kernel_name": "fused_attention_forward_triton",
        "algorithm": "FlashAttention-2 style online softmax forward",
        "tensor_layout": "batch, heads, seq, head_dim",
        "supported_dtypes": ["float16", "bfloat16"],
        "supported_head_dims": sorted(_SUPPORTED_HEAD_DIMS),
        "supports_non_square": True,
        "causal_semantics": "lower-right when seq_kv >= seq_q",
        "supports_dropout": False,
        "supports_backward": False,
        "requires_contiguous": True,
        "block_m": _TRITON_ATTENTION_DEFAULT_SCHEDULE[0],
        "block_n": _TRITON_ATTENTION_DEFAULT_SCHEDULE[1],
        "num_warps": _TRITON_ATTENTION_DEFAULT_SCHEDULE[2],
        "num_stages": _TRITON_ATTENTION_DEFAULT_SCHEDULE[3],
        "schedule_policy": "exact signature presets with explicit per-field override",
        "schedule_presets": [
            "sm89_fp16_decode_q1_kv1024_d64",
            "sm89_bf16_decode_q1_kv1024_d64",
        ],
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "hardware": "CUDA SM80+; performance evidence initially scoped to SM89",
        "source": (
            "Triton fused-attention tutorial and "
            "learn/flash_attention/06_flash_attention_v2.py"
        ),
    },
}


__all__ = [
    "TRITON_ATTENTION_KERNEL_METADATA",
    "TritonAttentionSchedule",
    "fused_attention_forward_reference",
    "fused_attention_forward_triton",
    "resolve_triton_attention_schedule",
]
