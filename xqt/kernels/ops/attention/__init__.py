"""attention kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from xqt.kernels.registry import register_kernel
from xqt.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)
from xqt.kernels.ops._legacy_api import load_legacy

_CUDA = frozenset({CapabilityRequirement.CUDA})


def _fused_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.TORCH,
        target="xqt.kernels.ops.attention:_fused_attention_torch",
        format_signature=FormatSignature(description="fused attention torch reference"),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.TRITON,
        target="xqt.kernels.ops._impl.triton.attention:fused_attention_forward_triton",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fused attention triton"),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.TILELANG,
        target="xqt.kernels.ops._impl.tilelang.attention:fused_attention_forward_tilelang",
        capabilities=_CUDA,
        format_signature=FormatSignature(description="fused attention tilelang"),
    )
)
register_kernel(
    KernelSpec(
        op="attention.fused_attention",
        backend=KernelBackend.SAGE,
        target=(
            "xqt.kernels.ops._impl.triton.sage_attention:sage_attention_forward_triton"
        ),
        capabilities=_CUDA,
        format_signature=FormatSignature(
            description=(
                "SageAttention-v1-style INT8 QK / FP16 PV forward attention "
                "(opt-in, measured slower than SDPA and Triton FA on sm_89)"
            )
        ),
    )
)

__all__ = [
    "_fused_attention_torch",
    "fused_attention",
    "fused_attention_sage",
    "fused_attention_tilelang",
    "fused_attention_triton",
    "recommend_attention_backend",
    "AttentionRoutingDecision",
]


def __getattr__(name: str) -> Any:
    return load_legacy("attention", name)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.TORCH)(
        q, k, v, is_causal=is_causal
    )


def fused_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.TRITON)(
        q, k, v, is_causal=is_causal
    )


def fused_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """Run the TileLang attention kernel through the unified registry."""

    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.TILELANG)(
        q,
        k,
        v,
        causal=causal,
        dropout_p=dropout_p,
        block_m=block_m,
        block_n=block_n,
        threads=threads,
        num_stages=num_stages,
        target_arch=target_arch,
    )


def fused_attention_sage(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """Run the opt-in SageAttention-v1-style kernel through the unified registry.

    This is an explicit opt-in path: the measured sm_89 length sweep found it
    slower than both SDPA and the Triton FA at almost every tested shape, so it
    is deliberately absent from the default backend precedence and is never
    auto-selected by :func:`recommend_attention_backend`.
    """

    from xqt.kernels.selector import get_kernel

    return get_kernel("attention.fused_attention", KernelBackend.SAGE)(
        q,
        k,
        v,
        causal=causal,
        dropout_p=dropout_p,
        scale=scale,
        smooth_k=smooth_k,
    )


@dataclass(frozen=True, slots=True)
class AttentionRoutingDecision:
    """One length-aware attention backend recommendation plus its reason."""

    backend: str
    reason: str
    bucket: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-friendly routing metadata."""

        return {
            "backend": self.backend,
            "reason": self.reason,
            "bucket": self.bucket,
        }


# Length buckets shared with the Triton attention schedule policy.  They keep
# the routing decision explainable without any tuning cache.
_DECODE_MAX_SEQ_Q = 1
_SHORT_MAX_SEQ_KV = 512
_MEDIUM_MAX_SEQ_KV = 2048

_BACKEND_SDPA = "sdpa"
_BACKEND_TRITON = "triton"
_BACKEND_TILELANG = "tilelang"


def _attention_length_bucket(*, seq_q: int, seq_kv: int) -> str:
    """Classify one shape into the decode/short/medium/long routing bucket."""

    if int(seq_q) <= _DECODE_MAX_SEQ_Q:
        return "decode"
    if int(seq_kv) <= _SHORT_MAX_SEQ_KV:
        return "short"
    if int(seq_kv) <= _MEDIUM_MAX_SEQ_KV:
        return "medium"
    return "long"


def _attention_backend_available(
    backend: str,
    available_backends: Sequence[str] | None,
) -> bool:
    """Return whether ``backend`` is selectable for the current runtime."""

    if available_backends is None:
        return True
    normalized = {str(name).strip().lower() for name in available_backends}
    return backend in normalized


def recommend_attention_backend(
    *,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    dtype: str = "float16",
    causal: bool = False,
    gqa: bool = False,
    available_backends: Sequence[str] | None = None,
) -> AttentionRoutingDecision:
    """Recommend a forward-attention backend from the measured length sweep.

    Policy (sm_89, directional starting point - not a hard speedup claim):

    - ``decode`` (``seq_q <= 1``): Triton, which won the short-KV decode cases.
    - GQA/MQA (``heads_q > heads_kv``): Triton, because TileLang has no GQA
      support and Triton measured fastest on the grouped-head shapes.
    - non-GQA ``medium``/``long`` prefill (``seq_kv > 512``): TileLang, which
      won prefill at medium and long lengths.
    - non-GQA ``short`` prefill (``seq_kv <= 512``): Triton.
    - anything unsupported by the preferred backend falls back to SDPA
      (``torch``).

    Sage is never returned: the sweep measured it slower than SDPA and Triton FA
    on sm_89 for almost all tested shapes, so it stays explicit opt-in.

    ``available_backends`` optionally narrows the selectable set (for example a
    CPU-only box, or a TileLang import failure).  When omitted every backend is
    treated as available.  This is a pure helper and is not wired into any
    decode hot path.
    """

    bucket = _attention_length_bucket(seq_q=int(seq_q), seq_kv=int(seq_kv))
    # ``head_dim``, ``dtype`` and ``causal`` are accepted so the policy signature
    # is stable; the measured sweep did not resolve per-field differences beyond
    # the length/GQA split, so they do not change the decision yet.

    if bucket == "decode":
        preferred, reason = (
            _BACKEND_TRITON,
            "decode: Triton won the short-KV decode cases",
        )
    elif bool(gqa):
        preferred, reason = (
            _BACKEND_TRITON,
            "GQA/MQA: TileLang has no GQA support and Triton won the grouped cases",
        )
    elif bucket in {"medium", "long"}:
        preferred, reason = (
            _BACKEND_TILELANG,
            f"non-GQA {bucket} prefill: TileLang won medium/long prefill",
        )
    else:
        preferred, reason = (
            _BACKEND_TRITON,
            "non-GQA short prefill: Triton won the short prefill cases",
        )

    if _attention_backend_available(preferred, available_backends):
        return AttentionRoutingDecision(
            backend=preferred,
            reason=reason,
            bucket=bucket,
        )

    fallback_reason = f"{reason}; preferred backend {preferred!r} unavailable"
    if _attention_backend_available(_BACKEND_SDPA, available_backends):
        return AttentionRoutingDecision(
            backend=_BACKEND_SDPA,
            reason=f"{fallback_reason}; falling back to SDPA",
            bucket=bucket,
        )

    return AttentionRoutingDecision(
        backend=_BACKEND_SDPA,
        reason=f"{fallback_reason}; SDPA not declared available, returning torch default",
        bucket=bucket,
    )
