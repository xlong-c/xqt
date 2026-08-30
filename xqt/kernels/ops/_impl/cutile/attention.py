"""CuTile Attention operator references and guarded entry points."""

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from ._common import require_cuda_tensors, require_cutile, require_fp16_tensors


@dataclass(frozen=True)
class CuTileAttentionDesign:
    """Reusable CuTile attention design metadata aligned with the TileLang pattern."""

    kernel_name: str = "fused_attention_forward"
    tensor_layout: str = "batch, heads, seq, head_dim"
    q_tile: str = "block_m x head_dim tile"
    k_tile: str = "block_n x head_dim tile"
    v_tile: str = "block_n x head_dim tile"
    accumulator: str = "block_m x head_dim fp32 accumulator"
    score_tile: str = "block_m x block_n fp32 score tile"
    softmax: str = "online softmax with running max/logsum"
    causal_mask: str = "lower-right aware mask when seq_kv >= seq_q"
    default_block_m: int = 64
    default_block_n: int = 64
    default_threads: int = 128
    source: str = "xqt.kernels.ops._impl.cutile.attention"
    production_status: str = "reference_guarded"
    limitations: tuple[str, ...] = (
        "The current CuTile entry validates CUDA fp16 inputs and uses SDPA reference fallback.",
        "Full cuTile JIT lowering is intentionally not compiled during package import.",
        "The design mirrors TileLang attention coverage for backend planning and artifacts.",
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limitations"] = list(self.limitations)
        return data


def build_cutile_attention_design(
    *,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
) -> CuTileAttentionDesign:
    """Return stable design metadata for a CuTile attention kernel."""

    return CuTileAttentionDesign(
        default_block_m=block_m,
        default_block_n=block_n,
        default_threads=threads,
    )


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise XQTBackendError(
            "CuTile attention expects 4D tensors shaped [batch, heads, seq, head_dim]"
        )
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise XQTBackendError(
            "CuTile attention requires matching batch size for q, k, v"
        )
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise XQTBackendError(
            "CuTile attention requires matching head count for q, k, v"
        )
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise XQTBackendError("CuTile attention requires matching head_dim for q, k, v")
    if k.shape[2] != v.shape[2]:
        raise XQTBackendError(
            "CuTile attention requires matching key/value sequence length"
        )
    if k.shape[3] != v.shape[3]:
        raise XQTBackendError("CuTile attention requires matching key/value head_dim")
    if k.shape[2] < q.shape[2]:
        raise XQTBackendError("CuTile attention currently requires seq_kv >= seq_q")
    if dropout_p != 0.0:
        raise XQTBackendError(
            "CuTile attention kernel does not yet support dropout_p != 0"
        )


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    if causal and q.size(-2) != k.size(-2):
        try:
            from torch.nn.attention.bias import causal_lower_right
        except Exception as exc:
            raise XQTBackendError(
                "non-square causal attention requires torch.nn.attention.bias.causal_lower_right"
            ) from exc
        attn_mask = causal_lower_right(q.size(-2), k.size(-2))
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=causal,
    )


def fused_attention_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Reference forward attention used by the CuTile backend."""

    return _sdpa_reference(q, k, v, causal=causal, dropout_p=dropout_p)


def fused_attention_forward_cutile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    dropout_p: float = 0.0,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
) -> torch.Tensor:
    """CUDA-only CuTile guarded attention entry point."""

    del block_m, block_n, threads
    require_cuda_tensors(q, k, v)
    require_fp16_tensors(q, k, v)
    _validate_attention_inputs(q, k, v, dropout_p=dropout_p)
    require_cutile()
    return fused_attention_forward_reference(
        q, k, v, causal=causal, dropout_p=dropout_p
    )


_ATTENTION_DESIGN = build_cutile_attention_design()

CUTILE_ATTENTION_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "attention": {
        "kernel_name": "fused_attention_forward",
        "block_m": _ATTENTION_DESIGN.default_block_m,
        "block_n": _ATTENTION_DESIGN.default_block_n,
        "threads": _ATTENTION_DESIGN.default_threads,
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "design": _ATTENTION_DESIGN.to_dict(),
        "production_status": "reference_guarded",
    },
}

__all__ = [
    "CUTILE_ATTENTION_KERNEL_METADATA",
    "CuTileAttentionDesign",
    "build_cutile_attention_design",
    "fused_attention_forward_cutile",
    "fused_attention_forward_reference",
]
