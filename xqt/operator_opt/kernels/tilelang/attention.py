"""TileLang Attention operator references and guarded entry points."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError

from xqt.operator_opt.kernels.tilelang._common import (
    require_cuda_tensors,
    require_fp16_or_bf16_tensors,
    require_tilelang,
)


@dataclass(frozen=True)
class TileLangAttentionDesign:
    """Reusable TileLang attention design extracted from learn/tilelang/flashatt.py."""

    kernel_name: str = "fused_attention_forward"
    tensor_layout: str = "batch, heads, seq, head_dim"
    q_tile: str = "block_m x head_dim shared tile"
    k_tile: str = "block_n x head_dim shared tile"
    v_tile: str = "block_n x head_dim shared tile"
    accumulator: str = "block_m x head_dim fp32 accumulator"
    score_tile: str = "block_m x block_n fp32 score tile"
    softmax: str = "online softmax with running max/logsum"
    causal_mask: str = "lower-right aware mask when seq_kv >= seq_q"
    gemm_policy: str = "T.GemmWarpPolicy.FullRow"
    default_block_m: int = 64
    default_block_n: int = 64
    default_threads: int = 128
    default_num_stages: int = 2
    source: str = "learn/tilelang/flashatt.py"
    production_status: str = "runtime_kernel"
    limitations: tuple[str, ...] = (
        "Only CUDA tensors are accepted by the guarded TileLang entry point.",
        "The production entry JIT-compiles a fixed-shape TileLang kernel and keeps SDPA as the reference/fallback path.",
        "The extracted learning design supports fp16/bf16 and assumes seq_kv >= seq_q for non-square causal cases.",
        "Bfloat16 inputs require head_dim to be divisible by 16 for the current TileLang MMA lowering.",
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limitations"] = list(self.limitations)
        return data


def build_tilelang_attention_design(
    *,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
) -> TileLangAttentionDesign:
    """Return stable design metadata for a TileLang FlashAttention-style kernel."""

    return TileLangAttentionDesign(
        default_block_m=block_m,
        default_block_n=block_n,
        default_threads=threads,
        default_num_stages=num_stages,
    )



def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise XQTBackendError("TileLang attention expects 4D tensors shaped [batch, heads, seq, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise XQTBackendError("TileLang attention requires matching batch size for q, k, v")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise XQTBackendError("TileLang attention requires matching head count for q, k, v")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise XQTBackendError("TileLang attention requires matching head_dim for q, k, v")
    if k.shape[2] != v.shape[2]:
        raise XQTBackendError("TileLang attention requires matching key/value sequence length")
    if k.shape[3] != v.shape[3]:
        raise XQTBackendError("TileLang attention requires matching key/value head_dim")
    if q.dtype == torch.bfloat16 and int(q.shape[3]) % 16 != 0:
        raise XQTBackendError(
            "TileLang bfloat16 attention requires head_dim divisible by 16"
        )
    if k.shape[2] < q.shape[2]:
        raise XQTBackendError("TileLang attention currently requires seq_kv >= seq_q")
    if dropout_p != 0.0:
        raise XQTBackendError("TileLang attention kernel does not yet support dropout_p != 0")



@lru_cache(maxsize=32)
def _build_tilelang_flashatt_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    causal: bool,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    input_dtype: str,
) -> Any:
    require_tilelang()
    from learn.tilelang.flashatt import build_tilelang_flashatt

    return build_tilelang_flashatt(
        batch=batch,
        heads=heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=head_dim,
        causal=causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
        input_dtype=input_dtype,
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
    """Reference forward attention used by the TileLang backend."""

    return _sdpa_reference(q, k, v, causal=causal, dropout_p=dropout_p)


def fused_attention_forward_tilelang(
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
) -> torch.Tensor:
    """CUDA-only TileLang attention entry point."""

    require_cuda_tensors(q, k, v)
    require_fp16_or_bf16_tensors(q, k, v)
    _validate_attention_inputs(q, k, v, dropout_p=dropout_p)
    input_dtype = (
        "float16" if q.dtype == torch.float16 else "bfloat16"
    )
    kernel = _build_tilelang_flashatt_kernel(
        batch=int(q.shape[0]),
        heads=int(q.shape[1]),
        seq_q=int(q.shape[2]),
        seq_kv=int(k.shape[2]),
        head_dim=int(q.shape[3]),
        causal=causal,
        block_m=int(block_m),
        block_n=int(block_n),
        num_stages=int(num_stages),
        threads=int(threads),
        input_dtype=input_dtype,
    )
    return kernel(q, k, v)



_ATTENTION_DESIGN = build_tilelang_attention_design()

TILELANG_ATTENTION_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "attention": {
        "kernel_name": "fused_attention_forward",
        "block_m": _ATTENTION_DESIGN.default_block_m,
        "block_n": _ATTENTION_DESIGN.default_block_n,
        "threads": _ATTENTION_DESIGN.default_threads,
        "num_stages": _ATTENTION_DESIGN.default_num_stages,
        "supported_dtypes": ["float16", "bfloat16"],
        "bfloat16_head_dim_multiple": 16,
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "design": _ATTENTION_DESIGN.to_dict(),
    },
}

__all__ = [
    "TILELANG_ATTENTION_KERNEL_METADATA",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
]
