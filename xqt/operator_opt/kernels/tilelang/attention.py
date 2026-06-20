"""TileLang operator optimization references and guarded entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError


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
    production_status: str = "design_extracted_reference_guarded"
    limitations: tuple[str, ...] = (
        "Only CUDA tensors are accepted by the guarded TileLang entry point.",
        "The current production entry uses SDPA reference fallback until TileLang JIT is wired and validated.",
        "The extracted learning design assumes fp16 and seq_kv >= seq_q for non-square causal cases.",
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


def _require_cuda_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise XQTBackendError("at least one tensor is required")
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("TileLang kernels require CUDA tensors")


def _require_tilelang() -> object:
    try:
        import tilelang
    except ImportError as exc:
        raise XQTBackendError(
            "tilelang is required for TileLang operator kernels. Install the optimization extras."
        ) from exc
    return tilelang


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

    del block_m, block_n, threads, num_stages
    _require_tilelang()
    _require_cuda_tensors(q, k, v)
    return fused_attention_forward_reference(
        q,
        k,
        v,
        causal=causal,
        dropout_p=dropout_p,
    )


def dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference dequantized GEMM with optional bias and activation epilogue."""

    weight = qweight.to(dtype=x.dtype, device=x.device) * scale.to(dtype=x.dtype, device=x.device)
    output = x.matmul(weight.t())
    if bias is not None:
        output = output + bias.to(dtype=output.dtype, device=output.device)
    if activation is None:
        return output
    if activation == "gelu":
        return F.gelu(output)
    if activation == "silu":
        return F.silu(output)
    if activation == "relu":
        return F.relu(output)
    raise ValueError(f"unsupported activation: {activation}")


def dequant_gemm_epilogue_tilelang(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
) -> torch.Tensor:
    """CUDA-only TileLang dequant GEMM epilogue entry point."""

    del block_m, block_n, threads, num_stages
    tensors = (x, qweight, scale) if bias is None else (x, qweight, scale, bias)
    _require_tilelang()
    _require_cuda_tensors(*tensors)
    return dequant_gemm_epilogue_reference(
        x,
        qweight,
        scale,
        bias,
        activation=activation,
    )


_ATTENTION_DESIGN = build_tilelang_attention_design()

TILELANG_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "attention": {
        "kernel_name": "fused_attention_forward",
        "block_m": _ATTENTION_DESIGN.default_block_m,
        "block_n": _ATTENTION_DESIGN.default_block_n,
        "threads": _ATTENTION_DESIGN.default_threads,
        "num_stages": _ATTENTION_DESIGN.default_num_stages,
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "design": _ATTENTION_DESIGN.to_dict(),
    },
    "dequant_gemm_epilogue": {
        "kernel_name": "dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "torch.matmul + epilogue",
        "usage": "Quantized Linear path with dequantize + matmul + bias/activation epilogue.",
    },
}


__all__ = [
    "TILELANG_KERNEL_METADATA",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
]
