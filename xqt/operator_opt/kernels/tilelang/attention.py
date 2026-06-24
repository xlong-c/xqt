"""TileLang operator optimization references and guarded entry points."""

from __future__ import annotations

from functools import lru_cache
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from .gemm_builder import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
    build_tilelang_fp4_unpack_dequant_kernel,
    build_tilelang_gemm_kernel,
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


def _require_fp16_tensors(*tensors: torch.Tensor) -> None:
    if not all(tensor.dtype == torch.float16 for tensor in tensors):
        raise XQTBackendError("TileLang FlashAttention path currently supports only float16 tensors")


def _validate_dequant_gemm_inputs(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    activation: str | None,
    block_m: int,
    block_n: int,
) -> None:
    if x.ndim != 2 or qweight.ndim != 2:
        raise XQTBackendError("TileLang dequant GEMM expects x and qweight to be 2D tensors")
    if scale.ndim not in {1, 2}:
        raise XQTBackendError("TileLang dequant GEMM expects scale to be 1D or 2D")
    if x.shape[1] != qweight.shape[1]:
        raise XQTBackendError("TileLang dequant GEMM requires x.shape[1] == qweight.shape[1]")
    if scale.ndim == 1 and scale.shape[0] != qweight.shape[0]:
        raise XQTBackendError("1D scale must match qweight out_features")
    if scale.ndim == 2 and scale.shape != qweight.shape:
        raise XQTBackendError("2D scale must match qweight shape")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != qweight.shape[0]):
        raise XQTBackendError("bias must be 1D and match qweight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[0] % block_m != 0 or qweight.shape[0] % block_n != 0:
        raise XQTBackendError(
            "minimal TileLang dequant GEMM currently requires batch and out_features to be multiples of block sizes"
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
) -> Any:
    _require_tilelang()
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

    _require_cuda_tensors(q, k, v)
    _require_fp16_tensors(q, k, v)
    _validate_attention_inputs(q, k, v, dropout_p=dropout_p)
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
    )
    return kernel(q, k, v)


def dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference dequantized GEMM with optional bias and activation epilogue."""

    weight_scale = scale.to(dtype=x.dtype, device=x.device)
    if weight_scale.ndim == 1:
        weight_scale = weight_scale.unsqueeze(-1)
    weight = qweight.to(dtype=x.dtype, device=x.device) * weight_scale
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


def _decode_packed_signed_int4(
    packed_weight: torch.Tensor,
    *,
    input_features: int,
) -> torch.Tensor:
    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(packed_weight.shape[0], -1)
    unpacked = unpacked[:, : int(input_features)]
    signed = torch.where(
        unpacked >= 8,
        unpacked.to(torch.int16) - 16,
        unpacked.to(torch.int16),
    )
    return signed.to(torch.float32)


def fp4_packed_dequant_gemm_epilogue_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None = None,
) -> torch.Tensor:
    """Reference packed FP4 dequantized GEMM with optional epilogue."""

    padded_input_features = int(packed_weight.shape[1]) * 2
    qweight = _decode_packed_signed_int4(
        packed_weight.to(device=x.device),
        input_features=padded_input_features,
    ).to(dtype=x.dtype, device=x.device)
    weight_scale = scale.to(dtype=x.dtype, device=x.device)
    grouped = qweight.reshape(qweight.shape[0], -1, int(group_size))
    weight = (grouped * weight_scale).reshape(qweight.shape[0], padded_input_features)[
        :, : int(input_features)
    ]
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
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only TileLang dequant GEMM epilogue entry point."""

    tensors = (x, qweight, scale) if bias is None else (x, qweight, scale, bias)
    _require_cuda_tensors(*tensors)
    _require_fp16_tensors(*tensors)
    _validate_dequant_gemm_inputs(
        x,
        qweight,
        scale,
        bias=bias,
        activation=activation,
        block_m=int(block_m),
        block_n=int(block_n),
    )
    _require_tilelang()
    dequantized = qweight * scale if scale.ndim == 2 else qweight * scale.unsqueeze(-1)
    kernel = build_tilelang_gemm_kernel(
        m=int(x.shape[0]),
        n=int(qweight.shape[0]),
        k=int(x.shape[1]),
        block_m=int(block_m),
        block_n=int(block_n),
        threads=int(threads),
        target_arch=target_arch,
    )
    output = kernel(x, dequantized)
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


def fp4_packed_dequant_gemm_epilogue_tilelang(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_features: int,
    group_size: int,
    activation: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    target_arch: str | None = None,
) -> torch.Tensor:
    """CUDA-only packed FP4 entry using fused TileLang unpack/dequant GEMM."""

    tensors = (x, packed_weight, scale) if bias is None else (x, packed_weight, scale, bias)
    _require_cuda_tensors(*tensors)
    if x.dtype != torch.float16 or scale.dtype != torch.float16:
        raise XQTBackendError("packed FP4 TileLang path currently requires float16 x and scale")
    if bias is not None and bias.dtype != torch.float16:
        raise XQTBackendError("packed FP4 TileLang path currently requires float16 bias")
    if packed_weight.dtype != torch.uint8:
        raise XQTBackendError("packed FP4 TileLang path expects uint8 packed_weight")
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise XQTBackendError("packed FP4 TileLang path expects 2D x and packed_weight")
    if x.shape[1] != int(input_features):
        raise XQTBackendError("packed FP4 TileLang path requires x.shape[1] == input_features")
    if scale.ndim != 3 or scale.shape[0] != packed_weight.shape[0] or scale.shape[2] != 1:
        raise XQTBackendError("packed FP4 TileLang path expects scale shaped [out_features, groups, 1]")
    if int(group_size) <= 0:
        raise XQTBackendError("group_size must be positive")
    padded_input_features = int(packed_weight.shape[1]) * 2
    if scale.shape[1] * int(group_size) != padded_input_features:
        raise XQTBackendError("scale groups must cover the packed padded input features")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != packed_weight.shape[0]):
        raise XQTBackendError("bias must be 1D and match packed_weight out_features")
    if activation not in {None, "gelu", "silu", "relu"}:
        raise XQTBackendError(f"unsupported activation: {activation}")
    if x.shape[0] % int(block_m) != 0 or packed_weight.shape[0] % int(block_n) != 0:
        raise XQTBackendError(
            "minimal packed FP4 TileLang path requires batch and out_features to be multiples of block sizes"
        )
    _require_tilelang()
    kernel = build_tilelang_fp4_fused_dequant_gemm_kernel(
        m=int(x.shape[0]),
        n=int(packed_weight.shape[0]),
        input_features=int(input_features),
        group_size=int(group_size),
        block_m=int(block_m),
        block_n=int(block_n),
        threads=int(threads),
        target_arch=target_arch,
        has_bias=bias is not None,
        activation=activation,
    )
    if bias is not None:
        return kernel(x, packed_weight, scale, bias)
    return kernel(x, packed_weight, scale)


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
    "fp4_packed_dequant_gemm_epilogue": {
        "kernel_name": "fp4_packed_dequant_gemm_epilogue",
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "threads": 128,
        "num_stages": 2,
        "baseline": "fused TileLang packed FP4 unpack/dequant GEMM + bias/activation epilogue",
        "usage": "ReferenceFP4Linear path that consumes packed uint8 weight and group-wise scale.",
        "unpack_stage": "tilelang_fused_gemm_kernel",
        "fusion_status": "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue",
        "epilogue_stage": "tilelang_fused_bias_activation",
    },
}


__all__ = [
    "TILELANG_KERNEL_METADATA",
    "TileLangAttentionDesign",
    "build_tilelang_attention_design",
    "dequant_gemm_epilogue_reference",
    "dequant_gemm_epilogue_tilelang",
    "fp4_packed_dequant_gemm_epilogue_reference",
    "fp4_packed_dequant_gemm_epilogue_tilelang",
    "fused_attention_forward_reference",
    "fused_attention_forward_tilelang",
]
