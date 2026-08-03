"""HF GPTQ/AWQ process_weights_after_loading into XQT Linear modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTArtifactError
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.quant.quantizers.fp4_weight_only import _pack_int4
from xqt.runtime.bridges.hf_int4_pack import (
    apply_zero_points,
    awq_qweight_to_codes_matrix,
    gptq_qweight_to_signed_matrix,
    group_qweight_keys,
    normalize_scales,
    replace_submodule,
)


@dataclass(frozen=True, slots=True)
class ProcessWeightsResult:
    """Materialized Linear plus whether act-order ``g_idx`` was absorbed."""

    module: AWQGPTQWeightOnlyLinear
    g_idx_applied: bool | None


def _sequential_g_idx(in_features: int, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(in_features, dtype=dtype) // max(int(group_size), 1)


def _is_sequential_g_idx(
    g_idx: torch.Tensor,
    *,
    in_features: int,
    group_size: int,
) -> bool:
    if g_idx.numel() != in_features:
        return False
    expected = _sequential_g_idx(in_features, group_size, g_idx.dtype)
    return bool(torch.equal(g_idx.detach().cpu(), expected.cpu()))


def _absorb_g_idx_into_sequential_storage(
    codes: torch.Tensor,
    scale: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fold non-sequential g_idx scales into dequant, re-pack sequential groups.

    XQT Linear storage is sequential groupwise. Act-order checkpoints assign
    each input column a group via ``g_idx``; we expand scales per column,
    reconstruct float weights, then re-quantize into sequential groups so
    ``dequantize_weight()`` matches the g_idx reference path.
    """

    groups = int(scale.shape[1])
    g = g_idx.detach().to(device=codes.device, dtype=torch.long).reshape(-1)
    if g.numel() != in_features:
        raise XQTArtifactError(
            f"g_idx numel {g.numel()} != in_features {in_features}"
        )
    g = torch.clamp(g, 0, max(groups - 1, 0))
    scale_cols = scale[:, g, 0]
    weight = codes.to(torch.float32) * scale_cols.to(torch.float32)

    padded_in = ((in_features + group_size - 1) // group_size) * group_size
    if padded_in != in_features:
        weight = F.pad(weight, (0, padded_in - in_features), value=0.0)
    max_code = 7.0 if bits == 4 else 127.0
    min_code = -8.0 if bits == 4 else -128.0
    grouped = weight.reshape(out_features, -1, group_size)
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    new_scale = torch.where(
        max_abs > 0,
        max_abs / max_code,
        torch.ones_like(max_abs),
    )
    q = torch.clamp(torch.round(grouped / new_scale), min=min_code, max=max_code)
    q = q.to(torch.int8).reshape(out_features, padded_in)
    if bits == 4:
        packed = _pack_int4(q)
    else:
        packed = q.contiguous()
    return packed, new_scale, padded_in


def process_weights_after_loading(
    *,
    method: str,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    bias: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    require_g_idx: bool = False,
) -> AWQGPTQWeightOnlyLinear | ProcessWeightsResult:
    """Convert HF GPTQ/AWQ packed tensors into XQT ``AWQGPTQWeightOnlyLinear``.

    When ``g_idx`` is sequential or absent, scales stay groupwise sequential.
    When ``g_idx`` is non-sequential (desc_act / act-order), scales are absorbed
    into sequential storage so dequant matches the g_idx reference. Set
    ``require_g_idx=True`` to return ``ProcessWeightsResult`` with
    ``g_idx_applied`` (callers that must refuse silent success).
    """

    if method not in {"gptq", "awq", "compressed_tensors"}:
        raise XQTArtifactError(
            f"process_weights_after_loading unsupported method {method!r}"
        )

    layout_method = "awq" if method == "awq" else "gptq"
    if layout_method == "awq":
        codes = awq_qweight_to_codes_matrix(
            qweight,
            bits=bits,
            in_features=in_features,
            out_features=out_features,
        )
    else:
        codes = gptq_qweight_to_signed_matrix(
            qweight,
            bits=bits,
            in_features=in_features,
            out_features=out_features,
        )

    codes = apply_zero_points(
        codes,
        qzeros,
        bits=bits,
        group_size=group_size,
        method=layout_method,
    )

    scale = normalize_scales(
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
    )

    g_idx_applied: bool | None
    if g_idx is None:
        g_idx_applied = None
        padded_in = ((in_features + group_size - 1) // group_size) * group_size
        if padded_in != in_features:
            codes = F.pad(codes, (0, padded_in - in_features), value=0.0)
        if bits == 4:
            q_store = torch.clamp(torch.round(codes), -8, 7).to(torch.int8)
            packed = _pack_int4(q_store)
        else:
            packed = torch.clamp(torch.round(codes), -128, 127).to(torch.int8)
    elif _is_sequential_g_idx(g_idx, in_features=in_features, group_size=group_size):
        g_idx_applied = True
        padded_in = ((in_features + group_size - 1) // group_size) * group_size
        if padded_in != in_features:
            codes = F.pad(codes, (0, padded_in - in_features), value=0.0)
        if bits == 4:
            q_store = torch.clamp(torch.round(codes), -8, 7).to(torch.int8)
            packed = _pack_int4(q_store)
        else:
            packed = torch.clamp(torch.round(codes), -128, 127).to(torch.int8)
    else:
        packed, scale, padded_in = _absorb_g_idx_into_sequential_storage(
            codes,
            scale,
            g_idx,
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
        )
        g_idx_applied = True

    bias_t = None if bias is None else bias.detach().to(torch.float32).reshape(-1)
    if bias_t is not None and bias_t.numel() != out_features:
        raise XQTArtifactError(
            f"bias numel {bias_t.numel()} != out_features {out_features}"
        )

    module = AWQGPTQWeightOnlyLinear(
        packed,
        scale,
        bias=bias_t,
        input_features=in_features,
        output_features=out_features,
        group_size=group_size,
        padded_input_features=padded_in,
        bits=bits,
        method=layout_method if method != "compressed_tensors" else "gptq",
    )
    if require_g_idx:
        return ProcessWeightsResult(module=module, g_idx_applied=g_idx_applied)
    return module


# Re-export pack helpers so existing imports from hf_int4_layout keep working.
__all__ = [
    "ProcessWeightsResult",
    "apply_zero_points",
    "awq_qweight_to_codes_matrix",
    "gptq_qweight_to_signed_matrix",
    "group_qweight_keys",
    "normalize_scales",
    "process_weights_after_loading",
    "replace_submodule",
]
