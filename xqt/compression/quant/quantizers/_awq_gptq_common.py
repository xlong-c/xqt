"""Shared AWQ/GPTQ calibration helpers (single fact source)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _awq_group_multiplier_for_bits(
    weight: torch.Tensor,
    stats: object | None,
    *,
    bits: int,
    output_features: int,
    input_features: int,
    padded_input_features: int,
    group_size: int,
    alpha: float,
) -> torch.Tensor | None:
    from xqt.contracts.packing_int4 import _pad_weight_for_groups, _safe_positive
    from xqt.contracts.weight_only import _signed_quant_bounds

    if stats is None:
        return None
    min_code, max_code = _signed_quant_bounds(bits)
    act_mean = getattr(stats, "activation_abs_mean", None)
    if act_mean is None:
        return None
    padded_weight, _ = _pad_weight_for_groups(
        weight.detach().to(torch.float32),
        input_features=input_features,
        group_size=group_size,
    )
    grouped_weight = padded_weight.reshape(output_features, -1, group_size)
    importance = _safe_positive(act_mean.to(torch.float32)).pow(float(alpha))
    if padded_input_features != int(importance.numel()):
        importance = F.pad(
            importance,
            (0, padded_input_features - int(importance.numel())),
            value=1.0,
        )
    grouped_importance = importance.reshape(1, -1, group_size).to(grouped_weight.device)
    group_max_abs = grouped_weight.abs().amax(dim=2, keepdim=True)
    base_scale = torch.where(
        group_max_abs > 0,
        group_max_abs / float(max_code),
        torch.ones_like(grouped_weight[..., :1]),
    )
    candidates = torch.tensor(
        (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0),
        device=grouped_weight.device,
        dtype=grouped_weight.dtype,
    )
    best_error = torch.full(
        grouped_weight.shape[:2],
        float("inf"),
        device=grouped_weight.device,
        dtype=grouped_weight.dtype,
    )
    best_multiplier = torch.ones_like(base_scale)
    for candidate in candidates:
        scale = base_scale * candidate
        quantized = torch.clamp(
            torch.round(grouped_weight / scale), min=float(min_code), max=float(max_code)
        )
        dequantized = quantized * scale
        error = ((grouped_weight - dequantized).square() * grouped_importance).mean(dim=2)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_multiplier = torch.where(improved.unsqueeze(-1), candidate, best_multiplier)
    return best_multiplier


def _awq_group_multiplier(
    weight: torch.Tensor,
    stats: object | None,
    *,
    output_features: int,
    input_features: int,
    padded_input_features: int,
    group_size: int,
    alpha: float,
) -> torch.Tensor | None:
    return _awq_group_multiplier_for_bits(
        weight,
        stats,
        bits=4,
        output_features=output_features,
        input_features=input_features,
        padded_input_features=padded_input_features,
        group_size=group_size,
        alpha=alpha,
    )


def _gptq_corrected_weight(
    weight: torch.Tensor,
    dequantized: torch.Tensor,
    stats: object | None,
    *,
    dampening: float,
) -> torch.Tensor:
    from xqt.contracts.packing_int4 import _safe_positive

    if stats is None:
        return weight
    h_diag = getattr(stats, "activation_hessian_diag", None)
    if h_diag is None:
        return weight
    hessian = _safe_positive(h_diag.to(device=weight.device, dtype=weight.dtype))
    mean_hessian = _safe_positive(hessian.mean())
    damping = mean_hessian * float(dampening)
    correction_gain = hessian / (hessian + damping)
    while correction_gain.ndim < weight.ndim:
        correction_gain = correction_gain.unsqueeze(0)
    return weight - (dequantized - weight) * correction_gain


__all__ = [
    "_awq_group_multiplier",
    "_awq_group_multiplier_for_bits",
    "_gptq_corrected_weight",
]
