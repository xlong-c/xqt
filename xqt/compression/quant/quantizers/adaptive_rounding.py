"""Adaptive Rounding Quantization (AdaRound / AutoRound style).

Minimizes local output reconstruction error for Linear layers by optimizing
integer rounding directions instead of naive round-to-nearest (RTN):
    min_V || W @ X - (floor(W/s) + h(V)) * s @ X ||_F^2
Significantly reduces quantization loss for low-bit (4-bit, 3-bit) weights
without global backpropagation or training pipelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.contracts.packing_int4 import (
    _normalize_group_size,
    _pack_int4,
    _pad_weight_for_groups,
)
from xqt.contracts.weight_only import (
    AWQGPTQWeightOnlyLinear,
    _signed_quant_bounds,
)
from xqt.compression.quant.policy import (
    QuantizationPolicy,
    should_quantize_module,
)
from xqt.compression.quant.quantizers.base import (
    call_model,
    iter_calibration_batches,
    move_batch_to_device,
    policy_from_mapping,
    replace_submodule,
)


def _rectified_sigmoid(v: torch.Tensor, zeta: float = 1.1, gamma: float = -0.1) -> torch.Tensor:
    """Rectified sigmoid stretching [0, 1] range to avoid vanishing gradients."""
    return torch.clamp(torch.sigmoid(v) * (zeta - gamma) + gamma, min=0.0, max=1.0)


def optimize_linear_rounding(
    linear: nn.Linear,
    input_activations: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    steps: int = 40,
    lr: float = 1e-2,
    reg_weight: float = 0.01,
) -> AWQGPTQWeightOnlyLinear:
    """Optimize rounding offsets for a single Linear module using input activations."""
    min_code, max_code = _signed_quant_bounds(bits)
    orig_device = linear.weight.device
    orig_dtype = linear.weight.dtype

    w_float = linear.weight.detach().to(device=orig_device, dtype=torch.float32)
    in_feat = linear.in_features
    out_feat = linear.out_features
    norm_group_size = _normalize_group_size(group_size, in_feat)

    # Pad and reshape to groups
    padded_w, padded_in_feat = _pad_weight_for_groups(
        w_float,
        input_features=in_feat,
        group_size=norm_group_size,
    )
    grouped_w = padded_w.reshape(out_feat, -1, norm_group_size)

    # Base scales
    max_abs = grouped_w.abs().amax(dim=2, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / float(max_code), torch.ones_like(max_abs))

    # Continuous ratio and floor integer baseline
    ratio = grouped_w / scale
    floor_w = torch.clamp(torch.floor(ratio), min=min_code, max=max_code)
    remainder = ratio - floor_w  # range [0, 1)

    # Initialize optimization variable V using inverse sigmoid
    init_v = -torch.log(
        torch.clamp((1.0 / torch.clamp(remainder, min=1e-4, max=1.0 - 1e-4)) - 1.0, min=1e-4)
    )
    v = nn.Parameter(init_v.clone())
    optimizer = torch.optim.Adam([v], lr=lr)

    # Format input activations [N, in_features]
    x = input_activations.detach().to(device=orig_device, dtype=torch.float32)
    if x.ndim > 2:
        x = x.reshape(-1, in_feat)
    # Target float output [N, out_features]
    target_y = F.linear(x, w_float, bias=None)

    # Optimization loop
    for step in range(steps):
        optimizer.zero_grad()
        soft_offset = _rectified_sigmoid(v)
        # STE: hard round forward, soft gradient backward
        hard_offset = (soft_offset >= 0.5).to(torch.float32)
        ste_offset = hard_offset + (soft_offset - soft_offset.detach())

        cand_q = torch.clamp(floor_w + ste_offset, min=min_code, max=max_code)
        cand_w = (cand_q * scale).reshape(out_feat, padded_in_feat)[:, :in_feat]

        pred_y = F.linear(x, cand_w, bias=None)
        recon_loss = F.mse_loss(pred_y, target_y)

        # Regularization driving soft values towards 0 or 1
        reg_loss = reg_weight * (1.0 - ((2.0 * soft_offset - 1.0).abs())).mean()
        loss = recon_loss + reg_loss

        loss.backward()
        optimizer.step()

    # Final hard quantized weight
    with torch.no_grad():
        final_offset = (_rectified_sigmoid(v) >= 0.5).to(torch.float32)
        final_q = torch.clamp(floor_w + final_offset, min=min_code, max=max_code)
        final_q_int8 = final_q.to(torch.int8).reshape(out_feat, padded_in_feat)

        if bits == 4:
            packed_weight = _pack_int4(final_q_int8)
        else:
            packed_weight = final_q_int8.contiguous()

    bias = None if linear.bias is None else linear.bias.detach().to(torch.float32)

    return AWQGPTQWeightOnlyLinear(
        packed_weight,
        scale,
        bias=bias,
        input_features=in_feat,
        output_features=out_feat,
        group_size=norm_group_size,
        padded_input_features=padded_in_feat,
        bits=bits,
        method="adaround",
    )


def quantize_with_adaptive_rounding(
    model: nn.Module,
    calibration_inputs: Iterable[Any],
    *,
    bits: int = 4,
    group_size: int = 128,
    steps: int = 30,
    policy: QuantizationPolicy | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Quantize all eligible linear layers using adaptive rounding optimization."""
    quant_policy = policy or QuantizationPolicy()
    device = next(model.parameters(), torch.empty((), device="cpu")).device

    # Find candidate linears
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_quantize_module(name, module, quant_policy):
            candidates.append((name, module))

    if not candidates:
        return model, {"quantized_modules": [], "count": 0}

    # Collect input activations for candidate layers
    inputs_dict: dict[str, list[torch.Tensor]] = {name: [] for name, _ in candidates}
    handles = []

    def _make_hook(mod_name: str) -> Any:
        def _hook(m: nn.Module, inps: tuple[Any, ...], out: Any) -> None:
            del m, out
            if inps and isinstance(inps[0], torch.Tensor):
                # Collect up to 256 tokens per batch to keep GPU memory light
                act = inps[0].detach().cpu()
                flattened = act.reshape(-1, act.shape[-1])[:256]
                inputs_dict[mod_name].append(flattened)

        return _hook

    for name, mod in candidates:
        handles.append(mod.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in iter_calibration_batches(calibration_inputs, sample_limit=4):
                batch_dev = move_batch_to_device(batch, device)
                call_model(model, batch_dev)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # Execute adaptive rounding on each candidate
    quantized_names: list[str] = []
    for name, mod in candidates:
        acts = inputs_dict.get(name, [])
        if acts:
            cat_acts = torch.cat(acts, dim=0).to(device=mod.weight.device)
        else:
            cat_acts = torch.randn(16, mod.in_features, device=mod.weight.device)

        replacement = optimize_linear_rounding(
            mod,
            cat_acts,
            bits=bits,
            group_size=group_size,
            steps=steps,
        )
        replace_submodule(model, name, replacement)
        quantized_names.append(name)

    return model, {
        "quantized_modules": quantized_names,
        "count": len(quantized_names),
        "bits": bits,
        "group_size": group_size,
        "algorithm": "adaptive_rounding",
    }


__all__ = [
    "optimize_linear_rounding",
    "quantize_with_adaptive_rounding",
]
