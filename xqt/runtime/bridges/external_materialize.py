"""Materialize HF packed weights onto a base nn.Module (C4 process step)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from xqt.core.errors import XQTArtifactError
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.runtime.bridges.hf_int4_layout import (
    group_qweight_keys,
    normalize_scales,
    process_weights_after_loading,
    replace_submodule,
)


@dataclass(frozen=True, slots=True)
class ModuleWeightPlan:
    """create_weights plan for one Linear target."""

    module_name: str
    in_features: int
    out_features: int
    has_bias: bool
    key_map: dict[str, str]


def create_weight_plans(
    base_model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> list[ModuleWeightPlan]:
    """Plan Linear targets that have packed qweight+scales in ``state``."""

    groups = group_qweight_keys(state)
    plans: list[ModuleWeightPlan] = []
    for prefix, keys in groups.items():
        if "qweight" not in keys and "weight_packed" not in keys:
            continue
        if "scales" not in keys and "weight_scale" not in keys:
            continue
        module_name = prefix.rstrip(".")
        if not module_name:
            continue
        try:
            module = base_model.get_submodule(module_name)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear):
            continue
        key_map = dict(keys)
        if "qweight" not in key_map and "weight_packed" in key_map:
            key_map["qweight"] = key_map["weight_packed"]
        if "scales" not in key_map and "weight_scale" in key_map:
            key_map["scales"] = key_map["weight_scale"]
        plans.append(
            ModuleWeightPlan(
                module_name=module_name,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                has_bias=module.bias is not None,
                key_map=key_map,
            )
        )
    return plans


def try_native_xqt_copy(
    base_model: nn.Module,
    state: Mapping[str, torch.Tensor],
    notes: list[str],
) -> int:
    """Copy tensors into existing AWQGPTQWeightOnlyLinear modules when shapes match."""

    replaced = 0
    for prefix, keys in group_qweight_keys(state).items():
        module_name = prefix.rstrip(".")
        if not module_name or "qweight" not in keys or "scales" not in keys:
            continue
        try:
            module = base_model.get_submodule(module_name)
        except AttributeError:
            continue
        if not isinstance(module, AWQGPTQWeightOnlyLinear):
            continue
        try:
            qweight = state[keys["qweight"]]
            scales = state[keys["scales"]]
            if module.quantized_weight.shape != qweight.shape:
                notes.append(f"shape_mismatch:{module_name}")
                continue
            module.quantized_weight.data.copy_(qweight.to(module.quantized_weight.dtype))
            if module.weight_scale.shape == scales.shape:
                module.weight_scale.data.copy_(scales.to(dtype=module.weight_scale.dtype))
            else:
                module.weight_scale.data.copy_(
                    normalize_scales(
                        scales,
                        out_features=module.output_features,
                        in_features=module.input_features,
                        group_size=module.group_size,
                    ).to(dtype=module.weight_scale.dtype)
                )
            replaced += 1
        except (RuntimeError, KeyError, AttributeError, XQTArtifactError) as exc:
            notes.append(f"copy_failed:{module_name}:{type(exc).__name__}")
    return replaced


def materialize_plans(
    base_model: nn.Module,
    state: Mapping[str, torch.Tensor],
    plans: list[ModuleWeightPlan],
    *,
    fmt: str,
    bits: int,
    group_size: int,
    notes: list[str],
) -> int:
    """Run process_weights_after_loading for each plan and replace modules."""

    replaced = 0
    for plan in plans:
        keys = plan.key_map
        try:
            qweight = state[keys["qweight"]]
            scales = state[keys["scales"]]
        except KeyError as exc:
            notes.append(f"missing_tensor:{plan.module_name}:{exc}")
            continue
        qzeros = state.get(keys["qzeros"]) if "qzeros" in keys else None
        bias = state.get(keys["bias"]) if "bias" in keys else None
        g_idx = state.get(keys["g_idx"]) if "g_idx" in keys else None
        try:
            processed = process_weights_after_loading(
                method=fmt,
                bits=bits,
                group_size=group_size if group_size > 0 else plan.in_features,
                in_features=plan.in_features,
                out_features=plan.out_features,
                qweight=qweight,
                scales=scales,
                qzeros=qzeros,
                bias=bias,
                g_idx=g_idx,
                require_g_idx=True,
            )
            if isinstance(processed, AWQGPTQWeightOnlyLinear):
                new_module = processed
                g_idx_applied: bool | None = None
            else:
                new_module = processed.module
                g_idx_applied = processed.g_idx_applied
            replace_submodule(base_model, plan.module_name, new_module)
            replaced += 1
            if g_idx is not None and g_idx_applied is not True:
                notes.append(f"desc_act_g_idx_not_fully_applied:{plan.module_name}")
            elif g_idx is not None and g_idx_applied is True:
                notes.append(f"g_idx_applied:{plan.module_name}")
        except XQTArtifactError as exc:
            notes.append(f"process_failed:{plan.module_name}:{exc}")
        except (RuntimeError, TypeError, ValueError) as exc:
            notes.append(
                f"process_failed:{plan.module_name}:{type(exc).__name__}:{exc}"
            )
    return replaced


__all__ = [
    "ModuleWeightPlan",
    "create_weight_plans",
    "materialize_plans",
    "try_native_xqt_copy",
]
