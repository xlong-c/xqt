"""Explicit PyTorch-side lowerings for deployment-oriented exports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from xqt.core.errors import XQTBackendError


@dataclass
class PreExportLoweringResult:
    """Result of an explicit deployment lowering applied before export."""

    model: nn.Module
    applied: bool
    mode: str
    inplace: bool
    lowered_modules: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_packed_weight_dequant_module(module: nn.Module) -> bool:
    """Recognize packed-weight storage via duck type (G6), not quantizer class alone."""
    if not callable(getattr(module, "dequantize_weight", None)):
        return False
    if not hasattr(module, "input_features") or not hasattr(module, "output_features"):
        return False
    return (
        hasattr(module, "packed_weight")
        or hasattr(module, "quantized_weight")
        or hasattr(module, "qweight")
        or hasattr(module, "qweight_t")
    )


def _fp4_weight_only_linear_to_dense(module: nn.Module) -> nn.Linear:
    """Materialize packed FP4-style storage as an equivalent dense Linear module."""

    if not _is_packed_weight_dequant_module(module):
        # Compatibility: still accept the historical quantizer class by name.
        type_name = type(module).__name__
        if type_name != "FP4WeightOnlyLinear" or not callable(
            getattr(module, "dequantize_weight", None)
        ):
            raise TypeError(
                "expected packed-weight module with dequantize_weight(), "
                f"got {type(module).__name__}"
            )
    weight = module.dequantize_weight().detach()
    bias = module.bias.detach() if module.bias is not None else None
    lowered = nn.Linear(
        module.input_features,
        module.output_features,
        bias=bias is not None,
    ).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lowered.weight.copy_(weight)
        if bias is not None and lowered.bias is not None:
            lowered.bias.copy_(bias.to(device=weight.device, dtype=weight.dtype))
    return lowered


def _replace_fp4_weight_only_linears(model: nn.Module) -> list[dict[str, Any]]:
    """Replace every FP4 storage Linear with its dense dequantized equivalent."""

    replacements: list[dict[str, Any]] = []
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and _is_packed_weight_dequant_module(module)
    ]
    for name, module in candidates:
        parent_path, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        lowered = _fp4_weight_only_linear_to_dense(module)
        setattr(parent, child_name, lowered)
        replacements.append(
            {
                "path": name,
                "source_module_type": type(module).__name__,
                "target_module_type": type(lowered).__name__,
                "source_weight_storage": "packed_fp4",
                "target_weight_storage": "dense_dequantized",
                "source_weight_dtype": str(
                    getattr(module, "packed_weight", getattr(module, "quantized_weight")).dtype
                ),
                "target_weight_dtype": str(lowered.weight.dtype),
            }
        )
    return replacements


def apply_pre_export_lowering(
    model: nn.Module,
    config: Mapping[str, Any] | None,
) -> PreExportLoweringResult:
    """Apply an explicitly configured module lowering before model export."""

    if not config or not bool(config.get("enabled", False)):
        return PreExportLoweringResult(
            model=model,
            applied=False,
            mode="disabled",
            inplace=True,
            metadata={"enabled": False},
        )

    mode = str(config.get("mode", "fp4_weight_only_to_dense_linear"))
    inplace = bool(config.get("inplace", False))
    if mode != "fp4_weight_only_to_dense_linear":
        raise XQTBackendError(f"Unsupported pre_export_lowering mode: {mode}")

    target_model = model if inplace else deepcopy(model)
    target_model.eval()
    if _is_packed_weight_dequant_module(target_model):
        lowered_root = _fp4_weight_only_linear_to_dense(target_model)
        replacements = [
            {
                "path": "",
                "source_module_type": type(target_model).__name__,
                "target_module_type": type(lowered_root).__name__,
                "source_weight_storage": "packed_fp4",
                "target_weight_storage": "dense_dequantized",
                "source_weight_dtype": str(
                    getattr(
                        target_model,
                        "packed_weight",
                        getattr(target_model, "quantized_weight"),
                    ).dtype
                ),
                "target_weight_dtype": str(lowered_root.weight.dtype),
            }
        ]
        target_model = lowered_root
    else:
        replacements = _replace_fp4_weight_only_linears(target_model)
    if not replacements:
        raise XQTBackendError(
            "pre_export_lowering mode=fp4_weight_only_to_dense_linear requires "
            "at least one packed-weight module with dequantize_weight()"
        )

    return PreExportLoweringResult(
        model=target_model,
        applied=True,
        mode=mode,
        inplace=inplace,
        lowered_modules=replacements,
        metadata={
            "enabled": True,
            "mode": mode,
            "inplace": inplace,
            "lowered_modules": replacements,
            "deployment_note": (
                "FP4 packed storage was materialized into dense dequantized weights "
                "for export; the resulting artifact is not a packed-FP4 runtime."
            ),
        },
    )


__all__ = ["PreExportLoweringResult", "apply_pre_export_lowering"]
