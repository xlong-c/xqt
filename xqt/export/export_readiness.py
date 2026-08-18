"""Export readiness after quant: can_export, blockers, suggested lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn


def _is_packed_dequant(module: nn.Module) -> bool:
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


def _is_specialized_non_export(module: nn.Module) -> bool:
    name = type(module).__name__
    if name in {
        "Int8MmaLinear",
        "W4StorageInt8MmaLinear",
        "Fp8MmaLinear",
        "SvdCompositeLinear",
        "ConvRotMixedPrecisionLinear",
    }:
        return True
    return bool(getattr(module, "_xqt_requires_custom_export", False))


@dataclass(frozen=True, slots=True)
class ExportReadinessReport:
    can_export: bool
    blockers: tuple[str, ...] = ()
    suggested_lowering: str | None = None
    packed_modules: tuple[str, ...] = ()
    specialized_modules: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_export": self.can_export,
            "blockers": list(self.blockers),
            "suggested_lowering": self.suggested_lowering,
            "packed_modules": list(self.packed_modules),
            "specialized_modules": list(self.specialized_modules),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


def assess_export_readiness(model: nn.Module) -> ExportReadinessReport:
    packed: list[str] = []
    specialized: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if _is_packed_dequant(module):
            packed.append(name)
        if _is_specialized_non_export(module):
            specialized.append(name)

    blockers: list[str] = []
    notes: list[str] = []
    suggested: str | None = None

    if packed:
        suggested = "fp4_weight_only_to_dense_linear"
        notes.append(
            "packed modules expose dequantize_weight; enable pre_export_lowering "
            f"mode={suggested!r} before ONNX/generic export"
        )
        blockers.append(
            f"packed_weight_modules:{len(packed)} "
            "(require pre_export_lowering to dense Linear)"
        )
    if specialized:
        blockers.append(
            f"specialized_runtime_modules:{len(specialized)} "
            "(no generic ONNX path without dedicated lowering)"
        )
        notes.append(
            "specialized modules (INT8 MMA / ConvRot / composite) need "
            "explicit materialize or stay non-exportable"
        )

    can_export = not blockers
    if can_export:
        notes.append("no packed or specialized blockers; generic export may proceed")

    return ExportReadinessReport(
        can_export=can_export,
        blockers=tuple(blockers),
        suggested_lowering=suggested,
        packed_modules=tuple(packed),
        specialized_modules=tuple(specialized),
        notes=tuple(notes),
        metadata={"packed_count": len(packed), "specialized_count": len(specialized)},
    )


__all__ = [
    "ExportReadinessReport",
    "assess_export_readiness",
]
