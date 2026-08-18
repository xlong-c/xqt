"""Three-axis public quant capability facts (DEBT-002 full API).

Axis 1 = quant method (algorithm: ``awq`` / ``gptq`` / ``svd`` / ...).
Axis 2 = storage / precision contract (weight dtype, granularity, group size,
activation dtype/mode), config-facing name is the canonical strategy string.
Axis 3 = compute / MMA contract (``dequant_fp16`` / ``w8a8_int8_mma`` /
``fp8_mma`` / ``qdq_static`` / ...), plus the operator engine that lowers it.

These helpers are the public three-axis view. The backend capability table
(``xqt.quant.capability``) consumes the same facts instead of hand-writing a
parallel matrix.
"""

from __future__ import annotations

from typing import Any

from xqt.core.schema import (
    CANONICAL_QUANT_COMPUTES,
    CANONICAL_QUANT_METHODS,
)
from xqt.contracts.quant_strategy import quantization_nature_for_compute

from .strategy import canonical_quant_strategies, strategy_scheme_templates
from .types import QuantizationNature

_METHOD_NOTES: dict[str, tuple[str, ...]] = {
    "none": ("No quantization algorithm; raw model path.",),
    "awq": (
        "AWQ is a calibration method producing weight-only scales; "
        "it is not an operator engine.",
    ),
    "gptq": (
        "GPTQ is a calibration method producing weight-only packed weights; "
        "it is not an operator engine.",
    ),
    "svd": (
        "SVDQuant is a low-rank + quantized residual method; "
        "compute contract is composite_add, not a standalone engine.",
    ),
    "convrot": (
        "ConvRot is an activation-rotation-aware method; "
        "unabsorbed parts declare required_kernels.",
    ),
    "turboquant": (
        "TurboQuant is a calibration/packing method; "
        "kernel selection stays in the engine registry.",
    ),
    "moe": (
        "MoE expert weight-only quantization; router/gating stays high precision.",
    ),
    "moe_weight_only": (
        "MoE expert weight-only quantization (registry name); "
        "alias of the moe method family.",
    ),
}

_COMPUTE_NOTES: dict[str, tuple[str, ...]] = {
    "dequant_fp16": (
        "Reference dequantized FP16 compute; no native low-precision MMA.",
    ),
    "w8a8_int8_mma": (
        "Native INT8 MMA compute contract; hardware/engine availability decides "
        "whether a forward realizes it.",
    ),
    "fp8_mma": (
        "Native FP8 MMA compute contract; CUDA verification pending where "
        "hardware is absent.",
    ),
    "qdq_static": (
        "Static QDQ graph compute contract (calibrated scales).",
    ),
    "qdq_dynamic": (
        "Dynamic QDQ graph compute contract (per-forward scales).",
    ),
    "dequant_gemm": (
        "Dequantized GEMM compute contract (packed weight + reference math).",
    ),
}

def _registered_quant_methods() -> list[str]:
    """Registered route method names (axis 1 union with config canonical list)."""

    from xqt.quant import quantizers as _quantizers  # noqa: F401  # register routes

    from .registry import iter_quant_routes

    names: list[str] = []
    seen: set[str] = set()
    for registration in iter_quant_routes():
        for method in registration.methods or ("none",):
            if method in {"none", ""} or method in seen:
                continue
            seen.add(method)
            names.append(method)
    return sorted(names)


def quant_method_specs() -> list[dict[str, Any]]:
    """Return the axis-1 (quant method) public fact list."""

    registered = set(_registered_quant_methods())
    methods = tuple(dict.fromkeys((*CANONICAL_QUANT_METHODS, *sorted(registered))))
    return [
        {
            "name": method,
            "axis": "method",
            "registered_route": method in registered,
            "notes": list(_METHOD_NOTES.get(method, ())),
        }
        for method in methods
    ]


def quant_storage_specs() -> list[dict[str, Any]]:
    """Return the axis-2 (storage + activation scheme) public fact list."""

    specs: list[dict[str, Any]] = []
    for strategy in canonical_quant_strategies():
        template = strategy_scheme_templates()[strategy]
        specs.append(
            {
                "strategy": strategy,
                "axis": "storage",
                "scheme": dict(template),
                "strategy_is_scheme_alias": True,
            }
        )
    return specs


def quant_compute_specs() -> list[dict[str, Any]]:
    """Return the axis-3 (compute / MMA contract) public fact list."""

    specs: list[dict[str, Any]] = []
    for compute in CANONICAL_QUANT_COMPUTES:
        nature = quantization_nature_for_compute(compute)
        specs.append(
            {
                "name": compute,
                "axis": "compute",
                "nature": nature.value,
                "notes": list(_COMPUTE_NOTES.get(compute, ())),
            }
        )
    return specs


def quant_axis_report() -> dict[str, Any]:
    """Aggregate three-axis public facts for capability / readiness reports."""

    return {
        "axes": ["method", "storage", "compute"],
        "strategy_is_scheme_alias": True,
        "method": quant_method_specs(),
        "storage": quant_storage_specs(),
        "compute": quant_compute_specs(),
    }


__all__ = [
    "quant_axis_report",
    "quant_compute_specs",
    "quant_method_specs",
    "quant_storage_specs",
]
