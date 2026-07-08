"""Selection policy metadata helpers for quantization execution."""

from __future__ import annotations

from typing import Any, Mapping

from xqt.quant.types import QuantizationComponentPlan

from .component import ordered_unique, prefix_module_names


def build_effective_selection_policy(component: QuantizationComponentPlan) -> dict[str, Any]:
    """Build the effective module selection policy for a component."""

    policy = dict(component.policy)
    include_module_names = list(policy.get("include_module_names") or [])
    exclude_module_names = list(policy.get("exclude_module_names") or [])
    include_module_names.extend(component.force_quantize)
    exclude_module_names.extend(component.skip_quantize)
    exclude_module_names.extend(component.keep_high_precision)
    if include_module_names:
        policy["include_module_names"] = ordered_unique(
            str(name) for name in include_module_names
        )
    if exclude_module_names:
        policy["exclude_module_names"] = ordered_unique(
            str(name) for name in exclude_module_names
        )
    if component.strategy is not None:
        policy.setdefault("strategy", component.strategy)
    return policy


def selection_policy_metadata(component: QuantizationComponentPlan) -> dict[str, Any]:
    """Return report metadata describing component selection controls."""

    raw = component.policy.get("selection_policy")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {
        "target_path": component.target_path,
        "selectors": {},
        "keep_high_precision": list(component.keep_high_precision),
        "skip_quantize": list(component.skip_quantize),
        "force_quantize": list(component.force_quantize),
    }


def module_selection_reason_metadata(
    component: QuantizationComponentPlan,
    *,
    quantized_modules: list[str],
    skipped_modules: list[str],
    high_precision_modules: list[str],
) -> dict[str, dict[str, str]]:
    """Explain why modules were quantized or skipped."""

    forced = set(prefix_module_names(component.force_quantize, component.target_path))
    explicit_skip = set(prefix_module_names(component.skip_quantize, component.target_path))
    high_precision = set(high_precision_modules)
    quantized: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for name in quantized_modules:
        quantized[name] = "force_quantize" if name in forced else "matched_selection_policy"
    for name in skipped_modules:
        if name in high_precision:
            skipped[name] = "keep_high_precision"
        elif name in explicit_skip:
            skipped[name] = "skip_quantize"
        else:
            skipped[name] = "filtered_by_selection_policy"

    return {
        "quantized": quantized,
        "skipped": skipped,
        "high_precision": {
            name: "keep_high_precision" for name in high_precision_modules
        },
        "fallback": {},
    }


__all__ = [
    "build_effective_selection_policy",
    "module_selection_reason_metadata",
    "selection_policy_metadata",
]
