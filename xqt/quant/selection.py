"""Selection policy metadata helpers for quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from xqt.contracts.quantized import QuantizedModel
from xqt.quant.types import QuantizationComponentPlan, QuantizationReport

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


@dataclass(frozen=True, slots=True)
class QuantHitFieldsReport:
    """Whether a quant result explains which modules were hit or skipped."""

    ok: bool
    quantized_modules: tuple[str, ...]
    has_selection_policy: bool
    has_module_selection_reasons: bool
    gaps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "quantized_modules": list(self.quantized_modules),
            "has_selection_policy": self.has_selection_policy,
            "has_module_selection_reasons": self.has_module_selection_reasons,
            "gaps": list(self.gaps),
        }


def summarize_quant_hit_fields(
    source: QuantizedModel | QuantizationReport | Mapping[str, Any],
    *,
    require_hits: bool = True,
) -> QuantHitFieldsReport:
    """Check quant outputs expose hit/skip explanation fields (not empty unknowns)."""

    if isinstance(source, QuantizedModel):
        modules = tuple(str(n) for n in source.quantized_modules)
        metadata: Mapping[str, Any] = dict(source.metadata)
    elif isinstance(source, QuantizationReport):
        modules = tuple(str(n) for n in source.quantized_modules)
        metadata = dict(source.metadata)
    elif isinstance(source, Mapping):
        raw_modules = source.get("quantized_modules", ())
        if isinstance(raw_modules, (list, tuple)):
            modules = tuple(str(n) for n in raw_modules)
        else:
            modules = ()
        raw_meta = source.get("metadata", source)
        metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
    else:
        raise TypeError(
            "summarize_quant_hit_fields expects QuantizedModel, "
            f"QuantizationReport, or mapping; got {type(source).__name__}"
        )

    has_policy = "selection_policy" in metadata and metadata["selection_policy"] is not None
    has_reasons = (
        "module_selection_reasons" in metadata
        and metadata["module_selection_reasons"] is not None
    )
    gaps: list[str] = []
    if require_hits and not modules:
        gaps.append("quantized_modules_empty")
    if not has_policy and not has_reasons and not modules:
        gaps.append("no_selection_explanation")
    ok = not gaps
    return QuantHitFieldsReport(
        ok=ok,
        quantized_modules=modules,
        has_selection_policy=bool(has_policy),
        has_module_selection_reasons=bool(has_reasons),
        gaps=tuple(gaps),
    )


__all__ = [
    "QuantHitFieldsReport",
    "build_effective_selection_policy",
    "module_selection_reason_metadata",
    "selection_policy_metadata",
    "summarize_quant_hit_fields",
]
