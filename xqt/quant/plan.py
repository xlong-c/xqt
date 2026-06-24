"""Quantization plan construction helpers."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from xqt.core.schema import QuantComponentPolicyConfig, QuantConfig

from .strategy import CANONICAL_QUANT_STRATEGIES, normalize_quant_strategy
from .types import QuantizationComponentPlan, QuantizationExecutionPlan


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _policy_selector_summary(
    *,
    target_path: Optional[str],
    policy: dict[str, Any],
    keep_high_precision: list[str],
    skip_quantize: list[str],
    force_quantize: list[str],
) -> dict[str, Any]:
    selector_keys = (
        "include_module_types",
        "exclude_module_types",
        "include_name_patterns",
        "exclude_name_patterns",
        "include_module_names",
        "exclude_module_names",
        "min_parameters",
    )
    return {
        "target_path": target_path,
        "selectors": {
            key: list(value) if isinstance(value, (list, tuple)) else value
            for key in selector_keys
            if (value := policy.get(key)) is not None
        },
        "keep_high_precision": list(keep_high_precision),
        "skip_quantize": list(skip_quantize),
        "force_quantize": list(force_quantize),
    }


def _merge_component_plan(
    quant_config: QuantConfig,
    component_config: Optional[QuantComponentPolicyConfig],
) -> QuantizationComponentPlan:
    if component_config is None:
        policy = dict(quant_config.policy)
        strategy = normalize_quant_strategy(quant_config.strategy, policy)
        keep_high_precision = list(quant_config.keep_high_precision)
        skip_quantize = list(quant_config.skip_quantize)
        force_quantize = list(quant_config.force_quantize)
        policy["selection_policy"] = _policy_selector_summary(
            target_path=None,
            policy=policy,
            keep_high_precision=keep_high_precision,
            skip_quantize=skip_quantize,
            force_quantize=force_quantize,
        )
        return QuantizationComponentPlan(
            name="model",
            backend=quant_config.backend,
            method=quant_config.method,
            strategy=str(strategy) if strategy is not None else None,
            policy=policy,
            keep_high_precision=keep_high_precision,
            skip_quantize=skip_quantize,
            force_quantize=force_quantize,
        )

    merged_policy = dict(quant_config.policy)
    merged_policy.update(component_config.policy)
    strategy = normalize_quant_strategy(
        component_config.strategy or quant_config.strategy,
        merged_policy,
    )
    keep_high_precision = _ordered_unique(
        [*quant_config.keep_high_precision, *component_config.keep_high_precision]
    )
    skip_quantize = _ordered_unique(
        [*quant_config.skip_quantize, *component_config.skip_quantize]
    )
    force_quantize = _ordered_unique(
        [*quant_config.force_quantize, *component_config.force_quantize]
    )
    merged_policy["selection_policy"] = _policy_selector_summary(
        target_path=component_config.target,
        policy=merged_policy,
        keep_high_precision=keep_high_precision,
        skip_quantize=skip_quantize,
        force_quantize=force_quantize,
    )
    return QuantizationComponentPlan(
        name=component_config.name,
        backend=component_config.backend or quant_config.backend,
        target_path=component_config.target,
        method=component_config.method or quant_config.method,
        strategy=str(strategy) if strategy is not None else None,
        policy=merged_policy,
        keep_high_precision=keep_high_precision,
        skip_quantize=skip_quantize,
        force_quantize=force_quantize,
        analysis_only=component_config.analysis_only,
    )


def build_quantization_plan(
    quant_config: QuantConfig,
    *,
    artifact_prefix: str = "quant",
) -> QuantizationExecutionPlan:
    """Build a normalized quantization execution plan from config."""

    if not quant_config.enabled:
        return QuantizationExecutionPlan(
            components=[],
            artifact_prefix=artifact_prefix,
        )

    components: list[QuantizationComponentPlan] = []
    if quant_config.component_policies:
        for component_config in quant_config.component_policies:
            if not component_config.enabled:
                continue
            components.append(_merge_component_plan(quant_config, component_config))
    else:
        components.append(_merge_component_plan(quant_config, None))

    return QuantizationExecutionPlan(
        components=components,
        artifact_prefix=artifact_prefix,
        metadata={
            "analysis_only_modules": list(quant_config.analysis_only_modules),
            "canonical_strategies": list(CANONICAL_QUANT_STRATEGIES),
        },
    )


__all__ = ["build_quantization_plan"]
