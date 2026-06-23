"""Quantization plan construction helpers."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from xqt.core.schema import QuantComponentPolicyConfig, QuantConfig

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


def _merge_component_plan(
    quant_config: QuantConfig,
    component_config: Optional[QuantComponentPolicyConfig],
) -> QuantizationComponentPlan:
    if component_config is None:
        policy = dict(quant_config.policy)
        strategy = quant_config.strategy or policy.get("strategy")
        return QuantizationComponentPlan(
            name="model",
            backend=quant_config.backend,
            method=quant_config.method,
            strategy=str(strategy) if strategy is not None else None,
            policy=policy,
            keep_high_precision=list(quant_config.keep_high_precision),
            skip_quantize=list(quant_config.skip_quantize),
            force_quantize=list(quant_config.force_quantize),
        )

    merged_policy = dict(quant_config.policy)
    merged_policy.update(component_config.policy)
    strategy = component_config.strategy or quant_config.strategy or merged_policy.get("strategy")
    return QuantizationComponentPlan(
        name=component_config.name,
        backend=component_config.backend or quant_config.backend,
        target_path=component_config.target,
        method=component_config.method or quant_config.method,
        strategy=str(strategy) if strategy is not None else None,
        policy=merged_policy,
        keep_high_precision=_ordered_unique(
            [*quant_config.keep_high_precision, *component_config.keep_high_precision]
        ),
        skip_quantize=_ordered_unique(
            [*quant_config.skip_quantize, *component_config.skip_quantize]
        ),
        force_quantize=_ordered_unique(
            [*quant_config.force_quantize, *component_config.force_quantize]
        ),
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
        },
    )


__all__ = ["build_quantization_plan"]
