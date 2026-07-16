"""Quantization plan construction helpers."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from omegaconf import OmegaConf

from xqt.core.schema import QuantComponentPolicyConfig, QuantConfig
from xqt.workflows.stage_specs import QuantStageSpec

from .capability import describe_quant_backend_capability
from .strategy import (
    CANONICAL_QUANT_STRATEGIES,
    require_supported_quant_compute,
    require_supported_quant_method,
    require_supported_quant_strategy,
)
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


def _component_policy_config(value: Any) -> QuantComponentPolicyConfig:
    if isinstance(value, QuantComponentPolicyConfig):
        return value
    merged = OmegaConf.merge(
        OmegaConf.structured(QuantComponentPolicyConfig),
        OmegaConf.create(dict(value)),
    )
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


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
        strategy = require_supported_quant_strategy(
            quant_config.strategy,
            location="quant.strategy",
            policy=policy,
        )
        method = require_supported_quant_method(
            quant_config.method,
            location="quant.method",
        )
        compute = require_supported_quant_compute(
            quant_config.compute,
            location="quant.compute",
        )
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
            method=method,
            strategy=strategy,
            compute=compute,
            policy=policy,
            composite_gemm=quant_config.composite_gemm,
            keep_high_precision=keep_high_precision,
            skip_quantize=skip_quantize,
            force_quantize=force_quantize,
        )

    merged_policy = dict(quant_config.policy)
    merged_policy.update(component_config.policy)
    strategy = require_supported_quant_strategy(
        component_config.strategy or quant_config.strategy,
        location=f"quant.component_policies[{component_config.name}].strategy",
        policy=merged_policy,
    )
    method = require_supported_quant_method(
        component_config.method or quant_config.method,
        location=f"quant.component_policies[{component_config.name}].method",
    )
    compute = require_supported_quant_compute(
        component_config.compute
        if component_config.compute is not None
        else quant_config.compute,
        location=f"quant.component_policies[{component_config.name}].compute",
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
        method=method,
        strategy=strategy,
        compute=compute,
        policy=merged_policy,
        composite_gemm=component_config.composite_gemm or quant_config.composite_gemm,
        keep_high_precision=keep_high_precision,
        skip_quantize=skip_quantize,
        force_quantize=force_quantize,
        analysis_only=component_config.analysis_only,
    )


def _validate_quant_config_for_plan(quant_config: QuantConfig) -> None:
    if not quant_config.backend:
        raise ValueError("quant.backend is required when quantization is enabled")
    has_selector = (
        quant_config.method is not None
        or quant_config.strategy is not None
        or quant_config.compute is not None
        or bool(quant_config.policy)
        or bool(quant_config.component_policies)
    )
    if not has_selector:
        raise ValueError(
            "quant must specify method, strategy, compute, policy, or "
            "component_policies when enabled=true"
        )
    if quant_config.component_policies:
        quant_config.component_policies = [
            _component_policy_config(component)
            for component in quant_config.component_policies
        ]
        for component_config in quant_config.component_policies:
            if not component_config.enabled:
                continue
            component_policy = dict(quant_config.policy)
            component_policy.update(component_config.policy)
            describe_quant_backend_capability(
                component_config.backend or quant_config.backend,
                method=component_config.method or quant_config.method,
                strategy=component_config.strategy or quant_config.strategy,
                compute=(
                    component_config.compute
                    if component_config.compute is not None
                    else quant_config.compute
                ),
                policy=component_policy,
            )
        return
    describe_quant_backend_capability(
        quant_config.backend,
        method=quant_config.method,
        strategy=quant_config.strategy,
        compute=quant_config.compute,
        policy=quant_config.policy,
    )


def build_quantization_plan(
    quant_config: QuantConfig | QuantStageSpec,
    *,
    artifact_prefix: str = "quant",
) -> QuantizationExecutionPlan:
    """Build a normalized quantization execution plan from config."""

    if isinstance(quant_config, QuantStageSpec):
        quant_config = QuantConfig(
            enabled=True,
            backend=quant_config.backend,
            method=quant_config.method,
            strategy=quant_config.strategy,
            compute=quant_config.compute,
            policy=dict(quant_config.policy),
            composite_gemm=quant_config.composite_gemm,
            keep_high_precision=list(quant_config.keep_high_precision),
            skip_quantize=list(quant_config.skip_quantize),
            force_quantize=list(quant_config.force_quantize),
            analysis_only_modules=list(quant_config.analysis_only_modules),
            component_policies=list(quant_config.component_policies),
        )

    if not quant_config.enabled:
        return QuantizationExecutionPlan(
            components=[],
            artifact_prefix=artifact_prefix,
        )
    _validate_quant_config_for_plan(quant_config)

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
