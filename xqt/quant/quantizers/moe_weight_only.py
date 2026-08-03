"""MoE expert weight-only quantization (C8); router stays high precision (smoke)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.types import XQTContext
from xqt.quant.execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from xqt.quant.execution.reporting import optional_calibration_summary
from xqt.quant.execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from xqt.quant.policy import (
    DEFAULT_MOE_EXPERT_NAME_PATTERNS,
    DEFAULT_MOE_ROUTER_NAME_PATTERNS,
    classify_moe_module,
    is_moe_expert_module,
    is_moe_router_module,
)
from xqt.quant.quantizers.awq_gptq_weight_only import quantize_with_awq_weight_only
from xqt.quant.strategy import normalize_quant_strategy
from xqt.quant.types import QuantizationComponentPlan, QuantizationNature, QuantizationReport


@dataclass
class MoEExpertQuantizationResult(QuantizedModel):
    """Result of MoE expert weight-only quantization."""

    backend: str = "pytorch"
    method: str | None = "moe_weight_only"
    strategy: str = "w4a16_int4"
    expert_modules: list[str] = field(default_factory=list)
    router_modules: list[str] = field(default_factory=list)
    shared_expert_modules: list[str] = field(default_factory=list)


def list_moe_module_roles(
    model: nn.Module,
    *,
    expert_patterns: Sequence[str] | None = None,
    router_patterns: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Classify Linear modules into expert / router / shared_expert / other."""

    roles: dict[str, list[str]] = {
        "expert": [],
        "router": [],
        "shared_expert": [],
        "other": [],
    }
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        role = classify_moe_module(
            name,
            expert_patterns=expert_patterns,
            router_patterns=router_patterns,
        )
        roles.setdefault(role, []).append(name)
    return roles


def quantize_moe_experts_weight_only(
    model: nn.Module,
    *,
    policy: Mapping[str, Any] | None = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: str | None = None,
    method: str = "awq",
    inplace: bool = True,
) -> MoEExpertQuantizationResult:
    """Quantize expert Linears; keep router / gate high precision."""

    policy_mapping = dict(policy or {})
    expert_patterns = tuple(
        policy_mapping.get("expert_name_patterns") or DEFAULT_MOE_EXPERT_NAME_PATTERNS
    )
    router_patterns = tuple(
        policy_mapping.get("router_name_patterns") or DEFAULT_MOE_ROUTER_NAME_PATTERNS
    )
    roles = list_moe_module_roles(
        model,
        expert_patterns=expert_patterns,
        router_patterns=router_patterns,
    )
    expert_names = list(roles["expert"]) + list(roles["shared_expert"])
    if not expert_names:
        return MoEExpertQuantizationResult(
            model=model if inplace else model,
            expert_modules=[],
            router_modules=list(roles["router"]),
            shared_expert_modules=list(roles["shared_expert"]),
            metadata={
                "note": "no_expert_modules_matched",
                "roles": roles,
            },
        )

    include_policy = {
        **policy_mapping,
        "include_module_names": expert_names,
        "include_module_types": ["Linear"],
        "exclude_name_patterns": list(router_patterns),
        "exclude_module_names": list(roles["router"]),
        "dtype": policy_mapping.get("dtype", "int4"),
        "scheme": policy_mapping.get("scheme", "weight_only"),
        "bits": int(policy_mapping.get("bits", 4) or 4),
        "group_size": int(policy_mapping.get("group_size", 128) or 128),
    }
    selected_strategy = (
        normalize_quant_strategy(strategy, include_policy) or "w4a16_int4"
    )
    if method not in {"awq", "gptq"}:
        raise ValueError(f"MoE expert quant method must be awq or gptq; got {method!r}")

    if method == "awq":
        result = quantize_with_awq_weight_only(
            model,
            policy=include_policy,
            calibration_inputs=calibration_inputs,
            strategy=selected_strategy,
            inplace=inplace,
        )
    else:
        from xqt.quant.quantizers.awq_gptq_weight_only import quantize_with_gptq_weight_only

        result = quantize_with_gptq_weight_only(
            model,
            policy=include_policy,
            calibration_inputs=calibration_inputs,
            strategy=selected_strategy,
            inplace=inplace,
        )

    expert_hit = [
        name
        for name in result.quantized_modules
        if is_moe_expert_module(name, patterns=expert_patterns)
        or name in roles["shared_expert"]
    ]
    bits = int(include_policy.get("bits", 4) or 4)
    group_size = int(include_policy.get("group_size", 128) or 128)
    from xqt.quant.quantizers.moe_layout_report import build_expert_layout_reports

    expert_layout_reports = build_expert_layout_reports(
        result.model,
        expert_hit,
        shared_expert_names=roles["shared_expert"],
    )
    return MoEExpertQuantizationResult(
        model=result.model,
        backend=result.backend,
        method="moe_weight_only",
        strategy=result.strategy,
        quantized_modules=list(result.quantized_modules),
        expert_modules=expert_hit,
        router_modules=list(roles["router"]),
        shared_expert_modules=list(roles["shared_expert"]),
        metadata={
            **dict(result.metadata),
            "moe_roles": roles,
            "expert_quantized_count": len(expert_hit),
            "router_kept_high_precision": list(roles["router"]),
            "base_method": method,
            "smoke_only": True,
            "note": "expert weight-only path; does not claim MoE serving speedup",
            "no_ep_dispatcher": True,
            "bits": bits,
            "group_size": group_size,
            "expert_layout_reports": expert_layout_reports,
        },
    )


def execute_moe_weight_only_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute MoE expert weight-only quantization for one component."""

    target = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    method = str(component.method or component.policy.get("base_method") or "awq")
    if method in {"moe_weight_only", "moe"}:
        method = str(component.policy.get("base_method", "awq"))
    calibration = context.calibration_inputs
    if calibration is None:
        calibration = context.example_inputs
    result = quantize_moe_experts_weight_only(
        target,
        policy=effective_policy,
        calibration_inputs=calibration,
        strategy=component.strategy,
        method=method,
        inplace=True,
    )
    updated = replace_component_model(root_model, component.target_path, result.model)
    high_precision = ordered_unique(
        [
            *prefix_module_names(component.keep_high_precision, component.target_path),
            *prefix_module_names(result.router_modules, component.target_path),
        ]
    )
    skipped = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision,
        ]
    )
    quantized = prefix_module_names(result.quantized_modules, component.target_path)
    calibration_samples, calibration_summary = optional_calibration_summary(
        context, component
    )
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method or "moe_weight_only",
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized,
        skipped_modules=skipped,
        high_precision_modules=high_precision,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=True,
        method_semantics="moe_expert_weight_only_router_high_precision",
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reason_metadata(
                component,
                quantized_modules=quantized,
                skipped_modules=skipped,
                high_precision_modules=high_precision,
            ),
            "executed": True,
            "expert_modules": prefix_module_names(
                result.expert_modules, component.target_path
            ),
            "router_modules": prefix_module_names(
                result.router_modules, component.target_path
            ),
            "shared_expert_modules": prefix_module_names(
                result.shared_expert_modules, component.target_path
            ),
        },
    )
    return updated, report


__all__ = [
    "MoEExpertQuantizationResult",
    "execute_moe_weight_only_component",
    "list_moe_module_roles",
    "quantize_moe_experts_weight_only",
]
