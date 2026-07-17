"""torchao quantization adapter."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext

from ..capability import _resolve_nature
from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import optional_calibration_summary
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..strategy import normalize_quant_strategy
from ..policy import QuantizationPolicy, should_quantize_module
from ..types import (
    QuantizationComponentPlan,
    QuantizationNature,
    QuantizationReport,
)


@dataclass
class TorchAOQuantizationResult(QuantizedModel):
    """Result returned by the torchao quantization adapter."""

    backend: str = "torchao"
    strategy: str = ""


def _import_torchao_quantization() -> Any:
    try:
        import torchao.quantization as quantization
    except ImportError as exc:
        raise XQTBackendError(
            "torchao is required for backend='torchao'. Install xdl[optimization]."
        ) from exc
    return quantization


def _get_strategy_factory(strategy: str) -> Callable[[], Any]:
    quantization = _import_torchao_quantization()
    normalized = normalize_quant_strategy(strategy)
    aliases = {
        "w8a8_int8": "Int8DynamicActivationInt8WeightConfig",
        "w8a16_int8": "Int8WeightOnlyConfig",
        "w4a16_int4": "Int4WeightOnlyConfig",
        "w8a8_fp8_e4m3": "Float8DynamicActivationFloat8WeightConfig",
        "w8a16_fp8_e4m3": "Float8WeightOnlyConfig",
    }
    attr_name = aliases.get(normalized or strategy)
    if attr_name is None:
        raise XQTBackendError(f"Unsupported torchao quantization strategy: {strategy}")
    if not hasattr(quantization, attr_name):
        raise XQTBackendError(f"torchao.quantization.{attr_name} is not available")

    return getattr(quantization, attr_name)


def _policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key == "dtype":
            kwargs["dtype"] = str(value)
        elif key == "scheme":
            kwargs["scheme"] = str(value)
        elif key in {
            "include_module_types",
            "exclude_module_types",
            "include_name_patterns",
            "exclude_name_patterns",
            "include_module_names",
            "exclude_module_names",
        }:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


def quantize_with_torchao(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> TorchAOQuantizationResult:
    """Quantize a model with torchao according to an XQT policy."""

    quantization = _import_torchao_quantization()
    if not hasattr(quantization, "quantize_"):
        raise XQTBackendError("torchao.quantization.quantize_ is not available")

    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    selected_strategy = strategy or str(getattr(quant_policy, "dtype", "int8"))
    if selected_strategy in {"int8", "int4", "fp8"}:
        selected_strategy = {
            "int8": "w8a8_int8",
            "int4": "w4a16_int4",
            "fp8": "w8a8_fp8_e4m3",
        }[selected_strategy]
    selected_strategy = normalize_quant_strategy(selected_strategy, {
        "dtype": quant_policy.dtype,
        "scheme": quant_policy.scheme,
    }) or selected_strategy

    strategy_factory = _get_strategy_factory(selected_strategy)
    quantization_config = strategy_factory()
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []

    def filter_fn(module: nn.Module, name: str) -> bool:
        should_quantize = should_quantize_module(name, module, quant_policy)
        if should_quantize:
            quantized_modules.append(name)
        return should_quantize

    quantization.quantize_(target_model, quantization_config, filter_fn)
    return TorchAOQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "policy": {
                "dtype": quant_policy.dtype,
                "scheme": quant_policy.scheme,
                "include_module_types": list(quant_policy.include_module_types),
                "exclude_module_types": list(quant_policy.exclude_module_types),
                "include_name_patterns": list(quant_policy.include_name_patterns),
                "exclude_name_patterns": list(quant_policy.exclude_name_patterns),
                "include_module_names": list(quant_policy.include_module_names),
                "exclude_module_names": list(quant_policy.exclude_module_names),
                "min_parameters": quant_policy.min_parameters,
            }
        },
    )


def execute_torchao_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_torchao,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute a torchao quantization component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    high_precision_modules = prefix_module_names(
        component.keep_high_precision,
        component.target_path,
    )
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    quantized_modules = prefix_module_names(result.quantized_modules, component.target_path)
    module_selection_reasons = module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = optional_calibration_summary(
        context,
        component,
    )
    nature = _resolve_nature(component.strategy, component.policy)
    algorithm_executable = True
    method_semantics = "torchao_executable_quantization"
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=nature,
        algorithm_executable=algorithm_executable,
        method_semantics=method_semantics,
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "quantization_nature_scope": "configured_torchao_route_not_runtime_observation",
            "runtime_precision_note": (
                "XQT delegates kernel selection to torchao and PyTorch. The selected "
                "strategy describes the configured W/A route, not a per-forward native "
                "MMA or speedup guarantee."
            ),
            "analysis_only": component.analysis_only,
            "algorithm_executable": algorithm_executable,
            "method_semantics": method_semantics,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
        },
    )
    return updated_model, report


__all__ = [
    "TorchAOQuantizationResult",
    "execute_torchao_component",
    "quantize_with_torchao",
]
