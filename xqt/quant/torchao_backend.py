"""torchao quantization adapter."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from torch import nn

from xqt.core.errors import XQTBackendError

from .strategy import normalize_quant_strategy
from .policy import QuantizationPolicy, should_quantize_module


@dataclass
class TorchAOQuantizationResult:
    """Result returned by the torchao quantization adapter."""

    model: nn.Module
    backend: str = "torchao"
    strategy: str = ""
    quantized_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
        "dynamic_int8": "Int8DynamicActivationInt8WeightConfig",
        "int8_dynamic_activation_int8_weight": "Int8DynamicActivationInt8WeightConfig",
        "weight_only_int8": "Int8WeightOnlyConfig",
        "int8_weight_only": "Int8WeightOnlyConfig",
        "weight_only_int4": "Int4WeightOnlyConfig",
        "int4_weight_only": "Int4WeightOnlyConfig",
        "fp8_dynamic": "Float8DynamicActivationFloat8WeightConfig",
        "float8_dynamic_activation_float8_weight": "Float8DynamicActivationFloat8WeightConfig",
        "fp8_weight_only": "Float8WeightOnlyConfig",
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
            "int8": "dynamic_int8",
            "int4": "weight_only_int4",
            "fp8": "fp8_dynamic",
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


__all__ = [
    "TorchAOQuantizationResult",
    "quantize_with_torchao",
]
