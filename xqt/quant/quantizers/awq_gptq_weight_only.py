"""Algorithmic AWQ/GPTQ weight-only quantization backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.types import XQTContext

from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import build_component_quantization_report
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from xqt.contracts.packing_int4 import (
    _normalize_group_size,
    _pad_weight_for_groups,
    _safe_positive,
)
from xqt.contracts.weight_only import (
    AWQGPTQWeightOnlyLinear,
    _quantize_grouped_weight,
    _signed_quant_bounds,
)
from .fp4_weight_only import (
    _LinearCalibrationStats,
    _collect_linear_calibration_stats,
    _gptq_corrected_weight,
)
from .base import (
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)


@dataclass
class AWQGPTQWeightOnlyQuantizationResult(QuantizedModel):
    """Result returned by algorithmic AWQ/GPTQ weight-only quantization."""

    backend: str = "pytorch"
    strategy: str = "w4a16_int4"


def _bits_from_strategy_policy(
    strategy: str | None,
    policy: Mapping[str, Any],
) -> int:
    normalized = normalize_quant_strategy(strategy, policy)
    if "int8" in str(normalized):
        return 8
    if "int4" in str(normalized):
        return 4
    bits = int(policy.get("bits", 4) or 4)
    if bits not in {4, 8}:
        raise ValueError(f"AWQ/GPTQ weight-only quantization supports 4 or 8 bits, got {bits}")
    return bits


def _awq_group_multiplier_for_bits(
    weight: torch.Tensor,
    stats: _LinearCalibrationStats | None,
    *,
    bits: int,
    output_features: int,
    input_features: int,
    padded_input_features: int,
    group_size: int,
    alpha: float,
) -> torch.Tensor | None:
    if stats is None:
        return None
    min_code, max_code = _signed_quant_bounds(bits)
    padded_weight, _ = _pad_weight_for_groups(
        weight.detach().to(torch.float32),
        input_features=input_features,
        group_size=group_size,
    )
    grouped_weight = padded_weight.reshape(output_features, -1, group_size)
    importance = _safe_positive(stats.activation_abs_mean.to(torch.float32)).pow(float(alpha))
    if padded_input_features != int(importance.numel()):
        importance = F.pad(
            importance,
            (0, padded_input_features - int(importance.numel())),
            value=1.0,
        )
    grouped_importance = importance.reshape(1, -1, group_size).to(grouped_weight.device)
    group_max_abs = grouped_weight.abs().amax(dim=2, keepdim=True)
    base_scale = torch.where(
        group_max_abs > 0,
        group_max_abs / float(max_code),
        torch.ones_like(grouped_weight[..., :1]),
    )
    candidates = torch.tensor(
        (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0),
        device=grouped_weight.device,
        dtype=grouped_weight.dtype,
    )
    best_error = torch.full(
        grouped_weight.shape[:2],
        float("inf"),
        device=grouped_weight.device,
        dtype=grouped_weight.dtype,
    )
    best_multiplier = torch.ones_like(base_scale)
    for candidate in candidates:
        scale = base_scale * candidate
        quantized = torch.clamp(torch.round(grouped_weight / scale), min=min_code, max=max_code)
        dequantized = quantized * scale
        error = ((grouped_weight - dequantized).square() * grouped_importance).mean(dim=2)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_multiplier = torch.where(improved.unsqueeze(-1), candidate, best_multiplier)
    return best_multiplier


def _from_linear_awq(
    module: nn.Linear,
    *,
    bits: int,
    group_size: int,
    stats: _LinearCalibrationStats | None,
    alpha: float = 0.5,
) -> AWQGPTQWeightOnlyLinear:
    normalized_group_size = _normalize_group_size(group_size, module.in_features)
    _, padded_input_features = _pad_weight_for_groups(
        module.weight.detach().to(torch.float32),
        input_features=module.in_features,
        group_size=normalized_group_size,
    )
    group_multiplier = _awq_group_multiplier_for_bits(
        module.weight,
        stats,
        bits=bits,
        output_features=module.out_features,
        input_features=module.in_features,
        padded_input_features=padded_input_features,
        group_size=normalized_group_size,
        alpha=alpha,
    )
    quantized_weight, scale, padded_input_features = _quantize_grouped_weight(
        module.weight,
        bits=bits,
        group_size=normalized_group_size,
        input_features=module.in_features,
        output_features=module.out_features,
        group_multiplier=group_multiplier,
    )
    bias = None if module.bias is None else module.bias.detach().to(torch.float32)
    return AWQGPTQWeightOnlyLinear(
        quantized_weight,
        scale,
        bias=bias,
        input_features=module.in_features,
        output_features=module.out_features,
        group_size=normalized_group_size,
        padded_input_features=padded_input_features,
        bits=bits,
        method="awq",
    )


def _from_linear_gptq(
    module: nn.Linear,
    *,
    bits: int,
    group_size: int,
    stats: _LinearCalibrationStats | None,
    dampening: float = 0.01,
) -> AWQGPTQWeightOnlyLinear:
    normalized_group_size = _normalize_group_size(group_size, module.in_features)
    baseline = AWQGPTQWeightOnlyLinear.from_linear(
        module,
        bits=bits,
        group_size=normalized_group_size,
        method="gptq",
    )
    corrected_weight = _gptq_corrected_weight(
        module.weight.detach().to(torch.float32),
        baseline.dequantize_weight(),
        stats,
        dampening=dampening,
    )
    quantized_weight, scale, padded_input_features = _quantize_grouped_weight(
        corrected_weight,
        bits=bits,
        group_size=normalized_group_size,
        input_features=module.in_features,
        output_features=module.out_features,
    )
    bias = None if module.bias is None else module.bias.detach().to(torch.float32)
    return AWQGPTQWeightOnlyLinear(
        quantized_weight,
        scale,
        bias=bias,
        input_features=module.in_features,
        output_features=module.out_features,
        group_size=normalized_group_size,
        padded_input_features=padded_input_features,
        bits=bits,
        method="gptq",
    )


def _quantize_with_calibrated_weight_only(
    model: nn.Module,
    *,
    method: str,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> AWQGPTQWeightOnlyQuantizationResult:
    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    bits = _bits_from_strategy_policy(strategy, policy_mapping)
    configured_group_size = int(policy_mapping.get("group_size", 128) or 128)
    sample_limit = policy_mapping.get("sample_limit")
    normalized_sample_limit = None if sample_limit is None else int(sample_limit)
    target_model = model if inplace else copy.deepcopy(model)
    candidate_modules = [
        (name, module)
        for name, module in list(target_model.named_modules())
        if isinstance(module, nn.Linear) and should_quantize_module(name, module, quant_policy)
    ]
    stats = _collect_linear_calibration_stats(
        target_model,
        module_names=[name for name, _ in candidate_modules],
        calibration_inputs=calibration_inputs,
        sample_limit=normalized_sample_limit,
    )
    quantized_modules: list[str] = []
    for name, module in candidate_modules:
        normalized_group_size = _normalize_group_size(configured_group_size, module.in_features)
        if method == "awq":
            replacement = _from_linear_awq(
                module,
                bits=bits,
                group_size=normalized_group_size,
                stats=stats.get(name),
                alpha=float(policy_mapping.get("awq_alpha", 0.5)),
            )
        elif method == "gptq":
            replacement = _from_linear_gptq(
                module,
                bits=bits,
                group_size=normalized_group_size,
                stats=stats.get(name),
                dampening=float(policy_mapping.get("gptq_dampening", 0.01)),
            )
        else:
            raise ValueError(f"Unsupported weight-only calibration method: {method}")
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "bits": bits,
                "dtype": getattr(quant_policy, "dtype", f"int{bits}"),
                "scheme": getattr(quant_policy, "scheme", "weight_only"),
            },
        )
        or f"weight_only_int{bits}"
    )
    from xqt.contracts.runtime_quant import (
        build_runtime_quant_contract,
        first_linear_shapes,
    )
    from xqt.quant.types import QuantScheme

    global_shape, local_shape = first_linear_shapes(target_model)
    storage = "xqt_awq_gptq_int4_v1" if bits == 4 else "xqt_awq_gptq_int8_v1"
    contract = build_runtime_quant_contract(
        quant_spec=QuantScheme(
            weight_dtype=f"int{bits}",
            weight_granularity="groupwise",
            group_size=configured_group_size,
            activation_dtype=None,
            activation_mode="none",
            sym=True,
        ),
        storage_layout=storage,
        required_kernels=("dequant_fp16",),
        global_shape=global_shape,
        local_shape=local_shape,
        prefill_supported=True,
        decode_supported=True,
    )
    from xqt.quant.layout_apply_report import (
        attach_layout_kernel_metadata,
        layout_report_for_awq_gptq_model,
    )

    layout = layout_report_for_awq_gptq_model(
        target_model,
        bits=bits,
        group_size=configured_group_size,
        selected_kernel="dequant_fp16_reference",
    )
    meta = {
        "implementation": f"{method}_weight_only_int{bits}_linear",
        "weight_encoding": "packed_signed_int4" if bits == 4 else "signed_int8",
        "bits": bits,
        "group_size": configured_group_size,
        "algorithm": method,
        "calibration_algorithm": (
            "activation_aware_scale_selection"
            if method == "awq"
            else "hessian_diag_residual_compensation"
        ),
        "calibrated_module_count": len(stats),
        "calibration_sample_count": sum(item.sample_count for item in stats.values()),
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
            "bits": bits,
            "group_size": configured_group_size,
            "sample_limit": normalized_sample_limit,
        },
    }
    result = AWQGPTQWeightOnlyQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata=attach_layout_kernel_metadata(meta, layout),
    )
    return result.with_runtime_quant_contract(contract)


def quantize_with_awq_weight_only(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> AWQGPTQWeightOnlyQuantizationResult:
    """Run algorithmic AWQ weight-only quantization for Linear modules."""

    return _quantize_with_calibrated_weight_only(
        model,
        method="awq",
        policy=policy,
        calibration_inputs=calibration_inputs,
        strategy=strategy,
        inplace=inplace,
    )


def quantize_with_gptq_weight_only(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> AWQGPTQWeightOnlyQuantizationResult:
    """Run algorithmic GPTQ-style weight-only quantization for Linear modules."""

    return _quantize_with_calibrated_weight_only(
        model,
        method="gptq",
        policy=policy,
        calibration_inputs=calibration_inputs,
        strategy=strategy,
        inplace=inplace,
    )


def execute_awq_gptq_weight_only_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute algorithmic AWQ/GPTQ weight-only quantization for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    strategy = component.strategy or effective_policy.get("strategy")
    if component.method == "awq":
        result = quantize_with_awq_weight_only(
            target_model,
            policy=effective_policy,
            calibration_inputs=context.calibration_inputs,
            strategy=strategy,
            inplace=True,
        )
    elif component.method == "gptq":
        result = quantize_with_gptq_weight_only(
            target_model,
            policy=effective_policy,
            calibration_inputs=context.calibration_inputs,
            strategy=strategy,
            inplace=True,
        )
    else:
        raise ValueError(f"Unsupported algorithmic weight-only method: {component.method}")

    updated_model = replace_component_model(root_model, component.target_path, result.model)
    calibrated_module_count = int(result.metadata.get("calibrated_module_count", 0) or 0)
    algorithm_executable = calibrated_module_count > 0
    bits = int(result.metadata["bits"])
    method_semantics = (
        f"awq_activation_aware_weight_only_int{bits}_quantization"
        if component.method == "awq" and algorithm_executable
        else f"awq_label_only_groupwise_weight_only_int{bits}_storage_quantization"
        if component.method == "awq"
        else f"gptq_hessian_aware_weight_only_int{bits}_quantization"
        if component.method == "gptq" and algorithm_executable
        else f"gptq_label_only_groupwise_weight_only_int{bits}_storage_quantization"
    )
    report = build_component_quantization_report(
        context,
        component,
        backend=component.backend,
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=algorithm_executable,
        method_semantics=method_semantics,
        effective_policy=effective_policy,
        result_metadata=result.metadata,
        execution_state=f"weight_only_int{bits}",
    )
    return updated_model, report


__all__ = [
    "AWQGPTQWeightOnlyLinear",
    "AWQGPTQWeightOnlyQuantizationResult",
    "execute_awq_gptq_weight_only_component",
    "quantize_with_awq_weight_only",
    "quantize_with_gptq_weight_only",
]
