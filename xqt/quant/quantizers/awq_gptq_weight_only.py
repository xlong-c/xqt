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
from ..execution.reporting import optional_calibration_summary
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from .fp4_weight_only import (
    _LinearCalibrationStats,
    _can_mutate_runtime_cache,
    _collect_linear_calibration_stats,
    _gptq_corrected_weight,
    _normalize_group_size,
    _pack_int4,
    _pad_weight_for_groups,
    _policy_from_mapping,
    _replace_submodule,
    _safe_positive,
    _unpack_int4,
)


@dataclass
class AWQGPTQWeightOnlyQuantizationResult(QuantizedModel):
    """Result returned by algorithmic AWQ/GPTQ weight-only quantization."""

    backend: str = "pytorch"
    strategy: str = "weight_only_int4"


def _bits_from_strategy_policy(
    strategy: str | None,
    policy: Mapping[str, Any],
) -> int:
    normalized = normalize_quant_strategy(strategy, policy)
    if normalized == "weight_only_int8":
        return 8
    if normalized == "weight_only_int4":
        return 4
    bits = int(policy.get("bits", 4) or 4)
    if bits not in {4, 8}:
        raise ValueError(f"AWQ/GPTQ weight-only quantization supports 4 or 8 bits, got {bits}")
    return bits


def _signed_quant_bounds(bits: int) -> tuple[int, int]:
    if bits == 8:
        return -128, 127
    if bits == 4:
        return -8, 7
    raise ValueError(f"Unsupported signed weight-only bit width: {bits}")


def _quantize_grouped_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    input_features: int,
    output_features: int,
    group_multiplier: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    min_code, max_code = _signed_quant_bounds(bits)
    weight, padded_input_features = _pad_weight_for_groups(
        weight.detach().to(torch.float32),
        input_features=input_features,
        group_size=group_size,
    )
    grouped_weight = weight.reshape(output_features, -1, group_size)
    max_abs = grouped_weight.abs().amax(dim=2, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / float(max_code), torch.ones_like(max_abs))
    if group_multiplier is not None:
        multiplier = group_multiplier.to(device=scale.device, dtype=scale.dtype)
        if multiplier.ndim == 2:
            multiplier = multiplier.unsqueeze(-1)
        scale = scale * _safe_positive(multiplier)
    quantized = torch.clamp(torch.round(grouped_weight / scale), min=min_code, max=max_code)
    quantized = quantized.to(torch.int8).reshape(output_features, padded_input_features)
    if bits == 4:
        return _pack_int4(quantized), scale, padded_input_features
    return quantized.contiguous(), scale, padded_input_features


def _decode_quantized_weight(
    quantized_weight: torch.Tensor,
    *,
    bits: int,
    padded_input_features: int,
) -> torch.Tensor:
    if bits == 4:
        return _unpack_int4(quantized_weight, padded_input_features)
    return quantized_weight.to(torch.float32)


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


class AWQGPTQWeightOnlyLinear(nn.Module):
    """Weight-only Linear produced by algorithmic AWQ/GPTQ calibration."""

    def __init__(
        self,
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        bits: int,
        method: str,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.bits = int(bits)
        self.method = str(method)
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._dense_bias_cache: dict[tuple[str, str], torch.Tensor | None] = {}
        if self.bits != 4:
            self.tilelang_packed_dequant_gemm_args = None
        storage_dtype = torch.uint8 if self.bits == 4 else torch.int8
        self.register_buffer("quantized_weight", quantized_weight.to(storage_dtype))
        self.register_buffer("weight_scale", scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        bits: int,
        group_size: int,
        method: str,
    ) -> "AWQGPTQWeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        quantized_weight, scale, padded_input_features = _quantize_grouped_weight(
            module.weight,
            bits=bits,
            group_size=normalized_group_size,
            input_features=module.in_features,
            output_features=module.out_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            quantized_weight,
            scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded_input_features,
            bits=bits,
            method=method,
        )

    @classmethod
    def from_linear_awq(
        cls,
        module: nn.Linear,
        *,
        bits: int,
        group_size: int,
        stats: _LinearCalibrationStats | None,
        alpha: float = 0.5,
    ) -> "AWQGPTQWeightOnlyLinear":
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
        return cls(
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

    @classmethod
    def from_linear_gptq(
        cls,
        module: nn.Linear,
        *,
        bits: int,
        group_size: int,
        stats: _LinearCalibrationStats | None,
        dampening: float = 0.01,
    ) -> "AWQGPTQWeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        baseline = cls.from_linear(
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
        return cls(
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

    def dequantize_weight(self) -> torch.Tensor:
        quantized = _decode_quantized_weight(
            self.quantized_weight,
            bits=self.bits,
            padded_input_features=self.padded_input_features,
        )
        grouped = quantized.reshape(self.output_features, -1, self.group_size)
        dequantized = grouped * self.weight_scale
        return dequantized.reshape(self.output_features, self.padded_input_features)[
            :, : self.input_features
        ]

    def quantized_weight_codes(self) -> torch.Tensor:
        """Return unpacked signed integer weight codes as a dense matrix."""

        return _decode_quantized_weight(
            self.quantized_weight,
            bits=self.bits,
            padded_input_features=self.padded_input_features,
        )[:, : self.input_features]

    def expanded_weight_scale(self) -> torch.Tensor:
        """Return per-element scale expanded from the stored group-wise scale."""

        expanded = self.weight_scale.expand(-1, -1, self.group_size).reshape(
            self.output_features,
            self.padded_input_features,
        )
        return expanded[:, : self.input_features]

    def tilelang_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        """Expose dense qweight/scale inputs for TileLang dequant GEMM wrappers."""

        qweight = self.quantized_weight_codes().to(device=device, dtype=dtype)
        scale = self.expanded_weight_scale().to(device=device, dtype=dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return qweight, scale, bias, None

    def tilelang_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        """Expose packed INT4 inputs for TileLang packed dequant GEMM wrappers."""

        if self.bits != 4:
            raise ValueError("Packed TileLang dequant GEMM bridge is only available for 4-bit weights")
        packed_weight = self.quantized_weight.to(device=device)
        scale = self.weight_scale.to(device=device, dtype=dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return packed_weight, scale, bias, None, self.input_features, self.group_size

    def dense_weight(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not _can_mutate_runtime_cache():
            return self.dequantize_weight().to(device=device, dtype=dtype).detach()
        key = (str(device), str(dtype))
        cached = self._dense_weight_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        weight = self.dequantize_weight().to(device=device, dtype=dtype).detach()
        self._dense_weight_cache[key] = weight
        return weight

    def dense_bias(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not _can_mutate_runtime_cache():
            return None if self.bias is None else self.bias.to(device=device, dtype=dtype).detach()
        key = (str(device), str(dtype))
        if key in self._dense_bias_cache:
            return self._dense_bias_cache[key]
        bias = None if self.bias is None else self.bias.to(device=device, dtype=dtype).detach()
        self._dense_bias_cache[key] = bias
        return bias

    def tilelang_dense_linear_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        """Expose cached dense weights for native linear fastpaths."""

        return (
            self.dense_weight(dtype=dtype, device=device),
            self.dense_bias(dtype=dtype, device=device),
            None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight(dtype=inputs.dtype, device=inputs.device)
        bias = self.dense_bias(dtype=inputs.dtype, device=inputs.device)
        return F.linear(inputs, weight, bias)


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
            replacement = AWQGPTQWeightOnlyLinear.from_linear_awq(
                module,
                bits=bits,
                group_size=normalized_group_size,
                stats=stats.get(name),
                alpha=float(policy_mapping.get("awq_alpha", 0.5)),
            )
        elif method == "gptq":
            replacement = AWQGPTQWeightOnlyLinear.from_linear_gptq(
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
    return AWQGPTQWeightOnlyQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
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
        },
    )


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
    report = QuantizationReport(
        component_name=component.name,
        backend=component.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=algorithm_executable,
        method_semantics=method_semantics,
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "algorithm_executable": algorithm_executable,
            "method_semantics": method_semantics,
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "execution_state": f"weight_only_int{bits}",
        },
    )
    return updated_model, report


__all__ = [
    "AWQGPTQWeightOnlyLinear",
    "AWQGPTQWeightOnlyQuantizationResult",
    "execute_awq_gptq_weight_only_component",
    "quantize_with_awq_weight_only",
    "quantize_with_gptq_weight_only",
]
