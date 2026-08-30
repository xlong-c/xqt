"""FP4 weight-only quantization backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.compilation import can_mutate_runtime_cache as _can_mutate_runtime_cache
from ._awq_gptq_common import _awq_group_multiplier, _gptq_corrected_weight

from xqt.contracts import QuantizedModel
from xqt.contracts.packing_int4 import (
    _normalize_group_size,
    _pad_weight_for_groups,
    _quantize_grouped_fp4_weight,
    _safe_positive,
    _unpack_int4,
)
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
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
from .base import (
    call_model as _call_model,
    iter_calibration_batches as _iter_calibration_batches,
    move_batch_to_device as _move_batch_to_device,
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)




@dataclass
class FP4QuantizationResult(QuantizedModel):
    """Result returned by the FP4 weight-only quantization backend."""

    backend: str = "pytorch"
    strategy: str = "w4a16_fp4"
    method: str | None = None


@dataclass(frozen=True)
class _LinearCalibrationStats:
    activation_abs_mean: torch.Tensor
    activation_hessian_diag: torch.Tensor
    sample_count: int


def _collect_linear_calibration_stats(
    model: nn.Module,
    *,
    module_names: Iterable[str],
    calibration_inputs: Iterable[Any] | None,
    sample_limit: int | None,
) -> dict[str, _LinearCalibrationStats]:
    if calibration_inputs is None:
        return {}
    wanted = {str(name) for name in module_names}
    if not wanted:
        return {}
    device = next(model.parameters(), torch.empty((), device="cpu")).device
    sums: dict[str, torch.Tensor] = {}
    sq_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles: list[Any] = []

    def _make_hook(name: str) -> Any:
        def _hook(module: nn.Module, inputs: tuple[Any, ...], _: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            activation = inputs[0].detach().to(torch.float32)
            if activation.ndim == 0:
                return
            flattened = activation.reshape(-1, activation.shape[-1])
            if flattened.shape[-1] != getattr(module, "in_features", flattened.shape[-1]):
                return
            sums[name] = sums.get(name, torch.zeros(flattened.shape[-1])) + flattened.abs().sum(dim=0).cpu()
            sq_sums[name] = sq_sums.get(name, torch.zeros(flattened.shape[-1])) + flattened.square().sum(dim=0).cpu()
            counts[name] = counts.get(name, 0) + int(flattened.shape[0])

        return _hook

    for name, module in model.named_modules():
        if name in wanted and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_make_hook(name)))
    if not handles:
        return {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            expected_input_count = infer_model_input_count(model)
            for batch in _iter_calibration_batches(calibration_inputs, sample_limit=sample_limit):
                inputs = extract_model_inputs(
                    batch,
                    expected_input_count=expected_input_count,
                )
                _call_model(model, _move_batch_to_device(inputs, device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    stats: dict[str, _LinearCalibrationStats] = {}
    for name, count in counts.items():
        if count <= 0:
            continue
        stats[name] = _LinearCalibrationStats(
            activation_abs_mean=sums[name] / float(count),
            activation_hessian_diag=sq_sums[name] / float(count),
            sample_count=count,
        )
    return stats


class FP4WeightOnlyLinear(nn.Module):
    """Weight-only Linear backed by packed 4-bit codes."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._dense_bias_cache: dict[tuple[str, str], torch.Tensor | None] = {}
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8))
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
        group_size: int,
    ) -> "FP4WeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        packed_weight, scale, padded_input_features = _quantize_grouped_fp4_weight(
            module.weight,
            group_size=normalized_group_size,
            input_features=module.in_features,
            output_features=module.out_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed_weight,
            scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded_input_features,
        )

    @classmethod
    def from_linear_awq(
        cls,
        module: nn.Linear,
        *,
        group_size: int,
        stats: _LinearCalibrationStats | None,
        alpha: float = 0.5,
    ) -> "FP4WeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        _, padded_input_features = _pad_weight_for_groups(
            module.weight.detach().to(torch.float32),
            input_features=module.in_features,
            group_size=normalized_group_size,
        )
        group_multiplier = _awq_group_multiplier(
            module.weight,
            stats,
            output_features=module.out_features,
            input_features=module.in_features,
            padded_input_features=padded_input_features,
            group_size=normalized_group_size,
            alpha=alpha,
        )
        packed_weight, scale, padded_input_features = _quantize_grouped_fp4_weight(
            module.weight,
            group_size=normalized_group_size,
            input_features=module.in_features,
            output_features=module.out_features,
            group_multiplier=group_multiplier,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed_weight,
            scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded_input_features,
        )

    @classmethod
    def from_linear_gptq(
        cls,
        module: nn.Linear,
        *,
        group_size: int,
        stats: _LinearCalibrationStats | None,
        dampening: float = 0.01,
    ) -> "FP4WeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        baseline = cls.from_linear(module, group_size=normalized_group_size)
        corrected_weight = _gptq_corrected_weight(
            module.weight.detach().to(torch.float32),
            baseline.dequantize_weight(),
            stats,
            dampening=dampening,
        )
        packed_weight, scale, padded_input_features = _quantize_grouped_fp4_weight(
            corrected_weight,
            group_size=normalized_group_size,
            input_features=module.in_features,
            output_features=module.out_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed_weight,
            scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded_input_features,
        )

    def dequantize_weight(self) -> torch.Tensor:
        quantized = _unpack_int4(self.packed_weight, self.padded_input_features)
        grouped = quantized.reshape(self.output_features, -1, self.group_size)
        dequantized = grouped * self.weight_scale
        return dequantized.reshape(self.output_features, self.padded_input_features)[
            :, : self.input_features
        ]

    def quantized_weight_codes(self) -> torch.Tensor:
        """Return the unpacked signed FP4 codes as a dense matrix."""

        return _unpack_int4(self.packed_weight, self.padded_input_features)[
            :, : self.input_features
        ]

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
        """Expose dequant GEMM inputs for TileLang operator wrappers."""

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
        """Expose packed FP4 inputs for TileLang operator wrappers."""

        packed_weight = self.packed_weight.to(device=device)
        scale = self.weight_scale.to(device=device, dtype=dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return packed_weight, scale, bias, None, self.input_features, self.group_size

    def triton_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        """Expose packed FP4 inputs for Triton operator wrappers."""

        return self.tilelang_packed_dequant_gemm_args(dtype=dtype, device=device)

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
        """Expose cached dense weights for Ada native linear fastpaths."""

        return (
            self.dense_weight(dtype=dtype, device=device),
            self.dense_bias(dtype=dtype, device=device),
            None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight(dtype=inputs.dtype, device=inputs.device)
        bias = self.dense_bias(dtype=inputs.dtype, device=inputs.device)
        return F.linear(inputs, weight, bias)


def quantize_with_fp4_weight_only(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> FP4QuantizationResult:
    """Quantize Linear modules with an FP4 weight-only path."""

    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = (
        dict(policy)
        if isinstance(policy, Mapping)
        else {}
    )
    configured_group_size = int(policy_mapping.get("group_size", 128) or 128)
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": getattr(quant_policy, "dtype", "fp4"),
                "scheme": getattr(quant_policy, "scheme", "weight_only"),
            },
        )
        or "w4a16_fp4"
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []

    for name, module in list(target_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue
        replacement = FP4WeightOnlyLinear.from_linear(
            module,
            group_size=configured_group_size,
        )
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)

    return FP4QuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "fp4_weight_only_linear",
            "weight_encoding": "packed_signed_int4",
            "group_size": configured_group_size,
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
                "group_size": configured_group_size,
            },
        },
    )


def quantize_with_awq_fp4(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> FP4QuantizationResult:
    """Run activation-aware FP4 weight quantization for Linear modules."""

    return _quantize_with_calibrated_fp4(
        model,
        method="awq",
        policy=policy,
        calibration_inputs=calibration_inputs,
        strategy=strategy,
        inplace=inplace,
    )


def quantize_with_gptq_fp4(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> FP4QuantizationResult:
    """Run Hessian-aware GPTQ-style FP4 weight quantization for Linear modules."""

    return _quantize_with_calibrated_fp4(
        model,
        method="gptq",
        policy=policy,
        calibration_inputs=calibration_inputs,
        strategy=strategy,
        inplace=inplace,
    )


def _quantize_with_calibrated_fp4(
    model: nn.Module,
    *,
    method: str,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    calibration_inputs: Iterable[Any] | None = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> FP4QuantizationResult:
    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
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
            replacement = FP4WeightOnlyLinear.from_linear_awq(
                module,
                group_size=normalized_group_size,
                stats=stats.get(name),
                alpha=float(policy_mapping.get("awq_alpha", 0.5)),
            )
        elif method == "gptq":
            replacement = FP4WeightOnlyLinear.from_linear_gptq(
                module,
                group_size=normalized_group_size,
                stats=stats.get(name),
                dampening=float(policy_mapping.get("gptq_dampening", 0.01)),
            )
        else:
            raise ValueError(f"Unsupported calibrated FP4 method: {method}")
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": getattr(quant_policy, "dtype", "fp4"),
                "scheme": getattr(quant_policy, "scheme", "weight_only"),
            },
        )
        or "w4a16_fp4"
    )
    return FP4QuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": f"{method}_fp4_weight_only_linear",
            "weight_encoding": "packed_signed_int4",
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
                "group_size": configured_group_size,
                "sample_limit": normalized_sample_limit,
            },
        },
    )


def execute_fp4_weight_only_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_fp4_weight_only,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the FP4 weight-only quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    if component.method == "awq":
        result = quantize_with_awq_fp4(
            target_model,
            policy=effective_policy,
            calibration_inputs=context.calibration_inputs,
            strategy=component.strategy or effective_policy.get("strategy"),
            inplace=True,
        )
    elif component.method == "gptq":
        result = quantize_with_gptq_fp4(
            target_model,
            policy=effective_policy,
            calibration_inputs=context.calibration_inputs,
            strategy=component.strategy or effective_policy.get("strategy"),
            inplace=True,
        )
    else:
        result = quantize_fn(
            target_model,
            policy=effective_policy,
            strategy=component.strategy or effective_policy.get("strategy"),
            inplace=True,
        )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    calibrated_method = component.method in {"awq", "gptq"}
    calibrated_module_count = int(result.metadata.get("calibrated_module_count", 0) or 0)
    algorithm_executable = (not calibrated_method) or calibrated_module_count > 0
    method_semantics = (
        "awq_activation_aware_fp4_weight_only_quantization"
        if component.method == "awq" and algorithm_executable
        else "awq_label_only_groupwise_fp4_weight_only_storage_quantization"
        if component.method == "awq"
        else "gptq_hessian_aware_fp4_weight_only_quantization"
        if component.method == "gptq" and algorithm_executable
        else "gptq_label_only_groupwise_fp4_weight_only_storage_quantization"
        if component.method == "gptq"
        else "groupwise_fp4_weight_only_storage_quantization"
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
        execution_state="w4a16_fp4",
    )
    return updated_model, report


__all__ = [
    "FP4QuantizationResult",
    "FP4WeightOnlyLinear",
    "execute_fp4_weight_only_component",
    "quantize_with_awq_fp4",
    "quantize_with_fp4_weight_only",
    "quantize_with_gptq_fp4",
]
