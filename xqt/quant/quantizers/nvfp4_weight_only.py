"""NVFP4 weight-only quantization backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.types import XQTContext

from xqt.contracts.nvfp4 import NVFP4LinearBridge, _normalize_group_scale
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
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)


_NVFP4_QUANT_CODEBOOK = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_NVFP4_SCALE_DTYPE = getattr(torch, "float8_e4m3fn", torch.float32)
_NVFP4_MAX_VALUE = 6.0
_NVFP4_SCALE_RANGE = 448.0
_NVFP4_GLOBAL_SCALE_RANGE = _NVFP4_MAX_VALUE * _NVFP4_SCALE_RANGE


@dataclass
class NVFP4QuantizationResult(QuantizedModel):
    """Result returned by the NVFP4 weight-only quantization backend."""

    backend: str = "pytorch"
    strategy: str = "w4a16_nvfp4"


def _normalize_group_size(group_size: int, input_features: int) -> int:
    return max(1, min(int(group_size), int(input_features)))


def _pad_weight_for_groups(
    weight: torch.Tensor,
    *,
    input_features: int,
    group_size: int,
) -> tuple[torch.Tensor, int]:
    padded_input_features = (
        (int(input_features) + int(group_size) - 1) // int(group_size)
    ) * int(group_size)
    if padded_input_features != input_features:
        weight = F.pad(weight, (0, padded_input_features - input_features))
    return weight, padded_input_features


def _pack_nvfp4_codes(codes: torch.Tensor) -> torch.Tensor:
    encoded = codes.to(torch.uint8)
    if encoded.shape[-1] % 2 != 0:
        encoded = F.pad(encoded, (0, 1), value=0)
    low = encoded[..., 0::2]
    high = encoded[..., 1::2] << 4
    return (low | high).contiguous()


def _quantize_to_nvfp4_codes(values: torch.Tensor) -> torch.Tensor:
    codebook = _NVFP4_QUANT_CODEBOOK.to(device=values.device, dtype=torch.float32)
    flat = values.reshape(-1, 1).to(torch.float32)
    indices = torch.argmin((flat - codebook.reshape(1, -1)).abs(), dim=1)
    return indices.reshape(values.shape).to(torch.uint8)


def _quantize_grouped_nvfp4_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    input_features: int,
    output_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    weight, padded_input_features = _pad_weight_for_groups(
        weight.detach().to(torch.float32),
        input_features=input_features,
        group_size=group_size,
    )
    grouped_weight = weight.reshape(output_features, -1, group_size)
    global_absmax = torch.clamp(weight.abs().amax(), min=1e-8)
    weight_global_scale = torch.tensor(
        [_NVFP4_GLOBAL_SCALE_RANGE / float(global_absmax.item())],
        dtype=torch.float32,
        device=weight.device,
    )
    group_absmax = grouped_weight.abs().amax(dim=2, keepdim=True)
    scale = group_absmax / _NVFP4_MAX_VALUE
    scaled_group_scale = torch.clamp(
        scale * weight_global_scale.reshape(1, 1, 1),
        min=0.0,
        max=_NVFP4_SCALE_RANGE,
    )
    stored_group_scale = scaled_group_scale.to(_NVFP4_SCALE_DTYPE)
    dequant_group_scale = stored_group_scale.to(torch.float32)
    output_scale = torch.where(
        dequant_group_scale > 0,
        weight_global_scale.reshape(1, 1, 1) / dequant_group_scale,
        torch.zeros_like(dequant_group_scale),
    )
    quantized = grouped_weight * output_scale
    codes = _quantize_to_nvfp4_codes(quantized)
    packed_weight = _pack_nvfp4_codes(codes.reshape(output_features, padded_input_features))
    return packed_weight, stored_group_scale, weight_global_scale, padded_input_features


class NVFP4WeightOnlyLinear(NVFP4LinearBridge):
    """Weight-only Linear backed by packed NVFP4 E2M1 groups."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        weight_global_scale: torch.Tensor,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
    ) -> None:
        nn.Module.__init__(self)
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.source_module_type = "NVFP4WeightOnlyLinear"
        self._xqt_nvfp4_weight_only = True
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._dense_bias_cache: dict[tuple[str, str], torch.Tensor | None] = {}
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8))
        self.register_buffer(
            "weight_scale",
            _normalize_group_scale(weight_scale).to(_NVFP4_SCALE_DTYPE),
        )
        self.register_buffer(
            "weight_global_scale",
            weight_global_scale.detach().clone().to(torch.float32),
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

    def _apply(self, fn: Any) -> "NVFP4WeightOnlyLinear":
        """Preserve NVFP4 scale/global-scale storage dtypes across .to() calls."""

        nn.Module._apply(self, fn)
        self._dense_weight_cache.clear()
        self._dense_bias_cache.clear()
        self.weight_scale = self.weight_scale.to(dtype=_NVFP4_SCALE_DTYPE)
        self.weight_global_scale = self.weight_global_scale.to(dtype=torch.float32)
        if self.bias is not None:
            self.bias = self.bias.to(dtype=torch.float32)
        return self

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        group_size: int = 16,
    ) -> "NVFP4WeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        packed_weight, weight_scale, weight_global_scale, _ = _quantize_grouped_nvfp4_weight(
            module.weight.detach(),
            group_size=normalized_group_size,
            input_features=module.in_features,
            output_features=module.out_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed_weight,
            weight_scale,
            weight_global_scale=weight_global_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
        )


def quantize_with_nvfp4_weight_only(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> NVFP4QuantizationResult:
    """Quantize Linear modules with an NVFP4 weight-only path."""

    quant_policy = (
        policy if isinstance(policy, QuantizationPolicy) else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    configured_group_size = int(policy_mapping.get("group_size", 16) or 16)
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": getattr(quant_policy, "dtype", "nvfp4"),
                "scheme": getattr(quant_policy, "scheme", "weight_only"),
            },
        )
        or "nvfp4_weight_only"
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []

    for name, module in list(target_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue
        replacement = NVFP4WeightOnlyLinear.from_linear(
            module,
            group_size=configured_group_size,
        )
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)

    return NVFP4QuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "nvfp4_weight_only_linear",
            "weight_encoding": "packed_nvfp4_e2m1",
            "scale_encoding": str(_NVFP4_SCALE_DTYPE),
            "global_scale_encoding": "float32_scalar",
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


def execute_nvfp4_weight_only_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_nvfp4_weight_only,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the NVFP4 weight-only quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    algorithm_executable = component.method not in {"awq", "gptq"}
    method_semantics = (
        "awq_gptq_label_only_groupwise_nvfp4_weight_only_storage_quantization"
        if component.method in {"awq", "gptq"}
        else "groupwise_nvfp4_weight_only_storage_quantization"
    )
    report = build_component_quantization_report(
        context,
        component,
        backend=result.backend,
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=algorithm_executable,
        method_semantics=method_semantics,
        effective_policy=effective_policy,
        result_metadata=result.metadata,
        execution_state="nvfp4_weight_only",
    )
    return updated_model, report


__all__ = [
    "NVFP4QuantizationResult",
    "NVFP4WeightOnlyLinear",
    "execute_nvfp4_weight_only_component",
    "quantize_with_nvfp4_weight_only",
]
