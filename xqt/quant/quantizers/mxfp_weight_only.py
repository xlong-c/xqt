"""MXFP weight-only quantization backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

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


def _can_mutate_runtime_cache() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    return not torch.jit.is_tracing()


@dataclass
class MXFPQuantizationResult(QuantizedModel):
    """Result returned by the MXFP weight-only quantization backend."""

    backend: str = "pytorch"
    strategy: str = "w4a16_mxfp4"


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


def _validate_mxfp_precision(precision: int) -> int:
    normalized = int(precision)
    if normalized not in {4, 6, 8}:
        raise ValueError(f"mxfp precision must be 4, 6, or 8, got {precision}")
    return normalized


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    encoded = torch.where(values < 0, values + 16, values).to(torch.uint8)
    if encoded.shape[-1] % 2 != 0:
        encoded = F.pad(encoded, (0, 1), value=0)
    low = encoded[..., 0::2]
    high = encoded[..., 1::2] << 4
    return (low | high).contiguous()


def _unpack_signed_int4(packed: torch.Tensor, block_size: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    unpacked = unpacked[..., : int(block_size)]
    signed = torch.where(
        unpacked >= 8,
        unpacked.to(torch.int16) - 16,
        unpacked.to(torch.int16),
    )
    return signed.to(torch.float32)


def _pack_mxfp_blocks(
    tensor: torch.Tensor,
    *,
    precision: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    normalized_precision = _validate_mxfp_precision(precision)
    weight = tensor.to(torch.float32)
    if weight.ndim != 2:
        raise ValueError("mxfp weight-only quantizer expects a 2D weight tensor")
    rows, input_features = weight.shape
    padded_input_features = ((int(input_features) + block_size - 1) // block_size) * block_size
    if padded_input_features != int(input_features):
        weight = F.pad(weight, (0, padded_input_features - int(input_features)))
    blocks = weight.reshape(rows, -1, block_size)
    block_max = torch.clamp(blocks.abs().amax(dim=2, keepdim=True), min=1e-10)
    max_int = (1 << (normalized_precision - 1)) - 1
    scale = block_max / max_int
    quantized = torch.round(blocks / scale).to(torch.int32)
    quantized = torch.clamp(quantized, min=-max_int, max=max_int)
    if normalized_precision == 8:
        packed = quantized.to(torch.int8).reshape(rows, padded_input_features)
    elif normalized_precision == 6:
        # Phase 1 weight-only path keeps one int8 lane per value.
        packed = quantized.to(torch.int8).reshape(rows, padded_input_features)
    else:
        packed = _pack_signed_int4(quantized.to(torch.int8).reshape(rows, padded_input_features))
    return packed.contiguous(), scale.reshape(rows, -1).to(torch.float32), padded_input_features


def _unpack_mxfp_blocks(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    precision: int,
    block_size: int,
    padded_input_features: int,
) -> torch.Tensor:
    normalized_precision = _validate_mxfp_precision(precision)
    rows = int(scale.shape[0])
    groups = int(scale.shape[1])
    if normalized_precision in {8, 6}:
        mantissas = packed.to(torch.float32).reshape(rows, groups, block_size)
    else:
        unpacked = _unpack_signed_int4(packed.to(torch.uint8), padded_input_features)
        mantissas = unpacked.reshape(rows, groups, block_size)
    dequantized = mantissas * scale.reshape(rows, groups, 1).to(
        dtype=torch.float32,
        device=mantissas.device,
    )
    return dequantized.reshape(rows, padded_input_features)


class MXFPWeightOnlyLinear(nn.Module):
    """Weight-only Linear backed by MXFP-packed per-row weight blocks."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        block_size: int,
        mx_precision: int,
        padded_input_features: int,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.block_size = int(block_size)
        self.mx_precision = int(mx_precision)
        self.padded_input_features = int(padded_input_features)
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._dense_bias_cache: dict[tuple[str, str], torch.Tensor | None] = {}
        if self.mx_precision == 4:
            self.register_buffer("packed_weight", packed_weight.to(torch.uint8))
        else:
            self.register_buffer("packed_weight", packed_weight.to(torch.int8))
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        mx_precision: int,
        block_size: int,
    ) -> "MXFPWeightOnlyLinear":
        normalized_precision = _validate_mxfp_precision(mx_precision)
        normalized_block_size = max(1, int(block_size))
        weight = module.weight.detach().to(torch.float32)
        packed, scale, padded_input_features = _pack_mxfp_blocks(
            weight,
            precision=normalized_precision,
            block_size=normalized_block_size,
        )
        rows = int(module.out_features)
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed,
            scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            block_size=normalized_block_size,
            mx_precision=normalized_precision,
            padded_input_features=padded_input_features,
        )

    def dequantize_weight(self) -> torch.Tensor:
        unpacked = _unpack_mxfp_blocks(
            self.packed_weight,
            self.weight_scale,
            precision=self.mx_precision,
            block_size=self.block_size,
            padded_input_features=self.padded_input_features,
        )
        return unpacked[:, : self.input_features]

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
        """Expose dense cache args for generic Linear engines."""

        return (
            self.dense_weight(dtype=dtype, device=device),
            self.dense_bias(dtype=dtype, device=device),
            None,
        )

    def tilelang_packed_mxfp_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        """Expose packed MXFP4 inputs for TileLang operator wrappers."""

        if self.mx_precision != 4:
            raise ValueError(
                "TileLang packed MXFP dequant GEMM path currently supports MXFP4 only"
            )
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return (
            self.packed_weight.to(device=device),
            self.weight_scale.to(device=device, dtype=dtype).unsqueeze(-1),
            bias,
            None,
            self.input_features,
            self.block_size,
        )

    def triton_mxfp_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:
        """Expose packed MXFP inputs for Triton operator wrappers."""

        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return (
            self.packed_weight.to(device=device),
            self.weight_scale.to(device=device, dtype=torch.float32),
            bias,
            self.block_size,
            self.mx_precision,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight(dtype=inputs.dtype, device=inputs.device)
        bias = self.dense_bias(dtype=inputs.dtype, device=inputs.device)
        return F.linear(inputs, weight, bias)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def quantize_with_mxfp_weight_only(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> MXFPQuantizationResult:
    """Quantize Linear modules with an MXFP weight-only path."""

    quant_policy = (
        policy if isinstance(policy, QuantizationPolicy) else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    configured_block_size = int(policy_mapping.get("block_size", 32) or 32)
    configured_precision = _validate_mxfp_precision(int(policy_mapping.get("precision", 8) or 8))
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": getattr(quant_policy, "dtype", "mxfp"),
                "scheme": getattr(quant_policy, "scheme", "weight_only"),
            },
        )
        or "mxfp_weight_only"
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []

    for name, module in list(target_model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue
        _replace_submodule(
            target_model,
            name,
            MXFPWeightOnlyLinear.from_linear(
                module,
                mx_precision=configured_precision,
                block_size=configured_block_size,
            ),
        )
        quantized_modules.append(name)

    return MXFPQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "mxfp_weight_only_linear",
            "weight_encoding": f"mxfp{configured_precision}_packed_shared_scale",
            "mx_precision": configured_precision,
            "block_size": configured_block_size,
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
                "precision": configured_precision,
                "block_size": configured_block_size,
            },
        },
    )


def execute_mxfp_weight_only_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_mxfp_weight_only,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the MXFP weight-only quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    high_precision_modules = prefix_module_names(component.keep_high_precision, component.target_path)
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
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=component.method not in {"awq", "gptq"},
        method_semantics=(
            "awq_gptq_label_only_groupwise_weight_only_storage_quantization"
            if component.method in {"awq", "gptq"}
            else "groupwise_mxfp_weight_only_storage_quantization"
        ),
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "algorithm_executable": component.method not in {"awq", "gptq"},
            "method_semantics": (
                "awq_gptq_label_only_groupwise_weight_only_storage_quantization"
                if component.method in {"awq", "gptq"}
                else "groupwise_mxfp_weight_only_storage_quantization"
            ),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "execution_state": "w4a16_mxfp4",
        },
    )
    return updated_model, report


__all__ = [
    "MXFPQuantizationResult",
    "MXFPWeightOnlyLinear",
    "execute_mxfp_weight_only_component",
    "quantize_with_mxfp_weight_only",
]
