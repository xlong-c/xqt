"""Reference FP4 weight-only quantization backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .policy import QuantizationPolicy, should_quantize_module


@dataclass
class FP4QuantizationResult:
    """Result returned by the reference FP4 quantization backend."""

    model: nn.Module
    backend: str = "pytorch"
    strategy: str = "fp4_weight_only"
    quantized_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _encode_signed_nibble(values: torch.Tensor) -> torch.Tensor:
    encoded = torch.where(values < 0, values + 16, values)
    return encoded.to(torch.uint8)


def _decode_signed_nibble(values: torch.Tensor) -> torch.Tensor:
    signed = torch.where(values >= 8, values.to(torch.int16) - 16, values.to(torch.int16))
    return signed.to(torch.float32)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    encoded = _encode_signed_nibble(values)
    if encoded.shape[-1] % 2 != 0:
        encoded = F.pad(encoded, (0, 1), value=0)
    low = encoded[..., 0::2]
    high = encoded[..., 1::2] << 4
    return (low | high).contiguous()


def _unpack_int4(packed: torch.Tensor, input_features: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    unpacked = unpacked[..., :input_features]
    return _decode_signed_nibble(unpacked)


class ReferenceFP4Linear(nn.Module):
    """Reference weight-only Linear backed by packed 4-bit codes."""

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
    ) -> "ReferenceFP4Linear":
        weight = module.weight.detach().to(torch.float32)
        normalized_group_size = max(1, min(int(group_size), module.in_features))
        padded_input_features = (
            (module.in_features + normalized_group_size - 1) // normalized_group_size
        ) * normalized_group_size
        if padded_input_features != module.in_features:
            weight = F.pad(weight, (0, padded_input_features - module.in_features))
        grouped_weight = weight.reshape(module.out_features, -1, normalized_group_size)
        max_abs = grouped_weight.abs().amax(dim=2, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
        quantized = torch.clamp(torch.round(grouped_weight / scale), min=-8, max=7).to(torch.int8)
        packed_weight = _pack_int4(quantized.reshape(module.out_features, padded_input_features))
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight().to(device=inputs.device, dtype=inputs.dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def quantize_with_reference_fp4(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> FP4QuantizationResult:
    """Quantize Linear modules with a reference FP4 weight-only path."""

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
    selected_strategy = strategy or str(getattr(quant_policy, "dtype", "fp4_weight_only"))
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
            ReferenceFP4Linear.from_linear(
                module,
                group_size=configured_group_size,
            ),
        )
        quantized_modules.append(name)

    return FP4QuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "reference_fp4_linear_weight_only",
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


__all__ = [
    "FP4QuantizationResult",
    "ReferenceFP4Linear",
    "quantize_with_reference_fp4",
]
