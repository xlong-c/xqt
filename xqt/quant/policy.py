"""Quantization policy helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn


DEFAULT_QUANTIZABLE_TYPES = ("Linear", "Conv2d", "MultiheadAttention")
DEFAULT_EXCLUDED_TYPES = (
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Embedding",
)

DEFAULT_MOE_EXPERT_NAME_PATTERNS: tuple[str, ...] = (
    r"experts?\.\d+",
    r"experts?\.[^.]+",
    r"shared_expert",
    r"shared_experts?",
    r"mlp\.experts?",
)
DEFAULT_MOE_ROUTER_NAME_PATTERNS: tuple[str, ...] = (
    r"router",
    r"gate(?!_proj)",
    r"gating",
    r"moe_gate",
    r"expert_gate",
)


@dataclass
class QuantizationPolicy:
    """Module selection policy for quantization passes."""

    dtype: str = "int8"
    scheme: str = "weight_only"
    include_module_types: tuple[str, ...] = DEFAULT_QUANTIZABLE_TYPES
    exclude_module_types: tuple[str, ...] = DEFAULT_EXCLUDED_TYPES
    include_name_patterns: tuple[str, ...] = ()
    exclude_name_patterns: tuple[str, ...] = ("head", "classifier")
    include_module_names: tuple[str, ...] = ()
    exclude_module_names: tuple[str, ...] = ()
    min_parameters: int = 0


@dataclass
class QuantizationCandidate:
    """Summary of whether a module should be quantized."""

    name: str
    module_type: str
    parameter_count: int
    quantize: bool
    reason: str


def _matches_pattern(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def should_quantize_module(
    name: str,
    module: nn.Module,
    policy: QuantizationPolicy,
) -> bool:
    """Return whether a module matches the quantization policy."""

    module_type = type(module).__name__

    if name in policy.include_module_names:
        return True
    if _matches_pattern(name, policy.include_name_patterns):
        return True

    if name in policy.exclude_module_names:
        return False
    if _matches_pattern(name, policy.exclude_name_patterns):
        return False
    if module_type in policy.exclude_module_types:
        return False
    if policy.include_module_types and module_type not in policy.include_module_types:
        return False

    return _module_parameter_count(module) >= policy.min_parameters


def list_quantizable_modules(
    model: nn.Module,
    policy: QuantizationPolicy | None = None,
) -> list[QuantizationCandidate]:
    """Inspect a model and return quantization candidates."""

    policy = policy or QuantizationPolicy()
    candidates: list[QuantizationCandidate] = []
    for name, module in model.named_modules():
        if not name:
            continue
        module_type = type(module).__name__
        parameter_count = _module_parameter_count(module)
        quantize = should_quantize_module(name, module, policy)
        reason = "matched policy" if quantize else "filtered by policy"
        candidates.append(
            QuantizationCandidate(
                name=name,
                module_type=module_type,
                parameter_count=parameter_count,
                quantize=quantize,
                reason=reason,
            )
        )
    return candidates


def is_moe_expert_module(
    name: str,
    *,
    patterns: Sequence[str] | None = None,
) -> bool:
    """Return True when ``name`` matches MoE expert path patterns."""

    return _matches_pattern(name, patterns or DEFAULT_MOE_EXPERT_NAME_PATTERNS)


def is_moe_router_module(
    name: str,
    *,
    patterns: Sequence[str] | None = None,
) -> bool:
    """Return True when ``name`` matches MoE router / gate path patterns."""

    return _matches_pattern(name, patterns or DEFAULT_MOE_ROUTER_NAME_PATTERNS)


def classify_moe_module(
    name: str,
    *,
    expert_patterns: Sequence[str] | None = None,
    router_patterns: Sequence[str] | None = None,
) -> str:
    """Classify a module path as expert / router / shared_expert / other."""

    if is_moe_router_module(name, patterns=router_patterns):
        return "router"
    if re.search(r"shared_expert", name):
        return "shared_expert"
    if is_moe_expert_module(name, patterns=expert_patterns):
        return "expert"
    return "other"


__all__ = [
    "DEFAULT_EXCLUDED_TYPES",
    "DEFAULT_MOE_EXPERT_NAME_PATTERNS",
    "DEFAULT_MOE_ROUTER_NAME_PATTERNS",
    "DEFAULT_QUANTIZABLE_TYPES",
    "QuantizationCandidate",
    "QuantizationPolicy",
    "classify_moe_module",
    "is_moe_expert_module",
    "is_moe_router_module",
    "list_quantizable_modules",
    "should_quantize_module",
]
