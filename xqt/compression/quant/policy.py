"""Quantization policy helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn


from xqt.contracts.model_structure import (
    ModelStructureContract,
    is_module_path_within,
    resolve_structure_role,
    structure_contract_keep_high_precision_paths,
)
from xqt.contracts.moe import (
    DEFAULT_MOE_EXPERT_NAME_PATTERNS,
    DEFAULT_MOE_ROUTER_NAME_PATTERNS,
    classify_moe_module,
    is_moe_expert_module,
    is_moe_router_module,
)

DEFAULT_EXCLUDED_TYPES = (
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Embedding",
)

DEFAULT_QUANTIZABLE_TYPES = ("Linear", "Conv2d", "MultiheadAttention")


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
    structure_role: str | None = None


def _matches_pattern(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def should_quantize_module(
    name: str,
    module: nn.Module,
    policy: QuantizationPolicy,
    *,
    structure_contract: ModelStructureContract | None = None,
) -> bool:
    """Return whether a module matches the quantization policy.

    ``structure_contract`` 可选: 契约声明 ``keep_high_precision`` 的组件
    路径 (含子模块) 一律不量化, 优先级高于 policy 的 include 规则; 缺省
    保持纯 policy 判定.
    """

    if structure_contract is not None and any(
        is_module_path_within(name, path)
        for path in structure_contract_keep_high_precision_paths(structure_contract)
    ):
        return False

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
    *,
    structure_contract: ModelStructureContract | None = None,
) -> list[QuantizationCandidate]:
    """Inspect a model and return quantization candidates."""

    policy = policy or QuantizationPolicy()
    protected = (
        structure_contract_keep_high_precision_paths(structure_contract)
        if structure_contract is not None
        else ()
    )
    candidates: list[QuantizationCandidate] = []
    for name, module in model.named_modules():
        if not name:
            continue
        module_type = type(module).__name__
        parameter_count = _module_parameter_count(module)
        quantize = should_quantize_module(
            name, module, policy, structure_contract=structure_contract
        )
        structure_role = (
            None
            if structure_contract is None
            else (resolve_structure_role(structure_contract, name) or "undeclared")
        )
        if quantize:
            reason = "matched policy"
        elif protected and any(is_module_path_within(name, path) for path in protected):
            reason = "structure contract keep_high_precision"
        else:
            reason = "filtered by policy"
        candidates.append(
            QuantizationCandidate(
                name=name,
                module_type=module_type,
                parameter_count=parameter_count,
                quantize=quantize,
                reason=reason,
                structure_role=structure_role,
            )
        )
    return candidates


__all__ = [
    "DEFAULT_EXCLUDED_TYPES",
    "DEFAULT_QUANTIZABLE_TYPES",
    "QuantizationCandidate",
    "QuantizationPolicy",
    "classify_moe_module",
    "is_moe_expert_module",
    "is_moe_router_module",
    "list_quantizable_modules",
    "should_quantize_module",
]
