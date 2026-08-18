"""Canonical quantization strategy, scheme, and nature facts."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from .quant_scheme import QuantScheme


class QuantizationNature(str, enum.Enum):
    """Static classification of a requested quantization compute contract."""

    TRUE = "true"
    PSEUDO = "pseudo"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QuantStrategyDefinition:
    """One config-facing strategy and its orthogonal storage semantics."""

    scheme: QuantScheme
    nature: QuantizationNature


CANONICAL_QUANT_METHODS = (
    "none",
    "awq",
    "gptq",
    "svd",
    "convrot",
    "turboquant",
    "moe",
    "moe_weight_only",
)

CANONICAL_QUANT_COMPUTES = (
    "dequant_fp16",
    "w8a8_int8_mma",
    "fp8_mma",
    "qdq_static",
    "qdq_dynamic",
    "dequant_gemm",
)

_STRATEGY_DEFINITIONS = {
    "w4a16_int4": QuantStrategyDefinition(
        QuantScheme("int4", "groupwise", group_size=128),
        QuantizationNature.PSEUDO,
    ),
    "w8a16_int8": QuantStrategyDefinition(
        QuantScheme("int8", "per_channel"),
        QuantizationNature.PSEUDO,
    ),
    "w4a16_fp4": QuantStrategyDefinition(
        QuantScheme("fp4", "groupwise", group_size=128),
        QuantizationNature.PSEUDO,
    ),
    "w4a16_nvfp4": QuantStrategyDefinition(
        QuantScheme("nvfp4", "block", group_size=16),
        QuantizationNature.PSEUDO,
    ),
    "w4a16_mxfp4": QuantStrategyDefinition(
        QuantScheme("mxfp4", "block", group_size=32),
        QuantizationNature.PSEUDO,
    ),
    "w8a16_mxfp8": QuantStrategyDefinition(
        QuantScheme("mxfp8", "block", group_size=32),
        QuantizationNature.PSEUDO,
    ),
    "w8a16_fp8_e4m3": QuantStrategyDefinition(
        QuantScheme("fp8_e4m3", "per_channel"),
        QuantizationNature.PSEUDO,
    ),
    "w8a16_fp8_e5m2": QuantStrategyDefinition(
        QuantScheme("fp8_e5m2", "per_channel"),
        QuantizationNature.PSEUDO,
    ),
    "w8a8_int8": QuantStrategyDefinition(
        QuantScheme(
            "int8",
            "per_channel",
            activation_dtype="int8",
            activation_mode="dynamic",
        ),
        QuantizationNature.UNKNOWN,
    ),
    "w8a8_fp8_e4m3": QuantStrategyDefinition(
        QuantScheme(
            "fp8_e4m3",
            "per_channel",
            activation_dtype="fp8_e4m3",
            activation_mode="dynamic",
        ),
        QuantizationNature.UNKNOWN,
    ),
    "w8a8_fp8_e5m2": QuantStrategyDefinition(
        QuantScheme(
            "fp8_e5m2",
            "per_channel",
            activation_dtype="fp8_e5m2",
            activation_mode="dynamic",
        ),
        QuantizationNature.UNKNOWN,
    ),
    "w4a4_int4": QuantStrategyDefinition(
        QuantScheme(
            "int4",
            "groupwise",
            group_size=128,
            activation_dtype="int4",
            activation_mode="dynamic",
        ),
        QuantizationNature.PSEUDO,
    ),
    "w4a4_fp4": QuantStrategyDefinition(
        QuantScheme(
            "fp4",
            "groupwise",
            group_size=128,
            activation_dtype="fp4",
            activation_mode="dynamic",
        ),
        QuantizationNature.PSEUDO,
    ),
    "w4a4_nvfp4": QuantStrategyDefinition(
        QuantScheme(
            "nvfp4",
            "block",
            group_size=16,
            activation_dtype="nvfp4",
            activation_mode="dynamic",
        ),
        QuantizationNature.PSEUDO,
    ),
    "w4a4_mxfp4": QuantStrategyDefinition(
        QuantScheme(
            "mxfp4",
            "block",
            group_size=32,
            activation_dtype="mxfp4",
            activation_mode="dynamic",
        ),
        QuantizationNature.PSEUDO,
    ),
}

QUANT_STRATEGY_DEFINITIONS: Mapping[str, QuantStrategyDefinition] = MappingProxyType(
    _STRATEGY_DEFINITIONS
)

QUANT_COMPUTE_NATURES: Mapping[str, QuantizationNature] = MappingProxyType(
    {
        "dequant_fp16": QuantizationNature.PSEUDO,
        "w8a8_int8_mma": QuantizationNature.TRUE,
        "fp8_mma": QuantizationNature.TRUE,
        "qdq_static": QuantizationNature.PSEUDO,
        "qdq_dynamic": QuantizationNature.PSEUDO,
        "dequant_gemm": QuantizationNature.PSEUDO,
    }
)

CANONICAL_QUANT_STRATEGIES = tuple(QUANT_STRATEGY_DEFINITIONS)


def canonical_quant_strategies() -> tuple[str, ...]:
    """Return canonical strategy names in declaration order."""

    return CANONICAL_QUANT_STRATEGIES


def strategy_scheme_templates() -> dict[str, dict[str, object]]:
    """Return mutable copies of canonical strategy scheme templates."""

    return {
        name: asdict(definition.scheme)
        for name, definition in QUANT_STRATEGY_DEFINITIONS.items()
    }


def quantization_nature_for_strategy(strategy: str) -> QuantizationNature:
    """Return the declared nature for one normalized strategy name."""

    definition = QUANT_STRATEGY_DEFINITIONS.get(strategy)
    if definition is None:
        return QuantizationNature.UNKNOWN
    return definition.nature


def quantization_nature_for_compute(compute: str) -> QuantizationNature:
    """Return the declared nature for one normalized compute contract."""

    return QUANT_COMPUTE_NATURES.get(compute, QuantizationNature.UNKNOWN)


__all__ = [
    "CANONICAL_QUANT_COMPUTES",
    "CANONICAL_QUANT_METHODS",
    "CANONICAL_QUANT_STRATEGIES",
    "QUANT_COMPUTE_NATURES",
    "QUANT_STRATEGY_DEFINITIONS",
    "QuantStrategyDefinition",
    "QuantizationNature",
    "canonical_quant_strategies",
    "quantization_nature_for_compute",
    "quantization_nature_for_strategy",
    "strategy_scheme_templates",
]
