"""Shared precision and operator-lowering contracts for XQT kernels.

``PrecisionPolicy`` is used by both ``xqt.kernels.ops`` GEMM dispatch and
``xqt.kernels.nn`` conversion, so it lives next to those packages rather
than inside ``nn`` (ops/_impl must not import the facade).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from xqt.contracts.module import (
    CompositePrecisionGemmSpec,
    coerce_composite_precision_gemm_spec,
)
from xqt.core.base import XQTBackendError


OperatorKind = Literal[
    "linear",
    "conv2d",
    "layernorm",
    "feedforward",
    "attention",
    "transformer_block",
]

_PRECISION_ALIASES = {
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "fp4e2m1": "fp4",
    "nv_fp4": "nvfp4",
    "nv-fp4": "nvfp4",
}
_SUPPORTED_PRECISION_NAMES = {
    "fp16",
    "bf16",
    "fp32",
    "int8",
    "fp8",
    "int4",
    "fp4",
    "nvfp4",
    "mxfp8",
    "mxfp6",
    "mxfp4",
}
_PRECISION_ROLE_ALIASES = {
    "a": "activation",
    "lhs": "activation",
    "input": "activation",
    "activation": "activation",
    "activation_dtype": "activation",
    "b": "weight",
    "rhs": "weight",
    "weight": "weight",
    "weight_dtype": "weight",
    "c": "bias",
    "bias": "bias",
    "bias_dtype": "bias",
    "addend": "bias",
    "addend_dtype": "bias",
    "mma": "mma",
    "mma_dtype": "mma",
    "acc": "accum",
    "accum": "accum",
    "accum_dtype": "accum",
    "accumulator": "accum",
    "accumulator_dtype": "accum",
    "o": "output",
    "out": "output",
    "output": "output",
    "output_dtype": "output",
}


def _canonical_precision_name(name: str) -> str:
    normalized = str(name).strip().lower()
    canonical = _PRECISION_ALIASES.get(normalized, normalized)
    if canonical not in _SUPPORTED_PRECISION_NAMES:
        choices = ", ".join(sorted(_SUPPORTED_PRECISION_NAMES))
        raise XQTBackendError(
            f"Unsupported precision name: {name}. Allowed: {choices}"
        )
    return canonical


def _canonical_precision_role(name: str) -> str:
    normalized = str(name).strip().lower()
    try:
        return _PRECISION_ROLE_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_PRECISION_ROLE_ALIASES))
        raise XQTBackendError(
            f"Unsupported precision field: {name}. Allowed: {choices}"
        ) from exc


@dataclass(frozen=True)
class PrecisionPolicy:
    """User-facing compute precision intent for model-side transformation."""

    activation: str = "fp16"
    weight: str = "fp16"
    bias: str = "fp16"
    mma: str = "fp16"
    accum: str = "fp32"
    output: str = "fp16"
    composite_gemm: CompositePrecisionGemmSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "activation": self.activation,
            "weight": self.weight,
            "bias": self.bias,
            "mma": self.mma,
            "accum": self.accum,
            "output": self.output,
        }
        if self.composite_gemm is not None:
            payload["composite_gemm"] = self.composite_gemm.to_dict()
        return payload

    @classmethod
    def from_matmul(
        cls,
        *,
        A: str | None = None,
        B: str | None = None,
        C: str | None = None,
        O: str | None = None,
        activation: str | None = None,
        weight: str | None = None,
        bias: str | None = None,
        mma: str = "fp16",
        accum: str = "fp32",
        output: str | None = None,
    ) -> "PrecisionPolicy":
        """Build a policy from ``A x B + C = O`` matrix role names."""

        return cls.from_roles(
            A=A,
            B=B,
            C=C,
            O=O,
            activation=activation,
            weight=weight,
            bias=bias,
            mma=mma,
            accum=accum,
            output=output,
        )

    @classmethod
    def from_roles(
        cls,
        *,
        A: str | None = None,
        B: str | None = None,
        C: str | None = None,
        O: str | None = None,
        activation: str | None = None,
        weight: str | None = None,
        bias: str | None = None,
        mma: str = "fp16",
        accum: str = "fp32",
        output: str | None = None,
    ) -> "PrecisionPolicy":
        """Build one shared policy from GEMM role names."""

        return cls.from_mapping(
            {
                "A": A
                if A is not None
                else activation
                if activation is not None
                else "fp16",
                "B": B if B is not None else weight if weight is not None else "fp16",
                "C": C if C is not None else bias if bias is not None else "fp16",
                "mma": mma,
                "accum": accum,
                "O": O if O is not None else output if output is not None else "fp16",
            }
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PrecisionPolicy":
        """Normalize semantic or ``A x B + C = O`` precision fields."""

        payload = cls.normalize_fields(values)
        activation = payload.get("activation", "fp16")
        return cls(
            activation=activation,
            weight=payload.get("weight", activation),
            bias=payload.get("bias", payload.get("output", activation)),
            mma=payload.get("mma", activation),
            accum=payload.get("accum", "fp32"),
            output=payload.get("output", activation),
            composite_gemm=coerce_composite_precision_gemm_spec(
                payload.get("composite_gemm")
            ),
        )

    @staticmethod
    def normalize_fields(values: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize precision aliases without adding omitted role defaults."""

        normalized: dict[str, Any] = {}
        for key, value in values.items():
            raw_key = str(key).strip().lower()
            if raw_key == "composite_gemm":
                normalized["composite_gemm"] = coerce_composite_precision_gemm_spec(
                    value
                )
                continue
            normalized[_canonical_precision_role(str(key))] = _canonical_precision_name(
                str(value)
            )
        return normalized

    @staticmethod
    def canonical_name(name: str, *, allow_auto: bool = False) -> str:
        """Normalize one precision name for a contract or runtime intent."""

        normalized = str(name).strip().lower()
        if allow_auto and normalized == "auto":
            return normalized
        return _canonical_precision_name(normalized)

    @staticmethod
    def canonical_field(name: str) -> str:
        """Normalize one semantic or GEMM role field name."""

        return _canonical_precision_role(name)


@dataclass(frozen=True)
class FeedForwardPrecisionPolicy:
    """Structured per-projection precision intent for FeedForward conversion."""

    default: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    proj_in: PrecisionPolicy | None = None
    proj_gate: PrecisionPolicy | None = None
    proj_out: PrecisionPolicy | None = None

    def projection_policies(self) -> dict[str, PrecisionPolicy]:
        policies: dict[str, PrecisionPolicy] = {}
        if self.proj_in is not None:
            policies["proj_in"] = self.proj_in
        if self.proj_gate is not None:
            policies["proj_gate"] = self.proj_gate
        if self.proj_out is not None:
            policies["proj_out"] = self.proj_out
        return policies


@dataclass(frozen=True)
class TensorStorageSpec:
    """Storage and logical representation for one model tensor role."""

    storage_dtype: str
    logical_dtype: str
    layout: str
    packed: bool = False
    group_size: int | None = None
    scale_dtype: str | None = None
    scale_layout: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "storage_dtype": self.storage_dtype,
            "logical_dtype": self.logical_dtype,
            "layout": self.layout,
            "packed": self.packed,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "scale_layout": self.scale_layout,
        }


@dataclass(frozen=True)
class FusionIntent:
    """Requested operator fusion semantics independent of one kernel engine."""

    patterns: tuple[str, ...] = ()
    epilogue: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "patterns": list(self.patterns),
            "epilogue": list(self.epilogue),
        }


@dataclass(frozen=True)
class ModuleContract:
    """Engine-independent lowering contract for one transformed module."""

    operator_kind: OperatorKind
    policy: PrecisionPolicy
    input_spec: TensorStorageSpec
    weight_spec: TensorStorageSpec
    output_dtype: str
    epilogue: tuple[str, ...] = ()
    fusion: FusionIntent = field(default_factory=FusionIntent)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_kind": self.operator_kind,
            "policy": self.policy.to_dict(),
            "input_spec": self.input_spec.to_dict(),
            "weight_spec": self.weight_spec.to_dict(),
            "output_dtype": self.output_dtype,
            "epilogue": list(self.epilogue),
            "fusion": self.fusion.to_dict(),
            "metadata": dict(self.metadata),
        }


OperatorContract = ModuleContract


__all__ = [
    "FeedForwardPrecisionPolicy",
    "FusionIntent",
    "ModuleContract",
    "OperatorContract",
    "OperatorKind",
    "PrecisionPolicy",
    "TensorStorageSpec",
]
