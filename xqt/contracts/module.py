"""Shared model-side contracts used by conversion and operator lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from xqt.operator_opt.backends.gemm_precision import MatmulPrecisionSpec


OperatorKind = Literal["linear", "conv2d", "layernorm", "feedforward"]


@dataclass(frozen=True)
class PrecisionPolicy:
    """User-facing compute precision intent for model-side transformation."""

    activation: str = "fp16"
    weight: str = "fp16"
    bias: str = "fp16"
    mma: str = "fp16"
    accum: str = "fp32"
    output: str = "fp16"

    def to_dict(self) -> dict[str, str]:
        return {
            "activation": self.activation,
            "weight": self.weight,
            "bias": self.bias,
            "mma": self.mma,
            "accum": self.accum,
            "output": self.output,
        }

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

        resolved = MatmulPrecisionSpec.from_roles(
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
        return cls(
            activation=resolved.activation,
            weight=resolved.weight,
            bias=resolved.bias,
            mma=resolved.mma,
            accum=resolved.accum,
            output=resolved.output,
        )


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
