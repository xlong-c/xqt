"""Shared model-side contracts used by conversion and operator lowering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from xqt.core.errors import XQTBackendError


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
CompositeExecutionMode = Literal["reject", "reference", "split", "fused"]
_SUPPORTED_COMPOSITE_EXECUTION_MODES = {
    "reject",
    "reference",
    "split",
    "fused",
}


def _canonical_composite_execution_mode(mode: str) -> CompositeExecutionMode:
    normalized = str(mode).strip().lower()
    if normalized not in _SUPPORTED_COMPOSITE_EXECUTION_MODES:
        choices = ", ".join(sorted(_SUPPORTED_COMPOSITE_EXECUTION_MODES))
        raise XQTBackendError(
            f"Unsupported composite execution mode: {mode}. Allowed: {choices}"
        )
    return normalized  # type: ignore[return-value]


def _default_composite_available_modes(
    *,
    backend: str | None = None,
) -> tuple[CompositeExecutionMode, ...]:
    # v1 only guarantees reference/split. Fused stays contract/capability/report only.
    del backend
    return ("split", "reference")


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
class CompositePrecisionPartitionSpec:
    """Static K-group partition used by composite-precision GEMM."""

    group_axis: Literal["k"] = "k"
    group_size: int = 128
    group_count: int = 1
    selected_groups: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        group_axis = str(self.group_axis).strip().lower()
        if group_axis != "k":
            raise XQTBackendError(
                f"CompositePrecisionPartitionSpec.group_axis must be 'k', got {self.group_axis}"
            )
        object.__setattr__(self, "group_axis", "k")
        if not isinstance(self.selected_groups, tuple):
            object.__setattr__(
                self,
                "selected_groups",
                tuple(int(item) for item in self.selected_groups),
            )
        if self.group_size <= 0:
            raise XQTBackendError("CompositePrecisionPartitionSpec.group_size must be positive")
        if self.group_count <= 0:
            raise XQTBackendError("CompositePrecisionPartitionSpec.group_count must be positive")
        seen: set[int] = set()
        for index in self.selected_groups:
            if index in seen:
                raise XQTBackendError(
                    f"CompositePrecisionPartitionSpec.selected_groups contains duplicate group index {index}"
                )
            if index < 0 or index >= self.group_count:
                raise XQTBackendError(
                    f"CompositePrecisionPartitionSpec.selected_groups index {index} "
                    f"out of range for group_count={self.group_count}"
                )
            seen.add(index)

    @property
    def partition_group_count(self) -> int:
        return len(self.selected_groups)

    @property
    def residual_group_count(self) -> int:
        return self.group_count - self.partition_group_count

    def partition_map(
        self,
        *,
        selected_branch: str,
        residual_branch: str,
    ) -> list[dict[str, Any]]:
        selected = set(self.selected_groups)
        return [
            {
                "group_index": group_index,
                "branch": selected_branch if group_index in selected else residual_branch,
                "selected": group_index in selected,
            }
            for group_index in range(self.group_count)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_axis": self.group_axis,
            "group_size": self.group_size,
            "group_count": self.group_count,
            "selected_groups": list(self.selected_groups),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompositePrecisionPartitionSpec":
        return cls(
            group_axis=str(payload.get("group_axis", "k")),
            group_size=int(payload.get("group_size", 128)),
            group_count=int(payload.get("group_count", 1)),
            selected_groups=tuple(
                int(item) for item in payload.get("selected_groups", ())
            ),
        )


@dataclass(frozen=True)
class CompositePrecisionBranchSpec:
    """One branch inside a composite-precision GEMM partition."""

    name: str
    format: str
    weight_format: str | None = None
    scale_format: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise XQTBackendError("CompositePrecisionBranchSpec.name must not be empty")
        if not str(self.format).strip():
            raise XQTBackendError("CompositePrecisionBranchSpec.format must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "format": self.format,
        }
        if self.weight_format is not None:
            payload["weight_format"] = self.weight_format
        if self.scale_format is not None:
            payload["scale_format"] = self.scale_format
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompositePrecisionBranchSpec":
        return cls(
            name=str(payload.get("name", "")),
            format=str(payload.get("format", "")),
            weight_format=(
                str(payload["weight_format"])
                if payload.get("weight_format") is not None
                else None
            ),
            scale_format=(
                str(payload["scale_format"])
                if payload.get("scale_format") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CompositePrecisionGemmSpec:
    """Canonical composite-precision contract for inference-side Linear/GEMM."""

    partition: CompositePrecisionPartitionSpec
    selected_branch: CompositePrecisionBranchSpec
    residual_branch: CompositePrecisionBranchSpec
    preferred_mode: CompositeExecutionMode = "split"
    allowed_modes: tuple[CompositeExecutionMode, ...] = ("split", "reference")
    accumulation_dtype: str = "fp32"
    fallback: Literal["reject"] = "reject"

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_modes, tuple):
            object.__setattr__(
                self,
                "allowed_modes",
                tuple(
                    _canonical_composite_execution_mode(str(item))
                    for item in self.allowed_modes
                ),
            )
        preferred_mode = _canonical_composite_execution_mode(str(self.preferred_mode))
        object.__setattr__(self, "preferred_mode", preferred_mode)
        if not self.allowed_modes:
            raise XQTBackendError("CompositePrecisionGemmSpec.allowed_modes must not be empty")
        if len(set(self.allowed_modes)) != len(self.allowed_modes):
            raise XQTBackendError(
                "CompositePrecisionGemmSpec.allowed_modes must not contain duplicates"
            )
        if preferred_mode not in self.allowed_modes:
            allowed = ", ".join(self.allowed_modes)
            raise XQTBackendError(
                "CompositePrecisionGemmSpec.preferred_mode must be one of "
                f"allowed_modes. preferred_mode={preferred_mode}, allowed_modes={allowed}"
            )
        if self.selected_branch.name == self.residual_branch.name:
            raise XQTBackendError(
                "CompositePrecisionGemmSpec requires distinct branch names for "
                "selected_branch and residual_branch"
            )
        object.__setattr__(
            self,
            "accumulation_dtype",
            PrecisionPolicy.canonical_name(self.accumulation_dtype),
        )
        if self.fallback != "reject":
            raise XQTBackendError(
                f"CompositePrecisionGemmSpec.fallback only supports 'reject', got {self.fallback}"
            )

    def branch_formats(self) -> dict[str, str]:
        return {
            self.selected_branch.name: self.selected_branch.format,
            self.residual_branch.name: self.residual_branch.format,
        }

    def resolve_runtime_plan(
        self,
        *,
        backend: str | None = None,
        available_modes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        requested_mode = self.preferred_mode
        available = tuple(
            _canonical_composite_execution_mode(str(item))
            for item in (
                available_modes
                if available_modes is not None
                else _default_composite_available_modes(backend=backend)
            )
        )
        actual_mode: CompositeExecutionMode = "reject"
        fallback_reason: str | None = None
        if requested_mode in available:
            actual_mode = requested_mode
        else:
            for candidate in self.allowed_modes:
                if candidate in available:
                    actual_mode = candidate
                    fallback_reason = f"requested_mode_{requested_mode}_unavailable"
                    break
            if actual_mode == "reject":
                fallback_reason = f"requested_mode_{requested_mode}_rejected"
        kernel_count = 0
        if actual_mode == "fused":
            kernel_count = 1
        elif actual_mode in {"split", "reference"}:
            kernel_count = 2
        return {
            "composite_precision": True,
            "requested_mode": requested_mode,
            "actual_mode": actual_mode,
            "backend": str(backend or "unknown"),
            "partition_group_count": self.partition.partition_group_count,
            "residual_group_count": self.partition.residual_group_count,
            "partition_map": self.partition.partition_map(
                selected_branch=self.selected_branch.name,
                residual_branch=self.residual_branch.name,
            ),
            "branch_formats": self.branch_formats(),
            "accumulation_dtype": self.accumulation_dtype,
            "kernel_count": kernel_count,
            "workspace_bytes": 0,
            "fallback_reason": fallback_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition.to_dict(),
            "selected_branch": self.selected_branch.to_dict(),
            "residual_branch": self.residual_branch.to_dict(),
            "preferred_mode": self.preferred_mode,
            "allowed_modes": list(self.allowed_modes),
            "accumulation_dtype": self.accumulation_dtype,
            "fallback": self.fallback,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CompositePrecisionGemmSpec":
        partition = payload.get("partition", {})
        selected_branch = payload.get("selected_branch", {})
        residual_branch = payload.get("residual_branch", {})
        if not isinstance(partition, Mapping):
            raise XQTBackendError("CompositePrecisionGemmSpec.partition must be a mapping")
        if not isinstance(selected_branch, Mapping):
            raise XQTBackendError(
                "CompositePrecisionGemmSpec.selected_branch must be a mapping"
            )
        if not isinstance(residual_branch, Mapping):
            raise XQTBackendError(
                "CompositePrecisionGemmSpec.residual_branch must be a mapping"
            )
        return cls(
            partition=CompositePrecisionPartitionSpec.from_mapping(partition),
            selected_branch=CompositePrecisionBranchSpec.from_mapping(selected_branch),
            residual_branch=CompositePrecisionBranchSpec.from_mapping(residual_branch),
            preferred_mode=_canonical_composite_execution_mode(
                str(payload.get("preferred_mode", "split"))
            ),
            allowed_modes=tuple(
                _canonical_composite_execution_mode(str(item))
                for item in payload.get("allowed_modes", ("split", "reference"))
            ),
            accumulation_dtype=str(payload.get("accumulation_dtype", "fp32")),
            fallback=str(payload.get("fallback", "reject")),
        )


def coerce_composite_precision_gemm_spec(
    value: CompositePrecisionGemmSpec | Mapping[str, Any] | None,
) -> CompositePrecisionGemmSpec | None:
    if value is None:
        return None
    if isinstance(value, CompositePrecisionGemmSpec):
        return value
    if not isinstance(value, Mapping):
        raise XQTBackendError(
            "composite_gemm must be a CompositePrecisionGemmSpec or mapping"
        )
    return CompositePrecisionGemmSpec.from_mapping(value)


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
    "CompositeExecutionMode",
    "CompositePrecisionBranchSpec",
    "CompositePrecisionGemmSpec",
    "CompositePrecisionPartitionSpec",
    "FeedForwardPrecisionPolicy",
    "FusionIntent",
    "ModuleContract",
    "OperatorContract",
    "OperatorKind",
    "PrecisionPolicy",
    "TensorStorageSpec",
]
