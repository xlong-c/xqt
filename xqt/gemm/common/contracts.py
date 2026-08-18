"""Public contracts owned by the XQT GEMM module.

The contracts in this module describe logical tensors and quantization metadata.
They deliberately do not import CUDA, CUTLASS, Triton, or model code.  Backend
implementations may add an execution layout, but they must not change the
logical ``A[M, K] @ W[N, K].T -> Y[M, N]`` convention.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


_DTYPES = frozenset(
    {
        "fp32",
        "int32",
        "fp16",
        "bf16",
        "int8",
        "int4",
        "int3",
        "int2",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp4",
        "mxfp8",
        "mxfp6",
        "mxfp4",
        "nvfp4",
        "codebook",
        "none",
    }
)
_GRANULARITIES = frozenset({"none", "per_tensor", "per_channel", "per_token", "groupwise", "blockwise"})
_SCALE_SOURCES = frozenset({"none", "weight_offline", "weight_load_time", "activation_static", "activation_dynamic"})
_OPS = frozenset({"dense", "batched", "grouped"})
_PHASES = frozenset({"generic", "prefill", "decode"})
_ACTIVATIONS = frozenset({"none", "relu", "gelu", "silu"})


def _shape(value: Sequence[int], *, field_name: str, rank: int | None = None) -> tuple[int, ...]:
    """Validate and normalize a shape without silently fixing it."""

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of ints")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"{field_name} entries must be positive ints")
        try:
            integer = int(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} entries must be positive ints") from exc
        if integer <= 0:
            raise ValueError(f"{field_name} entries must be positive, got {integer}")
        result.append(integer)
    if rank is not None and len(result) != rank:
        raise ValueError(f"{field_name} must have rank {rank}, got {len(result)}")
    return tuple(result)


def _tuple_strings(value: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} cannot contain empty strings")
    return result


@dataclass(frozen=True, slots=True)
class GemmProblem:
    """Logical GEMM problem.

    ``weight`` is always the logical PyTorch Linear layout ``[N, K]``.  The
    problem stores dimensions explicitly so a dispatcher never has to infer
    prefill/decode semantics from a scheduler or from a tensor's batch shape.
    """

    m: int
    n: int
    k: int
    op: str = "dense"
    phase: str = "generic"
    batch: int = 1
    group_count: int = 1
    device: str = "cpu"
    sm: int | None = None
    cuda_graph: bool = False

    def __post_init__(self) -> None:
        for name in ("n", "k", "batch", "group_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"GemmProblem.{name} must be a positive int")
            object.__setattr__(self, name, int(value))
        # A zero-row expert is a real routing outcome for grouped MoE GEMM.
        # N/K and all execution-count fields stay positive; only M can be zero.
        if isinstance(self.m, bool) or int(self.m) != self.m or int(self.m) < 0:
            raise ValueError("GemmProblem.m must be a non-negative int")
        object.__setattr__(self, "m", int(self.m))
        if self.op not in _OPS:
            raise ValueError(f"GemmProblem.op must be one of {sorted(_OPS)}, got {self.op!r}")
        if self.phase not in _PHASES:
            raise ValueError(
                f"GemmProblem.phase must be one of {sorted(_PHASES)}, got {self.phase!r}"
            )
        if self.op == "grouped" and self.group_count < 1:
            raise ValueError("grouped GEMM requires group_count >= 1")
        device = str(self.device)
        if not device:
            raise ValueError("GemmProblem.device must be a non-empty string")
        object.__setattr__(self, "device", device)
        if self.sm is not None:
            if isinstance(self.sm, bool) or int(self.sm) != self.sm or int(self.sm) < 0:
                raise ValueError("GemmProblem.sm must be a non-negative int or None")
            object.__setattr__(self, "sm", int(self.sm))
        if not isinstance(self.cuda_graph, bool):
            raise TypeError("GemmProblem.cuda_graph must be bool")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.m, self.n, self.k

    @property
    def arch(self) -> str:
        return "unknown" if self.sm is None else f"sm_{self.sm}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "op": self.op,
            "phase": self.phase,
            "batch": self.batch,
            "group_count": self.group_count,
            "device": self.device,
            "sm": self.sm,
            "cuda_graph": self.cuda_graph,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GemmProblem":
        if not isinstance(payload, Mapping):
            raise TypeError("GemmProblem.from_dict expects a mapping")
        return cls(
            m=int(payload["m"]),
            n=int(payload["n"]),
            k=int(payload["k"]),
            op=str(payload.get("op", "dense")),
            phase=str(payload.get("phase", "generic")),
            batch=int(payload.get("batch", 1)),
            group_count=int(payload.get("group_count", 1)),
            device=str(payload.get("device", "cpu")),
            sm=None if payload.get("sm") is None else int(payload["sm"]),
            cuda_graph=bool(payload.get("cuda_graph", False)),
        )


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """GEMM-side quantization contract.

    ``weight_scale_source`` and ``activation_scale_source`` intentionally use
    distinct vocabularies.  A static activation scale is not interchangeable
    with an online per-token reduction, and a load-time weight quantization is
    not a runtime dynamic activation path.
    """

    weight_dtype: str = "fp16"
    activation_dtype: str = "fp16"
    compute_dtype: str = "fp32"
    accum_dtype: str = "fp32"
    output_dtype: str = "fp16"
    weight_granularity: str = "per_tensor"
    activation_granularity: str = "per_tensor"
    group_axis: str = "k"
    group_size: int | None = None
    symmetric: bool = True
    weight_zero_point: bool = False
    activation_zero_point: bool = False
    weight_scale_source: str = "none"
    activation_scale_source: str = "none"
    storage_layout: str = "canonical"
    pack_version: str = "canonical-v1"

    def __post_init__(self) -> None:
        for field_name in (
            "weight_dtype",
            "activation_dtype",
            "compute_dtype",
            "accum_dtype",
            "output_dtype",
        ):
            value = str(getattr(self, field_name))
            if value not in _DTYPES:
                raise ValueError(f"QuantSpec.{field_name} unsupported dtype: {value!r}")
            object.__setattr__(self, field_name, value)
        for field_name in ("weight_granularity", "activation_granularity"):
            value = str(getattr(self, field_name))
            if value not in _GRANULARITIES:
                raise ValueError(f"QuantSpec.{field_name} unsupported value: {value!r}")
            object.__setattr__(self, field_name, value)
        if self.group_axis not in {"k", "n", "m"}:
            raise ValueError("QuantSpec.group_axis must be 'k', 'n', or 'm'")
        if self.weight_granularity in {"groupwise", "blockwise"}:
            if self.group_size is None or int(self.group_size) <= 0:
                raise ValueError("groupwise/blockwise weight quantization requires group_size > 0")
            object.__setattr__(self, "group_size", int(self.group_size))
        elif self.group_size is not None:
            raise ValueError("group_size is only valid for groupwise/blockwise weights")
        for field_name in ("symmetric", "weight_zero_point", "activation_zero_point"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"QuantSpec.{field_name} must be bool")
        if self.weight_dtype in {"fp8_e4m3", "fp8_e5m2"}:
            if not self.symmetric:
                raise ValueError("FP8 weights are signed and cannot declare symmetric=False")
            if self.weight_zero_point:
                raise ValueError("FP8 weights do not support integer zero points")
        if self.activation_dtype in {"fp8_e4m3", "fp8_e5m2"} and self.activation_zero_point:
            raise ValueError("FP8 activations do not support integer zero points")
        for field_name in ("weight_scale_source", "activation_scale_source"):
            value = str(getattr(self, field_name))
            if value not in _SCALE_SOURCES:
                raise ValueError(f"QuantSpec.{field_name} unsupported value: {value!r}")
            object.__setattr__(self, field_name, value)
        if self.weight_dtype in {
            "int4",
            "int8",
            "int3",
            "int2",
            "fp8_e4m3",
            "fp8_e5m2",
            "fp4",
            "mxfp8",
            "mxfp6",
            "mxfp4",
            "nvfp4",
            "codebook",
        }:
            if self.weight_scale_source == "none":
                raise ValueError("quantized weights require weight_scale_source")
        elif self.weight_scale_source != "none":
            raise ValueError("dense weights cannot declare a weight scale source")
        if self.activation_dtype in {"int8", "fp8_e4m3", "fp8_e5m2"}:
            if self.activation_scale_source not in {"activation_static", "activation_dynamic"}:
                raise ValueError(
                    "QuantSpec.activation_scale_source must be activation_static "
                    "or activation_dynamic for quantized activations"
                )
        elif self.activation_scale_source != "none":
            raise ValueError("non-quantized activations cannot declare an activation scale source")
        for field_name in ("storage_layout", "pack_version"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"QuantSpec.{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)

    @property
    def scale_mode(self) -> str:
        return f"w:{self.weight_granularity}/a:{self.activation_granularity}"

    @property
    def is_weight_only(self) -> bool:
        return self.activation_dtype not in {"int8", "fp8_e4m3", "fp8_e5m2"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "compute_dtype": self.compute_dtype,
            "accum_dtype": self.accum_dtype,
            "output_dtype": self.output_dtype,
            "weight_granularity": self.weight_granularity,
            "activation_granularity": self.activation_granularity,
            "group_axis": self.group_axis,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "weight_zero_point": self.weight_zero_point,
            "activation_zero_point": self.activation_zero_point,
            "weight_scale_source": self.weight_scale_source,
            "activation_scale_source": self.activation_scale_source,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuantSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("QuantSpec.from_dict expects a mapping")
        return cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class EpilogueSpec:
    """Post-accumulation operations that are part of a GEMM contract."""

    activation: str = "none"
    has_bias: bool = False
    has_residual: bool = False
    output_dtype: str = "fp16"

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"EpilogueSpec.activation must be one of {sorted(_ACTIVATIONS)}")
        if not isinstance(self.has_bias, bool) or not isinstance(self.has_residual, bool):
            raise TypeError("EpilogueSpec.has_bias and has_residual must be bool")
        if self.output_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("EpilogueSpec.output_dtype must be fp32, fp16, or bf16")

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation": self.activation,
            "has_bias": self.has_bias,
            "has_residual": self.has_residual,
            "output_dtype": self.output_dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpilogueSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("EpilogueSpec.from_dict expects a mapping")
        return cls(
            activation=str(payload.get("activation", "none")),
            has_bias=bool(payload.get("has_bias", False)),
            has_residual=bool(payload.get("has_residual", False)),
            output_dtype=str(payload.get("output_dtype", "fp16")),
        )


@dataclass(frozen=True, slots=True)
class GemmSpec:
    """Complete logical GEMM contract consumed by reference or native backends."""

    problem: GemmProblem
    quant: QuantSpec = field(default_factory=QuantSpec)
    epilogue: EpilogueSpec = field(default_factory=EpilogueSpec)

    def __post_init__(self) -> None:
        if self.epilogue.output_dtype != self.quant.output_dtype:
            raise ValueError("GemmSpec epilogue.output_dtype must match quant.output_dtype")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "quant": self.quant.to_dict(),
            "epilogue": self.epilogue.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GemmSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("GemmSpec.from_dict expects a mapping")
        return cls(
            problem=GemmProblem.from_dict(payload["problem"]),
            quant=QuantSpec.from_dict(payload.get("quant", {})),
            epilogue=EpilogueSpec.from_dict(payload.get("epilogue", {})),
        )


@dataclass(frozen=True, slots=True)
class PackedWeightMetadata:
    """Metadata that makes a packed weight reversible and dispatchable."""

    logical_shape: tuple[int, int]
    storage_layout: str
    pack_version: str
    weight_dtype: str
    padded_k: int
    group_size: int | None = None
    packed_bits: int | None = None
    nibble_order: str | None = None
    nibble_signed: bool = True
    local_shape: tuple[int, int] | None = None
    shard_axis: int | None = None
    global_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        logical = _shape(self.logical_shape, field_name="PackedWeightMetadata.logical_shape", rank=2)
        object.__setattr__(self, "logical_shape", logical)
        if int(self.padded_k) < logical[1]:
            raise ValueError("padded_k cannot be smaller than logical K")
        object.__setattr__(self, "padded_k", int(self.padded_k))
        if self.weight_dtype not in _DTYPES:
            raise ValueError(f"unsupported packed weight dtype: {self.weight_dtype!r}")
        if self.packed_bits not in {None, 2, 3, 4, 8}:
            raise ValueError("packed_bits must be None, 2, 3, 4, or 8")
        if self.packed_bits == 4 and self.nibble_order not in {"low_high", "high_low"}:
            raise ValueError("4-bit weights require nibble_order low_high or high_low")
        if self.packed_bits in {2, 3} and self.nibble_order is not None:
            raise ValueError("2/3-bit weights do not use nibble_order")
        if not isinstance(self.nibble_signed, bool):
            raise TypeError("nibble_signed must be bool")
        if self.local_shape is not None:
            object.__setattr__(self, "local_shape", _shape(self.local_shape, field_name="local_shape", rank=2))
        if self.shard_axis is not None and self.shard_axis not in {0, 1}:
            raise ValueError("shard_axis must be 0, 1, or None")
        if self.global_scale is not None:
            if not isinstance(self.global_scale, torch.Tensor) or self.global_scale.numel() != 1:
                raise ValueError("global_scale must be a scalar tensor")
            if not bool(torch.isfinite(self.global_scale).all()) or bool((self.global_scale <= 0).any()):
                raise ValueError("global_scale must be finite and positive")
            object.__setattr__(self, "global_scale", self.global_scale.detach().to(dtype=torch.float32).reshape(()))

    @property
    def padding_ratio(self) -> float:
        return float(self.padded_k - self.logical_shape[1]) / float(self.logical_shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_shape": list(self.logical_shape),
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
            "weight_dtype": self.weight_dtype,
            "padded_k": self.padded_k,
            "group_size": self.group_size,
            "packed_bits": self.packed_bits,
            "nibble_order": self.nibble_order,
            "nibble_signed": self.nibble_signed,
            "local_shape": None if self.local_shape is None else list(self.local_shape),
            "shard_axis": self.shard_axis,
            "padding_ratio": self.padding_ratio,
            "global_scale_present": self.global_scale is not None,
            "global_scale_shape": [] if self.global_scale is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PackedWeightMetadata":
        if not isinstance(payload, Mapping):
            raise TypeError("PackedWeightMetadata.from_dict expects a mapping")
        return cls(
            logical_shape=tuple(payload["logical_shape"]),
            storage_layout=str(payload["storage_layout"]),
            pack_version=str(payload["pack_version"]),
            weight_dtype=str(payload["weight_dtype"]),
            padded_k=int(payload["padded_k"]),
            group_size=None if payload.get("group_size") is None else int(payload["group_size"]),
            packed_bits=None if payload.get("packed_bits") is None else int(payload["packed_bits"]),
            nibble_order=None if payload.get("nibble_order") is None else str(payload["nibble_order"]),
            nibble_signed=bool(payload.get("nibble_signed", True)),
            local_shape=None if payload.get("local_shape") is None else tuple(payload["local_shape"]),
            shard_axis=None if payload.get("shard_axis") is None else int(payload["shard_axis"]),
        )


@dataclass(frozen=True, slots=True)
class PackedWeight:
    """Tensor payload plus enough metadata for reference/native decoding."""

    qweight: Any
    scales: Any | None
    zero_points: Any | None
    metadata: PackedWeightMetadata
    canonical_qweight: Any | None = None
    global_scale: torch.Tensor | None = None
    sparse_mask: torch.Tensor | None = None
    codebook: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.qweight, "shape"):
            raise TypeError("PackedWeight.qweight must be a tensor-like object")
        if self.scales is not None and not hasattr(self.scales, "shape"):
            raise TypeError("PackedWeight.scales must be tensor-like when provided")
        if self.zero_points is not None and not hasattr(self.zero_points, "shape"):
            raise TypeError("PackedWeight.zero_points must be tensor-like when provided")
        if self.canonical_qweight is not None and not hasattr(self.canonical_qweight, "shape"):
            raise TypeError("PackedWeight.canonical_qweight must be tensor-like when provided")
        if self.sparse_mask is not None:
            if not isinstance(self.sparse_mask, torch.Tensor) or self.sparse_mask.ndim != 2:
                raise TypeError("PackedWeight.sparse_mask must be a rank-2 bool tensor")
            expected_mask = self.metadata.logical_shape
            if tuple(self.sparse_mask.shape) != expected_mask:
                raise ValueError(
                    "PackedWeight.sparse_mask must match logical_shape "
                    f"{expected_mask}, got {tuple(self.sparse_mask.shape)}"
                )
            if self.sparse_mask.dtype != torch.bool:
                raise TypeError("PackedWeight.sparse_mask must use torch.bool")
        if self.codebook is not None:
            if not isinstance(self.codebook, torch.Tensor) or self.codebook.ndim != 2:
                raise TypeError("PackedWeight.codebook must be a rank-2 tensor")
            if self.codebook.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
                raise TypeError("PackedWeight.codebook must use a floating dtype")

    def to_metadata_dict(self) -> dict[str, Any]:
        return self.metadata.to_dict()


@dataclass(frozen=True, slots=True)
class GroupedGemmProblem:
    """Logical problems for one grouped launch, one item per expert/group."""

    problems: tuple[GemmProblem, ...]
    m_offsets: tuple[int, ...] | None = None
    output_rows: tuple[int, ...] | None = None
    cuda_graph: bool = False

    def __post_init__(self) -> None:
        if not self.problems:
            raise ValueError("GroupedGemmProblem requires at least one group")
        if any(problem.op == "grouped" for problem in self.problems):
            raise ValueError("individual grouped problems must use op='dense' or 'batched'")
        first = self.problems[0]
        if any(problem.n != first.n or problem.k != first.k for problem in self.problems[1:]):
            raise ValueError("grouped GEMM currently requires common N and K")
        if self.m_offsets is not None:
            offsets = tuple(int(item) for item in self.m_offsets)
            if len(offsets) != len(self.problems) + 1:
                raise ValueError("m_offsets must have group_count + 1 entries")
            if offsets[0] != 0 or any(right < left for left, right in zip(offsets, offsets[1:])):
                raise ValueError("m_offsets must be monotonic and start at zero")
            expected_offsets = [0]
            for problem in self.problems:
                expected_offsets.append(expected_offsets[-1] + problem.m)
            if offsets != tuple(expected_offsets):
                raise ValueError("m_offsets must match cumulative expert M offsets")
            object.__setattr__(self, "m_offsets", offsets)
        total_m = sum(problem.m for problem in self.problems)
        if self.output_rows is not None:
            rows = tuple(int(item) for item in self.output_rows)
            if len(rows) != total_m:
                raise ValueError("output_rows must contain one destination row per grouped input row")
            if any(item < 0 for item in rows):
                raise ValueError("output_rows entries must be non-negative")
            if set(rows) != set(range(total_m)):
                raise ValueError("output_rows must be a permutation of [0, total_m)")
            object.__setattr__(self, "output_rows", rows)
        if not isinstance(self.cuda_graph, bool):
            raise TypeError("GroupedGemmProblem.cuda_graph must be bool")

    @property
    def group_count(self) -> int:
        return len(self.problems)

    @property
    def n(self) -> int:
        return self.problems[0].n

    @property
    def k(self) -> int:
        return self.problems[0].k

    @property
    def total_m(self) -> int:
        """Return the packed activation row count, including no rows for empty experts."""

        return sum(problem.m for problem in self.problems)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problems": [problem.to_dict() for problem in self.problems],
            "m_offsets": None if self.m_offsets is None else list(self.m_offsets),
            "output_rows": None if self.output_rows is None else list(self.output_rows),
            "cuda_graph": self.cuda_graph,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroupedGemmProblem":
        if not isinstance(payload, Mapping):
            raise TypeError("GroupedGemmProblem.from_dict expects a mapping")
        return cls(
            problems=tuple(GemmProblem.from_dict(item) for item in payload["problems"]),
            m_offsets=None if payload.get("m_offsets") is None else tuple(payload["m_offsets"]),
            output_rows=None
            if payload.get("output_rows") is None
            else tuple(payload["output_rows"]),
            cuda_graph=bool(payload.get("cuda_graph", False)),
        )


__all__ = [
    "EpilogueSpec",
    "GemmProblem",
    "GemmSpec",
    "GroupedGemmProblem",
    "PackedWeight",
    "PackedWeightMetadata",
    "QuantSpec",
]
