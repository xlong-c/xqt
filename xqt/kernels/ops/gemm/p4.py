"""P4 GEMM contracts and reference paths.

This module owns the low-bit value/scale ABI used by the P4 work items.  It is
CPU-safe and deliberately keeps native promotion out of the reference code:
the latter is useful for correctness gates, MoE shape coverage, and report
generation even when a target architecture is unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    GroupedGemmProblem,
    PackedWeight,
    QuantSpec,
)
from .fp8 import FP8QuantizedTensor, quantize_fp8
from .layout import build_packed_weight, pack_int4_signed
from .quantize import Int8ActivationQuantization, quantize_int8_activation
from .reference import (
    dequantize_weight_reference,
    reference_gemm,
    reference_packed_grouped_gemm,
)


FP4_E2M1_CODEBOOK: tuple[float, ...] = (
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
)
FP4_FORMATS: tuple[str, ...] = ("fp4", "mxfp4", "nvfp4")
W4A8_ACTIVATION_FORMATS: tuple[str, ...] = ("int8", "fp8_e4m3", "fp8_e5m2")


def _positive_scalar(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    result = value.to(dtype=torch.float32).reshape(())
    if not bool(torch.isfinite(result)) or bool(result <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _canonical_fp4_format(format_name: str) -> str:
    value = str(format_name).lower().strip()
    aliases = {"fp4_e2m1": "fp4", "fp4_e2m1fn": "fp4"}
    value = aliases.get(value, value)
    if value not in FP4_FORMATS:
        raise ValueError(f"unsupported FP4 format {format_name!r}")
    return value


def _group_count(columns: int, group_size: int) -> int:
    if int(columns) <= 0 or int(group_size) <= 0:
        raise ValueError("columns and group_size must be positive")
    return (int(columns) + int(group_size) - 1) // int(group_size)


def _fp4_codes(values: torch.Tensor) -> torch.Tensor:
    """Quantize values already normalized to the E2M1 range."""

    thresholds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    absolute = values.to(torch.float32).abs()
    codes = torch.zeros_like(absolute, dtype=torch.uint8)
    for threshold in thresholds:
        codes = codes + (absolute > threshold).to(torch.uint8)
    return codes + torch.signbit(values).to(torch.uint8) * 8


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2 or codes.shape[1] % 2 != 0:
        raise ValueError("FP4 codes must be rank-2 with an even K")
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous()


def _unpack_fp4_codes(storage: torch.Tensor, *, padded_k: int) -> torch.Tensor:
    if storage.ndim != 2 or storage.dtype != torch.uint8:
        raise TypeError("FP4 storage must be uint8 rank-2")
    expected = (int(padded_k) + 1) // 2
    if int(storage.shape[1]) != expected:
        raise ValueError(f"FP4 storage must have {expected} columns, got {storage.shape[1]}")
    low = storage & 0x0F
    high = (storage >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(storage.shape[0], -1)[:, :padded_k]


def _quantize_mxfp_scale(raw: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    safe = raw.to(torch.float32).clamp_min(tiny)
    return torch.exp2(torch.ceil(torch.log2(safe))).where(raw > 0, torch.ones_like(raw))


def _quantize_nvfp4_scale(raw: torch.Tensor) -> torch.Tensor:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        return raw.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    try:
        encoded = raw.to(torch.float32).clamp_max(float(torch.finfo(dtype).max)).to(dtype)
        return encoded.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)
    except (RuntimeError, TypeError):
        return raw.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)


@dataclass(frozen=True, slots=True)
class FP4QuantizedTensor:
    """Packed E2M1 values and explicit per-group scales."""

    storage: torch.Tensor
    scale: torch.Tensor
    format_name: str
    group_size: int
    logical_shape: tuple[int, int]
    padded_k: int
    source: str = "weight_offline"
    global_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        format_name = _canonical_fp4_format(self.format_name)
        object.__setattr__(self, "format_name", format_name)
        if len(self.logical_shape) != 2:
            raise ValueError("FP4 logical_shape must be a rank-2 tuple")
        if int(self.group_size) <= 0:
            raise ValueError("FP4 group_size must be positive")
        n, k = (int(self.logical_shape[0]), int(self.logical_shape[1]))
        if n <= 0 or k <= 0 or int(self.padded_k) < k:
            raise ValueError("FP4 logical_shape/padded_k are invalid")
        if int(self.padded_k) % int(self.group_size) != 0:
            raise ValueError("FP4 padded_k must be divisible by group_size")
        expected_storage = (n, (int(self.padded_k) + 1) // 2)
        if self.storage.dtype != torch.uint8 or tuple(self.storage.shape) != expected_storage:
            raise ValueError(f"FP4 storage must be uint8 with shape {expected_storage}")
        expected_scale = (n, _group_count(int(self.padded_k), int(self.group_size)))
        if tuple(self.scale.shape) != expected_scale:
            raise ValueError(f"FP4 scale must have shape {expected_scale}, got {tuple(self.scale.shape)}")
        if not bool(torch.isfinite(self.scale.to(torch.float32)).all()) or bool((self.scale <= 0).any()):
            raise ValueError("FP4 scales must be finite and positive")
        if format_name == "nvfp4":
            object.__setattr__(self, "global_scale", _positive_scalar(self.global_scale, name="NVFP4 global_scale"))
        elif self.global_scale is not None:
            raise ValueError("global_scale is only valid for NVFP4")

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        codes = _unpack_fp4_codes(self.storage, padded_k=self.padded_k)
        codebook = torch.tensor(FP4_E2M1_CODEBOOK, device=self.storage.device, dtype=torch.float32)
        values = codebook[codes.long()]
        expanded = self.scale.to(device=values.device, dtype=torch.float32).repeat_interleave(
            int(self.group_size), dim=1
        )[:, : self.padded_k]
        if self.format_name == "nvfp4":
            assert self.global_scale is not None
            expanded = expanded / self.global_scale.to(device=values.device)
        return (values * expanded)[:, : self.logical_shape[1]].to(dtype=dtype)

    def to_packed_weight(
        self,
        *,
        activation_dtype: str = "fp16",
        output_dtype: str = "fp16",
        pack_version: str = "xqt-fp4-v1",
    ) -> PackedWeight:
        spec = QuantSpec(
            weight_dtype=self.format_name,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            weight_granularity="groupwise",
            group_size=int(self.group_size),
            weight_scale_source="weight_offline",
            activation_scale_source=(
                "activation_dynamic"
                if activation_dtype == "int8"
                else ("activation_static" if activation_dtype in W4A8_ACTIVATION_FORMATS else "none")
            ),
            storage_layout="xqt_fp4_nk_v1",
            pack_version=pack_version,
        )
        return build_packed_weight(
            self.storage,
            logical_shape=self.logical_shape,
            spec=spec,
            scales=self.scale,
            padded_k=self.padded_k,
            storage_layout="xqt_fp4_nk_v1",
            pack_version=pack_version,
            global_scale=self.global_scale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format_name,
            "storage_dtype": "uint8",
            "storage_layout": "xqt_fp4_nk_v1",
            "logical_shape": list(self.logical_shape),
            "padded_k": int(self.padded_k),
            "group_size": int(self.group_size),
            "scale_shape": list(self.scale.shape),
            "global_scale": None if self.global_scale is None else float(self.global_scale.item()),
            "source": self.source,
        }


def quantize_fp4_reference(
    values: torch.Tensor,
    *,
    format_name: str = "fp4",
    group_size: int | None = None,
    global_scale: torch.Tensor | None = None,
    source: str = "weight_offline",
) -> FP4QuantizedTensor:
    """Quantize a 2D tensor to FP4/MXFP4/NVFP4 reference storage."""

    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError("FP4 quantization expects a rank-2 tensor [rows, cols]")
    format_name = _canonical_fp4_format(format_name)
    default_group = {"fp4": 32, "mxfp4": 32, "nvfp4": 16}[format_name]
    normalized_group = default_group if group_size is None else int(group_size)
    if normalized_group <= 0:
        raise ValueError("FP4 group_size must be positive")
    if format_name == "mxfp4" and normalized_group != 32:
        raise ValueError("MXFP4 uses group_size=32")
    if format_name == "nvfp4" and normalized_group != 16:
        raise ValueError("NVFP4 uses group_size=16")
    if not bool(torch.isfinite(values.to(torch.float32)).all()):
        raise ValueError("FP4 quantization requires finite input values")
    rows, cols = (int(values.shape[0]), int(values.shape[1]))
    padded_k = _group_count(cols, normalized_group) * normalized_group
    if padded_k % 2:
        padded_k += normalized_group
    padded = F.pad(values.to(torch.float32), (0, padded_k - cols))
    grouped = padded.reshape(rows, padded_k // normalized_group, normalized_group)
    absmax = grouped.abs().amax(dim=-1)
    if format_name == "fp4":
        scale = absmax.clamp_min(1e-8) / 6.0
    elif format_name == "mxfp4":
        scale = _quantize_mxfp_scale(absmax / 6.0)
    else:
        resolved_global = _positive_scalar(global_scale, name="NVFP4 global_scale")
        scale = _quantize_nvfp4_scale(absmax * (resolved_global / 6.0))
        global_scale = resolved_global
    scale = scale.to(torch.float32).clamp_min(1e-8)
    if format_name == "nvfp4":
        assert global_scale is not None
        normalized = grouped * (global_scale / scale).unsqueeze(-1)
    else:
        normalized = grouped / scale.unsqueeze(-1)
    codes = _fp4_codes(normalized.reshape(rows, padded_k))
    storage = _pack_fp4_codes(codes)
    return FP4QuantizedTensor(
        storage=storage,
        scale=scale.contiguous(),
        format_name=format_name,
        group_size=normalized_group,
        logical_shape=(rows, cols),
        padded_k=padded_k,
        source=source,
        global_scale=global_scale,
    )


def pack_fp4_weight(
    values: torch.Tensor,
    *,
    format_name: str = "fp4",
    group_size: int | None = None,
    global_scale: torch.Tensor | None = None,
    activation_dtype: str = "fp16",
    output_dtype: str = "fp16",
) -> PackedWeight:
    """Quantize and materialize an FP4-family ``PackedWeight``."""

    return quantize_fp4_reference(
        values,
        format_name=format_name,
        group_size=group_size,
        global_scale=global_scale,
    ).to_packed_weight(activation_dtype=activation_dtype, output_dtype=output_dtype)


@dataclass(frozen=True, slots=True)
class W4A8Contract:
    """Explicit INT4 weight plus INT8/FP8 activation scale contract."""

    activation_dtype: str = "int8"
    weight_group_size: int = 32
    activation_granularity: str = "per_token"
    activation_scale_source: str = "activation_dynamic"
    output_dtype: str = "fp16"
    storage_layout: str = "xqt_int4_nk_v1"
    pack_version: str = "xqt-w4a8-v1"

    def __post_init__(self) -> None:
        if self.activation_dtype not in W4A8_ACTIVATION_FORMATS:
            raise ValueError(f"W4A8 activation_dtype must be one of {W4A8_ACTIVATION_FORMATS}")
        if int(self.weight_group_size) <= 0:
            raise ValueError("W4A8 weight_group_size must be positive")
        if self.activation_granularity not in {"per_tensor", "per_token", "blockwise"}:
            raise ValueError("W4A8 activation_granularity must be per_tensor, per_token, or blockwise")
        if self.activation_dtype == "int8" and self.activation_granularity == "blockwise":
            raise ValueError("INT8 W4A8 activation does not support blockwise scales")
        expected_source = "activation_dynamic" if self.activation_dtype == "int8" else "activation_static"
        if self.activation_scale_source != expected_source:
            raise ValueError(f"{self.activation_dtype} W4A8 requires {expected_source} scale source")
        if self.output_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("W4A8 output_dtype must be fp16, bf16, or fp32")

    @property
    def activation_scale_mode(self) -> str:
        return self.activation_granularity

    def quant_spec(self) -> QuantSpec:
        return QuantSpec(
            weight_dtype="int4",
            activation_dtype=self.activation_dtype,
            compute_dtype="fp32",
            accum_dtype="int32" if self.activation_dtype == "int8" else "fp32",
            output_dtype=self.output_dtype,
            weight_granularity="groupwise",
            activation_granularity=self.activation_granularity,
            group_size=int(self.weight_group_size),
            weight_scale_source="weight_offline",
            activation_scale_source=self.activation_scale_source,
            storage_layout=self.storage_layout,
            pack_version=self.pack_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "weight_dtype": "int4",
            "weight_granularity": "groupwise",
            "weight_group_size": int(self.weight_group_size),
            "activation_granularity": self.activation_granularity,
            "activation_scale_source": self.activation_scale_source,
            "output_dtype": self.output_dtype,
            "storage_layout": self.storage_layout,
            "pack_version": self.pack_version,
        }


W4A8Spec = W4A8Contract


def build_w4a8_gemm_spec(
    problem: GemmProblem,
    contract: W4A8Contract,
    *,
    has_bias: bool = False,
    has_residual: bool = False,
    activation: str = "none",
) -> GemmSpec:
    """Build a full ``GemmSpec`` from the compact W4A8 contract."""

    quant = contract.quant_spec()
    return GemmSpec(
        problem=problem,
        quant=quant,
        epilogue=EpilogueSpec(
            activation=activation,
            has_bias=has_bias,
            has_residual=has_residual,
            output_dtype=contract.output_dtype,
        ),
    )


def pack_w4a8_weight(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
    scales: torch.Tensor | None = None,
    contract: W4A8Contract | None = None,
    logical_shape: tuple[int, int] | None = None,
) -> PackedWeight:
    """Pack signed INT4 weights for the W4A8 canonical layout.

    Floating input is quantized with symmetric ``absmax / 7`` group scales;
    integer input is interpreted as already quantized values in ``[-8, 7]``.
    """

    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("W4A8 weight must be rank-2 [N,K]")
    resolved_contract = contract or W4A8Contract(weight_group_size=group_size)
    group_size = int(resolved_contract.weight_group_size)
    if weight.dtype == torch.uint8:
        if logical_shape is None:
            raise ValueError("packed W4A8 uint8 input requires logical_shape=(N,K)")
        n, k = (int(logical_shape[0]), int(logical_shape[1]))
    else:
        n, k = (int(weight.shape[0]), int(weight.shape[1]))
    padded_k = _group_count(k, group_size) * group_size
    if padded_k % 2:
        padded_k += group_size
    groups = _group_count(padded_k, group_size)

    def canonical_scales(value: torch.Tensor) -> torch.Tensor:
        """Normalize per-row shorthand scales to the explicit ``[N,G]`` ABI."""

        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=weight.device)
        value = value.to(device=weight.device, dtype=torch.float32)
        if value.ndim == 1 and int(value.numel()) == n:
            value = value.reshape(n, 1)
        if tuple(value.shape) == (n, 1):
            value = value.expand(n, groups)
        elif tuple(value.shape) != (n, groups):
            raise ValueError(
                "W4A8 scales must be [N], [N,1], or [N,groups], "
                f"got {tuple(value.shape)} for groups={groups}"
            )
        if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
            raise ValueError("W4A8 scales must be finite and positive")
        return value.contiguous()

    if weight.dtype == torch.uint8:
        if scales is None or int(weight.shape[1]) != (padded_k // 2):
            raise ValueError("packed W4A8 uint8 input requires matching scales and logical K")
        packed = weight
        resolved_scales = canonical_scales(scales)
    else:
        if weight.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
            codes = weight.to(torch.int16)
            if bool(((codes < -8) | (codes > 7)).any()):
                raise ValueError("W4A8 signed INT4 values must be in [-8, 7]")
            if scales is None:
                scales = torch.ones((n, groups), dtype=torch.float32, device=weight.device)
        else:
            values = weight.to(torch.float32)
            padded = F.pad(values, (0, padded_k - k))
            grouped = padded.reshape(n, _group_count(padded_k, group_size), group_size)
            derived = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
            scales = derived if scales is None else scales
            resolved = canonical_scales(scales)
            codes = torch.round(grouped / resolved.unsqueeze(-1)).clamp(-8, 7).reshape(n, padded_k).to(torch.int8)
        if scales is None:
            raise AssertionError("W4A8 scales must be resolved")
        resolved_scales = canonical_scales(scales)
        if tuple(codes.shape) == (n, k):
            codes = F.pad(codes, (0, padded_k - k))
        packed = pack_int4_signed(codes.to(torch.int8))
    spec = resolved_contract.quant_spec()
    return build_packed_weight(
        packed,
        logical_shape=(n, k),
        spec=spec,
        scales=resolved_scales,
        padded_k=padded_k,
        storage_layout=resolved_contract.storage_layout,
        pack_version=resolved_contract.pack_version,
    )


def quantize_w4a8_activation(
    activation: torch.Tensor,
    contract: W4A8Contract,
    *,
    scales: torch.Tensor | None = None,
) -> Int8ActivationQuantization | FP8QuantizedTensor:
    """Materialize activation values and scales under the W4A8 contract."""

    if contract.activation_dtype == "int8":
        return quantize_int8_activation(
            activation,
            granularity=contract.activation_granularity,
            source=contract.activation_scale_source,
            scale=scales,
        )
    block_k = contract.weight_group_size if contract.activation_granularity == "blockwise" else None
    return quantize_fp8(
        activation,
        format_name=contract.activation_dtype,
        granularity=contract.activation_granularity,
        role="activation",
        source=contract.activation_scale_source,
        scale=scales,
        block_k=block_k,
    )


def reference_w4a8_gemm(
    activation: torch.Tensor,
    weight: PackedWeight,
    *,
    contract: W4A8Contract | None = None,
    spec: GemmSpec | None = None,
    activation_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute the W4A8 dequantized reference with explicit scale semantics."""

    if not isinstance(weight, PackedWeight) or weight.metadata.weight_dtype != "int4":
        raise TypeError("W4A8 reference requires an INT4 PackedWeight")
    if spec is None:
        if contract is None:
            raise ValueError("reference_w4a8_gemm requires contract or spec")
        spec = build_w4a8_gemm_spec(
            GemmProblem(m=int(activation.shape[0]), n=weight.metadata.logical_shape[0], k=int(activation.shape[1])),
            contract,
            has_bias=bias is not None,
            has_residual=residual is not None,
        )
    if contract is not None and spec.quant != contract.quant_spec():
        raise ValueError("W4A8 contract and GemmSpec quantization fields disagree")
    quant = spec.quant
    runtime_activation = activation
    runtime_scales = activation_scales
    if activation.dtype not in {torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}:
        encoded = quantize_w4a8_activation(activation, contract or W4A8Contract(
            activation_dtype=quant.activation_dtype,
            weight_group_size=int(quant.group_size or 32),
            activation_granularity=quant.activation_granularity,
            activation_scale_source=quant.activation_scale_source,
            output_dtype=quant.output_dtype,
            storage_layout=quant.storage_layout,
            pack_version=quant.pack_version,
        ), scales=activation_scales)
        runtime_activation = encoded.values if isinstance(encoded, Int8ActivationQuantization) else encoded.storage
        runtime_scales = encoded.scales if isinstance(encoded, Int8ActivationQuantization) else encoded.scale
    return reference_gemm(
        runtime_activation,
        weight,
        spec=spec,
        weight_scales=weight.scales,
        activation_scales=runtime_scales,
        bias=bias,
        residual=residual,
    )


@dataclass(frozen=True, slots=True)
class W4A8GroupedReport:
    """Correctness/report metadata for a routed W4A8 reference launch."""

    group_count: int
    expert_rows: tuple[int, ...]
    empty_experts: tuple[int, ...]
    output_scatter: bool
    execution_mode: str = "reference_grouped"
    native: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "expert_rows": list(self.expert_rows),
            "empty_experts": list(self.empty_experts),
            "output_scatter": self.output_scatter,
            "execution_mode": self.execution_mode,
            "native": self.native,
        }


@dataclass(frozen=True, slots=True)
class W4A8GroupedResult:
    output: torch.Tensor
    report: W4A8GroupedReport


def reference_grouped_w4a8_gemm(
    grouped_problem: GroupedGemmProblem,
    activation: torch.Tensor,
    weights: Sequence[PackedWeight],
    *,
    contract: W4A8Contract,
    activation_scales: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    bias: Sequence[torch.Tensor | None] | None = None,
) -> W4A8GroupedResult:
    """Run a routed W4A8 reference including empty experts and scatter."""

    quant = contract.quant_spec()
    output = reference_packed_grouped_gemm(
        grouped_problem,
        activation,
        weights,
        quant_specs=quant,
        activation_scales=activation_scales,
        bias=bias,
    )
    rows = tuple(problem.m for problem in grouped_problem.problems)
    return W4A8GroupedResult(
        output=output,
        report=W4A8GroupedReport(
            group_count=grouped_problem.group_count,
            expert_rows=rows,
            empty_experts=tuple(index for index, row_count in enumerate(rows) if row_count == 0),
            output_scatter=grouped_problem.output_rows is not None,
        ),
    )


@dataclass(frozen=True, slots=True)
class SVDGemmContract:
    """GEMM-side contract for quantized main plus low-rank correction."""

    main_spec: GemmSpec
    rank: int
    outlier_channels: tuple[int, ...] = ()
    fusion_boundary: str = "separate_gemm"
    outlier_mode: str = "correction"

    def __post_init__(self) -> None:
        if int(self.rank) <= 0:
            raise ValueError("SVD rank must be positive")
        if self.fusion_boundary not in {"separate_gemm", "fuse_down", "fuse_up"}:
            raise ValueError("unsupported SVD fusion_boundary")
        if self.outlier_mode not in {"correction", "replacement"}:
            raise ValueError("unsupported SVD outlier_mode")
        channels = tuple(int(item) for item in self.outlier_channels)
        if len(set(channels)) != len(channels) or any(item < 0 or item >= self.main_spec.problem.k for item in channels):
            raise ValueError("SVD outlier_channels must be unique valid K indices")
        object.__setattr__(self, "outlier_channels", channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_spec": self.main_spec.to_dict(),
            "rank": int(self.rank),
            "outlier_channels": list(self.outlier_channels),
            "fusion_boundary": self.fusion_boundary,
            "outlier_mode": self.outlier_mode,
        }


@dataclass(frozen=True, slots=True)
class SVDGemmReport:
    """Branch-level numeric and fusion evidence for one SVD GEMM."""

    execution_mode: str
    fusion_boundary: str
    rank: int
    outlier_channels: tuple[int, ...]
    main_max_abs_error: float | None
    low_rank_max_abs_error: float | None
    outlier_max_abs_error: float | None
    merged_max_abs_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "fusion_boundary": self.fusion_boundary,
            "rank": self.rank,
            "outlier_channels": list(self.outlier_channels),
            "main_max_abs_error": self.main_max_abs_error,
            "low_rank_max_abs_error": self.low_rank_max_abs_error,
            "outlier_max_abs_error": self.outlier_max_abs_error,
            "merged_max_abs_error": self.merged_max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class SVDGemmResult:
    output: torch.Tensor
    main_output: torch.Tensor
    low_rank_output: torch.Tensor
    outlier_output: torch.Tensor
    report: SVDGemmReport


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor | None) -> float | None:
    if expected is None:
        return None
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError("SVD reference tensors must have matching shapes")
    return float((actual.to(torch.float32) - expected.to(torch.float32)).abs().max().item())


def reference_svd_dual_path(
    activation: torch.Tensor,
    main_weight: PackedWeight,
    *,
    contract: SVDGemmContract,
    low_rank_down: torch.Tensor,
    low_rank_up: torch.Tensor,
    main_weight_reference: torch.Tensor | None = None,
    outlier_weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> SVDGemmResult:
    """Run main quantized GEMM, low-rank GEMM, and optional outlier correction.

    ``outlier_weight`` stores exact columns in ``outlier_channels``.  In
    ``correction`` mode the quantized main columns are subtracted before the
    exact columns are added, preventing double counting.
    """

    spec = contract.main_spec
    if tuple(low_rank_down.shape) != (contract.rank, spec.problem.k):
        raise ValueError("low_rank_down must have shape [rank,K]")
    if tuple(low_rank_up.shape) != (spec.problem.n, contract.rank):
        raise ValueError("low_rank_up must have shape [N,rank]")
    if outlier_weight is not None:
        expected = (spec.problem.n, len(contract.outlier_channels))
        if tuple(outlier_weight.shape) != expected:
            raise ValueError(f"outlier_weight must have shape {expected}")
    main_output = reference_gemm(
        activation,
        main_weight,
        spec=spec,
        weight_scales=main_weight.scales,
        bias=bias,
    ).to(torch.float32)
    hidden = activation.to(torch.float32) @ low_rank_down.to(torch.float32).transpose(0, 1)
    low_rank_output = hidden @ low_rank_up.to(torch.float32).transpose(0, 1)
    outlier_output = torch.zeros_like(main_output)
    if outlier_weight is not None and contract.outlier_channels:
        indices = torch.tensor(contract.outlier_channels, device=activation.device, dtype=torch.long)
        exact = activation.to(torch.float32).index_select(1, indices) @ outlier_weight.to(torch.float32).transpose(0, 1)
        if contract.outlier_mode == "correction":
            decoded_main = dequantize_weight_reference(
                main_weight,
                spec=spec.quant,
                scales=main_weight.scales,
            )
            approx = activation.to(torch.float32).index_select(1, indices) @ decoded_main.index_select(1, indices).transpose(0, 1)
            outlier_output = exact - approx
        else:
            outlier_output = exact
    output = main_output + low_rank_output + outlier_output
    reference_main = None
    reference_low = None
    reference_outlier = None
    reference_merged = None
    if main_weight_reference is not None:
        reference_merged = activation.to(torch.float32) @ main_weight_reference.to(torch.float32).transpose(0, 1)
        reference_main = reference_merged
        reference_low = torch.zeros_like(low_rank_output)
        reference_outlier = torch.zeros_like(outlier_output)
    output_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(spec.quant.output_dtype, torch.float32)
    return SVDGemmResult(
        output=output.to(dtype=output_dtype),
        main_output=main_output,
        low_rank_output=low_rank_output,
        outlier_output=outlier_output,
        report=SVDGemmReport(
            execution_mode="reference_dual_path",
            fusion_boundary=contract.fusion_boundary,
            rank=contract.rank,
            outlier_channels=contract.outlier_channels,
            main_max_abs_error=_max_abs_error(main_output, reference_main),
            low_rank_max_abs_error=_max_abs_error(low_rank_output, reference_low),
            outlier_max_abs_error=_max_abs_error(outlier_output, reference_outlier),
            merged_max_abs_error=_max_abs_error(output, reference_merged),
        ),
    )


def reference_svd_outlier_hybrid(
    activation: torch.Tensor,
    main_weight: PackedWeight,
    *,
    contract: SVDGemmContract,
    low_rank_down: torch.Tensor,
    low_rank_up: torch.Tensor,
    outlier_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> SVDGemmResult:
    """Named convenience wrapper for the outlier-channel hybrid path."""

    return reference_svd_dual_path(
        activation,
        main_weight,
        contract=contract,
        low_rank_down=low_rank_down,
        low_rank_up=low_rank_up,
        outlier_weight=outlier_weight,
        bias=bias,
    )


__all__ = [
    "FP4_E2M1_CODEBOOK",
    "FP4_FORMATS",
    "FP4QuantizedTensor",
    "SVDGemmContract",
    "SVDGemmReport",
    "SVDGemmResult",
    "W4A8Contract",
    "W4A8GroupedReport",
    "W4A8GroupedResult",
    "W4A8Spec",
    "W4A8_ACTIVATION_FORMATS",
    "build_w4a8_gemm_spec",
    "pack_fp4_weight",
    "pack_w4a8_weight",
    "quantize_fp4_reference",
    "quantize_w4a8_activation",
    "reference_grouped_w4a8_gemm",
    "reference_svd_dual_path",
    "reference_svd_outlier_hybrid",
    "reference_w4a8_gemm",
]
