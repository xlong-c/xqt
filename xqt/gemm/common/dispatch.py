"""Registry-driven GEMM dispatch with explicit fallback reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from xqt.core.errors import XQTBackendError

from .contracts import GemmSpec, PackedWeight
from .layout import validate_w4a16_packed_weight
from .reference import reference_gemm, reference_w4a16_gemm, reference_w8a16_gemm
from .registry import GemmKernelRegistration, GemmKernelRegistry, default_registry


@dataclass(frozen=True, slots=True)
class GemmDispatchReport:
    """Machine-readable selection result; metadata-only is never executable."""

    requested_kernel: str | None
    selected_kernel: str
    backend: str
    kernel_family: str
    maturity: str
    arch: str
    tile_shape: tuple[int, int, int] | None
    warp_count: int | None
    stage_count: int | None
    scale_mode: str
    pack_version: str
    weight_scale_source: str = "none"
    activation_scale_source: str = "none"
    activation_granularity: str = "none"
    padding_ratio: float = 0.0
    fallback_reason: str | None = None
    fallback_chain: tuple[str, ...] = ()
    group_size: int | None = None
    native: bool = False
    shape_variant: str = "generic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_kernel": self.requested_kernel,
            "selected_kernel": self.selected_kernel,
            "backend": self.backend,
            "kernel_family": self.kernel_family,
            "maturity": self.maturity,
            "arch": self.arch,
            "tile_shape": None if self.tile_shape is None else list(self.tile_shape),
            "warp_count": self.warp_count,
            "stage_count": self.stage_count,
            "scale_mode": self.scale_mode,
            "pack_version": self.pack_version,
            "weight_scale_source": self.weight_scale_source,
            "activation_scale_source": self.activation_scale_source,
            "activation_granularity": self.activation_granularity,
            "padding_ratio": self.padding_ratio,
            "fallback_reason": self.fallback_reason,
            "fallback_chain": list(self.fallback_chain),
            "group_size": self.group_size,
            "native": self.native,
            "shape_variant": self.shape_variant,
        }


@dataclass(frozen=True, slots=True)
class GemmDispatchResult:
    output: torch.Tensor
    report: GemmDispatchReport


def _reference_entry(
    matches: tuple[GemmKernelRegistration, ...],
    *,
    weight: torch.Tensor | PackedWeight,
) -> GemmKernelRegistration | None:
    for entry in matches:
        if entry.implementation == "reference" and entry.maturity == "reference_guarded":
            if entry.kernel_family in {"w4a16_reference", "w8a16_reference"}:
                if not isinstance(weight, PackedWeight):
                    continue
                if entry.kernel_family == "w4a16_reference" and weight.metadata.storage_layout != "xqt_int4_nk_v1":
                    continue
                if entry.kernel_family == "w8a16_reference" and weight.metadata.storage_layout != "xqt_int8_nk_v1":
                    continue
            return entry
    return None


def _run_reference(
    entry: GemmKernelRegistration,
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None,
    activation_scales: torch.Tensor | None,
    weight_zero_points: torch.Tensor | None,
    activation_zero_points: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    if entry.kernel_family == "w4a16_reference":
        if not isinstance(weight, PackedWeight):
            raise TypeError("w4a16_reference requires a PackedWeight")
        return reference_w4a16_gemm(
            activation,
            weight,
            spec=spec,
            weight_scales=weight_scales,
            activation_scales=activation_scales,
            weight_zero_points=weight_zero_points,
            activation_zero_points=activation_zero_points,
            bias=bias,
            residual=residual,
        )
    if entry.kernel_family == "w8a16_reference":
        if not isinstance(weight, PackedWeight):
            raise TypeError("w8a16_reference requires a PackedWeight")
        return reference_w8a16_gemm(
            activation,
            weight,
            spec=spec,
            weight_scales=weight_scales,
            activation_scales=activation_scales,
            weight_zero_points=weight_zero_points,
            activation_zero_points=activation_zero_points,
            bias=bias,
            residual=residual,
        )
    return reference_gemm(
        activation,
        weight,
        spec=spec,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
        weight_zero_points=weight_zero_points,
        activation_zero_points=activation_zero_points,
        bias=bias,
        residual=residual,
    )


def _problem_spec(spec: GemmSpec) -> tuple[str, str]:
    return spec.problem.arch, spec.quant.scale_mode


def dispatch_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    spec: GemmSpec,
    weight_scales: torch.Tensor | None = None,
    activation_scales: torch.Tensor | None = None,
    weight_zero_points: torch.Tensor | None = None,
    activation_zero_points: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    registry: GemmKernelRegistry | None = None,
    requested_kernel: str | None = None,
    allow_reference: bool = True,
) -> GemmDispatchResult:
    """Dispatch one GEMM and return output plus a transparent selection report."""

    if not isinstance(spec, GemmSpec):
        raise TypeError("dispatch_gemm requires a complete GemmSpec")
    active_registry = default_registry() if registry is None else registry
    all_matches = active_registry.matching(spec.problem, spec.quant, spec.epilogue)
    if requested_kernel is not None:
        requested = active_registry.get(requested_kernel)
        if not requested.supports(spec.problem, spec.quant, spec.epilogue):
            raise ValueError(
                f"requested kernel {requested_kernel!r} does not support problem/quant/epilogue"
            )
        # A requested kernel is a preferred first candidate, not permission to
        # silently skip the registered fallback ladder.  This keeps explicit
        # experiments safe when an artifact is absent or a shape is rejected.
        matches = (requested,) + tuple(
            entry for entry in all_matches if entry.name != requested.name
        )
    else:
        matches = all_matches
    if not matches:
        raise RuntimeError(
            "no GEMM registry entry supports "
            f"arch={spec.problem.arch}, weight_dtype={spec.quant.weight_dtype}, "
            f"activation_dtype={spec.quant.activation_dtype}, scale_mode={spec.quant.scale_mode}"
        )
    if (
        spec.quant.weight_dtype == "int4"
        and isinstance(weight, PackedWeight)
        and weight.metadata.storage_layout == "xqt_int4_nk_v1"
        and spec.quant.activation_dtype in {"fp16", "bf16"}
    ):
        # The W4A16 native ABI is narrower than the W4A8 reference contract.
        # In particular, its kernel only accepts the documented 32/64/128 K
        # groups, so applying its validator to an INT8/FP8 activation path
        # would reject a valid reference-only W4A8 packed weight.
        validate_w4a16_packed_weight(
            weight,
            spec=spec.quant,
            logical_shape=(spec.problem.n, spec.problem.k),
        )
    fallback_reasons: list[str] = []
    report_entry: GemmKernelRegistration | None = None
    output: torch.Tensor | None = None
    native = False
    for candidate in matches:
        if not candidate.executable:
            if candidate.maturity in {"metadata_only", "planned"}:
                fallback_reasons.append(
                    f"candidate {candidate.name!r} is maturity={candidate.maturity}"
                )
            continue
        if candidate.executor is None:
            raise RuntimeError("registry marked a native GEMM executable without an executor")
        try:
            output = candidate.executor(
                activation,
                weight,
                spec=spec,
                weight_scales=weight_scales,
                activation_scales=activation_scales,
                weight_zero_points=weight_zero_points,
                activation_zero_points=activation_zero_points,
                bias=bias,
                residual=residual,
            )
            report_entry = candidate
            native = True
        except XQTBackendError as exc:
            fallback_reasons.append(f"candidate {candidate.name!r} unavailable: {exc}")
            if not allow_reference:
                raise
            continue
        break
    if report_entry is None:
        report_entry = _reference_entry(matches, weight=weight)
        if report_entry is None or not allow_reference:
            top = matches[0]
            details = "; ".join(fallback_reasons)
            suffix = f" ({details})" if details else ""
            raise RuntimeError(
                f"GEMM kernel {top.name!r} is maturity={top.maturity}, "
                f"and no executable/reference fallback is allowed{suffix}"
            )
        output = _run_reference(
            report_entry,
            activation,
            weight,
            spec=spec,
            weight_scales=weight_scales,
            activation_scales=activation_scales,
            weight_zero_points=weight_zero_points,
            activation_zero_points=activation_zero_points,
            bias=bias,
            residual=residual,
        )
        native = False
    if output is None:
        raise RuntimeError("GEMM dispatch selected a kernel without producing output")
    fallback_reason = "; ".join(fallback_reasons) if fallback_reasons else None
    arch, scale_mode = _problem_spec(spec)
    # Use the highest-priority non-reference candidate for padding accounting.
    # A reference fallback has alignment (1,1,1), but the report should still
    # expose the padding that the preferred native tile would have required.
    alignment_entry = next(
        (entry for entry in matches if entry.implementation != "reference"), report_entry
    )
    alignment = alignment_entry.alignment
    padded_m = ((spec.problem.m + alignment[0] - 1) // alignment[0]) * alignment[0]
    padded_n = ((spec.problem.n + alignment[1] - 1) // alignment[1]) * alignment[1]
    padded_k = ((spec.problem.k + alignment[2] - 1) // alignment[2]) * alignment[2]
    # The SM89 INT8 entry has a true DP4A M=1 GEMV branch; it does not pad the
    # row dimension even though its main MMA tile advertises M alignment 16.
    if report_entry.kernel_family == "w8a8_int8_mma" and spec.problem.m == 1:
        padded_m = spec.problem.m
    logical_elements = spec.problem.m * spec.problem.n * spec.problem.k
    padded_elements = padded_m * padded_n * padded_k
    kernel_family = report_entry.kernel_family
    tile_shape = report_entry.tile_shape
    warp_count = report_entry.warp_count
    stage_count = report_entry.stage_count
    if kernel_family == "w8a8_int8_mma" and spec.problem.m == 1:
        kernel_family = "w8a8_int8_gemv"
        tile_shape = None
        warp_count = 1
        stage_count = 1
    shape_variant = "generic"
    if native and report_entry.kernel_family == "w4a16_dequant_fallback":
        if spec.problem.m == 1:
            shape_variant = "m1_gemv"
        elif spec.problem.m <= 8:
            shape_variant = "small_m_2_8"
        else:
            shape_variant = "tile_m_8x16x32"
    elif native and report_entry.kernel_family == "w4a16_fused_cutlass_mma":
        shape_variant = "tile_m_16x8x16"
    elif native and report_entry.kernel_family == "w8a16":
        if spec.problem.m == 1:
            shape_variant = "m1_gemv"
        else:
            shape_variant = "general_m_quantize_int8"
    return GemmDispatchResult(
        output=output,
        report=GemmDispatchReport(
            requested_kernel=requested_kernel,
            selected_kernel=report_entry.name,
            backend=report_entry.backend,
            kernel_family=kernel_family,
            maturity=report_entry.maturity,
            arch=arch,
            tile_shape=tile_shape,
            warp_count=warp_count,
            stage_count=stage_count,
            scale_mode=scale_mode,
            pack_version=(
                weight.metadata.pack_version
                if isinstance(weight, PackedWeight)
                else spec.quant.pack_version
            ),
            weight_scale_source=spec.quant.weight_scale_source,
            activation_scale_source=spec.quant.activation_scale_source,
            activation_granularity=spec.quant.activation_granularity,
            padding_ratio=(
                float(padded_elements - logical_elements) / float(logical_elements)
            ),
            fallback_reason=fallback_reason,
            fallback_chain=tuple(entry.name for entry in matches),
            group_size=(
                weight.metadata.group_size
                if isinstance(weight, PackedWeight)
                else spec.quant.group_size
            ),
            native=native,
            shape_variant=shape_variant,
        ),
    )


def select_kernel(
    spec: GemmSpec,
    *,
    registry: GemmKernelRegistry | None = None,
) -> tuple[GemmKernelRegistration, ...]:
    """Return registry candidates in dispatch order without executing anything."""

    active_registry = default_registry() if registry is None else registry
    return active_registry.matching(spec.problem, spec.quant, spec.epilogue)


__all__ = ["GemmDispatchReport", "GemmDispatchResult", "dispatch_gemm", "select_kernel"]
