"""Unified GEMM dispatcher with family-oriented backend reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from xqt.contracts import PrecisionPolicy
from xqt.core.errors import XQTBackendError
from xqt.gemm import (
    EpilogueSpec,
    GemmProblem,
    GemmSpec,
    QuantSpec,
    default_registry,
    dense_gemm_reference,
    dispatch_gemm,
)

from . import run_tilelang_kernel, run_triton_kernel
from .gemm_selector import GemmShape, select_gemm_engine

MatmulPrecisionSpec = PrecisionPolicy


_DTYPE_PRECISIONS: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


_SUPPORTED_ACTIVATIONS = {None, "relu", "gelu", "silu"}


@dataclass(frozen=True)
class GemmKernelFamilySpec:
    """One GEMM-family dispatch contract shared by Triton and TileLang."""

    family: str
    mma: str
    engines: dict[str, str]
    supports_activation: bool = True
    requires_bias_vector: bool = False
    engine_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    engine_pattern_aliases: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class GemmVariantDispatchSpec:
    """Table-name to executable GEMM composition mapping."""

    op: str
    precision: str | None = None
    activation: str | None = None
    engine_patterns: dict[str, str | None] = field(default_factory=dict)
    allow_generic_engine_fallback: bool = True
    notes: str = ""


_TRITON_DENSE_FAMILIES: dict[str, GemmKernelFamilySpec] = {
    "fp16": GemmKernelFamilySpec(
        family="dense_gemm_2d",
        mma="fp16",
        engines={"triton": "gemm_fp16", "tilelang": "dense_linear_epilogue"},
        engine_pattern_aliases={
            "tilelang": {
                "dense_linear_epilogue": "dense_linear_epilogue",
                "linear": "linear",
                "half_linear": "linear",
                "linear_marlin": "linear_marlin",
            },
        },
    ),
    "bf16": GemmKernelFamilySpec(
        family="dense_gemm_2d",
        mma="bf16",
        engines={"triton": "gemm_bf16", "tilelang": "dense_linear_epilogue"},
        engine_pattern_aliases={
            "tilelang": {
                "dense_linear_epilogue": "dense_linear_epilogue",
                "linear": "linear",
                "half_linear": "linear",
                "linear_marlin": "linear_marlin",
            },
        },
    ),
    "int8": GemmKernelFamilySpec(
        family="dequant_gemm_or_true_w8a8",
        mma="int8",
        engines={"triton": "gemm_int8", "tilelang": "int8_linear"},
        engine_pattern_aliases={
            "tilelang": {
                "int8_mma": "int8_mma",
                "int8_linear": "int8_linear",
                "int8_linear_static_activation": "int8_linear_static_activation",
                "linear_marlin": "linear_marlin",
                "dequant_gemm_epilogue": "dequant_gemm_epilogue",
            },
        },
    ),
    "fp8": GemmKernelFamilySpec(
        family="quantized_activation_gemm",
        mma="fp8",
        engines={"triton": "gemm_fp8"},
    ),
    "int4": GemmKernelFamilySpec(
        family="dequant_gemm",
        mma="int4",
        engines={"triton": "gemm_int4_dequant"},
        engine_pattern_aliases={
            "tilelang": {
                "linear_marlin": "linear_marlin",
            },
        },
    ),
    "mxfp8": GemmKernelFamilySpec(
        family="packed_weight_gemm",
        mma="mxfp8",
        engines={"triton": "gemm_mxfp8"},
        engine_kwargs={"triton": {"mx_precision": 8}},
    ),
    "mxfp6": GemmKernelFamilySpec(
        family="packed_weight_gemm",
        mma="mxfp6",
        engines={"triton": "gemm_mxfp6"},
        engine_kwargs={"triton": {"mx_precision": 6}},
    ),
    "mxfp4": GemmKernelFamilySpec(
        family="packed_weight_gemm",
        mma="mxfp4",
        engines={"triton": "gemm_mxfp4"},
        engine_kwargs={"triton": {"mx_precision": 4}},
    ),
}


_TILELANG_PACKED_FAMILIES: dict[str, GemmKernelFamilySpec] = {
    "fp4": GemmKernelFamilySpec(
        family="packed_weight_gemm",
        mma="fp4",
        engines={
            "tilelang": "fp4_packed_dequant_gemm_epilogue",
            "triton": "gemm_int4_dequant",
        },
        engine_pattern_aliases={
            "triton": {
                "fp4_packed_dequant_gemm_epilogue": "gemm_int4_dequant",
            },
        },
    ),
    "mxfp4": GemmKernelFamilySpec(
        family="packed_weight_gemm",
        mma="mxfp4",
        engines={
            "tilelang": "mxfp4_packed_dequant_gemm_epilogue",
            "triton": "gemm_mxfp4",
        },
        engine_kwargs={"triton": {"mx_precision": 4}},
        engine_pattern_aliases={
            "triton": {
                "mxfp4_packed_dequant_gemm_epilogue": "gemm_mxfp4",
            },
        },
    ),
    "nvfp4": GemmKernelFamilySpec(
        family="packed_weight_gemm",
        mma="nvfp4",
        engines={
            "tilelang": "nvfp4_packed_dequant_gemm_epilogue",
            "triton": "gemm_nvfp4_packed_dequant",
        },
        engine_pattern_aliases={
            "triton": {
                "nvfp4_packed_dequant_gemm_epilogue": "gemm_nvfp4_packed_dequant",
            },
        },
    ),
}


_GEMM_KERNEL_FAMILY_SPECS: dict[str, GemmKernelFamilySpec] = {
    **_TRITON_DENSE_FAMILIES,
    **_TILELANG_PACKED_FAMILIES,
}


_GEMM_VARIANT_DISPATCH_TABLE: dict[str, GemmVariantDispatchSpec] = {
    "gemm_fp16_dense_2d": GemmVariantDispatchSpec(op="gemm", precision="fp16"),
    "gemm_bf16_dense_2d": GemmVariantDispatchSpec(op="gemm", precision="bf16"),
    "gemm_fp32_reference": GemmVariantDispatchSpec(op="gemm", precision="fp32"),
    "gemm_fp16_bias": GemmVariantDispatchSpec(op="gemm", precision="fp16"),
    "gemm_bf16_bias": GemmVariantDispatchSpec(op="gemm", precision="bf16"),
    "gemm_bias": GemmVariantDispatchSpec(op="gemm"),
    "gemm_bias_relu": GemmVariantDispatchSpec(op="gemm", activation="relu"),
    "gemm_bias_gelu": GemmVariantDispatchSpec(op="gemm", activation="gelu"),
    "gemm_bias_silu": GemmVariantDispatchSpec(op="gemm", activation="silu"),
    "gemm_residual_add": GemmVariantDispatchSpec(op="gemm_residual"),
    "gemm_int8_weight_only_reference": GemmVariantDispatchSpec(
        op="gemm",
        precision="int8",
        engine_patterns={"tilelang": "dequant_gemm_epilogue"},
    ),
    "gemm_true_w8a8": GemmVariantDispatchSpec(
        op="gemm",
        precision="int8",
        engine_patterns={"tilelang": "int8_linear"},
    ),
    "int8_linear": GemmVariantDispatchSpec(
        op="gemm",
        precision="int8",
        engine_patterns={"tilelang": "int8_linear"},
        allow_generic_engine_fallback=False,
    ),
    "int8_linear_static_activation": GemmVariantDispatchSpec(
        op="gemm",
        precision="int8",
        engine_patterns={"tilelang": "int8_linear_static_activation"},
        allow_generic_engine_fallback=False,
    ),
    "gemm_int4_weight_only_dequant": GemmVariantDispatchSpec(
        op="gemm",
        precision="int4",
        engine_patterns={"triton": "gemm_int4_dequant", "tilelang": "linear_marlin"},
        allow_generic_engine_fallback=False,
    ),
    "dequant_gemm_epilogue": GemmVariantDispatchSpec(
        op="gemm",
        precision="int8",
        engine_patterns={
            "triton": "gemm_int8",
            "tilelang": "dequant_gemm_epilogue",
        },
        allow_generic_engine_fallback=False,
    ),
    "gemm_fp4_packed_dequant": GemmVariantDispatchSpec(op="gemm", precision="fp4"),
    "fp4_packed_dequant_gemm_epilogue": GemmVariantDispatchSpec(
        op="gemm",
        precision="fp4",
    ),
    "mxfp4_packed_dequant_gemm_epilogue": GemmVariantDispatchSpec(
        op="gemm",
        precision="mxfp4",
    ),
    "gemm_nvfp4_packed_dequant": GemmVariantDispatchSpec(op="gemm", precision="nvfp4"),
    "nvfp4_packed_dequant_gemm_epilogue": GemmVariantDispatchSpec(
        op="gemm",
        precision="nvfp4",
    ),
    "gemm_mxfp8": GemmVariantDispatchSpec(op="gemm", precision="mxfp8"),
    "gemm_mxfp6": GemmVariantDispatchSpec(op="gemm", precision="mxfp6"),
    "gemm_mxfp4": GemmVariantDispatchSpec(op="gemm", precision="mxfp4"),
    "marlin_style_packed_gemm": GemmVariantDispatchSpec(
        op="gemm",
        engine_patterns={"tilelang": "linear_marlin"},
        allow_generic_engine_fallback=False,
    ),
    "linear_marlin": GemmVariantDispatchSpec(
        op="gemm",
        engine_patterns={"tilelang": "linear_marlin"},
        allow_generic_engine_fallback=False,
    ),
    "batched_gemm": GemmVariantDispatchSpec(op="batched"),
    "attention_score_matmul": GemmVariantDispatchSpec(op="attention_score"),
    "attention_score_gemm_composition": GemmVariantDispatchSpec(op="attention_score"),
    "qk_matmul_prefill": GemmVariantDispatchSpec(op="attention_score"),
    "qk_matmul_decode_kvcache": GemmVariantDispatchSpec(op="attention_score"),
    "attention_value_matmul": GemmVariantDispatchSpec(op="attention_value"),
    "attention_value_gemm_composition": GemmVariantDispatchSpec(op="attention_value"),
    "pv_matmul_prefill": GemmVariantDispatchSpec(op="attention_value"),
    "pv_matmul_decode_kvcache": GemmVariantDispatchSpec(op="attention_value"),
    "q_proj_gemm": GemmVariantDispatchSpec(op="projection"),
    "k_proj_gemm": GemmVariantDispatchSpec(op="projection"),
    "v_proj_gemm": GemmVariantDispatchSpec(op="projection"),
    "qkv_projection_gemm": GemmVariantDispatchSpec(op="projection"),
    "fused_qkv_gemm": GemmVariantDispatchSpec(op="projection"),
    "fused_qkv_gemm_prefill": GemmVariantDispatchSpec(op="projection"),
    "fused_qkv_gemm_decode": GemmVariantDispatchSpec(op="projection"),
    "o_projection_gemm": GemmVariantDispatchSpec(op="projection"),
    "o_proj_gemm": GemmVariantDispatchSpec(op="projection"),
    "o_proj_gemm_prefill": GemmVariantDispatchSpec(op="projection"),
    "o_proj_gemm_decode": GemmVariantDispatchSpec(op="projection"),
    "ffn_up_gemm": GemmVariantDispatchSpec(op="projection"),
    "ffn_gate_gemm": GemmVariantDispatchSpec(op="projection"),
    "ffn_down_gemm": GemmVariantDispatchSpec(op="projection"),
    "proj_in_gelu_epilogue": GemmVariantDispatchSpec(
        op="projection",
        activation="gelu",
    ),
    "proj_out_epilogue": GemmVariantDispatchSpec(op="projection"),
    "bias_gelu_epilogue_gemm": GemmVariantDispatchSpec(op="gemm", activation="gelu"),
    "router_gemm": GemmVariantDispatchSpec(op="projection"),
    "router_logits_gemm": GemmVariantDispatchSpec(op="projection"),
    "router_gemm_composition": GemmVariantDispatchSpec(op="projection"),
    "grouped_gemm": GemmVariantDispatchSpec(op="grouped"),
    "grouped_gemm_composition": GemmVariantDispatchSpec(op="grouped"),
    "expert_gemm": GemmVariantDispatchSpec(op="expert"),
    "expert_gemm_composition": GemmVariantDispatchSpec(op="expert"),
    "expert_up_grouped_gemm": GemmVariantDispatchSpec(op="expert"),
    "expert_gate_grouped_gemm": GemmVariantDispatchSpec(op="expert"),
    "expert_down_grouped_gemm": GemmVariantDispatchSpec(op="expert"),
    "shared_expert_gemm": GemmVariantDispatchSpec(op="projection"),
    "lm_head_gemm": GemmVariantDispatchSpec(op="projection"),
    "lm_head_gemm_prefill": GemmVariantDispatchSpec(op="projection"),
    "lm_head_gemm_decode": GemmVariantDispatchSpec(op="projection"),
    "conv1x1_as_gemm": GemmVariantDispatchSpec(op="conv1x1"),
    "conv3x3_im2col_gemm": GemmVariantDispatchSpec(op="conv3x3"),
    "patch_embed_gemm": GemmVariantDispatchSpec(op="projection"),
    "patch_embed_linear": GemmVariantDispatchSpec(op="projection"),
    "conv_bias_silu_epilogue": GemmVariantDispatchSpec(
        op="conv2d",
        activation="silu",
    ),
    "conv_bias_relu": GemmVariantDispatchSpec(op="conv2d", activation="relu"),
    "head_cls_gemm": GemmVariantDispatchSpec(op="projection"),
    "head_box_gemm": GemmVariantDispatchSpec(op="projection"),
    "dfl_projection_gemm": GemmVariantDispatchSpec(op="projection"),
}


def _require_supported_activation(activation: str | None) -> None:
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise XQTBackendError(
            f"Unsupported GEMM activation: {activation!r}. "
            f"Known: {sorted(a for a in _SUPPORTED_ACTIVATIONS if a is not None)}"
        )


def _precision_name_to_dtype(
    name: str,
    *,
    fallback: torch.dtype | None = None,
    role: str = "precision",
) -> torch.dtype:
    canonical = PrecisionPolicy.canonical_name(name)
    dtype = _DTYPE_PRECISIONS.get(canonical)
    if dtype is not None:
        return dtype
    if fallback is not None:
        return fallback
    raise XQTBackendError(
        f"GEMM dtype mapping is not defined for {role} precision {name}"
    )


def _resolve_matmul_precision(
    precision: str | MatmulPrecisionSpec | Mapping[str, Any],
) -> MatmulPrecisionSpec:
    if isinstance(precision, MatmulPrecisionSpec):
        return precision
    if isinstance(precision, str):
        canonical = PrecisionPolicy.canonical_name(precision)
        return PrecisionPolicy(
            activation=canonical,
            weight=canonical,
            bias=canonical,
            mma=canonical,
            accum="fp32",
            output=canonical,
        )
    return PrecisionPolicy.from_mapping(precision)


def _resolve_gemm_engine(
    *,
    engine: str,
) -> str:
    return str(engine).strip().lower()


def _resolve_gemm_family_spec(precision: MatmulPrecisionSpec) -> GemmKernelFamilySpec:
    try:
        return _GEMM_KERNEL_FAMILY_SPECS[precision.mma]
    except KeyError as exc:
        raise XQTBackendError(
            f"Unsupported GEMM precision family: {precision.mma}"
        ) from exc


def _resolve_requested_pattern(
    family: GemmKernelFamilySpec,
    engine: str,
    pattern: str | None,
) -> str:
    default_pattern = family.engines.get(engine)
    engine_aliases = family.engine_pattern_aliases.get(engine, {})
    if pattern is None:
        if default_pattern is None:
            raise XQTBackendError(
                f"{engine} engine is not configured for GEMM precision family {family.mma}"
            )
        return default_pattern
    requested = str(pattern).strip()
    if not requested:
        raise XQTBackendError("GEMM pattern must not be empty")
    resolved = engine_aliases.get(requested, requested)
    available = set(engine_aliases.values())
    if default_pattern is not None:
        available.add(default_pattern)
    if resolved not in available:
        raise XQTBackendError(
            f"GEMM pattern {requested!r} is not compatible with precision family "
            f"{family.mma!r} on engine {engine!r}. Available: {sorted(available)}"
        )
    return resolved


def _tensor_gemm_dtype(tensor: torch.Tensor) -> str:
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int8: "int8",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise XQTBackendError(
            f"GEMM contract does not support tensor dtype {tensor.dtype}"
        ) from exc


def _gemm_spec_from_precision_call(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    has_bias: bool,
    runtime_kwargs: Mapping[str, Any],
) -> GemmSpec:
    shape = _gemm_shape_from_tensors(a, b, transpose_b)
    sm = None
    if a.device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(a.device)
        sm = major * 10 + minor
    weight_dtype = {
        "fp8": "fp8_e4m3",
    }.get(precision.mma, precision.mma)
    if precision.mma in {"fp16", "bf16", "fp32"}:
        weight_dtype = _tensor_gemm_dtype(b)
    activation_dtype = _tensor_gemm_dtype(a)
    quantized_weights = weight_dtype not in {"fp16", "bf16", "fp32"}
    group_size_value = runtime_kwargs.get("group_size")
    group_size = None if group_size_value is None else int(group_size_value)
    weight_granularity = "groupwise" if group_size is not None else "per_tensor"
    activation_granularity = (
        "per_token" if activation_dtype in {"int8", "fp8_e4m3", "fp8_e5m2"}
        else "per_tensor"
    )
    output_dtype = (
        precision.output
        if precision.output in {"fp16", "bf16", "fp32"}
        else "bf16"
        if a.dtype == torch.bfloat16
        else "fp16"
    )
    return GemmSpec(
        problem=GemmProblem(
            m=shape.m,
            n=shape.n,
            k=shape.k,
            batch=shape.batch,
            device=str(a.device),
            sm=sm,
        ),
        quant=QuantSpec(
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            compute_dtype={"fp8": "fp8_e4m3"}.get(
                precision.mma, precision.mma
            ),
            accum_dtype=precision.accum,
            output_dtype=output_dtype,
            weight_granularity=weight_granularity,
            activation_granularity=activation_granularity,
            group_size=group_size,
            weight_scale_source=("weight_load_time" if quantized_weights else "none"),
            activation_scale_source=(
                "activation_dynamic"
                if activation_dtype in {"int8", "fp8_e4m3", "fp8_e5m2"}
                else "none"
            ),
        ),
        epilogue=EpilogueSpec(
            activation="none" if activation is None else activation,
            has_bias=has_bias,
            output_dtype=output_dtype,
        ),
    )


def gemm_with_precision(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    transpose_b: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Unified GEMM dispatcher with precision control.

    Args:
        a: Left matrix (M, K)
        b: Right matrix (N, K) if transpose_b else (K, N)
        bias: Optional addend vector C with shape (N,)
        precision: Precision mode or role spec. Supported names include
            "fp16", "bf16", "fp32", "int8", "fp8", "int4", "fp4",
            "nvfp4", "mxfp8", "mxfp6", "mxfp4". Mapping inputs may use
            A/B/C/O aliases for activation/weight/bias/output.
        engine: XQT GEMM engine - "triton", "tilelang", "torch", "auto"
        pattern: Optional concrete kernel pattern override within one precision
            family, for example "linear_marlin", "dequant_gemm_epilogue",
            "int8_linear", or "int8_linear_static_activation".
        activation: Optional activation - "relu", "gelu", "silu"
        transpose_b: Whether to transpose b before matmul
        **kwargs: Engine-specific parameters (scales, group_size, etc.)

    Returns:
        Output tensor (M, N)
    """
    precision_spec = _resolve_matmul_precision(precision)
    resolved_engine = _resolve_gemm_engine(engine=engine)
    selection = None
    if resolved_engine == "auto":
        selection = select_gemm_engine(
            precision=precision_spec,
            device=a.device,
            shape=_gemm_shape_from_tensors(a, b, transpose_b),
            fused_ops=_fused_ops_from_call(a, precision_spec.mma, activation),
        )
        resolved_engine = selection.selected_engine

    registry = default_registry()
    if selection is None:
        registrations = tuple(
            entry
            for entry in registry.entries()
            if entry.scope == "precision"
            and precision_spec.mma in entry.precision_mmas
            and entry.backend == resolved_engine
            and entry.dispatchable_by_precision
        )
        if not registrations:
            raise XQTBackendError(
                f"Unsupported GEMM engine {resolved_engine!r} for precision "
                f"{precision_spec.mma!r}"
            )
        candidate_kernels = (registrations[0].name,)
    else:
        candidate_kernels = tuple(
            candidate.kernel_name
            for candidate in selection.candidates
            if candidate.dispatchable_by_gemm_with_precision
        )
    spec = _gemm_spec_from_precision_call(
        a,
        b,
        precision=precision_spec,
        activation=activation,
        transpose_b=transpose_b,
        has_bias=bias is not None,
        runtime_kwargs=kwargs,
    )
    result = dispatch_gemm(
        a,
        b,
        spec=spec,
        bias=bias,
        registry=registry,
        requested_kernel=candidate_kernels[0],
        candidate_kernels=candidate_kernels,
        executor_kwargs={
            "precision_policy": precision_spec,
            "activation_name": activation,
            "transpose_b": transpose_b,
            "runtime_kwargs": dict(kwargs),
            "pattern": pattern,
        },
        allow_reference=False,
    )
    return result.output


def list_gemm_variant_dispatch_specs() -> dict[str, dict[str, Any]]:
    """Return table-name to GEMM composition dispatch metadata."""

    return {
        name: {
            "op": spec.op,
            "precision": spec.precision,
            "activation": spec.activation,
            "engine_patterns": dict(spec.engine_patterns),
            "notes": spec.notes,
        }
        for name, spec in sorted(_GEMM_VARIANT_DISPATCH_TABLE.items())
    }


def gemm_variant_with_precision(
    variant: str,
    *args: Any,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] | None = None,
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run a named GEMM variant from ``KERNEL_GUIDANCE_TABLE`` via reuse.

    This is intentionally a thin table-driven layer. It translates backlog
    names such as ``qkv_projection_gemm`` or ``conv3x3_im2col_gemm`` into the
    existing 2D GEMM, batched GEMM, grouped GEMM, projection, or conv-lowering
    composition paths.
    """

    name = str(variant).strip()
    try:
        spec = _GEMM_VARIANT_DISPATCH_TABLE[name]
    except KeyError as exc:
        known = ", ".join(sorted(_GEMM_VARIANT_DISPATCH_TABLE))
        raise XQTBackendError(
            f"Unsupported GEMM variant {variant!r}. Known GEMM variants: {known}"
        ) from exc
    resolved_engine = _resolve_gemm_engine(engine=engine)
    resolved_precision = precision if precision is not None else spec.precision or "fp16"
    resolved_activation = activation if activation is not None else spec.activation
    resolved_pattern = pattern
    if resolved_pattern is None and resolved_engine != "auto":
        resolved_pattern = spec.engine_patterns.get(resolved_engine)
    if (
        resolved_pattern is None
        and resolved_engine != "auto"
        and not spec.allow_generic_engine_fallback
    ):
        raise XQTBackendError(
            f"GEMM variant {variant!r} does not have a generic {resolved_engine} fallback"
        )
    runtime_kwargs = dict(kwargs)

    match spec.op:
        case "gemm":
            return gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "gemm_residual":
            residual = runtime_kwargs.pop("residual", None)
            if residual is None:
                raise XQTBackendError("gemm_residual_add requires residual=<tensor>")
            output = gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
            return output + residual.to(dtype=output.dtype, device=output.device)
        case "batched":
            return batched_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "projection":
            return projection_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "attention_score":
            return attention_score_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                **runtime_kwargs,
            )
        case "attention_value":
            return attention_value_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                **runtime_kwargs,
            )
        case "grouped":
            return grouped_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "expert":
            return expert_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "conv1x1":
            return conv1x1_as_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "conv3x3":
            return conv3x3_im2col_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case "conv2d":
            return conv2d_as_gemm_with_precision(
                *args,
                precision=resolved_precision,
                engine=engine,
                pattern=resolved_pattern,
                activation=resolved_activation,
                **runtime_kwargs,
            )
        case _:
            raise XQTBackendError(
                f"GEMM variant {variant!r} is wired to unknown op {spec.op!r}"
            )


def batched_gemm_with_precision(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    transpose_b: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Batched GEMM dispatcher that reuses the configured 2D GEMM path.

    If ``b`` is 2D, all batch slices share one weight matrix and the function
    flattens ``[B, M, K]`` into one ``[B*M, K]`` GEMM. If ``b`` is 3D, each
    batch slice is dispatched through the same 2D engine/pattern contract and
    the outputs are stacked.
    """

    if a.ndim != 3:
        raise XQTBackendError(
            f"batched_gemm_with_precision expects a to be 3D, got shape={tuple(a.shape)}"
        )
    batch, rows, k = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    if b.ndim == 2:
        output = gemm_with_precision(
            a.reshape(batch * rows, k),
            b,
            bias,
            precision=precision,
            engine=engine,
            pattern=pattern,
            activation=activation,
            transpose_b=transpose_b,
            **kwargs,
        )
        return output.reshape(batch, rows, int(output.shape[-1]))
    if b.ndim != 3:
        raise XQTBackendError(
            f"batched_gemm_with_precision expects b to be 2D or 3D, got shape={tuple(b.shape)}"
        )
    if int(b.shape[0]) != batch:
        raise XQTBackendError(
            f"batched GEMM batch mismatch: a batch={batch}, b batch={int(b.shape[0])}"
        )

    outputs: list[torch.Tensor] = []
    for index in range(batch):
        batch_bias = _select_batched_bias(bias, index)
        outputs.append(
            gemm_with_precision(
                a[index],
                b[index],
                batch_bias,
                precision=precision,
                engine=engine,
                pattern=pattern,
                activation=activation,
                transpose_b=transpose_b,
                **kwargs,
            )
        )
    return torch.stack(outputs, dim=0)


def grouped_gemm_with_precision(
    a_groups: Sequence[torch.Tensor],
    b_groups: Sequence[torch.Tensor],
    bias_groups: Sequence[torch.Tensor | None] | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    transpose_b: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """Grouped GEMM composition over the configured 2D GEMM dispatcher."""

    if len(a_groups) != len(b_groups):
        raise XQTBackendError(
            f"grouped GEMM requires matching a/b group counts, got {len(a_groups)} and {len(b_groups)}"
        )
    if bias_groups is not None and len(bias_groups) != len(a_groups):
        raise XQTBackendError(
            f"grouped GEMM bias group count must match inputs, got {len(bias_groups)} and {len(a_groups)}"
        )

    outputs: list[torch.Tensor] = []
    for index, (group_a, group_b) in enumerate(zip(a_groups, b_groups, strict=True)):
        group_bias = None if bias_groups is None else bias_groups[index]
        outputs.append(
            gemm_with_precision(
                group_a,
                group_b,
                group_bias,
                precision=precision,
                engine=engine,
                pattern=pattern,
                activation=activation,
                transpose_b=transpose_b,
                **kwargs,
            )
        )
    return tuple(outputs)


def expert_gemm_with_precision(
    token_groups: Sequence[torch.Tensor],
    expert_weights: Sequence[torch.Tensor],
    expert_biases: Sequence[torch.Tensor | None] | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    transpose_b: bool = True,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """MoE expert GEMM alias over grouped GEMM composition."""

    return grouped_gemm_with_precision(
        token_groups,
        expert_weights,
        expert_biases,
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        transpose_b=transpose_b,
        **kwargs,
    )


def attention_score_gemm_with_precision(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    scale: float | torch.Tensor | None = None,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Attention score GEMM: ``Q[..., Sq, Dh] x K[..., Sk, Dh]^T``."""

    flat_q, prefix, q_len, head_dim = _flatten_attention_operand(q, "q")
    flat_k, k_prefix, k_len, k_dim = _flatten_attention_operand(k, "k")
    if prefix != k_prefix:
        raise XQTBackendError(
            f"attention score GEMM prefix mismatch: q prefix={prefix}, k prefix={k_prefix}"
        )
    if head_dim != k_dim:
        raise XQTBackendError(
            f"attention score GEMM head-dim mismatch: q Dh={head_dim}, k Dh={k_dim}"
        )
    scores = batched_gemm_with_precision(
        flat_q,
        flat_k,
        precision=precision,
        engine=engine,
        pattern=pattern,
        transpose_b=True,
        **kwargs,
    ).reshape(*prefix, q_len, k_len)
    if scale is None:
        return scores
    return scores * scale


def attention_value_gemm_with_precision(
    probabilities: torch.Tensor,
    value: torch.Tensor,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Attention value GEMM: ``P[..., Sq, Sk] x V[..., Sk, Dh]``."""

    flat_p, prefix, q_len, k_len = _flatten_attention_operand(probabilities, "p")
    flat_v, v_prefix, v_len, head_dim = _flatten_attention_operand(value, "v")
    if prefix != v_prefix:
        raise XQTBackendError(
            f"attention value GEMM prefix mismatch: p prefix={prefix}, v prefix={v_prefix}"
        )
    if k_len != v_len:
        raise XQTBackendError(
            f"attention value GEMM sequence mismatch: p Sk={k_len}, v Sk={v_len}"
        )
    return batched_gemm_with_precision(
        flat_p,
        flat_v,
        precision=precision,
        engine=engine,
        pattern=pattern,
        transpose_b=False,
        **kwargs,
    ).reshape(*prefix, q_len, head_dim)


def router_gemm_with_precision(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Router logits GEMM for MoE gates over ``x[..., hidden]``."""

    return projection_gemm_with_precision(
        x,
        router_weight,
        bias,
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        transpose_b=True,
        **kwargs,
    )


def projection_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    transpose_b: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Projection GEMM over ``x[..., hidden]`` using a configured 2D GEMM."""

    flat_x, prefix, hidden = _flatten_projection_operand(x, "x")
    if weight.ndim != 2:
        raise XQTBackendError(
            f"projection GEMM expects 2D weight, got shape={tuple(weight.shape)}"
        )
    expected_hidden = int(weight.shape[1] if transpose_b else weight.shape[0])
    if expected_hidden != hidden:
        raise XQTBackendError(
            f"projection GEMM hidden mismatch: x hidden={hidden}, weight hidden={expected_hidden}"
        )
    output = gemm_with_precision(
        flat_x,
        weight,
        bias,
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        transpose_b=transpose_b,
        **kwargs,
    )
    return output.reshape(*prefix, int(output.shape[-1]))


def q_proj_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Q projection GEMM alias over ``projection_gemm_with_precision``."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def k_proj_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """K projection GEMM alias over ``projection_gemm_with_precision``."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def v_proj_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """V projection GEMM alias over ``projection_gemm_with_precision``."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def qkv_projection_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Packed QKV projection GEMM; output remains flattened as ``[..., 3*inner]``."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def o_proj_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Attention output projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def o_projection_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Attention output projection GEMM alias using the table name."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def ffn_up_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """FFN up projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def ffn_gate_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """FFN gate projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def ffn_down_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """FFN down projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def lm_head_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Vocabulary projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def router_logits_gemm_with_precision(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """MoE router logits GEMM alias using the table name."""

    return router_gemm_with_precision(x, router_weight, bias, **kwargs)


def shared_expert_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Shared expert dense GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def head_cls_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Detection classification head GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def head_box_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Detection box head GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def patch_embed_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Patch embedding projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def dfl_projection_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """DFL projection GEMM alias."""

    return projection_gemm_with_precision(x, weight, bias, **kwargs)


def conv2d_as_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Lower a groups=1 NCHW Conv2d into GEMM plus optional epilogue."""

    if int(groups) != 1:
        raise XQTBackendError("conv2d_as_gemm_with_precision currently supports groups=1")
    stride_pair = _normalize_int_pair(stride, "stride")
    padding_pair = _normalize_int_pair(padding, "padding")
    dilation_pair = _normalize_int_pair(dilation, "dilation")
    weight_matrix, kernel_size = _conv2d_weight_to_matrix(weight)
    if kernel_size == (1, 1):
        return conv1x1_as_gemm_with_precision(
            x,
            weight_matrix,
            bias,
            stride=stride_pair,
            padding=padding_pair,
            dilation=dilation_pair,
            precision=precision,
            engine=engine,
            pattern=pattern,
            activation=activation,
            **kwargs,
        )
    return _conv2d_im2col_gemm_with_precision(
        x,
        weight_matrix,
        bias,
        kernel_size=kernel_size,
        stride=stride_pair,
        padding=padding_pair,
        dilation=dilation_pair,
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        **kwargs,
    )


def conv1x1_as_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Lower NCHW 1x1 Conv2d into GEMM plus optional bias/activation."""

    stride_pair = _normalize_int_pair(stride, "stride")
    padding_pair = _normalize_int_pair(padding, "padding")
    dilation_pair = _normalize_int_pair(dilation, "dilation")
    weight_matrix, kernel_size = _conv2d_weight_to_matrix(weight)
    if kernel_size != (1, 1):
        raise XQTBackendError(
            f"conv1x1_as_gemm_with_precision expects 1x1 weight, got kernel={kernel_size}"
        )
    if stride_pair != (1, 1) or padding_pair != (0, 0) or dilation_pair != (1, 1):
        return _conv2d_im2col_gemm_with_precision(
            x,
            weight_matrix,
            bias,
            kernel_size=kernel_size,
            stride=stride_pair,
            padding=padding_pair,
            dilation=dilation_pair,
            precision=precision,
            engine=engine,
            pattern=pattern,
            activation=activation,
            **kwargs,
        )
    if x.ndim != 4:
        raise XQTBackendError(
            f"conv1x1_as_gemm_with_precision expects NCHW input, got shape={tuple(x.shape)}"
        )
    batch, in_channels, height, width = (int(dim) for dim in x.shape)
    if int(weight_matrix.shape[1]) != in_channels:
        raise XQTBackendError(
            f"conv1x1 GEMM channel mismatch: x channels={in_channels}, weight channels={int(weight_matrix.shape[1])}"
        )
    flat = x.permute(0, 2, 3, 1).contiguous().reshape(batch * height * width, in_channels)
    output = gemm_with_precision(
        flat,
        weight_matrix,
        bias,
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        transpose_b=True,
        **kwargs,
    )
    return output.reshape(batch, height, width, int(output.shape[-1])).permute(0, 3, 1, 2).contiguous()


def conv3x3_im2col_gemm_with_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
    pattern: str | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Lower NCHW 3x3 Conv2d into unfold/im2col plus configured GEMM."""

    if int(groups) != 1:
        raise XQTBackendError("conv3x3_im2col_gemm_with_precision currently supports groups=1")
    weight_matrix, kernel_size = _conv2d_weight_to_matrix(weight)
    if kernel_size != (3, 3):
        raise XQTBackendError(
            f"conv3x3_im2col_gemm_with_precision expects 3x3 weight, got kernel={kernel_size}"
        )
    return _conv2d_im2col_gemm_with_precision(
        x,
        weight_matrix,
        bias,
        kernel_size=kernel_size,
        stride=_normalize_int_pair(stride, "stride"),
        padding=_normalize_int_pair(padding, "padding"),
        dilation=_normalize_int_pair(dilation, "dilation"),
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        **kwargs,
    )


def _flatten_projection_operand(
    tensor: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    if tensor.ndim < 2:
        raise XQTBackendError(
            f"projection operand {name!r} expects rank >= 2, got shape={tuple(tensor.shape)}"
        )
    prefix = tuple(int(dim) for dim in tensor.shape[:-1])
    hidden = int(tensor.shape[-1])
    flattened = 1
    for dim in prefix:
        flattened *= dim
    return tensor.reshape(flattened, hidden), prefix, hidden


def _normalize_int_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise XQTBackendError(f"{name} must be an int or a pair of ints, got {value!r}")
    return pair[0], pair[1]


def _conv2d_weight_to_matrix(weight: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if weight.ndim == 2:
        return weight, (1, 1)
    if weight.ndim != 4:
        raise XQTBackendError(
            f"Conv GEMM lowering expects 2D or 4D weight, got shape={tuple(weight.shape)}"
        )
    out_channels = int(weight.shape[0])
    kernel_size = int(weight.shape[2]), int(weight.shape[3])
    return weight.reshape(out_channels, -1), kernel_size


def _conv_output_hw(
    height: int,
    width: int,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    out_h = (
        height
        + 2 * padding[0]
        - dilation[0] * (kernel_size[0] - 1)
        - 1
    ) // stride[0] + 1
    out_w = (
        width
        + 2 * padding[1]
        - dilation[1] * (kernel_size[1] - 1)
        - 1
    ) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise XQTBackendError(
            f"Conv GEMM lowering produced invalid output size {(out_h, out_w)}"
        )
    return out_h, out_w


def _conv2d_im2col_gemm_with_precision(
    x: torch.Tensor,
    weight_matrix: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    precision: str | MatmulPrecisionSpec | Mapping[str, Any],
    engine: str,
    pattern: str | None,
    activation: str | None,
    **kwargs: Any,
) -> torch.Tensor:
    if x.ndim != 4:
        raise XQTBackendError(
            f"Conv GEMM lowering expects NCHW input, got shape={tuple(x.shape)}"
        )
    batch, in_channels, height, width = (int(dim) for dim in x.shape)
    expected_k = in_channels * kernel_size[0] * kernel_size[1]
    if int(weight_matrix.shape[1]) != expected_k:
        raise XQTBackendError(
            f"Conv GEMM lowering channel mismatch: unfolded K={expected_k}, weight K={int(weight_matrix.shape[1])}"
        )
    out_h, out_w = _conv_output_hw(
        height,
        width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    unfolded = F.unfold(
        x,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    patches = unfolded.transpose(1, 2).contiguous().reshape(-1, expected_k)
    output = gemm_with_precision(
        patches,
        weight_matrix,
        bias,
        precision=precision,
        engine=engine,
        pattern=pattern,
        activation=activation,
        transpose_b=True,
        **kwargs,
    )
    return (
        output.reshape(batch, out_h * out_w, int(output.shape[-1]))
        .transpose(1, 2)
        .contiguous()
        .reshape(batch, int(output.shape[-1]), out_h, out_w)
    )


def _flatten_attention_operand(
    tensor: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    if tensor.ndim < 3:
        raise XQTBackendError(
            f"attention operand {name!r} expects rank >= 3, got shape={tuple(tensor.shape)}"
        )
    prefix = tuple(int(dim) for dim in tensor.shape[:-2])
    rows = int(tensor.shape[-2])
    cols = int(tensor.shape[-1])
    if not prefix:
        raise XQTBackendError(
            f"attention operand {name!r} expects at least one batch/head prefix dimension"
        )
    flattened_batch = 1
    for dim in prefix:
        flattened_batch *= dim
    return tensor.reshape(flattened_batch, rows, cols), prefix, rows, cols


def _select_batched_bias(bias: torch.Tensor | None, index: int) -> torch.Tensor | None:
    if bias is None:
        return None
    if bias.ndim == 1:
        return bias
    if bias.ndim == 2:
        return bias[index]
    raise XQTBackendError(
        f"batched GEMM bias must be 1D shared or 2D per-batch, got shape={tuple(bias.shape)}"
    )


def _select_engine(
    precision: MatmulPrecisionSpec,
    device: torch.device,
    *,
    shape: GemmShape,
    fused_ops: frozenset[str] = frozenset(),
) -> str:
    """Auto-select best XQT engine for given precision, device, and shape.

    Delegates to gemm_selector.select_gemm_engine and keeps only the chosen
    engine name; gemm_with_precision's call site does not need the full
    GemmEngineSelection candidate/rationale payload.
    """
    return select_gemm_engine(
        precision=precision,
        shape=shape,
        device=device,
        fused_ops=fused_ops,
    ).selected_engine


def _gemm_shape_from_tensors(
    a: torch.Tensor, b: torch.Tensor, transpose_b: bool
) -> GemmShape:
    """Derive (m, n, k) from the tensors gemm_with_precision was called with.

    Mirrors the same shape-extraction convention already used inline by
    _gemm_triton/_gemm_tilelang below: 2D a/b, transpose_b selects which of
    b's two dimensions is n.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"gemm_with_precision expects 2D tensors, got a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}"
        )
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[0]) if transpose_b else int(b.shape[1])
    return GemmShape(m=m, n=n, k=k)


def _fused_ops_from_call(
    a: torch.Tensor,
    precision_mma: str,
    activation: str | None,
) -> frozenset[str]:
    """Infer the fused_ops set implied by one gemm_with_precision call.

    Only int8 currently has more than one input contract in this
    dispatcher (see gemm_selector's int8 candidates): a.dtype != torch.int8
    means the caller has not pre-quantized the activation, i.e. it needs
    activation quantization fused into the kernel. This is read directly
    from the real tensor dtype rather than guessed from which kwargs are
    present, since e.g. a_scale can be supplied for a weight-only int8 call
    where `a` is still a float tensor.
    """
    ops: set[str] = set()
    if activation is not None:
        ops.add(activation)
    if precision_mma == "int8" and a.dtype != torch.int8:
        ops.add("activation_quant")
    return frozenset(ops)


def _shared_runtime_kwargs(
    precision: MatmulPrecisionSpec,
    activation: str | None,
    kwargs: Mapping[str, Any],
    *,
    include_output_dtype: bool = True,
) -> dict[str, Any]:
    runtime_kwargs = dict(kwargs)
    if activation is not None:
        runtime_kwargs["activation"] = activation
    if include_output_dtype:
        runtime_kwargs.setdefault(
            "output_dtype",
            _precision_name_to_dtype(
                precision.output,
                role="output",
                fallback=torch.float16 if precision.mma in {"fp4", "nvfp4", "int8"} else None,
            ),
        )
    return runtime_kwargs


def _tilelang_dense_linear_dispatch(
    pattern: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    if not transpose_b:
        raise XQTBackendError(f"TileLang pattern '{pattern}' requires transpose_b=True")
    if a.ndim != 2 or b.ndim != 2:
        raise XQTBackendError(f"TileLang pattern '{pattern}' expects 2D GEMM inputs")
    if int(a.shape[1]) != int(b.shape[1]):
        raise XQTBackendError(
            f"TileLang dense GEMM inner-dimension mismatch: a.shape[1]={int(a.shape[1])}, "
            f"b.shape[1]={int(b.shape[1])}"
        )
    compute_dtype = _precision_name_to_dtype(precision.mma, role="mma")
    if compute_dtype not in {torch.float16, torch.bfloat16}:
        raise XQTBackendError(
            f"TileLang dense GEMM supports only fp16 or bf16 MMA, got {precision.mma}"
        )
    named_tensors = [("activation", a), ("weight", b)]
    if bias is not None:
        named_tensors.append(("bias", bias))
    mismatched = [
        f"{name}={tensor.dtype}"
        for name, tensor in named_tensors
        if tensor.dtype != compute_dtype
    ]
    if mismatched:
        raise XQTBackendError(
            "TileLang dense GEMM requires activation, weight, and bias dtypes "
            f"to match {compute_dtype}: {', '.join(mismatched)}"
        )
    output_dtype = _precision_name_to_dtype(precision.output, role="output")
    if output_dtype != compute_dtype:
        raise XQTBackendError(
            "TileLang dense GEMM requires output precision to match the fp16/bf16 MMA dtype"
        )
    weight = b
    runtime_kwargs = _shared_runtime_kwargs(
        precision,
        activation,
        kwargs,
        include_output_dtype=False,
    )
    if pattern == "linear_marlin":
        runtime_kwargs["precision"] = precision.mma
        return run_tilelang_kernel(
            pattern,
            a,
            weight,
            None,
            bias,
            **runtime_kwargs,
        )
    return run_tilelang_kernel(
        pattern,
        a,
        weight,
        bias,
        **runtime_kwargs,
    )


def _tilelang_int8_dispatch(
    pattern: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    if not transpose_b:
        raise XQTBackendError("TileLang int8 GEMM requires transpose_b=True")
    if pattern == "int8_mma":
        if activation is not None:
            raise XQTBackendError("TileLang int8_mma does not support activation epilogues")
        if bias is not None:
            raise XQTBackendError("TileLang int8_mma does not support bias epilogues")
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise XQTBackendError("TileLang int8_mma expects int8 activation and weight tensors")
        runtime_kwargs = {
            key: value for key, value in kwargs.items() if key not in {"a_scale", "b_scale"}
        }
        return run_tilelang_kernel(
            "int8_mma",
            a,
            b.t().contiguous(),
            **runtime_kwargs,
        )
    if activation is not None:
        raise XQTBackendError(
            "TileLang int8 GEMM does not yet fuse activation epilogues; "
            "use triton or apply the activation at a higher composition layer"
        )
    a_scale = kwargs.get("a_scale")
    b_scale = kwargs.get("b_scale")
    runtime_kwargs = {
        key: value for key, value in kwargs.items() if key not in {"a_scale", "b_scale"}
    }
    if pattern == "int8_linear":
        if a.dtype != torch.int8:
            raise XQTBackendError(
                "TileLang int8_linear expects pre-quantized torch.int8 activations"
            )
        if b.dtype != torch.int8:
            raise XQTBackendError(
                "TileLang int8 GEMM expects torch.int8 weights for true W8A8 dispatch"
            )
        if a_scale is None or b_scale is None:
            raise ValueError("TileLang int8 GEMM requires a_scale and b_scale")
        return run_tilelang_kernel(
            "int8_linear",
            a,
            b.t().contiguous(),
            a_scale,
            b_scale,
            bias,
            output_dtype=_precision_name_to_dtype(
                precision.output, role="output", fallback=torch.float16
            ),
            **runtime_kwargs,
        )
    if pattern != "int8_linear_static_activation":
        raise XQTBackendError(f"Unsupported TileLang int8 pattern: {pattern}")
    if a.dtype == torch.int8:
        raise XQTBackendError(
            "TileLang int8_linear_static_activation expects fp16, bf16, or fp32 activations"
        )
    if b.dtype != torch.int8:
        raise XQTBackendError("TileLang fused activation-quant int8 GEMM expects torch.int8 weights")
    if a_scale is None or b_scale is None:
        raise ValueError("TileLang fused activation-quant int8 GEMM requires a_scale and b_scale")
    return run_tilelang_kernel(
        "int8_linear_static_activation",
        a,
        b.t().contiguous(),
        a_scale,
        b_scale,
        bias,
        output_dtype=_precision_name_to_dtype(
            precision.output, role="output", fallback=torch.float16
        ),
        **runtime_kwargs,
    )


def _tilelang_packed_dispatch(
    pattern: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    if not transpose_b:
        raise XQTBackendError(f"packed {precision.mma} TileLang GEMM requires transpose_b=True")
    runtime_kwargs = _shared_runtime_kwargs(
        precision,
        activation,
        kwargs,
        include_output_dtype=False,
    )
    scale = runtime_kwargs.pop("b_scale", None)
    if scale is None:
        raise ValueError(f"{precision.mma} precision requires b_scale")
    return run_tilelang_kernel(
        pattern,
        a,
        b,
        scale,
        bias,
        **runtime_kwargs,
    )


def _tilelang_dequant_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    if not transpose_b:
        raise XQTBackendError("TileLang dequant GEMM requires transpose_b=True")
    if a.ndim != 2 or b.ndim != 2:
        raise XQTBackendError("TileLang dequant GEMM expects 2D GEMM inputs")
    if int(a.shape[1]) != int(b.shape[1]):
        raise XQTBackendError(
            f"TileLang dequant GEMM inner-dimension mismatch: a.shape[1]={int(a.shape[1])}, "
            f"b.shape[1]={int(b.shape[1])}"
        )
    if b.dtype not in {torch.int8, torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError(
            "TileLang dequant GEMM expects qweight dtype in {int8, fp16, bf16, fp32}"
        )
    scale = kwargs.get("b_scale")
    if scale is None:
        raise ValueError("TileLang dequant GEMM requires b_scale")
    runtime_kwargs = _shared_runtime_kwargs(
        precision,
        activation,
        {key: value for key, value in kwargs.items() if key != "b_scale"},
        include_output_dtype=False,
    )
    qweight = b.to(dtype=a.dtype, device=a.device)
    scale_tensor = scale.to(dtype=a.dtype, device=a.device)
    runtime_bias = None if bias is None else bias.to(dtype=a.dtype, device=a.device)
    return run_tilelang_kernel(
        "dequant_gemm_epilogue",
        a,
        qweight,
        scale_tensor,
        runtime_bias,
        **runtime_kwargs,
    )


def _tilelang_marlin_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: Mapping[str, Any],
) -> torch.Tensor:
    if not transpose_b:
        raise XQTBackendError("TileLang Marlin GEMM requires transpose_b=True")
    if a.ndim != 2 or b.ndim != 2:
        raise XQTBackendError("TileLang Marlin GEMM expects 2D GEMM inputs")
    if precision.mma in {"int8", "int4"}:
        scale = kwargs.get("b_scale")
        if scale is None:
            raise ValueError(f"TileLang Marlin {precision.mma} GEMM requires b_scale")
    else:
        scale = None
    runtime_kwargs = _shared_runtime_kwargs(
        precision,
        activation,
        {key: value for key, value in kwargs.items() if key != "b_scale"},
        include_output_dtype=False,
    )
    runtime_kwargs.setdefault("precision", precision.mma)
    if scale is None:
        return run_tilelang_kernel(
            "linear_marlin",
            a,
            b,
            None,
            bias,
            **runtime_kwargs,
        )
    return run_tilelang_kernel(
        "linear_marlin",
        a,
        b,
        scale,
        bias,
        **runtime_kwargs,
    )


def _triton_dispatch(
    pattern: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: Mapping[str, Any],
    *,
    static_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    runtime_kwargs = _shared_runtime_kwargs(
        precision,
        activation,
        kwargs,
        include_output_dtype=pattern in {"gemm_fp16", "gemm_bf16"},
    )
    runtime_kwargs["transpose_b"] = transpose_b
    if pattern in {"gemm_fp16", "gemm_bf16"}:
        runtime_kwargs.setdefault(
            "accum_dtype",
            _precision_name_to_dtype(precision.accum, role="accum"),
        )
        runtime_kwargs["output_dtype"] = _precision_name_to_dtype(
            precision.output,
            role="output",
        )
    if static_kwargs:
        runtime_kwargs.update(static_kwargs)
    match pattern:
        case "gemm_int8" | "gemm_fp8":
            a_scale = runtime_kwargs.pop("a_scale", None)
            b_scale = runtime_kwargs.pop("b_scale", None)
            return run_triton_kernel(
                pattern,
                a,
                b,
                a_scale,
                b_scale,
                bias,
                **runtime_kwargs,
            )
        case "gemm_int4_dequant":
            b_scale = runtime_kwargs.pop("b_scale", None)
            if b_scale is None:
                raise ValueError("int4 precision requires b_scale")
            b_zero = runtime_kwargs.pop("b_zero", None)
            return run_triton_kernel(
                pattern,
                a,
                b,
                b_scale,
                b_zero,
                bias,
                **runtime_kwargs,
            )
        case "gemm_mxfp8" | "gemm_mxfp6" | "gemm_mxfp4":
            b_scales = runtime_kwargs.pop("b_scales", None)
            if b_scales is None:
                raise ValueError(f"{precision.mma} precision requires b_scales")
            runtime_kwargs.pop("output_dtype", None)
            return run_triton_kernel(
                pattern,
                a,
                b,
                b_scales,
                bias,
                **runtime_kwargs,
            )
        case _:
            return run_triton_kernel(
                pattern,
                a,
                b,
                bias,
                **runtime_kwargs,
            )


def _gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
    pattern: str | None = None,
) -> torch.Tensor:
    """Triton engine dispatcher shared by multiple GEMM families."""
    _require_supported_activation(activation)
    family = _resolve_gemm_family_spec(precision)
    resolved_pattern = _resolve_requested_pattern(family, "triton", pattern)
    return _triton_dispatch(
        resolved_pattern,
        a,
        b,
        bias,
        precision,
        activation,
        transpose_b,
        kwargs,
        static_kwargs=family.engine_kwargs.get("triton"),
    )


def _gemm_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
    pattern: str | None = None,
) -> torch.Tensor:
    """TileLang engine dispatcher shared by dense and packed GEMM families."""
    _require_supported_activation(activation)
    family = _resolve_gemm_family_spec(precision)
    resolved_pattern = _resolve_requested_pattern(family, "tilelang", pattern)
    merged_kwargs = dict(family.engine_kwargs.get("tilelang", {}))
    merged_kwargs.update(kwargs)
    if (
        pattern is None
        and resolved_pattern == "int8_linear"
        and a.dtype != torch.int8
    ):
        resolved_pattern = "int8_linear_static_activation"

    match resolved_pattern:
        case "dense_linear_epilogue" | "linear":
            return _tilelang_dense_linear_dispatch(
                resolved_pattern,
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                merged_kwargs,
            )
        case "int8_mma" | "int8_linear" | "int8_linear_static_activation":
            return _tilelang_int8_dispatch(
                resolved_pattern,
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                merged_kwargs,
            )
        case "dequant_gemm_epilogue":
            return _tilelang_dequant_dispatch(
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                merged_kwargs,
            )
        case "linear_marlin":
            return _tilelang_marlin_dispatch(
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                merged_kwargs,
            )
        case (
            "fp4_packed_dequant_gemm_epilogue"
            | "mxfp4_packed_dequant_gemm_epilogue"
            | "nvfp4_packed_dequant_gemm_epilogue"
        ):
            return _tilelang_packed_dispatch(
                resolved_pattern,
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                merged_kwargs,
            )
        case _:
            raise XQTBackendError(
                f"TileLang precision {precision.mma} is not wired for pattern "
                f"{resolved_pattern}"
            )


def _gemm_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """PyTorch fallback dispatcher."""
    del kwargs
    compute_dtype = _precision_name_to_dtype(
        precision.mma,
        role="mma",
    )
    lhs = a.to(
        _precision_name_to_dtype(
            precision.activation,
            fallback=compute_dtype,
            role="activation",
        )
    )
    rhs = b.to(
        _precision_name_to_dtype(
            precision.weight,
            fallback=compute_dtype,
            role="weight",
        )
    )
    bias_value = (
        None
        if bias is None
        else bias.to(
            _precision_name_to_dtype(
                precision.bias,
                fallback=compute_dtype,
                role="bias",
            )
        )
    )
    output = dense_gemm_reference(
        lhs, rhs, bias_value, activation=activation, transpose_b=transpose_b
    )
    return output.to(_precision_name_to_dtype(precision.output, role="output"))


def _execute_registered_gemm_engine(
    engine: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    runtime_kwargs: dict[str, Any],
    *,
    pattern: str | None,
) -> torch.Tensor:
    """Execute one engine selected by the canonical GEMM registry."""

    match engine:
        case "triton":
            return _gemm_triton(
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                runtime_kwargs,
                pattern=pattern,
            )
        case "tilelang":
            return _gemm_tilelang(
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                runtime_kwargs,
                pattern=pattern,
            )
        case "torch":
            return _gemm_torch(
                a,
                b,
                bias,
                precision,
                activation,
                transpose_b,
                runtime_kwargs,
            )
        case _:
            raise XQTBackendError(f"Unsupported registered GEMM engine: {engine}")


def describe_gemm_precision_capability(
    precision: str,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Return capability description for a given precision on device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    capability = {
        "precision": precision,
        "device": str(device),
        "available": False,
        "engine": "none",
        "hardware_native": False,
        "notes": [],
    }

    if device.type != "cuda":
        capability["notes"].append("CUDA required for most precision modes")
        match precision:
            case "fp16" | "bf16":
                capability["available"] = True
                capability["engine"] = "torch"
            case "fp4" | "nvfp4":
                capability["available"] = True
                capability["engine"] = "tilelang"
                capability["notes"].append(
                    "FP4/NVFP4 can use the TileLang reference path without CUDA native kernels"
                )
        return capability

    # 检查CUDA compute capability
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor

        match precision:
            case "fp16":
                capability["available"] = True
                capability["engine"] = "triton"
                capability["hardware_native"] = sm >= 70
                if sm >= 70:
                    capability["notes"].append(
                        "Tensor Core FP16 MMA available (SM70+)"
                    )
            case "bf16":
                capability["available"] = True
                capability["engine"] = "triton"
                capability["hardware_native"] = sm >= 80
                if sm >= 80:
                    capability["notes"].append(
                        "Tensor Core BF16 MMA available (SM80+ Ampere)"
                    )
            case "int8":
                capability["available"] = True
                capability["engine"] = "triton"
                capability["hardware_native"] = sm >= 75
                if sm >= 75:
                    capability["notes"].append(
                        "Tensor Core INT8 MMA available (SM75+ Turing)"
                    )
            case "fp8":
                if sm >= 89:
                    capability["available"] = True
                    capability["engine"] = "triton"
                    capability["hardware_native"] = True
                    capability["notes"].append(
                        "Tensor Core FP8 path available on Ada or newer NVIDIA architectures"
                    )
                else:
                    capability["notes"].append(
                        "FP8 requires SM89+ or newer NVIDIA architecture support"
                    )
            case "int4":
                capability["available"] = True
                capability["engine"] = "triton"
                capability["hardware_native"] = False
                capability["notes"].append("INT4 via unpacking + FP16 MMA")
            case "fp4" | "nvfp4":
                capability["available"] = True
                capability["engine"] = "tilelang"
                capability["hardware_native"] = sm >= 100
                if sm >= 100:
                    capability["notes"].append(
                        "FP4/NVFP4 tensor-core path is a Blackwell-first target"
                    )
                else:
                    capability["notes"].append(
                        "FP4/NVFP4 requires packed weight contracts and fused dequant GEMM kernels"
                    )
            case "mxfp8" | "mxfp6" | "mxfp4":
                if sm >= 100:
                    capability["available"] = True
                    capability["engine"] = "triton"
                    capability["hardware_native"] = True
                    capability["notes"].append(
                        "MXFP native support on Blackwell-class NVIDIA architectures"
                    )
                else:
                    capability["available"] = True
                    capability["engine"] = "triton"
                    capability["hardware_native"] = False
                    capability["notes"].append("MXFP emulated via block scaling")

    return capability


def list_available_precisions(device: torch.device | None = None) -> list[str]:
    """List all precisions available on the given device."""
    precisions = [
        "fp16",
        "bf16",
        "int8",
        "fp8",
        "int4",
        "fp4",
        "nvfp4",
        "mxfp8",
        "mxfp6",
        "mxfp4",
    ]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available = []
    for precision in precisions:
        cap = describe_gemm_precision_capability(precision, device)
        if cap["available"]:
            available.append(precision)

    return available


__all__ = [
    "attention_score_gemm_with_precision",
    "attention_value_gemm_with_precision",
    "batched_gemm_with_precision",
    "conv1x1_as_gemm_with_precision",
    "conv2d_as_gemm_with_precision",
    "conv3x3_im2col_gemm_with_precision",
    "describe_gemm_precision_capability",
    "dfl_projection_gemm_with_precision",
    "expert_gemm_with_precision",
    "ffn_down_gemm_with_precision",
    "ffn_gate_gemm_with_precision",
    "ffn_up_gemm_with_precision",
    "gemm_with_precision",
    "grouped_gemm_with_precision",
    "head_box_gemm_with_precision",
    "head_cls_gemm_with_precision",
    "k_proj_gemm_with_precision",
    "list_available_precisions",
    "list_gemm_variant_dispatch_specs",
    "lm_head_gemm_with_precision",
    "MatmulPrecisionSpec",
    "o_proj_gemm_with_precision",
    "o_projection_gemm_with_precision",
    "patch_embed_gemm_with_precision",
    "gemm_variant_with_precision",
    "projection_gemm_with_precision",
    "q_proj_gemm_with_precision",
    "qkv_projection_gemm_with_precision",
    "router_gemm_with_precision",
    "router_logits_gemm_with_precision",
    "shared_expert_gemm_with_precision",
    "v_proj_gemm_with_precision",
]
