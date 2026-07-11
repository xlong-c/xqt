"""Dtype/precision helpers and type aliases for XQT nn operator facades."""

from __future__ import annotations

from typing import Literal, Mapping

import torch

from xqt.contracts import PrecisionPolicy
from xqt.core.errors import XQTBackendError


NormKind = Literal["layernorm", "rmsnorm"]
ActivationKind = Literal[
    "gelu",
    "gelu-approximate",
    "geglu",
    "geglu-approximate",
    "swiglu",
    "linear-silu",
]
EngineKind = Literal["torch", "triton"]
ProjectionName = Literal["proj_in", "proj_gate", "proj_out"]

_TORCH_DTYPE_PRECISION_NAMES = {"fp16", "bf16", "fp32"}


def _resolve_engine_alias(
    *,
    engine: str | None,
    default: str = "torch",
    context: str,
) -> str:
    del context
    if engine is None:
        return default
    return str(engine).strip().lower()


def _canonical_precision_name(name: str) -> str:
    try:
        return PrecisionPolicy.canonical_name(name, allow_auto=True)
    except XQTBackendError as exc:
        raise ValueError(str(exc)) from exc


def _precision_name_from_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return "fp32"


def _precision_name_to_dtype(
    name: str,
    fallback: torch.dtype,
    *,
    allow_low_bit_fallback: bool = True,
    role: str = "precision",
) -> torch.dtype:
    canonical = _canonical_precision_name(name)
    if canonical == "auto":
        return fallback
    if canonical == "fp16":
        return torch.float16
    if canonical == "bf16":
        return torch.bfloat16
    if canonical == "fp32":
        return torch.float32
    if not allow_low_bit_fallback:
        raise ValueError(f"{role} precision {name} has no torch dtype runtime")
    return fallback


def _compute_output_precision_name(
    precision: Mapping[str, str], fallback: torch.dtype
) -> str:
    name = precision["output"]
    if name != "auto":
        return name
    return _precision_name_from_dtype(fallback)


def _default_runtime_precision() -> dict[str, str]:
    return {
        "activation": "auto",
        "weight": "auto",
        "bias": "auto",
        "mma": "auto",
        "accum": "auto",
        "output": "auto",
    }


def _canonical_precision_key(name: str) -> str:
    try:
        return PrecisionPolicy.canonical_field(name)
    except XQTBackendError as exc:
        raise ValueError(str(exc)) from exc
