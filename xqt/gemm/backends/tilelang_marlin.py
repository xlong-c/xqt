"""GEMM registry adapter for the TileLang Marlin-style Linear kernel.

The kernel body lives under xqt.operator_opt because TileLang kernels are
shared across operator optimization. This module owns the GEMM-side contract:
logical A[M,K] @ W[N,K].T validation, PackedWeight unpacking, and registry
promotion into xqt.gemm.
"""

from __future__ import annotations

import torch

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.backends.tilelang import run_tilelang_kernel

from ..contracts import GemmSpec, PackedWeight


def _activation_name(spec: GemmSpec) -> str | None:
    return None if spec.epilogue.activation == "none" else spec.epilogue.activation


def _packed_payload(weight: torch.Tensor | PackedWeight) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(weight, PackedWeight):
        qweight = weight.canonical_qweight if weight.canonical_qweight is not None else weight.qweight
        return qweight, weight.scales
    return weight, None


def _validate_common(
    activation: torch.Tensor,
    weight: torch.Tensor | PackedWeight,
    *,
    spec: GemmSpec,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if spec.problem.op != "dense":
        raise XQTBackendError("TileLang Marlin GEMM supports dense GEMM only")
    if spec.quant.activation_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("TileLang Marlin GEMM requires fp16 or bf16 activations")
    if spec.quant.output_dtype not in {"fp16", "bf16"}:
        raise XQTBackendError("TileLang Marlin GEMM supports fp16 or bf16 output only")
    if spec.epilogue.has_residual or residual is not None:
        raise XQTBackendError("TileLang Marlin GEMM has no residual epilogue")
    if spec.epilogue.has_bias and bias is None:
        raise XQTBackendError("TileLang Marlin GEMM epilogue declares bias but bias is missing")
    if activation.ndim != 2:
        raise XQTBackendError("TileLang Marlin GEMM expects activation [M,K]")
    if tuple(activation.shape) != (spec.problem.m, spec.problem.k):
        raise XQTBackendError("activation shape does not match GemmProblem")

    qweight, scales = _packed_payload(weight)
    if not isinstance(qweight, torch.Tensor):
        raise XQTBackendError("TileLang Marlin GEMM requires tensor weight payload")
    if qweight.ndim != 2:
        raise XQTBackendError("TileLang Marlin GEMM expects weight [N,K] or packed [N,K/2]")
    if int(qweight.shape[0]) != spec.problem.n:
        raise XQTBackendError("weight N dimension does not match GemmProblem")
    if bias is not None and (bias.ndim != 1 or int(bias.shape[0]) != spec.problem.n):
        raise XQTBackendError("TileLang Marlin GEMM bias must be [N]")
    return qweight, scales


def tilelang_marlin_executor(
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
) -> torch.Tensor:
    """Run the TileLang Marlin-style Linear kernel through the GEMM contract."""

    if activation_scales is not None or activation_zero_points is not None:
        raise XQTBackendError("TileLang Marlin GEMM does not support quantized activations")
    if weight_zero_points is not None or spec.quant.weight_zero_point:
        raise XQTBackendError("TileLang Marlin GEMM supports symmetric weights only")

    qweight, packed_scales = _validate_common(
        activation,
        weight,
        spec=spec,
        bias=bias,
        residual=residual,
    )
    scales = weight_scales if weight_scales is not None else packed_scales

    precision = spec.quant.weight_dtype
    if precision in {"fp16", "bf16"}:
        if scales is not None:
            raise XQTBackendError("dense TileLang Marlin GEMM does not consume weight scales")
        if int(qweight.shape[1]) != spec.problem.k:
            raise XQTBackendError("dense TileLang Marlin weight K does not match GemmProblem")
    elif precision in {"int8", "int4"}:
        if scales is None:
            raise XQTBackendError("quantized TileLang Marlin GEMM requires weight scales")
        expected_k = spec.problem.k if precision == "int8" else (spec.problem.k + 1) // 2
        if int(qweight.shape[1]) < expected_k:
            raise XQTBackendError("quantized TileLang Marlin packed weight has too few K columns")
    else:
        raise XQTBackendError(f"unsupported TileLang Marlin GEMM weight dtype: {precision!r}")

    return run_tilelang_kernel(
        "linear_marlin",
        activation,
        qweight,
        scales,
        bias,
        fallback="raise",
        precision=precision,
        input_features=spec.problem.k,
        group_size=spec.quant.group_size or spec.problem.k,
        activation=_activation_name(spec),
    )


__all__ = ["tilelang_marlin_executor"]
