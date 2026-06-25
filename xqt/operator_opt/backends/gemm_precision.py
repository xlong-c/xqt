"""Unified GEMM precision dispatcher for XQT operator optimization."""

from __future__ import annotations

from typing import Any

import torch

from xqt.core.errors import XQTBackendError


def gemm_with_precision(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: str = "fp16",
    backend: str = "triton",
    activation: str | None = None,
    transpose_b: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Unified GEMM dispatcher with precision control.

    Args:
        a: Left matrix (M, K)
        b: Right matrix (N, K) if transpose_b else (K, N)
        bias: Optional bias vector (N,)
        precision: Precision mode - "fp16", "bf16", "int8", "fp8", "int4", "mxfp8", "mxfp6", "mxfp4"
        backend: Backend - "triton", "tilelang", "torch", "auto"
        activation: Optional activation - "relu", "gelu", "silu"
        transpose_b: Whether to transpose b before matmul
        **kwargs: Backend-specific parameters (scales, group_size, etc.)

    Returns:
        Output tensor (M, N)
    """
    if backend == "auto":
        backend = _select_backend(precision, a.device)

    if backend == "triton":
        return _gemm_triton(a, b, bias, precision, activation, transpose_b, kwargs)
    elif backend == "tilelang":
        return _gemm_tilelang(a, b, bias, precision, activation, transpose_b, kwargs)
    elif backend == "torch":
        return _gemm_torch(a, b, bias, precision, activation, transpose_b, kwargs)
    else:
        raise XQTBackendError(f"Unsupported GEMM backend: {backend}")


def _select_backend(precision: str, device: torch.device) -> str:
    """Auto-select best backend for given precision and device."""
    if not device.type == "cuda":
        return "torch"

    # Triton优先（覆盖面广）
    if precision in {"fp16", "bf16", "int8", "fp8", "mxfp8", "mxfp6", "mxfp4"}:
        return "triton"

    # INT4需要特殊处理
    if precision == "int4":
        return "triton"

    return "torch"


def _gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: str,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Triton backend dispatcher."""
    from ..kernels.triton.gemm import (
        gemm_bf16_triton,
        gemm_fp16_triton,
        gemm_fp8_triton,
        gemm_int4_dequant_triton,
        gemm_int8_triton,
    )

    if precision == "fp16":
        return gemm_fp16_triton(a, b, bias, activation=activation, transpose_b=transpose_b, **kwargs)
    elif precision == "bf16":
        return gemm_bf16_triton(a, b, bias, activation=activation, transpose_b=transpose_b, **kwargs)
    elif precision == "int8":
        a_scale = kwargs.get("a_scale")
        b_scale = kwargs.get("b_scale")
        return gemm_int8_triton(
            a, b, a_scale, b_scale, bias,
            activation=activation,
            transpose_b=transpose_b,
            **{k: v for k, v in kwargs.items() if k not in {"a_scale", "b_scale"}},
        )
    elif precision == "fp8":
        a_scale = kwargs.get("a_scale")
        b_scale = kwargs.get("b_scale")
        fp8_format = kwargs.get("fp8_format", "e4m3")
        return gemm_fp8_triton(
            a, b, a_scale, b_scale, bias,
            activation=activation,
            transpose_b=transpose_b,
            fp8_format=fp8_format,
        )
    elif precision == "int4":
        b_scale = kwargs.get("b_scale")
        b_zero = kwargs.get("b_zero")
        group_size = kwargs.get("group_size", 128)
        if b_scale is None:
            raise ValueError("int4 precision requires b_scale")
        return gemm_int4_dequant_triton(
            a, b, b_scale, b_zero, bias,
            group_size=group_size,
            activation=activation,
        )
    elif precision in {"mxfp8", "mxfp6", "mxfp4"}:
        from ..kernels.triton.mxfp_gemm import gemm_mxfp_triton
        b_scales = kwargs.get("b_scales")
        block_size = kwargs.get("block_size", 32)
        mx_precision = int(precision.replace("mxfp", ""))
        return gemm_mxfp_triton(
            a, b, b_scales, bias,
            mx_precision=mx_precision,
            block_size=block_size,
            activation=activation,
            transpose_b=transpose_b,
        )
    else:
        raise XQTBackendError(f"Unsupported Triton precision: {precision}")


def _gemm_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: str,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """TileLang backend dispatcher."""
    from ..kernels.tilelang.gemm_builder import build_tilelang_gemm_kernel

    if precision == "fp16":
        M, K = a.shape
        if transpose_b:
            N, K_b = b.shape
        else:
            K_b, N = b.shape
        assert K == K_b

        kernel = build_tilelang_gemm_kernel(M, N, K, **kwargs)
        b_t = b.t().contiguous() if transpose_b else b.contiguous()
        output = torch.empty((M, N), device=a.device, dtype=torch.float16)
        kernel(a, b_t, output)

        if bias is not None:
            output = output + bias
        if activation == "relu":
            output = torch.relu(output)
        elif activation == "gelu":
            output = torch.nn.functional.gelu(output)
        elif activation == "silu":
            output = torch.nn.functional.silu(output)

        return output
    else:
        raise XQTBackendError(f"TileLang precision {precision} not implemented yet")


def _gemm_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: str,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """PyTorch fallback dispatcher."""
    from ..kernels.triton.gemm import gemm_reference

    return gemm_reference(a, b, bias, activation=activation, transpose_b=transpose_b)


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
        "backend": "none",
        "hardware_native": False,
        "notes": [],
    }

    if device.type != "cuda":
        capability["notes"].append("CUDA required for most precision modes")
        if precision in {"fp16", "bf16"}:
            capability["available"] = True
            capability["backend"] = "torch"
        return capability

    # 检查CUDA compute capability
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor

        if precision == "fp16":
            capability["available"] = True
            capability["backend"] = "triton"
            capability["hardware_native"] = sm >= 70
            if sm >= 70:
                capability["notes"].append("Tensor Core FP16 MMA available (SM70+)")
        elif precision == "bf16":
            capability["available"] = True
            capability["backend"] = "triton"
            capability["hardware_native"] = sm >= 80
            if sm >= 80:
                capability["notes"].append("Tensor Core BF16 MMA available (SM80+ Ampere)")
        elif precision == "int8":
            capability["available"] = True
            capability["backend"] = "triton"
            capability["hardware_native"] = sm >= 75
            if sm >= 75:
                capability["notes"].append("Tensor Core INT8 MMA available (SM75+ Turing)")
        elif precision == "fp8":
            if sm >= 89:
                capability["available"] = True
                capability["backend"] = "triton"
                capability["hardware_native"] = True
                capability["notes"].append("Tensor Core FP8 MMA available (SM89+ Hopper H100)")
            else:
                capability["notes"].append("FP8 requires SM89+ (Hopper H100)")
        elif precision == "int4":
            capability["available"] = True
            capability["backend"] = "triton"
            capability["hardware_native"] = False
            capability["notes"].append("INT4 via unpacking + FP16 MMA")
        elif precision in {"mxfp8", "mxfp6", "mxfp4"}:
            if sm >= 120:
                capability["available"] = True
                capability["backend"] = "triton"
                capability["hardware_native"] = True
                capability["notes"].append("MXFP native support (SM120+ Blackwell)")
            else:
                capability["available"] = True
                capability["backend"] = "triton"
                capability["hardware_native"] = False
                capability["notes"].append("MXFP emulated via block scaling")

    return capability


def list_available_precisions(device: torch.device | None = None) -> list[str]:
    """List all precisions available on the given device."""
    precisions = ["fp16", "bf16", "int8", "fp8", "int4", "mxfp8", "mxfp6", "mxfp4"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available = []
    for precision in precisions:
        cap = describe_gemm_precision_capability(precision, device)
        if cap["available"]:
            available.append(precision)

    return available


__all__ = [
    "describe_gemm_precision_capability",
    "gemm_with_precision",
    "list_available_precisions",
]
