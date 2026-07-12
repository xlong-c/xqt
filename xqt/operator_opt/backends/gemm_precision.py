"""Unified GEMM precision dispatcher for XQT operator optimization."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from xqt.contracts import PrecisionPolicy
from xqt.core.errors import XQTBackendError

from .gemm_selector import GemmShape, select_gemm_engine

MatmulPrecisionSpec = PrecisionPolicy


_DTYPE_PRECISIONS: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
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


def gemm_with_precision(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    precision: str | MatmulPrecisionSpec | Mapping[str, Any] = "fp16",
    engine: str = "triton",
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
        activation: Optional activation - "relu", "gelu", "silu"
        transpose_b: Whether to transpose b before matmul
        **kwargs: Engine-specific parameters (scales, group_size, etc.)

    Returns:
        Output tensor (M, N)
    """
    precision_spec = _resolve_matmul_precision(precision)
    resolved_engine = _resolve_gemm_engine(engine=engine)
    if resolved_engine == "auto":
        resolved_engine = _select_engine(
            precision_spec,
            a.device,
            shape=_gemm_shape_from_tensors(a, b, transpose_b),
            fused_ops=_fused_ops_from_call(a, precision_spec.mma, activation),
        )

    if resolved_engine == "triton":
        return _gemm_triton(a, b, bias, precision_spec, activation, transpose_b, kwargs)
    elif resolved_engine == "tilelang":
        return _gemm_tilelang(
            a, b, bias, precision_spec, activation, transpose_b, kwargs
        )
    elif resolved_engine == "torch":
        return _gemm_torch(a, b, bias, precision_spec, activation, transpose_b, kwargs)
    else:
        raise XQTBackendError(f"Unsupported GEMM engine: {resolved_engine}")


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


def _gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Triton engine dispatcher."""
    from ..kernels.triton.gemm import (
        gemm_bf16_triton,
        gemm_fp16_triton,
        gemm_fp8_triton,
        gemm_int4_dequant_triton,
        gemm_int8_triton,
    )

    if precision.mma == "fp16":
        return gemm_fp16_triton(
            a,
            b,
            bias,
            activation=activation,
            transpose_b=transpose_b,
            accum_dtype=_precision_name_to_dtype(precision.accum, role="accum"),
            output_dtype=_precision_name_to_dtype(precision.output, role="output"),
            **kwargs,
        )
    elif precision.mma == "bf16":
        return gemm_bf16_triton(
            a,
            b,
            bias,
            activation=activation,
            transpose_b=transpose_b,
            accum_dtype=_precision_name_to_dtype(precision.accum, role="accum"),
            output_dtype=_precision_name_to_dtype(precision.output, role="output"),
            **kwargs,
        )
    elif precision.mma == "int8":
        a_scale = kwargs.get("a_scale")
        b_scale = kwargs.get("b_scale")
        return gemm_int8_triton(
            a,
            b,
            a_scale,
            b_scale,
            bias,
            activation=activation,
            transpose_b=transpose_b,
            **{k: v for k, v in kwargs.items() if k not in {"a_scale", "b_scale"}},
        )
    elif precision.mma == "fp8":
        a_scale = kwargs.get("a_scale")
        b_scale = kwargs.get("b_scale")
        fp8_format = kwargs.get("fp8_format", "e4m3")
        return gemm_fp8_triton(
            a,
            b,
            a_scale,
            b_scale,
            bias,
            activation=activation,
            transpose_b=transpose_b,
            fp8_format=fp8_format,
        )
    elif precision.mma == "int4":
        b_scale = kwargs.get("b_scale")
        b_zero = kwargs.get("b_zero")
        group_size = kwargs.get("group_size", 128)
        if b_scale is None:
            raise ValueError("int4 precision requires b_scale")
        return gemm_int4_dequant_triton(
            a,
            b,
            b_scale,
            b_zero,
            bias,
            group_size=group_size,
            activation=activation,
        )
    elif precision.mma in {"mxfp8", "mxfp6", "mxfp4"}:
        from ..kernels.triton.mxfp_gemm import gemm_mxfp_triton

        b_scales = kwargs.get("b_scales")
        block_size = kwargs.get("block_size", 32)
        mx_precision = int(precision.mma.replace("mxfp", ""))
        return gemm_mxfp_triton(
            a,
            b,
            b_scales,
            bias,
            mx_precision=mx_precision,
            block_size=block_size,
            activation=activation,
            transpose_b=transpose_b,
        )
    else:
        raise XQTBackendError(f"Unsupported Triton precision: {precision.mma}")


def _gemm_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    precision: MatmulPrecisionSpec,
    activation: str | None,
    transpose_b: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """TileLang engine dispatcher."""
    from ..kernels.tilelang.gemm_builder import build_tilelang_gemm_kernel
    from ..kernels.tilelang.gemm import (
        fp4_packed_dequant_gemm_epilogue_reference,
        fp4_packed_dequant_gemm_epilogue_tilelang,
        nvfp4_packed_dequant_gemm_epilogue_reference,
        nvfp4_packed_dequant_gemm_epilogue_tilelang,
    )

    if precision.mma == "fp16":
        M, K = a.shape
        if transpose_b:
            N, K_b = b.shape
        else:
            K_b, N = b.shape
        assert K == K_b

        kernel = build_tilelang_gemm_kernel(M, N, K, **kwargs)
        b_t = b.t().contiguous() if transpose_b else b.contiguous()
        output = torch.empty(
            (M, N),
            device=a.device,
            dtype=_precision_name_to_dtype(precision.output, role="output"),
        )
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
    elif precision.mma == "fp4":
        if not transpose_b:
            raise XQTBackendError("packed fp4 TileLang GEMM requires transpose_b=True")
        b_scale = kwargs.get("b_scale")
        if b_scale is None:
            raise ValueError("fp4 precision requires b_scale")
        group_size = int(kwargs.get("group_size", 128))
        input_features = int(kwargs.get("input_features", a.shape[1]))
        can_run_tilelang = (
            a.is_cuda
            and b.is_cuda
            and isinstance(b_scale, torch.Tensor)
            and b_scale.is_cuda
            and a.dtype == torch.float16
            and b.dtype == torch.uint8
            and b_scale.dtype == torch.float16
            and (bias is None or (bias.is_cuda and bias.dtype == torch.float16))
        )
        if can_run_tilelang:
            return fp4_packed_dequant_gemm_epilogue_tilelang(
                a,
                b,
                b_scale,
                bias,
                input_features=input_features,
                group_size=group_size,
                activation=activation,
                block_m=int(kwargs.get("block_m", 64)),
                block_n=int(kwargs.get("block_n", 64)),
                threads=int(kwargs.get("threads", 128)),
                num_stages=int(kwargs.get("num_stages", 2)),
                target_arch=kwargs.get("target_arch"),
            )
        return fp4_packed_dequant_gemm_epilogue_reference(
            a,
            b,
            b_scale,
            bias,
            input_features=input_features,
            group_size=group_size,
            activation=activation,
        )
    elif precision.mma == "nvfp4":
        if not transpose_b:
            raise XQTBackendError("packed nvfp4 TileLang GEMM requires transpose_b=True")
        b_scale = kwargs.get("b_scale")
        if b_scale is None:
            raise ValueError("nvfp4 precision requires b_scale")
        group_size = int(kwargs.get("group_size", 16))
        input_features = int(kwargs.get("input_features", a.shape[1]))
        weight_global_scale = kwargs.get("weight_global_scale")
        can_run_tilelang = (
            a.is_cuda
            and b.is_cuda
            and isinstance(b_scale, torch.Tensor)
            and b_scale.is_cuda
            and a.dtype == torch.float16
            and b.dtype == torch.uint8
            and b_scale.dtype == torch.float16
            and (weight_global_scale is None or (isinstance(weight_global_scale, torch.Tensor) and weight_global_scale.is_cuda and weight_global_scale.dtype == torch.float16))
            and (bias is None or (bias.is_cuda and bias.dtype == torch.float16))
        )
        if can_run_tilelang:
            return nvfp4_packed_dequant_gemm_epilogue_tilelang(
                a,
                b,
                b_scale,
                bias,
                input_features=input_features,
                group_size=group_size,
                weight_global_scale=weight_global_scale,
                activation=activation,
                block_m=int(kwargs.get("block_m", 64)),
                block_n=int(kwargs.get("block_n", 16)),
                block_k=int(kwargs.get("block_k", 128)),
                threads=int(kwargs.get("threads", 128)),
                num_stages=int(kwargs.get("num_stages", 2)),
                target_arch=kwargs.get("target_arch"),
            )
        return nvfp4_packed_dequant_gemm_epilogue_reference(
            a,
            b,
            b_scale,
            bias,
            input_features=input_features,
            group_size=group_size,
            weight_global_scale=weight_global_scale,
            activation=activation,
        )
    elif precision.mma == "int8":
        from ..kernels.tilelang.int8_mma import int8_linear_tilelang

        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise XQTBackendError(
                "TileLang int8 GEMM expects a and b to already be torch.int8; "
                "pre-quantize inputs before calling with engine='tilelang', or "
                "use engine='triton' for W8A16 weight-only dequant GEMM"
            )
        a_scale = kwargs.get("a_scale")
        b_scale = kwargs.get("b_scale")
        if a_scale is None or b_scale is None:
            raise ValueError(
                "TileLang int8 GEMM requires a_scale (per-tensor activation, "
                "shape (1,) or scalar tensor) and b_scale (per-channel weight, "
                "shape [n]); use engine='triton' for reference dequant GEMM if "
                "scales are not precomputed"
            )
        M, K_a = a.shape
        if transpose_b:
            N, K_b = b.shape
            b_kn = b.t().contiguous()
        else:
            K_b, N = b.shape
            b_kn = b.contiguous()
        if K_a != K_b:
            raise XQTBackendError(
                f"int8 GEMM inner-dimension mismatch: a.shape[1]={K_a}, b K={K_b}"
            )
        output = int8_linear_tilelang(
            a,
            b_kn,
            a_scale,
            b_scale,
            bias,
            output_dtype=_precision_name_to_dtype(
                precision.output, role="output", fallback=torch.float16
            ),
            block_m=int(kwargs.get("block_m", 64)),
            block_n=int(kwargs.get("block_n", 64)),
            block_k=int(kwargs.get("block_k", 64)),
            threads=int(kwargs.get("threads", 128)),
            num_stages=int(kwargs.get("num_stages", 2)),
            target_arch=kwargs.get("target_arch"),
        )
        return output
    else:
        raise XQTBackendError(f"TileLang precision {precision.mma} not implemented yet")


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
    from ..kernels.triton.gemm import gemm_reference

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
    output = gemm_reference(
        lhs, rhs, bias_value, activation=activation, transpose_b=transpose_b
    )
    return output.to(_precision_name_to_dtype(precision.output, role="output"))


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
        if precision in {"fp16", "bf16"}:
            capability["available"] = True
            capability["engine"] = "torch"
        elif precision in {"fp4", "nvfp4"}:
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

        if precision == "fp16":
            capability["available"] = True
            capability["engine"] = "triton"
            capability["hardware_native"] = sm >= 70
            if sm >= 70:
                capability["notes"].append("Tensor Core FP16 MMA available (SM70+)")
        elif precision == "bf16":
            capability["available"] = True
            capability["engine"] = "triton"
            capability["hardware_native"] = sm >= 80
            if sm >= 80:
                capability["notes"].append(
                    "Tensor Core BF16 MMA available (SM80+ Ampere)"
                )
        elif precision == "int8":
            capability["available"] = True
            capability["engine"] = "triton"
            capability["hardware_native"] = sm >= 75
            if sm >= 75:
                capability["notes"].append(
                    "Tensor Core INT8 MMA available (SM75+ Turing)"
                )
        elif precision == "fp8":
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
        elif precision == "int4":
            capability["available"] = True
            capability["engine"] = "triton"
            capability["hardware_native"] = False
            capability["notes"].append("INT4 via unpacking + FP16 MMA")
        elif precision in {"fp4", "nvfp4"}:
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
        elif precision in {"mxfp8", "mxfp6", "mxfp4"}:
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
    "describe_gemm_precision_capability",
    "gemm_with_precision",
    "list_available_precisions",
    "MatmulPrecisionSpec",
]
