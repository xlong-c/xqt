"""Validation helpers for TileLang packed FP4 operator paths."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from xqt.quant.quantizers.reference_fp4 import ReferenceFP4Linear

from ..kernels.tilelang import (
    build_tilelang_fp4_fused_dequant_gemm_kernel,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
)


_SUPPORTED_ACTIVATIONS = {None, "gelu", "silu", "relu"}


@dataclass(frozen=True)
class TileLangFP4ValidationResult:
    """Structured result for packed FP4 TileLang fused GEMM validation."""

    status: str
    reason: str | None
    device: str | None
    dtype: str
    shape: dict[str, int]
    activation: str | None
    has_bias: bool
    target_arch: str | None
    max_abs_error: float | None
    mean_abs_error: float | None
    rtol: float
    atol: float
    allclose: bool | None
    compile_only: bool
    compile_status: str
    latency_ms_tilelang: float | None
    latency_ms_reference: float | None
    speedup: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-safe dictionary representation."""

        return asdict(self)


def _result(
    *,
    status: str,
    reason: str | None,
    shape: dict[str, int],
    activation: str | None,
    has_bias: bool,
    target_arch: str | None,
    rtol: float,
    atol: float,
    compile_only: bool,
    compile_status: str,
    device: str | None = None,
    max_abs_error: float | None = None,
    mean_abs_error: float | None = None,
    allclose: bool | None = None,
    latency_ms_tilelang: float | None = None,
    latency_ms_reference: float | None = None,
    speedup: float | None = None,
) -> TileLangFP4ValidationResult:
    return TileLangFP4ValidationResult(
        status=status,
        reason=reason,
        device=device,
        dtype=str(torch.float16),
        shape=shape,
        activation=activation,
        has_bias=has_bias,
        target_arch=target_arch,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        rtol=rtol,
        atol=atol,
        allclose=allclose,
        compile_only=compile_only,
        compile_status=compile_status,
        latency_ms_tilelang=latency_ms_tilelang,
        latency_ms_reference=latency_ms_reference,
        speedup=speedup,
    )


def _tilelang_available() -> bool:
    return importlib.util.find_spec("tilelang") is not None


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None or Path("/usr/local/cuda/bin/nvcc").exists()


def _shape_dict(
    *,
    m: int,
    in_features: int,
    out_features: int,
    group_size: int,
    block_m: int,
    block_n: int,
) -> dict[str, int]:
    return {
        "m": int(m),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "group_size": int(group_size),
        "block_m": int(block_m),
        "block_n": int(block_n),
    }


def _validate_shape(shape: dict[str, int]) -> None:
    for name, value in shape.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if shape["m"] % shape["block_m"] != 0:
        raise ValueError("m must be a multiple of block_m for the minimal TileLang kernel")
    if shape["out_features"] % shape["block_n"] != 0:
        raise ValueError(
            "out_features must be a multiple of block_n for the minimal TileLang kernel"
        )


def _resolve_target_arch(target_arch: str | None) -> str | None:
    if target_arch is not None:
        return str(target_arch)
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _benchmark_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def validate_tilelang_packed_fp4_fused_gemm(
    *,
    m: int = 64,
    in_features: int = 32,
    out_features: int = 64,
    group_size: int = 16,
    activation: str | None = "silu",
    has_bias: bool = True,
    target_arch: str | None = None,
    block_m: int = 64,
    block_n: int = 64,
    threads: int = 128,
    warmup: int = 5,
    iterations: int = 20,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    compile_only: bool = False,
    seed: int = 0,
) -> TileLangFP4ValidationResult:
    """Validate the packed FP4 fused TileLang GEMM path.

    The helper is intentionally runtime-aware: it returns ``skipped`` when
    TileLang, nvcc, or CUDA runtime requirements are unavailable, ``ok`` when
    the requested validation completes and passes, and ``error`` for compile,
    launch, or numeric failures.
    """

    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported activation: {activation}")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    shape = _shape_dict(
        m=m,
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        block_m=block_m,
        block_n=block_n,
    )
    _validate_shape(shape)
    resolved_arch = _resolve_target_arch(target_arch)

    if not _tilelang_available():
        return _result(
            status="skipped",
            reason="tilelang package is not importable",
            shape=shape,
            activation=activation,
            has_bias=has_bias,
            target_arch=resolved_arch,
            rtol=rtol,
            atol=atol,
            compile_only=compile_only,
            compile_status="unavailable",
        )

    if compile_only:
        if resolved_arch is None:
            return _result(
                status="skipped",
                reason="target_arch is required for compile-only validation when CUDA is unavailable",
                shape=shape,
                activation=activation,
                has_bias=has_bias,
                target_arch=resolved_arch,
                rtol=rtol,
                atol=atol,
                compile_only=True,
                compile_status="skipped",
            )
        if not _nvcc_available():
            return _result(
                status="skipped",
                reason="nvcc is not available for TileLang compile-only validation",
                shape=shape,
                activation=activation,
                has_bias=has_bias,
                target_arch=resolved_arch,
                rtol=rtol,
                atol=atol,
                compile_only=True,
                compile_status="unavailable",
            )
        try:
            build_tilelang_fp4_fused_dequant_gemm_kernel(
                m,
                out_features,
                in_features,
                group_size,
                block_m=block_m,
                block_n=block_n,
                threads=threads,
                target_arch=resolved_arch,
                has_bias=has_bias,
                activation=activation,
            )
        except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
            return _result(
                status="error",
                reason=str(exc),
                shape=shape,
                activation=activation,
                has_bias=has_bias,
                target_arch=resolved_arch,
                rtol=rtol,
                atol=atol,
                compile_only=True,
                compile_status="error",
            )
        return _result(
            status="ok",
            reason=None,
            shape=shape,
            activation=activation,
            has_bias=has_bias,
            target_arch=resolved_arch,
            rtol=rtol,
            atol=atol,
            compile_only=True,
            compile_status="ok",
        )

    if not torch.cuda.is_available():
        return _result(
            status="skipped",
            reason="CUDA runtime is not available for TileLang packed FP4 validation",
            shape=shape,
            activation=activation,
            has_bias=has_bias,
            target_arch=resolved_arch,
            rtol=rtol,
            atol=atol,
            compile_only=False,
            compile_status="not_requested",
        )
    if not _nvcc_available():
        return _result(
            status="skipped",
            reason="nvcc is not available for TileLang packed FP4 validation",
            shape=shape,
            activation=activation,
            has_bias=has_bias,
            target_arch=resolved_arch,
            rtol=rtol,
            atol=atol,
            compile_only=False,
            compile_status="unavailable",
            device="cuda",
        )

    try:
        device = torch.device("cuda")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            linear = torch.nn.Linear(in_features, out_features, bias=has_bias)
            fp4_linear = ReferenceFP4Linear.from_linear(
                linear,
                group_size=group_size,
            ).to(device)
            x = torch.randn(m, in_features, device=device, dtype=torch.float16)

        packed_weight = fp4_linear.packed_weight
        scale = fp4_linear.weight_scale.to(dtype=torch.float16)
        bias = (
            fp4_linear.bias.to(dtype=torch.float16)
            if fp4_linear.bias is not None
            else None
        )

        def run_tilelang() -> torch.Tensor:
            return fp4_packed_dequant_gemm_epilogue_tilelang(
                x,
                packed_weight,
                scale,
                bias,
                input_features=fp4_linear.input_features,
                group_size=fp4_linear.group_size,
                activation=activation,
                block_m=block_m,
                block_n=block_n,
                threads=threads,
                target_arch=resolved_arch,
            )

        def run_reference() -> torch.Tensor:
            return fp4_packed_dequant_gemm_epilogue_reference(
                x,
                packed_weight,
                scale,
                bias,
                input_features=fp4_linear.input_features,
                group_size=fp4_linear.group_size,
                activation=activation,
            )

        output = run_tilelang()
        reference = run_reference()
        torch.cuda.synchronize()
        diff = (output.float() - reference.float()).abs()
        max_abs_error = float(diff.max().item())
        mean_abs_error = float(diff.mean().item())
        allclose = bool(
            torch.allclose(output.float(), reference.float(), atol=atol, rtol=rtol)
        )
        latency_ms_tilelang = _benchmark_cuda(run_tilelang, warmup=warmup, iterations=iterations)
        latency_ms_reference = _benchmark_cuda(run_reference, warmup=warmup, iterations=iterations)
        speedup = (
            latency_ms_reference / latency_ms_tilelang
            if latency_ms_tilelang > 0
            else None
        )
    except Exception as exc:  # pragma: no cover - requires CUDA runtime failure coverage
        return _result(
            status="error",
            reason=str(exc),
            shape=shape,
            activation=activation,
            has_bias=has_bias,
            target_arch=resolved_arch,
            rtol=rtol,
            atol=atol,
            compile_only=False,
            compile_status="error",
            device="cuda",
        )

    return _result(
        status="ok" if allclose else "error",
        reason=None if allclose else "TileLang output did not match the reference output",
        shape=shape,
        activation=activation,
        has_bias=has_bias,
        target_arch=resolved_arch,
        rtol=rtol,
        atol=atol,
        compile_only=False,
        compile_status="ok",
        device=str(device),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        allclose=allclose,
        latency_ms_tilelang=latency_ms_tilelang,
        latency_ms_reference=latency_ms_reference,
        speedup=speedup,
    )


__all__ = [
    "TileLangFP4ValidationResult",
    "validate_tilelang_packed_fp4_fused_gemm",
]
