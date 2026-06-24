"""TileLang backend adapter for XQT operator optimization."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from xqt.core.errors import XQTBackendError

from ..kernels.tilelang import (
    TILELANG_KERNEL_METADATA,
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)


@dataclass(frozen=True)
class TileLangKernelSpec:
    """One TileLang kernel registry entry."""

    pattern: str
    reference: Callable[..., torch.Tensor]
    kernel: Callable[..., torch.Tensor]
    cuda_only: bool = True
    fallback: str = "eager"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "cuda_only": self.cuda_only,
            "fallback": self.fallback,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TileLangCompileSettings:
    """Resolved TileLang compile settings safe for manifest metadata."""

    target: str = "cuda"
    target_arch: str | None = None
    cache_dir: str | None = None
    threads: int = 128
    num_stages: int = 2
    pass_configs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cache_dir = str(Path(self.cache_dir)) if self.cache_dir is not None else None
        return {
            "target": self.target,
            "target_arch": self.target_arch,
            "cache_dir": cache_dir,
            "threads": self.threads,
            "num_stages": self.num_stages,
            "pass_configs": dict(self.pass_configs),
        }


TILELANG_DTYPE_VALIDATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "torch.float32": {"atol": 1e-5, "rtol": 1e-5},
    "torch.float16": {"atol": 1e-3, "rtol": 1e-3},
    "torch.bfloat16": {"atol": 1e-2, "rtol": 1e-2},
    "torch.float8_e4m3fn": {"atol": 5e-2, "rtol": 5e-2},
    "torch.float8_e5m2": {"atol": 1e-1, "rtol": 1e-1},
}


TILELANG_KERNEL_REGISTRY: dict[str, TileLangKernelSpec] = {
    "attention": TileLangKernelSpec(
        pattern="attention",
        reference=fused_attention_forward_reference,
        kernel=fused_attention_forward_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["attention"]),
    ),
    "dequant_gemm_epilogue": TileLangKernelSpec(
        pattern="dequant_gemm_epilogue",
        reference=dequant_gemm_epilogue_reference,
        kernel=dequant_gemm_epilogue_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["dequant_gemm_epilogue"]),
    ),
    "fp4_packed_dequant_gemm_epilogue": TileLangKernelSpec(
        pattern="fp4_packed_dequant_gemm_epilogue",
        reference=fp4_packed_dequant_gemm_epilogue_reference,
        kernel=fp4_packed_dequant_gemm_epilogue_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["fp4_packed_dequant_gemm_epilogue"]),
    ),
}


def tilelang_validation_thresholds(dtype: torch.dtype | str | None) -> dict[str, float]:
    """Return default TileLang numeric thresholds for a tensor dtype."""

    key = str(dtype) if dtype is not None else "torch.float32"
    return dict(TILELANG_DTYPE_VALIDATION_THRESHOLDS.get(key, TILELANG_DTYPE_VALIDATION_THRESHOLDS["torch.float32"]))


def get_tilelang_kernel_spec(pattern: str) -> TileLangKernelSpec:
    """Return a registered TileLang kernel spec."""

    try:
        return TILELANG_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(TILELANG_KERNEL_REGISTRY))
        raise XQTBackendError(f"Unsupported TileLang pattern: {pattern}. Known: {allowed}") from exc


def tilelang_version() -> str | None:
    """Return the installed TileLang version when importable."""

    try:
        module = importlib.import_module("tilelang")
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def build_tilelang_artifact_metadata(
    pattern: str,
    settings: TileLangCompileSettings | None = None,
) -> dict[str, Any]:
    """Build stable TileLang artifact metadata."""

    spec = get_tilelang_kernel_spec(pattern)
    resolved = settings or TileLangCompileSettings()
    cache_dir = Path(resolved.cache_dir) if resolved.cache_dir is not None else None
    artifact_path = (
        str(cache_dir / f"{pattern}.tilelang.json")
        if cache_dir is not None
        else None
    )
    return {
        "backend": "tilelang",
        "pattern": pattern,
        "version": tilelang_version(),
        "kernel": dict(spec.metadata),
        "compile": resolved.to_dict(),
        "compile_status": "metadata_only",
        "compile_latency_ms": None,
        "execution_latency_ms": None,
        "artifact_path": artifact_path,
        "artifact_exists": False,
        "validation_thresholds": {
            dtype: dict(thresholds)
            for dtype, thresholds in TILELANG_DTYPE_VALIDATION_THRESHOLDS.items()
        },
        "exportable": False,
    }


def run_tilelang_kernel(
    pattern: str,
    *args: torch.Tensor,
    fallback: str = "eager",
    **kwargs: Any,
) -> torch.Tensor:
    """Run a TileLang kernel when CUDA is available, otherwise use configured fallback."""

    spec = get_tilelang_kernel_spec(pattern)
    if not all(isinstance(arg, torch.Tensor) for arg in args if arg is not None):
        raise TypeError("TileLang kernel arguments must be tensors or None")
    tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    if spec.cuda_only and not all(arg.is_cuda for arg in tensor_args):
        if fallback == "eager":
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in allowed
            }
            return spec.reference(*args, **filtered_kwargs)
        raise XQTBackendError(f"TileLang pattern '{pattern}' requires CUDA tensors")
    return spec.kernel(*args, **kwargs)


def list_tilelang_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return TileLang kernel registry metadata."""

    return {name: spec.to_dict() for name, spec in sorted(TILELANG_KERNEL_REGISTRY.items())}


__all__ = [
    "TILELANG_DTYPE_VALIDATION_THRESHOLDS",
    "TILELANG_KERNEL_REGISTRY",
    "TileLangCompileSettings",
    "TileLangKernelSpec",
    "build_tilelang_artifact_metadata",
    "get_tilelang_kernel_spec",
    "list_tilelang_kernel_specs",
    "run_tilelang_kernel",
    "tilelang_validation_thresholds",
    "tilelang_version",
]
