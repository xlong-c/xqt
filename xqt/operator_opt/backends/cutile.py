"""CuTile backend adapter for XQT operator optimization.

CuTile is treated as a Python DSL engine, closer to TileLang/CuTe DSL than to
an nvcc-built custom CUDA extension. This adapter intentionally records
capability and artifact metadata without compiling at import time.
"""

from __future__ import annotations

import inspect
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from xqt.core.errors import XQTBackendError

from ..kernels.cutile import (
    CUTILE_KERNEL_METADATA,
    conv2d_cutile,
    conv2d_reference,
    dense_linear_epilogue_cutile,
    dense_linear_epilogue_reference,
    dequant_gemm_epilogue_cutile,
    dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_cutile,
    fp4_packed_dequant_gemm_epilogue_reference,
    fused_attention_forward_cutile,
    fused_attention_forward_reference,
    fused_bias_silu_cutile,
    fused_bias_silu_reference,
    half_linear_cutile,
    half_linear_reference,
    layer_norm_cutile,
    layer_norm_reference,
    nvfp4_packed_dequant_gemm_epilogue_cutile,
    nvfp4_packed_dequant_gemm_epilogue_reference,
)
from ..kernels.cutile._common import cutile_module_metadata


def cutile_available() -> bool:
    """Return whether the current cuTile Python runtime is importable."""

    for module_name in ("cuda.tile", "cutile"):
        try:
            if importlib.util.find_spec(module_name) is not None:
                return True
        except ModuleNotFoundError:
            continue
    return False


@dataclass(frozen=True)
class CuTileKernelSpec:
    """One CuTile kernel registry entry."""

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
class CuTileCompileSettings:
    """Resolved CuTile compile settings safe for manifest metadata."""

    target: str = "cuda"
    target_arch: str | None = None
    cache_dir: str | None = None
    threads: int = 128
    pass_configs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cache_dir = str(Path(self.cache_dir)) if self.cache_dir is not None else None
        return {
            "target": self.target,
            "target_arch": self.target_arch,
            "cache_dir": cache_dir,
            "threads": self.threads,
            "pass_configs": dict(self.pass_configs),
        }


CUTILE_KERNEL_REGISTRY: dict[str, CuTileKernelSpec] = {
    "attention": CuTileKernelSpec(
        pattern="attention",
        reference=fused_attention_forward_reference,
        kernel=fused_attention_forward_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["attention"]),
    ),
    "bias_silu": CuTileKernelSpec(
        pattern="bias_silu",
        reference=fused_bias_silu_reference,
        kernel=fused_bias_silu_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["bias_silu"]),
    ),
    "conv": CuTileKernelSpec(
        pattern="conv",
        reference=conv2d_reference,
        kernel=conv2d_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["conv"]),
    ),
    "dense_linear_epilogue": CuTileKernelSpec(
        pattern="dense_linear_epilogue",
        reference=dense_linear_epilogue_reference,
        kernel=dense_linear_epilogue_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["dense_linear_epilogue"]),
    ),
    "dequant_gemm_epilogue": CuTileKernelSpec(
        pattern="dequant_gemm_epilogue",
        reference=dequant_gemm_epilogue_reference,
        kernel=dequant_gemm_epilogue_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["dequant_gemm_epilogue"]),
    ),
    "fp4_packed_dequant_gemm_epilogue": CuTileKernelSpec(
        pattern="fp4_packed_dequant_gemm_epilogue",
        reference=fp4_packed_dequant_gemm_epilogue_reference,
        kernel=fp4_packed_dequant_gemm_epilogue_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["fp4_packed_dequant_gemm_epilogue"]),
    ),
    "linear": CuTileKernelSpec(
        pattern="linear",
        reference=half_linear_reference,
        kernel=half_linear_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["linear"]),
    ),
    "norm": CuTileKernelSpec(
        pattern="norm",
        reference=layer_norm_reference,
        kernel=layer_norm_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["norm"]),
    ),
    "nvfp4_packed_dequant_gemm_epilogue": CuTileKernelSpec(
        pattern="nvfp4_packed_dequant_gemm_epilogue",
        reference=nvfp4_packed_dequant_gemm_epilogue_reference,
        kernel=nvfp4_packed_dequant_gemm_epilogue_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["nvfp4_packed_dequant_gemm_epilogue"]),
    ),
}


def cutile_version() -> str | None:
    """Return the installed CuTile version when importable."""

    metadata = cutile_module_metadata()
    return str(metadata.get("version", "unknown")) if metadata else None


def get_cutile_kernel_spec(pattern: str) -> CuTileKernelSpec:
    """Return a registered CuTile kernel spec."""

    try:
        return CUTILE_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUTILE_KERNEL_REGISTRY))
        raise XQTBackendError(
            f"Unsupported CuTile pattern: {pattern}. Known: {allowed}"
        ) from exc


def build_cutile_artifact_metadata(
    pattern: str,
    settings: CuTileCompileSettings | None = None,
) -> dict[str, Any]:
    """Build stable CuTile artifact metadata."""

    spec = get_cutile_kernel_spec(pattern)
    resolved = settings or CuTileCompileSettings()
    cache_dir = Path(resolved.cache_dir) if resolved.cache_dir is not None else None
    artifact_path = (
        str(cache_dir / f"{pattern}.cutile.json") if cache_dir is not None else None
    )
    return {
        "engine": "cutile",
        "pattern": pattern,
        "version": cutile_version(),
        "module": cutile_module_metadata().get("module"),
        "kernel": dict(spec.metadata),
        "compile": resolved.to_dict(),
        "compile_status": "metadata_only",
        "compile_latency_ms": None,
        "execution_latency_ms": None,
        "artifact_path": artifact_path,
        "artifact_exists": False,
        "exportable": False,
    }


def run_cutile_kernel(
    pattern: str,
    *args: torch.Tensor,
    fallback: str = "eager",
    **kwargs: Any,
) -> torch.Tensor:
    """Run a CuTile kernel when CUDA is available, otherwise use configured fallback."""

    spec = get_cutile_kernel_spec(pattern)
    if not all(isinstance(arg, torch.Tensor) for arg in args if arg is not None):
        raise TypeError("CuTile kernel arguments must be tensors or None")
    tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    if spec.cuda_only and not all(arg.is_cuda for arg in tensor_args):
        if fallback == "eager":
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in allowed
            }
            return spec.reference(*args, **filtered_kwargs)
        raise XQTBackendError(f"CuTile pattern '{pattern}' requires CUDA tensors")
    return spec.kernel(*args, **kwargs)


def list_cutile_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return CuTile kernel registry metadata."""

    return {
        name: spec.to_dict() for name, spec in sorted(CUTILE_KERNEL_REGISTRY.items())
    }


__all__ = [
    "CUTILE_KERNEL_REGISTRY",
    "CuTileCompileSettings",
    "CuTileKernelSpec",
    "build_cutile_artifact_metadata",
    "cutile_available",
    "cutile_version",
    "get_cutile_kernel_spec",
    "list_cutile_kernel_specs",
    "run_cutile_kernel",
]
