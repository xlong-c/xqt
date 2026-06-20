"""CuTile backend adapter for XQT operator optimization.

CuTile is treated as a Python DSL backend, closer to TileLang/CuTe DSL than to
an nvcc-built custom CUDA extension. This adapter intentionally records
capability and artifact metadata without compiling at import time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from xqt.core.errors import XQTBackendError

from ..kernels.cutile import (
    CUTILE_KERNEL_METADATA,
    fused_bias_silu_cutile,
    fused_bias_silu_reference,
)


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
    "bias_silu": CuTileKernelSpec(
        pattern="bias_silu",
        reference=fused_bias_silu_reference,
        kernel=fused_bias_silu_cutile,
        metadata=dict(CUTILE_KERNEL_METADATA["bias_silu"]),
    ),
}


def cutile_version() -> str | None:
    """Return the installed CuTile version when importable."""

    try:
        module = importlib.import_module("cutile")
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def get_cutile_kernel_spec(pattern: str) -> CuTileKernelSpec:
    """Return a registered CuTile kernel spec."""

    try:
        return CUTILE_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUTILE_KERNEL_REGISTRY))
        raise XQTBackendError(f"Unsupported CuTile pattern: {pattern}. Known: {allowed}") from exc


def build_cutile_artifact_metadata(
    pattern: str,
    settings: CuTileCompileSettings | None = None,
) -> dict[str, Any]:
    """Build stable CuTile artifact metadata."""

    spec = get_cutile_kernel_spec(pattern)
    resolved = settings or CuTileCompileSettings()
    cache_dir = Path(resolved.cache_dir) if resolved.cache_dir is not None else None
    artifact_path = (
        str(cache_dir / f"{pattern}.cutile.json")
        if cache_dir is not None
        else None
    )
    return {
        "backend": "cutile",
        "pattern": pattern,
        "version": cutile_version(),
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
    if not all(isinstance(arg, torch.Tensor) for arg in args):
        raise TypeError("CuTile kernel arguments must be tensors")
    if spec.cuda_only and not all(arg.is_cuda for arg in args):
        if fallback == "eager":
            return spec.reference(*args, **kwargs)
        raise XQTBackendError(f"CuTile pattern '{pattern}' requires CUDA tensors")
    return spec.kernel(*args, **kwargs)


def list_cutile_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return CuTile kernel registry metadata."""

    return {name: spec.to_dict() for name, spec in sorted(CUTILE_KERNEL_REGISTRY.items())}


__all__ = [
    "CUTILE_KERNEL_REGISTRY",
    "CuTileCompileSettings",
    "CuTileKernelSpec",
    "build_cutile_artifact_metadata",
    "cutile_version",
    "get_cutile_kernel_spec",
    "list_cutile_kernel_specs",
    "run_cutile_kernel",
]
