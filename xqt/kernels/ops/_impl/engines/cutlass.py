"""CUTLASS Python backend adapter for XQT operator optimization."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from xqt.kernels.jit.utils.compile import artifact_metadata_path, serialize_compile_settings
from xqt.kernels.registry import mirror_legacy_operator_registry as _mirror_registry

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.cutlass import (
    CUTLASS_KERNEL_METADATA,
    gemm_epilogue_cutlass,
    gemm_epilogue_reference,
)


@dataclass(frozen=True)
class CutlassKernelSpec:
    """One CUTLASS Python kernel registry entry."""

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
class CutlassCompileSettings:
    """Resolved CUTLASS Python compile settings safe for manifest metadata."""

    target_arch: str | None = None
    cache_dir: str | None = None
    tile_shape: tuple[int, int, int] = (128, 128, 64)
    cluster_shape: tuple[int, int, int] | None = None
    pass_configs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return serialize_compile_settings(
            target_arch=self.target_arch,
            cache_dir=self.cache_dir,
            tile_shape=self.tile_shape,
            cluster_shape=self.cluster_shape,
            pass_configs=self.pass_configs,
        )


CUTLASS_KERNEL_REGISTRY: dict[str, CutlassKernelSpec] = {
    "gemm_epilogue": CutlassKernelSpec(
        pattern="gemm_epilogue",
        reference=gemm_epilogue_reference,
        kernel=gemm_epilogue_cutlass,
        metadata=dict(CUTLASS_KERNEL_METADATA["gemm_epilogue"]),
    ),
    "grouped_gemm": CutlassKernelSpec(
        pattern="grouped_gemm",
        reference=gemm_epilogue_reference,
        kernel=gemm_epilogue_cutlass,
        metadata=dict(CUTLASS_KERNEL_METADATA["grouped_gemm"]),
    ),
}

_mirror_registry(
    "cutlass",
    CUTLASS_KERNEL_REGISTRY,
    target="xqt.kernels.ops._impl.engines.cutlass:run_cutlass_kernel",
)


def cutlass_version() -> str | None:
    """Return the installed CUTLASS Python version when importable."""

    try:
        module = importlib.import_module("cutlass")
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def get_cutlass_kernel_spec(pattern: str) -> CutlassKernelSpec:
    """Return a registered CUTLASS Python kernel spec."""

    try:
        return CUTLASS_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUTLASS_KERNEL_REGISTRY))
        raise XQTBackendError(f"Unsupported CUTLASS pattern: {pattern}. Known: {allowed}") from exc


def build_cutlass_artifact_metadata(
    pattern: str,
    settings: CutlassCompileSettings | None = None,
) -> dict[str, Any]:
    """Build stable CUTLASS Python artifact metadata."""

    spec = get_cutlass_kernel_spec(pattern)
    resolved = settings or CutlassCompileSettings()
    artifact = artifact_metadata_path(resolved.cache_dir, pattern, "cutlass")
    return {
        "engine": "cutlass",
        "pattern": pattern,
        "version": cutlass_version(),
        "kernel": dict(spec.metadata),
        "compile": resolved.to_dict(),
        "compile_status": "metadata_only",
        "compile_latency_ms": None,
        "execution_latency_ms": None,
        "artifact_path": None if artifact is None else str(artifact),
        "artifact_exists": False,
        "exportable": False,
    }


def run_cutlass_kernel(
    pattern: str,
    *args: torch.Tensor,
    fallback: str = "eager",
    **kwargs: Any,
) -> torch.Tensor:
    """Run a CUTLASS Python kernel when CUDA is available, otherwise use fallback."""

    spec = get_cutlass_kernel_spec(pattern)
    if not all(isinstance(arg, torch.Tensor) for arg in args if arg is not None):
        raise TypeError("CUTLASS kernel arguments must be tensors or None")
    tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    if spec.cuda_only and not all(arg.is_cuda for arg in tensor_args):
        if fallback == "eager":
            return spec.reference(*args, **kwargs)
        raise XQTBackendError(f"CUTLASS pattern '{pattern}' requires CUDA tensors")
    return spec.kernel(*args, **kwargs)


def list_cutlass_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return CUTLASS Python kernel registry metadata."""

    return {name: spec.to_dict() for name, spec in sorted(CUTLASS_KERNEL_REGISTRY.items())}


__all__ = [
    "CUTLASS_KERNEL_REGISTRY",
    "CutlassCompileSettings",
    "CutlassKernelSpec",
    "build_cutlass_artifact_metadata",
    "cutlass_version",
    "get_cutlass_kernel_spec",
    "list_cutlass_kernel_specs",
    "run_cutlass_kernel",
]
