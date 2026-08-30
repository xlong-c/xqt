"""CuTe DSL backend adapter for XQT operator optimization."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from xqt.kernels.jit.utils.compile import artifact_metadata_path, serialize_compile_settings
from xqt.kernels.registry import mirror_legacy_operator_registry as _mirror_registry

from xqt.core.errors import XQTBackendError

from xqt.kernels.ops._impl.cute_dsl import (
    CUTE_DSL_KERNEL_METADATA,
    gemm_epilogue_cute_dsl,
    gemm_epilogue_reference,
)


@dataclass(frozen=True)
class CuteDSLKernelSpec:
    """One CuTe DSL kernel registry entry."""

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
class CuteDSLCompileSettings:
    """Resolved CuTe DSL compile settings safe for manifest metadata."""

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


CUTE_DSL_KERNEL_REGISTRY: dict[str, CuteDSLKernelSpec] = {
    "gemm_epilogue": CuteDSLKernelSpec(
        pattern="gemm_epilogue",
        reference=gemm_epilogue_reference,
        kernel=gemm_epilogue_cute_dsl,
        metadata=dict(CUTE_DSL_KERNEL_METADATA["gemm_epilogue"]),
    ),
    "grouped_gemm": CuteDSLKernelSpec(
        pattern="grouped_gemm",
        reference=gemm_epilogue_reference,
        kernel=gemm_epilogue_cute_dsl,
        metadata=dict(CUTE_DSL_KERNEL_METADATA["grouped_gemm"]),
    ),
}

_mirror_registry(
    "cute_dsl",
    CUTE_DSL_KERNEL_REGISTRY,
    target="xqt.kernels.ops._impl.engines.cute_dsl:run_cute_dsl_kernel",
)


def cute_dsl_version() -> str | None:
    """Return the installed CuTe DSL version when importable."""

    try:
        cute = importlib.import_module("cutlass.cute")
    except ImportError:
        return None
    version = getattr(cute, "__version__", None)
    if version is not None:
        return str(version)
    try:
        cutlass = importlib.import_module("cutlass")
    except ImportError:
        return "unknown"
    return str(getattr(cutlass, "__version__", "unknown"))


def get_cute_dsl_kernel_spec(pattern: str) -> CuteDSLKernelSpec:
    """Return a registered CuTe DSL kernel spec."""

    try:
        return CUTE_DSL_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(CUTE_DSL_KERNEL_REGISTRY))
        raise XQTBackendError(f"Unsupported CuTe DSL pattern: {pattern}. Known: {allowed}") from exc


def build_cute_dsl_artifact_metadata(
    pattern: str,
    settings: CuteDSLCompileSettings | None = None,
) -> dict[str, Any]:
    """Build stable CuTe DSL artifact metadata."""

    spec = get_cute_dsl_kernel_spec(pattern)
    resolved = settings or CuteDSLCompileSettings()
    artifact = artifact_metadata_path(resolved.cache_dir, pattern, "cute_dsl")
    return {
        "engine": "cute_dsl",
        "pattern": pattern,
        "version": cute_dsl_version(),
        "kernel": dict(spec.metadata),
        "compile": resolved.to_dict(),
        "compile_status": "metadata_only",
        "compile_latency_ms": None,
        "execution_latency_ms": None,
        "artifact_path": None if artifact is None else str(artifact),
        "artifact_exists": False,
        "exportable": False,
    }


def run_cute_dsl_kernel(
    pattern: str,
    *args: torch.Tensor,
    fallback: str = "eager",
    **kwargs: Any,
) -> torch.Tensor:
    """Run a CuTe DSL kernel when CUDA is available, otherwise use fallback."""

    spec = get_cute_dsl_kernel_spec(pattern)
    if not all(isinstance(arg, torch.Tensor) for arg in args if arg is not None):
        raise TypeError("CuTe DSL kernel arguments must be tensors or None")
    tensor_args = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
    if spec.cuda_only and not all(arg.is_cuda for arg in tensor_args):
        if fallback == "eager":
            return spec.reference(*args, **kwargs)
        raise XQTBackendError(f"CuTe DSL pattern '{pattern}' requires CUDA tensors")
    return spec.kernel(*args, **kwargs)


def list_cute_dsl_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return CuTe DSL kernel registry metadata."""

    return {name: spec.to_dict() for name, spec in sorted(CUTE_DSL_KERNEL_REGISTRY.items())}


__all__ = [
    "CUTE_DSL_KERNEL_REGISTRY",
    "CuteDSLCompileSettings",
    "CuteDSLKernelSpec",
    "build_cute_dsl_artifact_metadata",
    "cute_dsl_version",
    "get_cute_dsl_kernel_spec",
    "list_cute_dsl_kernel_specs",
    "run_cute_dsl_kernel",
]
