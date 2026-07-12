"""TileLang backend adapter for XQT operator optimization."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from xqt.core.errors import XQTBackendError

from ..kernels.tilelang.attention import (
    build_tilelang_attention_design,
    fused_attention_forward_reference,
    fused_attention_forward_tilelang,
)
from ..kernels.tilelang.conv import (
    TILELANG_CONV_KERNEL_METADATA,
    conv2d_reference,
    conv2d_tilelang,
    conv3d_1x1x1_reference,
    conv3d_1x1x1_tilelang,
)
from ..kernels.tilelang.gemm import (
    TILELANG_DEQUANT_GEMM_KERNEL_METADATA,
    dequant_gemm_epilogue_reference,
    dequant_gemm_epilogue_tilelang,
    fp4_packed_dequant_gemm_epilogue_reference,
    fp4_packed_dequant_gemm_epilogue_tilelang,
    nvfp4_packed_dequant_gemm_epilogue_reference,
    nvfp4_packed_dequant_gemm_epilogue_tilelang,
)
from ..kernels.tilelang.int8_mma import (
    TILELANG_INT8_MMA_KERNEL_METADATA,
    int8_linear_reference,
    int8_linear_static_activation_reference,
    int8_mma_reference,
    int8_mma_tilelang,
    int8_linear_static_activation_tilelang,
    int8_linear_tilelang,
)
from ..kernels.tilelang.linear import (
    TILELANG_LINEAR_KERNEL_METADATA,
    dense_linear_epilogue_reference,
    dense_linear_epilogue_tilelang,
    half_linear_reference,
    half_linear_tilelang,
)
from ..kernels.tilelang.linear_marlin import (
    TILELANG_MARLIN_LINEAR_KERNEL_METADATA,
    linear_marlin_reference,
    linear_marlin_tilelang,
)
from ..kernels.tilelang.norm import (
    TILELANG_NORM_KERNEL_METADATA,
    layer_norm_reference,
    layer_norm_tilelang,
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

_ATTENTION_DESIGN = build_tilelang_attention_design()

TILELANG_KERNEL_METADATA: dict[str, dict[str, Any]] = {
    "attention": {
        "kernel_name": "fused_attention_forward",
        "block_m": _ATTENTION_DESIGN.default_block_m,
        "block_n": _ATTENTION_DESIGN.default_block_n,
        "threads": _ATTENTION_DESIGN.default_threads,
        "num_stages": _ATTENTION_DESIGN.default_num_stages,
        "baseline": "torch.nn.functional.scaled_dot_product_attention",
        "design": _ATTENTION_DESIGN.to_dict(),
    },
    **TILELANG_CONV_KERNEL_METADATA,
    **TILELANG_DEQUANT_GEMM_KERNEL_METADATA,
    **TILELANG_INT8_MMA_KERNEL_METADATA,
    **TILELANG_LINEAR_KERNEL_METADATA,
    **TILELANG_MARLIN_LINEAR_KERNEL_METADATA,
    **TILELANG_NORM_KERNEL_METADATA,
}


TILELANG_KERNEL_REGISTRY: dict[str, TileLangKernelSpec] = {
    "attention": TileLangKernelSpec(
        pattern="attention",
        reference=fused_attention_forward_reference,
        kernel=fused_attention_forward_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["attention"]),
    ),
    "conv": TileLangKernelSpec(
        pattern="conv",
        reference=conv2d_reference,
        kernel=conv2d_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["conv"]),
        cuda_only=False,
    ),
    "conv3d_1x1x1": TileLangKernelSpec(
        pattern="conv3d_1x1x1",
        reference=conv3d_1x1x1_reference,
        kernel=conv3d_1x1x1_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["conv3d_1x1x1"]),
        cuda_only=False,
    ),
    "dequant_gemm_epilogue": TileLangKernelSpec(
        pattern="dequant_gemm_epilogue",
        reference=dequant_gemm_epilogue_reference,
        kernel=dequant_gemm_epilogue_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["dequant_gemm_epilogue"]),
    ),
    "dense_linear_epilogue": TileLangKernelSpec(
        pattern="dense_linear_epilogue",
        reference=dense_linear_epilogue_reference,
        kernel=dense_linear_epilogue_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["dense_linear_epilogue"]),
    ),
    "int8_mma": TileLangKernelSpec(
        pattern="int8_mma",
        reference=int8_mma_reference,
        kernel=int8_mma_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["int8_mma"]),
    ),
    "int8_linear": TileLangKernelSpec(
        pattern="int8_linear",
        reference=int8_linear_reference,
        kernel=int8_linear_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["int8_linear"]),
    ),
    "int8_linear_static_activation": TileLangKernelSpec(
        pattern="int8_linear_static_activation",
        reference=int8_linear_static_activation_reference,
        kernel=int8_linear_static_activation_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["int8_linear_static_activation"]),
    ),
    "linear": TileLangKernelSpec(
        pattern="linear",
        reference=half_linear_reference,
        kernel=half_linear_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["linear"]),
    ),
    "linear_marlin": TileLangKernelSpec(
        pattern="linear_marlin",
        reference=linear_marlin_reference,
        kernel=linear_marlin_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["linear_marlin"]),
    ),
    "norm": TileLangKernelSpec(
        pattern="norm",
        reference=layer_norm_reference,
        kernel=layer_norm_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["norm"]),
    ),
    "fp4_packed_dequant_gemm_epilogue": TileLangKernelSpec(
        pattern="fp4_packed_dequant_gemm_epilogue",
        reference=fp4_packed_dequant_gemm_epilogue_reference,
        kernel=fp4_packed_dequant_gemm_epilogue_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["fp4_packed_dequant_gemm_epilogue"]),
    ),
    "nvfp4_packed_dequant_gemm_epilogue": TileLangKernelSpec(
        pattern="nvfp4_packed_dequant_gemm_epilogue",
        reference=nvfp4_packed_dequant_gemm_epilogue_reference,
        kernel=nvfp4_packed_dequant_gemm_epilogue_tilelang,
        metadata=dict(TILELANG_KERNEL_METADATA["nvfp4_packed_dequant_gemm_epilogue"]),
    ),
}


def tilelang_validation_thresholds(dtype: torch.dtype | str | None) -> dict[str, float]:
    """Return default TileLang numeric thresholds for a tensor dtype."""

    key = str(dtype) if dtype is not None else "torch.float32"
    return dict(
        TILELANG_DTYPE_VALIDATION_THRESHOLDS.get(
            key, TILELANG_DTYPE_VALIDATION_THRESHOLDS["torch.float32"]
        )
    )


def get_tilelang_kernel_spec(pattern: str) -> TileLangKernelSpec:
    """Return a registered TileLang kernel spec."""

    try:
        return TILELANG_KERNEL_REGISTRY[pattern]
    except KeyError as exc:
        allowed = ", ".join(sorted(TILELANG_KERNEL_REGISTRY))
        raise XQTBackendError(
            f"Unsupported TileLang pattern: {pattern}. Known: {allowed}"
        ) from exc


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
        str(cache_dir / f"{pattern}.tilelang.json") if cache_dir is not None else None
    )
    return {
        "engine": "tilelang",
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
                key: value for key, value in kwargs.items() if key in allowed
            }
            return spec.reference(*args, **filtered_kwargs)
        raise XQTBackendError(f"TileLang pattern '{pattern}' requires CUDA tensors")
    if any(arg.is_cuda for arg in tensor_args):
        from xqt.operator_opt.kernels.tilelang._common import (
            tilelang_runtime_unavailability_reason,
            tilelang_runtime_usable,
        )

        if not tilelang_runtime_usable():
            raise XQTBackendError(
                tilelang_runtime_unavailability_reason()
                or "TileLang runtime is unavailable for XQT kernels"
            )
    return spec.kernel(*args, **kwargs)


def list_tilelang_kernel_specs() -> dict[str, dict[str, Any]]:
    """Return TileLang kernel registry metadata."""

    return {
        name: spec.to_dict() for name, spec in sorted(TILELANG_KERNEL_REGISTRY.items())
    }


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
