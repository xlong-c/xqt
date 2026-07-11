"""Wan 2.1 VAE types, constants, and foundational helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError


WAN21_VAE_REPO_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
WAN21_VAE_SUBFOLDER = "vae"
WAN21_VAE_OPTIMIZATION_KINDS = ("compile", "cuda_graph")
WAN21_VAE_RUN_MODES = ("decode", "encode")

Wan21VAERunMode = Literal["decode", "encode"]


@dataclass(frozen=True)
class Wan21VAECompileResult:
    """Whole-runner torch.compile result for Wan 2.1 VAE inference."""

    model: nn.Module
    run_mode: Wan21VAERunMode
    compile_engine: str
    compile_mode: str | None
    compile_time_ms: float
    warmup_iterations: int
    warmup_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "compile_engine": self.compile_engine,
            "compile_mode": self.compile_mode,
            "compile_time_ms": self.compile_time_ms,
            "warmup_iterations": self.warmup_iterations,
            "warmup_time_ms": self.warmup_time_ms,
        }


@dataclass(frozen=True)
class Wan21VAECudaGraphResult:
    """Fixed-shape CUDA Graph capture result for Wan 2.1 VAE inference."""

    model: nn.Module
    run_mode: Wan21VAERunMode
    graph_state: Mapping[str, Any]
    input_signature: tuple[tuple[Any, ...], ...]
    warmup_iterations: int
    capture_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "input_signature": [list(signature) for signature in self.input_signature],
            "warmup_iterations": self.warmup_iterations,
            "capture_time_ms": self.capture_time_ms,
        }


@dataclass(frozen=True)
class Wan21VAEPairedBenchmarkResult:
    """Paired eager-vs-candidate benchmark for Wan 2.1 VAE encode/decode."""

    run_mode: Wan21VAERunMode
    reference_report: dict[str, Any]
    candidate_report: dict[str, Any]
    paired_speedup_ratios: list[float]
    paired_speedup_p50: float
    max_abs_vs_eager: float
    mean_abs_vs_eager: float
    allclose_vs_eager: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "reference_report": dict(self.reference_report),
            "candidate_report": dict(self.candidate_report),
            "paired_speedup_ratios": list(self.paired_speedup_ratios),
            "paired_speedup_p50": self.paired_speedup_p50,
            "max_abs_vs_eager": self.max_abs_vs_eager,
            "mean_abs_vs_eager": self.mean_abs_vs_eager,
            "allclose_vs_eager": self.allclose_vs_eager,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass(frozen=True)
class Wan21VAEOptimizationSummary:
    """High-level optimization summary for one Wan 2.1 VAE fastpath."""

    run_mode: Wan21VAERunMode
    optimization_kind: str
    compile_engine: str | None
    compile_mode: str | None
    tiled: bool
    sliced: bool
    tile_sample_min_height: int | None
    tile_sample_min_width: int | None
    tile_sample_stride_height: int | None
    tile_sample_stride_width: int | None
    warmup_iterations: int
    materialized_conv3d_targets: int = 0
    materialized_rmsnorm_targets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "optimization_kind": self.optimization_kind,
            "compile_engine": self.compile_engine,
            "compile_mode": self.compile_mode,
            "tiled": self.tiled,
            "sliced": self.sliced,
            "tile_sample_min_height": self.tile_sample_min_height,
            "tile_sample_min_width": self.tile_sample_min_width,
            "tile_sample_stride_height": self.tile_sample_stride_height,
            "tile_sample_stride_width": self.tile_sample_stride_width,
            "warmup_iterations": self.warmup_iterations,
            "materialized_conv3d_targets": self.materialized_conv3d_targets,
            "materialized_rmsnorm_targets": self.materialized_rmsnorm_targets,
        }


def _cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"sm_{major}{minor}"


def _normalize_run_mode(run_mode: str) -> Wan21VAERunMode:
    normalized = str(run_mode).strip().lower()
    if normalized not in WAN21_VAE_RUN_MODES:
        allowed = ", ".join(WAN21_VAE_RUN_MODES)
        raise XQTBackendError(f"Unsupported Wan 2.1 VAE run mode: {run_mode}. Known: {allowed}")
    return normalized  # type: ignore[return-value]


def _normalize_optimization_kind(kind: str) -> str:
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized not in WAN21_VAE_OPTIMIZATION_KINDS:
        allowed = ", ".join(WAN21_VAE_OPTIMIZATION_KINDS)
        raise XQTBackendError(
            f"Unsupported Wan 2.1 VAE optimization kind: {kind}. Known: {allowed}"
        )
    return normalized


def _is_autoencoder_kl_wan(module: Any) -> bool:
    if type(module).__name__ == "AutoencoderKLWan":
        return True
    required = ("encode", "decode", "enable_tiling", "enable_slicing")
    return isinstance(module, nn.Module) and all(callable(getattr(module, name, None)) for name in required)


def _resolve_vae(model_or_pipeline: Any) -> tuple[nn.Module, str | None]:
    if isinstance(model_or_pipeline, nn.Module) and _is_autoencoder_kl_wan(model_or_pipeline):
        return model_or_pipeline, None
    vae = getattr(model_or_pipeline, "vae", None)
    if isinstance(vae, nn.Module) and _is_autoencoder_kl_wan(vae):
        return vae, "vae"
    raise XQTBackendError(
        "Wan 2.1 VAE optimization requires an AutoencoderKLWan instance or a pipeline with a .vae module"
    )
