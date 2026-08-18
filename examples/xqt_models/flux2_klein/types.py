"""FLUX.2 klein NVFP4 types, constants, and foundational helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.operator_opt import OperatorOptimizationTargetPlan
from xqt.quant import bridge_module_to_nvfp4_linear_shared


FLUX2_KLEIN_4B_REPO_ID = "black-forest-labs/FLUX.2-klein-4b"
FLUX2_KLEIN_4B_NVFP4_REPO_ID = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
FLUX2_KLEIN_4B_NVFP4_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"
FLUX2_KLEIN_NVFP4_ENGINES = ("cute_dsl", "cutile", "tilelang")

_ENGINE_ALIASES = {
    "cutedsl": "cute_dsl",
    "cute-dsl": "cute_dsl",
    "cute_dsl": "cute_dsl",
    "cutile": "cutile",
    "tilelang": "tilelang",
}

_MODEL_OPT_NVFP4_PARAMS = {"input_scale", "weight", "weight_scale", "weight_scale_2"}
_NON_QUANT_KEY_RENAMES = {
    "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
    "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
    "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
    "img_in.weight": "x_embedder.weight",
    "txt_in.weight": "context_embedder.weight",
    "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
    "final_layer.adaLN_modulation.1.weight": "norm_out.linear.weight",
    "final_layer.linear.weight": "proj_out.weight",
}


@dataclass(frozen=True)
class _MappedNVFP4Layer:
    source_name: str
    target_path: str
    chunk_index: int | None = None
    chunk_count: int | None = None


@dataclass(frozen=True)
class Flux2KleinNVFP4TargetSummary:
    """Small manifest entry for one bridgeable FLUX.2 NVFP4 Linear target."""

    name: str
    engine: str
    module_type: str
    input_features: int
    output_features: int
    group_size: int
    patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "module_type": self.module_type,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "group_size": self.group_size,
            "patterns": list(self.patterns),
        }


@dataclass(frozen=True)
class Flux2KleinNVFP4EngineResult:
    """Materialized engine inference result for a FLUX.2 klein NVFP4 model."""

    model: Any
    engine: str
    targets: list[OperatorOptimizationTargetPlan]
    target_summaries: list[Flux2KleinNVFP4TargetSummary]

    @property
    def target_count(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class Flux2KleinNVFP4CompiledTransformerResult:
    """Whole-transformer compile result for FLUX.2 klein NVFP4 inference."""

    model: nn.Module
    engine: str | None
    materialized_target_count: int
    compile_engine: str
    compile_mode: str | None
    compile_time_ms: float
    warmup_iterations: int
    warmup_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "materialized_target_count": self.materialized_target_count,
            "compile_engine": self.compile_engine,
            "compile_mode": self.compile_mode,
            "compile_time_ms": self.compile_time_ms,
            "warmup_iterations": self.warmup_iterations,
            "warmup_time_ms": self.warmup_time_ms,
        }


@dataclass(frozen=True)
class Flux2KleinNVFP4CudaGraphTransformerResult:
    """Whole-transformer CUDA Graph capture result for fixed-shape inference."""

    model: nn.Module
    engine: str | None
    materialized_target_count: int
    graph_state: Mapping[str, Any]
    input_signature: tuple[tuple[Any, ...], ...]
    warmup_iterations: int
    capture_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "materialized_target_count": self.materialized_target_count,
            "input_signature": [list(signature) for signature in self.input_signature],
            "warmup_iterations": self.warmup_iterations,
            "capture_time_ms": self.capture_time_ms,
        }


@dataclass(frozen=True)
class Flux2KleinNVFP4PairedBenchmarkResult:
    """Paired eager-vs-candidate benchmark for whole-transformer forward."""

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


class _Flux2KleinNVFP4Linear(nn.Module):
    """Minimal modelopt NVFP4 Linear shim for FLUX.2 transformer weights."""

    def __init__(
        self,
        *,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        input_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        source_name: str,
    ) -> None:
        super().__init__()
        self.in_features = int(packed_weight.shape[1]) * 2
        self.out_features = int(packed_weight.shape[0])
        self.source_name = source_name
        self.register_buffer("weight", packed_weight.to(torch.uint8))
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        self.register_buffer("weight_scale_2", weight_scale_2.reshape(1).to(torch.float32))
        if input_scale is None:
            self.register_buffer("input_scale", None)
        else:
            self.register_buffer("input_scale", input_scale.reshape(1).to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

    def _apply(self, fn: Any) -> "_Flux2KleinNVFP4Linear":
        super()._apply(fn)
        self.weight = self.weight.to(torch.uint8)
        self.weight_scale = self.weight_scale.to(torch.float32)
        self.weight_scale_2 = self.weight_scale_2.to(torch.float32)
        if self.input_scale is not None:
            self.input_scale = self.input_scale.to(torch.float32)
        if self.bias is not None:
            self.bias = self.bias.to(torch.float32)
        self.__dict__.pop("_xqt_nvfp4_bridge", None)
        return self

    def _bridge(self) -> Any:
        cached = self.__dict__.get("_xqt_nvfp4_bridge")
        if cached is not None:
            return cached
        bridge = bridge_module_to_nvfp4_linear_shared(self)
        if bridge is None:
            raise XQTBackendError(f"failed to build NVFP4 bridge for {self.source_name}")
        self.__dict__["_xqt_nvfp4_bridge"] = bridge
        return bridge

    def tilelang_dense_linear_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        return self._bridge().tilelang_dense_linear_args(dtype=dtype, device=device)

    def tilelang_packed_nvfp4_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None]:
        return self._bridge().tilelang_packed_nvfp4_dequant_gemm_args(
            dtype=dtype,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight, bias, _ = self.tilelang_dense_linear_args(
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return F.linear(inputs, weight, bias)


def normalize_flux2_klein_nvfp4_engine(engine: str) -> str:
    """Normalize user-facing engine names to XQT operator engine ids."""

    key = str(engine).strip().lower().replace(" ", "_")
    try:
        return _ENGINE_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(FLUX2_KLEIN_NVFP4_ENGINES)
        raise XQTBackendError(
            f"Unsupported FLUX.2 klein NVFP4 engine: {engine}. Known: {allowed}"
        ) from exc


def _resolve_flux2_klein_nvfp4_engine(
    *,
    engine: str | None,
    context: str,
) -> str:
    if engine is None:
        raise XQTBackendError(f"{context} requires engine=...")
    return normalize_flux2_klein_nvfp4_engine(engine)


def flux2_klein_nvfp4_single_file_url(
    *,
    repo_id: str = FLUX2_KLEIN_4B_NVFP4_REPO_ID,
    filename: str = FLUX2_KLEIN_4B_NVFP4_FILENAME,
) -> str:
    """Return the Hugging Face single-file URL for the NVFP4 transformer."""

    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def _cuda_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"sm_{major}{minor}"


def _resolve_module(model_or_pipeline: Any) -> tuple[nn.Module, str | None]:
    if isinstance(model_or_pipeline, nn.Module):
        return model_or_pipeline, None
    transformer = getattr(model_or_pipeline, "transformer", None)
    if isinstance(transformer, nn.Module):
        return transformer, "transformer"
    raise XQTBackendError(
        "FLUX.2 klein NVFP4 engine inference requires an nn.Module or a pipeline with an nn.Module transformer"
    )
