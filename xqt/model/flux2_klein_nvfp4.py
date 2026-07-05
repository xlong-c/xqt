"""FLUX.2 klein NVFP4 model-side backend inference helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    materialize_operator_candidate_models,
)
from xqt.operator_opt.compile_backend import compile_with_torch
from xqt.quant import bridge_module_to_nvfp4_linear_shared, infer_nvfp4_tensor_layout


FLUX2_KLEIN_4B_REPO_ID = "black-forest-labs/FLUX.2-klein-4b"
FLUX2_KLEIN_4B_NVFP4_REPO_ID = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
FLUX2_KLEIN_4B_NVFP4_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"
FLUX2_KLEIN_NVFP4_BACKENDS = ("cute_dsl", "cutile", "tilelang")

_BACKEND_ALIASES = {
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
    backend: str
    module_type: str
    input_features: int
    output_features: int
    group_size: int
    patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "module_type": self.module_type,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "group_size": self.group_size,
            "patterns": list(self.patterns),
        }


@dataclass(frozen=True)
class Flux2KleinNVFP4BackendResult:
    """Materialized backend inference result for a FLUX.2 klein NVFP4 model."""

    model: Any
    backend: str
    targets: list[OperatorOptimizationTargetPlan]
    target_summaries: list[Flux2KleinNVFP4TargetSummary]

    @property
    def target_count(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class Flux2KleinNVFP4CompiledTransformerResult:
    """Whole-transformer compile result for FLUX.2 klein NVFP4 inference."""

    model: nn.Module
    backend: str | None
    materialized_target_count: int
    compile_backend: str
    compile_mode: str | None
    compile_time_ms: float
    warmup_iterations: int
    warmup_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "materialized_target_count": self.materialized_target_count,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "compile_time_ms": self.compile_time_ms,
            "warmup_iterations": self.warmup_iterations,
            "warmup_time_ms": self.warmup_time_ms,
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


def normalize_flux2_klein_nvfp4_backend(backend: str) -> str:
    """Normalize user-facing backend names to XQT operator backend ids."""

    key = str(backend).strip().lower().replace(" ", "_")
    try:
        return _BACKEND_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(FLUX2_KLEIN_NVFP4_BACKENDS)
        raise XQTBackendError(
            f"Unsupported FLUX.2 klein NVFP4 backend: {backend}. Known: {allowed}"
        ) from exc


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
        "FLUX.2 klein NVFP4 backend inference requires an nn.Module or a pipeline with an nn.Module transformer"
    )


def _matches_filters(
    name: str,
    *,
    include_names: Sequence[str] | None,
    exclude_names: Sequence[str] | None,
) -> bool:
    if include_names is not None and name not in set(include_names):
        return False
    if exclude_names is not None and name in set(exclude_names):
        return False
    return True


def _backend_patterns(backend: str) -> list[str]:
    if backend == "tilelang":
        return ["dequant_gemm_epilogue"]
    if backend == "cutile":
        return ["nvfp4_packed_dequant_gemm_epilogue"]
    return ["gemm_epilogue"]


def _target_plan_for_backend(
    *,
    name: str,
    backend: str,
    target_arch: str | None,
    min_speedup: float,
) -> OperatorOptimizationTargetPlan:
    patterns = _backend_patterns(backend)
    common: dict[str, Any] = {
        "name": f"{name}_{backend}",
        "backend": backend,
        "target_path": name,
        "patterns": patterns,
        "fallback": "eager",
        "min_speedup": float(min_speedup),
        "validate": {"atol": 1e-2, "rtol": 1e-2},
    }
    if backend == "tilelang":
        common["tilelang"] = {
            "target": "cuda",
            "target_arch": target_arch,
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
        }
    elif backend == "cutile":
        common["cutile"] = {
            "target": "cuda",
            "target_arch": target_arch,
            "threads": 128,
        }
    else:
        common["cute_dsl"] = {
            "target_arch": target_arch,
            "tile_shape": [128, 128, 64],
            "cluster_shape": None,
        }
    return OperatorOptimizationTargetPlan(**common)


def collect_flux2_klein_nvfp4_targets(
    model_or_pipeline: Any,
    *,
    backend: str,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> tuple[list[OperatorOptimizationTargetPlan], list[Flux2KleinNVFP4TargetSummary]]:
    """Collect bridgeable NVFP4 Linear targets for one backend.

    Target paths are relative to the resolved module. For Diffusers pipelines this
    helper scans `pipeline.transformer`, and `materialize_flux2_klein_nvfp4_backend`
    writes the optimized transformer back to the pipeline.
    """

    backend = normalize_flux2_klein_nvfp4_backend(backend)
    module, _ = _resolve_module(model_or_pipeline)
    resolved_arch = target_arch or _cuda_arch()
    targets: list[OperatorOptimizationTargetPlan] = []
    summaries: list[Flux2KleinNVFP4TargetSummary] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if not _matches_filters(name, include_names=include_names, exclude_names=exclude_names):
            continue
        layout = infer_nvfp4_tensor_layout(child)
        if layout is None:
            continue
        target = _target_plan_for_backend(
            name=name,
            backend=backend,
            target_arch=resolved_arch,
            min_speedup=min_speedup,
        )
        targets.append(target)
        summaries.append(
            Flux2KleinNVFP4TargetSummary(
                name=name,
                backend=backend,
                module_type=type(child).__name__,
                input_features=layout.input_features,
                output_features=layout.output_features,
                group_size=layout.group_size,
                patterns=list(target.patterns),
            )
        )
        if max_targets is not None and len(targets) >= int(max_targets):
            break
    return targets, summaries


def collect_flux2_klein_nvfp4_backend_targets(
    model_or_pipeline: Any,
    *,
    backends: Iterable[str] = FLUX2_KLEIN_NVFP4_BACKENDS,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> dict[str, tuple[list[OperatorOptimizationTargetPlan], list[Flux2KleinNVFP4TargetSummary]]]:
    """Collect target plans for all requested FLUX.2 klein NVFP4 backends."""

    return {
        normalize_flux2_klein_nvfp4_backend(backend): collect_flux2_klein_nvfp4_targets(
            model_or_pipeline,
            backend=backend,
            target_arch=target_arch,
            max_targets=max_targets,
            include_names=include_names,
            exclude_names=exclude_names,
            min_speedup=min_speedup,
        )
        for backend in backends
    }


def materialize_flux2_klein_nvfp4_backend(
    model_or_pipeline: Any,
    *,
    backend: str,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    inplace: bool = False,
    min_speedup: float = 0.0,
) -> Flux2KleinNVFP4BackendResult:
    """Materialize a callable XQT backend inference path for FLUX.2 klein NVFP4."""

    backend = normalize_flux2_klein_nvfp4_backend(backend)
    module, pipeline_component = _resolve_module(model_or_pipeline)
    targets, summaries = collect_flux2_klein_nvfp4_targets(
        module,
        backend=backend,
        target_arch=target_arch,
        max_targets=max_targets,
        include_names=include_names,
        exclude_names=exclude_names,
        min_speedup=min_speedup,
    )
    if not targets:
        raise XQTBackendError("no bridgeable FLUX.2 klein NVFP4 Linear targets found")
    optimized_module = materialize_operator_candidate_models(
        module,
        targets,
        inplace=inplace,
    )
    if pipeline_component is None:
        return Flux2KleinNVFP4BackendResult(
            model=optimized_module,
            backend=backend,
            targets=targets,
            target_summaries=summaries,
        )
    pipeline = model_or_pipeline if inplace else copy.copy(model_or_pipeline)
    setattr(pipeline, pipeline_component, optimized_module)
    return Flux2KleinNVFP4BackendResult(
        model=pipeline,
        backend=backend,
        targets=targets,
        target_summaries=summaries,
    )


def compile_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    backend_name: str | None = None,
    materialized_target_count: int = 0,
    backend: str = "inductor",
    mode: str | None = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = False,
    options: Mapping[str, Any] | None = None,
) -> Flux2KleinNVFP4CompiledTransformerResult:
    """Compile a FLUX.2 klein transformer for steady-state inference."""

    if mode not in {None, "default"} and options:
        raise XQTBackendError(
            "torch.compile in PyTorch 2.12 does not allow mode and options at the same time"
        )
    compile_plan = OperatorOptimizationTargetPlan(
        name="flux2_klein_nvfp4_transformer_compile",
        backend="torch_compile",
        options={"backend": backend, **(dict(options) if options is not None else {})},
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    compiled_model, compile_time_ms = compile_with_torch(transformer, compile_plan)
    return Flux2KleinNVFP4CompiledTransformerResult(
        model=compiled_model,
        backend=None if backend_name is None else normalize_flux2_klein_nvfp4_backend(backend_name),
        materialized_target_count=int(materialized_target_count),
        compile_backend=str(backend),
        compile_mode=None if mode in {None, "default"} else str(mode),
        compile_time_ms=float(compile_time_ms),
        warmup_iterations=0,
        warmup_time_ms=0.0,
    )


def optimize_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    backend: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    compile_backend: str = "inductor",
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    compile_options: Mapping[str, Any] | None = None,
    hidden_states: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
    img_ids: torch.Tensor | None = None,
    txt_ids: torch.Tensor | None = None,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    warmup_iterations: int = 0,
    inplace: bool = False,
) -> Flux2KleinNVFP4CompiledTransformerResult:
    """Materialize an optional backend, compile the whole transformer, and optionally warm it up."""

    optimized_model = transformer if inplace else copy.deepcopy(transformer)
    materialized_target_count = 0
    normalized_backend: str | None = None
    if backend is not None:
        backend_result = materialize_flux2_klein_nvfp4_backend(
            optimized_model,
            backend=backend,
            target_arch=target_arch,
            max_targets=max_targets,
            include_names=include_names,
            exclude_names=exclude_names,
            inplace=True,
            min_speedup=min_speedup,
        )
        optimized_model = backend_result.model
        normalized_backend = backend_result.backend
        materialized_target_count = backend_result.target_count
    compiled = compile_flux2_klein_nvfp4_transformer(
        optimized_model,
        backend_name=normalized_backend,
        materialized_target_count=materialized_target_count,
        backend=compile_backend,
        mode=compile_mode,
        fullgraph=compile_fullgraph,
        dynamic=compile_dynamic,
        options=compile_options,
    )
    warmup_time_ms = 0.0
    if warmup_iterations > 0:
        required_inputs = (
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
        )
        if any(value is None for value in required_inputs):
            raise XQTBackendError(
                "warmup requires hidden_states, encoder_hidden_states, timestep, img_ids, and txt_ids"
            )
        warmup_time_ms = warmup_flux2_klein_nvfp4_transformer(
            compiled.model,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            warmup_iterations=warmup_iterations,
        )
    return Flux2KleinNVFP4CompiledTransformerResult(
        model=compiled.model,
        backend=compiled.backend,
        materialized_target_count=compiled.materialized_target_count,
        compile_backend=compiled.compile_backend,
        compile_mode=compiled.compile_mode,
        compile_time_ms=compiled.compile_time_ms,
        warmup_iterations=int(warmup_iterations),
        warmup_time_ms=float(warmup_time_ms),
    )


def warmup_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    warmup_iterations: int = 6,
    sync_cuda: bool = True,
) -> float:
    """Run explicit warmup for a FLUX.2 klein transformer forward path."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if warmup_iterations == 0:
        return 0.0

    def _forward_once() -> object:
        return transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )

    start = perf_counter()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _forward_once()
        if sync_cuda and any(tensor.is_cuda for tensor in (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)):
            torch.cuda.synchronize(hidden_states.device)
    return float((perf_counter() - start) * 1000.0)


def benchmark_flux2_klein_nvfp4_transformer_forward(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    warmup: int = 6,
    iterations: int = 20,
    sync_cuda: bool = True,
) -> dict[str, Any]:
    """Benchmark one FLUX.2 klein transformer forward path with explicit warmup."""

    def _forward_once() -> object:
        return transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )

    device = str(hidden_states.device)
    return benchmark_callable(
        _forward_once,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    ).to_dict()


def _resolve_flux2_klein_nvfp4_model_file(
    model_file: str | None,
    *,
    local_files_only: bool,
) -> str:
    if model_file is not None:
        return model_file
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return flux2_klein_nvfp4_single_file_url()
    return hf_hub_download(
        repo_id=FLUX2_KLEIN_4B_NVFP4_REPO_ID,
        filename=FLUX2_KLEIN_4B_NVFP4_FILENAME,
        local_files_only=local_files_only,
    )


def _local_safetensors_path(source: str) -> Path | None:
    try:
        path = Path(source)
    except TypeError:
        return None
    if path.is_file() and path.suffix == ".safetensors":
        return path
    return None


def _modelopt_nvfp4_layers(source: str) -> tuple[str, ...]:
    path = _local_safetensors_path(source)
    if path is None:
        return ()
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise XQTBackendError("safetensors is required to inspect FLUX.2 NVFP4 weights") from exc
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        raw_metadata = (handle.metadata() or {}).get("_quantization_metadata")
    if raw_metadata is None:
        return ()
    metadata = json.loads(raw_metadata)
    layers = metadata.get("layers")
    if not isinstance(layers, Mapping):
        return ()
    return tuple(
        str(name)
        for name, entry in layers.items()
        if isinstance(entry, Mapping) and entry.get("format") == "nvfp4"
    )


def _map_modelopt_nvfp4_layer(source_name: str) -> list[_MappedNVFP4Layer]:
    parts = source_name.split(".")
    if len(parts) < 4:
        return []
    if parts[0] == "double_blocks" and len(parts) == 4:
        block_index = parts[1]
        stream = parts[2]
        leaf = parts[3]
        block_prefix = f"transformer_blocks.{block_index}"
        if stream == "img_attn" and leaf == "proj":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_out.0",
                )
            ]
        if stream == "txt_attn" and leaf == "proj":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_add_out",
                )
            ]
        if stream == "img_attn" and leaf == "qkv":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_q",
                    chunk_index=0,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_k",
                    chunk_index=1,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.to_v",
                    chunk_index=2,
                    chunk_count=3,
                ),
            ]
        if stream == "txt_attn" and leaf == "qkv":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.add_q_proj",
                    chunk_index=0,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.add_k_proj",
                    chunk_index=1,
                    chunk_count=3,
                ),
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.attn.add_v_proj",
                    chunk_index=2,
                    chunk_count=3,
                ),
            ]
        if stream == "img_mlp" and leaf == "0":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff.linear_in",
                )
            ]
        if stream == "img_mlp" and leaf == "2":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff.linear_out",
                )
            ]
        if stream == "txt_mlp" and leaf == "0":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff_context.linear_in",
                )
            ]
        if stream == "txt_mlp" and leaf == "2":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.ff_context.linear_out",
                )
            ]
        return []
    if parts[0] == "single_blocks" and len(parts) == 3:
        block_index = parts[1]
        block_prefix = f"single_transformer_blocks.{block_index}.attn"
        if parts[2] == "linear1":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.to_qkv_mlp_proj",
                )
            ]
        if parts[2] == "linear2":
            return [
                _MappedNVFP4Layer(
                    source_name=source_name,
                    target_path=f"{block_prefix}.to_out",
                )
            ]
    return []


def _replace_submodule(root: nn.Module, target_path: str, module: nn.Module) -> None:
    parent_path, _, child_name = target_path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child_name, module)


def _map_modelopt_non_quant_key(source_key: str) -> str | None:
    renamed = _NON_QUANT_KEY_RENAMES.get(source_key)
    if renamed is not None:
        return renamed
    parts = source_key.split(".")
    if (
        len(parts) == 6
        and parts[0] == "double_blocks"
        and parts[3] == "norm"
        and parts[5] == "scale"
    ):
        block_prefix = f"transformer_blocks.{parts[1]}.attn"
        if parts[2] == "img_attn" and parts[4] == "query_norm":
            return f"{block_prefix}.norm_q.weight"
        if parts[2] == "img_attn" and parts[4] == "key_norm":
            return f"{block_prefix}.norm_k.weight"
        if parts[2] == "txt_attn" and parts[4] == "query_norm":
            return f"{block_prefix}.norm_added_q.weight"
        if parts[2] == "txt_attn" and parts[4] == "key_norm":
            return f"{block_prefix}.norm_added_k.weight"
    if (
        len(parts) == 5
        and parts[0] == "single_blocks"
        and parts[2] == "norm"
        and parts[4] == "scale"
    ):
        block_prefix = f"single_transformer_blocks.{parts[1]}.attn"
        if parts[3] == "query_norm":
            return f"{block_prefix}.norm_q.weight"
        if parts[3] == "key_norm":
            return f"{block_prefix}.norm_k.weight"
    return None


def _slice_rows(
    tensor: torch.Tensor | None,
    *,
    chunk_index: int | None,
    chunk_count: int | None,
) -> torch.Tensor | None:
    if tensor is None or chunk_index is None or chunk_count is None:
        return tensor
    if tensor.ndim == 0:
        return tensor
    rows = int(tensor.shape[0])
    if rows % int(chunk_count) != 0:
        raise XQTBackendError(
            f"cannot split FLUX.2 NVFP4 tensor with rows={rows} into {chunk_count} chunks"
        )
    chunk_rows = rows // int(chunk_count)
    start = int(chunk_index) * chunk_rows
    return tensor[start : start + chunk_rows]


def _load_modelopt_nvfp4_linear(handle: Any, mapped: _MappedNVFP4Layer) -> nn.Module:
    keys = set(handle.keys())
    source = mapped.source_name
    missing = [
        name
        for name in _MODEL_OPT_NVFP4_PARAMS
        if f"{source}.{name}" not in keys
    ]
    if missing:
        raise XQTBackendError(
            f"FLUX.2 NVFP4 layer {source} is missing tensors: {', '.join(sorted(missing))}"
        )
    bias_key = f"{source}.bias"
    bias = handle.get_tensor(bias_key) if bias_key in keys else None
    return _Flux2KleinNVFP4Linear(
        packed_weight=_slice_rows(
            handle.get_tensor(f"{source}.weight"),
            chunk_index=mapped.chunk_index,
            chunk_count=mapped.chunk_count,
        ),
        weight_scale=_slice_rows(
            handle.get_tensor(f"{source}.weight_scale"),
            chunk_index=mapped.chunk_index,
            chunk_count=mapped.chunk_count,
        ),
        weight_scale_2=handle.get_tensor(f"{source}.weight_scale_2"),
        input_scale=handle.get_tensor(f"{source}.input_scale"),
        bias=_slice_rows(
            bias,
            chunk_index=mapped.chunk_index,
            chunk_count=mapped.chunk_count,
        ),
        source_name=source,
    )


def _load_modelopt_nvfp4_transformer(
    source: str,
    *,
    transformer_cls: type[Any],
    config: str | None,
    config_subfolder: str | None,
    dtype: torch.dtype,
    device: str | torch.device | None,
    local_files_only: bool,
    **kwargs: Any,
) -> nn.Module:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise XQTBackendError("safetensors is required to load FLUX.2 NVFP4 weights") from exc
    if config is None:
        raise XQTBackendError(
            "modelopt FLUX.2 NVFP4 loading requires a Diffusers transformer config"
        )
    config_kwargs = {
        "pretrained_model_name_or_path": config,
        "local_files_only": local_files_only,
    }
    if config_subfolder is not None:
        config_kwargs["subfolder"] = config_subfolder
    config_revision = kwargs.get("config_revision")
    if config_revision is not None:
        config_kwargs["revision"] = config_revision
    model_config = transformer_cls.load_config(**config_kwargs)
    transformer = transformer_cls.from_config(model_config)
    layers = _modelopt_nvfp4_layers(source)
    replaced_targets = 0
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for layer in layers:
            for mapped in _map_modelopt_nvfp4_layer(layer):
                target_module = _load_modelopt_nvfp4_linear(handle, mapped)
                _replace_submodule(transformer, mapped.target_path, target_module)
                replaced_targets += 1
        state_dict = {}
        for source_key in keys:
            target_key = _map_modelopt_non_quant_key(source_key)
            if target_key is not None:
                state_dict[target_key] = handle.get_tensor(source_key)
    if replaced_targets == 0:
        raise XQTBackendError("no FLUX.2 modelopt NVFP4 Linear layers were mapped")
    transformer.load_state_dict(state_dict, strict=False)
    transformer.eval()
    if dtype is not None:
        transformer.to(dtype=dtype)
    if device is not None:
        transformer.to(device=device)
    return transformer


def load_flux2_klein_nvfp4_transformer(
    *,
    model_file: str | None = None,
    config: str | None = FLUX2_KLEIN_4B_REPO_ID,
    config_subfolder: str | None = "transformer",
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Lazy-load the FLUX.2 klein NVFP4 transformer single-file checkpoint."""

    try:
        from diffusers import Flux2Transformer2DModel
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load FLUX.2 klein NVFP4. Install a version exposing Flux2Transformer2DModel."
        ) from exc
    source = _resolve_flux2_klein_nvfp4_model_file(
        model_file,
        local_files_only=local_files_only,
    )
    load_kwargs = dict(kwargs)
    if config is not None and "original_config" not in load_kwargs:
        load_kwargs["config"] = config
    if (
        config_subfolder is not None
        and "subfolder" not in load_kwargs
        and "original_config" not in load_kwargs
    ):
        load_kwargs["subfolder"] = config_subfolder
    if _modelopt_nvfp4_layers(source):
        return _load_modelopt_nvfp4_transformer(
            source,
            transformer_cls=Flux2Transformer2DModel,
            config=config,
            config_subfolder=config_subfolder,
            dtype=dtype,
            device=device,
            local_files_only=local_files_only,
            **kwargs,
        )
    transformer = Flux2Transformer2DModel.from_single_file(
        source,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **load_kwargs,
    )
    transformer.eval()
    if device is not None:
        transformer.to(device=device)
    return transformer


def load_flux2_klein_nvfp4_pipeline(
    *,
    model_file: str | None = None,
    base_repo_id: str = FLUX2_KLEIN_4B_REPO_ID,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> Any:
    """Lazy-load a Diffusers FLUX.2 klein pipeline with the NVFP4 transformer."""

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load FLUX.2 klein pipelines. Install a version exposing Flux2KleinPipeline."
        ) from exc
    transformer = load_flux2_klein_nvfp4_transformer(
        model_file=model_file,
        config=base_repo_id,
        config_subfolder="transformer",
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )
    pipeline = Flux2KleinPipeline.from_pretrained(
        base_repo_id,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    if device is not None:
        pipeline.to(device)
    return pipeline


def run_flux2_klein_nvfp4_inference(
    pipeline: Any,
    *,
    prompt: str | list[str],
    backend: str,
    target_arch: str | None = None,
    max_targets: int | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Diffusers FLUX.2 klein pipeline after XQT backend materialization."""

    result = materialize_flux2_klein_nvfp4_backend(
        pipeline,
        backend=backend,
        target_arch=target_arch,
        max_targets=max_targets,
        inplace=True,
    )
    with torch.inference_mode():
        return result.model(prompt=prompt, **kwargs)


__all__ = [
    "FLUX2_KLEIN_4B_NVFP4_FILENAME",
    "FLUX2_KLEIN_4B_NVFP4_REPO_ID",
    "FLUX2_KLEIN_4B_REPO_ID",
    "FLUX2_KLEIN_NVFP4_BACKENDS",
    "Flux2KleinNVFP4BackendResult",
    "Flux2KleinNVFP4CompiledTransformerResult",
    "Flux2KleinNVFP4TargetSummary",
    "benchmark_flux2_klein_nvfp4_transformer_forward",
    "compile_flux2_klein_nvfp4_transformer",
    "collect_flux2_klein_nvfp4_backend_targets",
    "collect_flux2_klein_nvfp4_targets",
    "flux2_klein_nvfp4_single_file_url",
    "load_flux2_klein_nvfp4_pipeline",
    "load_flux2_klein_nvfp4_transformer",
    "materialize_flux2_klein_nvfp4_backend",
    "normalize_flux2_klein_nvfp4_backend",
    "optimize_flux2_klein_nvfp4_transformer",
    "run_flux2_klein_nvfp4_inference",
    "warmup_flux2_klein_nvfp4_transformer",
]
