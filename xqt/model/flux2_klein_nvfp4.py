"""FLUX.2 klein NVFP4 model-side engine inference helpers."""

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

from xqt.analysis.compare import compare_tensors
from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    materialize_operator_candidate_models,
)
from xqt.operator_opt.compile_backend import compile_with_torch
from xqt.operator_opt.executor import (
    _benchmark_paired_callables,
    _capture_cuda_graph_with_static_state,
    _cuda_graph_tensor_signature,
    _replay_cuda_graph_tensor_callable,
)
from xqt.quant import bridge_module_to_nvfp4_linear_shared, infer_nvfp4_tensor_layout


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


def _should_skip_tilelang_materialization_for_cuda_graph(
    *,
    engine: str | None,
    optimization_kind: str,
    target_arch: str | None,
) -> bool:
    normalized_engine = None if engine is None else normalize_flux2_klein_nvfp4_engine(engine)
    kind = str(optimization_kind).strip().lower().replace("-", "_")
    resolved_arch = target_arch or _cuda_arch()
    return (
        normalized_engine == "tilelang"
        and kind == "cuda_graph"
        and resolved_arch == "sm_89"
    )


def _resolve_module(model_or_pipeline: Any) -> tuple[nn.Module, str | None]:
    if isinstance(model_or_pipeline, nn.Module):
        return model_or_pipeline, None
    transformer = getattr(model_or_pipeline, "transformer", None)
    if isinstance(transformer, nn.Module):
        return transformer, "transformer"
    raise XQTBackendError(
        "FLUX.2 klein NVFP4 engine inference requires an nn.Module or a pipeline with an nn.Module transformer"
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


def _engine_patterns(engine: str) -> list[str]:
    if engine == "tilelang":
        return ["dequant_gemm_epilogue"]
    if engine == "cutile":
        return ["nvfp4_packed_dequant_gemm_epilogue"]
    return ["gemm_epilogue"]


def _flux2_forward_kwargs(
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
    joint_attention_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "guidance": guidance,
        "joint_attention_kwargs": (
            None if joint_attention_kwargs is None else dict(joint_attention_kwargs)
        ),
        "return_dict": False,
    }


def _flux2_dynamic_inputs(
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    dynamic_inputs = [
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
    ]
    if guidance is not None:
        dynamic_inputs.append(guidance)
    return tuple(dynamic_inputs)


def _split_flux2_dynamic_inputs(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if len(runtime_args) not in {5, 6}:
        raise XQTBackendError(
            "FLUX.2 transformer CUDA Graph replay expects 5 or 6 tensor inputs"
        )
    guidance = runtime_args[5] if len(runtime_args) == 6 else None
    return (
        runtime_args[0],
        runtime_args[1],
        runtime_args[2],
        runtime_args[3],
        runtime_args[4],
        guidance,
    )


def _forward_flux2_klein_nvfp4_transformer_once(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        output = transformer(
            **_flux2_forward_kwargs(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        )
    if not isinstance(output, tuple) or not output:
        raise XQTBackendError("FLUX.2 transformer forward must return a non-empty tuple")
    first = output[0]
    if not isinstance(first, torch.Tensor):
        raise XQTBackendError("FLUX.2 transformer forward[0] must be a tensor")
    return first


def _flux2_cuda_graph_signature(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(_cuda_graph_tensor_signature(tensor) for tensor in runtime_args)


class _Flux2KleinNVFP4CudaGraphModule(nn.Module):
    """Callable whole-transformer CUDA Graph wrapper with strict signature checks."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        graph_state: Mapping[str, Any],
        input_signature: tuple[tuple[Any, ...], ...],
        joint_attention_kwargs: Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self._graph_state = graph_state
        self._input_signature = input_signature
        self._joint_attention_kwargs = (
            None if joint_attention_kwargs is None else dict(joint_attention_kwargs)
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: Mapping[str, Any] | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        if return_dict:
            raise XQTBackendError("FLUX.2 CUDA Graph helper only supports return_dict=False")
        if joint_attention_kwargs is not None and dict(joint_attention_kwargs) != dict(
            self._joint_attention_kwargs or {}
        ):
            raise XQTBackendError(
                "FLUX.2 CUDA Graph replay requires the same joint_attention_kwargs used at capture time"
            )
        runtime_args = _flux2_dynamic_inputs(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        runtime_signature = _flux2_cuda_graph_signature(runtime_args)
        if runtime_signature != self._input_signature:
            raise XQTBackendError(
                "FLUX.2 CUDA Graph replay requires matching shape/stride/dtype/device inputs"
            )
        with torch.no_grad():
            output = _replay_cuda_graph_tensor_callable(self._graph_state, runtime_args)
        return (output,)


def _target_plan_for_engine(
    *,
    name: str,
    engine: str,
    target_arch: str | None,
    min_speedup: float,
) -> OperatorOptimizationTargetPlan:
    patterns = _engine_patterns(engine)
    common: dict[str, Any] = {
        "name": f"{name}_{engine}",
        "engine": engine,
        "target_path": name,
        "patterns": patterns,
        "fallback": "eager",
        "min_speedup": float(min_speedup),
        "validate": {"atol": 1e-2, "rtol": 1e-2},
    }
    if engine == "tilelang":
        common["tilelang"] = {
            "target": "cuda",
            "target_arch": target_arch,
            "linear_runtime": "auto",
            "linear_fastpath": "auto",
        }
    elif engine == "cutile":
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
    engine: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> tuple[list[OperatorOptimizationTargetPlan], list[Flux2KleinNVFP4TargetSummary]]:
    """Collect bridgeable NVFP4 Linear targets for one XQT engine.

    Target paths are relative to the resolved module. For Diffusers pipelines this
    helper scans `pipeline.transformer`, and `materialize_flux2_klein_nvfp4_engine`
    writes the optimized transformer back to the pipeline.
    """

    resolved_engine = _resolve_flux2_klein_nvfp4_engine(
        engine=engine,
        context="collect_flux2_klein_nvfp4_targets",
    )
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
        target = _target_plan_for_engine(
            name=name,
            engine=resolved_engine,
            target_arch=resolved_arch,
            min_speedup=min_speedup,
        )
        targets.append(target)
        summaries.append(
            Flux2KleinNVFP4TargetSummary(
                name=name,
                engine=resolved_engine,
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


def collect_flux2_klein_nvfp4_engine_targets(
    model_or_pipeline: Any,
    *,
    engines: Iterable[str] | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> dict[str, tuple[list[OperatorOptimizationTargetPlan], list[Flux2KleinNVFP4TargetSummary]]]:
    """Collect target plans for requested FLUX.2 klein NVFP4 engines."""

    requested_engines = FLUX2_KLEIN_NVFP4_ENGINES
    if engines is not None:
        requested_engines = tuple(engines)

    return {
        normalize_flux2_klein_nvfp4_engine(engine): collect_flux2_klein_nvfp4_targets(
            model_or_pipeline,
            engine=engine,
            target_arch=target_arch,
            max_targets=max_targets,
            include_names=include_names,
            exclude_names=exclude_names,
            min_speedup=min_speedup,
        )
        for engine in requested_engines
    }


def materialize_flux2_klein_nvfp4_engine(
    model_or_pipeline: Any,
    *,
    engine: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    inplace: bool = False,
    min_speedup: float = 0.0,
) -> Flux2KleinNVFP4EngineResult:
    """Materialize a callable XQT engine inference path for FLUX.2 klein NVFP4."""

    resolved_engine = _resolve_flux2_klein_nvfp4_engine(
        engine=engine,
        context="materialize_flux2_klein_nvfp4_engine",
    )
    module, pipeline_component = _resolve_module(model_or_pipeline)
    targets, summaries = collect_flux2_klein_nvfp4_targets(
        module,
        engine=resolved_engine,
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
        return Flux2KleinNVFP4EngineResult(
            model=optimized_module,
            engine=resolved_engine,
            targets=targets,
            target_summaries=summaries,
        )
    pipeline = model_or_pipeline if inplace else copy.copy(model_or_pipeline)
    setattr(pipeline, pipeline_component, optimized_module)
    return Flux2KleinNVFP4EngineResult(
        model=pipeline,
        engine=resolved_engine,
        targets=targets,
        target_summaries=summaries,
    )


def compile_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    engine_name: str | None = None,
    materialized_target_count: int = 0,
    compile_engine: str = "inductor",
    mode: str | None = None,
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
        engine="torch_compile",
        options={
            "engine": compile_engine,
            **(dict(options) if options is not None else {}),
        },
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    compiled_model, compile_time_ms = compile_with_torch(transformer, compile_plan)
    return Flux2KleinNVFP4CompiledTransformerResult(
        model=compiled_model,
        engine=None if engine_name is None else normalize_flux2_klein_nvfp4_engine(engine_name),
        materialized_target_count=int(materialized_target_count),
        compile_engine=str(compile_engine),
        compile_mode=None if mode in {None, "default"} else str(mode),
        compile_time_ms=float(compile_time_ms),
        warmup_iterations=0,
        warmup_time_ms=0.0,
    )


def capture_flux2_klein_nvfp4_transformer_cuda_graph(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
    engine_name: str | None = None,
    materialized_target_count: int = 0,
    warmup_iterations: int = 6,
) -> Flux2KleinNVFP4CudaGraphTransformerResult:
    """Capture a fixed-shape whole-transformer CUDA Graph replay path."""

    runtime_args = _flux2_dynamic_inputs(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
    )
    if not runtime_args:
        raise XQTBackendError("CUDA Graph capture requires tensor inputs")
    if not all(tensor.is_cuda for tensor in runtime_args):
        raise XQTBackendError("FLUX.2 CUDA Graph capture requires CUDA tensor inputs")
    signature = _flux2_cuda_graph_signature(runtime_args)
    capture_start = perf_counter()

    def _capture_body(*dynamic_runtime_args: torch.Tensor) -> torch.Tensor:
        (
            capture_hidden_states,
            capture_encoder_hidden_states,
            capture_timestep,
            capture_img_ids,
            capture_txt_ids,
            capture_guidance,
        ) = _split_flux2_dynamic_inputs(dynamic_runtime_args)
        return _forward_flux2_klein_nvfp4_transformer_once(
            transformer,
            hidden_states=capture_hidden_states,
            encoder_hidden_states=capture_encoder_hidden_states,
            timestep=capture_timestep,
            img_ids=capture_img_ids,
            txt_ids=capture_txt_ids,
            guidance=capture_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    graph_state = _capture_cuda_graph_with_static_state(
        runtime_args,
        body=_capture_body,
        warmup=warmup_iterations,
    )
    capture_time_ms = float((perf_counter() - capture_start) * 1000.0)
    wrapped = _Flux2KleinNVFP4CudaGraphModule(
        transformer=transformer,
        graph_state=graph_state,
        input_signature=signature,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    return Flux2KleinNVFP4CudaGraphTransformerResult(
        model=wrapped,
        engine=None if engine_name is None else normalize_flux2_klein_nvfp4_engine(engine_name),
        materialized_target_count=int(materialized_target_count),
        graph_state=graph_state,
        input_signature=signature,
        warmup_iterations=int(warmup_iterations),
        capture_time_ms=capture_time_ms,
    )


def optimize_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    engine: str | None = None,
    optimization_kind: str = "compile",
    target_arch: str | None = None,
    max_targets: int | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    compile_engine: str = "inductor",
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
) -> Flux2KleinNVFP4CompiledTransformerResult | Flux2KleinNVFP4CudaGraphTransformerResult:
    """Materialize an optional engine, then build compile or CUDA Graph whole-model fastpaths."""

    optimized_model = transformer if inplace else copy.deepcopy(transformer)
    materialized_target_count = 0
    normalized_engine: str | None = None
    kind = str(optimization_kind).strip().lower().replace("-", "_")
    if engine is not None:
        normalized_engine = _resolve_flux2_klein_nvfp4_engine(
            engine=engine,
            context="optimize_flux2_klein_nvfp4_transformer",
        )
    skip_tilelang_materialization = _should_skip_tilelang_materialization_for_cuda_graph(
        engine=normalized_engine,
        optimization_kind=kind,
        target_arch=target_arch,
    )
    if normalized_engine is not None and not skip_tilelang_materialization:
        engine_result = materialize_flux2_klein_nvfp4_engine(
            optimized_model,
            engine=normalized_engine,
            target_arch=target_arch,
            max_targets=max_targets,
            include_names=include_names,
            exclude_names=exclude_names,
            inplace=True,
            min_speedup=min_speedup,
        )
        optimized_model = engine_result.model
        normalized_engine = engine_result.engine
        materialized_target_count = engine_result.target_count
    required_inputs = (
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
    )
    if kind == "cuda_graph":
        if any(value is None for value in required_inputs):
            raise XQTBackendError(
                "cuda_graph optimization requires hidden_states, encoder_hidden_states, timestep, img_ids, and txt_ids"
            )
        return capture_flux2_klein_nvfp4_transformer_cuda_graph(
            optimized_model,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            engine_name=normalized_engine,
            materialized_target_count=materialized_target_count,
            warmup_iterations=warmup_iterations,
        )
    if kind != "compile":
        raise XQTBackendError(
            "optimization_kind must be either 'compile' or 'cuda_graph'"
        )
    compiled = compile_flux2_klein_nvfp4_transformer(
        optimized_model,
        engine_name=normalized_engine,
        materialized_target_count=materialized_target_count,
        compile_engine=compile_engine,
        mode=compile_mode,
        fullgraph=compile_fullgraph,
        dynamic=compile_dynamic,
        options=compile_options,
    )
    warmup_time_ms = 0.0
    if warmup_iterations > 0:
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
        engine=compiled.engine,
        materialized_target_count=compiled.materialized_target_count,
        compile_engine=compiled.compile_engine,
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
            **_flux2_forward_kwargs(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
            )
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
            **_flux2_forward_kwargs(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        )

    device = str(hidden_states.device)
    return benchmark_callable(
        _forward_once,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    ).to_dict()


def benchmark_flux2_klein_nvfp4_transformer_paired(
    *,
    reference_transformer: nn.Module,
    candidate_transformer: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
    warmup: int = 6,
    iterations: int = 20,
    sync_cuda: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Flux2KleinNVFP4PairedBenchmarkResult:
    """Benchmark whole-transformer candidate against eager with paired alternating timing."""

    kwargs = _flux2_forward_kwargs(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
    )

    def _reference_forward() -> object:
        return reference_transformer(**kwargs)

    def _candidate_forward() -> object:
        return candidate_transformer(**kwargs)

    reference_output = _forward_flux2_klein_nvfp4_transformer_once(
        reference_transformer,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    candidate_output = _forward_flux2_klein_nvfp4_transformer_once(
        candidate_transformer,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    diff = compare_tensors(
        reference_output,
        candidate_output,
        atol=atol,
        rtol=rtol,
        include_summary=False,
    )
    reference_report, candidate_report, paired_speedup_ratios = _benchmark_paired_callables(
        _reference_forward,
        _candidate_forward,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=str(hidden_states.device),
    )
    sorted_ratios = sorted(paired_speedup_ratios)
    paired_speedup_p50 = 0.0
    if sorted_ratios:
        midpoint = len(sorted_ratios) // 2
        if len(sorted_ratios) % 2 == 1:
            paired_speedup_p50 = float(sorted_ratios[midpoint])
        else:
            paired_speedup_p50 = float(
                (sorted_ratios[midpoint - 1] + sorted_ratios[midpoint]) / 2.0
            )
    return Flux2KleinNVFP4PairedBenchmarkResult(
        reference_report=reference_report.to_dict(),
        candidate_report=candidate_report.to_dict(),
        paired_speedup_ratios=paired_speedup_ratios,
        paired_speedup_p50=paired_speedup_p50,
        max_abs_vs_eager=float(diff.max_abs),
        mean_abs_vs_eager=float(diff.mean_abs),
        allclose_vs_eager=bool(diff.allclose),
        atol=float(atol),
        rtol=float(rtol),
    )


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
    engine: str | None = None,
    target_arch: str | None = None,
    max_targets: int | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Diffusers FLUX.2 klein pipeline after XQT engine materialization."""

    result = materialize_flux2_klein_nvfp4_engine(
        pipeline,
        engine=engine,
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
    "FLUX2_KLEIN_NVFP4_ENGINES",
    "Flux2KleinNVFP4EngineResult",
    "Flux2KleinNVFP4CompiledTransformerResult",
    "Flux2KleinNVFP4CudaGraphTransformerResult",
    "Flux2KleinNVFP4PairedBenchmarkResult",
    "Flux2KleinNVFP4TargetSummary",
    "benchmark_flux2_klein_nvfp4_transformer_paired",
    "benchmark_flux2_klein_nvfp4_transformer_forward",
    "capture_flux2_klein_nvfp4_transformer_cuda_graph",
    "compile_flux2_klein_nvfp4_transformer",
    "collect_flux2_klein_nvfp4_engine_targets",
    "collect_flux2_klein_nvfp4_targets",
    "flux2_klein_nvfp4_single_file_url",
    "load_flux2_klein_nvfp4_pipeline",
    "load_flux2_klein_nvfp4_transformer",
    "materialize_flux2_klein_nvfp4_engine",
    "normalize_flux2_klein_nvfp4_engine",
    "optimize_flux2_klein_nvfp4_transformer",
    "run_flux2_klein_nvfp4_inference",
    "warmup_flux2_klein_nvfp4_transformer",
]
