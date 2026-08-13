"""FLUX.2 klein NVFP4 optimize and benchmark entry points."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from xqt.analysis.compare import compare_tensors
from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.operator_opt._benchmark import _benchmark_paired_callables
from xqt.quant.quantizers.convrot_int8 import ConvRotInt8QuantizationResult

from .types import (
    Flux2KleinNVFP4CompiledTransformerResult,
    Flux2KleinNVFP4CudaGraphTransformerResult,
    Flux2KleinNVFP4PairedBenchmarkResult,
    _cuda_arch,
    _resolve_flux2_klein_nvfp4_engine,
    normalize_flux2_klein_nvfp4_engine,
)
from .targets import materialize_flux2_klein_nvfp4_engine
from .runtime import (
    _flux2_forward_kwargs,
    _forward_flux2_klein_nvfp4_transformer_once,
    capture_flux2_klein_nvfp4_transformer_cuda_graph,
    compile_flux2_klein_nvfp4_transformer,
    warmup_flux2_klein_nvfp4_transformer,
)


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


def optimize_flux2_klein_convrot_int8_transformer(
    transformer: nn.Module,
    *,
    policy: Mapping[str, Any] | None = None,
    calibration_inputs: Sequence[Any] | None = None,
    inplace: bool = False,
    engine: str = "cuda_sm89",
    fallback_engine: str = "torch_int_mm",
    min_int8_rows: int | None = None,
    optimization_kind: str = "compile",
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
) -> tuple[
    ConvRotInt8QuantizationResult,
    Flux2KleinNVFP4CompiledTransformerResult
    | Flux2KleinNVFP4CudaGraphTransformerResult,
]:
    """Quantize ConvRot W8A8, then build a whole-transformer fastpath."""

    from .load import quantize_flux2_klein_bf16_transformer_to_convrot_int8

    quantization = quantize_flux2_klein_bf16_transformer_to_convrot_int8(
        transformer,
        policy=policy,
        calibration_inputs=calibration_inputs,
        inplace=inplace,
        engine=engine,
        fallback_engine=fallback_engine,
        min_int8_rows=min_int8_rows,
    )
    optimized = optimize_flux2_klein_nvfp4_transformer(
        quantization.model,
        engine=None,
        optimization_kind=optimization_kind,
        compile_engine=compile_engine,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_options=compile_options,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
        warmup_iterations=warmup_iterations,
        inplace=True,
    )
    optimized_metadata = dict(quantization.metadata)
    optimized_metadata["whole_transformer_optimization"] = optimized.to_dict()
    optimized_quantization = ConvRotInt8QuantizationResult(
        model=optimized.model,
        backend=quantization.backend,
        method=quantization.method,
        strategy=quantization.strategy,
        quantized_modules=list(quantization.quantized_modules),
        metadata=optimized_metadata,
        compute_config=quantization.compute_config,
    )
    return optimized_quantization, optimized


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


def benchmark_flux2_klein_convrot_int8_transformer_forward(
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
    """Benchmark one Klein ConvRot W8A8 transformer forward path."""

    return benchmark_flux2_klein_nvfp4_transformer_forward(
        transformer,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
    )


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


def benchmark_flux2_klein_convrot_int8_transformer_paired(
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
    """Compare Klein ConvRot W8A8 against an eager transformer in pairs."""

    return benchmark_flux2_klein_nvfp4_transformer_paired(
        reference_transformer=reference_transformer,
        candidate_transformer=candidate_transformer,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        atol=atol,
        rtol=rtol,
    )
