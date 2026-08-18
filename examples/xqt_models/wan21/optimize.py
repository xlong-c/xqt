"""Wan 2.1 VAE optimize and benchmark entry points."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn

from xqt.analysis.compare import compare_tensors
from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.operator_opt._benchmark import _benchmark_paired_callables

from .types import (
    Wan21VAECompileResult,
    Wan21VAECudaGraphResult,
    Wan21VAEOptimizationSummary,
    Wan21VAEPairedBenchmarkResult,
    Wan21VAERunMode,
    _normalize_run_mode,
    _normalize_optimization_kind,
    _resolve_vae,
)
from .runtime import (
    _summary_from_vae,
    build_wan21_vae_runner,
    capture_wan21_vae_cuda_graph,
    compile_wan21_vae_runner,
    warmup_wan21_vae_runner,
)


def optimize_wan21_vae(
    model_or_pipeline: Any,
    *,
    run_mode: str = "decode",
    optimization_kind: str = "compile",
    compile_engine: str = "inductor",
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    compile_options: Mapping[str, Any] | None = None,
    tensor: torch.Tensor | None = None,
    enable_tiling: bool = False,
    enable_slicing: bool = False,
    tile_sample_min_height: int | None = None,
    tile_sample_min_width: int | None = None,
    tile_sample_stride_height: int | None = None,
    tile_sample_stride_width: int | None = None,
    materialize_conv3d_fastpath: bool = False,
    conv3d_target_arch: str | None = None,
    conv3d_include_names: list[str] | None = None,
    conv3d_exclude_names: list[str] | None = None,
    conv3d_min_speedup: float = 0.0,
    materialize_rmsnorm_fastpath: bool = False,
    rmsnorm_include_names: list[str] | None = None,
    rmsnorm_exclude_names: list[str] | None = None,
    rmsnorm_min_speedup: float = 0.0,
    rmsnorm_eps: float = 1e-6,
    rmsnorm_block_size: int = 1024,
    rmsnorm_num_warps: int = 4,
    rmsnorm_num_stages: int = 4,
    warmup_iterations: int = 0,
    inplace: bool = False,
) -> tuple[Wan21VAECompileResult | Wan21VAECudaGraphResult, Wan21VAEOptimizationSummary]:
    """Build compile or CUDA Graph whole-runner fastpaths for Wan 2.1 VAE."""

    normalized_run_mode = _normalize_run_mode(run_mode)
    normalized_kind = _normalize_optimization_kind(optimization_kind)
    runner, summary = build_wan21_vae_runner(
        model_or_pipeline,
        run_mode=normalized_run_mode,
        enable_tiling=enable_tiling,
        enable_slicing=enable_slicing,
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_sample_stride_height=tile_sample_stride_height,
        tile_sample_stride_width=tile_sample_stride_width,
        materialize_conv3d_fastpath=materialize_conv3d_fastpath,
        conv3d_target_arch=conv3d_target_arch,
        conv3d_include_names=conv3d_include_names,
        conv3d_exclude_names=conv3d_exclude_names,
        conv3d_min_speedup=conv3d_min_speedup,
        materialize_rmsnorm_fastpath=materialize_rmsnorm_fastpath,
        rmsnorm_include_names=rmsnorm_include_names,
        rmsnorm_exclude_names=rmsnorm_exclude_names,
        rmsnorm_min_speedup=rmsnorm_min_speedup,
        rmsnorm_eps=rmsnorm_eps,
        rmsnorm_block_size=rmsnorm_block_size,
        rmsnorm_num_warps=rmsnorm_num_warps,
        rmsnorm_num_stages=rmsnorm_num_stages,
        inplace=inplace,
    )
    if normalized_kind == "cuda_graph":
        if tensor is None:
            raise XQTBackendError("cuda_graph optimization requires tensor=... with a fixed-shape input")
        graph_result = capture_wan21_vae_cuda_graph(
            runner,
            tensor=tensor,
            run_mode=normalized_run_mode,
            warmup_iterations=warmup_iterations,
        )
        return graph_result, _summary_from_vae(
            runner.vae,  # type: ignore[attr-defined]
            run_mode=normalized_run_mode,
            optimization_kind=normalized_kind,
            compile_engine=None,
            compile_mode=None,
            warmup_iterations=warmup_iterations,
            materialized_conv3d_targets=summary.materialized_conv3d_targets,
            materialized_rmsnorm_targets=summary.materialized_rmsnorm_targets,
        )
    compiled = compile_wan21_vae_runner(
        runner,
        run_mode=normalized_run_mode,
        compile_engine=compile_engine,
        mode=compile_mode,
        fullgraph=compile_fullgraph,
        dynamic=compile_dynamic,
        options=compile_options,
    )
    warmup_time_ms = 0.0
    if warmup_iterations > 0:
        if tensor is None:
            raise XQTBackendError("warmup requires tensor=... for Wan 2.1 VAE optimization")
        warmup_time_ms = warmup_wan21_vae_runner(
            compiled.model,
            tensor=tensor,
            warmup_iterations=warmup_iterations,
        )
    compiled_result = Wan21VAECompileResult(
        model=compiled.model,
        run_mode=compiled.run_mode,
        compile_engine=compiled.compile_engine,
        compile_mode=compiled.compile_mode,
        compile_time_ms=compiled.compile_time_ms,
        warmup_iterations=int(warmup_iterations),
        warmup_time_ms=float(warmup_time_ms),
    )
    return compiled_result, _summary_from_vae(
        runner.vae,  # type: ignore[attr-defined]
        run_mode=normalized_run_mode,
        optimization_kind=normalized_kind,
        compile_engine=compiled.compile_engine,
        compile_mode=compiled.compile_mode,
        warmup_iterations=warmup_iterations,
        materialized_conv3d_targets=summary.materialized_conv3d_targets,
        materialized_rmsnorm_targets=summary.materialized_rmsnorm_targets,
    )


def benchmark_wan21_vae_runner(
    runner: nn.Module,
    *,
    tensor: torch.Tensor,
    run_mode: str = "decode",
    warmup: int = 6,
    iterations: int = 20,
    sync_cuda: bool = True,
) -> dict[str, Any]:
    """Benchmark one Wan 2.1 VAE encode/decode runner with explicit warmup."""

    normalized_run_mode = _normalize_run_mode(run_mode)

    def _forward_once() -> object:
        return runner(tensor)

    report = benchmark_callable(
        _forward_once,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=str(tensor.device),
    ).to_dict()
    report["run_mode"] = normalized_run_mode
    return report


def benchmark_wan21_vae_paired(
    *,
    eager_runner: nn.Module,
    candidate_runner: nn.Module,
    tensor: torch.Tensor,
    run_mode: str = "decode",
    warmup: int = 6,
    iterations: int = 20,
    sync_cuda: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> Wan21VAEPairedBenchmarkResult:
    """Benchmark one candidate Wan 2.1 VAE runner against eager."""

    normalized_run_mode = _normalize_run_mode(run_mode)

    def _reference_forward() -> object:
        return eager_runner(tensor)

    def _candidate_forward() -> object:
        return candidate_runner(tensor)

    reference_output = eager_runner(tensor)
    candidate_output = candidate_runner(tensor)
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
        device=str(tensor.device),
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
    return Wan21VAEPairedBenchmarkResult(
        run_mode=normalized_run_mode,
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
