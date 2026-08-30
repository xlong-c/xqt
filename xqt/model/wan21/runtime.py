"""Wan 2.1 VAE runner, compile, CUDA Graph, and warmup helpers."""

from __future__ import annotations

import copy
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.wrappers import OperatorOptimizationTargetPlan
from xqt.kernels.wrappers.compile_backend import compile_with_torch
from xqt.kernels.wrappers.runtime import (
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)

from .types import (
    Wan21VAECompileResult,
    Wan21VAECudaGraphResult,
    Wan21VAEOptimizationSummary,
    Wan21VAERunMode,
    _normalize_run_mode,
    _resolve_vae,
)
from .targets import (
    _maybe_clone_container,
    materialize_wan21_vae_conv3d_fastpath,
    materialize_wan21_vae_rmsnorm_fastpath,
)


def _configure_vae_runtime(
    vae: nn.Module,
    *,
    enable_tiling: bool,
    enable_slicing: bool,
    tile_sample_min_height: int | None,
    tile_sample_min_width: int | None,
    tile_sample_stride_height: int | None,
    tile_sample_stride_width: int | None,
) -> None:
    if enable_slicing:
        if not hasattr(vae, "enable_slicing"):
            raise XQTBackendError("Wan 2.1 VAE does not expose enable_slicing()")
        getattr(vae, "enable_slicing")()
    elif hasattr(vae, "disable_slicing"):
        getattr(vae, "disable_slicing")()

    if enable_tiling:
        if not hasattr(vae, "enable_tiling"):
            raise XQTBackendError("Wan 2.1 VAE does not expose enable_tiling()")
        getattr(vae, "enable_tiling")(
            tile_sample_min_height=tile_sample_min_height,
            tile_sample_min_width=tile_sample_min_width,
            tile_sample_stride_height=tile_sample_stride_height,
            tile_sample_stride_width=tile_sample_stride_width,
        )
    elif hasattr(vae, "disable_tiling"):
        getattr(vae, "disable_tiling")()


def _extract_encode_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "latent_dist"):
        latent_dist = output.latent_dist
        if hasattr(latent_dist, "mode"):
            sample = latent_dist.mode()
            if isinstance(sample, torch.Tensor):
                return sample
    if isinstance(output, tuple) and output:
        first = output[0]
        if hasattr(first, "mode"):
            sample = first.mode()
            if isinstance(sample, torch.Tensor):
                return sample
        if isinstance(first, torch.Tensor):
            return first
    raise XQTBackendError("Wan 2.1 VAE encode output is not a supported latent container")


def _extract_decode_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
        return output.sample
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise XQTBackendError("Wan 2.1 VAE decode output is not a supported tensor container")


def _vae_encode_tensor(vae: nn.Module, video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = vae.encode(video, return_dict=True)
    return _extract_encode_tensor(output)


def _vae_decode_tensor(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = vae.decode(latents, return_dict=True)
    return _extract_decode_tensor(output)


def _runner_output_tensor(run_mode: Wan21VAERunMode, output: Any) -> torch.Tensor:
    if run_mode == "decode":
        return _extract_decode_tensor(output)
    return _extract_encode_tensor(output)


def _summary_from_vae(
    vae: nn.Module,
    *,
    run_mode: Wan21VAERunMode,
    optimization_kind: str,
    compile_engine: str | None,
    compile_mode: str | None,
    warmup_iterations: int,
    materialized_conv3d_targets: int = 0,
    materialized_rmsnorm_targets: int = 0,
) -> Wan21VAEOptimizationSummary:
    return Wan21VAEOptimizationSummary(
        run_mode=run_mode,
        optimization_kind=optimization_kind,
        compile_engine=compile_engine,
        compile_mode=compile_mode,
        tiled=bool(getattr(vae, "use_tiling", False)),
        sliced=bool(getattr(vae, "use_slicing", False)),
        tile_sample_min_height=getattr(vae, "tile_sample_min_height", None),
        tile_sample_min_width=getattr(vae, "tile_sample_min_width", None),
        tile_sample_stride_height=getattr(vae, "tile_sample_stride_height", None),
        tile_sample_stride_width=getattr(vae, "tile_sample_stride_width", None),
        warmup_iterations=int(warmup_iterations),
        materialized_conv3d_targets=int(materialized_conv3d_targets),
        materialized_rmsnorm_targets=int(materialized_rmsnorm_targets),
    )


def _tensor_callable_signature(runtime_args: tuple[torch.Tensor, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(cuda_graph_tensor_signature(tensor) for tensor in runtime_args)


class _Wan21VAERunnerModule(nn.Module):
    """Thin nn.Module wrapper that normalizes Wan 2.1 VAE encode/decode outputs."""

    def __init__(self, *, vae: nn.Module, run_mode: Wan21VAERunMode) -> None:
        super().__init__()
        self.vae = vae
        self.run_mode = run_mode

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.run_mode == "decode":
            return _vae_decode_tensor(self.vae, tensor)
        return _vae_encode_tensor(self.vae, tensor)


class _Wan21VAECudaGraphModule(nn.Module):
    """Replayable fixed-shape Wan 2.1 VAE CUDA Graph wrapper."""

    def __init__(
        self,
        *,
        runner: nn.Module,
        graph_state: Mapping[str, Any],
        input_signature: tuple[tuple[Any, ...], ...],
    ) -> None:
        super().__init__()
        self.runner = runner
        self._graph_state = graph_state
        self._input_signature = input_signature

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        runtime_args = (tensor,)
        runtime_signature = _tensor_callable_signature(runtime_args)
        if runtime_signature != self._input_signature:
            raise XQTBackendError(
                "Wan 2.1 VAE CUDA Graph replay requires matching shape/stride/dtype/device inputs"
            )
        with torch.no_grad():
            return replay_cuda_graph_tensor_callable(self._graph_state, runtime_args)


def build_wan21_vae_runner(
    model_or_pipeline: Any,
    *,
    run_mode: str = "decode",
    enable_tiling: bool = False,
    enable_slicing: bool = False,
    tile_sample_min_height: int | None = None,
    tile_sample_min_width: int | None = None,
    tile_sample_stride_height: int | None = None,
    tile_sample_stride_width: int | None = None,
    materialize_conv3d_fastpath: bool = False,
    conv3d_target_arch: str | None = None,
    conv3d_include_names: Sequence[str] | None = None,
    conv3d_exclude_names: Sequence[str] | None = None,
    conv3d_min_speedup: float = 0.0,
    materialize_rmsnorm_fastpath: bool = False,
    rmsnorm_include_names: Sequence[str] | None = None,
    rmsnorm_exclude_names: Sequence[str] | None = None,
    rmsnorm_min_speedup: float = 0.0,
    rmsnorm_eps: float = 1e-6,
    rmsnorm_block_size: int = 1024,
    rmsnorm_num_warps: int = 4,
    rmsnorm_num_stages: int = 4,
    inplace: bool = False,
) -> tuple[nn.Module, Wan21VAEOptimizationSummary]:
    """Build a normalized Wan 2.1 VAE encode/decode runner."""

    normalized_run_mode = _normalize_run_mode(run_mode)
    container = model_or_pipeline if inplace else copy.deepcopy(model_or_pipeline)
    materialized_conv3d_targets = 0
    materialized_rmsnorm_targets = 0
    if materialize_conv3d_fastpath:
        container, materialized_conv3d_targets = materialize_wan21_vae_conv3d_fastpath(
            container,
            target_arch=conv3d_target_arch,
            include_names=conv3d_include_names,
            exclude_names=conv3d_exclude_names,
            min_speedup=conv3d_min_speedup,
            inplace=True,
        )
    if materialize_rmsnorm_fastpath:
        container, materialized_rmsnorm_targets = materialize_wan21_vae_rmsnorm_fastpath(
            container,
            include_names=rmsnorm_include_names,
            exclude_names=rmsnorm_exclude_names,
            min_speedup=rmsnorm_min_speedup,
            eps=rmsnorm_eps,
            block_size=rmsnorm_block_size,
            num_warps=rmsnorm_num_warps,
            num_stages=rmsnorm_num_stages,
            inplace=True,
        )
    vae, _ = _resolve_vae(container)
    _configure_vae_runtime(
        vae,
        enable_tiling=enable_tiling,
        enable_slicing=enable_slicing,
        tile_sample_min_height=tile_sample_min_height,
        tile_sample_min_width=tile_sample_min_width,
        tile_sample_stride_height=tile_sample_stride_height,
        tile_sample_stride_width=tile_sample_stride_width,
    )
    runner = _Wan21VAERunnerModule(vae=vae, run_mode=normalized_run_mode)
    summary = _summary_from_vae(
        vae,
        run_mode=normalized_run_mode,
        optimization_kind="eager",
        compile_engine=None,
        compile_mode=None,
        warmup_iterations=0,
        materialized_conv3d_targets=materialized_conv3d_targets,
        materialized_rmsnorm_targets=materialized_rmsnorm_targets,
    )
    return runner, summary


def compile_wan21_vae_runner(
    runner: nn.Module,
    *,
    run_mode: str = "decode",
    compile_engine: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool = False,
    options: Mapping[str, Any] | None = None,
) -> Wan21VAECompileResult:
    """Compile a Wan 2.1 VAE runner with torch.compile."""

    normalized_run_mode = _normalize_run_mode(run_mode)
    if mode not in {None, "default"} and options:
        raise XQTBackendError(
            "torch.compile in PyTorch 2.12 does not allow mode and options at the same time"
        )
    compile_plan = OperatorOptimizationTargetPlan(
        name=f"wan21_vae_{normalized_run_mode}_compile",
        engine="torch_compile",
        options={
            "engine": compile_engine,
            **(dict(options) if options is not None else {}),
        },
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    compiled_model, compile_time_ms = compile_with_torch(runner, compile_plan)
    return Wan21VAECompileResult(
        model=compiled_model,
        run_mode=normalized_run_mode,
        compile_engine=str(compile_engine),
        compile_mode=None if mode in {None, "default"} else str(mode),
        compile_time_ms=float(compile_time_ms),
        warmup_iterations=0,
        warmup_time_ms=0.0,
    )


def capture_wan21_vae_cuda_graph(
    runner: nn.Module,
    *,
    tensor: torch.Tensor,
    run_mode: str = "decode",
    warmup_iterations: int = 6,
) -> Wan21VAECudaGraphResult:
    """Capture a fixed-shape Wan 2.1 VAE runner CUDA Graph replay path."""

    normalized_run_mode = _normalize_run_mode(run_mode)
    runtime_args = (tensor,)
    if not tensor.is_cuda:
        raise XQTBackendError("Wan 2.1 VAE CUDA Graph capture requires a CUDA tensor input")
    signature = _tensor_callable_signature(runtime_args)
    capture_start = perf_counter()

    def _capture_body(*dynamic_runtime_args: torch.Tensor) -> torch.Tensor:
        if len(dynamic_runtime_args) != 1:
            raise XQTBackendError("Wan 2.1 VAE CUDA Graph capture expects exactly one tensor input")
        return runner(dynamic_runtime_args[0])

    graph_state = capture_cuda_graph_with_static_state(
        runtime_args,
        body=_capture_body,
        warmup=warmup_iterations,
    )
    capture_time_ms = float((perf_counter() - capture_start) * 1000.0)
    wrapped = _Wan21VAECudaGraphModule(
        runner=runner,
        graph_state=graph_state,
        input_signature=signature,
    )
    return Wan21VAECudaGraphResult(
        model=wrapped,
        run_mode=normalized_run_mode,
        graph_state=graph_state,
        input_signature=signature,
        warmup_iterations=int(warmup_iterations),
        capture_time_ms=capture_time_ms,
    )


def warmup_wan21_vae_runner(
    runner: nn.Module,
    *,
    tensor: torch.Tensor,
    warmup_iterations: int = 6,
    sync_cuda: bool = True,
) -> float:
    """Run explicit warmup for one Wan 2.1 VAE runner forward path."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if warmup_iterations == 0:
        return 0.0
    start = perf_counter()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            runner(tensor)
        if sync_cuda and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
    return float((perf_counter() - start) * 1000.0)
