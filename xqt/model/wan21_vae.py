"""Wan 2.1 VAE model-side high-performance inference helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch import nn

from xqt.analysis.compare import compare_tensors
from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.operator_opt import OperatorOptimizationTargetPlan
from xqt.operator_opt import materialize_operator_candidate_models
from xqt.operator_opt.compile_backend import compile_with_torch
from xqt.operator_opt._benchmark import _benchmark_paired_callables
from xqt.operator_opt.runtime import (
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)


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


def _maybe_clone_container(model_or_pipeline: Any, *, inplace: bool) -> Any:
    return model_or_pipeline if inplace else copy.copy(model_or_pipeline)


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


def collect_wan21_vae_conv3d_targets(
    model_or_pipeline: Any,
    *,
    target_arch: str | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
) -> list[OperatorOptimizationTargetPlan]:
    """Collect TileLang Conv3d 1x1x1 targets from a Wan 2.1 VAE."""

    vae, _ = _resolve_vae(model_or_pipeline)
    include_set = None if include_names is None else set(include_names)
    exclude_set = None if exclude_names is None else set(exclude_names)
    targets: list[OperatorOptimizationTargetPlan] = []
    for name, child in vae.named_modules():
        if not name or not isinstance(child, nn.Conv3d):
            continue
        if include_set is not None and name not in include_set:
            continue
        if exclude_set is not None and name in exclude_set:
            continue
        if (
            tuple(int(value) for value in child.kernel_size) != (1, 1, 1)
            or tuple(int(value) for value in child.stride) != (1, 1, 1)
            or tuple(int(value) for value in child.padding) != (0, 0, 0)
            or tuple(int(value) for value in child.dilation) != (1, 1, 1)
            or int(child.groups) != 1
        ):
            continue
        targets.append(
            OperatorOptimizationTargetPlan(
                name=f"{name}_tilelang_conv3d_1x1x1",
                engine="tilelang",
                target_path=name,
                patterns=["conv3d_1x1x1"],
                fallback="eager",
                min_speedup=float(min_speedup),
                validate={"atol": 1e-2, "rtol": 1e-2},
                tilelang={
                    "target": "cuda",
                    "target_arch": target_arch,
                    "conv_fastpath": "tilelang",
                },
            )
        )
    return targets


def materialize_wan21_vae_conv3d_fastpath(
    model_or_pipeline: Any,
    *,
    target_arch: str | None = None,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    inplace: bool = False,
) -> tuple[Any, int]:
    """Replace eligible Wan 2.1 VAE Conv3d 1x1x1 modules with TileLang wrappers."""

    container = model_or_pipeline if inplace else copy.deepcopy(model_or_pipeline)
    vae, component_name = _resolve_vae(container)
    targets = collect_wan21_vae_conv3d_targets(
        vae,
        target_arch=target_arch,
        include_names=include_names,
        exclude_names=exclude_names,
        min_speedup=min_speedup,
    )
    if not targets:
        return container, 0
    optimized_vae = materialize_operator_candidate_models(
        vae,
        targets,
        inplace=True,
    )
    if component_name is not None:
        setattr(container, component_name, optimized_vae)
    else:
        container = optimized_vae
    return container, len(targets)


def collect_wan21_vae_rmsnorm_targets(
    model_or_pipeline: Any,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    eps: float = 1e-6,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
) -> list[OperatorOptimizationTargetPlan]:
    """Collect Triton RMSNorm targets from a Wan 2.1 VAE."""

    vae, _ = _resolve_vae(model_or_pipeline)
    include_set = None if include_names is None else set(include_names)
    exclude_set = None if exclude_names is None else set(exclude_names)
    targets: list[OperatorOptimizationTargetPlan] = []
    for name, child in vae.named_modules():
        if not name:
            continue
        if include_set is not None and name not in include_set:
            continue
        if exclude_set is not None and name in exclude_set:
            continue
        if not (hasattr(child, "gamma") and hasattr(child, "scale")):
            continue
        targets.append(
            OperatorOptimizationTargetPlan(
                name=f"{name}_triton_rmsnorm",
                engine="triton",
                target_path=name,
                patterns=["rmsnorm"],
                fallback="eager",
                min_speedup=float(min_speedup),
                validate={"atol": 1e-2, "rtol": 1e-2},
                options={
                    "eps": float(eps),
                    "block_size": int(block_size),
                    "num_warps": int(num_warps),
                    "num_stages": int(num_stages),
                },
            )
        )
    return targets


def materialize_wan21_vae_rmsnorm_fastpath(
    model_or_pipeline: Any,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    min_speedup: float = 0.0,
    eps: float = 1e-6,
    block_size: int = 1024,
    num_warps: int = 4,
    num_stages: int = 4,
    inplace: bool = False,
) -> tuple[Any, int]:
    """Replace eligible Wan 2.1 VAE RMSNorm modules with Triton wrappers."""

    container = model_or_pipeline if inplace else copy.deepcopy(model_or_pipeline)
    vae, component_name = _resolve_vae(container)
    targets = collect_wan21_vae_rmsnorm_targets(
        vae,
        include_names=include_names,
        exclude_names=exclude_names,
        min_speedup=min_speedup,
        eps=eps,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if not targets:
        return container, 0
    optimized_vae = materialize_operator_candidate_models(
        vae,
        targets,
        inplace=True,
    )
    if component_name is not None:
        setattr(container, component_name, optimized_vae)
    else:
        container = optimized_vae
    return container, len(targets)


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


def load_wan21_vae(
    *,
    repo_id: str = WAN21_VAE_REPO_ID,
    subfolder: str = WAN21_VAE_SUBFOLDER,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Lazy-load a Wan 2.1 AutoencoderKLWan from Diffusers."""

    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise XQTBackendError(
            "diffusers is required to load Wan 2.1 VAE. Install a version exposing AutoencoderKLWan."
        ) from exc
    vae = AutoencoderKLWan.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    vae.eval()
    if device is not None:
        vae.to(device=device)
    return vae


def load_wan21_pipeline_with_vae(
    *,
    pipeline_cls: type[Any] | None = None,
    repo_id: str = WAN21_VAE_REPO_ID,
    vae_repo_id: str = WAN21_VAE_REPO_ID,
    vae_subfolder: str = WAN21_VAE_SUBFOLDER,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a Diffusers pipeline and replace its VAE with an explicit Wan 2.1 VAE."""

    if pipeline_cls is None:
        raise XQTBackendError(
            "load_wan21_pipeline_with_vae requires pipeline_cls=... to avoid baking a specific Wan pipeline variant into XQT"
        )
    vae = load_wan21_vae(
        repo_id=vae_repo_id,
        subfolder=vae_subfolder,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
    )
    pipeline = pipeline_cls.from_pretrained(
        repo_id,
        vae=vae,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        **kwargs,
    )
    if device is not None and hasattr(pipeline, "to"):
        pipeline.to(device)
    return pipeline


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


def run_wan21_vae_inference(
    model_or_pipeline: Any,
    *,
    tensor: torch.Tensor,
    run_mode: str = "decode",
    optimization_kind: str = "compile",
    compile_engine: str = "inductor",
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    compile_options: Mapping[str, Any] | None = None,
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
    warmup_iterations: int = 0,
    inplace: bool = False,
) -> dict[str, Any]:
    """Optimize one Wan 2.1 VAE path and execute it once."""

    result, summary = optimize_wan21_vae(
        model_or_pipeline,
        run_mode=run_mode,
        optimization_kind=optimization_kind,
        compile_engine=compile_engine,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_options=compile_options,
        tensor=tensor,
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
        warmup_iterations=warmup_iterations,
        inplace=inplace,
    )
    with torch.inference_mode():
        output = result.model(tensor)
    return {
        "summary": summary.to_dict(),
        "result": result.to_dict(),
        "output": output,
    }


__all__ = [
    "WAN21_VAE_OPTIMIZATION_KINDS",
    "WAN21_VAE_REPO_ID",
    "WAN21_VAE_RUN_MODES",
    "WAN21_VAE_SUBFOLDER",
    "Wan21VAECompileResult",
    "Wan21VAECudaGraphResult",
    "Wan21VAEOptimizationSummary",
    "Wan21VAEPairedBenchmarkResult",
    "benchmark_wan21_vae_paired",
    "benchmark_wan21_vae_runner",
    "build_wan21_vae_runner",
    "capture_wan21_vae_cuda_graph",
    "collect_wan21_vae_conv3d_targets",
    "collect_wan21_vae_rmsnorm_targets",
    "compile_wan21_vae_runner",
    "load_wan21_pipeline_with_vae",
    "load_wan21_vae",
    "materialize_wan21_vae_conv3d_fastpath",
    "materialize_wan21_vae_rmsnorm_fastpath",
    "optimize_wan21_vae",
    "run_wan21_vae_inference",
    "warmup_wan21_vae_runner",
]
