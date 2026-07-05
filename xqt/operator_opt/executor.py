"""Operator optimization execution helpers."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from xqt.benchmark import LatencyReport, benchmark_callable, measure_callable_ms
from xqt.core.errors import XQTBackendError
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.schema import OperatorOptimizationConfig
from xqt.core.types import XQTContext
from xqt.analysis.compare import compare_tensors
from xqt.export.input_utils import first_tensor_output, split_example_input
from xqt.quant.nvfp4_bridge import (
    NVFP4LinearBridge,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    infer_nvfp4_tensor_layout,
)
from xqt.quant.capability import describe_quant_backend_capability

from .capability import describe_operator_backend_capability
from .compile_backend import compile_with_torch
from .patterns import (
    scan_export_candidates,
    scan_fx_candidates,
    summarize_candidate_report,
)
from .backends.cutile import (
    CuTileCompileSettings,
    build_cutile_artifact_metadata,
    cutile_available,
    get_cutile_kernel_spec,
    list_cutile_kernel_specs,
    run_cutile_kernel,
)
from .backends.cutlass import (
    CutlassCompileSettings,
    build_cutlass_artifact_metadata,
    list_cutlass_kernel_specs,
)
from .backends.cute_dsl import (
    CuteDSLCompileSettings,
    build_cute_dsl_artifact_metadata,
    get_cute_dsl_kernel_spec,
    list_cute_dsl_kernel_specs,
    run_cute_dsl_kernel,
)
from .backends.tilelang import (
    TileLangCompileSettings,
    build_tilelang_artifact_metadata,
    get_tilelang_kernel_spec,
    run_tilelang_kernel,
    list_tilelang_kernel_specs,
    tilelang_validation_thresholds,
)
from .backends.triton import list_triton_kernel_specs
from .types import (
    OperatorOptimizationExecutionPlan,
    OperatorOptimizationExecutionResult,
    OperatorOptimizationReport,
    OperatorOptimizationTargetPlan,
)


def _iter_tensors(data: Any) -> Iterable[torch.Tensor]:
    if isinstance(data, torch.Tensor):
        yield data
        return
    if isinstance(data, Mapping):
        for value in data.values():
            yield from _iter_tensors(value)
        return
    if isinstance(data, (tuple, list)):
        for value in data:
            yield from _iter_tensors(value)


def _module_has_cuda_state(module: nn.Module) -> bool:
    for tensor in module.parameters():
        if tensor.is_cuda:
            return True
    for tensor in module.buffers():
        if tensor.is_cuda:
            return True
    return False


def _sync_if_needed(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_report_from_samples(
    *,
    samples_ms: list[float],
    warmup: int,
    iterations: int,
) -> LatencyReport:
    sorted_samples = sorted(samples_ms)
    mean_ms = sum(samples_ms) / len(samples_ms)
    return LatencyReport(
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean_ms,
        p50_ms=_percentile(sorted_samples, 50),
        p90_ms=_percentile(sorted_samples, 90),
        p99_ms=_percentile(sorted_samples, 99),
        samples_ms=samples_ms,
    )


def _benchmark_paired_callables(
    reference_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
) -> tuple[LatencyReport, LatencyReport, list[float]]:
    torch_device = torch.device(device) if device is not None else None
    should_sync_cuda = sync_cuda and (torch_device is None or torch_device.type == "cuda")
    reference_samples_ms: list[float] = []
    candidate_samples_ms: list[float] = []
    paired_speedup_ratios: list[float] = []

    with torch.no_grad():
        for index in range(warmup):
            if index % 2 == 0:
                reference_fn()
                candidate_fn()
            else:
                candidate_fn()
                reference_fn()
        _sync_if_needed(should_sync_cuda)

        for index in range(iterations):
            first_fn = reference_fn if index % 2 == 0 else candidate_fn
            second_fn = candidate_fn if index % 2 == 0 else reference_fn
            first_ms = measure_callable_ms(
                first_fn,
                sync_cuda=should_sync_cuda,
                device=device,
            )
            second_ms = measure_callable_ms(
                second_fn,
                sync_cuda=should_sync_cuda,
                device=device,
            )
            if index % 2 == 0:
                reference_samples_ms.append(first_ms)
                candidate_samples_ms.append(second_ms)
                if second_ms > 0.0:
                    paired_speedup_ratios.append(first_ms / second_ms)
            else:
                candidate_samples_ms.append(first_ms)
                reference_samples_ms.append(second_ms)
                if first_ms > 0.0:
                    paired_speedup_ratios.append(second_ms / first_ms)

    return (
        _latency_report_from_samples(
            samples_ms=reference_samples_ms,
            warmup=warmup,
            iterations=iterations,
        ),
        _latency_report_from_samples(
            samples_ms=candidate_samples_ms,
            warmup=warmup,
            iterations=iterations,
        ),
        paired_speedup_ratios,
    )


def _benchmark_paired_batched_callables(
    reference_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
    inner_iterations: int,
) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    reference_report, candidate_report, paired_speedup_ratios = _benchmark_paired_callables(
        _repeat_callable(reference_fn, inner_iterations),
        _repeat_callable(candidate_fn, inner_iterations),
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    )
    return (
        _per_call_latency(reference_report.to_dict(), inner_iterations),
        _per_call_latency(candidate_report.to_dict(), inner_iterations),
        paired_speedup_ratios,
    )


def _repeat_callable(fn: Callable[[], object], count: int) -> Callable[[], object]:
    if count <= 0:
        raise ValueError("inner_iterations must be positive")

    def repeated() -> object:
        output = fn()
        for _ in range(count - 1):
            output = fn()
        return output

    return repeated


def _per_call_latency(report: dict[str, Any], inner_iterations: int) -> dict[str, Any]:
    scaled = dict(report)
    for key in ("mean_ms", "p50_ms", "p90_ms", "p99_ms"):
        scaled[key] = float(report[key]) / float(inner_iterations)
    scaled["samples_ms"] = [
        float(sample_ms) / float(inner_iterations)
        for sample_ms in report["samples_ms"]
    ]
    scaled["inner_iterations"] = inner_iterations
    return scaled


def _benchmark_batched_callable(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
    inner_iterations: int,
) -> dict[str, Any]:
    batched_fn = _repeat_callable(fn, inner_iterations)
    report = benchmark_callable(
        batched_fn,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=sync_cuda,
        device=device,
    ).to_dict()
    return _per_call_latency(report, inner_iterations)


def _tilelang_inner_iterations(
    execution_detail: Mapping[str, Any],
) -> int:
    kernel_kind = execution_detail.get("kernel_kind")
    operator_family = execution_detail.get("operator_family")
    if kernel_kind not in {"minimal_cuda_jit", "cuda_graph_replay"}:
        return 1
    if operator_family not in {"attention", "norm", "linear", "conv"}:
        return 1
    return 100


def _benchmark_callable_for_execution(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    sync_cuda: bool,
    device: str | None,
    execution_detail: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    inner_iterations = _tilelang_inner_iterations(execution_detail)
    if inner_iterations <= 1:
        return (
            benchmark_callable(
                fn,
                warmup=warmup,
                iterations=iterations,
                sync_cuda=sync_cuda,
                device=device,
            ).to_dict(),
            "single_callable_mean",
        )
    return (
        _benchmark_batched_callable(
            fn,
            warmup=warmup,
            iterations=iterations,
            sync_cuda=sync_cuda,
            device=device,
            inner_iterations=inner_iterations,
        ),
        "steady_state_batched_mean",
    )


def _native_runtime_speedup_strategy(
    execution_detail: Mapping[str, Any],
) -> str | None:
    if execution_detail.get("kernel_kind") != "native_runtime_fastpath":
        return None
    return "paired_alternating_p50"


def _paired_steady_state_speedup_strategy(
    execution_detail: Mapping[str, Any],
) -> str | None:
    inner_iterations = _tilelang_inner_iterations(execution_detail)
    if inner_iterations <= 1:
        return None
    return "paired_steady_state_batched_mean"


def _effective_min_speedup(
    target: OperatorOptimizationTargetPlan,
    *,
    execution_detail: Mapping[str, Any],
) -> float:
    if (
        execution_detail.get("kernel_kind") == "native_runtime_fastpath"
        and float(target.min_speedup) <= 1.000001
    ):
        return 0.99
    return float(target.min_speedup)


def _native_runtime_near_equal(
    execution_detail: Mapping[str, Any],
    *,
    latency_before: Mapping[str, Any],
    latency_after: Mapping[str, Any],
) -> bool:
    if execution_detail.get("kernel_kind") != "native_runtime_fastpath":
        return False
    p50_before = latency_before.get("p50_ms")
    p50_after = latency_after.get("p50_ms")
    if not isinstance(p50_before, (float, int)) or not isinstance(p50_after, (float, int)):
        return False
    before_ms = float(p50_before)
    after_ms = float(p50_after)
    if before_ms <= 0.0 or after_ms <= 0.0:
        return False
    max_ms = max(before_ms, after_ms)
    if max_ms > 0.25:
        return False
    absolute_gap_ms = abs(before_ms - after_ms)
    if absolute_gap_ms <= 0.005:
        return True
    if max_ms <= 0.1 and absolute_gap_ms <= 0.007:
        return True
    relative_gap = absolute_gap_ms / max(before_ms, after_ms)
    return relative_gap <= 0.03


def _resolve_component_model(
    model: nn.Module,
    target_path: Optional[str],
) -> nn.Module:
    if not target_path:
        return model
    return model.get_submodule(target_path)


def _replace_component_model(
    model: nn.Module,
    target_path: Optional[str],
    replacement: nn.Module,
) -> nn.Module:
    if not target_path:
        return replacement
    parent_path, _, attribute = target_path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)
    return model


def _call_module(module: nn.Module, inputs: Any) -> Any:
    normalized = split_example_input(inputs)
    return module(*normalized.args, **normalized.kwargs)


def _call_module_no_grad(module: nn.Module, inputs: Any) -> Any:
    with torch.no_grad():
        return _call_module(module, inputs)


def _move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _shape_signature(inputs: Any) -> dict[str, Any]:
    normalized = split_example_input(inputs)
    values = list(normalized.args) + list(normalized.kwargs.values())
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    return {
        "structure": (
            "mapping"
            if normalized.kwargs
            else "tuple"
            if len(normalized.args) > 1
            else "tensor"
        ),
        "input_count": len(values),
        "tensor_shapes": [list(tensor.shape) for tensor in tensors],
        "tensor_dtypes": [str(tensor.dtype) for tensor in tensors],
        "tensor_devices": [str(tensor.device) for tensor in tensors],
    }


def _infer_module_device_dtype(
    module: nn.Module,
    inputs: Any,
) -> tuple[str | None, str | None]:
    device: str | None = None
    dtype: str | None = None
    for tensor in module.parameters():
        device = str(tensor.device)
        if tensor.is_floating_point():
            dtype = str(tensor.dtype)
            return device, dtype
        if dtype is None:
            dtype = str(tensor.dtype)
    for tensor in module.buffers():
        if device is None:
            device = str(tensor.device)
        if tensor.is_floating_point():
            dtype = str(tensor.dtype)
            return device, dtype
        if dtype is None:
            dtype = str(tensor.dtype)
    for tensor in _iter_tensors(inputs):
        if device is None:
            device = str(tensor.device)
        if tensor.is_floating_point():
            dtype = str(tensor.dtype)
            return device, dtype
        if dtype is None:
            dtype = str(tensor.dtype)
    return device, dtype


def _effective_validation_thresholds(
    target: OperatorOptimizationTargetPlan,
    *,
    baseline_output: torch.Tensor,
    optimized_output: torch.Tensor,
) -> dict[str, float]:
    thresholds = {
        "atol": float(target.validate.get("atol", 1e-5)),
        "rtol": float(target.validate.get("rtol", 1e-5)),
    }
    if target.backend != "tilelang":
        return thresholds
    patterns = list(getattr(target, "patterns", []) or [])
    if patterns == ["attention"]:
        thresholds["atol"] = max(thresholds["atol"], 1e-2)
        thresholds["rtol"] = max(thresholds["rtol"], 1e-2)
    dtype_defaults = tilelang_validation_thresholds(
        optimized_output.dtype if isinstance(optimized_output, torch.Tensor) else baseline_output.dtype
    )
    return {
        "atol": max(float(dtype_defaults["atol"]), thresholds["atol"]),
        "rtol": max(float(dtype_defaults["rtol"]), thresholds["rtol"]),
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _attach_tilelang_execution_metadata(
    module: nn.Module,
    metadata: Mapping[str, Any],
) -> nn.Module:
    setattr(module, "_xqt_tilelang_execution_metadata", dict(metadata))
    return module


_DEFAULT_TILELANG_CUDA_GRAPH_WARMUP = 2


def _cuda_graph_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _make_static_cuda_graph_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        size=tuple(int(dim) for dim in tensor.shape),
        stride=tuple(int(stride) for stride in tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _capture_cuda_graph_tensor_callable(
    sample_args: tuple[torch.Tensor, ...],
    *,
    body: Callable[..., torch.Tensor],
    warmup: int,
) -> dict[str, Any]:
    if not sample_args:
        raise XQTBackendError("CUDA Graph capture requires at least one sample tensor")
    if not all(tensor.is_cuda for tensor in sample_args):
        raise XQTBackendError("CUDA Graph capture requires CUDA tensor inputs")
    static_args = tuple(_make_static_cuda_graph_tensor(tensor) for tensor in sample_args)
    for static_arg, sample_arg in zip(static_args, sample_args):
        static_arg.copy_(sample_arg)
    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            body(*static_args)
        torch.cuda.synchronize(sample_args[0].device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = body(*static_args)
    return {
        "graph": graph,
        "static_args": static_args,
        "static_output": static_output,
    }


def _capture_cuda_graph_with_static_state(
    dynamic_args: tuple[torch.Tensor, ...],
    *,
    body: Callable[..., torch.Tensor],
    warmup: int,
) -> dict[str, Any]:
    if not dynamic_args:
        raise XQTBackendError("CUDA Graph capture requires at least one dynamic tensor")
    if not all(tensor.is_cuda for tensor in dynamic_args):
        raise XQTBackendError("CUDA Graph capture requires CUDA tensor inputs")
    static_dynamic_args = tuple(_make_static_cuda_graph_tensor(tensor) for tensor in dynamic_args)
    for static_arg, runtime_arg in zip(static_dynamic_args, dynamic_args):
        static_arg.copy_(runtime_arg)
    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            body(*static_dynamic_args)
        torch.cuda.synchronize(dynamic_args[0].device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = body(*static_dynamic_args)
    return {
        "graph": graph,
        "static_args": static_dynamic_args,
        "static_output": static_output,
    }


def _replay_cuda_graph_tensor_callable(
    state: Mapping[str, Any],
    runtime_args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    static_args = state.get("static_args")
    graph = state.get("graph")
    static_output = state.get("static_output")
    if (
        not isinstance(static_args, tuple)
        or graph is None
        or not isinstance(static_output, torch.Tensor)
        or len(static_args) != len(runtime_args)
    ):
        raise XQTBackendError("invalid CUDA Graph state")
    for static_arg, runtime_arg in zip(static_args, runtime_args):
        if not isinstance(static_arg, torch.Tensor) or not isinstance(runtime_arg, torch.Tensor):
            raise XQTBackendError("CUDA Graph state contains non-tensor inputs")
        static_arg.copy_(runtime_arg)
    graph.replay()
    return static_output


def _infer_module_runtime_spec(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    saw_only_float32 = False
    for tensor in module.parameters():
        if device is None:
            device = tensor.device
        if tensor.is_floating_point():
            dtype = tensor.dtype
            if tensor.dtype != torch.float32:
                return device, dtype
            saw_only_float32 = True
    if dtype is None:
        for tensor in module.buffers():
            if device is None:
                device = tensor.device
            if tensor.is_floating_point():
                dtype = tensor.dtype
                if tensor.dtype != torch.float32:
                    return device, dtype
                saw_only_float32 = True
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    elif saw_only_float32:
        dtype = torch.float16
    return device, dtype


def _make_artifact_key(prefix: str, target_name: str) -> str:
    if target_name == "model":
        return prefix
    return f"{prefix}_{target_name}"


def _quant_runtime_guard(
    context: XQTContext,
    plan: OperatorOptimizationTargetPlan,
) -> Optional[str]:
    quant_metrics = context.metrics.get("quant")
    if not isinstance(quant_metrics, dict):
        return None
    backend = str(quant_metrics.get("backend") or "")
    runtime = str(quant_metrics.get("components", [{}])[0].get("runtime") or quant_metrics.get("runtime") or "")
    if backend == "onnxruntime_qdq" or runtime == "onnxruntime":
        return "quantized runtime artifact is onnxruntime_qdq and cannot be rewritten as a PyTorch custom kernel"
    if backend:
        capability = describe_quant_backend_capability(backend)
        if capability.runtime != "pytorch":
            return (
                f"quantized runtime '{capability.runtime}' is not a PyTorch module runtime "
                "for operator optimization replacement"
            )
    del plan
    return None


def _backend_metadata(
    target: OperatorOptimizationTargetPlan,
    *,
    dtype: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if target.backend == "deployment_backend":
        metadata["deployment_target"] = {
            "runtime": target.options.get("runtime"),
            "stage": target.options.get("stage"),
            "semantic_target": target.options.get("semantic_target") or target.name,
            "fallback_reason": target.options.get("fallback_reason"),
        }
    if target.backend == "triton":
        metadata["kernel_registry"] = list_triton_kernel_specs()
    if target.backend == "tilelang":
        settings = TileLangCompileSettings(
            target=str(target.tilelang.get("target", "cuda")),
            target_arch=target.tilelang.get("target_arch"),
            cache_dir=target.tilelang.get("cache_dir"),
            threads=int(target.tilelang.get("threads", 128)),
            num_stages=int(target.tilelang.get("num_stages", 2)),
            pass_configs=dict(target.tilelang.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_tilelang_kernel_specs()
        selected_patterns = target.patterns or ["attention"]
        metadata["tilelang_artifacts"] = {
            pattern: build_tilelang_artifact_metadata(pattern, settings)
            for pattern in selected_patterns
            if pattern in metadata["kernel_registry"]
        }
        effective_thresholds = tilelang_validation_thresholds(dtype)
        effective_thresholds.update(dict(target.validate))
        metadata["validation_thresholds"] = effective_thresholds
        metadata["execution_mode"] = "not_run"
        metadata["execution_reason"] = None
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.backend == "cutile":
        settings = CuTileCompileSettings(
            target=str(target.cutile.get("target", "cuda")),
            target_arch=target.cutile.get("target_arch"),
            cache_dir=target.cutile.get("cache_dir"),
            threads=int(target.cutile.get("threads", 128)),
            pass_configs=dict(target.cutile.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_cutile_kernel_specs()
        selected_patterns = target.patterns or ["bias_silu"]
        metadata["cutile_artifacts"] = {
            pattern: build_cutile_artifact_metadata(pattern, settings)
            for pattern in selected_patterns
            if pattern in metadata["kernel_registry"]
        }
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.backend == "cutlass":
        tile_shape_raw = target.cutlass.get("tile_shape", [128, 128, 64])
        cluster_shape_raw = target.cutlass.get("cluster_shape")
        settings = CutlassCompileSettings(
            target_arch=target.cutlass.get("target_arch"),
            cache_dir=target.cutlass.get("cache_dir"),
            tile_shape=tuple(int(value) for value in tile_shape_raw),
            cluster_shape=(
                tuple(int(value) for value in cluster_shape_raw)
                if cluster_shape_raw is not None
                else None
            ),
            pass_configs=dict(target.cutlass.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_cutlass_kernel_specs()
        selected_patterns = target.patterns or ["gemm_epilogue"]
        metadata["cutlass_artifacts"] = {
            pattern: build_cutlass_artifact_metadata(pattern, settings)
            for pattern in selected_patterns
            if pattern in metadata["kernel_registry"]
        }
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    if target.backend == "cute_dsl":
        tile_shape_raw = target.cute_dsl.get("tile_shape", [128, 128, 64])
        cluster_shape_raw = target.cute_dsl.get("cluster_shape")
        settings = CuteDSLCompileSettings(
            target_arch=target.cute_dsl.get("target_arch"),
            cache_dir=target.cute_dsl.get("cache_dir"),
            tile_shape=tuple(int(value) for value in tile_shape_raw),
            cluster_shape=(
                tuple(int(value) for value in cluster_shape_raw)
                if cluster_shape_raw is not None
                else None
            ),
            pass_configs=dict(target.cute_dsl.get("pass_configs", {})),
        )
        metadata["kernel_registry"] = list_cute_dsl_kernel_specs()
        selected_patterns = target.patterns or ["gemm_epilogue"]
        metadata["cute_dsl_artifacts"] = {
            pattern: build_cute_dsl_artifact_metadata(pattern, settings)
            for pattern in selected_patterns
            if pattern in metadata["kernel_registry"]
        }
        metadata["latency"] = {
            "compile_latency_ms": None,
            "execution_latency_ms": None,
            "status": "not_executed",
        }
    return metadata


def _artifact_paths_from_backend_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    for backend in ("tilelang", "cutile", "cutlass", "cute_dsl"):
        backend_artifacts = metadata.get(f"{backend}_artifacts")
        if not isinstance(backend_artifacts, Mapping):
            continue
        for pattern, artifact_metadata in backend_artifacts.items():
            if not isinstance(artifact_metadata, Mapping):
                continue
            artifact_path = artifact_metadata.get("artifact_path")
            if artifact_path is None:
                continue
            artifact_paths[f"{backend}.{pattern}"] = str(artifact_path)
    return artifact_paths


class _TileLangAttentionWrapper(nn.Module):
    """Minimal executable wrapper for attention-pattern TileLang targets."""

    def __init__(
        self,
        attention: nn.MultiheadAttention,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "attention"
        self.last_fastpath = "none"
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _resolved_target_arch(self, q: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if q.is_cuda:
            major, minor = torch.cuda.get_device_capability(q.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_attention_fastpath(self, q: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return self._resolved_target_arch(q) == "sm_89"

    def _prefer_graph_attention_fastpath(self, q: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode in {"graph", "tilelang_graph"}:
            return True
        return False

    def _canonicalize_attention_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_input = query if self.attention.batch_first else query.transpose(0, 1)
        source_key = query if key is None else key
        if source_key is query:
            k_input = q_input
        else:
            k_input = source_key if self.attention.batch_first else source_key.transpose(0, 1)
        source_value = source_key if value is None else value
        if source_value is source_key:
            v_input = k_input
        elif source_value is query:
            v_input = q_input
        else:
            v_input = source_value if self.attention.batch_first else source_value.transpose(0, 1)
        return q_input, k_input, v_input

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not getattr(self.attention, "_qkv_same_embed_dim", True):
            raise XQTBackendError(
                "TileLang attention wrapper currently supports only qkv_same_embed_dim=True"
            )
        if self.attention.in_proj_weight is None:
            raise XQTBackendError("TileLang attention wrapper requires packed in_proj_weight")
        embed_dim = int(self.attention.embed_dim)
        q_proj = F.linear(
            query,
            self.attention.in_proj_weight[:embed_dim],
            None if self.attention.in_proj_bias is None else self.attention.in_proj_bias[:embed_dim],
        )
        k_proj = F.linear(
            key,
            self.attention.in_proj_weight[embed_dim : 2 * embed_dim],
            None
            if self.attention.in_proj_bias is None
            else self.attention.in_proj_bias[embed_dim : 2 * embed_dim],
        )
        v_proj = F.linear(
            value,
            self.attention.in_proj_weight[2 * embed_dim :],
            None
            if self.attention.in_proj_bias is None
            else self.attention.in_proj_bias[2 * embed_dim :],
        )
        return q_proj, k_proj, v_proj

    def _reshape_for_tilelang(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = tensor.shape
        num_heads = int(self.attention.num_heads)
        head_dim = embed_dim // num_heads
        return tensor.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _merge_from_tilelang(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim).contiguous()

    def _project_qkv_for_tilelang(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_proj, k_proj, v_proj = self._project_qkv(query, key, value)
        return (
            self._reshape_for_tilelang(q_proj),
            self._reshape_for_tilelang(k_proj),
            self._reshape_for_tilelang(v_proj),
        )

    def _finalize_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        merged = self._merge_from_tilelang(attn_output)
        return self.attention.out_proj(merged)

    @staticmethod
    def _attention_graph_runtime_args(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[tuple[int, int, int], tuple[torch.Tensor, ...]]:
        unique_args: list[torch.Tensor] = []
        arg_mapping: list[int] = []
        tensor_index: dict[int, int] = {}
        for tensor in (query, key, value):
            index = tensor_index.get(id(tensor))
            if index is None:
                index = len(unique_args)
                unique_args.append(tensor)
                tensor_index[id(tensor)] = index
            arg_mapping.append(index)
        return (arg_mapping[0], arg_mapping[1], arg_mapping[2]), tuple(unique_args)

    @staticmethod
    def _resolve_attention_graph_args(
        dynamic_args: tuple[torch.Tensor, ...],
        arg_mapping: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            dynamic_args[arg_mapping[0]],
            dynamic_args[arg_mapping[1]],
            dynamic_args[arg_mapping[2]],
        )

    def _attention_graph_cache_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> tuple[Any, ...]:
        arg_mapping, runtime_args = self._attention_graph_runtime_args(query, key, value)
        return (
            tuple(_cuda_graph_tensor_signature(tensor) for tensor in runtime_args),
            arg_mapping,
            bool(is_causal),
            float(self.attention.dropout),
            int(self.settings.get("block_m", 64)),
            int(self.settings.get("block_n", 64)),
            int(self.settings.get("threads", 128)),
            int(self.settings.get("num_stages", 2)),
            str(self.settings.get("target_arch") or ""),
        )

    def _run_tilelang_attention_body(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        return run_tilelang_kernel(
            "attention",
            q,
            k,
            v,
            causal=is_causal,
            dropout_p=float(self.attention.dropout),
            block_m=int(self.settings.get("block_m", 64)),
            block_n=int(self.settings.get("block_n", 64)),
            threads=int(self.settings.get("threads", 128)),
            num_stages=int(self.settings.get("num_stages", 2)),
            fallback=self.fallback,
        )

    def _run_tilelang_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv_for_tilelang(query, key, value)
        return self._finalize_attention_output(
            self._run_tilelang_attention_body(q, k, v, is_causal=is_causal)
        )

    def _run_native_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv_for_tilelang(query, key, value)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=float(self.attention.dropout),
            is_causal=is_causal,
        )
        return self._finalize_attention_output(attn_output)

    def _graph_capture_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> dict[str, Any]:
        arg_mapping, runtime_args = self._attention_graph_runtime_args(query, key, value)
        state = _capture_cuda_graph_with_static_state(
            runtime_args,
            body=lambda *dynamic_args: self._run_tilelang_attention_forward(
                *self._resolve_attention_graph_args(
                    tuple(dynamic_args),
                    arg_mapping,
                ),
                is_causal=is_causal,
            ),
            warmup=int(self.settings.get("cuda_graph_warmup", _DEFAULT_TILELANG_CUDA_GRAPH_WARMUP)),
        )
        state["kind"] = "tilelang_attention_full_forward"
        state["arg_mapping"] = arg_mapping
        return state

    def _run_attention_with_optional_graph(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if not self._prefer_graph_attention_fastpath(query):
            self.last_graph_state = "disabled"
            self.last_graph_reason = "attention_fastpath is not set to graph mode"
            return self._run_tilelang_attention_forward(query, key, value, is_causal=is_causal)
        _, runtime_args = self._attention_graph_runtime_args(query, key, value)
        cache_key = self._attention_graph_cache_key(query, key, value, is_causal=is_causal)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = self._graph_capture_attention(query, key, value, is_causal=is_causal)
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._run_tilelang_attention_forward(query, key, value, is_causal=is_causal)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return _replay_cuda_graph_tensor_callable(state, runtime_args)
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return _replay_cuda_graph_tensor_callable(state, runtime_args)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del average_attn_weights
        if attn_mask is not None or key_padding_mask is not None:
            raise XQTBackendError(
                "TileLang attention wrapper does not yet support attn_mask or key_padding_mask"
            )
        q_input, k_input, v_input = self._canonicalize_attention_inputs(query, key, value)
        input_is_cuda = q_input.is_cuda and k_input.is_cuda and v_input.is_cuda
        use_graph_tilelang = (
            input_is_cuda and self._prefer_graph_attention_fastpath(q_input)
        )
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if input_is_cuda and self._prefer_native_attention_fastpath(q_input)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if input_is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_sdpa"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_attention_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_attention_kernel"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {
                "cuda_tilelang_entry",
                "cuda_graph_tilelang_entry",
                "cuda_native_fastpath",
            }
            else "TileLang attention kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode != "cuda_graph_tilelang_entry":
            self.last_graph_state = "disabled"
            self.last_graph_reason = (
                None
                if self.last_execution_mode == "cuda_native_fastpath"
                else "graph fastpath was not selected"
            )
        if self.last_execution_mode == "cuda_native_fastpath":
            projected = self._run_native_attention_forward(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        elif self.last_execution_mode == "cuda_graph_tilelang_entry":
            projected = self._run_attention_with_optional_graph(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        else:
            projected = self._run_tilelang_attention_forward(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        output = projected if self.attention.batch_first else projected.transpose(0, 1)
        weights = None
        if need_weights:
            batch_size = int(q_input.shape[0]) if q_input.ndim == 3 else 0
            target_len = int(q_input.shape[1]) if q_input.ndim == 3 else 0
            source_len = int(k_input.shape[1]) if k_input.ndim == 3 else 0
            weights = output.new_zeros((batch_size, target_len, source_len))
        return output, weights

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "cuda_graph_replay"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "dropout_p": 0.0,
                "requires_seq_kv_gte_seq_q": True,
                "supported_patterns": ["attention"],
                "operator_families": ["attention"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
            },
        }


class _TileLangConvWrapper(nn.Module):
    """Conv2d operator-family wrapper with architecture-aware native runtime routing."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = conv
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "conv"
        self.last_fastpath = "none"

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_conv_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("conv_fastpath", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                "conv",
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                block_m=int(self.settings.get("block_m", 64)),
                block_n=int(self.settings.get("block_n", 64)),
                block_k=int(self.settings.get("block_k", 64)),
                threads=int(self.settings.get("threads", 128)),
                num_stages=int(self.settings.get("num_stages", 2)),
                target_arch=self.settings.get("target_arch"),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang conv runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_conv_fastpath(x)
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_cudnn_conv2d"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_half_conv2d_im2col_gemm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang conv fastpath requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_native_fastpath":
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._run_tilelang_or_reference(x)
        return F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["conv"],
                "operator_families": ["conv"],
                "supports_grouped_conv": False,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangLinearWrapper(nn.Module):
    """Standalone half Linear wrapper for direct TileLang operator targets."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.linear = linear
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "linear"
        self.last_fastpath = "none"

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_runtime", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def _selected_linear_pattern(self) -> str:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list) and "linear_marlin" in patterns:
            return "linear_marlin"
        return "linear"

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("TileLang linear target requires at least 1D input")
        if x.shape[-1] <= 0:
            raise XQTBackendError("TileLang linear target requires a non-empty trailing feature dimension")
        return x.reshape(-1, int(x.shape[-1])), tuple(int(dim) for dim in x.shape[:-1])

    @staticmethod
    def _restore_output(output: torch.Tensor, prefix_shape: tuple[int, ...]) -> torch.Tensor:
        return output.reshape(*prefix_shape, int(output.shape[-1]))

    def _run_tilelang_or_reference(
        self,
        x: torch.Tensor,
        flat_input: torch.Tensor,
    ) -> torch.Tensor:
        pattern = self._selected_linear_pattern()
        try:
            if pattern == "linear_marlin":
                return run_tilelang_kernel(
                    pattern,
                    flat_input,
                    self.linear.weight,
                    bias=self.linear.bias,
                    precision=str(self.settings.get("precision", "auto")),
                    block_m=int(self.settings.get("block_m", 64)),
                    block_n=int(self.settings.get("block_n", 64)),
                    block_k=int(self.settings.get("block_k", 64)),
                    threads=int(self.settings.get("threads", 128)),
                    num_stages=int(self.settings.get("num_stages", 2)),
                    target_arch=self.settings.get("target_arch"),
                    fallback=self.fallback,
                )
            return run_tilelang_kernel(
                pattern,
                flat_input,
                self.linear.weight,
                self.linear.bias,
                block_m=int(self.settings.get("block_m", 64)),
                block_n=int(self.settings.get("block_n", 64)),
                block_k=int(self.settings.get("block_k", 64)),
                threads=int(self.settings.get("threads", 128)),
                num_stages=int(self.settings.get("num_stages", 2)),
                target_arch=self.settings.get("target_arch"),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang linear runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self.linear(x).reshape(-1, int(self.linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix_shape = tuple(int(dim) for dim in x.shape[:-1])
        flat_input, _ = self._flatten_input(x)
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_linear_fastpath(x)
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_torch_linear"
            if self.last_execution_mode == "cuda_native_fastpath"
            else (
                "tilelang_marlin_linear_kernel"
                if self._selected_linear_pattern() == "linear_marlin"
                else "tilelang_half_linear_kernel"
            )
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang linear kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
            return self.linear(x)
        output = self._run_tilelang_or_reference(x, flat_input)
        return self._restore_output(output, prefix_shape)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_precisions": ["fp16", "bf16", "int8", "int4"],
                "supported_patterns": ["linear", "linear_marlin"],
                "operator_families": ["linear"],
                "supports_rank_gte_1_via_batch_flatten": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangNormWrapper(nn.Module):
    """Standalone half LayerNorm wrapper for direct TileLang operator targets."""

    def __init__(
        self,
        norm: nn.LayerNorm,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.norm = norm
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "norm"
        self.last_fastpath = "none"
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_norm_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("norm_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def _prefer_graph_norm_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("norm_fastpath", "auto"))
        if mode in {"graph", "tilelang_graph"}:
            return True
        return False

    def _norm_weight_bias(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized_dim = int(self.norm.normalized_shape[-1])
        weight = self.norm.weight
        bias = self.norm.bias
        if weight is None:
            weight = torch.ones(normalized_dim, device=x.device, dtype=x.dtype)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return weight.to(device=x.device, dtype=x.dtype), bias

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self._norm_weight_bias(x)
        try:
            return run_tilelang_kernel(
                "norm",
                x,
                weight,
                bias,
                eps=float(self.norm.eps),
                threads=int(self.settings.get("threads", 64)),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang norm runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self.norm(x)

    def _norm_graph_cache_key(
        self,
        x: torch.Tensor,
    ) -> tuple[Any, ...]:
        normalized_dim = int(self.norm.normalized_shape[-1])
        return (
            _cuda_graph_tensor_signature(x),
            normalized_dim,
            float(self.norm.eps),
            int(self.settings.get("threads", 64)),
        )

    def _run_norm_with_optional_graph(self, x: torch.Tensor) -> torch.Tensor:
        if not self._prefer_graph_norm_fastpath(x):
            self.last_graph_state = "disabled"
            self.last_graph_reason = "norm_fastpath is not set to graph mode"
            return self._run_tilelang_or_reference(x)
        weight, bias = self._norm_weight_bias(x)
        graph_bias = bias if bias is not None else torch.zeros_like(weight)
        cache_key = self._norm_graph_cache_key(x)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = _capture_cuda_graph_with_static_state(
                    (x,),
                    body=lambda x_arg: run_tilelang_kernel(
                        "norm",
                        x_arg,
                        weight,
                        graph_bias,
                        eps=float(self.norm.eps),
                        threads=int(self.settings.get("threads", 64)),
                        fallback=self.fallback,
                    ),
                    warmup=int(
                        self.settings.get("cuda_graph_warmup", _DEFAULT_TILELANG_CUDA_GRAPH_WARMUP)
                    ),
                )
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._run_tilelang_or_reference(x)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return _replay_cuda_graph_tensor_callable(state, (x,))
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return _replay_cuda_graph_tensor_callable(state, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_graph_tilelang = x.is_cuda and self._prefer_graph_norm_fastpath(x)
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_norm_fastpath(x)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_torch_layer_norm"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_half_layer_norm_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_half_layer_norm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {
                "cuda_native_fastpath",
                "cuda_graph_tilelang_entry",
                "cuda_tilelang_entry",
            }
            else "TileLang norm kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode != "cuda_graph_tilelang_entry":
            self.last_graph_state = "disabled"
            self.last_graph_reason = (
                None
                if self.last_execution_mode == "cuda_native_fastpath"
                else "graph fastpath was not selected"
            )
        if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
            return self.norm(x)
        if self.last_execution_mode == "cuda_graph_tilelang_entry":
            return self._run_norm_with_optional_graph(x)
        return self._run_tilelang_or_reference(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "cuda_graph_replay"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["norm"],
                "operator_families": ["norm"],
                "normalized_last_dim_only": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
            },
        }


class _TileLangEagerDenseLinearModule(nn.Module):
    """One-time dequantized dense Linear replacement for sm_89 native runtime parity."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.linear = linear
        self._execution_metadata = dict(metadata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self._execution_metadata)


class _TileLangDequantGemmWrapper(nn.Module):
    """Minimal executable wrapper for dequant GEMM TileLang targets."""

    def __init__(
        self,
        module: nn.Module,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.module = module
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_weight_source = "not_run"
        self.last_weight_representation = "unknown"
        self.last_consumes_packed_weight = False
        self.last_unpack_stage: str | None = None
        self.last_kernel_pattern = "dequant_gemm_epilogue"
        self.last_operator_family = "linear"
        self.last_fastpath = "none"
        self._cached_nvfp4_bridge: NVFP4LinearBridge | None = None
        self._preferred_patterns_config = self._preferred_patterns()
        self._dense_linear_bridge = getattr(self.module, "tilelang_dense_linear_args", None)
        self._packed_nvfp4_bridge = getattr(self.module, "tilelang_packed_nvfp4_dequant_gemm_args", None)
        self._packed_fp4_bridge = getattr(self.module, "tilelang_packed_dequant_gemm_args", None)
        self._dense_fp4_bridge = getattr(self.module, "tilelang_dequant_gemm_args", None)
        if self._packed_nvfp4_bridge is None and self._dense_fp4_bridge is None and self._packed_fp4_bridge is None:
            inferred_bridge = bridge_module_to_nvfp4_linear_shared(self.module)
            if inferred_bridge is not None:
                self._cached_nvfp4_bridge = inferred_bridge

    def _can_delegate_dense_native_linear(self, activation: str | None) -> bool:
        return (
            activation is None
            and hasattr(self.module, "dense_weight")
            and callable(getattr(self.module, "forward", None))
        )

    def _resolve_dense_linear_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, str | None] | None:
        if callable(self._dense_linear_bridge):
            if callable(self._packed_fp4_bridge) or callable(self._dense_fp4_bridge):
                self.last_weight_source = "reference_fp4_linear_dense_cache_bridge"
                self.last_weight_representation = "dense_dequantized_fp4_weight_cache"
            else:
                self.last_weight_source = "nvfp4_dense_cache_bridge"
                self.last_weight_representation = "dense_dequantized_weight_cache"
            return self._dense_linear_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_dense_cache_bridge"
        self.last_weight_representation = "dense_dequantized_weight_cache"
        return bridge.tilelang_dense_linear_args(dtype=x.dtype, device=x.device)

    def _resolve_packed_nvfp4_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None] | None:
        if callable(self._packed_nvfp4_bridge):
            self.last_weight_source = "compressed_tensors_nvfp4_packed_bridge"
            self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
            return self._packed_nvfp4_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_packed_bridge"
        self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
        return bridge.tilelang_packed_nvfp4_dequant_gemm_args(dtype=x.dtype, device=x.device)

    @staticmethod
    def _apply_activation(output: torch.Tensor, activation: str | None) -> torch.Tensor:
        if activation is None:
            return output
        if activation == "gelu":
            return F.gelu(output)
        if activation == "silu":
            return F.silu(output)
        if activation == "relu":
            return F.relu(output)
        raise XQTBackendError(f"unsupported activation: {activation}")

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _preferred_patterns(self) -> list[str]:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list):
            return [str(pattern) for pattern in patterns]
        return ["dequant_gemm_epilogue"]

    def _prefer_dense_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_fastpath", "auto"))
        if mode == "dense":
            return True
        if mode == "packed":
            return False
        target_arch = self._resolved_target_arch(x)
        return target_arch == "sm_89"

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_runtime", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        target_arch = self._resolved_target_arch(x)
        return target_arch == "sm_89"

    def _resolved_nvfp4_bridge(self) -> NVFP4LinearBridge | None:
        if callable(self._packed_fp4_bridge) or callable(self._dense_fp4_bridge):
            return None
        if self._cached_nvfp4_bridge is not None:
            return self._cached_nvfp4_bridge
        existing_bridge = getattr(self.module, "_bridge", None)
        if isinstance(existing_bridge, NVFP4LinearBridge):
            self._cached_nvfp4_bridge = existing_bridge
            return existing_bridge
        inferred_bridge = bridge_module_to_nvfp4_linear(self.module)
        if inferred_bridge is not None:
            self._cached_nvfp4_bridge = inferred_bridge
        return inferred_bridge

    def _run_tilelang_or_reference(
        self,
        kernel_pattern: str,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                kernel_pattern,
                *args,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            spec = get_tilelang_kernel_spec(kernel_pattern)
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in allowed
            }
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"TileLang {kernel_pattern} runtime fallback: {exc}"
            )
            self.last_fastpath = "eager_reference_fallback"
            self.last_unpack_stage = (
                "one_time_eager_dequant_cache"
                if kernel_pattern == "dense_linear_epilogue"
                else "eager_reference_fallback"
            )
            return spec.reference(*args, **filtered_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefer_dense_linear = self._prefer_dense_linear_fastpath(x) and self._preferred_patterns_config in (
            ["dequant_gemm_epilogue"],
            ["dense_linear_epilogue"],
        )
        if prefer_dense_linear:
            dense_args = self._resolve_dense_linear_args(x)
            if dense_args is not None:
                qweight, bias, activation = dense_args
                self.last_kernel_pattern = "dense_linear_epilogue"
                self.last_consumes_packed_weight = False
                self.last_unpack_stage = "one_time_eager_dequant_cache"
                self.last_fastpath = "one_time_eager_dequant_plus_dense_half_gemm"
                if x.is_cuda:
                    self.last_execution_mode = (
                        "cuda_native_fastpath"
                        if self._prefer_native_linear_fastpath(x)
                        else "cuda_tilelang_entry"
                    )
                    self.last_execution_reason = None
                else:
                    self.last_execution_mode = "reference_fallback"
                    self.last_execution_reason = (
                        "TileLang dequant GEMM kernel requires CUDA tensors; using configured fallback."
                    )
                if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
                    if self._can_delegate_dense_native_linear(activation):
                        return self.module(x)
                    return self._apply_activation(F.linear(x, qweight, bias), activation)
                if bias is not None:
                    return self._run_tilelang_or_reference(
                        "dense_linear_epilogue",
                        x,
                        qweight,
                        bias,
                        activation=activation,
                        block_m=int(self.settings.get("block_m", 64)),
                        block_n=int(self.settings.get("block_n", 64)),
                        block_k=int(self.settings.get("block_k", 64)),
                        threads=int(self.settings.get("threads", 128)),
                        num_stages=int(self.settings.get("num_stages", 2)),
                        target_arch=self.settings.get("target_arch"),
                        fallback=self.fallback,
                    )
                return self._run_tilelang_or_reference(
                    "dense_linear_epilogue",
                    x,
                    qweight,
                    activation=activation,
                    block_m=int(self.settings.get("block_m", 64)),
                    block_n=int(self.settings.get("block_n", 64)),
                    block_k=int(self.settings.get("block_k", 64)),
                    threads=int(self.settings.get("threads", 128)),
                    num_stages=int(self.settings.get("num_stages", 2)),
                    target_arch=self.settings.get("target_arch"),
                    fallback=self.fallback,
                )
        qweight: torch.Tensor | None = None
        scale: torch.Tensor | None = None
        bias: torch.Tensor | None = None
        activation: str | None = None
        kernel_pattern = "dequant_gemm_epilogue"
        extra_kwargs: dict[str, Any] = {}
        if callable(self._packed_fp4_bridge):
            packed_weight, scale, bias, activation, input_features, group_size = self._packed_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            qweight = packed_weight
            kernel_pattern = "fp4_packed_dequant_gemm_epilogue"
            extra_kwargs = {
                "input_features": int(input_features),
                "group_size": int(group_size),
            }
            self.last_weight_source = "reference_fp4_linear_packed_bridge"
            self.last_weight_representation = "packed_signed_int4_plus_group_scale"
            self.last_consumes_packed_weight = True
            self.last_fastpath = "packed_fp4_fused_tilelang_kernel"
        elif callable(self._dense_fp4_bridge):
            qweight, scale, bias, activation = self._dense_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            self.last_weight_source = "reference_fp4_linear_dense_bridge"
            self.last_weight_representation = "dense_unpacked_codes_plus_expanded_scale"
            self.last_consumes_packed_weight = False
            self.last_fastpath = "dense_codes_times_scale"
        else:
            packed_nvfp4_args = self._resolve_packed_nvfp4_args(x)
            if packed_nvfp4_args is not None:
                (
                    packed_weight,
                    scale,
                    bias,
                    activation,
                    input_features,
                    group_size,
                    weight_global_scale,
                ) = packed_nvfp4_args
                qweight = packed_weight
                kernel_pattern = "nvfp4_packed_dequant_gemm_epilogue"
                extra_kwargs = {
                    "input_features": int(input_features),
                    "group_size": int(group_size),
                    "weight_global_scale": weight_global_scale,
                }
                self.last_consumes_packed_weight = True
                self.last_fastpath = "packed_nvfp4_fused_tilelang_kernel"
            else:
                qweight = getattr(self.module, "qweight", None)
                scale = getattr(self.module, "scale", None)
                bias = getattr(self.module, "bias", None)
                activation = getattr(self.module, "activation", None)
                self.last_weight_source = "module_qweight_scale"
                self.last_weight_representation = "dense_qweight_plus_scale"
                self.last_consumes_packed_weight = False
                self.last_fastpath = "dense_codes_times_scale"
        if not isinstance(qweight, torch.Tensor) or (
            kernel_pattern != "dense_linear_epilogue" and not isinstance(scale, torch.Tensor)
        ):
            raise XQTBackendError(
                "TileLang dequant GEMM target requires qweight/scale tensors or a tilelang_dequant_gemm_args bridge"
            )
        tensors = (
            (x, qweight)
            if kernel_pattern == "dense_linear_epilogue" and bias is None
            else (x, qweight, bias)
            if kernel_pattern == "dense_linear_epilogue"
            else (x, qweight, scale)
            if bias is None
            else (x, qweight, scale, bias)
        )
        uses_cuda = all(tensor.is_cuda for tensor in tensors)
        prefers_native_linear = (
            kernel_pattern == "dense_linear_epilogue"
            and uses_cuda
            and self._prefer_native_linear_fastpath(x)
        )
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if prefers_native_linear
            else "cuda_tilelang_entry"
            if uses_cuda
            else "reference_fallback"
        )
        self.last_kernel_pattern = kernel_pattern
        self.last_unpack_stage = (
            "one_time_eager_dequant_cache"
            if kernel_pattern == "dense_linear_epilogue"
            else
            "tilelang_fused_gemm_kernel"
            if uses_cuda and kernel_pattern in {"fp4_packed_dequant_gemm_epilogue", "nvfp4_packed_dequant_gemm_epilogue"}
            else "eager_reference_fallback"
            if kernel_pattern in {"fp4_packed_dequant_gemm_epilogue", "nvfp4_packed_dequant_gemm_epilogue"}
            else None
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_tilelang_entry", "cuda_native_fastpath"}
            else "TileLang dequant GEMM kernel requires CUDA tensors; using configured fallback."
        )
        tile_kwargs: dict[str, Any] = {
            "activation": activation,
            **extra_kwargs,
            "block_m": int(self.settings.get("block_m", 64)),
            "block_n": int(
                self.settings.get(
                    "block_n",
                    16 if kernel_pattern == "nvfp4_packed_dequant_gemm_epilogue" else 64,
                )
            ),
            "threads": int(self.settings.get("threads", 128)),
            "num_stages": int(self.settings.get("num_stages", 2)),
            "target_arch": self.settings.get("target_arch"),
            "fallback": self.fallback,
        }
        if kernel_pattern == "dense_linear_epilogue":
            tile_kwargs["block_k"] = int(self.settings.get("block_k", 64))
        if kernel_pattern == "nvfp4_packed_dequant_gemm_epilogue":
            tile_kwargs["block_k"] = int(self.settings.get("block_k", 128))
        if kernel_pattern == "dense_linear_epilogue":
            if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
                return self._apply_activation(F.linear(x, qweight, bias), activation)
            if bias is not None:
                return self._run_tilelang_or_reference(
                    kernel_pattern,
                    x,
                    qweight,
                    bias,
                    **tile_kwargs,
                )
            return self._run_tilelang_or_reference(
                kernel_pattern,
                x,
                qweight,
                **tile_kwargs,
            )
        return self._run_tilelang_or_reference(
            kernel_pattern,
            x,
            qweight,
            scale,
            bias,
            **tile_kwargs,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "kernel_constraints": {
                "dtype": "float16",
                "batch_multiple_of_block_m": True,
                "out_features_multiple_of_block_n": True,
                "supported_activations": [None, "gelu", "silu", "relu"],
                "supported_patterns": [
                    "dense_linear_epilogue",
                    "dequant_gemm_epilogue",
                    "fp4_packed_dequant_gemm_epilogue",
                    "nvfp4_packed_dequant_gemm_epilogue",
                ],
                "operator_families": ["linear", "conv", "attention"],
                "supports_reference_fp4_linear_bridge": True,
                "supports_packed_fp4_bridge": True,
                "supports_packed_nvfp4_bridge": True,
            },
            "kernel_pattern": self.last_kernel_pattern,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "weight_source": self.last_weight_source,
            "weight_representation": self.last_weight_representation,
            "consumes_packed_weight": self.last_consumes_packed_weight,
            "unpack_stage": self.last_unpack_stage,
            "fusion_status": (
                "tilelang_dense_half_gemm_epilogue"
                if self.last_unpack_stage == "one_time_eager_dequant_cache"
                else
                "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue"
                if self.last_unpack_stage == "tilelang_fused_gemm_kernel"
                else None
            ),
            "epilogue_stage": (
                "torch_bias_activation"
                if self.last_unpack_stage == "one_time_eager_dequant_cache"
                else
                "tilelang_fused_bias_activation"
                if self.last_unpack_stage == "tilelang_fused_gemm_kernel"
                else None
            ),
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _ReferenceGuardedLinearWrapper(nn.Module):
    """Executable linear/dequant GEMM wrapper for reference-guarded backends."""

    _REFERENCE_ONLY_PRODUCTION_STATUSES = {"", "metadata_only", "reference_guarded"}

    def __init__(
        self,
        module: nn.Module,
        *,
        backend: str,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        if backend not in {"cutile", "cute_dsl"}:
            raise XQTBackendError(f"unsupported reference-guarded backend: {backend}")
        self.module = module
        self.backend = backend
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_weight_source = "not_run"
        self.last_weight_representation = "unknown"
        self.last_consumes_packed_weight = False
        self.last_unpack_stage: str | None = None
        self.last_kernel_pattern = self._default_kernel_pattern()
        self.last_operator_family = "linear"
        self.last_fastpath = "none"
        self._cached_nvfp4_bridge: NVFP4LinearBridge | None = None
        self._dense_linear_bridge = getattr(self.module, "tilelang_dense_linear_args", None)
        self._packed_nvfp4_bridge = getattr(
            self.module,
            "tilelang_packed_nvfp4_dequant_gemm_args",
            None,
        )
        self._packed_fp4_bridge = getattr(
            self.module,
            "tilelang_packed_dequant_gemm_args",
            None,
        )
        if self._packed_nvfp4_bridge is None and self._packed_fp4_bridge is None:
            inferred_bridge = bridge_module_to_nvfp4_linear_shared(self.module)
            if inferred_bridge is not None:
                self._cached_nvfp4_bridge = inferred_bridge

    def _preferred_patterns(self) -> list[str]:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list):
            return [str(pattern) for pattern in patterns]
        return [self._default_kernel_pattern()]

    def _default_kernel_pattern(self) -> str:
        if self.backend == "cute_dsl":
            return "gemm_epilogue"
        return "dense_linear_epilogue"

    def _backend_display_name(self) -> str:
        return "CuTe DSL" if self.backend == "cute_dsl" else "CuTile"

    def _get_kernel_spec(self, pattern: str) -> Any:
        if self.backend == "cute_dsl":
            return get_cute_dsl_kernel_spec(pattern)
        return get_cutile_kernel_spec(pattern)

    def _run_backend_kernel(
        self,
        pattern: str,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            if self.backend == "cute_dsl":
                return run_cute_dsl_kernel(
                    pattern,
                    *args,
                    fallback=self.fallback,
                    **kwargs,
                )
            return run_cutile_kernel(
                pattern,
                *args,
                fallback=self.fallback,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            spec = self._get_kernel_spec(pattern)
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in allowed
            }
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"{self._backend_display_name()} {pattern} runtime fallback: {exc}"
            )
            self.last_fastpath = "eager_reference_fallback"
            return spec.reference(*args, **filtered_kwargs)

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _resolved_nvfp4_bridge(self) -> NVFP4LinearBridge | None:
        if callable(self._packed_fp4_bridge):
            return None
        if self._cached_nvfp4_bridge is not None:
            return self._cached_nvfp4_bridge
        existing_bridge = getattr(self.module, "_bridge", None)
        if isinstance(existing_bridge, NVFP4LinearBridge):
            self._cached_nvfp4_bridge = existing_bridge
            return existing_bridge
        inferred_bridge = bridge_module_to_nvfp4_linear(self.module)
        if inferred_bridge is not None:
            self._cached_nvfp4_bridge = inferred_bridge
        return inferred_bridge

    def _resolve_dense_linear_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, str | None] | None:
        if callable(self._dense_linear_bridge):
            self.last_weight_source = f"{self.backend}_dense_cache_bridge"
            self.last_weight_representation = "dense_dequantized_weight_cache"
            return self._dense_linear_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is not None:
            self.last_weight_source = "auto_inferred_nvfp4_dense_cache_bridge"
            self.last_weight_representation = "dense_dequantized_weight_cache"
            return bridge.tilelang_dense_linear_args(dtype=x.dtype, device=x.device)
        if isinstance(self.module, nn.Linear):
            self.last_weight_source = "torch_linear_parameter"
            self.last_weight_representation = "dense_weight"
            bias = self.module.bias
            return (
                self.module.weight.to(dtype=x.dtype, device=x.device),
                None if bias is None else bias.to(dtype=x.dtype, device=x.device),
                None,
            )
        dense_quant = self._resolve_dense_quant_args(x)
        if dense_quant is not None:
            qweight, scale, bias, activation = dense_quant
            weight_scale = scale.to(dtype=x.dtype, device=x.device)
            if weight_scale.ndim == 1:
                weight_scale = weight_scale.unsqueeze(-1)
            weight = qweight.to(dtype=x.dtype, device=x.device) * weight_scale
            self.last_weight_source = "module_qweight_scale_dense_cache"
            self.last_weight_representation = "dense_dequantized_weight"
            return (
                weight,
                None if bias is None else bias.to(dtype=x.dtype, device=x.device),
                activation,
            )
        return None

    def _resolve_dense_quant_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str | None] | None:
        qweight = getattr(self.module, "qweight", None)
        scale = getattr(self.module, "scale", None)
        if not isinstance(qweight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            return None
        bias = getattr(self.module, "bias", None)
        activation = getattr(self.module, "activation", None)
        self.last_weight_source = "module_qweight_scale"
        self.last_weight_representation = "dense_qweight_plus_scale"
        return (
            qweight.to(dtype=x.dtype, device=x.device),
            scale.to(dtype=x.dtype, device=x.device),
            bias if isinstance(bias, torch.Tensor) else None,
            activation if isinstance(activation, str) else None,
        )

    def _resolve_packed_nvfp4_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None] | None:
        if callable(self._packed_nvfp4_bridge):
            self.last_weight_source = "compressed_tensors_nvfp4_packed_bridge"
            self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
            return self._packed_nvfp4_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_packed_bridge"
        self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
        return bridge.tilelang_packed_nvfp4_dequant_gemm_args(
            dtype=x.dtype,
            device=x.device,
        )

    def _select_cutile_pattern(self) -> str:
        patterns = self._preferred_patterns()
        if (
            "nvfp4_packed_dequant_gemm_epilogue" in patterns
            and self._cutile_pattern_has_runtime_kernel(
                "nvfp4_packed_dequant_gemm_epilogue"
            )
            and (
                callable(self._packed_nvfp4_bridge)
                or self._resolved_nvfp4_bridge() is not None
            )
        ):
            return "nvfp4_packed_dequant_gemm_epilogue"
        if (
            "nvfp4_packed_dequant_gemm_epilogue" in patterns
            and (
                callable(self._dense_linear_bridge)
                or self._resolved_nvfp4_bridge() is not None
            )
        ):
            return "dense_linear_epilogue"
        if "fp4_packed_dequant_gemm_epilogue" in patterns and callable(self._packed_fp4_bridge):
            return "fp4_packed_dequant_gemm_epilogue"
        if "dequant_gemm_epilogue" in patterns and self._resolve_dense_quant_args_for_selection():
            return "dequant_gemm_epilogue"
        if "dense_linear_epilogue" in patterns:
            return "dense_linear_epilogue"
        if "linear" in patterns:
            return "linear"
        return "dense_linear_epilogue"

    def _cutile_pattern_has_runtime_kernel(self, pattern: str) -> bool:
        if not cutile_available():
            return False
        try:
            metadata = self._get_kernel_spec(pattern).metadata
        except Exception:
            return False
        production_status = str(metadata.get("production_status", "")).lower()
        fusion_status = str(metadata.get("fusion_status", "")).lower()
        return (
            production_status not in self._REFERENCE_ONLY_PRODUCTION_STATUSES
            and "reference_guarded" not in fusion_status
        )

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("linear backends require at least one input dimension")
        prefix_shape = tuple(x.shape[:-1])
        return x.reshape(-1, x.shape[-1]), prefix_shape

    @staticmethod
    def _restore_flattened_output(
        output: torch.Tensor,
        prefix_shape: tuple[int, ...],
    ) -> torch.Tensor:
        return output.reshape(*prefix_shape, output.shape[-1])

    def _resolve_dense_quant_args_for_selection(self) -> bool:
        return isinstance(getattr(self.module, "qweight", None), torch.Tensor) and isinstance(
            getattr(self.module, "scale", None),
            torch.Tensor,
        )

    def _prepare_execution_state(
        self,
        *,
        x: torch.Tensor,
        pattern: str,
        consumes_packed_weight: bool,
        unpack_stage: str | None,
        fastpath: str,
    ) -> None:
        self.last_kernel_pattern = pattern
        self.last_consumes_packed_weight = consumes_packed_weight
        self.last_unpack_stage = unpack_stage
        self.last_fastpath = fastpath
        if x.is_cuda:
            self.last_execution_mode = f"cuda_{self.backend}_entry"
            self.last_execution_reason = None
        else:
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"{self._backend_display_name()} {pattern} requires CUDA tensors; using configured fallback."
            )

    def _forward_cute_dsl(self, x: torch.Tensor) -> torch.Tensor:
        dense_args = self._resolve_dense_linear_args(x)
        if dense_args is None:
            raise XQTBackendError(
                "CuTe DSL inference target requires nn.Linear, qweight/scale, or an NVFP4 dense bridge"
            )
        weight, bias, activation = dense_args
        self._prepare_execution_state(
            x=x,
            pattern="gemm_epilogue",
            consumes_packed_weight=False,
            unpack_stage="one_time_eager_dequant_cache",
            fastpath="cute_dsl_dense_gemm_epilogue",
        )
        return self._run_backend_kernel(
            "gemm_epilogue",
            x,
            weight,
            bias,
            activation=activation,
            tile_shape=tuple(self.settings.get("tile_shape", (128, 128, 64))),
            cluster_shape=self.settings.get("cluster_shape"),
        )

    def _forward_cutile_packed_nvfp4(self, x: torch.Tensor) -> torch.Tensor:
        packed_args = self._resolve_packed_nvfp4_args(x)
        if packed_args is None:
            raise XQTBackendError("CuTile NVFP4 target requires a packed NVFP4 bridge")
        packed_weight, scale, bias, activation, input_features, group_size, weight_global_scale = packed_args
        flat_x, prefix_shape = self._flatten_input(x)
        self._prepare_execution_state(
            x=x,
            pattern="nvfp4_packed_dequant_gemm_epilogue",
            consumes_packed_weight=True,
            unpack_stage=(
                "cutile_reference_guarded_unpack"
                if x.is_cuda
                else "eager_reference_fallback"
            ),
            fastpath="packed_nvfp4_cutile_reference_guarded_kernel",
        )
        output = self._run_backend_kernel(
            "nvfp4_packed_dequant_gemm_epilogue",
            flat_x,
            packed_weight,
            scale,
            bias,
            input_features=int(input_features),
            group_size=int(group_size),
            weight_global_scale=weight_global_scale,
            activation=activation,
            threads=int(self.settings.get("threads", 128)),
            target_arch=self._resolved_target_arch(x),
        )
        return self._restore_flattened_output(output, prefix_shape)

    def _forward_cutile_dequant(self, x: torch.Tensor) -> torch.Tensor:
        dense_quant = self._resolve_dense_quant_args(x)
        if dense_quant is None:
            raise XQTBackendError("CuTile dequant GEMM target requires qweight/scale tensors")
        qweight, scale, bias, activation = dense_quant
        flat_x, prefix_shape = self._flatten_input(x)
        self._prepare_execution_state(
            x=x,
            pattern="dequant_gemm_epilogue",
            consumes_packed_weight=False,
            unpack_stage="cutile_reference_guarded_dequant",
            fastpath="cutile_dequant_gemm_epilogue",
        )
        output = self._run_backend_kernel(
            "dequant_gemm_epilogue",
            flat_x,
            qweight,
            scale,
            bias,
            activation=activation,
            threads=int(self.settings.get("threads", 128)),
            target_arch=self._resolved_target_arch(x),
        )
        return self._restore_flattened_output(output, prefix_shape)

    def _forward_cutile_dense(self, x: torch.Tensor, pattern: str) -> torch.Tensor:
        dense_args = self._resolve_dense_linear_args(x)
        if dense_args is None:
            raise XQTBackendError(
                "CuTile dense Linear target requires nn.Linear, qweight/scale, or an NVFP4 dense bridge"
            )
        weight, bias, activation = dense_args
        flat_x, prefix_shape = self._flatten_input(x)
        kernel_pattern = "linear" if pattern == "linear" else "dense_linear_epilogue"
        if kernel_pattern == "linear":
            activation = None
        self._prepare_execution_state(
            x=x,
            pattern=kernel_pattern,
            consumes_packed_weight=False,
            unpack_stage="one_time_eager_dequant_cache",
            fastpath=f"cutile_{kernel_pattern}",
        )
        kernel_kwargs: dict[str, Any] = {
            "threads": int(self.settings.get("threads", 128)),
            "target_arch": self._resolved_target_arch(x),
        }
        if kernel_pattern != "linear":
            kernel_kwargs["activation"] = activation
        output = self._run_backend_kernel(
            kernel_pattern,
            flat_x,
            weight,
            bias,
            **kernel_kwargs,
        )
        return self._restore_flattened_output(output, prefix_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "cute_dsl":
            return self._forward_cute_dsl(x)
        pattern = self._select_cutile_pattern()
        if pattern == "nvfp4_packed_dequant_gemm_epilogue":
            return self._forward_cutile_packed_nvfp4(x)
        if pattern == "dequant_gemm_epilogue":
            return self._forward_cutile_dequant(x)
        return self._forward_cutile_dense(x, pattern)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "reference_guarded_cuda_entry"
            if self.last_execution_mode.startswith("cuda_")
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        try:
            kernel_metadata = dict(self._get_kernel_spec(self.last_kernel_pattern).metadata)
        except Exception:
            kernel_metadata = {}
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": list(self._preferred_patterns()),
                "operator_families": ["linear"],
                "supports_packed_nvfp4_bridge": self.backend == "cutile",
                "supports_dense_nvfp4_cache_bridge": True,
            },
            "kernel_pattern": self.last_kernel_pattern,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "weight_source": self.last_weight_source,
            "weight_representation": self.last_weight_representation,
            "consumes_packed_weight": self.last_consumes_packed_weight,
            "unpack_stage": self.last_unpack_stage,
            "fusion_status": kernel_metadata.get(
                "fusion_status",
                f"{self.backend}_reference_guarded_gemm_epilogue",
            ),
            "epilogue_stage": kernel_metadata.get("epilogue_stage", "torch_bias_activation"),
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


def _build_tilelang_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    patterns = target.patterns or ["attention"]
    settings = dict(target.tilelang)
    settings["preferred_patterns"] = list(patterns)
    if patterns == ["attention"]:
        if isinstance(target_model, nn.MultiheadAttention):
            return _TileLangAttentionWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        attention = getattr(target_model, "attention", None)
        if isinstance(attention, nn.MultiheadAttention):
            target_model = copy.deepcopy(target_model)
            target_model.attention = _TileLangAttentionWrapper(
                target_model.attention,
                fallback=target.fallback,
                settings=settings,
            )
            return target_model
        for child_name, child in target_model.named_children():
            nested_attention = getattr(child, "attention", None)
            if isinstance(nested_attention, nn.MultiheadAttention):
                target_model = copy.deepcopy(target_model)
                wrapped_child = target_model.get_submodule(child_name)
                wrapped_child.attention = _TileLangAttentionWrapper(
                    wrapped_child.attention,
                    fallback=target.fallback,
                    settings=settings,
                )
                return target_model
        raise XQTBackendError(
            "TileLang attention target requires nn.MultiheadAttention or a module with an .attention submodule"
        )
    if patterns == ["conv"]:
        if isinstance(target_model, nn.Conv2d):
            return _TileLangConvWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        conv = getattr(target_model, "conv", None)
        if isinstance(conv, nn.Conv2d):
            target_model = copy.deepcopy(target_model)
            target_model.conv = _TileLangConvWrapper(
                conv,
                fallback=target.fallback,
                settings=settings,
            )
            return target_model
        for child_name, child in target_model.named_children():
            if isinstance(child, nn.Conv2d):
                target_model = copy.deepcopy(target_model)
                setattr(
                    target_model,
                    child_name,
                    _TileLangConvWrapper(
                        child,
                        fallback=target.fallback,
                        settings=settings,
                    ),
                )
                return target_model
        raise XQTBackendError(
            "TileLang conv target requires nn.Conv2d or a module with a Conv2d child"
        )
    if patterns in (["linear"], ["linear_marlin"]):
        if isinstance(target_model, nn.Linear):
            return _TileLangLinearWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        linear = getattr(target_model, "linear", None)
        if isinstance(linear, nn.Linear):
            target_model = copy.deepcopy(target_model)
            target_model.linear = _TileLangLinearWrapper(
                linear,
                fallback=target.fallback,
                settings=settings,
            )
            return target_model
        for child_name, child in target_model.named_children():
            if isinstance(child, nn.Linear):
                target_model = copy.deepcopy(target_model)
                setattr(
                    target_model,
                    child_name,
                    _TileLangLinearWrapper(
                        child,
                        fallback=target.fallback,
                        settings=settings,
                    ),
                )
                return target_model
        raise XQTBackendError(
            "TileLang linear target requires nn.Linear or a module with a Linear child"
        )
    if patterns == ["norm"]:
        if isinstance(target_model, nn.LayerNorm):
            return _TileLangNormWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        norm = getattr(target_model, "norm", None)
        if isinstance(norm, nn.LayerNorm):
            target_model = copy.deepcopy(target_model)
            target_model.norm = _TileLangNormWrapper(
                norm,
                fallback=target.fallback,
                settings=settings,
            )
            return target_model
        for child_name, child in target_model.named_children():
            if isinstance(child, nn.LayerNorm):
                target_model = copy.deepcopy(target_model)
                setattr(
                    target_model,
                    child_name,
                    _TileLangNormWrapper(
                        child,
                        fallback=target.fallback,
                        settings=settings,
                    ),
                )
                return target_model
        raise XQTBackendError(
            "TileLang norm target requires nn.LayerNorm or a module with a LayerNorm child"
        )
    if patterns in (
        ["dequant_gemm_epilogue"],
        ["fp4_packed_dequant_gemm_epilogue"],
        ["nvfp4_packed_dequant_gemm_epilogue"],
    ):
        inferred_nvfp4_layout = infer_nvfp4_tensor_layout(target_model)
        if (
            patterns == ["dequant_gemm_epilogue"]
            and str(settings.get("target_arch") or "") == "sm_89"
            and inferred_nvfp4_layout is not None
            and _module_has_cuda_state(target_model)
        ):
            bridge = bridge_module_to_nvfp4_linear_shared(target_model)
            if bridge is not None:
                device, dtype = _infer_module_runtime_spec(target_model)
                weight, bias, _ = bridge.tilelang_dense_linear_args(
                    dtype=dtype,
                    device=device,
                )
                linear = nn.Linear(
                    bridge.input_features,
                    bridge.output_features,
                    bias=bias is not None,
                    device=device,
                    dtype=dtype,
                )
                with torch.no_grad():
                    linear.weight.copy_(weight)
                    if bias is not None and linear.bias is not None:
                        linear.bias.copy_(bias)
                metadata = {
                    "execution_mode": "cuda_native_fastpath",
                    "execution_reason": None,
                    "kernel_kind": "native_runtime_fastpath",
                    "kernel_constraints": {
                        "dtype": str(dtype),
                        "batch_multiple_of_block_m": True,
                        "out_features_multiple_of_block_n": True,
                        "supported_activations": [None, "gelu", "silu", "relu"],
                        "supported_patterns": [
                            "dense_linear_epilogue",
                            "dequant_gemm_epilogue",
                            "fp4_packed_dequant_gemm_epilogue",
                            "nvfp4_packed_dequant_gemm_epilogue",
                        ],
                        "operator_families": ["linear", "conv", "attention"],
                        "supports_reference_fp4_linear_bridge": True,
                        "supports_packed_fp4_bridge": True,
                        "supports_packed_nvfp4_bridge": True,
                    },
                    "kernel_pattern": "dense_linear_epilogue",
                    "operator_family": "linear",
                    "selected_fastpath": "eager_dense_native_linear_module",
                    "weight_source": "auto_inferred_nvfp4_dense_cache_bridge",
                    "weight_representation": "dense_dequantized_weight_cache",
                    "consumes_packed_weight": False,
                    "unpack_stage": "one_time_eager_dequant_cache",
                    "fusion_status": "tilelang_dense_half_gemm_epilogue",
                    "epilogue_stage": "torch_bias_activation",
                    "fallback": target.fallback,
                    "settings": dict(settings),
                }
                return _attach_tilelang_execution_metadata(
                    _TileLangEagerDenseLinearModule(
                        linear,
                        metadata=metadata,
                    ),
                    metadata,
                )
        if (
            patterns == ["dequant_gemm_epilogue"]
            and str(settings.get("target_arch") or "") == "sm_89"
            and callable(getattr(target_model, "tilelang_dense_linear_args", None))
            and hasattr(target_model, "dense_weight")
            and _module_has_cuda_state(target_model)
        ):
            return _attach_tilelang_execution_metadata(
                target_model,
                {
                    "execution_mode": "cuda_native_fastpath",
                    "execution_reason": None,
                    "kernel_kind": "native_runtime_fastpath",
                    "kernel_constraints": {
                        "dtype": "float16",
                        "batch_multiple_of_block_m": True,
                        "out_features_multiple_of_block_n": True,
                        "supported_activations": [None, "gelu", "silu", "relu"],
                        "supported_patterns": [
                            "dense_linear_epilogue",
                            "dequant_gemm_epilogue",
                            "fp4_packed_dequant_gemm_epilogue",
                            "nvfp4_packed_dequant_gemm_epilogue",
                        ],
                        "operator_families": ["linear", "conv", "attention"],
                        "supports_reference_fp4_linear_bridge": True,
                        "supports_packed_fp4_bridge": True,
                        "supports_packed_nvfp4_bridge": True,
                    },
                    "kernel_pattern": "dense_linear_epilogue",
                    "operator_family": "linear",
                    "selected_fastpath": "delegated_native_dense_linear",
                    "weight_source": "module_dense_weight",
                    "weight_representation": "dense_dequantized_weight_cache",
                    "consumes_packed_weight": False,
                    "unpack_stage": "one_time_eager_dequant_cache",
                    "fusion_status": "tilelang_dense_half_gemm_epilogue",
                    "epilogue_stage": "torch_bias_activation",
                    "fallback": target.fallback,
                    "settings": dict(settings),
                },
            )
        if (
            inferred_nvfp4_layout is not None
            or
            callable(getattr(target_model, "tilelang_packed_nvfp4_dequant_gemm_args", None))
            or callable(getattr(target_model, "tilelang_packed_dequant_gemm_args", None))
            or callable(getattr(target_model, "tilelang_dequant_gemm_args", None))
        ) or all(
            hasattr(target_model, name) for name in ("qweight", "scale")
        ):
            return _TileLangDequantGemmWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        for child_name, child in target_model.named_children():
            inferred_child_nvfp4_layout = infer_nvfp4_tensor_layout(child)
            if (
                inferred_child_nvfp4_layout is not None
                or
                callable(getattr(child, "tilelang_packed_nvfp4_dequant_gemm_args", None))
                or callable(getattr(child, "tilelang_packed_dequant_gemm_args", None))
                or callable(getattr(child, "tilelang_dequant_gemm_args", None))
            ) or all(
                hasattr(child, name) for name in ("qweight", "scale")
            ):
                target_model = copy.deepcopy(target_model)
                wrapped_child = target_model.get_submodule(child_name)
                setattr(
                    target_model,
                    child_name,
                    _TileLangDequantGemmWrapper(
                        wrapped_child,
                        fallback=target.fallback,
                        settings=settings,
                    ),
                )
                return target_model
        raise XQTBackendError(
            "TileLang dequant GEMM target requires a module with qweight/scale tensors or a tilelang_dequant_gemm_args bridge"
        )
    raise XQTBackendError(
        "built-in TileLang executor currently supports attention, conv, linear, norm, and dequant_gemm_epilogue patterns"
    )


def _supports_reference_guarded_linear_backend(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    if infer_nvfp4_tensor_layout(module) is not None:
        return True
    if callable(getattr(module, "tilelang_dense_linear_args", None)):
        return True
    if callable(getattr(module, "tilelang_packed_nvfp4_dequant_gemm_args", None)):
        return True
    if callable(getattr(module, "tilelang_packed_dequant_gemm_args", None)):
        return True
    return all(hasattr(module, name) for name in ("qweight", "scale"))


def _build_reference_guarded_linear_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
    *,
    backend: str,
) -> nn.Module:
    settings = dict(target.cutile if backend == "cutile" else target.cute_dsl)
    settings["preferred_patterns"] = list(target.patterns or [])
    if _supports_reference_guarded_linear_backend(target_model):
        return _ReferenceGuardedLinearWrapper(
            target_model,
            backend=backend,
            fallback=target.fallback,
            settings=settings,
        )
    for child_name, child in target_model.named_children():
        if _supports_reference_guarded_linear_backend(child):
            target_model = copy.deepcopy(target_model)
            wrapped_child = target_model.get_submodule(child_name)
            setattr(
                target_model,
                child_name,
                _ReferenceGuardedLinearWrapper(
                    wrapped_child,
                    backend=backend,
                    fallback=target.fallback,
                    settings=settings,
                ),
            )
            return target_model
    raise XQTBackendError(
        f"{backend} inference target requires nn.Linear, qweight/scale tensors, or an NVFP4 bridge"
    )


def _tilelang_execution_metadata(model: nn.Module) -> dict[str, Any]:
    attached = getattr(model, "_xqt_tilelang_execution_metadata", None)
    if isinstance(attached, dict):
        return dict(attached)
    if isinstance(model, _TileLangAttentionWrapper):
        return model.execution_metadata()
    if isinstance(model, _TileLangConvWrapper):
        return model.execution_metadata()
    if isinstance(model, _TileLangLinearWrapper):
        return model.execution_metadata()
    if isinstance(model, _TileLangNormWrapper):
        return model.execution_metadata()
    if isinstance(model, _TileLangDequantGemmWrapper):
        return model.execution_metadata()
    attention = getattr(model, "attention", None)
    if isinstance(attention, _TileLangAttentionWrapper):
        return attention.execution_metadata()
    conv = getattr(model, "conv", None)
    if isinstance(conv, _TileLangConvWrapper):
        return conv.execution_metadata()
    linear = getattr(model, "linear", None)
    if isinstance(linear, _TileLangLinearWrapper):
        return linear.execution_metadata()
    norm = getattr(model, "norm", None)
    if isinstance(norm, _TileLangNormWrapper):
        return norm.execution_metadata()
    wrapped_module = getattr(model, "module", None)
    if isinstance(wrapped_module, _TileLangDequantGemmWrapper):
        return wrapped_module.execution_metadata()
    for module in model.modules():
        if isinstance(
            module,
            (
                _TileLangAttentionWrapper,
                _TileLangConvWrapper,
                _TileLangLinearWrapper,
                _TileLangNormWrapper,
                _TileLangDequantGemmWrapper,
            ),
        ):
            return module.execution_metadata()
    return {
        "execution_mode": "unknown",
        "execution_reason": None,
    }


def _operator_backend_execution_metadata(
    model: nn.Module,
    *,
    backend: str,
) -> dict[str, Any]:
    if backend == "tilelang":
        return _tilelang_execution_metadata(model)
    if backend in {"cutile", "cute_dsl"}:
        if isinstance(model, _ReferenceGuardedLinearWrapper):
            return model.execution_metadata()
        for module in model.modules():
            if isinstance(module, _ReferenceGuardedLinearWrapper) and module.backend == backend:
                return module.execution_metadata()
        return {
            "execution_mode": "unknown",
            "execution_reason": None,
        }
    return {}


def _scan_candidate_report(
    scanner: Callable[[nn.Module, Any], list[Any]],
    model: nn.Module,
    example_input: Any,
) -> dict[str, Any]:
    try:
        report = summarize_candidate_report(scanner(model, example_input))
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "candidate_count": 0,
            "patterns": [],
            "recommended_backends": [],
            "candidates": [],
        }
    report["status"] = "ok"
    report["error"] = None
    return report


def _torch_compile_explain_report(
    module: nn.Module,
    inputs: Any,
) -> dict[str, Any]:
    """Collect a stable subset of torch._dynamo.explain for report metadata."""

    dynamo = getattr(torch, "_dynamo", None)
    explain = getattr(dynamo, "explain", None) if dynamo is not None else None
    if explain is None:
        return {
            "status": "unavailable",
            "error": "torch._dynamo.explain is not available",
            "graph_count": None,
            "graph_break_count": None,
            "break_reasons": [],
            "op_count": None,
            "compile_times": None,
        }
    normalized = split_example_input(inputs)
    try:
        explain_output = explain(module)(*normalized.args, **normalized.kwargs)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "graph_count": None,
            "graph_break_count": None,
            "break_reasons": [],
            "op_count": None,
            "compile_times": None,
        }
    break_reasons = []
    for reason in getattr(explain_output, "break_reasons", []) or []:
        break_reasons.append(str(reason))
    return {
        "status": "ok",
        "error": None,
        "graph_count": getattr(explain_output, "graph_count", None),
        "graph_break_count": getattr(explain_output, "graph_break_count", None),
        "break_reasons": break_reasons,
        "op_count": getattr(explain_output, "op_count", None),
        "compile_times": str(getattr(explain_output, "compile_times", "")) or None,
    }


def materialize_operator_candidate_model(
    model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> tuple[nn.Module, float | None]:
    """Build one candidate root model with the target optimization applied."""

    candidate_root = copy.deepcopy(model)
    candidate_target = _resolve_component_model(candidate_root, target.target_path)
    if target.backend == "torch_compile":
        compiled_candidate, compile_time_ms = compile_with_torch(candidate_target, target)
        candidate_root = _replace_component_model(
            candidate_root,
            target.target_path,
            compiled_candidate,
        )
        return candidate_root, compile_time_ms
    if target.backend == "tilelang":
        tilelang_candidate = _build_tilelang_candidate_model(candidate_target, target)
        candidate_root = _replace_component_model(
            candidate_root,
            target.target_path,
            tilelang_candidate,
        )
        return candidate_root, None
    if target.backend in {"cutile", "cute_dsl"}:
        backend_candidate = _build_reference_guarded_linear_candidate_model(
            candidate_target,
            target,
            backend=target.backend,
        )
        candidate_root = _replace_component_model(
            candidate_root,
            target.target_path,
            backend_candidate,
        )
        return candidate_root, None
    raise XQTBackendError(
        f"Operator optimization backend '{target.backend}' is not executable yet"
    )


def materialize_operator_candidate_models(
    model: nn.Module,
    targets: Sequence[OperatorOptimizationTargetPlan],
    *,
    inplace: bool = False,
) -> nn.Module:
    """Materialize a root model with multiple operator targets applied sequentially."""

    candidate_root = model if inplace else copy.deepcopy(model)
    for target in targets:
        candidate_target = _resolve_component_model(candidate_root, target.target_path)
        if target.backend == "torch_compile":
            compiled_candidate, _ = compile_with_torch(candidate_target, target)
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                compiled_candidate,
            )
            continue
        if target.backend == "tilelang":
            tilelang_candidate = _build_tilelang_candidate_model(candidate_target, target)
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                tilelang_candidate,
            )
            continue
        if target.backend in {"cutile", "cute_dsl"}:
            backend_candidate = _build_reference_guarded_linear_candidate_model(
                candidate_target,
                target,
                backend=target.backend,
            )
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                backend_candidate,
            )
            continue
        raise XQTBackendError(
            f"Operator optimization backend '{target.backend}' is not executable yet"
        )
    return candidate_root


def _planned_operator_skip_reason(target: OperatorOptimizationTargetPlan) -> str | None:
    return None


def build_operator_optimization_plan(
    operator_config: OperatorOptimizationConfig,
) -> OperatorOptimizationExecutionPlan:
    """Build a normalized operator optimization execution plan from config."""

    if not operator_config.enabled:
        return OperatorOptimizationExecutionPlan(
            targets=[],
            default_backend=operator_config.default_backend,
            stage=operator_config.stage,
        )

    targets: list[OperatorOptimizationTargetPlan] = []
    for target in operator_config.targets:
        targets.append(
            OperatorOptimizationTargetPlan(
                name=target.name,
                backend=target.backend or operator_config.default_backend,
                target_path=target.target,
                mode=target.mode,
                fullgraph=target.fullgraph,
                dynamic=target.dynamic,
                options=dict(target.options),
                patterns=list(target.patterns),
                fallback=target.fallback,
                min_speedup=target.min_speedup,
                validate={
                    "atol": float(target.validate.atol),
                    "rtol": float(target.validate.rtol),
                },
                tilelang={
                    "target": target.tilelang.target,
                    "target_arch": target.tilelang.target_arch,
                    "threads": target.tilelang.threads,
                    "num_stages": target.tilelang.num_stages,
                    "cache_dir": target.tilelang.cache_dir,
                    "pass_configs": dict(target.tilelang.pass_configs),
                    "linear_runtime": target.tilelang.linear_runtime,
                    "linear_fastpath": target.tilelang.linear_fastpath,
                    "attention_fastpath": target.tilelang.attention_fastpath,
                    "conv_fastpath": target.tilelang.conv_fastpath,
                    "norm_fastpath": target.tilelang.norm_fastpath,
                },
                cutile={
                    "target": target.cutile.target,
                    "target_arch": target.cutile.target_arch,
                    "threads": target.cutile.threads,
                    "cache_dir": target.cutile.cache_dir,
                    "pass_configs": dict(target.cutile.pass_configs),
                },
                cutlass={
                    "target_arch": target.cutlass.target_arch,
                    "cache_dir": target.cutlass.cache_dir,
                    "tile_shape": list(target.cutlass.tile_shape),
                    "cluster_shape": (
                        list(target.cutlass.cluster_shape)
                        if target.cutlass.cluster_shape is not None
                        else None
                    ),
                    "pass_configs": dict(target.cutlass.pass_configs),
                },
                cute_dsl={
                    "target_arch": target.cute_dsl.target_arch,
                    "cache_dir": target.cute_dsl.cache_dir,
                    "tile_shape": list(target.cute_dsl.tile_shape),
                    "cluster_shape": (
                        list(target.cute_dsl.cluster_shape)
                        if target.cute_dsl.cluster_shape is not None
                        else None
                    ),
                    "pass_configs": dict(target.cute_dsl.pass_configs),
                },
            )
        )
    return OperatorOptimizationExecutionPlan(
        targets=targets,
        default_backend=operator_config.default_backend,
        stage=operator_config.stage,
        metadata={"target_names": _ordered_unique(target.name for target in targets)},
    )


def execute_operator_optimization_plan(
    context: XQTContext,
    plan: OperatorOptimizationExecutionPlan,
) -> OperatorOptimizationExecutionResult:
    """Execute a normalized operator optimization plan and return unified reports."""

    target_device = torch.device(context.config.model.device)
    current_model = context.require_model().to(target_device)
    context.model = current_model
    reports: list[OperatorOptimizationReport] = []
    artifacts: dict[str, Any] = {}
    if not plan.targets:
        return OperatorOptimizationExecutionResult(
            model=current_model,
            reports=[],
            artifacts={},
        )

    if context.example_inputs is None:
        raise ValueError("example_inputs are required for operator optimization")
    root_batch = context.example_inputs
    root_inputs = _move_to_device(
        extract_model_inputs(
            root_batch,
            expected_input_count=infer_model_input_count(current_model),
        ),
        target_device,
    )
    candidate_reports = {
        "fx": _scan_candidate_report(scan_fx_candidates, current_model, root_inputs),
        "torch_export": _scan_candidate_report(
            scan_export_candidates,
            current_model,
            root_inputs,
        ),
    }
    artifacts["operator_optimization_candidates"] = candidate_reports

    for target in plan.targets:
        capability = describe_operator_backend_capability(target.backend)
        target_model = _resolve_component_model(current_model, target.target_path)
        module_inputs = root_inputs
        if target.target_path:
            expected_input_count = infer_model_input_count(target_model)
            module_inputs = _move_to_device(
                extract_model_inputs(
                    root_batch,
                    expected_input_count=expected_input_count,
                ),
                target_device,
            )
        device, dtype = _infer_module_device_dtype(target_model, module_inputs)
        backend_metadata = _backend_metadata(target, dtype=dtype)
        artifact_paths = _artifact_paths_from_backend_metadata(backend_metadata)
        compile_explain = (
            _torch_compile_explain_report(target_model, module_inputs)
            if target.backend == "torch_compile"
            else {
                "status": "not_applicable",
                "error": None,
                "graph_count": None,
                "graph_break_count": None,
                "break_reasons": [],
                "op_count": None,
                "compile_times": None,
            }
        )

        skip_reason = _quant_runtime_guard(context, target)
        if skip_reason is None and target.backend == "torch_compile" and not capability.available:
            skip_reason = "torch.compile is not available in the current PyTorch build"
        if skip_reason is None and target.backend in {"triton", "cutlass", "custom_cuda"}:
            if not torch.cuda.is_available():
                skip_reason = f"{target.backend} requires CUDA-capable hardware"
            else:
                skip_reason = _planned_operator_skip_reason(target) or (
                    f"{target.backend} backend is configured but not implemented in the built-in executor"
                )
        if (
            skip_reason is None
            and target.backend in {"cutile", "cute_dsl"}
            and not torch.cuda.is_available()
            and target.fallback != "eager"
        ):
            skip_reason = f"{target.backend} requires CUDA-capable hardware"
        if skip_reason is None and target.backend == "deployment_backend":
            skip_reason = "deployment_backend is metadata-only in the built-in executor"
        fallback_detail = {
            "backend": target.backend,
            "fallback": target.fallback,
            "reason": skip_reason,
            "graph_break_count": compile_explain.get("graph_break_count"),
            "graph_breaks": list(compile_explain.get("break_reasons", [])),
            "compiled_regions": compile_explain.get("graph_count"),
            "explain": compile_explain,
        }

        baseline_output = first_tensor_output(
            _call_module_no_grad(target_model, module_inputs)
        )
        baseline_execution_detail: dict[str, Any] = {}
        if target.backend in {"tilelang", "cutile", "cute_dsl"}:
            baseline_execution_detail = _operator_backend_execution_metadata(
                target_model,
                backend=target.backend,
            )
        latency_before, baseline_benchmark_strategy = _benchmark_callable_for_execution(
            lambda: _call_module_no_grad(target_model, module_inputs),
            warmup=context.config.benchmark.warmup,
            iterations=context.config.benchmark.iterations,
            sync_cuda=context.config.benchmark.sync_cuda,
            device=context.config.model.device,
            execution_detail=baseline_execution_detail,
        )

        if skip_reason is not None:
            reports.append(
                OperatorOptimizationReport(
                    target_name=target.name,
                    module_path=target.target_path,
                    backend=target.backend,
                    runtime=capability.runtime,
                    applied=False,
                    fallback=target.fallback,
                    skip_reason=skip_reason,
                    compile_time_ms=None,
                    latency_before=latency_before,
                    latency_after=latency_before,
                    speedup=1.0,
                    numeric_diff={
                        "allclose": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                    },
                    device=device,
                    dtype=dtype,
                    shape_signature=_shape_signature(module_inputs),
                    exportable=capability.exportable,
                    artifact_paths=artifact_paths,
                    metadata={
                        "execution_state": "skipped",
                        "fallback_detail": fallback_detail,
                        "graph_break_report": compile_explain,
                        "options": dict(target.options),
                        "mode": target.mode,
                        "patterns": list(target.patterns),
                        "capability": capability.to_dict(),
                        **backend_metadata,
                    },
                )
            )
            continue

        compiled_model = None
        compile_time_ms = None
        try:
            if target.backend == "torch_compile":
                compiled_model, compile_time_ms = compile_with_torch(target_model, target)
            elif target.backend == "tilelang":
                compiled_model = _build_tilelang_candidate_model(target_model, target)
            elif target.backend in {"cutile", "cute_dsl"}:
                compiled_model = _build_reference_guarded_linear_candidate_model(
                    target_model,
                    target,
                    backend=target.backend,
                )
            else:
                raise XQTBackendError(
                    f"Operator optimization backend '{target.backend}' is not executable yet"
                )
        except Exception as exc:
            reports.append(
                OperatorOptimizationReport(
                    target_name=target.name,
                    module_path=target.target_path,
                    backend=target.backend,
                    runtime=capability.runtime,
                    applied=False,
                    fallback=target.fallback,
                    skip_reason=str(exc),
                    compile_time_ms=compile_time_ms,
                    latency_before=latency_before,
                    latency_after=latency_before,
                    speedup=1.0,
                    numeric_diff={
                        "allclose": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                    },
                    device=device,
                    dtype=dtype,
                    shape_signature=_shape_signature(module_inputs),
                    exportable=capability.exportable,
                    artifact_paths=artifact_paths,
                    metadata={
                        "execution_state": "fallback",
                        "fallback_detail": {
                            "backend": target.backend,
                            "fallback": target.fallback,
                            "reason": str(exc),
                            "graph_break_count": compile_explain.get("graph_break_count"),
                            "graph_breaks": list(compile_explain.get("break_reasons", [])),
                            "compiled_regions": compile_explain.get("graph_count"),
                            "explain": compile_explain,
                        },
                        "graph_break_report": compile_explain,
                        "options": dict(target.options),
                        "mode": target.mode,
                        "patterns": list(target.patterns),
                        "capability": capability.to_dict(),
                        **backend_metadata,
                    },
                )
            )
            continue

        candidate_root = copy.deepcopy(current_model)
        candidate_target = _resolve_component_model(candidate_root, target.target_path)
        if target.backend == "torch_compile":
            compiled_candidate, _ = compile_with_torch(candidate_target, target)
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                compiled_candidate,
            )
        elif target.backend == "tilelang" and compiled_model is target_model:
            candidate_root = current_model
            candidate_target = compiled_model
        elif target.backend == "tilelang":
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                _build_tilelang_candidate_model(candidate_target, target),
            )
        elif target.backend in {"cutile", "cute_dsl"}:
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                _build_reference_guarded_linear_candidate_model(
                    candidate_target,
                    target,
                    backend=target.backend,
                ),
            )
        candidate_target = _resolve_component_model(candidate_root, target.target_path)
        optimized_output = first_tensor_output(
            _call_module_no_grad(candidate_target, module_inputs)
        )
        execution_detail: dict[str, Any] = {}
        if target.backend in {"tilelang", "cutile", "cute_dsl"}:
            execution_detail = _operator_backend_execution_metadata(
                candidate_target,
                backend=target.backend,
            )
        identity_candidate = candidate_target is target_model
        effective_thresholds = _effective_validation_thresholds(
            target,
            baseline_output=baseline_output,
            optimized_output=optimized_output,
        )
        if target.backend == "tilelang":
            backend_metadata["validation_thresholds"] = dict(effective_thresholds)
        numeric_diff = compare_tensors(
            baseline_output,
            optimized_output,
            atol=effective_thresholds["atol"],
            rtol=effective_thresholds["rtol"],
        ).to_dict()
        benchmark_strategy = baseline_benchmark_strategy
        speedup_metric = "mean_ms"
        speedup_statistics: dict[str, float | None] = {}
        native_speedup_strategy = _native_runtime_speedup_strategy(execution_detail)
        paired_steady_state_strategy = _paired_steady_state_speedup_strategy(execution_detail)
        if native_speedup_strategy is not None:
            if identity_candidate:
                benchmark_strategy = "identity_native_baseline"
                latency_after = dict(latency_before)
                speedup_metric = "p50_ms"
                speedup_statistics = {
                    "mean_ms": 1.0,
                    "p50_ms": 1.0,
                    "paired_ratio_p50": 1.0,
                }
                speedup = 1.0
            else:
                benchmark_strategy = native_speedup_strategy
                paired_before, paired_after, paired_speedup_ratios = _benchmark_paired_callables(
                    lambda: _call_module_no_grad(target_model, module_inputs),
                    lambda: _call_module_no_grad(candidate_target, module_inputs),
                    warmup=context.config.benchmark.warmup,
                    iterations=context.config.benchmark.iterations,
                    sync_cuda=context.config.benchmark.sync_cuda,
                    device=context.config.model.device,
                )
                latency_before = paired_before.to_dict()
                latency_after = paired_after.to_dict()
                speedup_metric = "p50_ms"
                mean_before = float(latency_before["mean_ms"])
                mean_after = float(latency_after["mean_ms"])
                p50_before = float(latency_before["p50_ms"])
                p50_after = float(latency_after["p50_ms"])
                paired_speedup_sorted = sorted(paired_speedup_ratios)
                paired_speedup_p50 = _percentile(paired_speedup_sorted, 50) if paired_speedup_sorted else None
                speedup_statistics = {
                    "mean_ms": (mean_before / mean_after) if mean_after > 0.0 else None,
                    "p50_ms": (p50_before / p50_after) if p50_after > 0.0 else None,
                    "paired_ratio_p50": paired_speedup_p50,
                }
                speedup = (
                    paired_speedup_p50
                    if paired_speedup_p50 is not None
                    else speedup_statistics["p50_ms"]
                )
        elif paired_steady_state_strategy is not None:
            benchmark_strategy = paired_steady_state_strategy
            inner_iterations = _tilelang_inner_iterations(execution_detail)
            latency_before, latency_after, paired_speedup_ratios = _benchmark_paired_batched_callables(
                lambda: _call_module_no_grad(target_model, module_inputs),
                lambda: _call_module_no_grad(candidate_target, module_inputs),
                warmup=context.config.benchmark.warmup,
                iterations=context.config.benchmark.iterations,
                sync_cuda=context.config.benchmark.sync_cuda,
                device=context.config.model.device,
                inner_iterations=inner_iterations,
            )
            mean_before = float(latency_before["mean_ms"])
            mean_after = float(latency_after["mean_ms"])
            p50_before = float(latency_before["p50_ms"])
            p50_after = float(latency_after["p50_ms"])
            paired_speedup_sorted = sorted(paired_speedup_ratios)
            paired_speedup_p50 = _percentile(paired_speedup_sorted, 50) if paired_speedup_sorted else None
            speedup_statistics = {
                "mean_ms": (mean_before / mean_after) if mean_after > 0.0 else None,
                "p50_ms": (p50_before / p50_after) if p50_after > 0.0 else None,
                "paired_ratio_p50": paired_speedup_p50,
            }
            speedup = speedup_statistics["mean_ms"]
        else:
            latency_after, benchmark_strategy = _benchmark_callable_for_execution(
                lambda: _call_module_no_grad(candidate_target, module_inputs),
                warmup=context.config.benchmark.warmup,
                iterations=context.config.benchmark.iterations,
                sync_cuda=context.config.benchmark.sync_cuda,
                device=context.config.model.device,
                execution_detail=execution_detail,
            )
            mean_before = float(latency_before["mean_ms"])
            mean_after = float(latency_after["mean_ms"])
            speedup = (mean_before / mean_after) if mean_after > 0.0 else None
            speedup_statistics = {
                "mean_ms": speedup,
                "p50_ms": (
                    float(latency_before["p50_ms"]) / float(latency_after["p50_ms"])
                    if float(latency_after["p50_ms"]) > 0.0
                    else None
                ),
            }
        if target.backend in {"tilelang", "cutile", "cute_dsl"}:
            execution_detail = _operator_backend_execution_metadata(
                candidate_target,
                backend=target.backend,
            )
        effective_min_speedup = _effective_min_speedup(
            target,
            execution_detail=execution_detail,
        )
        meets_numeric = bool(numeric_diff.get("allclose"))
        meets_speedup = speedup is not None and speedup >= effective_min_speedup
        if not meets_speedup and _native_runtime_near_equal(
            execution_detail,
            latency_before=latency_before,
            latency_after=latency_after,
        ):
            meets_speedup = True
        applied = meets_numeric and meets_speedup
        skip_reason = None
        if not meets_numeric:
            skip_reason = "numeric validation failed"
        elif not meets_speedup:
            skip_reason = (
                f"speedup {speedup:.4f} did not reach min_speedup {effective_min_speedup:.4f}"
                if speedup is not None
                else "latency_after is zero so speedup could not be computed"
            )
        if applied:
            current_model = _replace_component_model(current_model, target.target_path, compiled_model)
        elif target.backend == "tilelang" and compiled_model is target_model:
            if hasattr(compiled_model, "_xqt_tilelang_execution_metadata"):
                delattr(compiled_model, "_xqt_tilelang_execution_metadata")
        fallback_detail = {
            "backend": target.backend,
            "fallback": target.fallback,
            "reason": skip_reason,
            "graph_break_count": compile_explain.get("graph_break_count"),
            "graph_breaks": list(compile_explain.get("break_reasons", [])),
            "compiled_regions": compile_explain.get("graph_count"),
            "explain": compile_explain,
        }
        reports.append(
            OperatorOptimizationReport(
                target_name=target.name,
                module_path=target.target_path,
                backend=target.backend,
                runtime=capability.runtime,
                applied=applied,
                fallback=target.fallback,
                skip_reason=skip_reason,
                compile_time_ms=compile_time_ms,
                latency_before=latency_before,
                latency_after=latency_after,
                speedup=speedup,
                numeric_diff=numeric_diff,
                device=device,
                dtype=dtype,
                shape_signature=_shape_signature(module_inputs),
                exportable=capability.exportable,
                artifact_paths=artifact_paths,
                metadata={
                    "execution_state": "executed" if applied else "fallback",
                    "fallback_detail": fallback_detail,
                    "graph_break_report": compile_explain,
                    "options": dict(target.options),
                    "mode": target.mode,
                        "patterns": list(target.patterns),
                        "min_speedup": target.min_speedup,
                        "effective_min_speedup": effective_min_speedup,
                        "benchmark_strategy": benchmark_strategy,
                        "speedup_metric": speedup_metric,
                        "speedup_statistics": speedup_statistics,
                        "capability": capability.to_dict(),
                        **backend_metadata,
                        **execution_detail,
                },
            )
        )

    manifest_path = Path(context.config.project.artifact_dir) / "operator_optimization.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "targets": [report.to_dict() for report in reports],
                "candidates": candidate_reports,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["operator_optimization_report"] = manifest_path
    return OperatorOptimizationExecutionResult(
        model=current_model,
        reports=reports,
        artifacts=artifacts,
    )


def summarize_operator_optimization_reports(
    reports: list[OperatorOptimizationReport],
    *,
    candidate_reports: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a unified metrics payload from operator optimization reports."""

    items = [report.to_dict() for report in reports]
    applied = [report for report in reports if report.applied]
    skipped = [report for report in reports if not report.applied]
    summary = {
        "target_count": len(reports),
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "targets": items,
        "backends": _ordered_unique(report.backend for report in reports),
        "runtimes": _ordered_unique(report.runtime for report in reports),
        "applied_targets": [report.target_name for report in applied],
        "skipped_targets": [report.target_name for report in skipped],
    }
    if candidate_reports is not None:
        summary["candidates"] = dict(candidate_reports)
    return summary


__all__ = [
    "build_operator_optimization_plan",
    "execute_operator_optimization_plan",
    "materialize_operator_candidate_model",
    "summarize_operator_optimization_reports",
]
