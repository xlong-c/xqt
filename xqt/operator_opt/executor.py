"""Operator optimization execution helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
from torch import nn

from xqt.benchmark import benchmark_callable
from xqt.core.errors import XQTBackendError
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.schema import OperatorOptimizationConfig
from xqt.core.types import XQTContext
from xqt.analysis.compare import compare_tensors
from xqt.export.input_utils import first_tensor_output, split_example_input
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
    list_cutile_kernel_specs,
)
from .backends.cutlass import (
    CutlassCompileSettings,
    build_cutlass_artifact_metadata,
    list_cutlass_kernel_specs,
)
from .backends.tilelang import (
    TileLangCompileSettings,
    build_tilelang_artifact_metadata,
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


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


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
    return metadata


def _artifact_paths_from_backend_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    for backend in ("tilelang", "cutile", "cutlass"):
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
        q = query if self.attention.batch_first else query.transpose(0, 1)
        source_key = query if key is None else key
        source_value = source_key if value is None else value
        k = source_key if self.attention.batch_first else source_key.transpose(0, 1)
        v = source_value if self.attention.batch_first else source_value.transpose(0, 1)
        self.last_execution_mode = (
            "cuda_tilelang_entry"
            if q.is_cuda and k.is_cuda and v.is_cuda
            else "reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "TileLang attention kernel requires CUDA tensors; using configured fallback."
        )
        attn_output = run_tilelang_kernel(
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
        output = attn_output if self.attention.batch_first else attn_output.transpose(0, 1)
        weights = None
        if need_weights:
            batch_size = int(output.shape[0]) if output.ndim >= 3 else 0
            target_len = int(output.shape[1]) if output.ndim >= 3 else 0
            source_len = int(k.shape[1]) if k.ndim >= 3 else 0
            weights = output.new_zeros((batch_size, target_len, source_len))
        return output, weights

    def execution_metadata(self) -> dict[str, Any]:
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


def _build_tilelang_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    if "attention" not in (target.patterns or ["attention"]):
        raise XQTBackendError(
            "built-in TileLang executor currently supports only the attention pattern"
        )
    if isinstance(target_model, nn.MultiheadAttention):
        return _TileLangAttentionWrapper(
            target_model,
            fallback=target.fallback,
            settings=target.tilelang,
        )
    attention = getattr(target_model, "attention", None)
    if isinstance(attention, nn.MultiheadAttention):
        target_model = copy.deepcopy(target_model)
        target_model.attention = _TileLangAttentionWrapper(
            attention,
            fallback=target.fallback,
            settings=target.tilelang,
        )
        return target_model
    for child_name, child in target_model.named_children():
        nested_attention = getattr(child, "attention", None)
        if isinstance(nested_attention, nn.MultiheadAttention):
            target_model = copy.deepcopy(target_model)
            wrapped_child = target_model.get_submodule(child_name)
            wrapped_child.attention = _TileLangAttentionWrapper(
                nested_attention,
                fallback=target.fallback,
                settings=target.tilelang,
            )
            return target_model
    raise XQTBackendError(
        "TileLang attention target requires nn.MultiheadAttention or a module with an .attention submodule"
    )


def _tilelang_execution_metadata(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, _TileLangAttentionWrapper):
        return model.execution_metadata()
    attention = getattr(model, "attention", None)
    if isinstance(attention, _TileLangAttentionWrapper):
        return attention.execution_metadata()
    for module in model.modules():
        if isinstance(module, _TileLangAttentionWrapper):
            return module.execution_metadata()
    return {
        "execution_mode": "unknown",
        "execution_reason": None,
    }


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
    raise XQTBackendError(
        f"Operator optimization backend '{target.backend}' is not executable yet"
    )


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
        device = None
        dtype = None
        first_parameter = next(target_model.parameters(), None)
        if first_parameter is not None:
            device = str(first_parameter.device)
            dtype = str(first_parameter.dtype)
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
        if skip_reason is None and target.backend in {"triton", "cutile", "cutlass", "custom_cuda"}:
            if not torch.cuda.is_available():
                skip_reason = f"{target.backend} requires CUDA-capable hardware"
            else:
                skip_reason = _planned_operator_skip_reason(target) or (
                    f"{target.backend} backend is configured but not implemented in the built-in executor"
                )
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
        latency_before = benchmark_callable(
            lambda: _call_module_no_grad(target_model, module_inputs),
            warmup=context.config.benchmark.warmup,
            iterations=context.config.benchmark.iterations,
            sync_cuda=context.config.benchmark.sync_cuda,
            device=context.config.model.device,
        ).to_dict()

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
        elif target.backend == "tilelang":
            candidate_root = _replace_component_model(
                candidate_root,
                target.target_path,
                _build_tilelang_candidate_model(candidate_target, target),
            )
        candidate_target = _resolve_component_model(candidate_root, target.target_path)
        optimized_output = first_tensor_output(
            _call_module_no_grad(candidate_target, module_inputs)
        )
        execution_detail: dict[str, Any] = {}
        if target.backend == "tilelang":
            execution_detail = _tilelang_execution_metadata(candidate_target)
        numeric_diff = compare_tensors(
            baseline_output,
            optimized_output,
            atol=target.validate.get("atol", 1e-5),
            rtol=target.validate.get("rtol", 1e-5),
        ).to_dict()
        latency_after = benchmark_callable(
            lambda: _call_module_no_grad(candidate_target, module_inputs),
            warmup=context.config.benchmark.warmup,
            iterations=context.config.benchmark.iterations,
            sync_cuda=context.config.benchmark.sync_cuda,
            device=context.config.model.device,
        ).to_dict()
        mean_before = float(latency_before["mean_ms"])
        mean_after = float(latency_after["mean_ms"])
        speedup = (mean_before / mean_after) if mean_after > 0.0 else None
        meets_numeric = bool(numeric_diff.get("allclose"))
        meets_speedup = speedup is not None and speedup >= target.min_speedup
        applied = meets_numeric and meets_speedup
        skip_reason = None
        if not meets_numeric:
            skip_reason = "numeric validation failed"
        elif not meets_speedup:
            skip_reason = (
                f"speedup {speedup:.4f} did not reach min_speedup {target.min_speedup:.4f}"
                if speedup is not None
                else "latency_after is zero so speedup could not be computed"
            )
        if applied:
            current_model = _replace_component_model(current_model, target.target_path, compiled_model)
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
                    "capability": capability.to_dict(),
                    **backend_metadata,
                    "execution_mode": execution_detail.get("execution_mode"),
                    "execution_reason": execution_detail.get("execution_reason"),
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
