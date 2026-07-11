"""Preflight checks for XQT recipes."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from xqt.core.config import ConfigInput
from xqt.core.imports import resolve_target
from xqt.core.schema import (
    QuantComponentPolicyConfig,
    QuantConfig,
    PruneConfig,
    TASK_TYPES,
)
from xqt.operator_opt.capability import describe_operator_engine_capability
from xqt.operator_opt.cuda_extension import describe_custom_cuda_extension_capability
from xqt.prune import describe_prune_runtime_capability
from xqt.quant.capability import describe_quant_backend_capability
from xqt.export.tensorrt import validate_tensorrt_plugin_libraries
from xqt.workflows.optimization import (
    OptimizationConfig,
    OptimizationStageConfig,
    load_optimization_config,
)
from xqt.workflows.stage_specs import (
    AnalyzeStageSpec,
    BenchmarkStageSpec,
    DeployRuntimeHandleSpec,
    DeployStageSpec,
    ExportStageSpec,
    OperatorStageSpec,
    PruneStageSpec,
    QuantStageSpec,
    ensure_stage_spec,
)


@dataclass
class PreflightCheck:
    """Single preflight check result."""

    name: str
    passed: bool
    message: str
    level: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "level": self.level,
            "metadata": dict(self.metadata),
        }


@dataclass
class PreflightReport:
    """Preflight report for one XQT recipe."""

    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, message: str, **metadata: Any) -> None:
        level = str(metadata.pop("level", "info"))
        self.checks.append(
            PreflightCheck(
                name=name,
                passed=passed,
                message=message,
                level=level,
                metadata=metadata,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


def _package_available(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _check_target(report: PreflightReport, name: str, target: str | None) -> None:
    if not target:
        report.add(name, True, "target is not configured")
        return
    try:
        resolve_target(target)
    except Exception as exc:
        report.add(name, False, f"failed to resolve target: {exc}", target=target)
        return
    report.add(name, True, "target resolved", target=target)


def _check_dependency(report: PreflightReport, package_name: str) -> None:
    available = _package_available(package_name)
    report.add(
        f"dependency.{package_name}",
        available,
        "available" if available else "missing optional dependency",
        package=package_name,
    )


def _module_metadata(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return {}
    version = getattr(module, "__version__", None)
    metadata: dict[str, Any] = {}
    if version is not None:
        metadata["version"] = str(version)
    return metadata


def _cutile_available() -> bool:
    try:
        from xqt.operator_opt.backends.cutile import cutile_available
    except Exception:
        return _package_available("cutile")
    return cutile_available()


def _cutile_metadata() -> dict[str, Any]:
    try:
        from xqt.operator_opt.kernels.cutile._common import cutile_module_metadata
    except Exception:
        return _module_metadata("cutile")
    return cutile_module_metadata()


def _check_executable(report: PreflightReport, executable: str, name: str) -> None:
    path = shutil.which(executable)
    report.add(
        name,
        path is not None,
        f"found: {path}" if path is not None else "missing optional executable",
        executable=executable,
    )


def _check_optional_executable(
    report: PreflightReport,
    executable: str,
    name: str,
    *,
    dry_run: bool,
) -> None:
    path = shutil.which(executable)
    if path is not None:
        report.add(name, True, f"found: {path}", executable=executable, dry_run=dry_run)
        return
    report.add(
        name,
        bool(dry_run),
        "missing optional executable; dry-run command construction only"
        if dry_run
        else "missing optional executable",
        level="warning" if dry_run else "info",
        executable=executable,
        dry_run=dry_run,
    )


def _check_optional_dependency(
    report: PreflightReport,
    package_name: str,
    name: str,
    *,
    dry_run: bool,
) -> None:
    available = _package_available(package_name)
    report.add(
        name,
        available or bool(dry_run),
        "available"
        if available
        else "missing optional dependency; dry-run command construction only"
        if dry_run
        else "missing optional dependency",
        level="info" if available else "warning" if dry_run else "info",
        package=package_name,
        dry_run=dry_run,
    )


def _check_cuda(report: PreflightReport, name: str) -> None:
    available = torch.cuda.is_available()
    report.add(
        name,
        available,
        "CUDA available" if available else "CUDA is not available",
        device_count=torch.cuda.device_count(),
    )


def _check_model_device(report: PreflightReport, device: str | None) -> None:
    if not device:
        report.add("model.device", True, "model device is not configured")
        return
    try:
        torch_device = torch.device(device)
    except Exception as exc:
        report.add("model.device", False, f"invalid device: {exc}", device=device)
        return
    if torch_device.type != "cuda":
        report.add("model.device", True, "non-CUDA device", device=str(torch_device))
        return
    if not torch.cuda.is_available():
        report.add(
            "model.device",
            False,
            "CUDA is required by model.device but is not available",
            device=str(torch_device),
            device_count=torch.cuda.device_count(),
        )
        return
    if (
        torch_device.index is not None
        and torch_device.index >= torch.cuda.device_count()
    ):
        report.add(
            "model.device",
            False,
            "CUDA device index is out of range",
            device=str(torch_device),
            device_count=torch.cuda.device_count(),
        )
        return
    report.add(
        "model.device",
        True,
        "CUDA device available",
        device=str(torch_device),
        device_count=torch.cuda.device_count(),
    )


def _check_quant_backend_capability(
    report: PreflightReport,
    name: str,
    backend: str,
    *,
    method: str | None,
    strategy: str | None,
    policy: dict[str, Any],
    component_name: str | None = None,
) -> None:
    capability = describe_quant_backend_capability(
        backend,
        method=method,
        strategy=strategy,
        policy=policy,
    )
    metadata = capability.to_dict()
    if component_name is not None:
        metadata["component"] = component_name
    report.add(
        name,
        capability.status == "available",
        "quantization backend capability described",
        level="info" if capability.status == "available" else "warning",
        **metadata,
    )


def _check_external_calibration_inputs(
    report: PreflightReport,
    *,
    name: str,
    component_name: str | None = None,
) -> None:
    metadata: dict[str, Any] = {"source": "external_context.calibration_inputs"}
    if component_name is not None:
        metadata["component"] = component_name
    report.add(
        name,
        True,
        "ONNX QDQ requires external calibration_inputs at runtime",
        level="warning",
        **metadata,
    )


def _check_quant_component_policy(
    report: PreflightReport,
    quant_config: QuantConfig,
    component: QuantComponentPolicyConfig,
    *,
    prefix: str | None = None,
) -> None:
    prefix = prefix or f"compression.quant.component_policies.{component.name}"
    backend = component.backend or quant_config.backend
    if component.target is not None:
        report.add(
            f"{prefix}.target",
            True,
            "component target path configured",
            target=component.target,
        )
    report.add(
        f"{prefix}.backend",
        True,
        "component backend configured",
        backend=backend,
    )
    effective_policy = {**quant_config.policy, **component.policy}
    effective_strategy = component.strategy or quant_config.strategy
    _check_quant_backend_capability(
        report,
        f"{prefix}.capability",
        backend,
        method=component.method or quant_config.method,
        strategy=effective_strategy,
        policy=effective_policy,
        component_name=component.name,
    )
    if backend == "torchao":
        _check_dependency(report, "torchao")
        if describe_quant_backend_capability(
            backend,
            method=component.method or quant_config.method,
            strategy=effective_strategy,
            policy=effective_policy,
        ).requires_cuda:
            _check_cuda(report, f"{prefix}.hardware.cuda")
    if backend == "onnxruntime_qdq":
        _check_dependency(report, "onnxruntime")
        _check_external_calibration_inputs(
            report,
            name=f"{prefix}.data_source",
            component_name=component.name,
        )


def _check_quant_runtime_mix(
    report: PreflightReport,
    quant_config: QuantConfig,
    *,
    name: str = "compression.quant.runtime_mix",
) -> None:
    backends: list[str] = []
    if quant_config.component_policies:
        backends = [
            component.backend or quant_config.backend
            for component in quant_config.component_policies
            if component.enabled
        ]
    elif quant_config.enabled:
        backends = [quant_config.backend]
    unique_backends = sorted(set(backends))
    if len(unique_backends) <= 1:
        report.add(
            name,
            True,
            "single quantization runtime configured",
            backends=unique_backends,
        )
        return
    report.add(
        name,
        True,
        "multiple quantization runtimes configured",
        level="warning",
        backends=unique_backends,
    )


def _check_quant_config(
    report: PreflightReport,
    quant_config: QuantConfig,
    *,
    prefix: str,
    cuda_name: str,
) -> None:
    if not quant_config.enabled:
        return
    _check_quant_runtime_mix(report, quant_config, name=f"{prefix}.runtime_mix")
    _check_quant_backend_capability(
        report,
        f"{prefix}.capability",
        quant_config.backend,
        method=quant_config.method,
        strategy=quant_config.strategy,
        policy=quant_config.policy,
    )
    if quant_config.backend == "torchao":
        _check_dependency(report, "torchao")
        if describe_quant_backend_capability(
            quant_config.backend,
            method=quant_config.method,
            strategy=quant_config.strategy,
            policy=quant_config.policy,
        ).requires_cuda:
            _check_cuda(report, cuda_name)
    if quant_config.backend == "onnxruntime_qdq":
        _check_dependency(report, "onnxruntime")
        _check_external_calibration_inputs(
            report,
            name=f"{prefix}.calibration_inputs",
        )
    if quant_config.component_policies:
        report.add(
            f"{prefix}.component_policies",
            True,
            "component-level quantization policies configured",
            count=len(quant_config.component_policies),
        )
        for component in quant_config.component_policies:
            _check_quant_component_policy(
                report,
                quant_config,
                component,
                prefix=f"{prefix}.component_policies.{component.name}",
            )


def _check_operator_targets(
    report: PreflightReport,
    operator_targets: list[Any],
    *,
    default_engine: str,
    prefix: str = "operator_optimization",
) -> None:
    if not operator_targets:
        report.add(
            f"{prefix}.targets",
            False,
            "operator optimization is enabled but no targets are configured",
            level="error",
        )
        return
    report.add(
        f"{prefix}.targets",
        True,
        "operator optimization targets configured",
        count=len(operator_targets),
        default_engine=default_engine,
    )
    torch_compile_available = hasattr(torch, "compile")
    report.add(
        f"{prefix}.torch_compile",
        torch_compile_available,
        "torch.compile available"
        if torch_compile_available
        else "torch.compile unavailable",
        torch_version=torch.__version__,
    )
    target_engines = {
        getattr(target, "engine", None) or default_engine for target in operator_targets
    }
    if target_engines & {
        "triton",
        "tilelang",
        "cutile",
        "cutlass",
        "cute_dsl",
        "custom_cuda",
    }:
        _check_cuda(report, f"{prefix}.hardware.cuda")
    for package_engine in ("triton", "tilelang", "cutlass"):
        if package_engine in target_engines:
            _check_dependency(report, package_engine)
    if "cutile" in target_engines:
        available = _cutile_available()
        report.add(
            "dependency.cuda.tile",
            available,
            "available" if available else "missing optional dependency",
            package="cuda.tile",
            legacy_package="cutile",
        )
    if "cute_dsl" in target_engines:
        _check_dependency(report, "cutlass.cute")
    for index, target in enumerate(operator_targets):
        target_prefix = f"{prefix}.targets.{index}"
        target_engine = getattr(target, "engine", None) or default_engine
        capability = describe_operator_engine_capability(
            target_engine,
            torch_compile_available=torch_compile_available,
        )
        fallback_policy = str(getattr(target, "fallback_policy", "prefer_fallback"))
        fallback_policy_ok = fallback_policy in {"strict", "prefer_fallback"}
        report.add(
            f"{target_prefix}.capability",
            capability.available or capability.status == "planned",
            "operator optimization engine capability described",
            level="info"
            if capability.available or capability.status == "available"
            else "warning",
            target_name=getattr(target, "name", None),
            module_path=getattr(target, "target", None),
            **capability.to_dict(),
        )
        report.add(
            f"{target_prefix}.fallback_policy",
            fallback_policy_ok,
            "operator fallback policy recorded"
            if fallback_policy_ok
            else "operator fallback_policy must be 'strict' or 'prefer_fallback'",
            level="info" if fallback_policy_ok else "error",
            target_name=getattr(target, "name", None),
            module_path=getattr(target, "target", None),
            fallback_policy=fallback_policy,
        )
        if target_engine == "tilelang" and not _package_available("tilelang"):
            report.add(
                f"{target_prefix}.tilelang.runtime",
                True,
                "tilelang package is not importable; built-in executor will be limited to reference fallback",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "cutile" and not capability.available:
            report.add(
                f"{target_prefix}.cutile.runtime",
                False,
                "cutile engine is configured but cuda.tile is not importable",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "cutlass" and not capability.available:
            report.add(
                f"{target_prefix}.cutlass.runtime",
                False,
                "cutlass engine is configured but cutlass is not importable",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "cute_dsl" and not capability.available:
            report.add(
                f"{target_prefix}.cute_dsl.runtime",
                False,
                "cute_dsl engine is configured but cutlass.cute is not importable",
                level="warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
            )
        if target_engine == "tilelang":
            tilelang_config = getattr(target, "tilelang")
            tilelang_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target": tilelang_config.target,
                "target_arch": tilelang_config.target_arch,
                "cache_dir": tilelang_config.cache_dir,
                "threads": tilelang_config.threads,
                "num_stages": tilelang_config.num_stages,
                "pass_configs": dict(tilelang_config.pass_configs),
            }
            tilelang_metadata.update(_module_metadata("tilelang"))
            report.add(
                f"{target_prefix}.tilelang.config",
                True,
                "tilelang compile configuration recorded",
                **tilelang_metadata,
            )
        if target_engine == "cutile":
            cutile_config = getattr(target, "cutile")
            cutile_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target": cutile_config.target,
                "target_arch": cutile_config.target_arch,
                "cache_dir": cutile_config.cache_dir,
                "threads": cutile_config.threads,
                "pass_configs": dict(cutile_config.pass_configs),
            }
            cutile_metadata.update(_cutile_metadata())
            report.add(
                f"{target_prefix}.cutile.config",
                True,
                "cutile compile configuration recorded",
                **cutile_metadata,
            )
        if target_engine == "cutlass":
            cutlass_config = getattr(target, "cutlass")
            cutlass_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target_arch": cutlass_config.target_arch,
                "cache_dir": cutlass_config.cache_dir,
                "tile_shape": list(cutlass_config.tile_shape),
                "cluster_shape": (
                    list(cutlass_config.cluster_shape)
                    if cutlass_config.cluster_shape is not None
                    else None
                ),
                "pass_configs": dict(cutlass_config.pass_configs),
            }
            cutlass_metadata.update(_module_metadata("cutlass"))
            report.add(
                f"{target_prefix}.cutlass.config",
                True,
                "cutlass compile configuration recorded",
                **cutlass_metadata,
            )
        if target_engine == "cute_dsl":
            cute_dsl_config = getattr(target, "cute_dsl")
            cute_dsl_metadata = {
                "target_name": getattr(target, "name", None),
                "module_path": getattr(target, "target", None),
                "target_arch": cute_dsl_config.target_arch,
                "cache_dir": cute_dsl_config.cache_dir,
                "tile_shape": list(cute_dsl_config.tile_shape),
                "cluster_shape": (
                    list(cute_dsl_config.cluster_shape)
                    if cute_dsl_config.cluster_shape is not None
                    else None
                ),
                "pass_configs": dict(cute_dsl_config.pass_configs),
            }
            cute_dsl_metadata.update(_module_metadata("cutlass.cute"))
            report.add(
                f"{target_prefix}.cute_dsl.config",
                True,
                "cute_dsl compile configuration recorded",
                **cute_dsl_metadata,
            )
        if target_engine == "custom_cuda":
            extension = describe_custom_cuda_extension_capability()
            report.add(
                f"{target_prefix}.custom_cuda.extension",
                extension.available,
                "custom CUDA extension capability described",
                level="info" if extension.available else "warning",
                target_name=getattr(target, "name", None),
                module_path=getattr(target, "target", None),
                **extension.to_dict(),
            )


def _check_prune_config(
    report: PreflightReport,
    prune: PruneConfig,
    *,
    device: str | None,
    task_type: str,
    prefix: str,
) -> None:
    if not prune.enabled:
        return
    if prune.method == "nm_structured":
        pattern_raw = prune.selection.get("pattern") or prune.params.get("pattern")
        if isinstance(pattern_raw, (list, tuple)) and len(pattern_raw) == 2:
            pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
            capability = describe_prune_runtime_capability(
                method="nm_structured",
                device=device,
                pattern=pattern,
            ).to_dict()
            report.add(
                f"{prefix}.nm_backend",
                bool(capability["supported"]),
                str(capability["reason"]),
                pattern=list(pattern),
                runtime=capability["runtime"],
                speedup_verified=capability["speedup_verified"],
                pattern_present=capability["pattern_present"],
                level="info" if capability["supported"] else "warning",
            )
        else:
            report.add(
                f"{prefix}.nm_backend",
                False,
                "N:M structured pruning requires selection.pattern=[N, M]",
                level="error",
            )
    if prune.method == "block_sparse":
        block_shape_raw = prune.selection.get("block_shape") or prune.params.get(
            "block_shape"
        )
        if isinstance(block_shape_raw, (list, tuple)) and len(block_shape_raw) == 2:
            block_shape = (int(block_shape_raw[0]), int(block_shape_raw[1]))
            capability = describe_prune_runtime_capability(
                method="block_sparse",
                device=device,
                block_shape=block_shape,
            ).to_dict()
            report.add(
                f"{prefix}.block_sparse_backend",
                bool(capability["supported"]),
                str(capability["reason"]),
                block_shape=list(block_shape),
                runtime=capability["runtime"],
                speedup_verified=capability["speedup_verified"],
                pattern_present=capability["pattern_present"],
                level="info" if capability["supported"] else "warning",
            )
        else:
            report.add(
                f"{prefix}.block_sparse_backend",
                False,
                "block_sparse pruning requires selection.block_shape=[rows, cols]",
                level="error",
            )
    metadata = {
        "method": prune.method,
        "target_sparsity": prune.target_sparsity,
        "granularity": prune.granularity,
        "scope": prune.scope,
    }
    if task_type != "detection":
        return
    if prune.method == "global_l1_unstructured":
        report.add(
            f"{prefix}.detection_safety",
            True,
            "unstructured detection pruning records sparsity only; speedup is not claimed",
            **metadata,
        )
        return
    if prune.method != "structured":
        report.add(
            f"{prefix}.detection_safety",
            True,
            "detection pruning method does not rewrite detection head topology",
            **metadata,
        )
        return
    report.add(
        f"{prefix}.detection_safety",
        True,
        (
            "structured detection pruning is guarded; residual/CSP/C2f/SPPF/detect head "
            "dependency rewrite is not implemented"
        ),
        level="warning",
        support="unsupported_in_builtin_executor",
        **metadata,
    )


def _check_export_targets(
    report: PreflightReport,
    targets: list[Any],
    *,
    prefix: str = "export.targets",
) -> None:
    for index, target in enumerate(targets):
        target_format = getattr(target, "format", None)
        target_params = dict(getattr(target, "params", {}) or {})
        target_prefix = f"{prefix}.{index}.{target_format}"
        if target_format in {"torch_export", "torchscript"}:
            report.add(target_prefix, True, "built-in PyTorch export target")
        elif target_format == "onnx":
            _check_dependency(report, "onnx")
            onnx_config = getattr(target, "onnx", None)
            if onnx_config is None:
                report.add(
                    f"{target_prefix}.onnx",
                    False,
                    "ONNX export target requires typed onnx configuration",
                    level="error",
                )
                continue
            if onnx_config.runtime_diff:
                _check_dependency(report, "onnxruntime")
            optimization = onnx_config.optimization
            if optimization.enabled:
                backend = optimization.backend
                if backend == "onnxruntime":
                    _check_dependency(report, "onnxruntime")
                else:
                    report.add(
                        f"{target_prefix}.onnx_optimization",
                        False,
                        f"unsupported ONNX optimization backend: {backend}",
                        level="error",
                    )
        elif target_format == "tensorrt":
            tensorrt = target.tensorrt
            _check_optional_executable(
                report,
                tensorrt.trtexec_path,
                f"{target_prefix}.trtexec",
                dry_run=tensorrt.dry_run,
            )
            for plugin_index, plugin_path in enumerate(tensorrt.plugin_libraries):
                path = Path(plugin_path)
                plugin_validation = validate_tensorrt_plugin_libraries(
                    [path],
                    validate_loadability=tensorrt.validate_plugin_libraries_loadable,
                )
                plugin_check = plugin_validation.plugin_libraries[0]
                report.add(
                    f"{target_prefix}.plugin_libraries.{plugin_index}",
                    plugin_check.exists,
                    "TensorRT plugin library found"
                    if plugin_check.exists
                    else "TensorRT plugin library missing",
                    level="info" if plugin_check.exists else "warning",
                    path=plugin_check.path,
                    validation=plugin_check.to_dict(),
                )
                if tensorrt.validate_plugin_libraries_loadable:
                    if plugin_check.loadable is True:
                        report.add(
                            f"{target_prefix}.plugin_libraries.{plugin_index}.loadable",
                            True,
                            "TensorRT plugin library loaded with ctypes RTLD_GLOBAL",
                            level="info",
                            path=plugin_check.path,
                            loaded_plugin_libraries=plugin_check.loaded_plugin_libraries,
                            validation=plugin_check.to_dict(),
                        )
                    else:
                        report.add(
                            f"{target_prefix}.plugin_libraries.{plugin_index}.loadable",
                            False,
                            f"TensorRT plugin library failed to load: {plugin_check.error}",
                            level="error",
                            path=plugin_check.path,
                            validation=plugin_check.to_dict(),
                        )
        elif target_format == "openvino":
            openvino = target.openvino
            _check_optional_dependency(
                report,
                "openvino",
                "dependency.openvino",
                dry_run=openvino.dry_run,
            )
        elif target_format == "executorch":
            _check_optional_dependency(
                report,
                "executorch",
                "dependency.executorch",
                dry_run=target.executorch.dry_run,
            )
        elif target_format == "ncnn":
            ncnn = target.ncnn
            if ncnn.converter == "pnnx":
                _check_optional_executable(
                    report,
                    ncnn.pnnx_path,
                    f"{target_prefix}.pnnx",
                    dry_run=ncnn.dry_run,
                )
            else:
                _check_optional_executable(
                    report,
                    ncnn.onnx2ncnn_path,
                    f"{target_prefix}.onnx2ncnn",
                    dry_run=ncnn.dry_run,
                )
        elif target_format == "mnn":
            mnn = target.mnn
            _check_optional_executable(
                report,
                mnn.converter_path,
                f"{target_prefix}.MNNConvert",
                dry_run=mnn.dry_run,
            )
        else:
            report.add(target_prefix, False, "unsupported export target")


def _check_deploy_runtime_handle(
    report: PreflightReport,
    handle: DeployRuntimeHandleSpec | None,
    *,
    prefix: str,
) -> None:
    if handle is None or not handle.materialize:
        return
    runtime = handle.runtime
    supported = runtime in {"onnxruntime", "tensorrt"}
    report.add(
        f"{prefix}.runtime_handle.runtime",
        supported,
        "runtime handle is configured"
        if supported
        else "unsupported materialized runtime handle",
        level="info" if supported else "error",
        runtime=runtime,
        handle_kind=handle.handle_kind,
    )
    if runtime == "onnxruntime":
        _check_dependency(report, "onnxruntime")
        return
    if runtime != "tensorrt":
        return
    _check_dependency(report, "tensorrt")
    for index, plugin_path in enumerate(handle.tensorrt.plugin_libraries):
        validation = validate_tensorrt_plugin_libraries(
            [plugin_path],
            validate_loadability=True,
        )
        check = validation.plugin_libraries[0]
        report.add(
            f"{prefix}.runtime_handle.tensorrt.plugin_libraries.{index}",
            check.loadable is True,
            "TensorRT runtime plugin library loaded"
            if check.loadable is True
            else f"TensorRT runtime plugin library failed to load: {check.error}",
            level="info" if check.loadable is True else "error",
            path=check.path,
            validation=check.to_dict(),
        )


def preflight_optimization_config(
    config: ConfigInput | OptimizationConfig,
) -> PreflightReport:
    """Run lightweight dependency and target checks for a stage workflow."""

    loaded = (
        config
        if isinstance(config, OptimizationConfig)
        else load_optimization_config(config)
    )
    report = PreflightReport()
    report.add(
        "project.artifact_dir",
        True,
        "artifact directory configured",
        path=str(loaded.project.get("artifact_dir", "")),
    )
    report.add(
        "task.type",
        loaded.task.type in TASK_TYPES,
        "task type configured",
        task_type=loaded.task.type,
    )
    if loaded.task.type == "detection":
        report.add(
            "task.detection_postprocess",
            True,
            "detection postprocess configured",
            **{
                "format": loaded.task.detection_postprocess.format,
                "box_format": loaded.task.detection_postprocess.box_format,
                "score_threshold": loaded.task.detection_postprocess.score_threshold,
                "iou_threshold": loaded.task.detection_postprocess.iou_threshold,
                "max_detections": loaded.task.detection_postprocess.max_detections,
            },
        )
    _check_target(report, "model.target", loaded.model.target)
    _check_model_device(report, loaded.device or loaded.model.device)

    for index, stage in enumerate(loaded.stages):
        spec = ensure_stage_spec(stage)
        prefix = f"stages.{index}.{stage.name}"
        report.add(
            f"{prefix}.kind",
            True,
            "stage kind configured",
            kind=stage.kind,
        )
        if isinstance(spec, QuantStageSpec):
            _check_quant_config(
                report,
                QuantConfig(enabled=True, **spec.__dict__),
                prefix=prefix,
                cuda_name="hardware.cuda",
            )
        elif isinstance(spec, PruneStageSpec):
            _check_prune_config(
                report,
                PruneConfig(enabled=True, **spec.__dict__),
                device=loaded.device or loaded.model.device,
                task_type=loaded.task.type,
                prefix=prefix,
            )
        elif isinstance(spec, OperatorStageSpec):
            _check_operator_targets(
                report,
                spec.targets,
                default_engine=spec.default_engine,
                prefix=prefix,
            )
        elif isinstance(spec, (ExportStageSpec, DeployStageSpec)):
            _check_export_targets(
                report, list(spec.targets), prefix=f"{prefix}.targets"
            )
            if isinstance(spec, DeployStageSpec):
                _check_deploy_runtime_handle(report, spec.runtime_handle, prefix=prefix)
        elif isinstance(spec, AnalyzeStageSpec):
            report.add(
                f"{prefix}.analysis.metrics",
                bool(spec.metrics),
                "analysis metrics configured"
                if spec.metrics
                else "analysis metrics missing",
                level="info" if spec.metrics else "error",
                metrics=list(spec.metrics),
            )
        elif isinstance(spec, BenchmarkStageSpec):
            report.add(
                f"{prefix}.benchmark",
                True,
                "benchmark override recorded",
                warmup=spec.warmup,
                iterations=spec.iterations,
                measure_memory=spec.measure_memory,
            )
    return report


__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "preflight_optimization_config",
]
