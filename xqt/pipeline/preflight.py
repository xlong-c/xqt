"""Preflight checks for XQT recipes."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from xqt.core.config import ConfigInput, load_xqt_config
from xqt.core.imports import resolve_target
from xqt.core.schema import (
    QuantComponentPolicyConfig,
    QuantConfig,
    TASK_TYPES,
    XQTConfig,
)
from xqt.operator_opt.capability import describe_operator_backend_capability
from xqt.operator_opt.cuda_extension import describe_custom_cuda_extension_capability
from xqt.prune import describe_prune_runtime_capability
from xqt.quant.capability import describe_quant_backend_capability


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
    return importlib.util.find_spec(package_name) is not None


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
    if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
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
    strategy: str | None,
    policy: dict[str, Any],
    component_name: str | None = None,
) -> None:
    capability = describe_quant_backend_capability(
        backend,
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


def _check_qdq_calibration(report: PreflightReport, loaded: XQTConfig) -> None:
    _check_qdq_data_source(
        report,
        loaded,
        name="data.calibration",
        calibration_split=loaded.compression.quant.calibration_split,
        validation_split=loaded.compression.quant.validation_split,
    )


def _check_qdq_data_source(
    report: PreflightReport,
    loaded: XQTConfig,
    *,
    name: str,
    calibration_split: str | None,
    validation_split: str | None,
    component_name: str | None = None,
) -> None:
    calibration_name = calibration_split or "calibration"
    calibration = getattr(loaded.data, calibration_name, None)
    metadata: dict[str, Any] = {
        "calibration_split": calibration_name,
    }
    if validation_split is not None:
        metadata["validation_split"] = validation_split
    if component_name is not None:
        metadata["component"] = component_name
    if calibration is not None:
        report.add(
            name,
            True,
            "calibration data configured",
            **metadata,
            source="calibration",
        )
        return
    report.add(
        name,
        False,
        "explicit calibration data is required for ONNX QDQ",
        level="error",
        **metadata,
        missing=[calibration_name],
    )


def _check_quant_component_policy(
    report: PreflightReport,
    loaded: XQTConfig,
    quant_config: QuantConfig,
    component: QuantComponentPolicyConfig,
) -> None:
    prefix = f"compression.quant.component_policies.{component.name}"
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
        strategy=effective_strategy,
        policy=effective_policy,
        component_name=component.name,
    )
    if backend == "torchao":
        _check_dependency(report, "torchao")
        if describe_quant_backend_capability(
            backend,
            strategy=effective_strategy,
            policy=effective_policy,
        ).requires_cuda:
            _check_cuda(report, f"{prefix}.hardware.cuda")
    if backend == "onnxruntime_qdq":
        _check_dependency(report, "onnxruntime")
        _check_qdq_data_source(
            report,
            loaded,
            name=f"{prefix}.data_source",
            calibration_split=component.calibration_split or quant_config.calibration_split,
            validation_split=component.validation_split or quant_config.validation_split,
            component_name=component.name,
        )


def _check_quant_runtime_mix(report: PreflightReport, quant_config: QuantConfig) -> None:
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
            "compression.quant.runtime_mix",
            True,
            "single quantization runtime configured",
            backends=unique_backends,
        )
        return
    report.add(
        "compression.quant.runtime_mix",
        True,
        "multiple quantization runtimes configured",
        level="warning",
        backends=unique_backends,
    )


def _check_operator_optimization(report: PreflightReport, loaded: XQTConfig) -> None:
    operator_config = loaded.operator_optimization
    if not operator_config.enabled:
        return
    if not operator_config.targets:
        report.add(
            "operator_optimization.targets",
            False,
            "operator optimization is enabled but no targets are configured",
            level="error",
        )
        return
    report.add(
        "operator_optimization.targets",
        True,
        "operator optimization targets configured",
        count=len(operator_config.targets),
        default_backend=operator_config.default_backend,
    )
    torch_compile_available = hasattr(torch, "compile")
    report.add(
        "operator_optimization.torch_compile",
        torch_compile_available,
        "torch.compile available" if torch_compile_available else "torch.compile unavailable",
        torch_version=torch.__version__,
    )
    target_backends = {
        target.backend or operator_config.default_backend
        for target in operator_config.targets
    }
    if target_backends & {"triton", "tilelang", "cutile", "cutlass", "custom_cuda"}:
        _check_cuda(report, "operator_optimization.hardware.cuda")
    for package_backend in ("triton", "tilelang", "cutile", "cutlass"):
        if package_backend in target_backends:
            _check_dependency(report, package_backend)
    for index, target in enumerate(operator_config.targets):
        prefix = f"operator_optimization.targets.{index}"
        capability = describe_operator_backend_capability(
            target.backend or operator_config.default_backend,
            torch_compile_available=torch_compile_available,
        )
        report.add(
            f"{prefix}.capability",
            capability.available or capability.status == "planned",
            "operator optimization backend capability described",
            level="info"
            if capability.available or capability.status == "available"
            else "warning",
            target_name=target.name,
            module_path=target.target,
            **capability.to_dict(),
        )
        if target.backend == "tilelang" and not capability.available:
            report.add(
                f"{prefix}.tilelang.runtime",
                False,
                "tilelang backend is configured but tilelang is not importable",
                level="warning",
                target_name=target.name,
                module_path=target.target,
            )
        if target.backend == "cutile" and not capability.available:
            report.add(
                f"{prefix}.cutile.runtime",
                False,
                "cutile backend is configured but cutile is not importable",
                level="warning",
                target_name=target.name,
                module_path=target.target,
            )
        if target.backend == "cutlass" and not capability.available:
            report.add(
                f"{prefix}.cutlass.runtime",
                False,
                "cutlass backend is configured but cutlass is not importable",
                level="warning",
                target_name=target.name,
                module_path=target.target,
            )
        if target.backend == "tilelang":
            tilelang_metadata = {
                "target_name": target.name,
                "module_path": target.target,
                "target": target.tilelang.target,
                "target_arch": target.tilelang.target_arch,
                "cache_dir": target.tilelang.cache_dir,
                "threads": target.tilelang.threads,
                "num_stages": target.tilelang.num_stages,
                "pass_configs": dict(target.tilelang.pass_configs),
            }
            tilelang_metadata.update(_module_metadata("tilelang"))
            report.add(
                f"{prefix}.tilelang.config",
                True,
                "tilelang compile configuration recorded",
                **tilelang_metadata,
            )
        if target.backend == "cutile":
            cutile_metadata = {
                "target_name": target.name,
                "module_path": target.target,
                "target": target.cutile.target,
                "target_arch": target.cutile.target_arch,
                "cache_dir": target.cutile.cache_dir,
                "threads": target.cutile.threads,
                "pass_configs": dict(target.cutile.pass_configs),
            }
            cutile_metadata.update(_module_metadata("cutile"))
            report.add(
                f"{prefix}.cutile.config",
                True,
                "cutile compile configuration recorded",
                **cutile_metadata,
            )
        if target.backend == "cutlass":
            cutlass_metadata = {
                "target_name": target.name,
                "module_path": target.target,
                "target_arch": target.cutlass.target_arch,
                "cache_dir": target.cutlass.cache_dir,
                "tile_shape": list(target.cutlass.tile_shape),
                "cluster_shape": (
                    list(target.cutlass.cluster_shape)
                    if target.cutlass.cluster_shape is not None
                    else None
                ),
                "pass_configs": dict(target.cutlass.pass_configs),
            }
            cutlass_metadata.update(_module_metadata("cutlass"))
            report.add(
                f"{prefix}.cutlass.config",
                True,
                "cutlass compile configuration recorded",
                **cutlass_metadata,
            )
        if target.backend == "custom_cuda":
            extension = describe_custom_cuda_extension_capability()
            report.add(
                f"{prefix}.custom_cuda.extension",
                extension.available,
                "custom CUDA extension capability described",
                level="info" if extension.available else "warning",
                target_name=target.name,
                module_path=target.target,
                **extension.to_dict(),
            )


def _check_detection_prune_safety(report: PreflightReport, loaded: XQTConfig) -> None:
    prune = loaded.compression.prune
    if not prune.enabled or loaded.task.type != "detection":
        return
    metadata = {
        "method": prune.method,
        "target_sparsity": prune.target_sparsity,
        "granularity": prune.granularity,
        "scope": prune.scope,
    }
    if prune.method == "global_l1_unstructured":
        report.add(
            "compression.prune.detection_safety",
            True,
            "unstructured detection pruning records sparsity only; speedup is not claimed",
            **metadata,
        )
        return
    if prune.method != "structured":
        report.add(
            "compression.prune.detection_safety",
            True,
            "detection pruning method does not rewrite detection head topology",
            **metadata,
        )
        return
    report.add(
        "compression.prune.detection_safety",
        True,
        "structured detection pruning requires model-specific dependency checks at plan time",
        level="warning",
        **metadata,
    )


def preflight_xqt_config(config: ConfigInput | XQTConfig) -> PreflightReport:
    """Run lightweight dependency and target checks for a recipe."""

    loaded = config if isinstance(config, XQTConfig) else load_xqt_config(config)
    report = PreflightReport()
    report.add(
        "project.artifact_dir",
        True,
        "artifact directory configured",
        path=loaded.project.artifact_dir,
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
    _check_model_device(report, loaded.model.device)

    for split_name in ("train", "validation", "calibration", "prompts"):
        split = getattr(loaded.data, split_name)
        if split is None:
            continue
        if split.target and split.target not in {
            "prompt_file",
            "prompt_list",
            "synthetic_classification",
            "synthetic_detection",
            "hf_text_classification",
            "torchvision_image_classification",
            "xdl_dataset",
            "xdl_detection",
        }:
            _check_target(report, f"data.{split_name}.target", split.target)
        else:
            report.add(
                f"data.{split_name}.target",
                True,
                "built-in data target",
                target=split.target,
            )
        if loaded.task.type == "detection" and split.target in {
            "synthetic_detection",
            "xdl_detection",
            "xdl_dataset",
        }:
            params = dict(getattr(split, "params", {}) or {})
            report.add(
                f"data.{split_name}.detection",
                True,
                "detection data split configured",
                batch_size=getattr(split, "batch_size", None),
                sample_limit=getattr(split, "sample_limit", None),
                params=params,
            )
        if split.root is not None:
            root = Path(split.root).expanduser()
            report.add(
                f"data.{split_name}.root",
                root.exists(),
                "root exists" if root.exists() else "root does not exist",
                path=str(root),
            )

    quant = loaded.compression.quant
    if quant.enabled:
        _check_quant_runtime_mix(report, quant)
        _check_quant_backend_capability(
            report,
            "compression.quant.capability",
            quant.backend,
            strategy=quant.strategy,
            policy=quant.policy,
        )
        if quant.backend == "torchao":
            _check_dependency(report, "torchao")
            if describe_quant_backend_capability(
                quant.backend,
                strategy=quant.strategy,
                policy=quant.policy,
            ).requires_cuda:
                _check_cuda(report, "hardware.cuda")
        if quant.backend == "onnxruntime_qdq":
            _check_dependency(report, "onnxruntime")
            _check_qdq_calibration(report, loaded)
        if quant.component_policies:
            report.add(
                "compression.quant.component_policies",
                True,
                "component-level quantization policies configured",
                count=len(quant.component_policies),
            )
            for component in quant.component_policies:
                _check_quant_component_policy(report, loaded, quant, component)
    _check_operator_optimization(report, loaded)

    prune = loaded.compression.prune
    if prune.enabled and prune.method == "nm_structured":
        pattern_raw = prune.selection.get("pattern") or prune.params.get("pattern")
        if isinstance(pattern_raw, (list, tuple)) and len(pattern_raw) == 2:
            pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
            capability = describe_prune_runtime_capability(
                method="nm_structured",
                device=loaded.model.device,
                pattern=pattern,
            ).to_dict()
            report.add(
                "compression.prune.nm_backend",
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
                "compression.prune.nm_backend",
                False,
                "N:M structured pruning requires selection.pattern=[N, M]",
                level="error",
            )
    if prune.enabled and prune.method == "block_sparse":
        block_shape_raw = prune.selection.get("block_shape") or prune.params.get("block_shape")
        if isinstance(block_shape_raw, (list, tuple)) and len(block_shape_raw) == 2:
            block_shape = (int(block_shape_raw[0]), int(block_shape_raw[1]))
            capability = describe_prune_runtime_capability(
                method="block_sparse",
                device=loaded.model.device,
                block_shape=block_shape,
            ).to_dict()
            report.add(
                "compression.prune.block_sparse_backend",
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
                "compression.prune.block_sparse_backend",
                False,
                "block_sparse pruning requires selection.block_shape=[rows, cols]",
                level="error",
            )
    _check_detection_prune_safety(report, loaded)

    if loaded.model.target == "xqt.distill.build_hf_text_classification_bundle_from_params" or any(
        getattr(getattr(loaded.data, split_name), "target", None) == "hf_text_classification"
        for split_name in ("train", "validation", "calibration")
    ):
        _check_dependency(report, "transformers")
        _check_dependency(report, "datasets")

    for index, target in enumerate(loaded.export.targets):
        prefix = f"export.targets.{index}.{target.format}"
        if target.format in {"torch_export", "torchscript"}:
            report.add(prefix, True, "built-in PyTorch export target")
        elif target.format == "onnx":
            _check_dependency(report, "onnx")
            if bool(target.params.get("runtime_diff", True)):
                _check_dependency(report, "onnxruntime")
        elif target.format == "tensorrt":
            _check_optional_executable(
                report,
                str(target.params.get("trtexec_path", "trtexec")),
                f"{prefix}.trtexec",
                dry_run=bool(target.params.get("dry_run", False)),
            )
        elif target.format == "openvino":
            _check_optional_dependency(
                report,
                "openvino",
                "dependency.openvino",
                dry_run=bool(target.params.get("dry_run", False)),
            )
        elif target.format == "executorch":
            _check_dependency(report, "executorch")
        elif target.format == "ncnn":
            if target.params.get("converter", "onnx2ncnn") == "pnnx":
                _check_executable(
                    report,
                    str(target.params.get("pnnx_path", "pnnx")),
                    f"{prefix}.pnnx",
                )
            else:
                _check_executable(
                    report,
                    str(target.params.get("onnx2ncnn_path", "onnx2ncnn")),
                    f"{prefix}.onnx2ncnn",
                )
        elif target.format == "mnn":
            _check_executable(
                report,
                str(target.params.get("converter_path", "MNNConvert")),
                f"{prefix}.MNNConvert",
            )
        else:
            report.add(prefix, False, "unsupported export target")
    return report


__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "preflight_xqt_config",
]
