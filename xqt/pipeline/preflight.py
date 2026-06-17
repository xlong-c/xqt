"""Preflight checks for XQT recipes."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from xqt.core.config import ConfigInput, load_xqt_config
from xqt.core.imports import resolve_target
from xqt.core.schema import XQTConfig


@dataclass
class PreflightCheck:
    """Single preflight check result."""

    name: str
    passed: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
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
        self.checks.append(
            PreflightCheck(
                name=name,
                passed=passed,
                message=message,
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


def _check_executable(report: PreflightReport, executable: str, name: str) -> None:
    path = shutil.which(executable)
    report.add(
        name,
        path is not None,
        f"found: {path}" if path is not None else "missing optional executable",
        executable=executable,
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


def _quant_policy_requires_cuda(policy: dict[str, Any]) -> bool:
    if bool(policy.get("requires_cuda", False)):
        return True
    strategy = str(policy.get("strategy", policy.get("dtype", ""))).lower()
    return "fp8" in strategy or "float8" in strategy


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
    _check_target(report, "model.target", loaded.model.target)
    _check_model_device(report, loaded.model.device)

    for split_name in ("train", "validation", "calibration", "prompts"):
        split = getattr(loaded.data, split_name)
        if split is None:
            continue
        if split.target and split.target not in {
            "synthetic_classification",
            "hf_text_classification",
            "torchvision_image_classification",
        }:
            _check_target(report, f"data.{split_name}.target", split.target)
        else:
            report.add(
                f"data.{split_name}.target",
                True,
                "built-in data target",
                target=split.target,
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
        if quant.backend == "torchao":
            _check_dependency(report, "torchao")
            if _quant_policy_requires_cuda(quant.policy):
                _check_cuda(report, "hardware.cuda")
        if quant.backend == "onnxruntime_qdq":
            _check_dependency(report, "onnxruntime")

    distill = loaded.compression.distill
    if (
        distill.enabled
        and loaded.model.target
        == "xqt.distill.build_hf_text_classification_bundle_from_params"
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
            _check_executable(
                report,
                str(target.params.get("trtexec_path", "trtexec")),
                f"{prefix}.trtexec",
            )
        elif target.format == "openvino":
            _check_dependency(report, "openvino")
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
