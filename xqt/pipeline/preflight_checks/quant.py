"""Preflight checks for quantization stages."""

from __future__ import annotations

from typing import Any

from xqt.core.schema import QuantComponentPolicyConfig, QuantConfig
from xqt.quant.capability import describe_quant_backend_capability

from ._base import PreflightReport, _check_dependency
from .model import _check_cuda


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
