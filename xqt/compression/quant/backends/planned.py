"""Capability-only quantization backend reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from ..artifacts import artifact_key
from ..capability import describe_quant_backend_capability
from ..component import prefix_module_names
from ..selection import selection_policy_metadata
from ..types import QuantizationComponentPlan, QuantizationReport


def execute_planned_method_component(
    root_model: nn.Module | None,
    component: QuantizationComponentPlan,
) -> tuple[nn.Module | None, QuantizationReport, dict[str, Any]]:
    """Return a structured planned report for unsupported method/backend pairs."""

    capability = describe_quant_backend_capability(
        component.backend,
        method=component.method,
        strategy=component.strategy,
        policy=component.policy,
    )
    artifact_name = f"{component.name}_{component.backend}_{component.method or 'planned'}.json"
    artifact_dir = Path("artifacts")
    planned_artifact = artifact_dir / artifact_name
    report = QuantizationReport(
        component_name=component.name,
        backend=component.backend,
        runtime=capability.runtime,
        method=component.method,
        strategy=component.strategy,
        target_path=component.target_path,
        skipped_modules=prefix_module_names(component.skip_quantize, component.target_path),
        high_precision_modules=prefix_module_names(
            component.keep_high_precision,
            component.target_path,
        ),
        nature=capability.nature,
        algorithm_executable=False,
        method_semantics="capability_report_only_no_executable_algorithm",
        artifacts={"planned": str(planned_artifact)},
        metadata={
            "execution_state": "planned",
            "executed": False,
            "algorithm_executable": False,
            "method_semantics": "capability_report_only_no_executable_algorithm",
            "capability": capability.to_dict(),
            "planned_artifact": str(planned_artifact),
            "policy": dict(component.policy),
            "selection_policy": selection_policy_metadata(component),
        },
    )
    artifact_updates = {
        artifact_key("quant_plan", component.name): planned_artifact,
    }
    return root_model, report, artifact_updates


__all__ = ["execute_planned_method_component"]
