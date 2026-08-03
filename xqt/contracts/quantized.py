"""Quantized model artifact contracts shared by XQT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from xqt.contracts.module import coerce_composite_precision_gemm_spec

from .compute import ComputeConfig, compute_config_from_mapping, compute_config_to_dict
from .runtime import _json_safe_contract_value
from .runtime_quant import (
    RuntimeQuantContract,
    attach_runtime_quant_contract,
    extract_runtime_quant_contract,
)


def _resolve_module_contract(
    model: Any,
    module_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if module_contract is not None:
        return dict(module_contract)
    raw = getattr(model, "_xqt_module_contract", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    return None


def _component_name_from_metrics(metrics: Mapping[str, Any], *, fallback: str) -> str:
    components = metrics.get("components", [])
    if isinstance(components, list):
        for item in components:
            if isinstance(item, Mapping) and item.get("component_name") is not None:
                return str(item["component_name"])
    return fallback


def _resolve_composite_quant_artifacts(
    metrics: Mapping[str, Any],
    *,
    module_contract: Mapping[str, Any] | None,
    backend: str,
    fallback_component_name: str,
) -> list[dict[str, Any]]:
    raw_artifacts = metrics.get("composite_quant_artifacts", [])
    if isinstance(raw_artifacts, list):
        items = [dict(item) for item in raw_artifacts if isinstance(item, Mapping)]
        if items:
            return items
    raw_single = metrics.get("composite_quant_artifact")
    if isinstance(raw_single, Mapping):
        return [dict(raw_single)]
    if not isinstance(module_contract, Mapping):
        return []
    raw_policy = module_contract.get("policy")
    if not isinstance(raw_policy, Mapping):
        return []
    spec = coerce_composite_precision_gemm_spec(raw_policy.get("composite_gemm"))
    if spec is None:
        return []
    component_name = _component_name_from_metrics(
        metrics,
        fallback=fallback_component_name,
    )
    runtime_plan = spec.resolve_runtime_plan(backend=backend)
    branches = []
    for branch in (spec.selected_branch, spec.residual_branch):
        branches.append(
            {
                "name": branch.name,
                "format": branch.format,
                "weight_format": branch.weight_format or branch.format,
                "scale_format": branch.scale_format,
                "weight_artifact": f"{component_name}:{branch.name}:weight",
                "scale_artifact": (
                    f"{component_name}:{branch.name}:scale"
                    if branch.scale_format is not None
                    else None
                ),
            }
        )
    return [
        {
            "component_name": component_name,
            "backend": backend,
            "composite_precision": True,
            "requested_mode": str(runtime_plan["requested_mode"]),
            "actual_mode": str(runtime_plan["actual_mode"]),
            "partition_group_count": int(runtime_plan["partition_group_count"]),
            "residual_group_count": int(runtime_plan["residual_group_count"]),
            "partition_map": [
                dict(item) for item in runtime_plan.get("partition_map", []) or []
            ],
            "branch_formats": {
                str(key): str(value)
                for key, value in dict(runtime_plan.get("branch_formats", {})).items()
            },
            "accumulation_dtype": runtime_plan.get("accumulation_dtype"),
            "kernel_count": runtime_plan.get("kernel_count"),
            "workspace_bytes": runtime_plan.get("workspace_bytes"),
            "fallback_reason": runtime_plan.get("fallback_reason"),
            "branches": branches,
        }
    ]


@dataclass(kw_only=True)
class QuantizedModel:
    """Semantic result of a model-side quantization algorithm.

    Infer handoff uses ``model`` + optional ``compute_config`` only.
    ``backend`` / ``method`` / ``strategy`` remain for quant lineage / reports.
    """

    model: Any
    backend: str = "unknown"
    method: str | None = None
    strategy: str | None = None
    quantized_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    compute_config: ComputeConfig | dict[str, Any] | None = None

    def resolve_compute_config(self) -> ComputeConfig | None:
        """Return typed compute_config if present."""
        if self.compute_config is None:
            raw = self.metadata.get("compute_config")
            if isinstance(raw, Mapping):
                return compute_config_from_mapping(raw)
            return None
        if isinstance(self.compute_config, ComputeConfig):
            return self.compute_config
        if isinstance(self.compute_config, Mapping):
            return compute_config_from_mapping(self.compute_config)
        return None

    def resolve_runtime_quant_contract(self) -> RuntimeQuantContract | None:
        """Return the internal runtime quant contract if attached to metadata."""

        return extract_runtime_quant_contract(self.metadata)

    def with_runtime_quant_contract(
        self,
        contract: RuntimeQuantContract,
    ) -> "QuantizedModel":
        """Return a copy with ``contract`` serialized into metadata (fact source)."""

        return QuantizedModel(
            model=self.model,
            backend=self.backend,
            method=self.method,
            strategy=self.strategy,
            quantized_modules=list(self.quantized_modules),
            metadata=attach_runtime_quant_contract(self.metadata, contract),
            compute_config=self.compute_config,
        )

    def infer_handoff(self) -> dict[str, Any]:
        """Infer-facing view: model + compute_config only (no method identity)."""
        config = self.resolve_compute_config()
        return {
            "model": self.model,
            "compute_config": None if config is None else config.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without serializing the model object."""

        model_type: str | None
        if self.model is None:
            model_type = None
        else:
            model_type = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        payload = {
            "model_type": model_type,
            "backend": self.backend,
            "method": self.method,
            "strategy": self.strategy,
            "quantized_module_count": len(self.quantized_modules),
            "quantized_modules": list(self.quantized_modules),
            "metadata": _json_safe_contract_value(self.metadata),
        }
        config_dict = compute_config_to_dict(self.resolve_compute_config())
        if config_dict is not None:
            payload["compute_config"] = config_dict
        return payload


@dataclass(kw_only=True)
class QuantizedModelPayload(QuantizedModel):
    """Workflow provenance and artifacts for a quantized model contract."""

    stage_name: str
    source_model_stage: str
    calibration_samples: int | None = None
    calibration_summary: dict[str, Any] | None = None
    components: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    capability: dict[str, Any] | None = None
    algorithm_metadata: dict[str, Any] | None = None
    execution_policies: list[dict[str, Any]] = field(default_factory=list)
    composite_quant_artifacts: list[dict[str, Any]] = field(default_factory=list)
    module_contract: dict[str, Any] | None = None
    # compute_config inherited from QuantizedModel when set on instance
    artifact_kind: str = field(default="quantized_model", init=False)

    @classmethod
    def from_stage_metrics(
        cls,
        *,
        stage_name: str,
        source_model_stage: str,
        model: Any,
        metrics: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, str] | None = None,
        capability: Mapping[str, Any] | None = None,
        module_contract: Mapping[str, Any] | None = None,
    ) -> "QuantizedModelPayload":
        stage_params = dict(params or {})
        components = metrics.get("components", [])
        if not isinstance(components, list):
            components = []
        quantized_modules = metrics.get("quantized_modules", [])
        if not isinstance(quantized_modules, list):
            quantized_modules = []
        calibration_samples = metrics.get("calibration_samples")
        calibration_summary = metrics.get("calibration_summary")
        algorithm_metadata = metrics.get("algorithm_metadata")
        execution_policies = metrics.get("execution_policies", [])
        if not isinstance(execution_policies, list):
            execution_policies = []
        resolved_contract = _resolve_module_contract(model, module_contract)
        backend = str(metrics.get("backend") or stage_params.get("backend") or "unknown")
        raw_compute = metrics.get("compute_config")
        if raw_compute is None:
            raw_compute = stage_params.get("compute_config")
        compute_config = compute_config_from_mapping(
            raw_compute if isinstance(raw_compute, Mapping) else None
        )
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            model=model,
            backend=backend,
            method=str(metrics.get("method") or stage_params.get("method") or "unknown"),
            strategy=str(
                metrics.get("strategy") or stage_params.get("strategy") or "unknown"
            ),
            quantized_modules=[str(item) for item in quantized_modules],
            compute_config=compute_config,
            calibration_samples=(
                calibration_samples if isinstance(calibration_samples, int) else None
            ),
            calibration_summary=(
                dict(calibration_summary)
                if isinstance(calibration_summary, Mapping)
                else None
            ),
            components=[
                dict(item) for item in components if isinstance(item, Mapping)
            ],
            artifacts={str(key): str(value) for key, value in dict(artifacts or {}).items()},
            capability=dict(capability) if isinstance(capability, Mapping) else None,
            algorithm_metadata=(
                dict(algorithm_metadata)
                if isinstance(algorithm_metadata, Mapping)
                else None
            ),
            execution_policies=[
                dict(item) for item in execution_policies if isinstance(item, Mapping)
            ],
            composite_quant_artifacts=_resolve_composite_quant_artifacts(
                metrics,
                module_contract=resolved_contract,
                backend=backend,
                fallback_component_name=stage_name,
            ),
            module_contract=resolved_contract,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without serializing the model object."""

        payload = {
            "artifact_kind": self.artifact_kind,
            "stage_name": self.stage_name,
            "source_model_stage": self.source_model_stage,
            **super().to_dict(),
            "calibration_samples": self.calibration_samples,
            "calibration_summary": _json_safe_contract_value(self.calibration_summary),
            "components": _json_safe_contract_value(self.components),
            "artifacts": dict(self.artifacts),
            "capability": _json_safe_contract_value(self.capability),
            "algorithm_metadata": _json_safe_contract_value(self.algorithm_metadata),
            "execution_policies": _json_safe_contract_value(self.execution_policies),
        }
        if self.composite_quant_artifacts:
            payload["composite_quant_artifacts"] = _json_safe_contract_value(
                self.composite_quant_artifacts
            )
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        config_dict = compute_config_to_dict(self.resolve_compute_config())
        if config_dict is not None:
            payload["compute_config"] = config_dict
        return payload


__all__ = ["QuantizedModel", "QuantizedModelPayload"]
