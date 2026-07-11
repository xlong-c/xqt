"""Quantized model artifact contracts shared by XQT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from xqt.contracts.module import coerce_composite_precision_gemm_spec
from xqt.quant.types import build_composite_quantization_artifact

from .runtime import _json_safe_contract_value


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
    artifact = build_composite_quantization_artifact(
        spec,
        component_name=_component_name_from_metrics(
            metrics,
            fallback=fallback_component_name,
        ),
        backend=backend,
    )
    return [artifact.to_dict()]


@dataclass(kw_only=True)
class QuantizedModel:
    """Backend-neutral semantic result of a model-side quantization algorithm."""

    model: Any
    backend: str = "unknown"
    method: str | None = None
    strategy: str | None = None
    quantized_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without serializing the model object."""

        model_type: str | None
        if self.model is None:
            model_type = None
        else:
            model_type = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        return {
            "model_type": model_type,
            "backend": self.backend,
            "method": self.method,
            "strategy": self.strategy,
            "quantized_module_count": len(self.quantized_modules),
            "quantized_modules": list(self.quantized_modules),
            "metadata": _json_safe_contract_value(self.metadata),
        }


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
        return payload


__all__ = ["QuantizedModel", "QuantizedModelPayload"]
