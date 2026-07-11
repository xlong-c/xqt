"""Quantized model artifact contracts shared by XQT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            model=model,
            backend=str(
                metrics.get("backend") or stage_params.get("backend") or "unknown"
            ),
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
            module_contract=_resolve_module_contract(model, module_contract),
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
        }
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        return payload


__all__ = ["QuantizedModel", "QuantizedModelPayload"]
