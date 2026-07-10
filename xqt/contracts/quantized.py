"""Quantized model artifact contracts shared by XQT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runtime import _json_safe_contract_value


@dataclass(kw_only=True)
class QuantizedModelPayload:
    """Type-safe payload describing a quantized model-side stage."""

    stage_name: str
    source_model_stage: str
    model: Any
    backend: str
    method: str
    strategy: str
    quantized_module_count: int = 0
    quantized_modules: list[str] = field(default_factory=list)
    calibration_samples: int | None = None
    calibration_summary: dict[str, Any] | None = None
    components: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    capability: dict[str, Any] | None = None
    artifact_kind: str = field(default="quantized_model", init=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without serializing the model object."""

        model_type: str | None
        if self.model is None:
            model_type = None
        else:
            model_type = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        return {
            "artifact_kind": self.artifact_kind,
            "stage_name": self.stage_name,
            "source_model_stage": self.source_model_stage,
            "model_type": model_type,
            "backend": self.backend,
            "method": self.method,
            "strategy": self.strategy,
            "quantized_module_count": self.quantized_module_count,
            "quantized_modules": list(self.quantized_modules),
            "calibration_samples": self.calibration_samples,
            "calibration_summary": _json_safe_contract_value(self.calibration_summary),
            "components": _json_safe_contract_value(self.components),
            "artifacts": dict(self.artifacts),
            "capability": _json_safe_contract_value(self.capability),
        }


__all__ = ["QuantizedModelPayload"]
