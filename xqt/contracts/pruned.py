"""Pruned model artifact contract shared by XQT workflow stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .runtime import _json_safe_contract_value


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(kw_only=True)
class PrunedModelPayload:
    """Typed model-side payload describing one completed pruning stage."""

    stage_name: str
    source_model_stage: str
    model: Any
    method: str
    granularity: str | None = None
    target_sparsity: float | None = None
    sparsity: float | None = None
    execution_state: str = "unknown"
    applied: bool | None = None
    report: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    capability: dict[str, Any] | None = None
    module_contract: dict[str, Any] | None = None
    artifact_kind: str = field(default="pruned_model", init=False)

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
    ) -> "PrunedModelPayload":
        stage_params = dict(params or {})
        applied = metrics.get("applied")
        execution_state = str(
            metrics.get("execution_state") or ("applied" if metrics else "unknown")
        )
        granularity = metrics.get("granularity")
        if granularity is None:
            granularity = stage_params.get("granularity")
        resolved_contract = (
            dict(module_contract)
            if module_contract is not None
            else (
                dict(raw)
                if isinstance((raw := getattr(model, "_xqt_module_contract", None)), Mapping)
                else None
            )
        )
        return cls(
            stage_name=stage_name,
            source_model_stage=source_model_stage,
            model=model,
            method=str(metrics.get("method") or stage_params.get("method") or "unknown"),
            granularity=str(granularity) if granularity is not None else None,
            target_sparsity=_optional_float(
                metrics.get("target_sparsity", stage_params.get("target_sparsity"))
            ),
            sparsity=_optional_float(
                metrics.get("sparsity", metrics.get("final_sparsity"))
            ),
            execution_state=execution_state,
            applied=applied if isinstance(applied, bool) else execution_state == "applied",
            report=dict(metrics),
            artifacts={str(key): str(value) for key, value in dict(artifacts or {}).items()},
            capability=dict(capability) if isinstance(capability, Mapping) else None,
            module_contract=resolved_contract,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without serializing the model object."""

        model_type: str | None
        if self.model is None:
            model_type = None
        else:
            model_type = f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        payload = {
            "artifact_kind": self.artifact_kind,
            "stage_name": self.stage_name,
            "source_model_stage": self.source_model_stage,
            "model_type": model_type,
            "method": self.method,
            "granularity": self.granularity,
            "target_sparsity": self.target_sparsity,
            "sparsity": self.sparsity,
            "execution_state": self.execution_state,
            "applied": self.applied,
            "report": _json_safe_contract_value(self.report),
            "artifacts": dict(self.artifacts),
            "capability": _json_safe_contract_value(self.capability),
        }
        if self.module_contract is not None:
            payload["module_contract"] = _json_safe_contract_value(self.module_contract)
        return payload


__all__ = ["PrunedModelPayload"]
