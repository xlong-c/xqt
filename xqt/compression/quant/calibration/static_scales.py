"""Static activation scale resolution for scheme.activation_mode='static'."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext

from ..strategy import resolve_scheme
from ..types import QuantScheme, QuantizationComponentPlan
from .scale_artifact import (
    ActivationScaleArtifact,
    activation_scales_to_mapping,
    calibrate_activation_scales,
)


def resolve_int8_scheme(
    component: QuantizationComponentPlan,
    effective_policy: Mapping[str, Any],
) -> QuantScheme | None:
    """Prefer the plan-time scheme; fall back to strategy+policy resolution."""

    if component.scheme is not None:
        return component.scheme
    strategy = component.strategy or effective_policy.get("strategy")
    return resolve_scheme(strategy, effective_policy)


def coerce_provided_activation_scales(
    raw: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor | float]:
    """Accept tensors, floats, or ActivationScaleArtifact-shaped payloads."""

    if raw is None:
        return {}
    scales: dict[str, torch.Tensor | float] = {}
    for name, value in raw.items():
        path = str(name)
        if isinstance(value, ActivationScaleArtifact):
            scales[path] = value.scale
            continue
        if isinstance(value, Mapping) and "scale" in value:
            scales[path] = torch.as_tensor(
                value["scale"], dtype=torch.float32
            ).reshape(())
            continue
        if isinstance(value, (int, float)):
            scales[path] = float(value)
            continue
        if isinstance(value, torch.Tensor):
            scales[path] = value
            continue
        raise TypeError(
            "activation_scales values must be float, Tensor, "
            "ActivationScaleArtifact, or a mapping with 'scale'; got "
            f"{type(value).__name__} for {path!r}"
        )
    return scales


def resolve_static_activation_scales(
    context: XQTContext,
    target_model: nn.Module,
    component: QuantizationComponentPlan,
    scheme: QuantScheme,
    *,
    eps: float,
) -> tuple[dict[str, torch.Tensor | float], dict[str, Any]]:
    """Resolve static activation scales from policy or live calibration.

    ``scheme.activation_mode="static"`` is a hard requirement: missing scales
    and missing calibration inputs raise. Returns (scale mapping, lineage dict).
    """

    provided = coerce_provided_activation_scales(
        component.policy.get("activation_scales")
        if isinstance(component.policy.get("activation_scales"), Mapping)
        else None
    )
    if provided:
        lineage = {
            "source": "policy.activation_scales",
            "observer": str(component.policy.get("activation_observer", "provided")),
            "module_count": len(provided),
            "scales": {
                name: float(
                    torch.as_tensor(scale, dtype=torch.float32).reshape(()).item()
                )
                for name, scale in provided.items()
            },
        }
        return provided, lineage

    calibration_inputs = context.calibration_inputs
    if calibration_inputs is None:
        calibration_inputs = context.example_inputs
    if calibration_inputs is None:
        raise XQTBackendError(
            "static activation INT8 MMA requires either policy.activation_scales "
            "or context.calibration_inputs / example_inputs to produce "
            "ActivationScaleArtifact"
        )

    if isinstance(calibration_inputs, torch.Tensor) or not isinstance(
        calibration_inputs, (list, tuple)
    ):
        batches: list[Any] = [calibration_inputs]
    else:
        batches = list(calibration_inputs)
    if not batches:
        raise XQTBackendError(
            "static activation INT8 MMA received empty calibration inputs"
        )

    sample_limit = component.policy.get("sample_limit")
    if sample_limit is not None:
        batches = batches[: int(sample_limit)]

    observer = str(component.policy.get("activation_observer", "minmax"))
    artifacts = calibrate_activation_scales(
        target_model,
        batches,
        scheme,
        policy=None,
        observer=observer,
        eps=float(eps),
    )
    scales = activation_scales_to_mapping(artifacts)
    lineage = {
        "source": "calibrate_activation_scales",
        "observer": observer,
        "module_count": len(artifacts),
        "num_batches": len(batches),
        "scales": {
            path: artifact.to_dict() for path, artifact in artifacts.items()
        },
    }
    return scales, lineage


def force_static_scheme(scheme: QuantScheme | None) -> QuantScheme:
    """Return a static INT8 activation scheme, preserving weight fields when present."""

    if scheme is None:
        return QuantScheme(
            weight_dtype="int8",
            weight_granularity="per_channel",
            activation_dtype="int8",
            activation_mode="static",
        )
    if scheme.activation_mode == "static":
        return scheme
    return QuantScheme(
        weight_dtype=scheme.weight_dtype,
        weight_granularity=scheme.weight_granularity,
        group_size=scheme.group_size,
        activation_dtype=scheme.activation_dtype or "int8",
        activation_mode="static",
        sym=scheme.sym,
    )


__all__ = [
    "coerce_provided_activation_scales",
    "force_static_scheme",
    "resolve_int8_scheme",
    "resolve_static_activation_scales",
]
