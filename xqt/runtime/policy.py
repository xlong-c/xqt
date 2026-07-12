"""Execution policy helpers for hybrid inference.

Policy application only mutates runtime precision selectors on already
quantized modules. Quantization algorithms live under ``xqt.quant``.

Types, constants, and normalizers are imported from ``xqt.contracts``.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from torch import nn

from xqt.contracts import (
    SUPPORTED_COMPUTE_PRECISIONS,
    ExecutionPolicyPayload,
    SupportsComputePrecision,
    normalize_compute_precision,
)

from .channel import apply_channel_hybrid_policy

__all__ = [
    "SUPPORTED_COMPUTE_PRECISIONS",
    "SupportsComputePrecision",
    "apply_execution_policy",
    "build_execution_policy_payload",
    "collect_module_precision_map",
    "normalize_compute_precision",
    "precision_overrides_to_map",
    "set_module_compute_precision",
]


def precision_overrides_to_map(
    precision_overrides: Sequence[Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Convert override records into ``module_name -> precision`` mapping."""

    override_map: dict[str, str] = {}
    for item in precision_overrides or []:
        if not isinstance(item, Mapping) or "module" not in item:
            continue
        module_name = str(item["module"])
        precision = normalize_compute_precision(
            str(item.get("precision", "w8a8"))
        )
        override_map[module_name] = precision
    return override_map


def collect_module_precision_map(model: nn.Module) -> dict[str, str]:
    """Collect current compute precision for policy-aware modules."""

    precision_map: dict[str, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, SupportsComputePrecision):
            precision_map[name] = str(module.compute_precision)
    return precision_map


def build_execution_policy_payload(
    *,
    stage_name: str,
    source_model_stage: str,
    policy_kind: str = "mixed_precision",
    runtime: str = "pytorch",
    precision_overrides: Sequence[Mapping[str, Any]] | None = None,
    required_capabilities: Sequence[str] | None = None,
    preferred_engines: Sequence[str] | None = None,
    compute_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    module_count: int | None = None,
) -> ExecutionPolicyPayload:
    """Build a typed execution-policy artifact from override records."""

    overrides = [dict(item) for item in (precision_overrides or [])]
    resolved_count = (
        int(module_count)
        if module_count is not None
        else len({str(item.get("module")) for item in overrides if "module" in item})
    )
    caps = [str(item) for item in (required_capabilities or []) if str(item)]
    preferred = [
        str(item).strip().lower()
        for item in (preferred_engines or [])
        if str(item).strip()
    ]
    config_dict = dict(compute_config) if isinstance(compute_config, Mapping) else None
    if config_dict is not None and not caps:
        raw_modules = config_dict.get("modules", [])
        if isinstance(raw_modules, list):
            for module in raw_modules:
                if not isinstance(module, Mapping):
                    continue
                for cap in module.get("required_capabilities", []) or []:
                    name = str(cap)
                    if name and name not in caps:
                        caps.append(name)
    return ExecutionPolicyPayload(
        stage_name=stage_name,
        source_model_stage=source_model_stage,
        policy_kind=str(policy_kind),
        runtime=str(runtime),
        module_count=resolved_count,
        precision_overrides=overrides,
        required_capabilities=caps,
        preferred_engines=preferred,
        compute_config=config_dict,
        metadata=dict(metadata or {}),
    )


def set_module_compute_precision(module: nn.Module, precision: str) -> None:
    """Set one module's runtime compute precision without re-quantizing."""

    resolved = normalize_compute_precision(precision)
    if isinstance(module, SupportsComputePrecision):
        module.set_compute_precision(resolved)
        return
    setter = getattr(module, "set_compute_precision", None)
    if callable(setter):
        setter(resolved)
        return
    if hasattr(module, "compute_precision"):
        module.compute_precision = resolved
        ensure = getattr(module, "_ensure_int8_compute", None)
        if resolved == "w8a8" and callable(ensure):
            ensure()
        return
    raise TypeError(
        f"module type {type(module).__name__} does not support compute precision"
    )


def apply_execution_policy(
    model: nn.Module,
    *,
    precision_overrides: Sequence[Mapping[str, Any]] | None = None,
    channel_overrides: Sequence[Mapping[str, Any]] | None = None,
    default_precision: str = "w4a4",
    inplace: bool = True,
    policy: ExecutionPolicyPayload | Mapping[str, Any] | None = None,
) -> nn.Module:
    """Apply mixed-precision policy to an already-quantized model.

    This function never quantizes weights. It only switches runtime compute
    precision selectors and optional channel hybrid plans on modules that
    already hold quantized artifacts.
    """

    target_model = model if inplace else copy.deepcopy(model)
    overrides = list(precision_overrides or [])
    channel_plans = list(channel_overrides or [])
    if policy is not None:
        if isinstance(policy, ExecutionPolicyPayload):
            overrides = list(policy.precision_overrides) + overrides
            default_from_meta = policy.metadata.get("default_precision")
            if default_from_meta is not None and precision_overrides is None:
                default_precision = str(default_from_meta)
            raw_channel = policy.metadata.get("channel_overrides", [])
            if isinstance(raw_channel, list) and channel_overrides is None:
                channel_plans = [
                    dict(item) for item in raw_channel if isinstance(item, Mapping)
                ] + channel_plans
        elif isinstance(policy, Mapping):
            raw_overrides = policy.get("precision_overrides", [])
            if isinstance(raw_overrides, list):
                overrides = [
                    dict(item) for item in raw_overrides if isinstance(item, Mapping)
                ] + overrides
            if "default_precision" in policy and precision_overrides is None:
                default_precision = str(policy["default_precision"])
            raw_channel = policy.get("channel_overrides", [])
            if isinstance(raw_channel, list) and channel_overrides is None:
                channel_plans = [
                    dict(item) for item in raw_channel if isinstance(item, Mapping)
                ] + channel_plans
    override_map = precision_overrides_to_map(overrides)
    resolved_default = normalize_compute_precision(default_precision)

    for name, module in list(target_model.named_modules()):
        if not (
            isinstance(module, SupportsComputePrecision)
            or hasattr(module, "compute_precision")
        ):
            continue
        precision = override_map.get(name, resolved_default)
        set_module_compute_precision(module, precision)
    if channel_plans:
        apply_channel_hybrid_policy(
            target_model,
            channel_overrides=channel_plans,
            inplace=True,
        )
    return target_model
