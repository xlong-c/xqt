"""Execution policy helpers for hybrid inference.

Policy application only mutates runtime precision selectors on already
quantized modules. Quantization algorithms live under ``xqt.quant``.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from torch import nn

from xqt.contracts import ExecutionPolicyPayload

from .channel import apply_channel_hybrid_policy

SUPPORTED_COMPUTE_PRECISIONS: frozenset[str] = frozenset(
    {"w4a4", "w4a16", "w8a8", "bf16"}
)


@runtime_checkable
class SupportsComputePrecision(Protocol):
    """Minimal runtime contract for mixed-precision quantized modules."""

    compute_precision: str

    def set_compute_precision(self, precision: str) -> None:
        """Switch runtime compute precision in place."""


def normalize_compute_precision(precision: str) -> str:
    """Canonicalize one compute-precision name."""

    normalized = str(precision).strip().lower()
    aliases = {
        "fp16": "bf16",
        "float16": "bf16",
        "bfloat16": "bf16",
        "int4": "w4a4",
        "w4": "w4a16",
        "weight_only_4bit": "w4a16",
        "int8": "w8a8",
        "w8": "w8a8",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in SUPPORTED_COMPUTE_PRECISIONS:
        allowed = ", ".join(sorted(SUPPORTED_COMPUTE_PRECISIONS))
        raise ValueError(
            f"compute_precision must be one of {allowed}; got {precision!r}"
        )
    return resolved


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
    return ExecutionPolicyPayload(
        stage_name=stage_name,
        source_model_stage=source_model_stage,
        policy_kind=str(policy_kind),
        runtime=str(runtime),
        module_count=resolved_count,
        precision_overrides=overrides,
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
