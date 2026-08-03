"""Model-family safety guards for structured pruning.

Structured pruning rewrites real module topology, so before mutating a model
XQT must verify family-specific contracts that generic index validation cannot
see: attention head alignment, MLP pair and gated-MLP dimension symmetry,
Conv2d producer/consumer shape contracts, grouped-conv divisibility, and the
detection "do not touch the head" rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from .report import StructuredPruningAction

_DETECTION_HEAD_PATTERNS = (
    "detect",
    "head",
    "postprocess",
    "classifier",
    "bbox",
    "cls_",
    "output_head",
)


def _get_module(model: nn.Module, name: str) -> nn.Module:
    from .discovery import get_module

    return get_module(model, name)


@dataclass(frozen=True)
class PruneSafetyCheck:
    """One family-specific safety check result."""

    name: str
    family: str
    passed: bool
    blocked_modules: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "passed": self.passed,
            "blocked_modules": list(self.blocked_modules),
            "violations": list(self.violations),
            "notes": list(self.notes),
        }


@dataclass
class PruneSafetyReport:
    """Aggregate safety report for one structured pruning plan."""

    task_type: str | None
    checks: list[PruneSafetyCheck] = field(default_factory=list)

    @property
    def blocked_modules(self) -> list[str]:
        blocked: list[str] = []
        for check in self.checks:
            blocked.extend(check.blocked_modules)
        return sorted(set(blocked))

    @property
    def violations(self) -> list[str]:
        violations: list[str] = []
        for check in self.checks:
            violations.extend(check.violations)
        return violations

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "blocked_modules": self.blocked_modules,
            "violations": self.violations,
        }


def _action_modules(action: StructuredPruningAction) -> list[str]:
    names: list[str] = [action.module_name]
    for key in (
        "gate_proj_name",
        "up_proj_name",
        "down_proj_name",
        "router_name",
        "experts_name",
    ):
        value = action.metadata.get(key)
        if isinstance(value, str):
            names.append(value)
    consumers = action.metadata.get("consumers")
    if isinstance(consumers, list):
        for item in consumers:
            if isinstance(item, dict):
                consumer_name = item.get("name")
                if isinstance(consumer_name, str):
                    names.append(consumer_name)
    return names


def _attention_action_contracts(
    model: nn.Module,
    action: StructuredPruningAction,
) -> list[str]:
    violations: list[str] = []
    attention = _get_module(model, action.module_name)
    num_heads = getattr(attention, "num_heads", None)
    head_dim = getattr(attention, "head_dim", None)
    embed_dim = getattr(attention, "embed_dim", None)
    if not isinstance(num_heads, int) or num_heads <= 0:
        violations.append(f"{action.module_name} has invalid num_heads")
    if not isinstance(head_dim, int) or head_dim <= 0:
        violations.append(f"{action.module_name} has invalid head_dim")
    if isinstance(num_heads, int) and isinstance(head_dim, int):
        kept = len(action.keep_indices)
        if kept <= 0 or kept > num_heads:
            violations.append(
                f"{action.module_name} keeps {kept} heads, expected 1..{num_heads}"
            )
        if kept * head_dim > int(
            getattr(attention, "inner_dim", kept * head_dim)
        ):
            violations.append(
                f"{action.module_name} kept-head features exceed inner dimension"
            )
    variant = str(action.metadata.get("attention_variant", "mha"))
    if variant in {"gqa", "mqa"}:
        num_kv_heads = getattr(attention, "num_kv_heads", None)
        if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
            violations.append(f"{action.module_name} has invalid num_kv_heads")
        elif isinstance(num_heads, int) and num_heads % num_kv_heads != 0:
            violations.append(
                f"{action.module_name} num_heads={num_heads} is not divisible "
                f"by num_kv_heads={num_kv_heads}"
            )
    return violations


def _mlp_action_contracts(
    model: nn.Module,
    action: StructuredPruningAction,
) -> list[str]:
    violations: list[str] = []
    kept = len(action.keep_indices)
    if action.action_type == "mlp_neuron_group":
        fc1 = _get_module(model, action.module_name)
        if action.consumer_name is None:
            violations.append(f"{action.module_name} mlp_neuron_group has no consumer")
            return violations
        fc2 = _get_module(model, action.consumer_name)
        if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
            violations.append(f"{action.module_name} MLP pair is not Linear")
            return violations
        if fc1.out_features != fc2.in_features:
            violations.append(
                f"{action.module_name} MLP pair shape mismatch: "
                f"out={fc1.out_features}, in={fc2.in_features}"
            )
        if kept > fc1.out_features:
            violations.append(
                f"{action.module_name} keeps {kept} neurons, "
                f"larger than out_features={fc1.out_features}"
            )
        return violations

    if action.action_type == "gated_mlp_neuron_group":
        gate_name = action.metadata.get("gate_proj_name")
        up_name = action.metadata.get("up_proj_name")
        down_name = action.metadata.get("down_proj_name")
        if not all(
            isinstance(name, str)
            for name in (gate_name, up_name, down_name)
        ):
            violations.append(
                f"{action.module_name} gated_mlp_neuron_group is missing "
                "gate/up/down projection names"
            )
            return violations
        gate = _get_module(model, str(gate_name))
        up = _get_module(model, str(up_name))
        down = _get_module(model, str(down_name))
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear):
            violations.append(f"{action.module_name} gated MLP gate/up is not Linear")
            return violations
        if not isinstance(down, nn.Linear):
            violations.append(f"{action.module_name} gated MLP down is not Linear")
            return violations
        if gate.out_features != up.out_features or gate.out_features != down.in_features:
            violations.append(
                f"{action.module_name} gated MLP shape mismatch: "
                f"gate={gate.out_features}, up={up.out_features}, "
                f"down_in={down.in_features}"
            )
        if kept > gate.out_features:
            violations.append(
                f"{action.module_name} keeps {kept} neurons, "
                f"larger than gate out_features={gate.out_features}"
            )
        return violations
    return violations


def _conv_action_contracts(
    model: nn.Module,
    action: StructuredPruningAction,
) -> list[str]:
    from .rewrite_ops.linear_conv import validate_conv2d_keep_indices

    violations: list[str] = []
    if action.action_type == "conv_channel_group":
        producer = _get_module(model, action.module_name)
        if not isinstance(producer, nn.Conv2d):
            violations.append(f"{action.module_name} is not a Conv2d producer")
            return violations
        if len(action.keep_indices) > producer.out_channels:
            violations.append(
                f"{action.module_name} keeps {len(action.keep_indices)} channels, "
                f"larger than out_channels={producer.out_channels}"
            )
        try:
            validate_conv2d_keep_indices(producer, action.keep_indices, axis="out")
        except ValueError as exc:
            violations.append(f"{action.module_name}: {exc}")
        if action.consumer_name is None or action.consumer_type is None:
            return violations
        if action.consumer_type == "Conv2d":
            consumer = _get_module(model, action.consumer_name)
            if not isinstance(consumer, nn.Conv2d):
                violations.append(f"{action.consumer_name} is not a Conv2d consumer")
                return violations
            if consumer.in_channels != producer.out_channels:
                violations.append(
                    f"{action.module_name} conv shape contract broken: "
                    f"producer out={producer.out_channels}, "
                    f"consumer in={consumer.in_channels}"
                )
            try:
                validate_conv2d_keep_indices(consumer, action.keep_indices, axis="in")
            except ValueError as exc:
                violations.append(f"{action.consumer_name}: {exc}")
        elif action.consumer_type == "Linear":
            consumer = _get_module(model, action.consumer_name)
            if not isinstance(consumer, nn.Linear):
                violations.append(f"{action.consumer_name} is not a Linear consumer")
                return violations
            kept_features = len(action.keep_indices) * action.feature_block_size
            if consumer.in_features % producer.out_channels != 0:
                violations.append(
                    f"{action.module_name} conv->linear block contract broken: "
                    f"producer out={producer.out_channels}, "
                    f"consumer in={consumer.in_features}"
                )
            if kept_features > consumer.in_features:
                violations.append(
                    f"{action.consumer_name} keeps {kept_features} features, "
                    f"larger than in_features={consumer.in_features}"
                )
        return violations

    if action.action_type == "residual_stage_channels":
        producer = _get_module(model, action.module_name)
        consumers = action.metadata.get("consumers")
        if not isinstance(consumers, list):
            violations.append(f"{action.module_name} residual stage has no consumers")
            return violations
        producer_out = (
            producer.out_channels
            if isinstance(producer, nn.Conv2d)
            else None
        )
        for item in consumers:
            if not isinstance(item, dict):
                continue
            consumer_name = item.get("name")
            consumer_type = item.get("type")
            if consumer_type != "Conv2d" or not isinstance(consumer_name, str):
                continue
            consumer = _get_module(model, consumer_name)
            if not isinstance(consumer, nn.Conv2d):
                violations.append(f"{consumer_name} is not a Conv2d residual consumer")
                continue
            if producer_out is not None and consumer.in_channels != producer_out:
                violations.append(
                    f"{action.module_name} residual shape contract broken: "
                    f"producer out={producer_out}, "
                    f"consumer {consumer_name} in={consumer.in_channels}"
                )
            try:
                validate_conv2d_keep_indices(consumer, action.keep_indices, axis="in")
            except ValueError as exc:
                violations.append(f"{consumer_name}: {exc}")
        return violations
    return violations


def _detection_head_blocked_modules(
    actions: list[StructuredPruningAction],
) -> list[str]:
    blocked: list[str] = []
    for action in actions:
        for module_name in _action_modules(action):
            lowered = module_name.lower()
            if any(pattern in lowered for pattern in _DETECTION_HEAD_PATTERNS):
                blocked.append(module_name)
    return sorted(set(blocked))


def assess_prune_safety(
    model: nn.Module,
    actions: list[StructuredPruningAction],
    *,
    task_type: str | None = None,
) -> PruneSafetyReport:
    """Check family-specific safety contracts before a structured rewrite."""

    checks: list[PruneSafetyCheck] = []
    attention_actions = [
        action for action in actions if action.action_type == "attention_heads"
    ]
    if attention_actions:
        violations: list[str] = []
        for action in attention_actions:
            violations.extend(_attention_action_contracts(model, action))
        checks.append(
            PruneSafetyCheck(
                "transformer_attention",
                "transformer",
                passed=not violations,
                violations=tuple(violations),
                notes=(
                    "Attention head pruning rewrites q/k/v/out projections; "
                    "head divisibility and projection dimension contracts checked.",
                ),
            )
        )

    mlp_actions = [
        action
        for action in actions
        if action.action_type in {"mlp_neuron_group", "gated_mlp_neuron_group"}
    ]
    if mlp_actions:
        violations = []
        for action in mlp_actions:
            violations.extend(_mlp_action_contracts(model, action))
        checks.append(
            PruneSafetyCheck(
                "transformer_mlp",
                "transformer",
                passed=not violations,
                violations=tuple(violations),
                notes=(
                    "MLP neuron pruning keeps producer and consumer dimensions "
                    "aligned; gated MLP requires gate/up/down symmetry.",
                ),
            )
        )

    conv_actions = [
        action
        for action in actions
        if action.action_type
        in {"conv_channel_group", "residual_stage_channels", "concat_branch_channels"}
    ]
    if conv_actions:
        violations = []
        for action in conv_actions:
            violations.extend(_conv_action_contracts(model, action))
        checks.append(
            PruneSafetyCheck(
                "convnet_shape",
                "convnet",
                passed=not violations,
                violations=tuple(violations),
                notes=(
                    "ConvNet pruning keeps grouped-conv divisibility and "
                    "producer/consumer channel shape contracts.",
                ),
            )
        )

    if task_type == "detection":
        blocked = _detection_head_blocked_modules(actions)
        violations = (
            [f"structured detection pruning would rewrite blocked module {name}" for name in blocked]
            if blocked
            else []
        )
        checks.append(
            PruneSafetyCheck(
                "detection_head_guard",
                "detection",
                passed=not violations,
                blocked_modules=tuple(blocked),
                violations=tuple(violations),
                notes=(
                    "Detection postprocess/detect-head modules must never be "
                    "rewritten by model-side structured pruning.",
                ),
            )
        )

    if not checks:
        checks.append(
            PruneSafetyCheck(
                "generic_shape_contract",
                "generic",
                passed=True,
                notes=(
                    "No family-specific guard matched the selected actions; "
                    "index validation and forward diff remain active.",
                ),
            )
        )
    return PruneSafetyReport(task_type=task_type, checks=checks)


__all__ = [
    "PruneSafetyCheck",
    "PruneSafetyReport",
    "assess_prune_safety",
]
