"""Hybrid inference engine for quantized model artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from xqt.contracts import ExecutionPolicyPayload, QuantizedModel

from .channel import apply_channel_hybrid_policy, collect_channel_hybrid_map
from .policy import (
    apply_execution_policy,
    build_execution_policy_payload,
    collect_module_precision_map,
    normalize_compute_precision,
)


@dataclass
class HybridInferenceResult:
    """Result of one hybrid-inference forward pass."""

    output: Any
    precision_map: dict[str, str] = field(default_factory=dict)
    channel_hybrid_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy_kind: str | None = None
    runtime: str = "pytorch"


class HybridInferenceEngine:
    """Run mixed-precision inference over quantized modules.

    The engine consumes:
    - a model that already holds quantized storage (for example ConvRot buffers)
    - an optional execution policy describing per-module compute precision

    It never calls quantizers, calibration hooks, or sensitivity analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        default_precision: str = "w4a4",
        runtime: str = "pytorch",
        policy: ExecutionPolicyPayload | Mapping[str, Any] | None = None,
        apply_policy_on_init: bool = True,
    ) -> None:
        self.model = model
        self.default_precision = normalize_compute_precision(default_precision)
        self.runtime = str(runtime)
        self._policy: ExecutionPolicyPayload | None = None
        if policy is not None:
            self.bind_policy(policy, apply=apply_policy_on_init)
        elif apply_policy_on_init:
            apply_execution_policy(
                self.model,
                default_precision=self.default_precision,
                inplace=True,
            )

    @classmethod
    def from_quantized_model(
        cls,
        quantized: QuantizedModel,
        *,
        default_precision: str = "w4a4",
        runtime: str = "pytorch",
        stage_name: str = "quant",
        source_model_stage: str = "baseline",
        apply_policy_on_init: bool = True,
    ) -> "HybridInferenceEngine":
        """Build an engine from a quant-stage artifact without re-quantizing."""

        metadata = dict(quantized.metadata)
        handoff = (
            quantized.infer_handoff()
            if hasattr(quantized, "infer_handoff")
            else {"model": quantized.model, "compute_config": None}
        )
        compute_config = handoff.get("compute_config")
        raw_policies = metadata.get("execution_policies", [])
        if not raw_policies and isinstance(compute_config, Mapping):
            from xqt.contracts.compute import ComputeConfig

            parsed = ComputeConfig.from_mapping(compute_config)
            overrides = [] if parsed is None else parsed.precision_overrides()
            caps = [] if parsed is None else parsed.all_required_capabilities()
            preferred: list[str] = []
            if parsed is not None:
                for module in parsed.modules:
                    preferred.extend(module.preferred_engines)
            policy = build_execution_policy_payload(
                stage_name=stage_name,
                source_model_stage=source_model_stage,
                policy_kind="compute_config",
                runtime=runtime,
                precision_overrides=overrides,
                required_capabilities=caps,
                preferred_engines=preferred,
                compute_config=compute_config,
                metadata={
                    "default_precision": default_precision,
                    "source": "quantized_model.compute_config",
                },
                module_count=len(list(quantized.quantized_modules)),
            )
            return cls(
                quantized.model,
                default_precision=default_precision,
                runtime=runtime,
                policy=policy,
                apply_policy_on_init=apply_policy_on_init,
            )
        policy: ExecutionPolicyPayload | None = None
        if isinstance(raw_policies, list) and raw_policies:
            first = raw_policies[0]
            if isinstance(first, Mapping):
                overrides = first.get("precision_overrides", [])
                if not isinstance(overrides, list):
                    overrides = []
                channel_overrides = first.get("channel_overrides", [])
                if not isinstance(channel_overrides, list):
                    channel_overrides = []
                policy = build_execution_policy_payload(
                    stage_name=stage_name,
                    source_model_stage=source_model_stage,
                    policy_kind=str(first.get("policy_kind", "mixed_precision")),
                    runtime=str(first.get("runtime", runtime)),
                    precision_overrides=[
                        dict(item) for item in overrides if isinstance(item, Mapping)
                    ],
                    metadata={
                        "runtime_strategy": first.get(
                            "runtime_strategy",
                            metadata.get("implementation"),
                        ),
                        "mixed_ratio": first.get("mixed_ratio"),
                        "default_precision": default_precision,
                        "algorithm_metadata": metadata.get("algorithm_metadata"),
                        "channel_overrides": [
                            dict(item)
                            for item in channel_overrides
                            if isinstance(item, Mapping)
                        ],
                    },
                    module_count=len(list(quantized.quantized_modules)),
                )
        return cls(
            quantized.model,
            default_precision=default_precision,
            runtime=runtime,
            policy=policy,
            apply_policy_on_init=apply_policy_on_init,
        )

    @property
    def policy(self) -> ExecutionPolicyPayload | None:
        return self._policy

    def bind_policy(
        self,
        policy: ExecutionPolicyPayload | Mapping[str, Any],
        *,
        apply: bool = True,
        inplace: bool = True,
    ) -> ExecutionPolicyPayload:
        """Attach one execution policy and optionally materialize it."""

        if isinstance(policy, ExecutionPolicyPayload):
            bound = policy
        else:
            overrides = policy.get("precision_overrides", [])
            if not isinstance(overrides, list):
                overrides = []
            bound = build_execution_policy_payload(
                stage_name=str(policy.get("stage_name", "runtime")),
                source_model_stage=str(policy.get("source_model_stage", "quant")),
                policy_kind=str(policy.get("policy_kind", "mixed_precision")),
                runtime=str(policy.get("runtime", self.runtime)),
                precision_overrides=[
                    dict(item) for item in overrides if isinstance(item, Mapping)
                ],
                metadata=dict(policy.get("metadata", {}))
                if isinstance(policy.get("metadata"), Mapping)
                else {
                    key: value
                    for key, value in policy.items()
                    if key
                    not in {
                        "precision_overrides",
                        "policy_kind",
                        "runtime",
                        "stage_name",
                        "source_model_stage",
                    }
                },
            )
        self._policy = bound
        if apply:
            self.model = apply_execution_policy(
                self.model,
                policy=bound,
                default_precision=self.default_precision,
                inplace=inplace,
            )
        return bound

    def apply_policy(
        self,
        *,
        precision_overrides: Sequence[Mapping[str, Any]] | None = None,
        default_precision: str | None = None,
        inplace: bool = True,
    ) -> nn.Module:
        """Materialize the bound or provided precision overrides on the model."""

        resolved_default = (
            self.default_precision
            if default_precision is None
            else normalize_compute_precision(default_precision)
        )
        self.default_precision = resolved_default
        self.model = apply_execution_policy(
            self.model,
            precision_overrides=precision_overrides,
            default_precision=resolved_default,
            inplace=inplace,
            policy=self._policy if precision_overrides is None else None,
        )
        return self.model

    def set_module_precision(self, module_name: str, precision: str) -> None:
        from .policy import set_module_compute_precision

        module = self.model.get_submodule(module_name)
        set_module_compute_precision(module, precision)

    def set_module_channel_hybrid(
        self,
        module_name: str,
        *,
        high_precision_channels: Sequence[int],
        axis: str = "input",
        high_precision: str = "bf16",
        low_precision: str = "w4a4",
        enabled: bool = True,
    ) -> None:
        apply_channel_hybrid_policy(
            self.model,
            channel_overrides=[
                {
                    "module": module_name,
                    "axis": axis,
                    "high_precision_channels": list(high_precision_channels),
                    "high_precision": high_precision,
                    "low_precision": low_precision,
                    "enabled": enabled,
                }
            ],
            inplace=True,
        )

    def clear_module_channel_hybrid(self, module_name: str) -> None:
        module = self.model.get_submodule(module_name)
        setter = getattr(module, "set_channel_hybrid_spec", None)
        if callable(setter):
            setter(None)
            return
        raise TypeError(
            f"module {module_name!r} does not support channel hybrid inference"
        )

    def precision_map(self) -> dict[str, str]:
        return collect_module_precision_map(self.model)

    def channel_hybrid_map(self) -> dict[str, dict[str, Any]]:
        return collect_channel_hybrid_map(self.model)

    def eval(self) -> "HybridInferenceEngine":
        self.model.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> "HybridInferenceEngine":
        self.model = self.model.to(*args, **kwargs)
        return self

    @torch.inference_mode()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run one inference step on the quantized model."""

        return self.model(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @torch.inference_mode()
    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> HybridInferenceResult:
        """Run inference and return output plus live precision map."""

        output = self.forward(*args, **kwargs)
        return HybridInferenceResult(
            output=output,
            precision_map=self.precision_map(),
            channel_hybrid_map=self.channel_hybrid_map(),
            policy_kind=None if self._policy is None else self._policy.policy_kind,
            runtime=self.runtime,
        )
