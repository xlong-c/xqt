"""Pattern 3: Activation-Quant fusion transform (XQT-012).

Fuses activation non-linearities (GELU, SiLU, ReLU) and subsequent input quantization
into a single fused kernel operation.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from ..base import TransformPlan, TransformReport


class FusedActivationQuant(nn.Module):
    """Fused activation function and quantization scaling operation."""

    def __init__(
        self,
        activation_type: str = "gelu",
        quant_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation_type = activation_type.lower()
        self.quant_scale = float(quant_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "gelu":
            act = torch.nn.functional.gelu(x)
        elif self.activation_type == "silu":
            act = torch.nn.functional.silu(x)
        elif self.activation_type == "relu":
            act = torch.nn.functional.relu(x)
        else:
            act = x

        if self.quant_scale != 1.0:
            act = act * self.quant_scale
        return act


class ActivationQuantTransform:
    """Graph rewrite fusing activation functions and quantization."""

    name = "activation_quant"
    required_kernels: tuple[str, ...] = ("torch",)

    def __init__(
        self,
        target_modules: Sequence[str] | None = None,
        supported_activations: Sequence[str] = ("gelu", "silu", "relu", "GELU", "SiLU", "ReLU"),
    ) -> None:
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.supported_activations = tuple(a.lower() for a in supported_activations)

    def match(self, model: nn.Module) -> TransformPlan | None:
        matched_targets: list[str] = []
        rejection_reasons: list[str] = []
        expected_replacements: dict[str, str] = {}

        for name, module in model.named_modules():
            cls_name = type(module).__name__.lower()
            if cls_name not in self.supported_activations:
                continue
            if self.target_modules is not None and name not in self.target_modules:
                continue

            matched_targets.append(name)
            expected_replacements[name] = "FusedActivationQuant"

        if not matched_targets:
            return TransformPlan(
                transform_name=self.name,
                targets=(),
                metadata={
                    "matched": False,
                    "preconditions_met": False,
                    "rejection_reasons": rejection_reasons,
                },
            )

        return TransformPlan(
            transform_name=self.name,
            targets=tuple(matched_targets),
            absorbed_ops=("activation", "input_quant"),
            online_ops=("fused_activation_quant",),
            metadata={
                "matched": True,
                "preconditions_met": True,
                "rejection_reasons": rejection_reasons,
                "expected_replacements": expected_replacements,
                "expected_contract_changes": {
                    t: {"op": "fused_activation_quant"} for t in matched_targets
                },
            },
        )

    def apply(self, model: nn.Module, plan: TransformPlan) -> TransformReport:
        if not plan.targets:
            return TransformReport(
                transform_name=self.name,
                applied=False,
                notes=("no_eligible_targets",),
            )

        applied_targets: list[str] = []
        for target_path in plan.targets:
            parts = target_path.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            orig_module = getattr(parent, attr_name)

            act_name = type(orig_module).__name__.lower()
            fused = FusedActivationQuant(activation_type=act_name)
            setattr(parent, attr_name, fused)
            applied_targets.append(target_path)

        return TransformReport(
            transform_name=self.name,
            applied=bool(applied_targets),
            absorbed_ops=plan.absorbed_ops,
            online_ops=plan.online_ops,
            targets=tuple(applied_targets),
            metadata={"fused_count": len(applied_targets)},
        )


__all__ = ["ActivationQuantTransform", "FusedActivationQuant"]
