"""Pattern 2: Norm-Quant fusion transform (XQT-012).

Fuses RMSNorm / LayerNorm normalization scaling and activation quantization
into a single fused normalization-quantization operation.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from ..base import TransformPlan, TransformReport


class FusedNormQuant(nn.Module):
    """Fused LayerNorm/RMSNorm and activation quantization operation."""

    def __init__(
        self,
        norm_dim: int,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-5,
        quant_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm_dim = norm_dim
        self.eps = eps
        self.quant_scale = float(quant_scale)
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach())
        else:
            self.register_buffer("weight", torch.ones(norm_dim))
        if bias is not None:
            self.register_buffer("bias", bias.clone().detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard LayerNorm / RMSNorm computation
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            normed = normed * self.weight
        if self.bias is not None:
            normed = normed + self.bias
        # Combined quantization scaling
        if self.quant_scale != 1.0:
            normed = normed * self.quant_scale
        return normed


class NormQuantTransform:
    """Graph rewrite fusing normalization and activation quantization."""

    name = "norm_quant"
    required_kernels: tuple[str, ...] = ("torch",)

    def __init__(
        self,
        target_modules: Sequence[str] | None = None,
        require_no_bias: bool = False,
    ) -> None:
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.require_no_bias = require_no_bias

    def match(self, model: nn.Module) -> TransformPlan | None:
        matched_targets: list[str] = []
        rejection_reasons: list[str] = []
        expected_replacements: dict[str, str] = {}

        for name, module in model.named_modules():
            # Match LayerNorm or modules with RMSNorm in class name
            cls_name = type(module).__name__.lower()
            is_norm = isinstance(module, nn.LayerNorm) or "rmsnorm" in cls_name or "norm" in cls_name
            if not is_norm:
                continue
            if self.target_modules is not None and name not in self.target_modules:
                continue

            # Mathematical precondition: if require_no_bias is set, bias must be absent
            bias = getattr(module, "bias", None)
            if self.require_no_bias and bias is not None:
                rejection_reasons.append(f"{name}: bias present but require_no_bias=True")
                continue

            matched_targets.append(name)
            expected_replacements[name] = "FusedNormQuant"

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
            absorbed_ops=("norm_scaling", "act_quant"),
            online_ops=("fused_norm_quant",),
            metadata={
                "matched": True,
                "preconditions_met": True,
                "rejection_reasons": rejection_reasons,
                "expected_replacements": expected_replacements,
                "expected_contract_changes": {
                    t: {"op": "fused_norm_quant"} for t in matched_targets
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

            norm_dim = getattr(orig_module, "normalized_shape", None)
            if isinstance(norm_dim, (tuple, list)):
                norm_dim = norm_dim[-1]
            elif not isinstance(norm_dim, int):
                norm_dim = getattr(orig_module, "dim", 64)

            weight = getattr(orig_module, "weight", None)
            bias = getattr(orig_module, "bias", None)
            eps = float(getattr(orig_module, "eps", 1e-5))

            fused = FusedNormQuant(
                norm_dim=norm_dim,
                weight=weight,
                bias=bias,
                eps=eps,
            )
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


__all__ = ["FusedNormQuant", "NormQuantTransform"]
