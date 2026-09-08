"""Pattern 1: Dequant-GEMM fusion transform (XQT-012).

Fuses weight dequantization scaling and matrix multiplication into a single
fused linear operation. Declares supported layout, dtype and broadcast constraints.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from ..base import TransformPlan, TransformReport


class FusedDequantGemmLinear(nn.Module):
    """Fused weight-dequantization GEMM operation."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight.clone().detach())
        if weight_scale is not None:
            self.register_buffer("weight_scale", weight_scale.clone().detach())
        else:
            self.weight_scale = None
        if bias is not None:
            self.register_buffer("bias", bias.clone().detach())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.weight_scale is not None:
            w = w * self.weight_scale
        return torch.nn.functional.linear(x, w, self.bias)


class DequantGemmTransform:
    """Graph rewrite fusing weight dequantization and GEMM."""

    name = "dequant_gemm"
    required_kernels: tuple[str, ...] = ("torch",)

    def __init__(
        self,
        target_modules: Sequence[str] | None = None,
        supported_dtypes: Sequence[str] = ("float32", "float16", "bfloat16"),
    ) -> None:
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.supported_dtypes = tuple(supported_dtypes)

    def match(self, model: nn.Module) -> TransformPlan | None:
        matched_targets: list[str] = []
        rejection_reasons: list[str] = []
        expected_replacements: dict[str, str] = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.target_modules is not None and name not in self.target_modules:
                continue

            # Check preconditions: dtype compatibility and weight presence
            w = getattr(module, "weight", None)
            if w is None or not isinstance(w, torch.Tensor):
                rejection_reasons.append(f"{name}: missing weight tensor")
                continue

            dtype_name = str(w.dtype).replace("torch.", "")
            if dtype_name not in self.supported_dtypes:
                rejection_reasons.append(
                    f"{name}: dtype {dtype_name} not in supported {self.supported_dtypes}"
                )
                continue

            matched_targets.append(name)
            expected_replacements[name] = "FusedDequantGemmLinear"

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
            absorbed_ops=("dequant",),
            online_ops=("fused_gemm",),
            metadata={
                "matched": True,
                "preconditions_met": True,
                "rejection_reasons": rejection_reasons,
                "expected_replacements": expected_replacements,
                "expected_contract_changes": {
                    t: {"op": "fused_dequant_gemm"} for t in matched_targets
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

            if not isinstance(orig_module, nn.Linear):
                continue

            scale = getattr(orig_module, "weight_scale", None)
            fused = FusedDequantGemmLinear(
                weight=orig_module.weight,
                weight_scale=scale,
                bias=orig_module.bias,
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


__all__ = ["DequantGemmTransform", "FusedDequantGemmLinear"]
