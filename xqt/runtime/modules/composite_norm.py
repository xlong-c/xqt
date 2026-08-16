"""Runtime input transforms for generic additive composite artifacts."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from xqt.contracts.compute import ModuleComputeSpec
from xqt.contracts.composite import CompositeAddModule
from xqt.runtime.modules.composite_add import (
    materialize_composite_compute,
    materialize_composite_w4a4,
)


class RMSNormCompositeLinear(CompositeAddModule):
    """Apply RMSNorm before a generic additive composite linear module.

    RMSNorm fusion is an execution concern. Keeping it as a wrapper leaves the
    quantized composite artifact independent from model-specific input layout
    and native kernel choices.
    """

    xqt_storage_protocol = "composite_add_rmsnorm_input"

    def __init__(
        self,
        composite: CompositeAddModule,
        norm_weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(composite, CompositeAddModule):
            raise TypeError("composite must be a CompositeAddModule")
        if norm_weight.ndim != 1 or int(norm_weight.numel()) != composite.input_features:
            raise ValueError("norm_weight must match composite input_features")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        self.composite = composite
        self.input_features = composite.input_features
        self.output_features = composite.output_features
        self.rank = composite.rank
        self.group_size = composite.group_size
        self.padded_input_features = composite.padded_input_features
        self.quant_dtype = composite.quant_dtype
        self.fused_norm_eps = float(eps)
        self.register_buffer(
            "norm_weight",
            norm_weight.detach().to(torch.float32),
            persistent=False,
        )

    def _apply_rms_norm(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.norm_weight.to(device=inputs.device, dtype=torch.float32)
        row_scale = torch.rsqrt(
            inputs.float().pow(2).mean(dim=-1, keepdim=True) + self.fused_norm_eps
        )
        return (inputs.float() * row_scale * weight).to(inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the input transform, then execute the active composite."""

        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            raise ValueError(
                "RMSNormCompositeLinear input trailing dimension does not match "
                "input_features"
            )
        return self.composite(self._apply_rms_norm(inputs))

    def execution_metadata(self) -> dict[str, Any]:
        """Report the transform separately from the quantized artifact."""

        return {
            "implementation": "rmsnorm_composite_wrapper",
            "compute_contract": "composite_add",
            "input_transform": "rmsnorm",
            "fused_norm_eps": self.fused_norm_eps,
            "composite": self.composite.execution_metadata(),
        }

    def materialize_w4a4(
        self,
        *,
        native_fusion: bool = True,
        layout: str = "main",
    ) -> "RMSNormCompositeLinear":
        """Materialize the inner composite while retaining the input wrapper."""

        materialized = materialize_composite_w4a4(
            self.composite,
            native_fusion=native_fusion,
            layout=layout,
        )
        if not isinstance(materialized, CompositeAddModule):
            raise TypeError("W4A4 materialization must remain an additive composite")
        return type(self)(
            materialized,
            self.norm_weight,
            eps=self.fused_norm_eps,
        )

    def materialize_compute(
        self,
        spec: ModuleComputeSpec | Mapping[str, Any],
    ) -> nn.Module:
        """Materialize the inner compute path without moving the norm transform."""

        materialized = materialize_composite_compute(self.composite, spec)
        if materialized is self.composite:
            return self
        if not isinstance(materialized, CompositeAddModule):
            return self
        return type(self)(
            materialized,
            self.norm_weight,
            eps=self.fused_norm_eps,
        )


__all__ = ["RMSNormCompositeLinear"]
