"""FP8 compute views for generic additive composite artifacts."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.contracts.composite import CompositeAddLinear, CompositeAddModule
from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear


class CompositeAddFp8Linear(CompositeAddModule):
    """Split FP8 residual plus source-precision low-rank execution view."""

    xqt_storage_protocol = "composite_add_fp8_split"

    def __init__(
        self,
        *,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        residual_weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        output_dtype: torch.dtype,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if down_weight.ndim != 2 or up_weight.ndim != 2:
            raise ValueError("low-rank weights must be two-dimensional")
        if int(down_weight.shape[1]) != int(input_features):
            raise ValueError("down_weight width must match input_features")
        if int(up_weight.shape[0]) != int(output_features):
            raise ValueError("up_weight height must match output_features")
        if int(down_weight.shape[0]) != int(up_weight.shape[1]):
            raise ValueError("low-rank weights must have the same rank")

        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.rank = int(down_weight.shape[0])
        low_rank_dtype = (
            down_weight.dtype
            if down_weight.dtype in {torch.float16, torch.bfloat16}
            else output_dtype
        )
        self.down_proj = nn.Linear(
            self.input_features,
            self.rank,
            bias=False,
            dtype=low_rank_dtype,
            device=down_weight.device,
        )
        self.up_proj = nn.Linear(
            self.rank,
            self.output_features,
            bias=False,
            dtype=low_rank_dtype,
            device=up_weight.device,
        )
        with torch.no_grad():
            self.down_proj.weight.copy_(down_weight.to(low_rank_dtype))
            self.up_proj.weight.copy_(up_weight.to(low_rank_dtype))
        self.residual_fp8 = Fp8MmaLinear.from_dense_weight(
            residual_weight,
            bias=bias,
            input_features=self.input_features,
            output_features=self.output_features,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=min_fp8_rows,
            eps=eps,
        )
        self._fused_forward: Any = None

    @classmethod
    def from_composite(
        cls,
        module: CompositeAddLinear,
        *,
        output_dtype: torch.dtype,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> "CompositeAddFp8Linear":
        """Build an FP8 split view from the canonical packed artifact."""

        if not isinstance(module, CompositeAddLinear):
            raise TypeError("module must be a CompositeAddLinear artifact")
        return cls(
            down_weight=module.down_proj.weight.detach(),
            up_weight=module.up_proj.weight.detach(),
            residual_weight=module.dequantize_residual(),
            bias=None if module.bias is None else module.bias.detach(),
            input_features=module.input_features,
            output_features=module.output_features,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=min_fp8_rows,
            eps=eps,
        )

    def _apply(self, fn: Any) -> "CompositeAddFp8Linear":
        super()._apply(fn)
        self._fused_forward = None
        return self

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            return False
        try:
            self._fused_forward = compile_fn(self._compute_lean, mode=mode)
        except Exception:
            self._fused_forward = None
            return False
        return True

    def disable_fusion(self) -> None:
        self._fused_forward = None

    def _low_rank(self, inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        low_rank_inputs = inputs.to(
            device=self.down_proj.weight.device,
            dtype=self.down_proj.weight.dtype,
        )
        return self.up_proj(self.down_proj(low_rank_inputs)).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def _compute_lean(self, inputs: torch.Tensor) -> torch.Tensor:
        residual_output = self.residual_fp8.lean_forward(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def _compute_guarded(self, inputs: torch.Tensor) -> torch.Tensor:
        residual_output = self.residual_fp8(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            raise ValueError(
                "CompositeAddFp8Linear input trailing dimension does not match "
                "input_features"
            )
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        use_fused = (
            self._fused_forward is not None
            and inputs.is_cuda
            and not (
                0 < self.residual_fp8.min_fp8_rows
                and rows < self.residual_fp8.min_fp8_rows
            )
        )
        if use_fused:
            return self._fused_forward(inputs)
        return self._compute_guarded(inputs)

    def dequantize_residual(self) -> torch.Tensor:
        return self.residual_fp8.dequantize_weight()

    def low_rank_weight(self) -> torch.Tensor:
        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        low_rank_weight = self.low_rank_weight()
        return low_rank_weight + self.dequantize_residual().to(
            device=low_rank_weight.device,
            dtype=low_rank_weight.dtype,
        )

    def execution_metadata(self) -> dict[str, Any]:
        metadata = self.residual_fp8.execution_metadata()
        metadata.update(
            {
                "implementation": "composite_add_fp8_split",
                "compute_contract": "composite_add",
                "low_rank_branch": "source_precision",
                "residual_compute": "fp8_scaled_mm",
                "fusion_enabled": self._fused_forward is not None,
            }
        )
        return metadata


__all__ = ["CompositeAddFp8Linear"]
