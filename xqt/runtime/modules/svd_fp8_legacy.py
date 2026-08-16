"""Legacy SVDQuant FP8 split executor."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SVDQuantFp8Linear(nn.Module):
    """SVDQuant split module: source-precision low-rank plus FP8 residual."""

    def __init__(
        self,
        *,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        residual_weight: torch.Tensor,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        output_dtype: torch.dtype = torch.float16,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.output_dtype = output_dtype
        rank = int(down_weight.shape[0])
        self.rank = rank
        self.xqt_storage_protocol = "svd_low_rank_plus_residual"
        lr_dtype = (
            down_weight.dtype
            if down_weight.dtype in {torch.float16, torch.bfloat16}
            else output_dtype
        )
        self.down_proj = nn.Linear(
            self.input_features,
            rank,
            bias=False,
            dtype=lr_dtype,
        )
        self.down_proj.weight.data = down_weight.detach().to(lr_dtype).clone()
        self.up_proj = nn.Linear(
            rank,
            self.output_features,
            bias=False,
            dtype=lr_dtype,
        )
        self.up_proj.weight.data = up_weight.detach().to(lr_dtype).clone()
        self.residual_fp8 = Fp8MmaLinear.from_dense_weight(
            residual_weight,
            bias=bias,
            input_features=self.input_features,
            output_features=self.output_features,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=int(min_fp8_rows),
            eps=float(eps),
        )
        self._fused_forward: Any = None

    def _apply(self, fn: Any) -> "SVDQuantFp8Linear":
        """Move child runtime modules and discard device-specific compiled code."""

        super()._apply(fn)
        self._fused_forward = None
        return self

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        """Fold the branch chain with torch.compile when available."""

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

    def _low_rank(self, inputs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        low_rank_inputs = inputs.to(dtype=self.down_proj.weight.dtype)
        return self.up_proj(self.down_proj(low_rank_inputs)).to(
            device=ref.device,
            dtype=ref.dtype,
        )

    def _compute_lean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compile-friendly path without residual metadata side effects."""

        residual_output = self.residual_fp8.lean_forward(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def _compute_guarded(self, inputs: torch.Tensor) -> torch.Tensor:
        """Eager path with residual metadata and small-M fallback."""

        residual_output = self.residual_fp8(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def dequantize_residual(self) -> torch.Tensor:
        return self.residual_fp8.dequantize_weight()

    def low_rank_weight(self) -> torch.Tensor:
        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        low_rank_weight = self.low_rank_weight()
        return low_rank_weight + self.dequantize_residual().to(
            dtype=low_rank_weight.dtype,
            device=low_rank_weight.device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.input_features:
            raise ValueError(
                "SVDQuantFp8Linear input trailing dimension does not match "
                "input_features"
            )
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        residual = self.residual_fp8
        use_fused = (
            self._fused_forward is not None
            and inputs.is_cuda
            and not (0 < residual.min_fp8_rows and rows < residual.min_fp8_rows)
        )
        if use_fused:
            return self._fused_forward(inputs)
        return self._compute_guarded(inputs)

    def execution_metadata(self) -> dict[str, Any]:
        metadata = self.residual_fp8.execution_metadata()
        metadata.update(
            {
                "implementation": "composite_add_svd_low_rank_plus_fp8_residual",
                "compute_contract": "composite_add",
                "low_rank_branch": "source_precision",
                "residual_compute": "fp8_scaled_mm",
                "fusion_enabled": self._fused_forward is not None,
            }
        )
        return metadata


__all__ = ["SVDQuantFp8Linear"]
