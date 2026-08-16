"""Pure additive composite storage artifacts shared by quant and runtime."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.packing_int4 import _unpack_int4


class CompositeAddModule(nn.Module):
    """Marker base for additive composite artifacts and runtime views."""

    xqt_storage_protocol = "composite_add"


def initialize_composite_add_storage(
    module: CompositeAddModule,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    packed_residual: torch.Tensor,
    residual_scale: torch.Tensor,
    bias: torch.Tensor | None,
    input_features: int,
    output_features: int,
    group_size: int,
    padded_input_features: int,
    quant_dtype: str,
) -> None:
    """Initialize the shared packed storage layout on an artifact or view."""

    if not isinstance(module, CompositeAddModule):
        raise TypeError("module must be a CompositeAddModule")
    nn.Module.__init__(module)
    if down_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("low-rank weights must be two-dimensional")
    if int(down_weight.shape[1]) != int(input_features):
        raise ValueError("down_weight width must match input_features")
    if int(up_weight.shape[0]) != int(output_features):
        raise ValueError("up_weight height must match output_features")
    if int(down_weight.shape[0]) != int(up_weight.shape[1]):
        raise ValueError("low-rank weights must have the same rank")
    if int(group_size) < 1:
        raise ValueError("group_size must be positive")
    if int(padded_input_features) < int(input_features):
        raise ValueError("padded_input_features must cover input_features")

    module.input_features = int(input_features)
    module.output_features = int(output_features)
    module.group_size = int(group_size)
    module.padded_input_features = int(padded_input_features)
    module.rank = int(down_weight.shape[0])
    module.quant_dtype = str(quant_dtype)

    module.down_proj = nn.Linear(
        module.input_features,
        module.rank,
        bias=False,
        dtype=down_weight.dtype,
        device=down_weight.device,
    )
    module.up_proj = nn.Linear(
        module.rank,
        module.output_features,
        bias=False,
        dtype=up_weight.dtype,
        device=up_weight.device,
    )
    with torch.no_grad():
        module.down_proj.weight.copy_(down_weight)
        module.up_proj.weight.copy_(up_weight)

    module.register_buffer(
        "packed_residual",
        packed_residual.detach().to(torch.uint8),
    )
    module.register_buffer(
        "residual_scale",
        residual_scale.detach().to(torch.float32),
    )
    module.register_buffer(
        "bias",
        None if bias is None else bias.detach().to(torch.float32),
    )
    module._pending_activation_scale = None
    module._pending_activation_scale_mode = "dynamic"


class CompositeAddLinear(CompositeAddModule):
    """Canonical low-rank plus packed-residual model artifact.

    This class owns storage and reference execution only. Backend selection and
    compute materialization belong to ``xqt.runtime``.
    """

    xqt_storage_protocol = "composite_add_low_rank_plus_residual"

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        packed_residual: torch.Tensor,
        residual_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        quant_dtype: str = "int4",
    ) -> None:
        initialize_composite_add_storage(
            self,
            down_weight=down_weight,
            up_weight=up_weight,
            packed_residual=packed_residual,
            residual_scale=residual_scale,
            bias=bias,
            input_features=input_features,
            output_features=output_features,
            group_size=group_size,
            padded_input_features=padded_input_features,
            quant_dtype=quant_dtype,
        )

    @classmethod
    def from_packed_parts(
        cls,
        *,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        packed_residual: torch.Tensor,
        residual_scale: torch.Tensor,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        quant_dtype: str,
    ) -> "CompositeAddLinear":
        """Build the artifact from already-quantized model tensors."""

        return cls(
            down_weight=down_weight,
            up_weight=up_weight,
            packed_residual=packed_residual,
            residual_scale=residual_scale,
            bias=bias,
            input_features=input_features,
            output_features=output_features,
            group_size=group_size,
            padded_input_features=padded_input_features,
            quant_dtype=quant_dtype,
        )

    def dequantize_residual(self) -> torch.Tensor:
        """Reconstruct the residual branch for reference computation."""

        codes = _unpack_int4(self.packed_residual, self.padded_input_features)
        grouped = codes.reshape(
            self.output_features,
            -1,
            self.group_size,
        )
        dequantized = grouped * self.residual_scale.unsqueeze(-1)
        return dequantized.reshape(
            self.output_features,
            self.padded_input_features,
        )[:, : self.input_features]

    def low_rank_weight(self) -> torch.Tensor:
        """Reconstruct the dense low-rank branch weight."""

        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        """Reconstruct the complete reference weight."""

        return self.low_rank_weight() + self.dequantize_residual().to(
            device=self.down_proj.weight.device,
            dtype=self.down_proj.weight.dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the reference additive composite path."""

        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            raise ValueError(
                "CompositeAddLinear input trailing dimension does not match "
                "input_features"
            )
        dtype = inputs.dtype
        device = inputs.device
        residual_weight = self.dequantize_residual().to(device=device, dtype=dtype)
        output = F.linear(inputs, residual_weight)
        output = output + self.up_proj(self.down_proj(inputs))
        if self.bias is not None:
            output = output + self.bias.to(device=device, dtype=dtype)
        return output

    def execution_metadata(self) -> dict[str, Any]:
        """Describe the storage artifact without claiming a backend."""

        return {
            "implementation": "composite_add_reference",
            "compute_contract": "composite_add",
            "residual_storage": "packed_signed_int4_group_scale",
            "residual_compute": "dequant_fp16",
            "low_rank_branch": "source_precision",
            "quant_dtype": self.quant_dtype,
            "rank": self.rank,
            "group_size": self.group_size,
        }

    def set_activation_materialize_hint(
        self,
        *,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
    ) -> None:
        """Store activation settings for a later runtime materialization."""

        self._pending_activation_scale_mode = str(activation_scale_mode)
        self._pending_activation_scale = activation_scale


__all__ = [
    "CompositeAddLinear",
    "CompositeAddModule",
    "initialize_composite_add_storage",
]
