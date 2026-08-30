"""Weight-only quantized Linear module contract (AWQ/GPTQ storage layout).

Holds the packed-weight module produced by algorithmic AWQ/GPTQ quantization:
buffer storage, reference forward, and layout exposure for kernel wrappers.
Calibration algorithms live in ``xqt.compression.quant.quantizers.awq_gptq_weight_only``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.packing_int4 import (
    _normalize_group_size,
    _pack_int4,
    _pad_weight_for_groups,
    _safe_positive,
    _unpack_int4,
)
from xqt.core.compilation import can_mutate_runtime_cache as _can_mutate_runtime_cache


def _signed_quant_bounds(bits: int) -> tuple[int, int]:
    if bits == 8:
        return -128, 127
    if bits == 4:
        return -8, 7
    raise ValueError(f"Unsupported signed weight-only bit width: {bits}")


def _quantize_grouped_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    input_features: int,
    output_features: int,
    group_multiplier: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    min_code, max_code = _signed_quant_bounds(bits)
    weight, padded_input_features = _pad_weight_for_groups(
        weight.detach().to(torch.float32),
        input_features=input_features,
        group_size=group_size,
    )
    grouped_weight = weight.reshape(output_features, -1, group_size)
    max_abs = grouped_weight.abs().amax(dim=2, keepdim=True)
    scale = torch.where(max_abs > 0, max_abs / float(max_code), torch.ones_like(max_abs))
    if group_multiplier is not None:
        multiplier = group_multiplier.to(device=scale.device, dtype=scale.dtype)
        if multiplier.ndim == 2:
            multiplier = multiplier.unsqueeze(-1)
        scale = scale * _safe_positive(multiplier)
    quantized = torch.clamp(torch.round(grouped_weight / scale), min=min_code, max=max_code)
    quantized = quantized.to(torch.int8).reshape(output_features, padded_input_features)
    if bits == 4:
        return _pack_int4(quantized), scale, padded_input_features
    return quantized.contiguous(), scale, padded_input_features


def _decode_quantized_weight(
    quantized_weight: torch.Tensor,
    *,
    bits: int,
    padded_input_features: int,
) -> torch.Tensor:
    if bits == 4:
        return _unpack_int4(quantized_weight, padded_input_features)
    return quantized_weight.to(torch.float32)


class AWQGPTQWeightOnlyLinear(nn.Module):
    """Weight-only Linear produced by algorithmic AWQ/GPTQ calibration."""

    def __init__(
        self,
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        bits: int,
        method: str,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.bits = int(bits)
        self.method = str(method)
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._dense_bias_cache: dict[tuple[str, str], torch.Tensor | None] = {}
        if self.bits != 4:
            self.tilelang_packed_dequant_gemm_args = None
        storage_dtype = torch.uint8 if self.bits == 4 else torch.int8
        self.register_buffer("quantized_weight", quantized_weight.to(storage_dtype))
        self.register_buffer("weight_scale", scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        bits: int,
        group_size: int,
        method: str,
    ) -> "AWQGPTQWeightOnlyLinear":
        normalized_group_size = _normalize_group_size(group_size, module.in_features)
        quantized_weight, scale, padded_input_features = _quantize_grouped_weight(
            module.weight,
            bits=bits,
            group_size=normalized_group_size,
            input_features=module.in_features,
            output_features=module.out_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            quantized_weight,
            scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded_input_features,
            bits=bits,
            method=method,
        )

    def dequantize_weight(self) -> torch.Tensor:
        quantized = _decode_quantized_weight(
            self.quantized_weight,
            bits=self.bits,
            padded_input_features=self.padded_input_features,
        )
        grouped = quantized.reshape(self.output_features, -1, self.group_size)
        dequantized = grouped * self.weight_scale
        return dequantized.reshape(self.output_features, self.padded_input_features)[
            :, : self.input_features
        ]

    def quantized_weight_codes(self) -> torch.Tensor:
        """Return unpacked signed integer weight codes as a dense matrix."""

        return _decode_quantized_weight(
            self.quantized_weight,
            bits=self.bits,
            padded_input_features=self.padded_input_features,
        )[:, : self.input_features]

    def expanded_weight_scale(self) -> torch.Tensor:
        """Return per-element scale expanded from the stored group-wise scale."""

        expanded = self.weight_scale.expand(-1, -1, self.group_size).reshape(
            self.output_features,
            self.padded_input_features,
        )
        return expanded[:, : self.input_features]

    def tilelang_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        """Expose dense qweight/scale inputs for TileLang dequant GEMM wrappers."""

        qweight = self.quantized_weight_codes().to(device=device, dtype=dtype)
        scale = self.expanded_weight_scale().to(device=device, dtype=dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return qweight, scale, bias, None

    def tilelang_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        """Expose packed INT4 inputs for TileLang packed dequant GEMM wrappers."""

        if self.bits != 4:
            raise ValueError("Packed TileLang dequant GEMM bridge is only available for 4-bit weights")
        packed_weight = self.quantized_weight.to(device=device)
        scale = self.weight_scale.to(device=device, dtype=dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return packed_weight, scale, bias, None, self.input_features, self.group_size

    def triton_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        """Expose packed INT4 inputs for Triton operator wrappers."""

        return self.tilelang_packed_dequant_gemm_args(dtype=dtype, device=device)

    def dense_weight(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not _can_mutate_runtime_cache():
            return self.dequantize_weight().to(device=device, dtype=dtype).detach()
        key = (str(device), str(dtype))
        cached = self._dense_weight_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        weight = self.dequantize_weight().to(device=device, dtype=dtype).detach()
        self._dense_weight_cache[key] = weight
        return weight

    def dense_bias(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not _can_mutate_runtime_cache():
            return None if self.bias is None else self.bias.to(device=device, dtype=dtype).detach()
        key = (str(device), str(dtype))
        if key in self._dense_bias_cache:
            return self._dense_bias_cache[key]
        bias = None if self.bias is None else self.bias.to(device=device, dtype=dtype).detach()
        self._dense_bias_cache[key] = bias
        return bias

    def tilelang_dense_linear_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        """Expose cached dense weights for native linear fastpaths."""

        return (
            self.dense_weight(dtype=dtype, device=device),
            self.dense_bias(dtype=dtype, device=device),
            None,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight(dtype=inputs.dtype, device=inputs.device)
        bias = self.dense_bias(dtype=inputs.dtype, device=inputs.device)
        return F.linear(inputs, weight, bias)


__all__ = ["AWQGPTQWeightOnlyLinear"]
