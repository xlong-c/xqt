"""Reference storage semantics for packed W4 linear artifacts."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.core.base import XQTBackendError

from .int8_mma import Int8MmaLinear
from .packing_int4 import (
    _normalize_group_size,
    _quantize_grouped_fp4_weight,
    _unpack_int4,
)


def _packed_w4_from_float_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    output_features, input_features = int(weight.shape[0]), int(weight.shape[1])
    normalized_group_size = _normalize_group_size(group_size, input_features)
    packed_weight, scale, padded_input_features = _quantize_grouped_fp4_weight(
        weight,
        group_size=normalized_group_size,
        input_features=input_features,
        output_features=output_features,
    )
    return packed_weight, scale, normalized_group_size, padded_input_features


class W4StorageInt8MmaLinear(nn.Module):
    """Packed W4 artifact with reference INT8-retarget semantics."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        group_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        output_dtype: torch.dtype = torch.float32,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
        min_int8_rows: int = 0,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.engine = str(engine)
        self.fallback_engine = str(fallback_engine)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.threads = int(threads)
        self.num_stages = int(num_stages)
        self.output_dtype = output_dtype
        self.activation_scale_mode = str(activation_scale_mode)
        self.activation_quant_block_size = int(activation_quant_block_size)
        self.eps = float(eps)
        self.cache_int8_compute_view = bool(cache_int8_compute_view)
        self.min_int8_rows = int(min_int8_rows)
        self._activation_scale = activation_scale
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8).contiguous())
        self.register_buffer("group_scale", group_scale.to(torch.float32).contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())
        self._compute: Int8MmaLinear | None = None
        self._compute_signature: tuple[Any, ...] | None = None
        if self.cache_int8_compute_view:
            self._ensure_compute_view()

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        group_size: int = 128,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
    ) -> "W4StorageInt8MmaLinear":
        packed_weight, group_scale, normalized_group_size, padded = (
            _packed_w4_from_float_weight(
                module.weight.detach(),
                group_size=group_size,
            )
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed_weight,
            group_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=module.weight.dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
        )

    @classmethod
    def from_fp4_weight_only(
        cls,
        module: nn.Module,
        *,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
    ) -> "W4StorageInt8MmaLinear":
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            module.packed_weight.detach(),
            module.weight_scale.detach(),
            bias=bias,
            input_features=module.input_features,
            output_features=module.output_features,
            group_size=module.group_size,
            padded_input_features=module.padded_input_features,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=torch.float32 if module.bias is None else module.bias.dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
        )

    def _apply(self, fn: Any) -> "W4StorageInt8MmaLinear":
        super()._apply(fn)
        self.release_int8_compute_view()
        return self

    def dequantize_weight(self) -> torch.Tensor:
        quantized = _unpack_int4(
            self.packed_weight, self.padded_input_features
        ).reshape(self.output_features, -1, self.group_size)
        scale = self.group_scale.to(torch.float32)
        if scale.ndim == 2:
            scale = scale.unsqueeze(-1)
        expected_shape = quantized.shape[:2] + (1,)
        if scale.shape != expected_shape:
            raise ValueError(
                "group_scale must have shape (output_features, groups) or "
                "(output_features, groups, 1)"
            )
        return (quantized * scale).reshape(
            self.output_features, self.padded_input_features
        )[:, : self.input_features]

    def quantized_weight_codes(self) -> torch.Tensor:
        return _unpack_int4(self.packed_weight, self.padded_input_features)[
            :, : self.input_features
        ]

    def storage_nbytes(self) -> int:
        total = int(self.packed_weight.nbytes) + int(self.group_scale.nbytes)
        if self.bias is not None:
            total += int(self.bias.nbytes)
        return total

    def release_int8_compute_view(self) -> None:
        self._compute = None
        self._compute_signature = None

    def _compute_cache_signature(self) -> tuple[Any, ...]:
        return (
            str(self.packed_weight.device),
            int(getattr(self.packed_weight, "_version", 0)),
            str(self.group_scale.device),
            int(getattr(self.group_scale, "_version", 0)),
        )

    def _ensure_compute_view(self) -> Int8MmaLinear:
        signature = self._compute_cache_signature()
        if self._compute is not None and self._compute_signature == signature:
            return self._compute
        weight = self.dequantize_weight()
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        channel_scale = torch.where(
            max_abs > self.eps,
            max_abs / 127.0,
            torch.ones_like(max_abs),
        )
        qweight = torch.round(weight / channel_scale).clamp(-127, 127).to(torch.int8)
        compute = Int8MmaLinear(
            qweight.t().contiguous(),
            channel_scale.reshape(-1),
            bias=self.bias,
            input_features=self.input_features,
            output_features=self.output_features,
            engine="torch_int_mm",
            fallback_engine=self.fallback_engine,
            output_dtype=self.output_dtype,
            activation_scale_mode=self.activation_scale_mode,
            activation_scale=self._activation_scale,
            activation_quant_block_size=self.activation_quant_block_size,
            eps=self.eps,
            min_int8_rows=self.min_int8_rows,
        ).to(self.packed_weight.device)
        if self.cache_int8_compute_view:
            self._compute = compute
            self._compute_signature = signature
        return compute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "W4StorageInt8MmaLinear input trailing dimension does not match "
                "input_features"
            )
        compute = self._ensure_compute_view()
        if compute.qweight_t.device != inputs.device:
            compute.to(inputs.device)
        output = compute(inputs)
        metadata = dict(compute.execution_metadata())
        metadata.update(
            {
                "storage_dtype": "packed_signed_int4",
                "compute_dtype": "int8",
                "retarget": "w4_storage_int8_mma",
                "group_size": self.group_size,
                "storage_nbytes": self.storage_nbytes(),
                "int8_compute_view_cached": self._compute is not None,
                "quantization_nature": "true",
                "artifact_view": "contracts_reference",
            }
        )
        self.last_execution = metadata
        return output

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self.last_execution)

    def tilelang_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        packed_weight = self.packed_weight.to(device=device)
        scale = self.group_scale.to(device=device, dtype=dtype)
        bias = None if self.bias is None else self.bias.to(device=device, dtype=dtype)
        return packed_weight, scale, bias, None, self.input_features, self.group_size

    def triton_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        return self.tilelang_packed_dequant_gemm_args(dtype=dtype, device=device)


__all__ = ["W4StorageInt8MmaLinear"]
