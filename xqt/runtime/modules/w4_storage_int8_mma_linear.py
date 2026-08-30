"""Runtime packed W4 storage Linear with INT8 MMA retarget."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from xqt.contracts.packing_int4 import (
    _unpack_int4,
)
from xqt.contracts.w4_storage import (
    W4StorageInt8MmaLinear as W4StorageInt8MmaStorageLinear,
    _packed_w4_from_float_weight as _core_packed_w4_from_float_weight,
)
from xqt.core.errors import XQTBackendError
from xqt.runtime.modules.int8_mma_linear import Int8MmaLinear

_ACTIVATION_SCALE_MODES = {"dynamic", "static"}

def _channel_int8_from_float_weight(
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.detach().to(torch.float32)
    max_abs = weight_f.abs().amax(dim=1, keepdim=True)
    scale = torch.where(
        max_abs > float(eps),
        max_abs / 127.0,
        torch.ones_like(max_abs),
    )
    qweight = torch.round(weight_f / scale).clamp(-127, 127).to(torch.int8)
    return qweight.t().contiguous(), scale.reshape(-1).contiguous()

def _packed_w4_from_float_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    return _core_packed_w4_from_float_weight(weight, group_size=group_size)

def _float_weight_from_packed_w4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    input_features: int,
    output_features: int,
    group_size: int,
    padded_input_features: int,
) -> torch.Tensor:
    quantized = _unpack_int4(packed_weight, padded_input_features)
    grouped = quantized.reshape(output_features, -1, group_size)
    scale = weight_scale.to(dtype=torch.float32)
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    if scale.shape != grouped.shape[:2] + (1,):
        raise ValueError(
            "weight_scale must have shape (output_features, groups) or "
            "(output_features, groups, 1)"
        )
    dequantized = grouped * scale
    return dequantized.reshape(output_features, padded_input_features)[:, :input_features]

class W4StorageInt8MmaLinear(W4StorageInt8MmaStorageLinear):
    """Packed W4 storage Linear that delegates a requested W8A8 INT8 retarget."""

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
        nn.Module.__init__(self)
        if str(activation_scale_mode) not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
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
        if int(min_int8_rows) < 0:
            raise ValueError("min_int8_rows must be >= 0")
        self.min_int8_rows = int(min_int8_rows)
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8).contiguous())
        self.register_buffer("group_scale", group_scale.to(torch.float32).contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())
        self._compute: Int8MmaLinear | None = None
        self._compute_signature: tuple[Any, ...] | None = None
        self._activation_scale = activation_scale
        if self.cache_int8_compute_view:
            self._ensure_compute_view()

    def _apply(self, fn: Any) -> "W4StorageInt8MmaLinear":
        """Move storage tensors and invalidate derived INT8 compute state."""

        super()._apply(fn)
        self._compute = None
        self._compute_signature = None
        return self

    @classmethod
    def from_storage(
        cls,
        module: W4StorageInt8MmaStorageLinear,
        *,
        engine: str | None = None,
        fallback_engine: str | None = None,
    ) -> "W4StorageInt8MmaLinear":
        """Materialize a backend execution view from a contracts storage shell."""

        return cls(
            module.packed_weight.detach(),
            module.group_scale.detach(),
            bias=None if module.bias is None else module.bias.detach(),
            input_features=module.input_features,
            output_features=module.output_features,
            group_size=module.group_size,
            padded_input_features=module.padded_input_features,
            engine=engine or module.engine,
            fallback_engine=fallback_engine or module.fallback_engine,
            block_m=module.block_m,
            block_n=module.block_n,
            block_k=module.block_k,
            threads=module.threads,
            num_stages=module.num_stages,
            output_dtype=module.output_dtype,
            activation_scale_mode=module.activation_scale_mode,
            activation_scale=module._activation_scale,
            activation_quant_block_size=module.activation_quant_block_size,
            eps=module.eps,
            cache_int8_compute_view=module.cache_int8_compute_view,
            min_int8_rows=module.min_int8_rows,
        )

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
        packed_weight, group_scale, normalized_group_size, padded = _packed_w4_from_float_weight(
            module.weight.detach(),
            group_size=group_size,
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
            module.packed_weight.detach().to(torch.uint8).contiguous(),
            module.weight_scale.detach().to(torch.float32).contiguous(),
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

    def dequantize_weight(self) -> torch.Tensor:
        return _float_weight_from_packed_w4(
            self.packed_weight,
            self.group_scale,
            input_features=self.input_features,
            output_features=self.output_features,
            group_size=self.group_size,
            padded_input_features=self.padded_input_features,
        )

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
            tuple(int(dim) for dim in self.packed_weight.shape),
            str(self.group_scale.device),
            int(getattr(self.group_scale, "_version", 0)),
            tuple(int(dim) for dim in self.group_scale.shape),
            str(self.bias.device) if self.bias is not None else None,
            int(getattr(self.bias, "_version", 0)) if self.bias is not None else None,
        )

    def _ensure_compute_view(self) -> Int8MmaLinear:
        signature = self._compute_cache_signature()
        if self._compute is not None and self._compute_signature == signature:
            return self._compute
        self._compute = None
        self._compute_signature = None
        weight = self.dequantize_weight()
        qweight_t, channel_scale = _channel_int8_from_float_weight(weight, eps=self.eps)
        compute = Int8MmaLinear(
            qweight_t,
            channel_scale,
            bias=self.bias,
            input_features=self.input_features,
            output_features=self.output_features,
            engine=self.engine,
            fallback_engine=self.fallback_engine,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            threads=self.threads,
            num_stages=self.num_stages,
            output_dtype=self.output_dtype,
            activation_scale_mode=self.activation_scale_mode,
            activation_scale=self._activation_scale,
            activation_quant_block_size=self.activation_quant_block_size,
            eps=self.eps,
            min_int8_rows=self.min_int8_rows,
        )
        compute.to(device=self.packed_weight.device)
        if self.cache_int8_compute_view:
            self._compute = compute
            self._compute_signature = signature
        return compute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "W4StorageInt8MmaLinear input trailing dimension does not match input_features"
            )
        compute = self._ensure_compute_view()
        if compute.qweight_t.device != inputs.device:
            compute.to(device=inputs.device)
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
            }
        )
        self.last_execution = metadata
        return output

    def execution_metadata(self) -> dict[str, Any]:
        metadata = dict(self.last_execution)
        metadata["artifact_view"] = "runtime"
        return metadata

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
        """Expose packed INT4 inputs for Triton operator wrappers."""

        return self.tilelang_packed_dequant_gemm_args(dtype=dtype, device=device)

__all__ = ["W4StorageInt8MmaLinear"]
