"""NVFP4 weight bridge helpers for XQT TileLang operator optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_NVFP4_CODEBOOK = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _can_mutate_runtime_cache() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    return not torch.jit.is_tracing()


def unpack_nvfp4e2m1(packed_weight: torch.Tensor, *, input_features: int) -> torch.Tensor:
    """Decode packed NVFP4 E2M1 weights into float32 codes."""

    if packed_weight.dtype != torch.uint8:
        raise TypeError("packed_weight must be uint8")
    low = packed_weight & 0x0F
    high = (packed_weight >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(packed_weight.shape[0], -1)
    unpacked = unpacked[:, : int(input_features)]
    codebook = _NVFP4_CODEBOOK.to(device=packed_weight.device)
    return codebook[unpacked.long()]


def _normalize_group_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    if weight_scale.ndim == 2:
        return weight_scale.unsqueeze(-1)
    if weight_scale.ndim == 3 and weight_scale.shape[2] == 1:
        return weight_scale
    raise ValueError(
        "weight_scale must have shape [out_features, groups] or [out_features, groups, 1]"
    )


def expand_group_scale(
    weight_scale: torch.Tensor,
    *,
    group_size: int,
    input_features: int,
) -> torch.Tensor:
    """Expand per-group scale to `[out_features, input_features]`."""

    normalized = _normalize_group_scale(weight_scale)
    expanded = normalized.expand(-1, -1, int(group_size)).reshape(normalized.shape[0], -1)
    return expanded[:, : int(input_features)]


@dataclass(frozen=True)
class NVFP4TensorLayout:
    """Resolved tensor contract for packed NVFP4 linear weights."""

    input_features: int
    output_features: int
    group_size: int
    packed_weight_name: str
    weight_scale_name: str
    weight_global_scale_name: str | None
    bias_name: str | None
    invert_weight_global_scale: bool = False


class NVFP4LinearBridge(nn.Module):
    """Thin adapter that exposes packed NVFP4 weights to XQT TileLang wrappers."""

    def __init__(
        self,
        *,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        source_module_type: str = "unknown",
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.source_module_type = str(source_module_type)
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._dense_bias_cache: dict[tuple[str, str], torch.Tensor | None] = {}
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8))
        self.register_buffer("weight_scale", _normalize_group_scale(weight_scale).to(torch.float32))
        if weight_global_scale is None:
            self.register_buffer("weight_global_scale", None)
        else:
            self.register_buffer("weight_global_scale", weight_global_scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

    def _apply(self, fn: Any) -> "NVFP4LinearBridge":
        """Preserve calibration tensors in float32 across parent .to(dtype=...) calls."""

        super()._apply(fn)
        self._dense_weight_cache.clear()
        self._dense_bias_cache.clear()
        self.weight_scale = self.weight_scale.to(dtype=torch.float32)
        if self.weight_global_scale is not None:
            self.weight_global_scale = self.weight_global_scale.to(dtype=torch.float32)
        if self.bias is not None:
            self.bias = self.bias.to(dtype=torch.float32)
        return self

    @classmethod
    def from_tensor_layout(
        cls,
        module: nn.Module,
        *,
        layout: NVFP4TensorLayout,
        clone_tensors: bool = True,
    ) -> "NVFP4LinearBridge":
        packed_weight = getattr(module, layout.packed_weight_name)
        weight_scale = getattr(module, layout.weight_scale_name)
        weight_global_scale = (
            getattr(module, layout.weight_global_scale_name)
            if layout.weight_global_scale_name is not None
            else None
        )
        bias = getattr(module, layout.bias_name) if layout.bias_name is not None else None
        if clone_tensors:
            packed_weight = packed_weight.detach().clone()
            weight_scale = weight_scale.detach().clone()
            if weight_global_scale is not None:
                weight_global_scale = weight_global_scale.detach().clone()
            if bias is not None:
                bias = bias.detach().clone()
        if weight_global_scale is not None and layout.invert_weight_global_scale:
            weight_global_scale = torch.reciprocal(weight_global_scale)
        return cls(
            packed_weight=packed_weight,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            bias=bias,
            input_features=layout.input_features,
            output_features=layout.output_features,
            group_size=layout.group_size,
            source_module_type=type(module).__name__,
        )

    def expanded_weight_scale(self) -> torch.Tensor:
        scale = self.weight_scale
        if self.weight_global_scale is not None:
            scale = scale / self.weight_global_scale.reshape(1, 1, 1)
        return expand_group_scale(
            scale,
            group_size=self.group_size,
            input_features=self.input_features,
        )

    def dequantize_weight(self) -> torch.Tensor:
        codes = unpack_nvfp4e2m1(self.packed_weight, input_features=self.input_features)
        scale = self.expanded_weight_scale().to(device=codes.device, dtype=codes.dtype)
        return codes * scale

    def dense_weight(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a cached dense dequantized weight for static inference workloads."""

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
        """Return a cached dense bias matching the requested runtime dtype/device."""

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
        """Expose cached dense weights for Ada/Hopper native half GEMM fastpaths."""

        return (
            self.dense_weight(dtype=dtype, device=device),
            self.dense_bias(dtype=dtype, device=device),
            None,
        )

    def tilelang_packed_nvfp4_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None]:
        """Expose packed NVFP4 tensors for TileLang wrappers."""

        packed_weight = self.packed_weight.to(device=device)
        weight_scale = self.weight_scale.to(device=device, dtype=dtype)
        weight_global_scale = None
        if self.weight_global_scale is not None:
            weight_global_scale = self.weight_global_scale.to(device=device, dtype=dtype)
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=device, dtype=dtype)
        return (
            packed_weight,
            weight_scale,
            bias,
            None,
            self.input_features,
            self.group_size,
            weight_global_scale,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight(dtype=inputs.dtype, device=inputs.device)
        bias = self.dense_bias(dtype=inputs.dtype, device=inputs.device)
        return F.linear(inputs, weight, bias)


def infer_nvfp4_tensor_layout(module: nn.Module) -> NVFP4TensorLayout | None:
    """Infer a packed NVFP4 tensor contract from a module by attribute names."""

    packed_weight_name = None
    for candidate in ("qweight", "weight_packed", "packed_weight", "weight"):
        value = getattr(module, candidate, None)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 2:
            packed_weight_name = candidate
            break
    if packed_weight_name is None:
        return None

    weight_scale_name = None
    for candidate in ("weight_scale", "scales", "weight_scales"):
        value = getattr(module, candidate, None)
        if isinstance(value, torch.Tensor) and value.ndim in {2, 3}:
            weight_scale_name = candidate
            break
    if weight_scale_name is None:
        return None

    packed_weight = getattr(module, packed_weight_name)
    weight_scale = getattr(module, weight_scale_name)
    if weight_scale.ndim == 3 and weight_scale.shape[2] != 1:
        return None
    input_features = int(getattr(module, "in_features", 0) or 0)
    if input_features <= 0:
        input_features = int(packed_weight.shape[1]) * 2
    output_features = int(getattr(module, "out_features", 0) or 0)
    if output_features <= 0:
        output_features = int(packed_weight.shape[0])
    if input_features <= 0 or output_features <= 0:
        return None

    groups = int(weight_scale.shape[1])
    if groups <= 0:
        return None
    padded_input_features = int(packed_weight.shape[1]) * 2
    group_size = int(padded_input_features // groups)
    if group_size <= 0:
        return None
    if groups * group_size != padded_input_features:
        return None

    weight_global_scale_name = None
    invert_weight_global_scale = False
    for candidate in (
        "weight_global_scale",
        "global_scale",
        "weight_scale_global",
        "weight_scale_2",
    ):
        value = getattr(module, candidate, None)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            weight_global_scale_name = candidate
            invert_weight_global_scale = candidate == "weight_scale_2"
            break

    bias_name = "bias" if isinstance(getattr(module, "bias", None), torch.Tensor) else None

    return NVFP4TensorLayout(
        input_features=input_features,
        output_features=output_features,
        group_size=group_size,
        packed_weight_name=packed_weight_name,
        weight_scale_name=weight_scale_name,
        weight_global_scale_name=weight_global_scale_name,
        bias_name=bias_name,
        invert_weight_global_scale=invert_weight_global_scale,
    )


def bridge_module_to_nvfp4_linear(module: nn.Module) -> NVFP4LinearBridge | None:
    """Best-effort bridge from an external packed NVFP4 module to XQT protocol."""

    layout = infer_nvfp4_tensor_layout(module)
    if layout is None:
        return None
    return NVFP4LinearBridge.from_tensor_layout(module, layout=layout)


def bridge_module_to_nvfp4_linear_shared(module: nn.Module) -> NVFP4LinearBridge | None:
    """Create an NVFP4 bridge that shares source tensor storage with the module."""

    layout = infer_nvfp4_tensor_layout(module)
    if layout is None:
        return None
    return NVFP4LinearBridge.from_tensor_layout(
        module,
        layout=layout,
        clone_tensors=False,
    )


__all__ = [
    "NVFP4LinearBridge",
    "NVFP4TensorLayout",
    "bridge_module_to_nvfp4_linear",
    "bridge_module_to_nvfp4_linear_shared",
    "expand_group_scale",
    "infer_nvfp4_tensor_layout",
    "unpack_nvfp4e2m1",
]
