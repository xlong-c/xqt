"""Native ConvRot dynamic W8A8 kernels for Ada ``sm_89``."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.kernels.jit.utils.compile import CompileSpec, csrc_path, load_extension
from xqt.kernels.ops._impl.cute.svdq_w4a4_sm89 import pack_scale

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
_NUNCHAKU_INCLUDE = _REPO_ROOT / "learn" / "nunchaku"
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_NATIVE_ROT_SIZE = 256
_CUDA_SOURCE = csrc_path("quantization", "convrot_w8a8_sm89_kernel.cu")
_BINDING_SOURCE = csrc_path("quantization", "convrot_w8a8_sm89_binding.cpp")


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def native_convrot_w8a8_shape_supported(
    logical_input_features: int,
    rotated_input_features: int,
    output_features: int,
    rot_size: int,
) -> bool:
    logical_k = int(logical_input_features)
    rotated_k = int(rotated_input_features)
    return (
        logical_k > 0
        and logical_k <= rotated_k
        and int(output_features) > 0
        and int(output_features) % 4 == 0
        and int(rot_size) == _NATIVE_ROT_SIZE
        and rotated_k % _NATIVE_ROT_SIZE == 0
    )


def native_prerotated_w8a8_shape_supported(
    rotated_input_features: int,
    output_features: int,
) -> bool:
    return (
        int(rotated_input_features) > 0
        and int(output_features) > 0
        and int(output_features) % 4 == 0
    )


def _dtype_key(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise XQTBackendError("native ConvRot W8A8 requires float16 or bfloat16")


@lru_cache(maxsize=2)
def _load_extension(dtype_key: str) -> Any:
    if os.environ.get("XQT_DISABLE_CONVROT_W8A8_SM89", "0") == "1":
        raise XQTBackendError("native sm_89 ConvRot W8A8 backend is disabled")
    if not torch.cuda.is_available():
        raise XQTBackendError("native sm_89 ConvRot W8A8 backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"native ConvRot W8A8 currently targets sm_89, got sm_{major}{minor}"
        )
    if dtype_key not in {"fp16", "bf16"}:
        raise XQTBackendError(f"unsupported ConvRot W8A8 dtype key: {dtype_key}")
    if not _NUNCHAKU_INCLUDE.is_dir():
        raise XQTBackendError(
            f"Nunchaku kernel headers are missing at {_NUNCHAKU_INCLUDE}"
        )

    dtype_flags = ["-DXQT_W8A8_FP16=1"] if dtype_key == "fp16" else []
    try:
        return load_extension(
            CompileSpec(
                name=f"xqt_convrot_w8a8_sm89_{dtype_key}_v1",
                sources=(_BINDING_SOURCE, _CUDA_SOURCE),
                include_dirs=(_NUNCHAKU_INCLUDE,),
                cxx_flags=("-O3", "-std=c++20", *dtype_flags),
                cuda_flags=(
                    "-O3",
                    "-std=c++20",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DENABLE_BF16=1",
                    *dtype_flags,
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--generate-line-info",
                ),
                target_arch="sm_89",
            ),
            verbose=False,
        )
    except Exception as exc:
        raise XQTBackendError(
            f"failed to build native sm_89 ConvRot W8A8 {dtype_key} extension: {exc}"
        ) from exc


def native_convrot_w8a8_available(
    dtype: torch.dtype,
    *,
    build: bool = False,
) -> bool:
    if dtype not in _SUPPORTED_DTYPES:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        return False
    if os.environ.get("XQT_DISABLE_CONVROT_W8A8_SM89", "0") == "1":
        return False
    if not _NUNCHAKU_INCLUDE.is_dir():
        return False
    if not build:
        return True
    try:
        _load_extension(_dtype_key(dtype))
    except XQTBackendError:
        return False
    return True


def native_convrot_w8a8_version(dtype: torch.dtype) -> str:
    return str(_load_extension(_dtype_key(dtype)).version())


@dataclass(frozen=True)
class PackedConvRotW8A8Linear:
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    packed_bias: torch.Tensor
    input_features: int
    output_features: int
    padded_input_features: int
    padded_output_features: int
    dtype: torch.dtype


@dataclass
class ConvRotW8A8Workspace:
    quantized_activation: torch.Tensor
    activation_scales: torch.Tensor

    @property
    def padded_rows(self) -> int:
        return int(self.quantized_activation.shape[0])


def pack_convrot_w8a8_linear(
    qweight_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> PackedConvRotW8A8Linear:
    if qweight_t.ndim != 2 or qweight_t.dtype != torch.int8:
        raise XQTBackendError("qweight_t must be a two-dimensional int8 tensor")
    if not qweight_t.is_cuda or not qweight_t.is_contiguous():
        raise XQTBackendError("qweight_t must be contiguous CUDA storage")
    if dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError("native ConvRot W8A8 requires float16 or bfloat16")
    input_features, output_features = (int(dim) for dim in qweight_t.shape)
    if output_features % 4 != 0:
        raise XQTBackendError(
            "native ConvRot W8A8 requires output_features to be a multiple of 4"
        )
    if weight_scale.ndim != 1 or int(weight_scale.numel()) != output_features:
        raise ValueError("weight_scale must contain one value per output channel")
    if weight_scale.device != qweight_t.device:
        raise XQTBackendError("qweight_t and weight_scale must share a CUDA device")

    padded_input = _round_up(input_features, _NATIVE_ROT_SIZE)
    padded_output = _round_up(output_features, 128)
    padded_codes = F.pad(
        qweight_t.t().contiguous(),
        (0, padded_input - input_features, 0, padded_output - output_features),
    )
    raw_scales = F.pad(
        weight_scale.to(device=qweight_t.device, dtype=dtype).reshape(-1),
        (0, padded_output - output_features),
        value=1.0,
    ).contiguous()
    dense_weight = padded_codes.to(dtype=dtype) * raw_scales.reshape(-1, 1)
    packed_weight = torch.empty_like(dense_weight, dtype=torch.int8)
    extension = _load_extension(_dtype_key(dtype))
    extension.quantize_weight(dense_weight.contiguous(), raw_scales, packed_weight)

    if bias is None:
        padded_bias = torch.zeros(
            padded_output,
            dtype=dtype,
            device=qweight_t.device,
        )
    else:
        if bias.ndim != 1 or int(bias.numel()) != output_features:
            raise ValueError("bias must match output_features")
        padded_bias = F.pad(
            bias.to(device=qweight_t.device, dtype=dtype),
            (0, padded_output - output_features),
        )
    return PackedConvRotW8A8Linear(
        qweight=packed_weight,
        weight_scales=pack_scale(raw_scales),
        packed_bias=pack_scale(padded_bias.contiguous()),
        input_features=input_features,
        output_features=output_features,
        padded_input_features=padded_input,
        padded_output_features=padded_output,
        dtype=dtype,
    )


def allocate_convrot_w8a8_workspace(
    rows: int,
    packed: PackedConvRotW8A8Linear,
) -> ConvRotW8A8Workspace:
    padded_rows = _round_up(int(rows), 256)
    return ConvRotW8A8Workspace(
        quantized_activation=torch.empty(
            (padded_rows, packed.padded_input_features),
            dtype=torch.int8,
            device=packed.qweight.device,
        ),
        activation_scales=torch.empty(
            padded_rows,
            dtype=packed.dtype,
            device=packed.qweight.device,
        ),
    )


def convrot_w8a8_linear(
    inputs: torch.Tensor,
    packed: PackedConvRotW8A8Linear,
    *,
    rotated_input_features: int,
    rot_size: int,
    workspace: ConvRotW8A8Workspace | None = None,
) -> torch.Tensor:
    if inputs.ndim != 2 or not inputs.is_cuda:
        raise XQTBackendError("inputs must be a two-dimensional CUDA tensor")
    if inputs.dtype != packed.dtype:
        raise XQTBackendError("inputs must match the packed ConvRot W8A8 dtype")
    if inputs.device != packed.qweight.device:
        raise XQTBackendError("inputs and packed weights must share a CUDA device")
    logical_k = int(inputs.shape[1])
    rotated_k = int(rotated_input_features)
    if int(rot_size) == 1:
        supported = native_prerotated_w8a8_shape_supported(
            rotated_k,
            packed.output_features,
        )
    else:
        supported = native_convrot_w8a8_shape_supported(
            logical_k,
            rotated_k,
            packed.output_features,
            int(rot_size),
        )
    if not supported or rotated_k != packed.input_features:
        raise XQTBackendError("native ConvRot W8A8 shape or rotation size is unsupported")

    active_workspace = workspace or allocate_convrot_w8a8_workspace(
        int(inputs.shape[0]),
        packed,
    )
    if active_workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")
    if int(active_workspace.quantized_activation.shape[1]) != packed.padded_input_features:
        raise ValueError("workspace feature extent does not match packed weights")
    extension = _load_extension(_dtype_key(inputs.dtype))
    return extension.linear(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.qweight,
        packed.weight_scales,
        packed.packed_bias,
        rotated_k,
        int(rot_size),
        packed.output_features,
    )


def bind_convrot_w8a8_linear(
    packed: PackedConvRotW8A8Linear,
    workspace: ConvRotW8A8Workspace,
    *,
    rows: int,
    rotated_input_features: int,
    rot_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind validated packed state for the steady-state ConvRot hot path."""

    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    if int(workspace.quantized_activation.shape[1]) != packed.padded_input_features:
        raise ValueError("workspace feature extent does not match packed weights")
    extension = _load_extension(_dtype_key(packed.dtype))
    quantized_activation = workspace.quantized_activation
    activation_scales = workspace.activation_scales
    qweight = packed.qweight
    weight_scales = packed.weight_scales
    packed_bias = packed.packed_bias
    rotated_k = int(rotated_input_features)
    rotation_size = int(rot_size)
    output_features = packed.output_features

    def run(inputs: torch.Tensor) -> torch.Tensor:
        return extension.linear(
            inputs.contiguous(),
            quantized_activation,
            activation_scales,
            qweight,
            weight_scales,
            packed_bias,
            rotated_k,
            rotation_size,
            output_features,
        )

    return run


__all__ = [
    "ConvRotW8A8Workspace",
    "PackedConvRotW8A8Linear",
    "allocate_convrot_w8a8_workspace",
    "bind_convrot_w8a8_linear",
    "convrot_w8a8_linear",
    "native_convrot_w8a8_available",
    "native_convrot_w8a8_shape_supported",
    "native_convrot_w8a8_version",
    "native_prerotated_w8a8_shape_supported",
    "pack_convrot_w8a8_linear",
]
