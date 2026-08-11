"""Rowwise ConvRot W4A4 CUDA path for Ada ``sm_89``.

The activation kernel performs a regular-Hadamard rotation, dynamic rowwise
INT4 quantization, and row-major nibble packing without materializing the
rotated half tensor. The CUTLASS GEMM consumes the packed activations and
weights directly and applies both row/column scales plus bias in its epilogue.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from xqt.core.errors import XQTBackendError

_HERE = Path(__file__).resolve().parent
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_ROT_SIZE = 256
_MIN_K = 1024
_MAX_K = 32768


def _cutlass_include_dir() -> Path | None:
    spec = importlib.util.find_spec("tilelang")
    if spec is None or spec.origin is None:
        return None
    include_dir = Path(spec.origin).resolve().parent / "3rdparty" / "cutlass" / "include"
    if not (include_dir / "cutlass" / "cutlass.h").is_file():
        return None
    return include_dir


def native_rowwise_convrot_w4a4_shape_supported(
    input_features: int,
    output_features: int,
    rot_size: int,
) -> bool:
    """Return whether the half2/bf162 warp-FHT path covers this shape."""

    k = int(input_features)
    n = int(output_features)
    return (
        int(rot_size) == _ROT_SIZE
        and _MIN_K <= k <= _MAX_K
        and (k == _MIN_K or k % 2048 == 0)
        and n > 0
        and n % 8 == 0
    )


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    if os.environ.get("XQT_DISABLE_CONVROT_W4A4_ROWWISE_SM89", "0") == "1":
        raise XQTBackendError("rowwise ConvRot W4A4 backend is disabled by environment")
    if not torch.cuda.is_available():
        raise XQTBackendError("rowwise ConvRot W4A4 backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"rowwise ConvRot W4A4 currently targets sm_89, got sm_{major}{minor}"
        )
    cutlass_include = _cutlass_include_dir()
    if cutlass_include is None:
        raise XQTBackendError("CUTLASS headers bundled with TileLang are unavailable")

    from torch.utils.cpp_extension import load

    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_max_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ.setdefault("MAX_JOBS", "1")
    try:
        return load(
            name="xqt_convrot_w4a4_rowwise_sm89_v1",
            sources=[
                str(_HERE / "convrot_w4a4_rowwise_sm89_binding.cpp"),
                str(_HERE / "convrot_w4a4_rowwise_sm89_kernel.cu"),
            ],
            extra_include_paths=[str(cutlass_include)],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--generate-line-info",
            ],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:
        raise XQTBackendError(
            f"failed to build rowwise ConvRot W4A4 extension: {exc}"
        ) from exc
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list
        if old_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_max_jobs


def native_rowwise_convrot_w4a4_available(*, build: bool = False) -> bool:
    """Return whether the source-level rowwise backend can run here."""

    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    if os.environ.get("XQT_DISABLE_CONVROT_W4A4_ROWWISE_SM89", "0") == "1":
        return False
    if _cutlass_include_dir() is None:
        return False
    if not build:
        return True
    try:
        _load_extension()
    except XQTBackendError:
        return False
    return True


def native_rowwise_convrot_w4a4_version() -> str:
    return str(_load_extension().version())


def _require_half_cuda_2d(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise XQTBackendError(f"{name} must be a 2D tensor")
    if not tensor.is_cuda:
        raise XQTBackendError(f"{name} must be CUDA")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError(f"{name} must be float16 or bfloat16")


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or int(values.shape[1]) % 2 != 0:
        raise ValueError("signed INT4 values must be 2D with an even K extent")
    encoded = torch.bitwise_and(values.to(torch.int16), 0x0F).to(torch.uint8)
    packed = encoded[:, 0::2] | (encoded[:, 1::2] << 4)
    return packed.contiguous().view(torch.int8)


@dataclass(frozen=True)
class PackedConvRotW4A4Rowwise:
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    bias: torch.Tensor
    input_features: int
    output_features: int


@dataclass
class ConvRotW4A4RowwiseWorkspace:
    quantized_activation: torch.Tensor
    activation_scales: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.quantized_activation.shape[0])


def pack_convrot_w4a4_rowwise_weight(
    rotated_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> PackedConvRotW4A4Rowwise:
    """Quantize an offline-rotated weight with one FP32 scale per output row."""

    _require_half_cuda_2d(rotated_weight, "rotated_weight")
    output_features, input_features = (int(dim) for dim in rotated_weight.shape)
    if not native_rowwise_convrot_w4a4_shape_supported(
        input_features,
        output_features,
        _ROT_SIZE,
    ):
        raise XQTBackendError("rotated weight shape is unsupported by rowwise ConvRot W4A4")
    weight_fp32 = rotated_weight.float()
    absmax = weight_fp32.abs().amax(dim=1, keepdim=True)
    scales = torch.clamp(absmax / 7.0, min=1.0e-10)
    quantized = torch.clamp(torch.round(weight_fp32 / scales), min=-7, max=7).to(
        torch.int8
    )
    if bias is None:
        packed_bias = torch.zeros(
            output_features,
            dtype=torch.float32,
            device=rotated_weight.device,
        )
    else:
        if bias.ndim != 1 or int(bias.numel()) != output_features:
            raise ValueError("bias must match rotated_weight output features")
        packed_bias = bias.to(device=rotated_weight.device, dtype=torch.float32).contiguous()
    return PackedConvRotW4A4Rowwise(
        qweight=_pack_signed_int4(quantized),
        weight_scales=scales.reshape(output_features).to(torch.float32).contiguous(),
        bias=packed_bias,
        input_features=input_features,
        output_features=output_features,
    )


def wrap_convrot_w4a4_rowwise_weight(
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    input_features: int,
    output_features: int,
) -> PackedConvRotW4A4Rowwise:
    """Wrap canonical rowwise buffers without requantizing or copying qweight."""

    if packed_weight.dtype == torch.uint8:
        qweight = packed_weight.contiguous().view(torch.int8)
    elif packed_weight.dtype == torch.int8:
        qweight = packed_weight.contiguous()
    else:
        raise ValueError("packed_weight must be uint8 or int8")
    if tuple(qweight.shape) != (int(output_features), int(input_features) // 2):
        raise ValueError("packed_weight shape does not match rowwise ConvRot dimensions")
    scales = weight_scales.reshape(-1).to(dtype=torch.float32).contiguous()
    if int(scales.numel()) != int(output_features):
        raise ValueError("weight_scales must contain one value per output row")
    if bias is None:
        packed_bias = torch.zeros(
            int(output_features),
            dtype=torch.float32,
            device=qweight.device,
        )
    else:
        packed_bias = bias.to(device=qweight.device, dtype=torch.float32).contiguous()
    return PackedConvRotW4A4Rowwise(
        qweight=qweight,
        weight_scales=scales,
        bias=packed_bias,
        input_features=int(input_features),
        output_features=int(output_features),
    )


def allocate_convrot_w4a4_rowwise_workspace(
    rows: int,
    packed: PackedConvRotW4A4Rowwise,
) -> ConvRotW4A4RowwiseWorkspace:
    normalized_rows = int(rows)
    if normalized_rows < 1:
        raise ValueError("rows must be positive")
    return ConvRotW4A4RowwiseWorkspace(
        quantized_activation=torch.empty(
            (normalized_rows, packed.input_features // 2),
            dtype=torch.int8,
            device=packed.qweight.device,
        ),
        activation_scales=torch.empty(
            normalized_rows,
            dtype=torch.float32,
            device=packed.qweight.device,
        ),
    )


def convrot_w4a4_rowwise_linear(
    inputs: torch.Tensor,
    packed: PackedConvRotW4A4Rowwise,
    *,
    workspace: ConvRotW4A4RowwiseWorkspace | None = None,
) -> torch.Tensor:
    """Run warp-FHT rowwise INT4 quantization plus CUTLASS INT4 GEMM."""

    _require_half_cuda_2d(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device:
        raise XQTBackendError("inputs and packed weight must share a CUDA device")
    if not native_rowwise_convrot_w4a4_shape_supported(
        packed.input_features,
        packed.output_features,
        _ROT_SIZE,
    ):
        raise XQTBackendError("shape is unsupported by rowwise ConvRot W4A4")
    active_workspace = workspace or allocate_convrot_w4a4_rowwise_workspace(
        int(inputs.shape[0]),
        packed,
    )
    if active_workspace.rows != int(inputs.shape[0]):
        raise ValueError("workspace row extent does not match inputs")
    return _load_extension().linear(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.qweight,
        packed.weight_scales,
        packed.bias,
        packed.output_features,
    )


def bind_convrot_w4a4_rowwise_linear(
    packed: PackedConvRotW4A4Rowwise,
    workspace: ConvRotW4A4RowwiseWorkspace,
    *,
    rows: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind rowwise packed state so the hot call only receives activation."""

    if workspace.rows != int(rows):
        raise ValueError("workspace row extent does not match rows")
    return _load_extension().bind_linear(
        workspace.quantized_activation,
        workspace.activation_scales,
        packed.qweight,
        packed.weight_scales,
        packed.bias,
        int(rows),
        int(packed.input_features),
        int(packed.output_features),
    )


def bind_dynamic_convrot_w4a4_rowwise_linear(
    packed: PackedConvRotW4A4Rowwise,
    *,
    source_weight: torch.Tensor | None = None,
    source_weight_scales: torch.Tensor | None = None,
    source_bias: torch.Tensor | None = None,
    expected_dtype: torch.dtype | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind weights while C++ owns per-shape, per-stream activation workspaces."""

    expected_scalar_kind = (
        -1
        if expected_dtype is None
        else 0
        if expected_dtype == torch.float16
        else 1
        if expected_dtype == torch.bfloat16
        else -2
    )
    if expected_scalar_kind == -2:
        raise ValueError("expected_dtype must be float16, bfloat16, or None")
    return _load_extension().bind_dynamic_linear(
        packed.qweight,
        packed.weight_scales,
        packed.bias,
        int(packed.input_features),
        int(packed.output_features),
        source_weight,
        source_weight_scales,
        source_bias,
        expected_scalar_kind,
    )


__all__ = [
    "ConvRotW4A4RowwiseWorkspace",
    "PackedConvRotW4A4Rowwise",
    "allocate_convrot_w4a4_rowwise_workspace",
    "bind_convrot_w4a4_rowwise_linear",
    "bind_dynamic_convrot_w4a4_rowwise_linear",
    "convrot_w4a4_rowwise_linear",
    "native_rowwise_convrot_w4a4_available",
    "native_rowwise_convrot_w4a4_shape_supported",
    "native_rowwise_convrot_w4a4_version",
    "pack_convrot_w4a4_rowwise_weight",
    "wrap_convrot_w4a4_rowwise_weight",
]
