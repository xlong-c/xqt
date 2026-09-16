"""Native Nunchaku-style BF16 W8A8 SVDQuant kernels for Ada ``sm_89``."""

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
from xqt.kernels.ops._impl.cute.svdq_w4a4_sm89 import (
    pack_lowrank_weight,
    pack_scale,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[4]
_NUNCHAKU_INCLUDE = _REPO_ROOT / "learn" / "nunchaku"
_VECTOR_ALIGNMENT = 4
_CUDA_SOURCE = csrc_path("quantization", "svdq_w8a8_sm89_kernel.cu")
_BINDING_SOURCE = csrc_path("quantization", "svdq_w8a8_sm89_binding.cpp")
_TVM_BINDING_SOURCE = csrc_path("quantization", "svdq_w8a8_sm89_tvm_binding.cpp")


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def native_svdq_w8a8_shape_supported(
    input_features: int,
    output_features: int,
    rank: int,
) -> bool:
    return (
        int(input_features) > 0
        and int(output_features) > 0
        and int(input_features) % _VECTOR_ALIGNMENT == 0
        and int(output_features) % _VECTOR_ALIGNMENT == 0
        and 0 < int(rank) <= 1024
    )


def _require_bf16_cuda_2d(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise XQTBackendError(f"{name} must be a 2D tensor")
    if not tensor.is_cuda:
        raise XQTBackendError(f"{name} must be CUDA")
    if tensor.dtype != torch.bfloat16:
        raise XQTBackendError(f"{name} must be bfloat16")


@lru_cache(maxsize=2)
def _load_extension(backend: str | None = None) -> Any:
    selected_backend = backend or os.environ.get("XQT_JIT_BACKEND", "tvm_ffi")
    if os.environ.get("XQT_DISABLE_SVDQ_W8A8_SM89", "0") == "1":
        raise XQTBackendError("native sm_89 SVDQuant W8A8 backend is disabled")
    if not torch.cuda.is_available():
        raise XQTBackendError("native sm_89 SVDQuant W8A8 backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"native SVDQuant W8A8 currently targets sm_89, got sm_{major}{minor}"
        )
    if not _NUNCHAKU_INCLUDE.is_dir():
        raise XQTBackendError(
            f"Nunchaku kernel headers are missing at {_NUNCHAKU_INCLUDE}"
        )

    binding_source = (
        _TVM_BINDING_SOURCE if selected_backend == "tvm_ffi" else _BINDING_SOURCE
    )
    ext_suffix = "_tvm_v1" if selected_backend == "tvm_ffi" else "_v1"

    try:
        return load_extension(
            CompileSpec(
                name=f"xqt_svdq_w8a8_sm89{ext_suffix}",
                sources=(binding_source, _CUDA_SOURCE),
                include_dirs=(_NUNCHAKU_INCLUDE,),
                cxx_flags=("-O3", "-std=c++20"),
                cuda_flags=(
                    "-O3",
                    "-std=c++20",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DENABLE_BF16=1",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--generate-line-info",
                ),
                target_arch="sm_89",
                backend=selected_backend,
            ),
            verbose=False,
        )
    except Exception as exc:
        raise XQTBackendError(
            f"failed to build native sm_89 SVDQuant W8A8 extension: {exc}"
        ) from exc


def native_svdq_w8a8_available(*, build: bool = False) -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    if os.environ.get("XQT_DISABLE_SVDQ_W8A8_SM89", "0") == "1":
        return False
    if not build:
        return _NUNCHAKU_INCLUDE.is_dir()
    try:
        _load_extension()
    except XQTBackendError:
        return False
    return True


def native_svdq_w8a8_version() -> str:
    return str(_load_extension().version())


@dataclass(frozen=True)
class PackedSVDQW8A8Linear:
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    packed_bias: torch.Tensor
    packed_down: torch.Tensor
    packed_up: torch.Tensor
    input_features: int
    output_features: int
    padded_input_features: int
    padded_output_features: int
    rank: int
    padded_rank: int


@dataclass
class W8A8SVDQWorkspace:
    quantized_activation: torch.Tensor
    activation_scales: torch.Tensor
    lora_activation: torch.Tensor

    @property
    def padded_rows(self) -> int:
        return int(self.quantized_activation.shape[0])


def pack_svdq_w8a8_linear(
    residual_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> PackedSVDQW8A8Linear:
    _require_bf16_cuda_2d(residual_weight, "residual_weight")
    _require_bf16_cuda_2d(down_weight, "down_weight")
    _require_bf16_cuda_2d(up_weight, "up_weight")
    output_features, input_features = (int(dim) for dim in residual_weight.shape)
    rank = int(down_weight.shape[0])
    if not native_svdq_w8a8_shape_supported(
        input_features,
        output_features,
        rank,
    ):
        raise XQTBackendError(
            "native SVDQuant W8A8 requires N/K multiples of 4 and rank <= 1024"
        )
    if tuple(int(dim) for dim in down_weight.shape) != (rank, input_features):
        raise ValueError("down_weight must be [rank, input_features]")
    if tuple(int(dim) for dim in up_weight.shape) != (output_features, rank):
        raise ValueError("up_weight must be [output_features, rank]")

    padded_input = _round_up(input_features, 128)
    padded_output = _round_up(output_features, 128)
    padded_rank = _round_up(rank, 16)
    padded_weight = F.pad(
        residual_weight.contiguous(),
        (0, padded_input - input_features, 0, padded_output - output_features),
    )
    maximum = padded_weight.abs().amax(dim=1)
    raw_scales = torch.where(
        maximum > 0,
        maximum / 127.0,
        torch.ones_like(maximum),
    ).to(torch.bfloat16)
    qweight = torch.empty_like(padded_weight, dtype=torch.int8)
    _load_extension().quantize_weight(padded_weight, raw_scales, qweight)

    down = F.pad(
        down_weight,
        (0, padded_input - input_features, 0, padded_rank - rank),
    )
    up = F.pad(
        up_weight,
        (0, padded_rank - rank, 0, padded_output - output_features),
    )
    if bias is None:
        padded_bias = torch.zeros(
            padded_output,
            dtype=torch.bfloat16,
            device=residual_weight.device,
        )
    else:
        if bias.ndim != 1 or int(bias.numel()) != output_features:
            raise ValueError("bias must match output_features")
        padded_bias = F.pad(
            bias.to(device=residual_weight.device, dtype=torch.bfloat16),
            (0, padded_output - output_features),
        )
    return PackedSVDQW8A8Linear(
        qweight=qweight,
        weight_scales=pack_scale(raw_scales),
        packed_bias=pack_scale(padded_bias),
        packed_down=pack_lowrank_weight(down, down=True),
        packed_up=pack_lowrank_weight(up, down=False),
        input_features=input_features,
        output_features=output_features,
        padded_input_features=padded_input,
        padded_output_features=padded_output,
        rank=rank,
        padded_rank=padded_rank,
    )


def allocate_svdq_w8a8_workspace(
    rows: int,
    packed: PackedSVDQW8A8Linear,
) -> W8A8SVDQWorkspace:
    padded_rows = _round_up(int(rows), 256)
    return W8A8SVDQWorkspace(
        quantized_activation=torch.empty(
            (padded_rows, packed.padded_input_features),
            dtype=torch.int8,
            device=packed.qweight.device,
        ),
        activation_scales=torch.empty(
            padded_rows,
            dtype=torch.bfloat16,
            device=packed.qweight.device,
        ),
        lora_activation=torch.empty(
            (padded_rows, packed.padded_rank),
            dtype=torch.float32,
            device=packed.qweight.device,
        ),
    )


def svdq_w8a8_linear(
    inputs: torch.Tensor,
    packed: PackedSVDQW8A8Linear,
    *,
    workspace: W8A8SVDQWorkspace | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    _require_bf16_cuda_2d(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device:
        raise XQTBackendError("inputs and packed weights must share a CUDA device")
    active_workspace = workspace or allocate_svdq_w8a8_workspace(
        int(inputs.shape[0]),
        packed,
    )
    if active_workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")
    extension = _load_extension()
    if hasattr(extension, "svdq_linear"):
        return extension.svdq_linear(
            inputs.contiguous(),
            active_workspace.quantized_activation,
            active_workspace.activation_scales,
            packed.packed_down,
            active_workspace.lora_activation,
            packed.qweight,
            packed.weight_scales,
            packed.packed_up,
            packed.packed_bias,
            packed.output_features,
            float(lora_scale),
        )

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=torch.bfloat16, device=inputs.device)
    extension.quantize_act_lora(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.packed_down,
        active_workspace.lora_activation,
    )
    extension.gemm_lora(
        active_workspace.quantized_activation,
        packed.qweight,
        output,
        active_workspace.activation_scales,
        packed.weight_scales,
        active_workspace.lora_activation,
        packed.packed_up,
        packed.packed_bias,
        float(lora_scale),
    )
    return output


def bind_svdq_w8a8_linear(
    packed: PackedSVDQW8A8Linear,
    workspace: W8A8SVDQWorkspace,
    *,
    rows: int,
    lora_scale: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind validated packed state for the steady-state BF16 hot path."""

    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    extension = _load_extension()
    quantized_activation = workspace.quantized_activation
    activation_scales = workspace.activation_scales
    lora_activation = workspace.lora_activation
    packed_down = packed.packed_down
    qweight = packed.qweight
    weight_scales = packed.weight_scales
    packed_up = packed.packed_up
    packed_bias = packed.packed_bias
    output_features = packed.output_features
    scale = float(lora_scale)

    if hasattr(extension, "svdq_linear"):
        def run(inputs: torch.Tensor) -> torch.Tensor:
            return extension.svdq_linear(
                inputs.contiguous(),
                quantized_activation,
                activation_scales,
                packed_down,
                lora_activation,
                qweight,
                weight_scales,
                packed_up,
                packed_bias,
                output_features,
                scale,
            )

        return run

    def run_tvm(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty((inputs.shape[0], output_features), dtype=torch.bfloat16, device=inputs.device)
        extension.quantize_act_lora(
            inputs.contiguous(),
            quantized_activation,
            activation_scales,
            packed_down,
            lora_activation,
        )
        extension.gemm_lora(
            quantized_activation,
            qweight,
            output,
            activation_scales,
            weight_scales,
            lora_activation,
            packed_up,
            packed_bias,
            scale,
        )
        return output

    return run_tvm


def w8a8_linear(
    inputs: torch.Tensor,
    packed: PackedSVDQW8A8Linear,
    *,
    workspace: W8A8SVDQWorkspace | None = None,
) -> torch.Tensor:
    """Run the same dynamic W8A8 path without either low-rank epilogue."""

    _require_bf16_cuda_2d(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device:
        raise XQTBackendError("inputs and packed weights must share a CUDA device")
    active_workspace = workspace or allocate_svdq_w8a8_workspace(
        int(inputs.shape[0]),
        packed,
    )
    if active_workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")
    extension = _load_extension()
    if hasattr(extension, "linear"):
        return extension.linear(
            inputs.contiguous(),
            active_workspace.quantized_activation,
            active_workspace.activation_scales,
            packed.qweight,
            packed.weight_scales,
            packed.packed_bias,
            packed.output_features,
        )

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=torch.bfloat16, device=inputs.device)
    extension.quantize_act(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
    )
    extension.gemm(
        active_workspace.quantized_activation,
        packed.qweight,
        output,
        active_workspace.activation_scales,
        packed.weight_scales,
        packed.packed_bias,
    )
    return output


__all__ = [
    "PackedSVDQW8A8Linear",
    "W8A8SVDQWorkspace",
    "allocate_svdq_w8a8_workspace",
    "bind_svdq_w8a8_linear",
    "native_svdq_w8a8_available",
    "native_svdq_w8a8_shape_supported",
    "native_svdq_w8a8_version",
    "pack_svdq_w8a8_linear",
    "svdq_w8a8_linear",
    "w8a8_linear",
]
