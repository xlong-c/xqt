"""Native Nunchaku-style W4A4 and SVDQuant kernels for Ada ``sm_89``."""

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

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
_NUNCHAKU_INCLUDE = _REPO_ROOT / "learn" / "nunchaku"
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_VECTOR_ALIGNMENT = 4
_SUPPORTED_ROTATION_SIZES = frozenset({1, 4, 16, 64, 256})


def _round_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _require_native_shape(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise XQTBackendError(f"{name} must be a 2D tensor")
    if not tensor.is_cuda:
        raise XQTBackendError(f"{name} must be CUDA")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise XQTBackendError(f"{name} must be float16 or bfloat16")


def native_w4a4_shape_supported(
    input_features: int,
    output_features: int,
) -> bool:
    """Return whether logical feature strides are safe for vectorized IO."""

    return (
        int(input_features) > 0
        and int(output_features) > 0
        and int(input_features) % _VECTOR_ALIGNMENT == 0
        and int(output_features) % _VECTOR_ALIGNMENT == 0
    )


def native_convrot_w4a4_shape_supported(
    logical_input_features: int,
    rotated_input_features: int,
    output_features: int,
    rot_size: int,
) -> bool:
    """Return whether fused Hadamard rotation can feed native W4A4 safely."""

    logical_k = int(logical_input_features)
    rotated_k = int(rotated_input_features)
    rotation = int(rot_size)
    return (
        logical_k > 0
        and logical_k <= rotated_k
        and rotation in _SUPPORTED_ROTATION_SIZES
        and rotated_k % rotation == 0
        and native_w4a4_shape_supported(rotated_k, output_features)
    )


def _require_vector_aligned_features(
    input_features: int,
    output_features: int,
) -> None:
    if native_w4a4_shape_supported(input_features, output_features):
        return
    raise XQTBackendError(
        "native W4A4 requires input_features and output_features to be "
        f"multiples of {_VECTOR_ALIGNMENT} for vectorized activation IO"
    )


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    if os.environ.get("XQT_DISABLE_SVDQ_W4A4_SM89", "0") == "1":
        raise XQTBackendError("native sm_89 W4A4 backend is disabled by environment")
    if not torch.cuda.is_available():
        raise XQTBackendError("native sm_89 W4A4 backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"native W4A4 backend currently targets sm_89, got sm_{major}{minor}"
        )
    if not _NUNCHAKU_INCLUDE.is_dir():
        raise XQTBackendError(
            f"Nunchaku kernel headers are missing at {_NUNCHAKU_INCLUDE}"
        )

    from torch.utils.cpp_extension import load

    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_max_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ.setdefault("MAX_JOBS", "1")
    try:
        return load(
            name="xqt_svdq_w4a4_sm89_v2",
            sources=[
                str(_HERE / "svdq_w4a4_sm89_binding.cpp"),
                str(_HERE / "svdq_w4a4_sm89_kernel.cu"),
            ],
            extra_include_paths=[str(_NUNCHAKU_INCLUDE)],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
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
            ],
            with_cuda=True,
            verbose=False,
        )
    except Exception as exc:
        raise XQTBackendError(
            f"failed to build native sm_89 W4A4 extension: {exc}"
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


def native_w4a4_available(*, build: bool = False) -> bool:
    """Return whether the native backend can run on the current device."""

    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    if os.environ.get("XQT_DISABLE_SVDQ_W4A4_SM89", "0") == "1":
        return False
    if not build:
        return _NUNCHAKU_INCLUDE.is_dir()
    try:
        _load_extension()
    except XQTBackendError:
        return False
    return True


def native_w4a4_version() -> str:
    return str(_load_extension().version())


def pack_scale(values: torch.Tensor, *, warp_n: int = 128) -> torch.Tensor:
    """Pack one per-channel FP16/BF16 vector in the official scale layout."""

    if values.ndim != 1:
        raise ValueError("scale values must be one-dimensional")
    if values.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("scale values must be float16 or bfloat16")
    if int(values.numel()) % int(warp_n) != 0:
        raise ValueError("scale length must be padded to a multiple of warp_n")
    pack_size = min(max(int(warp_n) // 32, 2), 8)
    lanes = min(32, int(warp_n) // pack_size)
    packs = int(warp_n) // (pack_size * lanes)
    packed = values.reshape(
        int(values.numel()) // int(warp_n),
        packs,
        lanes // 4,
        pack_size // 2,
        4,
        2,
        1,
    )
    return packed.permute(0, 6, 1, 2, 4, 3, 5).contiguous().view(-1)


def pack_lowrank_weight(weight: torch.Tensor, *, down: bool) -> torch.Tensor:
    """Pack padded LoRA/SVD factors for the official MMA fragment layout."""

    if weight.ndim != 2:
        raise ValueError("low-rank weight must be two-dimensional")
    if weight.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("low-rank weight must be float16 or bfloat16")
    frag_n = 16
    frag_k = 16
    weight = F.pad(
        weight,
        (0, (-int(weight.shape[1])) % frag_k, 0, (-int(weight.shape[0])) % frag_n),
    )
    if down:
        rank, channels = (int(dim) for dim in weight.shape)
        rank_frags = rank // frag_n
        channel_frags = channels // frag_k
        packed = weight.view(rank_frags, frag_n, channel_frags, frag_k).permute(
            2, 0, 1, 3
        )
    else:
        channels, rank = (int(dim) for dim in weight.shape)
        channel_frags = channels // frag_n
        rank_frags = rank // frag_k
        packed = weight.view(channel_frags, frag_n, rank_frags, frag_k).permute(
            0, 2, 1, 3
        )
    packed = packed.reshape(channel_frags, rank_frags, 2, 8, 2, 4, 2)
    return packed.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
        channels, rank
    )


@dataclass(frozen=True)
class PackedW4A4Linear:
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    packed_bias: torch.Tensor
    packed_smooth: torch.Tensor
    input_features: int
    output_features: int
    padded_input_features: int
    padded_output_features: int


@dataclass(frozen=True)
class PackedSVDQW4A4Linear(PackedW4A4Linear):
    packed_down: torch.Tensor
    packed_up: torch.Tensor
    rank: int
    padded_rank: int


@dataclass
class W4A4Workspace:
    quantized_activation: torch.Tensor
    activation_scales: torch.Tensor
    lora_activation: torch.Tensor | None = None

    @property
    def padded_rows(self) -> int:
        return int(self.quantized_activation.shape[0])


def pack_w4a4_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> PackedW4A4Linear:
    """Quantize and prepack a dense residual weight for native W4A4 GEMM."""

    _require_native_shape(weight, "weight")
    output_features, input_features = (int(dim) for dim in weight.shape)
    _require_vector_aligned_features(input_features, output_features)
    padded_output = _round_up(output_features, 128)
    padded_input = _round_up(input_features, 128)
    padded_weight = F.pad(
        weight.contiguous(),
        (0, padded_input - input_features, 0, padded_output - output_features),
    )
    qweight = torch.empty(
        (padded_output, padded_input // 2),
        dtype=torch.int8,
        device=weight.device,
    )
    weight_scales = torch.empty(
        (padded_input // 64, padded_output),
        dtype=weight.dtype,
        device=weight.device,
    )
    _load_extension().quantize_weight(padded_weight, qweight, weight_scales)
    if bias is None:
        padded_bias = torch.zeros(
            padded_output, dtype=weight.dtype, device=weight.device
        )
    else:
        if bias.ndim != 1 or int(bias.numel()) != output_features:
            raise ValueError("bias must match weight output features")
        padded_bias = F.pad(
            bias.to(device=weight.device, dtype=weight.dtype).contiguous(),
            (0, padded_output - output_features),
        )
    return PackedW4A4Linear(
        qweight=qweight,
        weight_scales=weight_scales,
        packed_bias=pack_scale(padded_bias),
        packed_smooth=pack_scale(
            torch.ones(
                padded_input, dtype=weight.dtype, device=weight.device
            )
        ),
        input_features=input_features,
        output_features=output_features,
        padded_input_features=padded_input,
        padded_output_features=padded_output,
    )


def pack_svdq_w4a4_linear(
    residual_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    smooth: torch.Tensor | None = None,
) -> PackedSVDQW4A4Linear:
    """Prepack the residual and low-rank branches for two-stage SVDQuant."""

    base = pack_w4a4_linear(residual_weight, bias)
    if down_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("down_weight and up_weight must be two-dimensional")
    rank = int(down_weight.shape[0])
    if int(down_weight.shape[1]) != base.input_features:
        raise ValueError("down_weight must be [rank, input_features]")
    if tuple(int(dim) for dim in up_weight.shape) != (base.output_features, rank):
        raise ValueError("up_weight must be [output_features, rank]")
    padded_rank = _round_up(rank, 16)
    if padded_rank > 1024:
        raise ValueError("native SVDQuant supports rank up to 1024")
    down = F.pad(
        down_weight.to(device=residual_weight.device, dtype=residual_weight.dtype),
        (
            0,
            base.padded_input_features - base.input_features,
            0,
            padded_rank - rank,
        ),
    )
    up = F.pad(
        up_weight.to(device=residual_weight.device, dtype=residual_weight.dtype),
        (0, padded_rank - rank, 0, base.padded_output_features - base.output_features),
    )
    if smooth is None:
        smooth_values = torch.ones(
            base.padded_input_features,
            dtype=residual_weight.dtype,
            device=residual_weight.device,
        )
    else:
        if smooth.ndim != 1 or int(smooth.numel()) != base.input_features:
            raise ValueError("smooth must match input_features")
        smooth_values = F.pad(
            smooth.to(device=residual_weight.device, dtype=residual_weight.dtype),
            (0, base.padded_input_features - base.input_features),
            value=1.0,
        )
    base_values = dict(base.__dict__)
    base_values["packed_smooth"] = pack_scale(smooth_values)
    return PackedSVDQW4A4Linear(
        **base_values,
        packed_down=pack_lowrank_weight(down, down=True),
        packed_up=pack_lowrank_weight(up, down=False),
        rank=rank,
        padded_rank=padded_rank,
    )


def allocate_w4a4_workspace(
    rows: int,
    packed: PackedW4A4Linear,
    *,
    with_lora_rank: int = 0,
) -> W4A4Workspace:
    padded_rows = _round_up(int(rows), 256)
    qactivation = torch.empty(
        (padded_rows, packed.padded_input_features // 2),
        dtype=torch.int8,
        device=packed.qweight.device,
    )
    activation_scales = torch.empty(
        (packed.padded_input_features // 64, padded_rows),
        dtype=packed.weight_scales.dtype,
        device=packed.qweight.device,
    )
    lora_activation = None
    if int(with_lora_rank) > 0:
        lora_activation = torch.empty(
            (padded_rows, int(with_lora_rank)),
            dtype=torch.float32,
            device=packed.qweight.device,
        )
    return W4A4Workspace(
        quantized_activation=qactivation,
        activation_scales=activation_scales,
        lora_activation=lora_activation,
    )


def w4a4_linear(
    inputs: torch.Tensor,
    packed: PackedW4A4Linear,
    *,
    workspace: W4A4Workspace | None = None,
    smooth: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dynamic activation INT4 quantization followed by native W4A4 GEMM."""

    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_w4a4_workspace(
        int(inputs.shape[0]), packed
    )
    if active_workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")
    if smooth is None:
        packed_smooth = packed.packed_smooth
    else:
        packed_smooth = smooth
    extension = _load_extension()
    return extension.linear(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed_smooth,
        packed.qweight,
        packed.weight_scales,
        packed.packed_bias,
        packed.output_features,
    )


def convrot_w4a4_linear(
    inputs: torch.Tensor,
    packed: PackedW4A4Linear,
    *,
    rot_size: int,
    workspace: W4A4Workspace | None = None,
) -> torch.Tensor:
    """Fuse regular-Hadamard rotation with dynamic INT4 activation packing."""

    _require_native_shape(inputs, "inputs")
    if not native_convrot_w4a4_shape_supported(
        int(inputs.shape[1]),
        packed.input_features,
        packed.output_features,
        int(rot_size),
    ):
        raise XQTBackendError(
            "input, rotated feature, output, or rot_size is unsupported by native ConvRot W4A4"
        )
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_w4a4_workspace(
        int(inputs.shape[0]),
        packed,
    )
    if active_workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")
    extension = _load_extension()
    return extension.convrot_linear(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.qweight,
        packed.weight_scales,
        packed.packed_bias,
        int(packed.input_features),
        int(rot_size),
        packed.output_features,
    )


def bind_convrot_w4a4_linear(
    packed: PackedW4A4Linear,
    workspace: W4A4Workspace,
    *,
    rows: int,
    logical_input_features: int,
    rotated_input_features: int,
    rot_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind validated ConvRot state for the steady-state W4A4 hot path."""

    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    if not native_convrot_w4a4_shape_supported(
        int(logical_input_features),
        int(rotated_input_features),
        int(packed.output_features),
        int(rot_size),
    ):
        raise XQTBackendError(
            "input, rotated feature, output, or rot_size is unsupported by native ConvRot W4A4"
        )
    if int(rotated_input_features) != int(packed.input_features):
        raise ValueError("rotated_input_features must match packed input_features")
    extension = _load_extension()
    return extension.bind_convrot_linear(
        workspace.quantized_activation,
        workspace.activation_scales,
        packed.qweight,
        packed.weight_scales,
        packed.packed_bias,
        int(rows),
        int(logical_input_features),
        int(rotated_input_features),
        int(packed.output_features),
        int(rot_size),
    )


def svdq_w4a4_linear(
    inputs: torch.Tensor,
    packed: PackedSVDQW4A4Linear,
    *,
    workspace: W4A4Workspace | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Run the Nunchaku two-stage dynamic W4A4 plus low-rank fusion."""

    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_w4a4_workspace(
        int(inputs.shape[0]), packed, with_lora_rank=packed.padded_rank
    )
    if active_workspace.lora_activation is None:
        raise ValueError("SVDQuant workspace requires lora_activation")
    if active_workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")
    extension = _load_extension()
    return extension.svdq_linear(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.packed_down,
        active_workspace.lora_activation,
        packed.packed_smooth,
        packed.qweight,
        packed.weight_scales,
        packed.packed_up,
        packed.packed_bias,
        packed.output_features,
        float(lora_scale),
    )


def bind_svdq_w4a4_linear(
    packed: PackedSVDQW4A4Linear,
    workspace: W4A4Workspace,
    *,
    rows: int,
    lora_scale: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind validated packed state for the steady-state SVDQuant hot path."""

    if workspace.lora_activation is None:
        raise ValueError("SVDQuant workspace requires lora_activation")
    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    extension = _load_extension()
    return extension.bind_svdq_linear(
        workspace.quantized_activation,
        workspace.activation_scales,
        packed.packed_down,
        workspace.lora_activation,
        packed.packed_smooth,
        packed.qweight,
        packed.weight_scales,
        packed.packed_up,
        packed.packed_bias,
        int(rows),
        int(packed.input_features),
        int(packed.output_features),
        float(lora_scale),
    )


__all__ = [
    "PackedSVDQW4A4Linear",
    "PackedW4A4Linear",
    "W4A4Workspace",
    "allocate_w4a4_workspace",
    "bind_convrot_w4a4_linear",
    "bind_svdq_w4a4_linear",
    "convrot_w4a4_linear",
    "native_convrot_w4a4_shape_supported",
    "native_w4a4_available",
    "native_w4a4_shape_supported",
    "native_w4a4_version",
    "pack_lowrank_weight",
    "pack_scale",
    "pack_svdq_w4a4_linear",
    "pack_w4a4_linear",
    "svdq_w4a4_linear",
    "w4a4_linear",
]
