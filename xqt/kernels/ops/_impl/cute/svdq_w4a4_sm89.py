"""Native Nunchaku-style W4A4 and SVDQuant kernels for Ada ``sm_89``."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.kernels.jit.utils.compile import (
    CompileSpec,
    csrc_path,
    include_root,
    load_extension,
)

_NUNCHAKU_INCLUDE = include_root() / "nunchaku"
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_VECTOR_ALIGNMENT = 4
_SUPPORTED_ROTATION_SIZES = frozenset({1, 4, 16, 64, 256})
_QKV_HEAD_DIM = 128
_CUDA_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_kernel.cu")
_NORM_CUDA_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_norm_kernels.cu")
_SMALLN_CUDA_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_smalln_kernel.cu")
_BINDING_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_binding.cpp")
_SMALLN_BINDING_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_smalln_binding.cpp")


_TVM_BINDING_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_tvm_binding.cpp")


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


@lru_cache(maxsize=2)
def _load_extension(backend: str | None = None) -> Any:
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

    selected_backend = backend or os.environ.get("XQT_JIT_BACKEND", "tvm_ffi")
    binding_source = (
        _TVM_BINDING_SOURCE if selected_backend == "tvm_ffi" else _BINDING_SOURCE
    )
    ext_suffix = "_tvm_v6" if selected_backend == "tvm_ffi" else "_v6"

    try:
        return load_extension(
            CompileSpec(
                name=f"xqt_svdq_w4a4_sm89{ext_suffix}",
                sources=(
                    binding_source,
                    _CUDA_SOURCE,
                    _NORM_CUDA_SOURCE,
                ),
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
            f"failed to build native sm_89 W4A4 extension: {exc}"
        ) from exc


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


def pack_svdq_w4a4_rotary_emb(
    rotary_emb: torch.Tensor,
    *,
    rows: int,
    padded_rows: int | None = None,
) -> torch.Tensor:
    """Pack raw RoPE sin/cos values for the fused QKV epilogue.

    ``rotary_emb`` must contain one raw RoPE record per flattened input row,
    with shape ``[B, S, 64, 1, 2]`` or ``[M, 64, 1, 2]``. Padding is applied
    after flattening so its row extent exactly matches the W4A4 workspace.
    """

    actual_rows = int(rows)
    target_rows = (
        _round_up(actual_rows, 256)
        if padded_rows is None
        else int(padded_rows)
    )
    if actual_rows <= 0:
        raise ValueError("rows must be positive")
    if target_rows < actual_rows or target_rows % 16 != 0:
        raise ValueError("padded_rows must cover rows and be divisible by 16")
    if rotary_emb.dtype != torch.float32:
        raise ValueError("rotary_emb must be float32")
    if rotary_emb.ndim == 5:
        flattened_rows = int(rotary_emb.shape[0]) * int(rotary_emb.shape[1])
    elif rotary_emb.ndim == 4:
        flattened_rows = int(rotary_emb.shape[0])
    else:
        raise ValueError(
            "rotary_emb must have shape [B, S, 64, 1, 2] or [M, 64, 1, 2]"
        )
    if tuple(int(dim) for dim in rotary_emb.shape[-3:]) != (
        _QKV_HEAD_DIM // 2,
        1,
        2,
    ):
        raise ValueError("rotary_emb trailing shape must be [64, 1, 2]")
    if flattened_rows != actual_rows:
        raise ValueError("rotary_emb row count must match rows")

    flattened = rotary_emb.reshape(
        1,
        actual_rows,
        _QKV_HEAD_DIM // 2,
        1,
        2,
    )
    if target_rows != actual_rows:
        padded = torch.zeros(
            (1, target_rows, _QKV_HEAD_DIM // 2, 1, 2),
            dtype=rotary_emb.dtype,
            device=rotary_emb.device,
        )
        padded[:, :actual_rows].copy_(flattened)
        flattened = padded

    packed = flattened.reshape(
        1,
        target_rows // 16,
        16,
        _QKV_HEAD_DIM // 8,
        8,
    )
    packed = packed.permute(0, 1, 3, 2, 4)
    packed = packed.reshape(*packed.shape[:3], 2, 8, 4, 2)
    packed = packed.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
    return packed.view(1, target_rows, _QKV_HEAD_DIM)


_SMALLN_TVM_BINDING_SOURCE = csrc_path("quantization", "svdq_w4a4_sm89_smalln_tvm_binding.cpp")


@lru_cache(maxsize=2)
def _load_smalln_extension(backend: str | None = None) -> Any:
    if os.environ.get("XQT_DISABLE_SVDQ_W4A4_SM89", "0") == "1":
        raise XQTBackendError("native sm_89 W4A4 backend is disabled by environment")
    if not torch.cuda.is_available():
        raise XQTBackendError("native sm_89 W4A4 small-N backend requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise XQTBackendError(
            f"native W4A4 small-N backend currently targets sm_89, got sm_{major}{minor}"
        )
    if not _NUNCHAKU_INCLUDE.is_dir():
        raise XQTBackendError(
            f"Nunchaku kernel headers are missing at {_NUNCHAKU_INCLUDE}"
        )

    selected_backend = backend or os.environ.get("XQT_JIT_BACKEND", "tvm_ffi")
    binding_source = (
        _SMALLN_TVM_BINDING_SOURCE if selected_backend == "tvm_ffi" else _SMALLN_BINDING_SOURCE
    )
    ext_suffix = "_tvm_v3" if selected_backend == "tvm_ffi" else "_v3"

    try:
        return load_extension(
            CompileSpec(
                name=f"xqt_svdq_w4a4_sm89_smalln{ext_suffix}",
                sources=(
                    binding_source,
                    _SMALLN_CUDA_SOURCE,
                    _NORM_CUDA_SOURCE,
                ),
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
            f"failed to build native sm_89 W4A4 small-N extension: {exc}"
        ) from exc


def native_w4a4_smalln_available(*, build: bool = False) -> bool:
    """Return whether the small-BLOCK_N (64) W4A4 GEMM variant can run."""

    if not native_w4a4_available(build=False):
        return False
    if not build:
        return _SMALLN_CUDA_SOURCE.is_file()
    try:
        _load_smalln_extension()
    except XQTBackendError:
        return False
    return True


def native_w4a4_smalln_version() -> str:
    return str(_load_smalln_extension().version())


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
    norm_weight: torch.Tensor | None
    rank: int
    padded_rank: int


@dataclass
class W4A4Workspace:
    quantized_activation: torch.Tensor
    activation_scales: torch.Tensor
    lora_activation: torch.Tensor | None = None
    row_scales: torch.Tensor | None = None

    @property
    def padded_rows(self) -> int:
        return int(self.quantized_activation.shape[0])


@dataclass(frozen=True)
class SVDQW4A4GeluMLPWorkspace:
    fc1: W4A4Workspace
    fc2: W4A4Workspace


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
    norm_weight: torch.Tensor | None = None,
) -> PackedSVDQW4A4Linear:
    """Prepack the residual and low-rank branches for two-stage SVDQuant."""

    base = pack_w4a4_linear(residual_weight, bias)
    return _pack_svdq_lora_parts(
        base,
        down_weight,
        up_weight,
        smooth,
        norm_weight,
    )


def _pack_svdq_lora_parts(
    base: PackedW4A4Linear,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    smooth: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
) -> PackedSVDQW4A4Linear:
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
        down_weight.to(device=base.qweight.device, dtype=base.weight_scales.dtype),
        (
            0,
            base.padded_input_features - base.input_features,
            0,
            padded_rank - rank,
        ),
    )
    up = F.pad(
        up_weight.to(device=base.qweight.device, dtype=base.weight_scales.dtype),
        (0, padded_rank - rank, 0, base.padded_output_features - base.output_features),
    )
    if smooth is None:
        smooth_values = torch.ones(
            base.padded_input_features,
            dtype=base.weight_scales.dtype,
            device=base.qweight.device,
        )
    else:
        if smooth.ndim != 1 or int(smooth.numel()) != base.input_features:
            raise ValueError("smooth must match input_features")
        smooth_values = F.pad(
            smooth.to(device=base.qweight.device, dtype=base.weight_scales.dtype),
            (0, base.padded_input_features - base.input_features),
            value=1.0,
        )
    base_values = dict(base.__dict__)
    base_values["packed_smooth"] = pack_scale(smooth_values)
    padded_norm_weight = None
    if norm_weight is not None:
        if norm_weight.ndim != 1 or int(norm_weight.numel()) != base.input_features:
            raise ValueError("norm_weight must match input_features")
        padded_norm_weight = F.pad(
            norm_weight.to(
                device=base.qweight.device,
                dtype=base.weight_scales.dtype,
            ),
            (0, base.padded_input_features - base.input_features),
            value=1.0,
        ).contiguous()
    return PackedSVDQW4A4Linear(
        **base_values,
        packed_down=pack_lowrank_weight(down, down=True),
        packed_up=pack_lowrank_weight(up, down=False),
        norm_weight=padded_norm_weight,
        rank=rank,
        padded_rank=padded_rank,
    )


def pack_w4a4_linear_smalln(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> PackedW4A4Linear:
    """Quantize and prepack a dense residual weight for the BLOCK_N=64 GEMM.

    The packed activation and packed LoRA layouts are shared with the base
    BLOCK_N=128 extension, so the same workspace and low-rank packs apply;
    only the residual weight/scales/bias use the WARP_N=64 fragment layout.
    """

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
    _load_smalln_extension().quantize_weight(padded_weight, qweight, weight_scales)
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
        packed_bias=pack_scale(padded_bias, warp_n=64),
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


def pack_svdq_w4a4_linear_smalln(
    residual_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    smooth: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
) -> PackedSVDQW4A4Linear:
    """Prepack two-stage SVDQuant for the BLOCK_N=64 GEMM variant."""

    base = pack_w4a4_linear_smalln(residual_weight, bias)
    return _pack_svdq_lora_parts(
        base,
        down_weight,
        up_weight,
        smooth,
        norm_weight,
    )


def smalln_w4a4_beneficial(
    padded_rows: int,
    padded_output_features: int,
) -> bool:
    """Provisional heuristic for choosing the BLOCK_N=64 GEMM.

    The base 128-wide tile yields only ``padded_n / 128`` CTAs on the N axis;
    short-prefill shapes (one 256-row M block) underutilize the SMs, so the
    64-wide tile is expected to win there.  This predicate is provisional
    until per-shape CUDA-event evidence lands in the optimization records.
    """

    return int(padded_rows) <= 256 and int(padded_output_features) >= 512


def _require_smalln_inputs(inputs: torch.Tensor, packed: PackedW4A4Linear) -> W4A4Workspace:
    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    workspace = allocate_w4a4_workspace(int(inputs.shape[0]), packed)
    return workspace


def w4a4_linear_smalln(
    inputs: torch.Tensor,
    packed: PackedW4A4Linear,
    *,
    workspace: W4A4Workspace | None = None,
    smooth: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dynamic INT4 activation quantization plus the BLOCK_N=64 W4A4 GEMM."""

    if workspace is None:
        workspace = _require_smalln_inputs(inputs, packed)
    else:
        _require_native_shape(inputs, "inputs")
        if workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
            raise ValueError("workspace row extent does not match inputs")
    if smooth is None:
        packed_smooth = packed.packed_smooth
    else:
        packed_smooth = smooth
    base_extension = _load_extension()
    smalln_extension = _load_smalln_extension()
    output = torch.empty(
        (int(inputs.shape[0]), packed.output_features),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    base_extension.quantize_act(
        inputs.contiguous(),
        workspace.quantized_activation,
        workspace.activation_scales,
        packed_smooth,
    )
    smalln_extension.gemm(
        workspace.quantized_activation,
        packed.qweight,
        output,
        workspace.activation_scales,
        packed.weight_scales,
        packed.packed_bias,
    )
    return output


def svdq_w4a4_linear_smalln(
    inputs: torch.Tensor,
    packed: PackedSVDQW4A4Linear,
    *,
    workspace: W4A4Workspace | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Two-stage dynamic W4A4 plus low-rank fusion with the BLOCK_N=64 GEMM.

    Mirrors ``svdq_w4a4_linear`` as a single fused C++ call so the op-level
    latency is not dominated by Python-side allocation and pybind overhead.
    """

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
    smalln_extension = _load_smalln_extension()
    if hasattr(smalln_extension, "svdq_linear"):
        return smalln_extension.svdq_linear(
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

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=inputs.dtype, device=inputs.device)
    smalln_extension.quantize_act_lora(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.packed_down,
        active_workspace.lora_activation,
        packed.packed_smooth,
    )
    smalln_extension.gemm_lora(
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


def bind_svdq_w4a4_linear_smalln(
    packed: PackedSVDQW4A4Linear,
    workspace: W4A4Workspace,
    *,
    rows: int,
    lora_scale: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind validated small-N packed state for the steady-state hot path."""

    if workspace.lora_activation is None:
        raise ValueError("SVDQuant workspace requires lora_activation")
    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    smalln_extension = _load_smalln_extension()
    if hasattr(smalln_extension, "bind_svdq_linear"):
        return smalln_extension.bind_svdq_linear(
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

    def run_bound_smalln(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty((int(rows), packed.output_features), dtype=inputs.dtype, device=inputs.device)
        smalln_extension.quantize_act_lora(
            inputs.contiguous(),
            workspace.quantized_activation,
            workspace.activation_scales,
            packed.packed_down,
            workspace.lora_activation,
            packed.packed_smooth,
        )
        smalln_extension.gemm_lora(
            workspace.quantized_activation,
            packed.qweight,
            output,
            workspace.activation_scales,
            packed.weight_scales,
            workspace.lora_activation,
            packed.packed_up,
            packed.packed_bias,
            float(lora_scale),
        )
        return output

    return run_bound_smalln


def svdq_w4a4_linear_smalln_norm(
    inputs: torch.Tensor,
    packed: PackedSVDQW4A4Linear,
    *,
    workspace: W4A4Workspace | None = None,
    lora_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the RMSNorm-fused SVDQuant path with the BLOCK_N=64 GEMM."""

    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_w4a4_workspace(
        int(inputs.shape[0]),
        packed,
        with_lora_rank=packed.padded_rank,
        with_row_scales=True,
    )
    _require_norm_workspace(inputs, active_workspace)
    if packed.norm_weight is None:
        raise ValueError("norm-fused packed state requires norm_weight")
    smalln_extension = _load_smalln_extension()
    if hasattr(smalln_extension, "svdq_linear_norm"):
        return smalln_extension.svdq_linear_norm(
            inputs.contiguous(),
            packed.norm_weight,
            active_workspace.row_scales,
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
            float(eps),
        )

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=inputs.dtype, device=inputs.device)
    smalln_extension.norm_quantize_act_lora(
        inputs.contiguous(),
        packed.norm_weight,
        active_workspace.row_scales,
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.packed_down,
        active_workspace.lora_activation,
        packed.packed_smooth,
    )
    smalln_extension.gemm_lora(
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


def bind_svdq_w4a4_linear_smalln_norm(
    packed: PackedSVDQW4A4Linear,
    workspace: W4A4Workspace,
    *,
    rows: int,
    lora_scale: float = 1.0,
    eps: float = 1e-6,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind the RMSNorm-fused BLOCK_N=64 SVDQuant steady-state runner."""

    if workspace.lora_activation is None or workspace.row_scales is None:
        raise ValueError("SVDQuant norm workspace requires lora_activation and row_scales")
    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    if packed.norm_weight is None:
        raise ValueError("norm-fused packed state requires norm_weight")
    smalln_extension = _load_smalln_extension()
    if hasattr(smalln_extension, "bind_svdq_linear_norm"):
        return smalln_extension.bind_svdq_linear_norm(
            packed.norm_weight,
            workspace.row_scales,
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
            float(eps),
        )

    def run_bound_smalln_norm(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty((int(rows), packed.output_features), dtype=inputs.dtype, device=inputs.device)
        smalln_extension.norm_quantize_act_lora(
            inputs.contiguous(),
            packed.norm_weight,
            workspace.row_scales,
            workspace.quantized_activation,
            workspace.activation_scales,
            packed.packed_down,
            workspace.lora_activation,
            packed.packed_smooth,
        )
        smalln_extension.gemm_lora(
            workspace.quantized_activation,
            packed.qweight,
            output,
            workspace.activation_scales,
            packed.weight_scales,
            workspace.lora_activation,
            packed.packed_up,
            packed.packed_bias,
            float(lora_scale),
        )
        return output

    return run_bound_smalln_norm


def allocate_w4a4_workspace(
    rows: int,
    packed: PackedW4A4Linear,
    *,
    with_lora_rank: int = 0,
    with_row_scales: bool = False,
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
    row_scales = None
    if with_row_scales:
        row_scales = torch.empty(
            (padded_rows,),
            dtype=torch.float32,
            device=packed.qweight.device,
        )
    return W4A4Workspace(
        quantized_activation=qactivation,
        activation_scales=activation_scales,
        lora_activation=lora_activation,
        row_scales=row_scales,
    )


def allocate_svdq_w4a4_gelu_mlp_workspace(
    rows: int,
    fc1: PackedSVDQW4A4Linear,
    fc2: PackedSVDQW4A4Linear,
) -> SVDQW4A4GeluMLPWorkspace:
    """Allocate both activation/LoRA workspaces for a fused GELU MLP."""

    if fc1.output_features != fc2.input_features:
        raise ValueError("fc1 output_features must match fc2 input_features")
    if fc1.padded_output_features != fc2.padded_input_features:
        raise ValueError("fc1 and fc2 padded hidden features must match")
    return SVDQW4A4GeluMLPWorkspace(
        fc1=allocate_w4a4_workspace(
            rows,
            fc1,
            with_lora_rank=fc1.padded_rank,
        ),
        fc2=allocate_w4a4_workspace(
            rows,
            fc2,
            with_lora_rank=fc2.padded_rank,
        ),
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
    if hasattr(extension, "linear"):
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

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=inputs.dtype, device=inputs.device)
    extension.quantize_act(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed_smooth,
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
    if hasattr(extension, "convrot_linear"):
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

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=inputs.dtype, device=inputs.device)
    extension.quantize_rotated_act(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        int(packed.input_features),
        int(rot_size),
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
    if hasattr(extension, "bind_convrot_linear"):
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

    def run_bound_convrot(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty((int(rows), packed.output_features), dtype=inputs.dtype, device=inputs.device)
        extension.quantize_rotated_act(
            inputs.contiguous(),
            workspace.quantized_activation,
            workspace.activation_scales,
            int(rotated_input_features),
            int(rot_size),
        )
        extension.gemm(
            workspace.quantized_activation,
            packed.qweight,
            output,
            workspace.activation_scales,
            packed.weight_scales,
            packed.packed_bias,
        )
        return output

    return run_bound_convrot


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
    if hasattr(extension, "svdq_linear"):
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

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=inputs.dtype, device=inputs.device)
    extension.quantize_act_lora(
        inputs.contiguous(),
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.packed_down,
        active_workspace.lora_activation,
        packed.packed_smooth,
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
    if hasattr(extension, "bind_svdq_linear"):
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

    def run_bound_svdq(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty((int(rows), packed.output_features), dtype=inputs.dtype, device=inputs.device)
        extension.quantize_act_lora(
            inputs.contiguous(),
            workspace.quantized_activation,
            workspace.activation_scales,
            packed.packed_down,
            workspace.lora_activation,
            packed.packed_smooth,
        )
        extension.gemm_lora(
            workspace.quantized_activation,
            packed.qweight,
            output,
            workspace.activation_scales,
            packed.weight_scales,
            workspace.lora_activation,
            packed.packed_up,
            packed.packed_bias,
            float(lora_scale),
        )
        return output

    return run_bound_svdq


def bind_svdq_w4a4_qkv_rmsnorm_rope(
    packed: PackedSVDQW4A4Linear,
    workspace: W4A4Workspace,
    norm_q: torch.Tensor,
    norm_k: torch.Tensor,
    *,
    rows: int,
    lora_scale: float = 1.0,
    eps: float = 1e-6,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Bind the QKV projection, Q/K RMSNorm, and RoPE epilogue hot path."""

    if workspace.lora_activation is None:
        raise ValueError("SVDQuant QKV workspace requires lora_activation")
    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    if packed.output_features != packed.padded_output_features:
        raise ValueError("QKV output_features must not require output padding")
    if packed.output_features % (3 * _QKV_HEAD_DIM) != 0:
        raise ValueError("QKV output_features must be divisible by 384")
    for tensor, name in ((norm_q, "norm_q"), (norm_k, "norm_k")):
        if tensor.ndim != 1 or int(tensor.numel()) != _QKV_HEAD_DIM:
            raise ValueError(f"{name} must have 128 elements")
        if tensor.device != packed.qweight.device:
            raise ValueError(f"{name} must match the packed weight device")
        if tensor.dtype != packed.weight_scales.dtype:
            raise ValueError(f"{name} must match the packed weight dtype")
    extension = _load_extension()
    return extension.bind_svdq_qkv_rmsnorm_rope(
        workspace.quantized_activation,
        workspace.activation_scales,
        packed.packed_down,
        workspace.lora_activation,
        packed.packed_smooth,
        packed.qweight,
        packed.weight_scales,
        packed.packed_up,
        packed.packed_bias,
        norm_q.contiguous(),
        norm_k.contiguous(),
        int(rows),
        int(packed.input_features),
        int(packed.output_features),
        float(lora_scale),
        float(eps),
    )


def svdq_w4a4_qkv_rmsnorm_rope(
    inputs: torch.Tensor,
    packed: PackedSVDQW4A4Linear,
    norm_q: torch.Tensor,
    norm_k: torch.Tensor,
    rotary_emb: torch.Tensor,
    *,
    workspace: W4A4Workspace | None = None,
    lora_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run one fused QKV projection with an already packed RoPE tensor."""

    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_w4a4_workspace(
        int(inputs.shape[0]),
        packed,
        with_lora_rank=packed.padded_rank,
    )
    runner = bind_svdq_w4a4_qkv_rmsnorm_rope(
        packed,
        active_workspace,
        norm_q,
        norm_k,
        rows=int(inputs.shape[0]),
        lora_scale=lora_scale,
        eps=eps,
    )
    return runner(inputs.contiguous(), rotary_emb.contiguous())


def svdq_w4a4_attention_fp16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the official Nunchaku-style FP16 online-softmax attention kernel."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention query, key, and value must be [B, H, S, 128]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("attention query, key, and value shapes must match")
    if int(query.shape[-1]) != _QKV_HEAD_DIM:
        raise ValueError("attention head dimension must be 128")
    if query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16:
        raise ValueError("FP16 attention query, key, and value must use float16")
    if not query.is_cuda or key.device != query.device or value.device != query.device:
        raise XQTBackendError("FP16 attention tensors must share a CUDA device")
    if output is None:
        output = torch.empty(
            (int(query.shape[0]), int(query.shape[2]), int(query.shape[1]) * _QKV_HEAD_DIM),
            device=query.device,
            dtype=query.dtype,
        )
    if output.shape != (
        int(query.shape[0]),
        int(query.shape[2]),
        int(query.shape[1]) * _QKV_HEAD_DIM,
    ):
        raise ValueError("attention output shape does not match query")
    if output.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("attention output must use float16 or bfloat16")
    extension = _load_extension()
    return extension.attention_fp16(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        output,
        float(scale),
    )


def bind_svdq_w4a4_gelu_mlp(
    fc1: PackedSVDQW4A4Linear,
    fc2: PackedSVDQW4A4Linear,
    workspace: SVDQW4A4GeluMLPWorkspace,
    *,
    rows: int,
    fc1_lora_scale: float = 1.0,
    fc2_lora_scale: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind an official-semantics INT4 SVDQuant GELU MLP hot path.

    The first GEMM epilogue applies LoRA-up and bias, tanh-approximate GELU, the
    Nunchaku ``+0.171875`` shift, unsigned INT4 quantization, and the second
    layer LoRA-down. The second GEMM consumes unsigned activations and applies
    its LoRA-up and bias.
    """

    if fc1.output_features != fc2.input_features:
        raise ValueError("fc1 output_features must match fc2 input_features")
    if fc1.padded_output_features != fc2.padded_input_features:
        raise ValueError("fc1 and fc2 padded hidden features must match")
    expected_rows = _round_up(int(rows), 256)
    if workspace.fc1.padded_rows != expected_rows:
        raise ValueError("fc1 workspace row extent does not match rows")
    if workspace.fc2.padded_rows != expected_rows:
        raise ValueError("fc2 workspace row extent does not match rows")
    if workspace.fc1.lora_activation is None:
        raise ValueError("fc1 workspace requires lora_activation")
    if workspace.fc2.lora_activation is None:
        raise ValueError("fc2 workspace requires lora_activation")
    extension = _load_extension()
    return extension.bind_svdq_gelu_mlp(
        workspace.fc1.quantized_activation,
        workspace.fc1.activation_scales,
        fc1.packed_down,
        workspace.fc1.lora_activation,
        fc1.packed_smooth,
        fc1.qweight,
        fc1.weight_scales,
        fc1.packed_up,
        fc1.packed_bias,
        workspace.fc2.quantized_activation,
        workspace.fc2.activation_scales,
        fc2.packed_down,
        workspace.fc2.lora_activation,
        fc2.packed_smooth,
        fc2.qweight,
        fc2.weight_scales,
        fc2.packed_up,
        fc2.packed_bias,
        int(rows),
        int(fc1.input_features),
        int(fc1.output_features),
        int(fc2.output_features),
        float(fc1_lora_scale),
        float(fc2_lora_scale),
    )


def svdq_w4a4_gelu_mlp(
    inputs: torch.Tensor,
    fc1: PackedSVDQW4A4Linear,
    fc2: PackedSVDQW4A4Linear,
    *,
    workspace: SVDQW4A4GeluMLPWorkspace | None = None,
    fc1_lora_scale: float = 1.0,
    fc2_lora_scale: float = 1.0,
) -> torch.Tensor:
    """Run the bound official-semantics SVDQuant GELU MLP once."""

    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != fc1.input_features:
        raise ValueError("inputs do not match fc1 input_features")
    if inputs.device != fc1.qweight.device or inputs.dtype != fc1.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_svdq_w4a4_gelu_mlp_workspace(
        int(inputs.shape[0]),
        fc1,
        fc2,
    )
    runner = bind_svdq_w4a4_gelu_mlp(
        fc1,
        fc2,
        active_workspace,
        rows=int(inputs.shape[0]),
        fc1_lora_scale=fc1_lora_scale,
        fc2_lora_scale=fc2_lora_scale,
    )
    return runner(inputs.contiguous())


def _require_norm_workspace(
    inputs: torch.Tensor,
    workspace: W4A4Workspace,
) -> None:
    if workspace.lora_activation is None:
        raise ValueError("SVDQuant norm workspace requires lora_activation")
    if workspace.row_scales is None:
        raise ValueError("SVDQuant norm workspace requires row_scales")
    if workspace.padded_rows != _round_up(int(inputs.shape[0]), 256):
        raise ValueError("workspace row extent does not match inputs")


def svdq_w4a4_linear_norm(
    inputs: torch.Tensor,
    packed: PackedSVDQW4A4Linear,
    *,
    workspace: W4A4Workspace | None = None,
    lora_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm-fused two-stage W4A4: ``inputs`` is the PRE-norm activation.

    The per-channel norm weight is stored explicitly in ``packed``. The row
    RMS kernel computes ``rsqrt(mean(x^2) + eps)`` and the activation kernel
    applies both factors before sharing the fragment with W4A4 quantization
    and LoRA-down.
    """

    _require_native_shape(inputs, "inputs")
    if int(inputs.shape[1]) != packed.input_features:
        raise ValueError("inputs do not match packed input_features")
    if inputs.device != packed.qweight.device or inputs.dtype != packed.weight_scales.dtype:
        raise XQTBackendError("inputs must match packed weight device and dtype")
    active_workspace = workspace or allocate_w4a4_workspace(
        int(inputs.shape[0]),
        packed,
        with_lora_rank=packed.padded_rank,
        with_row_scales=True,
    )
    _require_norm_workspace(inputs, active_workspace)
    if packed.norm_weight is None:
        raise ValueError("norm-fused packed state requires norm_weight")
    extension = _load_extension()
    if hasattr(extension, "svdq_linear_norm"):
        return extension.svdq_linear_norm(
            inputs.contiguous(),
            packed.norm_weight,
            active_workspace.row_scales,
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
            float(eps),
        )

    output = torch.empty((inputs.shape[0], packed.output_features), dtype=inputs.dtype, device=inputs.device)
    extension.norm_quantize_act_lora(
        inputs.contiguous(),
        packed.norm_weight,
        active_workspace.row_scales,
        active_workspace.quantized_activation,
        active_workspace.activation_scales,
        packed.packed_down,
        active_workspace.lora_activation,
        packed.packed_smooth,
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


def bind_svdq_w4a4_linear_norm(
    packed: PackedSVDQW4A4Linear,
    workspace: W4A4Workspace,
    *,
    rows: int,
    lora_scale: float = 1.0,
    eps: float = 1e-6,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind validated norm-fused packed state for the steady-state hot path."""

    if workspace.lora_activation is None or workspace.row_scales is None:
        raise ValueError("SVDQuant norm workspace requires lora_activation and row_scales")
    if workspace.padded_rows != _round_up(int(rows), 256):
        raise ValueError("workspace row extent does not match rows")
    if packed.norm_weight is None:
        raise ValueError("norm-fused packed state requires norm_weight")
    extension = _load_extension()
    if hasattr(extension, "bind_svdq_linear_norm"):
        return extension.bind_svdq_linear_norm(
            packed.norm_weight,
            workspace.row_scales,
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
            float(eps),
        )

    def run_bound_norm(inputs: torch.Tensor) -> torch.Tensor:
        output = torch.empty((int(rows), packed.output_features), dtype=inputs.dtype, device=inputs.device)
        extension.norm_quantize_act_lora(
            inputs.contiguous(),
            packed.norm_weight,
            workspace.row_scales,
            workspace.quantized_activation,
            workspace.activation_scales,
            packed.packed_down,
            workspace.lora_activation,
            packed.packed_smooth,
        )
        extension.gemm_lora(
            workspace.quantized_activation,
            packed.qweight,
            output,
            workspace.activation_scales,
            packed.weight_scales,
            workspace.lora_activation,
            packed.packed_up,
            packed.packed_bias,
            float(lora_scale),
        )
        return output

    return run_bound_norm


__all__ = [
    "PackedSVDQW4A4Linear",
    "PackedW4A4Linear",
    "SVDQW4A4GeluMLPWorkspace",
    "W4A4Workspace",
    "allocate_svdq_w4a4_gelu_mlp_workspace",
    "allocate_w4a4_workspace",
    "bind_convrot_w4a4_linear",
    "bind_svdq_w4a4_linear",
    "bind_svdq_w4a4_linear_norm",
    "bind_svdq_w4a4_qkv_rmsnorm_rope",
    "bind_svdq_w4a4_linear_smalln",
    "bind_svdq_w4a4_linear_smalln_norm",
    "bind_svdq_w4a4_gelu_mlp",
    "convrot_w4a4_linear",
    "native_convrot_w4a4_shape_supported",
    "native_w4a4_available",
    "native_w4a4_shape_supported",
    "native_w4a4_smalln_available",
    "native_w4a4_smalln_version",
    "native_w4a4_version",
    "pack_lowrank_weight",
    "pack_scale",
    "pack_svdq_w4a4_rotary_emb",
    "pack_svdq_w4a4_linear",
    "pack_svdq_w4a4_linear_smalln",
    "pack_w4a4_linear",
    "pack_w4a4_linear_smalln",
    "smalln_w4a4_beneficial",
    "svdq_w4a4_linear",
    "svdq_w4a4_gelu_mlp",
    "svdq_w4a4_linear_norm",
    "svdq_w4a4_qkv_rmsnorm_rope",
    "svdq_w4a4_attention_fp16",
    "svdq_w4a4_linear_smalln",
    "svdq_w4a4_linear_smalln_norm",
    "w4a4_linear",
    "w4a4_linear_smalln",
]
