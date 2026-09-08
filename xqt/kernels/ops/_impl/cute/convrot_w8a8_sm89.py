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
_REPO_ROOT = _HERE.parents[4]
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
                name=f"xqt_convrot_w8a8_sm89_{dtype_key}_v4",
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
    raw_qweight_t: torch.Tensor
    raw_weight_scales: torch.Tensor
    raw_bias: torch.Tensor


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
        raw_qweight_t=qweight_t,
        raw_weight_scales=weight_scale.to(
            device=qweight_t.device,
            dtype=torch.float32,
        ).contiguous(),
        raw_bias=(
            torch.zeros(output_features, device=qweight_t.device, dtype=torch.float32)
            if bias is None
            else bias.to(device=qweight_t.device, dtype=torch.float32).contiguous()
        ),
    )


def allocate_convrot_w8a8_workspace(
    rows: int,
    packed: PackedConvRotW8A8Linear,
) -> ConvRotW8A8Workspace:
    padded_rows = _round_up(int(rows), 16) if int(rows) <= 128 else _round_up(int(rows), 256)
    activation_features = (
        int(packed.raw_qweight_t.shape[0])
        if int(rows) <= 128
        else packed.padded_input_features
    )
    return ConvRotW8A8Workspace(
        quantized_activation=torch.empty(
            (padded_rows, activation_features),
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
    expected_rows = (
        _round_up(int(inputs.shape[0]), 16)
        if int(inputs.shape[0]) <= 128
        else _round_up(int(inputs.shape[0]), 256)
    )
    if active_workspace.padded_rows != expected_rows:
        raise ValueError("workspace row extent does not match inputs")
    expected_features = (
        int(packed.raw_qweight_t.shape[0])
        if int(inputs.shape[0]) <= 128
        else packed.padded_input_features
    )
    if int(active_workspace.quantized_activation.shape[1]) != expected_features:
        raise ValueError("workspace feature extent does not match packed weights")
    extension = _load_extension(_dtype_key(inputs.dtype))
    if int(inputs.shape[0]) <= 128:
        return extension.small_m_linear(
            inputs.contiguous(),
            active_workspace.quantized_activation,
            active_workspace.activation_scales,
            packed.raw_qweight_t,
            packed.raw_weight_scales,
            packed.raw_bias,
            rotated_k,
            int(rot_size),
        )
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

    expected_rows = _round_up(int(rows), 16) if int(rows) <= 128 else _round_up(int(rows), 256)
    if workspace.padded_rows != expected_rows:
        raise ValueError("workspace row extent does not match rows")
    expected_features = (
        int(packed.raw_qweight_t.shape[0])
        if int(rows) <= 128
        else packed.padded_input_features
    )
    if int(workspace.quantized_activation.shape[1]) != expected_features:
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
    raw_qweight_t = packed.raw_qweight_t
    raw_weight_scales = packed.raw_weight_scales
    raw_bias = packed.raw_bias

    def run(inputs: torch.Tensor) -> torch.Tensor:
        if int(inputs.shape[0]) <= 128:
            return extension.small_m_linear(
                inputs.contiguous(),
                quantized_activation,
                activation_scales,
                raw_qweight_t,
                raw_weight_scales,
                raw_bias,
                rotated_k,
                rotation_size,
            )
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


def quantize_rotated_activation_sm89(
    inputs: torch.Tensor,
    workspace: ConvRotW8A8Workspace,
    *,
    rotated_input_features: int,
    rot_size: int = _NATIVE_ROT_SIZE,
    norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute online Hadamard rotation and dynamic INT8 quantization once into workspace."""
    if inputs.ndim != 2 or not inputs.is_cuda:
        raise XQTBackendError("inputs must be a two-dimensional CUDA tensor")
    extension = _load_extension(_dtype_key(inputs.dtype))
    m = int(inputs.shape[0])
    weight = norm_weight.contiguous() if norm_weight is not None else None
    if m <= 128:
        extension.small_quantize(
            inputs.contiguous(),
            workspace.quantized_activation,
            workspace.activation_scales,
            int(rotated_input_features),
            int(rot_size),
            weight,
            float(eps),
        )
    else:
        extension.quantize_rotated_act(
            inputs.contiguous(),
            workspace.quantized_activation,
            workspace.activation_scales,
            int(rotated_input_features),
            int(rot_size),
            weight,
            float(eps),
        )
    return workspace.quantized_activation, workspace.activation_scales


def gemm_convrot_w8a8_sm89(
    quantized_activation: torch.Tensor,
    activation_scales: torch.Tensor,
    packed: PackedConvRotW8A8Linear,
    *,
    output: torch.Tensor | None = None,
    actual_rows: int | None = None,
) -> torch.Tensor:
    """Execute INT8 Tensor Core GEMM on pre-quantized/rotated activation."""
    m = int(actual_rows if actual_rows is not None else quantized_activation.shape[0])
    if output is None:
        output = torch.empty(
            (m, packed.output_features),
            dtype=packed.dtype,
            device=packed.qweight.device,
        )
    extension = _load_extension(_dtype_key(packed.dtype))
    if m <= 128:
        extension.small_gemm(
            quantized_activation,
            activation_scales,
            packed.raw_qweight_t,
            packed.raw_weight_scales,
            packed.raw_bias,
            output,
        )
    else:
        extension.gemm(
            quantized_activation,
            packed.qweight,
            output,
            activation_scales,
            packed.weight_scales,
            packed.packed_bias,
        )
    return output


def fused_swiglu(gate_up: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorized fused SwiGLU: silu(gate) * up with zero intermediate tensor allocations."""
    if gate_up.ndim < 2:
        raise XQTBackendError("gate_up must have at least 2 dimensions")
    orig_shape = gate_up.shape[:-1]
    total_d = int(gate_up.shape[-1])
    if total_d % 2 != 0:
        raise XQTBackendError("last dimension of gate_up must be even")
    d = total_d // 2
    flat_in = gate_up.reshape(-1, total_d).contiguous()
    m = int(flat_in.shape[0])
    if output is None:
        flat_out = torch.empty((m, d), dtype=gate_up.dtype, device=gate_up.device)
    else:
        flat_out = output.reshape(-1, d)
    extension = _load_extension(_dtype_key(gate_up.dtype))
    extension.fused_swiglu(flat_in, flat_out)
    if gate_up.ndim > 2:
        return flat_out.reshape(*orig_shape, d)
    return flat_out


class SharedConvRotW8A8Group:
    """Group of ConvRot W8A8 projections sharing input rotation and quantization."""

    def __init__(
        self,
        packeds: list[PackedConvRotW8A8Linear],
        *,
        rot_size: int = _NATIVE_ROT_SIZE,
        norm_weight: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> None:
        if not packeds:
            raise ValueError("packeds must contain at least one projection")
        self.packeds = list(packeds)
        self.rot_size = int(rot_size)
        self.input_features = self.packeds[0].input_features
        self.padded_input_features = self.packeds[0].padded_input_features
        self.dtype = self.packeds[0].dtype
        self.device = self.packeds[0].qweight.device
        self.norm_weight = norm_weight.to(device=self.device, dtype=self.dtype) if norm_weight is not None else None
        self.eps = float(eps)
        self._workspace: ConvRotW8A8Workspace | None = None

    def _get_workspace(self, rows: int) -> ConvRotW8A8Workspace:
        expected_rows = _round_up(int(rows), 16) if int(rows) <= 128 else _round_up(int(rows), 256)
        if (
            self._workspace is None
            or self._workspace.padded_rows != expected_rows
        ):
            self._workspace = allocate_convrot_w8a8_workspace(rows, self.packeds[0])
        return self._workspace

    def __call__(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.ndim != 2:
            flat = inputs.reshape(-1, self.input_features)
        else:
            flat = inputs
        m = int(flat.shape[0])
        workspace = self._get_workspace(m)
        quantize_rotated_activation_sm89(
            flat,
            workspace,
            rotated_input_features=self.padded_input_features,
            rot_size=self.rot_size,
            norm_weight=self.norm_weight,
            eps=self.eps,
        )
        outputs: list[torch.Tensor] = []
        for packed in self.packeds:
            out = torch.empty((m, packed.output_features), dtype=self.dtype, device=self.device)
            gemm_convrot_w8a8_sm89(
                workspace.quantized_activation,
                workspace.activation_scales,
                packed,
                output=out,
                actual_rows=m,
            )
            if inputs.ndim > 2:
                out = out.reshape(*inputs.shape[:-1], packed.output_features)
            outputs.append(out)
        return outputs


class FusedConvRotW8A8LinearGroup:
    """Group of ConvRot W8A8 projections fused into a single horizontal GEMM."""

    def __init__(
        self,
        packeds: list[PackedConvRotW8A8Linear],
        *,
        rot_size: int = _NATIVE_ROT_SIZE,
        norm_weight: torch.Tensor | None = None,
        eps: float = 1e-6,
        min_int8_rows: int = 256,
    ) -> None:
        if not packeds:
            raise ValueError("packeds must contain at least one projection")
        self.packeds = list(packeds)
        self.rot_size = int(rot_size)
        self.input_features = self.packeds[0].input_features
        self.padded_input_features = self.packeds[0].padded_input_features
        self.dtype = self.packeds[0].dtype
        self.device = self.packeds[0].qweight.device
        self.split_sizes = [p.output_features for p in self.packeds]
        self.total_output_features = sum(self.split_sizes)
        self.norm_weight = norm_weight.to(device=self.device, dtype=self.dtype) if norm_weight is not None else None
        self.eps = float(eps)
        self.min_int8_rows = int(min_int8_rows)

        cat_qw = torch.cat([p.raw_qweight_t for p in self.packeds], dim=1)
        cat_ws = torch.cat([p.raw_weight_scales for p in self.packeds], dim=0)
        has_bias = any(p.raw_bias.abs().sum().item() > 0 for p in self.packeds)
        if has_bias:
            cat_b = torch.cat([p.raw_bias for p in self.packeds], dim=0)
        else:
            cat_b = None

        self.fused_packed = pack_convrot_w8a8_linear(
            cat_qw,
            cat_ws,
            cat_b,
            dtype=self.dtype,
        )
        self.dequant_weight = torch.cat(
            [p.raw_qweight_t.float() * p.raw_weight_scales.unsqueeze(0) for p in self.packeds],
            dim=1,
        ).to(device=self.device, dtype=self.dtype).detach()
        self.cat_bias = (
            cat_b.to(device=self.device, dtype=self.dtype).detach()
            if cat_b is not None
            else None
        )

        self._workspace: ConvRotW8A8Workspace | None = None
        self._decode_fused_output: torch.Tensor | None = None
        self._decode_outputs: list[torch.Tensor] | None = None
        self._decode_swiglu_output: torch.Tensor | None = None

    def _get_workspace(self, rows: int) -> ConvRotW8A8Workspace:
        expected_rows = _round_up(int(rows), 16) if int(rows) <= 128 else _round_up(int(rows), 256)
        if (
            self._workspace is None
            or self._workspace.padded_rows != expected_rows
        ):
            self._workspace = allocate_convrot_w8a8_workspace(rows, self.fused_packed)
        return self._workspace

    def __call__(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        if inputs.ndim != 2:
            flat = inputs.reshape(-1, self.input_features)
        else:
            flat = inputs
        m = int(flat.shape[0])

        if 0 < self.min_int8_rows and m < self.min_int8_rows:
            if self.norm_weight is not None:
                normed = (flat.float() * torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_weight.float()).to(self.dtype)
            else:
                normed = flat
            normed = normed.detach()
            if m == 1:
                if self._decode_fused_output is None:
                    self._decode_fused_output = torch.empty(
                        (1, self.total_output_features),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self._decode_outputs = [
                        torch.empty((1, s), dtype=self.dtype, device=self.device)
                        for s in self.split_sizes
                    ]
                torch.mm(normed, self.dequant_weight, out=self._decode_fused_output)
                if self.cat_bias is not None:
                    self._decode_fused_output.add_(self.cat_bias)
                col = 0
                for i, s in enumerate(self.split_sizes):
                    self._decode_outputs[i].copy_(self._decode_fused_output[:, col : col + s])
                    col += s
                if inputs.ndim > 2:
                    orig_shape = inputs.shape[:-1]
                    return [out.reshape(*orig_shape, -1) for out in self._decode_outputs]
                return self._decode_outputs

            fused_output = F.linear(normed, self.dequant_weight.t(), self.cat_bias)
            outputs = list(torch.split(fused_output, self.split_sizes, dim=-1))
            if inputs.ndim > 2:
                orig_shape = inputs.shape[:-1]
                outputs = [out.reshape(*orig_shape, -1) for out in outputs]
            return outputs

        workspace = self._get_workspace(m)
        quantize_rotated_activation_sm89(
            flat,
            workspace,
            rotated_input_features=self.padded_input_features,
            rot_size=self.rot_size,
            norm_weight=self.norm_weight,
            eps=self.eps,
        )
        if m == 1:
            if self._decode_fused_output is None:
                self._decode_fused_output = torch.empty(
                    (1, self.total_output_features),
                    dtype=self.dtype,
                    device=self.device,
                )
                self._decode_outputs = [
                    torch.empty((1, s), dtype=self.dtype, device=self.device)
                    for s in self.split_sizes
                ]
            gemm_convrot_w8a8_sm89(
                workspace.quantized_activation,
                workspace.activation_scales,
                self.fused_packed,
                output=self._decode_fused_output,
                actual_rows=1,
            )
            col = 0
            for i, s in enumerate(self.split_sizes):
                self._decode_outputs[i].copy_(self._decode_fused_output[:, col : col + s])
                col += s
            if inputs.ndim > 2:
                orig_shape = inputs.shape[:-1]
                return [out.reshape(*orig_shape, -1) for out in self._decode_outputs]
            return self._decode_outputs

        fused_output = torch.empty(
            (m, self.total_output_features),
            dtype=self.dtype,
            device=self.device,
        )
        gemm_convrot_w8a8_sm89(
            workspace.quantized_activation,
            workspace.activation_scales,
            self.fused_packed,
            output=fused_output,
            actual_rows=m,
        )
        outputs = list(torch.split(fused_output, self.split_sizes, dim=-1))
        if inputs.ndim > 2:
            orig_shape = inputs.shape[:-1]
            outputs = [out.reshape(*orig_shape, -1) for out in outputs]
        return outputs

    def forward_swiglu(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run Gate-Up GEMM followed immediately by fused SwiGLU in one pass."""
        if len(self.packeds) != 2:
            raise XQTBackendError("forward_swiglu requires exactly two packed projections (Gate and Up)")
        if inputs.ndim != 2:
            flat = inputs.reshape(-1, self.input_features)
        else:
            flat = inputs
        m = int(flat.shape[0])

        if 0 < self.min_int8_rows and m < self.min_int8_rows:
            if self.norm_weight is not None:
                normed = (flat.float() * torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_weight.float()).to(self.dtype)
            else:
                normed = flat
            normed = normed.detach()
            if m == 1:
                if self._decode_fused_output is None:
                    self._decode_fused_output = torch.empty(
                        (1, self.total_output_features),
                        dtype=self.dtype,
                        device=self.device,
                    )
                if self._decode_swiglu_output is None:
                    self._decode_swiglu_output = torch.empty(
                        (1, self.split_sizes[0]),
                        dtype=self.dtype,
                        device=self.device,
                    )
                torch.mm(normed, self.dequant_weight, out=self._decode_fused_output)
                if self.cat_bias is not None:
                    self._decode_fused_output.add_(self.cat_bias)
                fused_swiglu(self._decode_fused_output, output=self._decode_swiglu_output)
                if inputs.ndim > 2:
                    return self._decode_swiglu_output.reshape(*inputs.shape[:-1], self.split_sizes[0])
                return self._decode_swiglu_output

            fused_output = F.linear(normed, self.dequant_weight.t(), self.cat_bias)
            out = fused_swiglu(fused_output)
            if inputs.ndim > 2:
                return out.reshape(*inputs.shape[:-1], self.split_sizes[0])
            return out

        workspace = self._get_workspace(m)
        quantize_rotated_activation_sm89(
            flat,
            workspace,
            rotated_input_features=self.padded_input_features,
            rot_size=self.rot_size,
            norm_weight=self.norm_weight,
            eps=self.eps,
        )
        if m == 1:
            if self._decode_fused_output is None:
                self._decode_fused_output = torch.empty(
                    (1, self.total_output_features),
                    dtype=self.dtype,
                    device=self.device,
                )
            if self._decode_swiglu_output is None:
                self._decode_swiglu_output = torch.empty(
                    (1, self.split_sizes[0]),
                    dtype=self.dtype,
                    device=self.device,
                )
            gemm_convrot_w8a8_sm89(
                workspace.quantized_activation,
                workspace.activation_scales,
                self.fused_packed,
                output=self._decode_fused_output,
                actual_rows=1,
            )
            fused_swiglu(self._decode_fused_output, output=self._decode_swiglu_output)
            if inputs.ndim > 2:
                return self._decode_swiglu_output.reshape(*inputs.shape[:-1], self.split_sizes[0])
            return self._decode_swiglu_output

        fused_output = torch.empty(
            (m, self.total_output_features),
            dtype=self.dtype,
            device=self.device,
        )
        gemm_convrot_w8a8_sm89(
            workspace.quantized_activation,
            workspace.activation_scales,
            self.fused_packed,
            output=fused_output,
            actual_rows=m,
        )
        out = fused_swiglu(fused_output)
        if inputs.ndim > 2:
            return out.reshape(*inputs.shape[:-1], self.split_sizes[0])
        return out


__all__ = [
    "ConvRotW8A8Workspace",
    "FusedConvRotW8A8LinearGroup",
    "PackedConvRotW8A8Linear",
    "SharedConvRotW8A8Group",
    "allocate_convrot_w8a8_workspace",
    "bind_convrot_w8a8_linear",
    "convrot_w8a8_linear",
    "fused_swiglu",
    "gemm_convrot_w8a8_sm89",
    "native_convrot_w8a8_available",
    "native_convrot_w8a8_shape_supported",
    "native_convrot_w8a8_version",
    "native_prerotated_w8a8_shape_supported",
    "pack_convrot_w8a8_linear",
    "quantize_rotated_activation_sm89",
]
