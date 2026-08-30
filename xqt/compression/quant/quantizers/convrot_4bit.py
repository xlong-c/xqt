"""ConvRot-inspired group-wise regular Hadamard 4-bit quantization.

Quantization produces packed weight artifacts, rotation matrices, activation
scales, and optional mixed-precision execution policies.

Inference / mixed-precision dispatch lives in ``xqt.runtime`` and only consumes
those artifacts. This module never owns a hybrid inference engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import (
    ChannelHybridSpec,
    QuantizedModel,
    SUPPORTED_COMPUTE_PRECISIONS,
    compute_hybrid_linear,
    normalize_channel_axis,
    normalize_compute_precision,
)
from xqt.core.types import XQTContext

from ..channel_helpers import (
    build_channel_mask,
    normalize_channel_indices,
    select_outlier_channels,
)

from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import build_component_quantization_report
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..sensitivity import (
    LayerSensitivityRecord,
    analyze_layer_sensitivity,
    suggest_high_precision_modules,
)
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from xqt.contracts.packing_int4 import (
    _pack_int4,
    _quantize_grouped_fp4_weight,
    _unpack_int4,
)
from xqt.core.compilation import can_mutate_runtime_cache as _can_mutate_runtime_cache
from .int8_mma import Int8MmaLinear
from .base import (
    call_model as _call_model,
    iter_calibration_batches as _iter_calibration_batches,
    move_batch_to_device as _move_batch_to_device,
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)

_ACTIVATION_SCALE_MODES = frozenset({"dynamic", "static"})
_W4A4_RUNTIME_BACKENDS = frozenset({"auto", "rowwise", "nunchaku", "reference"})


def _normalize_w4a4_runtime_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _W4A4_RUNTIME_BACKENDS:
        allowed = ", ".join(sorted(_W4A4_RUNTIME_BACKENDS))
        raise ValueError(f"w4a4_runtime_backend must be one of: {allowed}")
    return normalized


def _base_regular_hadamard4() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


@lru_cache(maxsize=16)
def _cached_regular_hadamard_matrix(order: int) -> torch.Tensor:
    """Build one CPU matrix once; callers receive a clone before mutation."""

    if order == 1:
        return torch.ones((1, 1), dtype=torch.float32)
    matrix = _base_regular_hadamard4()
    current_order = 4
    while current_order < order:
        matrix = torch.kron(matrix, _base_regular_hadamard4())
        current_order *= 4
    return matrix.contiguous()


def build_regular_hadamard_matrix(order: int) -> torch.Tensor:
    """Return the regular Hadamard matrix from ConvRot Theorem 3.3 for 4^k orders."""

    normalized_order = int(order)
    if normalized_order < 1:
        raise ValueError("order must be positive")
    if normalized_order != 1:
        value = normalized_order
        while value % 4 == 0:
            value //= 4
        if value != 1:
            raise ValueError("regular Hadamard order must be a power of four")
    return _cached_regular_hadamard_matrix(normalized_order).clone()


def _normalize_rot_size(rot_size: int, input_features: int) -> int:
    """Validate and retain the requested regular-Hadamard block size.

    ``input_features`` is intentionally not used to downsize the block.  ConvRot
    pads a short or non-divisible feature dimension internally so the configured
    ``N0`` remains the actual rotation block described by the recipe.
    """

    features = int(input_features)
    if features < 1:
        raise ValueError("input_features must be positive")
    requested = int(rot_size)
    if requested < 1:
        raise ValueError("rot_size must be positive")
    if requested != 1:
        value = requested
        while value % 4 == 0:
            value //= 4
        if value != 1:
            raise ValueError("rot_size must be one or a power of four")
    return requested


def _normalize_convrot_group_size(group_size: int) -> int:
    """Validate ConvRot's weight quantization group size without clamping it.

    The generic weight-only quantizer clamps a group to the logical input width.
    ConvRot instead pads the logical width to the requested group/block alignment,
    so a default group such as 128 remains 128 even for a smaller Linear.
    """

    normalized = int(group_size)
    if normalized < 1:
        raise ValueError("group_size must be positive")
    return normalized


def _rotation_padded_features(
    input_features: int,
    rot_size: int,
    *,
    alignment: int = 1,
) -> int:
    """Return the internal K extent aligned for rotation and its consumer."""

    features = int(input_features)
    rotation = int(rot_size)
    consumer_alignment = int(alignment)
    if features < 1 or rotation < 1 or consumer_alignment < 1:
        raise ValueError("feature and alignment sizes must be positive")
    common = math.lcm(rotation, consumer_alignment)
    return ((features + common - 1) // common) * common


def _pad_last_dim(tensor: torch.Tensor, padded_features: int) -> torch.Tensor:
    current = int(tensor.shape[-1])
    target = int(padded_features)
    if target < current:
        raise ValueError(
            f"padded feature extent {target} is smaller than tensor extent {current}"
        )
    if target == current:
        return tensor
    return F.pad(tensor, (0, target - current))


def _normalized_regular_hadamard(order: int, *, device: torch.device) -> torch.Tensor:
    matrix = build_regular_hadamard_matrix(order).to(device=device)
    return matrix / float(order) ** 0.5


def _apply_groupwise_rotation(
    tensor: torch.Tensor,
    *,
    rot_size: int,
    rotation_matrix: torch.Tensor | None = None,
    return_padded: bool = False,
) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("rotation tensor rank must be >= 1")
    feature_dim = int(tensor.shape[-1])
    normalized_rot_size = int(rot_size)
    if normalized_rot_size < 1:
        raise ValueError("rot_size must be positive")
    _normalize_rot_size(normalized_rot_size, feature_dim)
    padded_feature_dim = (
        (feature_dim + normalized_rot_size - 1) // normalized_rot_size
    ) * normalized_rot_size
    rotation = (
        rotation_matrix
        if rotation_matrix is not None
        else _normalized_regular_hadamard(normalized_rot_size, device=tensor.device)
    )
    if tuple(rotation.shape) != (normalized_rot_size, normalized_rot_size):
        raise ValueError("rotation_matrix shape must match rot_size")
    padded = _pad_last_dim(tensor, padded_feature_dim)
    reshaped = padded.reshape(
        -1,
        padded_feature_dim // normalized_rot_size,
        normalized_rot_size,
    )
    rotated = torch.matmul(reshaped, rotation.to(dtype=tensor.dtype, device=tensor.device))
    rotated = rotated.reshape(*padded.shape)
    return rotated if return_padded else rotated[..., :feature_dim]


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _build_execution_policies(
    sensitivity_records: Sequence[LayerSensitivityRecord],
    *,
    mixed_ratio: float,
    runtime_strategy: str,
) -> list[dict[str, Any]]:
    if not sensitivity_records:
        return []
    total = len(sensitivity_records)
    mixed_count = max(1, int(round(float(mixed_ratio) * float(total))))
    selected = suggest_high_precision_modules(sensitivity_records, top_k=mixed_count)
    overrides = [
        {
            "module": name,
            "precision": "w8a8",
            "reason": "sensitivity_topk",
            "runtime_strategy": runtime_strategy,
        }
        for name in selected
    ]
    return [
        {
            "policy_kind": "mixed_precision",
            "runtime": "pytorch",
            "runtime_strategy": runtime_strategy,
            "mixed_ratio": float(mixed_ratio),
            "precision_overrides": overrides,
        }
    ]


@dataclass
class ConvRot4BitQuantizationResult(QuantizedModel):
    """Result returned by the ConvRot-inspired 4-bit helper."""

    backend: str = "pytorch"
    method: str | None = "convrot"
    strategy: str = "w4a4_int4"
    compute: str = "dequant_fp16"


class ConvRotMixedPrecisionLinear(nn.Module):
    """ConvRot packed-weight Linear artifact with selectable runtime precision."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        rot_size: int,
        activation_scale: torch.Tensor | float,
        reference_weight: torch.Tensor,
        rotation_matrix: torch.Tensor,
        compute_precision: str = "w4a4",
        activation_scale_mode: str = "dynamic",
        w4a4_runtime_backend: str = "auto",
        high_precision_channel_mask: torch.Tensor | None = None,
        channel_hybrid_axis: str = "input",
        channel_high_precision: str = "bf16",
        channel_low_precision: str = "w4a4",
        channel_hybrid_enabled: bool = False,
    ) -> None:
        super().__init__()
        # Quantizer construction creates a serializable storage artifact.
        # Native dispatch is enabled only by ConvRotW4A4ExecutionView.
        self._xqt_runtime_execution_enabled = False
        self._xqt_convrot_storage_kind = "w4a4"
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.rot_size = int(rot_size)
        if self.input_features < 1 or self.output_features < 1:
            raise ValueError("input_features and output_features must be positive")
        self.rot_size = _normalize_rot_size(self.rot_size, self.input_features)
        if self.group_size < 1 or self.rot_size < 1:
            raise ValueError("group_size and rot_size must be positive")
        if self.padded_input_features < self.input_features:
            raise ValueError("padded_input_features must cover input_features")
        if self.padded_input_features % self.rot_size != 0:
            raise ValueError("padded_input_features must be divisible by rot_size")
        if self.padded_input_features % self.group_size != 0:
            raise ValueError("padded_input_features must be divisible by group_size")
        if tuple(packed_weight.shape) != (
            self.output_features,
            (self.padded_input_features + 1) // 2,
        ):
            raise ValueError("packed_weight shape does not match padded_input_features")
        if tuple(rotation_matrix.shape) != (self.rot_size, self.rot_size):
            raise ValueError("rotation_matrix shape must match rot_size")
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8).contiguous())
        self.register_buffer("weight_scale", weight_scale.to(torch.float32).contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())
        self.register_buffer(
            "activation_scale",
            torch.as_tensor(activation_scale, dtype=torch.float32).reshape(()),
        )
        self.register_buffer(
            "rotation_matrix",
            rotation_matrix.to(torch.float32).contiguous(),
        )
        reference = reference_weight.detach().to(torch.float32).contiguous()
        if tuple(reference.shape) != (self.output_features, self.padded_input_features):
            raise ValueError(
                "reference_weight must have shape "
                f"({self.output_features}, {self.padded_input_features})"
            )
        self.register_buffer("reference_weight", reference)
        axis = normalize_channel_axis(channel_hybrid_axis)
        dim_size = self.input_features if axis == "input" else self.output_features
        if high_precision_channel_mask is None:
            mask = torch.zeros(dim_size, dtype=torch.bool)
        else:
            mask = high_precision_channel_mask.to(torch.bool).reshape(-1).contiguous()
            if int(mask.numel()) != dim_size:
                raise ValueError(
                    f"high_precision_channel_mask length {int(mask.numel())} "
                    f"!= axis dim {dim_size}"
                )
        self.register_buffer("high_precision_channel_mask", mask)
        self.channel_hybrid_axis = axis
        self.channel_high_precision = str(channel_high_precision)
        self.channel_low_precision = str(channel_low_precision)
        self.channel_hybrid_enabled = bool(channel_hybrid_enabled) and bool(mask.any().item())
        scale_mode = str(activation_scale_mode).strip().lower()
        if scale_mode not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
        self.activation_scale_mode = scale_mode
        self.w4a4_runtime_backend = _normalize_w4a4_runtime_backend(
            w4a4_runtime_backend
        )
        self.input_already_rotated = False
        self._int8_compute: Int8MmaLinear | None = None
        self._dequantized_weight_cache: dict[
            tuple[str, str, bool], tuple[tuple[Any, ...], torch.Tensor]
        ] = {}
        self._runtime_rotation_cache: dict[
            tuple[str, int, int], torch.Tensor
        ] = {}
        self._rowwise_w4a4_packed_cache: tuple[tuple[Any, ...], Any] | None = None
        self._rowwise_w4a4_runner_cache: tuple[
            int | None, torch.dtype, tuple[int, int, int], Any, Any
        ] | None = None
        self._native_w4a4_packed_cache: tuple[tuple[Any, ...], Any] | None = None
        self._native_w4a4_workspace_cache: dict[tuple[Any, ...], Any] = {}
        self._native_w4a4_hot_cache: dict[
            tuple[Any, ...], tuple[tuple[Any, ...], Any, Any, Any]
        ] = {}
        self._last_native_w4a4_used = False
        self._last_native_w4a4_backend: str | None = None
        self._last_native_w4a4_fallback_reason: str | None = None
        self.compute_precision = normalize_compute_precision(compute_precision)
        self._ensure_int8_compute()

    def _clear_runtime_caches(self) -> None:
        self._dequantized_weight_cache.clear()
        self._runtime_rotation_cache.clear()
        self._rowwise_w4a4_packed_cache = None
        self._rowwise_w4a4_runner_cache = None
        self._native_w4a4_packed_cache = None
        self._native_w4a4_workspace_cache.clear()
        self._native_w4a4_hot_cache.clear()

    def _apply(self, fn: Any) -> "ConvRotMixedPrecisionLinear":
        """Move registered tensors, then invalidate device/dtype-derived views."""

        super()._apply(fn)
        self._clear_runtime_caches()
        self._last_native_w4a4_used = False
        self._last_native_w4a4_backend = None
        self._last_native_w4a4_fallback_reason = None
        return self

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rot_size: int = 256,
        group_size: int = 128,
        activation_scale: torch.Tensor | float = 1.0,
        activation_scale_mode: str = "dynamic",
        compute_precision: str = "w4a4",
        w4a4_runtime_backend: str = "auto",
        high_precision_channels: Sequence[int] | torch.Tensor | None = None,
        channel_hybrid_axis: str = "input",
        channel_high_precision: str = "bf16",
        channel_low_precision: str = "w4a4",
    ) -> "ConvRotMixedPrecisionLinear":
        weight = module.weight.detach().to(torch.float32)
        input_features = int(module.in_features)
        output_features = int(module.out_features)
        normalized_rot_size = _normalize_rot_size(rot_size, input_features)
        normalized_group_size = _normalize_convrot_group_size(group_size)
        padded_input_features = _rotation_padded_features(
            input_features,
            normalized_rot_size,
            alignment=normalized_group_size,
        )
        padded_weight = _pad_last_dim(weight, padded_input_features)
        rotation = _normalized_regular_hadamard(
            normalized_rot_size,
            device=weight.device,
        )
        rotated_weight = _apply_groupwise_rotation(
            padded_weight,
            rot_size=normalized_rot_size,
            rotation_matrix=rotation,
            return_padded=True,
        )
        packed_weight, weight_scale, padded_input_features = _quantize_grouped_fp4_weight(
            rotated_weight,
            group_size=normalized_group_size,
            input_features=padded_input_features,
            output_features=output_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        axis = normalize_channel_axis(channel_hybrid_axis)
        dim_size = input_features if axis == "input" else output_features
        selected_channels: Sequence[int] | torch.Tensor
        selected_channels = () if high_precision_channels is None else high_precision_channels
        mask = build_channel_mask(
            dim_size,
            selected_channels,
            device=weight.device,
        )
        instance = cls(
            packed_weight,
            weight_scale,
            bias=bias,
            input_features=input_features,
            output_features=output_features,
            group_size=normalized_group_size,
            padded_input_features=padded_input_features,
            rot_size=normalized_rot_size,
            activation_scale=activation_scale,
            reference_weight=rotated_weight,
            rotation_matrix=rotation,
            compute_precision=compute_precision,
            activation_scale_mode=activation_scale_mode,
            w4a4_runtime_backend=w4a4_runtime_backend,
            high_precision_channel_mask=mask,
            channel_hybrid_axis=axis,
            channel_high_precision=channel_high_precision,
            channel_low_precision=channel_low_precision,
            channel_hybrid_enabled=bool(mask.any().item()),
        )
        already = getattr(module, "input_already_rotated", None)
        if already is None:
            buf = getattr(module, "_xqt_input_already_rotated", None)
            if buf is not None:
                already = bool(buf.item() if hasattr(buf, "item") else buf)
        if already is not None:
            instance.input_already_rotated = bool(already)
        return instance

    def quantized_weight_codes(self) -> torch.Tensor:
        return _unpack_int4(self.packed_weight, self.padded_input_features)[
            :, : self.input_features
        ]

    def dequantized_weight(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        include_padding: bool = False,
    ) -> torch.Tensor:
        can_cache = _can_mutate_runtime_cache()
        if not can_cache:
            codes = _unpack_int4(self.packed_weight, self.padded_input_features).to(
                device=device,
                dtype=torch.float32,
            )
            grouped = codes.reshape(self.output_features, -1, self.group_size)
            dequantized = grouped * self.weight_scale.to(
                device=device, dtype=torch.float32
            )
            dense = dequantized.reshape(self.output_features, self.padded_input_features)
            if not include_padding:
                dense = dense[:, : self.input_features]
            return dense.to(dtype=dtype)

        cache_key = (str(device), str(dtype), bool(include_padding))
        cache_signature = (
            str(self.packed_weight.device),
            int(getattr(self.packed_weight, "_version", 0)),
            tuple(int(dim) for dim in self.packed_weight.shape),
            str(self.weight_scale.device),
            int(getattr(self.weight_scale, "_version", 0)),
            tuple(int(dim) for dim in self.weight_scale.shape),
        )
        cached = self._dequantized_weight_cache.get(cache_key)
        if cached is not None and cached[0] == cache_signature:
            return cached[1]
        codes = _unpack_int4(self.packed_weight, self.padded_input_features).to(
            device=device,
            dtype=torch.float32,
        )
        grouped = codes.reshape(self.output_features, -1, self.group_size)
        dequantized = grouped * self.weight_scale.to(device=device, dtype=torch.float32)
        dense = dequantized.reshape(self.output_features, self.padded_input_features)
        if not include_padding:
            dense = dense[:, : self.input_features]
        result = dense.to(dtype=dtype).detach()
        self._dequantized_weight_cache[cache_key] = (cache_signature, result)
        return result

    def _bias_for(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(device=device, dtype=dtype)

    def _rotate_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = _pad_last_dim(inputs, self.padded_input_features)
        if self.input_already_rotated:
            return padded.to(torch.float32)
        can_cache = _can_mutate_runtime_cache()
        if not can_cache:
            return _apply_groupwise_rotation(
                padded.to(torch.float32),
                rot_size=self.rot_size,
                rotation_matrix=self.rotation_matrix.to(
                    device=inputs.device,
                    dtype=torch.float32,
                ),
                return_padded=True,
            )
        rotation_version = int(getattr(self.rotation_matrix, "_version", 0))
        rotation_key = (
            str(inputs.device),
            int(self.rotation_matrix.data_ptr()),
            rotation_version,
        )
        rotation = None
        rotation = self._runtime_rotation_cache.get(rotation_key)
        if rotation is None:
            rotation = self.rotation_matrix.to(
                device=inputs.device,
                dtype=torch.float32,
            )
            self._runtime_rotation_cache[rotation_key] = rotation
        return _apply_groupwise_rotation(
            padded.to(torch.float32),
            rot_size=self.rot_size,
            rotation_matrix=rotation,
            return_padded=True,
        )

    def _ensure_int8_compute(self) -> None:
        if self.compute_precision != "w8a8" or self._int8_compute is not None:
            return
        dense_weight = self.reference_weight.to(torch.float32)
        dense_bias = None if self.bias is None else self.bias.to(torch.float32)
        linear = nn.Linear(
            self.padded_input_features,
            self.output_features,
            bias=dense_bias is not None,
        )
        linear.weight.data.copy_(dense_weight)
        if dense_bias is not None and linear.bias is not None:
            linear.bias.data.copy_(dense_bias)
        self._int8_compute = Int8MmaLinear.from_linear(
            linear,
            activation_scale_mode=self.activation_scale_mode,
            activation_scale=self.activation_scale,
        )

    def set_compute_precision(self, precision: str) -> None:
        resolved = normalize_compute_precision(precision)
        self.compute_precision = resolved
        if resolved == "w8a8":
            if self._int8_compute is None:
                self._ensure_int8_compute()
            return
        self._int8_compute = None

    def with_compute_precision(self, precision: str) -> "ConvRotMixedPrecisionLinear":
        cloned = copy.deepcopy(self)
        cloned.set_compute_precision(precision)
        return cloned

    def channel_hybrid_spec(self) -> ChannelHybridSpec | None:
        if not self.channel_hybrid_enabled:
            return None
        mask = self.high_precision_channel_mask
        channels = tuple(
            int(item) for item in mask.nonzero(as_tuple=False).reshape(-1).tolist()
        )
        if not channels:
            return None
        return ChannelHybridSpec(
            axis=self.channel_hybrid_axis,
            high_precision_channels=channels,
            high_precision=self.channel_high_precision,
            low_precision=self.channel_low_precision,
            enabled=True,
        )

    def set_channel_hybrid_spec(self, spec: ChannelHybridSpec | None) -> None:
        if spec is None or not spec.enabled or not spec.high_precision_channels:
            axis = self.channel_hybrid_axis
            dim_size = (
                self.input_features if axis == "input" else self.output_features
            )
            self.high_precision_channel_mask = torch.zeros(
                dim_size,
                dtype=torch.bool,
                device=self.high_precision_channel_mask.device,
            )
            self.channel_hybrid_enabled = False
            return
        axis = normalize_channel_axis(spec.axis)
        dim_size = self.input_features if axis == "input" else self.output_features
        mask = build_channel_mask(
            dim_size,
            spec.high_precision_channels,
            device=self.high_precision_channel_mask.device,
        )
        self.channel_hybrid_axis = axis
        self.channel_high_precision = str(spec.high_precision)
        self.channel_low_precision = str(spec.low_precision)
        self.high_precision_channel_mask = mask
        self.channel_hybrid_enabled = bool(mask.any().item())

    def _activation_scale(self, rotated: torch.Tensor) -> torch.Tensor:
        if self.activation_scale_mode == "static":
            scale = self.activation_scale.to(device=rotated.device, dtype=torch.float32)
            return torch.clamp(scale, min=1e-6)
        amax = rotated.detach().abs().amax(dim=-1, keepdim=True)
        return torch.clamp(amax / 7.0, min=1e-6)

    def _quantize_rotated_activation(self, rotated: torch.Tensor) -> torch.Tensor:
        scale = self._activation_scale(rotated)
        quantized = torch.clamp(torch.round(rotated / scale), min=-8, max=7)
        return quantized * scale

    def _quantize_activation(self, inputs: torch.Tensor) -> torch.Tensor:
        rotated = self._rotate_inputs(inputs)
        return self._quantize_rotated_activation(rotated).to(dtype=inputs.dtype)

    def _w4a4_compute_view_signature(self, inputs: torch.Tensor) -> tuple[Any, ...]:
        tensors = (self.packed_weight, self.weight_scale, self.bias)
        return (
            str(inputs.device),
            str(inputs.dtype),
            self.padded_input_features,
            self.output_features,
            self.group_size,
            *(
                None
                if tensor is None
                else (
                    str(tensor.device),
                    str(tensor.dtype),
                    int(tensor.data_ptr()),
                    int(getattr(tensor, "_version", 0)),
                    tuple(int(dim) for dim in tensor.shape),
                )
                for tensor in tensors
            ),
        )

    def _rowwise_w4a4_requested(self) -> bool:
        return self.w4a4_runtime_backend == "rowwise" or (
            self.w4a4_runtime_backend == "auto"
            and self.group_size == self.padded_input_features
        )

    def _rowwise_w4a4_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        if not self._rowwise_w4a4_requested():
            return False, (
                "rowwise W4A4 requires w4a4_runtime_backend=rowwise or an "
                "auto-selected whole-row weight scale"
            )
        if self.compute_precision != "w4a4":
            return False, "rowwise W4A4 requires w4a4 compute precision"
        if self.activation_scale_mode != "dynamic":
            return False, "rowwise W4A4 requires dynamic activation scales"
        if self.input_already_rotated:
            return False, "rowwise ConvRot requires an unrotated activation input"
        if self.input_features != self.padded_input_features:
            return False, "rowwise ConvRot does not materialize padded activation features"
        if not _can_mutate_runtime_cache():
            return False, "rowwise W4A4 eager cache is unavailable while compiling or tracing"
        if not inputs.is_cuda:
            return False, "rowwise W4A4 requires CUDA activations"
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            return False, "rowwise W4A4 requires float16 or bfloat16 activations"
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            return False, "rowwise W4A4 input trailing dimension is invalid"
        major, minor = torch.cuda.get_device_capability(inputs.device)
        if (major, minor) != (8, 9):
            return False, f"rowwise W4A4 currently targets sm_89, got sm_{major}{minor}"
        try:
            from xqt.kernels.ops.quantization import (
                native_rowwise_convrot_w4a4_available,
                native_rowwise_convrot_w4a4_shape_supported,
            )

            if not native_rowwise_convrot_w4a4_shape_supported(
                self.input_features,
                self.output_features,
                self.rot_size,
            ):
                return False, "rowwise ConvRot W4A4 shape or rotation size is unsupported"
            if not native_rowwise_convrot_w4a4_available(build=False):
                return False, "rowwise ConvRot W4A4 backend is unavailable"
        except Exception as exc:
            return False, f"rowwise ConvRot W4A4 capability check failed: {exc}"
        return True, "warp-FHT rowwise INT4 quantization and CUTLASS W4A4 are available"

    def _rowwise_w4a4_packed(self, inputs: torch.Tensor) -> Any:
        from xqt.kernels.ops.quantization import (
            pack_convrot_w4a4_rowwise_weight,
        )

        signature = self._w4a4_compute_view_signature(inputs)
        cached = self._rowwise_w4a4_packed_cache
        if cached is not None and cached[0] == signature:
            return cached[1]
        weight = self.dequantized_weight(
            dtype=inputs.dtype,
            device=inputs.device,
            include_padding=True,
        )
        bias = self._bias_for(dtype=inputs.dtype, device=inputs.device)
        packed = pack_convrot_w4a4_rowwise_weight(weight, bias)
        self._rowwise_w4a4_packed_cache = (signature, packed)
        self._rowwise_w4a4_runner_cache = None
        return packed

    def _rowwise_w4a4_hot_forward(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if (
            self.compute_precision != "w4a4"
            or not self._rowwise_w4a4_requested()
            or self.activation_scale_mode != "dynamic"
            or self.input_already_rotated
            or self.channel_hybrid_enabled
            or not inputs.is_cuda
        ):
            return None
        cached = self._rowwise_w4a4_runner_cache
        if cached is None:
            return None
        device_index, dtype, source_ids, _packed, native_forward = cached
        if (
            device_index != inputs.device.index
            or dtype != inputs.dtype
            or source_ids
            != (id(self.packed_weight), id(self.weight_scale), id(self.bias))
        ):
            self._rowwise_w4a4_runner_cache = None
            return None
        try:
            output = native_forward(inputs)
        except RuntimeError as exc:
            if "XQT_ROWWISE_W4A4_STALE_STATE" not in str(exc):
                raise
            self._rowwise_w4a4_runner_cache = None
            return None
        backend = "native_convrot_w4a4_rowwise_dynamic_runner"
        if self._last_native_w4a4_backend != backend:
            self._last_native_w4a4_used = True
            self._last_native_w4a4_backend = backend
            self._last_native_w4a4_fallback_reason = None
        return output

    def _rowwise_w4a4_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        from xqt.kernels.ops.quantization import (
            bind_dynamic_convrot_w4a4_rowwise_linear,
        )

        packed = self._rowwise_w4a4_packed(inputs)
        native_forward = bind_dynamic_convrot_w4a4_rowwise_linear(
            packed,
            source_weight=self.packed_weight,
            source_weight_scales=self.weight_scale,
            source_bias=self.bias,
            expected_dtype=inputs.dtype,
        )
        output = native_forward(inputs)
        self._rowwise_w4a4_runner_cache = (
            inputs.device.index,
            inputs.dtype,
            (id(self.packed_weight), id(self.weight_scale), id(self.bias)),
            packed,
            native_forward,
        )
        self._last_native_w4a4_used = True
        self._last_native_w4a4_backend = "native_convrot_w4a4_rowwise_dynamic_runner"
        self._last_native_w4a4_fallback_reason = None
        return output

    def _native_w4a4_gate(self, inputs: torch.Tensor) -> tuple[bool, str]:
        if self.w4a4_runtime_backend == "reference":
            return False, "native W4A4 is disabled by w4a4_runtime_backend=reference"
        if self.compute_precision != "w4a4":
            return False, "native W4A4 requires w4a4 compute precision"
        if self.activation_scale_mode != "dynamic":
            return False, "native W4A4 requires dynamic activation scales"
        if not _can_mutate_runtime_cache():
            return False, "native W4A4 eager cache is unavailable while compiling or tracing"
        if not inputs.is_cuda:
            return False, "native W4A4 requires CUDA activations"
        if inputs.dtype not in {torch.float16, torch.bfloat16}:
            return False, "native W4A4 requires float16 or bfloat16 activations"
        if inputs.ndim < 1 or int(inputs.shape[-1]) != self.input_features:
            return False, "native W4A4 input trailing dimension is invalid"
        major, minor = torch.cuda.get_device_capability(inputs.device)
        if (major, minor) != (8, 9):
            return False, f"native W4A4 currently targets sm_89, got sm_{major}{minor}"
        try:
            from xqt.kernels.ops.quantization import (
                native_convrot_w4a4_shape_supported,
                native_w4a4_available,
                native_w4a4_shape_supported,
            )

            if self.input_already_rotated:
                shape_supported = native_w4a4_shape_supported(
                    self.padded_input_features,
                    self.output_features,
                )
            else:
                shape_supported = native_convrot_w4a4_shape_supported(
                    self.input_features,
                    self.padded_input_features,
                    self.output_features,
                    self.rot_size,
                )
            if not shape_supported:
                return False, "native ConvRot W4A4 shape or rotation size is unsupported"
            if not native_w4a4_available(build=False):
                return False, "native W4A4 backend is unavailable"
        except Exception as exc:
            return False, f"native W4A4 capability check failed: {exc}"
        return True, "native ConvRot rotation plus dynamic W4A4 quantization is available"

    def _native_w4a4_packed(self, inputs: torch.Tensor) -> Any:
        from xqt.kernels.ops.quantization import pack_w4a4_linear

        signature = self._w4a4_compute_view_signature(inputs)
        cached = self._native_w4a4_packed_cache
        if cached is not None and cached[0] == signature:
            return cached[1]
        weight = self.dequantized_weight(
            dtype=inputs.dtype,
            device=inputs.device,
            include_padding=True,
        )
        bias = self._bias_for(dtype=inputs.dtype, device=inputs.device)
        packed = pack_w4a4_linear(weight, bias)
        self._native_w4a4_packed_cache = (signature, packed)
        self._native_w4a4_workspace_cache.clear()
        return packed

    def _native_w4a4_workspace(self, inputs: torch.Tensor, packed: Any) -> Any:
        from xqt.kernels.ops.quantization import (
            allocate_w4a4_workspace,
        )

        rows = int(inputs.shape[0])
        padded_rows = ((rows + 255) // 256) * 256
        stream_id = int(torch.cuda.current_stream(inputs.device).cuda_stream)
        key = (
            str(inputs.device),
            str(inputs.dtype),
            padded_rows,
            int(packed.padded_input_features),
            stream_id,
        )
        workspace = self._native_w4a4_workspace_cache.get(key)
        if workspace is not None:
            return workspace
        workspace = allocate_w4a4_workspace(rows, packed)
        if len(self._native_w4a4_workspace_cache) >= 8:
            self._native_w4a4_workspace_cache.clear()
        self._native_w4a4_workspace_cache[key] = workspace
        return workspace

    def _native_w4a4_state_signature(self) -> tuple[Any, ...]:
        tensors = (self.packed_weight, self.weight_scale, self.bias)
        return tuple(
            item
            for tensor in tensors
            for item in (
                0 if tensor is None else id(tensor),
                0 if tensor is None else int(getattr(tensor, "_version", 0)),
            )
        )

    @staticmethod
    def _native_w4a4_hot_key(inputs: torch.Tensor) -> tuple[Any, ...]:
        stream_id = int(torch.cuda.current_stream(inputs.device).cuda_stream)
        return (
            inputs.device.index,
            inputs.dtype,
            int(inputs.shape[0]),
            stream_id,
        )

    def _native_w4a4_hot_forward(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if (
            self.compute_precision != "w4a4"
            or self.w4a4_runtime_backend == "reference"
            or self.activation_scale_mode != "dynamic"
            or self.input_already_rotated
            or self.channel_hybrid_enabled
            or not inputs.is_cuda
        ):
            return None
        flat = inputs if inputs.ndim == 2 else inputs.reshape(-1, self.input_features)
        key = self._native_w4a4_hot_key(flat)
        cached = self._native_w4a4_hot_cache.get(key)
        if cached is None:
            return None
        state_signature, _packed, _workspace, native_forward = cached
        if state_signature != self._native_w4a4_state_signature():
            self._native_w4a4_hot_cache.pop(key, None)
            return None
        output = native_forward(flat)
        backend = "native_convrot_w4a4_dynamic_bound"
        if self._last_native_w4a4_backend != backend:
            self._last_native_w4a4_used = True
            self._last_native_w4a4_backend = backend
            self._last_native_w4a4_fallback_reason = None
        if inputs.ndim == 2:
            return output
        return output.reshape(*inputs.shape[:-1], self.output_features)

    def _native_w4a4_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        from xqt.kernels.ops.quantization import (
            bind_convrot_w4a4_linear,
            w4a4_linear,
        )

        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat = inputs.reshape(-1, self.input_features)
        packed = self._native_w4a4_packed(flat)
        workspace = self._native_w4a4_workspace(flat, packed)
        if self.input_already_rotated:
            padded = _pad_last_dim(flat, self.padded_input_features)
            output = w4a4_linear(padded, packed, workspace=workspace)
            backend = "native_w4a4_dynamic_pre_rotated"
        else:
            native_forward = bind_convrot_w4a4_linear(
                packed,
                workspace,
                rows=int(flat.shape[0]),
                logical_input_features=self.input_features,
                rotated_input_features=self.padded_input_features,
                rot_size=self.rot_size,
            )
            output = native_forward(flat)
            if len(self._native_w4a4_hot_cache) >= 8:
                self._native_w4a4_hot_cache.clear()
            self._native_w4a4_hot_cache[self._native_w4a4_hot_key(flat)] = (
                self._native_w4a4_state_signature(),
                packed,
                workspace,
                native_forward,
            )
            backend = "native_convrot_w4a4_dynamic_bound"
        self._last_native_w4a4_used = True
        self._last_native_w4a4_backend = backend
        self._last_native_w4a4_fallback_reason = None
        return output.reshape(*original_shape, self.output_features)

    def _channel_hybrid_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        device = inputs.device
        bias = self._bias_for(dtype=dtype, device=device)
        mask = self.high_precision_channel_mask.to(device=device)
        rotated = self._rotate_inputs(inputs)
        low_activation = self._quantize_rotated_activation(rotated).to(dtype=dtype)
        high_activation = rotated.to(dtype=dtype)
        low_weight = self.dequantized_weight(
            dtype=dtype,
            device=device,
            include_padding=True,
        )
        high_weight = self.reference_weight.to(device=device, dtype=dtype)
        axis = self.channel_hybrid_axis
        if axis == "input":
            padded_mask = torch.zeros(
                self.padded_input_features,
                dtype=torch.bool,
                device=device,
            )
            padded_mask[: self.input_features] = mask
            low_w = low_weight[:, ~padded_mask]
            high_w = high_weight[:, padded_mask]
            low_act = low_activation[..., ~padded_mask]
            high_act = high_activation[..., padded_mask]
        else:
            low_w = low_weight[~mask, :]
            high_w = high_weight[mask, :]
            low_act = low_activation
            high_act = high_activation
        return compute_hybrid_linear(
            inputs,
            low_weight=low_w,
            high_weight=high_w,
            low_activation=low_act,
            high_activation=high_act,
            bias=bias,
            axis=axis,
            high_precision_mask=mask,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 1:
            raise ValueError("ConvRotMixedPrecisionLinear input rank must be >= 1")
        if int(inputs.shape[-1]) != self.input_features:
            raise ValueError(
                "ConvRotMixedPrecisionLinear input trailing dimension does not "
                "match input_features"
            )
        if not self._xqt_runtime_execution_enabled:
            return self._reference_forward(inputs)
        hot_output = self._rowwise_w4a4_hot_forward(inputs)
        if hot_output is not None:
            return hot_output
        hot_output = self._native_w4a4_hot_forward(inputs)
        if hot_output is not None:
            return hot_output
        self._last_native_w4a4_used = False
        self._last_native_w4a4_backend = None
        self._last_native_w4a4_fallback_reason = None
        if self.channel_hybrid_enabled and bool(
            self.high_precision_channel_mask.any().item()
        ):
            self._last_native_w4a4_fallback_reason = (
                "native full-tensor W4A4 is disabled for channel-hybrid execution"
            )
            return self._channel_hybrid_forward(inputs)

        precision = self.compute_precision
        bias = self._bias_for(dtype=inputs.dtype, device=inputs.device)

        if precision == "w4a4":
            fallback_reasons: list[str] = []
            rowwise_allowed, rowwise_reason = self._rowwise_w4a4_gate(inputs)
            if rowwise_allowed:
                try:
                    return self._rowwise_w4a4_forward(inputs)
                except Exception as exc:
                    rowwise_reason = f"rowwise ConvRot W4A4 execution failed: {exc}"
            fallback_reasons.append(f"rowwise: {rowwise_reason}")
            native_allowed, native_reason = self._native_w4a4_gate(inputs)
            if native_allowed:
                try:
                    return self._native_w4a4_forward(inputs)
                except Exception as exc:
                    native_reason = f"native ConvRot W4A4 execution failed: {exc}"
            fallback_reasons.append(f"nunchaku: {native_reason}")
            self._last_native_w4a4_used = False
            self._last_native_w4a4_backend = None
            self._last_native_w4a4_fallback_reason = "; ".join(fallback_reasons)
            quantized_activation = self._quantize_activation(inputs)
            weight = self.dequantized_weight(
                dtype=inputs.dtype,
                device=inputs.device,
                include_padding=True,
            )
            return F.linear(quantized_activation, weight, bias)

        rotated_inputs = self._rotate_inputs(inputs).to(dtype=inputs.dtype)

        if precision == "w4a16":
            weight = self.dequantized_weight(
                dtype=inputs.dtype,
                device=inputs.device,
                include_padding=True,
            )
            return F.linear(rotated_inputs, weight, bias)

        if precision == "bf16":
            return F.linear(
                rotated_inputs,
                self.reference_weight.to(device=inputs.device, dtype=inputs.dtype),
                bias,
            )

        if precision == "w8a8":
            self._ensure_int8_compute()
            if self._int8_compute is None:
                raise RuntimeError("w8a8 compute path is not materializable")
            return self._int8_compute(rotated_inputs)

        allowed = ", ".join(sorted(SUPPORTED_COMPUTE_PRECISIONS))
        raise ValueError(f"unsupported compute_precision {precision!r}; expected {allowed}")

    def _reference_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.channel_hybrid_enabled and bool(
            self.high_precision_channel_mask.any().item()
        ):
            return self._channel_hybrid_forward(inputs)
        precision = self.compute_precision
        bias = self._bias_for(dtype=inputs.dtype, device=inputs.device)
        if precision == "w4a4":
            quantized_activation = self._quantize_activation(inputs)
            weight = self.dequantized_weight(
                dtype=inputs.dtype,
                device=inputs.device,
                include_padding=True,
            )
            return F.linear(quantized_activation, weight, bias)
        rotated_inputs = self._rotate_inputs(inputs).to(dtype=inputs.dtype)
        if precision == "w4a16":
            weight = self.dequantized_weight(
                dtype=inputs.dtype,
                device=inputs.device,
                include_padding=True,
            )
            return F.linear(rotated_inputs, weight, bias)
        if precision == "bf16":
            return F.linear(
                rotated_inputs,
                self.reference_weight.to(device=inputs.device, dtype=inputs.dtype),
                bias,
            )
        if precision == "w8a8":
            self._ensure_int8_compute()
            if self._int8_compute is None:
                raise RuntimeError("w8a8 compute path is not materializable")
            return self._int8_compute(rotated_inputs)
        allowed = ", ".join(sorted(SUPPORTED_COMPUTE_PRECISIONS))
        raise ValueError(f"unsupported compute_precision {precision!r}; expected {allowed}")

    def execution_metadata(self) -> dict[str, Any]:
        rowwise_backend = (
            self._last_native_w4a4_backend
            == "native_convrot_w4a4_rowwise_dynamic_runner"
        )
        return {
            "implementation": (
                self._last_native_w4a4_backend
                if self._last_native_w4a4_used
                else "convrot_pytorch_reference"
            ),
            "method": "convrot",
            "strategy": "w4a4_int4",
            "compute_precision": self.compute_precision,
            "w4a4_runtime_backend": self.w4a4_runtime_backend,
            "resolved_w4a4_runtime_backend": (
                "rowwise"
                if rowwise_backend
                else "nunchaku"
                if self._last_native_w4a4_used
                else "reference"
            ),
            "rotation_kind": "regular_hadamard",
            "rotation_scope": "groupwise",
            "rot_size": self.rot_size,
            "logical_input_features": self.input_features,
            "padded_input_features": self.padded_input_features,
            "input_already_rotated": bool(self.input_already_rotated),
            "rotation_quant_fused": bool(self._last_native_w4a4_used),
            "native_w4a4_used": bool(self._last_native_w4a4_used),
            "rowwise_w4a4_used": rowwise_backend,
            "native_w4a4_backend": self._last_native_w4a4_backend,
            "native_w4a4_fallback_reason": self._last_native_w4a4_fallback_reason,
            "runtime_weight_layout": (
                "row_major_signed_int4_rowwise"
                if rowwise_backend
                else "nunchaku_packed_int4"
                if self._last_native_w4a4_used
                else "grouped_artifact"
            ),
            "runtime_weight_requantized_from_artifact": bool(
                self._last_native_w4a4_used
            ),
            "fused_epilogue": (
                "activation_scale_weight_scale_bias"
                if rowwise_backend
                else None
            ),
            "norm_fused": False,
            "artifact_view": "contracts_reference",
        }


def _collect_convrot_activation_stats(
    model: nn.Module,
    *,
    module_names: Iterable[str],
    calibration_inputs: Iterable[Any] | None,
    sample_limit: int | None,
    rot_size: int,
    padding_alignment: int = 1,
    quant_max: float = 7.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collect per-tensor activation scales and per-channel abs-max scores."""

    if calibration_inputs is None:
        return {}, {}
    wanted = {str(name) for name in module_names}
    if not wanted:
        return {}, {}
    device = next(model.parameters(), torch.empty((), device="cpu")).device
    maxima: dict[str, float] = {}
    channel_maxima: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    module_rot_sizes: dict[str, int] = {}
    module_padded_features: dict[str, int] = {}
    for name in wanted:
        hook_module = model.get_submodule(name)
        in_features = int(getattr(hook_module, "in_features", 0) or 0)
        module_rot_sizes[name] = (
            _normalize_rot_size(rot_size, in_features) if in_features > 0 else int(rot_size)
        )
        if in_features > 0:
            module_padded_features[name] = _rotation_padded_features(
                in_features,
                module_rot_sizes[name],
                alignment=int(padding_alignment),
            )

    def _make_hook(name: str) -> Any:
        def _hook(module: nn.Module, inputs: tuple[Any, ...], _: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            activation = inputs[0].detach().to(torch.float32)
            if activation.ndim < 1:
                return
            if activation.shape[-1] != getattr(
                module, "in_features", activation.shape[-1]
            ):
                return
            padded_features = module_padded_features.get(name)
            if padded_features is None:
                rotated = _apply_groupwise_rotation(
                    activation,
                    rot_size=module_rot_sizes.get(name, int(rot_size)),
                )
            else:
                rotated = _apply_groupwise_rotation(
                    _pad_last_dim(activation, padded_features),
                    rot_size=module_rot_sizes.get(name, int(rot_size)),
                    return_padded=True,
                )
            flattened = rotated.reshape(-1, rotated.shape[-1])
            current_max = float(flattened.abs().amax().item())
            previous = maxima.get(name, 0.0)
            maxima[name] = max(previous, current_max)
            channel_score = flattened.abs().amax(dim=0)
            previous_channel = channel_maxima.get(name)
            if previous_channel is None:
                channel_maxima[name] = channel_score.clone()
            else:
                channel_maxima[name] = torch.maximum(previous_channel, channel_score)

        return _hook

    for name in wanted:
        hook_module = model.get_submodule(name)
        handles.append(hook_module.register_forward_hook(_make_hook(name)))
    try:
        for batch in _iter_calibration_batches(calibration_inputs, sample_limit=sample_limit):
            moved = _move_batch_to_device(batch, device)
            with torch.no_grad():
                _call_model(model, moved)
    finally:
        for handle in handles:
            handle.remove()
    if float(quant_max) <= 0.0:
        raise ValueError("quant_max must be positive")
    scales = {
        name: torch.tensor(max(value / float(quant_max), 1e-6), dtype=torch.float32)
        for name, value in maxima.items()
    }
    return scales, channel_maxima


def quantize_with_convrot_4bit(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    calibration_inputs: Iterable[Any] | None = None,
    inplace: bool = True,
) -> ConvRot4BitQuantizationResult:
    """Quantize Linear modules with ConvRot-inspired group-wise W4A4."""

    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    configured_rot_size = int(
        policy_mapping["rot_size"] if "rot_size" in policy_mapping else 256
    )
    configured_group_size = int(
        policy_mapping["group_size"] if "group_size" in policy_mapping else 128
    )
    _normalize_rot_size(configured_rot_size, 1)
    _normalize_convrot_group_size(configured_group_size)
    sample_limit = (
        int(policy_mapping["sample_limit"]) if "sample_limit" in policy_mapping else None
    )
    mixed_ratio = float(policy_mapping.get("mixed_precision_ratio", 0.2) or 0.2)
    channel_hybrid_ratio = float(policy_mapping.get("channel_hybrid_ratio", 0.0) or 0.0)
    channel_hybrid_axis = normalize_channel_axis(
        str(policy_mapping.get("channel_hybrid_axis", "input") or "input")
    )
    channel_high_precision = str(
        policy_mapping.get("channel_high_precision", "bf16") or "bf16"
    )
    channel_low_precision = str(
        policy_mapping.get("channel_low_precision", "w4a4") or "w4a4"
    )
    activation_scale_mode = str(
        policy_mapping.get("activation_scale_mode", "dynamic") or "dynamic"
    ).strip().lower()
    if activation_scale_mode not in _ACTIVATION_SCALE_MODES:
        raise ValueError("activation_scale_mode must be dynamic or static")
    default_compute_precision = normalize_compute_precision(
        str(policy_mapping.get("default_compute_precision", "w4a4") or "w4a4")
    )
    w4a4_runtime_backend = _normalize_w4a4_runtime_backend(
        str(policy_mapping.get("w4a4_runtime_backend", "auto") or "auto")
    )
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": "int4",
                "scheme": "convrot_w4a4",
            },
        )
        or "convrot_w4a4"
    )
    target_model = model if inplace else copy.deepcopy(model)
    candidate_names: list[str] = []
    for name, module in list(target_model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue
        candidate_names.append(name)
    activation_scales, channel_scores = _collect_convrot_activation_stats(
        target_model,
        module_names=candidate_names,
        calibration_inputs=calibration_inputs,
        sample_limit=sample_limit,
        rot_size=configured_rot_size,
        padding_alignment=configured_group_size,
    )
    if (
        activation_scales
        and activation_scale_mode == "dynamic"
        and "activation_scale_mode" not in policy_mapping
    ):
        activation_scale_mode = "static"
    quantized_modules: list[str] = []
    channel_overrides: list[dict[str, Any]] = []
    module_feature_shapes: dict[str, dict[str, int]] = {}
    for name, module in list(target_model.named_modules()):
        if name not in candidate_names:
            continue
        activation_scale = activation_scales.get(name, torch.tensor(1.0, dtype=torch.float32))
        high_precision_channels: tuple[int, ...] = ()
        if channel_hybrid_ratio > 0.0:
            if channel_hybrid_axis == "input":
                scores = channel_scores.get(name)
                if scores is None:
                    scores = module.weight.detach().to(torch.float32).abs().amax(dim=0)
            else:
                scores = module.weight.detach().to(torch.float32).abs().amax(dim=1)
            high_precision_channels = select_outlier_channels(
                scores,
                ratio=channel_hybrid_ratio,
            )
        replacement = ConvRotMixedPrecisionLinear.from_linear(
            module,
            rot_size=configured_rot_size,
            group_size=configured_group_size,
            activation_scale=activation_scale,
            activation_scale_mode=activation_scale_mode,
            compute_precision=default_compute_precision,
            w4a4_runtime_backend=w4a4_runtime_backend,
            high_precision_channels=high_precision_channels,
            channel_hybrid_axis=channel_hybrid_axis,
            channel_high_precision=channel_high_precision,
            channel_low_precision=channel_low_precision,
        )
        replacement._xqt_runtime_execution_enabled = False
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)
        module_feature_shapes[name] = {
            "logical_input_features": int(replacement.input_features),
            "padded_input_features": int(replacement.padded_input_features),
            "rotation_size": int(replacement.rot_size),
            "group_size": int(replacement.group_size),
        }
        if high_precision_channels:
            channel_overrides.append(
                {
                    "module": name,
                    "axis": channel_hybrid_axis,
                    "high_precision_channels": list(high_precision_channels),
                    "high_precision": channel_high_precision,
                    "low_precision": channel_low_precision,
                    "enabled": True,
                    "reason": "channel_absmax_topk",
                }
            )

    execution_policies: list[dict[str, Any]] = []
    sensitivity_records: list[LayerSensitivityRecord] = []
    if calibration_inputs is not None and quantized_modules:
        first_batch = next(
            iter(_iter_calibration_batches(calibration_inputs, sample_limit=1)),
            None,
        )
        if first_batch is not None:
            moved = _move_batch_to_device(
                first_batch,
                next(target_model.parameters(), torch.empty((), device="cpu")).device,
            )
            sensitivity_records = analyze_layer_sensitivity(
                model,
                target_model,
                moved,
                module_names=quantized_modules,
            )
            execution_policies = _build_execution_policies(
                sensitivity_records,
                mixed_ratio=mixed_ratio,
                runtime_strategy="convrot_mixed_w4a4_w8a8",
            )
    if channel_overrides:
        if execution_policies:
            execution_policies[0]["channel_overrides"] = channel_overrides
            execution_policies[0]["runtime_strategy"] = (
                "convrot_channel_hybrid_w4a4_bf16"
            )
            execution_policies[0]["policy_kind"] = "channel_mixed_precision"
        else:
            execution_policies = [
                {
                    "policy_kind": "channel_mixed_precision",
                    "runtime": "pytorch",
                    "runtime_strategy": "convrot_channel_hybrid_w4a4_bf16",
                    "channel_hybrid_ratio": float(channel_hybrid_ratio),
                    "channel_hybrid_axis": channel_hybrid_axis,
                    "precision_overrides": [],
                    "channel_overrides": channel_overrides,
                }
            ]

    return ConvRot4BitQuantizationResult(
        model=target_model,
        method="convrot",
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "convrot_groupwise_regular_hadamard_w4a4",
            "quantization_nature": "pseudo",
            "quantization_nature_scope": "current_xqt_runtime_implementation",
            "weight_encoding": "packed_signed_int4",
            "activation_encoding": (
                "symmetric_int4_per_token_reference"
                if activation_scale_mode == "dynamic"
                else "symmetric_int4_per_tensor_reference"
            ),
            "activation_granularity": (
                "per_token" if activation_scale_mode == "dynamic" else "per_tensor"
            ),
            "activation_scale_mode": activation_scale_mode,
            "default_compute_precision": default_compute_precision,
            "w4a4_runtime_backend": w4a4_runtime_backend,
            "supported_compute_precisions": sorted(SUPPORTED_COMPUTE_PRECISIONS),
            "precision_description": {
                "quantization_time": {
                    "weight": "offline rotated and packed signed INT4 with group scales",
                    "activation": (
                        "not stored as an activation artifact; runtime uses "
                        f"{activation_scale_mode} symmetric INT4 "
                        f"({'per-token' if activation_scale_mode == 'dynamic' else 'per-tensor'}) "
                        "when compute_precision=w4a4"
                    ),
                },
                "runtime": {
                    "w4a4": (
                        "auto preserves grouped-artifact semantics through the native "
                        "Nunchaku layout unless the artifact has one whole-row scale; "
                        "rowwise explicitly selects warp-FHT rowwise INT4 plus CUTLASS "
                        "INT4 Tensor Core GEMM; reference uses dequantized F.linear"
                    ),
                    "w4a16": "packed W4 is dequantized; activation stays in the input float dtype",
                    "bf16": "uses the retained floating-point reference weight",
                    "w8a8": "delegates to Int8MmaLinear; inspect its runtime_precision metadata",
                    "precision_change": (
                        "there is no automatic precision fallback in this module; "
                        "only an explicit execution policy or channel-hybrid override changes compute_precision"
                    ),
                },
            },
            "group_size": configured_group_size,
            "rot_size": configured_rot_size,
            "feature_padding": "internal_to_rotation_and_weight_group_alignment",
            "module_feature_shapes": module_feature_shapes,
            "channel_hybrid_ratio": float(channel_hybrid_ratio),
            "channel_hybrid_axis": channel_hybrid_axis,
            "algorithm_metadata": {
                "rotation_kind": "regular_hadamard",
                "rotation_scope": "groupwise",
                "rot_size": configured_rot_size,
                "group_size": configured_group_size,
                "feature_padding": "internal_to_rotation_and_weight_group_alignment",
                "weight_bits": 4,
                "activation_bits": 4,
                "activation_scale_mode": activation_scale_mode,
                "compute_precisions": sorted(SUPPORTED_COMPUTE_PRECISIONS),
                "w4a4_runtime_backend": w4a4_runtime_backend,
                "channel_hybrid": {
                    "axis": channel_hybrid_axis,
                    "ratio": float(channel_hybrid_ratio),
                    "high_precision": channel_high_precision,
                    "low_precision": channel_low_precision,
                },
            },
            "execution_policies": execution_policies,
            "sensitivity": [
                {
                    "name": record.name,
                    "module_type": record.module_type,
                    "mean_abs": record.diff.mean_abs,
                    "max_abs": record.diff.max_abs,
                }
                for record in sensitivity_records
            ],
            "policy": {
                "dtype": quant_policy.dtype,
                "scheme": quant_policy.scheme,
                "include_module_types": list(quant_policy.include_module_types),
                "exclude_module_types": list(quant_policy.exclude_module_types),
                "include_name_patterns": list(quant_policy.include_name_patterns),
                "exclude_name_patterns": list(quant_policy.exclude_name_patterns),
                "include_module_names": list(quant_policy.include_module_names),
                "exclude_module_names": list(quant_policy.exclude_module_names),
                "min_parameters": quant_policy.min_parameters,
                "group_size": configured_group_size,
                "rot_size": configured_rot_size,
                "mixed_precision_ratio": mixed_ratio,
                "channel_hybrid_ratio": float(channel_hybrid_ratio),
                "channel_hybrid_axis": channel_hybrid_axis,
                "activation_scale_mode": activation_scale_mode,
                "default_compute_precision": default_compute_precision,
                "w4a4_runtime_backend": w4a4_runtime_backend,
            },
        },
    )


def execute_convrot_4bit_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_convrot_4bit,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the ConvRot 4-bit quantizer for one component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        calibration_inputs=context.calibration_inputs,
        inplace=True,
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    raw_execution_policies = result.metadata.get("execution_policies", [])
    execution_policies: list[dict[str, Any]] = []
    if isinstance(raw_execution_policies, list):
        for item in raw_execution_policies:
            if not isinstance(item, Mapping):
                continue
            policy_item = dict(item)
            overrides = policy_item.get("precision_overrides", [])
            if isinstance(overrides, list):
                policy_item["precision_overrides"] = [
                    {
                        **dict(override),
                        "module": prefix_module_names(
                            [str(override["module"])],
                            component.target_path,
                        )[0],
                    }
                    for override in overrides
                    if isinstance(override, Mapping) and "module" in override
                ]
            channel_items = policy_item.get("channel_overrides", [])
            if isinstance(channel_items, list):
                policy_item["channel_overrides"] = [
                    {
                        **dict(override),
                        "module": prefix_module_names(
                            [str(override["module"])],
                            component.target_path,
                        )[0],
                    }
                    for override in channel_items
                    if isinstance(override, Mapping) and "module" in override
                ]
            execution_policies.append(policy_item)
    recommended_high_precision = []
    if execution_policies:
        overrides = execution_policies[0].get("precision_overrides", [])
        if isinstance(overrides, list):
            recommended_high_precision = [
                str(item["module"])
                for item in overrides
                if isinstance(item, Mapping) and "module" in item
            ]
    report = build_component_quantization_report(
        context,
        component,
        backend=component.backend,
        method=component.method or "convrot",
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=True,
        method_semantics="groupwise_regular_hadamard_rotation_w4a4_quantization",
        effective_policy=effective_policy,
        result_metadata=result.metadata,
        execution_state="convrot_4bit",
        extra_metadata={
            "execution_policies": execution_policies,
            "recommended_high_precision_modules": recommended_high_precision,
        },
    )
    return updated_model, report


__all__ = [
    "ConvRotMixedPrecisionLinear",
    "ConvRot4BitQuantizationResult",
    "build_regular_hadamard_matrix",
    "execute_convrot_4bit_component",
    "quantize_with_convrot_4bit",
]
