"""ConvRot-inspired group-wise regular Hadamard 4-bit quantization.

Quantization produces packed weight artifacts, rotation matrices, activation
scales, and optional mixed-precision execution policies.

Inference / mixed-precision dispatch lives in ``xqt.runtime`` and only consumes
those artifacts. This module never owns a hybrid inference engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
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
from ..execution.reporting import optional_calibration_summary
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
from .fp4_weight_only import (
    _normalize_group_size,
    _pack_int4,
    _quantize_grouped_fp4_weight,
    _unpack_int4,
)
from .int8_mma import Int8MmaLinear

_ACTIVATION_SCALE_MODES = frozenset({"dynamic", "static"})


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


def build_regular_hadamard_matrix(order: int) -> torch.Tensor:
    """Return the regular Hadamard matrix from ConvRot Theorem 3.3 for 4^k orders."""

    normalized_order = int(order)
    if normalized_order < 1:
        raise ValueError("order must be positive")
    if normalized_order == 1:
        return torch.ones((1, 1), dtype=torch.float32)
    if normalized_order == 4:
        return _base_regular_hadamard4()
    value = normalized_order
    while value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError("regular Hadamard order must be a power of four")
    matrix = _base_regular_hadamard4()
    current_order = 4
    while current_order < normalized_order:
        matrix = torch.kron(matrix, _base_regular_hadamard4())
        current_order *= 4
    return matrix


def _normalize_rot_size(rot_size: int, input_features: int) -> int:
    """Pick the largest regular-Hadamard order <= rot_size that divides features."""

    features = int(input_features)
    if features < 1:
        raise ValueError("input_features must be positive")
    requested = max(1, min(int(rot_size), features))
    best = 1
    order = 1
    while order <= requested:
        if features % order == 0:
            best = order
        if order == 1:
            order = 4
        else:
            order *= 4
    return best


def _normalized_regular_hadamard(order: int, *, device: torch.device) -> torch.Tensor:
    matrix = build_regular_hadamard_matrix(order).to(device=device)
    return matrix / float(order) ** 0.5


def _apply_groupwise_rotation(
    tensor: torch.Tensor,
    *,
    rot_size: int,
    rotation_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("rotation tensor rank must be >= 1")
    feature_dim = int(tensor.shape[-1])
    if feature_dim % int(rot_size) != 0:
        raise ValueError(
            f"feature dim {feature_dim} must be divisible by rot_size {int(rot_size)}"
        )
    rotation = (
        rotation_matrix
        if rotation_matrix is not None
        else _normalized_regular_hadamard(int(rot_size), device=tensor.device)
    )
    reshaped = tensor.reshape(-1, feature_dim // int(rot_size), int(rot_size))
    rotated = torch.matmul(reshaped, rotation.to(dtype=tensor.dtype, device=tensor.device))
    return rotated.reshape(*tensor.shape)


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    if isinstance(batch, Mapping):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    return batch


def _call_model(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    return model(inputs)


def _iter_calibration_batches(
    calibration_inputs: Iterable[Any] | None,
    *,
    sample_limit: int | None,
) -> Iterable[Any]:
    if calibration_inputs is None:
        return
    for index, batch in enumerate(calibration_inputs):
        if sample_limit is not None and index >= sample_limit:
            break
        yield batch


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def _policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key == "dtype":
            kwargs["dtype"] = str(value)
        elif key == "scheme":
            kwargs["scheme"] = str(value)
        elif key in {
            "include_module_types",
            "exclude_module_types",
            "include_name_patterns",
            "exclude_name_patterns",
            "include_module_names",
            "exclude_module_names",
        }:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


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
    strategy: str = "convrot_w4a4"


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
        high_precision_channel_mask: torch.Tensor | None = None,
        channel_hybrid_axis: str = "input",
        channel_high_precision: str = "bf16",
        channel_low_precision: str = "w4a4",
        channel_hybrid_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.rot_size = int(rot_size)
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
        self.register_buffer(
            "reference_weight",
            reference_weight.detach().to(torch.float32).contiguous(),
        )
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
        self._int8_compute: Int8MmaLinear | None = None
        self.compute_precision = normalize_compute_precision(compute_precision)
        self._ensure_int8_compute()

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
        high_precision_channels: Sequence[int] | torch.Tensor | None = None,
        channel_hybrid_axis: str = "input",
        channel_high_precision: str = "bf16",
        channel_low_precision: str = "w4a4",
    ) -> "ConvRotMixedPrecisionLinear":
        weight = module.weight.detach().to(torch.float32)
        input_features = int(module.in_features)
        output_features = int(module.out_features)
        normalized_rot_size = _normalize_rot_size(rot_size, input_features)
        if input_features % normalized_rot_size != 0:
            raise ValueError(
                "ConvRot mixed-precision linear requires input_features divisible by rot_size"
            )
        rotation = _normalized_regular_hadamard(
            normalized_rot_size,
            device=weight.device,
        )
        rotated_weight = _apply_groupwise_rotation(
            weight,
            rot_size=normalized_rot_size,
            rotation_matrix=rotation,
        )
        normalized_group_size = _normalize_group_size(group_size, input_features)
        packed_weight, weight_scale, padded_input_features = _quantize_grouped_fp4_weight(
            rotated_weight,
            group_size=normalized_group_size,
            input_features=input_features,
            output_features=output_features,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        axis = normalize_channel_axis(channel_hybrid_axis)
        dim_size = input_features if axis == "input" else output_features
        mask = build_channel_mask(
            dim_size,
            high_precision_channels or (),
            device=weight.device,
        )
        return cls(
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
            high_precision_channel_mask=mask,
            channel_hybrid_axis=axis,
            channel_high_precision=channel_high_precision,
            channel_low_precision=channel_low_precision,
            channel_hybrid_enabled=bool(mask.any().item()),
        )

    def quantized_weight_codes(self) -> torch.Tensor:
        return _unpack_int4(self.packed_weight, self.padded_input_features)[
            :, : self.input_features
        ]

    def dequantized_weight(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        codes = _unpack_int4(self.packed_weight, self.padded_input_features).to(
            device=device,
            dtype=torch.float32,
        )
        grouped = codes.reshape(self.output_features, -1, self.group_size)
        dequantized = grouped * self.weight_scale.to(device=device, dtype=torch.float32)
        dense = dequantized.reshape(self.output_features, self.padded_input_features)[
            :, : self.input_features
        ]
        return dense.to(dtype=dtype)

    def _bias_for(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(device=device, dtype=dtype)

    def _rotate_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return _apply_groupwise_rotation(
            inputs.to(torch.float32),
            rot_size=self.rot_size,
            rotation_matrix=self.rotation_matrix.to(device=inputs.device),
        )

    def _ensure_int8_compute(self) -> None:
        if self.compute_precision != "w8a8" or self._int8_compute is not None:
            return
        dense_weight = self.reference_weight.to(torch.float32)
        dense_bias = None if self.bias is None else self.bias.to(torch.float32)
        linear = nn.Linear(self.input_features, self.output_features, bias=dense_bias is not None)
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
        amax = rotated.detach().abs().amax()
        return torch.clamp(amax / 7.0, min=1e-6)

    def _quantize_rotated_activation(self, rotated: torch.Tensor) -> torch.Tensor:
        scale = self._activation_scale(rotated)
        quantized = torch.clamp(torch.round(rotated / scale), min=-8, max=7)
        return quantized * scale

    def _quantize_activation(self, inputs: torch.Tensor) -> torch.Tensor:
        rotated = self._rotate_inputs(inputs)
        return self._quantize_rotated_activation(rotated).to(dtype=inputs.dtype)

    def _channel_hybrid_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        device = inputs.device
        bias = self._bias_for(dtype=dtype, device=device)
        mask = self.high_precision_channel_mask.to(device=device)
        rotated = self._rotate_inputs(inputs)
        low_activation = self._quantize_rotated_activation(rotated).to(dtype=dtype)
        high_activation = rotated.to(dtype=dtype)
        low_weight = self.dequantized_weight(dtype=dtype, device=device)
        high_weight = self.reference_weight.to(device=device, dtype=dtype)
        axis = self.channel_hybrid_axis
        if axis == "input":
            low_w = low_weight[:, ~mask]
            high_w = high_weight[:, mask]
            low_act = low_activation[..., ~mask]
            high_act = high_activation[..., mask]
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
        if self.channel_hybrid_enabled and bool(
            self.high_precision_channel_mask.any().item()
        ):
            return self._channel_hybrid_forward(inputs)

        precision = self.compute_precision
        bias = self._bias_for(dtype=inputs.dtype, device=inputs.device)

        if precision == "w4a4":
            quantized_activation = self._quantize_activation(inputs)
            weight = self.dequantized_weight(dtype=inputs.dtype, device=inputs.device)
            return F.linear(quantized_activation, weight, bias)

        rotated_inputs = self._rotate_inputs(inputs).to(dtype=inputs.dtype)

        if precision == "w4a16":
            weight = self.dequantized_weight(dtype=inputs.dtype, device=inputs.device)
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


def _collect_convrot_activation_stats(
    model: nn.Module,
    *,
    module_names: Iterable[str],
    calibration_inputs: Iterable[Any] | None,
    sample_limit: int | None,
    rot_size: int,
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
    for name in wanted:
        hook_module = model.get_submodule(name)
        in_features = int(getattr(hook_module, "in_features", 0) or 0)
        module_rot_sizes[name] = (
            _normalize_rot_size(rot_size, in_features) if in_features > 0 else int(rot_size)
        )

    def _make_hook(name: str) -> Any:
        def _hook(module: nn.Module, inputs: tuple[Any, ...], _: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            activation = inputs[0].detach().to(torch.float32)
            if activation.shape[-1] != getattr(module, "in_features", activation.shape[-1]):
                return
            rotated = _apply_groupwise_rotation(
                activation,
                rot_size=module_rot_sizes.get(name, int(rot_size)),
            )
            current_max = float(rotated.abs().amax().item())
            previous = maxima.get(name, 0.0)
            maxima[name] = max(previous, current_max)
            channel_score = rotated.abs().amax(dim=tuple(range(rotated.ndim - 1)))
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
    scales = {
        name: torch.tensor(max(value / 7.0, 1e-6), dtype=torch.float32)
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
    configured_rot_size = int(policy_mapping.get("rot_size", 256) or 256)
    configured_group_size = int(policy_mapping.get("group_size", 128) or 128)
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
    )
    if (
        activation_scales
        and activation_scale_mode == "dynamic"
        and "activation_scale_mode" not in policy_mapping
    ):
        activation_scale_mode = "static"
    quantized_modules: list[str] = []
    channel_overrides: list[dict[str, Any]] = []
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
            high_precision_channels=high_precision_channels,
            channel_hybrid_axis=channel_hybrid_axis,
            channel_high_precision=channel_high_precision,
            channel_low_precision=channel_low_precision,
        )
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)
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
            "weight_encoding": "packed_signed_int4",
            "activation_encoding": "symmetric_int4_reference",
            "activation_scale_mode": activation_scale_mode,
            "default_compute_precision": default_compute_precision,
            "supported_compute_precisions": sorted(SUPPORTED_COMPUTE_PRECISIONS),
            "group_size": configured_group_size,
            "rot_size": configured_rot_size,
            "channel_hybrid_ratio": float(channel_hybrid_ratio),
            "channel_hybrid_axis": channel_hybrid_axis,
            "algorithm_metadata": {
                "rotation_kind": "regular_hadamard",
                "rotation_scope": "groupwise",
                "rot_size": configured_rot_size,
                "group_size": configured_group_size,
                "weight_bits": 4,
                "activation_bits": 4,
                "activation_scale_mode": activation_scale_mode,
                "compute_precisions": sorted(SUPPORTED_COMPUTE_PRECISIONS),
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
    high_precision_modules = prefix_module_names(
        component.keep_high_precision,
        component.target_path,
    )
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    quantized_modules = prefix_module_names(result.quantized_modules, component.target_path)
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
    module_selection_reasons = module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = optional_calibration_summary(
        context,
        component,
    )
    report = QuantizationReport(
        component_name=component.name,
        backend=component.backend,
        runtime="pytorch",
        method=component.method or "convrot",
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=True,
        method_semantics="groupwise_regular_hadamard_rotation_w4a4_quantization",
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "execution_policies": execution_policies,
            "algorithm_executable": True,
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "execution_state": "convrot_4bit",
            "recommended_high_precision_modules": recommended_high_precision,
            "executed": True,
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
