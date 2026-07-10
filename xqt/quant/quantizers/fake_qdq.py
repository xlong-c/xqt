"""PyTorch fake-QDQ surrogate helpers for analysis-only layer inspection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.schema import QuantConfig
from xqt.core.types import XQTContext
from xqt.workflows.stage_specs import QuantStageSpec

from ..plan import build_quantization_plan
from ..policy import QuantizationPolicy, list_quantizable_modules
from ..types import QuantizationComponentPlan


_LINEAR_MODULE_TYPES = {"Linear", "Conv1d", "Conv2d", "Conv3d"}
_PER_CHANNEL_WEIGHT_DIMS = {
    "Linear": 0,
    "Conv1d": 0,
    "Conv2d": 0,
    "Conv3d": 0,
}
_OP_TYPE_TO_MODULE_TYPES = {
    "Conv": ("Conv1d", "Conv2d", "Conv3d"),
    "Gemm": ("Linear",),
    "MatMul": ("Linear",),
}


@dataclass(frozen=True)
class FakeQDQSurrogateResult:
    """Analysis-only fake-QDQ model plus lightweight metadata."""

    model: nn.Module
    quantized_modules: list[str]
    activation_statistics: dict[str, dict[str, float]]
    sample_count: int


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _prefix_module_names(names: Iterable[str], prefix: Optional[str]) -> list[str]:
    if not prefix:
        return [str(name) for name in names]
    prefixed: list[str] = []
    for name in names:
        string_name = str(name)
        prefixed.append(f"{prefix}.{string_name}" if string_name else prefix)
    return prefixed


def _resolve_calibration_inputs(context: XQTContext) -> Iterable[Any]:
    if context.calibration_inputs is None:
        return ()
    return context.calibration_inputs


def _build_policy(component: QuantizationComponentPlan) -> QuantizationPolicy:
    policy = dict(component.policy)
    configured_include_types = policy.get("include_module_types")
    requested_op_types = [
        str(item) for item in policy.get("op_types_to_quantize") or ()
    ]
    derived_include_types: list[str] = []
    for op_type in requested_op_types:
        derived_include_types.extend(_OP_TYPE_TO_MODULE_TYPES.get(op_type, ()))
    include_module_types = tuple(
        str(item)
        for item in (
            configured_include_types
            or derived_include_types
            or ("Linear", "Conv1d", "Conv2d", "Conv3d")
        )
    )
    exclude_module_types = tuple(
        str(item)
        for item in policy.get("exclude_module_types")
        or (
            "LayerNorm",
            "BatchNorm1d",
            "BatchNorm2d",
            "BatchNorm3d",
            "Embedding",
        )
    )
    include_name_patterns = tuple(
        str(item) for item in policy.get("include_name_patterns") or ()
    )
    exclude_name_patterns = tuple(
        str(item) for item in policy.get("exclude_name_patterns") or ("head", "classifier")
    )
    include_module_names = _ordered_unique(
        [
            *[str(item) for item in policy.get("include_module_names") or ()],
            *[str(item) for item in component.force_quantize],
        ]
    )
    exclude_module_names = _ordered_unique(
        [
            *[str(item) for item in policy.get("exclude_module_names") or ()],
            *[str(item) for item in component.skip_quantize],
            *[str(item) for item in component.keep_high_precision],
        ]
    )
    min_parameters = int(policy.get("min_parameters", 0))
    return QuantizationPolicy(
        dtype=str(policy.get("dtype", "int8")),
        scheme=str(policy.get("scheme", "weight_only")),
        include_module_types=include_module_types,
        exclude_module_types=exclude_module_types,
        include_name_patterns=include_name_patterns,
        exclude_name_patterns=exclude_name_patterns,
        include_module_names=tuple(include_module_names),
        exclude_module_names=tuple(exclude_module_names),
        min_parameters=min_parameters,
    )


def _quant_range_for_dtype(dtype: str, *, symmetric: bool) -> tuple[int, int]:
    normalized = dtype.lower()
    if normalized in {"qint8", "int8"}:
        return (-127, 127) if symmetric else (-128, 127)
    if normalized in {"quint8", "uint8"}:
        return (0, 255)
    raise ValueError(f"unsupported fake-QDQ dtype: {dtype}")


def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(scale.dtype).eps
    return torch.where(scale > 0, scale, torch.full_like(scale, epsilon))


def _reshape_channel_scale(
    scale: torch.Tensor,
    tensor: torch.Tensor,
    channel_axis: int,
) -> torch.Tensor:
    shape = [1] * tensor.ndim
    shape[channel_axis] = int(scale.numel())
    return scale.reshape(shape)


def _fake_quantize_tensor(
    tensor: torch.Tensor,
    *,
    dtype: str,
    symmetric: bool,
    per_channel_axis: Optional[int] = None,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor

    qmin, qmax = _quant_range_for_dtype(dtype, symmetric=symmetric)
    value = tensor.detach().to(dtype=torch.float32)
    if per_channel_axis is None:
        max_abs = value.abs().max()
        scale = _safe_scale(max_abs / float(qmax if qmax > 0 else 1))
        if qmin >= 0:
            minimum = value.min()
            maximum = value.max()
            scale = _safe_scale((maximum - minimum) / float(qmax - qmin))
            zero_point = torch.round(torch.tensor(qmin, dtype=torch.float32) - minimum / scale)
            zero_point = torch.clamp(zero_point, qmin, qmax)
        else:
            zero_point = torch.zeros((), dtype=torch.float32)
        quantized = torch.clamp(torch.round(value / scale + zero_point), qmin, qmax)
        return (quantized - zero_point) * scale

    reduce_dims = [dim for dim in range(value.ndim) if dim != per_channel_axis]
    max_abs = value.abs().amax(dim=reduce_dims)
    scale = _safe_scale(max_abs / float(qmax if qmax > 0 else 1))
    scale_view = _reshape_channel_scale(scale, value, per_channel_axis)
    if qmin >= 0:
        minimum = value.amin(dim=reduce_dims)
        maximum = value.amax(dim=reduce_dims)
        scale = _safe_scale((maximum - minimum) / float(qmax - qmin))
        scale_view = _reshape_channel_scale(scale, value, per_channel_axis)
        zero_point = torch.round(
            torch.tensor(qmin, dtype=torch.float32, device=value.device)
            - minimum / scale
        )
        zero_point = torch.clamp(zero_point, qmin, qmax)
        zero_point_view = _reshape_channel_scale(zero_point, value, per_channel_axis)
    else:
        zero_point_view = torch.zeros_like(scale_view)
    quantized = torch.clamp(
        torch.round(value / scale_view + zero_point_view),
        qmin,
        qmax,
    )
    return (quantized - zero_point_view) * scale_view


def _target_names_for_component(
    model: nn.Module,
    component: QuantizationComponentPlan,
    policy: QuantizationPolicy,
) -> list[str]:
    if component.target_path:
        target_model = model.get_submodule(component.target_path)
        return _prefix_module_names(
            [
                candidate.name
                for candidate in list_quantizable_modules(target_model, policy)
                if candidate.quantize
            ],
            component.target_path,
        )
    return [
        candidate.name
        for candidate in list_quantizable_modules(model, policy)
        if candidate.quantize
    ]


def _apply_weight_fake_qdq(
    model: nn.Module,
    component: QuantizationComponentPlan,
) -> list[str]:
    policy = _build_policy(component)
    dtype = str(component.policy.get("weight_type") or policy.dtype or "QInt8")
    symmetric = bool(component.policy.get("weight_symmetric", True))
    per_channel = bool(component.policy.get("per_channel", False))
    quantized_names: list[str] = []
    for name in _target_names_for_component(model, component, policy):
        module = model.get_submodule(name)
        module_type = type(module).__name__
        if module_type not in _LINEAR_MODULE_TYPES:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        per_channel_axis = (
            _PER_CHANNEL_WEIGHT_DIMS[module_type]
            if per_channel and module_type in _PER_CHANNEL_WEIGHT_DIMS
            else None
        )
        fake_weight = _fake_quantize_tensor(
            weight,
            dtype=dtype,
            symmetric=symmetric,
            per_channel_axis=per_channel_axis,
        ).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight.copy_(fake_weight)
        quantized_names.append(name)
    return quantized_names


def _iter_input_batches(
    dataloader: Iterable[Any],
    *,
    expected_input_count: Optional[int],
    sample_limit: Optional[int],
) -> list[Any]:
    batches: list[Any] = []
    for index, batch in enumerate(dataloader):
        if sample_limit is not None and index >= sample_limit:
            break
        batches.append(
            extract_model_inputs(batch, expected_input_count=expected_input_count)
        )
    return batches


def _module_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _activation_stat_collector(
    model: nn.Module,
    module_names: Sequence[str],
    batches: Sequence[Any],
) -> dict[str, dict[str, float]]:
    if not module_names or not batches:
        return {}
    modules = dict(model.named_modules())
    model_device = _module_device(model)
    statistics: dict[str, dict[str, float]] = {
        name: {"minimum": float("inf"), "maximum": float("-inf")}
        for name in module_names
    }
    handles = []
    try:
        for name in module_names:
            module = modules.get(name)
            if module is None:
                continue

            def make_hook(module_name: str):
                def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
                    if isinstance(output, (tuple, list)):
                        output = output[0] if output else output
                    if not isinstance(output, torch.Tensor):
                        return
                    flat = output.detach().to(dtype=torch.float32, device="cpu")
                    statistics[module_name]["minimum"] = min(
                        statistics[module_name]["minimum"],
                        float(flat.min().item()),
                    )
                    statistics[module_name]["maximum"] = max(
                        statistics[module_name]["maximum"],
                        float(flat.max().item()),
                    )

                return hook

            handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for batch_inputs in batches:
                moved_batch_inputs = _move_to_device(batch_inputs, model_device)
                if isinstance(moved_batch_inputs, Mapping):
                    model(**dict(moved_batch_inputs))
                elif isinstance(moved_batch_inputs, tuple):
                    model(*moved_batch_inputs)
                elif isinstance(moved_batch_inputs, list):
                    model(*moved_batch_inputs)
                else:
                    model(moved_batch_inputs)
    finally:
        while handles:
            handles.pop().remove()
    return {
        name: value
        for name, value in statistics.items()
        if value["minimum"] != float("inf") and value["maximum"] != float("-inf")
    }


class _FakeActivationObserver:
    def __init__(
        self,
        *,
        dtype: str,
        symmetric: bool,
    ) -> None:
        self.dtype = dtype
        self.symmetric = symmetric
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def observe(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        detached = tensor.detach().to(dtype=torch.float32, device="cpu")
        self.minimum = min(self.minimum, float(detached.min().item()))
        self.maximum = max(self.maximum, float(detached.max().item()))

    def fake_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.minimum == float("inf") or self.maximum == float("-inf"):
            return tensor
        reference = torch.tensor(
            [self.minimum, self.maximum],
            dtype=torch.float32,
            device=tensor.device,
        )
        qmin, qmax = _quant_range_for_dtype(self.dtype, symmetric=self.symmetric)
        if qmin >= 0:
            minimum = reference[0]
            maximum = reference[1]
            scale = _safe_scale((maximum - minimum) / float(qmax - qmin))
            zero_point = torch.round(
                torch.tensor(qmin, dtype=torch.float32, device=tensor.device)
                - minimum / scale
            )
            zero_point = torch.clamp(zero_point, qmin, qmax)
            quantized = torch.clamp(
                torch.round(tensor.to(dtype=torch.float32) / scale + zero_point),
                qmin,
                qmax,
            )
            return ((quantized - zero_point) * scale).to(dtype=tensor.dtype)

        max_abs = reference.abs().max()
        scale = _safe_scale(max_abs / float(qmax if qmax > 0 else 1))
        quantized = torch.clamp(
            torch.round(tensor.to(dtype=torch.float32) / scale),
            qmin,
            qmax,
        )
        return (quantized * scale).to(dtype=tensor.dtype)


def _attach_activation_fake_qdq(
    model: nn.Module,
    *,
    module_names: Sequence[str],
    activation_statistics: Mapping[str, Mapping[str, float]],
    dtype: str,
    symmetric: bool,
) -> None:
    if not module_names:
        return
    modules = dict(model.named_modules())
    for name in module_names:
        module = modules.get(name)
        if module is None:
            continue
        stats = activation_statistics.get(name)
        if not isinstance(stats, Mapping):
            continue
        observer = _FakeActivationObserver(dtype=dtype, symmetric=symmetric)
        minimum = stats.get("minimum")
        maximum = stats.get("maximum")
        if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
            observer.minimum = float(minimum)
            observer.maximum = float(maximum)

        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object, *, _observer: _FakeActivationObserver = observer) -> object:
            if isinstance(output, torch.Tensor):
                return _observer.fake_quantize(output)
            if isinstance(output, tuple):
                if output and isinstance(output[0], torch.Tensor):
                    items = list(output)
                    items[0] = _observer.fake_quantize(items[0])
                    return tuple(items)
                return output
            if isinstance(output, list):
                if output and isinstance(output[0], torch.Tensor):
                    items = list(output)
                    items[0] = _observer.fake_quantize(items[0])
                    return items
                return output
            return output

        module.register_forward_hook(hook)


def build_fake_qdq_surrogate(
    context: XQTContext,
    quant_config: QuantConfig | QuantStageSpec | None = None,
) -> Optional[FakeQDQSurrogateResult]:
    """Build a PyTorch fake-QDQ surrogate for analysis-only layer inspection."""

    model = context.model
    if not isinstance(model, nn.Module):
        return None

    resolved_quant = quant_config or context.quant_config
    if resolved_quant is None:
        raise ValueError("XQTContext.quant_config is required")
    enabled = resolved_quant.enabled if isinstance(resolved_quant, QuantConfig) else True
    backend = resolved_quant.backend
    if not enabled or backend != "onnxruntime_qdq":
        return None

    plan = build_quantization_plan(resolved_quant)
    qdq_components = [
        component
        for component in plan.components
        if component.backend == "onnxruntime_qdq" and not component.analysis_only
    ]
    if not qdq_components:
        return None

    surrogate = copy.deepcopy(model)
    quantized_modules: list[str] = []
    activation_statistics: dict[str, dict[str, float]] = {}
    activation_settings: dict[str, tuple[str, bool]] = {}
    sample_count = 0
    expected_input_count = infer_model_input_count(surrogate)

    for component in qdq_components:
        quantized_modules.extend(_apply_weight_fake_qdq(surrogate, component))
        calibration_inputs = _resolve_calibration_inputs(context)
        sample_limit = component.policy.get("sample_limit")
        batches = _iter_input_batches(
            calibration_inputs,
            expected_input_count=expected_input_count,
            sample_limit=int(sample_limit) if sample_limit is not None else None,
        )
        if not batches:
            continue
        sample_count += len(batches)
        policy = _build_policy(component)
        module_names = _target_names_for_component(surrogate, component, policy)
        activation_statistics.update(
            _activation_stat_collector(surrogate, module_names, batches)
        )
        activation_type = str(component.policy.get("activation_type", "QUInt8"))
        extra_options = component.policy.get("extra_options")
        activation_symmetric = False
        if isinstance(extra_options, Mapping):
            activation_symmetric = bool(extra_options.get("ActivationSymmetric", False))
        for module_name in module_names:
            activation_settings[module_name] = (activation_type, activation_symmetric)

    grouped_module_names: dict[tuple[str, bool], list[str]] = {}
    for module_name, setting in activation_settings.items():
        grouped_module_names.setdefault(setting, []).append(module_name)
    for (activation_type, activation_symmetric), module_names in grouped_module_names.items():
        _attach_activation_fake_qdq(
            surrogate,
            module_names=module_names,
            activation_statistics=activation_statistics,
            dtype=activation_type,
            symmetric=activation_symmetric,
        )

    return FakeQDQSurrogateResult(
        model=surrogate,
        quantized_modules=_ordered_unique(quantized_modules),
        activation_statistics=activation_statistics,
        sample_count=sample_count,
    )


__all__ = [
    "FakeQDQSurrogateResult",
    "build_fake_qdq_surrogate",
]
