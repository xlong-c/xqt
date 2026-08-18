"""Lightweight quantizer contracts for XQT model-side quantization algorithms.

New executable quantizer PR checklist (U12 / T4):

1. Implement algorithm under ``xqt/quant/quantizers/`` (or backends for adapters).
2. Register via ``register_quant_route(QuantRouteRegistration(...))`` in
   ``quantizers/__init__.py`` (or the package that owns the route).
3. If ``maturity="executable"``, **must** set non-empty ``primary_kernel`` and
   ``reference_kernel`` (scheme x kernel pair). Missing either raises at
   registration time.
4. Prefer attaching ``RuntimeQuantContract`` + ``layout_kernel`` on success
   (see ``layout_apply_report`` / ``build_runtime_quant_contract``).
5. Do not mark cutile/cute_dsl/cutlass as the executable primary for new routes
   unless the engine registration maturity is actually executable and dispatchable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.types import XQTContext

from ..component import replace_component_model
from ..policy import QuantizationPolicy
from ..types import QuantizationComponentPlan, QuantizationReport


@dataclass(frozen=True)
class QuantizerOptions:
    """Resolved options passed from a quantization component to a quantizer."""

    backend: str
    method: str | None = None
    strategy: str | None = None
    policy: Mapping[str, Any] = field(default_factory=dict)
    inplace: bool = True


QuantizerResult = QuantizedModel


class Quantizer(Protocol):
    """Protocol implemented by concrete quantizer algorithms."""

    def quantize(self, model: nn.Module, options: QuantizerOptions) -> QuantizedModel:
        """Quantize a model or component and return a normalized result."""


class QuantizerTemplate(ABC):
    """Template method base for model-side quantizer implementations."""

    def quantize(self, model: nn.Module, options: QuantizerOptions) -> QuantizedModel:
        """Resolve shared policy semantics before invoking the algorithm hook."""

        policy = policy_from_mapping(options.policy)
        return self.quantize_model(model, options=options, policy=policy)

    @abstractmethod
    def quantize_model(
        self,
        model: nn.Module,
        *,
        options: QuantizerOptions,
        policy: QuantizationPolicy,
    ) -> QuantizedModel:
        """Execute algorithm-specific packing and module construction."""


_POLICY_SEQUENCE_FIELDS = frozenset(
    {
        "include_module_types",
        "exclude_module_types",
        "include_name_patterns",
        "exclude_name_patterns",
        "include_module_names",
        "exclude_module_names",
    }
)


def policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    """Build the shared module-selection policy from one mapping."""

    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key in {"dtype", "scheme"}:
            kwargs[key] = str(value)
        elif key in _POLICY_SEQUENCE_FIELDS:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


def replace_submodule(
    root: nn.Module,
    path: str,
    replacement: nn.Module,
) -> None:
    """Replace one nested module through the canonical component helper."""

    replace_component_model(root, path, replacement)


def iter_calibration_batches(
    calibration_inputs: Iterable[Any] | None,
    *,
    sample_limit: int | None,
) -> Iterable[Any]:
    """Yield at most ``sample_limit`` representative calibration batches."""

    if calibration_inputs is None:
        return
    for index, batch in enumerate(calibration_inputs):
        if sample_limit is not None and index >= sample_limit:
            break
        yield batch


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move one calibration batch to a device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    if isinstance(batch, Mapping):
        return {
            key: move_batch_to_device(value, device)
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def call_model(model: nn.Module, inputs: Any) -> Any:
    """Call a model with mapping, sequence, or tensor inputs."""

    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


_RouteResult = tuple[Optional[nn.Module], QuantizationReport, dict[str, Any]]
_ComponentExecutor = Callable[..., tuple[Optional[nn.Module], QuantizationReport]]


def component_route_handler(
    executor: _ComponentExecutor,
    *,
    executor_kwargs: Mapping[str, Any] | None = None,
    require_model: bool = False,
    missing_model_message: str | None = None,
) -> Callable[..., _RouteResult]:
    """Create the standard registry adapter for a component executor."""

    static_kwargs = dict(executor_kwargs or {})

    @wraps(executor)
    def handler(
        context: XQTContext,
        model: Optional[nn.Module],
        component: QuantizationComponentPlan,
        *,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> _RouteResult:
        del runtime
        if require_model and model is None:
            raise ValueError(
                missing_model_message
                or f"{executor.__name__} requires a PyTorch model"
            )
        current_model, report = executor(
            context,
            model,
            component,
            **static_kwargs,
        )
        return current_model, report, {}

    return handler


__all__ = [
    "Quantizer",
    "QuantizerOptions",
    "QuantizerResult",
    "QuantizerTemplate",
    "call_model",
    "component_route_handler",
    "iter_calibration_batches",
    "move_batch_to_device",
    "policy_from_mapping",
    "replace_submodule",
]
