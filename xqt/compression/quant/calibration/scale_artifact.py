"""Static activation scale artifacts and minmax calibration (C5)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn

from ..policy import QuantizationPolicy, list_quantizable_modules
from ..types import QuantScheme


@dataclass(frozen=True)
class ActivationScaleArtifact:
    """Calibrated activation scale for one module (C5 static activation path).

    First-batch support is symmetric per-tensor minmax for signed INT8
    (``qmax=127``). The scale is a 0-d float32 tensor so it can be registered
    as a module buffer without further conversion.
    """

    module_path: str
    scale: torch.Tensor
    observer: str
    num_samples: int
    granularity: str = "per_tensor"
    qmax: int = 127
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not str(self.module_path).strip():
            raise ValueError("ActivationScaleArtifact.module_path must be non-empty")
        if int(self.num_samples) <= 0:
            raise ValueError("ActivationScaleArtifact.num_samples must be positive")
        if int(self.qmax) <= 0:
            raise ValueError("ActivationScaleArtifact.qmax must be positive")
        scale = torch.as_tensor(self.scale, dtype=torch.float32).reshape(())
        if float(scale.item()) <= 0.0:
            raise ValueError("ActivationScaleArtifact.scale must be positive")
        object.__setattr__(self, "scale", scale.detach().cpu().clone())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact for quant-pair lineage and reports."""

        return {
            "module_path": self.module_path,
            "scale": float(self.scale.item()),
            "observer": self.observer,
            "num_samples": int(self.num_samples),
            "granularity": self.granularity,
            "qmax": int(self.qmax),
            "eps": float(self.eps),
        }


@contextmanager
def preserve_module_inference_state(model: nn.Module):
    """Temporarily evaluate ``model`` without leaking mode or buffer changes."""

    modules = tuple(model.modules())
    training_modes = {module: bool(module.training) for module in modules}
    buffers = {
        (module, name): buffer.detach().clone()
        for module in modules
        for name, buffer in module._buffers.items()
        if buffer is not None
    }
    model.eval()
    try:
        yield
    finally:
        for (module, name), value in buffers.items():
            current = module._buffers.get(name)
            if current is None:
                module._buffers[name] = value.clone()
            elif current.shape == value.shape and current.dtype == value.dtype:
                current.copy_(value.to(device=current.device))
            else:
                module._buffers[name] = value.to(
                    device=current.device,
                    dtype=current.dtype,
                )
        # Module.train() recursively changes children; direct assignment keeps
        # intentionally mixed training modes intact.
        for module, training in training_modes.items():
            module.training = training


def _batch_has_empty_tensor(batch: Any) -> bool:
    if isinstance(batch, torch.Tensor):
        return batch.numel() == 0
    if isinstance(batch, Mapping):
        return any(_batch_has_empty_tensor(item) for item in batch.values())
    if isinstance(batch, (tuple, list)):
        return any(_batch_has_empty_tensor(item) for item in batch)
    return False


def run_calibration_batches(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> int:
    """Forward every calibration batch; return how many batches were run."""

    kwargs = dict(forward_kwargs or {})
    batch_count = 0
    with preserve_module_inference_state(model), torch.no_grad():
        for batch in batches:
            if _batch_has_empty_tensor(batch):
                raise ValueError("calibration batch contains an empty tensor")
            if isinstance(batch, tuple):
                model(*batch, **kwargs)
            elif isinstance(batch, dict):
                model(**batch, **kwargs)
            else:
                model(batch, **kwargs)
            batch_count += 1
    if batch_count == 0:
        raise ValueError("calibration_inputs iterable must yield at least one batch; got 0")
    return batch_count


def _symmetric_scale_from_max_abs(
    max_abs: float,
    *,
    qmax: int,
    eps: float,
) -> torch.Tensor:
    value = max(float(max_abs), float(eps)) / float(qmax)
    return torch.tensor(value, dtype=torch.float32)


def calibrate_activation_scales(
    model: nn.Module,
    calibration_inputs: Iterable[Any],
    scheme: QuantScheme,
    *,
    module_names: Optional[Sequence[str]] = None,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
    policy: Optional[QuantizationPolicy] = None,
    observer: str = "minmax",
    qmax: int = 127,
    eps: float = 1e-6,
) -> dict[str, ActivationScaleArtifact]:
    """Calibrate static per-tensor activation scales from module *inputs*.

    Captures module inputs via pre-hooks so scales match W8A8 INT8 MMA runtime
    encoding. Only ``activation_mode="static"`` schemes are accepted. First
    batch supports ``observer="minmax"`` only (``max_abs / qmax``).
    """

    if scheme.activation_mode != "static":
        raise ValueError(
            "calibrate_activation_scales requires scheme.activation_mode='static'; "
            f"got {scheme.activation_mode!r}"
        )
    if observer != "minmax":
        raise ValueError(
            "calibrate_activation_scales currently supports observer='minmax' only; "
            f"got {observer!r}"
        )
    if int(qmax) <= 0:
        raise ValueError("qmax must be a positive int")

    names = list(module_names) if module_names is not None else [
        candidate.name
        for candidate in list_quantizable_modules(model, policy)
        if candidate.quantize
    ]
    if not names:
        return {}

    modules = dict(model.named_modules())
    missing = [name for name in names if name not in modules]
    if missing:
        raise KeyError(f"Modules not found: {missing}")

    max_abs: dict[str, float] = {name: 0.0 for name in names}
    sample_counts: dict[str, int] = {name: 0 for name in names}
    handles: list[Any] = []
    try:
        for name in names:
            module = modules[name]

            def make_pre_hook(module_name: str):
                def hook(
                    _module: nn.Module,
                    inputs: tuple[Any, ...],
                ) -> None:
                    if not inputs:
                        return
                    value: Any = inputs[0]
                    if isinstance(value, (tuple, list)):
                        value = value[0] if value else value
                    if not isinstance(value, torch.Tensor):
                        return
                    flat = value.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
                    if flat.numel() == 0:
                        return
                    current = float(flat.abs().amax().item())
                    if current > max_abs[module_name]:
                        max_abs[module_name] = current
                    sample_counts[module_name] += int(flat.numel())

                return hook

            handles.append(module.register_forward_pre_hook(make_pre_hook(name)))

        batch_count = run_calibration_batches(
            model,
            calibration_inputs,
            forward_kwargs=forward_kwargs,
        )
    finally:
        while handles:
            handles.pop().remove()

    if batch_count <= 0:
        raise ValueError(
            "calibrate_activation_scales requires at least one calibration batch"
        )

    artifacts: dict[str, ActivationScaleArtifact] = {}
    empty: list[str] = []
    for name in names:
        if sample_counts[name] <= 0:
            empty.append(name)
            continue
        scale = _symmetric_scale_from_max_abs(
            max_abs[name],
            qmax=int(qmax),
            eps=float(eps),
        )
        artifacts[name] = ActivationScaleArtifact(
            module_path=name,
            scale=scale,
            observer=observer,
            num_samples=sample_counts[name],
            granularity="per_tensor",
            qmax=int(qmax),
            eps=float(eps),
        )
    if empty:
        raise ValueError(
            "calibrate_activation_scales collected no activation samples for: "
            + ", ".join(empty)
        )
    return artifacts


def activation_scales_to_mapping(
    artifacts: Mapping[str, ActivationScaleArtifact],
) -> dict[str, torch.Tensor]:
    """Flatten scale artifacts into the mapping consumed by int8 MMA quantizer."""

    return {
        path: artifact.scale.detach().clone()
        for path, artifact in artifacts.items()
    }


__all__ = [
    "ActivationScaleArtifact",
    "activation_scales_to_mapping",
    "calibrate_activation_scales",
    "preserve_module_inference_state",
    "run_calibration_batches",
]
