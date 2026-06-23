"""PyTorch baseline evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
from torch import nn

from xdl.metric import TopKAccuracy as _XDLTopKAccuracy
from xqt.data.input_utils import split_batch


MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass
class EvaluationReport:
    """Aggregated evaluation metrics."""

    samples: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Convert the report to a JSON-serializable dictionary."""

        return {
            "samples": self.samples,
            "metrics": dict(self.metrics),
        }


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    k: int = 1,
) -> float:
    """Compute top-k accuracy for classification logits."""

    return _XDLTopKAccuracy(k=k)(logits, targets)


def _split_batch(batch: Any) -> tuple[Any, Optional[torch.Tensor]]:
    split = split_batch(batch)
    return split.inputs, split.targets


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


def _call_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        output = model(**inputs)
    elif isinstance(inputs, tuple):
        output = model(*inputs)
    else:
        output = model(inputs)
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        "model output must be a Tensor, tuple/list with Tensor[0], "
        "or mapping with logits"
    )


def evaluate_pytorch_model(
    model: nn.Module,
    dataloader: Iterable[Any],
    *,
    metrics: Optional[Mapping[str, MetricFn]] = None,
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
) -> EvaluationReport:
    """Evaluate a PyTorch model on an iterable dataloader."""

    torch_device = torch.device(device)
    model.to(torch_device)
    was_training = model.training
    model.eval()

    metric_fns = dict(metrics or {"top1": topk_accuracy})
    metric_sums: dict[str, float] = {name: 0.0 for name in metric_fns}
    total_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs, targets = _split_batch(batch)
            inputs = _move_to_device(inputs, torch_device)
            targets = _move_to_device(targets, torch_device) if targets is not None else None
            outputs = _call_model(model, inputs)

            if targets is None:
                total_samples += int(outputs.shape[0]) if outputs.ndim > 0 else 1
                continue

            batch_size = int(targets.shape[0]) if targets.ndim > 0 else 1
            total_samples += batch_size
            for name, metric_fn in metric_fns.items():
                metric_sums[name] += float(metric_fn(outputs, targets)) * batch_size

    if was_training:
        model.train()

    if total_samples == 0:
        return EvaluationReport(
            samples=0,
            metrics={name: 0.0 for name in metric_sums},
        )

    return EvaluationReport(
        samples=total_samples,
        metrics={name: value / total_samples for name, value in metric_sums.items()},
    )


__all__ = [
    "EvaluationReport",
    "MetricFn",
    "evaluate_pytorch_model",
    "topk_accuracy",
]
