"""Small PyTorch training loop for classification distillation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import nn

from xqt.data.input_utils import split_batch

from .losses import distillation_loss


@dataclass
class DistillationTrainReport:
    """Metrics collected from a distillation training run."""

    steps: int
    samples: int
    mean_loss: float
    last_loss: float
    loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "samples": self.samples,
            "mean_loss": self.mean_loss,
            "last_loss": self.last_loss,
            "loss_history": list(self.loss_history),
        }


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
    raise TypeError("model output must be a Tensor, tuple/list Tensor[0], or logits mapping")


def train_logit_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str | torch.device = "cpu",
    max_steps: Optional[int] = None,
) -> DistillationTrainReport:
    """Train a student with teacher logit distillation on a small iterable."""

    torch_device = torch.device(device)
    student.to(torch_device)
    teacher.to(torch_device)
    teacher_was_training = teacher.training
    student_was_training = student.training
    teacher.eval()
    student.train()

    history: list[float] = []
    sample_count = 0
    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        inputs, targets = _split_batch(batch)
        inputs = _move_to_device(inputs, torch_device)
        targets = _move_to_device(targets, torch_device) if targets is not None else None

        with torch.no_grad():
            teacher_logits = _call_model(teacher, inputs)
        student_logits = _call_model(student, inputs)
        loss_breakdown = distillation_loss(
            student_logits,
            teacher_logits,
            targets=targets,
            temperature=temperature,
            alpha=alpha,
        )
        optimizer.zero_grad()
        loss_breakdown.total.backward()
        optimizer.step()

        history.append(float(loss_breakdown.total.detach().cpu().item()))
        if isinstance(student_logits, torch.Tensor) and student_logits.ndim > 0:
            sample_count += int(student_logits.shape[0])

    if teacher_was_training:
        teacher.train()
    if not student_was_training:
        student.eval()

    steps = len(history)
    mean_loss = sum(history) / steps if steps else 0.0
    last_loss = history[-1] if history else 0.0
    return DistillationTrainReport(
        steps=steps,
        samples=sample_count,
        mean_loss=mean_loss,
        last_loss=last_loss,
        loss_history=history,
    )


__all__ = [
    "DistillationTrainReport",
    "train_logit_distillation",
]
