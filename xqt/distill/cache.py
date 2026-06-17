"""Teacher output cache helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from .hooks import ModuleOutputCapture


@dataclass
class TeacherOutput:
    """Teacher logits, optional features, and metadata."""

    logits: torch.Tensor
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherCacheRecord:
    """A cached teacher output location."""

    key: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class TeacherOutputCache:
    """Disk cache for teacher logits and feature tensors."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.pt"

    def exists(self, key: str) -> bool:
        return self.path_for_key(key).is_file()

    def write(
        self,
        key: str,
        output: TeacherOutput,
    ) -> TeacherCacheRecord:
        path = self.path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": output.logits.detach().cpu(),
                "features": {
                    name: value.detach().cpu()
                    for name, value in output.features.items()
                },
                "metadata": dict(output.metadata),
            },
            path,
        )
        return TeacherCacheRecord(key=key, path=path, metadata=dict(output.metadata))

    def read(self, key: str) -> TeacherOutput:
        payload = torch.load(self.path_for_key(key), map_location="cpu")
        features = payload.get("features", {})
        if not isinstance(features, Mapping):
            raise ValueError("Cached teacher features must be a mapping")
        return TeacherOutput(
            logits=payload["logits"],
            features=dict(features),
            metadata=dict(payload.get("metadata", {})),
        )


def _batch_inputs(batch: Any) -> Any:
    if isinstance(batch, Mapping):
        if "inputs" in batch:
            return batch["inputs"]
        if "input" in batch:
            return batch["input"]
        if "x" in batch:
            return batch["x"]
        return {
            key: value
            for key, value in batch.items()
            if key not in {"targets", "target", "y"}
        }
    if isinstance(batch, (tuple, list)):
        return batch[0] if len(batch) == 2 else tuple(batch[:-1])
    return batch


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
    raise TypeError("teacher output must be a Tensor, tuple/list Tensor[0], or logits mapping")


def cache_teacher_outputs(
    teacher: nn.Module,
    batches: Sequence[Any],
    cache: TeacherOutputCache,
    *,
    feature_module_names: Sequence[str] = (),
    device: str | torch.device = "cpu",
    key_prefix: str = "batch",
    max_batches: Optional[int] = None,
) -> list[TeacherCacheRecord]:
    """Run a teacher over batches and cache logits plus optional feature tensors."""

    torch_device = torch.device(device)
    teacher.to(torch_device)
    was_training = teacher.training
    teacher.eval()

    records: list[TeacherCacheRecord] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            key = f"{key_prefix}_{batch_index}"
            inputs = _move_to_device(_batch_inputs(batch), torch_device)

            if feature_module_names:
                with ModuleOutputCapture(
                    teacher,
                    feature_module_names,
                    to_cpu=True,
                ) as capture:
                    logits = _call_model(teacher, inputs)
                features = {
                    name: value
                    for name, value in capture.outputs.items()
                    if isinstance(value, torch.Tensor)
                }
            else:
                logits = _call_model(teacher, inputs)
                features = {}

            records.append(
                cache.write(
                    key,
                    TeacherOutput(
                        logits=logits,
                        features=features,
                        metadata={"batch_index": batch_index},
                    ),
                )
            )

    if was_training:
        teacher.train()
    return records


__all__ = [
    "TeacherCacheRecord",
    "TeacherOutput",
    "TeacherOutputCache",
    "cache_teacher_outputs",
]
