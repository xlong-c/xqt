"""Teacher output cache helpers."""

from __future__ import annotations

import hashlib
import json
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
    sample_identity: Optional[str] = None
    dataset_signature: Optional[str] = None
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

    def key_for_identity(self, sample_identity: str, *, prefix: str = "sample") -> str:
        """Build a stable cache key from a sample identity."""

        digest = hashlib.sha256(sample_identity.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def write(
        self,
        key: str,
        output: TeacherOutput,
        *,
        sample_identity: Optional[str] = None,
        dataset_signature: Optional[str] = None,
    ) -> TeacherCacheRecord:
        path = self.path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "key": key,
                "sample_identity": sample_identity,
                "dataset_signature": dataset_signature,
                "logits": output.logits.detach().cpu(),
                "features": {
                    name: value.detach().cpu()
                    for name, value in output.features.items()
                },
                "metadata": dict(output.metadata),
            },
            path,
        )
        return TeacherCacheRecord(
            key=key,
            path=path,
            sample_identity=sample_identity,
            dataset_signature=dataset_signature,
            metadata=dict(output.metadata),
        )

    def read(
        self,
        key: str,
        *,
        expected_identity: Optional[str] = None,
        expected_dataset_signature: Optional[str] = None,
    ) -> TeacherOutput:
        payload = torch.load(self.path_for_key(key), map_location="cpu")
        features = payload.get("features", {})
        if not isinstance(features, Mapping):
            raise ValueError("Cached teacher features must be a mapping")
        sample_identity = payload.get("sample_identity")
        dataset_signature = payload.get("dataset_signature")
        if expected_identity is not None and sample_identity != expected_identity:
            raise ValueError("Cached teacher sample identity does not match expected identity")
        if (
            expected_dataset_signature is not None
            and dataset_signature != expected_dataset_signature
        ):
            raise ValueError("Cached teacher dataset signature does not match expected signature")
        metadata = dict(payload.get("metadata", {}))
        if sample_identity is not None:
            metadata["sample_identity"] = sample_identity
        if dataset_signature is not None:
            metadata["dataset_signature"] = dataset_signature
        return TeacherOutput(
            logits=payload["logits"],
            features=dict(features),
            metadata=metadata,
        )


def batch_identity(batch: Any) -> str:
    """Build a deterministic identity string for a teacher-cache batch."""

    def normalize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            return {
                "type": "tensor",
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
            }
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps(normalize(batch), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_signature(sample_identities: Sequence[str]) -> str:
    """Build a deterministic signature for a calibration or distillation dataset slice."""

    payload = json.dumps(list(sample_identities), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    sample_identities: Optional[Sequence[str]] = None,
    data_version: Optional[str] = None,
) -> list[TeacherCacheRecord]:
    """Run a teacher over batches and cache logits plus optional feature tensors."""

    torch_device = torch.device(device)
    teacher.to(torch_device)
    was_training = teacher.training
    teacher.eval()

    batch_limit = min(len(batches), max_batches) if max_batches is not None else len(batches)
    if sample_identities is not None and len(sample_identities) < batch_limit:
        raise ValueError("sample_identities must cover every cached batch")
    resolved_sample_identities = [
        sample_identities[index] if sample_identities is not None else batch_identity(batches[index])
        for index in range(batch_limit)
    ]
    resolved_dataset_signature = data_version or dataset_signature(resolved_sample_identities)

    records: list[TeacherCacheRecord] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            sample_identity = resolved_sample_identities[batch_index]
            key = cache.key_for_identity(sample_identity, prefix=key_prefix)
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
                        metadata={
                            "batch_index": batch_index,
                            "sample_identity": sample_identity,
                            "dataset_signature": resolved_dataset_signature,
                        },
                    ),
                    sample_identity=sample_identity,
                    dataset_signature=resolved_dataset_signature,
                )
            )

    if was_training:
        teacher.train()
    return records


__all__ = [
    "TeacherCacheRecord",
    "TeacherOutput",
    "TeacherOutputCache",
    "batch_identity",
    "cache_teacher_outputs",
    "dataset_signature",
]
