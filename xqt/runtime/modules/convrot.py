"""Explicit runtime execution views for ConvRot storage artifacts."""

from __future__ import annotations

import copy
from typing import Any

from torch import nn


class ConvRotExecutionView(nn.Module):
    """Enable native ConvRot dispatch only after explicit materialization."""

    storage_kind: str

    def __init__(self, storage: nn.Module, *, storage_kind: str) -> None:
        super().__init__()
        actual_kind = getattr(storage, "_xqt_convrot_storage_kind", None)
        if actual_kind != storage_kind:
            raise TypeError(
                f"expected ConvRot storage kind {storage_kind!r}, got {actual_kind!r}"
            )
        self.storage = storage
        self.storage_kind = storage_kind
        setattr(self.storage, "_xqt_runtime_execution_enabled", True)

    @classmethod
    def from_storage(
        cls,
        storage: nn.Module,
        *,
        copy_storage: bool = True,
    ) -> "ConvRotExecutionView":
        materialized = copy.deepcopy(storage) if copy_storage else storage
        return cls(materialized, storage_kind=cls.storage_kind)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.storage(*args, **kwargs)

    def execution_metadata(self) -> dict[str, Any]:
        metadata_fn = getattr(self.storage, "execution_metadata", None)
        metadata = dict(metadata_fn()) if callable(metadata_fn) else {}
        metadata["artifact_view"] = "runtime_execution"
        metadata["storage_kind"] = self.storage_kind
        return metadata


class ConvRotInt8ExecutionView(ConvRotExecutionView):
    storage_kind = "int8"


class ConvRotW4A4ExecutionView(ConvRotExecutionView):
    storage_kind = "w4a4"


__all__ = [
    "ConvRotExecutionView",
    "ConvRotInt8ExecutionView",
    "ConvRotW4A4ExecutionView",
]
