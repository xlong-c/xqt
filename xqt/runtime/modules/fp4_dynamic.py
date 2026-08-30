"""Runtime execution view for FP4DynamicLinear storage artifacts."""

from __future__ import annotations

import copy
from typing import Any

from torch import nn


class FP4DynamicExecutionView(nn.Module):
    """Enable native FP4 dynamic dispatch only after explicit materialization."""

    _xqt_runtime_execution_view = True

    def __init__(self, storage: nn.Module) -> None:
        super().__init__()
        if not hasattr(storage, "packed_weight"):
            raise TypeError("expected FP4DynamicLinear storage")
        self.storage = storage
        setattr(self.storage, "_xqt_runtime_execution_enabled", True)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as error:
            modules = self.__dict__.get("_modules", {})
            storage = modules.get("storage")
            if storage is not None:
                try:
                    return getattr(storage, name)
                except AttributeError:
                    pass
            raise error

    @classmethod
    def from_storage(
        cls,
        storage: nn.Module,
        *,
        copy_storage: bool = True,
    ) -> "FP4DynamicExecutionView":
        materialized = copy.deepcopy(storage) if copy_storage else storage
        return cls(materialized)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.storage(*args, **kwargs)

    def execution_metadata(self) -> dict[str, Any]:
        metadata_fn = getattr(self.storage, "execution_metadata", None)
        metadata = dict(metadata_fn()) if callable(metadata_fn) else {}
        metadata["artifact_view"] = "runtime_execution"
        return metadata


def materialize_fp4_dynamic_execution_views(
    model: nn.Module,
    *,
    inplace: bool = True,
    copy_storage: bool = True,
) -> nn.Module:
    target = model if inplace else copy.deepcopy(model)
    for name, module in list(target.named_modules()):
        if isinstance(module, FP4DynamicExecutionView):
            continue
        if getattr(module, "_xqt_storage_only", False) and hasattr(module, "packed_weight"):
            view = FP4DynamicExecutionView.from_storage(
                module,
                copy_storage=copy_storage if inplace else False,
            )
            if name:
                parent_path, _, attr = name.rpartition(".")
                parent = target.get_submodule(parent_path) if parent_path else target
                if attr.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
                    parent[int(attr)] = view
                else:
                    setattr(parent, attr, view)
            else:
                target = view
                break
    return target


__all__ = ["FP4DynamicExecutionView", "materialize_fp4_dynamic_execution_views"]
