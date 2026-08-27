"""Explicit runtime execution views for ConvRot storage artifacts."""

from __future__ import annotations

import copy
from typing import Any, TypeAlias

from torch import nn


class ConvRotExecutionView(nn.Module):
    """Enable native ConvRot dispatch only after explicit materialization."""

    storage_kind: str
    _xqt_runtime_execution_view = True

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

    def __getattr__(self, name: str) -> Any:
        """Expose storage policy attributes without copying runtime state."""

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


ConvRotViewType: TypeAlias = type[ConvRotExecutionView]
_VIEW_TYPES: dict[str, ConvRotViewType] = {
    "int8": ConvRotInt8ExecutionView,
    "w4a4": ConvRotW4A4ExecutionView,
}


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def materialize_convrot_execution_views(
    model: nn.Module,
    *,
    inplace: bool = True,
    copy_storage: bool = True,
) -> nn.Module:
    """Replace marked ConvRot storage modules with runtime execution views.

    The marker is the only cross-layer handoff contract. Runtime therefore
    does not import quantizer implementations, while quantized models remain
    serializable until this function is called.
    """

    target = model if inplace else copy.deepcopy(model)
    modules = list(target.named_modules())
    view_roots = {
        name for name, module in modules if isinstance(module, ConvRotExecutionView)
    }
    for name, module in modules:
        if any(
            root == "" or root == name or (root and name.startswith(f"{root}."))
            for root in view_roots
        ):
            continue
        storage_kind = getattr(module, "_xqt_convrot_storage_kind", None)
        view_type = _VIEW_TYPES.get(str(storage_kind))
        if view_type is None:
            continue
        view = view_type.from_storage(
            module,
            copy_storage=copy_storage if inplace else False,
        )
        if name:
            _replace_submodule(target, name, view)
        else:
            target = view
            break
    return target


__all__ = [
    "ConvRotExecutionView",
    "ConvRotInt8ExecutionView",
    "ConvRotW4A4ExecutionView",
    "materialize_convrot_execution_views",
]
