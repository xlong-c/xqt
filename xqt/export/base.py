"""Shared contracts for XQT export adapters and results."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ExportResultBase:
    """Common result surface implemented by artifact-producing exporters."""

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        raise NotImplementedError

    @property
    def succeeded(self) -> bool:
        dry_run = bool(getattr(self, "dry_run", False))
        return dry_run or bool(self.artifact_paths) and all(
            path.is_file() or path.is_dir() for path in self.artifact_paths
        )


@runtime_checkable
class ExportAdapter(Protocol):
    """Uniform callable adapter contract used by export orchestration."""

    name: str

    def export(self, *args: Any, **kwargs: Any) -> ExportResultBase:
        ...


class FunctionExportAdapter:
    """Adapt an existing export function to the shared adapter Protocol."""

    def __init__(
        self,
        name: str,
        exporter: Callable[..., ExportResultBase],
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = str(name)
        self._exporter = exporter
        self._defaults = dict(defaults or {})

    def export(self, *args: Any, **kwargs: Any) -> ExportResultBase:
        options = dict(self._defaults)
        options.update(kwargs)
        return self._exporter(*args, **options)


__all__ = ["ExportAdapter", "ExportResultBase", "FunctionExportAdapter"]
