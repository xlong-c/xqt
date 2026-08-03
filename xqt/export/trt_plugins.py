"""TensorRT plugin library validation and loading helpers."""

from __future__ import annotations

import ctypes
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Sequence

from xqt.core.errors import XQTBackendError

from .trt_types import TensorRTPluginLibraryCheck, TensorRTPluginValidationResult


def _import_tensorrt() -> Any:
    try:
        return import_module("tensorrt")
    except ImportError as exc:
        raise XQTBackendError("tensorrt is required for TensorRT python_api builds") from exc


def _normalize_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
) -> list[Path]:
    normalized: list[Path] = []
    for raw in plugin_libraries or ():
        path = Path(raw)
        if path in normalized:
            continue
        normalized.append(path)
    return normalized


def _load_tensorrt_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
) -> list[str]:
    loaded: list[str] = []
    for path in _normalize_plugin_libraries(plugin_libraries):
        if not path.is_file():
            raise XQTBackendError(f"TensorRT plugin library not found: {path}")
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise XQTBackendError(
                f"Failed to load TensorRT plugin library '{path}': {exc}"
            ) from exc
        loaded.append(str(path))
    return loaded


def validate_tensorrt_plugin_libraries(
    plugin_libraries: Optional[Sequence[str | Path]],
    *,
    validate_loadability: bool = False,
) -> TensorRTPluginValidationResult:
    """Validate TensorRT plugin shared libraries without building an engine."""

    checks: list[TensorRTPluginLibraryCheck] = []
    for path in _normalize_plugin_libraries(plugin_libraries):
        exists = path.is_file()
        if not exists:
            checks.append(
                TensorRTPluginLibraryCheck(
                    path=str(path),
                    exists=False,
                    load_requested=validate_loadability,
                    error=f"TensorRT plugin library not found: {path}",
                )
            )
            continue
        if not validate_loadability:
            checks.append(
                TensorRTPluginLibraryCheck(
                    path=str(path),
                    exists=True,
                    load_requested=False,
                )
            )
            continue
        try:
            loaded = _load_tensorrt_plugin_libraries([path])
        except Exception as exc:
            checks.append(
                TensorRTPluginLibraryCheck(
                    path=str(path),
                    exists=True,
                    load_requested=True,
                    loadable=False,
                    error=str(exc),
                )
            )
            continue
        checks.append(
            TensorRTPluginLibraryCheck(
                path=str(path),
                exists=True,
                load_requested=True,
                loadable=True,
                loaded_plugin_libraries=loaded,
            )
        )

    if not checks:
        status = "not_requested"
    elif any(not check.exists for check in checks):
        status = "missing"
    elif validate_loadability and any(check.loadable is False for check in checks):
        status = "load_failed"
    elif validate_loadability:
        status = "ok"
    else:
        status = "present"
    return TensorRTPluginValidationResult(
        status=status,
        loadability_requested=validate_loadability,
        plugin_libraries=checks,
    )
