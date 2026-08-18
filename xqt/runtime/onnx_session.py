"""ONNX Runtime session factory for runtime-side inference runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from xqt.core.errors import XQTBackendError


def create_onnxruntime_session(
    onnx_path: str | Path,
    *,
    providers: Sequence[str] | None = None,
) -> Any:
    """Materialize an ONNX Runtime inference session for one exported model."""

    path = Path(onnx_path)
    if not path.is_file():
        raise XQTBackendError(f"ONNX model file not found: {path}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise XQTBackendError(
            "onnxruntime is required to materialize an ONNX Runtime handle"
        ) from exc
    return ort.InferenceSession(
        str(path),
        providers=list(providers or ["CPUExecutionProvider"]),
    )


__all__ = ["create_onnxruntime_session"]
