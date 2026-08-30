"""Stable cache keys and metadata paths for JIT extension builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .paths import cache_dir
from .spec import CompileSpec


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def compile_cache_key(spec: CompileSpec, *, extra: Mapping[str, object] | None = None) -> str:
    payload: dict[str, object] = {
        "spec": spec.to_dict(),
        "source_digests": {
            str(path): _source_digest(path) for path in spec.sources
        },
    }
    if spec.env:
        env_payload = json.dumps(
            _jsonable(spec.env), sort_keys=True, separators=(",", ":")
        ).encode()
        payload["env_digest"] = hashlib.sha256(env_payload).hexdigest()
    if extra:
        payload["extra"] = _jsonable(extra)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_directory(spec: CompileSpec, *, extra: Mapping[str, object] | None = None) -> Path:
    return cache_dir(spec.cache_dir) / f"{spec.name}_{compile_cache_key(spec, extra=extra)}"


def metadata_path(spec: CompileSpec, *, extra: Mapping[str, object] | None = None) -> Path:
    return build_directory(spec, extra=extra) / "compile.json"


__all__ = ["build_directory", "compile_cache_key", "metadata_path"]
