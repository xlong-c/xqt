"""Lazy PyTorch extension loader with isolated cache and build environment."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from ..arch import architecture_env
from .cache import build_directory, metadata_path
from .cpp_args import extension_load_kwargs
from .spec import CompileSpec


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_extension(
    spec: CompileSpec,
    *,
    verbose: bool = False,
    extra: dict[str, object] | None = None,
    max_jobs: int | None = 1,
    backend: str | None = None,
) -> Any:
    """Build/load one extension only when explicitly called."""

    selected_backend = backend or os.environ.get("XQT_JIT_BACKEND") or spec.backend or "tvm_ffi"
    if selected_backend == "tvm_ffi":
        from .tvm_ffi_loader import load_tvm_ffi_extension

        return load_tvm_ffi_extension(
            spec, verbose=verbose, extra=extra, max_jobs=max_jobs
        )

    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build an XQT JIT extension") from exc
    directory = build_directory(spec, extra=extra)
    directory.mkdir(parents=True, exist_ok=True)
    kwargs = extension_load_kwargs(spec)
    kwargs["build_directory"] = str(directory)
    kwargs["verbose"] = verbose
    env = dict(spec.env)
    env.update(architecture_env(spec.target_arch))
    if max_jobs is not None:
        env.setdefault("MAX_JOBS", str(max_jobs))
    with _temporary_environment(env):
        extension = load(**kwargs)
    metadata_path(spec, extra=extra).write_text(
        __import__("json").dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return extension


load_jit = load_extension


__all__ = ["load_extension", "load_jit"]
