"""Lazy TVM FFI extension builder and loader with isolated cache and build environment."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..arch import normalize_cuda_arch
from .cache import build_directory, metadata_path
from .spec import CompileSpec


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
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


def _cuda_arch_target(target_arch: str | int | tuple[int, int] | None) -> str | None:
    if target_arch is None:
        return None
    normalized = normalize_cuda_arch(target_arch)  # e.g., "sm_89"
    if normalized.startswith("sm_"):
        digits = normalized[3:]
        if len(digits) == 2:
            return f"{digits[0]}.{digits[1]}"
        if len(digits) == 3:
            return f"{digits[:2]}.{digits[2:]}"
    return None


def load_tvm_ffi_extension(
    spec: CompileSpec,
    *,
    verbose: bool = False,
    extra: dict[str, object] | None = None,
    max_jobs: int | None = None,
) -> Any:
    """Build and load one C++/CUDA extension using TVM FFI."""

    try:
        import tvm_ffi
        import tvm_ffi.cpp
    except ImportError as exc:
        raise RuntimeError("tvm_ffi is required to build a TVM FFI extension") from exc

    directory = build_directory(spec, extra=extra)
    directory.mkdir(parents=True, exist_ok=True)

    env = dict(spec.env)
    arch_target = _cuda_arch_target(spec.target_arch)
    if arch_target is not None:
        env["TVM_FFI_CUDA_ARCH_LIST"] = arch_target
    if max_jobs is not None:
        env.setdefault("MAX_JOBS", str(max_jobs))

    # Strip any -gencode flags from cuda_flags since TVM_FFI_CUDA_ARCH_LIST handles them
    cuda_flags = [f for f in spec.cuda_flags if not f.startswith("-gencode")]

    # Include dirs for TVM FFI
    include_paths = [str(p.resolve()) for p in spec.include_dirs]
    if spec.with_cuda:
        cuda_home = Path(tvm_ffi.cpp.extension._find_cuda_home())
        cuda_inc = str((cuda_home / "include").resolve())
        if cuda_inc not in include_paths:
            include_paths.append(cuda_inc)

    cpp_files = []
    cuda_files = []
    for s in spec.sources:
        s_str = str(s.resolve())
        if s_str.endswith(".cu"):
            cuda_files.append(s_str)
        else:
            cpp_files.append(s_str)

    with _temporary_environment(env):
        lib_path = tvm_ffi.cpp.build(
            name=spec.name,
            cpp_files=cpp_files or None,
            cuda_files=cuda_files or None,
            extra_cflags=list(spec.cxx_flags),
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=list(spec.link_flags),
            extra_include_paths=include_paths,
            build_directory=str(directory),
            backend="cuda" if spec.with_cuda else None,
        )
        module = tvm_ffi.load_module(lib_path)

    metadata_path(spec, extra=extra).write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return module
