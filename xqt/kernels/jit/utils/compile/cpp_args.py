"""Compiler argument construction kept independent from the loader."""

from __future__ import annotations

from pathlib import Path

from ..arch import make_jit_cuda_arch
from .spec import CompileSpec


def include_flags(include_dirs: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(flag for path in include_dirs for flag in ("-I", str(path)))


def cxx_args(spec: CompileSpec) -> tuple[str, ...]:
    return (*spec.cxx_flags, *include_flags(spec.include_dirs))


def cuda_args(spec: CompileSpec) -> tuple[str, ...]:
    flags = list(spec.cuda_flags)
    if spec.target_arch is not None:
        arch = make_jit_cuda_arch(spec.target_arch)
        flags.extend((f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}",))
    flags.extend(include_flags(spec.include_dirs))
    return tuple(flags)


def extension_load_kwargs(spec: CompileSpec) -> dict[str, object]:
    """Translate a spec to ``torch.utils.cpp_extension.load`` kwargs."""

    return {
        "name": spec.name,
        "sources": [str(path) for path in spec.sources],
        "extra_cflags": list(cxx_args(spec)),
        "extra_cuda_cflags": list(cuda_args(spec)),
        "extra_ldflags": list(spec.link_flags),
        "with_cuda": spec.with_cuda,
    }


__all__ = ["cuda_args", "cxx_args", "extension_load_kwargs", "include_flags"]
