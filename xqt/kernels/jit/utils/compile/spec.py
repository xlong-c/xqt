"""Typed, backend-neutral inputs for a lazy C++/CUDA extension build."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _paths(values: tuple[str | Path, ...]) -> tuple[Path, ...]:
    return tuple(Path(value).expanduser().resolve() for value in values)


@dataclass(frozen=True, slots=True)
class CompileSpec:
    """Complete build inputs consumed by :func:`load_extension`."""

    name: str
    sources: tuple[str | Path, ...]
    include_dirs: tuple[str | Path, ...] = ()
    cxx_flags: tuple[str, ...] = ("-O3", "-std=c++20")
    cuda_flags: tuple[str, ...] = ("-O3", "-std=c++20")
    link_flags: tuple[str, ...] = ()
    target_arch: str | int | tuple[int, int] | None = None
    cache_dir: str | Path | None = None
    with_cuda: bool = True
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("extension name must be non-empty and contain no whitespace")
        if not self.sources:
            raise ValueError("at least one source is required")
        source_paths = _paths(self.sources)
        if any(not source.is_file() for source in source_paths):
            missing = next(source for source in source_paths if not source.is_file())
            raise FileNotFoundError(missing)
        object.__setattr__(self, "sources", source_paths)
        object.__setattr__(self, "include_dirs", _paths(self.include_dirs))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser().resolve())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata without environment secrets."""

        return {
            "name": self.name,
            "sources": [str(path) for path in self.sources],
            "include_dirs": [str(path) for path in self.include_dirs],
            "cxx_flags": list(self.cxx_flags),
            "cuda_flags": list(self.cuda_flags),
            "link_flags": list(self.link_flags),
            "target_arch": self.target_arch,
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
            "with_cuda": self.with_cuda,
        }


ExtensionSpec = CompileSpec


def serialize_compile_settings(**settings: Any) -> dict[str, Any]:
    """Return stable, JSON-safe backend compile settings metadata."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, set):
            return sorted(convert(item) for item in value)
        return value

    serialized: dict[str, Any] = {}
    for name, value in settings.items():
        serialized[name] = convert(value)
    return serialized


__all__ = ["CompileSpec", "ExtensionSpec", "serialize_compile_settings"]
