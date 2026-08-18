"""Shared config input types for XQT workflow loaders."""

from pathlib import Path
from typing import Any, Mapping, Union

from omegaconf import OmegaConf

ConfigInput = Union[str, Path, Mapping[str, Any]]


def register_default_resolvers() -> None:
    """Register XQT default OmegaConf resolvers (idempotent)."""

    if not OmegaConf.has_resolver("xdl.join_path"):
        OmegaConf.register_new_resolver(
            "xdl.join_path",
            lambda *parts: str(Path(*[str(part) for part in parts])),
        )

    if not OmegaConf.has_resolver("xdl.abspath"):
        OmegaConf.register_new_resolver(
            "xdl.abspath",
            lambda *parts: str(
                Path(*[str(part) for part in parts]).expanduser().resolve()
            ),
        )


__all__ = ["ConfigInput", "register_default_resolvers"]
