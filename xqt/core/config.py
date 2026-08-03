"""Shared config input types for XQT workflow loaders."""

from pathlib import Path
from typing import Any, Mapping, Union

ConfigInput = Union[str, Path, Mapping[str, Any]]

__all__ = ["ConfigInput"]
