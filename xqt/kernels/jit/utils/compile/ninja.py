"""Minimal Ninja availability and manifest helpers."""

from __future__ import annotations

import shutil
from pathlib import Path


def ninja_path() -> str | None:
    return shutil.which("ninja")


def ninja_available() -> bool:
    return ninja_path() is not None


def write_ninja_marker(directory: str | Path) -> Path:
    """Record that the shared loader may use PyTorch's Ninja backend."""

    output = Path(directory) / "ninja.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("ninja_available=" + str(ninja_available()) + "\n", encoding="utf-8")
    return output


__all__ = ["ninja_available", "ninja_path", "write_ninja_marker"]
