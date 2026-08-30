"""Lazy host/CUDA toolchain discovery for diagnostics and smoke builds."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    return output.strip() or None


def find_nvcc() -> str | None:
    configured = os.environ.get("NVCC")
    candidate = configured or shutil.which("nvcc")
    return None if candidate is None else str(candidate)


def find_cxx() -> str | None:
    configured = os.environ.get("CXX")
    candidate = configured or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    return None if candidate is None else str(candidate)


def tool_version(command: str | None) -> str | None:
    return None if command is None else _command_output([command, "--version"])


@dataclass(frozen=True, slots=True)
class ToolchainInfo:
    nvcc: str | None
    nvcc_version: str | None
    cxx: str | None
    cxx_version: str | None
    ninja: str | None

    @property
    def available(self) -> bool:
        return self.nvcc is not None and self.cxx is not None

    def to_dict(self) -> dict[str, str | None | bool]:
        return {
            "nvcc": self.nvcc,
            "nvcc_version": self.nvcc_version,
            "cxx": self.cxx,
            "cxx_version": self.cxx_version,
            "ninja": self.ninja,
            "available": self.available,
        }


def detect_toolchain() -> ToolchainInfo:
    nvcc = find_nvcc()
    cxx = find_cxx()
    return ToolchainInfo(
        nvcc=nvcc,
        nvcc_version=tool_version(nvcc),
        cxx=cxx,
        cxx_version=tool_version(cxx),
        ninja=shutil.which("ninja"),
    )


__all__ = ["ToolchainInfo", "detect_toolchain", "find_cxx", "find_nvcc", "tool_version"]
