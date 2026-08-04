#!/usr/bin/env python3
"""Print a reproducible NVIDIA profiling environment snapshot as JSON."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from io import StringIO
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "status": "error",
            "error": str(exc),
        }
    return {
        "command": command,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _tool_snapshot(name: str, version_command: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"path": None, "status": "missing"}
    result = _run(version_command)
    return {"path": path, **result}


def _nvidia_smi_snapshot() -> dict[str, Any]:
    path = shutil.which("nvidia-smi")
    if path is None:
        return {"path": None, "status": "missing", "devices": []}
    command = [
        path,
        "--query-gpu=index,name,uuid,driver_version,pstate,power.limit,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]
    result = _run(command)
    devices: list[dict[str, str]] = []
    if result["status"] == "ok":
        fields = [
            "index",
            "name",
            "uuid",
            "driver_version",
            "pstate",
            "power_limit_watts",
            "sm_clock_mhz",
            "memory_clock_mhz",
        ]
        for row in csv.reader(StringIO(str(result["stdout"]))):
            if len(row) != len(fields):
                continue
            devices.append(
                {field: value.strip() for field, value in zip(fields, row, strict=True)}
            )
    return {"path": path, "devices": devices, **result}


def _torch_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"status": "missing", "error": str(exc), "devices": []}

    devices: list[dict[str, Any]] = []
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "sm": f"sm_{properties.major}{properties.minor}",
                    "total_memory_bytes": properties.total_memory,
                    "multiprocessor_count": properties.multi_processor_count,
                }
            )
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "devices": devices,
    }


def main() -> None:
    snapshot = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "host": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "tools": {
            "ncu": _tool_snapshot("ncu", ["ncu", "--version"]),
            "nsys": _tool_snapshot("nsys", ["nsys", "--version"]),
        },
        "nvidia_smi": _nvidia_smi_snapshot(),
        "torch": _torch_snapshot(),
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
