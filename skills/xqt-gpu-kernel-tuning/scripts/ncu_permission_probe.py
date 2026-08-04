#!/usr/bin/env python3
"""Probe whether the current user can collect basic Nsight Compute counters."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any


_WORKLOAD = """
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA_UNAVAILABLE")
x = torch.ones((16, 16), device="cuda")
torch.cuda.synchronize()
x @ x
torch.cuda.synchronize()
"""


def _result(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details}


def _probe() -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        return _result("missing_tool", ncu_path=None)

    try:
        import torch
    except ImportError as exc:
        return _result("missing_torch", ncu_path=ncu, error=str(exc))
    if not torch.cuda.is_available():
        return _result("cuda_unavailable", ncu_path=ncu)

    command = [
        ncu,
        "--target-processes",
        "all",
        "--set",
        "basic",
        "--launch-skip",
        "0",
        "--launch-count",
        "1",
        sys.executable,
        "-c",
        _WORKLOAD,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result("probe_error", ncu_path=ncu, command=command, error=str(exc))

    output = (completed.stdout + completed.stderr).strip()
    if "ERR_NVGPUCTRPERM" in output:
        status = "counter_permission_denied"
    elif "CUDA_UNAVAILABLE" in output:
        status = "cuda_unavailable"
    elif completed.returncode == 0:
        status = "ready"
    else:
        status = "probe_failed"
    return _result(
        status,
        ncu_path=ncu,
        command=command,
        returncode=completed.returncode,
        output=output,
    )


def main() -> None:
    print(json.dumps(_probe(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
