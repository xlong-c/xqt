"""Preflight checks for model and device."""

from __future__ import annotations

import torch

from ._base import PreflightReport


def _check_cuda(report: PreflightReport, name: str) -> None:
    available = torch.cuda.is_available()
    report.add(
        name,
        available,
        "CUDA available" if available else "CUDA is not available",
        device_count=torch.cuda.device_count(),
    )


def _check_model_device(report: PreflightReport, device: str | None) -> None:
    if not device:
        report.add("model.device", True, "model device is not configured")
        return
    try:
        torch_device = torch.device(device)
    except Exception as exc:
        report.add("model.device", False, f"invalid device: {exc}", device=device)
        return
    if torch_device.type != "cuda":
        report.add("model.device", True, "non-CUDA device", device=str(torch_device))
        return
    if not torch.cuda.is_available():
        report.add(
            "model.device",
            False,
            "CUDA is required by model.device but is not available",
            device=str(torch_device),
            device_count=torch.cuda.device_count(),
        )
        return
    if (
        torch_device.index is not None
        and torch_device.index >= torch.cuda.device_count()
    ):
        report.add(
            "model.device",
            False,
            "CUDA device index is out of range",
            device=str(torch_device),
            device_count=torch.cuda.device_count(),
        )
        return
    report.add(
        "model.device",
        True,
        "CUDA device available",
        device=str(torch_device),
        device_count=torch.cuda.device_count(),
    )
