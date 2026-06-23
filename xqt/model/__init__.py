"""Smoke-only model helpers used by XQT recipes."""

from .smoke_detection import SmokeDetectionModule, build_smoke_detection_module

__all__ = [
    "SmokeDetectionModule",
    "build_smoke_detection_module",
]
