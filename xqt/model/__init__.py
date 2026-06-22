"""Model adapters used by XQT recipes."""

from .toy_detection import ToyDetectionModule, build_toy_detection_module

__all__ = [
    "ToyDetectionModule",
    "build_toy_detection_module",
]
