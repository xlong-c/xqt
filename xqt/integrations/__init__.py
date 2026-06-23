"""Lightweight model-output integration helpers for XQT."""

from .detection import DetectionPrediction, decode_detection_output

__all__ = [
    "DetectionPrediction",
    "decode_detection_output",
]
