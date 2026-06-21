"""Model adapters used by XQT recipes."""

from .toy_detection import ToyDetectionModule, build_toy_detection_module
from .ultralytics_yolo import (
    UltralyticsDatasetInfo,
    UltralyticsDetectionModule,
    build_ultralytics_detection_module,
    export_ultralytics_reference,
    resolve_ultralytics_dataset,
    ultralytics_class_names,
)

__all__ = [
    "ToyDetectionModule",
    "UltralyticsDatasetInfo",
    "UltralyticsDetectionModule",
    "build_toy_detection_module",
    "build_ultralytics_detection_module",
    "export_ultralytics_reference",
    "resolve_ultralytics_dataset",
    "ultralytics_class_names",
]
