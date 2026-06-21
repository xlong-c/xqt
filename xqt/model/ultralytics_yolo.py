"""Ultralytics YOLO adapters for XQT detection practice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from xqt.core.errors import XQTBackendError


def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise XQTBackendError("ultralytics is required for YOLO detection recipes") from exc
    return YOLO


@dataclass
class UltralyticsDatasetInfo:
    """Resolved detection dataset metadata from Ultralytics YAML."""

    yaml_path: str
    root: str
    train: Optional[str]
    val: Optional[str]
    names: list[str]
    nc: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "yaml_path": self.yaml_path,
            "root": self.root,
            "train": self.train,
            "val": self.val,
            "names": list(self.names),
            "nc": self.nc,
        }


class UltralyticsDetectionModule(nn.Module):
    """Thin nn.Module wrapper over an Ultralytics detection backbone."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        super().__init__()
        YOLO = _require_ultralytics()
        self.weights = weights
        bundle = YOLO(weights)
        if getattr(bundle, "task", None) != "detect":
            raise XQTBackendError(f"Ultralytics model task must be detect, got {bundle.task!r}")
        self.detector = bundle.model
        self.detector.eval()
        self.names = dict(getattr(bundle, "names", {}))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return self.detector(x)


def build_ultralytics_detection_module(weights: str = "yolo11n.pt") -> UltralyticsDetectionModule:
    """Build a wrapped Ultralytics detection module suitable for XQT."""

    return UltralyticsDetectionModule(weights=weights)


def resolve_ultralytics_dataset(
    dataset: str,
    *,
    autodownload: bool = True,
) -> UltralyticsDatasetInfo:
    """Resolve an Ultralytics detection dataset YAML, downloading if requested."""

    try:
        from ultralytics.data.utils import check_det_dataset
    except ImportError as exc:
        raise XQTBackendError("ultralytics is required to resolve detection datasets") from exc

    info = check_det_dataset(dataset, autodownload=autodownload)
    names_raw = info.get("names", {})
    if isinstance(names_raw, dict):
        names = [str(name) for _, name in sorted(names_raw.items())]
    else:
        names = [str(name) for name in names_raw]
    return UltralyticsDatasetInfo(
        yaml_path=str(dataset),
        root=str(info.get("path")),
        train=str(info.get("train")) if info.get("train") is not None else None,
        val=str(info.get("val")) if info.get("val") is not None else None,
        names=names,
        nc=int(info.get("nc", len(names))),
    )


def ultralytics_class_names(weights: str = "yolo11n.pt") -> list[str]:
    """Return stable class names from a YOLO checkpoint."""

    module = UltralyticsDetectionModule(weights=weights)
    return [str(name) for _, name in sorted(module.names.items())]


def export_ultralytics_reference(
    weights: str,
    *,
    format: str = "onnx",
    imgsz: int = 640,
    data: Optional[str] = None,
    fraction: Optional[float] = None,
    int8: bool = False,
    half: bool = False,
    dynamic: bool = False,
    nms: bool = False,
    device: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Run Ultralytics native export for baseline/reference artifacts."""

    YOLO = _require_ultralytics()
    model = YOLO(weights)
    kwargs: dict[str, Any] = {
        "format": format,
        "imgsz": imgsz,
        "int8": int8,
        "half": half,
        "dynamic": dynamic,
        "nms": nms,
    }
    if data is not None:
        kwargs["data"] = data
    if fraction is not None:
        kwargs["fraction"] = fraction
    if device is not None:
        kwargs["device"] = device
    if project is not None:
        kwargs["project"] = project
    if name is not None:
        kwargs["name"] = name
    return model.export(**kwargs)


__all__ = [
    "UltralyticsDatasetInfo",
    "UltralyticsDetectionModule",
    "build_ultralytics_detection_module",
    "export_ultralytics_reference",
    "resolve_ultralytics_dataset",
    "ultralytics_class_names",
]
