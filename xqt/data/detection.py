"""Detection data helpers for XQT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from xdl.config.builder import build_collate_fn, build_dataloader, build_dataset
from xdl.dataset.collate import DetectionCollate
from xdl.dataset.transforms import ImageBoxesTransform
from xqt.model import resolve_ultralytics_dataset


@dataclass
class SyntheticDetectionSpec:
    """Specification for generating synthetic detection samples."""

    sample_limit: int
    batch_size: int
    image_shape: list[int]
    num_classes: int = 3
    boxes_per_image: int = 2
    seed: int = 0


class _SyntheticDetectionDataset(Dataset[dict[str, Any]]):
    """Small synthetic detection dataset used by tests and smoke recipes."""

    def __init__(self, spec: SyntheticDetectionSpec) -> None:
        self.spec = spec
        self._generator = torch.Generator().manual_seed(spec.seed)

    def __len__(self) -> int:
        return self.spec.sample_limit

    def __getitem__(self, index: int) -> dict[str, Any]:
        channels, height, width = self.spec.image_shape
        image = torch.rand(
            channels,
            height,
            width,
            generator=self._generator,
        )
        boxes: list[list[float]] = []
        labels: list[int] = []
        for box_index in range(self.spec.boxes_per_image):
            x1 = float((box_index + 1) * width / (self.spec.boxes_per_image + 2))
            y1 = float((box_index + 1) * height / (self.spec.boxes_per_image + 2))
            x2 = min(float(width - 1), x1 + max(4.0, width * 0.2))
            y2 = min(float(height - 1), y1 + max(4.0, height * 0.2))
            boxes.append([x1, y1, x2, y2])
            labels.append(box_index % max(self.spec.num_classes, 1))
        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }


class _UltralyticsDetectionDataset(Dataset[dict[str, Any]]):
    """Dataset adapter for Ultralytics detection labels/images."""

    def __init__(
        self,
        image_dir: str | Path,
        *,
        class_names: list[str],
        transform: Optional[ImageBoxesTransform] = None,
        sample_limit: Optional[int] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.class_names = list(class_names)
        self.transform = transform
        image_paths = sorted(
            [
                path
                for path in self.image_dir.rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        )
        if sample_limit is not None:
            image_paths = image_paths[: int(sample_limit)]
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        images_root = self.image_dir.parent
        labels_root = images_root.parent / "labels"
        relative = image_path.relative_to(images_root)
        return (labels_root / relative).with_suffix(".txt")

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        labels: list[int] = []
        label_path = self._label_path(image_path)
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                class_id, cx, cy, box_w, box_h = (float(part) for part in stripped.split())
                x1 = (cx - box_w / 2.0) * width
                y1 = (cy - box_h / 2.0) * height
                x2 = (cx + box_w / 2.0) * width
                y2 = (cy + box_h / 2.0) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))
        if self.transform is not None:
            image_value, box_tensor = self.transform(image, boxes)
        else:
            image_value = load_image_tensor(image_path)
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
        return {
            "image": image_value,
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "image_path": str(image_path),
            "class_names": list(self.class_names),
        }


def build_synthetic_detection_loader(spec: SyntheticDetectionSpec) -> DataLoader[Any]:
    """Build a synthetic detection loader with variable-size targets preserved."""

    dataset = _SyntheticDetectionDataset(spec)
    return DataLoader(
        dataset,
        batch_size=spec.batch_size,
        collate_fn=DetectionCollate(),
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected a mapping, got {type(value).__name__}")


def _read_split_field(split: Any, name: str, default: Any = None) -> Any:
    if isinstance(split, Mapping):
        return split.get(name, default)
    return getattr(split, name, default)


def build_xdl_detection_loader(split: Any) -> Any:
    """Build an XDL detection dataset split with detection-aware defaults."""

    params = _as_mapping(_read_split_field(split, "params", {}))
    dataset_config = _as_mapping(params.get("dataset"))
    if not dataset_config:
        raise ValueError("xdl_detection data split requires params.dataset")

    transform_config = _as_mapping(params.get("transform"))
    if transform_config.get("target") == "xqt.data.detection.build_image_boxes_transform":
        transform_params = _as_mapping(transform_config.get("params"))
        dataset_config.setdefault("params", {})
        dataset_config["params"]["transform"] = build_image_boxes_transform(**transform_params)

    dataset = build_dataset(dataset_config)
    dataloader_params = _as_mapping(params.get("dataloader"))
    batch_size = int(_read_split_field(split, "batch_size", 1))
    dataloader_config = {
        "params": {
            "batch_size": batch_size,
            **dataloader_params,
        }
    }
    collate_config = params.get("collate_fn", dataloader_params.pop("collate_fn", None))
    collate_fn = (
        build_collate_fn(collate_config)
        if collate_config is not None
        else DetectionCollate()
    )
    return build_dataloader(dataset, dataloader_config, collate_fn=collate_fn)


def build_ultralytics_detection_loader(split: Any) -> DataLoader[Any]:
    """Build a dataloader from an Ultralytics detection dataset YAML."""

    params = _as_mapping(_read_split_field(split, "params", {}))
    dataset_yaml = str(
        params.get("dataset")
        or params.get("yaml")
        or _read_split_field(split, "root", "coco8.yaml")
    )
    autodownload = bool(params.get("autodownload", True))
    subset = str(params.get("subset", "val"))
    info = resolve_ultralytics_dataset(dataset_yaml, autodownload=autodownload)
    image_dir = info.val if subset == "val" else info.train
    if image_dir is None:
        raise ValueError(f"Ultralytics dataset subset {subset!r} is not available")
    transform_params = _as_mapping(params.get("transform"))
    transform = (
        build_image_boxes_transform(**transform_params)
        if transform_params
        else build_image_boxes_transform(size=params.get("image_size", [640, 640]))
    )
    dataset = _UltralyticsDetectionDataset(
        image_dir,
        class_names=info.names,
        transform=transform,
        sample_limit=_read_split_field(split, "sample_limit", None),
    )
    return DataLoader(
        dataset,
        batch_size=int(_read_split_field(split, "batch_size", 1)),
        shuffle=bool(params.get("shuffle", False)),
        num_workers=int(params.get("num_workers", 0)),
        collate_fn=DetectionCollate(),
    )


def build_image_boxes_transform(
    size: Iterable[int] | None = None,
    *,
    random_flip: bool = False,
    normalize: bool = True,
) -> ImageBoxesTransform:
    """Build an :class:`ImageBoxesTransform` from recipe-friendly params."""

    resolved_size = tuple(int(value) for value in size) if size is not None else (640, 640)
    return ImageBoxesTransform(
        height=int(resolved_size[0]),
        width=int(resolved_size[1]),
        random_flip=random_flip,
        normalize=normalize,
    )


def load_image_tensor(path: str | Path) -> torch.Tensor:
    """Load one image file as a CHW float tensor in [0, 1]."""

    image = Image.open(Path(path)).convert("RGB")
    tensor = torch.from_numpy(np.array(image, copy=True))
    return tensor.permute(2, 0, 1).float() / 255.0


__all__ = [
    "SyntheticDetectionSpec",
    "build_image_boxes_transform",
    "build_synthetic_detection_loader",
    "build_ultralytics_detection_loader",
    "build_xdl_detection_loader",
    "load_image_tensor",
]
