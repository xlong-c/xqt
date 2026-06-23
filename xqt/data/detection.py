"""Detection data helpers for XQT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from xdl.config.builder import build_collate_fn, build_dataloader, build_dataset
from xdl.dataset.collate import DetectionCollate
from xdl.dataset.transforms import ImageBoxesTransform

@dataclass
class SyntheticDetectionSpec:
    """Specification for generating synthetic detection samples."""

    sample_limit: int
    batch_size: int
    image_shape: list[int]
    num_classes: int = 3
    boxes_per_image: int = 2
    seed: int = 0
    include_model_inputs: bool = False
    input_image_key: str = "images"
    input_size_key: str = "orig_target_sizes"


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
        sample = {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }
        if self.spec.include_model_inputs:
            sample[self.spec.input_image_key] = image
            sample[self.spec.input_size_key] = torch.tensor([height, width], dtype=torch.long)
        return sample


@dataclass
class Coco8DetectionSpec:
    """Specification for loading the local COCO8-style detection sample set."""

    root: str = "data/coco8"
    split: str = "val"
    batch_size: int = 1
    sample_limit: int | None = None
    include_model_inputs: bool = False
    input_image_key: str = "images"
    input_size_key: str = "orig_target_sizes"
    normalize: bool = True
    resize: tuple[int, int] | None = (640, 640)


class _Coco8DetectionDataset(Dataset[dict[str, Any]]):
    """Small loader for the local COCO8 sample set with YOLO txt labels."""

    def __init__(self, spec: Coco8DetectionSpec) -> None:
        self.spec = spec
        root = Path(spec.root)
        self.image_dir = root / "images" / spec.split
        self.label_dir = root / "labels" / spec.split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"COCO8 image dir not found: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"COCO8 label dir not found: {self.label_dir}")
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        if spec.sample_limit is not None:
            self.image_paths = self.image_paths[: int(spec.sample_limit)]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, bw, bh = parts
                center_x = float(cx) * width
                center_y = float(cy) * height
                box_w = float(bw) * width
                box_h = float(bh) * height
                x1 = max(0.0, center_x - box_w / 2.0)
                y1 = max(0.0, center_y - box_h / 2.0)
                x2 = min(float(width - 1), center_x + box_w / 2.0)
                y2 = min(float(height - 1), center_y + box_h / 2.0)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        if self.spec.resize is not None:
            transform = ImageBoxesTransform(
                height=int(self.spec.resize[0]),
                width=int(self.spec.resize[1]),
                random_flip=False,
                normalize=self.spec.normalize,
            )
            tensor, box_tensor = transform(image, boxes)
        else:
            tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
            if self.spec.normalize:
                tensor = tensor / 255.0
            box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        sample = {
            "image": tensor,
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "image_path": str(image_path),
        }
        if self.spec.include_model_inputs:
            sample[self.spec.input_image_key] = tensor
            sample[self.spec.input_size_key] = torch.tensor([height, width], dtype=torch.long)
        return sample



def build_synthetic_detection_loader(spec: SyntheticDetectionSpec) -> DataLoader[Any]:
    """Build a synthetic detection loader with variable-size targets preserved."""

    dataset = _SyntheticDetectionDataset(spec)
    loader = DataLoader(
        dataset,
        batch_size=spec.batch_size,
        collate_fn=DetectionCollate(),
    )
    loader.xqt_detection_metadata = {
        "target": "synthetic_detection",
        "sample_limit": int(spec.sample_limit),
        "batch_size": int(spec.batch_size),
        "image_shape": [int(value) for value in spec.image_shape],
        "resize": [int(spec.image_shape[1]), int(spec.image_shape[2])],
        "resize_mode": "direct_resize",
        "letterbox": False,
        "num_classes": int(spec.num_classes),
        "boxes_per_image": int(spec.boxes_per_image),
        "seed": int(spec.seed),
        "include_model_inputs": bool(spec.include_model_inputs),
        "input_image_key": str(spec.input_image_key),
        "input_size_key": str(spec.input_size_key),
    }
    return loader


def build_coco8_detection_loader(spec: Coco8DetectionSpec) -> DataLoader[Any]:
    """Build a detection loader for the local COCO8 sample set."""

    dataset = _Coco8DetectionDataset(spec)
    loader = DataLoader(
        dataset,
        batch_size=spec.batch_size,
        collate_fn=DetectionCollate(),
    )
    loader.xqt_detection_metadata = {
        "target": "coco8_detection",
        "root": str(spec.root),
        "split": str(spec.split),
        "sample_limit": int(spec.sample_limit) if spec.sample_limit is not None else None,
        "batch_size": int(spec.batch_size),
        "include_model_inputs": bool(spec.include_model_inputs),
        "input_image_key": str(spec.input_image_key),
        "input_size_key": str(spec.input_size_key),
        "normalize": bool(spec.normalize),
        "resize_mode": "direct_resize",
        "letterbox": False,
        "resize": (
            [int(value) for value in spec.resize]
            if spec.resize is not None
            else None
        ),
    }
    return loader


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
    "Coco8DetectionSpec",
    "SyntheticDetectionSpec",
    "build_coco8_detection_loader",
    "build_image_boxes_transform",
    "build_synthetic_detection_loader",
    "build_xdl_detection_loader",
    "load_image_tensor",
]
