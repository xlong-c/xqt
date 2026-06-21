"""Small deterministic detection module for XQT smoke recipes."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ToyDetectionModule(nn.Module):
    """Emit deterministic YOLO-style raw detections matching synthetic targets."""

    def __init__(
        self,
        *,
        num_classes: int = 3,
        boxes_per_image: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.boxes_per_image = boxes_per_image
        self.stem = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.stem.weight.zero_()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        batch_size, _, height, width = x.shape
        anchor = self.stem(x).mean(dim=(1, 2, 3), keepdim=False) * 0.0
        output = x.new_full(
            (batch_size, 4 + self.num_classes, self.boxes_per_image),
            fill_value=-10.0,
        )
        for box_index in range(self.boxes_per_image):
            x1 = float((box_index + 1) * width / (self.boxes_per_image + 2))
            y1 = float((box_index + 1) * height / (self.boxes_per_image + 2))
            x2 = min(float(width - 1), x1 + max(4.0, width * 0.2))
            y2 = min(float(height - 1), y1 + max(4.0, height * 0.2))
            output[:, 0, box_index] = x1
            output[:, 1, box_index] = y1
            output[:, 2, box_index] = x2
            output[:, 3, box_index] = y2
            output[:, 4 + (box_index % max(self.num_classes, 1)), box_index] = 10.0
        return output + anchor.view(batch_size, 1, 1)


def build_toy_detection_module(
    num_classes: int = 3,
    boxes_per_image: int = 2,
) -> ToyDetectionModule:
    """Build a deterministic detection model for synthetic smoke tests."""

    return ToyDetectionModule(
        num_classes=num_classes,
        boxes_per_image=boxes_per_image,
    )


__all__ = [
    "ToyDetectionModule",
    "build_toy_detection_module",
]
