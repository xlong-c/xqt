"""Smoke-only multimodal module and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class SmokeMultimodalClassifier(nn.Module):
    """Tiny vision + text multimodal classifier for CPU smoke tests."""

    def __init__(
        self,
        *,
        image_channels: int = 3,
        image_size: int = 16,
        vision_hidden: int = 16,
        text_hidden: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(image_channels, vision_hidden, kernel_size=4, stride=4),
            nn.Flatten(),
        )
        vision_features = vision_hidden * (image_size // 4) ** 2
        self.vision_proj = nn.Linear(vision_features, vision_hidden)
        self.text_embed = nn.Linear(text_hidden, text_hidden)
        self.cross_attn = nn.MultiheadAttention(
            text_hidden,
            1,
            batch_first=True,
        )
        self.head = nn.Linear(text_hidden, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.vision_proj(self.vision_encoder(image))
        text_hidden = self.text_embed(text)
        attended, _ = self.cross_attn(text_hidden, visual, visual, need_weights=False)
        return self.head(attended.mean(dim=1))


def build_smoke_multimodal_classifier(
    *,
    image_channels: int = 3,
    image_size: int = 16,
    vision_hidden: int = 16,
    text_hidden: int = 16,
    num_classes: int = 4,
) -> SmokeMultimodalClassifier:
    """Build a deterministic small multimodal smoke model."""

    return SmokeMultimodalClassifier(
        image_channels=image_channels,
        image_size=image_size,
        vision_hidden=vision_hidden,
        text_hidden=text_hidden,
        num_classes=num_classes,
    )


@dataclass(frozen=True)
class MultimodalInputSignature:
    """Model-side multimodal input signature metadata."""

    modalities: tuple[tuple[str, str, tuple[int, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": [
                {"name": name, "modality": modality, "shape": list(shape)}
                for name, modality, shape in self.modalities
            ]
        }


def multimodal_input_signature(
    specs: Sequence[Mapping[str, Any]],
) -> MultimodalInputSignature:
    """Normalize multimodal input signature metadata.

    Each spec may contain ``name``, ``modality`` and ``shape``. This is pure
    metadata; no dataloader or input batch is constructed here.
    """

    modalities: list[tuple[str, str, tuple[int, ...]]] = []
    for spec in specs:
        name = str(spec.get("name", ""))
        modality = str(spec.get("modality", "unknown"))
        raw_shape = spec.get("shape", ())
        shape = tuple(int(item) for item in raw_shape)
        modalities.append((name, modality, shape))
    return MultimodalInputSignature(tuple(modalities))


@dataclass(frozen=True)
class VisualTokenCompressionMetadata:
    """Model-side visual token compression metadata (no compression impl)."""

    ratio: float
    method: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "method": self.method,
            "notes": list(self.notes),
        }


def visual_token_compression_metadata(
    *,
    ratio: float,
    method: str,
) -> VisualTokenCompressionMetadata:
    """Record visual token compression intent as model-side metadata."""

    if float(ratio) <= 0.0:
        raise ValueError("ratio must be positive")
    if not str(method).strip():
        raise ValueError("method must not be empty")
    return VisualTokenCompressionMetadata(
        ratio=float(ratio),
        method=str(method).strip(),
        notes=("metadata only; XQT 不实现 token 压缩算法.",),
    )


@dataclass(frozen=True)
class EncoderCacheMetadata:
    """Model-side encoder cache metadata (no cache management)."""

    cacheable: bool
    owner: str = "external_runtime"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cacheable": self.cacheable,
            "owner": self.owner,
            "notes": list(self.notes),
        }


def encoder_cache_metadata(
    *,
    cacheable: bool,
    notes: Sequence[str] = (),
) -> EncoderCacheMetadata:
    """Record encoder cache intent; cache storage belongs to external runtime."""

    return EncoderCacheMetadata(
        cacheable=bool(cacheable),
        owner="external_runtime",
        notes=tuple(str(item) for item in notes),
    )


__all__ = [
    "EncoderCacheMetadata",
    "MultimodalInputSignature",
    "SmokeMultimodalClassifier",
    "VisualTokenCompressionMetadata",
    "build_smoke_multimodal_classifier",
    "encoder_cache_metadata",
    "multimodal_input_signature",
    "visual_token_compression_metadata",
]
