"""Torchvision-based image classification data loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import warnings

from torch.utils.data import DataLoader, Subset


@dataclass
class TorchvisionImageClassificationSpec:
    """Specification for a torchvision image classification dataset.

    If *sample_limit* is set, only the first *sample_limit* samples are used.
    """

    dataset_name: str
    root: str
    train: bool = False
    download: bool = False
    sample_limit: int | None = 1
    batch_size: int = 1
    transform_params: Dict[str, Any] = field(default_factory=dict)
    shuffle: bool = False
    num_workers: int = 0


def build_torchvision_image_classification_loader(
    spec: TorchvisionImageClassificationSpec,
) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` from a torchvision
    image classification dataset.

    Returns
    -------
    DataLoader
        Yields ``(inputs, targets)`` tuples where *inputs* is a ``(B, C, H, W)``
        image tensor and *targets* are integer class labels.
    """
    try:
        import torchvision.datasets  # type: ignore[import-untyped]
        import torchvision.transforms  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for torchvision image classification data. "
            "Install it with: pip install torchvision"
        ) from exc

    dataset_cls = getattr(torchvision.datasets, spec.dataset_name, None)
    if dataset_cls is None:
        available = [
            name
            for name in dir(torchvision.datasets)
            if not name.startswith("_") and isinstance(
                getattr(torchvision.datasets, name, None),
                type,
            )
        ]
        raise ValueError(
            f"Unknown torchvision dataset: {spec.dataset_name}. "
            f"Available: {available}"
        )

    transform = _build_transform(spec.transform_params)

    dataset_kwargs: Dict[str, Any] = {
        "root": spec.root,
        "train": spec.train,
        "transform": transform,
    }
    if "download" in dataset_cls.__init__.__code__.co_varnames:  # type: ignore[union-attr]
        dataset_kwargs["download"] = spec.download

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = dataset_cls(**dataset_kwargs)

    if spec.sample_limit is not None and len(dataset) > spec.sample_limit:
        indices = list(range(spec.sample_limit))
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=spec.shuffle,
        num_workers=spec.num_workers,
    )


def _build_transform(
    params: Dict[str, Any],
) -> Any:
    """Build a torchvision transform from the given parameters dict."""
    try:
        import torchvision.transforms  # type: ignore[import-untyped]
    except ImportError:
        return None

    ops: list[Any] = []

    image_size = params.get("image_size")
    if image_size is not None:
        ops.append(torchvision.transforms.Resize((image_size, image_size)))

    ops.append(torchvision.transforms.ToTensor())

    normalize_mean = params.get("normalize_mean", params.get("mean"))
    normalize_std = params.get("normalize_std", params.get("std"))
    if normalize_mean is not None and normalize_std is not None:
        ops.append(
            torchvision.transforms.Normalize(
                mean=normalize_mean,
                std=normalize_std,
            )
        )

    if len(ops) == 1:
        return ops[0]
    return torchvision.transforms.Compose(ops)


__all__ = [
    "TorchvisionImageClassificationSpec",
    "build_torchvision_image_classification_loader",
]
