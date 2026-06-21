"""Built-in XQT data split builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import Subset

from xdl.config.builder import build_collate_fn, build_dataloader, build_dataset

from .detection import (
    SyntheticDetectionSpec,
    build_synthetic_detection_loader,
    build_ultralytics_detection_loader,
    build_xdl_detection_loader,
)
from .hf_text import build_hf_text_classification_loader
from .prompts import build_prompt_list, build_prompt_list_from_file
from .samples import SyntheticClassificationSpec, build_synthetic_classification_loader
from .torchvision import (
    TorchvisionImageClassificationSpec,
    build_torchvision_image_classification_loader,
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


def _build_synthetic_split(
    split: Any,
    *,
    model_params: Mapping[str, Any] | None,
    default_seed: int,
) -> Any:
    params = _as_mapping(_read_split_field(split, "params", {}))
    model_config = dict(model_params or {})
    input_shape = params.get("input_shape")
    num_classes = int(params.get("num_classes", model_config.get("num_classes", 2)))
    input_dim = int(params.get("input_dim", model_config.get("in_features", 4)))
    spec = SyntheticClassificationSpec(
        sample_limit=int(_read_split_field(split, "sample_limit", 1)),
        batch_size=int(_read_split_field(split, "batch_size", 1)),
        input_shape=list(input_shape) if input_shape is not None else None,
        num_classes=num_classes,
        input_dim=input_dim,
        seed=int(params.get("seed", default_seed)),
    )
    return build_synthetic_classification_loader(spec)


def _build_torchvision_split(split: Any) -> Any:
    params = _as_mapping(_read_split_field(split, "params", {}))
    transform_params = _as_mapping(params.get("transform_params"))
    if "normalize_mean" not in transform_params and "mean" in transform_params:
        transform_params["normalize_mean"] = transform_params["mean"]
    if "normalize_std" not in transform_params and "std" in transform_params:
        transform_params["normalize_std"] = transform_params["std"]
    spec = TorchvisionImageClassificationSpec(
        dataset_name=str(params["dataset_name"]),
        root=str(_read_split_field(split, "root", "data")),
        train=bool(params.get("train", False)),
        download=bool(params.get("download", False)),
        sample_limit=_read_split_field(split, "sample_limit", None),
        batch_size=int(_read_split_field(split, "batch_size", 1)),
        transform_params=transform_params,
        shuffle=bool(params.get("shuffle", False)),
        num_workers=int(params.get("num_workers", 0)),
    )
    return build_torchvision_image_classification_loader(spec)


def _build_synthetic_detection_split(
    split: Any,
    *,
    default_seed: int,
) -> Any:
    params = _as_mapping(_read_split_field(split, "params", {}))
    input_shape = params.get("input_shape") or params.get("image_shape") or [3, 64, 64]
    spec = SyntheticDetectionSpec(
        sample_limit=int(_read_split_field(split, "sample_limit", 1)),
        batch_size=int(_read_split_field(split, "batch_size", 1)),
        image_shape=list(input_shape),
        num_classes=int(params.get("num_classes", 3)),
        boxes_per_image=int(params.get("boxes_per_image", 2)),
        seed=int(params.get("seed", default_seed)),
    )
    return build_synthetic_detection_loader(spec)


def _build_xdl_dataset_split(split: Any) -> Any:
    params = _as_mapping(_read_split_field(split, "params", {}))
    dataset_config = _as_mapping(params.get("dataset"))
    if not dataset_config:
        raise ValueError("xdl_dataset data split requires params.dataset")
    dataset = build_dataset(dataset_config)

    sample_limit = _read_split_field(split, "sample_limit", None)
    if sample_limit is not None and len(dataset) > int(sample_limit):
        dataset = Subset(dataset, list(range(int(sample_limit))))

    dataloader_params = _as_mapping(params.get("dataloader"))
    dataloader_config = {
        "params": {
            "batch_size": int(_read_split_field(split, "batch_size", 1)),
            **dataloader_params,
        }
    }
    collate_config = params.get("collate_fn", dataloader_params.pop("collate_fn", None))
    collate_fn = build_collate_fn(collate_config) if collate_config is not None else None
    return build_dataloader(dataset, dataloader_config, collate_fn=collate_fn)


def build_data_split(
    split_name: str,
    split: Any,
    *,
    model_params: Mapping[str, Any] | None = None,
    default_seed: int = 0,
) -> Any:
    """Build one configured XQT data split."""

    target = _read_split_field(split, "target")
    if target == "synthetic_classification":
        return _build_synthetic_split(
            split,
            model_params=model_params,
            default_seed=default_seed,
        )
    if target == "torchvision_image_classification":
        return _build_torchvision_split(split)
    if target == "synthetic_detection":
        return _build_synthetic_detection_split(
            split,
            default_seed=default_seed,
        )
    if target == "hf_text_classification":
        return build_hf_text_classification_loader(
            split_name,
            model_params=model_params,
            split_params=_as_mapping(_read_split_field(split, "params", {})),
            batch_size=int(_read_split_field(split, "batch_size", 1)),
            sample_limit=_read_split_field(split, "sample_limit", None),
        )
    if target == "xdl_dataset":
        return _build_xdl_dataset_split(split)
    if target == "xdl_detection":
        return build_xdl_detection_loader(split)
    if target == "ultralytics_detection":
        return build_ultralytics_detection_loader(split)
    if target == "prompt_list":
        params = _as_mapping(_read_split_field(split, "params", {}))
        return build_prompt_list(
            params.get("prompts", []),
            sample_limit=_read_split_field(split, "sample_limit", None),
        )
    if target == "prompt_file":
        params = _as_mapping(_read_split_field(split, "params", {}))
        path = (
            params.get("path")
            or params.get("prompt_file")
            or _read_split_field(split, "root", None)
        )
        if path is None:
            raise ValueError("prompt_file data split requires params.path or root")
        return build_prompt_list_from_file(
            path,
            sample_limit=_read_split_field(split, "sample_limit", None),
        )
    raise ValueError(f"Unsupported built-in data target: {target}")


__all__ = ["build_data_split"]
