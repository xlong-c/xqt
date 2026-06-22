"""Bridge XDL dataset templates into XQT recipe data splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from torch.utils.data import DataLoader, Subset

from xdl.config.builder import build_collate_fn, build_dataloader, build_dataset
from xdl.config.errors import ConfigValidationError


@dataclass
class XDLDatasetBridgeSpec:
    """Bridge one XDL dataset config into an XQT dataloader."""

    dataset: dict[str, Any]
    dataloader: dict[str, Any] = field(default_factory=dict)
    batch_size: Optional[int] = 1
    sample_limit: Optional[int] = None
    collate_fn: Any = None


def _normalize_dataloader_config(
    config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Any]:
    if config is None:
        return {}, None
    if not isinstance(config, Mapping):
        raise ConfigValidationError("xdl_dataset dataloader config must be a mapping")

    config_dict = dict(config)
    collate_config = config_dict.get("collate_fn")
    if "params" in config_dict:
        params = dict(config_dict.get("params") or {})
        extras = {
            key: value
            for key, value in config_dict.items()
            if key not in {"params", "collate_fn"}
        }
        params = {**extras, **params}
        return params, collate_config

    return (
        {key: value for key, value in config_dict.items() if key != "collate_fn"},
        collate_config,
    )


def _apply_sample_limit(dataset: Any, sample_limit: Optional[int]) -> Any:
    if sample_limit is None:
        return dataset
    if int(sample_limit) <= 0:
        raise ValueError("sample_limit must be positive")
    try:
        total = len(dataset)
    except TypeError as exc:
        raise TypeError("xdl_dataset bridge requires a sized dataset for sample_limit") from exc
    return Subset(dataset, range(min(int(sample_limit), total)))


def build_xdl_dataset_loader(spec: XDLDatasetBridgeSpec) -> DataLoader[Any]:
    """Build a DataLoader from an XDL dataset config."""

    if not spec.dataset:
        raise ConfigValidationError("xdl_dataset requires params.dataset")

    dataset = build_dataset(dict(spec.dataset))
    dataset = _apply_sample_limit(dataset, spec.sample_limit)

    loader_params, nested_collate = _normalize_dataloader_config(spec.dataloader)
    if "batch_size" not in loader_params:
        loader_params["batch_size"] = spec.batch_size
    collate_fn = build_collate_fn(spec.collate_fn or nested_collate)
    return build_dataloader(
        dataset,
        {"params": loader_params},
        collate_fn=collate_fn,
    )


__all__ = ["XDLDatasetBridgeSpec", "build_xdl_dataset_loader"]
