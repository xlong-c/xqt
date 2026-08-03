"""Weight file discovery and state_dict load for external quant bridges.

Supported auto materialize:
- Single file: ``model.safetensors``, ``pytorch_model.bin``, ``model.pt``, ``model.bin``
- Sharded: ``model.safetensors.index.json`` with ``weight_map`` (U5 multi-file merge)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from xqt.core.errors import XQTArtifactError


def find_weight_file(root: Path) -> Path | None:
    """Return the preferred weight artifact under ``root`` (single file first)."""

    for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
        "model.bin",
    ):
        candidate = root / name
        if candidate.is_file():
            return candidate
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return index
    return None


def _load_single_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise XQTArtifactError(
            "loading .safetensors requires the safetensors package"
        ) from exc
    return dict(load_file(str(path)))


def _load_sharded_safetensors(index_path: Path) -> dict[str, torch.Tensor]:
    """Merge tensors listed in ``model.safetensors.index.json`` weight_map."""

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise XQTArtifactError(
            f"invalid safetensors index JSON: {index_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise XQTArtifactError(
            f"safetensors index must be a JSON object: {index_path}"
        )
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise XQTArtifactError(
            f"sharded safetensors index missing non-empty weight_map: {index_path}"
        )

    root = index_path.parent
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not key:
            raise XQTArtifactError(
                f"weight_map keys must be non-empty strings in {index_path}"
            )
        if not isinstance(shard_name, str) or not shard_name:
            raise XQTArtifactError(
                f"weight_map[{key!r}] must be a non-empty shard filename"
            )
        if Path(shard_name).is_absolute() or ".." in Path(shard_name).parts:
            raise XQTArtifactError(
                f"shard path must be a relative file under the index root: {shard_name}"
            )
        shard_to_keys.setdefault(shard_name, []).append(key)

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise XQTArtifactError(
            "loading sharded .safetensors requires the safetensors package"
        ) from exc

    merged: dict[str, torch.Tensor] = {}
    for shard_name, expected_keys in shard_to_keys.items():
        shard_path = (root / shard_name).resolve()
        try:
            shard_path.relative_to(root.resolve())
        except ValueError as exc:
            raise XQTArtifactError(
                f"shard path escapes index root: {shard_name}"
            ) from exc
        if not shard_path.is_file():
            raise XQTArtifactError(
                f"missing safetensors shard: {shard_name} (required by {index_path})"
            )
        shard_tensors = dict(load_file(str(shard_path)))
        for key in expected_keys:
            if key not in shard_tensors:
                raise XQTArtifactError(
                    f"key {key!r} listed in weight_map for shard {shard_name} "
                    f"but not found in file"
                )
            if key in merged:
                raise XQTArtifactError(
                    f"duplicate weight_map key across shards: {key!r}"
                )
            tensor = shard_tensors[key]
            if not isinstance(tensor, torch.Tensor):
                raise XQTArtifactError(
                    f"shard entry {key!r} is not a tensor in {shard_name}"
                )
            merged[key] = tensor
    return merged


def load_weight_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a flat tensor state_dict from pt/bin/safetensors or a shard index."""

    if path.name.endswith(".index.json") or (
        path.suffix == ".json" and "index" in path.name
    ):
        return _load_sharded_safetensors(path)
    if path.suffix == ".json":
        raise XQTArtifactError(
            f"unsupported weight JSON (expected *.safetensors.index.json): {path}"
        )
    if path.suffix == ".safetensors":
        return _load_single_safetensors(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, nn.Module):
        return dict(payload.state_dict())
    if isinstance(payload, Mapping):
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return {
                str(k): v
                for k, v in payload["state_dict"].items()
                if isinstance(v, torch.Tensor)
            }
        return {
            str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)
        }
    raise XQTArtifactError(f"unsupported weight payload type at {path}")


__all__ = ["find_weight_file", "load_weight_state_dict"]
