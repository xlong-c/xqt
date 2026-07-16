"""Flat Infer delivery: one weights file + quant.json sidecar.

Consumes already-quantized storage + optional compute_config.
Never runs quantizers, calibration, or sensitivity analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from xqt.contracts.compute import (
    ComputeConfig,
    compute_config_from_mapping,
    compute_config_to_dict,
)
from xqt.contracts.quantized import QuantizedModel
from xqt.core.artifact import file_sha256, utc_timestamp
from xqt.core.errors import XQTArtifactError
from xqt.core.serialization import json_safe_value

from .quant_pair_schema import (
    DEFAULT_SIDECAR_NAME,
    DEFAULT_WEIGHTS_NAME,
    QUANT_SIDECAR_ARTIFACT_TYPE,
    QUANT_SIDECAR_SCHEMA_VERSION,
    WEIGHTS_FORMAT_TORCH_STATE_DICT,
    LoadedQuantPair,
    QuantPairManifest,
)


def _json_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    safe_value = json_safe_value(dict(value))
    if not isinstance(safe_value, dict):
        raise XQTArtifactError(f"{name} must serialize to a JSON object")
    return safe_value


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise XQTArtifactError(f"{name} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise XQTArtifactError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise XQTArtifactError(f"{name} must contain a JSON object: {path}")
    return payload


def _resolve_pair_file(pair_dir: Path, relative_path: str, *, name: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise XQTArtifactError(f"{name} must be a relative path: {relative_path}")
    root = pair_dir.resolve()
    resolved = (pair_dir / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise XQTArtifactError(
            f"{name} escapes the quant pair root: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise XQTArtifactError(f"{name} file not found: {resolved}")
    return resolved


def _resolve_sidecar_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        sidecar = candidate / DEFAULT_SIDECAR_NAME
    elif candidate.name == DEFAULT_SIDECAR_NAME or candidate.suffix == ".json":
        sidecar = candidate
    else:
        raise XQTArtifactError(
            "load_quant_pair expects a directory containing quant.json "
            "or a quant.json path"
        )
    if not sidecar.is_file():
        raise XQTArtifactError(f"quant.json file not found: {sidecar}")
    return sidecar.resolve()


def _resolve_compute_config_dict(
    compute_config: ComputeConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if compute_config is None:
        return None
    if isinstance(compute_config, ComputeConfig):
        return compute_config.to_dict()
    return compute_config_to_dict(compute_config_from_mapping(compute_config))


def write_quant_pair(
    model: nn.Module,
    output_dir: str | Path,
    *,
    compute_config: ComputeConfig | Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    weights_name: str = DEFAULT_WEIGHTS_NAME,
    sidecar_name: str = DEFAULT_SIDECAR_NAME,
) -> Path:
    """Write model.pt + quant.json. Model must already hold quantized storage."""

    pair_dir = Path(output_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    weights_path = pair_dir / weights_name
    torch.save(model.state_dict(), weights_path)

    meta = _json_mapping(metadata, name="metadata")
    meta.setdefault("producer", "xqt")
    meta.setdefault("created_at", utc_timestamp())
    QuantPairManifest(
        weights={
            "path": weights_name,
            "format": WEIGHTS_FORMAT_TORCH_STATE_DICT,
            "checksum": file_sha256(weights_path),
        },
        compute_config=_resolve_compute_config_dict(compute_config),
        lineage=_json_mapping(lineage, name="lineage"),
        metadata=meta,
    ).write_json(pair_dir / sidecar_name)
    return pair_dir


def write_quant_pair_from_quantized(
    quantized: QuantizedModel,
    output_dir: str | Path,
    *,
    weights_name: str = DEFAULT_WEIGHTS_NAME,
    sidecar_name: str = DEFAULT_SIDECAR_NAME,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a quant pair from QuantizedModel via infer_handoff() only."""

    handoff = quantized.infer_handoff()
    model = handoff["model"]
    if not isinstance(model, nn.Module):
        raise XQTArtifactError(
            "write_quant_pair_from_quantized requires QuantizedModel.model "
            "to be a torch.nn.Module"
        )
    return write_quant_pair(
        model,
        output_dir,
        compute_config=handoff.get("compute_config"),
        lineage={
            "backend": quantized.backend,
            "method": quantized.method,
            "strategy": quantized.strategy,
        },
        metadata=metadata,
        weights_name=weights_name,
        sidecar_name=sidecar_name,
    )


def load_quant_pair(path: str | Path) -> LoadedQuantPair:
    """Load and validate one weights + quant.json pair (no model mutation)."""

    sidecar_path = _resolve_sidecar_path(path)
    pair_dir = sidecar_path.parent
    manifest = QuantPairManifest.from_dict(
        _load_json_mapping(sidecar_path, name="quant.json")
    )
    weights_path = _resolve_pair_file(
        pair_dir,
        str(manifest.weights["path"]),
        name="weights.path",
    )
    expected = manifest.weights.get("checksum")
    if isinstance(expected, str) and expected:
        actual = file_sha256(weights_path)
        if actual != expected:
            raise XQTArtifactError(
                "quant pair weights checksum mismatch: "
                f"expected {expected}, got {actual}"
            )
    weights_format = str(manifest.weights.get("format", ""))
    if weights_format != WEIGHTS_FORMAT_TORCH_STATE_DICT:
        raise XQTArtifactError(
            "load_quant_pair currently supports format="
            f"{WEIGHTS_FORMAT_TORCH_STATE_DICT!r}, got {weights_format!r}"
        )
    return LoadedQuantPair(
        pair_dir=pair_dir.resolve(),
        sidecar_path=sidecar_path,
        weights_path=weights_path,
        manifest=manifest,
        compute_config=compute_config_from_mapping(manifest.compute_config),
    )


def load_quant_pair_into_model(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> QuantizedModel:
    """Load state_dict into a quantized module shell; never re-quantizes."""

    loaded = load_quant_pair(path)
    try:
        state = torch.load(
            loaded.weights_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(loaded.weights_path, map_location=map_location)
    if not isinstance(state, Mapping):
        raise XQTArtifactError(
            f"quant pair weights must be a state_dict mapping: {loaded.weights_path}"
        )
    model.load_state_dict(dict(state), strict=strict)
    return loaded.to_quantized_model(model)


__all__ = [
    "DEFAULT_SIDECAR_NAME",
    "DEFAULT_WEIGHTS_NAME",
    "LoadedQuantPair",
    "QUANT_SIDECAR_ARTIFACT_TYPE",
    "QUANT_SIDECAR_SCHEMA_VERSION",
    "QuantPairManifest",
    "WEIGHTS_FORMAT_TORCH_STATE_DICT",
    "load_quant_pair",
    "load_quant_pair_into_model",
    "write_quant_pair",
    "write_quant_pair_from_quantized",
]
