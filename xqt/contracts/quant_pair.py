"""Flat Infer delivery: one weights file + quant.json sidecar.

Consumes already-quantized storage + optional compute_config.
Never runs quantizers, calibration, or sensitivity analysis.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import torch
from torch import nn

from xqt.contracts.compute import (
    ComputeConfig,
    compute_config_from_mapping,
    compute_config_to_dict,
)
from xqt.contracts.quantized import QuantizedModel
from xqt.contracts.runtime_quant import (
    RUNTIME_QUANT_CONTRACT_KEY,
    RuntimeQuantContract,
    attach_runtime_quant_contract,
    extract_runtime_quant_contract,
)
from xqt.core.base import XQTArtifactError, file_sha256, json_safe_value, utc_timestamp

from .quant_pair_schema import (
    DEFAULT_SAFETENSORS_NAME,
    DEFAULT_SIDECAR_NAME,
    DEFAULT_WEIGHTS_NAME,
    QUANT_SIDECAR_ARTIFACT_TYPE,
    QUANT_SIDECAR_SCHEMA_VERSION,
    SUPPORTED_WEIGHTS_FORMATS,
    WEIGHTS_FORMAT_SAFETENSORS,
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


def _validate_pair_relative_path(relative_path: str, *, name: str) -> Path:
    """Reject paths that cannot safely live inside one quant-pair directory."""

    candidate = Path(relative_path)
    if not str(relative_path).strip() or candidate.is_absolute():
        raise XQTArtifactError(f"{name} must be a non-empty relative path")
    if any(part == ".." for part in candidate.parts):
        raise XQTArtifactError(f"{name} must not contain parent traversal: {relative_path}")
    if candidate.name in {"", "."}:
        raise XQTArtifactError(f"{name} must name a file: {relative_path}")
    return candidate


def _write_pair_weights(
    model: nn.Module,
    weights_path: Path,
    *,
    weights_format: str,
) -> None:
    """Write the weight payload into an already isolated staging directory."""

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_format == WEIGHTS_FORMAT_SAFETENSORS:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise XQTArtifactError(
                "saving .safetensors requires the safetensors package"
            ) from exc
        state = {
            str(key): (value.detach().contiguous() if isinstance(value, torch.Tensor) else value)
            for key, value in model.state_dict().items()
        }
        save_file(state, str(weights_path))
        return
    torch.save(model.state_dict(), weights_path)


def _publish_staged_pair(staging_dir: Path, pair_dir: Path) -> None:
    """Atomically promote a verified staged directory, retaining old output on error."""

    parent = pair_dir.parent
    backup_dir: Path | None = None
    try:
        if pair_dir.exists():
            backup_dir = parent / f".{pair_dir.name}.backup_{uuid4().hex}"
            os.replace(pair_dir, backup_dir)
        os.replace(staging_dir, pair_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not pair_dir.exists():
            os.replace(backup_dir, pair_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def _resolve_sidecar_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        sidecar = candidate / DEFAULT_SIDECAR_NAME
    elif candidate.name == DEFAULT_SIDECAR_NAME or candidate.suffix == ".json":
        sidecar = candidate
    elif candidate.is_file() and candidate.suffix in (".safetensors", ".pt", ".bin"):
        sidecar = candidate.parent / DEFAULT_SIDECAR_NAME
    else:
        raise XQTArtifactError(
            "load_quant_pair expects a directory containing quant.json, "
            "a quant.json path, or a paired weights file"
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


def _attach_runtime_manifest_if_useful(meta: dict[str, Any]) -> dict[str, Any]:
    """Persist RuntimeManifest when contract or layout diagnostics are present (V1)."""

    from xqt.contracts.runtime_manifest import (
        RUNTIME_MANIFEST_KEY,
        build_runtime_manifest,
    )

    if isinstance(meta.get(RUNTIME_MANIFEST_KEY), Mapping):
        return meta
    has_contract = extract_runtime_quant_contract(meta) is not None
    has_layout = isinstance(meta.get("layout_kernel"), Mapping) or isinstance(
        meta.get("layout_reports"), (list, tuple)
    )
    if not has_contract and not has_layout:
        return meta
    manifest = build_runtime_manifest(meta)
    out = dict(meta)
    out[RUNTIME_MANIFEST_KEY] = manifest.to_dict()
    return out


def _merge_pair_metadata(
    base: Mapping[str, Any] | None,
    *,
    runtime_contract: RuntimeQuantContract | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build sidecar metadata; promote RuntimeQuantContract under the canonical key."""

    meta = _json_mapping(base, name="metadata")
    meta.setdefault("producer", "xqt")
    meta.setdefault("created_at", utc_timestamp())
    if runtime_contract is None:
        return _attach_runtime_manifest_if_useful(meta)
    if isinstance(runtime_contract, RuntimeQuantContract):
        return _attach_runtime_manifest_if_useful(
            attach_runtime_quant_contract(meta, runtime_contract)
        )
    if isinstance(runtime_contract, Mapping):
        parsed = RuntimeQuantContract.from_dict(runtime_contract)
        return _attach_runtime_manifest_if_useful(
            attach_runtime_quant_contract(meta, parsed)
        )
    raise XQTArtifactError(
        "runtime_quant_contract must be RuntimeQuantContract or mapping; "
        f"got {type(runtime_contract).__name__}"
    )


def write_quant_pair(
    model: nn.Module,
    output_dir: str | Path,
    *,
    compute_config: ComputeConfig | Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    runtime_quant_contract: RuntimeQuantContract | Mapping[str, Any] | None = None,
    weights_name: str | None = None,
    sidecar_name: str = DEFAULT_SIDECAR_NAME,
    weights_format: str | None = None,
) -> Path:
    """Write model weights (safetensors or .pt) + quant.json. Model must already hold quantized storage.

    When ``runtime_quant_contract`` is set (or present inside ``metadata``), it is
    stored under ``runtime_quant_contract`` as the internal fact source. HF
    ``quantization_config`` is never invented here.
    """

    if weights_format is not None and weights_format not in SUPPORTED_WEIGHTS_FORMATS:
        raise XQTArtifactError(
            f"unsupported weights_format={weights_format!r}; "
            f"supported formats are {SUPPORTED_WEIGHTS_FORMATS}"
        )

    if weights_format == WEIGHTS_FORMAT_SAFETENSORS:
        resolved_weights_name = weights_name or DEFAULT_SAFETENSORS_NAME
        resolved_format = WEIGHTS_FORMAT_SAFETENSORS
    elif weights_format == WEIGHTS_FORMAT_TORCH_STATE_DICT:
        resolved_weights_name = weights_name or DEFAULT_WEIGHTS_NAME
        resolved_format = WEIGHTS_FORMAT_TORCH_STATE_DICT
    else:
        if weights_name is None:
            resolved_weights_name = DEFAULT_WEIGHTS_NAME
            resolved_format = WEIGHTS_FORMAT_TORCH_STATE_DICT
        elif weights_name.endswith(".safetensors"):
            resolved_weights_name = weights_name
            resolved_format = WEIGHTS_FORMAT_SAFETENSORS
        else:
            resolved_weights_name = weights_name
            resolved_format = WEIGHTS_FORMAT_TORCH_STATE_DICT

    weights_relative = _validate_pair_relative_path(
        resolved_weights_name,
        name="weights_name",
    )
    sidecar_relative = _validate_pair_relative_path(
        sidecar_name,
        name="sidecar_name",
    )
    pair_dir = Path(output_dir)
    if pair_dir.exists() and pair_dir.is_symlink():
        raise XQTArtifactError("output_dir must not be a symbolic link")
    if not pair_dir.name:
        raise XQTArtifactError("output_dir must name a quant-pair directory")
    pair_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = pair_dir.parent / f".{pair_dir.name}.staging_{uuid4().hex}"
    staging_dir.mkdir()
    try:
        weights_path = staging_dir / weights_relative
        _write_pair_weights(model, weights_path, weights_format=resolved_format)
        contract = runtime_quant_contract
        if contract is None and metadata is not None:
            contract = extract_runtime_quant_contract(metadata)
        meta = _merge_pair_metadata(metadata, runtime_contract=contract)
        manifest = QuantPairManifest(
            weights={
                "path": str(weights_relative),
                "format": resolved_format,
                "checksum": file_sha256(weights_path),
            },
            compute_config=_resolve_compute_config_dict(compute_config),
            lineage=_json_mapping(lineage, name="lineage"),
            metadata=meta,
        )
        sidecar_path = staging_dir / sidecar_relative
        manifest.write_json(sidecar_path)
        verified = QuantPairManifest.from_dict(
            _load_json_mapping(sidecar_path, name="quant.json")
        )
        verified_weights = _resolve_pair_file(
            staging_dir,
            str(verified.weights["path"]),
            name="weights.path",
        )
        if file_sha256(verified_weights) != verified.weights["checksum"]:
            raise XQTArtifactError("staged quant pair checksum verification failed")
        _publish_staged_pair(staging_dir, pair_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return pair_dir


def write_quant_pair_from_quantized(
    quantized: QuantizedModel,
    output_dir: str | Path,
    *,
    weights_name: str | None = None,
    sidecar_name: str = DEFAULT_SIDECAR_NAME,
    metadata: Mapping[str, Any] | None = None,
    weights_format: str | None = None,
) -> Path:
    """Write a quant pair from QuantizedModel via infer_handoff() + contract."""

    handoff = quantized.infer_handoff()
    model = handoff["model"]
    if not isinstance(model, nn.Module):
        raise XQTArtifactError(
            "write_quant_pair_from_quantized requires QuantizedModel.model "
            "to be a torch.nn.Module"
        )
    merged: dict[str, Any] = {}
    if isinstance(quantized.metadata, Mapping):
        merged.update(dict(quantized.metadata))
    if metadata is not None:
        merged.update(dict(metadata))
    contract = quantized.resolve_runtime_quant_contract()
    if contract is None:
        contract = extract_runtime_quant_contract(merged)
    merged.pop(RUNTIME_QUANT_CONTRACT_KEY, None)
    return write_quant_pair(
        model,
        output_dir,
        compute_config=handoff.get("compute_config"),
        lineage={
            "backend": quantized.backend,
            "method": quantized.method,
            "strategy": quantized.strategy,
        },
        metadata=merged,
        runtime_quant_contract=contract,
        weights_name=weights_name,
        sidecar_name=sidecar_name,
        weights_format=weights_format,
    )


def load_quant_pair(path: str | Path) -> LoadedQuantPair:
    """Load and validate one weights + quant.json pair (no model mutation).

    ``LoadedQuantPair.resolve_runtime_quant_contract()`` returns the sidecar
    contract when present, otherwise ``None`` (honest absence; no HF invent).
    """

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
    if weights_format not in SUPPORTED_WEIGHTS_FORMATS:
        raise XQTArtifactError(
            f"load_quant_pair currently supports formats {SUPPORTED_WEIGHTS_FORMATS!r}, "
            f"got {weights_format!r}"
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
    weights_format = str(loaded.manifest.weights.get("format", ""))

    if weights_format == WEIGHTS_FORMAT_SAFETENSORS:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise XQTArtifactError(
                "loading .safetensors requires the safetensors package"
            ) from exc
        raw_state = load_file(str(loaded.weights_path))
        target_device = (
            torch.device(map_location)
            if isinstance(map_location, str)
            else map_location
        )
        state = {k: v.to(target_device) for k, v in raw_state.items()}
    elif weights_format == WEIGHTS_FORMAT_TORCH_STATE_DICT:
        try:
            state = torch.load(
                loaded.weights_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(loaded.weights_path, map_location=map_location)
    else:
        raise XQTArtifactError(
            f"unsupported weights format in quant pair: {weights_format!r}"
        )

    if not isinstance(state, Mapping):
        raise XQTArtifactError(
            f"quant pair weights must be a state_dict mapping: {loaded.weights_path}"
        )
    model.load_state_dict(dict(state), strict=strict)
    return loaded.to_quantized_model(model)


__all__ = [
    "DEFAULT_SAFETENSORS_NAME",
    "DEFAULT_SIDECAR_NAME",
    "DEFAULT_WEIGHTS_NAME",
    "LoadedQuantPair",
    "QUANT_SIDECAR_ARTIFACT_TYPE",
    "QUANT_SIDECAR_SCHEMA_VERSION",
    "QuantPairManifest",
    "SUPPORTED_WEIGHTS_FORMATS",
    "WEIGHTS_FORMAT_SAFETENSORS",
    "WEIGHTS_FORMAT_TORCH_STATE_DICT",
    "load_quant_pair",
    "load_quant_pair_into_model",
    "write_quant_pair",
    "write_quant_pair_from_quantized",
]
