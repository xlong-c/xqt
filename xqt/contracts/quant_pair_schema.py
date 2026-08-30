"""Schema types for the flat quant pair (weights + quant.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from torch import nn

from xqt.contracts.compute import ComputeConfig
from xqt.contracts.quantized import QuantizedModel
from xqt.contracts.runtime_manifest import RUNTIME_MANIFEST_KEY
from xqt.contracts.runtime_quant import (
    RUNTIME_QUANT_CONTRACT_KEY,
    extract_runtime_quant_contract,
)
from xqt.core.base import XQTArtifactError

QUANT_SIDECAR_SCHEMA_VERSION = "1.0"
QUANT_SIDECAR_ARTIFACT_TYPE = "xqt_quant_sidecar"
DEFAULT_WEIGHTS_NAME = "model.pt"
DEFAULT_SIDECAR_NAME = "quant.json"
WEIGHTS_FORMAT_TORCH_STATE_DICT = "torch_state_dict"


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass
class QuantPairManifest:
    """Parsed quant.json sidecar (Infer-facing, not a quant recipe)."""

    schema_version: str = QUANT_SIDECAR_SCHEMA_VERSION
    artifact_type: str = QUANT_SIDECAR_ARTIFACT_TYPE
    weights: dict[str, Any] = field(default_factory=dict)
    compute_config: dict[str, Any] | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "weights": dict(self.weights),
            "lineage": dict(self.lineage),
            "metadata": dict(self.metadata),
        }
        if self.compute_config is not None:
            payload["compute_config"] = dict(self.compute_config)
        return payload

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuantPairManifest":
        if payload.get("artifact_type") != QUANT_SIDECAR_ARTIFACT_TYPE:
            raise XQTArtifactError(
                "quant.json has unsupported artifact_type: "
                f"{payload.get('artifact_type')!r}"
            )
        if str(payload.get("schema_version")) != QUANT_SIDECAR_SCHEMA_VERSION:
            raise XQTArtifactError(
                "quant.json has unsupported schema_version: "
                f"{payload.get('schema_version')!r}"
            )
        weights = payload.get("weights")
        if not isinstance(weights, Mapping):
            raise XQTArtifactError("quant.json.weights is required")
        weights_path = weights.get("path")
        weights_format = weights.get("format")
        if not isinstance(weights_path, str) or not weights_path:
            raise XQTArtifactError("quant.json.weights.path must be a non-empty string")
        if not isinstance(weights_format, str) or not weights_format:
            raise XQTArtifactError(
                "quant.json.weights.format must be a non-empty string"
            )
        raw_compute = payload.get("compute_config")
        if raw_compute is None:
            compute_config: dict[str, Any] | None = None
        elif isinstance(raw_compute, Mapping):
            compute_config = dict(raw_compute)
        else:
            raise XQTArtifactError("quant.json.compute_config must be a JSON object")
        lineage = payload.get("lineage", {})
        metadata = payload.get("metadata", {})
        if not isinstance(lineage, Mapping):
            raise XQTArtifactError("quant.json.lineage must be a JSON object")
        if not isinstance(metadata, Mapping):
            raise XQTArtifactError("quant.json.metadata must be a JSON object")
        return cls(
            schema_version=str(payload["schema_version"]),
            artifact_type=str(payload["artifact_type"]),
            weights={str(key): value for key, value in weights.items()},
            compute_config=compute_config,
            lineage=dict(lineage),
            metadata=dict(metadata),
        )


@dataclass
class LoadedQuantPair:
    """Resolved weights + sidecar view used by Infer consumers."""

    pair_dir: Path
    sidecar_path: Path
    weights_path: Path
    manifest: QuantPairManifest
    compute_config: ComputeConfig | None = None

    def resolve_runtime_quant_contract(self):
        """Parse RuntimeQuantContract from sidecar metadata, if present."""

        return extract_runtime_quant_contract(self.manifest.metadata)

    def to_quantized_model(
        self,
        model: nn.Module,
        *,
        backend: str = "unknown",
        method: str | None = None,
        strategy: str | None = None,
    ) -> QuantizedModel:
        lineage = self.manifest.lineage
        pair_meta = dict(self.manifest.metadata)
        metadata: dict[str, Any] = {
            "quant_pair_dir": str(self.pair_dir),
            "weights_path": str(self.weights_path),
            "sidecar_path": str(self.sidecar_path),
            "lineage": dict(self.manifest.lineage),
            "pair_metadata": pair_meta,
        }
        raw_contract = pair_meta.get(RUNTIME_QUANT_CONTRACT_KEY)
        if raw_contract is not None:
            metadata[RUNTIME_QUANT_CONTRACT_KEY] = raw_contract
        raw_manifest = pair_meta.get(RUNTIME_MANIFEST_KEY)
        if raw_manifest is not None:
            metadata[RUNTIME_MANIFEST_KEY] = raw_manifest
        raw_layout = pair_meta.get("layout_kernel")
        if isinstance(raw_layout, dict):
            metadata["layout_kernel"] = raw_layout
        return QuantizedModel(
            model=model,
            backend=str(lineage.get("backend") or backend or "unknown"),
            method=optional_str(
                method if method is not None else lineage.get("method")
            ),
            strategy=optional_str(
                strategy if strategy is not None else lineage.get("strategy")
            ),
            compute_config=self.compute_config,
            metadata=metadata,
        )


__all__ = [
    "DEFAULT_SIDECAR_NAME",
    "DEFAULT_WEIGHTS_NAME",
    "LoadedQuantPair",
    "QUANT_SIDECAR_ARTIFACT_TYPE",
    "QUANT_SIDECAR_SCHEMA_VERSION",
    "QuantPairManifest",
    "WEIGHTS_FORMAT_TORCH_STATE_DICT",
    "optional_str",
]
