"""Model-side inference input/output contract schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from xqt.core.base import XQTConfigError

INFERENCE_CONTRACT_SCHEMA_VERSION = "1.0"
DEFAULT_INFERENCE_ADAPTER = "tensor"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise XQTConfigError(
            f"InferenceContract.{name} must be a JSON object; "
            f"got {type(value).__name__}"
        )
    invalid_keys = [key for key in value if not isinstance(key, str)]
    if invalid_keys:
        raise XQTConfigError(
            f"InferenceContract.{name} keys must be strings; got {invalid_keys[0]!r}"
        )
    return dict(value)


def _entries(value: Any, *, name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise XQTConfigError(
            f"InferenceContract.{name} must be a sequence; got {type(value).__name__}"
        )
    parsed: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise XQTConfigError(
                f"InferenceContract.{name}[{index}] must be a JSON object"
            )
        invalid_keys = [key for key in item if not isinstance(key, str)]
        if invalid_keys:
            raise XQTConfigError(
                f"InferenceContract.{name}[{index}] keys must be strings; "
                f"got {invalid_keys[0]!r}"
            )
        entry = dict(item)
        raw_name = entry.get("name")
        if raw_name is not None:
            if not isinstance(raw_name, str) or not raw_name:
                raise XQTConfigError(
                    f"InferenceContract.{name}[{index}].name must be a non-empty string"
                )
            if raw_name in names:
                raise XQTConfigError(
                    f"InferenceContract.{name} contains duplicate name {raw_name!r}"
                )
            names.add(raw_name)
        raw_semantic = entry.get("semantic")
        if raw_semantic is not None and (
            not isinstance(raw_semantic, str) or not raw_semantic
        ):
            raise XQTConfigError(
                f"InferenceContract.{name}[{index}].semantic must be a non-empty string"
            )
        parsed.append(entry)
    return tuple(parsed)


@dataclass
class InferenceContractConfig:
    """Mutable OmegaConf-facing schema for one model inference contract."""

    schema_version: str = INFERENCE_CONTRACT_SCHEMA_VERSION
    family: str = "unknown"
    adapter: str = DEFAULT_INFERENCE_ADAPTER
    adapter_version: str = "1"
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "inputs": [dict(entry) for entry in self.inputs],
            "outputs": [dict(entry) for entry in self.outputs],
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }

    def to_contract(
        self,
        *,
        io: Mapping[str, Any] | None = None,
    ) -> "InferenceContract":
        payload = self.to_dict()
        if not self.inputs:
            payload.pop("inputs", None)
        if not self.outputs:
            payload.pop("outputs", None)
        return InferenceContract.from_dict(payload, io=io)


@dataclass(frozen=True, slots=True)
class InferenceContract:
    """Stable semantic contract between a model package and an adapter."""

    schema_version: str = INFERENCE_CONTRACT_SCHEMA_VERSION
    family: str = "unknown"
    adapter: str = DEFAULT_INFERENCE_ADAPTER
    adapter_version: str = "1"
    inputs: tuple[dict[str, Any], ...] = ()
    outputs: tuple[dict[str, Any], ...] = ()
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(
            str(entry["name"])
            for entry in self.inputs
            if isinstance(entry.get("name"), str)
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(
            str(entry["name"])
            for entry in self.outputs
            if isinstance(entry.get("name"), str)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "inputs": [dict(entry) for entry in self.inputs],
            "outputs": [dict(entry) for entry in self.outputs],
            "config": dict(self.config or {}),
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | InferenceContractConfig | None,
        *,
        io: Mapping[str, Any] | None = None,
    ) -> "InferenceContract":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, InferenceContractConfig):
            return payload.to_contract(io=io)
        raw = {} if payload is None else payload
        if not isinstance(raw, Mapping):
            raise XQTConfigError(
                f"InferenceContract expects a JSON object; got {type(raw).__name__}"
            )
        version = raw.get("schema_version", INFERENCE_CONTRACT_SCHEMA_VERSION)
        if not isinstance(version, str) or not version:
            raise XQTConfigError(
                "InferenceContract.schema_version must be a non-empty string"
            )
        if version != INFERENCE_CONTRACT_SCHEMA_VERSION:
            raise XQTConfigError(
                f"InferenceContract has unsupported schema_version: {version!r}"
            )
        family = raw.get("family", "unknown")
        adapter = raw.get("adapter", DEFAULT_INFERENCE_ADAPTER)
        adapter_version = raw.get("adapter_version", "1")
        for name, value in (
            ("family", family),
            ("adapter", adapter),
            ("adapter_version", adapter_version),
        ):
            if not isinstance(value, str) or not value:
                raise XQTConfigError(
                    f"InferenceContract.{name} must be a non-empty string"
                )
        fallback_io = io if isinstance(io, Mapping) else {}
        inputs = raw.get("inputs", fallback_io.get("inputs"))
        outputs = raw.get("outputs", fallback_io.get("outputs"))
        return cls(
            schema_version=version,
            family=family,
            adapter=adapter,
            adapter_version=adapter_version,
            inputs=_entries(inputs, name="inputs"),
            outputs=_entries(outputs, name="outputs"),
            config=_mapping(raw.get("config"), name="config"),
            metadata=_mapping(raw.get("metadata"), name="metadata"),
        )


__all__ = [
    "DEFAULT_INFERENCE_ADAPTER",
    "INFERENCE_CONTRACT_SCHEMA_VERSION",
    "InferenceContract",
    "InferenceContractConfig",
]
