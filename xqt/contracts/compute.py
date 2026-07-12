"""Compute-precision and compute-config contracts shared by quant and runtime.

Pure types, constants, protocols, and normalizers.
No quantizer / calibration / policy-apply logic.

Infer handoff (DEBT-003):
  Infer consumes model + optional ComputeConfig.
  required_capabilities is first-class; required_engine is not a schema key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

SUPPORTED_COMPUTE_PRECISIONS: frozenset[str] = frozenset(
    {"w4a4", "w4a16", "w8a8", "bf16"}
)

# Compute / MMA contracts (axis 3). Extensible; not tied to quant method names.
SUPPORTED_COMPUTE_CONTRACTS: frozenset[str] = frozenset(
    {
        "int8_mma",
        "fp4_mma",
        "int4_mma",
        "fp16_mma",
        "w4_storage_int8_mma",
        "mix_fp4_int8_mma",
        "generic",
    }
)

COMPUTE_CONFIG_SCHEMA_VERSION = "1.0"

# Forbidden as primary keys on compute_config (loader strips / ignores).
_FORBIDDEN_ENGINE_PRIMARY_KEYS: frozenset[str] = frozenset(
    {"required_engine", "engine", "force_engine"}
)


@runtime_checkable
class SupportsComputePrecision(Protocol):
    """Minimal runtime contract for mixed-precision quantized modules."""

    compute_precision: str

    def set_compute_precision(self, precision: str) -> None:
        """Switch runtime compute precision in place."""


@runtime_checkable
class SupportsPackedWeightDequant(Protocol):
    """Duck-type storage protocol for pre-export dense materialization (G6)."""

    input_features: int
    output_features: int

    def dequantize_weight(self) -> Any:
        """Return dense weight tensor equivalent to packed storage."""


def normalize_compute_precision(precision: str) -> str:
    """Canonicalize one compute-precision name.

    Accepted aliases: ``"fp16"``, ``"float16"``, ``"bfloat16"`` → ``"bf16"``;
    ``"int4"`` → ``"w4a4"``; ``"w4"``, ``"weight_only_4bit"`` → ``"w4a16"``;
    ``"int8"``, ``"w8"`` → ``"w8a8"``.
    """
    normalized = str(precision).strip().lower()
    aliases: dict[str, str] = {
        "fp16": "bf16",
        "float16": "bf16",
        "bfloat16": "bf16",
        "int4": "w4a4",
        "w4": "w4a16",
        "weight_only_4bit": "w4a16",
        "int8": "w8a8",
        "w8": "w8a8",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in SUPPORTED_COMPUTE_PRECISIONS:
        allowed = ", ".join(sorted(SUPPORTED_COMPUTE_PRECISIONS))
        raise ValueError(
            f"compute_precision must be one of {allowed}; got {precision!r}"
        )
    return resolved


def normalize_compute_contract(contract: str | None) -> str | None:
    """Canonicalize one compute/MMA contract name, or None if unset."""
    if contract is None:
        return None
    normalized = str(contract).strip().lower()
    if not normalized:
        return None
    aliases: dict[str, str] = {
        "w8a8_mma": "int8_mma",
        "dynamic_int8_mma": "int8_mma",
        "true_int8_mma": "int8_mma",
        "nvfp4_mma": "fp4_mma",
        "fp4": "fp4_mma",
        "int4": "int4_mma",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in SUPPORTED_COMPUTE_CONTRACTS:
        allowed = ", ".join(sorted(SUPPORTED_COMPUTE_CONTRACTS))
        raise ValueError(
            f"compute_contract must be one of {allowed}; got {contract!r}"
        )
    return resolved


def _normalize_capability_list(values: Sequence[Any] | None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _normalize_preferred_engines(values: Sequence[Any] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        name = str(item).strip().lower()
        if not name or name in seen or name == "auto":
            continue
        seen.add(name)
        result.append(name)
    return result


@dataclass(kw_only=True)
class ModuleComputeSpec:
    """Per-module compute contract for Infer handoff (not quant method identity)."""

    name: str
    compute_contract: str | None = None
    precision: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    preferred_engines: list[str] = field(default_factory=list)
    storage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.compute_contract is not None:
            payload["compute_contract"] = self.compute_contract
        if self.precision is not None:
            payload["precision"] = self.precision
        if self.required_capabilities:
            payload["required_capabilities"] = list(self.required_capabilities)
        if self.preferred_engines:
            payload["preferred_engines"] = list(self.preferred_engines)
        if self.storage:
            payload["storage"] = dict(self.storage)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModuleComputeSpec":
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("ModuleComputeSpec requires non-empty name")
        forbidden_hits = [
            key for key in _FORBIDDEN_ENGINE_PRIMARY_KEYS if key in payload
        ]
        meta = dict(payload.get("metadata") or {})
        if forbidden_hits:
            meta.setdefault(
                "ignored_forbidden_engine_keys",
                [str(item) for item in forbidden_hits],
            )
        raw_caps = payload.get("required_capabilities")
        if raw_caps is None and isinstance(payload.get("capabilities"), list):
            raw_caps = payload.get("capabilities")
        raw_preferred = payload.get("preferred_engines")
        if raw_preferred is None and isinstance(payload.get("engines_hint"), list):
            raw_preferred = payload.get("engines_hint")
        precision_raw = payload.get("precision")
        storage_raw = payload.get("storage")
        return cls(
            name=name,
            compute_contract=normalize_compute_contract(
                None
                if payload.get("compute_contract") is None
                else str(payload.get("compute_contract"))
            ),
            precision=(
                normalize_compute_precision(str(precision_raw))
                if precision_raw is not None
                else None
            ),
            required_capabilities=_normalize_capability_list(
                raw_caps
                if isinstance(raw_caps, Sequence) and not isinstance(raw_caps, (str, bytes))
                else None
            ),
            preferred_engines=_normalize_preferred_engines(
                raw_preferred
                if isinstance(raw_preferred, Sequence)
                and not isinstance(raw_preferred, (str, bytes))
                else None
            ),
            storage=dict(storage_raw) if isinstance(storage_raw, Mapping) else {},
            metadata=meta,
        )


@dataclass(kw_only=True)
class ComputeConfig:
    """Optional Infer-side compute configuration (model handoff companion).

    Does not carry quant method (awq/gptq/...) as primary keys.
    Does not use required_engine as a schema primary field.
    """

    schema_version: str = COMPUTE_CONFIG_SCHEMA_VERSION
    default_precision: str | None = None
    modules: list[ModuleComputeSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "modules": [module.to_dict() for module in self.modules],
        }
        if self.default_precision is not None:
            payload["default_precision"] = self.default_precision
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def all_required_capabilities(self) -> list[str]:
        """Union of module-level required_capabilities, stable order."""
        seen: set[str] = set()
        result: list[str] = []
        for module in self.modules:
            for cap in module.required_capabilities:
                if cap not in seen:
                    seen.add(cap)
                    result.append(cap)
        return result

    def precision_overrides(self) -> list[dict[str, str]]:
        """Project to HybridInferenceEngine-style precision_overrides records."""
        overrides: list[dict[str, str]] = []
        for module in self.modules:
            if module.precision is None:
                continue
            overrides.append({"module": module.name, "precision": module.precision})
        return overrides

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ComputeConfig | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("compute_config must be a mapping")
        forbidden_hits = [
            key for key in _FORBIDDEN_ENGINE_PRIMARY_KEYS if key in payload
        ]
        meta = dict(payload.get("metadata") or {})
        if forbidden_hits:
            meta.setdefault(
                "ignored_forbidden_engine_keys",
                [str(item) for item in forbidden_hits],
            )
        raw_modules = payload.get("modules", [])
        modules: list[ModuleComputeSpec] = []
        if isinstance(raw_modules, list):
            for item in raw_modules:
                if isinstance(item, Mapping):
                    modules.append(ModuleComputeSpec.from_mapping(item))
        default_precision = payload.get("default_precision")
        version = str(payload.get("schema_version") or COMPUTE_CONFIG_SCHEMA_VERSION)
        return cls(
            schema_version=version,
            default_precision=(
                normalize_compute_precision(str(default_precision))
                if default_precision is not None
                else None
            ),
            modules=modules,
            metadata=meta,
        )

    @classmethod
    def from_modules(
        cls,
        *,
        module_names: Sequence[str],
        compute_contract: str,
        precision: str | None = None,
        required_capabilities: Sequence[str] | None = None,
        preferred_engines: Sequence[str] | None = None,
        default_precision: str | None = None,
        storage: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ComputeConfig":
        """Build a uniform compute_config for a list of quantized modules."""
        contract = normalize_compute_contract(compute_contract)
        if contract is None:
            raise ValueError("compute_contract is required")
        caps = _normalize_capability_list(required_capabilities)
        if not caps:
            caps = [contract]
        preferred = _normalize_preferred_engines(preferred_engines)
        resolved_precision = (
            normalize_compute_precision(precision) if precision is not None else None
        )
        storage_dict = dict(storage or {})
        modules = [
            ModuleComputeSpec(
                name=str(name),
                compute_contract=contract,
                precision=resolved_precision,
                required_capabilities=list(caps),
                preferred_engines=list(preferred),
                storage=dict(storage_dict),
            )
            for name in module_names
            if str(name)
        ]
        return cls(
            default_precision=(
                normalize_compute_precision(default_precision)
                if default_precision is not None
                else resolved_precision
            ),
            modules=modules,
            metadata=dict(metadata or {}),
        )


def compute_config_from_mapping(
    payload: Mapping[str, Any] | None,
) -> ComputeConfig | None:
    """Public alias for ComputeConfig.from_mapping."""
    return ComputeConfig.from_mapping(payload)


def compute_config_to_dict(
    config: ComputeConfig | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Serialize ComputeConfig or mapping to a JSON-safe dict."""
    if config is None:
        return None
    if isinstance(config, ComputeConfig):
        return config.to_dict()
    if isinstance(config, Mapping):
        parsed = ComputeConfig.from_mapping(config)
        return None if parsed is None else parsed.to_dict()
    raise TypeError(f"unsupported compute_config type: {type(config)!r}")


__all__ = [
    "COMPUTE_CONFIG_SCHEMA_VERSION",
    "ComputeConfig",
    "ModuleComputeSpec",
    "SUPPORTED_COMPUTE_CONTRACTS",
    "SUPPORTED_COMPUTE_PRECISIONS",
    "SupportsComputePrecision",
    "SupportsPackedWeightDequant",
    "compute_config_from_mapping",
    "compute_config_to_dict",
    "normalize_compute_contract",
    "normalize_compute_precision",
]
