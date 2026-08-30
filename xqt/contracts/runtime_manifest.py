"""Aggregated runtime manifest for quantized model / quant pair artifacts.

Internal fact source: scheme, layout reports, kernel selection, prefill/decode
flags. Optional HF/serving adapters may *derive* from this; they do not define it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from xqt.contracts.layout_kernel_report import LayoutKernelReport
from xqt.contracts.quantized import QuantizedModel
from xqt.contracts.runtime_quant import (
    RUNTIME_QUANT_CONTRACT_KEY,
    RuntimeQuantContract,
    extract_runtime_quant_contract,
)
from xqt.contracts.runtime_features import (
    RUNTIME_FEATURES_KEY,
    RuntimeFeatureMetadata,
)
from xqt.core.base import XQTConfigError

RUNTIME_MANIFEST_KEY = "runtime_manifest"
RUNTIME_MANIFEST_SCHEMA_VERSION = 1


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise XQTConfigError(
            "RuntimeManifest kernel fields must be a sequence of str"
        )
    return tuple(str(item) for item in value)


def _layout_from_any(value: Any) -> LayoutKernelReport | None:
    if value is None:
        return None
    if isinstance(value, LayoutKernelReport):
        return value
    if isinstance(value, Mapping):
        return LayoutKernelReport.from_dict(value)
    raise XQTConfigError(
        "layout_reports entries must be LayoutKernelReport or mapping; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """One artifact-level answer for scheme, layout, kernels, and phase flags."""

    contract: RuntimeQuantContract | None = None
    layout_reports: tuple[LayoutKernelReport, ...] = ()
    selected_kernels: tuple[str, ...] = ()
    fallback_kernels: tuple[str, ...] = ()
    prefill_supported: bool | None = None
    decode_supported: bool | None = None
    runtime_features: RuntimeFeatureMetadata | None = None
    notes: tuple[str, ...] = ()
    schema_version: int = RUNTIME_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        prefill = self.prefill_supported
        decode = self.decode_supported
        if self.contract is not None:
            if prefill is None:
                prefill = self.contract.prefill_supported
            if decode is None:
                decode = self.contract.decode_supported
        return {
            "schema_version": int(self.schema_version),
            "contract": None if self.contract is None else self.contract.to_dict(),
            "layout_reports": [item.to_dict() for item in self.layout_reports],
            "selected_kernels": list(self.selected_kernels),
            "fallback_kernels": list(self.fallback_kernels),
            "prefill_supported": prefill,
            "decode_supported": decode,
            "runtime_features": (
                None
                if self.runtime_features is None
                else self.runtime_features.to_dict()
            ),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeManifest:
        if not isinstance(payload, Mapping):
            raise XQTConfigError(
                "RuntimeManifest.from_dict expects a mapping; "
                f"got {type(payload).__name__}"
            )
        raw_contract = payload.get("contract")
        contract: RuntimeQuantContract | None
        if raw_contract is None:
            contract = None
        elif isinstance(raw_contract, RuntimeQuantContract):
            contract = raw_contract
        elif isinstance(raw_contract, Mapping):
            contract = RuntimeQuantContract.from_dict(raw_contract)
        else:
            raise XQTConfigError(
                "RuntimeManifest.contract must be mapping, RuntimeQuantContract, "
                f"or None; got {type(raw_contract).__name__}"
            )
        raw_layouts = payload.get("layout_reports", ())
        if raw_layouts is None:
            layouts: list[LayoutKernelReport] = []
        elif isinstance(raw_layouts, (list, tuple)):
            layouts = []
            for item in raw_layouts:
                parsed = _layout_from_any(item)
                if parsed is not None:
                    layouts.append(parsed)
        else:
            raise XQTConfigError("RuntimeManifest.layout_reports must be a sequence")
        prefill = payload.get("prefill_supported")
        decode = payload.get("decode_supported")
        if prefill is not None and not isinstance(prefill, bool):
            raise XQTConfigError("RuntimeManifest.prefill_supported must be bool or None")
        if decode is not None and not isinstance(decode, bool):
            raise XQTConfigError("RuntimeManifest.decode_supported must be bool or None")
        raw_features = payload.get("runtime_features")
        runtime_features: RuntimeFeatureMetadata | None
        if raw_features is None:
            runtime_features = None
        elif isinstance(raw_features, RuntimeFeatureMetadata):
            runtime_features = raw_features
        elif isinstance(raw_features, Mapping):
            runtime_features = RuntimeFeatureMetadata.from_dict(raw_features)
        else:
            raise XQTConfigError(
                "RuntimeManifest.runtime_features must be a mapping, "
                "RuntimeFeatureMetadata, or None"
            )
        version = payload.get("schema_version", RUNTIME_MANIFEST_SCHEMA_VERSION)
        try:
            version_i = int(version)
        except (TypeError, ValueError) as exc:
            raise XQTConfigError(
                f"RuntimeManifest.schema_version must be int; got {version!r}"
            ) from exc
        return cls(
            contract=contract,
            layout_reports=tuple(layouts),
            selected_kernels=_as_str_tuple(payload.get("selected_kernels")),
            fallback_kernels=_as_str_tuple(payload.get("fallback_kernels")),
            prefill_supported=prefill,
            decode_supported=decode,
            runtime_features=runtime_features,
            notes=_as_str_tuple(payload.get("notes")),
            schema_version=version_i,
        )


def _layouts_from_metadata(metadata: Mapping[str, Any]) -> list[LayoutKernelReport]:
    out: list[LayoutKernelReport] = []
    single = metadata.get("layout_kernel")
    parsed = _layout_from_any(single) if single is not None else None
    if parsed is not None:
        out.append(parsed)
    multi = metadata.get("layout_kernels") or metadata.get("layout_reports")
    if isinstance(multi, (list, tuple)):
        for item in multi:
            item_parsed = _layout_from_any(item)
            if item_parsed is not None:
                out.append(item_parsed)
    return out


def _kernels_from_layouts(
    layouts: Sequence[LayoutKernelReport],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected: list[str] = []
    fallback: list[str] = []
    for layout in layouts:
        if layout.selected_kernel:
            selected.append(str(layout.selected_kernel))
        if layout.fallback_reason:
            fallback.append(str(layout.fallback_reason))
    return tuple(dict.fromkeys(selected)), tuple(dict.fromkeys(fallback))


def build_runtime_manifest(
    source: QuantizedModel | Mapping[str, Any] | str | Any,
) -> RuntimeManifest:
    """Build RuntimeManifest from QuantizedModel, quant pair path, or metadata map.

    Accepts:
    - ``QuantizedModel``
    - quant pair directory / ``quant.json`` path (lazy import load_quant_pair)
    - raw metadata ``Mapping``
    """

    notes: list[str] = []
    contract: RuntimeQuantContract | None = None
    layouts: list[LayoutKernelReport] = []
    metadata: dict[str, Any] = {}

    if isinstance(source, QuantizedModel):
        contract = source.resolve_runtime_quant_contract()
        metadata = dict(source.metadata)
        layouts = _layouts_from_metadata(metadata)
        if contract is None:
            notes.append("runtime_quant_contract_absent")
    elif isinstance(source, Mapping):
        metadata = dict(source)
        contract = extract_runtime_quant_contract(metadata)
        if contract is None and isinstance(metadata.get("contract"), Mapping):
            contract = RuntimeQuantContract.from_dict(metadata["contract"])
        layouts = _layouts_from_metadata(metadata)
        raw_manifest = metadata.get(RUNTIME_MANIFEST_KEY)
        if isinstance(raw_manifest, Mapping) and contract is None:
            return RuntimeManifest.from_dict(raw_manifest)
        if contract is None:
            notes.append("runtime_quant_contract_absent")
    else:
        from pathlib import Path

        from xqt.contracts.quant_pair import load_quant_pair

        loaded = load_quant_pair(Path(source) if not isinstance(source, Path) else source)
        contract = loaded.resolve_runtime_quant_contract()
        metadata = dict(loaded.manifest.metadata)
        layouts = _layouts_from_metadata(metadata)
        if contract is None:
            notes.append("runtime_quant_contract_absent")

    selected, fallback = _kernels_from_layouts(layouts)
    if contract is not None and not selected and contract.required_kernels:
        selected = tuple(contract.required_kernels)
    prefill = None if contract is None else contract.prefill_supported
    decode = None if contract is None else contract.decode_supported
    runtime_features: RuntimeFeatureMetadata | None = None
    raw_features = metadata.get(RUNTIME_FEATURES_KEY)
    if raw_features is not None:
        if isinstance(raw_features, RuntimeFeatureMetadata):
            runtime_features = raw_features
        elif isinstance(raw_features, Mapping):
            runtime_features = RuntimeFeatureMetadata.from_dict(raw_features)
        else:
            raise XQTConfigError(
                f"metadata[{RUNTIME_FEATURES_KEY!r}] must be a mapping or "
                "RuntimeFeatureMetadata"
            )
    return RuntimeManifest(
        contract=contract,
        layout_reports=tuple(layouts),
        selected_kernels=selected,
        fallback_kernels=fallback,
        prefill_supported=prefill,
        decode_supported=decode,
        runtime_features=runtime_features,
        notes=tuple(notes),
    )


__all__ = [
    "RUNTIME_MANIFEST_KEY",
    "RUNTIME_MANIFEST_SCHEMA_VERSION",
    "RuntimeManifest",
    "build_runtime_manifest",
]
