"""External quantized checkpoint probing (C4), aligned with vLLM load semantics.

Mirrors serving-engine contract without a serving runtime:

1. Multi-source config detection (conflict = hard error).
2. override_quantization_method-style alias / upgrade chain.
3. Canonical method for bridges
   (create_weights / load / process_weights_after_loading / apply).

First-batch formats: compressed_tensors, gptq, awq.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from xqt.core.errors import XQTArtifactError
from xqt.contracts.external_methods import (
    iter_config_filenames,
    list_supported_external_formats,
    normalize_external_format,
    override_external_format,
)
from xqt.contracts.external_types import ExternalQuantInfo


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _bits_from_config(quant_config: Mapping[str, Any]) -> int | None:
    bits = _as_int(quant_config.get("bits") or quant_config.get("weight_bits"))
    if bits is not None:
        return bits
    weights = quant_config.get("weights")
    if isinstance(weights, Mapping):
        return _as_int(weights.get("num_bits") or weights.get("bits"))
    groups = quant_config.get("config_groups")
    if isinstance(groups, Mapping):
        for group in groups.values():
            if not isinstance(group, Mapping):
                continue
            w = group.get("weights")
            if isinstance(w, Mapping):
                b = _as_int(w.get("num_bits") or w.get("bits"))
                if b is not None:
                    return b
    return None


def _group_size_from_config(quant_config: Mapping[str, Any]) -> int | None:
    raw = quant_config.get("group_size")
    if raw is None:
        raw = quant_config.get("groupsize")
    group_size = _as_int(raw)
    if group_size is not None:
        return group_size
    weights = quant_config.get("weights")
    if isinstance(weights, Mapping):
        return _as_int(weights.get("group_size") or weights.get("groupsize"))
    groups = quant_config.get("config_groups")
    if isinstance(groups, Mapping):
        for group in groups.values():
            if not isinstance(group, Mapping):
                continue
            w = group.get("weights")
            if isinstance(w, Mapping):
                g = _as_int(w.get("group_size") or w.get("groupsize"))
                if g is not None:
                    return g
    return None


def _infer_method_name(quant_config: Mapping[str, Any]) -> str | None:
    raw = (
        quant_config.get("quant_method")
        or quant_config.get("quantization_method")
        or quant_config.get("quant_algo")
        or quant_config.get("format")
        or ""
    )
    text = str(raw).strip().lower()
    if "compressed" in text or quant_config.get("config_groups"):
        return "compressed_tensors"
    if text in {"", "none", "null"}:
        if "bits" in quant_config or "group_size" in quant_config:
            if quant_config.get("zero_point") is True:
                return "awq"
            return "gptq"
        return None
    return normalize_external_format(text)


def _infer_from_quant_config(
    quant_config: Mapping[str, Any],
    *,
    source_file: str,
) -> ExternalQuantInfo | None:
    method = _infer_method_name(quant_config)
    if method is None:
        return None
    bits = _bits_from_config(quant_config)
    group_size = _group_size_from_config(quant_config)
    sym = _as_bool(quant_config.get("sym"))
    zero_point = _as_bool(quant_config.get("zero_point"))
    if sym is None and zero_point is not None:
        sym = not zero_point
    if zero_point is None and sym is not None:
        zero_point = not sym
    desc_act = _as_bool(quant_config.get("desc_act"))
    raw_method = (
        quant_config.get("quant_method")
        or quant_config.get("quantization_method")
        or quant_config.get("format")
    )
    return ExternalQuantInfo(
        format=method,
        source_file=source_file,
        raw_config=dict(quant_config),
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=desc_act,
        zero_point=zero_point,
        quant_method_raw=str(raw_method) if raw_method is not None else None,
        source_files=(source_file,),
    )


def _collect_findings(root: Path) -> list[ExternalQuantInfo]:
    findings: list[ExternalQuantInfo] = []
    config = _load_json(root / "config.json")
    if config is not None:
        for field_name in ("quantization_config", "compression_config"):
            body = config.get(field_name)
            if isinstance(body, Mapping) and body:
                info = _infer_from_quant_config(
                    body, source_file=f"config.json#{field_name}"
                )
                if info is not None:
                    findings.append(info)
    for filename in iter_config_filenames():
        payload = _load_json(root / filename)
        if payload is None:
            continue
        nested = payload.get("quantization_config")
        if isinstance(nested, Mapping) and nested:
            body: Mapping[str, Any] = nested
            source = f"{filename}#quantization_config"
        else:
            body = payload
            source = filename
        if not body:
            continue
        info = _infer_from_quant_config(body, source_file=source)
        if info is not None:
            findings.append(info)
    return findings


def probe_external_quant_config(model_path: str | Path) -> ExternalQuantInfo | None:
    """Probe HF-style quantization metadata under ``model_path``."""

    root = Path(model_path)
    if not root.exists():
        raise XQTArtifactError(f"model path does not exist: {root}")
    findings = _collect_findings(root)
    if not findings:
        return None
    formats = {item.format for item in findings}
    if len(formats) > 1:
        detail = ", ".join(f"{item.format}@{item.source_file}" for item in findings)
        raise XQTArtifactError(
            f"conflicting external quant formats detected under {root}: {detail}"
        )
    primary = findings[0]
    sources = tuple(item.source_file for item in findings)
    return ExternalQuantInfo(
        format=primary.format,
        source_file=primary.source_file,
        raw_config=dict(primary.raw_config),
        bits=primary.bits,
        group_size=primary.group_size,
        sym=primary.sym,
        desc_act=primary.desc_act,
        zero_point=primary.zero_point,
        quant_method_raw=primary.quant_method_raw,
        extra=dict(primary.extra),
        source_files=sources,
    )


def resolve_external_quantization(
    model_path: str | Path,
    *,
    user_quant: str | None = None,
) -> ExternalQuantInfo:
    """Full vLLM-style resolve: probe -> conflict check -> override chain."""

    root = Path(model_path)
    info = probe_external_quant_config(root)
    if info is None:
        raise XQTArtifactError(
            f"no external quant config found under {root}; "
            f"supported methods: {list_supported_external_formats()}"
        )
    resolved = override_external_format(info, user_quant)
    if resolved == info.format:
        return info
    return ExternalQuantInfo(
        format=resolved,
        source_file=info.source_file,
        raw_config=dict(info.raw_config),
        bits=info.bits,
        group_size=info.group_size,
        sym=info.sym,
        desc_act=info.desc_act,
        zero_point=info.zero_point,
        quant_method_raw=info.quant_method_raw,
        extra={**dict(info.extra), "user_quant": user_quant},
        source_files=info.source_files or (info.source_file,),
    )


__all__ = [
    "ExternalQuantInfo",
    "iter_config_filenames",
    "list_supported_external_formats",
    "normalize_external_format",
    "override_external_format",
    "probe_external_quant_config",
    "resolve_external_quantization",
]
