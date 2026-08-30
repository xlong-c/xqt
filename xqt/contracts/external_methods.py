"""External quant method aliases and override chain (vLLM-aligned)."""

from __future__ import annotations

from typing import Final

from xqt.core.base import XQTArtifactError
from xqt.contracts.external_types import ExternalQuantInfo

_SUPPORTED_METHODS: Final[frozenset[str]] = frozenset(
    {"compressed_tensors", "gptq", "awq"}
)

_METHOD_ALIASES: Final[dict[str, str]] = {
    "compressed-tensors": "compressed_tensors",
    "compressed_tensors": "compressed_tensors",
    "compressedtensors": "compressed_tensors",
    "gptq": "gptq",
    "gptq_marlin": "gptq",
    "gptq-marlin": "gptq",
    "auto_gptq": "gptq",
    "auto-gptq": "gptq",
    "awq": "awq",
    "awq_marlin": "awq",
    "awq-marlin": "awq",
    "auto_awq": "awq",
    "auto-awq": "awq",
    "autoawq": "awq",
    "marlin": "gptq",
}

_COMPATIBLE_USER_METHODS: Final[dict[str, frozenset[str | None]]] = {
    "gptq": frozenset(
        {None, "gptq", "gptq_marlin", "gptq-marlin", "auto_gptq", "auto-gptq", "marlin"}
    ),
    "awq": frozenset(
        {
            None,
            "awq",
            "awq_marlin",
            "awq-marlin",
            "auto_awq",
            "auto-awq",
            "autoawq",
            "marlin",
        }
    ),
    "compressed_tensors": frozenset(
        {None, "compressed_tensors", "compressed-tensors", "compressedtensors"}
    ),
}

_SIDECAR_CONFIG_FILES: Final[tuple[str, ...]] = (
    "quantize_config.json",
    "quant_config.json",
    "hf_quant_config.json",
    "quantization_config.json",
    "compress_config.json",
)


def normalize_external_format(name: str | None) -> str | None:
    """Normalize a user or checkpoint method alias to a canonical name."""

    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    if text in _METHOD_ALIASES:
        return _METHOD_ALIASES[text]
    return text.replace("-", "_")


def list_supported_external_formats() -> list[str]:
    """Return sorted first-batch supported external method names."""

    return sorted(_SUPPORTED_METHODS)


def override_external_format(
    info: ExternalQuantInfo,
    user_format: str | None,
) -> str:
    """Resolve checkpoint method against an optional user override (vLLM-style)."""

    checkpoint = str(info.format)
    if checkpoint not in _SUPPORTED_METHODS:
        raise XQTArtifactError(
            f"external quant format {checkpoint!r} is not in the first-batch "
            f"supported set: {sorted(_SUPPORTED_METHODS)}"
        )
    if user_format is None or not str(user_format).strip():
        return checkpoint
    user_raw = str(user_format).strip().lower()
    user_norm = normalize_external_format(user_raw)
    compatible = _COMPATIBLE_USER_METHODS.get(checkpoint, frozenset())
    if user_raw in compatible or user_norm == checkpoint:
        return checkpoint
    if user_norm in _SUPPORTED_METHODS and user_norm != checkpoint:
        raise XQTArtifactError(
            f"user quantization {user_format!r} is incompatible with checkpoint "
            f"method {checkpoint!r} (source={info.source_file}); "
            f"compatible overrides: {sorted(x for x in compatible if x)}"
        )
    raise XQTArtifactError(
        f"override format {user_format!r} is not supported; "
        f"supported: {sorted(_SUPPORTED_METHODS)}"
    )


def iter_config_filenames() -> tuple[str, ...]:
    """Return sidecar config filenames searched during probe."""

    return _SIDECAR_CONFIG_FILES


def supported_methods() -> frozenset[str]:
    """Return the frozenset of first-batch methods."""

    return _SUPPORTED_METHODS


__all__ = [
    "iter_config_filenames",
    "list_supported_external_formats",
    "normalize_external_format",
    "override_external_format",
    "supported_methods",
]
