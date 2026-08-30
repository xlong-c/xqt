"""ComfyUI-compatible ``comfy_quant`` marker encode/decode helpers.

XQT does not depend on ComfyUI. These helpers only speak the safetensors side
convention used by stock ComfyUI ``int8_tensorwise`` loaders and offline tools
such as comfy-quants / comfy-model-tools / convert_to_quant.

Stock marker (preferred):
    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

Legacy INT8-Fast marker (accepted on import, rewritten on export):
    {"convrot": true, "convrot_groupsize": 256, "per_row": true}
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import torch

STOCK_INT8_FORMAT = "int8_tensorwise"
DEFAULT_CONVROT_GROUP_SIZE = 256


def encode_comfy_quant_marker(payload: Mapping[str, Any]) -> torch.Tensor:
    """Encode a marker mapping to a uint8 byte tensor (safetensors-friendly)."""

    text = json.dumps(dict(payload), separators=(",", ":"), ensure_ascii=True)
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def decode_comfy_quant_marker(marker: torch.Tensor | bytes | bytearray | str) -> dict[str, Any]:
    """Decode a marker tensor / raw bytes / JSON string into a plain dict."""

    if isinstance(marker, str):
        text = marker
    elif isinstance(marker, (bytes, bytearray)):
        text = bytes(marker).decode("utf-8")
    elif isinstance(marker, torch.Tensor):
        if marker.dtype != torch.uint8:
            raise TypeError(
                f"comfy_quant marker tensor must be uint8, got {marker.dtype}"
            )
        flat = marker.detach().cpu().reshape(-1).tolist()
        text = bytes(int(value) & 0xFF for value in flat).decode("utf-8")
    else:
        raise TypeError(
            "comfy_quant marker must be a uint8 tensor, bytes, or JSON string"
        )
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("comfy_quant marker JSON must be an object")
    return {str(key): value for key, value in payload.items()}


def build_int8_tensorwise_marker(
    *,
    convrot: bool = False,
    convrot_groupsize: int = DEFAULT_CONVROT_GROUP_SIZE,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stock ComfyUI ``int8_tensorwise`` marker payload."""

    payload: dict[str, Any] = {"format": STOCK_INT8_FORMAT}
    if convrot:
        payload["convrot"] = True
        payload["convrot_groupsize"] = int(convrot_groupsize)
    if extra:
        for key, value in extra.items():
            if key == "format":
                continue
            payload[str(key)] = value
    return payload


def normalize_int8_tensorwise_marker(
    payload: Mapping[str, Any],
    *,
    default_groupsize: int = DEFAULT_CONVROT_GROUP_SIZE,
) -> dict[str, Any]:
    """Normalize legacy / stock markers into the stock shape with ``format``."""

    raw = {str(key): value for key, value in payload.items()}
    convrot = bool(raw.get("convrot", False))
    group_size = raw.get("convrot_groupsize", raw.get("group_size", default_groupsize))
    extra = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "format",
            "convrot",
            "convrot_groupsize",
            "group_size",
            "per_row",
        }
    }
    return build_int8_tensorwise_marker(
        convrot=convrot,
        convrot_groupsize=int(group_size),
        extra=extra,
    )


def encode_int8_tensorwise_marker(
    *,
    convrot: bool = False,
    convrot_groupsize: int = DEFAULT_CONVROT_GROUP_SIZE,
    extra: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Convenience: build stock marker and encode to uint8 tensor."""

    return encode_comfy_quant_marker(
        build_int8_tensorwise_marker(
            convrot=convrot,
            convrot_groupsize=convrot_groupsize,
            extra=extra,
        )
    )


def marker_reports_convrot(payload: Mapping[str, Any]) -> bool:
    """Return whether a decoded marker enables ConvRot."""

    return bool(payload.get("convrot", False))


def marker_convrot_groupsize(
    payload: Mapping[str, Any],
    *,
    default: int = DEFAULT_CONVROT_GROUP_SIZE,
) -> int:
    """Return ConvRot group size from a decoded marker."""

    value = payload.get("convrot_groupsize", payload.get("group_size", default))
    return int(value)


__all__ = [
    "DEFAULT_CONVROT_GROUP_SIZE",
    "STOCK_INT8_FORMAT",
    "build_int8_tensorwise_marker",
    "decode_comfy_quant_marker",
    "encode_comfy_quant_marker",
    "encode_int8_tensorwise_marker",
    "marker_convrot_groupsize",
    "marker_reports_convrot",
    "normalize_int8_tensorwise_marker",
]
