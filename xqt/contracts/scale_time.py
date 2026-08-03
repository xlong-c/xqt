"""Explicit scale lifetime labels (GUIDE / survey static-dynamic-online axes)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ScaleTime(str, Enum):
    """When scales are fixed relative to model load and forward."""

    WEIGHT_OFFLINE = "weight_offline"
    WEIGHT_LOAD_TIME = "weight_load_time"
    ACTIVATION_STATIC = "activation_static"
    ACTIVATION_DYNAMIC = "activation_dynamic"
    KV_SCALE = "kv_scale"

    @classmethod
    def parse(cls, value: str | ScaleTime | None) -> ScaleTime | None:
        if value is None:
            return None
        if isinstance(value, ScaleTime):
            return value
        text = str(value).strip().lower()
        for item in cls:
            if item.value == text:
                return item
        raise ValueError(
            f"unknown scale_time {value!r}; expected one of "
            f"{[item.value for item in cls]}"
        )


def scale_time_payload(
    scale_time: ScaleTime | str | None,
    *,
    activation_granularity: str | None = None,
) -> dict[str, Any]:
    """JSON-safe scale-time block for reports."""

    parsed = ScaleTime.parse(scale_time) if scale_time is not None else None
    return {
        "scale_time": None if parsed is None else parsed.value,
        "activation_granularity": activation_granularity,
    }


__all__ = ["ScaleTime", "scale_time_payload"]
