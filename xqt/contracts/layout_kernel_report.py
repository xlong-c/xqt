"""Layout / kernel diagnostic report (Marlin-style fields, model-side only).

Used by load/process/apply paths so failures are not reduced to a bare
``method=awq`` string. Keys are always present in ``to_dict()`` even when null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from xqt.core.base import XQTConfigError

LAYOUT_KERNEL_REPORT_KEYS: Final[tuple[str, ...]] = (
    "bits",
    "group_size",
    "symmetric",
    "zero_point",
    "desc_act",
    "g_idx_applied",
    "global_shape",
    "local_shape",
    "padding_ratio",
    "storage_layout",
    "selected_kernel",
    "fallback_reason",
    "sm",
    "min_capability",
    "scale_time",
    "activation_granularity",
)


def _as_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise XQTConfigError(
            f"LayoutKernelReport.{field_name} must be int or None; got {value!r}"
        ) from exc


def _as_optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise XQTConfigError(
        f"LayoutKernelReport.{field_name} must be bool or None; "
        f"got {type(value).__name__}"
    )


def _as_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise XQTConfigError(
            f"LayoutKernelReport.{field_name} must be float or None; got {value!r}"
        ) from exc


def _as_optional_shape(value: Any, *, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise XQTConfigError(
            f"LayoutKernelReport.{field_name} must be a sequence or None"
        )
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            raise XQTConfigError(
                f"LayoutKernelReport.{field_name} entries must be int; got {item!r}"
            ) from exc
    return tuple(out)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


@dataclass(frozen=True, slots=True)
class LayoutKernelReport:
    """Diagnostic fields for packed-weight layout and kernel selection."""

    bits: int | None = None
    group_size: int | None = None
    symmetric: bool | None = None
    zero_point: bool | None = None
    desc_act: bool | None = None
    g_idx_applied: bool | None = None
    global_shape: tuple[int, ...] | None = None
    local_shape: tuple[int, ...] | None = None
    padding_ratio: float | None = None
    storage_layout: str | None = None
    selected_kernel: str | None = None
    fallback_reason: str | None = None
    sm: int | None = None
    min_capability: int | None = None
    scale_time: str | None = None
    activation_granularity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "zero_point": self.zero_point,
            "desc_act": self.desc_act,
            "g_idx_applied": self.g_idx_applied,
            "global_shape": (
                None if self.global_shape is None else list(self.global_shape)
            ),
            "local_shape": (
                None if self.local_shape is None else list(self.local_shape)
            ),
            "padding_ratio": self.padding_ratio,
            "storage_layout": self.storage_layout,
            "selected_kernel": self.selected_kernel,
            "fallback_reason": self.fallback_reason,
            "sm": self.sm,
            "min_capability": self.min_capability,
            "scale_time": self.scale_time,
            "activation_granularity": self.activation_granularity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LayoutKernelReport:
        if not isinstance(payload, Mapping):
            raise XQTConfigError(
                "LayoutKernelReport.from_dict expects a mapping; "
                f"got {type(payload).__name__}"
            )
        # Require storage_layout key presence for partial-reject tests that omit
        # the whole diagnostic block; empty/null storage_layout is still allowed
        # on empty reports, but from_dict used for non-empty reconstruction must
        # see the key when other diagnostic fields are present.
        if "storage_layout" not in payload and any(
            key in payload for key in LAYOUT_KERNEL_REPORT_KEYS if key != "storage_layout"
        ):
            raise XQTConfigError(
                "LayoutKernelReport missing required field: storage_layout"
            )
        if not payload:
            raise XQTConfigError(
                "LayoutKernelReport missing required field: storage_layout"
            )
        # Explicit: single-field partial without storage_layout
        if "bits" in payload and "storage_layout" not in payload:
            raise XQTConfigError(
                "LayoutKernelReport missing required field: storage_layout"
            )
        return cls(
            bits=_as_optional_int(payload.get("bits"), field_name="bits"),
            group_size=_as_optional_int(
                payload.get("group_size"), field_name="group_size"
            ),
            symmetric=_as_optional_bool(
                payload.get("symmetric"), field_name="symmetric"
            ),
            zero_point=_as_optional_bool(
                payload.get("zero_point"), field_name="zero_point"
            ),
            desc_act=_as_optional_bool(payload.get("desc_act"), field_name="desc_act"),
            g_idx_applied=_as_optional_bool(
                payload.get("g_idx_applied"), field_name="g_idx_applied"
            ),
            global_shape=_as_optional_shape(
                payload.get("global_shape"), field_name="global_shape"
            ),
            local_shape=_as_optional_shape(
                payload.get("local_shape"), field_name="local_shape"
            ),
            padding_ratio=_as_optional_float(
                payload.get("padding_ratio"), field_name="padding_ratio"
            ),
            storage_layout=_as_optional_str(payload.get("storage_layout")),
            selected_kernel=_as_optional_str(payload.get("selected_kernel")),
            fallback_reason=_as_optional_str(payload.get("fallback_reason")),
            sm=_as_optional_int(payload.get("sm"), field_name="sm"),
            min_capability=_as_optional_int(
                payload.get("min_capability"), field_name="min_capability"
            ),
            scale_time=_as_optional_str(payload.get("scale_time")),
            activation_granularity=_as_optional_str(
                payload.get("activation_granularity")
            ),
        )


def empty_layout_kernel_report(
    *,
    fallback_reason: str | None = None,
    bits: int | None = None,
    group_size: int | None = None,
    symmetric: bool | None = None,
    zero_point: bool | None = None,
    desc_act: bool | None = None,
    scale_time: str | None = None,
    activation_granularity: str | None = None,
) -> LayoutKernelReport:
    """Build a report with all keys present (nulls) plus optional probe fields."""

    return LayoutKernelReport(
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        zero_point=zero_point,
        desc_act=desc_act,
        g_idx_applied=None,
        global_shape=None,
        local_shape=None,
        padding_ratio=None,
        storage_layout=None,
        selected_kernel=None,
        fallback_reason=fallback_reason,
        sm=None,
        min_capability=None,
        scale_time=scale_time,
        activation_granularity=activation_granularity,
    )


def layout_report_from_module_shapes(
    *,
    bits: int | None,
    group_size: int | None,
    symmetric: bool | None,
    zero_point: bool | None,
    desc_act: bool | None,
    g_idx_applied: bool | None,
    out_features: int,
    in_features: int,
    padded_in_features: int | None,
    storage_layout: str,
    selected_kernel: str | None,
    fallback_reason: str | None = None,
    sm: int | None = None,
    min_capability: int | None = None,
    scale_time: str | None = None,
    activation_granularity: str | None = None,
) -> LayoutKernelReport:
    """Fill shape/padding fields from Linear-like dimensions."""

    padded = int(padded_in_features) if padded_in_features is not None else int(in_features)
    pad_ratio: float | None
    if in_features <= 0:
        pad_ratio = None
    else:
        pad_ratio = max(0.0, float(padded - in_features) / float(in_features))
    shape = (int(out_features), int(in_features))
    local = (int(out_features), int(padded))
    return LayoutKernelReport(
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        zero_point=zero_point,
        desc_act=desc_act,
        g_idx_applied=g_idx_applied,
        global_shape=shape,
        local_shape=local,
        padding_ratio=pad_ratio,
        storage_layout=storage_layout,
        selected_kernel=selected_kernel,
        fallback_reason=fallback_reason,
        sm=sm,
        min_capability=min_capability,
        scale_time=scale_time,
        activation_granularity=activation_granularity,
    )


__all__ = [
    "LAYOUT_KERNEL_REPORT_KEYS",
    "LayoutKernelReport",
    "empty_layout_kernel_report",
    "layout_report_from_module_shapes",
]
