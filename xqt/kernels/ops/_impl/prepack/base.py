from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from xqt.core.errors import XQTBackendError


@dataclass(frozen=True)
class PrepackLayoutId:
    dtype: str
    arch: str
    role: str

    def key(self) -> str:
        return f"{self.dtype}:{self.arch}:{self.role}"


@dataclass
class PrepackSpec:
    layout_id: PrepackLayoutId
    description: str
    status: str
    math_shape_order: str
    packed_shape_order: str
    pack: Callable[..., PrepackResult] | None = None
    unpack: Callable[..., torch.Tensor] | None = None
    notes: str = ""
    metadata_defaults: dict[str, Any] = field(default_factory=dict)

    def is_implemented(self) -> bool:
        return self.status == "implemented" and self.pack is not None and self.unpack is not None


@dataclass
class PrepackResult:
    packed: torch.Tensor
    layout_id: PrepackLayoutId
    math_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout_id.key(),
            "dtype": self.layout_id.dtype,
            "arch": self.layout_id.arch,
            "role": self.layout_id.role,
            "math_shape": list(self.math_shape),
            "packed_shape": list(self.packed_shape),
            "metadata": dict(self.metadata),
            "packed_dtype": str(self.packed.dtype),
            "packed_device": str(self.packed.device),
            "packed_nbytes": int(self.packed.numel() * self.packed.element_size()),
        }


PREPACK_REGISTRY: dict[str, PrepackSpec] = {}


def register_prepack_spec(spec: PrepackSpec) -> PrepackSpec:
    PREPACK_REGISTRY[spec.layout_id.key()] = spec
    return spec


def get_prepack_spec(layout_key: str) -> PrepackSpec:
    if layout_key not in PREPACK_REGISTRY:
        raise XQTBackendError(
            f"Unknown prepack layout {layout_key!r}. Available: {sorted(PREPACK_REGISTRY)}"
        )
    return PREPACK_REGISTRY[layout_key]


def list_prepack_specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, spec in sorted(PREPACK_REGISTRY.items()):
        rows.append(
            {
                "layout": key,
                "status": spec.status,
                "description": spec.description,
                "math_shape_order": spec.math_shape_order,
                "packed_shape_order": spec.packed_shape_order,
                "notes": spec.notes,
            }
        )
    return rows


def prepack_weight(weight: torch.Tensor, layout_key: str, **kwargs: Any) -> PrepackResult:
    spec = get_prepack_spec(layout_key)
    if not spec.is_implemented() or spec.pack is None:
        raise XQTBackendError(
            f"Prepack layout {layout_key!r} is {spec.status}. {spec.notes}"
        )
    return spec.pack(weight, **kwargs)


def unpack_weight(
    packed: torch.Tensor,
    layout_key: str,
    *,
    math_shape: tuple[int, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    spec = get_prepack_spec(layout_key)
    if not spec.is_implemented() or spec.unpack is None:
        raise XQTBackendError(
            f"Prepack layout {layout_key!r} is {spec.status}. {spec.notes}"
        )
    return spec.unpack(packed, math_shape=math_shape, metadata=metadata, **kwargs)


def _register_placeholders() -> None:
    placeholders = [
        PrepackSpec(
            layout_id=PrepackLayoutId("int4", "sm_89", "b_mma"),
            description="INT4 weight prepack for sm_89 INT8-retarget / native int4 paths",
            status="placeholder",
            math_shape_order="K,N",
            packed_shape_order="N,K_packed",
            notes="Placeholder: pack after W4 unpack or direct int4 MMA when kernel lands.",
        ),
        PrepackSpec(
            layout_id=PrepackLayoutId("fp4", "sm_89", "b_mma"),
            description="FP4 / NVFP4 weight prepack for non-native FP4 devices (retarget path)",
            status="placeholder",
            math_shape_order="K,N",
            packed_shape_order="N,K_packed",
            notes="Placeholder: share W4 storage; prepack applies to INT8 compute view or future FP4 MMA.",
        ),
        PrepackSpec(
            layout_id=PrepackLayoutId("fp8", "sm_89", "b_mma"),
            description="FP8 e4m3/e5m2 weight prepack for Ada FP8 Tensor Core layouts",
            status="placeholder",
            math_shape_order="K,N",
            packed_shape_order="N,K_swizzled",
            notes="Placeholder: needs m16n8k32 fp8 fragment layout + scale recipe.",
        ),
        PrepackSpec(
            layout_id=PrepackLayoutId("int8", "sm_80", "b_mma"),
            description="INT8 B prepack for Ampere sm_80",
            status="placeholder",
            math_shape_order="K,N",
            packed_shape_order="N,K",
            notes="Placeholder: reuse int8:sm_89:b_nk when Ampere kernel is specialized.",
        ),
        PrepackSpec(
            layout_id=PrepackLayoutId("int8", "sm_90", "b_mma"),
            description="INT8 B prepack for Hopper (WGMMA-oriented layout TBD)",
            status="placeholder",
            math_shape_order="K,N",
            packed_shape_order="N,K_wgmma",
            notes="Placeholder: Hopper prefers different swizzle / WGMMA atom layout.",
        ),
    ]
    for spec in placeholders:
        register_prepack_spec(spec)


_register_placeholders()
