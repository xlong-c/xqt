"""MoE module classification helpers (model-side, no quant dependency)."""

from __future__ import annotations

import re
from typing import Sequence

DEFAULT_MOE_EXPERT_NAME_PATTERNS: tuple[str, ...] = (
    r"experts?\.\d+",
    r"experts?\.[^.]+",
    r"shared_expert",
    r"shared_experts?",
    r"mlp\.experts?",
)
DEFAULT_MOE_ROUTER_NAME_PATTERNS: tuple[str, ...] = (
    r"router",
    r"gate",
)


def _matches_pattern(name: str, patterns: Sequence[str]) -> bool:
    for pat in patterns:
        if re.search(pat, name):
            return True
    return False


def is_moe_expert_module(
    name: str,
    *,
    patterns: Sequence[str] | None = None,
) -> bool:
    return _matches_pattern(name, patterns or DEFAULT_MOE_EXPERT_NAME_PATTERNS)


def is_moe_router_module(
    name: str,
    *,
    patterns: Sequence[str] | None = None,
) -> bool:
    return _matches_pattern(name, patterns or DEFAULT_MOE_ROUTER_NAME_PATTERNS)


def classify_moe_module(
    name: str,
    *,
    expert_patterns: Sequence[str] | None = None,
    router_patterns: Sequence[str] | None = None,
) -> str:
    if is_moe_router_module(name, patterns=router_patterns):
        return "router"
    if re.search(r"shared_expert", name):
        return "shared_expert"
    if is_moe_expert_module(name, patterns=expert_patterns):
        return "expert"
    return "other"


__all__ = [
    "DEFAULT_MOE_EXPERT_NAME_PATTERNS",
    "DEFAULT_MOE_ROUTER_NAME_PATTERNS",
    "classify_moe_module",
    "is_moe_expert_module",
    "is_moe_router_module",
]
