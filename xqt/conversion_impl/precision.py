"""Precision policy helpers used by the conversion pipeline."""

from __future__ import annotations

from typing import Any, Mapping

from xqt.contracts import FeedForwardPrecisionPolicy, PrecisionPolicy
from xqt.core.errors import XQTBackendError


def _resolve_engine_alias(
    *,
    engine: str | None,
    context: str,
) -> str:
    if engine is None:
        raise XQTBackendError(f"{context} requires engine=...")
    return str(engine).strip().lower()


def _runtime_precision_dict(
    policy: PrecisionPolicy | Mapping[str, str],
) -> dict[str, Any]:
    if isinstance(policy, PrecisionPolicy):
        return policy.to_dict()
    if isinstance(policy, FeedForwardPrecisionPolicy):
        return policy.default.to_dict()
    return PrecisionPolicy.from_mapping(policy).to_dict()


def _projection_precision_dict(
    policy: PrecisionPolicy
    | Mapping[str, str]
    | FeedForwardPrecisionPolicy,
) -> dict[str, dict[str, Any]] | None:
    if isinstance(policy, FeedForwardPrecisionPolicy):
        return {
            name: projection_policy.to_dict()
            for name, projection_policy in policy.projection_policies().items()
        }
    if isinstance(policy, PrecisionPolicy):
        return None
    return {
        str(name): _projection_policy_dict(projection_policy)
        for name, projection_policy in policy.items()
    }


def _precision_policy_from_mapping(policy: Mapping[str, Any]) -> PrecisionPolicy:
    return PrecisionPolicy.from_mapping(policy)


def _projection_policy_dict(
    policy: PrecisionPolicy | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(policy, PrecisionPolicy):
        return policy.to_dict()
    return PrecisionPolicy.normalize_fields(policy)
