"""Module conversion facade for operator-oriented XQT engines.

Public API: convert(), ConvertResult.
Implementation resides in xqt.kernels.nn.conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn

from xqt.kernels.precision import (
    FeedForwardPrecisionPolicy,
    OperatorContract,
    PrecisionPolicy,
    TensorStorageSpec,
)
from xqt.core.errors import XQTBackendError
from xqt.core.schema import CONVERT_ENGINE_NAMES

from xqt.kernels.nn.conversion.converter import _ModuleConverter
from xqt.kernels.nn.conversion.precision import (
    _precision_policy_from_mapping,
    _resolve_engine_alias,
)

EngineKind = Literal["torch", "triton", "tilelang", "cutile", "cute_dsl"]
MatmulPrecisionSpec = PrecisionPolicy


@dataclass(frozen=True)
class ConvertResult:
    """Structured conversion result for one module."""

    model: nn.Module
    engine: str
    target: str
    contract: OperatorContract
    converted: bool
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "target": self.target,
            "contract": self.contract.to_dict(),
            "converted": self.converted,
            "report": dict(self.report),
        }


def convert(
    module: nn.Module,
    *,
    engine: str | None = None,
    target: str = "cuda",
    policy: PrecisionPolicy
    | FeedForwardPrecisionPolicy
    | Mapping[str, str]
    | None = None,
    projection_policies: Mapping[str, PrecisionPolicy | Mapping[str, str]]
    | None = None,
    fallback: str = "eager",
    target_arch: str | None = None,
    inplace: bool = False,
    return_result: bool = False,
) -> nn.Module | ConvertResult:
    """Convert one module through the XQT operator facade (contract + optional materialize).

    ``engine`` is an optional materialize *preference* (DEBT-001), not a quant/infer
    handoff primary key. When omitted, defaults to ``"torch"`` (intent / minimal path).
    Capability-based engine selection for Infer lives in ``xqt.kernels.engine_resolve``.

    This public API is intentionally function-shaped. Internally it delegates to a
    stateful converter class so future recursive model conversion can share lowering
    context and capability caches.
    """

    if isinstance(policy, FeedForwardPrecisionPolicy):
        resolved_policy = policy.default
        resolved_projection_policies = policy.projection_policies()
        if projection_policies is not None:
            raise XQTBackendError(
                "xqt.convert does not allow both FeedForwardPrecisionPolicy and projection_policies"
            )
    elif isinstance(policy, Mapping):
        resolved_policy = _precision_policy_from_mapping(policy)
        resolved_projection_policies = projection_policies
    else:
        resolved_policy = policy or PrecisionPolicy()
        resolved_projection_policies = projection_policies
    resolved_engine = _resolve_engine_alias(
        engine="torch" if engine is None else engine,
        context="xqt.convert",
    )
    if resolved_engine not in CONVERT_ENGINE_NAMES:
        allowed = ", ".join(CONVERT_ENGINE_NAMES)
        raise XQTBackendError(
            f"xqt.convert engine={resolved_engine!r} is not a convert "
            f"materialize preference. Allowed: {allowed}"
        )
    result = _ModuleConverter(
        engine=resolved_engine,
        target=target,
        policy=resolved_policy,
        projection_policies=resolved_projection_policies,
        fallback=fallback,
        target_arch=target_arch,
        inplace=inplace,
        policy_was_explicit=policy is not None,
    ).convert_module(module)
    if return_result:
        return result
    return result.model


__all__ = [
    "ConvertResult",
    "EngineKind",
    "FeedForwardPrecisionPolicy",
    "MatmulPrecisionSpec",
    "OperatorContract",
    "PrecisionPolicy",
    "TensorStorageSpec",
    "convert",
]
