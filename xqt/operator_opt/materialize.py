"""Cross-engine operator candidate materialization and contract validation."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Sequence

from torch import nn

from xqt.core.errors import XQTBackendError

from .block_kernels import build_block_kernel_candidate
from .compile_backend import compile_with_torch
from .reference_wrappers import build_reference_guarded_linear_candidate_model
from .tilelang_wrappers import build_tilelang_candidate_model
from .triton_wrappers import build_triton_candidate_model
from .types import OperatorOptimizationTargetPlan

if TYPE_CHECKING:
    from xqt.contracts import ModuleContract


_CONTRACT_PATTERNS: dict[str, frozenset[str]] = {
    "linear": frozenset(
        {
            "linear",
            "dense_linear_epilogue",
            "dequant_gemm_epilogue",
            "fp4_packed_dequant_gemm_epilogue",
            "mxfp4_packed_dequant_gemm_epilogue",
            "nvfp4_packed_dequant_gemm_epilogue",
            "gemm_int4_dequant",
            "gemm_mxfp8",
            "gemm_mxfp6",
            "gemm_mxfp4",
            "gemm_nvfp4_packed_dequant",
        }
    ),
    "conv2d": frozenset({"conv"}),
    "feedforward": frozenset({"feedforward"}),
    "layernorm": frozenset({"norm"}),
    "attention": frozenset({"attention"}),
    "transformer_block": frozenset({"attention", "feedforward"}),
}


def _resolve_component_model(
    model: nn.Module,
    target_path: Optional[str],
) -> nn.Module:
    if not target_path:
        return model
    return model.get_submodule(target_path)


def _replace_component_model(
    model: nn.Module,
    target_path: Optional[str],
    replacement: nn.Module,
) -> nn.Module:
    if not target_path:
        return replacement
    parent_path, _, attribute = target_path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)
    return model


def _materialize_target(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> tuple[nn.Module, float | None]:
    if target.engine == "torch_compile":
        return compile_with_torch(target_model, target)
    if target.candidate_kind == "block_kernel":
        return build_block_kernel_candidate(target_model, target), None
    if target.engine == "tilelang":
        return build_tilelang_candidate_model(target_model, target), None
    if target.engine == "triton":
        return build_triton_candidate_model(target_model, target), None
    if target.engine in {"cutile", "cute_dsl"}:
        return (
            build_reference_guarded_linear_candidate_model(
                target_model,
                target,
                engine=target.engine,
            ),
            None,
        )
    raise XQTBackendError(
        f"Operator optimization engine '{target.engine}' is not executable yet"
    )


def materialize_operator_candidate_model(
    model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> tuple[nn.Module, float | None]:
    """Build one candidate root model with one target optimization applied."""

    candidate_root = copy.deepcopy(model)
    candidate_target = _resolve_component_model(candidate_root, target.target_path)
    replacement, compile_time_ms = _materialize_target(candidate_target, target)
    return (
        _replace_component_model(candidate_root, target.target_path, replacement),
        compile_time_ms,
    )


def materialize_module(
    module: nn.Module,
    *,
    contract: "ModuleContract",
    target: OperatorOptimizationTargetPlan,
) -> tuple[nn.Module, float | None]:
    """Materialize one module from a shared module contract and target plan."""

    allowed_patterns = _CONTRACT_PATTERNS.get(contract.operator_kind)
    if allowed_patterns is None:
        raise XQTBackendError(
            "operator materialization does not support module contract kind "
            f"'{contract.operator_kind}'"
        )
    unsupported_patterns = sorted(set(target.patterns) - allowed_patterns)
    if unsupported_patterns:
        raise XQTBackendError(
            f"module contract kind '{contract.operator_kind}' does not support "
            f"operator pattern(s): {', '.join(unsupported_patterns)}"
        )
    candidate, compile_time_ms = materialize_operator_candidate_model(module, target)
    setattr(candidate, "_xqt_module_contract", contract.to_dict())
    return candidate, compile_time_ms


def materialize_operator_candidate_models(
    model: nn.Module,
    targets: Sequence[OperatorOptimizationTargetPlan],
    *,
    inplace: bool = False,
) -> nn.Module:
    """Materialize a root model with multiple targets applied sequentially."""

    candidate_root = model if inplace else copy.deepcopy(model)
    for target in targets:
        candidate_target = _resolve_component_model(candidate_root, target.target_path)
        replacement, _ = _materialize_target(candidate_target, target)
        candidate_root = _replace_component_model(
            candidate_root,
            target.target_path,
            replacement,
        )
    return candidate_root


__all__ = [
    "materialize_module",
    "materialize_operator_candidate_model",
    "materialize_operator_candidate_models",
]
