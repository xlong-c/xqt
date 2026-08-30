"""Registry for hand-written runtime kernels that replace a whole model block."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from xqt.core.errors import XQTBackendError

from .types import OperatorOptimizationTargetPlan


BlockKernelBuilder = Callable[[nn.Module, OperatorOptimizationTargetPlan], nn.Module]

_BLOCK_KERNEL_BUILDERS: dict[str, BlockKernelBuilder] = {}


def register_block_kernel_builder(
    name: str,
) -> Callable[[BlockKernelBuilder], BlockKernelBuilder]:
    """Register a named builder for a hand-written block-level replacement."""

    normalized_name = str(name).strip()
    if not normalized_name:
        raise ValueError("block kernel builder name must not be empty")

    def decorator(builder: BlockKernelBuilder) -> BlockKernelBuilder:
        if normalized_name in _BLOCK_KERNEL_BUILDERS:
            raise ValueError(
                f"block kernel builder '{normalized_name}' is already registered"
            )
        _BLOCK_KERNEL_BUILDERS[normalized_name] = builder
        return builder

    return decorator


def available_block_kernel_builders() -> tuple[str, ...]:
    """Return registered hand-written block-kernel builder names."""

    return tuple(sorted(_BLOCK_KERNEL_BUILDERS))


def build_block_kernel_candidate(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Materialize one named block replacement without falling back to an operator wrapper."""

    if target.candidate_kind != "block_kernel":
        raise ValueError(
            "block kernel materialization requires candidate_kind=block_kernel"
        )
    builder_name = target.block_kernel
    if not builder_name:
        raise XQTBackendError(
            "block_kernel candidates require a named block_kernel builder; "
            "use candidate_kind=single_kernel for an operator wrapper"
        )
    builder = _BLOCK_KERNEL_BUILDERS.get(builder_name)
    if builder is None:
        available = ", ".join(available_block_kernel_builders()) or "none"
        raise XQTBackendError(
            f"block kernel builder '{builder_name}' is not registered; available: {available}"
        )
    candidate = builder(target_model, target)
    if not isinstance(candidate, nn.Module):
        raise XQTBackendError(
            f"block kernel builder '{builder_name}' must return torch.nn.Module"
        )
    metadata = getattr(candidate, "_xqt_block_kernel_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    candidate._xqt_block_kernel_metadata = {
        **metadata,
        "builder": builder_name,
        "engine": target.engine,
        "candidate_kind": target.candidate_kind,
    }
    return candidate


__all__ = [
    "BlockKernelBuilder",
    "available_block_kernel_builders",
    "build_block_kernel_candidate",
    "register_block_kernel_builder",
]
