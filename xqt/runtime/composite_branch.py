"""Composite branch module protocol, fusion mixin, and materialize registry.

Dispatch for composite mixed-precision modules used to be ``if isinstance(...)``
on concrete class names. This module replaces that with contracts and a registry
so new branch algorithms plug in without editing dispatch sites:

- :class:`CompositeBranchModule` - base class carrying ``branches``
  (:class:`nn.ModuleList`), ``combine_strategy`` (:class:`CombineStrategy`),
  torch.compile fusion of ``_compute_lean``, small-M / non-CUDA routing, and
  ``reconstruct_weight`` via the combine strategy.
- :class:`SupportsStaticActivationCalibration` - duck-type surface the Infer
  static-scale calibrator hooks, replacing ``isinstance(m, (Int8, Fp8))``.
- the materialize registry (:func:`register_materializer` / :func:`get_materializer`)
  maps ``(composite_kind, mode, compute_precision)`` to a builder, so a storage
  shell's ``materialize_compute`` is a table lookup instead of a branch ladder.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch
from torch import nn

from xqt.runtime.composite_combine import CombineStrategy, get_combine_strategy

# ============================================================================
# Protocol: what the Infer static-scale calibrator hooks
# ============================================================================


@runtime_checkable
class SupportsStaticActivationCalibration(Protocol):
    """Duck-type surface for static activation-scale calibration.

    Modules that carry these attributes / methods participate in
    :func:`calibrate_static_activation_scales` without ``isinstance`` checks.
    """

    activation_scale_mode: str
    input_features: int
    eps: float
    quant_max: float
    static_activation_scale: torch.Tensor

    def set_static_activation_scale(self, scale: torch.Tensor | float) -> None:
        ...


# ============================================================================
# Base class for composite mixed-precision modules
# ============================================================================


class CompositeBranchModule(nn.Module):
    """Shared base for composite mixed-precision modules.

    A *composite* module computes ``combine(branch_0(x), branch_1(x), ...)``.
    The physical submodules are stored in ``self.branches`` (a
    :class:`nn.ModuleList`); a :class:`CombineStrategy` held in
    ``self.combine_strategy`` binds them into one output and one reconstructed
    weight.

    Subclasses override two methods:

    - ``_compute_lean(inputs)`` - pure-tensor forward (for torch.compile).
    - ``_compute_guarded(inputs)`` - eager full-path forward with per-branch
      guarding, fallback routing, and metadata writes.
    """

    combine_strategy: CombineStrategy
    branches: nn.ModuleList

    def __init__(
        self,
        *,
        combine_strategy: CombineStrategy | str,
        combine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(combine_strategy, str):
            kw = dict(combine_kwargs or {})
            combine_strategy = get_combine_strategy(combine_strategy, **kw)  # type: ignore[arg-type]
        self.combine_strategy = combine_strategy
        self.branches = nn.ModuleList()
        self._fused_forward: object | None = None  # torch.compile returns opaque

    # ---- fusion -----------------------------------------------------------

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            return False
        try:
            fused = compile_fn(self._compute_lean, mode=mode)
            self._fused_forward = fused
        except Exception:
            self._fused_forward = None
            return False
        return True

    def disable_fusion(self) -> None:
        self._fused_forward = None

    # ---- forwarding -------------------------------------------------------

    def _compute_lean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pure-tensor forward - override in subclass."""
        raise NotImplementedError

    def _compute_guarded(self, inputs: torch.Tensor) -> torch.Tensor:
        """Eager guarded forward - override in subclass."""
        raise NotImplementedError

    def _should_fuse(self, inputs: torch.Tensor) -> bool:
        return self._fused_forward is not None and inputs.is_cuda

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._should_fuse(inputs):
            return self._fused_forward(inputs)  # type: ignore[operator]
        return self._compute_guarded(inputs)

    # ---- weight reconstruction --------------------------------------------

    def branch_weights(self) -> list[torch.Tensor]:
        """Collect dense weight per branch for the combine strategy.

        Default walks ``self.branches`` and calls ``dequantize_weight()`` on
        each; override when branches need a different weight attribute.
        """

        weights: list[torch.Tensor] = []
        for branch in self.branches:
            deq = getattr(branch, "dequantize_weight", None)
            if callable(deq):
                ret = deq()
                if isinstance(ret, torch.Tensor):
                    weights.append(ret)
                continue
            weight = getattr(branch, "weight", None)
            if isinstance(weight, torch.Tensor):
                weights.append(weight.detach())
                continue
            raise AttributeError(
                f"branch {type(branch).__name__} has no dequantize_weight() "
                "or weight; override branch_weights()"
            )
        return weights

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the full dense weight via the combine strategy."""
        return self.combine_strategy.reconstruct_weight(self.branch_weights())

    # ---- introspection ----------------------------------------------------

    def execution_metadata(self) -> dict[str, Any]:
        return {
            "implementation": f"composite_{self.combine_strategy.name}",
            "compute_contract": f"composite_{self.combine_strategy.name}",
            "fusion_enabled": self._fused_forward is not None,
            "branch_count": len(self.branches),
        }


# ============================================================================
# Materialize registry: (composite_kind, mode, compute_precision) -> builder
# ============================================================================

MaterializerFunc = Callable[..., nn.Module]
_MATERIALIZER_REGISTRY: dict[tuple[str, str, str], MaterializerFunc] = {}


def register_materializer(
    composite_kind: str,
    *,
    mode: str = "split",
    compute_precision: str = "w8a8",
) -> Callable[[MaterializerFunc], MaterializerFunc]:
    """Decorator: register a builder for a (kind, mode, precision) triple.

    The decorated function receives a storage shell (that knows the weight
    tensors and shapes) plus keyword arguments for the execution policy; it
    returns a materialized :class:`nn.Module`.
    """

    canonical_mode = str(mode).strip().lower()
    canonical_precision = str(compute_precision).strip().lower()
    key = (str(composite_kind).strip().lower(), canonical_mode, canonical_precision)

    def decorator(fn: MaterializerFunc) -> MaterializerFunc:
        _MATERIALIZER_REGISTRY[key] = fn
        return fn

    return decorator


def get_materializer(
    composite_kind: str,
    *,
    mode: str = "split",
    compute_precision: str = "w8a8",
) -> MaterializerFunc | None:
    """Look up a materializer for a (kind, mode, precision) triple.

    Returns ``None`` when no builder is registered; the shell's
    ``materialize_compute`` then returns itself (identity passthrough).
    """

    return _MATERIALIZER_REGISTRY.get(
        (
            str(composite_kind).strip().lower(),
            str(mode).strip().lower(),
            str(compute_precision).strip().lower(),
        )
    )


# ============================================================================
# Submodule replacement helper
# ============================================================================


def replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace one named submodule in the tree, supporting
    :class:`nn.Sequential` / :class:`nn.ModuleList` numeric indices.
    """

    parent_path, _, attr = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attr.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attr)] = replacement
        return
    setattr(parent, attr, replacement)


__all__ = [
    "CompositeBranchModule",
    "MaterializerFunc",
    "SupportsStaticActivationCalibration",
    "get_materializer",
    "register_materializer",
    "replace_submodule",
]
