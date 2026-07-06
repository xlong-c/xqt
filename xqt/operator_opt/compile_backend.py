"""torch.compile helpers for XQT operator optimization."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
from torch import nn

from xqt.core.errors import XQTBackendError

from .types import OperatorOptimizationTargetPlan


def compile_with_torch(
    module: nn.Module,
    plan: OperatorOptimizationTargetPlan,
) -> tuple[nn.Module, float]:
    """Compile a module with torch.compile and return compile time in milliseconds."""

    if not hasattr(torch, "compile"):
        raise XQTBackendError("torch.compile is not available in the current PyTorch build")

    engine = str(plan.options.get("engine", "inductor"))
    mode = None if plan.mode in {None, "default"} else plan.mode
    compile_options: dict[str, Any] | None = dict(plan.options) if plan.options else None
    if compile_options is not None:
        compile_options.pop("engine", None)
        if not compile_options:
            compile_options = None
    start = perf_counter()
    try:
        compiled = torch.compile(
            module,
            fullgraph=plan.fullgraph,
            dynamic=plan.dynamic,
            backend=engine,
            mode=mode,
            options=compile_options,
        )
    except Exception as exc:
        raise XQTBackendError(f"torch.compile failed: {exc}") from exc
    return compiled, (perf_counter() - start) * 1000.0


__all__ = ["compile_with_torch"]
