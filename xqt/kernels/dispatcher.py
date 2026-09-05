"""Adaptive operator dispatcher with timing-based routing and transparent fallback."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from xqt.kernels.selector import get_fastest_kernel, select_fastest_kernel
from xqt.kernels.spec import KernelBackend, KernelSpec

logger = logging.getLogger(__name__)


class AutoTunedOperator:
    """Dynamic runtime dispatcher routing calls to the fastest available kernel."""

    def __init__(
        self,
        op: str,
        fallback_fn: Optional[Callable[..., Any]] = None,
        extra: str = "",
        preferred_backends: Optional[Sequence[KernelBackend]] = None,
    ) -> None:
        self.op = op
        self.fallback_fn = fallback_fn
        self.extra = extra
        self.preferred_backends = preferred_backends
        self._last_selected_spec: Optional[KernelSpec] = None

    @property
    def last_selected_spec(self) -> Optional[KernelSpec]:
        return self._last_selected_spec

    def _extract_shape_and_dtype(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[tuple[int, ...] | None, str | None]:
        """Infer canonical problem shape and dtype from arguments."""
        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    tensor = v
                    break

        if tensor is not None:
            return tuple(tensor.shape), str(tensor.dtype).replace("torch.", "")
        return None, None

    def resolve(
        self,
        shape: Optional[tuple[int, ...] | Sequence[int]] = None,
        dtype: Optional[str] = None,
    ) -> KernelSpec:
        """Resolve the optimal KernelSpec for the specified problem shape."""
        spec = select_fastest_kernel(
            op=self.op,
            shape=shape,
            dtype=dtype,
            extra=self.extra,
            preferred_backends=self.preferred_backends,
        )
        self._last_selected_spec = spec
        return spec

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch operator to the fastest kernel with fallback protection."""
        shape, dtype = self._extract_shape_and_dtype(args, kwargs)

        try:
            kernel_fn = get_fastest_kernel(
                op=self.op,
                shape=shape,
                dtype=dtype,
                extra=self.extra,
            )
            return kernel_fn(*args, **kwargs)
        except Exception as e:
            if self.fallback_fn is not None:
                logger.warning(
                    "Fastest kernel dispatch for op %r failed (%s); falling back to reference implementation.",
                    self.op,
                    e,
                )
                return self.fallback_fn(*args, **kwargs)
            raise
