"""Runtime intent mixin shared by XQT semantic module facades."""

from __future__ import annotations

from typing import Any

from ._precision import (
    _canonical_precision_name,
    _default_runtime_precision,
    _resolve_engine_alias,
)


class _SemanticModuleMixin:
    """Runtime intent shared by XQT semantic module facades."""

    def _init_runtime_intent(self, *, engine: str | None) -> None:
        resolved = _resolve_engine_alias(
            engine=engine,
            context=type(self).__name__,
        )
        if resolved not in {"torch", "triton", "tilelang", "cutile", "cute_dsl"}:
            raise ValueError(f"unsupported engine: {resolved}")
        self.engine = resolved
        self.runtime_precision = _default_runtime_precision()
        self.runtime_fallback: dict[str, Any] | None = None
        self.runtime_fallback_count = 0

    def configure_runtime(
        self,
        *,
        engine: str | None = None,
        activation_dtype: str | None = None,
        weight_dtype: str | None = None,
        bias_dtype: str | None = None,
        mma_dtype: str | None = None,
        accum_dtype: str | None = None,
        output_dtype: str | None = None,
    ) -> None:
        if engine is not None:
            self._init_runtime_intent(engine=engine)
        updates = {
            "activation": activation_dtype,
            "weight": weight_dtype,
            "bias": bias_dtype,
            "mma": mma_dtype,
            "accum": accum_dtype,
            "output": output_dtype,
        }
        for name, value in updates.items():
            if value is not None:
                self.runtime_precision[name] = _canonical_precision_name(value)

    def runtime_config(self) -> dict[str, str]:
        return {"engine": self.engine, **dict(self.runtime_precision)}
