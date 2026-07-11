"""Runtime-configured Linear facade backed by the unified GEMM dispatcher."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from xqt.contracts import PrecisionPolicy
from xqt.core.errors import XQTBackendError
from xqt.operator_opt.backends.gemm_precision import gemm_with_precision

from .precision import _resolve_engine_alias, _runtime_precision_dict


class _RuntimeLinearModule(nn.Module):
    """Runtime-configured Linear facade backed by the unified GEMM dispatcher."""

    def __init__(
        self,
        module: nn.Linear,
        *,
        engine: str,
        precision: PrecisionPolicy,
    ) -> None:
        super().__init__()
        if engine not in {"torch", "triton"}:
            raise XQTBackendError(f"unsupported runtime Linear engine: {engine}")
        self.module = module
        self.engine = engine
        self.runtime_precision = precision.to_dict()

    def configure_runtime(
        self,
        *,
        engine: str | None = None,
        precision: PrecisionPolicy | Mapping[str, str] | None = None,
        activation_dtype: str | None = None,
        weight_dtype: str | None = None,
        bias_dtype: str | None = None,
        mma_dtype: str | None = None,
        accum_dtype: str | None = None,
        output_dtype: str | None = None,
    ) -> None:
        if engine is not None:
            resolved_engine = _resolve_engine_alias(
                engine=engine,
                context="Runtime Linear configuration",
            )
            if resolved_engine not in {"torch", "triton"}:
                raise XQTBackendError(
                    f"unsupported runtime Linear engine: {resolved_engine}"
                )
            self.engine = resolved_engine
        if precision is not None:
            self.runtime_precision = _runtime_precision_dict(precision)
        updates = {
            "activation": activation_dtype,
            "weight": weight_dtype,
            "bias": bias_dtype,
            "mma": mma_dtype,
            "accum": accum_dtype,
            "output": output_dtype,
        }
        for key, value in updates.items():
            if value is None:
                continue
            self.runtime_precision[key] = str(value)

    def runtime_config(self) -> dict[str, str]:
        return {
            "engine": self.engine,
            "activation": self.runtime_precision["activation"],
            "weight": self.runtime_precision["weight"],
            "bias": self.runtime_precision["bias"],
            "mma": self.runtime_precision["mma"],
            "accum": self.runtime_precision["accum"],
            "output": self.runtime_precision["output"],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix_shape = tuple(x.shape[:-1])
        flat_x = x.reshape(-1, int(x.shape[-1]))
        weight = self.module.weight.to(device=x.device)
        bias = (
            None if self.module.bias is None else self.module.bias.to(device=x.device)
        )
        output = gemm_with_precision(
            flat_x,
            weight,
            bias,
            precision=PrecisionPolicy(**self.runtime_precision),
            engine=self.engine,
            transpose_b=True,
        )
        return output.reshape(*prefix_shape, int(output.shape[-1]))
