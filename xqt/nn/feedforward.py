"""Configurable FFN facade with optional fused activation fastpaths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import FusionIntent
from xqt.operator_opt.backends import run_triton_kernel
from xqt.operator_opt.kernels.triton.gemm import (
    gemm_bf16_triton,
    gemm_fp16_triton,
)
from xqt.gemm import dense_gemm_reference

from .linear import LayerNorm, Linear
from .norm import RMSNorm
from ._precision import (
    _TORCH_DTYPE_PRECISION_NAMES,
    ActivationKind,
    EngineKind,
    NormKind,
    ProjectionName,
    _canonical_precision_key,
    _canonical_precision_name,
    _compute_output_precision_name,
    _default_runtime_precision,
    _precision_name_from_dtype,
    _precision_name_to_dtype,
    _resolve_engine_alias,
)


@dataclass(frozen=True)
class FeedForwardFusionConfig:
    """Runtime fusion intent for the FeedForward facade."""

    enabled: bool = True
    prefer_single_kernel: bool = False
    fuse_norm: bool = False
    fuse_linear_activation: bool = True
    fuse_gate: bool = True
    fuse_output: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "enabled": self.enabled,
            "prefer_single_kernel": self.prefer_single_kernel,
            "fuse_norm": self.fuse_norm,
            "fuse_linear_activation": self.fuse_linear_activation,
            "fuse_gate": self.fuse_gate,
            "fuse_output": self.fuse_output,
        }


class FeedForward(nn.Module):
    """Configurable FFN facade with optional fused activation fastpaths."""

    def __init__(
        self,
        dim: int,
        *,
        dim_out: int | None = None,
        inner_dim: int | None = None,
        mult: int = 4,
        activation: ActivationKind = "gelu",
        norm: NormKind | None = None,
        eps: float = 1e-5,
        dropout: float = 0.0,
        final_dropout: bool = False,
        bias: bool = True,
        engine: EngineKind | None = None,
        fusion: bool | FeedForwardFusionConfig = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.dim_out = int(dim if dim_out is None else dim_out)
        self.inner_dim = int(self.dim * mult if inner_dim is None else inner_dim)
        self.activation = activation
        self.norm_kind = norm
        self.dropout_p = float(dropout)
        self.final_dropout_enabled = bool(final_dropout)
        resolved_engine = _resolve_engine_alias(
            engine=engine,
            context="FeedForward",
        )
        if resolved_engine not in {"torch", "triton"}:
            raise ValueError(f"unsupported engine: {resolved_engine}")
        self.engine = resolved_engine
        self.fusion_config = (
            fusion
            if isinstance(fusion, FeedForwardFusionConfig)
            else FeedForwardFusionConfig(enabled=bool(fusion))
        )
        self.runtime_precision = _default_runtime_precision()
        self.runtime_fallback: dict[str, Any] | None = None
        self.runtime_fallback_count = 0

        if norm is None:
            self.norm: nn.Module | None = None
        elif norm == "layernorm":
            self.norm = LayerNorm(self.dim, eps=eps, engine=self.engine)
        elif norm == "rmsnorm":
            self.norm = RMSNorm(self.dim, eps=eps)
        else:
            raise ValueError(f"unsupported norm: {norm}")

        if activation in {
            "gelu",
            "gelu-approximate",
            "geglu-approximate",
            "linear-silu",
        }:
            self.proj_in = Linear(self.dim, self.inner_dim, bias=bias, engine=self.engine)
            self.proj_gate = None
        elif activation in {"geglu", "swiglu"}:
            self.proj_in = Linear(self.dim, self.inner_dim, bias=bias, engine=self.engine)
            self.proj_gate = Linear(
                self.dim,
                self.inner_dim,
                bias=bias,
                engine=self.engine,
            )
        else:
            raise ValueError(f"unsupported activation: {activation}")

        self.proj_out = Linear(
            self.inner_dim,
            self.dim_out,
            bias=bias,
            engine=self.engine,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.final_dropout = (
            nn.Dropout(self.dropout_p) if self.final_dropout_enabled else None
        )
        self.projection_precision = {
            name: _default_runtime_precision() for name in self._projection_names()
        }

    def configure_runtime(
        self,
        *,
        engine: EngineKind | None = None,
        activation_dtype: str | None = None,
        weight_dtype: str | None = None,
        bias_dtype: str | None = None,
        mma_dtype: str | None = None,
        accum_dtype: str | None = None,
        output_dtype: str | None = None,
        projection_policies: Mapping[str, Mapping[str, str] | None] | None = None,
        fusion: bool | FeedForwardFusionConfig | None = None,
    ) -> None:
        if engine is not None:
            resolved_engine = _resolve_engine_alias(
                engine=engine,
                context="FeedForward runtime",
            )
            if resolved_engine not in {"torch", "triton"}:
                raise ValueError(f"unsupported engine: {resolved_engine}")
            self.engine = resolved_engine
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
            self.runtime_precision[key] = _canonical_precision_name(value)
        if projection_policies is not None:
            for name, policy in projection_policies.items():
                self._update_projection_precision(name, policy)
        if fusion is not None:
            self.fusion_config = (
                fusion
                if isinstance(fusion, FeedForwardFusionConfig)
                else FeedForwardFusionConfig(enabled=bool(fusion))
            )
        for name in self._projection_names():
            projection = getattr(self, name)
            if isinstance(projection, Linear):
                policy = self._effective_projection_precision(name)
                projection.configure_runtime(
                    engine=self.engine,
                    activation_dtype=policy["activation"],
                    weight_dtype=policy["weight"],
                    bias_dtype=policy["bias"],
                    mma_dtype=policy["mma"],
                    accum_dtype=policy["accum"],
                    output_dtype=policy["output"],
                )
        if isinstance(self.norm, LayerNorm):
            self.norm.configure_runtime(engine=self.engine, **{
                f"{name}_dtype": value
                for name, value in self.runtime_precision.items()
            })

    def runtime_config(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "activation": self.runtime_precision["activation"],
            "weight": self.runtime_precision["weight"],
            "bias": self.runtime_precision["bias"],
            "mma": self.runtime_precision["mma"],
            "accum": self.runtime_precision["accum"],
            "output": self.runtime_precision["output"],
            "fusion": self._fusion_report(),
            "projections": {
                name: self._effective_projection_precision(name)
                for name in self._projection_names()
            },
            "fallback": (
                None if self.runtime_fallback is None else dict(self.runtime_fallback)
            ),
            "fallback_count": self.runtime_fallback_count,
        }

    def _record_runtime_fallback(
        self,
        *,
        stage: str,
        reason: Exception,
    ) -> None:
        self.runtime_fallback_count += 1
        self.runtime_fallback = {
            "engine": self.engine,
            "stage": stage,
            "reason": str(reason),
        }

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            return x
        return self.norm(x)

    def _projection_names(self) -> tuple[ProjectionName, ...]:
        names: list[ProjectionName] = ["proj_in"]
        if self.proj_gate is not None:
            names.append("proj_gate")
        names.append("proj_out")
        return tuple(names)

    def _update_projection_precision(
        self,
        name: str,
        policy: Mapping[str, str] | None,
    ) -> None:
        projection_name = str(name).strip().lower()
        if projection_name not in self.projection_precision:
            available = ", ".join(self._projection_names())
            raise ValueError(
                f"unsupported FFN projection policy: {name}. Available: {available}"
            )
        if policy is None:
            self.projection_precision[projection_name] = _default_runtime_precision()
            return
        for key, value in policy.items():
            canonical_key = _canonical_precision_key(key)
            self.projection_precision[projection_name][canonical_key] = (
                _canonical_precision_name(value)
            )

    def _effective_projection_precision(self, name: ProjectionName) -> dict[str, str]:
        effective = dict(self.runtime_precision)
        if name == "proj_out":
            effective["output"] = self.runtime_precision["output"]
        else:
            effective["output"] = self.runtime_precision["activation"]
        overrides = self.projection_precision[name]
        for key, value in overrides.items():
            if value == "auto":
                continue
            effective[key] = value
        return effective

    def _preferred_compute_precision_name(
        self,
        precision: Mapping[str, str],
        fallback: torch.dtype,
    ) -> str:
        mma_name = precision["mma"]
        if mma_name != "auto":
            canonical_mma = _canonical_precision_name(mma_name)
            if canonical_mma not in _TORCH_DTYPE_PRECISION_NAMES:
                raise ValueError(f"mma precision {mma_name} has no torch dtype runtime")
            return mma_name
        for key in ("activation", "weight"):
            name = precision[key]
            if (
                name != "auto"
                and _canonical_precision_name(name) in _TORCH_DTYPE_PRECISION_NAMES
            ):
                return name
        return _precision_name_from_dtype(fallback)

    def fusion_intent(self) -> FusionIntent:
        realized_patterns: list[str] = []
        config = self.fusion_config
        if config.enabled:
            if (
                self.activation in {"gelu", "linear-silu"}
                and config.fuse_linear_activation
            ):
                realized_patterns.append(f"proj_in_{self.activation}_epilogue")
            if self.activation in {"swiglu", "geglu"} and config.fuse_gate:
                realized_patterns.append(self.activation)
            if self.norm is not None and config.fuse_norm:
                realized_patterns.append(f"{self.norm_kind}_norm_requested")
            if config.fuse_output:
                realized_patterns.append("proj_out_epilogue_requested")
        epilogue: list[str] = []
        if self.dropout_p > 0.0:
            epilogue.append("dropout")
        if self.final_dropout_enabled:
            epilogue.append("final_dropout")
        return FusionIntent(
            patterns=tuple(realized_patterns),
            epilogue=tuple(epilogue),
        )

    def _fusion_report(self) -> dict[str, Any]:
        intent = self.fusion_intent()
        return {
            **self.fusion_config.to_dict(),
            "realized_patterns": list(intent.patterns),
            "epilogue": list(intent.epilogue),
        }

    def _linear_epilogue(
        self,
        x: torch.Tensor,
        linear: nn.Linear,
        *,
        activation: str | None = None,
        precision: Mapping[str, str] | None = None,
    ) -> torch.Tensor:
        flat = x.reshape(-1, int(x.shape[-1]))
        runtime_precision = dict(self.runtime_precision)
        if precision is not None:
            runtime_precision = _default_runtime_precision()
            for key, value in precision.items():
                runtime_precision[_canonical_precision_key(key)] = (
                    _canonical_precision_name(value)
                )
        compute_name = self._preferred_compute_precision_name(
            runtime_precision, x.dtype
        )
        compute_dtype = _precision_name_to_dtype(compute_name, x.dtype)
        accum_name = runtime_precision["accum"]
        output_name = _compute_output_precision_name(runtime_precision, x.dtype)
        accum_dtype = _precision_name_to_dtype(
            accum_name,
            torch.float32,
            allow_low_bit_fallback=False,
            role="accum",
        )
        output_dtype = _precision_name_to_dtype(
            output_name,
            x.dtype,
            allow_low_bit_fallback=False,
            role="output",
        )
        bias_dtype = _precision_name_to_dtype(runtime_precision["bias"], output_dtype)
        flat_compute = flat.to(device=x.device, dtype=compute_dtype)
        weight = linear.weight.to(device=x.device, dtype=compute_dtype)
        bias = (
            None
            if linear.bias is None
            else linear.bias.to(device=x.device, dtype=bias_dtype)
        )
        if self.engine == "triton":
            try:
                if compute_dtype == torch.float16:
                    out = gemm_fp16_triton(
                        flat_compute,
                        weight,
                        bias,
                        activation=activation,
                        transpose_b=True,
                        accum_dtype=accum_dtype,
                        output_dtype=output_dtype,
                    )
                elif compute_dtype == torch.bfloat16:
                    out = gemm_bf16_triton(
                        flat_compute,
                        weight,
                        bias,
                        activation=activation,
                        transpose_b=True,
                        accum_dtype=accum_dtype,
                        output_dtype=output_dtype,
                    )
                else:
                    raise RuntimeError(
                        f"unsupported Triton compute dtype for FFN: {compute_name}"
                    )
                return out.reshape(*x.shape[:-1], int(out.shape[-1]))
            except Exception as exc:
                self._record_runtime_fallback(
                    stage="linear_epilogue",
                    reason=exc,
                )
        out = dense_gemm_reference(
            flat_compute,
            weight,
            bias,
            activation=activation,
            transpose_b=True,
        )
        out = out.to(dtype=output_dtype)
        return out.reshape(*x.shape[:-1], int(out.shape[-1]))

    def _fused_gate(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        *,
        pattern: str,
    ) -> torch.Tensor:
        if self.engine == "triton":
            try:
                return run_triton_kernel(pattern, gate, up)
            except Exception as exc:
                self._record_runtime_fallback(
                    stage=f"{pattern}_gate",
                    reason=exc,
                )
        if pattern == "swiglu":
            return F.silu(gate) * up
        if pattern == "geglu":
            return F.gelu(gate) * up
        raise ValueError(f"unsupported gate pattern: {pattern}")

    def _activation_hidden(self, x: torch.Tensor) -> torch.Tensor:
        proj_in_precision = self._effective_projection_precision("proj_in")
        hidden_precision = _compute_output_precision_name(proj_in_precision, x.dtype)
        if self.activation == "gelu":
            proj_in_precision["output"] = hidden_precision
            if self.fusion_config.enabled and self.fusion_config.fuse_linear_activation:
                return self._linear_epilogue(
                    x,
                    self.proj_in,
                    activation="gelu",
                    precision=proj_in_precision,
                )
            return F.gelu(
                self._linear_epilogue(
                    x,
                    self.proj_in,
                    activation=None,
                    precision=proj_in_precision,
                )
            )
        if self.activation == "gelu-approximate":
            proj_in_precision["output"] = hidden_precision
            hidden = self._linear_epilogue(
                x,
                self.proj_in,
                activation=None,
                precision=proj_in_precision,
            )
            return F.gelu(hidden, approximate="tanh")
        if self.activation == "geglu-approximate":
            proj_in_precision["output"] = hidden_precision
            hidden = self._linear_epilogue(
                x,
                self.proj_in,
                activation=None,
                precision=proj_in_precision,
            )
            return hidden * torch.sigmoid(1.702 * hidden)
        if self.activation == "linear-silu":
            proj_in_precision["output"] = hidden_precision
            if self.fusion_config.enabled and self.fusion_config.fuse_linear_activation:
                return self._linear_epilogue(
                    x,
                    self.proj_in,
                    activation="silu",
                    precision=proj_in_precision,
                )
            return F.silu(
                self._linear_epilogue(
                    x,
                    self.proj_in,
                    activation=None,
                    precision=proj_in_precision,
                )
            )
        if self.proj_gate is None:
            raise RuntimeError("gated activation requires proj_gate")
        proj_gate_precision = self._effective_projection_precision("proj_gate")
        proj_in_precision["output"] = hidden_precision
        proj_gate_precision["output"] = _compute_output_precision_name(
            proj_gate_precision, x.dtype
        )
        up = self._linear_epilogue(
            x,
            self.proj_in,
            activation=None,
            precision=proj_in_precision,
        )
        gate = self._linear_epilogue(
            x,
            self.proj_gate,
            activation=None,
            precision=proj_gate_precision,
        )
        if self.activation == "swiglu":
            if self.fusion_config.enabled and self.fusion_config.fuse_gate:
                return self._fused_gate(gate, up, pattern="swiglu")
            return F.silu(gate) * up
        if self.activation == "geglu":
            if self.fusion_config.enabled and self.fusion_config.fuse_gate:
                return self._fused_gate(gate, up, pattern="geglu")
            return F.gelu(gate) * up
        raise ValueError(f"unsupported activation: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._apply_norm(x)
        hidden = self._activation_hidden(hidden)
        hidden = self.dropout(hidden)
        proj_out_precision = self._effective_projection_precision("proj_out")
        out = self._linear_epilogue(
            hidden,
            self.proj_out,
            activation=None,
            precision=proj_out_precision,
        )
        if self.final_dropout is not None:
            out = self.final_dropout(out)
        return out
