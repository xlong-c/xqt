"""FeedForward conversion mixin for _ModuleConverter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import xqt.kernels.nn as xqt_nn
from xqt.kernels.precision import (
    FusionIntent,
    OperatorContract,
    PrecisionPolicy,
    TensorStorageSpec,
)
from xqt.core.errors import XQTBackendError
import xqt.kernels.wrappers as _opt

from .precision import _projection_precision_dict

if TYPE_CHECKING:
    from xqt.kernels.nn.convert import ConvertResult


class _FeedForwardConversionMixin:
    """Provide feedforward contract building and conversion to _ModuleConverter."""

    def _build_feedforward_contract(
        self, module: xqt_nn.FeedForward
    ) -> OperatorContract:
        norm_kind = "none" if module.norm is None else str(module.norm_kind)
        has_gate = module.proj_gate is not None
        fusion = module.fusion_intent()
        runtime_fusion = module.runtime_config()["fusion"]
        return OperatorContract(
            operator_kind="feedforward",
            policy=self.policy,
            input_spec=TensorStorageSpec(
                storage_dtype=self.policy.activation,
                logical_dtype=self.policy.activation,
                layout="row_major_dense",
            ),
            weight_spec=TensorStorageSpec(
                storage_dtype=self.policy.weight,
                logical_dtype=self.policy.weight,
                layout="row_major_dense",
            ),
            output_dtype=self.policy.output,
            epilogue=fusion.epilogue,
            fusion=fusion,
            metadata={
                "source_module_type": type(module).__name__,
                "dim": int(module.dim),
                "inner_dim": int(module.inner_dim),
                "dim_out": int(module.dim_out),
                "activation": str(module.activation),
                "norm": norm_kind,
                "has_gate": has_gate,
                "engine": module.engine,
                "fusion": runtime_fusion,
            },
        )

    def _convert_feedforward(
        self,
        module: xqt_nn.FeedForward,
        contract: OperatorContract,
    ) -> "ConvertResult":
        from xqt.kernels.nn.convert import ConvertResult

        if self.engine not in {"torch", "triton"}:
            raise XQTBackendError(
                "xqt.convert FeedForward currently supports only "
                f"engine='torch' or engine='triton', got {self.engine}"
            )
        if self.engine == "torch":
            module.configure_runtime(
                engine=self.engine,
                activation_dtype=self.policy.activation,
                weight_dtype=self.policy.weight,
                bias_dtype=self.policy.bias,
                mma_dtype=self.policy.mma,
                accum_dtype=self.policy.accum,
                output_dtype=self.policy.output,
                projection_policies=self._feedforward_projection_policies(),
            )
            return self._result(
                model=module,
                contract=contract,
                converted=False,
                report={
                    "reason": "torch engine keeps the original FeedForward implementation",
                    "runtime_config": module.runtime_config(),
                    "fusion": module.runtime_config()["fusion"],
                },
            )
        target_plan = _opt.OperatorOptimizationTargetPlan(
            name=f"{type(module).__name__}_{self.engine}",
            engine=self.engine,
            target_path=None,
            patterns=["feedforward"],
            fallback=self.fallback,
            min_speedup=0.0,
            validate={"atol": 1e-2, "rtol": 1e-2},
            options={
                "activation_dtype": self.policy.activation,
                "weight_dtype": self.policy.weight,
                "bias_dtype": self.policy.bias,
                "mma_dtype": self.policy.mma,
                "accum_dtype": self.policy.accum,
                "output_dtype": self.policy.output,
                "projection_policies": self._feedforward_projection_policies(),
            },
        )
        converted_module, _ = _opt.materialize_module(
            module,
            contract=contract,
            target=target_plan,
        )
        return self._result(
            model=converted_module,
            contract=contract,
            converted=True,
            report={
                "reason": "Triton FeedForward candidate materialized from shared module contract",
                "runtime_config": converted_module.runtime_config(),
                "fusion": converted_module.runtime_config()["fusion"],
                "target_plan": target_plan.to_dict(),
            },
        )

    def _feedforward_projection_policies(self) -> dict[str, dict[str, str]] | None:
        if self.projection_policies is None:
            return None
        return _projection_precision_dict(self.projection_policies)
