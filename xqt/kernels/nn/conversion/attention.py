"""Attention conversion mixin for _ModuleConverter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

import xqt.kernels.nn as xqt_nn
from xqt.kernels.precision import OperatorContract, TensorStorageSpec
from xqt.core.errors import XQTBackendError
import xqt.kernels.wrappers as _opt

if TYPE_CHECKING:
    from xqt.kernels.nn.convert import ConvertResult


class _AttentionConversionMixin:
    """Provide attention contract building and conversion to _ModuleConverter."""

    def _build_attention_contract(
        self, module: xqt_nn.Attention
    ) -> OperatorContract:
        return OperatorContract(
            operator_kind="attention",
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
            metadata={
                "source_module_type": type(module).__name__,
                "dim": int(module.dim),
                "dim_out": int(module.dim_out),
                "heads": int(module.heads),
                "head_dim": int(module.head_dim),
                "causal": bool(module.causal),
                "dropout": float(module.dropout_p),
                "engine": module.engine,
            },
        )

    def _convert_attention(
        self,
        module: xqt_nn.Attention,
        contract: OperatorContract,
    ) -> "ConvertResult":
        from xqt.kernels.nn.convert import ConvertResult

        if self.engine not in {"torch", "tilelang"}:
            raise XQTBackendError(
                "xqt.convert Attention currently supports only "
                f"engine='torch' or engine='tilelang', got {self.engine}"
            )
        module.configure_runtime(
            engine=self.engine,
            activation_dtype=self.policy.activation,
            weight_dtype=self.policy.weight,
            bias_dtype=self.policy.bias,
            mma_dtype=self.policy.mma,
            accum_dtype=self.policy.accum,
            output_dtype=self.policy.output,
        )
        if self.engine == "torch":
            return self._result(
                model=module,
                contract=contract,
                converted=False,
                report={
                    "reason": "torch engine keeps the Attention facade with SDPA forward",
                    "runtime_config": module.runtime_config(),
                },
            )
        target_plan = _opt.OperatorOptimizationTargetPlan(
            name=f"{type(module).__name__}_{self.engine}",
            engine=self.engine,
            target_path=None,
            patterns=["attention"],
            fallback=self.fallback,
            min_speedup=0.0,
            validate={"atol": 1e-2, "rtol": 1e-2},
            tilelang={
                "target": self.target,
                "target_arch": self.target_arch,
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
                "reason": "Attention candidate materialized from shared module contract",
                "target_plan": target_plan.to_dict(),
            },
        )
