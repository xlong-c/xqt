"""TransformerBlock conversion mixin for _ModuleConverter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

import xqt.nn as xqt_nn
from xqt.contracts import OperatorContract, TensorStorageSpec
from xqt.core.errors import XQTBackendError
import xqt.operator_opt as _opt

if TYPE_CHECKING:
    from xqt.conversion import ConvertResult


class _TransformerBlockConversionMixin:
    """Provide transformer block contract building and conversion to _ModuleConverter."""

    def _build_transformer_block_contract(
        self, module: xqt_nn.TransformerBlock
    ) -> OperatorContract:
        return OperatorContract(
            operator_kind="transformer_block",
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
                "norm": "none" if module.norm_kind is None else str(module.norm_kind),
                "ffn_activation": str(module.ffn.activation),
                "engine": module.engine,
            },
        )

    def _convert_transformer_block(
        self,
        module: xqt_nn.TransformerBlock,
        contract: OperatorContract,
    ) -> "ConvertResult":
        from xqt.conversion import ConvertResult

        if self.engine not in {"torch", "tilelang"}:
            raise XQTBackendError(
                "xqt.convert TransformerBlock currently supports only "
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
                    "reason": "torch engine keeps the TransformerBlock facade composition",
                    "runtime_config": module.runtime_config(),
                },
            )
        target_plan = _opt.OperatorOptimizationTargetPlan(
            name=f"{type(module).__name__}_attn_{self.engine}",
            engine=self.engine,
            target_path="attn",
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
                "reason": (
                    "TransformerBlock tilelang: internal Attention materialized via "
                    "shared module contract; full block-level single-kernel fusion "
                    "remains future work"
                ),
                "target_plan": target_plan.to_dict(),
                "runtime_config": (
                    converted_module.runtime_config()
                    if hasattr(converted_module, "runtime_config")
                    else module.runtime_config()
                ),
            },
        )
