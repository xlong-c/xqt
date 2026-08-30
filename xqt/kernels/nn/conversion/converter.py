"""Internal converter class - thin orchestrator assembled from target-aware mixins."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Mapping

from torch import nn

from xqt.kernels.precision import OperatorContract, PrecisionPolicy
from xqt.core.errors import XQTBackendError

import xqt.kernels.nn as xqt_nn

from .attention import _AttentionConversionMixin
from .block import _TransformerBlockConversionMixin
from .feedforward import _FeedForwardConversionMixin
from .linear import _LinearConversionMixin

if TYPE_CHECKING:
    from xqt.kernels.nn.convert import ConvertResult


class _ModuleConverter(
    _LinearConversionMixin,
    _FeedForwardConversionMixin,
    _AttentionConversionMixin,
    _TransformerBlockConversionMixin,
):
    """Internal stateful converter behind the public ``xqt.convert`` facade."""

    def __init__(
        self,
        *,
        engine: str,
        target: str,
        policy: PrecisionPolicy,
        projection_policies: Mapping[str, PrecisionPolicy | Mapping[str, str]]
        | None = None,
        fallback: str = "eager",
        target_arch: str | None = None,
        inplace: bool = False,
        policy_was_explicit: bool = False,
    ) -> None:
        self.engine = str(engine).strip().lower()
        self.target = str(target).strip().lower()
        self.policy = policy
        self.projection_policies = projection_policies
        self.fallback = fallback
        self.target_arch = target_arch
        self.inplace = inplace
        self.policy_was_explicit = policy_was_explicit

    def convert_module(self, module: nn.Module) -> "ConvertResult":
        from xqt.kernels.nn.convert import ConvertResult

        working = module if self.inplace else copy.deepcopy(module)
        contract = self._build_contract(working)
        if (
            self.projection_policies is not None
            and contract.operator_kind != "feedforward"
        ):
            raise XQTBackendError(
                "xqt.convert projection_policies are supported only for FeedForward modules"
            )
        if contract.operator_kind == "linear":
            return self._convert_linear(working, contract)
        if contract.operator_kind == "conv2d":
            return self._convert_conv2d(working, contract)
        if contract.operator_kind == "layernorm":
            return self._convert_layernorm(working, contract)
        if contract.operator_kind == "feedforward":
            return self._convert_feedforward(working, contract)
        if contract.operator_kind == "attention":
            return self._convert_attention(working, contract)
        if contract.operator_kind == "transformer_block":
            return self._convert_transformer_block(working, contract)
        raise XQTBackendError(
            f"Unsupported conversion operator kind: {contract.operator_kind}"
        )

    def _build_contract(self, module: nn.Module) -> OperatorContract:
        from xqt.compression.quant import FP4WeightOnlyLinear, infer_nvfp4_tensor_layout

        if (
            isinstance(module, (nn.Linear, FP4WeightOnlyLinear))
            or infer_nvfp4_tensor_layout(module) is not None
        ):
            return self._build_linear_contract(module)
        if isinstance(module, nn.Conv2d):
            return self._build_conv2d_contract(module)
        if isinstance(module, nn.LayerNorm):
            return self._build_layernorm_contract(module)
        if isinstance(module, xqt_nn.FeedForward):
            return self._build_feedforward_contract(module)
        if isinstance(module, xqt_nn.Attention):
            return self._build_attention_contract(module)
        if isinstance(module, xqt_nn.TransformerBlock):
            return self._build_transformer_block_contract(module)
        raise XQTBackendError(
            "xqt.convert currently supports Linear, Conv2d, LayerNorm, FeedForward, "
            "Attention, TransformerBlock, FP4WeightOnlyLinear, and bridgeable NVFP4 Linear modules"
        )

    def _attach_contract(
        self,
        module: nn.Module,
        contract: OperatorContract,
    ) -> nn.Module:
        setattr(module, "_xqt_module_contract", contract.to_dict())
        return module

    def _result(
        self,
        *,
        model: nn.Module,
        contract: OperatorContract,
        converted: bool,
        report: dict[str, Any],
    ) -> "ConvertResult":
        from xqt.kernels.nn.convert import ConvertResult

        model = self._attach_contract(model, contract)
        payload = dict(report)
        payload.setdefault("contract", contract.to_dict())
        payload.setdefault("engine", self.engine)
        payload.setdefault("converted", converted)
        return ConvertResult(
            model=model,
            engine=self.engine,
            target=self.target,
            contract=contract,
            converted=converted,
            report=payload,
        )
