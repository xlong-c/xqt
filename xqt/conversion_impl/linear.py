"""Linear conversion mixin for _ModuleConverter.

Provides contract building, lowering, and materialization for linear,
conv2d, and layernorm modules.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from xqt.contracts import OperatorContract, TensorStorageSpec
from xqt.core.errors import XQTBackendError
import xqt.operator_opt as _opt
from xqt.quant import FP4WeightOnlyLinear, infer_nvfp4_tensor_layout

from .runtime_linear import _RuntimeLinearModule


class _LinearConversionMixin:
    """Provide linear/conv2d/layernorm conversion to _ModuleConverter."""

    # ------------------------------------------------------------------
    # Linear contract & conversion
    # ------------------------------------------------------------------

    def _build_linear_contract(self, module: nn.Module) -> OperatorContract:
        if isinstance(module, FP4WeightOnlyLinear):
            weight_spec = TensorStorageSpec(
                storage_dtype="fp4_packed",
                logical_dtype=self.policy.weight,
                layout="row_major_grouped",
                packed=True,
                group_size=int(module.group_size),
                scale_dtype="fp32",
                scale_layout="per_group",
            )
            metadata = {
                "source_module_type": type(module).__name__,
                "input_features": int(module.input_features),
                "output_features": int(module.output_features),
                "path": "fp4_weight_only",
            }
        else:
            nvfp4_layout = infer_nvfp4_tensor_layout(module)
            if nvfp4_layout is not None:
                weight_spec = TensorStorageSpec(
                    storage_dtype="nvfp4_packed",
                    logical_dtype=self.policy.weight,
                    layout="row_major_grouped",
                    packed=True,
                    group_size=int(nvfp4_layout.group_size),
                    scale_dtype="fp32",
                    scale_layout="per_group",
                )
                metadata = {
                    "source_module_type": type(module).__name__,
                    "input_features": int(nvfp4_layout.input_features),
                    "output_features": int(nvfp4_layout.output_features),
                    "path": "bridgeable_nvfp4",
                }
            elif isinstance(module, nn.Linear):
                weight_spec = TensorStorageSpec(
                    storage_dtype=self.policy.weight,
                    logical_dtype=self.policy.weight,
                    layout="row_major_dense",
                    packed=False,
                )
                metadata = {
                    "source_module_type": type(module).__name__,
                    "input_features": int(module.in_features),
                    "output_features": int(module.out_features),
                    "path": "dense_linear",
                }
            else:
                raise XQTBackendError(
                    f"Unsupported linear module type for xqt.convert: {type(module).__name__}"
                )
        return OperatorContract(
            operator_kind="linear",
            policy=self.policy,
            input_spec=TensorStorageSpec(
                storage_dtype=self.policy.activation,
                logical_dtype=self.policy.activation,
                layout="row_major_dense",
            ),
            weight_spec=weight_spec,
            output_dtype=self.policy.output,
            epilogue=("bias",) if getattr(module, "bias", None) is not None else (),
            metadata=metadata,
        )

    def _convert_linear(
        self, module: nn.Module, contract: OperatorContract
    ) -> "ConvertResult":
        from xqt.conversion import ConvertResult

        if self.engine == "torch":
            if not isinstance(module, nn.Linear) or not self.policy_was_explicit:
                return self._result(
                    model=module,
                    contract=contract,
                    converted=False,
                    report={
                        "reason": "torch engine keeps the original Linear-compatible implementation",
                    },
                )
        if isinstance(module, nn.Linear) and self.engine in {"torch", "triton"}:
            runtime_linear = _RuntimeLinearModule(
                module,
                engine=self.engine,
                precision=self.policy,
            )
            converted = self.engine == "triton"
            reason = (
                "torch engine uses a runtime-configured Linear wrapper"
                if self.engine == "torch"
                else "triton engine uses a runtime-configured Linear wrapper"
            )
            return self._result(
                model=runtime_linear,
                contract=contract,
                converted=converted,
                report={
                    "reason": reason,
                    "runtime_config": runtime_linear.runtime_config(),
                },
            )
        target_plan = self._lower_linear_contract_to_target_plan(contract)
        converted_module, _ = _opt.materialize_module(
            module,
            contract=contract,
            target=target_plan,
        )
        return self._result(
            model=converted_module,
            contract=contract,
            converted=True,
            report={"target_plan": target_plan.to_dict()},
        )

    def _build_conv2d_contract(self, module: nn.Conv2d) -> OperatorContract:
        return OperatorContract(
            operator_kind="conv2d",
            policy=self.policy,
            input_spec=TensorStorageSpec(
                storage_dtype=self.policy.activation,
                logical_dtype=self.policy.activation,
                layout="nchw_dense",
            ),
            weight_spec=TensorStorageSpec(
                storage_dtype=self.policy.weight,
                logical_dtype=self.policy.weight,
                layout="oihw_dense",
            ),
            output_dtype=self.policy.output,
            epilogue=("bias",) if module.bias is not None else (),
            metadata={
                "source_module_type": type(module).__name__,
                "kernel_size": tuple(int(v) for v in module.kernel_size),
                "stride": tuple(int(v) for v in module.stride),
                "padding": tuple(int(v) for v in module.padding),
            },
        )

    def _convert_conv2d(
        self, module: nn.Conv2d, contract: OperatorContract
    ) -> "ConvertResult":
        from xqt.conversion import ConvertResult

        if self.engine == "torch":
            return self._result(
                model=module,
                contract=contract,
                converted=False,
                report={
                    "reason": "torch engine keeps the original Conv2d implementation",
                },
            )
        if self.engine != "tilelang":
            raise XQTBackendError(
                f"xqt.convert Conv2d currently supports only engine='tilelang', got {self.engine}"
            )
        target_plan = _opt.OperatorOptimizationTargetPlan(
            name=f"{type(module).__name__}_{self.engine}",
            engine=self.engine,
            target_path=None,
            patterns=["conv"],
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
            report={"target_plan": target_plan.to_dict()},
        )

    def _build_layernorm_contract(self, module: nn.LayerNorm) -> OperatorContract:
        return OperatorContract(
            operator_kind="layernorm",
            policy=self.policy,
            input_spec=TensorStorageSpec(
                storage_dtype=self.policy.activation,
                logical_dtype=self.policy.activation,
                layout="last_dim_dense",
            ),
            weight_spec=TensorStorageSpec(
                storage_dtype=self.policy.weight,
                logical_dtype=self.policy.weight,
                layout="vector_dense",
            ),
            output_dtype=self.policy.output,
            epilogue=(),
            metadata={
                "source_module_type": type(module).__name__,
                "normalized_shape": tuple(int(v) for v in module.normalized_shape),
                "eps": float(module.eps),
            },
        )

    def _convert_layernorm(
        self, module: nn.LayerNorm, contract: OperatorContract
    ) -> "ConvertResult":
        from xqt.conversion import ConvertResult

        if self.engine == "torch":
            return self._result(
                model=module,
                contract=contract,
                converted=False,
                report={
                    "reason": "torch engine keeps the original LayerNorm implementation",
                },
            )
        if self.engine != "tilelang":
            raise XQTBackendError(
                f"xqt.convert LayerNorm currently supports only engine='tilelang', got {self.engine}"
            )
        target_plan = _opt.OperatorOptimizationTargetPlan(
            name=f"{type(module).__name__}_{self.engine}",
            engine=self.engine,
            target_path=None,
            patterns=["norm"],
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
            report={"target_plan": target_plan.to_dict()},
        )

    # ------------------------------------------------------------------
    # Linear lowering helpers
    # ------------------------------------------------------------------

    def _lower_linear_contract_to_target_plan(
        self,
        contract: OperatorContract,
    ) -> _opt.OperatorOptimizationTargetPlan:
        patterns = self._linear_patterns_for_contract(contract)
        target_kwargs: dict[str, Any] = {
            "name": f"{contract.metadata.get('source_module_type', 'Linear')}_{self.engine}",
            "engine": self.engine,
            "target_path": None,
            "patterns": patterns,
            "fallback": self.fallback,
            "min_speedup": 0.0,
            "validate": {"atol": 1e-2, "rtol": 1e-2},
        }
        if self.engine == "tilelang":
            target_kwargs["tilelang"] = {
                "target": self.target,
                "target_arch": self.target_arch,
                "linear_runtime": "tilelang"
                if contract.weight_spec.packed
                and contract.weight_spec.storage_dtype == "nvfp4_packed"
                else "auto",
                "linear_fastpath": "auto",
            }
        elif self.engine in {"cutile", "cute_dsl"}:
            key = "cutile" if self.engine == "cutile" else "cute_dsl"
            target_kwargs[key] = {
                "target": self.target,
                "target_arch": self.target_arch,
            }
        else:
            raise XQTBackendError(
                f"xqt.convert Linear currently supports engines tilelang/cutile/cute_dsl/torch, got {self.engine}"
            )
        return _opt.OperatorOptimizationTargetPlan(**target_kwargs)

    def _linear_patterns_for_contract(self, contract: OperatorContract) -> list[str]:
        weight_storage = contract.weight_spec.storage_dtype
        if self.engine == "tilelang":
            if weight_storage == "nvfp4_packed":
                return ["nvfp4_packed_dequant_gemm_epilogue"]
            if weight_storage == "fp4_packed":
                return ["fp4_packed_dequant_gemm_epilogue"]
            if weight_storage == "fp16":
                return ["linear"]
            return ["dequant_gemm_epilogue"]
        if self.engine in {"cutile", "cute_dsl"}:
            if weight_storage == "nvfp4_packed":
                return ["nvfp4_packed_dequant_gemm_epilogue"]
            raise XQTBackendError(
                f"xqt.convert engine={self.engine} currently supports only NVFP4 packed Linear contracts"
            )
        raise XQTBackendError(f"Unsupported Linear engine: {self.engine}")
