"""Module conversion facade for operator-oriented XQT engines."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Literal

import torch
from torch.nn import functional as F
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.contracts import (
    FeedForwardPrecisionPolicy,
    FusionIntent,
    OperatorContract,
    PrecisionPolicy,
    TensorStorageSpec,
)
from xqt import nn as xqt_nn
from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    materialize_module,
)
from xqt.operator_opt.backends.gemm_precision import (
    MatmulPrecisionSpec,
    gemm_with_precision,
)
from xqt.quant import (
    FP4WeightOnlyLinear,
    infer_nvfp4_tensor_layout,
)


EngineKind = Literal["torch", "triton", "tilelang", "cutile", "cute_dsl"]


def _resolve_engine_alias(
    *,
    engine: str | None,
    context: str,
) -> str:
    if engine is None:
        raise XQTBackendError(f"{context} requires engine=...")
    return str(engine).strip().lower()


def _runtime_precision_dict(
    policy: PrecisionPolicy | MatmulPrecisionSpec | Mapping[str, str],
) -> dict[str, str]:
    if isinstance(policy, PrecisionPolicy):
        return policy.to_dict()
    if isinstance(policy, FeedForwardPrecisionPolicy):
        return policy.default.to_dict()
    if isinstance(policy, MatmulPrecisionSpec):
        return policy.to_dict()
    return MatmulPrecisionSpec.from_roles(**_matmul_role_kwargs(policy)).to_dict()


def _projection_precision_dict(
    policy: PrecisionPolicy
    | MatmulPrecisionSpec
    | Mapping[str, str]
    | FeedForwardPrecisionPolicy,
) -> dict[str, dict[str, str]] | None:
    if isinstance(policy, FeedForwardPrecisionPolicy):
        return {
            name: projection_policy.to_dict()
            for name, projection_policy in policy.projection_policies().items()
        }
    if isinstance(policy, PrecisionPolicy):
        return None
    return {
        str(name): _projection_policy_dict(projection_policy)
        for name, projection_policy in policy.items()
    }


def _precision_policy_from_matmul_spec(policy: MatmulPrecisionSpec) -> PrecisionPolicy:
    return PrecisionPolicy(
        activation=policy.activation,
        weight=policy.weight,
        bias=policy.bias,
        mma=policy.mma,
        accum=policy.accum,
        output=policy.output,
    )


def _precision_policy_from_mapping(policy: Mapping[str, str]) -> PrecisionPolicy:
    return _precision_policy_from_matmul_spec(
        MatmulPrecisionSpec.from_roles(**_matmul_role_kwargs(policy))
    )


def _matmul_spec_from_precision_policy(policy: PrecisionPolicy) -> MatmulPrecisionSpec:
    return MatmulPrecisionSpec(
        activation=policy.activation,
        weight=policy.weight,
        bias=policy.bias,
        mma=policy.mma,
        accum=policy.accum,
        output=policy.output,
    )


def _matmul_role_kwargs(policy: Mapping[str, str]) -> dict[str, str]:
    key_aliases = {
        "a": "A",
        "activation": "activation",
        "activation_dtype": "activation",
        "input": "activation",
        "lhs": "activation",
        "b": "B",
        "weight": "weight",
        "weight_dtype": "weight",
        "rhs": "weight",
        "c": "C",
        "bias": "bias",
        "bias_dtype": "bias",
        "addend": "bias",
        "addend_dtype": "bias",
        "mma": "mma",
        "mma_dtype": "mma",
        "acc": "accum",
        "accum": "accum",
        "accum_dtype": "accum",
        "accumulator": "accum",
        "accumulator_dtype": "accum",
        "o": "O",
        "out": "output",
        "output": "output",
        "output_dtype": "output",
    }
    payload: dict[str, str] = {}
    for key, value in policy.items():
        normalized = str(key).strip().lower()
        try:
            canonical_key = key_aliases[normalized]
        except KeyError as exc:
            allowed = ", ".join(sorted(key_aliases))
            raise XQTBackendError(
                f"unsupported precision policy field: {key}. Allowed: {allowed}"
            ) from exc
        payload[canonical_key] = str(value)
    return payload


def _projection_policy_dict(
    policy: PrecisionPolicy | MatmulPrecisionSpec | Mapping[str, str],
) -> dict[str, str]:
    if isinstance(policy, PrecisionPolicy):
        return policy.to_dict()
    if isinstance(policy, MatmulPrecisionSpec):
        return policy.to_dict()
    role_kwargs = _matmul_role_kwargs(policy)
    key_aliases = {
        "A": "activation",
        "B": "weight",
        "C": "bias",
        "O": "output",
    }
    return {key_aliases.get(key, key): str(value) for key, value in role_kwargs.items()}


@dataclass(frozen=True)
class ConvertResult:
    """Structured conversion result for one module."""

    model: nn.Module
    engine: str
    target: str
    contract: OperatorContract
    converted: bool
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "target": self.target,
            "contract": self.contract.to_dict(),
            "converted": self.converted,
            "report": dict(self.report),
        }


class _RuntimeLinearModule(nn.Module):
    """Runtime-configured Linear facade backed by the unified GEMM dispatcher."""

    def __init__(
        self,
        module: nn.Linear,
        *,
        engine: str,
        precision: MatmulPrecisionSpec,
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
        precision: MatmulPrecisionSpec | Mapping[str, str] | None = None,
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
            precision=MatmulPrecisionSpec(**self.runtime_precision),
            engine=self.engine,
            transpose_b=True,
        )
        return output.reshape(*prefix_shape, int(output.shape[-1]))


class _ModuleConverter:
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

    def convert_module(self, module: nn.Module) -> ConvertResult:
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
        raise XQTBackendError(
            f"Unsupported conversion operator kind: {contract.operator_kind}"
        )

    def _build_contract(self, module: nn.Module) -> OperatorContract:
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
        raise XQTBackendError(
            "xqt.convert currently supports Linear, Conv2d, LayerNorm, FeedForward, FP4WeightOnlyLinear, and bridgeable NVFP4 Linear modules"
        )

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

    def _build_feedforward_contract(
        self, module: xqt_nn.FeedForward
    ) -> OperatorContract:
        norm_kind = "none" if module.norm is None else str(module.norm_kind)
        has_gate = module.proj_gate is not None
        epilogue = ("dropout",) if module.dropout_p > 0.0 else ()
        if module.final_dropout_enabled:
            epilogue = (*epilogue, "final_dropout")
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
            epilogue=epilogue,
            fusion=FusionIntent(
                patterns=tuple(runtime_fusion["realized_patterns"]),
                epilogue=epilogue,
            ),
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

    def _convert_linear(
        self, module: nn.Module, contract: OperatorContract
    ) -> ConvertResult:
        if self.engine == "torch":
            if not isinstance(module, nn.Linear) or not self.policy_was_explicit:
                report = {
                    "engine": self.engine,
                    "converted": False,
                    "reason": "torch engine keeps the original Linear-compatible implementation",
                }
                return ConvertResult(
                    model=module,
                    engine=self.engine,
                    target=self.target,
                    contract=contract,
                    converted=False,
                    report=report,
                )
        if isinstance(module, nn.Linear) and self.engine in {"torch", "triton"}:
            runtime_linear = _RuntimeLinearModule(
                module,
                engine=self.engine,
                precision=_matmul_spec_from_precision_policy(self.policy),
            )
            converted = self.engine == "triton"
            reason = (
                "torch engine uses a runtime-configured Linear wrapper"
                if self.engine == "torch"
                else "triton engine uses a runtime-configured Linear wrapper"
            )
            return ConvertResult(
                model=runtime_linear,
                engine=self.engine,
                target=self.target,
                contract=contract,
                converted=converted,
                report={
                    "engine": self.engine,
                    "converted": converted,
                    "reason": reason,
                    "runtime_config": runtime_linear.runtime_config(),
                    "contract": contract.to_dict(),
                },
            )
        target_plan = self._lower_linear_contract_to_target_plan(contract)
        converted_module, _ = materialize_module(
            module,
            contract=contract,
            target=target_plan,
        )
        report = {
            "engine": self.engine,
            "converted": True,
            "target_plan": target_plan.to_dict(),
            "contract": contract.to_dict(),
        }
        return ConvertResult(
            model=converted_module,
            engine=self.engine,
            target=self.target,
            contract=contract,
            converted=True,
            report=report,
        )

    def _convert_conv2d(
        self, module: nn.Conv2d, contract: OperatorContract
    ) -> ConvertResult:
        if self.engine == "torch":
            return ConvertResult(
                model=module,
                engine=self.engine,
                target=self.target,
                contract=contract,
                converted=False,
                report={
                    "engine": self.engine,
                    "converted": False,
                    "reason": "torch engine keeps the original Conv2d implementation",
                },
            )
        if self.engine != "tilelang":
            raise XQTBackendError(
                f"xqt.convert Conv2d currently supports only engine='tilelang', got {self.engine}"
            )
        target_plan = OperatorOptimizationTargetPlan(
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
        converted_module, _ = materialize_module(
            module,
            contract=contract,
            target=target_plan,
        )
        return ConvertResult(
            model=converted_module,
            engine=self.engine,
            target=self.target,
            contract=contract,
            converted=True,
            report={
                "engine": self.engine,
                "converted": True,
                "target_plan": target_plan.to_dict(),
                "contract": contract.to_dict(),
            },
        )

    def _convert_layernorm(
        self, module: nn.LayerNorm, contract: OperatorContract
    ) -> ConvertResult:
        if self.engine == "torch":
            return ConvertResult(
                model=module,
                engine=self.engine,
                target=self.target,
                contract=contract,
                converted=False,
                report={
                    "engine": self.engine,
                    "converted": False,
                    "reason": "torch engine keeps the original LayerNorm implementation",
                },
            )
        if self.engine != "tilelang":
            raise XQTBackendError(
                f"xqt.convert LayerNorm currently supports only engine='tilelang', got {self.engine}"
            )
        target_plan = OperatorOptimizationTargetPlan(
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
        converted_module, _ = materialize_module(
            module,
            contract=contract,
            target=target_plan,
        )
        return ConvertResult(
            model=converted_module,
            engine=self.engine,
            target=self.target,
            contract=contract,
            converted=True,
            report={
                "engine": self.engine,
                "converted": True,
                "target_plan": target_plan.to_dict(),
                "contract": contract.to_dict(),
            },
        )

    def _convert_feedforward(
        self,
        module: xqt_nn.FeedForward,
        contract: OperatorContract,
    ) -> ConvertResult:
        if self.engine not in {"torch", "triton"}:
            raise XQTBackendError(
                f"xqt.convert FeedForward currently supports only engine='torch' or engine='triton', got {self.engine}"
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
            return ConvertResult(
                model=module,
                engine=self.engine,
                target=self.target,
                contract=contract,
                converted=False,
                report={
                    "engine": self.engine,
                    "converted": False,
                    "reason": "torch engine keeps the original FeedForward implementation",
                    "runtime_config": module.runtime_config(),
                    "fusion": module.runtime_config()["fusion"],
                    "contract": contract.to_dict(),
                },
            )
        target_plan = OperatorOptimizationTargetPlan(
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
        converted_module, _ = materialize_module(
            module,
            contract=contract,
            target=target_plan,
        )
        return ConvertResult(
            model=converted_module,
            engine=self.engine,
            target=self.target,
            contract=contract,
            converted=True,
            report={
                "engine": self.engine,
                "converted": True,
                "reason": "Triton FeedForward candidate materialized from shared module contract",
                "runtime_config": converted_module.runtime_config(),
                "fusion": converted_module.runtime_config()["fusion"],
                "target_plan": target_plan.to_dict(),
                "contract": contract.to_dict(),
            },
        )

    def _feedforward_projection_policies(self) -> dict[str, dict[str, str]] | None:
        if self.projection_policies is None:
            return None
        return _projection_precision_dict(self.projection_policies)

    def _lower_linear_contract_to_target_plan(
        self,
        contract: OperatorContract,
    ) -> OperatorOptimizationTargetPlan:
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
        return OperatorOptimizationTargetPlan(**target_kwargs)

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


def convert(
    module: nn.Module,
    *,
    engine: str | None = None,
    target: str = "cuda",
    policy: PrecisionPolicy
    | FeedForwardPrecisionPolicy
    | MatmulPrecisionSpec
    | Mapping[str, str]
    | None = None,
    projection_policies: Mapping[str, PrecisionPolicy | Mapping[str, str]]
    | None = None,
    fallback: str = "eager",
    target_arch: str | None = None,
    inplace: bool = False,
    return_result: bool = False,
) -> nn.Module | ConvertResult:
    """Convert one module through the XQT operator engine facade.

    This public API is intentionally function-shaped. Internally it delegates to a
    stateful converter class so future recursive model conversion can share lowering
    context and capability caches.
    """

    if isinstance(policy, FeedForwardPrecisionPolicy):
        resolved_policy = policy.default
        resolved_projection_policies = policy.projection_policies()
        if projection_policies is not None:
            raise XQTBackendError(
                "xqt.convert does not allow both FeedForwardPrecisionPolicy and projection_policies"
            )
    elif isinstance(policy, MatmulPrecisionSpec):
        resolved_policy = _precision_policy_from_matmul_spec(policy)
        resolved_projection_policies = projection_policies
    elif isinstance(policy, Mapping):
        resolved_policy = _precision_policy_from_mapping(policy)
        resolved_projection_policies = projection_policies
    else:
        resolved_policy = policy or PrecisionPolicy()
        resolved_projection_policies = projection_policies
    resolved_engine = _resolve_engine_alias(
        engine=engine,
        context="xqt.convert",
    )
    result = _ModuleConverter(
        engine=resolved_engine,
        target=target,
        policy=resolved_policy,
        projection_policies=resolved_projection_policies,
        fallback=fallback,
        target_arch=target_arch,
        inplace=inplace,
        policy_was_explicit=policy is not None,
    ).convert_module(module)
    if return_result:
        return result
    return result.model


__all__ = [
    "ConvertResult",
    "FeedForwardPrecisionPolicy",
    "MatmulPrecisionSpec",
    "OperatorContract",
    "PrecisionPolicy",
    "TensorStorageSpec",
    "convert",
]
