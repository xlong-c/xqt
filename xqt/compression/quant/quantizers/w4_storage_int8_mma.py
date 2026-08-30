"""W4 storage to requested W8A8 INT8 MMA compute retarget quantizer."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.contracts.int8_mma import Int8MmaLinear
from xqt.contracts.w4_storage import W4StorageInt8MmaLinear
from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext

from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import build_component_quantization_report
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from xqt.contracts.packing_int4 import (
    _normalize_group_size,
    _quantize_grouped_fp4_weight,
    _unpack_int4,
)
from .fp4_weight_only import FP4WeightOnlyLinear
from .base import (
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)


_ACTIVATION_SCALE_MODES = {"dynamic", "static"}
_SELECTION_MODES = {"default", "include_only"}
_SOURCE_KINDS = {"linear", "fp4_weight_only", "auto"}


@dataclass
class W4StorageInt8MmaQuantizationResult(QuantizedModel):
    backend: str = "pytorch"
    strategy: str = "w4a16_int4"
    compute: str = "w8a8_int8_mma"


def _matches_name_patterns(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _should_quantize_module(
    name: str,
    module: nn.Module,
    policy: QuantizationPolicy,
    *,
    selection_mode: str,
) -> bool:
    if isinstance(module, W4StorageInt8MmaLinear):
        return False
    if not isinstance(module, (nn.Linear, FP4WeightOnlyLinear)):
        return False
    if selection_mode == "default":
        if name in policy.include_module_names:
            return True
        if _matches_name_patterns(name, policy.include_name_patterns):
            return True
        if name in policy.exclude_module_names:
            return False
        if _matches_name_patterns(name, policy.exclude_name_patterns):
            return False
        if "Linear" in policy.exclude_module_types:
            return False
        if policy.include_module_types and "Linear" not in policy.include_module_types:
            return False
        return _module_parameter_count(module) >= policy.min_parameters
    if selection_mode != "include_only":
        raise ValueError("selection_mode must be default or include_only")
    included = name in policy.include_module_names or _matches_name_patterns(
        name,
        policy.include_name_patterns,
    )
    if not included:
        return False
    if name in policy.exclude_module_names:
        return False
    if _matches_name_patterns(name, policy.exclude_name_patterns):
        return False
    if "Linear" in policy.exclude_module_types:
        return False
    if policy.include_module_types and "Linear" not in policy.include_module_types:
        return False
    return _module_parameter_count(module) >= policy.min_parameters



def quantize_with_w4_storage_int8_mma(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
    group_size: int = 128,
    engine: str = "tilelang",
    fallback_engine: str = "torch_int_mm",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    activation_scale_mode: str = "dynamic",
    activation_scales: Optional[Mapping[str, torch.Tensor | float]] = None,
    activation_quant_block_size: int = 256,
    eps: float = 1e-6,
    cache_int8_compute_view: bool = True,
    source: str = "auto",
) -> W4StorageInt8MmaQuantizationResult:
    """Replace Linear / FP4 weight-only modules with W4 storage and W8A8 retarget."""

    if str(source) not in _SOURCE_KINDS:
        raise ValueError("source must be one of auto, linear, fp4_weight_only")
    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    selection_mode = "default"
    policy_mapping: dict[str, Any] = {}
    if isinstance(policy, Mapping):
        policy_mapping = dict(policy)
        selection_mode = str(policy.get("selection_mode", "default"))
    if selection_mode not in _SELECTION_MODES:
        raise ValueError("selection_mode must be default or include_only")
    configured_group_size = int(policy_mapping.get("group_size", group_size) or group_size)
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": "int8",
                "scheme": "w4_storage_int8_mma",
                "engine": engine,
            },
        )
        or "w4_storage_int8_mma"
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []
    static_scales = dict(activation_scales or {})
    static_scale_modules = 0
    dynamic_fallback_modules = 0
    source_linear_count = 0
    source_fp4_count = 0

    for name, module in list(target_model.named_modules()):
        if not name:
            continue
        if not _should_quantize_module(
            name,
            module,
            quant_policy,
            selection_mode=selection_mode,
        ):
            continue
        module_activation_scale_mode = activation_scale_mode
        module_activation_scale = static_scales.get(name)
        if activation_scale_mode == "static":
            if module_activation_scale is None:
                module_activation_scale_mode = "dynamic"
                dynamic_fallback_modules += 1
            else:
                static_scale_modules += 1

        if isinstance(module, FP4WeightOnlyLinear) and source in {"auto", "fp4_weight_only"}:
            replacement = W4StorageInt8MmaLinear.from_fp4_weight_only(
                module,
                engine=engine,
                fallback_engine=fallback_engine,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                threads=threads,
                num_stages=num_stages,
                activation_scale_mode=module_activation_scale_mode,
                activation_scale=module_activation_scale,
                activation_quant_block_size=activation_quant_block_size,
                eps=eps,
                cache_int8_compute_view=cache_int8_compute_view,
            )
            source_fp4_count += 1
        elif isinstance(module, nn.Linear) and source in {"auto", "linear"}:
            replacement = W4StorageInt8MmaLinear.from_linear(
                module,
                group_size=configured_group_size,
                engine=engine,
                fallback_engine=fallback_engine,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                threads=threads,
                num_stages=num_stages,
                activation_scale_mode=module_activation_scale_mode,
                activation_scale=module_activation_scale,
                activation_quant_block_size=activation_quant_block_size,
                eps=eps,
                cache_int8_compute_view=cache_int8_compute_view,
            )
            source_linear_count += 1
        else:
            continue
        _replace_submodule(target_model, name, replacement)
        quantized_modules.append(name)

    return W4StorageInt8MmaQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "w4_storage_int8_mma_linear",
            "quantization_nature": "true",
            "quantization_nature_scope": "requested_compute_contract_not_runtime_observation",
            "method_semantics": "w4_storage_int8_mma_compute_retarget",
            "storage_encoding": "packed_signed_int4_group_scale",
            "compute_encoding": "w8a8_int8_mma",
            "activation_encoding": f"{activation_scale_mode}_signed_int8_per_tensor",
            "accumulation": "int32",
            "precision_description": {
                "quantization_time": {
                    "weight": "offline packed signed INT4 with group scales",
                    "activation": (
                        "not stored as an activation artifact; each forward uses "
                        f"{activation_scale_mode} signed INT8 per-tensor encoding"
                    ),
                },
                "runtime": {
                    "weight_retarget": (
                        "dequantize packed W4 then re-encode a per-output-channel "
                        "INT8 compute view"
                    ),
                    "requested_compute": "W8A8 INT8 MMA with INT32 accumulation",
                    "actual_execution_source": "W4StorageInt8MmaLinear.execution_metadata.runtime_precision",
                    "native_mma": "only true when the delegated per-forward metadata reports it",
                    "small_batch_float_fallback": {
                        "enabled": False,
                        "condition": "input_rows < min_int8_rows",
                        "note": "quantize_with_w4_storage_int8_mma constructs min_int8_rows=0",
                    },
                },
            },
            "engine": engine,
            "fallback_engine": fallback_engine,
            "group_size": configured_group_size,
            "cache_int8_compute_view": bool(cache_int8_compute_view),
            "source_linear_module_count": source_linear_count,
            "source_fp4_module_count": source_fp4_count,
            "activation_scale_mode": activation_scale_mode,
            "static_scale_module_count": static_scale_modules,
            "dynamic_fallback_module_count": dynamic_fallback_modules,
            "activation_quant_block_size": int(activation_quant_block_size),
            "block_m": int(block_m),
            "block_n": int(block_n),
            "block_k": int(block_k),
            "threads": int(threads),
            "num_stages": int(num_stages),
            "selection_mode": selection_mode,
            "notes": (
                "HBM state_dict keeps packed W4 weights; INT8 compute view is rebuilt from "
                "dequantized W4 (near-lossless re-encode of discrete levels to channel int8). "
                "Not bit-exact native FP4 MMA."
            ),
            "policy": {
                "dtype": quant_policy.dtype,
                "scheme": quant_policy.scheme,
                "include_module_types": list(quant_policy.include_module_types),
                "exclude_module_types": list(quant_policy.exclude_module_types),
                "include_name_patterns": list(quant_policy.include_name_patterns),
                "exclude_name_patterns": list(quant_policy.exclude_name_patterns),
                "include_module_names": list(quant_policy.include_module_names),
                "exclude_module_names": list(quant_policy.exclude_module_names),
                "min_parameters": quant_policy.min_parameters,
                "group_size": configured_group_size,
            },
        },
    )


def execute_w4_storage_int8_mma_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_w4_storage_int8_mma,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute W4-storage / INT8-MMA retarget for one quantization component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
        group_size=int(component.policy.get("group_size", 128)),
        engine=str(component.policy.get("engine", "tilelang")),
        fallback_engine=str(component.policy.get("fallback_engine", "torch_int_mm")),
        block_m=int(component.policy.get("block_m", 64)),
        block_n=int(component.policy.get("block_n", 64)),
        block_k=int(component.policy.get("block_k", 64)),
        threads=int(component.policy.get("threads", 128)),
        num_stages=int(component.policy.get("num_stages", 2)),
        activation_scale_mode=str(component.policy.get("activation_scale_mode", "dynamic")),
        activation_scales=component.policy.get("activation_scales"),
        activation_quant_block_size=int(component.policy.get("activation_quant_block_size", 256)),
        eps=float(component.policy.get("eps", 1e-6)),
        cache_int8_compute_view=bool(component.policy.get("cache_int8_compute_view", True)),
        source=str(component.policy.get("source", "auto")),
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    method_semantics = "w4_storage_int8_mma_compute_retarget"
    report = build_component_quantization_report(
        context,
        component,
        backend=result.backend,
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        nature=QuantizationNature.TRUE,
        algorithm_executable=True,
        method_semantics=method_semantics,
        effective_policy=effective_policy,
        result_metadata=result.metadata,
        execution_state=result.strategy,
    )
    return updated_model, report


__all__ = [
    "W4StorageInt8MmaLinear",
    "W4StorageInt8MmaQuantizationResult",
    "execute_w4_storage_int8_mma_component",
    "quantize_with_w4_storage_int8_mma",
]
