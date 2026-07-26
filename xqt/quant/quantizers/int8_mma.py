"""W8A8 INT8 MMA contract quantizer with per-forward runtime observation."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.contracts import ComputeConfig, QuantizedModel
from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext
from xqt.runtime.engine_resolve import (
    normalize_engine_name,
    resolve_int8_mma_engine,
)
from xqt.runtime.modules import Int8MmaLinear

_PTX_SM89_ENGINES = frozenset({"ptx_sm89", "native_sm89"})
_CUDA_SM89_ENGINES = frozenset({"cuda_sm89"})
_VALID_ENGINES = (
    frozenset({"auto", "triton", "tilelang", "torch_int_mm"})
    | _PTX_SM89_ENGINES
    | _CUDA_SM89_ENGINES
)


from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import optional_calibration_summary
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport


_ACTIVATION_SCALE_MODES = {"dynamic", "static"}
_OUTPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_SELECTION_MODES = {"default", "include_only"}


@dataclass
class Int8MmaQuantizationResult(QuantizedModel):
    """Result returned by the true INT8 MMA quantization helper."""

    backend: str = "pytorch"
    strategy: str = "w8a8_int8"
    compute: str = "w8a8_int8_mma"


def _policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key == "dtype":
            kwargs["dtype"] = str(value)
        elif key == "scheme":
            kwargs["scheme"] = str(value)
        elif key in {
            "include_module_types",
            "exclude_module_types",
            "include_name_patterns",
            "exclude_name_patterns",
            "include_module_names",
            "exclude_module_names",
        }:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


def _matches_name_patterns(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=False))


def _should_quantize_int8_mma_module(
    name: str,
    module: nn.Module,
    policy: QuantizationPolicy,
    *,
    selection_mode: str,
) -> bool:
    if selection_mode == "default":
        return should_quantize_module(name, module, policy)
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
    if type(module).__name__ in policy.exclude_module_types:
        return False
    if policy.include_module_types and type(module).__name__ not in policy.include_module_types:
        return False
    return _module_parameter_count(module) >= policy.min_parameters


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)




def quantize_with_int8_mma(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
    engine: str = "auto",
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
) -> Int8MmaQuantizationResult:
    """Replace Linear modules with a requested W8A8 INT8 MMA runtime contract."""

    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    selection_mode = "default"
    if isinstance(policy, Mapping):
        selection_mode = str(policy.get("selection_mode", "default"))
    if selection_mode not in _SELECTION_MODES:
        raise ValueError("selection_mode must be default or include_only")
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": "int8",
                "scheme": "dynamic_mma",
                "engine": engine,
            },
        )
        or "dynamic_int8_mma"
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []
    static_scales = dict(activation_scales or {})
    static_scale_modules = 0
    dynamic_fallback_modules = 0

    for name, module in list(target_model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _should_quantize_int8_mma_module(
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
        _replace_submodule(
            target_model,
            name,
            Int8MmaLinear.from_linear(
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
            ),
        )
        quantized_modules.append(name)

    preferred_hint = [] if normalize_engine_name(engine) == "auto" else [normalize_engine_name(engine)]
    compute_config = ComputeConfig.from_modules(
        module_names=quantized_modules,
        compute_contract="int8_mma",
        precision="w8a8",
        required_capabilities=["int8_mma"],
        preferred_engines=preferred_hint,
        default_precision="w8a8",
        storage={
            "format": "int8_per_out_channel",
            "layout": "qweight_t",
        },
        metadata={
            "activation_scale_mode": activation_scale_mode,
            "fallback_engine": fallback_engine,
        },
    )
    return Int8MmaQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        compute_config=compute_config,
        metadata={
            "implementation": "dynamic_w8a8_int8_mma_linear",
            "quantization_nature": "true",
            "quantization_nature_scope": "requested_compute_contract_not_runtime_observation",
            "activation_encoding": f"{activation_scale_mode}_signed_int8_per_tensor",
            "weight_encoding": "signed_int8_per_output_channel",
            "accumulation": "int32",
            "precision_description": {
                "quantization_time": {
                    "weight": "offline static signed INT8 per output channel",
                    "activation": (
                        "not stored as an activation artifact; each forward uses "
                        f"{activation_scale_mode} signed INT8 per-tensor encoding"
                    ),
                },
                "runtime": {
                    "requested_compute": "W8A8 INT8 MMA with INT32 accumulation",
                    "actual_execution_source": "Int8MmaLinear.execution_metadata.runtime_precision",
                    "native_mma": "only true when the per-forward metadata reports it",
                    "small_batch_float_fallback": {
                        "enabled": False,
                        "condition": "input_rows < min_int8_rows",
                        "note": "quantize_with_int8_mma constructs min_int8_rows=0",
                    },
                },
            },
            "engine_preference": normalize_engine_name(engine),
            "preferred_engines": preferred_hint,
            "fallback_engine": fallback_engine,
            "compute_config": compute_config.to_dict(),
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
            },
        },
    )


def execute_int8_mma_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_int8_mma,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the W8A8 INT8 MMA contract quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
        engine=str(component.policy.get("engine", "auto")),
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
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    high_precision_modules = prefix_module_names(component.keep_high_precision, component.target_path)
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    quantized_modules = prefix_module_names(result.quantized_modules, component.target_path)
    module_selection_reasons = module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = optional_calibration_summary(
        context,
        component,
    )
    method_semantics = "w8a8_int8_mma_runtime_quantization_contract"
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method,
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.TRUE,
        algorithm_executable=True,
        method_semantics=method_semantics,
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "executed": True,
            "execution_state": result.strategy,
            "algorithm_executable": True,
            "method_semantics": method_semantics,
        },
    )
    return updated_model, report


__all__ = [
    "execute_int8_mma_component",
    "Int8MmaLinear",
    "Int8MmaQuantizationResult",
    "quantize_with_int8_mma",
]
