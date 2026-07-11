"""W4 storage with INT8 MMA compute retarget for non-native FP4 hardware."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext

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
from ..policy import QuantizationPolicy
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from .fp4_weight_only import (
    FP4WeightOnlyLinear,
    _normalize_group_size,
    _quantize_grouped_fp4_weight,
    _unpack_int4,
)
from .int8_mma import Int8MmaLinear


_ACTIVATION_SCALE_MODES = {"dynamic", "static"}
_SELECTION_MODES = {"default", "include_only"}
_SOURCE_KINDS = {"linear", "fp4_weight_only", "auto"}


@dataclass
class W4StorageInt8MmaQuantizationResult(QuantizedModel):
    backend: str = "pytorch"
    strategy: str = "w4_storage_int8_mma"


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


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def _channel_int8_from_float_weight(
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f = weight.detach().to(torch.float32)
    max_abs = weight_f.abs().amax(dim=1, keepdim=True)
    scale = torch.where(
        max_abs > float(eps),
        max_abs / 127.0,
        torch.ones_like(max_abs),
    )
    qweight = torch.round(weight_f / scale).clamp(-127, 127).to(torch.int8)
    return qweight.t().contiguous(), scale.reshape(-1).contiguous()


def _packed_w4_from_float_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    output_features, input_features = int(weight.shape[0]), int(weight.shape[1])
    normalized_group_size = _normalize_group_size(group_size, input_features)
    packed_weight, scale, padded_input_features = _quantize_grouped_fp4_weight(
        weight,
        group_size=normalized_group_size,
        input_features=input_features,
        output_features=output_features,
    )
    return packed_weight, scale, normalized_group_size, padded_input_features


def _float_weight_from_packed_w4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    input_features: int,
    output_features: int,
    group_size: int,
    padded_input_features: int,
) -> torch.Tensor:
    quantized = _unpack_int4(packed_weight, padded_input_features)
    grouped = quantized.reshape(output_features, -1, group_size)
    dequantized = grouped * weight_scale.to(dtype=torch.float32)
    return dequantized.reshape(output_features, padded_input_features)[:, :input_features]


class W4StorageInt8MmaLinear(nn.Module):
    """Packed W4 storage Linear with INT8 MMA compute retarget."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        group_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        output_dtype: torch.dtype = torch.float32,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
    ) -> None:
        super().__init__()
        if str(activation_scale_mode) not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.engine = str(engine)
        self.fallback_engine = str(fallback_engine)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.threads = int(threads)
        self.num_stages = int(num_stages)
        self.output_dtype = output_dtype
        self.activation_scale_mode = str(activation_scale_mode)
        self.activation_quant_block_size = int(activation_quant_block_size)
        self.eps = float(eps)
        self.cache_int8_compute_view = bool(cache_int8_compute_view)
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("packed_weight", packed_weight.to(torch.uint8).contiguous())
        self.register_buffer("group_scale", group_scale.to(torch.float32).contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())
        self._compute: Int8MmaLinear | None = None
        self._activation_scale = activation_scale
        if self.cache_int8_compute_view:
            self._ensure_compute_view()

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        group_size: int = 128,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
    ) -> "W4StorageInt8MmaLinear":
        packed_weight, group_scale, normalized_group_size, padded = _packed_w4_from_float_weight(
            module.weight.detach(),
            group_size=group_size,
        )
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            packed_weight,
            group_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=normalized_group_size,
            padded_input_features=padded,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=module.weight.dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
        )

    @classmethod
    def from_fp4_weight_only(
        cls,
        module: FP4WeightOnlyLinear,
        *,
        engine: str = "tilelang",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        cache_int8_compute_view: bool = True,
    ) -> "W4StorageInt8MmaLinear":
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            module.packed_weight.detach().to(torch.uint8).contiguous(),
            module.weight_scale.detach().to(torch.float32).contiguous(),
            bias=bias,
            input_features=module.input_features,
            output_features=module.output_features,
            group_size=module.group_size,
            padded_input_features=module.padded_input_features,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=torch.float32 if module.bias is None else module.bias.dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
        )

    def dequantize_weight(self) -> torch.Tensor:
        return _float_weight_from_packed_w4(
            self.packed_weight,
            self.group_scale,
            input_features=self.input_features,
            output_features=self.output_features,
            group_size=self.group_size,
            padded_input_features=self.padded_input_features,
        )

    def quantized_weight_codes(self) -> torch.Tensor:
        return _unpack_int4(self.packed_weight, self.padded_input_features)[
            :, : self.input_features
        ]

    def storage_nbytes(self) -> int:
        total = int(self.packed_weight.nbytes) + int(self.group_scale.nbytes)
        if self.bias is not None:
            total += int(self.bias.nbytes)
        return total

    def release_int8_compute_view(self) -> None:
        self._compute = None

    def _ensure_compute_view(self) -> Int8MmaLinear:
        if self._compute is not None:
            return self._compute
        weight = self.dequantize_weight()
        qweight_t, channel_scale = _channel_int8_from_float_weight(weight, eps=self.eps)
        compute = Int8MmaLinear(
            qweight_t,
            channel_scale,
            bias=self.bias,
            input_features=self.input_features,
            output_features=self.output_features,
            engine=self.engine,
            fallback_engine=self.fallback_engine,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            threads=self.threads,
            num_stages=self.num_stages,
            output_dtype=self.output_dtype,
            activation_scale_mode=self.activation_scale_mode,
            activation_scale=self._activation_scale,
            activation_quant_block_size=self.activation_quant_block_size,
            eps=self.eps,
        )
        compute.to(device=self.packed_weight.device)
        if self.cache_int8_compute_view:
            self._compute = compute
        return compute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "W4StorageInt8MmaLinear input trailing dimension does not match input_features"
            )
        compute = self._ensure_compute_view()
        if compute.qweight_t.device != inputs.device:
            compute.to(device=inputs.device)
        output = compute(inputs)
        metadata = dict(compute.execution_metadata())
        metadata.update(
            {
                "storage_dtype": "packed_signed_int4",
                "compute_dtype": "int8",
                "retarget": "w4_storage_int8_mma",
                "group_size": self.group_size,
                "storage_nbytes": self.storage_nbytes(),
                "int8_compute_view_cached": self._compute is not None,
                "quantization_nature": "true",
            }
        )
        self.last_execution = metadata
        return output

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self.last_execution)

    def tilelang_packed_dequant_gemm_args(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int]:
        packed_weight = self.packed_weight.to(device=device)
        scale = self.group_scale.to(device=device, dtype=dtype)
        bias = None if self.bias is None else self.bias.to(device=device, dtype=dtype)
        return packed_weight, scale, bias, None, self.input_features, self.group_size


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
    """Replace Linear / FP4 weight-only modules with W4 storage + INT8 MMA compute."""

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
            "method_semantics": "w4_storage_int8_mma_compute_retarget",
            "storage_encoding": "packed_signed_int4_group_scale",
            "compute_encoding": "w8a8_int8_mma",
            "activation_encoding": f"{activation_scale_mode}_signed_int8_per_tensor",
            "accumulation": "int32",
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
    method_semantics = "w4_storage_int8_mma_compute_retarget"
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
    "W4StorageInt8MmaLinear",
    "W4StorageInt8MmaQuantizationResult",
    "execute_w4_storage_int8_mma_component",
    "quantize_with_w4_storage_int8_mma",
]
