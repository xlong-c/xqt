"""SVDQuant method: dual-branch storage quant (runtime modules in xqt.runtime.modules)."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import CompositeAddLinear, ComputeConfig, QuantizedModel
from xqt.analysis.svd_analysis import (
    SVDQuantAnalysis,
    decompose_weight_svd,
)
from xqt.core.inputs import extract_model_inputs, infer_model_input_count
from xqt.core.types import XQTContext

from ..capability import _resolve_nature
from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..execution.reporting import build_component_quantization_report
from xqt.contracts.packing_int4 import _pack_int4
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from .base import (
    call_model as _call_model,
    move_batch_to_device as _move_calibration_batch,
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)


# ── SVDQuant result dataclass ────────────────────────────────────────────


@dataclass
class SVDQuantResult(QuantizedModel):
    """Result returned by the SVDQuant quant method (pytorch backend)."""

    backend: str = "pytorch"
    method: str | None = "svd"
    strategy: str = "w4a16_fp4"
    compute: str | None = "dequant_fp16"
    svd_analysis: Optional[SVDQuantAnalysis] = None


_SUPPORTED_RESIDUAL_QUANT_DTYPES = frozenset({"fp4", "int4"})
_SUPPORTED_RESIDUAL_COMPUTE = frozenset({"reference", "int8_mma"})


def _quant_dtype_from_strategy(strategy: str | None, default: str = "fp4") -> str:
    """Derive quant_dtype from canonical WxAy+format strategy."""
    if strategy is None:
        return default
    text = strategy.lower()
    if "fp4" in text:
        return "fp4"
    if "int4" in text:
        return "int4"
    if "int8" in text:
        return "int8"
    return default


def _residual_compute_from_compute(compute: str | None) -> str:
    """Derive residual_compute from canonical compute field."""
    if compute == "w8a8_int8_mma":
        return "int8_mma"
    return "reference"


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _quantize_residual_int4(
    weight_residual: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize one SVD residual into the generic composite artifact format."""

    output_features, input_features = weight_residual.shape
    normalized_group_size = max(1, min(int(group_size), int(input_features)))
    padded_input_features = (
        (int(input_features) + normalized_group_size - 1)
        // normalized_group_size
    ) * normalized_group_size
    residual = weight_residual.detach().to(torch.float32)
    if padded_input_features != input_features:
        residual = F.pad(
            residual,
            (0, padded_input_features - input_features),
        )
    grouped = residual.reshape(
        int(output_features),
        -1,
        normalized_group_size,
    )
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scale = torch.where(
        max_abs > 0,
        max_abs / 7.0,
        torch.ones_like(max_abs),
    )
    quantized = torch.clamp(
        torch.round(grouped / (scale + 1e-12)),
        min=-8,
        max=7,
    ).to(torch.int8)
    packed = _pack_int4(
        quantized.reshape(int(output_features), padded_input_features)
    )
    return packed, scale.squeeze(-1).to(torch.float32), padded_input_features


def _build_composite_add_linear(
    module: nn.Linear,
    *,
    rank: int,
    group_size: int,
    quant_dtype: str,
) -> tuple[CompositeAddLinear, Any]:
    """Create a generic composite artifact from one dense Linear module."""

    weight = module.weight.detach().reshape(module.out_features, module.in_features)
    decomposition = decompose_weight_svd(weight, rank=rank)
    down_weight, up_weight = decomposition.low_rank_components()
    residual = decomposition.residual_weight(weight).to(torch.float32)
    packed_residual, residual_scale, padded_input_features = _quantize_residual_int4(
        residual,
        group_size=group_size,
    )
    normalized_group_size = max(1, min(int(group_size), int(module.in_features)))
    artifact = CompositeAddLinear.from_packed_parts(
        down_weight=down_weight,
        up_weight=up_weight,
        packed_residual=packed_residual,
        residual_scale=residual_scale,
        bias=None if module.bias is None else module.bias.detach(),
        input_features=module.in_features,
        output_features=module.out_features,
        group_size=normalized_group_size,
        padded_input_features=padded_input_features,
        quant_dtype=quant_dtype,
    )
    return artifact, decomposition


def _collect_static_activation_scales(
    model: nn.Module,
    *,
    module_names: Iterable[str],
    calibration_inputs: Iterable[Any] | None,
    sample_limit: int | None,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Collect symmetric per-tensor activation scales for selected Linear modules."""

    if calibration_inputs is None:
        return {}
    wanted = {str(name) for name in module_names}
    if not wanted:
        return {}
    maxima: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], _: Any) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            activation = inputs[0].detach()
            if activation.ndim == 0 or activation.shape[-1] != module.in_features:
                return
            current = activation.to(torch.float32).abs().amax()
            previous = maxima.get(name)
            maxima[name] = current if previous is None else torch.maximum(previous, current)

        return hook

    for name, module in model.named_modules():
        if name in wanted and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
    if not handles:
        return {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            expected_input_count = infer_model_input_count(model)
            for index, batch in enumerate(calibration_inputs):
                if sample_limit is not None and index >= int(sample_limit):
                    break
                inputs = extract_model_inputs(
                    batch,
                    expected_input_count=expected_input_count,
                )
                _call_model(
                    model,
                    _move_calibration_batch(inputs, _model_device(model)),
                )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {
        name: (value / 127.0).clamp_min(float(eps)).detach().to("cpu")
        for name, value in maxima.items()
    }


# ── Low-rank branch ──────────────────────────────────────────────────────



# ── Packed INT4 residual weight helpers ──────────────────────────────────




# ── Combined SVDQuant Linear module ──────────────────────────────────────




# ── Submodule replacement ─────────────────────────────────────────────────


# ── Main quantization entry point ────────────────────────────────────────


def quantize_with_svd(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    compute: Optional[str] = None,
    rank: int = 32,
    group_size: int = 128,
    quant_dtype: str = "int4",
    residual_compute: str = "reference",
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    activation_scale_mode: str = "dynamic",
    activation_scales: Optional[Mapping[str, torch.Tensor | float]] = None,
    calibration_inputs: Iterable[Any] | None = None,
    calibration_sample_limit: int | None = None,
    activation_quant_block_size: int = 256,
    eps: float = 1e-6,
    cache_int8_compute_view: bool = True,
    inplace: bool = True,
    collect_analysis: bool = True,
) -> SVDQuantResult:
    """Apply SVDQuant to all qualifying Linear layers in a model.

    For each Linear layer:
      1. SVD decompose weight -> low-rank branch (L1, L2) + residual
      2. Quantize residual to INT4/FP4 with per-group scales
      3. Replace the reference path with a generic CompositeAddLinear artifact

    Args:
        model: PyTorch model to quantize.
        policy: Module selection policy (which layers to quantize).
        strategy: Canonical WxAy+format strategy (e.g. "w4a16_fp4", "w4a16_int4").
            Determines storage quant_dtype.
        compute: Compute kernel (e.g. "dequant_fp16", "w8a8_int8_mma").
            Determines residual compute path.
        rank: Low-rank branch rank (r). Typical: 16-64.
        group_size: Per-group quantization granularity.
        quant_dtype: Residual quantization dtype ("fp4" or "int4").
        residual_compute: "reference" or "int8_mma" for the residual branch.
        engine: Preferred INT8 MMA engine when residual_compute is "int8_mma".
        calibration_inputs: Representative model inputs for static INT8 activation
            scale collection when activation_scale_mode is "static".
        inplace: If True, modify model in-place.
        collect_analysis: If True, collect SVD metrics for all layers.

    Returns:
        SVDQuantResult with the modified model and analysis.
    """
    quant_policy = (
        policy
        if isinstance(policy, QuantizationPolicy)
        else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    configured_rank = int(policy_mapping.get("rank", rank))
    configured_group_size = int(policy_mapping.get("group_size", group_size) or group_size)
    configured_quant_dtype = str(
        policy_mapping.get("quant_dtype", quant_dtype)
    ).lower()

    selected_strategy = normalize_quant_strategy(strategy, policy_mapping)
    if selected_strategy is None:
        selected_strategy = f"w4a16_{configured_quant_dtype}"
    from xqt.compression.quant.strategy import normalize_quant_compute

    selected_compute = normalize_quant_compute(compute, policy_mapping)
    if selected_compute is None and policy_mapping:
        selected_compute = normalize_quant_compute(policy_mapping.get("compute"))
    if selected_compute is None:
        selected_compute = "dequant_fp16"

    if "quant_dtype" not in policy_mapping and selected_strategy is not None:
        configured_quant_dtype = _quant_dtype_from_strategy(
            selected_strategy, configured_quant_dtype
        )
    if "residual_compute" in policy_mapping:
        configured_residual_compute = str(policy_mapping["residual_compute"]).lower()
    elif compute is not None or (
        policy_mapping.get("compute") is not None
    ):
        configured_residual_compute = _residual_compute_from_compute(selected_compute)
    else:
        configured_residual_compute = str(residual_compute).lower()
    if configured_quant_dtype not in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
        allowed = ", ".join(sorted(_SUPPORTED_RESIDUAL_QUANT_DTYPES))
        raise ValueError(f"quant_dtype must be one of {allowed}")
    if configured_residual_compute not in _SUPPORTED_RESIDUAL_COMPUTE:
        allowed = ", ".join(sorted(_SUPPORTED_RESIDUAL_COMPUTE))
        raise ValueError(f"residual_compute must be one of {allowed}")

    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []
    svd_analysis = SVDQuantAnalysis() if collect_analysis else None
    static_scales = dict(activation_scales or {})
    static_scale_modules = 0
    dynamic_fallback_modules = 0
    selected_module_names = [
        name
        for name, module in target_model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and should_quantize_module(name, module, quant_policy)
    ]
    selected_module_name_set = set(selected_module_names)
    calibrated_static_scales: dict[str, torch.Tensor] = {}
    if (
        configured_residual_compute == "int8_mma"
        and activation_scale_mode == "static"
        and calibration_inputs is not None
    ):
        calibrated_static_scales = _collect_static_activation_scales(
            target_model,
            module_names=selected_module_names,
            calibration_inputs=calibration_inputs,
            sample_limit=calibration_sample_limit,
            eps=float(policy_mapping.get("eps", eps)),
        )
        for name, scale in calibrated_static_scales.items():
            static_scales.setdefault(name, scale)

    module_activation_hints: dict[str, dict[str, Any]] = {}
    for name, module in list(target_model.named_modules()):
        if name not in selected_module_name_set or not isinstance(module, nn.Linear):
            continue

        svd_module, decomposition = _build_composite_add_linear(
            module,
            rank=configured_rank,
            group_size=configured_group_size,
            quant_dtype=configured_quant_dtype,
        )
        if configured_residual_compute == "int8_mma":
            activation_scale = static_scales.get(name)
            module_activation_scale_mode = activation_scale_mode
            if activation_scale_mode == "static":
                if activation_scale is None:
                    module_activation_scale_mode = "dynamic"
                    dynamic_fallback_modules += 1
                else:
                    static_scale_modules += 1
            svd_module.set_activation_materialize_hint(
                activation_scale_mode=module_activation_scale_mode,
                activation_scale=activation_scale,
            )
            module_activation_hints[name] = {
                "activation_scale_mode": module_activation_scale_mode,
                "activation_scale": activation_scale,
            }

        _replace_submodule(target_model, name, svd_module)
        quantized_modules.append(name)

        if svd_analysis is not None:
            svd_analysis.decompositions[name] = decomposition

    configured_engine = str(policy_mapping.get("engine", engine))
    preferred_engines = (
        [] if configured_engine in {"", "auto"} else [configured_engine]
    )
    residual_branch_contract = (
        "w4_storage_int8_mma"
        if configured_residual_compute == "int8_mma"
        else "generic"
    )
    residual_caps = (
        ["int8_mma", "w4_storage_int8_mma", "composite_split"]
        if configured_residual_compute == "int8_mma"
        else ["fp16_mma", "composite_split"]
    )
    branches = [
        {
            "name": "low_rank",
            "compute_contract": "fp16_mma",
            "precision": "source_precision",
            "storage": {
                "format": "dense_source_precision",
                "kind": "low_rank_factors",
            },
            "required_capabilities": ["fp16_mma"],
        },
        {
            "name": "quant_residual",
            "compute_contract": residual_branch_contract,
            "precision": (
                "w8a8" if configured_residual_compute == "int8_mma" else "w4a16"
            ),
            "storage": {
                "format": "packed_signed_int4_group_scale",
                "quant_dtype": configured_quant_dtype,
            },
            "required_capabilities": list(residual_caps),
            "preferred_engines": list(preferred_engines),
        },
    ]
    compute_config = ComputeConfig.from_modules(
        module_names=quantized_modules,
        compute_contract="composite_add",
        precision="w8a8" if configured_residual_compute == "int8_mma" else "w4a16",
        required_capabilities=(
            ["composite_add", "int8_mma", "fp16_mma"]
            if configured_residual_compute == "int8_mma"
            else ["composite_add", "fp16_mma"]
        ),
        preferred_engines=preferred_engines,
        default_precision=(
            "w8a8" if configured_residual_compute == "int8_mma" else "w4a16"
        ),
        storage={
            "kind": "svd_low_rank_plus_residual",
            "decomposition": "additive",
            "rank": configured_rank,
            "group_size": configured_group_size,
            "quant_dtype": configured_quant_dtype,
        },
        branches=branches,
        combine="add",
        preferred_mode="split",
        execution={"activation_scale_mode": activation_scale_mode},
        metadata={
            "residual_compute": configured_residual_compute,
            "quant_dtype": configured_quant_dtype,
            "fallback_engine": str(
                policy_mapping.get("fallback_engine", fallback_engine)
            ),
            "activation_scale_mode": activation_scale_mode,
            "block_m": int(policy_mapping.get("block_m", block_m)),
            "block_n": int(policy_mapping.get("block_n", block_n)),
            "block_k": int(policy_mapping.get("block_k", block_k)),
            "threads": int(policy_mapping.get("threads", threads)),
            "num_stages": int(policy_mapping.get("num_stages", num_stages)),
            "activation_quant_block_size": int(
                policy_mapping.get(
                    "activation_quant_block_size", activation_quant_block_size
                )
            ),
            "eps": float(policy_mapping.get("eps", eps)),
            "cache_int8_compute_view": bool(
                policy_mapping.get("cache_int8_compute_view", cache_int8_compute_view)
            ),
        },
    )
    for spec in compute_config.modules:
        spec.metadata.update(compute_config.metadata)

    if bool(policy_mapping.get("materialize_compute", False)):
        raise ValueError(
            "SVDQuant quantizer returns storage artifacts only; "
            "materialize compute in xqt.runtime"
        )

    return SVDQuantResult(
        model=target_model,
        backend="pytorch",
        method="svd",
        strategy=selected_strategy,
        compute=selected_compute,
        quantized_modules=quantized_modules,
        svd_analysis=svd_analysis,
        compute_config=compute_config,
        metadata={
            "implementation": "composite_add_svd_artifact",
            "quant_method": "svd",
            "rank": configured_rank,
            "group_size": configured_group_size,
            "quant_dtype": configured_quant_dtype,
            "low_rank_branch_dtype": "source_precision",
            "residual_compute": configured_residual_compute,
            "residual_storage": "packed_signed_int4_group_scale",
            "compute_contract": "composite_add",
            "combine": "add",
            "preferred_mode": "split",
            "fusion_status": "none",
            "fusion_note": (
                "Dual-branch composite_add: low-rank source-precision + quantized "
                "residual. FUSE_DOWN/FUSE_UP kernels are not materialized."
            ),
            "compute_config": compute_config.to_dict(),
            "activation_scale_mode": activation_scale_mode,
            "static_scale_module_count": static_scale_modules,
            "dynamic_fallback_module_count": dynamic_fallback_modules,
            "calibrated_static_scale_module_count": len(calibrated_static_scales),
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
                "rank": configured_rank,
                "group_size": configured_group_size,
                "quant_dtype": configured_quant_dtype,
                "residual_compute": configured_residual_compute,
                "calibration_sample_limit": calibration_sample_limit,
            },
            "svd_analysis": svd_analysis.to_dict() if svd_analysis is not None else None,
        },
    )


def execute_svdquant_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_svd,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the SVDQuant quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    configured_rank = int(effective_policy.get("rank", 32))
    configured_group_size = int(effective_policy.get("group_size", 128))
    strategy_name = str(component.strategy or effective_policy.get("strategy") or "")
    configured_quant_dtype = str(
        effective_policy.get("quant_dtype", _quant_dtype_from_strategy(strategy_name, "fp4"))
    )
    configured_residual_compute = str(
        effective_policy.get(
            "residual_compute",
            _residual_compute_from_compute(component.compute),
        )
    )
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        compute=component.compute,
        rank=configured_rank,
        group_size=configured_group_size,
        quant_dtype=configured_quant_dtype,
        residual_compute=configured_residual_compute,
        engine=str(effective_policy.get("engine", "auto")),
        fallback_engine=str(effective_policy.get("fallback_engine", "torch_int_mm")),
        block_m=int(effective_policy.get("block_m", 64)),
        block_n=int(effective_policy.get("block_n", 64)),
        block_k=int(effective_policy.get("block_k", 64)),
        threads=int(effective_policy.get("threads", 128)),
        num_stages=int(effective_policy.get("num_stages", 2)),
        activation_scale_mode=str(
            effective_policy.get("activation_scale_mode", "dynamic")
        ),
        activation_scales=effective_policy.get("activation_scales"),
        calibration_inputs=context.calibration_inputs,
        calibration_sample_limit=effective_policy.get("calibration_sample_limit"),
        activation_quant_block_size=int(
            effective_policy.get("activation_quant_block_size", 256)
        ),
        eps=float(effective_policy.get("eps", 1e-6)),
        cache_int8_compute_view=bool(
            effective_policy.get("cache_int8_compute_view", True)
        ),
        inplace=True,
        collect_analysis=True,
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    is_int8_mma = result.compute == "w8a8_int8_mma" or str(
        result.metadata.get("residual_compute", "")
    ) == "int8_mma"
    nature = QuantizationNature.TRUE if is_int8_mma else _resolve_nature(
        component.strategy,
        component.policy,
        compute=result.compute,
    )
    method_semantics = (
        "svd_method_composite_add_low_rank_plus_w4_residual"
        if is_int8_mma
        else "svd_method_composite_add_reference"
    )
    report = build_component_quantization_report(
        context,
        component,
        backend="pytorch",
        method=component.method or result.method or "svd",
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        nature=nature,
        algorithm_executable=True,
        method_semantics=method_semantics,
        effective_policy=effective_policy,
        result_metadata=result.metadata,
        execution_state="composite_add_artifact",
        extra_metadata={
            "rank": configured_rank,
            "group_size": configured_group_size,
            "quant_dtype": configured_quant_dtype,
        },
    )
    return updated_model, report


__all__ = [
    "SVDQuantResult",
    "execute_svdquant_component",
    "quantize_with_svd",
]
