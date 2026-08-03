"""External GPTQ/AWQ/compressed-tensors load bridge (C4), vLLM-aligned.

Lifecycle (model-side only, mirrors QuantizeMethodBase):

1. resolve_external_quantization
2. create_weight_plans
3. load state_dict
4. process_weights_after_loading
5. apply via module forward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from xqt.contracts import QuantizedModel
from xqt.contracts.layout_kernel_report import (
    LayoutKernelReport,
    empty_layout_kernel_report,
    layout_report_from_module_shapes,
)
from xqt.core.errors import XQTArtifactError
from xqt.quant.external import resolve_external_quantization
from xqt.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear
from xqt.runtime.bridges.external_materialize import (
    create_weight_plans,
    materialize_plans,
    try_native_xqt_copy,
)
from xqt.runtime.bridges.hf_int4_layout import process_weights_after_loading
from xqt.runtime.bridges.weight_io import find_weight_file, load_weight_state_dict


@dataclass(frozen=True, slots=True)
class ExternalLoadReport:
    """Result of attempting to load an external quantized checkpoint."""

    format: str
    model_path: str
    loaded: bool
    module_count: int = 0
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    layout_kernel: LayoutKernelReport | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "model_path": self.model_path,
            "loaded": self.loaded,
            "module_count": self.module_count,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.layout_kernel is not None:
            payload["layout_kernel"] = self.layout_kernel.to_dict()
        return payload


def _probe_layout(
    *,
    bits: int | None,
    group_size: int | None,
    symmetric: bool | None,
    zero_point: bool | None,
    desc_act: bool | None,
    fallback_reason: str | None,
) -> LayoutKernelReport:
    return empty_layout_kernel_report(
        fallback_reason=fallback_reason,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        zero_point=zero_point,
        desc_act=desc_act,
        scale_time="weight_offline",
        activation_granularity=None,
    )


def _success_layout_from_model(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    symmetric: bool | None,
    zero_point: bool | None,
    desc_act: bool | None,
    g_idx_applied: bool | None,
    storage_layout: str,
    selected_kernel: str,
    scale_time: str | None = "weight_offline",
    activation_granularity: str | None = None,
) -> LayoutKernelReport:
    for module in model.modules():
        if isinstance(module, AWQGPTQWeightOnlyLinear):
            return layout_report_from_module_shapes(
                bits=int(module.bits),
                group_size=int(module.group_size),
                symmetric=symmetric,
                zero_point=zero_point,
                desc_act=desc_act,
                g_idx_applied=g_idx_applied,
                out_features=int(module.output_features),
                in_features=int(module.input_features),
                padded_in_features=int(module.padded_input_features),
                storage_layout=storage_layout,
                selected_kernel=selected_kernel,
                fallback_reason=None,
                scale_time=scale_time,
                activation_granularity=activation_granularity,
            )
    return layout_report_from_module_shapes(
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        zero_point=zero_point,
        desc_act=desc_act,
        g_idx_applied=g_idx_applied,
        out_features=0,
        in_features=0,
        padded_in_features=0,
        storage_layout=storage_layout,
        selected_kernel=selected_kernel,
        fallback_reason=None,
        scale_time=scale_time,
        activation_granularity=activation_granularity,
    )


def load_external_quantized_model(
    model_path: str | Path,
    *,
    override: str | None = None,
    base_model: nn.Module | None = None,
) -> tuple[QuantizedModel | None, ExternalLoadReport]:
    """Load external quant checkpoint with vLLM-aligned resolve + materialize."""

    root = Path(model_path)
    info = resolve_external_quantization(root, user_quant=override)
    fmt = info.format
    bits = int(info.bits) if info.bits is not None else None
    group_size = int(info.group_size) if info.group_size is not None else None
    weight_file = find_weight_file(root)
    metadata: dict[str, Any] = {
        "probe": info.to_dict(),
        "resolved_format": fmt,
        "weight_file": str(weight_file) if weight_file is not None else None,
        "lifecycle": (
            "resolve",
            "create_weights",
            "load_state_dict",
            "process_weights_after_loading",
            "apply",
        ),
    }

    def _fail(
        notes: tuple[str, ...],
        *,
        fallback: str,
    ) -> tuple[QuantizedModel | None, ExternalLoadReport]:
        layout = _probe_layout(
            bits=bits,
            group_size=group_size,
            symmetric=info.sym,
            zero_point=info.zero_point,
            desc_act=info.desc_act,
            fallback_reason=fallback,
        )
        return None, ExternalLoadReport(
            format=fmt,
            model_path=str(root),
            loaded=False,
            notes=notes,
            metadata=metadata,
            layout_kernel=layout,
        )

    if base_model is None:
        return _fail(
            ("probe_only: pass base_model to materialize Linear modules",),
            fallback="probe_only",
        )
    if weight_file is None:
        return _fail(("weight_file_missing",), fallback="weight_file_missing")

    try:
        state = load_weight_state_dict(weight_file)
    except XQTArtifactError as exc:
        return _fail(
            (f"state_dict_load_failed:{exc}",),
            fallback="state_dict_load_failed",
        )

    resolved_bits = int(bits) if bits is not None else 4
    resolved_group = int(group_size) if group_size is not None else 128
    if resolved_bits not in {4, 8}:
        return _fail((f"unsupported_bits:{resolved_bits}",), fallback="unsupported_bits")

    plans = create_weight_plans(base_model, state)
    metadata["create_weights_targets"] = [p.module_name for p in plans]
    notes: list[str] = []
    replaced = try_native_xqt_copy(base_model, state, notes)
    lifecycle = "process_weights_after_loading:xqt_native_copy"
    selected_kernel = "xqt_native_buffer_copy"
    if replaced == 0:
        replaced = materialize_plans(
            base_model,
            state,
            plans,
            fmt=fmt,
            bits=resolved_bits,
            group_size=resolved_group,
            notes=notes,
        )
        lifecycle = "process_weights_after_loading"
        selected_kernel = "dequant_fp16_reference"

    loaded = replaced > 0
    if not loaded and not notes:
        notes.append("no_qweight_groups_found")
    if not loaded:
        return _fail(
            tuple(notes),
            fallback=notes[0] if notes else "materialize_failed",
        )

    storage = (
        "xqt_awq_gptq_int4_v1" if resolved_bits == 4 else "xqt_awq_gptq_int8_v1"
    )
    g_idx_applied: bool | None = None
    if any(str(n).startswith("g_idx_applied:") for n in notes):
        g_idx_applied = True
    elif any("desc_act_g_idx_not_fully_applied" in str(n) for n in notes):
        g_idx_applied = False
    elif info.desc_act is True:
        g_idx_applied = False
    layout = _success_layout_from_model(
        base_model,
        bits=resolved_bits,
        group_size=resolved_group,
        symmetric=info.sym,
        zero_point=info.zero_point,
        desc_act=info.desc_act,
        g_idx_applied=g_idx_applied,
        storage_layout=storage,
        selected_kernel=selected_kernel,
        scale_time="weight_offline",
        activation_granularity=None,
    )
    from xqt.contracts.runtime_quant import (
        build_runtime_quant_contract,
        first_linear_shapes,
    )
    from xqt.quant.types import QuantScheme

    global_shape, local_shape = first_linear_shapes(base_model)
    weight_dtype = "int4" if resolved_bits == 4 else "int8"
    contract = build_runtime_quant_contract(
        quant_spec=QuantScheme(
            weight_dtype=weight_dtype,
            weight_granularity="groupwise",
            group_size=resolved_group,
            activation_dtype=None,
            activation_mode="none",
            sym=True if info.sym is None else bool(info.sym),
        ),
        storage_layout=storage,
        required_kernels=(selected_kernel,),
        global_shape=global_shape,
        local_shape=local_shape,
        prefill_supported=True,
        decode_supported=True,
    )
    quantized_model = QuantizedModel(
        model=base_model,
        backend="external",
        method=fmt,
        strategy="w4a16_int4" if resolved_bits == 4 else "w8a16_int8",
        quantized_modules=[p.module_name for p in plans] if plans else [],
        compute_config={"compute": "dequant_fp16"},
        metadata={
            "external_load": True,
            "format": fmt,
            "replaced_modules": replaced,
            "bits": resolved_bits,
            "group_size": resolved_group,
            "lifecycle": lifecycle,
            "layout_kernel": layout.to_dict(),
        },
    ).with_runtime_quant_contract(contract)
    return quantized_model, ExternalLoadReport(
        format=fmt,
        model_path=str(root),
        loaded=True,
        module_count=replaced,
        notes=tuple(notes),
        metadata=metadata,
        layout_kernel=layout,
    )


__all__ = [
    "ExternalLoadReport",
    "create_weight_plans",
    "load_external_quantized_model",
    "process_weights_after_loading",
]
