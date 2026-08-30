"""Build LayoutKernelReport for quant apply paths (U6 / V2).

Attaches selected/fallback kernel facts so RuntimeManifest and quant pair
sidecars do not rely on free-form note strings alone. Preference is written at
quantize time; realized engine is refreshed after forward via
``refresh_layout_kernel_after_forward``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from torch import nn

from xqt.contracts.layout_kernel_report import (
    LayoutKernelReport,
    layout_report_from_module_shapes,
)
from xqt.kernels.engine_resolve import (
    get_engine_registration,
    map_primary_kernel_to_engine,
    normalize_engine_name,
)
from xqt.compression.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear


def resolve_selected_kernel_name(
    preferred: str | None,
    *,
    fallback: str | None = None,
) -> tuple[str, str | None]:
    """Normalize a preferred engine/kernel token; return (selected, fallback_reason)."""

    if preferred is None or not str(preferred).strip():
        selected = str(fallback or "torch")
        return selected, "preferred_kernel_absent"
    token = str(preferred).strip()
    mapped = map_primary_kernel_to_engine(token) or normalize_engine_name(token)
    registration = get_engine_registration(mapped)
    if registration is None:
        return str(fallback or "torch"), f"unknown_kernel:{token}"
    if registration.maturity not in {"executable", "reference_guarded"}:
        return (
            str(fallback or registration.name),
            f"non_executable_maturity:{registration.maturity}",
        )
    return registration.name, None


def layout_report_for_awq_gptq_model(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    selected_kernel: str = "dequant_fp16_reference",
    fallback_reason: str | None = None,
    scale_time: str = "weight_offline",
    desc_act: bool | None = False,
    g_idx_applied: bool | None = False,
) -> LayoutKernelReport:
    """First AWQ/GPTQ Linear shapes + selected kernel for apply metadata."""

    for module in model.modules():
        if isinstance(module, AWQGPTQWeightOnlyLinear):
            return layout_report_from_module_shapes(
                bits=int(module.bits),
                group_size=int(module.group_size),
                symmetric=True,
                zero_point=False,
                desc_act=desc_act,
                g_idx_applied=g_idx_applied,
                out_features=int(module.output_features),
                in_features=int(module.input_features),
                padded_in_features=int(module.padded_input_features),
                storage_layout=(
                    "xqt_awq_gptq_int4_v1"
                    if int(module.bits) == 4
                    else "xqt_awq_gptq_int8_v1"
                ),
                selected_kernel=selected_kernel,
                fallback_reason=fallback_reason,
                scale_time=scale_time,
            )
    return layout_report_from_module_shapes(
        bits=bits,
        group_size=group_size,
        symmetric=True,
        zero_point=False,
        desc_act=desc_act,
        g_idx_applied=g_idx_applied,
        out_features=0,
        in_features=0,
        padded_in_features=0,
        storage_layout=(
            "xqt_awq_gptq_int4_v1" if bits == 4 else "xqt_awq_gptq_int8_v1"
        ),
        selected_kernel=selected_kernel,
        fallback_reason=fallback_reason or "no_awq_gptq_linear",
        scale_time=scale_time,
    )


def layout_report_for_int8_mma(
    model: nn.Module,
    *,
    engine_preference: str,
    fallback_engine: str,
    activation_scale_mode: str,
    scale_time: str,
) -> LayoutKernelReport:
    """Int8 MMA apply report: preferred engine mapped into selected_kernel."""

    selected, fallback_reason = resolve_selected_kernel_name(
        engine_preference,
        fallback=fallback_engine,
    )
    out_f, in_f = 0, 0
    for module in model.modules():
        out_attr = getattr(module, "output_features", None)
        in_attr = getattr(module, "input_features", None)
        if out_attr is not None and in_attr is not None:
            out_f, in_f = int(out_attr), int(in_attr)
            break
        if isinstance(module, nn.Linear):
            out_f, in_f = int(module.out_features), int(module.in_features)
            break
    registration = get_engine_registration(selected)
    min_cap = None if registration is None else registration.min_capability
    return layout_report_from_module_shapes(
        bits=8,
        group_size=None,
        symmetric=True,
        zero_point=False,
        desc_act=False,
        g_idx_applied=None,
        out_features=out_f,
        in_features=in_f,
        padded_in_features=in_f,
        storage_layout="xqt_int8_mma_v1",
        selected_kernel=selected,
        fallback_reason=fallback_reason,
        min_capability=min_cap,
        scale_time=scale_time,
        activation_granularity="per_tensor",
    )


def layout_report_for_fp4_dynamic(
    model: nn.Module,
    *,
    engine_preference: str,
    fp4_format: str,
) -> LayoutKernelReport:
    """FP4 dynamic apply report with resolved selected_kernel."""

    selected, fallback_reason = resolve_selected_kernel_name(
        engine_preference,
        fallback="torch",
    )
    out_f, in_f = 0, 0
    for module in model.modules():
        out_attr = getattr(module, "output_features", None)
        in_attr = getattr(module, "input_features", None)
        if out_attr is not None and in_attr is not None:
            out_f, in_f = int(out_attr), int(in_attr)
            break
        if isinstance(module, nn.Linear):
            out_f, in_f = int(module.out_features), int(module.in_features)
            break
    return layout_report_from_module_shapes(
        bits=4,
        group_size=None,
        symmetric=True,
        zero_point=False,
        desc_act=False,
        g_idx_applied=None,
        out_features=out_f,
        in_features=in_f,
        padded_in_features=in_f,
        storage_layout=f"xqt_fp4_dynamic_{fp4_format}_v1",
        selected_kernel=selected,
        fallback_reason=fallback_reason,
        scale_time="activation_dynamic",
        activation_granularity="block",
    )


def attach_layout_kernel_metadata(
    metadata: dict[str, Any],
    report: LayoutKernelReport,
) -> dict[str, Any]:
    """Copy metadata and set layout_kernel (canonical apply diagnostic)."""

    out = dict(metadata)
    out["layout_kernel"] = report.to_dict()
    return out


def selected_kernel_from_execution_metadata(
    execution: Mapping[str, Any] | None,
    *,
    default: str = "not_run",
) -> str:
    """Read engine from module execution_metadata after a forward."""

    if not isinstance(execution, Mapping):
        return default
    engine = execution.get("engine")
    if engine is None or not str(engine).strip():
        return default
    return str(engine)


def with_realized_selected_kernel(
    report: LayoutKernelReport,
    *,
    selected_kernel: str,
    fallback_reason: str | None = None,
) -> LayoutKernelReport:
    """Return a copy of ``report`` with the realized engine after forward (V2)."""

    return replace(
        report,
        selected_kernel=str(selected_kernel),
        fallback_reason=(
            report.fallback_reason if fallback_reason is None else fallback_reason
        ),
    )


def collect_realized_kernel(
    model: nn.Module,
) -> tuple[str | None, str | None]:
    """Scan modules for the first post-forward ``execution_metadata`` engine."""

    for module in model.modules():
        getter = getattr(module, "execution_metadata", None)
        if not callable(getter):
            continue
        execution = getter()
        if not isinstance(execution, Mapping):
            continue
        engine = selected_kernel_from_execution_metadata(execution)
        if engine == "not_run":
            continue
        fallback: str | None = None
        reason = execution.get("reason")
        if isinstance(reason, str) and "fallback" in reason.lower():
            fallback = reason
        raw_fallbacks = execution.get("runtime_fallbacks")
        if isinstance(raw_fallbacks, (list, tuple)) and raw_fallbacks:
            last = raw_fallbacks[-1]
            if isinstance(last, Mapping) and last.get("reason") is not None:
                fallback = str(last["reason"])
        return engine, fallback
    return None, None


def refresh_layout_kernel_after_forward(
    metadata: Mapping[str, Any] | None,
    model: nn.Module,
) -> dict[str, Any]:
    """Update ``layout_kernel.selected_kernel`` from realized forward engine (V2)."""

    out: dict[str, Any] = {} if metadata is None else dict(metadata)
    realized, fallback = collect_realized_kernel(model)
    if realized is None:
        return out
    raw = out.get("layout_kernel")
    if isinstance(raw, Mapping):
        from xqt.core.errors import XQTConfigError

        try:
            report = LayoutKernelReport.from_dict(raw)
        except (XQTConfigError, TypeError, ValueError, KeyError):
            report = LayoutKernelReport(
                storage_layout=str(raw.get("storage_layout") or "unknown"),
                selected_kernel=(
                    str(raw.get("selected_kernel"))
                    if raw.get("selected_kernel") is not None
                    else None
                ),
            )
        updated = with_realized_selected_kernel(
            report,
            selected_kernel=realized,
            fallback_reason=fallback,
        )
        out["layout_kernel"] = updated.to_dict()
    else:
        out["layout_kernel"] = LayoutKernelReport(
            storage_layout="unknown",
            selected_kernel=realized,
            fallback_reason=fallback,
        ).to_dict()
    return out


__all__ = [
    "attach_layout_kernel_metadata",
    "collect_realized_kernel",
    "layout_report_for_awq_gptq_model",
    "layout_report_for_fp4_dynamic",
    "layout_report_for_int8_mma",
    "refresh_layout_kernel_after_forward",
    "resolve_selected_kernel_name",
    "selected_kernel_from_execution_metadata",
    "with_realized_selected_kernel",
]
