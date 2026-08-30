"""Per-expert LayoutKernelReport helpers for MoE weight-only quant (T11)."""

from __future__ import annotations

from typing import Any, Sequence

from torch import nn

from xqt.contracts.layout_kernel_report import layout_report_from_module_shapes
from xqt.compression.quant.quantizers.awq_gptq_weight_only import AWQGPTQWeightOnlyLinear


def build_expert_layout_reports(
    model: nn.Module,
    expert_names: Sequence[str],
    *,
    shared_expert_names: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build T2-style layout dicts for each quantized expert Linear."""

    shared = set(shared_expert_names)
    reports: list[dict[str, Any]] = []
    for name in expert_names:
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        if not isinstance(module, AWQGPTQWeightOnlyLinear):
            continue
        reports.append(
            {
                "module": name,
                "role": "shared_expert" if name in shared else "expert",
                **layout_report_from_module_shapes(
                    bits=int(module.bits),
                    group_size=int(module.group_size),
                    symmetric=True,
                    zero_point=False,
                    desc_act=False,
                    g_idx_applied=False,
                    out_features=int(module.output_features),
                    in_features=int(module.input_features),
                    padded_in_features=int(module.padded_input_features),
                    storage_layout=(
                        "xqt_awq_gptq_int4_v1"
                        if int(module.bits) == 4
                        else "xqt_awq_gptq_int8_v1"
                    ),
                    selected_kernel="dequant_fp16_reference",
                    scale_time="weight_offline",
                ).to_dict(),
            }
        )
    return reports


__all__ = ["build_expert_layout_reports"]
