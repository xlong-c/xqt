from __future__ import annotations

import pytest
import torch

from xqt.contracts import QuantizedModelPayload, RuntimePlanPayload
from xqt.core.errors import XQTConfigError
from xqt.core.reporting import build_stage_report
from xqt.workflows.stage_specs import build_stage_spec


def _composite_gemm_spec(
    *,
    selected_groups: list[int] | None = None,
    preferred_mode: str = "fused",
    allowed_modes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "partition": {
            "group_axis": "k",
            "group_size": 64,
            "group_count": 4,
            "selected_groups": list(selected_groups or [1, 3]),
        },
        "selected_branch": {
            "name": "selected",
            "format": "w4a4",
            "weight_format": "int4",
            "scale_format": "fp16",
        },
        "residual_branch": {
            "name": "residual",
            "format": "bf16",
            "weight_format": "bf16",
        },
        "preferred_mode": preferred_mode,
        "allowed_modes": list(allowed_modes or ["fused", "split"]),
        "accumulation_dtype": "fp32",
        "fallback": "reject",
    }


def _module_contract(
    *,
    preferred_mode: str = "fused",
    allowed_modes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "operator_kind": "linear",
        "policy": {
            "activation": "fp16",
            "weight": "fp16",
            "bias": "fp16",
            "mma": "fp16",
            "accum": "fp32",
            "output": "fp16",
            "composite_gemm": _composite_gemm_spec(
                preferred_mode=preferred_mode,
                allowed_modes=allowed_modes,
            ),
        },
    }


def test_quant_stage_spec_rejects_duplicate_composite_selected_groups() -> None:
    with pytest.raises(XQTConfigError, match="duplicate group index 1"):
        build_stage_spec(
            "quant",
            {
                "backend": "pytorch",
                "method": "awq",
                "strategy": "weight_only_int4",
                "composite_gemm": _composite_gemm_spec(selected_groups=[1, 1]),
            },
        )


def test_quant_stage_spec_rejects_preferred_mode_outside_allowed_modes() -> None:
    with pytest.raises(XQTConfigError, match="preferred_mode"):
        build_stage_spec(
            "quant",
            {
                "backend": "pytorch",
                "method": "awq",
                "strategy": "weight_only_int4",
                "composite_gemm": _composite_gemm_spec(
                    preferred_mode="fused",
                    allowed_modes=["split", "reference"],
                ),
            },
        )


def test_quantized_model_payload_materializes_composite_partition_artifact() -> None:
    payload = QuantizedModelPayload.from_stage_metrics(
        stage_name="quant",
        source_model_stage="baseline",
        model=torch.nn.Linear(8, 8),
        metrics={
            "backend": "pytorch",
            "method": "awq",
            "strategy": "weight_only_int4",
            "components": [{"component_name": "model"}],
        },
        module_contract=_module_contract(),
    )

    serialized = payload.to_dict()
    artifacts = serialized["composite_quant_artifacts"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["requested_mode"] == "fused"
    assert artifact["actual_mode"] == "split"
    assert artifact["partition_group_count"] == 2
    assert artifact["residual_group_count"] == 2
    assert len(artifact["partition_map"]) == 4
    assert {item["branch"] for item in artifact["partition_map"]} == {
        "selected",
        "residual",
    }
    assert len(artifact["branches"]) == 2
    branch_names = {branch["name"] for branch in artifact["branches"]}
    assert branch_names == {"selected", "residual"}


def test_runtime_plan_payload_downgrades_requested_fused_mode_to_split() -> None:
    payload = RuntimePlanPayload.from_stage_metrics(
        stage_name="operator",
        source_model_stage="quant",
        metrics={"targets": [{"engine": "tilelang"}]},
        module_contract=_module_contract(),
    )

    serialized = payload.to_dict()
    assert serialized["composite_precision"] is True
    assert serialized["requested_mode"] == "fused"
    assert serialized["actual_mode"] == "split"
    assert serialized["kernel_count"] == 2
    assert serialized["fallback_reason"] == "requested_mode_fused_unavailable"


def test_stage_report_records_requested_and_actual_composite_modes() -> None:
    report = build_stage_report(
        stage_name="operator_linear",
        stage_kind="operator",
        accepted=True,
        message="ok",
        metrics={
            "targets": [
                {
                    "engine": "tilelang",
                    "module_contract": _module_contract(),
                }
            ]
        },
        artifacts={},
    ).to_dict()

    execution = report["execution"]
    assert execution["composite_precision"] is True
    assert execution["requested_mode"] == "fused"
    assert execution["actual_mode"] == "split"
