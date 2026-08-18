"""Tests for C5 static activation scale calibration and INT8 MMA consumption."""

from __future__ import annotations

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext
from xqt.quant.calibration import (
    ActivationScaleArtifact,
    calibrate_activation_scales,
)
from xqt.quant.quantizers.int8_mma import (
    Int8MmaLinear,
    execute_int8_mma_component,
    quantize_with_int8_mma,
)
from xqt.quant.types import QuantScheme, QuantizationComponentPlan
from xqt.contracts.quant_pair import write_quant_pair


class _TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def test_calibrate_activation_scales_produces_per_tensor_minmax() -> None:
    model = _TinyLinearModel().eval()
    batches = [torch.full((2, 8), 2.0), torch.full((2, 8), -4.0)]
    scheme = QuantScheme(
        weight_dtype="int8",
        weight_granularity="per_channel",
        activation_dtype="int8",
        activation_mode="static",
    )

    artifacts = calibrate_activation_scales(model, batches, scheme)

    assert set(artifacts) == {"fc"}
    artifact = artifacts["fc"]
    assert isinstance(artifact, ActivationScaleArtifact)
    assert artifact.observer == "minmax"
    assert artifact.granularity == "per_tensor"
    assert artifact.num_samples > 0
    # max_abs of inputs is 4.0 -> scale = 4/127
    expected = 4.0 / 127.0
    assert abs(float(artifact.scale.item()) - expected) < 1e-6
    payload = artifact.to_dict()
    assert payload["module_path"] == "fc"
    assert payload["scale"] == float(artifact.scale.item())


def test_calibrate_activation_scales_rejects_non_static_scheme() -> None:
    model = _TinyLinearModel().eval()
    scheme = QuantScheme(
        weight_dtype="int8",
        weight_granularity="per_channel",
        activation_dtype="int8",
        activation_mode="dynamic",
    )
    try:
        calibrate_activation_scales(model, [torch.randn(2, 8)], scheme)
        raised = False
    except ValueError as exc:
        raised = True
        assert "static" in str(exc)
    assert raised


def test_quantize_with_int8_mma_static_scales_bake_into_module() -> None:
    model = _TinyLinearModel().eval()
    scale = 0.05
    result = quantize_with_int8_mma(
        model,
        policy={"include_module_types": ["Linear"], "exclude_name_patterns": []},
        engine="torch_int_mm",
        inplace=False,
        activation_scale_mode="static",
        activation_scales={"fc": scale},
    )

    assert isinstance(result.model.fc, Int8MmaLinear)
    assert result.model.fc.activation_scale_mode == "static"
    assert abs(float(result.model.fc.static_activation_scale.item()) - scale) < 1e-6
    assert result.metadata["static_scale_module_count"] == 1
    assert result.metadata["dynamic_fallback_module_count"] == 0
    output = result.model(torch.randn(3, 8))
    assert output.shape == (3, 4)


def test_execute_int8_mma_static_calibrates_from_context_inputs(tmp_path) -> None:
    model = _TinyLinearModel().eval()
    context = XQTContext(
        model=model,
        calibration_inputs=[torch.full((2, 8), 3.0)],
        artifact_dir=str(tmp_path),
    )
    component = QuantizationComponentPlan(
        name="root",
        backend="pytorch",
        strategy="w8a8_int8",
        compute="w8a8_int8_mma",
        policy={
            "include_module_types": ["Linear"],
            "exclude_name_patterns": [],
            "engine": "torch_int_mm",
            "activation_mode": "static",
        },
        scheme=QuantScheme(
            weight_dtype="int8",
            weight_granularity="per_channel",
            activation_dtype="int8",
            activation_mode="static",
        ),
    )

    updated, report = execute_int8_mma_component(context, model, component)

    assert isinstance(updated.fc, Int8MmaLinear)
    assert updated.fc.activation_scale_mode == "static"
    expected = 3.0 / 127.0
    assert abs(float(updated.fc.static_activation_scale.item()) - expected) < 1e-6
    lineage = report.metadata["activation_scale_lineage"]
    assert lineage["source"] == "calibrate_activation_scales"
    assert lineage["observer"] == "minmax"
    assert "fc" in lineage["scales"]
    assert report.calibration_summary is not None
    assert "activation_scale_lineage" in report.calibration_summary

    pair_dir = write_quant_pair(
        updated,
        tmp_path / "pair",
        compute_config=report.metadata.get("compute_config"),
        lineage={
            "backend": report.backend,
            "method": report.method,
            "strategy": report.strategy,
            "activation_scale_lineage": lineage,
        },
        metadata={"activation_scale_mode": "static"},
    )
    sidecar = (pair_dir / "quant.json").read_text(encoding="utf-8")
    assert "activation_scale_lineage" in sidecar
    assert "minmax" in sidecar


def test_execute_int8_mma_static_without_inputs_raises() -> None:
    model = _TinyLinearModel().eval()
    context = XQTContext(model=model)
    component = QuantizationComponentPlan(
        name="root",
        backend="pytorch",
        strategy="w8a8_int8",
        compute="w8a8_int8_mma",
        policy={
            "include_module_types": ["Linear"],
            "engine": "torch_int_mm",
            "activation_mode": "static",
        },
        scheme=QuantScheme(
            weight_dtype="int8",
            weight_granularity="per_channel",
            activation_dtype="int8",
            activation_mode="static",
        ),
    )
    try:
        execute_int8_mma_component(context, model, component)
        raised = False
    except XQTBackendError as exc:
        raised = True
        assert "ActivationScaleArtifact" in str(exc) or "calibration" in str(exc)
    assert raised
