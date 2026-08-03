"""Tests for prune granularity vocabulary, safety guards, topology diff, FLOPs,
forward diff and export readiness."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from xqt.core.types import XQTContext
from xqt.prune import (
    StructuredPruningAction,
    StructuredPruningPlan,
    apply_block_sparse_pruning,
    apply_structured_pruning,
    apply_structured_pruning_plan,
    assess_prune_safety,
    describe_prune_granularity,
    diff_module_dimensions,
    estimate_model_flops,
    normalize_prune_granularity,
    rewrite_supported_granularities,
    snapshot_module_dimensions,
    supported_prune_granularities,
)
from xqt.prune.toy_models import StructuredPruningToyCNN
from xqt.pipeline.passes import run_prune_stage
from xqt.workflows.stage_specs import PruneStageSpec


def test_granularity_vocabulary_normalizes_aliases() -> None:
    assert normalize_prune_granularity("attention_head") == "head"
    assert normalize_prune_granularity("HEAD") == "head"

    canonical = {
        item["canonical_granularity"] for item in supported_prune_granularities()
    }
    for granularity in (
        "channel",
        "filter",
        "mlp_neuron",
        "head",
        "block",
        "token",
    ):
        assert granularity in canonical

    token = describe_prune_granularity("token")
    assert token["rewrites_structure"] is False
    assert token["runtime_support"] == "metadata_only"
    assert "token" not in rewrite_supported_granularities()


def test_structured_plan_rejects_metadata_only_granularity() -> None:
    from xqt.prune import plan_structured_pruning

    with pytest.raises(ValueError, match="metadata_only"):
        plan_structured_pruning(
            StructuredPruningToyCNN(),
            target_sparsity=0.3,
            granularity="token",
        )


def test_structured_plan_rejects_unknown_granularity() -> None:
    from xqt.prune import plan_structured_pruning

    with pytest.raises(ValueError, match="Unsupported prune granularity"):
        plan_structured_pruning(
            StructuredPruningToyCNN(),
            target_sparsity=0.3,
            granularity="not_a_granularity",
        )


def _apply_toy_cnn_prune(*, example_input: object) -> tuple[StructuredPruningToyCNN, object]:
    model = StructuredPruningToyCNN()
    report = apply_structured_pruning(
        model,
        0.3,
        granularity="channel",
        scope="global",
        example_input=example_input,
    )
    return model, report


def test_structured_report_carries_topology_diff_flops_forward_and_export() -> None:
    example_input = torch.randn(2, 3, 8, 8)
    model, report = _apply_toy_cnn_prune(example_input=example_input)

    assert report.safety["passed"] is True
    assert report.forward_diff["checked"] is True
    assert report.forward_diff["status"] == "passed"
    assert report.forward_diff["tensor_count"] >= 1
    assert report.export_readiness["can_export"] is True

    # Structured pruning must be a real topology change, not mask-only.
    assert report.mask_only_modules == []
    assert report.changed_dimensions
    assert report.parameter_count_after < report.parameter_count_before
    assert report.flops_after is not None
    assert report.flops_after < report.flops_before
    assert report.flops_reduction_ratio is not None and report.flops_reduction_ratio > 0.0

    payload = report.to_dict()
    for key in (
        "removed_modules",
        "changed_dimensions",
        "mask_only_modules",
        "flops_before",
        "flops_after",
        "flops_reduction_ratio",
        "safety",
        "forward_diff",
        "export_readiness",
    ):
        assert key in payload

    # Pruned model still runs and keeps its output shape.
    model.eval()
    with torch.no_grad():
        assert model(example_input).shape == (2, 4)


def test_structured_forward_diff_is_not_run_without_example_input() -> None:
    _, report = _apply_toy_cnn_prune(example_input=None)

    assert report.forward_diff["checked"] is False
    assert report.forward_diff["status"] == "not_run"
    assert report.forward_checked is False


def test_safety_guard_rejects_broken_mlp_contract() -> None:
    class _Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    model = _Pair()
    action = StructuredPruningAction(
        action_type="mlp_neuron_group",
        module_name="fc1",
        module_type="Linear",
        granularity="mlp_neuron",
        original_units=8,
        keep_indices=list(range(10)),
        prune_indices=[],
        consumer_name="fc2",
        consumer_type="Linear",
    )

    safety = assess_prune_safety(model, [action], task_type="transformer")
    assert safety.passed is False
    assert any("larger than out_features" in item for item in safety.violations)

    plan = StructuredPruningPlan(
        method="structured",
        granularity="mlp_neuron",
        scope="global",
        target_sparsity=0.5,
        importance_metric="l1",
        adapters=["mlp_pair_adapter"],
        structure_families=["transformer"],
        actions=[action],
    )
    with pytest.raises(ValueError, match="safety guard rejected"):
        apply_structured_pruning_plan(model, plan)


def test_safety_guard_blocks_detection_head_modules() -> None:
    class _Detection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(8, 8)
            self.detect_head = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.detect_head(self.backbone(x))

    model = _Detection()
    action = StructuredPruningAction(
        action_type="mlp_neuron_group",
        module_name="detect_head",
        module_type="Linear",
        granularity="mlp_neuron",
        original_units=8,
        keep_indices=[0, 1, 2],
        prune_indices=[3, 4, 5, 6, 7],
        consumer_name=None,
        consumer_type=None,
    )

    safety = assess_prune_safety(model, [action], task_type="detection")
    assert "detect_head" in safety.blocked_modules
    assert safety.passed is False


def test_mask_based_reports_list_mask_only_modules() -> None:
    model = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(4, 4, bias=False))

    nm_report = apply_block_sparse_pruning(
        model,
        target_sparsity=0.5,
        block_shape=(2, 2),
    )
    assert nm_report.mask_only_modules == [layer.module_name for layer in nm_report.layers]

    from xqt.prune import apply_nm_structured_sparsity

    model = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(4, 4, bias=False))
    nm = apply_nm_structured_sparsity(model, pattern_n=2, pattern_m=4)
    assert nm.mask_only_modules == [layer.module_name for layer in nm.layers]


def test_dimension_snapshot_diff_reports_removed_and_changed() -> None:
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.proj(x))

    model = _Model()
    before = snapshot_module_dimensions(model)
    model.proj = nn.Linear(2, 4)
    delattr(model, "head")
    removed, changed = diff_module_dimensions(before, snapshot_module_dimensions(model))

    assert [item["module_name"] for item in removed] == ["head"]
    proj_change = next(item for item in changed if item["module_name"] == "proj")
    assert proj_change["changed_dimensions"][0]["dimension"] == "in_features"
    assert proj_change["changed_dimensions"][0]["before"] == 4
    assert proj_change["changed_dimensions"][0]["after"] == 2


def test_flops_estimate_is_shape_based_and_separate_from_latency() -> None:
    model = StructuredPruningToyCNN()
    estimate = estimate_model_flops(model)

    assert estimate["flops"] > 0.0
    assert estimate["estimate_kind"] == "shape_based_per_output_position"
    assert estimate["per_module"]


def _prune_stage_context(*, task_type: str) -> tuple[XQTContext, object]:
    model = StructuredPruningToyCNN()
    context = XQTContext(
        model=model,
        reference_model=copy.deepcopy(model),
        example_inputs=torch.randn(2, 3, 8, 8),
        device="cpu",
        artifact_dir="artifacts/xqt/tests/prune_topology_stage",
        project_name="prune_topology_stage",
        task_type=task_type,
    )
    return context, torch.randn(2, 3, 8, 8)


def test_run_prune_stage_structured_writes_topology_and_readiness() -> None:
    context, _ = _prune_stage_context(task_type="classification")
    output = run_prune_stage(
        context,
        PruneStageSpec(
            method="structured",
            target_sparsity=0.3,
            granularity="channel",
        ),
    )

    prune = output.metrics["prune"]
    assert prune["safety"]["passed"] is True
    assert prune["forward_diff"]["checked"] is True
    assert prune["forward_diff"]["status"] == "passed"
    assert prune["export_readiness"]["can_export"] is True
    assert prune["flops_after"] < prune["flops_before"]
    assert prune["mask_only_modules"] == []
    assert prune["changed_dimensions"]


def test_run_prune_stage_detection_structured_skip_keeps_honest_safety() -> None:
    context, _ = _prune_stage_context(task_type="detection")
    output = run_prune_stage(
        context,
        PruneStageSpec(
            method="structured",
            target_sparsity=0.3,
        ),
    )

    prune = output.metrics["prune"]
    assert prune["execution_state"] == "skipped"
    assert prune["applied"] is False
    assert prune["canonical_granularity"] == "channel"
    assert prune["safety"]["passed"] is True
    assert "detection_head_guard" in {
        check["name"] for check in prune["safety"]["checks"]
    }
