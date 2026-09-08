"""Tests for XQT-002 (Stage transaction and rollback) and XQT-003 (Lineage and identity)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from xqt.workflows import XQTOptimizationSession


class _SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_stage_rejected_rolls_back_model_and_context(tmp_path: Path) -> None:
    """XQT-002: When a stage is rejected (accepted=False), model and context revert."""
    model = _SimpleModel().eval()
    initial_state = _clone_state(model)

    session = XQTOptimizationSession(
        project={"name": "test_reject", "artifact_dir": str(tmp_path / "artifacts")},
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    # Run a quant stage with an impossible acceptance threshold (e.g. min_speedup=999.0)
    result = session.quant(
        name="quant_rejected",
        backend="pytorch",
        method="awq",
        strategy="w4a16_int4",
        policy={"include_module_names": ["fc1"], "group_size": 8, "bits": 4},
        accept={"min_speedup": 999.0},
    )

    assert result.accepted is False
    # Context model must match initial state
    current_state = _clone_state(session._state.context.model)
    assert set(current_state.keys()) == set(initial_state.keys())
    for k in initial_state:
        assert torch.equal(current_state[k], initial_state[k])

    # Model topology unchanged
    module_names = [name for name, _ in session._state.context.model.named_modules()]
    assert "fc1" in module_names and "fc2" in module_names
    assert isinstance(session._state.context.model.fc1, nn.Linear)

    # Config stages should not contain the rejected stage
    assert "quant_rejected" not in [s.name for s in session._state.config.stages]
    # Current stage should remain baseline
    assert session.current_stage == "baseline"


def test_stage_exception_rolls_back_and_allows_retry(tmp_path: Path) -> None:
    """XQT-002: When a runner raises an exception, model reverts and session can retry."""
    model = _SimpleModel().eval()
    initial_state = _clone_state(model)

    session = XQTOptimizationSession(
        project={"name": "test_exception", "artifact_dir": str(tmp_path / "artifacts")},
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    # Simulate an exception in runner
    with patch("xqt.workflows.optimization._run_quant", side_effect=RuntimeError("simulated crash")):
        with pytest.raises(RuntimeError, match="simulated crash"):
            session.quant(
                name="quant_crashed",
                backend="pytorch",
                method="awq",
                strategy="w4a16_int4",
                policy={"include_module_names": ["fc1"], "group_size": 8, "bits": 4},
            )

    # Model must be restored
    current_state = _clone_state(session._state.context.model)
    for k in initial_state:
        assert torch.equal(current_state[k], initial_state[k])

    # Config stages must not contain crashed stage
    assert "quant_crashed" not in [s.name for s in session._state.config.stages]

    # Session can retry immediately with a working stage
    retry_result = session.quant(
        name="quant_retry",
        backend="pytorch",
        method="awq",
        strategy="w4a16_int4",
        policy={"include_module_names": ["fc1"], "group_size": 8, "bits": 4},
    )
    assert retry_result.accepted is True
    assert "quant_retry" in [s.name for s in session._state.config.stages]
    assert session.current_stage == "quant_retry"


def test_lineage_dag_and_use_switching(tmp_path: Path) -> None:
    """XQT-003: Switching via use() updates current_stage and avoids DAG self-reference."""
    model = _SimpleModel().eval()

    session = XQTOptimizationSession(
        project={"name": "test_lineage", "artifact_dir": str(tmp_path / "artifacts")},
        model=model,
        example_inputs=torch.randn(2, 8),
    )

    assert session.baseline_stage == "baseline"
    assert session.current_stage == "baseline"

    # 1. Baseline -> Stage A (quant)
    stage_a = session.quant(
        name="stage_A",
        backend="pytorch",
        method="awq",
        strategy="w4a16_int4",
        policy={"include_module_names": ["fc1"], "group_size": 8, "bits": 4},
    )
    assert stage_a.accepted is True
    assert session.current_stage == "stage_A"
    stage_a_record = next(s for s in session.session_stages if s.name == "stage_A")
    assert stage_a_record.created_by.from_stage == "baseline"

    # 2. Switch back to baseline
    session.use("baseline")
    assert session.current_stage == "baseline"

    # 3. Baseline -> Stage B (should have from_stage == "baseline", NOT "stage_A")
    stage_b = session.quant(
        name="stage_B",
        backend="pytorch",
        method="awq",
        strategy="w4a16_int4",
        policy={"include_module_names": ["fc2"], "group_size": 8, "bits": 4},
    )
    assert stage_b.accepted is True
    stage_b_record = next(s for s in session.session_stages if s.name == "stage_B")
    assert stage_b_record.created_by.from_stage == "baseline"
    assert session.current_stage == "stage_B"

    # 4. Observation stage (benchmark / analyze) does NOT advance current_stage
    bench_result = session.benchmark(name="bench_B")
    assert bench_result.accepted is True
    assert session.current_stage == "stage_B"

    # 5. Check no self-reference in any stage reports or lineage
    for stage in session.session_stages:
        if stage.created_by is not None:
            assert stage.created_by.from_stage != stage.name
