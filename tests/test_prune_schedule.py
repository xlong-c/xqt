import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.core.errors import XQTBackendError
from xqt.prune import PruningSchedule, run_prune_kd_loop


def test_pruning_schedule_values_linear_and_one_shot() -> None:
    assert PruningSchedule(target_sparsity=0.6, steps=3).values() == pytest.approx(
        [0.2, 0.4, 0.6]
    )
    assert PruningSchedule(
        target_sparsity=0.5,
        start_sparsity=0.1,
        steps=4,
    ).values() == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert PruningSchedule(
        target_sparsity=0.7,
        steps=3,
        schedule="one_shot",
    ).values() == [0.7]


@pytest.mark.parametrize(
    "schedule",
    [
        PruningSchedule(target_sparsity=-0.1),
        PruningSchedule(target_sparsity=1.1),
        PruningSchedule(target_sparsity=0.5, steps=0),
        PruningSchedule(target_sparsity=0.5, schedule="cosine"),
    ],
)
def test_pruning_schedule_rejects_invalid_values(schedule: PruningSchedule) -> None:
    with pytest.raises(ValueError):
        schedule.values()


class RecordingRecoveryProvider:
    def __init__(self) -> None:
        self.jobs = []

    def __call__(self, job):
        self.jobs.append(job)
        return {
            "steps": 1,
            "samples": 3,
            "mean_loss": 0.1,
            "last_loss": 0.1,
            "loss_history": [0.1],
        }


def test_run_prune_kd_loop_applies_schedule_and_delegates_recovery() -> None:
    student = nn.Linear(4, 2)
    teacher = nn.Linear(4, 2)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 4), torch.randint(0, 2, (6,))),
        batch_size=3,
    )
    provider = RecordingRecoveryProvider()

    report = run_prune_kd_loop(
        student,
        teacher,
        loader,
        schedule=PruningSchedule(target_sparsity=0.5, steps=2),
        kd_steps_per_prune=1,
        training_provider=provider,
    )

    assert len(report.steps) == 2
    assert report.final_sparsity == pytest.approx(0.5)
    assert report.steps[0].distillation is not None
    assert report.steps[0].distillation.steps == 1
    assert [job.mode for job in provider.jobs] == ["distill", "distill"]
    assert report.to_dict()["final_sparsity"] == pytest.approx(0.5)


def test_run_prune_kd_loop_rejects_legacy_optimizer_recovery_without_provider() -> None:
    student = nn.Linear(4, 2)
    teacher = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

    with pytest.raises(XQTBackendError, match="delegated to a provider"):
        run_prune_kd_loop(
            student,
            teacher,
            [torch.randn(2, 4)],
            schedule=PruningSchedule(target_sparsity=0.5, steps=1),
            optimizer=optimizer,
        )
