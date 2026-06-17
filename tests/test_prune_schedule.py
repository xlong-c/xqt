import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def test_run_prune_kd_loop_applies_schedule_and_distills() -> None:
    student = nn.Linear(4, 2)
    teacher = nn.Linear(4, 2)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 4), torch.randint(0, 2, (6,))),
        batch_size=3,
    )
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

    report = run_prune_kd_loop(
        student,
        teacher,
        loader,
        schedule=PruningSchedule(target_sparsity=0.5, steps=2),
        optimizer=optimizer,
        kd_steps_per_prune=1,
    )

    assert len(report.steps) == 2
    assert report.final_sparsity == pytest.approx(0.5)
    assert report.steps[0].distillation is not None
    assert report.steps[0].distillation.steps == 1
    assert report.to_dict()["final_sparsity"] == pytest.approx(0.5)
