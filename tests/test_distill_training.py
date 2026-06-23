import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.core.errors import XQTBackendError
from xqt.distill.training import train_logit_distillation


class RecordingTrainingProvider:
    def __init__(self) -> None:
        self.jobs = []

    def __call__(self, job):
        self.jobs.append(job)
        return {
            "provider": "recording",
            "steps": 2,
            "samples": 8,
            "mean_loss": 0.25,
            "last_loss": 0.2,
            "loss_history": [0.3, 0.2],
        }


def test_train_logit_distillation_delegates_to_provider() -> None:
    teacher = nn.Linear(3, 2)
    student = nn.Linear(3, 2)
    before = student.weight.detach().clone()
    inputs = torch.randn(8, 3)
    targets = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    provider = RecordingTrainingProvider()

    report = train_logit_distillation(
        student,
        teacher,
        loader,
        optimizer=None,
        temperature=2.0,
        alpha=0.7,
        training_provider=provider,
    )

    assert len(provider.jobs) == 1
    assert provider.jobs[0].mode == "distill"
    assert provider.jobs[0].model is student
    assert provider.jobs[0].teacher is teacher
    assert provider.jobs[0].train_data is loader
    assert provider.jobs[0].params["temperature"] == 2.0
    assert report.steps == 2
    assert report.samples == 8
    assert report.mean_loss == pytest.approx(0.25)
    assert report.loss_history == [0.3, 0.2]
    assert torch.equal(before, student.weight)


def test_train_logit_distillation_requires_provider() -> None:
    teacher = nn.Linear(3, 2)
    student = nn.Linear(3, 2)
    loader = [torch.randn(4, 3)]

    with pytest.raises(XQTBackendError, match="no longer owns"):
        train_logit_distillation(
            student,
            teacher,
            loader,
            optimizer=None,
        )
