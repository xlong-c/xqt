import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.distill.training import train_logit_distillation


def test_train_logit_distillation_updates_student_and_reports_metrics() -> None:
    teacher = nn.Linear(3, 2)
    student = nn.Linear(3, 2)
    with torch.no_grad():
        teacher.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        teacher.bias.zero_()

    before = student.weight.detach().clone()
    inputs = torch.randn(8, 3)
    targets = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    report = train_logit_distillation(
        student,
        teacher,
        loader,
        optimizer,
        temperature=2.0,
        alpha=0.7,
    )

    assert report.steps == 2
    assert report.samples == 8
    assert report.mean_loss > 0.0
    assert len(report.loss_history) == 2
    assert not torch.equal(before, student.weight)
    assert report.to_dict()["steps"] == 2
