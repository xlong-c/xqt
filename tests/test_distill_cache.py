import torch
from torch import nn

from xqt.distill.cache import (
    TeacherOutput,
    TeacherOutputCache,
    cache_teacher_outputs,
)


def test_teacher_output_cache_writes_and_reads_tensors(tmp_path) -> None:
    cache = TeacherOutputCache(tmp_path)
    output = TeacherOutput(
        logits=torch.tensor([[1.0, 2.0]]),
        features={"hidden": torch.tensor([[3.0, 4.0]])},
        metadata={"sample": 1},
    )

    record = cache.write("sample-1", output)
    loaded = cache.read("sample-1")

    assert record.path.is_file()
    assert cache.exists("sample-1") is True
    assert torch.equal(loaded.logits, output.logits)
    assert torch.equal(loaded.features["hidden"], output.features["hidden"])
    assert loaded.metadata == {"sample": 1}


def test_cache_teacher_outputs_saves_logits_and_features(tmp_path) -> None:
    teacher = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    teacher.train()
    batches = [
        (torch.ones(2, 3), torch.tensor([0, 1])),
        (torch.zeros(1, 3), torch.tensor([1])),
    ]
    cache = TeacherOutputCache(tmp_path)

    records = cache_teacher_outputs(
        teacher,
        batches,
        cache,
        feature_module_names=["0"],
        key_prefix="teacher",
    )

    assert len(records) == 2
    assert teacher.training is True
    first = cache.read("teacher_0")
    assert first.logits.shape == (2, 2)
    assert first.features["0"].shape == (2, 4)
    assert first.metadata == {"batch_index": 0}
