import pytest
import torch
from torch import nn

from xqt.distill.cache import (
    TeacherOutput,
    TeacherOutputCache,
    batch_identity,
    cache_teacher_outputs,
    dataset_signature,
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
    first_identity = batch_identity(batches[0])
    first_key = cache.key_for_identity(first_identity, prefix="teacher")
    first = cache.read(first_key, expected_identity=first_identity)
    assert first.logits.shape == (2, 2)
    assert first.features["0"].shape == (2, 4)
    assert first.metadata["batch_index"] == 0
    assert first.metadata["sample_identity"] == first_identity
    assert records[0].sample_identity == first_identity


def test_teacher_output_cache_rejects_identity_mismatch(tmp_path) -> None:
    cache = TeacherOutputCache(tmp_path)
    output = TeacherOutput(logits=torch.tensor([[1.0, 2.0]]))
    record = cache.write("sample-1", output, sample_identity="identity-a")

    assert record.sample_identity == "identity-a"
    with pytest.raises(ValueError, match="identity"):
        cache.read("sample-1", expected_identity="identity-b")


def test_teacher_output_cache_reads_dataset_signature(tmp_path) -> None:
    cache = TeacherOutputCache(tmp_path)
    output = TeacherOutput(logits=torch.tensor([[1.0, 2.0]]))

    record = cache.write(
        "sample-1",
        output,
        sample_identity="identity-a",
        dataset_signature="dataset-v1",
    )
    loaded = cache.read(
        "sample-1",
        expected_identity="identity-a",
        expected_dataset_signature="dataset-v1",
    )

    assert record.dataset_signature == "dataset-v1"
    assert loaded.metadata["dataset_signature"] == "dataset-v1"


def test_teacher_output_cache_rejects_dataset_signature_mismatch(tmp_path) -> None:
    cache = TeacherOutputCache(tmp_path)
    output = TeacherOutput(logits=torch.tensor([[1.0, 2.0]]))
    cache.write("sample-1", output, dataset_signature="dataset-v1")

    with pytest.raises(ValueError, match="signature"):
        cache.read("sample-1", expected_dataset_signature="dataset-v2")


def test_cache_teacher_outputs_records_dataset_signature(tmp_path) -> None:
    teacher = nn.Linear(3, 2)
    batches = [
        (torch.ones(2, 3), torch.tensor([0, 1])),
        (torch.zeros(1, 3), torch.tensor([1])),
    ]
    sample_identities = ["sample-a", "sample-b"]
    expected_signature = dataset_signature(sample_identities)
    cache = TeacherOutputCache(tmp_path)

    records = cache_teacher_outputs(
        teacher,
        batches,
        cache,
        key_prefix="teacher",
        sample_identities=sample_identities,
    )

    assert [record.dataset_signature for record in records] == [
        expected_signature,
        expected_signature,
    ]
    loaded = cache.read(
        cache.key_for_identity("sample-a", prefix="teacher"),
        expected_identity="sample-a",
        expected_dataset_signature=expected_signature,
    )
    assert loaded.metadata["dataset_signature"] == expected_signature


def test_cache_teacher_outputs_allows_data_version_override(tmp_path) -> None:
    teacher = nn.Linear(3, 2)
    batches = [(torch.ones(1, 3), torch.tensor([0]))]
    cache = TeacherOutputCache(tmp_path)

    records = cache_teacher_outputs(
        teacher,
        batches,
        cache,
        key_prefix="teacher",
        sample_identities=["sample-a"],
        data_version="dataset-v2",
    )

    assert records[0].dataset_signature == "dataset-v2"
    loaded = cache.read(
        cache.key_for_identity("sample-a", prefix="teacher"),
        expected_identity="sample-a",
        expected_dataset_signature="dataset-v2",
    )
    assert loaded.metadata["dataset_signature"] == "dataset-v2"
