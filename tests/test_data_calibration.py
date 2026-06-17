import torch
from torch.utils.data import DataLoader, TensorDataset

from xqt.data.calibration import (
    calibration_sample_count,
    calibration_sweep,
    extract_calibration_samples,
)


def test_extract_calibration_samples_from_tensor_batches() -> None:
    loader = DataLoader(
        TensorDataset(torch.ones(8, 3), torch.zeros(8, dtype=torch.long)),
        batch_size=4,
    )

    samples = extract_calibration_samples(loader, sample_limit=1)

    assert len(samples) == 1
    assert samples[0].shape == (4, 3)


def test_extract_calibration_samples_respects_sample_limit() -> None:
    loader = DataLoader(
        TensorDataset(torch.ones(20, 2)),
        batch_size=1,
    )

    samples = extract_calibration_samples(loader, sample_limit=3)

    assert len(samples) == 3
    assert all(s.shape == (1, 2) for s in samples)


def test_extract_calibration_samples_from_tuple_batches() -> None:
    loader = DataLoader(
        TensorDataset(torch.ones(4, 3), torch.zeros(4, dtype=torch.long)),
        batch_size=2,
    )

    # input_index=0 picks the first tensor
    samples = extract_calibration_samples(loader, sample_limit=1, input_index=0)

    assert len(samples) == 1
    assert samples[0].shape == (2, 3)


def test_extract_calibration_samples_from_mapping_batches() -> None:
    class MappingDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"input": torch.ones(1, 2) * (idx + 1), "label": torch.tensor(idx)}

    loader = DataLoader(MappingDataset(), batch_size=2)
    samples = extract_calibration_samples(loader, sample_limit=1, input_index=0)

    assert len(samples) == 1
    # default_collate stacks along dim 0: (2, 1, 2) from two (1, 2) samples
    assert samples[0].shape == (2, 1, 2)


def test_calibration_sample_count_returns_dataset_length() -> None:
    loader = DataLoader(
        TensorDataset(torch.ones(7, 4), torch.zeros(7, dtype=torch.long)),
        batch_size=3,
    )

    assert calibration_sample_count(loader) == 7


def test_calibration_sweep_yields_increasing_limits() -> None:
    loader = DataLoader(
        TensorDataset(torch.ones(10, 2)),
        batch_size=1,
    )

    results = list(calibration_sweep(loader, [2, 5, 8]))

    assert len(results) == 3
    assert [limit for limit, _ in results] == [2, 5, 8]
    assert [len(samples) for _, samples in results] == [2, 5, 8]
