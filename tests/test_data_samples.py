import torch

from xqt.data.samples import build_example_input


def test_build_example_input_default_batch_and_dtype() -> None:
    x = build_example_input([3, 16, 16])

    assert x.shape == (1, 3, 16, 16)
    assert x.dtype == torch.float32


def test_build_example_input_with_batch_and_dtype() -> None:
    x = build_example_input([10], batch_size=4, dtype=torch.float64)

    assert x.shape == (4, 10)
    assert x.dtype == torch.float64


def test_build_example_input_deterministic_with_seed() -> None:
    a = build_example_input([4], seed=42)
    b = build_example_input([4], seed=42)

    assert torch.equal(a, b)


def test_build_example_input_different_seeds_different_tensors() -> None:
    a = build_example_input([4], seed=1)
    b = build_example_input([4], seed=2)

    assert not torch.equal(a, b)
