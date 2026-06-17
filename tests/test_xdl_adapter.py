from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xqt.xdl_adapter import (
    load_checkpoint_into_model,
    xdl_checkpoint_to_xqt_context,
    xdl_setup_to_xqt_context,
)


@dataclass
class FakeTrainSetup:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    device: str = "cpu"
    batch_size: int = 2
    num_epochs: int = 1


def _loader() -> DataLoader:
    dataset = TensorDataset(torch.randn(4, 2), torch.randint(0, 2, (4,)))
    return DataLoader(dataset, batch_size=2)


def test_xdl_setup_to_xqt_context_maps_model_and_dataloaders(tmp_path) -> None:
    model = nn.Linear(2, 2)
    setup = FakeTrainSetup(
        model=model,
        train_loader=_loader(),
        val_loader=_loader(),
        test_loader=_loader(),
    )

    context = xdl_setup_to_xqt_context(
        setup,
        {
            "project": {"artifact_dir": str(tmp_path / "artifacts")},
            "model": {"device": "cpu"},
        },
        include_test_loader=True,
    )

    assert context.model is model
    assert set(context.data) == {"train", "validation", "test"}
    assert context.metrics["xdl_setup"]["device"] == "cpu"
    assert context.manifest is not None


def test_load_checkpoint_into_model_supports_plain_and_xdl_state(tmp_path) -> None:
    source = nn.Linear(2, 2)
    plain = tmp_path / "plain.pt"
    torch.save(source.state_dict(), plain)
    loaded = nn.Linear(2, 2)
    load_checkpoint_into_model(loaded, plain)

    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])

    xdl_path = tmp_path / "xdl.pt"
    torch.save({"state_dict": {"model": source.state_dict()}}, xdl_path)
    loaded_xdl = nn.Linear(2, 2)
    load_checkpoint_into_model(loaded_xdl, xdl_path)

    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, loaded_xdl.state_dict()[name])


def test_xdl_checkpoint_to_xqt_context_sets_source_checkpoint(tmp_path) -> None:
    source = nn.Linear(2, 2)
    checkpoint = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint)
    model = nn.Linear(2, 2)

    context = xdl_checkpoint_to_xqt_context(
        model,
        checkpoint,
        {
            "project": {"artifact_dir": str(tmp_path / "artifacts")},
            "model": {"device": "cpu"},
        },
    )

    assert context.model is model
    assert context.manifest is not None
    assert context.manifest.source_checkpoint == str(checkpoint)
