import pytest
import torch
from torch import nn

from xqt.prune import (
    collect_module_importance,
    apply_global_l1_unstructured_pruning,
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_out_features,
    rank_prune_candidates,
    remove_pruning_reparameterization,
    summarize_pruning,
    tensor_sparsity,
)


def test_global_l1_unstructured_pruning_adds_masks_and_reports_sparsity() -> None:
    model = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter).float())

    report = apply_global_l1_unstructured_pruning(model, amount=0.5)

    assert report.total_parameters == 18
    assert report.zero_parameters == 9
    assert report.sparsity == pytest.approx(0.5)
    assert hasattr(model[0], "weight_mask")
    assert hasattr(model[2], "weight_mask")

    remove_pruning_reparameterization(model)

    assert not hasattr(model[0], "weight_mask")
    assert not hasattr(model[2], "weight_mask")
    assert summarize_pruning(model).sparsity == pytest.approx(0.5)


def test_global_pruning_rejects_invalid_amount() -> None:
    with pytest.raises(ValueError, match="amount must be"):
        apply_global_l1_unstructured_pruning(nn.Linear(2, 2), amount=1.1)


def test_global_pruning_supports_root_module() -> None:
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(8).reshape(2, 4).float())

    report = apply_global_l1_unstructured_pruning(model, amount=0.5)
    remove_pruning_reparameterization(model)
    final_report = summarize_pruning(model)

    assert report.entries[0].module_name == "<root>"
    assert final_report.sparsity == pytest.approx(0.5)


def test_tensor_sparsity_handles_empty_and_regular_tensors() -> None:
    assert tensor_sparsity(torch.empty(0)) == 0.0
    assert tensor_sparsity(torch.tensor([0.0, 1.0, 0.0])) == pytest.approx(2 / 3)


def test_prune_linear_out_and_in_features_preserves_selected_weights() -> None:
    linear = nn.Linear(4, 3)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(12).reshape(3, 4).float())
        linear.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))

    out_pruned = prune_linear_out_features(linear, [0, 2])
    in_pruned = prune_linear_in_features(linear, [1, 3])

    assert out_pruned.out_features == 2
    assert out_pruned.in_features == 4
    assert torch.equal(out_pruned.weight, linear.weight[[0, 2]])
    assert torch.equal(out_pruned.bias, linear.bias[[0, 2]])
    assert in_pruned.in_features == 2
    assert in_pruned.out_features == 3
    assert torch.equal(in_pruned.weight, linear.weight[:, [1, 3]])


def test_prune_conv2d_out_and_in_channels_preserves_selected_weights() -> None:
    conv = nn.Conv2d(3, 4, kernel_size=1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(12).reshape(4, 3, 1, 1).float())
        conv.bias.copy_(torch.arange(4).float())

    out_pruned = prune_conv2d_out_channels(conv, [1, 3])
    in_pruned = prune_conv2d_in_channels(conv, [0, 2])

    assert out_pruned.out_channels == 2
    assert out_pruned.in_channels == 3
    assert torch.equal(out_pruned.weight, conv.weight[[1, 3]])
    assert torch.equal(out_pruned.bias, conv.bias[[1, 3]])
    assert in_pruned.in_channels == 2
    assert in_pruned.out_channels == 4
    assert torch.equal(in_pruned.weight, conv.weight[:, [0, 2]])


def test_prune_batchnorm_channels_preserves_running_stats() -> None:
    batchnorm = nn.BatchNorm2d(4)
    with torch.no_grad():
        batchnorm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        batchnorm.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
        batchnorm.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        batchnorm.running_var.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    pruned = prune_batchnorm_channels(batchnorm, [0, 3])

    assert pruned.num_features == 2
    assert torch.equal(pruned.weight, batchnorm.weight[[0, 3]])
    assert torch.equal(pruned.bias, batchnorm.bias[[0, 3]])
    assert torch.equal(pruned.running_mean, batchnorm.running_mean[[0, 3]])
    assert torch.equal(pruned.running_var, batchnorm.running_var[[0, 3]])


def test_structured_pruning_helpers_reject_empty_or_duplicate_indices() -> None:
    linear = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="must not be empty"):
        prune_linear_out_features(linear, [])
    with pytest.raises(ValueError, match="must be unique"):
        prune_linear_out_features(linear, [0, 0])


class TinyPruneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def test_collect_module_importance_returns_weight_statistics() -> None:
    model = TinyPruneModel()
    with torch.no_grad():
        model.features[0].weight.fill_(0.5)
        model.features[2].weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0],
                ]
            )
        )

    records = collect_module_importance(model)

    assert [record.name for record in records] == ["features.0", "features.2"]
    by_name = {record.name: record for record in records}
    assert by_name["features.0"].parameter_count == 12
    assert by_name["features.0"].l1_mean == pytest.approx(0.5)
    assert by_name["features.0"].max_abs == pytest.approx(0.5)
    assert by_name["features.2"].parameter_count == 8
    assert by_name["features.2"].l1_mean == pytest.approx(2.5)
    assert by_name["features.2"].to_dict()["module_type"] == "Linear"


def test_rank_prune_candidates_supports_importance_only() -> None:
    model = TinyPruneModel()
    with torch.no_grad():
        model.features[0].weight.fill_(0.1)
        model.features[2].weight.fill_(2.0)

    candidates = rank_prune_candidates(model, [], top_k=1)

    assert len(candidates) == 1
    assert candidates[0].name == "features.0"
    assert candidates[0].rank == 1
    assert candidates[0].sensitivity_available is False
    assert candidates[0].combined_score == pytest.approx(0.0)


def test_rank_prune_candidates_combines_importance_and_sensitivity() -> None:
    class Diff:
        def __init__(self, mean_abs: float) -> None:
            self.mean_abs = mean_abs

    class Record:
        def __init__(self, name: str, mean_abs: float) -> None:
            self.name = name
            self.diff = Diff(mean_abs)

    candidate = TinyPruneModel()
    with torch.no_grad():
        candidate.features[0].weight.fill_(0.1)
        candidate.features[2].weight.fill_(2.0)
    sensitivity_records = [
        Record("features.0", 0.05),
        Record("features.2", 0.8),
    ]
    candidates = rank_prune_candidates(candidate, sensitivity_records)

    assert [record.name for record in candidates] == ["features.0", "features.2"]
    assert candidates[0].sensitivity_available is True
    assert candidates[0].combined_score <= candidates[1].combined_score
    assert candidates[1].sensitivity_score >= candidates[0].sensitivity_score


def test_rank_prune_candidates_can_require_sensitivity() -> None:
    model = TinyPruneModel()

    candidates = rank_prune_candidates(
        model,
        [],
        require_sensitivity=True,
    )

    assert candidates == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"importance_weight": -1.0}, "non-negative"),
        ({"sensitivity_weight": -1.0}, "non-negative"),
        ({"importance_weight": 0.0, "sensitivity_weight": 0.0}, "positive"),
        ({"top_k": 0}, "top_k must be positive"),
    ],
)
def test_rank_prune_candidates_rejects_invalid_parameters(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rank_prune_candidates(TinyPruneModel(), [], **kwargs)
