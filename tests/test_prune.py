from pathlib import Path

import pytest
import torch
from torch import nn

from xdl.model import VisionTransformer, resnet18, resnet50
from xqt.benchmark.latency import benchmark_callable
from xqt.prune import (
    collect_module_importance,
    apply_block_sparse_pruning,
    apply_global_l1_unstructured_pruning,
    apply_nm_structured_sparsity,
    apply_structured_pruning,
    apply_structured_pruning_plan,
    find_structured_pruning_targets,
    PruningTarget,
    StructuredPruningToyCNN,
    plan_structured_pruning,
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_out_features,
    rank_prune_candidates,
    remove_pruning_reparameterization,
    summarize_pruning,
    tensor_sparsity,
    validate_conv2d_keep_indices,
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


def test_prune_grouped_conv2d_in_channels_preserves_group_structure() -> None:
    conv = nn.Conv2d(4, 6, kernel_size=1, groups=2, bias=True)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel()).reshape_as(conv.weight).float())
        conv.bias.copy_(torch.arange(conv.bias.numel()).float())

    pruned = prune_conv2d_in_channels(conv, [0, 1, 2, 3])
    reduced = prune_conv2d_in_channels(conv, [0, 2])

    assert pruned.groups == 2
    assert pruned.in_channels == 4
    assert pruned.out_channels == 6
    assert reduced.groups == 2
    assert reduced.in_channels == 2
    assert reduced.out_channels == 6
    assert reduced.weight.shape == (6, 1, 1, 1)
    assert torch.equal(reduced.bias, conv.bias)


def test_validate_conv2d_keep_indices_rejects_uneven_group_pruning() -> None:
    conv = nn.Conv2d(4, 6, kernel_size=1, groups=2, bias=False)

    with pytest.raises(ValueError, match="same number of channels in every group"):
        validate_conv2d_keep_indices(conv, [0, 1, 2], axis="in")


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


def test_plan_structured_pruning_builds_actions_for_chain_cnn() -> None:
    model = StructuredPruningToyCNN(hidden_channels=8, out_channels=16, num_classes=4)

    plan = plan_structured_pruning(
        model,
        0.25,
        granularity="channel",
        scope="global",
        importance={"metric": "l1"},
        selection={"min_keep": 1},
    )

    assert plan.method == "structured"
    assert plan.granularity == "channel"
    assert plan.scope == "global"
    assert len(plan.targets) == 2
    assert len(plan.actions) == 2
    assert plan.total_units == 24
    assert plan.pruned_units == 6
    assert plan.sparsity == pytest.approx(0.25)
    assert isinstance(plan.targets[0], PruningTarget)
    assert plan.actions[0].module_name == "features.0"
    assert plan.actions[0].consumer_name == "features.3"
    assert plan.actions[1].consumer_name == "head"
    assert plan.actions[0].action_type == "conv_channel_group"
    assert plan.actions[0].adapter == "cnn_chain_adapter"
    assert plan.actions[0].structure_family == "cnn"
    assert plan.adapters == ["cnn_chain_adapter"]
    assert plan.structure_families == ["cnn"]
    assert plan.dependency_graph["groups"][0]["producer"] == "features.0"
    assert plan.actions[0].metadata["consumer_name"] == "features.3"


def test_apply_structured_pruning_plan_rewrites_conv_bn_and_linear_shapes() -> None:
    model = StructuredPruningToyCNN(hidden_channels=8, out_channels=16, num_classes=4)
    example_input = torch.randn(2, 3, 16, 16)
    baseline = model(example_input)
    before_parameters = sum(parameter.numel() for parameter in model.parameters())
    plan = plan_structured_pruning(
        model,
        0.25,
        granularity="channel",
        scope="global",
    )

    report = apply_structured_pruning_plan(
        model,
        plan,
        example_input=example_input,
    )
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 4)
    assert report.forward_checked is True
    assert report.parameter_count_before == before_parameters
    assert report.parameter_count_after < before_parameters
    assert report.parameter_reduction > 0
    assert report.parameter_reduction_ratio > 0.0
    assert report.export_status["attempted"] is False
    assert report.benchmark_status["attempted"] is False
    report_dict = report.to_dict()
    assert report_dict["export_status"]["artifacts"] == []
    assert report_dict["benchmark_status"]["latency"] is None
    assert model.features[0].out_channels == len(plan.actions[0].keep_indices)
    assert model.features[1].num_features == len(plan.actions[0].keep_indices)
    assert model.features[3].in_channels == len(plan.actions[0].keep_indices)
    assert model.features[3].out_channels == len(plan.actions[1].keep_indices)
    assert model.features[4].num_features == len(plan.actions[1].keep_indices)
    assert model.head.in_features == len(plan.actions[1].keep_indices)


class TinyGroupedConvCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=1, groups=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class TinyConcatBranchCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.branch1 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.fuse = nn.Conv2d(10, 8, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class TinyMBConvBlock(nn.Module):
    def __init__(self, in_channels: int = 4, mid_channels: int = 8, out_channels: int = 4) -> None:
        super().__init__()
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_act = nn.ReLU()
        self.depthwise_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_act = nn.ReLU()
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand_act(self.expand_bn(self.expand_conv(x)))
        x = self.depthwise_act(self.depthwise_bn(self.depthwise_conv(x)))
        x = self.project_bn(self.project_conv(x))
        return x


class TinyMBConvCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.block = TinyMBConvBlock(in_channels=4, mid_channels=8, out_channels=4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class TinySameWidthCNNStage(nn.Module):
    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TinyMultiStageCNN(nn.Module):
    def __init__(self, channels: int = 8, depth: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.stages = nn.ModuleList(
            [TinySameWidthCNNStage(channels=channels) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class TinyMoEBlock(nn.Module):
    def __init__(self, embed_dim: int = 8, num_experts: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.expert_usage = torch.tensor([0.60, 0.10, 0.25, 0.05])
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        router_probs = torch.softmax(self.router(pooled), dim=-1)
        expert_outputs = torch.stack(
            [expert(pooled) for expert in self.experts],
            dim=1,
        )
        mixed = torch.sum(router_probs.unsqueeze(-1) * expert_outputs, dim=1)
        return self.head(mixed)


def test_plan_and_apply_structured_pruning_auto_aligns_grouped_conv_channels() -> None:
    model = TinyGroupedConvCNN()
    model.eval()
    example_input = torch.randn(2, 3, 8, 8)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="channel",
        scope="per_layer",
        importance={"metric": "l1"},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before
    assert plan.actions[0].module_name == "features.0"
    assert len(plan.actions[0].keep_indices) == 2
    assert sorted(index // 2 for index in plan.actions[0].keep_indices) == [0, 1]
    assert model.features[0].out_channels == 2
    assert model.features[1].num_features == 2
    assert model.features[3].in_channels == 2
    assert model.features[3].groups == 2
    assert model.features[3].weight.shape[1] == 1
    assert model.features[3].out_channels == 4
    assert model.features[4].num_features == 4
    assert model.head.in_features == 4


def test_find_structured_pruning_targets_supports_concat_branch_channels() -> None:
    model = TinyConcatBranchCNN()

    targets = find_structured_pruning_targets(
        model,
        granularity="channel",
        importance={"metric": "l1"},
    )

    target_names = [target.module_name for target in targets]
    assert "branch1.0" in target_names
    assert "branch2.0" in target_names
    concat_targets = [target for target in targets if target.metadata.get("merge") == "concat"]
    assert len(concat_targets) == 2
    assert concat_targets[0].adapter == "concat_branch_adapter"
    assert concat_targets[0].structure_family == "cnn_branch"
    assert concat_targets[0].metadata["consumer_name"] == "fuse"


def test_plan_and_apply_structured_pruning_supports_concat_branch_channels() -> None:
    model = TinyConcatBranchCNN()
    example_input = torch.randn(2, 3, 8, 8)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="channel",
        selection={
            "keep_indices": {
                "branch1.0": [0, 2],
                "branch2.0": [1, 3, 5],
            }
        },
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert report.forward_checked is True
    assert model.branch1[0].out_channels == 2
    assert model.branch1[1].num_features == 2
    assert model.branch2[0].out_channels == 3
    assert model.branch2[1].num_features == 3
    assert model.fuse.in_channels == 5
    assert model.fuse.out_channels == 8
    assert report.parameter_count_after < report.parameter_count_before


def test_find_structured_pruning_targets_supports_mbconv_mid_channels() -> None:
    model = TinyMBConvCNN()

    targets = find_structured_pruning_targets(
        model,
        granularity="channel",
        importance={"metric": "l1"},
    )

    mbconv_targets = [target for target in targets if target.adapter == "mbconv_adapter"]
    assert len(mbconv_targets) == 1
    assert mbconv_targets[0].module_name == "block"
    assert mbconv_targets[0].structure_family == "cnn_mbconv"
    assert mbconv_targets[0].group_size == 8
    assert mbconv_targets[0].metadata["project_conv_name"] == "block.project_conv"


def test_plan_and_apply_structured_pruning_supports_mbconv_mid_channels() -> None:
    model = TinyMBConvCNN()
    example_input = torch.randn(2, 3, 8, 8)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="channel",
        selection={"keep_indices": {"block": [0, 2, 4, 6]}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert report.forward_checked is True
    assert model.block.expand_conv.out_channels == 4
    assert model.block.expand_bn.num_features == 4
    assert model.block.depthwise_conv.in_channels == 4
    assert model.block.depthwise_conv.out_channels == 4
    assert model.block.depthwise_conv.groups == 4
    assert model.block.depthwise_bn.num_features == 4
    assert model.block.project_conv.in_channels == 4
    assert model.block.project_conv.out_channels == 4
    assert report.parameter_count_after < report.parameter_count_before


def test_find_structured_pruning_targets_supports_cnn_stage_pruning() -> None:
    model = TinyMultiStageCNN()

    targets = find_structured_pruning_targets(
        model,
        granularity="stage",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "stages"
    assert targets[0].adapter == "cnn_stage_adapter"
    assert targets[0].structure_family == "cnn_stage"
    assert targets[0].group_size == 4
    assert targets[0].metadata["shape_compatible"] is True
    assert targets[0].metadata["stage_names"] == [
        "stages.0",
        "stages.1",
        "stages.2",
        "stages.3",
    ]


def test_plan_and_apply_structured_pruning_supports_cnn_stage_pruning() -> None:
    model = TinyMultiStageCNN()
    example_input = torch.randn(2, 3, 8, 8)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="stage",
        selection={"keep_indices": {"stages": [0, 2]}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)
    benchmark = benchmark_callable(
        lambda: model(example_input),
        warmup=1,
        iterations=2,
        sync_cuda=False,
        device="cpu",
    )

    assert baseline.shape == candidate.shape == (2, 3)
    assert len(model.stages) == 2
    assert plan.actions[0].action_type == "drop_stages"
    assert plan.actions[0].adapter == "cnn_stage_adapter"
    assert plan.actions[0].structure_family == "cnn_stage"
    assert plan.actions[0].keep_indices == [0, 2]
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before
    assert benchmark.iterations == 2
    assert len(benchmark.samples_ms) == 2


def test_apply_structured_pruning_runs_end_to_end() -> None:
    model = StructuredPruningToyCNN(hidden_channels=8, out_channels=16, num_classes=4)
    example_input = torch.randn(1, 3, 16, 16)

    report = apply_structured_pruning(
        model,
        0.25,
        granularity="filter",
        scope="global",
        importance={"metric": "l2"},
        selection={"min_keep": 1},
        example_input=example_input,
    )

    assert report.method == "structured"
    assert report.granularity == "filter"
    assert report.importance_metric == "l2"
    assert report.parameter_count_after < report.parameter_count_before
    assert report.sparsity == pytest.approx(0.25)


def test_find_structured_pruning_targets_returns_conv_targets() -> None:
    model = StructuredPruningToyCNN(hidden_channels=8, out_channels=16, num_classes=4)

    targets = find_structured_pruning_targets(
        model,
        granularity="channel",
        importance={"metric": "bn_gamma"},
    )

    assert [target.module_name for target in targets] == ["features.0", "features.3"]
    assert targets[0].module_type == "Conv2d"
    assert targets[0].dependency_group == "features.0"
    assert targets[0].metadata["normalization_name"] == "features.1"


def test_plan_structured_pruning_rejects_model_without_supported_candidate_chain() -> None:
    class ResidualLikeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(self.conv(x)) + x[:, :4]

    with pytest.raises(ValueError, match="No supported channel structured pruning candidates"):
        plan_structured_pruning(ResidualLikeModel(), 0.25)


def test_find_structured_pruning_targets_returns_residual_stage_targets_for_resnet18() -> None:
    model = resnet18(num_classes=7)

    targets = find_structured_pruning_targets(
        model,
        granularity="channel",
        importance={"metric": "l1"},
    )

    assert [target.module_name for target in targets] == ["layer2", "layer3", "layer4"]
    assert targets[0].adapter == "residual_cnn_adapter"
    assert targets[0].structure_family == "cnn_residual"
    assert targets[0].dependency_group == "layer2"
    assert targets[0].metadata["block_type"] == "BasicBlock"
    assert targets[0].metadata["merge"] == "add"
    assert targets[0].metadata["consumer_name"] == "layer3.0.conv1"


def test_apply_structured_pruning_supports_resnet18_residual_stage_pruning() -> None:
    model = resnet18(num_classes=7)
    model.eval()
    example_input = torch.randn(1, 3, 64, 64)
    baseline = model(example_input)
    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="channel",
        selection={"keep_indices": {"layer2": list(range(64))}},
    )

    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (1, 7)
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before
    assert report.adapters == ["residual_cnn_adapter"]
    assert report.structure_families == ["cnn_residual"]
    assert plan.actions[0].module_name == "layer2"
    assert plan.actions[0].pruned_units == 64
    assert plan.actions[1].module_name == "layer3"
    assert plan.actions[1].pruned_units == 0
    assert model.layer2[0].conv1.out_channels == 64
    assert model.layer2[0].bn1.num_features == 64
    assert model.layer2[0].conv2.in_channels == 64
    assert model.layer2[0].conv2.out_channels == 64
    assert model.layer2[0].bn2.num_features == 64
    assert model.layer2[0].downsample[0].out_channels == 64
    assert model.layer2[0].downsample[1].num_features == 64
    assert model.layer2[1].conv1.in_channels == 64
    assert model.layer2[1].conv1.out_channels == 64
    assert model.layer3[0].conv1.in_channels == 64
    assert model.layer3[0].downsample[0].in_channels == 64


def test_resnet18_residual_stage_pruning_exports_to_onnx(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    model = resnet18(num_classes=7)
    model.eval()
    example_input = torch.randn(1, 3, 64, 64)
    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="channel",
        selection={"keep_indices": {"layer2": list(range(64))}},
    )
    apply_structured_pruning_plan(model, plan, example_input=example_input)
    export_path = tmp_path / "resnet18_residual_pruned.onnx"

    torch.onnx.export(
        model,
        example_input,
        str(export_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
    )

    assert export_path.exists()
    assert export_path.stat().st_size > 0


def test_apply_structured_pruning_supports_resnet50_bottleneck_stage_output_pruning() -> None:
    model = resnet50(num_classes=7)
    model.eval()
    example_input = torch.randn(1, 3, 64, 64)
    baseline = model(example_input)
    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="channel",
        selection={"keep_indices": {"layer2": list(range(256))}},
    )

    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (1, 7)
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before
    assert plan.actions[1].module_name == "layer2"
    assert plan.actions[1].pruned_units == 256
    assert model.layer2[0].conv1.in_channels == 256
    assert model.layer2[0].conv1.out_channels == 128
    assert model.layer2[0].conv3.in_channels == 128
    assert model.layer2[0].conv3.out_channels == 256
    assert model.layer2[0].bn3.num_features == 256
    assert model.layer2[0].downsample[0].out_channels == 256
    assert model.layer2[0].downsample[1].num_features == 256
    assert model.layer2[1].conv1.in_channels == 256
    assert model.layer2[1].conv1.out_channels == 128
    assert model.layer3[0].conv1.in_channels == 256
    assert model.layer3[0].downsample[0].in_channels == 256


def _tiny_vit() -> VisionTransformer:
    return VisionTransformer(
        img_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=5,
        embed_dim=32,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
    )


class TinySplitProjectionAttention(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.embed_dim)
        return self.out_proj(x)


class TinySplitAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = TinySplitProjectionAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm(x))
        return self.head(x.mean(dim=1))


class TinyCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.attention_role = "cross"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        context_batch_size, context_token_count, _ = context.shape
        assert context_batch_size == batch_size
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        k = self.k_proj(context).reshape(
            context_batch_size,
            context_token_count,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(context).reshape(
            context_batch_size,
            context_token_count,
            self.num_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.embed_dim)
        return self.proj_dropout(self.out_proj(x))


class TinyCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = TinyCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.query_norm(x), self.context_norm(context))
        return self.head(x.mean(dim=1))


class TinyGroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, self.embed_dim)
        return self.proj_dropout(self.out_proj(x))


class TinyGroupedQueryAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, num_kv_heads: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = TinyGroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm(x))
        return self.head(x.mean(dim=1))


class TinyMultiQueryAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = TinyGroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=1,
        )
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.norm(x))
        return self.head(x.mean(dim=1))


class TinyGatedMLPBlock(nn.Module):
    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = TinyGatedMLP(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return self.head(x.mean(dim=1))


class TinyGatedResidualBlock(nn.Module):
    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = TinyGatedMLP(embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


class TinyGatedMLP(nn.Module):
    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.up_proj = nn.Linear(embed_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyResidualMLPBlock(nn.Module):
    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc2(self.act(self.fc1(self.norm(x))))
        return residual + x


class TinyHeterogeneousBlockStack(nn.Module):
    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TinyResidualMLPBlock(embed_dim=embed_dim),
                TinyGatedResidualBlock(embed_dim=embed_dim, hidden_dim=32),
                TinyResidualMLPBlock(embed_dim=embed_dim),
            ]
        )
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


class TinyDiTBlock(nn.Module):
    def __init__(self, embed_dim: int = 16, num_heads: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.time_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = TinyCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TinyGatedMLP(embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.time_proj(time_emb).unsqueeze(1)
        x = x + self.attn(self.norm1(x), context)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyDiTModel(nn.Module):
    def __init__(self, embed_dim: int = 16, depth: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.time_embed = nn.Embedding(32, embed_dim)
        self.blocks = nn.ModuleList(
            [
                TinyDiTBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=embed_dim * 2)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        time_emb = self.time_embed(timesteps)
        for block in self.blocks:
            x = block(x, time_emb, context)
        return self.out_proj(self.norm(x))


def test_find_structured_pruning_targets_returns_mlp_targets_for_vit() -> None:
    model = _tiny_vit()

    targets = find_structured_pruning_targets(
        model,
        granularity="mlp_neuron",
        importance={"metric": "l1"},
    )

    assert len(targets) == 4
    assert targets[0].module_name == "blocks.0.mlp.fc1"
    assert targets[0].module_type == "Linear"
    assert targets[0].granularity == "mlp_neuron"
    assert targets[0].adapter == "mlp_pair_adapter"
    assert targets[0].structure_family == "transformer"
    assert targets[0].metadata["partner_name"] == "blocks.0.mlp.fc2"


def test_plan_and_apply_structured_pruning_supports_mlp_neuron_for_vit() -> None:
    model = _tiny_vit()
    example_input = torch.randn(2, 3, 32, 32)
    baseline = model(example_input)
    plan = plan_structured_pruning(
        model,
        0.25,
        granularity="mlp_neuron",
        scope="per_layer",
        importance={"metric": "l2"},
    )

    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 5)
    assert report.parameter_count_after < report.parameter_count_before
    assert len(plan.targets) == 4
    assert all(action.action_type == "mlp_neuron_group" for action in plan.actions)
    assert model.blocks[0].mlp.fc1.out_features == 48
    assert model.blocks[0].mlp.fc2.in_features == 48
    assert report.sparsity == pytest.approx(0.25)


def test_find_structured_pruning_targets_supports_gated_mlp() -> None:
    model = TinyGatedMLPBlock()

    targets = find_structured_pruning_targets(
        model,
        granularity="mlp_neuron",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "mlp.up_proj"
    assert targets[0].adapter == "mlp_pair_adapter"
    assert targets[0].structure_family == "transformer"
    assert targets[0].metadata["mlp_kind"] == "gated"
    assert targets[0].metadata["gate_proj_name"] == "mlp.gate_proj"
    assert targets[0].metadata["down_proj_name"] == "mlp.down_proj"


def test_plan_and_apply_structured_pruning_supports_gated_mlp_neurons() -> None:
    model = TinyGatedMLPBlock()
    example_input = torch.randn(2, 5, 16)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="mlp_neuron",
        scope="per_layer",
        importance={"metric": "l2"},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert len(plan.targets) == 1
    assert plan.actions[0].action_type == "gated_mlp_neuron_group"
    assert plan.actions[0].metadata["mlp_kind"] == "gated"
    assert model.mlp.gate_proj.out_features == 16
    assert model.mlp.up_proj.out_features == 16
    assert model.mlp.down_proj.in_features == 16
    assert model.mlp.down_proj.out_features == 16
    assert report.parameter_count_after < report.parameter_count_before
    assert report.sparsity == pytest.approx(0.5)


def test_find_structured_pruning_targets_returns_head_targets_for_vit() -> None:
    model = _tiny_vit()

    targets = find_structured_pruning_targets(
        model,
        granularity="head",
        importance={"metric": "l1"},
    )

    assert len(targets) == 4
    assert targets[0].module_name == "blocks.0.attn"
    assert targets[0].group_size == 4
    assert targets[0].adapter == "attention_adapter"
    assert targets[0].structure_family == "transformer"
    assert targets[0].metadata["attention_kind"] == "fused_qkv"
    assert targets[0].metadata["num_heads"] == 4
    assert targets[0].metadata["head_dim"] == 8


def test_plan_and_apply_structured_pruning_supports_head_pruning_for_vit() -> None:
    model = _tiny_vit()
    example_input = torch.randn(2, 3, 32, 32)
    baseline = model(example_input)
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="head",
        scope="per_layer",
        importance={"metric": "l2"},
    )

    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 5)
    assert len(plan.targets) == 4
    assert all(action.action_type == "attention_heads" for action in plan.actions)
    assert model.blocks[0].attn.num_heads == 2
    assert model.blocks[0].attn.embed_dim == 32
    assert model.blocks[0].attn.qkv.out_features == 48
    assert model.blocks[0].attn.proj.in_features == 16
    assert model.blocks[0].attn.proj.out_features == 32
    assert report.parameter_count_after < report.parameter_count_before
    assert report.adapters == ["attention_adapter"]
    assert report.structure_families == ["transformer"]
    assert report.sparsity == pytest.approx(0.5)


def test_find_structured_pruning_targets_supports_split_projection_attention() -> None:
    model = TinySplitAttentionBlock()

    targets = find_structured_pruning_targets(
        model,
        granularity="head",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "attn"
    assert targets[0].group_size == 4
    assert targets[0].adapter == "attention_adapter"
    assert targets[0].metadata["attention_kind"] == "split_qkv"
    assert targets[0].metadata["attention_role"] == "self"
    assert targets[0].metadata["q_proj_name"] == "attn.q_proj"
    assert targets[0].metadata["out_proj_name"] == "attn.out_proj"


def test_plan_and_apply_structured_pruning_supports_split_projection_head_pruning() -> None:
    model = TinySplitAttentionBlock()
    example_input = torch.randn(2, 5, 32)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="head",
        scope="per_layer",
        importance={"metric": "l2"},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert len(plan.targets) == 1
    assert plan.actions[0].metadata["attention_kind"] == "split_qkv"
    assert model.attn.num_heads == 2
    assert model.attn.embed_dim == 32
    assert model.attn.q_proj.out_features == 16
    assert model.attn.k_proj.out_features == 16
    assert model.attn.v_proj.out_features == 16
    assert model.attn.out_proj.in_features == 16
    assert model.attn.out_proj.out_features == 32
    assert report.parameter_count_after < report.parameter_count_before
    assert report.adapters == ["attention_adapter"]
    assert report.sparsity == pytest.approx(0.5)


def test_find_structured_pruning_targets_supports_cross_attention() -> None:
    model = TinyCrossAttentionBlock()

    targets = find_structured_pruning_targets(
        model,
        granularity="head",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "attn"
    assert targets[0].metadata["attention_kind"] == "split_qkv"
    assert targets[0].metadata["attention_role"] == "cross"
    assert targets[0].metadata["q_proj_name"] == "attn.q_proj"
    assert targets[0].metadata["k_proj_name"] == "attn.k_proj"
    assert targets[0].metadata["v_proj_name"] == "attn.v_proj"
    assert targets[0].metadata["out_proj_name"] == "attn.out_proj"


def test_plan_and_apply_structured_pruning_supports_cross_attention_head_pruning() -> None:
    model = TinyCrossAttentionBlock()
    example_input = (
        torch.randn(2, 5, 32),
        torch.randn(2, 7, 32),
    )
    baseline = model(*example_input)

    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="head",
        scope="per_layer",
        importance={"metric": "l2"},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(*example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert len(plan.targets) == 1
    assert plan.actions[0].metadata["attention_kind"] == "split_qkv"
    assert plan.actions[0].metadata["attention_role"] == "cross"
    assert model.attn.attention_role == "cross"
    assert model.attn.num_heads == 2
    assert model.attn.embed_dim == 32
    assert model.attn.q_proj.out_features == 16
    assert model.attn.k_proj.out_features == 16
    assert model.attn.v_proj.out_features == 16
    assert model.attn.out_proj.in_features == 16
    assert model.attn.out_proj.out_features == 32
    assert report.parameter_count_after < report.parameter_count_before
    assert report.forward_checked is True
    assert report.adapters == ["attention_adapter"]
    assert report.sparsity == pytest.approx(0.5)


def test_find_structured_pruning_targets_supports_grouped_query_attention() -> None:
    model = TinyGroupedQueryAttentionBlock()

    targets = find_structured_pruning_targets(
        model,
        granularity="head",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "attn"
    assert targets[0].group_size == 2
    assert targets[0].metadata["attention_kind"] == "split_qkv"
    assert targets[0].metadata["attention_variant"] == "gqa"
    assert targets[0].metadata["num_heads"] == 4
    assert targets[0].metadata["num_kv_heads"] == 2


def test_plan_and_apply_structured_pruning_supports_grouped_query_attention() -> None:
    model = TinyGroupedQueryAttentionBlock()
    example_input = torch.randn(2, 5, 32)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="head",
        selection={"keep_indices": {"attn": [0]}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert plan.targets[0].group_size == 2
    assert plan.actions[0].metadata["attention_variant"] == "gqa"
    assert model.attn.num_heads == 2
    assert model.attn.num_kv_heads == 1
    assert model.attn.q_proj.out_features == 16
    assert model.attn.k_proj.out_features == 8
    assert model.attn.v_proj.out_features == 8
    assert model.attn.out_proj.in_features == 16
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before


def test_plan_and_apply_structured_pruning_supports_multi_query_attention() -> None:
    model = TinyMultiQueryAttentionBlock()
    example_input = torch.randn(2, 5, 32)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="head",
        selection={"keep_indices": {"attn": [0]}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert plan.targets[0].group_size == 1
    assert plan.actions[0].metadata["attention_variant"] == "mqa"
    assert model.attn.num_heads == 4
    assert model.attn.num_kv_heads == 1
    assert model.attn.q_proj.out_features == 32
    assert model.attn.k_proj.out_features == 8
    assert model.attn.v_proj.out_features == 8
    assert model.attn.out_proj.in_features == 32
    assert report.forward_checked is True


def test_plan_structured_pruning_rejects_out_of_range_kv_keep_indices_for_gqa() -> None:
    model = TinyGroupedQueryAttentionBlock()

    with pytest.raises(ValueError, match="must reference kv heads"):
        plan_structured_pruning(
            model,
            0.0,
            granularity="head",
            selection={"keep_indices": {"attn": [2]}},
        )


def test_plan_and_apply_structured_pruning_supports_block_pruning_for_vit() -> None:
    model = _tiny_vit()
    example_input = torch.randn(1, 3, 32, 32)
    baseline = model(example_input)
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="block",
        scope="per_layer",
        importance={"metric": "l1"},
        selection={"min_keep": 1},
    )

    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (1, 5)
    assert len(plan.targets) == 1
    assert plan.targets[0].module_name == "blocks"
    assert plan.actions[0].action_type == "drop_blocks"
    assert len(model.blocks) == 2
    assert report.sparsity == pytest.approx(0.5)
    assert report.parameter_count_after < report.parameter_count_before


def test_block_pruning_reduces_measured_latency_for_tiny_vit() -> None:
    model = VisionTransformer(
        img_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=5,
        embed_dim=64,
        depth=8,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
    )
    example_input = torch.randn(2, 3, 32, 32)
    baseline = benchmark_callable(
        lambda: model(example_input),
        warmup=1,
        iterations=5,
        sync_cuda=False,
        device="cpu",
    )
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="block",
        scope="per_layer",
        importance={"metric": "l1"},
        selection={"min_keep": 1},
    )

    apply_structured_pruning_plan(model, plan, example_input=example_input)
    pruned = benchmark_callable(
        lambda: model(example_input),
        warmup=1,
        iterations=5,
        sync_cuda=False,
        device="cpu",
    )

    assert len(model.blocks) == 4
    assert pruned.mean_ms < baseline.mean_ms


def test_plan_structured_pruning_supports_explicit_keep_indices() -> None:
    model = _tiny_vit()

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="block",
        selection={"keep_indices": {"blocks": [0, 3]}},
    )

    assert plan.actions[0].keep_indices == [0, 3]
    assert plan.actions[0].prune_indices == [1, 2]


def test_find_structured_pruning_targets_supports_heterogeneous_block_container() -> None:
    model = TinyHeterogeneousBlockStack()

    targets = find_structured_pruning_targets(
        model,
        granularity="block",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "blocks"
    assert targets[0].adapter == "container_adapter"
    assert targets[0].metadata["container_type"] == "ModuleList"
    assert targets[0].metadata["heterogeneous_children"] is True
    assert targets[0].metadata["block_types"] == [
        "TinyResidualMLPBlock",
        "TinyGatedResidualBlock",
        "TinyResidualMLPBlock",
    ]


def test_plan_and_apply_structured_pruning_supports_heterogeneous_block_container() -> None:
    model = TinyHeterogeneousBlockStack()
    example_input = torch.randn(2, 4, 16)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="block",
        selection={"keep_indices": {"blocks": [0, 2]}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 2)
    assert len(plan.targets) == 1
    assert plan.targets[0].metadata["heterogeneous_children"] is True
    assert plan.actions[0].keep_indices == [0, 2]
    assert len(model.blocks) == 2
    assert isinstance(model.blocks[0], TinyResidualMLPBlock)
    assert isinstance(model.blocks[1], TinyResidualMLPBlock)
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before


def test_plan_and_apply_structured_pruning_supports_toy_dit_block_pruning_smoke() -> None:
    model = TinyDiTModel()
    example_input = (
        torch.randn(2, 4, 16),
        torch.randint(0, 32, (2,), dtype=torch.long),
        torch.randn(2, 6, 16),
    )
    baseline = model(*example_input)

    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="block",
        scope="per_layer",
        importance={"metric": "l1"},
        selection={"min_keep": 1},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(*example_input)

    assert baseline.shape == candidate.shape == (2, 4, 16)
    assert len(plan.targets) == 1
    assert plan.targets[0].module_name == "blocks"
    assert plan.targets[0].metadata["container_type"] == "ModuleList"
    assert len(model.blocks) == 2
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before


def test_find_structured_pruning_targets_supports_vit_hidden_width() -> None:
    model = _tiny_vit()

    targets = find_structured_pruning_targets(
        model,
        granularity="hidden_width",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "<root>"
    assert targets[0].adapter == "vit_hidden_width_adapter"
    assert targets[0].structure_family == "transformer_width"
    assert targets[0].group_size == 32
    assert targets[0].metadata["model_family"] == "xdl_vit"
    assert targets[0].metadata["embed_dim"] == 32
    assert targets[0].metadata["num_heads"] == 4
    assert targets[0].metadata["width_kind"] == "hidden"


def test_plan_and_apply_structured_pruning_supports_vit_hidden_width() -> None:
    model = _tiny_vit()
    example_input = torch.randn(2, 3, 32, 32)
    baseline = model(example_input)
    keep_indices = list(range(0, 32, 2))

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="hidden_width",
        selection={"keep_indices": {"<root>": keep_indices}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 5)
    assert plan.actions[0].action_type == "vit_hidden_width"
    assert plan.actions[0].keep_indices == keep_indices
    assert plan.actions[0].prune_indices == list(range(1, 32, 2))
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before
    assert model.embed_dim == 16
    assert model.patch_embed.proj.out_channels == 16
    assert model.cls_token.shape == (1, 1, 16)
    assert model.pos_embed.shape[-1] == 16
    assert model.norm.normalized_shape == (16,)
    assert model.head.in_features == 16
    assert model.blocks[0].norm1.normalized_shape == (16,)
    assert model.blocks[0].attn.embed_dim == 16
    assert model.blocks[0].attn.head_dim == 4
    assert model.blocks[0].attn.qkv.in_features == 16
    assert model.blocks[0].attn.qkv.out_features == 48
    assert model.blocks[0].attn.proj.in_features == 16
    assert model.blocks[0].attn.proj.out_features == 16
    assert model.blocks[0].mlp.fc1.in_features == 16
    assert model.blocks[0].mlp.fc2.out_features == 16


def test_structured_vit_hidden_width_pruned_model_can_continue_training() -> None:
    model = _tiny_vit()
    example_input = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="hidden_width",
        scope="per_layer",
        selection={"min_keep": 4},
    )
    apply_structured_pruning_plan(model, plan, example_input=example_input)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    logits = model(example_input)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()

    assert logits.shape == (2, 5)
    assert torch.isfinite(loss)
    assert model.embed_dim == 16


def test_find_structured_pruning_targets_supports_vit_embedding_width_alias() -> None:
    model = _tiny_vit()

    targets = find_structured_pruning_targets(
        model,
        granularity="embedding_width",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    assert targets[0].granularity == "embedding_width"
    assert targets[0].adapter == "vit_embedding_width_adapter"
    assert targets[0].metadata["width_kind"] == "embedding"


def test_find_structured_pruning_targets_supports_usage_aware_experts() -> None:
    model = TinyMoEBlock()

    targets = find_structured_pruning_targets(
        model,
        granularity="expert",
        importance={"metric": "usage"},
    )

    assert len(targets) == 1
    assert targets[0].module_name == "<root>"
    assert targets[0].adapter == "moe_expert_adapter"
    assert targets[0].structure_family == "moe_expert"
    assert targets[0].group_size == 4
    assert targets[0].metadata["router_name"] == "router"
    assert targets[0].metadata["experts_name"] == "experts"
    assert targets[0].metadata["usage_source"] == "module_usage"
    assert targets[0].metadata["usage_scores"] == [0.6000000238418579, 0.10000000149011612, 0.25, 0.05000000074505806]


def test_plan_and_apply_structured_pruning_supports_usage_aware_experts() -> None:
    model = TinyMoEBlock()
    example_input = torch.randn(2, 4, 8)
    baseline = model(example_input)

    plan = plan_structured_pruning(
        model,
        0.0,
        granularity="expert",
        importance={"metric": "usage"},
        selection={"keep_indices": {"<root>": [0, 2]}},
    )
    report = apply_structured_pruning_plan(model, plan, example_input=example_input)
    candidate = model(example_input)

    assert baseline.shape == candidate.shape == (2, 3)
    assert plan.actions[0].action_type == "drop_experts"
    assert plan.actions[0].keep_indices == [0, 2]
    assert plan.actions[0].prune_indices == [1, 3]
    assert plan.topology_changes[0]["kind"] == "expert"
    assert plan.topology_changes[0]["kept_indices"] == [0, 2]
    assert plan.topology_changes[0]["pruned_indices"] == [1, 3]
    assert plan.topology_changes[0]["pruned_usage"] == pytest.approx([0.1, 0.05])
    assert model.router.out_features == 2
    assert len(model.experts) == 2
    assert model.num_experts == 2
    assert torch.equal(model.expert_usage, torch.tensor([0.60, 0.25]))
    assert report.forward_checked is True
    assert report.parameter_count_after < report.parameter_count_before
    assert report.topology_changes[0]["pruned_indices"] == [1, 3]


def test_plan_structured_pruning_rejects_usage_metric_for_non_experts() -> None:
    model = _tiny_vit()

    with pytest.raises(ValueError, match="usage only supports expert"):
        plan_structured_pruning(
            model,
            0.5,
            granularity="block",
            importance={"metric": "usage"},
        )


def test_apply_nm_structured_sparsity_enforces_two_of_four_pattern() -> None:
    model = nn.Sequential(
        nn.Linear(8, 4, bias=False),
        nn.ReLU(),
        nn.Linear(4, 2, bias=False),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.arange(1, parameter.numel() + 1).reshape_as(parameter).float())

    report = apply_nm_structured_sparsity(
        model,
        pattern_n=2,
        pattern_m=4,
        module_types=(nn.Linear,),
    )

    assert report.method == "nm_structured"
    assert report.granularity == "nm"
    assert report.pattern_n == 2
    assert report.pattern_m == 4
    assert report.sparsity == pytest.approx(0.5)
    assert report.compliance_ratio == pytest.approx(1.0)
    assert len(report.layers) == 2
    assert report.layers[0].compliance_ratio == pytest.approx(1.0)


def test_apply_nm_structured_sparsity_rejects_invalid_pattern() -> None:
    model = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="pattern_n and pattern_m must be positive"):
        apply_nm_structured_sparsity(model, pattern_n=0, pattern_m=4)
    with pytest.raises(ValueError, match="pattern_n must be smaller than pattern_m"):
        apply_nm_structured_sparsity(model, pattern_n=4, pattern_m=4)


def test_apply_block_sparse_pruning_zeroes_low_score_blocks() -> None:
    model = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 5.0, 5.0],
                    [1.0, 1.0, 5.0, 5.0],
                    [4.0, 4.0, 2.0, 2.0],
                    [4.0, 4.0, 2.0, 2.0],
                ]
            )
        )

    report = apply_block_sparse_pruning(
        model,
        target_sparsity=0.5,
        block_shape=(2, 2),
        module_types=(nn.Linear,),
    )

    assert report.method == "block_sparse"
    assert report.granularity == "block_sparse"
    assert report.block_shape == (2, 2)
    assert report.sparsity == pytest.approx(0.5)
    assert report.parameter_sparsity == pytest.approx(0.5)
    assert report.layers[0].total_blocks == 4
    assert report.layers[0].zero_blocks == 2
    assert report.layers[0].pruned_blocks == 2
    assert torch.count_nonzero(model.weight[:2, :2]) == 0
    assert torch.count_nonzero(model.weight[2:, 2:]) == 0
    assert torch.count_nonzero(model.weight[:2, 2:]) == 4
    assert torch.count_nonzero(model.weight[2:, :2]) == 4
    assert report.to_dict()["layers"][0]["block_shape"] == [2, 2]


def test_apply_block_sparse_pruning_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="target_sparsity must be"):
        apply_block_sparse_pruning(nn.Linear(4, 4), target_sparsity=1.5)
    with pytest.raises(ValueError, match="block_shape values must be positive"):
        apply_block_sparse_pruning(nn.Linear(4, 4), target_sparsity=0.5, block_shape=(0, 2))


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
