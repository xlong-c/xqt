import torch
from torch import nn

from xdl.model.vit import VisionTransformer
from xqt.prune import (
    apply_structured_pruning,
    apply_structured_pruning_plan,
    find_structured_pruning_targets,
    plan_structured_pruning,
)


class _ToyGroupedQueryAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_role = "self"
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 8
        self.embed_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.q_proj(x))


class _ToyAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _ToyGroupedQueryAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class _ToyPlainMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _ToyGatedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 12)
        self.up_proj = nn.Linear(8, 12)
        self.down_proj = nn.Linear(12, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class _ToyMLPCandidateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.plain = _ToyPlainMLP()
        self.gated = _ToyGatedMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gated(self.plain(x))


class _ToyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(8, 3, bias=False)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.expert_usage = [8.0, 3.0, 1.0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.experts[0](x)


class _ToyMoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = _ToyMoE()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)


class _ToySpatialStage(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )


class _ToyContainerCandidateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.Sequential(_ToySpatialStage(), _ToySpatialStage())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)


def test_find_structured_pruning_targets_detects_grouped_query_attention_heads() -> None:
    targets = find_structured_pruning_targets(
        _ToyAttentionModel(),
        granularity="head",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.module_name == "attn"
    assert target.dependency_group == "attn"
    assert target.adapter == "attention_adapter"
    assert target.structure_family == "transformer"
    assert target.group_size == 2
    assert target.metadata["attention_kind"] == "split_qkv"
    assert target.metadata["attention_role"] == "self"
    assert target.metadata["attention_variant"] == "gqa"
    assert target.metadata["num_heads"] == 4
    assert target.metadata["num_kv_heads"] == 2
    assert target.metadata["q_proj_name"] == "attn.q_proj"
    assert target.metadata["out_proj_name"] == "attn.out_proj"


def test_find_structured_pruning_targets_detects_plain_and_gated_mlp_neurons() -> None:
    targets = find_structured_pruning_targets(
        _ToyMLPCandidateModel(),
        granularity="mlp_neuron",
        importance={"metric": "l1"},
    )

    targets_by_group = {target.dependency_group: target for target in targets}
    assert set(targets_by_group) == {"plain", "gated"}

    plain = targets_by_group["plain"]
    assert plain.module_name == "plain.fc1"
    assert plain.adapter == "mlp_pair_adapter"
    assert plain.structure_family == "transformer"
    assert plain.group_size == 12
    assert plain.metadata["parent_name"] == "plain"
    assert plain.metadata["partner_name"] == "plain.fc2"

    gated = targets_by_group["gated"]
    assert gated.module_name == "gated.up_proj"
    assert gated.adapter == "mlp_pair_adapter"
    assert gated.structure_family == "transformer"
    assert gated.group_size == 12
    assert gated.metadata["parent_name"] == "gated"
    assert gated.metadata["mlp_kind"] == "gated"
    assert gated.metadata["gate_proj_name"] == "gated.gate_proj"
    assert gated.metadata["up_proj_name"] == "gated.up_proj"
    assert gated.metadata["down_proj_name"] == "gated.down_proj"


def test_find_structured_pruning_targets_prefers_recorded_moe_usage() -> None:
    targets = find_structured_pruning_targets(
        _ToyMoEModel(),
        granularity="expert",
        importance={"metric": "usage"},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.module_name == "moe"
    assert target.dependency_group == "moe"
    assert target.adapter == "moe_expert_adapter"
    assert target.structure_family == "moe_expert"
    assert target.group_size == 3
    assert target.metadata["router_name"] == "moe.router"
    assert target.metadata["experts_name"] == "moe.experts"
    assert target.metadata["expert_names"] == [
        "moe.experts.0",
        "moe.experts.1",
        "moe.experts.2",
    ]
    assert target.metadata["usage_scores"] == [8.0, 3.0, 1.0]
    assert target.metadata["usage_source"] == "module_usage"


def test_find_structured_pruning_targets_detects_compatible_cnn_stages() -> None:
    targets = find_structured_pruning_targets(
        _ToyContainerCandidateModel(),
        granularity="stage",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.module_name == "stages"
    assert target.dependency_group == "stages"
    assert target.adapter == "cnn_stage_adapter"
    assert target.structure_family == "cnn_stage"
    assert target.group_size == 2
    assert target.metadata["container_type"] == "Sequential"
    assert target.metadata["stage_count"] == 2
    assert target.metadata["stage_names"] == ["stages.0", "stages.1"]
    assert target.metadata["input_channels"] == 4
    assert target.metadata["output_channels"] == 4
    assert target.metadata["shape_compatible"] is True


def test_find_structured_pruning_targets_detects_composite_blocks() -> None:
    targets = find_structured_pruning_targets(
        _ToyContainerCandidateModel(),
        granularity="block",
        importance={"metric": "l2"},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.module_name == "stages"
    assert target.dependency_group == "stages"
    assert target.adapter == "container_adapter"
    assert target.structure_family == "container"
    assert target.group_size == 2
    assert target.metadata["container_type"] == "Sequential"
    assert target.metadata["block_type"] == "_ToySpatialStage"
    assert target.metadata["block_types"] == ["_ToySpatialStage", "_ToySpatialStage"]
    assert target.metadata["heterogeneous_children"] is False


def _make_toy_vit() -> VisionTransformer:
    return VisionTransformer(
        img_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=3,
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
    )


def test_find_structured_pruning_targets_detects_vit_hidden_width() -> None:
    targets = find_structured_pruning_targets(
        _make_toy_vit(),
        granularity="hidden_width",
        importance={"metric": "l1"},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.module_name == "<root>"
    assert target.dependency_group == "<root>"
    assert target.adapter == "vit_hidden_width_adapter"
    assert target.structure_family == "transformer_width"
    assert target.group_size == 16
    assert target.metadata["width_kind"] == "hidden"
    assert target.metadata["model_family"] == "xdl_vit"
    assert target.metadata["num_heads"] == 4
    assert target.metadata["patch_proj_name"] == "patch_embed.proj"
    assert target.metadata["head_name"] == "head"
    assert target.metadata["block_count"] == 2
    constraints = target.metadata["group_alignment_constraints"]
    assert isinstance(constraints, list)
    assert constraints[0]["channels_per_group"] == 4


def test_find_structured_pruning_targets_detects_vit_embedding_width() -> None:
    targets = find_structured_pruning_targets(
        _make_toy_vit(),
        granularity="embedding_width",
        importance={"metric": "l2"},
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.module_name == "<root>"
    assert target.adapter == "vit_embedding_width_adapter"
    assert target.structure_family == "transformer_width"
    assert target.group_size == 16
    assert target.metadata["width_kind"] == "embedding"
    assert target.metadata["norm_name"] == "norm"


def test_plan_structured_pruning_uses_selection_for_grouped_query_attention() -> None:
    plan = plan_structured_pruning(
        _ToyAttentionModel(),
        0.5,
        granularity="head",
        importance={"metric": "l1"},
        selection={"keep_indices": {"attn": [0]}},
    )

    assert plan.granularity == "head"
    assert plan.scope == "global"
    assert plan.importance_metric == "l1"
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.action_type == "attention_heads"
    assert action.module_name == "attn"
    assert action.keep_indices == [0]
    assert action.prune_indices == [1]
    assert action.metadata["attention_variant"] == "gqa"


def test_plan_structured_pruning_keeps_vit_hidden_width_group_alignment() -> None:
    plan = plan_structured_pruning(
        _make_toy_vit(),
        0.5,
        granularity="hidden_width",
        scope="per_layer",
        importance={"metric": "l1"},
    )

    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.action_type == "vit_hidden_width"
    assert len(action.keep_indices) == 8
    assert len(action.prune_indices) == 8

    pruned_counts_by_offset = {offset: 0 for offset in range(4)}
    for index in action.prune_indices:
        pruned_counts_by_offset[index % 4] += 1
    assert set(pruned_counts_by_offset.values()) <= {0, 4}


def test_plan_structured_pruning_rejects_unknown_selection_candidates() -> None:
    try:
        plan_structured_pruning(
            _ToyMLPCandidateModel(),
            0.5,
            granularity="mlp_neuron",
            importance={"metric": "l1"},
            selection={"keep_indices": {"missing.fc1": [0]}},
        )
    except ValueError as exc:
        assert "unknown candidate" in str(exc)
    else:
        raise AssertionError("expected unknown selection.keep_indices candidate to fail")


def test_apply_structured_pruning_plan_rewrites_grouped_query_attention() -> None:
    model = _ToyAttentionModel()
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="head",
        importance={"metric": "l1"},
        selection={"keep_indices": {"attn": [0]}},
    )

    report = apply_structured_pruning_plan(
        model,
        plan,
        example_input=torch.randn(2, 3, 32),
    )

    assert report.forward_checked is True
    assert model.attn.num_heads == 2
    assert model.attn.num_kv_heads == 1
    assert model.attn.q_proj.out_features == 16
    assert model.attn.k_proj.out_features == 8
    assert model(torch.randn(2, 3, 32)).shape == (2, 3, 32)


def test_apply_structured_pruning_rewrites_moe_experts_and_usage() -> None:
    model = _ToyMoEModel()

    report = apply_structured_pruning(
        model,
        1.0 / 3.0,
        granularity="expert",
        importance={"metric": "usage"},
        selection={"keep_indices": {"moe": [0, 1]}},
        example_input=torch.randn(2, 8),
    )

    assert report.forward_checked is True
    assert model.moe.router.out_features == 2
    assert len(model.moe.experts) == 2
    assert model.moe.expert_usage == [8.0, 3.0]
    assert report.topology_changes[0]["kind"] == "expert"
    assert report.topology_changes[0]["pruned_experts"] == ["moe.experts.2"]


def test_apply_structured_pruning_plan_rewrites_vit_hidden_width() -> None:
    model = _make_toy_vit()
    plan = plan_structured_pruning(
        model,
        0.5,
        granularity="hidden_width",
        scope="per_layer",
        importance={"metric": "l1"},
    )

    report = apply_structured_pruning_plan(
        model,
        plan,
        example_input=torch.randn(2, 3, 8, 8),
    )

    assert report.forward_checked is True
    assert model.embed_dim == 8
    assert model.patch_embed.proj.out_channels == 8
    assert model.norm.normalized_shape == (8,)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 3)
