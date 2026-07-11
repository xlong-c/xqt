"""Orchestrator: apply_structured_pruning_plan."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from ..report import StructuredPruningPlan, StructuredPruningReport
from .attention import (
    PrunedGroupedQueryAttention,
    PrunedMultiHeadAttention,
    PrunedSplitProjectionAttention,
)
from .linear_conv import (
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
    prune_linear_out_features,
)
from .residual import _apply_concat_branch_action, _apply_mbconv_action, _apply_residual_stage_action
from .topology import (
    _expand_linear_keep_indices,
    _get_module,
    _rewrite_indexed_container,
    _set_module,
    topology_changes_from_actions,
)
from .vit_moe import _apply_drop_experts_action, _apply_vit_hidden_width_action


def _validate_forward(model: nn.Module, example_input: Any) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if isinstance(example_input, Mapping):
            model(**example_input)
        elif isinstance(example_input, tuple):
            model(*example_input)
        else:
            model(example_input)
    model.train(was_training)


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def apply_structured_pruning_plan(
    model: nn.Module,
    plan: StructuredPruningPlan,
    *,
    example_input: Any = None,
) -> StructuredPruningReport:
    """Apply a planned structured pruning rewrite in-place."""

    parameter_count_before = _count_parameters(model)
    concat_branch_keep_map = {
        item.module_name: list(item.keep_indices)
        for item in plan.actions
        if item.action_type == "concat_branch_channels"
    }
    rewritten_concat_consumers: set[str] = set()

    for action in plan.actions:
        if not action.prune_indices:
            continue
        if action.action_type == "residual_stage_channels":
            _apply_residual_stage_action(model, action)
            continue
        if action.action_type == "concat_branch_channels":
            _apply_concat_branch_action(
                model,
                action,
                concat_branch_keep_map,
                rewritten_concat_consumers,
            )
            continue
        if action.action_type == "mbconv_mid_channels":
            _apply_mbconv_action(model, action)
            continue
        if action.action_type == "vit_hidden_width":
            _apply_vit_hidden_width_action(model, action)
            continue
        if action.action_type == "drop_experts":
            _apply_drop_experts_action(model, action)
            continue
        if action.action_type == "conv_channel_group":
            producer = _get_module(model, action.module_name)
            if not isinstance(producer, nn.Conv2d):
                raise TypeError(f"{action.module_name} is not a Conv2d module")
            _set_module(
                model,
                action.module_name,
                prune_conv2d_out_channels(producer, action.keep_indices),
            )
            if action.normalization_name is not None:
                normalization = _get_module(model, action.normalization_name)
                if not isinstance(normalization, nn.modules.batchnorm._BatchNorm):
                    raise TypeError(f"{action.normalization_name} is not a BatchNorm module")
                _set_module(
                    model,
                    action.normalization_name,
                    prune_batchnorm_channels(normalization, action.keep_indices),
                )
            if action.consumer_name is None or action.consumer_type is None:
                continue
            consumer = _get_module(model, action.consumer_name)
            if action.consumer_type == "Conv2d":
                if not isinstance(consumer, nn.Conv2d):
                    raise TypeError(f"{action.consumer_name} is not a Conv2d module")
                _set_module(
                    model,
                    action.consumer_name,
                    prune_conv2d_in_channels(consumer, action.keep_indices),
                )
                continue
            if action.consumer_type == "Linear":
                if not isinstance(consumer, nn.Linear):
                    raise TypeError(f"{action.consumer_name} is not a Linear module")
                feature_keep_indices = _expand_linear_keep_indices(
                    action.keep_indices,
                    block_size=action.feature_block_size,
                )
                _set_module(
                    model,
                    action.consumer_name,
                    prune_linear_in_features(consumer, feature_keep_indices),
                )
                continue
            raise ValueError(f"Unsupported consumer type: {action.consumer_type}")

        if action.action_type == "mlp_neuron_group":
            fc1 = _get_module(model, action.module_name)
            if not isinstance(fc1, nn.Linear):
                raise TypeError(f"{action.module_name} is not a Linear module")
            if action.consumer_name is None:
                raise ValueError("mlp_neuron_group requires a consumer_name")
            fc2 = _get_module(model, action.consumer_name)
            if not isinstance(fc2, nn.Linear):
                raise TypeError(f"{action.consumer_name} is not a Linear module")
            _set_module(
                model,
                action.module_name,
                prune_linear_out_features(fc1, action.keep_indices),
            )
            _set_module(
                model,
                action.consumer_name,
                prune_linear_in_features(fc2, action.keep_indices),
            )
            continue

        if action.action_type == "gated_mlp_neuron_group":
            gate_proj_name = action.metadata.get("gate_proj_name")
            up_proj_name = action.metadata.get("up_proj_name")
            down_proj_name = action.metadata.get("down_proj_name")
            if not all(
                isinstance(name, str)
                for name in (gate_proj_name, up_proj_name, down_proj_name)
            ):
                raise ValueError(
                    "gated_mlp_neuron_group requires gate_proj_name/up_proj_name/down_proj_name"
                )
            gate_proj = _get_module(model, str(gate_proj_name))
            up_proj = _get_module(model, str(up_proj_name))
            down_proj = _get_module(model, str(down_proj_name))
            if not isinstance(gate_proj, nn.Linear):
                raise TypeError(f"{gate_proj_name} is not a Linear module")
            if not isinstance(up_proj, nn.Linear):
                raise TypeError(f"{up_proj_name} is not a Linear module")
            if not isinstance(down_proj, nn.Linear):
                raise TypeError(f"{down_proj_name} is not a Linear module")
            _set_module(
                model,
                str(gate_proj_name),
                prune_linear_out_features(gate_proj, action.keep_indices),
            )
            _set_module(
                model,
                str(up_proj_name),
                prune_linear_out_features(up_proj, action.keep_indices),
            )
            _set_module(
                model,
                str(down_proj_name),
                prune_linear_in_features(down_proj, action.keep_indices),
            )
            continue

        if action.action_type in {"drop_blocks", "drop_stages"}:
            container = _get_module(model, action.module_name)
            _set_module(
                model,
                action.module_name,
                _rewrite_indexed_container(container, action.keep_indices),
            )
            continue

        if action.action_type == "attention_heads":
            attention = _get_module(model, action.module_name)
            head_dim = getattr(attention, "head_dim", None)
            if not isinstance(head_dim, int):
                raise TypeError(f"{action.module_name} is not a supported attention module")
            attention_kind = str(action.metadata.get("attention_kind", "fused_qkv"))
            attention_variant = str(action.metadata.get("attention_variant", "mha"))
            if attention_kind == "split_qkv":
                if attention_variant in {"gqa", "mqa"}:
                    _set_module(
                        model,
                        action.module_name,
                        PrunedGroupedQueryAttention(attention, action.keep_indices),
                    )
                    continue
                _set_module(
                    model,
                    action.module_name,
                    PrunedSplitProjectionAttention(attention, action.keep_indices),
                )
                continue
            _set_module(
                model,
                action.module_name,
                PrunedMultiHeadAttention(attention, action.keep_indices),
            )
            continue

        raise ValueError(f"Unsupported structured action type: {action.action_type}")

    forward_checked = example_input is not None
    if forward_checked:
        _validate_forward(model, example_input)

    return StructuredPruningReport(
        method=plan.method,
        granularity=plan.granularity,
        scope=plan.scope,
        target_sparsity=plan.target_sparsity,
        importance_metric=plan.importance_metric,
        parameter_count_before=parameter_count_before,
        parameter_count_after=_count_parameters(model),
        forward_checked=forward_checked,
        adapters=list(plan.adapters),
        structure_families=list(plan.structure_families),
        blocked_modules=list(plan.blocked_modules),
        dependency_graph=dict(plan.dependency_graph),
        topology_changes=topology_changes_from_actions(plan.actions),
        export_status={"attempted": False, "passed": None, "artifacts": []},
        benchmark_status={
            "attempted": False,
            "passed": None,
            "latency": None,
            "memory": None,
        },
        targets=plan.targets,
        actions=plan.actions,
    )
