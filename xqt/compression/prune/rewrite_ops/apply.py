"""Orchestrator: apply_structured_pruning_plan."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from xqt.analysis.compare import compare_tensors


def _assess_export_readiness(model: nn.Module) -> Any:
    from xqt.export.export_readiness import assess_export_readiness

    return assess_export_readiness(model)

from ..dimensions import diff_module_dimensions, snapshot_module_dimensions
from ..flops import estimate_model_flops
from ..report import StructuredPruningPlan, StructuredPruningReport
from ..safety import assess_prune_safety
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


def _forward_outputs(model: nn.Module, example_input: Any) -> list[torch.Tensor]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if isinstance(example_input, Mapping):
            outputs = model(**example_input)
        elif isinstance(example_input, tuple):
            outputs = model(*example_input)
        else:
            outputs = model(example_input)
    model.train(was_training)
    return _collect_tensors(outputs)


def _collect_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            tensors.extend(_collect_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_collect_tensors(item))
    return tensors


def _forward_diff_report(
    reference_outputs: list[torch.Tensor],
    candidate_outputs: list[torch.Tensor],
) -> dict[str, Any]:
    if len(reference_outputs) != len(candidate_outputs):
        return {
            "checked": True,
            "status": "failed",
            "tensor_count": len(candidate_outputs),
            "max_abs": None,
            "mean_abs": None,
            "cosine_similarity": None,
            "allclose": None,
            "message": (
                f"output tensor count changed from {len(reference_outputs)} "
                f"to {len(candidate_outputs)}"
            ),
        }
    if not candidate_outputs:
        return {
            "checked": True,
            "status": "failed",
            "tensor_count": 0,
            "max_abs": None,
            "mean_abs": None,
            "cosine_similarity": None,
            "allclose": None,
            "message": "model produced no tensor outputs to compare",
        }

    max_abs_values: list[float] = []
    mean_abs_values: list[float] = []
    cosine_values: list[float] = []
    allclose_values: list[bool] = []
    for reference, candidate in zip(reference_outputs, candidate_outputs):
        if reference.shape != candidate.shape:
            return {
                "checked": True,
                "status": "failed",
                "tensor_count": len(candidate_outputs),
                "max_abs": None,
                "mean_abs": None,
                "cosine_similarity": None,
                "allclose": None,
                "message": (
                    f"output tensor shape mismatch: reference={tuple(reference.shape)}, "
                    f"candidate={tuple(candidate.shape)}"
                ),
            }
        diff = compare_tensors(reference, candidate, atol=1e-4, rtol=1e-3)
        max_abs_values.append(diff.max_abs)
        mean_abs_values.append(diff.mean_abs)
        allclose_values.append(diff.allclose)
        if diff.cosine_similarity is not None:
            cosine_values.append(diff.cosine_similarity)

    return {
        "checked": True,
        "status": "passed",
        "tensor_count": len(candidate_outputs),
        "max_abs": max(max_abs_values),
        "mean_abs": max(mean_abs_values),
        "cosine_similarity": min(cosine_values) if cosine_values else None,
        "allclose": all(allclose_values),
        "message": "baseline vs pruned forward diff computed",
    }


def _action_target_module_names(plan: StructuredPruningPlan) -> set[str]:
    names: set[str] = set()
    for action in plan.actions:
        names.add(action.module_name)
        for key in (
            "gate_proj_name",
            "up_proj_name",
            "down_proj_name",
            "router_name",
            "experts_name",
        ):
            value = action.metadata.get(key)
            if isinstance(value, str):
                names.add(value)
    return names


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def apply_structured_pruning_plan(
    model: nn.Module,
    plan: StructuredPruningPlan,
    *,
    example_input: Any = None,
    task_type: str | None = None,
) -> StructuredPruningReport:
    """Apply a planned structured pruning rewrite in-place."""

    safety = assess_prune_safety(model, plan.actions, task_type=task_type)
    if not safety.passed:
        raise ValueError(
            "Structured pruning safety guard rejected the plan: "
            + "; ".join(safety.violations)
        )

    parameter_count_before = _count_parameters(model)
    flops_before = estimate_model_flops(model)["flops"]
    dimensions_before = snapshot_module_dimensions(model)
    reference_outputs = (
        _forward_outputs(model, example_input) if example_input is not None else []
    )
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
    forward_diff: dict[str, Any]
    if forward_checked:
        candidate_outputs = _forward_outputs(model, example_input)
        forward_diff = _forward_diff_report(reference_outputs, candidate_outputs)
    else:
        forward_diff = {
            "checked": False,
            "status": "not_run",
            "tensor_count": 0,
            "max_abs": None,
            "mean_abs": None,
            "cosine_similarity": None,
            "allclose": None,
            "message": "no example_input provided; forward diff not run",
        }

    dimensions_after = snapshot_module_dimensions(model)
    removed_modules, changed_dimensions = diff_module_dimensions(
        dimensions_before,
        dimensions_after,
    )
    action_targets = _action_target_module_names(plan)
    mask_only_modules = sorted(
        name
        for name in action_targets
        if name in dimensions_after
        and dimensions_before.get(name) is not None
        and dimensions_before[name]["dimensions"]
        == dimensions_after[name]["dimensions"]
    )
    flops_after = estimate_model_flops(model)["flops"]
    flops_reduction_ratio = (
        (flops_before - flops_after) / flops_before
        if flops_before and flops_before > 0.0
        else None
    )

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
        removed_modules=removed_modules,
        changed_dimensions=changed_dimensions,
        mask_only_modules=mask_only_modules,
        flops_before=flops_before,
        flops_after=flops_after,
        flops_reduction_ratio=flops_reduction_ratio,
        safety=safety.to_dict(),
        forward_diff=forward_diff,
        export_readiness=_assess_export_readiness(model).to_dict(),
        targets=plan.targets,
        actions=plan.actions,
    )
