"""Residual stage, concat branch, and MBConv action appliers."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from ..report import StructuredPruningAction
from .linear_conv import (
    prune_batchnorm_channels,
    prune_conv2d_in_channels,
    prune_conv2d_out_channels,
    prune_linear_in_features,
)
from .topology import _expand_linear_keep_indices, _get_module, _set_module


def _apply_residual_stage_action(
    model: nn.Module,
    action: StructuredPruningAction,
) -> None:
    stage = _get_module(model, action.module_name)
    if not isinstance(stage, nn.Sequential):
        raise TypeError(f"{action.module_name} is not a Sequential stage")
    block_type = str(action.metadata.get("block_type"))
    blocks = action.metadata.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("residual_stage_channels requires metadata.blocks")

    keep = action.keep_indices
    stage_out_channels = int(action.metadata.get("stage_out_channels", action.original_units))
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("metadata.blocks entries must be mappings")
        conv1_name = str(block["conv1_name"])
        bn1_name = str(block["bn1_name"])
        conv2_name = str(block["conv2_name"])
        bn2_name = str(block["bn2_name"])
        conv1 = _get_module(model, conv1_name)

        if block_type == "BasicBlock":
            bn1 = _get_module(model, bn1_name)
            conv2 = _get_module(model, conv2_name)
            bn2 = _get_module(model, bn2_name)
            if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d):
                raise TypeError("BasicBlock residual stage pruning requires conv1/conv2")
            if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(
                bn2, nn.modules.batchnorm._BatchNorm
            ):
                raise TypeError("BasicBlock residual stage pruning requires bn1/bn2")
            if conv1.in_channels == stage_out_channels:
                _set_module(model, conv1_name, prune_conv2d_in_channels(conv1, keep))
                conv1 = _get_module(model, conv1_name)
                if not isinstance(conv1, nn.Conv2d):
                    raise TypeError(f"{conv1_name} is not a Conv2d module")
            _set_module(model, conv1_name, prune_conv2d_out_channels(conv1, keep))
            _set_module(model, bn1_name, prune_batchnorm_channels(bn1, keep))
            conv2 = _get_module(model, conv2_name)
            if not isinstance(conv2, nn.Conv2d):
                raise TypeError(f"{conv2_name} is not a Conv2d module")
            _set_module(model, conv2_name, prune_conv2d_in_channels(conv2, keep))
            conv2 = _get_module(model, conv2_name)
            if not isinstance(conv2, nn.Conv2d):
                raise TypeError(f"{conv2_name} is not a Conv2d module")
            _set_module(model, conv2_name, prune_conv2d_out_channels(conv2, keep))
            _set_module(model, bn2_name, prune_batchnorm_channels(bn2, keep))
        elif block_type == "Bottleneck":
            conv3_name = str(block["conv3_name"])
            bn3_name = str(block["bn3_name"])
            if not isinstance(conv1, nn.Conv2d):
                raise TypeError(f"{conv1_name} is not a Conv2d module")
            if conv1.in_channels == stage_out_channels:
                _set_module(model, conv1_name, prune_conv2d_in_channels(conv1, keep))
            conv3 = _get_module(model, conv3_name)
            bn3 = _get_module(model, bn3_name)
            if not isinstance(conv3, nn.Conv2d):
                raise TypeError(f"{conv3_name} is not a Conv2d module")
            if not isinstance(bn3, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{bn3_name} is not a BatchNorm module")
            _set_module(model, conv3_name, prune_conv2d_out_channels(conv3, keep))
            _set_module(model, bn3_name, prune_batchnorm_channels(bn3, keep))
        else:
            raise ValueError(f"Unsupported residual block_type: {block_type}")

        downsample_conv_name = block.get("downsample_conv_name")
        downsample_bn_name = block.get("downsample_bn_name")
        if isinstance(downsample_conv_name, str) and isinstance(downsample_bn_name, str):
            downsample_conv = _get_module(model, downsample_conv_name)
            downsample_bn = _get_module(model, downsample_bn_name)
            if not isinstance(downsample_conv, nn.Conv2d):
                raise TypeError(f"{downsample_conv_name} is not a Conv2d module")
            if not isinstance(downsample_bn, nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{downsample_bn_name} is not a BatchNorm module")
            _set_module(
                model,
                downsample_conv_name,
                prune_conv2d_out_channels(downsample_conv, keep),
            )
            _set_module(model, downsample_bn_name, prune_batchnorm_channels(downsample_bn, keep))

    consumers = action.metadata.get("consumers")
    if not isinstance(consumers, list):
        raise ValueError("residual_stage_channels requires metadata.consumers")
    for item in consumers:
        if not isinstance(item, Mapping):
            raise ValueError("metadata.consumers entries must be mappings")
        consumer_name = item.get("name")
        consumer_type = item.get("type")
        feature_block_size = int(item.get("feature_block_size", 1))
        if not isinstance(consumer_name, str) or not isinstance(consumer_type, str):
            continue
        consumer = _get_module(model, consumer_name)
        if consumer_type == "Conv2d":
            if not isinstance(consumer, nn.Conv2d):
                raise TypeError(f"{consumer_name} is not a Conv2d module")
            _set_module(model, consumer_name, prune_conv2d_in_channels(consumer, keep))
            continue
        if consumer_type == "Linear":
            if not isinstance(consumer, nn.Linear):
                raise TypeError(f"{consumer_name} is not a Linear module")
            feature_keep_indices = _expand_linear_keep_indices(
                keep,
                block_size=feature_block_size,
            )
            _set_module(model, consumer_name, prune_linear_in_features(consumer, feature_keep_indices))
            continue
        raise ValueError(f"Unsupported residual consumer type: {consumer_type}")


def _apply_concat_branch_action(
    model: nn.Module,
    action: StructuredPruningAction,
    branch_keep_map: Mapping[str, list[int]],
    rewritten_consumers: set[str],
) -> None:
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

    if action.consumer_name is None or action.consumer_type != "Conv2d":
        return
    if action.consumer_name in rewritten_consumers:
        return
    branch_specs = action.metadata.get("branch_specs")
    if not isinstance(branch_specs, list):
        raise ValueError("concat_branch_channels requires metadata.branch_specs")
    consumer = _get_module(model, action.consumer_name)
    if not isinstance(consumer, nn.Conv2d):
        raise TypeError(f"{action.consumer_name} is not a Conv2d module")
    input_keep_indices: list[int] = []
    offset = 0
    for branch_spec in branch_specs:
        if not isinstance(branch_spec, Mapping):
            raise ValueError("metadata.branch_specs entries must be mappings")
        conv_name = branch_spec.get("conv_name")
        out_channels = int(branch_spec.get("out_channels", 0))
        if not isinstance(conv_name, str):
            raise ValueError("metadata.branch_specs[*].conv_name must be a string")
        branch_keep = branch_keep_map.get(conv_name)
        if branch_keep is None:
            current_branch = _get_module(model, conv_name)
            if not isinstance(current_branch, nn.Conv2d):
                raise TypeError(f"{conv_name} is not a Conv2d module")
            branch_keep = list(range(current_branch.out_channels))
        input_keep_indices.extend(offset + int(index) for index in branch_keep)
        offset += out_channels
    _set_module(
        model,
        action.consumer_name,
        prune_conv2d_in_channels(consumer, input_keep_indices),
    )
    rewritten_consumers.add(action.consumer_name)


def _apply_mbconv_action(model: nn.Module, action: StructuredPruningAction) -> None:
    expand_conv_name = str(action.metadata["expand_conv_name"])
    expand_bn_name = str(action.metadata["expand_bn_name"])
    depthwise_conv_name = str(action.metadata["depthwise_conv_name"])
    depthwise_bn_name = str(action.metadata["depthwise_bn_name"])
    project_conv_name = str(action.metadata["project_conv_name"])
    expand_conv = _get_module(model, expand_conv_name)
    expand_bn = _get_module(model, expand_bn_name)
    depthwise_conv = _get_module(model, depthwise_conv_name)
    depthwise_bn = _get_module(model, depthwise_bn_name)
    project_conv = _get_module(model, project_conv_name)
    if not all(isinstance(module, nn.Conv2d) for module in (expand_conv, depthwise_conv, project_conv)):
        raise TypeError("MBConv pruning requires Conv2d modules")
    if not all(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in (expand_bn, depthwise_bn)
    ):
        raise TypeError("MBConv pruning requires BatchNorm modules")

    keep = action.keep_indices
    _set_module(model, expand_conv_name, prune_conv2d_out_channels(expand_conv, keep))
    _set_module(model, expand_bn_name, prune_batchnorm_channels(expand_bn, keep))
    depthwise_conv = _get_module(model, depthwise_conv_name)
    if not isinstance(depthwise_conv, nn.Conv2d):
        raise TypeError(f"{depthwise_conv_name} is not a Conv2d module")
    _set_module(model, depthwise_conv_name, prune_conv2d_in_channels(depthwise_conv, keep))
    _set_module(model, depthwise_bn_name, prune_batchnorm_channels(depthwise_bn, keep))
    _set_module(model, project_conv_name, prune_conv2d_in_channels(project_conv, keep))
