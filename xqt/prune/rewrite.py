"""Structured pruning rewrite helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _as_index_tensor(indices: Sequence[int], device: torch.device) -> torch.LongTensor:
    if not indices:
        raise ValueError("indices must not be empty")
    index_tensor = torch.tensor(list(indices), dtype=torch.long, device=device)
    if torch.unique(index_tensor).numel() != index_tensor.numel():
        raise ValueError("indices must be unique")
    return index_tensor


def _clone_training_state(source: nn.Module, target: nn.Module) -> nn.Module:
    target.train(source.training)
    return target


def prune_linear_out_features(
    module: nn.Linear,
    keep_indices: Sequence[int],
) -> nn.Linear:
    """Return a new linear layer with selected output channels."""

    keep = _as_index_tensor(keep_indices, module.weight.device)
    new_module = nn.Linear(
        module.in_features,
        keep.numel(),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(module.weight.data.index_select(0, keep))
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    return _clone_training_state(module, new_module)


def prune_linear_in_features(
    module: nn.Linear,
    keep_indices: Sequence[int],
) -> nn.Linear:
    """Return a new linear layer with selected input channels."""

    keep = _as_index_tensor(keep_indices, module.weight.device)
    new_module = nn.Linear(
        keep.numel(),
        module.out_features,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(module.weight.data.index_select(1, keep))
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    return _clone_training_state(module, new_module)


def _validate_conv_groups(module: nn.Conv2d) -> None:
    if module.groups == 1:
        return
    if module.groups == module.in_channels == module.out_channels:
        return
    raise ValueError(
        "Only groups=1 or depthwise Conv2d are supported by the structured pruning helpers"
    )


def prune_conv2d_out_channels(
    module: nn.Conv2d,
    keep_indices: Sequence[int],
) -> nn.Conv2d:
    """Return a new Conv2d layer with selected output channels."""

    _validate_conv_groups(module)
    keep = _as_index_tensor(keep_indices, module.weight.device)
    if module.groups == 1:
        new_in_channels = module.in_channels
        new_groups = 1
        weight = module.weight.data.index_select(0, keep)
    else:
        new_in_channels = keep.numel()
        new_groups = keep.numel()
        weight = module.weight.data.index_select(0, keep)
    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=keep.numel(),
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=new_groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(weight)
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    return _clone_training_state(module, new_module)


def prune_conv2d_in_channels(
    module: nn.Conv2d,
    keep_indices: Sequence[int],
) -> nn.Conv2d:
    """Return a new Conv2d layer with selected input channels."""

    _validate_conv_groups(module)
    keep = _as_index_tensor(keep_indices, module.weight.device)
    if module.groups == 1:
        new_in_channels = keep.numel()
        new_groups = 1
        weight = module.weight.data.index_select(1, keep)
    else:
        new_in_channels = keep.numel()
        new_groups = keep.numel()
        weight = module.weight.data.index_select(0, keep)
    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=module.out_channels if module.groups == 1 else keep.numel(),
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=new_groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(weight)
    if module.bias is not None and new_module.bias is not None:
        if module.groups == 1:
            new_module.bias.data.copy_(module.bias.data)
        else:
            new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    return _clone_training_state(module, new_module)


def prune_batchnorm_channels(
    module: nn.modules.batchnorm._BatchNorm,
    keep_indices: Sequence[int],
) -> nn.modules.batchnorm._BatchNorm:
    """Return a new BatchNorm layer with selected channels."""

    keep = _as_index_tensor(keep_indices, module.weight.device)
    new_module = type(module)(
        num_features=keep.numel(),
        eps=module.eps,
        momentum=module.momentum,
        affine=module.affine,
        track_running_stats=module.track_running_stats,
        device=module.weight.device if module.affine else None,
        dtype=module.weight.dtype if module.affine else None,
    )
    if module.affine and module.weight is not None and new_module.weight is not None:
        new_module.weight.data.copy_(module.weight.data.index_select(0, keep))
    if module.affine and module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
    if module.track_running_stats:
        new_module.running_mean.data.copy_(module.running_mean.data.index_select(0, keep))
        new_module.running_var.data.copy_(module.running_var.data.index_select(0, keep))
        new_module.num_batches_tracked.data.copy_(module.num_batches_tracked.data)
    return _clone_training_state(module, new_module)


__all__ = [
    "prune_batchnorm_channels",
    "prune_conv2d_in_channels",
    "prune_conv2d_out_channels",
    "prune_linear_in_features",
    "prune_linear_out_features",
]
