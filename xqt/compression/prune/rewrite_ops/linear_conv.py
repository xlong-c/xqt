"""Linear and Conv2d channel pruning helpers."""

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


def prune_linear_in_out_features(
    module: nn.Linear,
    keep_in_indices: Sequence[int],
    keep_out_indices: Sequence[int],
) -> nn.Linear:
    """Return a new linear layer with selected input and output channels."""

    keep_in = _as_index_tensor(keep_in_indices, module.weight.device)
    keep_out = _as_index_tensor(keep_out_indices, module.weight.device)
    new_module = nn.Linear(
        keep_in.numel(),
        keep_out.numel(),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    new_module.weight.data.copy_(
        module.weight.data.index_select(0, keep_out).index_select(1, keep_in)
    )
    if module.bias is not None and new_module.bias is not None:
        new_module.bias.data.copy_(module.bias.data.index_select(0, keep_out))
    return _clone_training_state(module, new_module)


def _validate_conv_groups(module: nn.Conv2d) -> None:
    if module.groups == 1:
        return
    if module.groups == module.in_channels == module.out_channels:
        return
    if module.in_channels % module.groups != 0 or module.out_channels % module.groups != 0:
        raise ValueError("Grouped Conv2d must satisfy in_channels/out_channels divisibility")


def _group_keep_local_indices(
    *,
    total_channels: int,
    groups: int,
    keep_indices: Sequence[int],
) -> tuple[list[list[int]], int]:
    if total_channels % groups != 0:
        raise ValueError("total_channels must be divisible by groups")
    channels_per_group = total_channels // groups
    local_keep_by_group: list[list[int]] = [[] for _ in range(groups)]
    for index in keep_indices:
        if index < 0 or index >= total_channels:
            raise ValueError("keep_indices contains out-of-range channel index")
        group_index = index // channels_per_group
        local_keep_by_group[group_index].append(index % channels_per_group)
    keep_counts = {len(local_keep) for local_keep in local_keep_by_group}
    if len(keep_counts) != 1:
        raise ValueError(
            "Grouped Conv2d pruning requires keeping the same number of channels in every group"
        )
    kept_per_group = keep_counts.pop()
    if kept_per_group <= 0:
        raise ValueError("Grouped Conv2d pruning must keep at least one channel in every group")
    return [sorted(local_keep) for local_keep in local_keep_by_group], kept_per_group


def validate_conv2d_keep_indices(
    module: nn.Conv2d,
    keep_indices: Sequence[int],
    *,
    axis: str,
) -> None:
    """Validate keep indices for grouped Conv2d pruning."""

    if axis not in {"in", "out"}:
        raise ValueError("axis must be 'in' or 'out'")
    if module.groups == 1:
        return
    if module.groups == module.in_channels == module.out_channels:
        return
    total_channels = module.in_channels if axis == "in" else module.out_channels
    _group_keep_local_indices(
        total_channels=total_channels,
        groups=module.groups,
        keep_indices=keep_indices,
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
    elif module.groups == module.in_channels == module.out_channels:
        new_in_channels = keep.numel()
        new_groups = keep.numel()
        weight = module.weight.data.index_select(0, keep)
    else:
        _group_keep_local_indices(
            total_channels=module.out_channels,
            groups=module.groups,
            keep_indices=keep_indices,
        )
        new_in_channels = module.in_channels
        new_groups = module.groups
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
    elif module.groups == module.in_channels == module.out_channels:
        new_in_channels = keep.numel()
        new_groups = keep.numel()
        weight = module.weight.data.index_select(0, keep)
    else:
        local_keep_by_group, kept_per_group = _group_keep_local_indices(
            total_channels=module.in_channels,
            groups=module.groups,
            keep_indices=keep_indices,
        )
        out_channels_per_group = module.out_channels // module.groups
        grouped_weights: list[torch.Tensor] = []
        for group_index, local_keep in enumerate(local_keep_by_group):
            out_start = group_index * out_channels_per_group
            out_end = out_start + out_channels_per_group
            grouped_weights.append(
                module.weight.data[out_start:out_end].index_select(
                    1,
                    torch.tensor(
                        local_keep,
                        dtype=torch.long,
                        device=module.weight.device,
                    ),
                )
            )
        new_in_channels = keep.numel()
        new_groups = module.groups
        weight = torch.cat(grouped_weights, dim=0)
    new_module = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=(
            keep.numel()
            if module.groups == module.in_channels == module.out_channels
            else module.out_channels
        ),
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
        elif module.groups == module.in_channels == module.out_channels:
            new_module.bias.data.copy_(module.bias.data.index_select(0, keep))
        else:
            new_module.bias.data.copy_(module.bias.data)
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
