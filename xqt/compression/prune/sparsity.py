"""In-place N:M and block-sparse pruning for module weights."""

from __future__ import annotations

import math

import torch
from torch import nn

from .report import (
    BlockSparseLayerReport,
    BlockSparsePruningReport,
    NMStructuredLayerReport,
    NMStructuredPruningReport,
)


def _tensor_zero_count(tensor: torch.Tensor) -> int:
    return int(torch.count_nonzero(tensor == 0).item())


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _count_zero_parameters(
    model: nn.Module,
    *,
    module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> int:
    zero_parameters = 0
    for module in model.modules():
        if not isinstance(module, module_types):
            continue
        parameter = getattr(module, parameter_name, None)
        if isinstance(parameter, torch.Tensor):
            zero_parameters += _tensor_zero_count(parameter)
    return zero_parameters


def _nm_group_compliance(
    flat_tensor: torch.Tensor,
    *,
    pattern_n: int,
    pattern_m: int,
) -> tuple[int, int]:
    total_groups = int(flat_tensor.numel()) // pattern_m
    if total_groups == 0:
        return 0, 0
    groups = flat_tensor[: total_groups * pattern_m].reshape(total_groups, pattern_m)
    zero_counts = torch.count_nonzero(groups == 0, dim=1)
    compliant_groups = int(torch.count_nonzero(zero_counts == (pattern_m - pattern_n)).item())
    return compliant_groups, total_groups


def apply_nm_structured_sparsity(
    model: nn.Module,
    *,
    pattern_n: int,
    pattern_m: int,
    module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> NMStructuredPruningReport:
    """Apply in-place N:M structured sparsity to supported module weights."""

    if pattern_n <= 0 or pattern_m <= 0:
        raise ValueError("pattern_n and pattern_m must be positive")
    if pattern_n >= pattern_m:
        raise ValueError("pattern_n must be smaller than pattern_m")

    parameter_count_before = _count_parameters(model)
    zero_before = _count_zero_parameters(
        model,
        module_types=module_types,
        parameter_name=parameter_name,
    )
    layer_reports: list[NMStructuredLayerReport] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, module_types):
            continue
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            continue

        weight = parameter.data
        flat = weight.view(-1)
        usable = (flat.numel() // pattern_m) * pattern_m
        if usable == 0:
            layer_reports.append(
                NMStructuredLayerReport(
                    module_name=module_name or "<root>",
                    module_type=type(module).__name__,
                    parameter_name=parameter_name,
                    total_parameters=int(flat.numel()),
                    zero_parameters=_tensor_zero_count(weight),
                    sparsity=float(_tensor_zero_count(weight) / flat.numel()) if flat.numel() else 0.0,
                    pattern_n=pattern_n,
                    pattern_m=pattern_m,
                    compliant_groups=0,
                    total_groups=0,
                    compliance_ratio=0.0,
                )
            )
            continue

        groups = flat[:usable].view(-1, pattern_m)
        _, prune_indices = torch.topk(
            groups.abs(),
            k=pattern_m - pattern_n,
            dim=1,
            largest=False,
        )
        groups.scatter_(1, prune_indices, 0.0)

        zero_parameters = _tensor_zero_count(weight)
        compliant_groups, total_groups = _nm_group_compliance(
            flat,
            pattern_n=pattern_n,
            pattern_m=pattern_m,
        )
        layer_reports.append(
            NMStructuredLayerReport(
                module_name=module_name or "<root>",
                module_type=type(module).__name__,
                parameter_name=parameter_name,
                total_parameters=int(flat.numel()),
                zero_parameters=zero_parameters,
                sparsity=float(zero_parameters / flat.numel()) if flat.numel() else 0.0,
                pattern_n=pattern_n,
                pattern_m=pattern_m,
                compliant_groups=compliant_groups,
                total_groups=total_groups,
                compliance_ratio=(compliant_groups / total_groups) if total_groups else 0.0,
            )
        )

    return NMStructuredPruningReport(
        method="nm_structured",
        granularity="nm",
        parameter_count_before=parameter_count_before,
        parameter_count_after=_count_parameters(model),
        zero_parameters_before=zero_before,
        zero_parameters_after=_count_zero_parameters(
            model,
            module_types=module_types,
            parameter_name=parameter_name,
        ),
        pattern_n=pattern_n,
        pattern_m=pattern_m,
        module_types=[module_type.__name__ for module_type in module_types],
        layers=layer_reports,
        mask_only_modules=[layer.module_name for layer in layer_reports],
    )


def _block_sparse_matrix_view(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        return weight
    if weight.ndim == 4:
        return weight.reshape(weight.shape[0], -1)
    raise ValueError("block_sparse pruning only supports 2D Linear or 4D Conv2d weights")


def _block_sparse_zero_blocks(
    matrix: torch.Tensor,
    *,
    block_rows: int,
    block_cols: int,
) -> tuple[int, int]:
    usable_rows = (matrix.shape[0] // block_rows) * block_rows
    usable_cols = (matrix.shape[1] // block_cols) * block_cols
    if usable_rows == 0 or usable_cols == 0:
        return 0, 0
    block_view = matrix[:usable_rows, :usable_cols].reshape(
        usable_rows // block_rows,
        block_rows,
        usable_cols // block_cols,
        block_cols,
    )
    block_view = block_view.permute(0, 2, 1, 3).reshape(-1, block_rows * block_cols)
    total_blocks = int(block_view.shape[0])
    zero_blocks = int(torch.count_nonzero(torch.count_nonzero(block_view, dim=1) == 0).item())
    return zero_blocks, total_blocks


def apply_block_sparse_pruning(
    model: nn.Module,
    *,
    target_sparsity: float,
    block_shape: tuple[int, int] = (4, 4),
    module_types: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    parameter_name: str = "weight",
) -> BlockSparsePruningReport:
    """Apply in-place block-sparse pruning to supported module weights."""

    if target_sparsity < 0.0 or target_sparsity > 1.0:
        raise ValueError("target_sparsity must be in [0, 1]")
    block_rows, block_cols = int(block_shape[0]), int(block_shape[1])
    if block_rows <= 0 or block_cols <= 0:
        raise ValueError("block_shape values must be positive")

    parameter_count_before = _count_parameters(model)
    zero_before = _count_zero_parameters(
        model,
        module_types=module_types,
        parameter_name=parameter_name,
    )
    layer_reports: list[BlockSparseLayerReport] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, module_types):
            continue
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            continue
        matrix = _block_sparse_matrix_view(parameter.data)
        usable_rows = (matrix.shape[0] // block_rows) * block_rows
        usable_cols = (matrix.shape[1] // block_cols) * block_cols
        zero_blocks_before, total_blocks = _block_sparse_zero_blocks(
            matrix,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        pruned_blocks = 0
        if total_blocks > 0:
            block_region = matrix[:usable_rows, :usable_cols].reshape(
                usable_rows // block_rows,
                block_rows,
                usable_cols // block_cols,
                block_cols,
            )
            block_region = block_region.permute(0, 2, 1, 3)
            flat_blocks = block_region.detach().reshape(total_blocks, block_rows, block_cols)
            block_scores = flat_blocks.abs().sum(dim=(1, 2))
            existing_zero_blocks = torch.count_nonzero(flat_blocks, dim=(1, 2)) == 0
            block_scores = block_scores.masked_fill(existing_zero_blocks, float("inf"))
            target_zero_blocks = int(round(total_blocks * target_sparsity))
            nonzero_block_count = total_blocks - zero_blocks_before
            prune_count = max(0, min(target_zero_blocks - zero_blocks_before, nonzero_block_count))
            if prune_count > 0:
                ranked = torch.argsort(block_scores)[:prune_count]
                row_block_count = usable_rows // block_rows
                col_block_count = usable_cols // block_cols
                for flat_index in ranked.tolist():
                    row_block_index = int(flat_index) // col_block_count
                    col_block_index = int(flat_index) % col_block_count
                    row_start = row_block_index * block_rows
                    col_start = col_block_index * block_cols
                    matrix[
                        row_start : row_start + block_rows,
                        col_start : col_start + block_cols,
                    ] = 0
                pruned_blocks = int(prune_count)

        zero_blocks_after, total_blocks_after = _block_sparse_zero_blocks(
            matrix,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        zero_parameters = _tensor_zero_count(parameter.data)
        total_parameters = int(parameter.data.numel())
        layer_reports.append(
            BlockSparseLayerReport(
                module_name=module_name or "<root>",
                module_type=type(module).__name__,
                parameter_name=parameter_name,
                block_shape=(block_rows, block_cols),
                total_blocks=total_blocks_after,
                zero_blocks=zero_blocks_after,
                pruned_blocks=pruned_blocks,
                block_sparsity=(
                    zero_blocks_after / total_blocks_after if total_blocks_after else 0.0
                ),
                total_parameters=total_parameters,
                zero_parameters=zero_parameters,
                parameter_sparsity=(
                    zero_parameters / total_parameters if total_parameters else 0.0
                ),
            )
        )

    return BlockSparsePruningReport(
        method="block_sparse",
        granularity="block_sparse",
        target_sparsity=target_sparsity,
        block_shape=(block_rows, block_cols),
        parameter_count_before=parameter_count_before,
        parameter_count_after=_count_parameters(model),
        zero_parameters_before=zero_before,
        zero_parameters_after=_count_zero_parameters(
            model,
            module_types=module_types,
            parameter_name=parameter_name,
        ),
        module_types=[module_type.__name__ for module_type in module_types],
        layers=layer_reports,
        mask_only_modules=[layer.module_name for layer in layer_reports],
    )


__all__ = [
    "apply_block_sparse_pruning",
    "apply_nm_structured_sparsity",
]
