"""Branch-combine strategies for composite mixed-precision modules.

A composite module (SVD low-rank + quantized residual, outlier-split, sparse +
dense, ...) computes several branches and merges them. The *merge* is a small
strategy object rather than a bare ``"add"`` string, because each topology has
its own three questions:

- how per-branch outputs merge into one tensor (``combine_outputs``),
- how per-branch dense weights reconstruct the full weight for collapse / export
  (``reconstruct_weight``),
- whether the branches may legally fold into a single quantized GEMM
  (``can_collapse``) and whether the merge is expressible as a single-kernel
  epilogue (``supports_epilogue_fusion``).

Three strategies cover the current algorithms:

- ``add``    - every branch spans the full (out, in) map; y = sum(y_i),
               W = sum(W_i). Always collapsible. (SVDQuant, sparse+dense)
- ``concat`` - branches own disjoint *output* channels; y = cat(y_i, -1),
               W = cat(W_i, 0). Collapsible only when every branch shares one
               precision (else a high-precision block would be requantized).
- ``select`` - branches own disjoint *input* channels (mask); y = sum(y_i)
               over zero-padded columns, W scatters each branch's columns back.
               Never collapsible - keeping columns at different precision is the
               whole point (outlier-split).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

SUPPORTED_COMBINE_STRATEGIES: frozenset[str] = frozenset({"add", "concat", "select"})


class CombineStrategy(ABC):
    """How a composite module merges its branches (outputs and weights)."""

    #: Canonical strategy name, matches ``ModuleComputeSpec.combine``.
    name: str

    @abstractmethod
    def combine_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Merge per-branch runtime outputs into the module output."""

    @abstractmethod
    def reconstruct_weight(
        self, branch_weights: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Rebuild the full dense weight from per-branch dense weights.

        Used by ``collapse`` (fold to one quantized GEMM) and by export /
        validation. Branch order must match ``combine_outputs``.
        """

    @abstractmethod
    def can_collapse(self, branch_precisions: Sequence[str]) -> bool:
        """Whether folding all branches into one quantized GEMM is lossless-legal.

        ``branch_precisions`` are per-branch precision tags (e.g. ``"fp8"``,
        ``"source_precision"``); a strategy inspects them to reject folds that
        would requantize a deliberately high-precision branch.
        """

    @property
    @abstractmethod
    def supports_epilogue_fusion(self) -> bool:
        """Whether the merge can be a single-kernel epilogue (L2 fusion)."""


class AddCombine(CombineStrategy):
    """Additive merge: y = sum(y_i), W = sum(W_i). Always collapsible."""

    name = "add"

    def combine_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not outputs:
            raise ValueError("AddCombine requires at least one branch output")
        total = outputs[0]
        for extra in outputs[1:]:
            total = total + extra.to(device=total.device, dtype=total.dtype)
        return total

    def reconstruct_weight(
        self, branch_weights: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if not branch_weights:
            raise ValueError("AddCombine requires at least one branch weight")
        total = branch_weights[0]
        for extra in branch_weights[1:]:
            total = total + extra.to(device=total.device, dtype=total.dtype)
        return total

    def can_collapse(self, branch_precisions: Sequence[str]) -> bool:
        # Summed weights are one dense matrix; a single quant GEMM is exact up to
        # the (chosen) collapse precision, independent of per-branch precision.
        return True

    @property
    def supports_epilogue_fusion(self) -> bool:
        return True


class ConcatCombine(CombineStrategy):
    """Output-channel partition: y = cat(y_i, -1), W = cat(W_i, 0).

    Each branch owns a contiguous slice of output channels. Collapsible only
    when every branch shares one precision - otherwise a high-precision output
    block would be requantized down by the fold.
    """

    name = "concat"

    def combine_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not outputs:
            raise ValueError("ConcatCombine requires at least one branch output")
        ref = outputs[0]
        cast = [ref] + [o.to(device=ref.device, dtype=ref.dtype) for o in outputs[1:]]
        return torch.cat(cast, dim=-1)

    def reconstruct_weight(
        self, branch_weights: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if not branch_weights:
            raise ValueError("ConcatCombine requires at least one branch weight")
        ref = branch_weights[0]
        cast = [ref] + [
            w.to(device=ref.device, dtype=ref.dtype) for w in branch_weights[1:]
        ]
        return torch.cat(cast, dim=0)

    def can_collapse(self, branch_precisions: Sequence[str]) -> bool:
        tags = {str(p).strip().lower() for p in branch_precisions}
        return len(tags) <= 1

    @property
    def supports_epilogue_fusion(self) -> bool:
        # Disjoint output tiles can be written by one kernel with per-tile scale.
        return True


class SelectCombine(CombineStrategy):
    """Input-channel decomposition by column index; y = sum(y_i), W scatters back.

    Each branch reads a disjoint set of *input* columns (an outlier set kept at
    high precision, the rest quantized). Branch outputs sum because the column
    partitions are disjoint. Never collapsible - keeping distinct columns at
    distinct precision is the point.
    """

    name = "select"

    def __init__(self, column_index: Sequence[Sequence[int]], in_features: int) -> None:
        self.in_features = int(in_features)
        self.column_index: list[torch.Tensor] = [
            torch.as_tensor(list(idx), dtype=torch.long) for idx in column_index
        ]

    def combine_outputs(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not outputs:
            raise ValueError("SelectCombine requires at least one branch output")
        total = outputs[0]
        for extra in outputs[1:]:
            total = total + extra.to(device=total.device, dtype=total.dtype)
        return total

    def reconstruct_weight(
        self, branch_weights: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if len(branch_weights) != len(self.column_index):
            raise ValueError(
                "SelectCombine expects one weight per column-index group; got "
                f"{len(branch_weights)} weights for {len(self.column_index)} groups"
            )
        ref = branch_weights[0]
        out_features = ref.shape[0]
        full = torch.zeros(
            out_features, self.in_features, device=ref.device, dtype=ref.dtype
        )
        for weight, index in zip(branch_weights, self.column_index):
            full[:, index.to(full.device)] = weight.to(device=ref.device, dtype=ref.dtype)
        return full

    def can_collapse(self, branch_precisions: Sequence[str]) -> bool:
        return False

    @property
    def supports_epilogue_fusion(self) -> bool:
        # Masked accumulate: each branch writes its columns' partial sums.
        return True


_STATELESS_STRATEGIES: dict[str, CombineStrategy] = {
    "add": AddCombine(),
    "concat": ConcatCombine(),
}


def get_combine_strategy(name: str, **kwargs: object) -> CombineStrategy:
    """Resolve a combine strategy by name.

    ``add`` / ``concat`` are stateless singletons. ``select`` is stateful and
    requires ``column_index`` and ``in_features`` keyword arguments.
    """

    key = str(name).strip().lower()
    if key == "select":
        column_index = kwargs.get("column_index")
        in_features = kwargs.get("in_features")
        if column_index is None or in_features is None:
            raise ValueError(
                "select combine requires column_index and in_features"
            )
        return SelectCombine(column_index, int(in_features))  # type: ignore[arg-type]
    strategy = _STATELESS_STRATEGIES.get(key)
    if strategy is None:
        allowed = ", ".join(sorted(SUPPORTED_COMBINE_STRATEGIES))
        raise ValueError(f"combine must be one of {allowed}; got {name!r}")
    return strategy


__all__ = [
    "AddCombine",
    "CombineStrategy",
    "ConcatCombine",
    "SelectCombine",
    "SUPPORTED_COMBINE_STRATEGIES",
    "get_combine_strategy",
]
