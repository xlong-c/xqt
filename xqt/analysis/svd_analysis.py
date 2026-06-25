"""SVD decomposition analysis for low-rank quantization (SVDQuant).

Reference:
  SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models
  (Li et al., NVIDIA/MIT, 2024) — https://arxiv.org/abs/2411.05007

The core idea:
  1. Decompose weight W via SVD: W = U @ diag(S) @ Vh
  2. Extract top-r components as a low-rank branch:
       L1 = U[:, :r] @ diag(sqrt(S[:r]))     # down-projection  (in × r)
       L2 = diag(sqrt(S[:r])) @ Vh[:r, :]     # up-projection    (r × out)
       W_lr = L2^T @ L1^T ≈ U_r @ diag(S_r) @ Vh_r
  3. Residual W_res = W - W_lr  → quantize to INT4/FP4
  4. At inference: y = (W_res_q @ x_q) + (L2 @ (L1 @ x_q))
     where down-proj + act-quantize share input, up-proj + 4-bit GEMM share output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.linalg


@dataclass
class SVDDecomposition:
    """Result of SVD decomposition for one weight matrix.

    W ≈ L1 @ L2  (low-rank approximation, rank=r)
    W_res = W - L1 @ L2  (residual to be quantized)

    Shapes (assuming W is out_features × in_features):
      - L1: (out_features, r)  — down-projection, applied after activation
      - L2: (r, in_features)   — up-projection, applied after MMA
    """

    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor
    rank: int
    singular_value_ratio: float = 0.0
    low_rank_error: float = 0.0

    def low_rank_components(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (L1, L2) such that W_lr = L1 @ L2.

        L1 shape: (out_features, r)  — used as x @ L1^T (or L1 @ x^T)
        L2 shape: (r, in_features)   — used as y @ L2^T (or L2 @ y^T)

        In the SVDQuant convention:
          down_proj:  x @ L1^T    (batch × in) @ (in × r) → (batch × r)
          up_proj:    z @ L2^T    (batch × r) @ (r × out) → (batch × out)

        So L1 is stored as (r, in_features) for PyTorch Linear,
        and L2 is stored as (out_features, r) for PyTorch Linear.
        """
        dtype = self.U.dtype
        r = min(self.rank, len(self.S))
        sqrt_S = torch.sqrt(torch.clamp(self.S[:r], min=0.0))
        # L1: down-projection weight = sqrt(S_r) @ Vh_r, shape (r, in_features)
        L1 = torch.diag(sqrt_S.to(dtype=dtype)) @ self.Vh[:r, :].to(dtype=dtype)
        # L2: up-projection weight = U_r @ sqrt(S_r), shape (out_features, r)
        L2 = self.U[:, :r].to(dtype=dtype) @ torch.diag(sqrt_S.to(dtype=dtype))
        return L1, L2  # type: ignore[return-value]

    def low_rank_weight(self) -> torch.Tensor:
        """Reconstruct the low-rank approximation W_lr = L1 @ L2."""
        L1, L2 = self.low_rank_components()
        return L2 @ L1  # (out_features, r) @ (r, in_features) = (out_features, in_features)

    def residual_weight(self, original_weight: torch.Tensor) -> torch.Tensor:
        """Compute W_res = W - L1 @ L2."""
        return original_weight - self.low_rank_weight()

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "singular_value_count": int(self.S.numel()),
            "singular_value_ratio": float(self.singular_value_ratio),
            "low_rank_error": float(self.low_rank_error),
            "low_rank_shape": [
                list(self.low_rank_weight().shape),
            ],
        }


def decompose_weight_svd(
    weight: torch.Tensor,
    *,
    rank: int,
    full_matrices: bool = False,
) -> SVDDecomposition:
    """Decompose a 2-D weight matrix via SVD.

    Args:
        weight: Weight tensor of shape (out_features, in_features).
        rank: Number of singular components to retain for the low-rank branch.
        full_matrices: Passed to torch.linalg.svd.

    Returns:
        SVDDecomposition with the full U, S, Vh plus pre-computed metrics.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2-D weight tensor, got shape {weight.shape}")

    effective_rank = min(rank, min(weight.shape) - 1)
    if effective_rank <= 0:
        raise ValueError(
            f"Rank {rank} is invalid for weight shape {weight.shape}. "
            f"Effective rank must be >= 1, got {effective_rank}."
        )

    dtype = weight.dtype
    compute_dtype = torch.float32
    weight_f32 = weight.detach().to(dtype=compute_dtype)

    U, S, Vh = torch.linalg.svd(weight_f32, full_matrices=full_matrices)

    # Metrics
    total_variance = (S**2).sum()
    retained_variance = (S[:effective_rank] ** 2).sum()
    singular_value_ratio = float(
        (retained_variance / total_variance).item()
    ) if total_variance > 0 else 1.0

    # Low-rank error = ||W - W_lr||_F / ||W||_F
    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vh_r = Vh[:effective_rank, :]
    W_lr = U_r @ torch.diag(S_r) @ Vh_r
    low_rank_error = float(
        (torch.linalg.matrix_norm(weight_f32 - W_lr, ord="fro")
         / (torch.linalg.matrix_norm(weight_f32, ord="fro") + 1e-12))
        .item()
    )

    return SVDDecomposition(
        U=U.to(dtype=dtype),
        S=S.to(dtype=torch.float32),
        Vh=Vh.to(dtype=dtype),
        rank=effective_rank,
        singular_value_ratio=singular_value_ratio,
        low_rank_error=low_rank_error,
    )


def compute_residual_weight(
    weight: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
) -> torch.Tensor:
    """Compute residual weight after subtracting low-rank approximation.

    W_res = W - L2 @ L1
    where L1 shape is (r, in_features), L2 shape is (out_features, r).
    """
    W_lr = torch.mm(L2, L1)
    return weight - W_lr


@dataclass
class SVDQuantAnalysis:
    """Aggregated SVD analysis results for a set of Linear modules.

    Provides a ranked list of which modules benefit most from
    low-rank decomposition (largest relative singular value drop-off).
    """

    decompositions: dict[str, SVDDecomposition] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def best_candidates(
        self,
        *,
        min_singular_value_ratio: float = 0.0,
        max_low_rank_error: float = 1.0,
        top_k: Optional[int] = None,
    ) -> list[str]:
        """Return module names sorted by low-rank quality (highest singular_value_ratio)."""
        candidates: list[tuple[str, float]] = []
        for name, decomp in self.decompositions.items():
            if decomp.singular_value_ratio >= min_singular_value_ratio:
                if decomp.low_rank_error <= max_low_rank_error:
                    candidates.append((name, decomp.singular_value_ratio))
        candidates.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            candidates = candidates[:top_k]
        return [name for name, _ in candidates]

    def worst_candidates(self, *, top_k: Optional[int] = None) -> list[str]:
        """Return module names that need high-precision fallback (lowest singular_value_ratio)."""
        sorted_items = sorted(
            self.decompositions.items(),
            key=lambda kv: kv[1].singular_value_ratio,
        )
        if top_k is not None:
            sorted_items = sorted_items[:top_k]
        return [name for name, _ in sorted_items]

    def to_dict(self) -> dict[str, object]:
        return {
            "module_count": len(self.decompositions),
            "decompositions": {
                name: decomp.to_dict() for name, decomp in self.decompositions.items()
            },
            "metadata": dict(self.metadata),
        }


__all__ = [
    "SVDDecomposition",
    "SVDQuantAnalysis",
    "compute_residual_weight",
    "decompose_weight_svd",
]
