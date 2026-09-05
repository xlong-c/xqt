"""Orthogonal Rotation Quantization Transform (SpinQuant / QuaRot style).

Applies offline orthogonal rotation to weight pairs to suppress activation
outliers before quantization. For two connected linear operations:
    Y = X @ W_pred.T @ W_succ.T
We insert an orthogonal matrix R (R @ R.T = I):
    W_pred' = R.T @ W_pred
    W_succ' = W_succ @ R
The mathematical composition remains strictly identical while activation
variance and outliers across channels are evenly distributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import torch
from torch import nn

from .base import GraphQuantTransform, TransformPlan, TransformReport


def build_random_orthogonal_matrix(
    dim: int,
    *,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a deterministic random orthogonal matrix via QR decomposition (Haar distribution)."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(
        (dim, dim),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    q, r = torch.linalg.qr(random_matrix)
    d = torch.diag(r, 0)
    ph = d.sign()
    q = q * ph
    return q.to(torch.float32)


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    curr = root
    for p in path.split("."):
        if p.isdigit():
            curr = curr[int(p)]
        else:
            curr = getattr(curr, p)
    return curr


@dataclass(frozen=True)
class OrthogonalRotationTransform:
    """Offline orthogonal rotation transform for adjacent Linear pairs."""

    name: str = "orthogonal_rotation"
    required_kernels: tuple[str, ...] = ()
    matrix_kind: str = "random_orthogonal"
    seed: int = 42

    def _generate_rotation(self, dim: int) -> torch.Tensor:
        if self.matrix_kind == "random_orthogonal":
            return build_random_orthogonal_matrix(dim, seed=self.seed)
        raise ValueError(f"unsupported matrix_kind: {self.matrix_kind}")

    def match(self, model: nn.Module) -> TransformPlan | None:
        """Find adjacent Linear module pairs with matching dimensions."""
        linears: list[tuple[str, nn.Linear]] = [
            (name, mod)
            for name, mod in model.named_modules()
            if name and isinstance(mod, nn.Linear)
        ]
        pairs: list[dict[str, Any]] = []

        for i in range(len(linears) - 1):
            pred_name, pred_mod = linears[i]
            succ_name, succ_mod = linears[i + 1]

            # Condition: pred output dimension matches succ input dimension
            if pred_mod.out_features == succ_mod.in_features:
                dim = pred_mod.out_features
                pairs.append(
                    {
                        "predecessor": pred_name,
                        "successor": succ_name,
                        "dim": dim,
                    }
                )

        if not pairs:
            return None

        targets: list[str] = []
        absorbed: list[str] = []
        for pair in pairs:
            targets.extend([pair["predecessor"], pair["successor"]])
            absorbed.append(f"{pair['predecessor']}->{pair['successor']}")

        return TransformPlan(
            transform_name=self.name,
            targets=tuple(sorted(set(targets))),
            absorbed_ops=tuple(absorbed),
            metadata={"pairs": pairs, "matrix_kind": self.matrix_kind, "seed": self.seed},
        )

    def apply(self, model: nn.Module, plan: TransformPlan) -> TransformReport:
        """Apply orthogonal rotations in-place to the matched pairs."""
        pairs = plan.metadata.get("pairs", [])
        if not pairs:
            return TransformReport(
                transform_name=self.name,
                applied=False,
                notes=("no_pairs_in_plan",),
            )

        applied_pairs: list[str] = []

        with torch.no_grad():
            for pair_info in pairs:
                pred_name = pair_info["predecessor"]
                succ_name = pair_info["successor"]
                dim = pair_info["dim"]

                pred_mod = _get_submodule(model, pred_name)
                succ_mod = _get_submodule(model, succ_name)
                if not isinstance(pred_mod, nn.Linear) or not isinstance(succ_mod, nn.Linear):
                    continue

                r = self._generate_rotation(dim).to(
                    device=pred_mod.weight.device,
                    dtype=pred_mod.weight.dtype,
                )

                # W_pred: shape [out_features, in_features]
                # W_pred' = R.T @ W_pred
                new_pred_weight = r.t() @ pred_mod.weight.data
                pred_mod.weight.copy_(new_pred_weight)

                # Bias of predecessor is also transformed: bias' = R.T @ bias
                if pred_mod.bias is not None:
                    new_pred_bias = (r.t() @ pred_mod.bias.data.unsqueeze(-1)).squeeze(-1)
                    pred_mod.bias.copy_(new_pred_bias)

                # W_succ: shape [out_features, in_features]
                # W_succ' = W_succ @ R
                new_succ_weight = succ_mod.weight.data @ r
                succ_mod.weight.copy_(new_succ_weight)

                applied_pairs.append(f"{pred_name}->{succ_name}")

        return TransformReport(
            transform_name=self.name,
            applied=bool(applied_pairs),
            absorbed_ops=tuple(applied_pairs),
            targets=plan.targets,
            metadata={
                "applied_pairs_count": len(applied_pairs),
                "matrix_kind": self.matrix_kind,
                "seed": self.seed,
            },
        )


__all__ = [
    "OrthogonalRotationTransform",
    "build_random_orthogonal_matrix",
]
