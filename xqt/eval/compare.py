"""Output comparison helpers."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TensorDiff:
    """Tensor difference metrics."""

    max_abs: float
    mean_abs: float
    mean_squared: float
    cosine_similarity: Optional[float]
    allclose: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, float | bool | None]:
        """Convert diff metrics to a plain dictionary."""

        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "mean_squared": self.mean_squared,
            "cosine_similarity": self.cosine_similarity,
            "allclose": self.allclose,
            "atol": self.atol,
            "rtol": self.rtol,
        }


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TensorDiff:
    """Compare two tensors using common deployment validation metrics."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"Tensor shapes differ: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )

    ref = reference.detach().to(dtype=torch.float32, device="cpu")
    cand = candidate.detach().to(dtype=torch.float32, device="cpu")
    delta = ref - cand
    flat_ref = ref.flatten()
    flat_cand = cand.flatten()

    cosine_similarity: Optional[float]
    if flat_ref.numel() == 0:
        cosine_similarity = None
    elif torch.linalg.vector_norm(flat_ref) == 0 or torch.linalg.vector_norm(flat_cand) == 0:
        cosine_similarity = None
    else:
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(flat_ref, flat_cand, dim=0).item()
        )

    return TensorDiff(
        max_abs=float(delta.abs().max().item()) if delta.numel() else 0.0,
        mean_abs=float(delta.abs().mean().item()) if delta.numel() else 0.0,
        mean_squared=float((delta * delta).mean().item()) if delta.numel() else 0.0,
        cosine_similarity=cosine_similarity,
        allclose=bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
        atol=atol,
        rtol=rtol,
    )


__all__ = ["TensorDiff", "compare_tensors"]
