"""Output comparison helpers."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch


@dataclass
class TensorSummary:
    """Serializable tensor summary for reporting and visualization."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    mean: float
    std: float
    minimum: float
    maximum: float
    quantiles: dict[str, float]
    zero_ratio: float
    nan_count: int
    inf_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert the summary to a plain dictionary."""

        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "numel": self.numel,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "quantiles": dict(self.quantiles),
            "zero_ratio": self.zero_ratio,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass
class TensorDiff:
    """Tensor difference metrics."""

    max_abs: float
    mean_abs: float
    mean_squared: float
    sqnr_db: Optional[float]
    relative_error: Optional[float]
    cosine_similarity: Optional[float]
    correlation: Optional[float]
    argmax_mismatch_rate: Optional[float]
    allclose: bool
    atol: float
    rtol: float
    valid: bool
    message: str
    reference_summary: Optional[TensorSummary] = None
    candidate_summary: Optional[TensorSummary] = None
    details: Optional[dict[str, object]] = None

    def to_dict(self) -> dict[str, object]:
        """Convert diff metrics to a plain dictionary."""

        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "mean_squared": self.mean_squared,
            "sqnr_db": self.sqnr_db,
            "relative_error": self.relative_error,
            "cosine_similarity": self.cosine_similarity,
            "correlation": self.correlation,
            "argmax_mismatch_rate": self.argmax_mismatch_rate,
            "allclose": self.allclose,
            "atol": self.atol,
            "rtol": self.rtol,
            "valid": self.valid,
            "message": self.message,
            "reference_summary": (
                self.reference_summary.to_dict() if self.reference_summary is not None else None
            ),
            "candidate_summary": (
                self.candidate_summary.to_dict() if self.candidate_summary is not None else None
            ),
            "details": dict(self.details) if self.details is not None else None,
        }


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis if axis >= 0 else ndim + axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of range for ndim={ndim}")
    return normalized


def _structured_diff_for_axis(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    axis: int,
) -> dict[str, object]:
    normalized_axis = _normalize_axis(axis, reference.ndim)
    ref = reference.detach().to(dtype=torch.float32, device="cpu")
    cand = candidate.detach().to(dtype=torch.float32, device="cpu")
    delta = (ref - cand).abs().movedim(normalized_axis, 0)
    slices = delta.reshape(delta.shape[0], -1)
    return {
        "axis": normalized_axis,
        "size": int(delta.shape[0]),
        "mean_abs": [float(value.item()) for value in slices.mean(dim=1)],
        "max_abs": [float(value.item()) for value in slices.max(dim=1).values],
    }


def _build_structured_details(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    structured: bool | Mapping[str, int],
) -> Optional[dict[str, object]]:
    if structured is False:
        return None
    if structured is True:
        if reference.ndim == 0:
            return None
        axes: dict[str, int] = {"per_channel": -1}
    else:
        axes = {str(name): int(axis) for name, axis in structured.items()}
    if not axes:
        return None

    details: dict[str, object] = {}
    for name, axis in axes.items():
        try:
            details[name] = _structured_diff_for_axis(reference, candidate, axis=axis)
        except ValueError:
            continue
    return details or None


def _compute_argmax_mismatch_rate(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> Optional[float]:
    if reference.ndim == 0 or reference.shape[-1] <= 1:
        return None
    ref = reference.detach().to(dtype=torch.float32, device="cpu")
    cand = candidate.detach().to(dtype=torch.float32, device="cpu")
    if ref.ndim == 1:
        return float(ref.argmax().item() != cand.argmax().item())
    ref_indices = ref.reshape(-1, ref.shape[-1]).argmax(dim=-1)
    cand_indices = cand.reshape(-1, cand.shape[-1]).argmax(dim=-1)
    return float((ref_indices != cand_indices).float().mean().item())


def summarize_tensor(
    tensor: torch.Tensor,
    *,
    quantile_points: tuple[float, ...] = (0.01, 0.5, 0.99),
) -> TensorSummary:
    """Summarize a tensor into stable scalar statistics."""

    detached = tensor.detach()
    numel = int(detached.numel())
    flat = detached.to(dtype=torch.float32, device="cpu").flatten()
    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    finite = flat[~nan_mask & ~inf_mask]

    quantiles: dict[str, float] = {}
    if finite.numel():
        quantile_tensor = torch.tensor(quantile_points, dtype=torch.float32)
        quantile_values = torch.quantile(finite, quantile_tensor)
        quantiles = {
            f"p{int(point * 100):02d}": float(value.item())
            for point, value in zip(quantile_points, quantile_values)
        }
        mean = float(finite.mean().item())
        std = float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        zero_ratio = float((finite == 0).sum().item()) / float(finite.numel())
    else:
        mean = 0.0
        std = 0.0
        minimum = 0.0
        maximum = 0.0
        zero_ratio = 0.0

    return TensorSummary(
        shape=tuple(detached.shape),
        dtype=str(detached.dtype),
        device=str(detached.device),
        numel=numel,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        quantiles=quantiles,
        zero_ratio=zero_ratio,
        nan_count=int(nan_mask.sum().item()),
        inf_count=int(inf_mask.sum().item()),
    )


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    include_summary: bool = True,
    structured: bool | Mapping[str, int] = False,
) -> TensorDiff:
    """Compare two tensors using common deployment validation metrics."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"Tensor shapes differ: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )

    reference_summary = summarize_tensor(reference) if include_summary else None
    candidate_summary = summarize_tensor(candidate) if include_summary else None

    ref = reference.detach().to(dtype=torch.float32, device="cpu")
    cand = candidate.detach().to(dtype=torch.float32, device="cpu")
    delta = ref - cand
    flat_ref = ref.flatten()
    flat_cand = cand.flatten()
    signal_power = float(torch.dot(flat_ref, flat_ref).item()) if flat_ref.numel() else 0.0
    noise_power = float(torch.dot(delta.flatten(), delta.flatten()).item()) if delta.numel() else 0.0
    sqnr_db: Optional[float]
    if delta.numel() == 0:
        sqnr_db = None
    elif noise_power == 0.0:
        sqnr_db = float("inf") if signal_power > 0.0 else None
    elif signal_power == 0.0:
        sqnr_db = None
    else:
        sqnr_db = float(10.0 * torch.log10(torch.tensor(signal_power / noise_power)).item())

    cosine_similarity: Optional[float]
    if flat_ref.numel() == 0:
        cosine_similarity = None
    elif torch.linalg.vector_norm(flat_ref) == 0 or torch.linalg.vector_norm(flat_cand) == 0:
        cosine_similarity = None
    else:
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(flat_ref, flat_cand, dim=0).item()
        )

    correlation: Optional[float]
    if flat_ref.numel() <= 1:
        correlation = None
    else:
        ref_centered = flat_ref - flat_ref.mean()
        cand_centered = flat_cand - flat_cand.mean()
        ref_norm = torch.linalg.vector_norm(ref_centered)
        cand_norm = torch.linalg.vector_norm(cand_centered)
        if ref_norm == 0 or cand_norm == 0:
            correlation = None
        else:
            correlation = float(torch.dot(ref_centered, cand_centered).item() / (ref_norm * cand_norm))

    reference_norm = torch.linalg.vector_norm(flat_ref)
    relative_error = (
        float(torch.linalg.vector_norm(delta).item() / reference_norm.item())
        if delta.numel() and reference_norm.item() > 0.0
        else None
    )
    details = _build_structured_details(reference, candidate, structured=structured)
    argmax_mismatch_rate = _compute_argmax_mismatch_rate(reference, candidate)

    return TensorDiff(
        max_abs=float(delta.abs().max().item()) if delta.numel() else 0.0,
        mean_abs=float(delta.abs().mean().item()) if delta.numel() else 0.0,
        mean_squared=float((delta * delta).mean().item()) if delta.numel() else 0.0,
        sqnr_db=sqnr_db,
        relative_error=relative_error,
        cosine_similarity=cosine_similarity,
        correlation=correlation,
        argmax_mismatch_rate=argmax_mismatch_rate,
        allclose=bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
        atol=atol,
        rtol=rtol,
        valid=True,
        message="ok",
        reference_summary=reference_summary,
        candidate_summary=candidate_summary,
        details=details,
    )


__all__ = ["TensorDiff", "TensorSummary", "compare_tensors", "summarize_tensor"]
