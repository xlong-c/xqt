"""End-to-end model compression quality assessment toolkit.

Provides comprehensive multi-dimensional fidelity, resource, and latency metrics
for comparing quantized or transformed models against their floating-point baselines:
- Numerical fidelity: Cosine similarity, MAE, MaxAbsError, MSE, RMSE, Relative L2.
- Prediction agreement: Top-1 match rate on logits.
- Footprint reduction: Parameter count, tensor storage bytes, compression ratio.
- Latency & speedup: Warmup-aware median latency benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F


def _extract_primary_tensor(output: Any) -> torch.Tensor | None:
    """Extract primary output tensor from tuple, dict, dataclass, or tensor."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _extract_primary_tensor(output[0])
    if isinstance(output, Mapping):
        if "logits" in output and isinstance(output["logits"], torch.Tensor):
            return output["logits"]
        if "last_hidden_state" in output and isinstance(output["last_hidden_state"], torch.Tensor):
            return output["last_hidden_state"]
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(output, "logits") and isinstance(getattr(output, "logits"), torch.Tensor):
        return getattr(output, "logits")
    return None


def _calculate_model_bytes(model: nn.Module) -> int:
    """Calculate total storage size in bytes of parameters and buffers."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes + buf_bytes


@dataclass(frozen=True)
class ModelCompressionQualityReport:
    """Comprehensive diagnostic report comparing compressed model against baseline."""

    mean_cosine_similarity: float
    min_cosine_similarity: float
    max_abs_error: float
    mean_abs_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    relative_l2_error: float
    top_1_agreement_rate: float | None
    float_parameters: int
    quant_parameters: int
    float_size_bytes: int
    quant_size_bytes: int
    compression_ratio: float
    sample_count: int
    median_latency_ms_float: float | None = None
    median_latency_ms_quant: float | None = None
    speedup_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def passes_gate(
        self,
        *,
        min_cosine: float = 0.99,
        max_mae: float = 0.1,
        min_top_1: float | None = None,
    ) -> bool:
        """Evaluate whether compressed model satisfies downstream quality threshold."""
        if self.mean_cosine_similarity < float(min_cosine):
            return False
        if self.mean_abs_error > float(max_mae):
            return False
        if min_top_1 is not None and self.top_1_agreement_rate is not None:
            if self.top_1_agreement_rate < float(min_top_1):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_cosine_similarity": float(self.mean_cosine_similarity),
            "min_cosine_similarity": float(self.min_cosine_similarity),
            "max_abs_error": float(self.max_abs_error),
            "mean_abs_error": float(self.mean_abs_error),
            "mean_squared_error": float(self.mean_squared_error),
            "root_mean_squared_error": float(self.root_mean_squared_error),
            "relative_l2_error": float(self.relative_l2_error),
            "top_1_agreement_rate": (
                float(self.top_1_agreement_rate)
                if self.top_1_agreement_rate is not None
                else None
            ),
            "float_parameters": int(self.float_parameters),
            "quant_parameters": int(self.quant_parameters),
            "float_size_bytes": int(self.float_size_bytes),
            "quant_size_bytes": int(self.quant_size_bytes),
            "compression_ratio": float(self.compression_ratio),
            "sample_count": int(self.sample_count),
            "median_latency_ms_float": self.median_latency_ms_float,
            "median_latency_ms_quant": self.median_latency_ms_quant,
            "speedup_ratio": self.speedup_ratio,
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        """Render a readable GitHub markdown summary table."""
        lines = [
            "### Model Compression Quality Assessment Report",
            "",
            "| Metric | Float Baseline | Compressed Model | Change / Ratio |",
            "| :--- | :--- | :--- | :--- |",
            f"| Storage Size | {self.float_size_bytes / 1024**2:.2f} MB | {self.quant_size_bytes / 1024**2:.2f} MB | {self.compression_ratio:.2f}x compression |",
            f"| Parameters | {self.float_parameters:,} | {self.quant_parameters:,} | - |",
            f"| Mean Cosine Similarity | 1.0000 | {self.mean_cosine_similarity:.4f} | {'PASS' if self.mean_cosine_similarity >= 0.99 else 'WARN'} |",
            f"| Min Cosine Similarity | 1.0000 | {self.min_cosine_similarity:.4f} | - |",
            f"| Mean Absolute Error | 0.0000 | {self.mean_abs_error:.6f} | - |",
            f"| Max Absolute Error | 0.0000 | {self.max_abs_error:.6f} | - |",
            f"| Relative L2 Error | 0.0000 | {self.relative_l2_error:.6f} | - |",
        ]
        if self.top_1_agreement_rate is not None:
            lines.append(
                f"| Top-1 Agreement | 100.0% | {self.top_1_agreement_rate * 100:.2f}% | - |"
            )
        if self.speedup_ratio is not None and self.median_latency_ms_float and self.median_latency_ms_quant:
            lines.append(
                f"| Median Latency | {self.median_latency_ms_float:.2f} ms | {self.median_latency_ms_quant:.2f} ms | {self.speedup_ratio:.2f}x speedup |"
            )
        return "\n".join(lines)


def evaluate_model_compression_quality(
    float_model: nn.Module,
    quant_model: nn.Module,
    dataloader: Iterable[Any],
    *,
    device: torch.device | str | None = None,
    sample_limit: int | None = 10,
    measure_latency: bool = False,
    latency_warmup: int = 3,
    latency_repeat: int = 10,
) -> ModelCompressionQualityReport:
    """Run end-to-end multi-dimensional quality evaluation on compressed model."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    float_model.eval()
    quant_model.eval()

    cos_sims: list[float] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    max_err = 0.0
    rel_l2_errors: list[float] = []
    top1_matches: list[float] = []
    total_samples = 0

    first_batch: Any = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if sample_limit is not None and i >= sample_limit:
                break
            if first_batch is None:
                first_batch = batch

            if isinstance(batch, torch.Tensor):
                dev_batch = batch.to(target_device)
                float_out = float_model(dev_batch)
                quant_out = quant_model(dev_batch)
            elif isinstance(batch, (list, tuple)):
                dev_batch = [b.to(target_device) if isinstance(b, torch.Tensor) else b for b in batch]
                float_out = float_model(*dev_batch)
                quant_out = quant_model(*dev_batch)
            elif isinstance(batch, Mapping):
                dev_batch = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                float_out = float_model(**dev_batch)
                quant_out = quant_model(**dev_batch)
            else:
                float_out = float_model(batch)
                quant_out = quant_model(batch)

            ref_tensor = _extract_primary_tensor(float_out)
            cand_tensor = _extract_primary_tensor(quant_out)

            if ref_tensor is None or cand_tensor is None:
                continue

            ref_f = ref_tensor.detach().to(device=target_device, dtype=torch.float32)
            cand_f = cand_tensor.detach().to(device=target_device, dtype=torch.float32)

            batch_size = ref_f.shape[0] if ref_f.ndim >= 1 else 1
            total_samples += int(batch_size)

            # Cosine similarity
            flat_ref = ref_f.reshape(batch_size, -1)
            flat_cand = cand_f.reshape(batch_size, -1)
            cos = F.cosine_similarity(flat_ref, flat_cand, dim=1).clamp(-1.0, 1.0)
            cos_sims.extend([float(v) for v in cos.cpu().tolist()])

            # Absolute error
            diff = (flat_ref - flat_cand).abs()
            curr_max = float(diff.amax().item()) if diff.numel() > 0 else 0.0
            if curr_max > max_err:
                max_err = curr_max
            abs_errors.append(float(diff.mean().item()))

            # Squared error
            sq_errors.append(float(diff.square().mean().item()))

            # Relative L2 error
            norm_ref = torch.norm(flat_ref, p=2, dim=1)
            norm_diff = torch.norm(diff, p=2, dim=1)
            rel_l2 = norm_diff / (norm_ref + 1e-8)
            rel_l2_errors.extend([float(v) for v in rel_l2.cpu().tolist()])

            # Top-1 agreement if logits
            if ref_f.ndim >= 2 and ref_f.shape[-1] > 1:
                match = (ref_f.argmax(dim=-1) == cand_f.argmax(dim=-1)).to(torch.float32)
                top1_matches.extend([float(v) for v in match.cpu().reshape(-1).tolist()])

    mean_cos = float(sum(cos_sims) / max(len(cos_sims), 1)) if cos_sims else 1.0
    min_cos = float(min(cos_sims)) if cos_sims else 1.0
    mean_mae = float(sum(abs_errors) / max(len(abs_errors), 1)) if abs_errors else 0.0
    mean_mse = float(sum(sq_errors) / max(len(sq_errors), 1)) if sq_errors else 0.0
    rmse = float(mean_mse ** 0.5)
    mean_rel_l2 = float(sum(rel_l2_errors) / max(len(rel_l2_errors), 1)) if rel_l2_errors else 0.0
    top1_rate = (
        float(sum(top1_matches) / max(len(top1_matches), 1))
        if top1_matches
        else None
    )

    float_params = sum(p.numel() for p in float_model.parameters())
    quant_params = sum(p.numel() for p in quant_model.parameters())
    float_bytes = _calculate_model_bytes(float_model)
    quant_bytes = _calculate_model_bytes(quant_model)
    comp_ratio = float_bytes / max(quant_bytes, 1)

    # Optional latency measurement
    med_float: float | None = None
    med_quant: float | None = None
    speedup: float | None = None

    if measure_latency and first_batch is not None:
        def _run_forward(m: nn.Module) -> None:
            if isinstance(first_batch, torch.Tensor):
                m(first_batch.to(target_device))
            elif isinstance(first_batch, (list, tuple)):
                m(*[b.to(target_device) if isinstance(b, torch.Tensor) else b for b in first_batch])
            elif isinstance(first_batch, Mapping):
                m(**{k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in first_batch.items()})

        # Warmup
        for _ in range(latency_warmup):
            _run_forward(float_model)
            _run_forward(quant_model)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

        # Measure float
        float_times: list[float] = []
        for _ in range(latency_repeat):
            t0 = time.perf_counter()
            _run_forward(float_model)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            float_times.append((time.perf_counter() - t0) * 1000.0)
        float_times.sort()
        med_float = float_times[len(float_times) // 2]

        # Measure quant
        quant_times: list[float] = []
        for _ in range(latency_repeat):
            t0 = time.perf_counter()
            _run_forward(quant_model)
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            quant_times.append((time.perf_counter() - t0) * 1000.0)
        quant_times.sort()
        med_quant = quant_times[len(quant_times) // 2]

        if med_quant > 0:
            speedup = med_float / med_quant

    return ModelCompressionQualityReport(
        mean_cosine_similarity=mean_cos,
        min_cosine_similarity=min_cos,
        max_abs_error=max_err,
        mean_abs_error=mean_mae,
        mean_squared_error=mean_mse,
        root_mean_squared_error=rmse,
        relative_l2_error=mean_rel_l2,
        top_1_agreement_rate=top1_rate,
        float_parameters=float_params,
        quant_parameters=quant_params,
        float_size_bytes=float_bytes,
        quant_size_bytes=quant_bytes,
        compression_ratio=comp_ratio,
        sample_count=total_samples,
        median_latency_ms_float=med_float,
        median_latency_ms_quant=med_quant,
        speedup_ratio=speedup,
    )


__all__ = [
    "ModelCompressionQualityReport",
    "evaluate_model_compression_quality",
]
