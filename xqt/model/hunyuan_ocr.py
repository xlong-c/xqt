"""HunyuanOCR model-side SVD method with composite_add Infer handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.workflows.optimization import XQTOptimizationSession

if TYPE_CHECKING:
    from xqt.workflows.optimization import OptimizationStageResult


HUNYUAN_OCR_REPO_ID = "tencent/HunyuanOCR"
HUNYUAN_OCR_DFLASH_SUBFOLDER = "dflash"
HUNYUAN_OCR_SVD_FP4_INT8_MMA_STRATEGY = "w4a16_fp4"


@dataclass(frozen=True)
class HunyuanOCRSVDQuantResult:
    """Quantized HunyuanOCR model with the accepted XQT quantization stage."""

    model: nn.Module
    stage: "OptimizationStageResult"
    compute_config: dict[str, Any] | None


def _model_device(model: nn.Module) -> torch.device:
    """Infer the device of the first parameter or buffer, defaulting to CPU."""

    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def load_hunyuan_ocr(
    *,
    repo_id: str = HUNYUAN_OCR_REPO_ID,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> nn.Module:
    """Load HunyuanOCR with its repository-provided Transformers implementation."""

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise XQTBackendError(
            "transformers is required to load HunyuanOCR from Hugging Face"
        ) from exc
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if subfolder is not None:
        normalized_subfolder = str(subfolder).strip().strip("/")
        if not normalized_subfolder:
            raise ValueError("subfolder must not be empty")
        kwargs["subfolder"] = normalized_subfolder
    model = AutoModel.from_pretrained(repo_id, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("HunyuanOCR AutoModel loader did not return torch.nn.Module")
    if device is not None:
        model = model.to(device)
    return model.eval()


def load_hunyuan_ocr_dflash(
    *,
    repo_id: str = HUNYUAN_OCR_REPO_ID,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load the Transformers-compatible DFlash model package from ``dflash``."""

    return load_hunyuan_ocr(
        repo_id=repo_id,
        revision=revision,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only,
        subfolder=HUNYUAN_OCR_DFLASH_SUBFOLDER,
    )


def optimize_hunyuan_ocr_svd_fp4_int8_mma(
    model: nn.Module,
    *,
    artifact_dir: str | Path = "artifacts/xqt/hunyuan_ocr_svd_fp4_int8_mma",
    rank: int = 32,
    group_size: int = 128,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    policy: Mapping[str, Any] | None = None,
    example_inputs: Any | None = None,
    calibration_inputs: Sequence[Any] | None = None,
) -> HunyuanOCRSVDQuantResult:
    """Apply SVD method: dual-branch storage + composite_add compute_config.

    Quant stage writes low-rank (source precision) + packed W4 residual storage,
    then optionally materializes residual INT8 MMA for Infer. The returned
    ``model`` keeps remote-code APIs (``generate`` / ``chat``). Qualifying
    ``nn.Linear`` modules become composite dual-branch modules
    (``SVDQuantInt8MmaLinear`` when residual compute is int8_mma).

    Infer handoff is ``model + compute_config`` with
    ``compute_contract=composite_add`` and branches
    ``low_rank`` / ``quant_residual``. ``engine`` is only a preferred_engines
    hint, not a required_engine primary key.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    linear_count = sum(
        1 for child in model.modules() if isinstance(child, nn.Linear)
    )
    if linear_count == 0:
        raise XQTBackendError("HunyuanOCR model does not expose any nn.Linear modules")
    quant_policy = dict(policy or {})
    quant_policy.update(
        {
            "dtype": "fp4",
            "scheme": "svd_int8_mma",
            "rank": int(rank),
            "group_size": int(group_size),
            "quant_dtype": "fp4",
            "residual_compute": "int8_mma",
            "engine": str(engine),
            "fallback_engine": str(fallback_engine),
        }
    )
    quant_policy.setdefault("include_module_types", ["Linear"])
    if calibration_inputs is not None:
        quant_policy.setdefault("activation_scale_mode", "static")
    session = XQTOptimizationSession(
        project={
            "name": "hunyuan_ocr_svd_fp4_int8_mma",
            "artifact_dir": str(artifact_dir),
        },
        model=model.eval(),
        device=str(_model_device(model)),
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )
    stage = session.quant(
        name=HUNYUAN_OCR_SVD_FP4_INT8_MMA_STRATEGY,
        backend="pytorch",
        method="svd",
        strategy=HUNYUAN_OCR_SVD_FP4_INT8_MMA_STRATEGY,
        compute="w8a8_int8_mma",
        policy=quant_policy,
    )
    if not isinstance(session.model, nn.Module):
        raise RuntimeError("XQT quantization did not return a torch.nn.Module")
    metadata = stage.metrics.get("metadata", {})
    compute_config = metadata.get("compute_config")
    if compute_config is not None and not isinstance(compute_config, Mapping):
        raise TypeError("quantization stage returned an invalid compute_config")
    return HunyuanOCRSVDQuantResult(
        model=session.model,
        stage=stage,
        compute_config=None if compute_config is None else dict(compute_config),
    )


def optimize_hunyuan_ocr_dflash_svd_fp4_int8_mma(
    model: nn.Module,
    *,
    artifact_dir: str | Path = "artifacts/xqt/hunyuan_ocr_dflash_svd_fp4_int8_mma",
    rank: int = 32,
    group_size: int = 128,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    policy: Mapping[str, Any] | None = None,
    example_inputs: Any | None = None,
    calibration_inputs: Sequence[Any] | None = None,
) -> HunyuanOCRSVDQuantResult:
    """Quantize a loaded HunyuanOCR DFlash model with SVDQuant and INT8 MMA."""

    return optimize_hunyuan_ocr_svd_fp4_int8_mma(
        model,
        artifact_dir=artifact_dir,
        rank=rank,
        group_size=group_size,
        engine=engine,
        fallback_engine=fallback_engine,
        policy=policy,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )


__all__ = [
    "HUNYUAN_OCR_REPO_ID",
    "HUNYUAN_OCR_DFLASH_SUBFOLDER",
    "HUNYUAN_OCR_SVD_FP4_INT8_MMA_STRATEGY",
    "HunyuanOCRSVDQuantResult",
    "load_hunyuan_ocr",
    "load_hunyuan_ocr_dflash",
    "optimize_hunyuan_ocr_svd_fp4_int8_mma",
    "optimize_hunyuan_ocr_dflash_svd_fp4_int8_mma",
]
