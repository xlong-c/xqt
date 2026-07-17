"""HunyuanOCR INT4 SVD quantization with block-level inference compilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.operator_opt.compile_backend import compile_with_torch
from xqt.operator_opt.types import OperatorOptimizationTargetPlan
from xqt.runtime.modules import SVDQuantInt8MmaLinear
from xqt.workflows.optimization import XQTOptimizationSession

if TYPE_CHECKING:
    from xqt.workflows.optimization import OptimizationStageResult


HUNYUAN_OCR_REPO_ID = "tencent/HunyuanOCR"
HUNYUAN_OCR_DFLASH_SUBFOLDER = "dflash"
HUNYUAN_OCR_SVD_INT4_STRATEGY = "w4a16_int4"


@dataclass(frozen=True)
class HunyuanOCRBlockOptimization:
    """In-memory block compilation result for a quantized HunyuanOCR model."""

    block_paths: tuple[str, ...]
    engine: str
    dynamic: bool
    compile_time_ms: float
    warmup_input_source: str

    @property
    def compiled_block_count(self) -> int:
        """Return the number of transformer blocks materialized for inference."""

        return len(self.block_paths)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe block optimization metadata."""

        return {
            "level": "block",
            "engine": self.engine,
            "dynamic": self.dynamic,
            "block_paths": list(self.block_paths),
            "compiled_block_count": self.compiled_block_count,
            "compile_time_ms": self.compile_time_ms,
            "warmup_input_source": self.warmup_input_source,
        }


@dataclass(frozen=True)
class HunyuanOCRInt4QuantResult:
    """Quantized HunyuanOCR model and its block-level inference materialization."""

    model: nn.Module
    stage: "OptimizationStageResult"
    compute_config: dict[str, Any] | None
    block_optimization: HunyuanOCRBlockOptimization


def _model_device(model: nn.Module) -> torch.device:
    """Infer the device of the first parameter or buffer, defaulting to CPU."""

    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return torch.device("cpu") if buffer is None else buffer.device


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move structured model inputs to the model device."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _call_model(model: nn.Module, inputs: Any) -> Any:
    """Invoke a model with tensor, positional, or keyword example inputs."""

    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    return model(inputs)


def _quantized_linear_count(module: nn.Module) -> int:
    """Count materialized SVD INT4 linear modules below one candidate block."""

    return sum(isinstance(child, SVDQuantInt8MmaLinear) for child in module.modules())


def _discover_hunyuan_ocr_block_paths(model: nn.Module) -> tuple[str, ...]:
    """Find outermost ModuleList containers whose children are quantized blocks."""

    containers: list[tuple[str, tuple[str, ...]]] = []
    for container_name, container in model.named_modules():
        if not isinstance(container, nn.ModuleList):
            continue
        block_paths = tuple(
            f"{container_name}.{index}" if container_name else str(index)
            for index, block in enumerate(container)
            if _quantized_linear_count(block) >= 2
        )
        if block_paths:
            containers.append((container_name, block_paths))

    outermost_paths: list[str] = []
    selected_containers: list[str] = []
    for container_name, block_paths in sorted(
        containers,
        key=lambda item: item[0].count("."),
    ):
        if any(
            container_name == selected or container_name.startswith(f"{selected}.")
            for selected in selected_containers
        ):
            continue
        selected_containers.append(container_name)
        outermost_paths.extend(block_paths)

    if not outermost_paths:
        raise XQTBackendError(
            "HunyuanOCR block optimization requires an outer ModuleList whose "
            "children each contain at least two quantized Linear modules"
        )
    return tuple(outermost_paths)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace a named submodule while preserving ModuleList indexing semantics."""

    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def _compile_hunyuan_ocr_blocks(
    model: nn.Module,
    *,
    engine: str,
    mode: str | None,
    dynamic: bool,
) -> HunyuanOCRBlockOptimization:
    """Materialize torch.compile independently for each discovered transformer block."""

    block_paths = _discover_hunyuan_ocr_block_paths(model)
    total_compile_time_ms = 0.0
    for block_path in block_paths:
        plan = OperatorOptimizationTargetPlan(
            name=f"{block_path.replace('.', '_')}_hunyuan_ocr_block",
            engine="torch_compile",
            target_path=block_path,
            mode=mode,
            dynamic=dynamic,
            patterns=["transformer_block"],
            fallback="error",
            fallback_policy="strict",
            options={"engine": engine},
        )
        compiled_block, compile_time_ms = compile_with_torch(
            model.get_submodule(block_path),
            plan,
        )
        _replace_submodule(model, block_path, compiled_block)
        total_compile_time_ms += compile_time_ms
    return HunyuanOCRBlockOptimization(
        block_paths=block_paths,
        engine=engine,
        dynamic=dynamic,
        compile_time_ms=total_compile_time_ms,
        warmup_input_source="pending",
    )


def _resolve_warmup_inputs(
    *,
    example_inputs: Any | None,
    calibration_inputs: Sequence[Any] | None,
) -> tuple[Any, str]:
    """Use explicit example inputs first, then a calibration sample for warmup."""

    if example_inputs is not None:
        return example_inputs, "example_inputs"
    if calibration_inputs:
        return calibration_inputs[0], "calibration_inputs[0]"
    raise ValueError(
        "block-level HunyuanOCR optimization requires example_inputs or "
        "at least one calibration input for compilation warmup"
    )


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


def optimize_hunyuan_ocr_svd_int4_blocks(
    model: nn.Module,
    *,
    artifact_dir: str | Path = "artifacts/xqt/hunyuan_ocr_svd_int4_blocks",
    rank: int = 32,
    group_size: int = 128,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    block_engine: str = "inductor",
    block_mode: str | None = None,
    block_dynamic: bool = True,
    policy: Mapping[str, Any] | None = None,
    example_inputs: Any | None = None,
    calibration_inputs: Sequence[Any] | None = None,
) -> HunyuanOCRInt4QuantResult:
    """Quantize HunyuanOCR to INT4 and compile every logical transformer block.

    The quant stage stores SVD residuals as packed signed INT4 and materializes
    the existing INT8 MMA residual compute view. Inference is then optimized at
    the block boundary: each qualifying child of an outer ``nn.ModuleList`` is
    independently compiled and warmed with a real model invocation. This keeps
    remote-code APIs such as ``generate`` and ``chat`` on the returned model.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    warmup_inputs, warmup_input_source = _resolve_warmup_inputs(
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )
    linear_count = sum(1 for child in model.modules() if isinstance(child, nn.Linear))
    if linear_count == 0:
        raise XQTBackendError("HunyuanOCR model does not expose any nn.Linear modules")
    quant_policy = dict(policy or {})
    quant_policy.update(
        {
            "dtype": "int4",
            "scheme": "svd_int4_int8_mma",
            "rank": int(rank),
            "group_size": int(group_size),
            "quant_dtype": "int4",
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
            "name": "hunyuan_ocr_svd_int4_blocks",
            "artifact_dir": str(artifact_dir),
        },
        model=model.eval(),
        device=str(_model_device(model)),
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )
    stage = session.quant(
        name=HUNYUAN_OCR_SVD_INT4_STRATEGY,
        backend="pytorch",
        method="svd",
        strategy=HUNYUAN_OCR_SVD_INT4_STRATEGY,
        compute="w8a8_int8_mma",
        policy=quant_policy,
    )
    if not isinstance(session.model, nn.Module):
        raise RuntimeError("XQT quantization did not return a torch.nn.Module")
    metadata = stage.metrics.get("metadata", {})
    compute_config = metadata.get("compute_config")
    if compute_config is not None and not isinstance(compute_config, Mapping):
        raise TypeError("quantization stage returned an invalid compute_config")

    optimized_model = session.model.eval()
    block_optimization = _compile_hunyuan_ocr_blocks(
        optimized_model,
        engine=block_engine,
        mode=block_mode,
        dynamic=block_dynamic,
    )
    with torch.inference_mode():
        _call_model(
            optimized_model,
            _move_to_device(warmup_inputs, _model_device(optimized_model)),
        )
    block_optimization = HunyuanOCRBlockOptimization(
        block_paths=block_optimization.block_paths,
        engine=block_optimization.engine,
        dynamic=block_optimization.dynamic,
        compile_time_ms=block_optimization.compile_time_ms,
        warmup_input_source=warmup_input_source,
    )
    stage.metrics["hunyuan_ocr_block_optimization"] = block_optimization.to_dict()
    return HunyuanOCRInt4QuantResult(
        model=optimized_model,
        stage=stage,
        compute_config=None if compute_config is None else dict(compute_config),
        block_optimization=block_optimization,
    )


def optimize_hunyuan_ocr_dflash_svd_int4_blocks(
    model: nn.Module,
    *,
    artifact_dir: str | Path = "artifacts/xqt/hunyuan_ocr_dflash_svd_int4_blocks",
    rank: int = 32,
    group_size: int = 128,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    block_engine: str = "inductor",
    block_mode: str | None = None,
    block_dynamic: bool = True,
    policy: Mapping[str, Any] | None = None,
    example_inputs: Any | None = None,
    calibration_inputs: Sequence[Any] | None = None,
) -> HunyuanOCRInt4QuantResult:
    """Quantize a HunyuanOCR DFlash model to INT4 and compile its blocks."""

    return optimize_hunyuan_ocr_svd_int4_blocks(
        model,
        artifact_dir=artifact_dir,
        rank=rank,
        group_size=group_size,
        engine=engine,
        fallback_engine=fallback_engine,
        block_engine=block_engine,
        block_mode=block_mode,
        block_dynamic=block_dynamic,
        policy=policy,
        example_inputs=example_inputs,
        calibration_inputs=calibration_inputs,
    )


__all__ = [
    "HUNYUAN_OCR_REPO_ID",
    "HUNYUAN_OCR_DFLASH_SUBFOLDER",
    "HUNYUAN_OCR_SVD_INT4_STRATEGY",
    "HunyuanOCRBlockOptimization",
    "HunyuanOCRInt4QuantResult",
    "load_hunyuan_ocr",
    "load_hunyuan_ocr_dflash",
    "optimize_hunyuan_ocr_svd_int4_blocks",
    "optimize_hunyuan_ocr_dflash_svd_int4_blocks",
]
