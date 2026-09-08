"""OvisOCR2 BF16 baseline and ConvRot W8A8 inference helpers.

Ovis owns multimodal preprocessing and generation through ``chat`` and
``generate``. XQT observes only the language-model Linear inputs, so the
visual tokenizer, visual embedding, and language-model output head stay BF16.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

import torch
from torch import nn

from xqt.compression.quant.policy import QuantizationPolicy, should_quantize_module
from xqt.compression.quant.quantizers.convrot_4bit import (
    _apply_groupwise_rotation,
    _normalize_rot_size,
)
from xqt.compression.quant.quantizers.convrot_int8 import (
    ConvRotInt8QuantizationResult,
    quantize_with_convrot_int8,
)
from xqt.runtime.modules.convrot import materialize_convrot_execution_views


OVISOCR2_REPO_ID = "ATH-MaaS/OvisOCR2"
OVISOCR2_CONVROT_INT8_STRATEGY = "w8a8_int8"

_DEFAULT_EXCLUDE_NAME_PATTERNS = (
    r"^visual_tokenizer(?:\.|$)",
    r"^visual(?:\.|$)",
    r"^vte(?:\.|$)",
    r".*lm_head$",
)

CalibrationCall = Callable[[nn.Module], Any]


@dataclass(frozen=True)
class OvisOcr2ConvRotCalibration:
    """Static rotated-activation scales collected from an official Ovis call."""

    activation_scales: dict[str, float]
    target_module_names: tuple[str, ...]
    observed_module_names: tuple[str, ...]
    rot_size_by_module: dict[str, int]
    min_input_rows_by_module: dict[str, int]
    max_input_rows_by_module: dict[str, int]

    @property
    def missing_module_names(self) -> tuple[str, ...]:
        observed = set(self.observed_module_names)
        return tuple(name for name in self.target_module_names if name not in observed)

    def int8_eligible_module_names(self, min_input_rows: int) -> tuple[str, ...]:
        threshold = max(0, int(min_input_rows))
        return tuple(
            name
            for name in self.target_module_names
            if self.min_input_rows_by_module.get(name, -1) >= threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_mode": "static_signed_int8_per_tensor_rotated",
            "target_module_count": len(self.target_module_names),
            "observed_module_count": len(self.observed_module_names),
            "missing_module_names": list(self.missing_module_names),
            "rot_size_by_module": dict(self.rot_size_by_module),
            "min_input_rows_by_module": dict(self.min_input_rows_by_module),
            "max_input_rows_by_module": dict(self.max_input_rows_by_module),
        }


@dataclass(frozen=True)
class OvisOcr2ConvRotInt8Result:
    """OvisOCR2 ConvRot storage artifact and optional calibration lineage."""

    model: nn.Module
    quantization: ConvRotInt8QuantizationResult
    calibration: OvisOcr2ConvRotCalibration | None

    @property
    def quantized_modules(self) -> list[str]:
        return self.quantization.quantized_modules

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": OVISOCR2_REPO_ID,
            "method": "convrot",
            "strategy": OVISOCR2_CONVROT_INT8_STRATEGY,
            "quantized_module_count": len(self.quantized_modules),
            "quantization_metadata": dict(self.quantization.metadata),
            "calibration": None
            if self.calibration is None
            else self.calibration.to_dict(),
        }


def ovisocr2_convrot_default_policy() -> dict[str, Any]:
    """Return the OvisOCR2 LLM-only ConvRot selection policy."""

    return {
        "dtype": "int8",
        "scheme": "convrot_w8a8",
        "include_module_types": ["Linear"],
        "exclude_name_patterns": list(_DEFAULT_EXCLUDE_NAME_PATTERNS),
        "min_parameters": 65536,
    }


def _merged_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = ovisocr2_convrot_default_policy()
    if policy is not None:
        merged.update(dict(policy))
    return merged


def _selection_policy(policy: Mapping[str, Any]) -> QuantizationPolicy:
    return QuantizationPolicy(
        dtype=str(policy.get("dtype", "int8")),
        scheme=str(policy.get("scheme", "convrot_w8a8")),
        include_module_types=tuple(
            str(value) for value in policy.get("include_module_types", ())
        ),
        exclude_module_types=tuple(
            str(value) for value in policy.get("exclude_module_types", ())
        ),
        include_name_patterns=tuple(
            str(value) for value in policy.get("include_name_patterns", ())
        ),
        exclude_name_patterns=tuple(
            str(value) for value in policy.get("exclude_name_patterns", ())
        ),
        include_module_names=tuple(
            str(value) for value in policy.get("include_module_names", ())
        ),
        exclude_module_names=tuple(
            str(value) for value in policy.get("exclude_module_names", ())
        ),
        min_parameters=int(policy.get("min_parameters", 0)),
    )


def select_ovisocr2_convrot_modules(
    model: nn.Module,
    *,
    policy: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Select eligible OvisOCR2 language-model Linear modules."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    policy_mapping = _merged_policy(policy)
    effective_policy = _selection_policy(policy_mapping)
    selection_mode = (
        str(policy_mapping.get("selection_mode", "default")).strip().lower()
    )
    if selection_mode not in {"default", "include_only"}:
        raise ValueError("selection_mode must be 'default' or 'include_only'")
    include_names = set(effective_policy.include_module_names)
    include_patterns = tuple(effective_policy.include_name_patterns)
    selected: list[str] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if not (
            name.startswith(
                (
                    "llm.",
                    "language_model.",
                    "model.language_model.",
                    "model.layers.",
                    "model.model.layers.",
                )
            )
            or "self_attn." in name
            or "mlp." in name
        ) or name.endswith("lm_head"):
            continue
        if selection_mode == "include_only" and not (
            name in include_names
            or any(re.search(pattern, name) for pattern in include_patterns)
        ):
            continue
        if should_quantize_module(name, module, effective_policy):
            selected.append(name)
    return tuple(selected)


def calibrate_ovisocr2_convrot_activation_scales(
    model: nn.Module,
    *,
    calibration_call: CalibrationCall,
    policy: Mapping[str, Any] | None = None,
    rot_size: int = 256,
    eps: float = 1e-6,
) -> OvisOcr2ConvRotCalibration:
    """Observe one official ``chat`` or ``generate`` call for static W8A8 scales."""

    if not callable(calibration_call):
        raise TypeError("calibration_call must be callable")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    target_names = select_ovisocr2_convrot_modules(model, policy=policy)
    if not target_names:
        raise ValueError("OvisOCR2 ConvRot policy selected no nn.Linear modules")
    rot_size_by_module = {
        name: _normalize_rot_size(
            int(rot_size), int(model.get_submodule(name).in_features)
        )
        for name in target_names
    }
    maxima: dict[str, float] = {}
    min_rows: dict[str, int] = {}
    max_rows: dict[str, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if (
                not inputs
                or not isinstance(inputs[0], torch.Tensor)
                or inputs[0].numel() == 0
            ):
                return
            values = inputs[0].detach()
            rows = int(values.reshape(-1, values.shape[-1]).shape[0])
            min_rows[name] = min(min_rows.get(name, rows), rows)
            max_rows[name] = max(max_rows.get(name, 0), rows)
            rotated = _apply_groupwise_rotation(
                values.to(torch.float32), rot_size=rot_size_by_module[name]
            )
            maxima[name] = max(
                maxima.get(name, 0.0), float(rotated.abs().amax().item())
            )

        return hook

    try:
        for name in target_names:
            handles.append(
                model.get_submodule(name).register_forward_hook(make_hook(name))
            )
        with torch.inference_mode():
            calibration_call(model)
    finally:
        for handle in handles:
            handle.remove()
    return OvisOcr2ConvRotCalibration(
        activation_scales={
            name: max(value / 127.0, float(eps)) for name, value in maxima.items()
        },
        target_module_names=target_names,
        observed_module_names=tuple(maxima),
        rot_size_by_module=rot_size_by_module,
        min_input_rows_by_module=min_rows,
        max_input_rows_by_module=max_rows,
    )


def load_ovisocr2(
    *,
    repo_id: str = OVISOCR2_REPO_ID,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load OvisOCR2 through its repository-provided Transformers model class."""

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required to load OvisOCR2") from exc
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(
            "OvisOCR2 AutoModelForCausalLM loader did not return torch.nn.Module"
        )
    if device is not None:
        model = model.to(device)
    return model.eval()


def quantize_ovisocr2_convrot_int8(
    model: nn.Module,
    *,
    calibration_call: CalibrationCall | None = None,
    policy: Mapping[str, Any] | None = None,
    activation_scales: Mapping[str, torch.Tensor | float] | None = None,
    activation_scale_mode: str = "dynamic",
    rot_size: int = 256,
    engine: str = "auto",
    fallback_engine: str = "torch_int_mm",
    mse_clip: bool = True,
    mse_clip_grid: int = 80,
    min_int8_rows: int = 256,
    only_int8_eligible_modules: bool = False,
    eps: float = 1e-6,
    inplace: bool = True,
) -> OvisOcr2ConvRotInt8Result:
    """Quantize OvisOCR2 LLM projections with rotation-aware W8A8 storage."""

    if activation_scale_mode not in {"static", "dynamic"}:
        raise ValueError("activation_scale_mode must be 'static' or 'dynamic'")
    effective_policy = _merged_policy(policy)
    effective_policy.update(
        {
            "rot_size": int(rot_size),
            "engine": str(engine),
            "fallback_engine": str(fallback_engine),
            "activation_scale_mode": str(activation_scale_mode),
            "mse_clip": bool(mse_clip),
            "mse_clip_grid": int(mse_clip_grid),
            "min_int8_rows": int(min_int8_rows),
            "eps": float(eps),
        }
    )
    calibration: OvisOcr2ConvRotCalibration | None = None
    scales = dict(activation_scales or {})
    candidate_names = select_ovisocr2_convrot_modules(model, policy=effective_policy)
    if not candidate_names:
        raise ValueError("OvisOCR2 ConvRot policy selected no LLM projection modules")
    effective_policy.update(
        {
            "include_module_names": list(candidate_names),
            "include_name_patterns": [],
            "selection_mode": "include_only",
        }
    )
    needs_calibration = activation_scale_mode == "static" and not scales
    if only_int8_eligible_modules:
        needs_calibration = True
    if needs_calibration:
        if calibration_call is None:
            raise ValueError("this ConvRot mode requires a real calibration_call")
        calibration = calibrate_ovisocr2_convrot_activation_scales(
            model,
            calibration_call=calibration_call,
            policy=effective_policy,
            rot_size=rot_size,
            eps=eps,
        )
        scales.update(calibration.activation_scales)
    if only_int8_eligible_modules:
        if calibration is None:
            raise RuntimeError("OvisOCR2 calibration was not collected")
        eligible_modules = calibration.int8_eligible_module_names(min_int8_rows)
        if not eligible_modules:
            raise ValueError(
                "calibration found no module reaching min_int8_rows for ConvRot"
            )
        effective_policy.update(
            {
                "include_module_names": list(eligible_modules),
                "include_name_patterns": [],
                "selection_mode": "include_only",
            }
        )
        scales = {name: scales[name] for name in eligible_modules if name in scales}
    quantization = quantize_with_convrot_int8(
        model,
        policy=effective_policy,
        strategy=OVISOCR2_CONVROT_INT8_STRATEGY,
        inplace=inplace,
        engine=engine,
        fallback_engine=fallback_engine,
        activation_scale_mode=activation_scale_mode,
        activation_scales=scales,
        eps=eps,
        mse_clip=mse_clip,
        mse_clip_grid=mse_clip_grid,
        min_int8_rows=min_int8_rows,
    )
    return OvisOcr2ConvRotInt8Result(
        model=quantization.model, quantization=quantization, calibration=calibration
    )


def materialize_ovisocr2_convrot_int8_runtime(
    model: nn.Module,
    *,
    inplace: bool = True,
) -> nn.Module:
    """Enable the explicit ConvRot runtime view for OvisOCR2 inference."""

    return materialize_convrot_execution_views(model, inplace=inplace)


def make_shared_convrot_w8a8_group(
    projections: list[nn.Module],
    *,
    sample_inputs: torch.Tensor | None = None,
    rot_size: int = 256,
    norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> Any:
    """Create a SharedConvRotW8A8Group from a list of ConvRot W8A8 Linear layers."""

    from xqt.kernels.ops._impl.cute.convrot_w8a8_sm89 import (
        PackedConvRotW8A8Linear,
        SharedConvRotW8A8Group,
    )

    packeds: list[PackedConvRotW8A8Linear] = []
    for proj in projections:
        target = getattr(proj, "storage", proj)
        if hasattr(target, "_native_w8a8_packed"):
            if sample_inputs is not None:
                dummy = sample_inputs
            else:
                buf = next(target.buffers(), None)
                if buf is not None:
                    device = buf.device
                elif hasattr(target, "int8_compute"):
                    device = target.int8_compute.qweight_t.device
                else:
                    device = torch.device("cuda")
                dummy = torch.empty(
                    (1, target.input_features),
                    dtype=torch.bfloat16,
                    device=device,
                )
            packeds.append(target._native_w8a8_packed(dummy))
        elif isinstance(target, PackedConvRotW8A8Linear):
            packeds.append(target)
        else:
            raise TypeError(
                f"Module {proj} does not expose ConvRot W8A8 packed state"
            )
    return SharedConvRotW8A8Group(
        packeds,
        rot_size=rot_size,
        norm_weight=norm_weight,
        eps=eps,
    )


def make_fused_convrot_w8a8_group(
    projections: list[nn.Module],
    *,
    sample_inputs: torch.Tensor | None = None,
    rot_size: int = 256,
    norm_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    min_int8_rows: int = 256,
) -> Any:
    """Create a FusedConvRotW8A8LinearGroup fusing projections into a single horizontal GEMM."""

    from xqt.kernels.ops._impl.cute.convrot_w8a8_sm89 import (
        FusedConvRotW8A8LinearGroup,
        PackedConvRotW8A8Linear,
    )

    packeds: list[PackedConvRotW8A8Linear] = []
    for proj in projections:
        target = getattr(proj, "storage", proj)
        if hasattr(target, "_native_w8a8_packed"):
            if sample_inputs is not None:
                dummy = sample_inputs
            else:
                buf = next(target.buffers(), None)
                if buf is not None:
                    device = buf.device
                elif hasattr(target, "int8_compute"):
                    device = target.int8_compute.qweight_t.device
                else:
                    device = torch.device("cuda")
                dummy = torch.empty(
                    (1, target.input_features),
                    dtype=torch.bfloat16,
                    device=device,
                )
            packeds.append(target._native_w8a8_packed(dummy))
        elif isinstance(target, PackedConvRotW8A8Linear):
            packeds.append(target)
        else:
            raise TypeError(
                f"Module {proj} does not expose ConvRot W8A8 packed state"
            )
    return FusedConvRotW8A8LinearGroup(
        packeds,
        rot_size=rot_size,
        norm_weight=norm_weight,
        eps=eps,
        min_int8_rows=min_int8_rows,
    )


__all__ = [
    "OVISOCR2_CONVROT_INT8_STRATEGY",
    "OVISOCR2_REPO_ID",
    "OvisOcr2ConvRotCalibration",
    "OvisOcr2ConvRotInt8Result",
    "calibrate_ovisocr2_convrot_activation_scales",
    "load_ovisocr2",
    "make_fused_convrot_w8a8_group",
    "make_shared_convrot_w8a8_group",
    "materialize_ovisocr2_convrot_int8_runtime",
    "ovisocr2_convrot_default_policy",
    "quantize_ovisocr2_convrot_int8",
    "select_ovisocr2_convrot_modules",
]
