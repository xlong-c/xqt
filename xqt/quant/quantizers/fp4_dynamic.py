"""Dynamic FP4 runtime quantization for Linear modules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from xqt.contracts import ComputeConfig, QuantizedModel
from xqt.core.errors import XQTBackendError
from xqt.core.types import XQTContext
from xqt.operator_opt.kernels.fp4_quant_common import (
    TILELANG_NVFP4_WEIGHT_BLOCK_K,
    TILELANG_NVFP4_WEIGHT_BLOCK_N,
    dequantize_nvfp4_codes,
    prepack_nvfp4_weight_for_tilelang,
)
from xqt.contracts.engine_resolve import resolve_engine

from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import build_component_quantization_report
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from .base import (
    policy_from_mapping as _policy_from_mapping,
    replace_submodule as _replace_submodule,
)


_VALID_ENGINES = frozenset({"auto", "tilelang", "triton", "torch"})
_VALID_FORMATS = frozenset({"nvfp4", "mxfp4"})
_NVFP4_GLOBAL_SCALE_RANGE = 6.0 * 448.0


def _can_mutate_runtime_cache() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling) and bool(is_compiling()):
            return False
    return not torch.jit.is_tracing()


def _tilelang_fp4_api() -> dict[str, Any]:
    from xqt.operator_opt.kernels.tilelang import (
        mxfp4_packed_activation_gemm_epilogue_reference,
        mxfp4_packed_activation_gemm_epilogue_tilelang,
        mxfp4_packed_dequant_gemm_epilogue_reference,
        mxfp4_packed_dequant_gemm_epilogue_tilelang,
        nvfp4_packed_activation_gemm_epilogue_reference,
        nvfp4_packed_activation_gemm_epilogue_tilelang,
        scaled_mxfp4_quant_tilelang,
        scaled_nvfp4_quant_tilelang,
        nvfp4_packed_dequant_gemm_epilogue_reference,
        nvfp4_packed_dequant_gemm_epilogue_tilelang,
    )

    return {
        "scaled_nvfp4_quant_tilelang": scaled_nvfp4_quant_tilelang,
        "scaled_mxfp4_quant_tilelang": scaled_mxfp4_quant_tilelang,
        "mxfp4_packed_activation_reference": mxfp4_packed_activation_gemm_epilogue_reference,
        "mxfp4_packed_activation_tilelang": mxfp4_packed_activation_gemm_epilogue_tilelang,
        "mxfp4_gemm_reference": mxfp4_packed_dequant_gemm_epilogue_reference,
        "mxfp4_gemm_tilelang": mxfp4_packed_dequant_gemm_epilogue_tilelang,
        "nvfp4_packed_activation_reference": nvfp4_packed_activation_gemm_epilogue_reference,
        "nvfp4_packed_activation_tilelang": nvfp4_packed_activation_gemm_epilogue_tilelang,
        "nvfp4_gemm_reference": nvfp4_packed_dequant_gemm_epilogue_reference,
        "nvfp4_gemm_tilelang": nvfp4_packed_dequant_gemm_epilogue_tilelang,
    }


def _triton_fp4_api() -> dict[str, Any]:
    from xqt.operator_opt.kernels.triton import (
        gemm_mxfp_packed_activation_reference,
        gemm_mxfp_packed_activation_triton,
        gemm_mxfp_reference,
        gemm_mxfp_triton,
        gemm_nvfp4_packed_activation_reference,
        gemm_nvfp4_packed_activation_triton,
        gemm_nvfp4_packed_dequant_reference,
        gemm_nvfp4_packed_dequant_triton,
        scaled_mxfp4_quant_triton,
        scaled_nvfp4_quant_triton,
    )

    return {
        "scaled_nvfp4_quant_triton": scaled_nvfp4_quant_triton,
        "scaled_mxfp4_quant_triton": scaled_mxfp4_quant_triton,
        "mxfp4_packed_activation_reference": gemm_mxfp_packed_activation_reference,
        "mxfp4_packed_activation_triton": gemm_mxfp_packed_activation_triton,
        "mxfp4_gemm_reference": gemm_mxfp_reference,
        "mxfp4_gemm_triton": gemm_mxfp_triton,
        "nvfp4_packed_activation_reference": gemm_nvfp4_packed_activation_reference,
        "nvfp4_packed_activation_triton": gemm_nvfp4_packed_activation_triton,
        "nvfp4_gemm_reference": gemm_nvfp4_packed_dequant_reference,
        "nvfp4_gemm_triton": gemm_nvfp4_packed_dequant_triton,
    }


def _reference_fp4_api() -> dict[str, Any]:
    from xqt.operator_opt.kernels.fp4_quant_common import (
        scaled_mxfp4_quant_reference,
        scaled_nvfp4_quant_reference,
    )
    from xqt.operator_opt.kernels.tilelang import (
        mxfp4_packed_activation_gemm_epilogue_reference,
        mxfp4_packed_dequant_gemm_epilogue_reference,
        nvfp4_packed_activation_gemm_epilogue_reference,
        nvfp4_packed_dequant_gemm_epilogue_reference,
    )

    return {
        "scaled_nvfp4_quant_reference": scaled_nvfp4_quant_reference,
        "scaled_mxfp4_quant_reference": scaled_mxfp4_quant_reference,
        "mxfp4_packed_activation_reference": mxfp4_packed_activation_gemm_epilogue_reference,
        "mxfp4_gemm_reference": mxfp4_packed_dequant_gemm_epilogue_reference,
        "nvfp4_packed_activation_reference": nvfp4_packed_activation_gemm_epilogue_reference,
        "nvfp4_gemm_reference": nvfp4_packed_dequant_gemm_epilogue_reference,
    }


def _mxfp_weight_only_api() -> dict[str, Any]:
    from .mxfp_weight_only import _pack_mxfp_blocks, _unpack_mxfp_blocks

    return {
        "pack_mxfp_blocks": _pack_mxfp_blocks,
        "unpack_mxfp_blocks": _unpack_mxfp_blocks,
    }


def _resolve_target_arch(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _resolve_engine(engine: str, inputs: torch.Tensor) -> str:
    normalized = str(engine).strip().lower()
    if normalized not in _VALID_ENGINES:
        allowed = ", ".join(sorted(_VALID_ENGINES))
        raise ValueError(f"engine must be one of {allowed}")
    if normalized == "auto":
        return "tilelang" if inputs.is_cuda else "torch"
    return normalized


def _engine_candidates(requested_engine: str, inputs: torch.Tensor) -> tuple[str, ...]:
    normalized = _resolve_engine(requested_engine, inputs)
    if normalized == "torch" or not inputs.is_cuda:
        return ("torch",)
    preferred = [] if normalized == "auto" else [normalized]
    resolved = resolve_engine(
        required_capabilities=["fp4_mma"],
        preferred_engines=preferred,
        fallback="torch",
    )
    supported = tuple(
        candidate
        for candidate in resolved.candidates
        if candidate in {"tilelang", "triton"}
    )
    if not supported:
        return ("torch",)
    return tuple(dict.fromkeys((*supported, "torch")))


def _normalize_fp4_format(format_name: str) -> str:
    normalized = str(format_name).strip().lower()
    aliases = {"fp4": "nvfp4", "mxfp": "mxfp4"}
    resolved = aliases.get(normalized, normalized)
    if resolved not in _VALID_FORMATS:
        allowed = ", ".join(sorted(_VALID_FORMATS))
        raise ValueError(f"fp4 format must be one of {allowed}")
    return resolved


def _nvfp4_global_scale_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    max_abs = tensor.detach().to(torch.float32).abs().amax().clamp_min(1e-8)
    return torch.tensor(
        [_NVFP4_GLOBAL_SCALE_RANGE / float(max_abs.item())],
        dtype=torch.float32,
        device=tensor.device,
    )


def _runtime_fallback_record(
    *,
    engine: str,
    stage: str,
    error: Exception,
) -> dict[str, str]:
    message = str(error).strip()
    detail = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return {
        "engine": str(engine),
        "stage": str(stage),
        "error_type": type(error).__name__,
        "reason": f"{engine} {stage} runtime fallback: {detail}",
    }


def _execution_mode_for_engine(engine: str, inputs: torch.Tensor) -> str:
    if engine == "tilelang" and inputs.is_cuda:
        return "cuda_tilelang_entry"
    if engine == "triton" and inputs.is_cuda:
        return "cuda_triton_entry"
    return "reference_fallback"


@dataclass
class FP4DynamicQuantizationResult(QuantizedModel):
    """Result returned by the dynamic FP4 runtime quantization helpers."""

    backend: str = "pytorch"
    strategy: str = "w4a4_nvfp4"


class FP4DynamicLinear(nn.Module):
    """Linear module backed by packed FP4 weight storage and dynamic FP4 activations."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        fp4_format: str,
        engine: str = "auto",
        weight_global_scale: torch.Tensor | None = None,
        group_size: int | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.fp4_format = _normalize_fp4_format(fp4_format)
        self.engine = str(engine).strip().lower()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size or (16 if self.fp4_format == "nvfp4" else 32))
        self.padded_input_features = int(packed_weight.shape[1]) * 2
        self.output_dtype = output_dtype
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self._dense_weight_cache: dict[tuple[str, str], torch.Tensor] = {}
        packed_weight = packed_weight.to(torch.uint8).contiguous()
        weight_scale = weight_scale.to(torch.float32).contiguous()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        tiled_weight: torch.Tensor | None = None
        tiled_scale: torch.Tensor | None = None
        if (
            self.fp4_format == "nvfp4"
            and self.output_features % TILELANG_NVFP4_WEIGHT_BLOCK_N == 0
            and self.padded_input_features % TILELANG_NVFP4_WEIGHT_BLOCK_K == 0
        ):
            tiled_weight, tiled_scale = prepack_nvfp4_weight_for_tilelang(
                packed_weight,
                weight_scale,
                group_size=self.group_size,
                block_n=TILELANG_NVFP4_WEIGHT_BLOCK_N,
                block_k=TILELANG_NVFP4_WEIGHT_BLOCK_K,
            )
        self.register_buffer("tilelang_packed_weight", tiled_weight)
        self.register_buffer("tilelang_weight_scale", tiled_scale)
        if weight_global_scale is None:
            self.register_buffer("weight_global_scale", None)
        else:
            self.register_buffer(
                "weight_global_scale",
                weight_global_scale.detach().to(torch.float32).reshape(()),
            )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())

    def _apply(self, fn: Any) -> "FP4DynamicLinear":
        nn.Module._apply(self, fn)
        self._dense_weight_cache.clear()
        if self.weight_global_scale is not None:
            self.weight_global_scale = self.weight_global_scale.to(dtype=torch.float32)
        if self.bias is not None:
            self.bias = self.bias.to(dtype=torch.float32)
        self.weight_scale = self.weight_scale.to(dtype=torch.float32)
        if self.tilelang_weight_scale is not None:
            self.tilelang_weight_scale = self.tilelang_weight_scale.to(dtype=torch.float32)
        return self

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        fp4_format: str,
        engine: str = "auto",
    ) -> "FP4DynamicLinear":
        format_name = _normalize_fp4_format(fp4_format)
        weight = module.weight.detach().to(torch.float32)
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        ref_api = _reference_fp4_api()
        if format_name == "nvfp4":
            weight_global_scale = _nvfp4_global_scale_from_tensor(weight)
            packed_weight, weight_scale = ref_api["scaled_nvfp4_quant_reference"](
                weight,
                weight_global_scale,
                group_size=16,
            )
            return cls(
                packed_weight,
                weight_scale,
                bias=bias,
                input_features=module.in_features,
                output_features=module.out_features,
                fp4_format=format_name,
                engine=engine,
                weight_global_scale=weight_global_scale,
                group_size=16,
                output_dtype=module.weight.dtype,
            )
        mxfp_api = _mxfp_weight_only_api()
        packed_weight, weight_scale, _ = mxfp_api["pack_mxfp_blocks"](
            weight,
            precision=4,
            block_size=32,
        )
        return cls(
            packed_weight,
            weight_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            fp4_format=format_name,
            engine=engine,
            weight_global_scale=None,
            group_size=32,
            output_dtype=module.weight.dtype,
        )

    def dense_weight(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if not _can_mutate_runtime_cache():
            return self.dequantize_weight().to(device=device, dtype=dtype).detach()
        key = (str(device), str(dtype))
        cached = self._dense_weight_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        weight = self.dequantize_weight().to(device=device, dtype=dtype).detach()
        self._dense_weight_cache[key] = weight
        return weight

    def dense_bias(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.bias is None:
            return None
        return self.bias.to(device=device, dtype=dtype)

    def dequantize_weight(self) -> torch.Tensor:
        if self.fp4_format == "mxfp4":
            mxfp_api = _mxfp_weight_only_api()
            unpacked = mxfp_api["unpack_mxfp_blocks"](
                self.packed_weight,
                self.weight_scale,
                precision=4,
                block_size=self.group_size,
                padded_input_features=self.padded_input_features,
            )
            return unpacked[:, : self.input_features]
        return dequantize_nvfp4_codes(
            self.packed_weight,
            self.weight_scale,
            input_features=self.input_features,
            group_size=self.group_size,
            global_scale=self.weight_global_scale,
            output_dtype=torch.float32,
        )

    def _quantize_activation(
        self,
        flat_inputs: torch.Tensor,
        *,
        selected_engine: str,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self.fp4_format == "nvfp4":
            global_scale = _nvfp4_global_scale_from_tensor(flat_inputs)
            if selected_engine == "tilelang" and flat_inputs.is_cuda:
                api = _tilelang_fp4_api()
                packed, scale = api["scaled_nvfp4_quant_tilelang"](
                    flat_inputs,
                    global_scale,
                    group_size=self.group_size,
                    target_arch=_resolve_target_arch(flat_inputs.device),
                )
                return packed, scale, "tilelang_nvfp4_quant"
            if selected_engine == "triton" and flat_inputs.is_cuda:
                api = _triton_fp4_api()
                packed, scale = api["scaled_nvfp4_quant_triton"](
                    flat_inputs,
                    global_scale,
                    group_size=self.group_size,
                )
                return packed, scale, "triton_nvfp4_quant"
            api = _reference_fp4_api()
            packed, scale = api["scaled_nvfp4_quant_reference"](
                flat_inputs,
                global_scale,
                group_size=self.group_size,
            )
            return packed, scale, "reference_nvfp4_quant"
        if selected_engine == "tilelang" and flat_inputs.is_cuda:
            api = _tilelang_fp4_api()
            packed, scale = api["scaled_mxfp4_quant_tilelang"](
                flat_inputs,
                group_size=self.group_size,
                target_arch=_resolve_target_arch(flat_inputs.device),
            )
            return packed, scale, "tilelang_mxfp4_quant"
        if selected_engine == "triton" and flat_inputs.is_cuda:
            api = _triton_fp4_api()
            packed, scale = api["scaled_mxfp4_quant_triton"](
                flat_inputs,
                group_size=self.group_size,
            )
            return packed, scale, "triton_mxfp4_quant"
        api = _reference_fp4_api()
        packed, scale = api["scaled_mxfp4_quant_reference"](
            flat_inputs,
            group_size=self.group_size,
        )
        return packed, scale, "reference_mxfp4_quant"

    def _run_packed_gemm(
        self,
        packed_activation: torch.Tensor,
        activation_scale: torch.Tensor,
        *,
        selected_engine: str,
        activation_global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, str, str]:
        if self.fp4_format == "nvfp4":
            if selected_engine == "tilelang" and packed_activation.is_cuda:
                if (
                    self.tilelang_packed_weight is None
                    or self.tilelang_weight_scale is None
                ):
                    raise XQTBackendError(
                        "TileLang NVFP4 CTA-tiled prepack is unavailable for this weight shape"
                    )
                api = _tilelang_fp4_api()
                output = api["nvfp4_packed_activation_tilelang"](
                    packed_activation,
                    activation_scale,
                    self.tilelang_packed_weight.to(device=packed_activation.device),
                    self.tilelang_weight_scale.to(
                        device=packed_activation.device,
                        dtype=activation_scale.dtype,
                    ),
                    None
                    if self.bias is None
                    else self.bias.to(
                        device=packed_activation.device,
                        dtype=activation_scale.dtype,
                    ),
                    input_features=self.input_features,
                    group_size=self.group_size,
                    activation_global_scale=activation_global_scale,
                    weight_global_scale=(
                        None
                        if self.weight_global_scale is None
                        else self.weight_global_scale.to(device=packed_activation.device).reshape(1)
                    ),
                    target_arch=_resolve_target_arch(packed_activation.device),
                    weight_is_prepacked=True,
                )
                return (
                    output,
                    "tilelang_prepacked_nvfp4_weight_gemm",
                    "packed_activation_api",
                )
            if selected_engine == "triton" and packed_activation.is_cuda:
                api = _triton_fp4_api()
                output = api["nvfp4_packed_activation_triton"](
                    packed_activation,
                    activation_scale,
                    self.packed_weight.to(device=packed_activation.device),
                    self.weight_scale.to(device=packed_activation.device, dtype=torch.float32),
                    None if self.bias is None else self.bias.to(device=packed_activation.device, dtype=torch.float32),
                    input_features=self.input_features,
                    group_size=self.group_size,
                    activation_global_scale=activation_global_scale,
                    weight_global_scale=(
                        None
                        if self.weight_global_scale is None
                        else self.weight_global_scale.to(device=packed_activation.device, dtype=torch.float32).reshape(1)
                    ),
                )
                return output, "triton_packed_nvfp4_weight_gemm", "packed_activation_api"
            api = _reference_fp4_api()
            output = api["nvfp4_packed_activation_reference"](
                packed_activation,
                activation_scale,
                self.packed_weight.to(device=packed_activation.device),
                self.weight_scale.to(device=packed_activation.device),
                None
                if self.bias is None
                else self.bias.to(
                    device=packed_activation.device,
                    dtype=torch.float32,
                ),
                input_features=self.input_features,
                group_size=self.group_size,
                activation_global_scale=activation_global_scale,
                weight_global_scale=(
                    None
                    if self.weight_global_scale is None
                    else self.weight_global_scale.to(device=packed_activation.device).reshape(1)
                ),
            )
            return output, "reference_packed_nvfp4_weight_gemm", "packed_activation_api"
        if selected_engine == "tilelang" and packed_activation.is_cuda:
            api = _tilelang_fp4_api()
            output = api["mxfp4_packed_activation_tilelang"](
                packed_activation,
                activation_scale,
                self.packed_weight.to(device=packed_activation.device),
                self.weight_scale.to(device=packed_activation.device),
                None if self.bias is None else self.bias.to(device=packed_activation.device),
                input_features=self.input_features,
                group_size=self.group_size,
                target_arch=_resolve_target_arch(packed_activation.device),
            )
            return output, "tilelang_packed_mxfp4_weight_gemm", "packed_activation_api"
        if selected_engine == "triton" and packed_activation.is_cuda:
            api = _triton_fp4_api()
            output = api["mxfp4_packed_activation_triton"](
                packed_activation,
                activation_scale,
                self.packed_weight.to(device=packed_activation.device),
                self.weight_scale.to(
                    device=packed_activation.device,
                    dtype=torch.float32,
                ),
                None
                if self.bias is None
                else self.bias.to(
                    device=packed_activation.device,
                    dtype=torch.float32,
                ),
                input_features=self.input_features,
                mx_precision=4,
                block_size=self.group_size,
            )
            return output, "triton_packed_mxfp4_weight_gemm", "packed_activation_api"
        api = _reference_fp4_api()
        output = api["mxfp4_packed_activation_reference"](
            packed_activation,
            activation_scale,
            self.packed_weight.to(device=packed_activation.device),
            self.weight_scale.to(device=packed_activation.device),
            None
            if self.bias is None
            else self.bias.to(
                device=packed_activation.device,
                dtype=torch.float32,
            ),
            input_features=self.input_features,
            group_size=self.group_size,
        )
        return output, "reference_packed_mxfp4_weight_gemm", "packed_activation_api"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "FP4DynamicLinear input trailing dimension does not match input_features"
            )
        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat_inputs = inputs.reshape(-1, self.input_features)
        activation_global_scale = (
            _nvfp4_global_scale_from_tensor(flat_inputs)
            if self.fp4_format == "nvfp4"
            else None
        )
        requested_engine = self.engine
        engine_candidates = _engine_candidates(requested_engine, flat_inputs)
        fallback_records: list[dict[str, str]] = []
        for selected_engine in engine_candidates:
            active_stage = "activation_quant"
            try:
                packed_activation, activation_scale, activation_quant_engine = (
                    self._quantize_activation(
                        flat_inputs,
                        selected_engine=selected_engine,
                    )
                )
                active_stage = "weight_gemm"
                output, weight_gemm_engine, weight_gemm_entry = self._run_packed_gemm(
                    packed_activation,
                    activation_scale,
                    selected_engine=selected_engine,
                    activation_global_scale=activation_global_scale,
                )
            except Exception as exc:
                failure = _runtime_fallback_record(
                    engine=selected_engine,
                    stage=active_stage,
                    error=exc,
                )
                if selected_engine != engine_candidates[-1]:
                    fallback_records.append(failure)
                    continue
                if fallback_records:
                    attempted = ", ".join(
                        f"{record['engine']}@{record['stage']}" for record in fallback_records
                    )
                    raise XQTBackendError(
                        "FP4DynamicLinear exhausted runtime fallbacks after "
                        f"{attempted}; final failure was {failure['reason']}"
                    ) from exc
                raise
            self.last_execution = {
                "engine": selected_engine,
                "requested_engine": requested_engine,
                "engine_candidates": list(engine_candidates),
                "execution_mode": _execution_mode_for_engine(selected_engine, flat_inputs),
                "execution_reason": (
                    fallback_records[-1]["reason"] if fallback_records else None
                ),
                "fp4_format": self.fp4_format,
                "activation_quant_engine": activation_quant_engine,
                "weight_gemm_engine": weight_gemm_engine,
                "weight_gemm_entry": weight_gemm_entry,
                "activation_global_scale": (
                    None
                    if activation_global_scale is None
                    else float(activation_global_scale.detach().cpu().item())
                ),
                "activation_storage": "packed_fp4_uint8",
                "weight_storage": (
                    "packed_nvfp4_e2m1_uint8"
                    if self.fp4_format == "nvfp4"
                    else "packed_mxfp4_int4_uint8"
                ),
                "tilelang_weight_layout": (
                    "cta_tiled_n16_k128"
                    if self.tilelang_packed_weight is not None
                    else None
                ),
                "quantization_nature": "pseudo",
                "consumes_packed_weight": True,
                "input_features": self.input_features,
                "output_features": self.output_features,
                "group_size": self.group_size,
                "padded_input_features": self.padded_input_features,
                "fallback_count": len(fallback_records),
                "fallback_reason": (
                    fallback_records[-1]["reason"] if fallback_records else None
                ),
                "runtime_fallbacks": list(fallback_records),
            }
            return output.to(self.output_dtype).reshape(*original_shape, self.output_features)
        raise RuntimeError("unreachable")

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self.last_execution)


def _dynamic_fp4_strategy_for_format(fp4_format: str) -> str:
    return "w4a4_nvfp4" if _normalize_fp4_format(fp4_format) == "nvfp4" else "w4a4_mxfp4"


def quantize_with_dynamic_fp4(
    model: nn.Module,
    *,
    fp4_format: str,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
    engine: str = "auto",
) -> FP4DynamicQuantizationResult:
    """Replace Linear modules with dynamic FP4 runtime modules."""

    format_name = _normalize_fp4_format(fp4_format)
    quant_policy = (
        policy if isinstance(policy, QuantizationPolicy) else _policy_from_mapping(policy or {})
    )
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []
    for name, module in list(target_model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue
        _replace_submodule(
            target_model,
            name,
            FP4DynamicLinear.from_linear(
                module,
                fp4_format=format_name,
                engine=engine,
            ),
        )
        quantized_modules.append(name)
    selected_strategy = (
        normalize_quant_strategy(
            strategy,
            {
                "dtype": format_name,
                "scheme": "dynamic",
                "engine": engine,
            },
        )
        or _dynamic_fp4_strategy_for_format(format_name)
    )
    preferred_engines = [] if engine == "auto" else [str(engine)]
    compute_config = ComputeConfig.from_modules(
        module_names=quantized_modules,
        compute_contract="fp4_mma",
        precision="w4a4",
        required_capabilities=["fp4_mma"],
        preferred_engines=preferred_engines,
        default_precision="w4a4",
        storage={
            "format": format_name,
            "layout": "packed_weight",
        },
        metadata={
            "engine": engine,
            "activation_quantization": "dynamic_fp4",
        },
    )
    from xqt.quant.layout_apply_report import (
        attach_layout_kernel_metadata,
        layout_report_for_fp4_dynamic,
    )

    layout = layout_report_for_fp4_dynamic(
        target_model,
        engine_preference=str(engine),
        fp4_format=format_name,
    )
    meta = {
        "implementation": f"{format_name}_dynamic_activation_fp4_linear",
        "quantization_nature": "pseudo",
        "quantization_nature_scope": "current_xqt_runtime_implementation",
        "activation_encoding": f"dynamic_{format_name}_packed_fp4",
        "weight_encoding": (
            "packed_nvfp4_e2m1_plus_fp8_scale"
            if format_name == "nvfp4"
            else "packed_mxfp4_int4_plus_block_scale"
        ),
        "precision_description": {
            "quantization_time": {
                "weight": f"offline packed {format_name} weight with scales",
                "activation": "not stored; quantized to packed FP4 for every forward",
            },
            "runtime": {
                "operand_contract": "packed W4A4 FP4 inputs and weights",
                "compute": (
                    "current XQT path is PSEUDO: packed dequant or reference "
                    "GEMM, not a claimed native FP4 MMA result"
                ),
                "actual_execution_source": "FP4DynamicLinear.execution_metadata",
                "engine_fallback": (
                    "auto or an unavailable requested engine tries its ordered "
                    "runtime candidates; runtime_fallbacks records each taken fallback"
                ),
            },
        },
        "engine_preference": str(engine),
        "preferred_engines": preferred_engines,
        "compute_config": compute_config.to_dict(),
        "policy": {
            "dtype": quant_policy.dtype,
            "scheme": quant_policy.scheme,
            "include_module_types": list(quant_policy.include_module_types),
            "exclude_module_types": list(quant_policy.exclude_module_types),
            "include_name_patterns": list(quant_policy.include_name_patterns),
            "exclude_name_patterns": list(quant_policy.exclude_name_patterns),
            "include_module_names": list(quant_policy.include_module_names),
            "exclude_module_names": list(quant_policy.exclude_module_names),
            "min_parameters": quant_policy.min_parameters,
            "engine": str(engine),
            "fp4_format": format_name,
        },
    }
    return FP4DynamicQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        compute_config=compute_config,
        metadata=attach_layout_kernel_metadata(meta, layout),
    )


def quantize_with_nvfp4_dynamic(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
    engine: str = "auto",
) -> FP4DynamicQuantizationResult:
    return quantize_with_dynamic_fp4(
        model,
        fp4_format="nvfp4",
        policy=policy,
        strategy=strategy,
        inplace=inplace,
        engine=engine,
    )


def quantize_with_mxfp4_dynamic(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
    engine: str = "auto",
) -> FP4DynamicQuantizationResult:
    return quantize_with_dynamic_fp4(
        model,
        fp4_format="mxfp4",
        policy=policy,
        strategy=strategy,
        inplace=inplace,
        engine=engine,
    )


def execute_dynamic_fp4_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    fp4_format: str,
    quantize_fn: Any,
) -> tuple[nn.Module, QuantizationReport]:
    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
        engine=str(component.policy.get("engine", "auto")),
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    method_semantics = f"dynamic_{_normalize_fp4_format(fp4_format)}_activation_with_packed_weight_runtime"
    report = build_component_quantization_report(
        context,
        component,
        backend=result.backend,
        strategy=result.strategy,
        quantized_modules=result.quantized_modules,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=True,
        method_semantics=method_semantics,
        effective_policy=effective_policy,
        result_metadata=result.metadata,
        execution_state=result.strategy,
    )
    return updated_model, report


__all__ = [
    "FP4DynamicLinear",
    "FP4DynamicQuantizationResult",
    "execute_dynamic_fp4_component",
    "quantize_with_dynamic_fp4",
    "quantize_with_mxfp4_dynamic",
    "quantize_with_nvfp4_dynamic",
]
