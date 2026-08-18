"""Runtime W8A8 INT8 MMA Linear module (Infer-facing)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.int8_mma import Int8MmaLinear as Int8MmaStorageLinear
from xqt.contracts.engine_resolve import (
    normalize_engine_name,
    resolve_int8_mma_engine,
)
from xqt.core.errors import XQTBackendError

_PTX_SM89_ENGINES = frozenset({"ptx_sm89", "native_sm89"})
_CUDA_SM89_ENGINES = frozenset({"cuda_sm89"})
_VALID_ENGINES = (
    frozenset({"auto", "tilelang", "triton", "torch_int_mm"})
    | _PTX_SM89_ENGINES
    | _CUDA_SM89_ENGINES
)
_ACTIVATION_SCALE_MODES = {"dynamic", "static"}
_OUTPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

def _tilelang_int8_api():
    """Lazy import TileLang INT8 kernels (not at quantizer module import time)."""
    from xqt.operator_opt.kernels.tilelang.int8_mma import (
        int8_linear_static_activation_m1_tilelang,
        int8_linear_static_activation_tilelang,
        int8_linear_tilelang,
        int8_mma_reference,
        pad_rows_to_block,
        static_activation_quantize_tilelang,
    )

    return {
        "int8_linear_tilelang": int8_linear_tilelang,
        "int8_linear_static_activation_tilelang": int8_linear_static_activation_tilelang,
        "int8_linear_static_activation_m1_tilelang": int8_linear_static_activation_m1_tilelang,
        "int8_mma_reference": int8_mma_reference,
        "pad_rows_to_block": pad_rows_to_block,
        "static_activation_quantize_tilelang": static_activation_quantize_tilelang,
    }


def _triton_int8_api():
    """Lazy import Triton INT8 GEMM kernels."""
    from xqt.operator_opt.kernels.triton.gemm import (
        gemm_int8_triton,
        quantize_int8_rowwise_triton,
    )

    return {
        "gemm_int8_triton": gemm_int8_triton,
        "quantize_int8_rowwise_triton": quantize_int8_rowwise_triton,
    }

def _target_arch_from_device(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"

class Int8MmaLinear(Int8MmaStorageLinear):
    """W8A8 INT8 Linear replacement that records the realized runtime path."""

    quant_max: float = 127.0

    def __init__(
        self,
        qweight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        engine: str = "auto",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        output_dtype: torch.dtype = torch.float32,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        preferred_engines: list[str] | tuple[str, ...] | None = None,
        min_int8_rows: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        normalized_engine = normalize_engine_name(engine)
        if normalized_engine == "native_sm89":
            normalized_engine = "ptx_sm89"
        if normalized_engine not in _VALID_ENGINES:
            raise ValueError(
                "engine must be one of auto, triton, tilelang, torch_int_mm, ptx_sm89, cuda_sm89"
            )
        self.preferred_engines = [
            normalize_engine_name(item)
            for item in (preferred_engines or [])
            if normalize_engine_name(item) != "auto"
        ]
        if str(fallback_engine) not in {"torch_int_mm", "reference"}:
            raise ValueError("fallback_engine must be torch_int_mm or reference")
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.engine = normalized_engine
        self.fallback_engine = str(fallback_engine)
        if not hasattr(self, "preferred_engines"):
            self.preferred_engines = []
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.threads = int(threads)
        self.num_stages = int(num_stages)
        if output_dtype not in _OUTPUT_DTYPES:
            raise ValueError("output_dtype must be float16, bfloat16, or float32")
        self.output_dtype = output_dtype
        if str(activation_scale_mode) not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
        self.activation_scale_mode = str(activation_scale_mode)
        self.activation_quant_block_size = int(activation_quant_block_size)
        self.eps = float(eps)
        if int(min_int8_rows) < 0:
            raise ValueError("min_int8_rows must be >= 0")
        self.min_int8_rows = int(min_int8_rows)
        # Lazily cached dense bf16-domain weight for the small-M fallback path.
        self._bf16_weight: torch.Tensor | None = None
        self._auto_engine_cache: dict[tuple[str, int | None, int, str, bool, bool], str] = {}
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("qweight_t", qweight_t.to(torch.int8).contiguous())
        self.register_buffer("weight_scale", weight_scale.to(torch.float32).contiguous())
        self._qweight_prepacked_b: torch.Tensor | None = None
        self._qweight_prepacked_b_version: int | None = None
        self._cuda_sm89_scale_bias_cache: torch.Tensor | None = None
        self._cuda_sm89_scale_bias_cache_key: tuple[int, int | None, int, torch.device] | None = None
        if normalized_engine in {"ptx_sm89", "cuda_sm89"}:
            self._ensure_ptx_prepacked_b()
        initial_activation_scale = torch.tensor(0.0, dtype=torch.float32)
        self._has_static_activation_scale = activation_scale is not None
        if activation_scale is not None:
            initial_activation_scale = torch.as_tensor(activation_scale, dtype=torch.float32).reshape(())
        self.register_buffer("static_activation_scale", initial_activation_scale)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())

    @classmethod
    def from_storage(
        cls,
        module: Int8MmaStorageLinear,
        *,
        engine: str | None = None,
        fallback_engine: str | None = None,
    ) -> "Int8MmaLinear":
        """Materialize a backend execution view from a contracts storage shell."""

        activation_scale = (
            module.static_activation_scale.detach()
            if module._has_static_activation_scale
            else None
        )
        return cls(
            module.qweight_t.detach(),
            module.weight_scale.detach(),
            bias=None if module.bias is None else module.bias.detach(),
            input_features=module.input_features,
            output_features=module.output_features,
            engine=engine or module.engine,
            fallback_engine=fallback_engine or module.fallback_engine,
            block_m=module.block_m,
            block_n=module.block_n,
            block_k=module.block_k,
            threads=module.threads,
            num_stages=module.num_stages,
            output_dtype=module.output_dtype,
            activation_scale_mode=module.activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=module.activation_quant_block_size,
            eps=module.eps,
            preferred_engines=module.preferred_engines,
            min_int8_rows=module.min_int8_rows,
        )

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        engine: str = "auto",
        fallback_engine: str = "torch_int_mm",
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        threads: int = 128,
        num_stages: int = 2,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
        activation_quant_block_size: int = 256,
        eps: float = 1e-6,
        preferred_engines: list[str] | tuple[str, ...] | None = None,
        min_int8_rows: int = 0,
    ) -> "Int8MmaLinear":
        weight = module.weight.detach().to(torch.float32)
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = torch.where(
            max_abs > float(eps),
            max_abs / 127.0,
            torch.ones_like(max_abs),
        )
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            qweight.t().contiguous(),
            scale.reshape(-1),
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=module.weight.dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            preferred_engines=preferred_engines,
            min_int8_rows=min_int8_rows,
        )

    def _dense_bf16_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Reconstruct the dense weight (out, in) from INT8 storage for fallback.

        Cached because decode repeatedly hits the same layer; caching trades an
        extra dense-weight copy (~1x source precision) for avoiding a re-dequant
        every forward. Only reached when ``min_int8_rows`` routing is enabled.
        """

        cached = self._bf16_weight
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        # qweight_t is (in_features, out_features) int8; weight_scale is per-out-channel.
        weight = (
            self.qweight_t.to(torch.float32).t()
            * self.weight_scale.to(torch.float32).reshape(-1, 1)
        ).to(device=device, dtype=dtype)
        self._bf16_weight = weight
        return weight

    def _run_bf16_fallback(self, inputs: torch.Tensor, flat_rows: int) -> torch.Tensor:
        """Dense GEMM in a float domain; used when rows < ``min_int8_rows``."""

        compute_dtype = inputs.dtype if inputs.dtype in _OUTPUT_DTYPES else torch.float32
        weight = self._dense_bf16_weight(compute_dtype, inputs.device)
        bias = (
            None
            if self.bias is None
            else self.bias.to(device=inputs.device, dtype=compute_dtype)
        )
        output = F.linear(inputs.to(compute_dtype), weight, bias)
        self.last_execution = {
            "engine": "bf16_fallback",
            "reason": "rows_below_min_int8_rows",
            "true_int8_mma": False,
            "min_int8_rows": self.min_int8_rows,
            "input_rows": int(flat_rows),
            "input_features": self.input_features,
            "output_features": self.output_features,
        }
        return output.to(self.output_dtype)

    def _runtime_precision_summary(self) -> dict[str, Any]:
        """Describe the operands and fallback state of the most recent forward."""

        execution = self.last_execution
        engine = str(execution.get("engine", "not_run"))
        float_fallback = engine == "bf16_fallback"
        native_mma = bool(execution.get("true_int8_mma", False))
        int8_operands = (
            not float_fallback
            and execution.get("activation_dtype") == "int8"
            and execution.get("weight_dtype") == "int8"
        )
        if engine == "not_run":
            execution_kind = "not_run"
        elif float_fallback:
            execution_kind = "floating_point_fallback"
        elif native_mma:
            execution_kind = "native_w8a8_int8_mma"
        elif int8_operands:
            execution_kind = "w8a8_int8_reference"
        else:
            execution_kind = "unclassified"

        small_batch_status = (
            "disabled"
            if self.min_int8_rows == 0
            else "taken"
            if float_fallback
            else "not_taken"
        )
        engine_error_fallback_taken = str(execution.get("reason", "")).startswith(
            "int8_mma_fallback:"
        )
        engine_error_fallback_status = (
            "disabled"
            if self.fallback_engine != "torch_int_mm"
            else "taken"
            if engine_error_fallback_taken
            else "not_taken"
        )
        activation_quant_engine = str(execution.get("activation_quant_engine", ""))
        activation_granularity = str(
            execution.get(
                "activation_granularity",
                "per_token"
                if "per_token" in activation_quant_engine
                else "per_tensor",
            )
        )
        return {
            "requested": "w8a8_int8_mma",
            "weight_storage": "signed_int8_per_output_channel",
            "activation_encoding": f"signed_int8_{activation_granularity}",
            "activation_granularity": activation_granularity,
            "execution_kind": execution_kind,
            "int8_operands_executed": int8_operands,
            "native_mma_executed": native_mma,
            "float_fallback_taken": float_fallback,
            "float_fallback_reason": execution.get("reason") if float_fallback else None,
            "small_batch_float_fallback": {
                "enabled": self.min_int8_rows > 0,
                "condition": "input_rows < min_int8_rows",
                "min_int8_rows": self.min_int8_rows,
                "status": small_batch_status,
            },
            "engine_error_int8_fallback": {
                "enabled": self.fallback_engine == "torch_int_mm",
                "condition": "selected non-reference INT8 engine raises or fails its runtime check",
                "status": engine_error_fallback_status,
            },
        }

    @staticmethod
    def activation_scale_from_inputs(inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Return the symmetric per-tensor INT8 activation scale for sample inputs."""

        max_abs = inputs.detach().reshape(-1).to(torch.float32).abs().amax().clamp_min(float(eps))
        return (max_abs / 127.0).reshape(())

    def set_static_activation_scale(self, scale: torch.Tensor | float) -> None:
        """Set a calibrated static activation scale used by the static fast path."""

        value = torch.as_tensor(scale, dtype=torch.float32, device=self.static_activation_scale.device).reshape(())
        if float(value.detach().cpu().item()) <= 0.0:
            raise ValueError("static activation scale must be positive")
        self.static_activation_scale.copy_(value)
        self._has_static_activation_scale = True
        self.activation_scale_mode = "static"
        self._cuda_sm89_scale_bias_cache = None
        self._cuda_sm89_scale_bias_cache_key = None

    def calibrate_static_activation_scale(self, sample_inputs: torch.Tensor) -> torch.Tensor:
        """Calibrate this layer's static activation scale from representative inputs."""

        scale = self.activation_scale_from_inputs(sample_inputs, eps=self.eps).to(
            device=self.static_activation_scale.device,
            dtype=torch.float32,
        )
        self.set_static_activation_scale(scale)
        return self.static_activation_scale.detach().clone()

    def _static_activation_scale(self, device: torch.device) -> torch.Tensor:
        if not self._has_static_activation_scale:
            raise XQTBackendError(
                "static activation scale mode requires calibrate_static_activation_scale() "
                "or set_static_activation_scale() before forward"
            )
        return self.static_activation_scale.to(device=device, dtype=torch.float32)

    def _quantize_activation(
        self,
        inputs: torch.Tensor,
        *,
        prefer_tilelang: bool = False,
        prefer_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        flat_input = inputs.reshape(-1, self.input_features)
        if self.activation_scale_mode == "static":
            scale = self._static_activation_scale(flat_input.device)
            if prefer_tilelang and flat_input.is_cuda and inputs.dtype in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }:
                api = _tilelang_int8_api()
                qactivation = api["static_activation_quantize_tilelang"](
                    flat_input,
                    scale,
                    block_size=self.activation_quant_block_size,
                    target_arch=_target_arch_from_device(flat_input.device),
                )
                return qactivation.contiguous(), scale, "tilelang_static"
        else:
            if (
                prefer_triton
                and flat_input.is_cuda
                and inputs.dtype
                in {torch.float16, torch.bfloat16, torch.float32}
            ):
                api = _triton_int8_api()
                qactivation, scale = api["quantize_int8_rowwise_triton"](
                    flat_input.contiguous()
                )
                return qactivation, scale, "triton_dynamic_per_token"
            flat = flat_input.to(torch.float32)
            scale = self.activation_scale_from_inputs(flat, eps=self.eps).to(device=flat.device)
            qactivation = torch.round(flat / scale).clamp(-127, 127).to(torch.int8)
            return qactivation.contiguous(), scale, f"torch_{self.activation_scale_mode}"
        qactivation = torch.round(flat_input.to(torch.float32) / scale).clamp(-127, 127).to(torch.int8)
        return qactivation.contiguous(), scale, f"torch_{self.activation_scale_mode}"

    def _can_use_tilelang(self, qactivation: torch.Tensor) -> tuple[bool, str]:
        if not qactivation.is_cuda or not self.qweight_t.is_cuda:
            return False, "TileLang INT8 MMA requires CUDA tensors"
        if self.output_features % self.block_n != 0:
            return False, "output_features is not block_n aligned"
        if self.input_features % self.block_k != 0:
            return False, "input_features is not block_k aligned"
        return True, "ok"

    def _run_torch_int_mm(self, qactivation: torch.Tensor) -> torch.Tensor:
        int_mm = getattr(torch, "_int_mm", None)
        if qactivation.is_cuda and self.qweight_t.is_cuda and callable(int_mm):
            return int_mm(qactivation, self.qweight_t)
        api = _tilelang_int8_api()
        return api["int8_mma_reference"](qactivation, self.qweight_t)

    def _can_use_ptx_sm89(self, qactivation: torch.Tensor) -> tuple[bool, str]:
        if not qactivation.is_cuda or not self.qweight_t.is_cuda:
            return False, "ptx_sm89 INT8 MMA requires CUDA tensors"
        major, minor = torch.cuda.get_device_capability(qactivation.device)
        if (major, minor) != (8, 9):
            return False, f"ptx_sm89 requires sm_89, got sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.cute.int8mma_binding import int8mma_available
        except Exception as exc:
            return False, f"ptx_sm89 import failed: {exc}"
        if not int8mma_available():
            return False, "ptx_sm89 shared library not built"
        return True, "ok"

    def _can_use_cuda_sm89(self, qactivation: torch.Tensor) -> tuple[bool, str]:
        """Check the fixed-shape CUTLASS W8A8 fast path constraints."""

        if not qactivation.is_cuda or not self.qweight_t.is_cuda:
            return False, "cuda_sm89 W8A8 requires CUDA tensors"
        if self.output_dtype != torch.float16:
            return False, "cuda_sm89 W8A8 fast path requires float16 output"
        if self.input_features % 32 != 0:
            return False, "cuda_sm89 W8A8 requires input_features % 32 == 0"
        if self.output_features % 8 != 0:
            return False, "cuda_sm89 W8A8 requires output_features % 8 == 0"
        major, minor = torch.cuda.get_device_capability(qactivation.device)
        if (major, minor) != (8, 9):
            return False, f"cuda_sm89 W8A8 requires sm_89, got sm_{major}{minor}"
        try:
            from xqt.operator_opt.kernels.cute.int8mma_binding import int8mma_available
        except Exception as exc:
            return False, f"cuda_sm89 import failed: {exc}"
        if not int8mma_available():
            return False, "cuda_sm89 shared library not built"
        return True, "ok"

    def _ensure_ptx_prepacked_b(self) -> torch.Tensor | None:
        if self.engine not in {"ptx_sm89", "cuda_sm89", "auto"}:
            return None
        qweight_version = self.qweight_t._version
        if (
            self._qweight_prepacked_b is not None
            and self._qweight_prepacked_b_version == qweight_version
        ):
            if self._qweight_prepacked_b.device != self.qweight_t.device:
                self._qweight_prepacked_b = self._qweight_prepacked_b.to(
                    device=self.qweight_t.device
                )
            return self._qweight_prepacked_b
        try:
            from xqt.operator_opt.kernels.cute.int8mma_binding import (
                prepack_qweight_t_for_ptx_sm89,
            )
        except Exception:
            return None
        packed = prepack_qweight_t_for_ptx_sm89(self.qweight_t)
        self._qweight_prepacked_b = packed.to(device=self.qweight_t.device)
        self._qweight_prepacked_b_version = qweight_version
        return self._qweight_prepacked_b

    def _cuda_sm89_scale_bias(self, activation_scale: torch.Tensor) -> torch.Tensor:
        """Return the cached CUTLASS ``[N, 2]`` scale/bias epilogue vector."""

        cached = self._cuda_sm89_scale_bias_cache
        cache_key = (
            self.weight_scale._version,
            None if self.bias is None else self.bias._version,
            self.static_activation_scale._version,
            activation_scale.device,
        )
        if (
            self.activation_scale_mode == "static"
            and cached is not None
            and self._cuda_sm89_scale_bias_cache_key == cache_key
        ):
            return cached
        weight_scale = self.weight_scale.to(
            device=activation_scale.device,
            dtype=torch.float32,
        ).reshape(-1)
        output_scale = activation_scale.to(
            device=activation_scale.device,
            dtype=torch.float32,
        ).reshape(()) * weight_scale
        bias = (
            torch.zeros_like(output_scale)
            if self.bias is None
            else self.bias.to(device=activation_scale.device, dtype=torch.float32).reshape(-1)
        )
        scale_bias = torch.stack((output_scale, bias), dim=1).contiguous()
        if self.activation_scale_mode == "static":
            self._cuda_sm89_scale_bias_cache = scale_bias
            self._cuda_sm89_scale_bias_cache_key = cache_key
        return scale_bias

    def _run_cuda_sm89(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.int8mma_binding import int8_linear_cutlass_sm89

        prepacked = self._ensure_ptx_prepacked_b()
        return int8_linear_cutlass_sm89(
            qactivation,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=self.output_dtype,
            prepacked_b=prepacked,
            scale_bias=self._cuda_sm89_scale_bias(activation_scale),
        )

    def _run_ptx_sm89(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.int8mma_binding import int8_linear_ptx_sm89

        prepacked = self._ensure_ptx_prepacked_b()
        return int8_linear_ptx_sm89(
            qactivation,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            prepacked_b=prepacked,
        )

    def _can_use_ptx_sm89_static_fused(self, flat_inputs: torch.Tensor) -> tuple[bool, str]:
        if flat_inputs.dtype != torch.float16:
            return False, "ptx_sm89 fused static path requires float16 activations"
        return self._can_use_ptx_sm89(flat_inputs)

    def _run_ptx_sm89_static_fused(
        self,
        flat_inputs: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        from xqt.operator_opt.kernels.cute.int8mma_binding import (
            int8_linear_fused_static_ptx_sm89,
        )

        prepacked = self._ensure_ptx_prepacked_b()
        return int8_linear_fused_static_ptx_sm89(
            flat_inputs,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            prepacked_b=prepacked,
        )

    def _run_tilelang(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, str]:
        api = _tilelang_int8_api()
        padded, original_rows = api["pad_rows_to_block"](qactivation, self.block_m)
        target_arch = _target_arch_from_device(padded.device)
        output = api["int8_linear_tilelang"](
            padded,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            threads=self.threads,
            num_stages=self.num_stages,
            target_arch=target_arch,
        )
        return output[:original_rows], int(padded.shape[0]), target_arch or "auto"

    def _can_use_triton(self, qactivation: torch.Tensor) -> tuple[bool, str]:
        if not qactivation.is_cuda or not self.qweight_t.is_cuda:
            return False, "Triton INT8 GEMM requires CUDA tensors"
        return True, "ok"

    def _pick_auto_engine(
        self,
        flat_inputs: torch.Tensor,
        *,
        candidates: tuple[str, ...] | list[str],
        resolved_engine: str,
    ) -> str:
        """Pick a reachable engine for this shape/device from the resolve chain.

        Capability table order (C10) is a preference, not a hard force: if the
        head engine cannot run here (no CUDA, missing tilelang, etc.), walk the
        candidate list, then apply the historical static/sm89 specializations.
        """

        if not flat_inputs.is_cuda or not self.qweight_t.is_cuda:
            return self.fallback_engine

        selected = resolved_engine
        ordered = list(candidates) if candidates else [resolved_engine]
        if resolved_engine not in ordered:
            ordered = [resolved_engine, *ordered]

        for candidate in ordered:
            if candidate == "tilelang":
                ok, _ = self._can_use_tilelang(flat_inputs)
                if self.activation_scale_mode == "static":
                    ok, _ = self._can_use_tilelang_static_fused(flat_inputs)
                if ok:
                    selected = "tilelang"
                    break
            elif candidate == "triton":
                ok, _ = self._can_use_triton(flat_inputs)
                if ok:
                    selected = "triton"
                    break
            elif candidate == "torch_int_mm":
                selected = "torch_int_mm"
                break
            elif candidate in {"ptx_sm89", "cuda_sm89", "torch", "reference"}:
                continue

        if (
            selected == "triton"
            and int(flat_inputs.shape[0]) == 1
            and self.activation_scale_mode == "static"
        ):
            ptx_ok, _ = self._can_use_ptx_sm89(flat_inputs)
            if ptx_ok:
                selected = "ptx_sm89"
        if selected == "ptx_sm89":
            m_rows = int(flat_inputs.shape[0])
            ptx_ok, _ = self._can_use_ptx_sm89(flat_inputs)
            if not (ptx_ok and m_rows >= 192 and flat_inputs.is_cuda):
                selected = (
                    "triton"
                    if flat_inputs.is_cuda and self.qweight_t.is_cuda
                    else self.fallback_engine
                )
        if (
            selected in {"triton", "tilelang"}
            and self.activation_scale_mode == "static"
            and int(flat_inputs.shape[0]) >= 32
        ):
            cuda_ok, _ = self._can_use_cuda_sm89(flat_inputs)
            if cuda_ok:
                selected = "cuda_sm89"
        return selected

    def _run_triton(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, str]:
        api = _triton_int8_api()
        output = api["gemm_int8_triton"](
            qactivation.contiguous(),
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            transpose_b=False,
            block_m=128,
            block_n=128,
            block_k=64,
            group_m=8,
            num_warps=4,
            num_stages=3,
            output_dtype=output_dtype,
        )
        return output, int(qactivation.shape[0]), _target_arch_from_device(qactivation.device) or "auto"

    def _run_triton_static_fused(
        self,
        flat_inputs: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, str]:
        api = _tilelang_int8_api()
        qactivation = api["static_activation_quantize_tilelang"](
            flat_inputs,
            activation_scale,
            block_size=self.activation_quant_block_size,
            target_arch=_target_arch_from_device(flat_inputs.device),
        )
        return self._run_triton(qactivation, activation_scale, output_dtype)

    def run_quantized_activation(
        self,
        qactivation: torch.Tensor,
        activation_scale: torch.Tensor | float,
        *,
        activation_quant_engine: str = "prequantized",
        execution_reason: str = "true_int8_mma_prequantized_activation",
    ) -> torch.Tensor:
        """Run the INT8 GEMM backend on an already-quantized activation.

        Fused input kernels such as ConvRot Norm+Hadamard produce the INT8
        activation themselves.  Calling :meth:`forward` here would quantize
        that tensor a second time, so this narrow entry point shares only the
        GEMM/dequantization dispatch and records the fused producer.
        """

        if qactivation.ndim != 2:
            raise XQTBackendError(
                "prequantized INT8 activation must be a 2D tensor"
            )
        if qactivation.dtype != torch.int8:
            raise XQTBackendError(
                "prequantized INT8 activation must have torch.int8 dtype"
            )
        if int(qactivation.shape[1]) != self.input_features:
            raise XQTBackendError(
                "prequantized INT8 activation trailing dimension does not match "
                "input_features"
            )
        scale = torch.as_tensor(
            activation_scale,
            device=qactivation.device,
            dtype=torch.float32,
        ).reshape(())
        # This entry point is used from a CUDA fused producer.  Calling
        # ``.item()`` here would synchronize the stream on every inference
        # invocation and erase the benefit of avoiding a second quantization.
        # Public setters validate scales before they reach this path; retain
        # the eager validation for CPU/reference calls where it is free.
        if scale.device.type != "cuda":
            if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
                raise XQTBackendError("prequantized activation scale must be positive")

        selected_engine = self.engine
        if selected_engine == "auto":
            cache_key = (
                qactivation.device.type,
                qactivation.device.index,
                int(qactivation.shape[0]),
                self.activation_scale_mode,
                bool(qactivation.is_cuda),
                bool(self.qweight_t.is_cuda),
            )
            selected_engine = self._auto_engine_cache.get(cache_key)
            if selected_engine is None:
                resolved = resolve_int8_mma_engine(
                    "auto",
                    preferred_engines=self.preferred_engines,
                    fallback=self.fallback_engine,
                )
                selected_engine = self._pick_auto_engine(
                    qactivation,
                    candidates=resolved.candidates,
                    resolved_engine=resolved.engine,
                )
                self._auto_engine_cache[cache_key] = selected_engine

        used_engine = str(selected_engine)
        padded_rows = int(qactivation.shape[0])
        target_arch = _target_arch_from_device(qactivation.device)
        output: torch.Tensor
        reason = str(execution_reason)
        try:
            if used_engine == "ptx_sm89":
                allowed, availability_reason = self._can_use_ptx_sm89(qactivation)
                if not allowed:
                    raise XQTBackendError(availability_reason)
                output = self._run_ptx_sm89(qactivation, scale, self.output_dtype)
                reason = f"{execution_reason}_ptx_sm89"
            elif used_engine == "cuda_sm89":
                allowed, availability_reason = self._can_use_cuda_sm89(qactivation)
                if not allowed:
                    raise XQTBackendError(availability_reason)
                output = self._run_cuda_sm89(qactivation, scale)
                reason = f"{execution_reason}_cuda_sm89"
            elif used_engine == "tilelang":
                allowed, availability_reason = self._can_use_tilelang(qactivation)
                if not allowed:
                    raise XQTBackendError(availability_reason)
                output, padded_rows, target_arch = self._run_tilelang(
                    qactivation,
                    scale,
                    self.output_dtype,
                )
                reason = f"{execution_reason}_tilelang"
            elif used_engine == "triton":
                allowed, availability_reason = self._can_use_triton(qactivation)
                if not allowed:
                    raise XQTBackendError(availability_reason)
                output, padded_rows, target_arch = self._run_triton(
                    qactivation,
                    scale,
                    self.output_dtype,
                )
                reason = f"{execution_reason}_triton"
            else:
                used_engine = "torch_int_mm"
                acc = self._run_torch_int_mm(qactivation)
                output = acc.to(torch.float32) * (
                    scale
                    * self.weight_scale.to(
                        device=acc.device,
                        dtype=torch.float32,
                    )
                )
                if self.bias is not None:
                    output = output + self.bias.to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                reason = f"{execution_reason}_torch_int_mm"
        except Exception as exc:
            if self.fallback_engine != "torch_int_mm":
                raise
            used_engine = "torch_int_mm"
            reason = f"{execution_reason}_fallback: {exc}"
            acc = self._run_torch_int_mm(qactivation)
            output = acc.to(torch.float32) * (
                scale
                * self.weight_scale.to(
                    device=acc.device,
                    dtype=torch.float32,
                )
            )
            if self.bias is not None:
                output = output + self.bias.to(
                    device=output.device,
                    dtype=output.dtype,
                )

        self.last_execution = {
            "engine": used_engine,
            "reason": reason,
            "true_int8_mma": bool(qactivation.is_cuda and self.qweight_t.is_cuda),
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "accumulation_dtype": "int32",
            "activation_scale_mode": self.activation_scale_mode,
            "activation_quant_engine": str(activation_quant_engine),
            "fused_static_status": "prequantized_activation_to_gemm",
            "input_rows": int(qactivation.shape[0]),
            "padded_rows": int(padded_rows),
            "input_features": self.input_features,
            "output_features": self.output_features,
            "target_arch": target_arch,
            "prepacked_b": (
                self._qweight_prepacked_b is not None
                if used_engine in {"ptx_sm89", "cuda_sm89"}
                else False
            ),
        }
        return output.to(self.output_dtype)

    def _can_use_tilelang_static_fused(self, flat_inputs: torch.Tensor) -> tuple[bool, str]:
        if not flat_inputs.is_cuda or not self.qweight_t.is_cuda:
            return False, "TileLang fused INT8 Linear requires CUDA tensors"
        if flat_inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False, "TileLang fused INT8 Linear requires fp16, bf16, or fp32 activations"
        if self.output_features % self.block_n != 0:
            return False, "output_features is not block_n aligned"
        if self.input_features % self.block_k != 0:
            return False, "input_features is not block_k aligned"
        return True, "ok"

    def _run_m1_int8_gemv(
        self,
        flat_inputs: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, str]:
        """True M=1 INT8 GEMV: prefer PTX DP4A, fall back to float-domain products."""

        if (
            flat_inputs.is_cuda
            and flat_inputs.dtype == torch.float16
            and self.qweight_t.is_cuda
        ):
            try:
                from xqt.operator_opt.kernels.cute.int8mma_binding import (
                    int8_gemv_m1_fused_static_ptx_sm89,
                    int8mma_available,
                )

                if int8mma_available():
                    prepacked = self._ensure_ptx_prepacked_b()
                    output = int8_gemv_m1_fused_static_ptx_sm89(
                        flat_inputs,
                        self.qweight_t,
                        activation_scale,
                        self.weight_scale,
                        self.bias,
                        output_dtype=output_dtype,
                        prepacked_b=prepacked,
                    )
                    return output, "ptx_sm89_m1_dp4a_gemv"
            except Exception:
                pass
        api = _tilelang_int8_api()
        output = api["int8_linear_static_activation_m1_tilelang"](
            flat_inputs,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            target_arch=_target_arch_from_device(flat_inputs.device),
        )
        return output, "torch_m1_int8_products"

    def _run_tilelang_static_fused(
        self,
        flat_inputs: torch.Tensor,
        activation_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, str]:
        api = _tilelang_int8_api()
        target_arch = _target_arch_from_device(flat_inputs.device)
        if int(flat_inputs.shape[0]) == 1:
            output, backend = self._run_m1_int8_gemv(
                flat_inputs,
                activation_scale,
                output_dtype,
            )
            return output, 1, backend
        padded, original_rows = api["pad_rows_to_block"](flat_inputs, self.block_m)
        target_arch = _target_arch_from_device(padded.device)
        output = api["int8_linear_static_activation_tilelang"](
            padded,
            self.qweight_t,
            activation_scale,
            self.weight_scale,
            self.bias,
            output_dtype=output_dtype,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            threads=self.threads,
            num_stages=self.num_stages,
            target_arch=target_arch,
        )
        return output[:original_rows], int(padded.shape[0]), target_arch or "auto"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "Int8MmaLinear input trailing dimension does not match input_features"
            )
        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        selected_engine = self.engine
        flat_inputs = inputs.reshape(-1, self.input_features)
        if 0 < self.min_int8_rows and flat_inputs.shape[0] < self.min_int8_rows:
            output = self._run_bf16_fallback(flat_inputs, flat_inputs.shape[0])
            return output.reshape(*original_shape, self.output_features)
        if selected_engine == "auto":
            cache_key = (
                flat_inputs.device.type,
                flat_inputs.device.index,
                int(flat_inputs.shape[0]),
                self.activation_scale_mode,
                bool(inputs.is_cuda),
                bool(self.qweight_t.is_cuda),
            )
            cached_engine = self._auto_engine_cache.get(cache_key)
            if cached_engine is not None:
                selected_engine = cached_engine
            else:
                resolved = resolve_int8_mma_engine(
                    "auto",
                    preferred_engines=self.preferred_engines,
                    fallback=self.fallback_engine,
                )
                selected_engine = self._pick_auto_engine(
                    flat_inputs,
                    candidates=resolved.candidates,
                    resolved_engine=resolved.engine,
                )
                self._auto_engine_cache[cache_key] = selected_engine
        prefer_tilelang = selected_engine == "tilelang"
        prefer_ptx = selected_engine == "ptx_sm89"
        prefer_triton = selected_engine == "triton"
        prefer_cuda_sm89 = selected_engine == "cuda_sm89"
        prefer_tilelang = selected_engine == "tilelang" or (
            selected_engine == "auto" and inputs.is_cuda and self.qweight_t.is_cuda
        )
        if (
            prefer_cuda_sm89
            and self.activation_scale_mode == "static"
            and self.fallback_engine == "torch_int_mm"
        ):
            activation_scale = self._static_activation_scale(flat_inputs.device)
            used_engine = "cuda_sm89"
            reason = "true_int8_mma_cuda_sm89_cutlass_64x128_fused_scale_bias"
            fused_static_status = "not_attempted"
            padded_rows = int(flat_inputs.shape[0])
            target_arch = _target_arch_from_device(flat_inputs.device)
            allowed, availability_reason = self._can_use_cuda_sm89(flat_inputs)
            if allowed:
                try:
                    qactivation, activation_scale, activation_quant_engine = (
                        self._quantize_activation(flat_inputs, prefer_tilelang=True)
                    )
                    output = self._run_cuda_sm89(qactivation, activation_scale)
                    self.last_execution = {
                        "engine": used_engine,
                        "reason": reason,
                        "true_int8_mma": True,
                        "activation_dtype": "int8",
                        "weight_dtype": "int8",
                        "accumulation_dtype": "int32",
                        "activation_scale_mode": self.activation_scale_mode,
                        "activation_quant_engine": activation_quant_engine,
                        "fused_static_status": "two_kernel_quant_then_cuda_gemm",
                        "input_rows": int(qactivation.shape[0]),
                        "padded_rows": padded_rows,
                        "input_features": self.input_features,
                        "output_features": self.output_features,
                        "target_arch": target_arch,
                        "prepacked_b": self._qweight_prepacked_b is not None,
                    }
                    return output.reshape(*original_shape, self.output_features)
                except Exception as exc:
                    reason = f"cuda_sm89_static_fallback: {exc}"
                    fused_static_status = reason
            else:
                reason = f"cuda_sm89_static_unavailable: {availability_reason}"
                fused_static_status = reason
        if (
            prefer_ptx
            and self.activation_scale_mode == "static"
            and self.fallback_engine == "torch_int_mm"
        ):
            activation_scale = self._static_activation_scale(flat_inputs.device)
            used_engine = "ptx_sm89"
            reason = "true_int8_mma_ptx_sm89_fused_static"
            fused_static_status = "not_attempted"
            padded_rows = int(flat_inputs.shape[0])
            target_arch = _target_arch_from_device(flat_inputs.device)
            allowed, reason = self._can_use_ptx_sm89_static_fused(flat_inputs)
            if allowed:
                try:
                    output = self._run_ptx_sm89_static_fused(
                        flat_inputs,
                        activation_scale,
                        self.output_dtype,
                    )
                    self.last_execution = {
                        "engine": used_engine,
                        "reason": "true_int8_mma_ptx_sm89_fused_static_prepacked_b",
                        "true_int8_mma": True,
                        "activation_dtype": "int8",
                        "weight_dtype": "int8",
                        "accumulation_dtype": "int32",
                        "activation_scale_mode": self.activation_scale_mode,
                        "activation_quant_engine": "ptx_sm89_fused_static",
                        "fused_static_status": "used",
                        "input_rows": int(flat_inputs.shape[0]),
                        "padded_rows": padded_rows,
                        "input_features": self.input_features,
                        "output_features": self.output_features,
                        "target_arch": target_arch,
                        "prepacked_b": self._qweight_prepacked_b is not None,
                    }
                    output = output.to(self.output_dtype)
                    return output.reshape(*original_shape, self.output_features)
                except Exception as exc:
                    reason = f"ptx_sm89_fused_static_fallback: {exc}"
                    fused_static_status = reason
            else:
                reason = f"ptx_sm89_fused_static_unavailable: {reason}"
                fused_static_status = reason
        if (
            prefer_triton
            and not prefer_ptx
            and self.activation_scale_mode == "static"
            and self.fallback_engine == "torch_int_mm"
        ):
            activation_scale = self._static_activation_scale(flat_inputs.device)
            used_engine = "triton"
            reason = "true_int8_mma_triton_static_tilelang_quant"
            fused_static_status = "not_attempted"
            padded_rows = int(flat_inputs.shape[0])
            target_arch = _target_arch_from_device(flat_inputs.device)
            allowed, reason = self._can_use_triton(flat_inputs)
            if allowed:
                try:
                    output, padded_rows, target_arch = self._run_triton_static_fused(
                        flat_inputs,
                        activation_scale,
                        self.output_dtype,
                    )
                    self.last_execution = {
                        "engine": used_engine,
                        "reason": "true_int8_mma_triton_static_tilelang_quant",
                        "true_int8_mma": True,
                        "activation_dtype": "int8",
                        "weight_dtype": "int8",
                        "accumulation_dtype": "int32",
                        "activation_scale_mode": self.activation_scale_mode,
                        "activation_quant_engine": "tilelang_static",
                        "fused_static_status": "two_kernel_quant_then_gemm",
                        "input_rows": int(flat_inputs.shape[0]),
                        "padded_rows": padded_rows,
                        "input_features": self.input_features,
                        "output_features": self.output_features,
                        "target_arch": target_arch,
                    }
                    output = output.to(self.output_dtype)
                    return output.reshape(*original_shape, self.output_features)
                except Exception as exc:
                    reason = f"triton_static_fallback: {exc}"
                    fused_static_status = reason
            else:
                reason = f"triton_static_unavailable: {reason}"
                fused_static_status = reason
        if (
            prefer_tilelang
            and not prefer_ptx
            and self.activation_scale_mode == "static"
            and self.fallback_engine == "torch_int_mm"
        ):
            activation_scale = self._static_activation_scale(flat_inputs.device)
            used_engine = "tilelang"
            reason = "true_int8_mma_fused_static_activation"
            fused_static_status = "not_attempted"
            padded_rows = int(flat_inputs.shape[0])
            target_arch = _target_arch_from_device(flat_inputs.device)
            allowed, reason = self._can_use_tilelang_static_fused(flat_inputs)
            if allowed:
                try:
                    output, padded_rows, target_arch = self._run_tilelang_static_fused(
                        flat_inputs,
                        activation_scale,
                        self.output_dtype,
                    )
                    is_m1 = int(flat_inputs.shape[0]) == 1 and int(padded_rows) == 1
                    quant_engine = (
                        str(target_arch)
                        if is_m1
                        else "tilelang_fused_static"
                    )
                    self.last_execution = {
                        "engine": (
                            "ptx_sm89"
                            if is_m1 and str(target_arch).startswith("ptx")
                            else used_engine
                        ),
                        "reason": (
                            "true_int8_m1_dp4a_gemv"
                            if is_m1 and str(target_arch).startswith("ptx")
                            else "true_int8_m1_static_gemv"
                            if is_m1
                            else reason
                        ),
                        "true_int8_mma": True,
                        "activation_dtype": "int8",
                        "weight_dtype": "int8",
                        "accumulation_dtype": "int32",
                        "activation_scale_mode": self.activation_scale_mode,
                        "activation_quant_engine": quant_engine,
                        "fused_static_status": "used",
                        "input_rows": int(flat_inputs.shape[0]),
                        "padded_rows": padded_rows,
                        "input_features": self.input_features,
                        "output_features": self.output_features,
                        "target_arch": (
                            "sm_89"
                            if is_m1
                            else target_arch
                        ),
                    }
                    output = output.to(self.output_dtype)
                    return output.reshape(*original_shape, self.output_features)
                except Exception as exc:
                    reason = f"tilelang_fused_static_fallback: {exc}"
                    fused_static_status = reason
            else:
                reason = f"tilelang_fused_static_unavailable: {reason}"
                fused_static_status = reason
        qactivation, activation_scale, activation_quant_engine = self._quantize_activation(
            inputs,
            prefer_tilelang=prefer_tilelang and not prefer_ptx,
            prefer_triton=selected_engine == "triton",
        )
        if selected_engine == "auto":
            if qactivation.is_cuda:
                ptx_ok, _ = self._can_use_ptx_sm89(qactivation)
                selected_engine = "triton" if not ptx_ok else "ptx_sm89"
            else:
                selected_engine = "torch_int_mm"

        used_engine = selected_engine
        reason = reason if "reason" in locals() else "true_int8_mma"
        padded_rows = int(qactivation.shape[0])
        target_arch = _target_arch_from_device(qactivation.device)
        try:
            if selected_engine == "ptx_sm89":
                allowed, reason = self._can_use_ptx_sm89(qactivation)
                if not allowed:
                    raise XQTBackendError(reason)
                output = self._run_ptx_sm89(
                    qactivation,
                    activation_scale,
                    self.output_dtype,
                )
                reason = (
                    "true_int8_mma_ptx_sm89_prepacked_b"
                    if self._qweight_prepacked_b is not None
                    else "true_int8_mma_ptx_sm89"
                )
            elif selected_engine == "cuda_sm89":
                allowed, reason = self._can_use_cuda_sm89(qactivation)
                if not allowed:
                    raise XQTBackendError(reason)
                output = self._run_cuda_sm89(qactivation, activation_scale)
                reason = "true_int8_mma_cuda_sm89_cutlass_64x128_fused_scale_bias"
            elif selected_engine == "tilelang":
                allowed, reason = self._can_use_tilelang(qactivation)
                if not allowed:
                    raise XQTBackendError(reason)
                output, padded_rows, target_arch = self._run_tilelang(
                    qactivation,
                    activation_scale,
                    self.output_dtype,
                )
            elif selected_engine == "triton":
                allowed, reason = self._can_use_triton(qactivation)
                if not allowed:
                    raise XQTBackendError(reason)
                output, padded_rows, target_arch = self._run_triton(
                    qactivation,
                    activation_scale,
                    self.output_dtype,
                )
            else:
                acc = self._run_torch_int_mm(qactivation)
                output = acc.to(torch.float32) * (
                    activation_scale.to(torch.float32) * self.weight_scale.to(
                        device=acc.device,
                        dtype=torch.float32,
                    )
                )
        except Exception as exc:
            if self.fallback_engine != "torch_int_mm":
                raise
            used_engine = "torch_int_mm"
            reason = f"int8_mma_fallback: {exc}"
            acc = self._run_torch_int_mm(qactivation)
            output = acc.to(torch.float32) * (
                activation_scale.to(torch.float32) * self.weight_scale.to(
                    device=acc.device,
                    dtype=torch.float32,
                )
            )
        if self.bias is not None and used_engine not in {
            "tilelang",
            "triton",
            "ptx_sm89",
            "cuda_sm89",
        }:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        self.last_execution = {
            "engine": used_engine,
            "reason": reason,
            "true_int8_mma": bool(qactivation.is_cuda and self.qweight_t.is_cuda),
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "accumulation_dtype": "int32",
            "activation_scale_mode": self.activation_scale_mode,
            "activation_quant_engine": activation_quant_engine,
            "fused_static_status": fused_static_status if "fused_static_status" in locals() else "not_attempted",
            "input_rows": int(qactivation.shape[0]),
            "padded_rows": padded_rows,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "target_arch": target_arch,
        }
        output = output.to(self.output_dtype)
        return output.reshape(*original_shape, self.output_features)

    def execution_metadata(self) -> dict[str, Any]:
        metadata = dict(self.last_execution)
        metadata["runtime_precision"] = self._runtime_precision_summary()
        metadata["artifact_view"] = "runtime"
        return metadata

__all__ = ["Int8MmaLinear"]
