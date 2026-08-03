"""Runtime W8A8-FP8 Linear backed by cuBLAS ``torch._scaled_mm`` (Ada sm_89+).

Measured on consumer Ada (sm_89): per-tensor (``tensorwise``) FP8 GEMM reaches
~1.8x fp16 at large M, while row-wise scaling is *slower* than fp16 — so this
module is tensorwise only. The per-forward activation-quant ``amax`` reduction
erases the win, so ``activation_scale_mode`` defaults to ``static`` (calibrated
scale). Small-M shapes (decode) fall back to a dense fp16 GEMM via
``min_fp8_rows`` because FP8 loses there on square layers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xqt.core.errors import XQTBackendError

_E4M3 = torch.float8_e4m3fn
_ACTIVATION_SCALE_MODES = {"dynamic", "static"}
_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}


def _scaled_mm_available() -> bool:
    return callable(getattr(torch, "_scaled_mm", None))


def _device_supports_fp8(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def _quantize_fp8_per_tensor(
    tensor: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 (e4m3) quantization; returns (q, fp32 scale)."""

    amax = tensor.detach().abs().amax().clamp_min(float(eps))
    scale = (amax / Fp8MmaLinear.quant_max).to(torch.float32).reshape(())
    quantized = (
        (tensor / scale.to(tensor.dtype))
        .clamp(-Fp8MmaLinear.quant_max, Fp8MmaLinear.quant_max)
        .to(_E4M3)
    )
    return quantized, scale


class Fp8MmaLinear(nn.Module):
    """Linear replacement backed by tensorwise FP8 ``_scaled_mm``."""

    quant_max: float = 448.0  # e4m3 representable maximum

    def __init__(
        self,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        output_dtype: torch.dtype = torch.float16,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if str(activation_scale_mode) not in _ACTIVATION_SCALE_MODES:
            raise ValueError("activation_scale_mode must be dynamic or static")
        if output_dtype not in _OUTPUT_DTYPES:
            raise ValueError("output_dtype must be float16 or bfloat16")
        if int(min_fp8_rows) < 0:
            raise ValueError("min_fp8_rows must be >= 0")
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.output_dtype = output_dtype
        self.activation_scale_mode = str(activation_scale_mode)
        self.min_fp8_rows = int(min_fp8_rows)
        self.eps = float(eps)
        self.last_execution: dict[str, Any] = {"engine": "not_run"}
        self.register_buffer("qweight", qweight.to(_E4M3).contiguous())
        self.register_buffer(
            "weight_scale", weight_scale.to(torch.float32).reshape(())
        )
        # _scaled_mm requires bias in the output float dtype (not fp32).
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(output_dtype).contiguous())
        initial = torch.tensor(0.0, dtype=torch.float32)
        self._has_static_activation_scale = activation_scale is not None
        if activation_scale is not None:
            initial = torch.as_tensor(activation_scale, dtype=torch.float32).reshape(())
        self.register_buffer("static_activation_scale", initial)
        self._dense_weight: torch.Tensor | None = None

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        output_dtype: torch.dtype | None = None,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> "Fp8MmaLinear":
        out_dtype = output_dtype or (
            module.weight.dtype if module.weight.dtype in _OUTPUT_DTYPES else torch.float16
        )
        qweight, weight_scale = _quantize_fp8_per_tensor(
            module.weight.detach(), eps=eps
        )
        bias = None if module.bias is None else module.bias.detach()
        return cls(
            qweight,
            weight_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            output_dtype=out_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=min_fp8_rows,
            eps=eps,
        )

    @classmethod
    def from_dense_weight(
        cls,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        output_dtype: torch.dtype = torch.float16,
        activation_scale_mode: str = "static",
        activation_scale: torch.Tensor | float | None = None,
        min_fp8_rows: int = 0,
        eps: float = 1e-8,
    ) -> "Fp8MmaLinear":
        """Build from an already-reconstructed dense weight (e.g. SVD collapse)."""

        qweight, weight_scale = _quantize_fp8_per_tensor(weight.detach(), eps=eps)
        return cls(
            qweight,
            weight_scale,
            bias=bias,
            input_features=input_features,
            output_features=output_features,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=min_fp8_rows,
            eps=eps,
        )

    def set_static_activation_scale(self, scale: torch.Tensor | float) -> None:
        value = torch.as_tensor(
            scale, dtype=torch.float32, device=self.static_activation_scale.device
        ).reshape(())
        if float(value.detach().cpu().item()) <= 0.0:
            raise ValueError("static activation scale must be positive")
        self.static_activation_scale.copy_(value)
        self._has_static_activation_scale = True
        self.activation_scale_mode = "static"

    def calibrate_static_activation_scale(self, sample_inputs: torch.Tensor) -> torch.Tensor:
        _, scale = _quantize_fp8_per_tensor(sample_inputs, eps=self.eps)
        self.set_static_activation_scale(scale.to(self.static_activation_scale.device))
        return self.static_activation_scale.detach().clone()

    def dequantize_weight(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.weight_scale.to(torch.float32)

    def _dense_fp16_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cached = self._dense_weight
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        weight = (self.qweight.to(torch.float32) * self.weight_scale.to(torch.float32)).to(
            device=device, dtype=dtype
        )
        self._dense_weight = weight
        return weight

    def _run_fp16_fallback(self, inputs: torch.Tensor, reason: str) -> torch.Tensor:
        compute_dtype = self.output_dtype
        weight = self._dense_fp16_weight(compute_dtype, inputs.device)
        bias = None if self.bias is None else self.bias.to(compute_dtype)
        output = F.linear(inputs.to(compute_dtype), weight, bias)
        self.last_execution = {
            "engine": "fp16_fallback",
            "reason": reason,
            "true_fp8_mma": False,
            "min_fp8_rows": self.min_fp8_rows,
            "input_rows": int(inputs.shape[0]),
            "input_features": self.input_features,
            "output_features": self.output_features,
        }
        return output.to(self.output_dtype)

    def _activation_scale(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        if self.activation_scale_mode == "static":
            if not self._has_static_activation_scale:
                raise XQTBackendError(
                    "static activation scale mode requires "
                    "calibrate_static_activation_scale() or "
                    "set_static_activation_scale() before forward"
                )
            return self.static_activation_scale.to(flat_inputs.device)
        amax = flat_inputs.detach().abs().amax().clamp_min(self.eps)
        return (amax / Fp8MmaLinear.quant_max).to(torch.float32).reshape(())

    def lean_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pure-tensor FP8 GEMM path for torch.compile fusion.

        No metadata writes, no try/except, no fallback branch — a clean sequence
        of tensor ops so Inductor can fuse without graph breaks. The caller must
        guarantee the FP8 fast path applies (CUDA + sm_89 + rows adequate); use
        :meth:`forward` for the guarded/fallback path.
        """

        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat_inputs = inputs.reshape(-1, self.input_features)
        act_scale = self._activation_scale(flat_inputs)
        qact = (
            (flat_inputs / act_scale.to(flat_inputs.dtype))
            .clamp(-Fp8MmaLinear.quant_max, Fp8MmaLinear.quant_max)
            .to(_E4M3)
        )
        output = torch._scaled_mm(
            qact,
            self.qweight.t(),
            scale_a=act_scale,
            scale_b=self.weight_scale,
            out_dtype=self.output_dtype,
            bias=None if self.bias is None else self.bias.to(self.output_dtype),
        )
        return output.reshape(*original_shape, self.output_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise XQTBackendError(
                "Fp8MmaLinear input trailing dimension does not match input_features"
            )
        original_shape = tuple(int(dim) for dim in inputs.shape[:-1])
        flat_inputs = inputs.reshape(-1, self.input_features)
        rows = int(flat_inputs.shape[0])

        # Route to dense fp16 when FP8 cannot win or is unavailable.
        if not (inputs.is_cuda and _scaled_mm_available() and _device_supports_fp8(inputs.device)):
            out = self._run_fp16_fallback(flat_inputs, "fp8_unavailable")
            return out.reshape(*original_shape, self.output_features)
        if 0 < self.min_fp8_rows and rows < self.min_fp8_rows:
            out = self._run_fp16_fallback(flat_inputs, "rows_below_min_fp8_rows")
            return out.reshape(*original_shape, self.output_features)

        act_scale = self._activation_scale(flat_inputs)
        qact = (
            (flat_inputs / act_scale.to(flat_inputs.dtype))
            .clamp(-Fp8MmaLinear.quant_max, Fp8MmaLinear.quant_max)
            .to(_E4M3)
        )
        weight_t = self.qweight.t()  # (in, out) column-major view for _scaled_mm
        bias = None if self.bias is None else self.bias.to(self.output_dtype)
        try:
            output = torch._scaled_mm(
                qact,
                weight_t,
                scale_a=act_scale,
                scale_b=self.weight_scale,
                out_dtype=self.output_dtype,
                bias=bias,
            )
        except Exception as exc:  # numeric/layout guard → dense fp16
            out = self._run_fp16_fallback(flat_inputs, f"scaled_mm_fallback: {exc}")
            return out.reshape(*original_shape, self.output_features)

        self.last_execution = {
            "engine": "scaled_mm_fp8",
            "reason": "true_fp8_mma_tensorwise",
            "true_fp8_mma": True,
            "activation_dtype": "float8_e4m3fn",
            "weight_dtype": "float8_e4m3fn",
            "activation_scale_mode": self.activation_scale_mode,
            "input_rows": rows,
            "input_features": self.input_features,
            "output_features": self.output_features,
        }
        return output.reshape(*original_shape, self.output_features)

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self.last_execution)


__all__ = ["Fp8MmaLinear"]
