"""Runtime dual-branch SVD composite Linear modules."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.analysis.svd_analysis import decompose_weight_svd
from xqt.runtime.composite_branch import get_materializer, register_materializer
from xqt.runtime.modules.packing_int4 import _pack_int4, _unpack_int4
from xqt.runtime.modules.w4_storage_int8_mma_linear import W4StorageInt8MmaLinear

_SUPPORTED_RESIDUAL_QUANT_DTYPES = frozenset({"fp4", "int4"})

def _quantize_residual_int4(
    weight_res: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize residual weight to signed INT4 with per-group scales.

    Returns (packed_weight, scale, padded_in_features).
    """
    out_features, in_features = weight_res.shape
    normalized_group_size = max(1, min(group_size, in_features))
    padded_in_features = (
        (in_features + normalized_group_size - 1) // normalized_group_size
    ) * normalized_group_size

    # Pad to group_size boundary
    weight_f32 = weight_res.detach().to(torch.float32)
    if padded_in_features != in_features:
        weight_f32 = F.pad(weight_f32, (0, padded_in_features - in_features))

    # Per-group absmax quantization to [-8, 7]
    grouped = weight_f32.reshape(out_features, -1, normalized_group_size)
    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scale = torch.where(
        max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs)
    )  # shape: (out_features, num_groups, 1)
    quantized = torch.clamp(
        torch.round(grouped / (scale + 1e-12)), min=-8, max=7
    ).to(torch.int8)
    scale = scale.squeeze(-1)  # (out_features, num_groups)

    packed = _pack_int4(quantized.reshape(out_features, padded_in_features))
    return packed, scale.to(torch.float32), padded_in_features

def _dequantize_residual_int4(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    input_features: int,
    group_size: int,
    padded_input_features: int,
) -> torch.Tensor:
    """Dequantize packed INT4 residual weight back to float."""
    codes = _unpack_int4(packed_weight, padded_input_features)
    grouped = codes.reshape(codes.shape[0], -1, group_size)  # (out, num_groups, gs)
    scale_expanded = scale.unsqueeze(-1)  # (out, num_groups, 1)
    dequantized = grouped * scale_expanded
    return dequantized.reshape(codes.shape[0], padded_input_features)[
        :, :input_features
    ]

class LowRankBranch(nn.Module):
    """Two-layer low-rank branch that absorbs weight outliers.

    W_lr = L2 @ L1, where:
      L1: (r, in_features) — down-projection
      L2: (out_features, r) — up-projection

    Forward: y = L2(L1(x)) = x @ L1.T @ L2.T
    """

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        rank, in_features = down_weight.shape
        out_features, rank2 = up_weight.shape
        if rank != rank2:
            raise ValueError(
                f"Rank mismatch: down_proj rank={rank}, up_proj rank={rank2}"
            )
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(rank, out_features, bias=False)
        self.up_proj.weight.data = up_weight.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank correction: L2(L1(x))."""
        return self.up_proj(self.down_proj(x))

class SVDQuantLinear(nn.Module):
    """SVDQuant Linear: low-rank FP16 branch + quantized INT4/FP4 residual.

    Shapes:
      - down_proj.weight: (r, in_features)       ← L1
      - up_proj.weight:   (out_features, r)       ← L2
      - packed_residual:  (out_features, padded_in // 2) — uint8 packed INT4
      - residual_scale:   (out_features, num_groups)
      - bias (optional):  (out_features,)

    Forward (reference path):
      x_q    = act_quantize(x)                    # simulated: pass-through in reference
      y_main = dequant_gemm(x, packed_residual)   # dequant + fp16 matmul
      y_lora = up_proj(down_proj(x))              # low-rank correction
      return y_main + y_lora + bias
    """

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        packed_residual: torch.Tensor,
        residual_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        quant_dtype: str = "int4",
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.quant_dtype = str(quant_dtype)
        rank = int(down_weight.shape[0])
        self.rank = rank
        self.xqt_storage_protocol = "svd_low_rank_plus_residual"
        self.down_proj = nn.Linear(input_features, rank, bias=False)
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(rank, output_features, bias=False)
        self.up_proj.weight.data = up_weight.detach().clone()
        self.register_buffer("packed_residual", packed_residual.to(torch.uint8))
        self.register_buffer("residual_scale", residual_scale.to(torch.float32))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))
        self._pending_activation_scale: torch.Tensor | float | None = None
        self._pending_activation_scale_mode: str = "dynamic"

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rank: int,
        group_size: int = 128,
        quant_dtype: str = "int4",
    ) -> "SVDQuantLinear":
        """Build an SVDQuantLinear from a regular nn.Linear via SVD decomposition."""
        weight = module.weight.detach()
        weight_2d = weight.reshape(module.out_features, module.in_features)
        if weight_2d.shape[0] != module.out_features:
            weight_2d = weight_2d.reshape(module.out_features, -1)

        # 1. SVD decomposition
        decomp = decompose_weight_svd(weight_2d, rank=rank)

        # 2. Extract low-rank components
        L1, L2 = decomp.low_rank_components()  # L1: (r, in), L2: (out, r)

        # 3. Compute residual and quantize
        weight_res = decomp.residual_weight(weight_2d).to(torch.float32)
        if quant_dtype in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
            packed_residual, residual_scale, padded_in = _quantize_residual_int4(
                weight_res, group_size=group_size
            )
        else:
            raise ValueError(
                f"Unsupported quant_dtype '{quant_dtype}' for SVDQuant. "
                "Supported: fp4, int4"
            )

        bias = None if module.bias is None else module.bias.detach().to(torch.float32)

        return cls(
            down_weight=L1,
            up_weight=L2,
            packed_residual=packed_residual,
            residual_scale=residual_scale,
            bias=bias,
            input_features=module.in_features,
            output_features=module.out_features,
            group_size=group_size,
            padded_input_features=padded_in,
            quant_dtype=quant_dtype,
        )

    def set_activation_materialize_hint(
        self,
        *,
        activation_scale_mode: str = "dynamic",
        activation_scale: torch.Tensor | float | None = None,
    ) -> None:
        self._pending_activation_scale_mode = str(activation_scale_mode)
        self._pending_activation_scale = activation_scale

    def materialize_compute(self, spec: Any) -> nn.Module:
        """Bind residual compute from ModuleComputeSpec (Infer / post-quant).

        Dispatch via the materialize registry keyed on
        ``(self.xqt_storage_protocol, preferred_mode, compute_precision)``.
        Falls back to the concrete ``SVDQuantInt8MmaLinear`` when no builder
        is registered for the requested triple.

        ``spec.execution`` carries the ``activation_scale_mode`` (static enables
        fused activation-quant kernels) and ``min_int8_rows`` (small-M bf16
        fallback) knobs, applied to whichever runtime module is produced.
        """
        from xqt.contracts.compute import (
            ModuleComputeSpec,
            ModuleExecutionSpec,
            normalize_compute_contract,
        )

        if not isinstance(spec, ModuleComputeSpec):
            if isinstance(spec, Mapping):
                spec = ModuleComputeSpec.from_mapping(spec)
            else:
                raise TypeError("materialize_compute expects ModuleComputeSpec")

        residual_contract = None
        preferred: list[str] = list(spec.preferred_engines)
        for branch in spec.branches:
            if str(branch.get("name", "")) != "quant_residual":
                continue
            raw = branch.get("compute_contract")
            if raw is not None:
                residual_contract = normalize_compute_contract(str(raw))
            preferred = list(branch.get("preferred_engines") or preferred)
            break
        if residual_contract is None:
            residual_contract = normalize_compute_contract(spec.compute_contract)
        if residual_contract not in {
            "int8_mma",
            "w4_storage_int8_mma",
            "mix_fp4_int8_mma",
        }:
            return self

        engine = preferred[0] if preferred else "auto"
        meta = dict(spec.metadata or {})
        fallback = str(meta.get("fallback_engine", "torch_int_mm"))
        execution = spec.execution or ModuleExecutionSpec()
        if execution.activation_scale_mode != "dynamic":
            act_mode = execution.activation_scale_mode
        else:
            act_mode = str(
                meta.get("activation_scale_mode", self._pending_activation_scale_mode)
            )
        act_scale = meta.get("activation_scale", self._pending_activation_scale)
        min_rows = int(execution.min_int8_rows)
        mode = spec.preferred_mode or "split"
        compute_precision = execution.compute_precision

        # Registry lookup
        builder = get_materializer(
            self.xqt_storage_protocol,
            mode=mode,
            compute_precision=compute_precision,
        )
        if builder is not None:
            return builder(
                self,
                act_mode=act_mode,
                act_scale=act_scale,
                min_rows=min_rows,
                meta=meta,
                engine=engine,
                fallback=fallback,
            )

        # No builder registered for this (kind, mode, precision) triple.
        # Return self unchanged - callers can detect and raise.
        return self

    def dequantize_residual(self) -> torch.Tensor:
        """Dequantize the packed residual weight for reference computation."""
        if self.quant_dtype in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
            packed = self.packed_residual
            scale = self.residual_scale
            if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
                raise RuntimeError(
                    "packed_residual and residual_scale must be tensors"
                )
            return _dequantize_residual_int4(
                packed,
                scale,
                self.input_features,
                self.group_size,
                self.padded_input_features,
            )
        raise RuntimeError(f"Unsupported quant_dtype: {self.quant_dtype}")

    def low_rank_weight(self) -> torch.Tensor:
        """Reconstruct the low-rank branch weight W_lr = L2 @ L1."""
        return self.up_proj.weight.data @ self.down_proj.weight.data

    def full_weight_dequant(self) -> torch.Tensor:
        """Reconstruct the full dequantized weight: W_lr + W_res_deq."""
        return self.low_rank_weight() + self.dequantize_residual()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference forward pass.

        Fusion opportunities (Phase 2-3):
          - FUSE_DOWN: down_proj(x) + act_quantize(x) share input → single kernel
          - FUSE_UP: up_proj(h) + dequant_gemm_epilogue share accumulator → single kernel
        """
        device = x.device
        dtype = x.dtype

        # Main path: dequant residual → fp16 GEMM ([FUSE_UP candidate: epilogue fusion])
        weight_deq = self.dequantize_residual().to(device=device, dtype=dtype)
        y_main = F.linear(x, weight_deq)

        # Low-rank correction ([FUSE_DOWN candidate: input-sharing with act_quantize])
        h = self.down_proj(x)  # (batch, r)
        y_lora = self.up_proj(h)  # (batch, out)  [FUSE_UP candidate: add to accum]

        y = y_main + y_lora
        if self.bias is not None:
            y = y + self.bias.to(device=device, dtype=dtype)
        return y

class SVDQuantInt8MmaLinear(nn.Module):
    """SVDQuant Linear with packed 4-bit residual storage and W8A8 MMA execution.

    The low-rank branch remains at the source module precision. The residual is
    stored as groupwise packed signed 4-bit values (``quant_dtype`` is ``fp4`` or
    ``int4``) and re-targeted to the existing W8A8 INT8 MMA runtime. This is not
    native FP4/INT4 MMA.
    """

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        packed_residual: torch.Tensor,
        residual_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
        group_size: int,
        padded_input_features: int,
        output_dtype: torch.dtype,
        quant_dtype: str = "fp4",
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
        cache_int8_compute_view: bool = True,
        min_int8_rows: int = 0,
    ) -> None:
        super().__init__()
        residual_dtype = str(quant_dtype).lower()
        if residual_dtype not in _SUPPORTED_RESIDUAL_QUANT_DTYPES:
            allowed = ", ".join(sorted(_SUPPORTED_RESIDUAL_QUANT_DTYPES))
            raise ValueError(
                f"Unsupported quant_dtype '{quant_dtype}' for SVDQuantInt8MmaLinear. "
                f"Supported: {allowed}"
            )
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.group_size = int(group_size)
        self.padded_input_features = int(padded_input_features)
        self.quant_dtype = residual_dtype
        self.down_proj = nn.Linear(
            self.input_features,
            int(down_weight.shape[0]),
            bias=False,
            dtype=down_weight.dtype,
            device=down_weight.device,
        )
        self.down_proj.weight.data = down_weight.detach().clone()
        self.up_proj = nn.Linear(
            int(up_weight.shape[1]),
            self.output_features,
            bias=False,
            dtype=up_weight.dtype,
            device=up_weight.device,
        )
        self.up_proj.weight.data = up_weight.detach().clone()
        self.residual_int8 = W4StorageInt8MmaLinear(
            packed_residual,
            residual_scale,
            bias=None,
            input_features=self.input_features,
            output_features=self.output_features,
            group_size=self.group_size,
            padded_input_features=self.padded_input_features,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
            min_int8_rows=min_int8_rows,
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().to(torch.float32))

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        rank: int,
        group_size: int = 128,
        quant_dtype: str = "fp4",
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
        cache_int8_compute_view: bool = True,
    ) -> "SVDQuantInt8MmaLinear":
        """Build the W4-storage / INT8-MMA SVDQuant runtime module."""

        reference = SVDQuantLinear.from_linear(
            module,
            rank=rank,
            group_size=group_size,
            quant_dtype=quant_dtype,
        )
        return cls(
            down_weight=reference.down_proj.weight.detach(),
            up_weight=reference.up_proj.weight.detach(),
            packed_residual=reference.packed_residual.detach(),
            residual_scale=reference.residual_scale.detach(),
            bias=reference.bias,
            input_features=reference.input_features,
            output_features=reference.output_features,
            group_size=reference.group_size,
            padded_input_features=reference.padded_input_features,
            output_dtype=module.weight.dtype,
            quant_dtype=reference.quant_dtype,
            engine=engine,
            fallback_engine=fallback_engine,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            threads=threads,
            num_stages=num_stages,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            activation_quant_block_size=activation_quant_block_size,
            eps=eps,
            cache_int8_compute_view=cache_int8_compute_view,
        )

    def dequantize_residual(self) -> torch.Tensor:
        """Return the residual weight reconstructed from packed W4 storage."""

        return self.residual_int8.dequantize_weight()

    def low_rank_weight(self) -> torch.Tensor:
        """Return the dense weight reconstructed from the low-rank branch."""

        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        """Return the complete reconstructed weight for validation or export."""

        low_rank_weight = self.low_rank_weight()
        return low_rank_weight + self.dequantize_residual().to(
            dtype=low_rank_weight.dtype,
            device=low_rank_weight.device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run W8A8 MMA on the residual and add the high-precision low-rank path."""

        if inputs.shape[-1] != self.input_features:
            raise ValueError(
                "SVDQuantInt8MmaLinear input trailing dimension does not match "
                "input_features"
            )
        residual_output = self.residual_int8(inputs)
        low_rank_inputs = inputs.to(
            device=self.down_proj.weight.device,
            dtype=self.down_proj.weight.dtype,
        )
        low_rank_output = self.up_proj(self.down_proj(low_rank_inputs)).to(
            device=residual_output.device,
            dtype=residual_output.dtype,
        )
        output = residual_output + low_rank_output
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def execution_metadata(self) -> dict[str, Any]:
        """Expose residual INT8 execution details with SVDQuant context."""

        metadata = self.residual_int8.execution_metadata()
        metadata.update(
            {
                "implementation": "composite_add_svd_w4_residual_int8_mma",
                "compute_contract": "composite_add",
                "low_rank_branch": "source_precision",
                "residual_storage": "packed_signed_int4_group_scale",
                "residual_compute": "w8a8_int8_mma",
                "quant_dtype": self.quant_dtype,
            }
        )
        return metadata

class SVDQuantFp8Linear(nn.Module):
    """SVDQuant split module: fp16 low-rank branch + tensorwise FP8 residual.

    The SVDQuant-faithful form for Ada (sm_89+): the low-rank branch stays at
    source precision (absorbs outliers), while the residual runs as a tensorwise
    FP8 ``_scaled_mm`` (``Fp8MmaLinear``). The two extra low-rank GEMMs are the
    known split overhead; call :meth:`enable_fusion` to fold the branch chain
    with ``torch.compile`` and recover most of it at large M. Small-M inputs are
    routed to a dense fp16 GEMM inside the residual module (``min_fp8_rows``).
    """

    def __init__(
        self,
        *,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        residual_weight: torch.Tensor,
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
        from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.output_dtype = output_dtype
        rank = int(down_weight.shape[0])
        self.rank = rank
        self.xqt_storage_protocol = "svd_low_rank_plus_residual"
        lr_dtype = down_weight.dtype if down_weight.dtype in {
            torch.float16,
            torch.bfloat16,
        } else output_dtype
        self.down_proj = nn.Linear(self.input_features, rank, bias=False, dtype=lr_dtype)
        self.down_proj.weight.data = down_weight.detach().to(lr_dtype).clone()
        self.up_proj = nn.Linear(rank, self.output_features, bias=False, dtype=lr_dtype)
        self.up_proj.weight.data = up_weight.detach().to(lr_dtype).clone()
        # Residual GEMM carries the bias so a single _scaled_mm epilogue applies it.
        self.residual_fp8 = Fp8MmaLinear.from_dense_weight(
            residual_weight,
            bias=bias,
            input_features=self.input_features,
            output_features=self.output_features,
            output_dtype=output_dtype,
            activation_scale_mode=activation_scale_mode,
            activation_scale=activation_scale,
            min_fp8_rows=int(min_fp8_rows),
            eps=float(eps),
        )
        self._fused_forward: Any = None

    def enable_fusion(self, *, mode: str = "reduce-overhead") -> bool:
        """Fold the branch chain with torch.compile; return True on success.

        Only the lean compute (clean tensor ops) is compiled — metadata writes
        and the small-M fallback stay in eager ``forward`` so Inductor sees no
        graph breaks. Fusion recovers the low-rank overhead only in large-M
        (compute-bound) regimes and degrades gracefully to eager on failure.
        """
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            return False
        try:
            self._fused_forward = compile_fn(self._compute_lean, mode=mode)
        except Exception:
            self._fused_forward = None
            return False
        return True

    def disable_fusion(self) -> None:
        self._fused_forward = None

    def _low_rank(self, inputs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        low_rank_inputs = inputs.to(dtype=self.down_proj.weight.dtype)
        return self.up_proj(self.down_proj(low_rank_inputs)).to(
            device=ref.device, dtype=ref.dtype
        )

    def _compute_lean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compile-friendly path: lean FP8 residual + low-rank, no side effects."""
        residual_output = self.residual_fp8.lean_forward(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def _compute_guarded(self, inputs: torch.Tensor) -> torch.Tensor:
        """Eager path: full residual forward (metadata + small-M fallback)."""
        residual_output = self.residual_fp8(inputs)
        return residual_output + self._low_rank(inputs, residual_output)

    def dequantize_residual(self) -> torch.Tensor:
        return self.residual_fp8.dequantize_weight()

    def low_rank_weight(self) -> torch.Tensor:
        return self.up_proj.weight @ self.down_proj.weight

    def full_weight_dequant(self) -> torch.Tensor:
        low_rank_weight = self.low_rank_weight()
        return low_rank_weight + self.dequantize_residual().to(
            dtype=low_rank_weight.dtype, device=low_rank_weight.device
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_features:
            raise ValueError(
                "SVDQuantFp8Linear input trailing dimension does not match "
                "input_features"
            )
        # Route small-M / non-CUDA to the guarded eager path (fp16 fallback);
        # only send adequately-sized CUDA inputs to the fused lean path.
        rows = int(inputs.reshape(-1, self.input_features).shape[0])
        residual = self.residual_fp8
        use_fused = (
            self._fused_forward is not None
            and inputs.is_cuda
            and not (0 < residual.min_fp8_rows and rows < residual.min_fp8_rows)
        )
        if use_fused:
            return self._fused_forward(inputs)
        return self._compute_guarded(inputs)

    def execution_metadata(self) -> dict[str, Any]:
        metadata = self.residual_fp8.execution_metadata()
        metadata.update(
            {
                "implementation": "composite_add_svd_low_rank_plus_fp8_residual",
                "compute_contract": "composite_add",
                "low_rank_branch": "source_precision",
                "residual_compute": "fp8_scaled_mm",
                "fusion_enabled": self._fused_forward is not None,
            }
        )
        return metadata

# ============================================================================
# Materialize registry builders for svd_low_rank_plus_residual
# ============================================================================


@register_materializer("svd_low_rank_plus_residual", mode="collapse", compute_precision="w8a8")
def _build_svd_collapsed_int8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Merge low-rank + residual into one per-channel INT8 GEMM."""
    from xqt.runtime.modules.int8_mma_linear import Int8MmaLinear
    from xqt.runtime.modules.w4_storage_int8_mma_linear import (
        _channel_int8_from_float_weight,
    )

    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    eps = float(meta.get("eps", 1e-6))
    dense_weight = shell.full_weight_dequant()
    qweight_t, channel_scale = _channel_int8_from_float_weight(dense_weight, eps=eps)
    return Int8MmaLinear(
        qweight_t,
        channel_scale,
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        engine=str(options["engine"]),
        fallback_engine=str(options["fallback"]),
        block_m=int(meta.get("block_m", 64)),
        block_n=int(meta.get("block_n", 64)),
        block_k=int(meta.get("block_k", 64)),
        threads=int(meta.get("threads", 128)),
        num_stages=int(meta.get("num_stages", 2)),
        output_dtype=shell.down_proj.weight.dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        activation_quant_block_size=int(meta.get("activation_quant_block_size", 256)),
        eps=eps,
        min_int8_rows=int(options["min_rows"]),
    )


@register_materializer("svd_low_rank_plus_residual", mode="collapse", compute_precision="fp8")
def _build_svd_collapsed_fp8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Fold low-rank + residual into one tensorwise FP8 Linear."""
    from xqt.runtime.modules.fp8_mma_linear import Fp8MmaLinear

    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    source_dtype = shell.down_proj.weight.dtype
    output_dtype = (
        source_dtype
        if source_dtype in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    dense_weight = shell.full_weight_dequant()
    return Fp8MmaLinear.from_dense_weight(
        dense_weight,
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        output_dtype=output_dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        min_fp8_rows=int(options["min_rows"]),
        eps=float(meta.get("eps", 1e-8)),
    )


@register_materializer("svd_low_rank_plus_residual", mode="split", compute_precision="fp8")
def _build_svd_split_fp8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Keep fp16 low-rank branch + tensorwise FP8 residual."""
    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    source_dtype = shell.down_proj.weight.dtype
    output_dtype = (
        source_dtype
        if source_dtype in {torch.float16, torch.bfloat16}
        else torch.float16
    )
    return SVDQuantFp8Linear(
        down_weight=shell.down_proj.weight.detach(),
        up_weight=shell.up_proj.weight.detach(),
        residual_weight=shell.dequantize_residual(),
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        output_dtype=output_dtype,
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        min_fp8_rows=int(options["min_rows"]),
        eps=float(meta.get("eps", 1e-8)),
    )


@register_materializer("svd_low_rank_plus_residual", mode="split", compute_precision="w8a8")
def _build_svd_split_int8(shell: "SVDQuantLinear", **options: Any) -> nn.Module:
    """Keep fp16 low-rank branch + W4-storage / INT8-MMA residual."""
    meta: Mapping[str, Any] = options["meta"]  # type: ignore[assignment]
    residual_dtype = str(
        meta.get("quant_dtype", getattr(shell, "quant_dtype", "fp4"))
    ).lower()
    return SVDQuantInt8MmaLinear(
        down_weight=shell.down_proj.weight.detach(),
        up_weight=shell.up_proj.weight.detach(),
        packed_residual=shell.packed_residual.detach(),
        residual_scale=shell.residual_scale.detach(),
        bias=None if shell.bias is None else shell.bias.detach(),
        input_features=shell.input_features,
        output_features=shell.output_features,
        group_size=shell.group_size,
        padded_input_features=shell.padded_input_features,
        output_dtype=shell.down_proj.weight.dtype,
        quant_dtype=residual_dtype,
        engine=str(options["engine"]),
        fallback_engine=str(options["fallback"]),
        block_m=int(meta.get("block_m", 64)),
        block_n=int(meta.get("block_n", 64)),
        block_k=int(meta.get("block_k", 64)),
        threads=int(meta.get("threads", 128)),
        num_stages=int(meta.get("num_stages", 2)),
        activation_scale_mode=str(options["act_mode"]),
        activation_scale=options.get("act_scale"),
        activation_quant_block_size=int(meta.get("activation_quant_block_size", 256)),
        eps=float(meta.get("eps", 1e-6)),
        cache_int8_compute_view=bool(meta.get("cache_int8_compute_view", True)),
        min_int8_rows=int(options["min_rows"]),
    )


__all__ = [
    "LowRankBranch",
    "SVDQuantFp8Linear",
    "SVDQuantLinear",
    "SVDQuantInt8MmaLinear",
    "_build_svd_collapsed_fp8",
    "_build_svd_collapsed_int8",
    "_build_svd_split_fp8",
    "_build_svd_split_int8",
]
