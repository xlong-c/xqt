"""Model-side KV-scale attention reference entity (C9 residual).

This module consumes the canonical KV scale artifacts (``KvScaleArtifact``
payloads, i.e. per-tensor k/v scales) and an optional ``RuntimeQuantContract``,
and runs a torch SDPA reference path with int8 quantize/dequantize of K and V.

It is a model-side *reference entity*: there is no page table, block pool,
cache eviction, or serving scheduler here. ``selected_kernel`` and fallback
reasons are recorded for every forward. The default path stays the torch SDPA
reference; an opt-in CUDA fused path (``preferred_kernel="auto"|"tilelang"``)
runs the TileLang KV-int8 fused attention kernel, which consumes int8 K/V
storage and dequantizes inside the kernel. The fused path is only reported as
verified after a fused forward actually ran on device; any unavailability or
kernel error falls back to the reference path with an honest
``fallback_reason``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts.contract_consume import (
    ContractConsumeReport,
    consume_runtime_quant_contract,
)
from xqt.contracts.runtime_quant import (
    RUNTIME_QUANT_CONTRACT_KEY,
    RuntimeQuantContract,
)
from xqt.core.errors import XQTBackendError, XQTConfigError


@dataclass(frozen=True, slots=True)
class KvCacheMetadata:
    """Model-side KV cache metadata; no cache storage or management."""

    dtype: str
    mode: str
    layer_scales: tuple[tuple[str, float, float], ...] = ()
    cache_block_size: int | None = None
    hit_rate: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "mode": self.mode,
            "layer_scales": [
                {
                    "layer_path": layer_path,
                    "k_scale": float(k_scale),
                    "v_scale": float(v_scale),
                }
                for layer_path, k_scale, v_scale in self.layer_scales
            ],
            "cache_block_size": self.cache_block_size,
            "hit_rate": self.hit_rate,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KvCacheMetadata:
        if not isinstance(payload, Mapping):
            raise XQTConfigError(
                "KvCacheMetadata.from_dict expects a mapping; "
                f"got {type(payload).__name__}"
            )
        dtype = str(payload.get("dtype", "int8")).strip().lower()
        mode = str(payload.get("mode", "per_tensor_scale")).strip().lower()
        scales: list[tuple[str, float, float]] = []
        raw_scales = payload.get("layer_scales", ())
        if not isinstance(raw_scales, (list, tuple)):
            raise XQTConfigError("KvCacheMetadata.layer_scales must be a sequence")
        for item in raw_scales:
            if not isinstance(item, Mapping):
                raise XQTConfigError(
                    "KvCacheMetadata.layer_scales items must be mappings"
                )
            layer_path = str(item.get("layer_path", ""))
            k_scale = item.get("k_scale", item.get("attn.k_scale"))
            v_scale = item.get("v_scale", item.get("attn.v_scale"))
            if k_scale is None or v_scale is None:
                raise XQTConfigError(
                    f"KvCacheMetadata layer {layer_path!r} requires k_scale/v_scale"
                )
            scales.append(
                (layer_path, float(k_scale), float(v_scale))
            )
        raw_block_size = payload.get("cache_block_size")
        raw_hit_rate = payload.get("hit_rate")
        raw_notes = payload.get("notes", ())
        if not isinstance(raw_notes, (list, tuple)):
            raise XQTConfigError("KvCacheMetadata.notes must be a sequence")
        return cls(
            dtype=dtype,
            mode=mode,
            layer_scales=tuple(scales),
            cache_block_size=(
                None if raw_block_size is None else int(raw_block_size)
            ),
            hit_rate=None if raw_hit_rate is None else float(raw_hit_rate),
            notes=tuple(str(item) for item in raw_notes),
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Mapping[str, Any],
        *,
        dtype: str | None = None,
        mode: str | None = None,
    ) -> KvCacheMetadata:
        """Extract per-layer K/V scales from canonical KV quant metadata."""

        if not isinstance(artifacts, Mapping):
            raise XQTConfigError(
                "KvCacheMetadata.from_artifacts expects a mapping"
            )
        raw_layers = artifacts.get("layers")
        layer_source: Mapping[str, Any]
        if isinstance(raw_layers, Mapping):
            layer_source = raw_layers
        else:
            layer_source = artifacts
        scales: list[tuple[str, float, float]] = []
        for layer_path, raw_artifact in layer_source.items():
            if not isinstance(raw_artifact, Mapping):
                continue
            k_scale = raw_artifact.get(
                "k_scale", raw_artifact.get("attn.k_scale")
            )
            v_scale = raw_artifact.get(
                "v_scale", raw_artifact.get("attn.v_scale")
            )
            if k_scale is None or v_scale is None:
                continue
            scales.append(
                (str(layer_path), float(k_scale), float(v_scale))
            )
        if not scales:
            raise XQTConfigError(
                "KvCacheMetadata.from_artifacts found no k_scale/v_scale layers"
            )
        return cls(
            dtype=str(dtype or artifacts.get("dtype", "int8")).strip().lower(),
            mode=str(
                mode or artifacts.get("mode", "per_tensor_scale")
            ).strip().lower(),
            layer_scales=tuple(scales),
            notes=(
                "model-side metadata only; no cache storage or management",
            ),
        )

    def scales_for(self, layer_path: str) -> tuple[float, float] | None:
        for candidate_path, k_scale, v_scale in self.layer_scales:
            if candidate_path == layer_path:
                return k_scale, v_scale
        return None


class KvScaleAttention(nn.Module):
    """Reference self-attention entity consuming KV scale artifacts."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        head_dim: int | None = None,
        k_scale: float,
        v_scale: float,
        qmax: int = 127,
        dtype: str = "int8",
        causal: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        layer_path: str = "",
        contract: RuntimeQuantContract | Mapping[str, Any] | None = None,
        cache_block_size: int | None = None,
        hit_rate: float | None = None,
        preferred_kernel: str = "reference",
        fused_block_m: int = 64,
        fused_block_n: int = 64,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0:
            raise ValueError("dim and heads must be positive")
        resolved_head_dim = int(dim // heads if head_dim is None else head_dim)
        if resolved_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if head_dim is None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when head_dim is omitted")
        if float(k_scale) <= 0.0 or float(v_scale) <= 0.0:
            raise ValueError("k_scale and v_scale must be positive")
        if int(qmax) <= 0:
            raise ValueError("qmax must be positive")
        normalized_kernel = str(preferred_kernel).strip().lower()
        if normalized_kernel not in {"reference", "auto", "tilelang"}:
            raise ValueError(
                "preferred_kernel must be one of 'reference', 'auto', 'tilelang'"
            )
        if int(fused_block_m) <= 0 or int(fused_block_n) <= 0:
            raise ValueError("fused_block_m and fused_block_n must be positive")

        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = resolved_head_dim
        self.inner_dim = self.heads * self.head_dim
        self.qmax = int(qmax)
        self.kv_cache_dtype = str(dtype).strip().lower()
        self.causal = bool(causal)
        self.dropout_p = float(dropout)
        self.layer_path = str(layer_path)
        self._contract = contract
        self._cache_block_size = (
            None if cache_block_size is None else int(cache_block_size)
        )
        self._hit_rate = None if hit_rate is None else float(hit_rate)
        self.preferred_kernel = normalized_kernel
        self._fused_block_m = int(fused_block_m)
        self._fused_block_n = int(fused_block_n)
        self._fallback_reason: str | None = None
        self._selected_kernel = "torch_sdpa_kv_scale_reference"
        self._cuda_fused_verified = False

        self.q_proj = nn.Linear(self.dim, self.inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.dim, self.inner_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.inner_dim, self.dim, bias=out_bias)
        self.register_buffer(
            "k_scale",
            torch.tensor(float(k_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "v_scale",
            torch.tensor(float(v_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "attn_k_scale",
            torch.tensor(float(k_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "attn_v_scale",
            torch.tensor(float(v_scale), dtype=torch.float32),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(self._contract, RuntimeQuantContract):
            payload[RUNTIME_QUANT_CONTRACT_KEY] = self._contract.to_dict()
        elif isinstance(self._contract, Mapping):
            payload[RUNTIME_QUANT_CONTRACT_KEY] = dict(self._contract)
        return payload

    def _consume_contract(self) -> ContractConsumeReport:
        if self._contract is None:
            return ContractConsumeReport(
                ok=False,
                contract=None,
                errors=("runtime_quant_contract_absent",),
                notes=(
                    "reference entity can run without a contract; a contract is "
                    "required to claim kernel/layout semantics",
                ),
            )
        report = consume_runtime_quant_contract(
            self._contract,
            require_kernels=False,
            require_shapes=False,
        )
        contract = report.contract
        if (
            report.ok
            and contract is not None
            and contract.kv_cache_dtype is not None
            and str(contract.kv_cache_dtype).strip().lower() != self.kv_cache_dtype
        ):
            return ContractConsumeReport(
                ok=False,
                contract=contract,
                errors=("kv_cache_dtype_mismatch",),
                notes=report.notes
                + (
                    f"contract kv_cache_dtype={contract.kv_cache_dtype}, "
                    f"entity dtype={self.kv_cache_dtype}",
                ),
            )
        return report

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = tensor.shape
        return (
            tensor.reshape(batch, seq, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _heads, seq, head_dim = tensor.shape
        return (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, seq, self.heads * head_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "KvScaleAttention expects 3D input shaped [batch, seq, dim]"
            )
        if int(x.shape[-1]) != self.dim:
            raise ValueError(
                f"input last dim {int(x.shape[-1])} does not match dim {self.dim}"
            )
        q = self._reshape(self.q_proj(x))
        k_int8 = self._quantize_kv_int8(self.k_proj(x), self.k_scale)
        v_int8 = self._quantize_kv_int8(self.v_proj(x), self.v_scale)
        if self.preferred_kernel != "reference":
            fused_attn = self._try_fused_attention(q, k_int8, v_int8)
            if fused_attn is not None:
                return self.out_proj(self._merge_heads(fused_attn))
        k = self._reshape(self._dequantize_kv(k_int8, self.k_scale, x.dtype))
        v = self._reshape(self._dequantize_kv(v_int8, self.v_scale, x.dtype))
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
            is_causal=self.causal,
        )
        return self.out_proj(self._merge_heads(attn))

    def _quantize_kv_int8(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """per-tensor int8 量化, reference 与 fused 路径共用同一份 int8."""

        return (
            torch.round(tensor / scale)
            .clamp(-self.qmax, self.qmax)
            .to(dtype=torch.int8)
        )

    def _dequantize_kv(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return quantized.to(dtype=dtype) * scale

    def _quantize_kv(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return self._dequantize_kv(
            self._quantize_kv_int8(tensor, scale),
            scale,
            tensor.dtype,
        )

    def _fused_unavailability_reason(self, q: torch.Tensor) -> str | None:
        """返回 fused 路径不可用的原因; 可用时返回 None."""

        if not q.is_cuda:
            return "cuda_unavailable"
        if q.dtype != torch.float16:
            return "dtype_not_fp16"
        if self.dropout_p != 0.0:
            return "dropout_unsupported"
        if self.head_dim % 16 != 0:
            return "head_dim_not_multiple_of_16"
        from xqt.operator_opt.kernels.tilelang._common import (
            tilelang_runtime_unavailability_reason,
            tilelang_runtime_usable,
        )

        if not tilelang_runtime_usable():
            return tilelang_runtime_unavailability_reason() or "tilelang_unusable"
        return None

    def _try_fused_attention(
        self,
        q: torch.Tensor,
        k_int8: torch.Tensor,
        v_int8: torch.Tensor,
    ) -> torch.Tensor | None:
        """尝试 TileLang KV-int8 fused 路径; 不可用时诚实记录并返回 None."""

        reason = self._fused_unavailability_reason(q)
        if reason is not None:
            self._selected_kernel = "torch_sdpa_kv_scale_reference"
            self._fallback_reason = reason
            return None
        from xqt.operator_opt.kernels.tilelang.kv_int8_attention import (
            KV_INT8_FUSED_KERNEL_NAME,
            fused_kv_int8_attention_forward_tilelang,
        )

        try:
            attn = fused_kv_int8_attention_forward_tilelang(
                q,
                self._reshape(k_int8),
                self._reshape(v_int8),
                float(self.k_scale.item()),
                float(self.v_scale.item()),
                causal=self.causal,
                block_m=self._fused_block_m,
                block_n=self._fused_block_n,
            )
        except (XQTBackendError, RuntimeError) as exc:
            self._selected_kernel = "torch_sdpa_kv_scale_reference"
            self._fallback_reason = f"tilelang_kernel_error:{type(exc).__name__}"
            return None
        self._selected_kernel = KV_INT8_FUSED_KERNEL_NAME
        self._cuda_fused_verified = True
        self._fallback_reason = None
        return attn

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Offline prefill phase wrapper (full sequence)."""

        return self.forward(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Offline decode phase wrapper (single token)."""

        return self.forward(x)

    def kv_cache_metadata(self) -> KvCacheMetadata:
        fused_note = (
            f"CUDA fused kernel verified on device: {self._selected_kernel}"
            if self._cuda_fused_verified
            else "CUDA fused-kernel verification pending"
        )
        return KvCacheMetadata(
            dtype=self.kv_cache_dtype,
            mode="per_tensor_scale",
            layer_scales=(
                (self.layer_path, float(self.k_scale.item()), float(self.v_scale.item())),
            ),
            cache_block_size=self._cache_block_size,
            hit_rate=self._hit_rate,
            notes=(
                "model-side metadata only; no cache storage or management",
                fused_note,
            ),
        )

    def report(self) -> dict[str, Any]:
        consume = self._consume_contract()
        # contract 缺失不覆盖已记录的 kernel fallback 原因, 也不把已成功
        # 运行的 fused 路径标成 fallback.
        if (
            not consume.ok
            and not self._cuda_fused_verified
            and self._fallback_reason is None
        ):
            self._fallback_reason = ";".join(consume.errors) or "contract_invalid"
        note = (
            f"torch SDPA reference entity; CUDA fused kernel verified on device: "
            f"{self._selected_kernel}"
            if self._cuda_fused_verified
            else "torch SDPA reference entity; CUDA fused-kernel verification pending"
        )
        return {
            "entity": "kv_scale_attention_reference",
            "layer_path": self.layer_path,
            "preferred_kernel": self.preferred_kernel,
            "selected_kernel": self._selected_kernel,
            "cuda_fused_verified": self._cuda_fused_verified,
            "fallback_reason": self._fallback_reason,
            "contract_ok": consume.ok,
            "contract_errors": list(consume.errors),
            "contract_notes": list(consume.notes),
            "kv_cache_metadata": self.kv_cache_metadata().to_dict(),
            "notes": (note,),
        }

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Mapping[str, Any],
        *,
        dim: int,
        heads: int,
        head_dim: int | None = None,
        layer_path: str = "",
        contract: RuntimeQuantContract | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> KvScaleAttention:
        """Build one layer entity from canonical KV quant metadata."""

        metadata = KvCacheMetadata.from_artifacts(artifacts)
        scales = metadata.scales_for(layer_path)
        if scales is None:
            if metadata.layer_scales:
                first_path, k_scale, v_scale = metadata.layer_scales[0]
                layer_path = first_path
                scales = (k_scale, v_scale)
            else:
                raise XQTConfigError("no K/V scale layer available")
        k_scale, v_scale = scales
        return cls(
            dim,
            heads=heads,
            head_dim=head_dim,
            k_scale=k_scale,
            v_scale=v_scale,
            layer_path=layer_path,
            contract=contract,
            dtype=str(metadata.dtype),
            **kwargs,
        )


__all__ = [
    "KvCacheMetadata",
    "KvScaleAttention",
]
