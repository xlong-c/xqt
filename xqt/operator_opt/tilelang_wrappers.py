"""TileLang runtime wrappers extracted from materialize.py.

This module keeps the minimal executable wrapper family for direct TileLang
operator targets without changing candidate materialization or executor
behavior.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from xqt.core.errors import XQTBackendError
from xqt.quant.bridges.nvfp4 import (
    NVFP4LinearBridge,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    infer_nvfp4_tensor_layout,
)
from xqt.quant.capability import describe_quant_backend_capability

from .backends.tilelang import get_tilelang_kernel_spec, run_tilelang_kernel
from .runtime import (
    DEFAULT_CUDA_GRAPH_WARMUP,
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)
from .types import OperatorOptimizationTargetPlan


class _TileLangConvWrapper(nn.Module):
    """Conv2d operator-family wrapper with architecture-aware native runtime routing."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = conv
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "conv"
        self.last_fastpath = "none"

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_conv_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("conv_fastpath", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                "conv",
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                block_m=int(self.settings.get("block_m", 64)),
                block_n=int(self.settings.get("block_n", 64)),
                block_k=int(self.settings.get("block_k", 64)),
                threads=int(self.settings.get("threads", 128)),
                num_stages=int(self.settings.get("num_stages", 2)),
                target_arch=self.settings.get("target_arch"),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang conv runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_conv_fastpath(x)
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_cudnn_conv2d"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_half_conv2d_im2col_gemm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang conv fastpath requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_native_fastpath":
            return F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._run_tilelang_or_reference(x)
        return F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["conv"],
                "operator_families": ["conv"],
                "supports_grouped_conv": False,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangConv3dWrapper(nn.Module):
    """Conv3d 1x1x1 operator-family wrapper with TileLang fastpath routing."""

    def __init__(
        self,
        conv: nn.Conv3d,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.conv = conv
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "conv"
        self.last_fastpath = "none"

    def _is_supported_fastpath(self) -> bool:
        return (
            tuple(int(value) for value in self.conv.kernel_size) == (1, 1, 1)
            and tuple(int(value) for value in self.conv.stride) == (1, 1, 1)
            and tuple(int(value) for value in self.conv.padding) == (0, 0, 0)
            and tuple(int(value) for value in self.conv.dilation) == (1, 1, 1)
            and int(self.conv.groups) == 1
        )

    def _run_eager(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                "conv3d_1x1x1",
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                block_m=int(self.settings.get("block_m", 64)),
                block_n=int(self.settings.get("block_n", 64)),
                block_k=int(self.settings.get("block_k", 64)),
                threads=int(self.settings.get("threads", 128)),
                num_stages=int(self.settings.get("num_stages", 2)),
                target_arch=self.settings.get("target_arch"),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang conv3d runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self._run_eager(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_supported_fastpath():
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                "TileLang Conv3d fastpath requires kernel_size=stride=dilation=(1,1,1), "
                "padding=(0,0,0), and groups=1."
            )
            self.last_fastpath = "eager_reference_fallback"
            return self._run_eager(x)
        self.last_execution_mode = (
            "cuda_tilelang_entry" if x.is_cuda else "reference_fallback"
        )
        self.last_fastpath = (
            "tilelang_half_conv3d_1x1x1_gemm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "TileLang conv3d fastpath requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._run_tilelang_or_reference(x)
        return self._run_eager(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "kernel_pattern": "conv3d_1x1x1",
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["conv3d_1x1x1"],
                "operator_families": ["conv"],
                "supports_grouped_conv": False,
                "requires_kernel_size": [1, 1, 1],
                "requires_stride": [1, 1, 1],
                "requires_padding": [0, 0, 0],
                "requires_dilation": [1, 1, 1],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangLinearWrapper(nn.Module):
    """Standalone half Linear wrapper for direct TileLang operator targets."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.linear = linear
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "linear"
        self.last_fastpath = "none"

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_runtime", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def _selected_linear_pattern(self) -> str:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list) and "linear_marlin" in patterns:
            return "linear_marlin"
        return "linear"

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("TileLang linear target requires at least 1D input")
        if x.shape[-1] <= 0:
            raise XQTBackendError(
                "TileLang linear target requires a non-empty trailing feature dimension"
            )
        return x.reshape(-1, int(x.shape[-1])), tuple(int(dim) for dim in x.shape[:-1])

    @staticmethod
    def _restore_output(output: torch.Tensor, prefix_shape: tuple[int, ...]) -> torch.Tensor:
        return output.reshape(*prefix_shape, int(output.shape[-1]))

    def _run_tilelang_or_reference(
        self,
        x: torch.Tensor,
        flat_input: torch.Tensor,
    ) -> torch.Tensor:
        pattern = self._selected_linear_pattern()
        try:
            if pattern == "linear_marlin":
                return run_tilelang_kernel(
                    pattern,
                    flat_input,
                    self.linear.weight,
                    bias=self.linear.bias,
                    precision=str(self.settings.get("precision", "auto")),
                    block_m=int(self.settings.get("block_m", 64)),
                    block_n=int(self.settings.get("block_n", 64)),
                    block_k=int(self.settings.get("block_k", 64)),
                    threads=int(self.settings.get("threads", 128)),
                    num_stages=int(self.settings.get("num_stages", 2)),
                    target_arch=self.settings.get("target_arch"),
                    fallback=self.fallback,
                )
            return run_tilelang_kernel(
                pattern,
                flat_input,
                self.linear.weight,
                self.linear.bias,
                block_m=int(self.settings.get("block_m", 64)),
                block_n=int(self.settings.get("block_n", 64)),
                block_k=int(self.settings.get("block_k", 64)),
                threads=int(self.settings.get("threads", 128)),
                num_stages=int(self.settings.get("num_stages", 2)),
                target_arch=self.settings.get("target_arch"),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang linear runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self.linear(x).reshape(-1, int(self.linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix_shape = tuple(int(dim) for dim in x.shape[:-1])
        flat_input, _ = self._flatten_input(x)
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_linear_fastpath(x)
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_torch_linear"
            if self.last_execution_mode == "cuda_native_fastpath"
            else (
                "tilelang_marlin_linear_kernel"
                if self._selected_linear_pattern() == "linear_marlin"
                else "tilelang_half_linear_kernel"
            )
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang linear kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
            return self.linear(x)
        output = self._run_tilelang_or_reference(x, flat_input)
        return self._restore_output(output, prefix_shape)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_precisions": ["fp16", "bf16", "int8", "int4"],
                "supported_patterns": ["linear", "linear_marlin"],
                "operator_families": ["linear"],
                "supports_rank_gte_1_via_batch_flatten": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangNormWrapper(nn.Module):
    """Standalone half LayerNorm wrapper for direct TileLang operator targets."""

    def __init__(
        self,
        norm: nn.LayerNorm,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.norm = norm
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "norm"
        self.last_fastpath = "none"
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_norm_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("norm_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def _prefer_graph_norm_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("norm_fastpath", "auto"))
        if mode in {"graph", "tilelang_graph"}:
            return True
        return False

    def _norm_weight_bias(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized_dim = int(self.norm.normalized_shape[-1])
        weight = self.norm.weight
        bias = self.norm.bias
        if weight is None:
            weight = torch.ones(normalized_dim, device=x.device, dtype=x.dtype)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return weight.to(device=x.device, dtype=x.dtype), bias

    def _run_tilelang_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self._norm_weight_bias(x)
        try:
            return run_tilelang_kernel(
                "norm",
                x,
                weight,
                bias,
                eps=float(self.norm.eps),
                threads=int(self.settings.get("threads", 64)),
                fallback=self.fallback,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang norm runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self.norm(x)

    def _norm_graph_cache_key(
        self,
        x: torch.Tensor,
    ) -> tuple[Any, ...]:
        normalized_dim = int(self.norm.normalized_shape[-1])
        return (
            cuda_graph_tensor_signature(x),
            normalized_dim,
            float(self.norm.eps),
            int(self.settings.get("threads", 64)),
        )

    def _run_norm_with_optional_graph(self, x: torch.Tensor) -> torch.Tensor:
        if not self._prefer_graph_norm_fastpath(x):
            self.last_graph_state = "disabled"
            self.last_graph_reason = "norm_fastpath is not set to graph mode"
            return self._run_tilelang_or_reference(x)
        weight, bias = self._norm_weight_bias(x)
        graph_bias = bias if bias is not None else torch.zeros_like(weight)
        cache_key = self._norm_graph_cache_key(x)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = capture_cuda_graph_with_static_state(
                    (x,),
                    body=lambda x_arg: run_tilelang_kernel(
                        "norm",
                        x_arg,
                        weight,
                        graph_bias,
                        eps=float(self.norm.eps),
                        threads=int(self.settings.get("threads", 64)),
                        fallback=self.fallback,
                    ),
                    warmup=int(
                        self.settings.get("cuda_graph_warmup", DEFAULT_CUDA_GRAPH_WARMUP)
                    ),
                )
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._run_tilelang_or_reference(x)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return replay_cuda_graph_tensor_callable(state, (x,))
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return replay_cuda_graph_tensor_callable(state, (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_graph_tilelang = x.is_cuda and self._prefer_graph_norm_fastpath(x)
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if x.is_cuda and self._prefer_native_norm_fastpath(x)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if x.is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_torch_layer_norm"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_half_layer_norm_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_half_layer_norm"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {
                "cuda_native_fastpath",
                "cuda_graph_tilelang_entry",
                "cuda_tilelang_entry",
            }
            else "TileLang norm kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode != "cuda_graph_tilelang_entry":
            self.last_graph_state = "disabled"
            self.last_graph_reason = (
                None
                if self.last_execution_mode == "cuda_native_fastpath"
                else "graph fastpath was not selected"
            )
        if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
            return self.norm(x)
        if self.last_execution_mode == "cuda_graph_tilelang_entry":
            return self._run_norm_with_optional_graph(x)
        return self._run_tilelang_or_reference(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "cuda_graph_replay"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": ["norm"],
                "operator_families": ["norm"],
                "normalized_last_dim_only": True,
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
            },
        }


class _TileLangAttentionWrapper(nn.Module):
    """Minimal executable wrapper for attention-pattern TileLang targets."""

    def __init__(
        self,
        attention: nn.MultiheadAttention,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "attention"
        self.last_fastpath = "none"
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _resolved_target_arch(self, q: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if q.is_cuda:
            major, minor = torch.cuda.get_device_capability(q.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_attention_fastpath(self, q: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return self._resolved_target_arch(q) == "sm_89"

    def _prefer_graph_attention_fastpath(self, q: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode in {"graph", "tilelang_graph"}:
            return True
        return False

    def _canonicalize_attention_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_input = query if self.attention.batch_first else query.transpose(0, 1)
        source_key = query if key is None else key
        if source_key is query:
            k_input = q_input
        else:
            k_input = source_key if self.attention.batch_first else source_key.transpose(0, 1)
        source_value = source_key if value is None else value
        if source_value is source_key:
            v_input = k_input
        elif source_value is query:
            v_input = q_input
        else:
            v_input = source_value if self.attention.batch_first else source_value.transpose(0, 1)
        return q_input, k_input, v_input

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not getattr(self.attention, "_qkv_same_embed_dim", True):
            raise XQTBackendError(
                "TileLang attention wrapper currently supports only qkv_same_embed_dim=True"
            )
        if self.attention.in_proj_weight is None:
            raise XQTBackendError("TileLang attention wrapper requires packed in_proj_weight")
        embed_dim = int(self.attention.embed_dim)
        q_proj = F.linear(
            query,
            self.attention.in_proj_weight[:embed_dim],
            None if self.attention.in_proj_bias is None else self.attention.in_proj_bias[:embed_dim],
        )
        k_proj = F.linear(
            key,
            self.attention.in_proj_weight[embed_dim : 2 * embed_dim],
            None
            if self.attention.in_proj_bias is None
            else self.attention.in_proj_bias[embed_dim : 2 * embed_dim],
        )
        v_proj = F.linear(
            value,
            self.attention.in_proj_weight[2 * embed_dim :],
            None
            if self.attention.in_proj_bias is None
            else self.attention.in_proj_bias[2 * embed_dim :],
        )
        return q_proj, k_proj, v_proj

    def _reshape_for_tilelang(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = tensor.shape
        num_heads = int(self.attention.num_heads)
        head_dim = embed_dim // num_heads
        return tensor.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _merge_from_tilelang(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim).contiguous()

    def _project_qkv_for_tilelang(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_proj, k_proj, v_proj = self._project_qkv(query, key, value)
        return (
            self._reshape_for_tilelang(q_proj),
            self._reshape_for_tilelang(k_proj),
            self._reshape_for_tilelang(v_proj),
        )

    def _finalize_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        merged = self._merge_from_tilelang(attn_output)
        return self.attention.out_proj(merged)

    @staticmethod
    def _attention_graph_runtime_args(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[tuple[int, int, int], tuple[torch.Tensor, ...]]:
        unique_args: list[torch.Tensor] = []
        arg_mapping: list[int] = []
        tensor_index: dict[int, int] = {}
        for tensor in (query, key, value):
            index = tensor_index.get(id(tensor))
            if index is None:
                index = len(unique_args)
                unique_args.append(tensor)
                tensor_index[id(tensor)] = index
            arg_mapping.append(index)
        return (arg_mapping[0], arg_mapping[1], arg_mapping[2]), tuple(unique_args)

    @staticmethod
    def _resolve_attention_graph_args(
        dynamic_args: tuple[torch.Tensor, ...],
        arg_mapping: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            dynamic_args[arg_mapping[0]],
            dynamic_args[arg_mapping[1]],
            dynamic_args[arg_mapping[2]],
        )

    def _attention_graph_cache_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> tuple[Any, ...]:
        arg_mapping, runtime_args = self._attention_graph_runtime_args(query, key, value)
        return (
            tuple(cuda_graph_tensor_signature(tensor) for tensor in runtime_args),
            arg_mapping,
            bool(is_causal),
            float(self.attention.dropout),
            int(self.settings.get("block_m", 64)),
            int(self.settings.get("block_n", 64)),
            int(self.settings.get("threads", 128)),
            int(self.settings.get("num_stages", 2)),
            str(self.settings.get("target_arch") or ""),
        )

    def _run_tilelang_attention_body(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        return run_tilelang_kernel(
            "attention",
            q,
            k,
            v,
            causal=is_causal,
            dropout_p=float(self.attention.dropout),
            block_m=int(self.settings.get("block_m", 64)),
            block_n=int(self.settings.get("block_n", 64)),
            threads=int(self.settings.get("threads", 128)),
            num_stages=int(self.settings.get("num_stages", 2)),
            fallback=self.fallback,
        )

    def _run_tilelang_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv_for_tilelang(query, key, value)
        return self._finalize_attention_output(
            self._run_tilelang_attention_body(q, k, v, is_causal=is_causal)
        )

    def _run_native_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv_for_tilelang(query, key, value)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=float(self.attention.dropout),
            is_causal=is_causal,
        )
        return self._finalize_attention_output(attn_output)

    def _graph_capture_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> dict[str, Any]:
        arg_mapping, runtime_args = self._attention_graph_runtime_args(query, key, value)
        state = capture_cuda_graph_with_static_state(
            runtime_args,
            body=lambda *dynamic_args: self._run_tilelang_attention_forward(
                *self._resolve_attention_graph_args(
                    tuple(dynamic_args),
                    arg_mapping,
                ),
                is_causal=is_causal,
            ),
            warmup=int(self.settings.get("cuda_graph_warmup", DEFAULT_CUDA_GRAPH_WARMUP)),
        )
        state["kind"] = "tilelang_attention_full_forward"
        state["arg_mapping"] = arg_mapping
        return state

    def _run_attention_with_optional_graph(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if not self._prefer_graph_attention_fastpath(query):
            self.last_graph_state = "disabled"
            self.last_graph_reason = "attention_fastpath is not set to graph mode"
            return self._run_tilelang_attention_forward(query, key, value, is_causal=is_causal)
        _, runtime_args = self._attention_graph_runtime_args(query, key, value)
        cache_key = self._attention_graph_cache_key(query, key, value, is_causal=is_causal)
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = self._graph_capture_attention(query, key, value, is_causal=is_causal)
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = f"CUDA Graph capture failed: {exc}"
                return self._run_tilelang_attention_forward(query, key, value, is_causal=is_causal)
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
            return replay_cuda_graph_tensor_callable(state, runtime_args)
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return replay_cuda_graph_tensor_callable(state, runtime_args)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        *,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del average_attn_weights
        if attn_mask is not None or key_padding_mask is not None:
            raise XQTBackendError(
                "TileLang attention wrapper does not yet support attn_mask or key_padding_mask"
            )
        q_input, k_input, v_input = self._canonicalize_attention_inputs(query, key, value)
        input_is_cuda = q_input.is_cuda and k_input.is_cuda and v_input.is_cuda
        use_graph_tilelang = (
            input_is_cuda and self._prefer_graph_attention_fastpath(q_input)
        )
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if input_is_cuda and self._prefer_native_attention_fastpath(q_input)
            else "cuda_graph_tilelang_entry"
            if use_graph_tilelang
            else "cuda_tilelang_entry"
            if input_is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_sdpa"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_attention_cuda_graph"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else "tilelang_attention_kernel"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {
                "cuda_tilelang_entry",
                "cuda_graph_tilelang_entry",
                "cuda_native_fastpath",
            }
            else "TileLang attention kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode != "cuda_graph_tilelang_entry":
            self.last_graph_state = "disabled"
            self.last_graph_reason = (
                None
                if self.last_execution_mode == "cuda_native_fastpath"
                else "graph fastpath was not selected"
            )
        if self.last_execution_mode == "cuda_native_fastpath":
            projected = self._run_native_attention_forward(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        elif self.last_execution_mode == "cuda_graph_tilelang_entry":
            projected = self._run_attention_with_optional_graph(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        else:
            projected = self._run_tilelang_attention_forward(
                q_input,
                k_input,
                v_input,
                is_causal=is_causal,
            )
        output = projected if self.attention.batch_first else projected.transpose(0, 1)
        weights = None
        if need_weights:
            batch_size = int(q_input.shape[0]) if q_input.ndim == 3 else 0
            target_len = int(q_input.shape[1]) if q_input.ndim == 3 else 0
            source_len = int(k_input.shape[1]) if k_input.ndim == 3 else 0
            weights = output.new_zeros((batch_size, target_len, source_len))
        return output, weights

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "cuda_graph_replay"
            if self.last_execution_mode == "cuda_graph_tilelang_entry"
            else
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "dropout_p": 0.0,
                "requires_seq_kv_gte_seq_q": True,
                "supported_patterns": ["attention"],
                "operator_families": ["attention"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
            },
        }

class _TileLangXqtAttentionWrapper(nn.Module):
    """Executable wrapper for xqt.nn.Attention with TileLang kernel fastpath."""

    def __init__(
        self,
        attention: "xqt_nn.Attention",
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_operator_family = "attention"
        self.last_fastpath = "none"

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.attention._reshape_qkv(self.attention.q_proj(x)),
            self.attention._reshape_qkv(self.attention.k_proj(x)),
            self.attention._reshape_qkv(self.attention.v_proj(x)),
        )

    def _run_tilelang_attention_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        return run_tilelang_kernel(
            "attention",
            q,
            k,
            v,
            causal=is_causal,
            dropout_p=float(self.attention.dropout_p),
            block_m=int(self.settings.get("block_m", 64)),
            block_n=int(self.settings.get("block_n", 64)),
            threads=int(self.settings.get("threads", 128)),
            num_stages=int(self.settings.get("num_stages", 2)),
            fallback=self.fallback,
        )

    def _run_tilelang_xqt_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        attn_output = self._run_tilelang_attention_kernel(
            q, k, v, is_causal=bool(self.attention.causal)
        )
        merged = self.attention._merge_heads(attn_output)
        return self.attention.out_proj(merged)

    def _run_sdpa_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=float(self.attention.dropout_p),
            is_causal=bool(self.attention.causal),
        )
        merged = self.attention._merge_heads(attn_output)
        return self.attention.out_proj(merged)

    def _forward_prefer_tilelang(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self._run_tilelang_xqt_attention_forward(x)
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"TileLang attention runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self._run_sdpa_attention_forward(x)

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _prefer_native_attention_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("attention_fastpath", "auto"))
        if mode == "native":
            return True
        if mode in {"tilelang", "graph", "tilelang_graph"}:
            return False
        return self._resolved_target_arch(x) == "sm_89"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_is_cuda = x.is_cuda
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if input_is_cuda and self._prefer_native_attention_fastpath(x)
            else "cuda_tilelang_entry"
            if input_is_cuda
            else "reference_fallback"
        )
        self.last_fastpath = (
            "native_sdpa"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "tilelang_xqt_attention_kernel"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_native_fastpath", "cuda_tilelang_entry"}
            else "TileLang attention kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_native_fastpath":
            return self._run_sdpa_attention_forward(x)
        if self.last_execution_mode == "cuda_tilelang_entry":
            return self._forward_prefer_tilelang(x)
        return self._run_sdpa_attention_forward(x)

    def runtime_config(self) -> dict[str, Any]:
        base = (
            self.attention.runtime_config()
            if hasattr(self.attention, "runtime_config")
            else {"engine": "tilelang"}
        )
        return {
            **base,
            "execution": self.execution_metadata(),
        }

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": "float16",
                "dropout_p": 0.0,
                "requires_seq_kv_gte_seq_q": True,
                "supported_patterns": ["attention"],
                "operator_families": ["attention"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TileLangEagerDenseLinearModule(nn.Module):
    """One-time dequantized dense Linear replacement for sm_89 native runtime parity."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.linear = linear
        self._execution_metadata = dict(metadata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def execution_metadata(self) -> dict[str, Any]:
        return dict(self._execution_metadata)


class _TileLangDequantGemmWrapper(nn.Module):
    """Minimal executable wrapper for dequant GEMM TileLang targets."""

    def __init__(
        self,
        module: nn.Module,
        *,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.module = module
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_weight_source = "not_run"
        self.last_weight_representation = "unknown"
        self.last_consumes_packed_weight = False
        self.last_unpack_stage: str | None = None
        self.last_kernel_pattern = "dequant_gemm_epilogue"
        self.last_operator_family = "linear"
        self.last_fastpath = "none"
        self._cached_nvfp4_bridge: NVFP4LinearBridge | None = None
        self._preferred_patterns_config = self._preferred_patterns()
        self._dense_linear_bridge = getattr(self.module, "tilelang_dense_linear_args", None)
        self._packed_nvfp4_bridge = getattr(self.module, "tilelang_packed_nvfp4_dequant_gemm_args", None)
        self._packed_fp4_bridge = getattr(self.module, "tilelang_packed_dequant_gemm_args", None)
        self._dense_fp4_bridge = getattr(self.module, "tilelang_dequant_gemm_args", None)
        if self._packed_nvfp4_bridge is None and self._dense_fp4_bridge is None and self._packed_fp4_bridge is None:
            inferred_bridge = bridge_module_to_nvfp4_linear_shared(self.module)
            if inferred_bridge is not None:
                self._cached_nvfp4_bridge = inferred_bridge

    def _weight_only_bridge_metadata(
        self,
        *,
        packed: bool,
        direct_dense_bridge: bool = False,
    ) -> tuple[str, str]:
        bits = getattr(self.module, "bits", None)
        method = getattr(self.module, "method", None)
        if isinstance(bits, int) and method in {"awq", "gptq"}:
            strategy = f"{method}_weight_only_int{bits}"
            if packed:
                return (
                    f"{strategy}_linear_packed_bridge",
                    "packed_signed_int4_plus_group_scale",
                )
            return (
                f"{strategy}_linear_dense_cache_bridge",
                f"dense_dequantized_{strategy}_weight_cache",
            )
        if packed:
            return (
                "fp4_weight_only_linear_packed_bridge",
                "packed_signed_int4_plus_group_scale",
            )
        if direct_dense_bridge:
            return (
                "fp4_weight_only_linear_dense_bridge",
                "dense_unpacked_codes_plus_expanded_scale",
            )
        return (
            "fp4_weight_only_linear_dense_cache_bridge",
            "dense_dequantized_fp4_weight_cache",
        )

    def _can_delegate_dense_native_linear(self, activation: str | None) -> bool:
        return (
            activation is None
            and hasattr(self.module, "dense_weight")
            and callable(getattr(self.module, "forward", None))
        )

    def _resolve_dense_linear_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, str | None] | None:
        if callable(self._dense_linear_bridge):
            if callable(self._packed_fp4_bridge) or callable(self._dense_fp4_bridge):
                self.last_weight_source, self.last_weight_representation = (
                    self._weight_only_bridge_metadata(packed=False)
                )
            elif hasattr(self.module, "mx_precision") and hasattr(self.module, "block_size"):
                mx_precision = int(getattr(self.module, "mx_precision"))
                self.last_weight_source = "mxfp_weight_only_linear_dense_cache_bridge"
                self.last_weight_representation = (
                    f"dense_dequantized_mxfp{mx_precision}_weight_cache"
                )
            else:
                self.last_weight_source = "module_dense_cache_bridge"
                self.last_weight_representation = "dense_dequantized_weight_cache"
            return self._dense_linear_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_dense_cache_bridge"
        self.last_weight_representation = "dense_dequantized_weight_cache"
        return bridge.tilelang_dense_linear_args(dtype=x.dtype, device=x.device)

    def _resolve_packed_nvfp4_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, int, int, torch.Tensor | None] | None:
        if callable(self._packed_nvfp4_bridge):
            self.last_weight_source = "compressed_tensors_nvfp4_packed_bridge"
            self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
            return self._packed_nvfp4_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_packed_bridge"
        self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
        return bridge.tilelang_packed_nvfp4_dequant_gemm_args(dtype=x.dtype, device=x.device)

    @staticmethod
    def _apply_activation(output: torch.Tensor, activation: str | None) -> torch.Tensor:
        if activation is None:
            return output
        if activation == "gelu":
            return F.gelu(output)
        if activation == "silu":
            return F.silu(output)
        if activation == "relu":
            return F.relu(output)
        raise XQTBackendError(f"unsupported activation: {activation}")

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _preferred_patterns(self) -> list[str]:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list):
            return [str(pattern) for pattern in patterns]
        return ["dequant_gemm_epilogue"]

    def _prefer_dense_linear_fastpath(self, x: torch.Tensor) -> bool:
        if (
            callable(self._dense_linear_bridge)
            and self._packed_nvfp4_bridge is None
            and self._packed_fp4_bridge is None
            and self._dense_fp4_bridge is None
        ):
            return True
        mode = str(self.settings.get("linear_fastpath", "auto"))
        if mode == "dense":
            return True
        if mode == "packed":
            return False
        target_arch = self._resolved_target_arch(x)
        return target_arch == "sm_89"

    def _prefer_native_linear_fastpath(self, x: torch.Tensor) -> bool:
        mode = str(self.settings.get("linear_runtime", "auto"))
        if mode == "native":
            return True
        if mode == "tilelang":
            return False
        target_arch = self._resolved_target_arch(x)
        return target_arch == "sm_89"

    def _resolved_nvfp4_bridge(self) -> NVFP4LinearBridge | None:
        if callable(self._packed_fp4_bridge) or callable(self._dense_fp4_bridge):
            return None
        if self._cached_nvfp4_bridge is not None:
            return self._cached_nvfp4_bridge
        existing_bridge = getattr(self.module, "_bridge", None)
        if isinstance(existing_bridge, NVFP4LinearBridge):
            self._cached_nvfp4_bridge = existing_bridge
            return existing_bridge
        inferred_bridge = bridge_module_to_nvfp4_linear(self.module)
        if inferred_bridge is not None:
            self._cached_nvfp4_bridge = inferred_bridge
        return inferred_bridge

    def _run_tilelang_or_reference(
        self,
        kernel_pattern: str,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            return run_tilelang_kernel(
                kernel_pattern,
                *args,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            spec = get_tilelang_kernel_spec(kernel_pattern)
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in allowed
            }
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"TileLang {kernel_pattern} runtime fallback: {exc}"
            )
            self.last_fastpath = "eager_reference_fallback"
            self.last_unpack_stage = (
                "one_time_eager_dequant_cache"
                if kernel_pattern == "dense_linear_epilogue"
                else "eager_reference_fallback"
            )
            return spec.reference(*args, **filtered_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefer_dense_linear = self._prefer_dense_linear_fastpath(x) and self._preferred_patterns_config in (
            ["dequant_gemm_epilogue"],
            ["dense_linear_epilogue"],
        )
        if prefer_dense_linear:
            dense_args = self._resolve_dense_linear_args(x)
            if dense_args is not None:
                qweight, bias, activation = dense_args
                self.last_kernel_pattern = "dense_linear_epilogue"
                self.last_consumes_packed_weight = False
                self.last_unpack_stage = "one_time_eager_dequant_cache"
                self.last_fastpath = "one_time_eager_dequant_plus_dense_half_gemm"
                if x.is_cuda:
                    self.last_execution_mode = (
                        "cuda_native_fastpath"
                        if self._prefer_native_linear_fastpath(x)
                        else "cuda_tilelang_entry"
                    )
                    self.last_execution_reason = None
                else:
                    self.last_execution_mode = "reference_fallback"
                    self.last_execution_reason = (
                        "TileLang dequant GEMM kernel requires CUDA tensors; using configured fallback."
                    )
                if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
                    if self._can_delegate_dense_native_linear(activation):
                        return self.module(x)
                    return self._apply_activation(F.linear(x, qweight, bias), activation)
                if bias is not None:
                    return self._run_tilelang_or_reference(
                        "dense_linear_epilogue",
                        x,
                        qweight,
                        bias,
                        activation=activation,
                        block_m=int(self.settings.get("block_m", 64)),
                        block_n=int(self.settings.get("block_n", 64)),
                        block_k=int(self.settings.get("block_k", 64)),
                        threads=int(self.settings.get("threads", 128)),
                        num_stages=int(self.settings.get("num_stages", 2)),
                        target_arch=self.settings.get("target_arch"),
                        fallback=self.fallback,
                    )
                return self._run_tilelang_or_reference(
                    "dense_linear_epilogue",
                    x,
                    qweight,
                    activation=activation,
                    block_m=int(self.settings.get("block_m", 64)),
                    block_n=int(self.settings.get("block_n", 64)),
                    block_k=int(self.settings.get("block_k", 64)),
                    threads=int(self.settings.get("threads", 128)),
                    num_stages=int(self.settings.get("num_stages", 2)),
                    target_arch=self.settings.get("target_arch"),
                    fallback=self.fallback,
                )
        qweight: torch.Tensor | None = None
        scale: torch.Tensor | None = None
        bias: torch.Tensor | None = None
        activation: str | None = None
        kernel_pattern = "dequant_gemm_epilogue"
        extra_kwargs: dict[str, Any] = {}
        if callable(self._packed_fp4_bridge):
            packed_weight, scale, bias, activation, input_features, group_size = self._packed_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            qweight = packed_weight
            kernel_pattern = "fp4_packed_dequant_gemm_epilogue"
            extra_kwargs = {
                "input_features": int(input_features),
                "group_size": int(group_size),
            }
            self.last_weight_source, self.last_weight_representation = (
                self._weight_only_bridge_metadata(packed=True)
            )
            self.last_consumes_packed_weight = True
            self.last_fastpath = "packed_fp4_fused_tilelang_kernel"
        elif callable(self._dense_fp4_bridge):
            qweight, scale, bias, activation = self._dense_fp4_bridge(
                dtype=x.dtype,
                device=x.device,
            )
            self.last_weight_source, self.last_weight_representation = (
                self._weight_only_bridge_metadata(
                    packed=False,
                    direct_dense_bridge=True,
                )
            )
            self.last_consumes_packed_weight = False
            self.last_fastpath = "dense_codes_times_scale"
        else:
            packed_nvfp4_args = self._resolve_packed_nvfp4_args(x)
            if packed_nvfp4_args is not None:
                (
                    packed_weight,
                    scale,
                    bias,
                    activation,
                    input_features,
                    group_size,
                    weight_global_scale,
                ) = packed_nvfp4_args
                qweight = packed_weight
                kernel_pattern = "nvfp4_packed_dequant_gemm_epilogue"
                extra_kwargs = {
                    "input_features": int(input_features),
                    "group_size": int(group_size),
                    "weight_global_scale": weight_global_scale,
                }
                self.last_consumes_packed_weight = True
                self.last_fastpath = "packed_nvfp4_fused_tilelang_kernel"
            else:
                qweight = getattr(self.module, "qweight", None)
                scale = getattr(self.module, "scale", None)
                bias = getattr(self.module, "bias", None)
                activation = getattr(self.module, "activation", None)
                self.last_weight_source = "module_qweight_scale"
                self.last_weight_representation = "dense_qweight_plus_scale"
                self.last_consumes_packed_weight = False
                self.last_fastpath = "dense_codes_times_scale"
        if not isinstance(qweight, torch.Tensor) or (
            kernel_pattern != "dense_linear_epilogue" and not isinstance(scale, torch.Tensor)
        ):
            raise XQTBackendError(
                "TileLang dequant GEMM target requires qweight/scale tensors or a tilelang_dequant_gemm_args bridge"
            )
        tensors = (
            (x, qweight)
            if kernel_pattern == "dense_linear_epilogue" and bias is None
            else (x, qweight, bias)
            if kernel_pattern == "dense_linear_epilogue"
            else (x, qweight, scale)
            if bias is None
            else (x, qweight, scale, bias)
        )
        uses_cuda = all(tensor.is_cuda for tensor in tensors)
        prefers_native_linear = (
            kernel_pattern == "dense_linear_epilogue"
            and uses_cuda
            and self._prefer_native_linear_fastpath(x)
        )
        self.last_execution_mode = (
            "cuda_native_fastpath"
            if prefers_native_linear
            else "cuda_tilelang_entry"
            if uses_cuda
            else "reference_fallback"
        )
        self.last_kernel_pattern = kernel_pattern
        self.last_unpack_stage = (
            "one_time_eager_dequant_cache"
            if kernel_pattern == "dense_linear_epilogue"
            else
            "tilelang_fused_gemm_kernel"
            if uses_cuda and kernel_pattern in {"fp4_packed_dequant_gemm_epilogue", "nvfp4_packed_dequant_gemm_epilogue"}
            else "eager_reference_fallback"
            if kernel_pattern in {"fp4_packed_dequant_gemm_epilogue", "nvfp4_packed_dequant_gemm_epilogue"}
            else None
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode in {"cuda_tilelang_entry", "cuda_native_fastpath"}
            else "TileLang dequant GEMM kernel requires CUDA tensors; using configured fallback."
        )
        tile_kwargs: dict[str, Any] = {
            "activation": activation,
            **extra_kwargs,
            "block_m": int(self.settings.get("block_m", 64)),
            "block_n": int(
                self.settings.get(
                    "block_n",
                    16 if kernel_pattern == "nvfp4_packed_dequant_gemm_epilogue" else 64,
                )
            ),
            "threads": int(self.settings.get("threads", 128)),
            "num_stages": int(self.settings.get("num_stages", 2)),
            "target_arch": self.settings.get("target_arch"),
            "fallback": self.fallback,
        }
        if kernel_pattern == "dense_linear_epilogue":
            tile_kwargs["block_k"] = int(self.settings.get("block_k", 64))
        if kernel_pattern == "nvfp4_packed_dequant_gemm_epilogue":
            tile_kwargs["block_k"] = int(self.settings.get("block_k", 128))
        if kernel_pattern == "dense_linear_epilogue":
            if self.last_execution_mode in {"cuda_native_fastpath", "reference_fallback"}:
                return self._apply_activation(F.linear(x, qweight, bias), activation)
            if bias is not None:
                return self._run_tilelang_or_reference(
                    kernel_pattern,
                    x,
                    qweight,
                    bias,
                    **tile_kwargs,
                )
            return self._run_tilelang_or_reference(
                kernel_pattern,
                x,
                qweight,
                **tile_kwargs,
            )
        return self._run_tilelang_or_reference(
            kernel_pattern,
            x,
            qweight,
            scale,
            bias,
            **tile_kwargs,
        )

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "native_runtime_fastpath"
            if self.last_execution_mode == "cuda_native_fastpath"
            else
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_tilelang_entry"
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "kernel_constraints": {
                "dtype": "float16",
                "batch_multiple_of_block_m": True,
                "out_features_multiple_of_block_n": True,
                "supported_activations": [None, "gelu", "silu", "relu"],
                "supported_patterns": [
                    "dense_linear_epilogue",
                    "dequant_gemm_epilogue",
                    "fp4_packed_dequant_gemm_epilogue",
                    "nvfp4_packed_dequant_gemm_epilogue",
                ],
                "operator_families": ["linear", "conv", "attention"],
                "supports_fp4_weight_only_linear_bridge": True,
                "supports_packed_fp4_bridge": True,
                "supports_packed_nvfp4_bridge": True,
            },
            "kernel_pattern": self.last_kernel_pattern,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "weight_source": self.last_weight_source,
            "weight_representation": self.last_weight_representation,
            "consumes_packed_weight": self.last_consumes_packed_weight,
            "unpack_stage": self.last_unpack_stage,
            "fusion_status": (
                "tilelang_dense_half_gemm_epilogue"
                if self.last_unpack_stage == "one_time_eager_dequant_cache"
                else
                "single_tilelang_kernel_for_unpack_dequant_gemm_epilogue"
                if self.last_unpack_stage == "tilelang_fused_gemm_kernel"
                else None
            ),
            "epilogue_stage": (
                "torch_bias_activation"
                if self.last_unpack_stage == "one_time_eager_dequant_cache"
                else
                "tilelang_fused_bias_activation"
                if self.last_unpack_stage == "tilelang_fused_gemm_kernel"
                else None
            ),
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }

def _module_has_cuda_state(module: nn.Module) -> bool:
    return any(tensor.is_cuda for tensor in module.parameters()) or any(
        tensor.is_cuda for tensor in module.buffers()
    )


def _attach_tilelang_execution_metadata(
    module: nn.Module,
    metadata: Mapping[str, Any],
) -> nn.Module:
    setattr(module, "_xqt_tilelang_execution_metadata", dict(metadata))
    return module


def _infer_module_runtime_spec(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    saw_only_float32 = False
    for tensor in tuple(module.parameters()) + tuple(module.buffers()):
        device = device or tensor.device
        if not tensor.is_floating_point():
            continue
        dtype = tensor.dtype
        if dtype != torch.float32:
            return device, dtype
        saw_only_float32 = True
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        return device, torch.float32
    return device, torch.float16 if saw_only_float32 else dtype


def _has_explicit_xqt_dequant_linear_bridge(module: nn.Module) -> bool:
    return bool(
        callable(getattr(module, "tilelang_packed_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_dequant_gemm_args", None))
        or (
            hasattr(module, "mx_precision")
            and callable(getattr(module, "tilelang_dense_linear_args", None))
        )
    )


def _wrap_direct_or_member(
    target_model: nn.Module,
    *,
    module_type: type[nn.Module],
    member_name: str,
    make_wrapper: Callable[[nn.Module], nn.Module],
    error_message: str,
    nested_member: bool = False,
) -> nn.Module:
    """Wrap a direct target or the first supported member without changing topology."""

    if isinstance(target_model, module_type):
        return make_wrapper(target_model)
    member = getattr(target_model, member_name, None)
    if isinstance(member, module_type):
        copied = copy.deepcopy(target_model)
        setattr(copied, member_name, make_wrapper(member))
        return copied
    for child_name, child in target_model.named_children():
        if nested_member:
            nested = getattr(child, member_name, None)
            if not isinstance(nested, module_type):
                continue
            copied = copy.deepcopy(target_model)
            copied_child = copied.get_submodule(child_name)
            setattr(copied_child, member_name, make_wrapper(getattr(copied_child, member_name)))
            return copied
        if isinstance(child, module_type):
            copied = copy.deepcopy(target_model)
            setattr(copied, child_name, make_wrapper(child))
            return copied
    raise XQTBackendError(error_message)


def _native_dense_metadata(
    *,
    dtype: str,
    selected_fastpath: str,
    weight_source: str,
    fallback: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_mode": "cuda_native_fastpath",
        "execution_reason": None,
        "kernel_kind": "native_runtime_fastpath",
        "kernel_constraints": {
            "dtype": dtype,
            "batch_multiple_of_block_m": True,
            "out_features_multiple_of_block_n": True,
            "supported_activations": [None, "gelu", "silu", "relu"],
            "supported_patterns": [
                "dense_linear_epilogue",
                "dequant_gemm_epilogue",
                "fp4_packed_dequant_gemm_epilogue",
                "nvfp4_packed_dequant_gemm_epilogue",
            ],
            "operator_families": ["linear", "conv", "attention"],
            "supports_fp4_weight_only_linear_bridge": True,
            "supports_packed_fp4_bridge": True,
            "supports_packed_nvfp4_bridge": True,
        },
        "kernel_pattern": "dense_linear_epilogue",
        "operator_family": "linear",
        "selected_fastpath": selected_fastpath,
        "weight_source": weight_source,
        "weight_representation": "dense_dequantized_weight_cache",
        "consumes_packed_weight": False,
        "unpack_stage": "one_time_eager_dequant_cache",
        "fusion_status": "tilelang_dense_half_gemm_epilogue",
        "epilogue_stage": "torch_bias_activation",
        "fallback": fallback,
        "settings": dict(settings),
    }


def _is_dequant_linear_target(module: nn.Module) -> bool:
    return bool(
        infer_nvfp4_tensor_layout(module) is not None
        or callable(getattr(module, "tilelang_dense_linear_args", None))
        or callable(getattr(module, "tilelang_packed_nvfp4_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_packed_dequant_gemm_args", None))
        or callable(getattr(module, "tilelang_dequant_gemm_args", None))
        or all(hasattr(module, name) for name in ("qweight", "scale"))
    )


def _build_tilelang_dequant_candidate(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
    settings: Mapping[str, Any],
) -> nn.Module:
    inferred_layout = infer_nvfp4_tensor_layout(target_model)
    is_sm89 = str(settings.get("target_arch") or "") == "sm_89"
    is_dense_pattern = target.patterns == ["dequant_gemm_epilogue"]
    if (
        is_dense_pattern
        and is_sm89
        and inferred_layout is not None
        and not _has_explicit_xqt_dequant_linear_bridge(target_model)
        and _module_has_cuda_state(target_model)
    ):
        bridge = bridge_module_to_nvfp4_linear_shared(target_model)
        if bridge is not None:
            device, dtype = _infer_module_runtime_spec(target_model)
            weight, bias, _ = bridge.tilelang_dense_linear_args(dtype=dtype, device=device)
            linear = nn.Linear(
                bridge.input_features,
                bridge.output_features,
                bias=bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(weight)
                if bias is not None and linear.bias is not None:
                    linear.bias.copy_(bias)
            metadata = _native_dense_metadata(
                dtype=str(dtype),
                selected_fastpath="eager_dense_native_linear_module",
                weight_source="auto_inferred_nvfp4_dense_cache_bridge",
                fallback=target.fallback,
                settings=settings,
            )
            return _attach_tilelang_execution_metadata(
                _TileLangEagerDenseLinearModule(linear, metadata=metadata),
                metadata,
            )
    if (
        is_dense_pattern
        and is_sm89
        and callable(getattr(target_model, "tilelang_dense_linear_args", None))
        and not _has_explicit_xqt_dequant_linear_bridge(target_model)
        and hasattr(target_model, "dense_weight")
        and _module_has_cuda_state(target_model)
    ):
        return _attach_tilelang_execution_metadata(
            target_model,
            _native_dense_metadata(
                dtype="float16",
                selected_fastpath="delegated_native_dense_linear",
                weight_source="module_dense_weight",
                fallback=target.fallback,
                settings=settings,
            ),
        )
    if _is_dequant_linear_target(target_model):
        return _TileLangDequantGemmWrapper(
            target_model,
            fallback=target.fallback,
            settings=dict(settings),
        )
    for child_name, child in target_model.named_children():
        if _is_dequant_linear_target(child):
            copied = copy.deepcopy(target_model)
            copied_child = copied.get_submodule(child_name)
            setattr(
                copied,
                child_name,
                _TileLangDequantGemmWrapper(
                    copied_child,
                    fallback=target.fallback,
                    settings=dict(settings),
                ),
            )
            return copied
    raise XQTBackendError(
        "TileLang dequant GEMM target requires a module with qweight/scale tensors or a tilelang_dequant_gemm_args bridge"
    )


def build_tilelang_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Build a TileLang candidate for one supported target pattern."""

    patterns = target.patterns or ["attention"]
    settings = dict(target.tilelang)
    settings["preferred_patterns"] = list(patterns)
    if patterns == ["attention"]:
        from xqt import nn as xqt_nn

        if isinstance(target_model, xqt_nn.Attention):
            return _TileLangXqtAttentionWrapper(
                target_model,
                fallback=target.fallback,
                settings=settings,
            )
        for child_name, child in target_model.named_children():
            if isinstance(child, xqt_nn.Attention):
                wrapped = copy.deepcopy(target_model)
                setattr(
                    wrapped,
                    child_name,
                    _TileLangXqtAttentionWrapper(
                        getattr(wrapped, child_name),
                        fallback=target.fallback,
                        settings=settings,
                    ),
                )
                return wrapped
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.MultiheadAttention,
            member_name="attention",
            make_wrapper=lambda module: _TileLangAttentionWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message=(
                "TileLang attention target requires xqt.nn.Attention, "
                "nn.MultiheadAttention, or a module with an attention submodule"
            ),
            nested_member=True,
        )
    if patterns == ["conv"]:
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.Conv2d,
            member_name="conv",
            make_wrapper=lambda module: _TileLangConvWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang conv target requires nn.Conv2d or a module with a Conv2d child",
        )
    if patterns == ["conv3d_1x1x1"]:
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.Conv3d,
            member_name="conv",
            make_wrapper=lambda module: _TileLangConv3dWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang conv3d_1x1x1 target requires nn.Conv3d or a module with a Conv3d child",
        )
    if patterns in (["linear"], ["linear_marlin"]):
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.Linear,
            member_name="linear",
            make_wrapper=lambda module: _TileLangLinearWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang linear target requires nn.Linear or a module with a Linear child",
        )
    if patterns == ["norm"]:
        return _wrap_direct_or_member(
            target_model,
            module_type=nn.LayerNorm,
            member_name="norm",
            make_wrapper=lambda module: _TileLangNormWrapper(
                module, fallback=target.fallback, settings=settings
            ),
            error_message="TileLang norm target requires nn.LayerNorm or a module with a LayerNorm child",
        )
    if patterns in (
        ["dequant_gemm_epilogue"],
        ["fp4_packed_dequant_gemm_epilogue"],
        ["nvfp4_packed_dequant_gemm_epilogue"],
    ):
        return _build_tilelang_dequant_candidate(target_model, target, settings)
    raise XQTBackendError(
        "built-in TileLang executor currently supports attention, conv, conv3d_1x1x1, linear, norm, and dequant_gemm_epilogue patterns"
    )


__all__ = [
    "_TileLangAttentionWrapper",
    "_TileLangConv3dWrapper",
    "_TileLangConvWrapper",
    "_TileLangDequantGemmWrapper",
    "_TileLangEagerDenseLinearModule",
    "_TileLangLinearWrapper",
    "_TileLangNormWrapper",
    "_TileLangXqtAttentionWrapper",
    "build_tilelang_candidate_model",
]
