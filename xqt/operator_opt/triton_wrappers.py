"""Triton candidate materialization and execution metadata."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Callable

import torch
from torch import nn

from xqt.core.errors import XQTBackendError

from .triton_dequant_wrappers import (
    _TritonDequantGemmWrapper,
    build_triton_dequant_candidate_model,
)
from .backends.triton import get_triton_kernel_spec, run_triton_kernel
from .kernels.triton.gemm import (
    resolve_triton_bf16_gemm_schedule,
    resolve_triton_fp16_gemm_schedule,
)
from .runtime import (
    DEFAULT_CUDA_GRAPH_WARMUP,
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)
from .types import OperatorOptimizationTargetPlan
from .wrappers._common import _resolved_target_arch, _target_arch_mismatch_reason


_TRITON_LINEAR_PATTERNS = frozenset({"linear", "gemm_fp16", "gemm_bf16"})


class _TritonLinearWrapper(nn.Module):
    """Inference-only FP16/BF16 Linear wrapper with explicit weight layout."""

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
        self.last_kernel_pattern = "unknown"
        self.last_kernel_dtype = "unknown"
        self.last_kernel_schedule: dict[str, int | str | None] | None = None
        self._precision = self._resolve_configured_precision()
        self._linear_fastpath = str(
            self.settings.get("linear_fastpath", "eager")
        ).strip().lower()
        if self._linear_fastpath not in {"eager", "graph"}:
            raise XQTBackendError(
                "Triton linear linear_fastpath must be 'eager' or 'graph'"
            )
        self._cuda_graph_warmup = int(
            self.settings.get("cuda_graph_warmup", DEFAULT_CUDA_GRAPH_WARMUP)
        )
        if self._cuda_graph_warmup < 0:
            raise XQTBackendError(
                "Triton linear cuda_graph_warmup must be non-negative"
            )
        self.settings["linear_fastpath"] = self._linear_fastpath
        self.settings["cuda_graph_warmup"] = self._cuda_graph_warmup
        self._weight_layout = str(
            self.settings.get("weight_layout", "transpose_stride")
        ).strip()
        if self._weight_layout not in {"transpose_stride", "prepacked_kn"}:
            raise XQTBackendError(
                "Triton linear weight_layout must be 'transpose_stride' or "
                "'prepacked_kn'"
            )
        self._target_arch = self.settings.get("target_arch")
        self._block_m = self._optional_int_setting("block_m")
        self._block_n = self._optional_int_setting("block_n")
        self._block_k = self._optional_int_setting("block_k")
        self._group_m = self._optional_int_setting("group_m")
        self._num_warps = self._optional_int_setting("num_warps")
        self._num_stages = self._optional_int_setting("num_stages")
        self._cached_schedule_kwargs = self._schedule_kwargs()
        self._cached_schedule_signature = tuple(
            sorted(self._cached_schedule_kwargs.items())
        )
        self._last_schedule_key: tuple[Any, ...] | None = None
        self._last_schedule_signature: tuple[tuple[str, Any], ...] = ()
        self._resolved_arch_device: torch.device | None = None
        self._fp16_kernel = get_triton_kernel_spec("gemm_fp16").kernel
        self._bf16_kernel = get_triton_kernel_spec("gemm_bf16").kernel
        self._active_dtype: torch.dtype | None = None
        self._active_pattern: str | None = None
        self._active_kernel: Any = None
        self._active_kernel_kwargs: dict[str, Any] = {}
        self.last_graph_state = "disabled"
        self.last_graph_reason: str | None = None
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._graph_parameter_signature: tuple[Any, ...] | None = None
        self._graph_last_state: dict[str, Any] | None = None
        self._graph_last_fast_signature: tuple[Any, ...] | None = None
        self._graph_replay_shape: torch.Size | None = None
        self._graph_replay_stride: tuple[int, ...] | None = None
        self._graph_replay_dtype: torch.dtype | None = None
        self._graph_replay_device: torch.device | None = None
        self._graph_replay_weight: torch.Tensor | None = None
        self._graph_replay_weight_version: int | None = None
        self._graph_replay_bias: torch.Tensor | None = None
        self._graph_replay_bias_version: int | None = None
        self._graph_replay_static_input: torch.Tensor | None = None
        self._graph_replay_graph: Any = None
        self._graph_replay_output: torch.Tensor | None = None
        self.register_buffer("_prepared_weight_kn", None, persistent=False)
        self._prepared_weight_source_id: int | None = None
        self._prepared_weight_source_version: int | None = None
        self._prepack_refreshes = 0
        if self._weight_layout == "prepacked_kn" and linear.weight.device.type != "meta":
            self._refresh_prepacked_weight()
        if linear.weight.dtype in {torch.float16, torch.bfloat16}:
            self._configure_runtime_dtype(linear.weight.dtype)

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> _TritonLinearWrapper:
        module = super()._apply(fn, recurse=recurse)
        self._invalidate_graph_cache("module device or dtype changed")
        self._prepared_weight_source_id = None
        self._prepared_weight_source_version = None
        self._active_dtype = None
        self._active_pattern = None
        self._active_kernel = None
        self._active_kernel_kwargs = {}
        self._last_schedule_key = None
        self._last_schedule_signature = ()
        self._resolved_arch_device = None
        self.last_execution_mode = "not_run"
        self.last_execution_reason = "module device or dtype changed"
        self.last_fastpath = "none"
        self.last_kernel_pattern = "unknown"
        self.last_kernel_dtype = "unknown"
        return module

    def _invalidate_graph_cache(self, reason: str) -> None:
        self._graph_cache.clear()
        self._graph_parameter_signature = None
        self._graph_last_state = None
        self._graph_last_fast_signature = None
        self._graph_replay_shape = None
        self._graph_replay_stride = None
        self._graph_replay_dtype = None
        self._graph_replay_device = None
        self._graph_replay_weight = None
        self._graph_replay_weight_version = None
        self._graph_replay_bias = None
        self._graph_replay_bias_version = None
        self._graph_replay_static_input = None
        self._graph_replay_graph = None
        self._graph_replay_output = None
        self.last_graph_state = "disabled"
        self.last_graph_reason = reason

    def _resolve_configured_precision(self) -> str:
        patterns = self.settings.get("preferred_patterns")
        pattern_precision = None
        if patterns == ["gemm_fp16"]:
            pattern_precision = "fp16"
        elif patterns == ["gemm_bf16"]:
            pattern_precision = "bf16"
        precision = str(self.settings.get("precision", pattern_precision or "auto"))
        precision = precision.strip().lower()
        if precision not in {"auto", "fp16", "bf16"}:
            raise XQTBackendError(
                "Triton linear precision must be 'auto', 'fp16', or 'bf16'"
            )
        if pattern_precision is not None and precision != pattern_precision:
            raise XQTBackendError(
                f"Triton linear pattern {patterns[0]!r} requires precision "
                f"{pattern_precision!r}"
            )
        return precision

    def _optional_int_setting(self, name: str) -> int | None:
        value = self.settings.get(name)
        return None if value is None else int(value)

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("Triton linear target requires at least 1D input")
        if int(x.shape[-1]) <= 0:
            raise XQTBackendError(
                "Triton linear target requires a non-empty trailing feature dimension"
            )
        prefix_shape = tuple(int(dim) for dim in x.shape[:-1])
        return x.reshape(-1, int(x.shape[-1])), prefix_shape

    @staticmethod
    def _restore_output(
        output: torch.Tensor,
        prefix_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if len(prefix_shape) == 1 and prefix_shape[0] == int(output.shape[0]):
            return output
        return output.reshape(*prefix_shape, int(output.shape[-1]))

    def _refresh_prepacked_weight(self) -> torch.Tensor:
        weight = self.linear.weight
        if weight.device.type == "meta":
            raise XQTBackendError("Triton linear cannot prepack a meta-device weight")
        if self._graph_cache or self._graph_parameter_signature is not None:
            self._invalidate_graph_cache("prepacked weight refreshed")
        prepared = weight.detach().t().contiguous()
        self._prepared_weight_kn = prepared
        self._prepared_weight_source_id = id(weight)
        self._prepared_weight_source_version = self._tensor_version(weight)
        self._prepack_refreshes += 1
        return prepared

    def _runtime_weight(self) -> tuple[torch.Tensor, bool]:
        if self._weight_layout == "transpose_stride":
            return self.linear.weight, True
        weight = self.linear.weight
        if (
            self._prepared_weight_kn is None
            or self._prepared_weight_source_id != id(weight)
            or self._prepared_weight_source_version
            != self._tensor_version(weight)
        ):
            return self._refresh_prepacked_weight(), False
        return self._prepared_weight_kn, False

    def _configure_runtime_dtype(self, dtype: torch.dtype) -> None:
        self.last_kernel_dtype = str(dtype).removeprefix("torch.")
        if dtype == torch.float16:
            runtime_precision = "fp16"
            pattern = "gemm_fp16"
            kernel = self._fp16_kernel
        elif dtype == torch.bfloat16:
            runtime_precision = "bf16"
            pattern = "gemm_bf16"
            kernel = self._bf16_kernel
        else:
            raise XQTBackendError(
                "Triton linear supports only float16 or bfloat16 runtime tensors"
            )
        if self._precision != "auto" and runtime_precision != self._precision:
            raise XQTBackendError(
                f"Triton linear configured precision {self._precision!r} does not "
                f"match runtime dtype {self.last_kernel_dtype!r}"
            )
        self._active_dtype = dtype
        self._active_pattern = pattern
        self._active_kernel = kernel
        self._active_kernel_kwargs = {
            "transpose_b": self._weight_layout == "transpose_stride",
            "accum_dtype": torch.float32,
            "output_dtype": dtype,
            **self._cached_schedule_kwargs,
        }
        if pattern in {"gemm_fp16", "gemm_bf16"} and isinstance(
            self._target_arch, str
        ):
            self._active_kernel_kwargs["target_arch"] = self._target_arch

    def _runtime_target_arch(self, x: torch.Tensor) -> str | None:
        if isinstance(self._target_arch, str) and self._target_arch:
            return self._target_arch
        if self._resolved_arch_device != x.device:
            self._target_arch = _resolved_target_arch(self.settings, x)
            self._resolved_arch_device = x.device
        return self._target_arch

    def _schedule_kwargs(self) -> dict[str, int]:
        values = {
            "block_m": self._block_m,
            "block_n": self._block_n,
            "block_k": self._block_k,
            "group_m": self._group_m,
            "num_warps": self._num_warps,
            "num_stages": self._num_stages,
        }
        return {name: value for name, value in values.items() if value is not None}

    def _record_schedule(
        self,
        pattern: str,
        m: int,
        *,
        target_arch: str | None,
    ) -> None:
        schedule_key = (
            pattern,
            int(m),
            target_arch,
            self.linear.bias is not None,
        )
        if self._last_schedule_key == schedule_key:
            return
        if pattern in {"gemm_fp16", "gemm_bf16"}:
            schedule_kwargs = {
                "m": int(m),
                "n": int(self.linear.out_features),
                "k": int(self.linear.in_features),
                "has_bias": self.linear.bias is not None,
                "activation": None,
                "block_m": self._block_m,
                "block_n": self._block_n,
                "block_k": self._block_k,
                "group_m": self._group_m,
                "num_warps": self._num_warps,
                "num_stages": self._num_stages,
                "target_arch": target_arch,
            }
            if pattern == "gemm_fp16":
                schedule = resolve_triton_fp16_gemm_schedule(
                    **schedule_kwargs,
                    transpose_b=self._weight_layout == "transpose_stride",
                )
            else:
                schedule = resolve_triton_bf16_gemm_schedule(**schedule_kwargs)
            self.last_kernel_schedule = schedule.to_dict()
            self._last_schedule_signature = tuple(
                sorted(self.last_kernel_schedule.items())
            )
            self._last_schedule_key = schedule_key
            return
        self.last_kernel_schedule = {
            "block_m": 128 if self._block_m is None else self._block_m,
            "block_n": 128 if self._block_n is None else self._block_n,
            "block_k": 32 if self._block_k is None else self._block_k,
            "group_m": 8 if self._group_m is None else self._group_m,
            "num_warps": 4 if self._num_warps is None else self._num_warps,
            "num_stages": 3 if self._num_stages is None else self._num_stages,
            "target_arch": target_arch,
            "preset": "default_or_explicit",
        }
        self._last_schedule_signature = tuple(
            sorted(self.last_kernel_schedule.items())
        )
        self._last_schedule_key = schedule_key

    @staticmethod
    def _tensor_version(tensor: torch.Tensor | None) -> int | None:
        if tensor is None:
            return None
        try:
            return int(tensor._version)
        except RuntimeError:
            return None

    @staticmethod
    def _tensor_state_signature(
        tensor: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        if tensor is None:
            return (None,)
        data_ptr = None if tensor.device.type == "meta" else int(tensor.data_ptr())
        return (
            id(tensor),
            _TritonLinearWrapper._tensor_version(tensor),
            data_ptr,
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )

    def _schedule_signature(self) -> tuple[tuple[str, Any], ...]:
        return self._last_schedule_signature

    def _linear_graph_fast_signature(
        self,
        x: torch.Tensor,
        *,
        pattern: str,
        target_arch: str | None,
        runtime_weight: torch.Tensor,
        transpose_b: bool,
    ) -> tuple[Any, ...]:
        weight = self.linear.weight
        bias = self.linear.bias
        return (
            x.shape,
            x.stride(),
            x.dtype,
            x.device,
            pattern,
            self._weight_layout,
            target_arch,
            bool(transpose_b),
            self._cached_schedule_signature,
            self._last_schedule_key,
            id(weight),
            self._tensor_version(weight),
            int(weight.data_ptr()),
            id(bias),
            self._tensor_version(bias),
            None if bias is None else int(bias.data_ptr()),
            id(runtime_weight),
            int(runtime_weight.data_ptr()),
        )

    def _install_graph_replay_state(
        self,
        state: dict[str, Any],
        x: torch.Tensor,
    ) -> None:
        weight = self.linear.weight
        bias = self.linear.bias
        static_args = state.get("static_args")
        graph = state.get("graph")
        output = state.get("static_output")
        if (
            not isinstance(static_args, tuple)
            or len(static_args) != 1
            or not isinstance(static_args[0], torch.Tensor)
            or graph is None
            or not isinstance(output, torch.Tensor)
        ):
            raise XQTBackendError("invalid Triton linear CUDA Graph state")
        self._graph_replay_shape = x.shape
        self._graph_replay_stride = x.stride()
        self._graph_replay_dtype = x.dtype
        self._graph_replay_device = x.device
        self._graph_replay_weight = weight
        self._graph_replay_weight_version = self._tensor_version(weight)
        self._graph_replay_bias = bias
        self._graph_replay_bias_version = self._tensor_version(bias)
        self._graph_replay_static_input = static_args[0]
        self._graph_replay_graph = graph
        self._graph_replay_output = output

    def _try_replay_last_graph(
        self,
        x: torch.Tensor,
    ) -> tuple[bool, torch.Tensor | None]:
        state = self._graph_last_state
        weight = self.linear.weight
        bias = self.linear.bias
        if (
            state is None
            or x.shape != self._graph_replay_shape
            or x.stride() != self._graph_replay_stride
            or x.dtype != self._graph_replay_dtype
            or x.device != self._graph_replay_device
            or weight is not self._graph_replay_weight
            or self._tensor_version(weight) != self._graph_replay_weight_version
            or bias is not self._graph_replay_bias
            or (
                bias is not None
                and self._tensor_version(bias) != self._graph_replay_bias_version
            )
        ):
            return False, None
        try:
            output = self._graph_replay_output
            if output is None:
                raise XQTBackendError("invalid Triton linear CUDA Graph replay state")
            replay_cuda_graph_tensor_callable(state, (x,))
        except Exception as exc:
            reason = f"CUDA Graph replay failed: {type(exc).__name__}: {exc}"
            self._invalidate_graph_cache(reason)
            self.last_graph_state = "fallback_eager"
            return True, None
        self.last_execution_mode = "cuda_graph_triton_entry"
        self.last_execution_reason = None
        self.last_kernel_pattern = str(state["pattern"])
        self.last_kernel_dtype = str(x.dtype).removeprefix("torch.")
        self.last_fastpath = str(state["fastpath"])
        self.last_graph_state = "replayed"
        self.last_graph_reason = None
        return True, output

    def _sync_graph_parameter_signature(
        self,
        runtime_weight: torch.Tensor,
    ) -> tuple[Any, ...]:
        signature = (
            self._tensor_state_signature(self.linear.weight),
            self._tensor_state_signature(self.linear.bias),
            self._tensor_state_signature(runtime_weight),
        )
        if (
            self._graph_parameter_signature is not None
            and self._graph_parameter_signature != signature
        ):
            self._invalidate_graph_cache("linear weight or bias changed")
        self._graph_parameter_signature = signature
        return signature

    def _linear_graph_cache_key(
        self,
        x: torch.Tensor,
        *,
        pattern: str,
        target_arch: str | None,
        runtime_weight: torch.Tensor,
        transpose_b: bool,
    ) -> tuple[Any, ...]:
        parameter_signature = self._sync_graph_parameter_signature(runtime_weight)
        return (
            cuda_graph_tensor_signature(x),
            pattern,
            self._weight_layout,
            target_arch,
            bool(transpose_b),
            self._schedule_signature(),
            parameter_signature,
        )

    def _run_eager_triton(
        self,
        x: torch.Tensor,
        runtime_weight: torch.Tensor,
    ) -> torch.Tensor:
        flat_input, prefix_shape = self._flatten_input(x)
        output = self._active_kernel(
            flat_input,
            runtime_weight,
            self.linear.bias,
            **self._active_kernel_kwargs,
        )
        return self._restore_output(output, prefix_shape)

    def _capture_linear_graph(
        self,
        x: torch.Tensor,
        *,
        runtime_weight: torch.Tensor,
    ) -> dict[str, Any]:
        kernel = self._active_kernel
        kernel_kwargs = dict(self._active_kernel_kwargs)
        bias = self.linear.bias

        def body(x_arg: torch.Tensor) -> torch.Tensor:
            flat_input, prefix_shape = self._flatten_input(x_arg)
            output = kernel(
                flat_input,
                runtime_weight,
                bias,
                **kernel_kwargs,
            )
            return self._restore_output(output, prefix_shape)

        state = capture_cuda_graph_with_static_state(
            (x,),
            body=body,
            warmup=self._cuda_graph_warmup,
        )
        state["kind"] = "triton_linear_full_forward"
        state["output_storage"] = "graph_owned"
        return state

    def _run_graph_triton(
        self,
        x: torch.Tensor,
        *,
        pattern: str,
        target_arch: str | None,
        runtime_weight: torch.Tensor,
        transpose_b: bool,
    ) -> torch.Tensor | None:
        fast_signature = self._linear_graph_fast_signature(
            x,
            pattern=pattern,
            target_arch=target_arch,
            runtime_weight=runtime_weight,
            transpose_b=transpose_b,
        )
        if (
            self._graph_last_state is not None
            and self._graph_last_fast_signature == fast_signature
        ):
            self.last_graph_state = "replayed"
            self.last_graph_reason = None
            try:
                return replay_cuda_graph_tensor_callable(
                    self._graph_last_state,
                    (x,),
                )
            except Exception as exc:
                self._invalidate_graph_cache(
                    f"CUDA Graph replay failed: {type(exc).__name__}: {exc}"
                )
                self.last_graph_state = "fallback_eager"
                return None
        cache_key = self._linear_graph_cache_key(
            x,
            pattern=pattern,
            target_arch=target_arch,
            runtime_weight=runtime_weight,
            transpose_b=transpose_b,
        )
        state = self._graph_cache.get(cache_key)
        if state is None:
            try:
                state = self._capture_linear_graph(
                    x,
                    runtime_weight=runtime_weight,
                )
            except Exception as exc:
                self.last_graph_state = "fallback_eager"
                self.last_graph_reason = (
                    f"CUDA Graph capture failed: {type(exc).__name__}: {exc}"
                )
                return None
            self._graph_cache[cache_key] = state
            self.last_graph_state = "captured"
            self.last_graph_reason = None
        else:
            self.last_graph_state = "replayed"
            self.last_graph_reason = None
        state["fast_signature"] = fast_signature
        state["pattern"] = pattern
        state["fastpath"] = (
            f"triton_{pattern.removeprefix('gemm_')}_linear_"
            f"{self._weight_layout}_cuda_graph"
        )
        self._graph_last_state = state
        self._graph_last_fast_signature = fast_signature
        self._install_graph_replay_state(state, x)
        try:
            return replay_cuda_graph_tensor_callable(state, (x,))
        except Exception as exc:
            reason = f"CUDA Graph replay failed: {type(exc).__name__}: {exc}"
            self._invalidate_graph_cache(reason)
            self.last_graph_state = "fallback_eager"
            return None

    def _reference_or_raise(
        self,
        x: torch.Tensor,
        reason: str,
        *,
        cause: Exception | None = None,
    ) -> torch.Tensor:
        self.last_execution_mode = "reference_fallback"
        self.last_execution_reason = reason
        self.last_fastpath = "eager_reference_fallback"
        if self.fallback != "eager":
            raise XQTBackendError(reason) from cause
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            self.last_kernel_dtype = str(x.dtype).removeprefix("torch.")
            self.last_graph_state = "disabled"
            self.last_graph_reason = "CUDA Graph requires CUDA tensor inputs"
            return self._reference_or_raise(
                x,
                "Triton linear kernel requires CUDA tensors; using configured fallback.",
            )
        mismatch = _target_arch_mismatch_reason(self.settings, x)
        if mismatch is not None:
            self.last_graph_state = "disabled"
            self.last_graph_reason = mismatch
            return self._reference_or_raise(
                x,
                f"Triton linear runtime fallback: {mismatch}",
            )
        if torch.is_grad_enabled() and x.requires_grad:
            self.last_graph_state = "disabled"
            self.last_graph_reason = "autograd_unsupported"
            return self._reference_or_raise(
                x,
                "Triton linear kernel is inference-only; autograd input requires reference fallback.",
            )
        graph_replay_attempted = False
        if self._linear_fastpath == "graph":
            graph_replay_attempted, graph_output = self._try_replay_last_graph(x)
            if graph_output is not None:
                return graph_output
        try:
            if x.ndim == 0:
                raise XQTBackendError(
                    "Triton linear target requires at least 1D input"
                )
            if int(x.shape[-1]) <= 0:
                raise XQTBackendError(
                    "Triton linear target requires a non-empty trailing feature dimension"
                )
            weight = self.linear.weight
            bias = self.linear.bias
            if x.dtype != weight.dtype or (bias is not None and bias.dtype != x.dtype):
                self.last_kernel_dtype = "mixed"
                raise XQTBackendError(
                    "Triton linear requires matching float16 or bfloat16 "
                    "activation, weight, and bias tensors"
                )
            if self._active_dtype != x.dtype:
                self._configure_runtime_dtype(x.dtype)
            pattern = self._active_pattern
            if pattern is None or self._active_kernel is None:
                raise XQTBackendError("Triton linear runtime kernel is not configured")
            runtime_weight, transpose_b = self._runtime_weight()
            target_arch = self._runtime_target_arch(x)
            self._record_schedule(
                pattern,
                int(x.numel()) // int(x.shape[-1]),
                target_arch=target_arch,
            )
            kernel_kwargs = self._active_kernel_kwargs
            if kernel_kwargs["transpose_b"] != transpose_b:
                kernel_kwargs["transpose_b"] = transpose_b
            if (
                pattern in {"gemm_fp16", "gemm_bf16"}
                and kernel_kwargs.get("target_arch") != target_arch
            ):
                kernel_kwargs["target_arch"] = target_arch
            if self._linear_fastpath == "graph":
                if not graph_replay_attempted:
                    output = self._run_graph_triton(
                        x,
                        pattern=pattern,
                        target_arch=target_arch,
                        runtime_weight=runtime_weight,
                        transpose_b=transpose_b,
                    )
                    if output is not None:
                        self.last_execution_mode = "cuda_graph_triton_entry"
                        self.last_execution_reason = None
                        self.last_kernel_pattern = pattern
                        self.last_fastpath = (
                            f"triton_{pattern.removeprefix('gemm_')}_linear_"
                            f"{self._weight_layout}_cuda_graph"
                        )
                        return output
            else:
                self.last_graph_state = "disabled"
                self.last_graph_reason = (
                    "linear_fastpath is not set to graph mode"
                )
            output = self._run_eager_triton(x, runtime_weight)
        except Exception as exc:
            return self._reference_or_raise(
                x,
                f"Triton linear runtime fallback: {type(exc).__name__}: {exc}",
                cause=exc,
            )
        self.last_execution_mode = "cuda_triton_entry"
        self.last_execution_reason = None
        self.last_kernel_pattern = pattern
        self.last_fastpath = f"triton_{pattern.removeprefix('gemm_')}_linear_{self._weight_layout}"
        return output

    def execution_metadata(self) -> dict[str, Any]:
        prepared_bytes = (
            0
            if self._prepared_weight_kn is None
            else int(
                self._prepared_weight_kn.numel()
                * self._prepared_weight_kn.element_size()
            )
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": (
                "cuda_graph_replay"
                if self.last_execution_mode == "cuda_graph_triton_entry"
                else "minimal_cuda_jit"
                if self.last_execution_mode == "cuda_triton_entry"
                else "reference_fallback"
                if self.last_execution_mode == "reference_fallback"
                else "unknown"
            ),
            "operator_family": self.last_operator_family,
            "kernel_pattern": self.last_kernel_pattern,
            "selected_fastpath": self.last_fastpath,
            "kernel_schedule": (
                None
                if self.last_kernel_schedule is None
                else dict(self.last_kernel_schedule)
            ),
            "kernel_constraints": {
                "dtype": self.last_kernel_dtype,
                "supported_dtypes": ["float16", "bfloat16"],
                "supported_patterns": sorted(_TRITON_LINEAR_PATTERNS),
                "operator_families": ["linear"],
                "supports_rank_gte_1_via_batch_flatten": True,
                "inference_only": True,
                "weight_layout": self._weight_layout,
                "prepared_weight_bytes": prepared_bytes,
                "prepack_refreshes": self._prepack_refreshes,
                "supported_linear_fastpaths": ["eager", "graph"],
                "cuda_graph_static_parameters": True,
            },
            "linear_fastpath": self._linear_fastpath,
            "cuda_graph": {
                "state": self.last_graph_state,
                "reason": self.last_graph_reason,
                "cache_size": len(self._graph_cache),
                "output_storage": (
                    "graph_owned"
                    if self.last_graph_state in {"captured", "replayed"}
                    and self._graph_cache
                    else None
                ),
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


class _TritonRMSNormWrapper(nn.Module):
    """Standalone half RMSNorm wrapper for Triton operator targets."""

    def __init__(
        self,
        norm: nn.Module,
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
        self.last_dtype = "unknown"
        self.last_kernel_pattern = "rmsnorm"
        self.last_channel_layout = "last_dim"

    def _supports_rmsnorm(self) -> bool:
        return hasattr(self.norm, "gamma") and hasattr(self.norm, "scale")

    def _rmsnorm_weight(self, x: torch.Tensor) -> torch.Tensor:
        gamma = getattr(self.norm, "gamma", None)
        scale = getattr(self.norm, "scale", None)
        if not isinstance(gamma, torch.Tensor) or scale is None:
            raise XQTBackendError("Triton RMSNorm wrapper requires gamma and scale")
        weight = gamma.to(device=x.device, dtype=x.dtype)
        return weight.reshape(-1).contiguous() * float(scale)

    def _rmsnorm_bias(self, x: torch.Tensor) -> torch.Tensor | None:
        bias = getattr(self.norm, "bias", None)
        if isinstance(bias, torch.Tensor):
            return bias.to(device=x.device, dtype=x.dtype).reshape(-1).contiguous()
        return None

    def _is_channel_first_norm(self, x: torch.Tensor) -> bool:
        channel_first = bool(getattr(self.norm, "channel_first", False))
        return channel_first and x.ndim >= 3

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def _precision_name(self, x: torch.Tensor) -> str:
        if x.dtype == torch.float16:
            return "half"
        if x.dtype == torch.bfloat16:
            return "bf16"
        return str(x.dtype).removeprefix("torch.")

    def _run_triton_or_reference(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._rmsnorm_weight(x)
        bias = self._rmsnorm_bias(x)
        if self._is_channel_first_norm(x):
            kernel_pattern = "rmsnorm_channel_first"
            eps = float(self.settings.get("eps", 1e-12))
            kernel_args: tuple[torch.Tensor, ...] = (x, weight)
            kernel_kwargs = {
                "bias": bias,
                "eps": eps,
                "block_size": int(self.settings.get("block_size", 1024)),
                "sites_per_program": int(self.settings.get("sites_per_program", 4)),
                "num_warps": int(self.settings.get("num_warps", 4)),
                "num_stages": int(self.settings.get("num_stages", 4)),
                "fallback": self.fallback,
            }
            self.last_channel_layout = "channel_first"
        else:
            kernel_pattern = "rmsnorm"
            eps = float(self.settings.get("eps", 1e-6))
            kernel_args = (x, weight)
            kernel_kwargs = {
                "eps": eps,
                "block_size": int(self.settings.get("block_size", 1024)),
                "num_warps": int(self.settings.get("num_warps", 4)),
                "num_stages": int(self.settings.get("num_stages", 4)),
                "fallback": self.fallback,
            }
            self.last_channel_layout = "last_dim"
        self.last_kernel_pattern = kernel_pattern
        try:
            return run_triton_kernel(kernel_pattern, *kernel_args, **kernel_kwargs)
        except Exception as exc:
            if self.fallback != "eager":
                raise
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = f"Triton rmsnorm runtime fallback: {exc}"
            self.last_fastpath = "eager_reference_fallback"
            return self._reference(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._supports_rmsnorm():
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = "module does not expose RMSNorm gamma/scale parameters"
            self.last_fastpath = "eager_reference_fallback"
            self.last_dtype = str(x.dtype).removeprefix("torch.")
            return self._reference(x)
        self.last_dtype = str(x.dtype).removeprefix("torch.")
        precision_name = self._precision_name(x)
        is_channel_first = self._is_channel_first_norm(x)
        self.last_kernel_pattern = (
            "rmsnorm_channel_first" if is_channel_first else "rmsnorm"
        )
        self.last_channel_layout = "channel_first" if is_channel_first else "last_dim"
        self.last_execution_mode = "cuda_triton_entry" if x.is_cuda else "reference_fallback"
        kernel_suffix = "channel_first_norm" if is_channel_first else "rmsnorm"
        self.last_fastpath = (
            f"triton_{precision_name}_{kernel_suffix}"
            if self.last_execution_mode == "cuda_triton_entry"
            else "eager_reference_fallback"
        )
        self.last_execution_reason = (
            None
            if self.last_execution_mode == "cuda_triton_entry"
            else "Triton rmsnorm kernel requires CUDA tensors; using configured fallback."
        )
        if self.last_execution_mode == "cuda_triton_entry":
            return self._run_triton_or_reference(x)
        return self._reference(x)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "minimal_cuda_jit"
            if self.last_execution_mode == "cuda_triton_entry"
            else "reference_fallback"
        )
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "operator_family": self.last_operator_family,
            "kernel_pattern": self.last_kernel_pattern,
            "selected_fastpath": self.last_fastpath,
            "kernel_constraints": {
                "dtype": self.last_dtype,
                "supported_patterns": ["rmsnorm", "rmsnorm_channel_first"],
                "operator_families": ["norm"],
                "normalized_last_dim_only": self.last_channel_layout == "last_dim",
                "supports_channel_first": True,
                "channel_layout": self.last_channel_layout,
                "supported_dtypes": ["float16", "bfloat16"],
            },
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


def supports_triton_rmsnorm(module: nn.Module | None) -> bool:
    """Return whether a module exposes the RMSNorm protocol used by Triton."""

    return module is not None and hasattr(module, "gamma") and hasattr(module, "scale")


def supports_triton_linear(module: nn.Module | None) -> bool:
    """Return whether a module is a dense Linear target supported by Triton."""

    return isinstance(module, nn.Linear)


def _triton_linear_settings(target: OperatorOptimizationTargetPlan) -> dict[str, Any]:
    settings = dict(target.options)
    settings["preferred_patterns"] = list(target.patterns or ["linear"])
    return settings


def _build_triton_linear_candidate(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    settings = _triton_linear_settings(target)

    def wrap(linear: nn.Linear) -> _TritonLinearWrapper:
        return _TritonLinearWrapper(
            linear,
            fallback=target.fallback,
            settings=settings,
        )

    if isinstance(target_model, nn.Linear):
        return wrap(target_model)
    member = getattr(target_model, "linear", None)
    if isinstance(member, nn.Linear):
        candidate = copy.deepcopy(target_model)
        candidate.linear = wrap(candidate.get_submodule("linear"))
        return candidate
    for child_name, child in target_model.named_children():
        if not isinstance(child, nn.Linear):
            continue
        candidate = copy.deepcopy(target_model)
        setattr(candidate, child_name, wrap(candidate.get_submodule(child_name)))
        return candidate
    raise XQTBackendError(
        "Triton linear target requires nn.Linear or a module with a Linear child"
    )


def _feedforward_type() -> type[nn.Module]:
    """Resolve the XQT FeedForward facade without a module-import cycle."""

    from xqt.nn import FeedForward

    return FeedForward


def supports_triton_feedforward(module: nn.Module | None) -> bool:
    """Return whether a module is the XQT FeedForward Triton facade."""

    return module is not None and isinstance(module, _feedforward_type())


def _triton_feedforward_settings(target: OperatorOptimizationTargetPlan) -> dict[str, Any]:
    """Validate and normalize runtime options for a FeedForward candidate."""

    settings = dict(target.options)
    settings["preferred_patterns"] = list(target.patterns or ["feedforward"])
    projection_policies = settings.get("projection_policies")
    if projection_policies is not None and not isinstance(projection_policies, Mapping):
        raise XQTBackendError(
            "Triton feedforward projection_policies must be a mapping when provided"
        )
    return settings


def _build_triton_feedforward_candidate(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Copy and configure an XQT FeedForward for the Triton runtime path."""

    if not supports_triton_feedforward(target_model):
        raise XQTBackendError(
            "Triton feedforward target requires an xqt.nn.FeedForward module"
        )
    settings = _triton_feedforward_settings(target)
    candidate = copy.deepcopy(target_model)
    runtime_options = {
        key: settings[key]
        for key in (
            "activation_dtype",
            "weight_dtype",
            "bias_dtype",
            "mma_dtype",
            "accum_dtype",
            "output_dtype",
            "projection_policies",
        )
        if key in settings
    }
    candidate.configure_runtime(engine="triton", **runtime_options)
    setattr(candidate, "_xqt_triton_target_settings", settings)
    return candidate


def build_triton_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    """Materialize one supported Triton candidate for one target."""

    patterns = target.patterns or ["rmsnorm"]
    settings = dict(target.options)
    settings["preferred_patterns"] = list(patterns)
    dequant_patterns = {
        "gemm_int4_dequant",
        "gemm_mxfp8",
        "gemm_mxfp6",
        "gemm_mxfp4",
        "gemm_nvfp4_packed_dequant",
        "fp4_packed_dequant_gemm_epilogue",
        "nvfp4_packed_dequant_gemm_epilogue",
    }
    if patterns == ["feedforward"]:
        return _build_triton_feedforward_candidate(target_model, target)
    if len(patterns) == 1 and patterns[0] in _TRITON_LINEAR_PATTERNS:
        return _build_triton_linear_candidate(target_model, target)
    if set(patterns).issubset(dequant_patterns):
        return build_triton_dequant_candidate_model(target_model, target)
    if patterns != ["rmsnorm"]:
        raise XQTBackendError(
            "built-in Triton executor currently supports linear, rmsnorm, "
            "feedforward, and low-bit dequant GEMM patterns"
        )
    if supports_triton_rmsnorm(target_model):
        return _TritonRMSNormWrapper(
            target_model,
            fallback=target.fallback,
            settings=settings,
        )
    norm = getattr(target_model, "norm", None)
    if supports_triton_rmsnorm(norm):
        candidate = copy.deepcopy(target_model)
        candidate_norm = candidate.get_submodule("norm")
        candidate.norm = _TritonRMSNormWrapper(
            candidate_norm,
            fallback=target.fallback,
            settings=settings,
        )
        return candidate
    for child_name, child in target_model.named_children():
        if supports_triton_rmsnorm(child):
            candidate = copy.deepcopy(target_model)
            setattr(
                candidate,
                child_name,
                _TritonRMSNormWrapper(
                    candidate.get_submodule(child_name),
                    fallback=target.fallback,
                    settings=settings,
                ),
            )
            return candidate
    raise XQTBackendError(
        "Triton rmsnorm target requires a module exposing gamma/scale parameters or a child module with that interface"
    )


def triton_execution_metadata(model: nn.Module) -> dict[str, Any]:
    """Read Triton wrapper execution metadata from a candidate model tree."""

    if supports_triton_feedforward(model):
        runtime_config = model.runtime_config()
        fallback = runtime_config.get("fallback")
        if isinstance(fallback, Mapping):
            execution_mode = "reference_fallback"
            execution_reason = str(fallback.get("reason") or "Triton runtime fallback")
        else:
            execution_mode = "triton_runtime_configured"
            execution_reason = None
        fusion = runtime_config.get("fusion")
        realized_patterns = (
            list(fusion.get("realized_patterns", []))
            if isinstance(fusion, Mapping)
            else []
        )
        return {
            "execution_mode": execution_mode,
            "execution_reason": execution_reason,
            "kernel_kind": (
                "reference_fallback"
                if execution_mode == "reference_fallback"
                else "triton_composed_runtime"
            ),
            "operator_family": "feedforward",
            "kernel_pattern": "feedforward",
            "selected_fastpath": (
                "eager_reference_fallback"
                if execution_mode == "reference_fallback"
                else "triton_feedforward_runtime"
            ),
            "kernel_constraints": {
                "supported_patterns": ["feedforward"],
                "operator_families": ["feedforward"],
                "realized_patterns": realized_patterns,
            },
            "fallback": "eager",
            "settings": dict(
                getattr(model, "_xqt_triton_target_settings", {})
            ),
            "runtime_config": runtime_config,
        }
    if isinstance(model, _TritonRMSNormWrapper):
        return model.execution_metadata()
    if isinstance(model, _TritonLinearWrapper):
        return model.execution_metadata()
    if isinstance(model, _TritonDequantGemmWrapper):
        return model.execution_metadata()
    norm = getattr(model, "norm", None)
    if isinstance(norm, _TritonRMSNormWrapper):
        return norm.execution_metadata()
    for module in model.modules():
        if isinstance(
            module,
            (
                _TritonLinearWrapper,
                _TritonRMSNormWrapper,
                _TritonDequantGemmWrapper,
            ),
        ):
            return module.execution_metadata()
    return {
        "execution_mode": "unknown",
        "execution_reason": None,
    }


__all__ = [
    "_TritonLinearWrapper",
    "_TritonRMSNormWrapper",
    "build_triton_candidate_model",
    "supports_triton_feedforward",
    "supports_triton_linear",
    "supports_triton_rmsnorm",
    "triton_execution_metadata",
]
