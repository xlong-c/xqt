"""Reference-guarded CuTile and CuTe DSL linear materialization."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.runtime.bridges.nvfp4 import (
    NVFP4LinearBridge,
    bridge_module_to_nvfp4_linear,
    bridge_module_to_nvfp4_linear_shared,
    infer_nvfp4_tensor_layout,
)

from .backends.cutile import (
    cutile_available,
    get_cutile_kernel_spec,
    run_cutile_kernel,
)
from .backends.cute_dsl import (
    get_cute_dsl_kernel_spec,
    run_cute_dsl_kernel,
)
from .types import OperatorOptimizationTargetPlan


class _ReferenceGuardedLinearWrapper(nn.Module):
    """Execute CuTile or CuTe DSL linear paths with an explicit reference guard."""

    _REFERENCE_ONLY_PRODUCTION_STATUSES = {"", "metadata_only", "reference_guarded"}

    def __init__(
        self,
        module: nn.Module,
        *,
        engine: str,
        fallback: str,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        if engine not in {"cutile", "cute_dsl"}:
            raise XQTBackendError(f"unsupported reference-guarded engine: {engine}")
        self.module = module
        self.engine = engine
        self.fallback = fallback
        self.settings = dict(settings)
        self.last_execution_mode = "not_run"
        self.last_execution_reason: str | None = None
        self.last_weight_source = "not_run"
        self.last_weight_representation = "unknown"
        self.last_consumes_packed_weight = False
        self.last_unpack_stage: str | None = None
        self.last_kernel_pattern = self._default_kernel_pattern()
        self.last_operator_family = "linear"
        self.last_fastpath = "none"
        self._cached_nvfp4_bridge: NVFP4LinearBridge | None = None
        self._dense_linear_bridge = getattr(self.module, "tilelang_dense_linear_args", None)
        self._packed_nvfp4_bridge = getattr(
            self.module,
            "tilelang_packed_nvfp4_dequant_gemm_args",
            None,
        )
        self._packed_fp4_bridge = getattr(
            self.module,
            "tilelang_packed_dequant_gemm_args",
            None,
        )
        if self._packed_nvfp4_bridge is None and self._packed_fp4_bridge is None:
            inferred_bridge = bridge_module_to_nvfp4_linear_shared(self.module)
            if inferred_bridge is not None:
                self._cached_nvfp4_bridge = inferred_bridge

    def _preferred_patterns(self) -> list[str]:
        patterns = self.settings.get("preferred_patterns")
        if isinstance(patterns, list):
            return [str(pattern) for pattern in patterns]
        return [self._default_kernel_pattern()]

    def _default_kernel_pattern(self) -> str:
        if self.engine == "cute_dsl":
            return "gemm_epilogue"
        return "dense_linear_epilogue"

    def _engine_display_name(self) -> str:
        return "CuTe DSL" if self.engine == "cute_dsl" else "CuTile"

    def _get_kernel_spec(self, pattern: str) -> Any:
        if self.engine == "cute_dsl":
            return get_cute_dsl_kernel_spec(pattern)
        return get_cutile_kernel_spec(pattern)

    def _run_engine_kernel(
        self,
        pattern: str,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            if self.engine == "cute_dsl":
                return run_cute_dsl_kernel(
                    pattern,
                    *args,
                    fallback=self.fallback,
                    **kwargs,
                )
            return run_cutile_kernel(
                pattern,
                *args,
                fallback=self.fallback,
                **kwargs,
            )
        except Exception as exc:
            if self.fallback != "eager":
                raise
            spec = self._get_kernel_spec(pattern)
            allowed = set(inspect.signature(spec.reference).parameters)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in allowed
            }
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"{self._engine_display_name()} {pattern} runtime fallback: {exc}"
            )
            self.last_fastpath = "eager_reference_fallback"
            return spec.reference(*args, **filtered_kwargs)

    def _resolved_target_arch(self, x: torch.Tensor) -> str | None:
        target_arch = self.settings.get("target_arch")
        if isinstance(target_arch, str) and target_arch:
            return target_arch
        if x.is_cuda:
            major, minor = torch.cuda.get_device_capability(x.device)
            return f"sm_{major}{minor}"
        return None

    def _resolved_nvfp4_bridge(self) -> NVFP4LinearBridge | None:
        if callable(self._packed_fp4_bridge):
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

    def _resolve_dense_linear_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, str | None] | None:
        if callable(self._dense_linear_bridge):
            self.last_weight_source = f"{self.engine}_dense_cache_bridge"
            self.last_weight_representation = "dense_dequantized_weight_cache"
            return self._dense_linear_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is not None:
            self.last_weight_source = "auto_inferred_nvfp4_dense_cache_bridge"
            self.last_weight_representation = "dense_dequantized_weight_cache"
            return bridge.tilelang_dense_linear_args(dtype=x.dtype, device=x.device)
        if isinstance(self.module, nn.Linear):
            self.last_weight_source = "torch_linear_parameter"
            self.last_weight_representation = "dense_weight"
            bias = self.module.bias
            return (
                self.module.weight.to(dtype=x.dtype, device=x.device),
                None if bias is None else bias.to(dtype=x.dtype, device=x.device),
                None,
            )
        dense_quant = self._resolve_dense_quant_args(x)
        if dense_quant is not None:
            qweight, scale, bias, activation = dense_quant
            weight_scale = scale.to(dtype=x.dtype, device=x.device)
            if weight_scale.ndim == 1:
                weight_scale = weight_scale.unsqueeze(-1)
            weight = qweight.to(dtype=x.dtype, device=x.device) * weight_scale
            self.last_weight_source = "module_qweight_scale_dense_cache"
            self.last_weight_representation = "dense_dequantized_weight"
            return (
                weight,
                None if bias is None else bias.to(dtype=x.dtype, device=x.device),
                activation,
            )
        return None

    def _resolve_dense_quant_args(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, str | None] | None:
        qweight = getattr(self.module, "qweight", None)
        scale = getattr(self.module, "scale", None)
        if not isinstance(qweight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            return None
        bias = getattr(self.module, "bias", None)
        activation = getattr(self.module, "activation", None)
        self.last_weight_source = "module_qweight_scale"
        self.last_weight_representation = "dense_qweight_plus_scale"
        return (
            qweight.to(dtype=x.dtype, device=x.device),
            scale.to(dtype=x.dtype, device=x.device),
            bias if isinstance(bias, torch.Tensor) else None,
            activation if isinstance(activation, str) else None,
        )

    def _resolve_packed_nvfp4_args(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        None,
        int,
        int,
        torch.Tensor | None,
    ] | None:
        if callable(self._packed_nvfp4_bridge):
            self.last_weight_source = "compressed_tensors_nvfp4_packed_bridge"
            self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
            return self._packed_nvfp4_bridge(dtype=x.dtype, device=x.device)
        bridge = self._resolved_nvfp4_bridge()
        if bridge is None:
            return None
        self.last_weight_source = "auto_inferred_nvfp4_packed_bridge"
        self.last_weight_representation = "packed_nvfp4_e2m1_plus_group_scale"
        return bridge.tilelang_packed_nvfp4_dequant_gemm_args(
            dtype=x.dtype,
            device=x.device,
        )

    def _select_cutile_pattern(self) -> str:
        patterns = self._preferred_patterns()
        if (
            "nvfp4_packed_dequant_gemm_epilogue" in patterns
            and self._cutile_pattern_has_runtime_kernel(
                "nvfp4_packed_dequant_gemm_epilogue"
            )
            and (
                callable(self._packed_nvfp4_bridge)
                or self._resolved_nvfp4_bridge() is not None
            )
        ):
            return "nvfp4_packed_dequant_gemm_epilogue"
        if (
            "nvfp4_packed_dequant_gemm_epilogue" in patterns
            and (
                callable(self._dense_linear_bridge)
                or self._resolved_nvfp4_bridge() is not None
            )
        ):
            return "dense_linear_epilogue"
        if (
            "fp4_packed_dequant_gemm_epilogue" in patterns
            and callable(self._packed_fp4_bridge)
        ):
            return "fp4_packed_dequant_gemm_epilogue"
        if (
            "dequant_gemm_epilogue" in patterns
            and self._resolve_dense_quant_args_for_selection()
        ):
            return "dequant_gemm_epilogue"
        if "dense_linear_epilogue" in patterns:
            return "dense_linear_epilogue"
        if "linear" in patterns:
            return "linear"
        return "dense_linear_epilogue"

    def _cutile_pattern_has_runtime_kernel(self, pattern: str) -> bool:
        if not cutile_available():
            return False
        try:
            metadata = self._get_kernel_spec(pattern).metadata
        except Exception:
            return False
        production_status = str(metadata.get("production_status", "")).lower()
        fusion_status = str(metadata.get("fusion_status", "")).lower()
        return (
            production_status not in self._REFERENCE_ONLY_PRODUCTION_STATUSES
            and "reference_guarded" not in fusion_status
        )

    @staticmethod
    def _flatten_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if x.ndim == 0:
            raise XQTBackendError("linear engines require at least one input dimension")
        prefix_shape = tuple(x.shape[:-1])
        return x.reshape(-1, x.shape[-1]), prefix_shape

    @staticmethod
    def _restore_flattened_output(
        output: torch.Tensor,
        prefix_shape: tuple[int, ...],
    ) -> torch.Tensor:
        return output.reshape(*prefix_shape, output.shape[-1])

    def _resolve_dense_quant_args_for_selection(self) -> bool:
        return isinstance(getattr(self.module, "qweight", None), torch.Tensor) and isinstance(
            getattr(self.module, "scale", None),
            torch.Tensor,
        )

    def _prepare_execution_state(
        self,
        *,
        x: torch.Tensor,
        pattern: str,
        consumes_packed_weight: bool,
        unpack_stage: str | None,
        fastpath: str,
    ) -> None:
        self.last_kernel_pattern = pattern
        self.last_consumes_packed_weight = consumes_packed_weight
        self.last_unpack_stage = unpack_stage
        self.last_fastpath = fastpath
        if x.is_cuda:
            self.last_execution_mode = f"cuda_{self.engine}_entry"
            self.last_execution_reason = None
        else:
            self.last_execution_mode = "reference_fallback"
            self.last_execution_reason = (
                f"{self._engine_display_name()} {pattern} requires CUDA tensors; using configured fallback."
            )

    def _forward_cute_dsl(self, x: torch.Tensor) -> torch.Tensor:
        dense_args = self._resolve_dense_linear_args(x)
        if dense_args is None:
            raise XQTBackendError(
                "CuTe DSL inference target requires nn.Linear, qweight/scale, or an NVFP4 dense bridge"
            )
        weight, bias, activation = dense_args
        self._prepare_execution_state(
            x=x,
            pattern="gemm_epilogue",
            consumes_packed_weight=False,
            unpack_stage="one_time_eager_dequant_cache",
            fastpath="cute_dsl_dense_gemm_epilogue",
        )
        return self._run_engine_kernel(
            "gemm_epilogue",
            x,
            weight,
            bias,
            activation=activation,
            tile_shape=tuple(self.settings.get("tile_shape", (128, 128, 64))),
            cluster_shape=self.settings.get("cluster_shape"),
        )

    def _forward_cutile_packed_nvfp4(self, x: torch.Tensor) -> torch.Tensor:
        packed_args = self._resolve_packed_nvfp4_args(x)
        if packed_args is None:
            raise XQTBackendError("CuTile NVFP4 target requires a packed NVFP4 bridge")
        (
            packed_weight,
            scale,
            bias,
            activation,
            input_features,
            group_size,
            weight_global_scale,
        ) = packed_args
        flat_x, prefix_shape = self._flatten_input(x)
        self._prepare_execution_state(
            x=x,
            pattern="nvfp4_packed_dequant_gemm_epilogue",
            consumes_packed_weight=True,
            unpack_stage=(
                "cutile_reference_guarded_unpack"
                if x.is_cuda
                else "eager_reference_fallback"
            ),
            fastpath="packed_nvfp4_cutile_reference_guarded_kernel",
        )
        output = self._run_engine_kernel(
            "nvfp4_packed_dequant_gemm_epilogue",
            flat_x,
            packed_weight,
            scale,
            bias,
            input_features=int(input_features),
            group_size=int(group_size),
            weight_global_scale=weight_global_scale,
            activation=activation,
            threads=int(self.settings.get("threads", 128)),
            target_arch=self._resolved_target_arch(x),
        )
        return self._restore_flattened_output(output, prefix_shape)

    def _forward_cutile_dequant(self, x: torch.Tensor) -> torch.Tensor:
        dense_quant = self._resolve_dense_quant_args(x)
        if dense_quant is None:
            raise XQTBackendError("CuTile dequant GEMM target requires qweight/scale tensors")
        qweight, scale, bias, activation = dense_quant
        flat_x, prefix_shape = self._flatten_input(x)
        self._prepare_execution_state(
            x=x,
            pattern="dequant_gemm_epilogue",
            consumes_packed_weight=False,
            unpack_stage="cutile_reference_guarded_dequant",
            fastpath="cutile_dequant_gemm_epilogue",
        )
        output = self._run_engine_kernel(
            "dequant_gemm_epilogue",
            flat_x,
            qweight,
            scale,
            bias,
            activation=activation,
            threads=int(self.settings.get("threads", 128)),
            target_arch=self._resolved_target_arch(x),
        )
        return self._restore_flattened_output(output, prefix_shape)

    def _forward_cutile_dense(self, x: torch.Tensor, pattern: str) -> torch.Tensor:
        dense_args = self._resolve_dense_linear_args(x)
        if dense_args is None:
            raise XQTBackendError(
                "CuTile dense Linear target requires nn.Linear, qweight/scale, or an NVFP4 dense bridge"
            )
        weight, bias, activation = dense_args
        flat_x, prefix_shape = self._flatten_input(x)
        kernel_pattern = "linear" if pattern == "linear" else "dense_linear_epilogue"
        if kernel_pattern == "linear":
            activation = None
        self._prepare_execution_state(
            x=x,
            pattern=kernel_pattern,
            consumes_packed_weight=False,
            unpack_stage="one_time_eager_dequant_cache",
            fastpath=f"cutile_{kernel_pattern}",
        )
        kernel_kwargs: dict[str, Any] = {
            "threads": int(self.settings.get("threads", 128)),
            "target_arch": self._resolved_target_arch(x),
        }
        if kernel_pattern != "linear":
            kernel_kwargs["activation"] = activation
        output = self._run_engine_kernel(
            kernel_pattern,
            flat_x,
            weight,
            bias,
            **kernel_kwargs,
        )
        return self._restore_flattened_output(output, prefix_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.engine == "cute_dsl":
            return self._forward_cute_dsl(x)
        pattern = self._select_cutile_pattern()
        if pattern == "nvfp4_packed_dequant_gemm_epilogue":
            return self._forward_cutile_packed_nvfp4(x)
        if pattern == "dequant_gemm_epilogue":
            return self._forward_cutile_dequant(x)
        return self._forward_cutile_dense(x, pattern)

    def execution_metadata(self) -> dict[str, Any]:
        kernel_kind = (
            "reference_guarded_cuda_entry"
            if self.last_execution_mode.startswith("cuda_")
            else "reference_fallback"
            if self.last_execution_mode == "reference_fallback"
            else "unknown"
        )
        try:
            kernel_metadata = dict(self._get_kernel_spec(self.last_kernel_pattern).metadata)
        except Exception:
            kernel_metadata = {}
        return {
            "execution_mode": self.last_execution_mode,
            "execution_reason": self.last_execution_reason,
            "kernel_kind": kernel_kind,
            "kernel_constraints": {
                "dtype": "float16",
                "supported_patterns": list(self._preferred_patterns()),
                "operator_families": ["linear"],
                "supports_packed_nvfp4_bridge": self.engine == "cutile",
                "supports_dense_nvfp4_cache_bridge": True,
            },
            "kernel_pattern": self.last_kernel_pattern,
            "operator_family": self.last_operator_family,
            "selected_fastpath": self.last_fastpath,
            "weight_source": self.last_weight_source,
            "weight_representation": self.last_weight_representation,
            "consumes_packed_weight": self.last_consumes_packed_weight,
            "unpack_stage": self.last_unpack_stage,
            "fusion_status": kernel_metadata.get(
                "fusion_status",
                f"{self.engine}_reference_guarded_gemm_epilogue",
            ),
            "epilogue_stage": kernel_metadata.get(
                "epilogue_stage",
                "torch_bias_activation",
            ),
            "fallback": self.fallback,
            "settings": dict(self.settings),
        }


def supports_reference_guarded_linear_engine(module: nn.Module) -> bool:
    """Return whether a module exposes one supported guarded-linear protocol."""

    if isinstance(module, nn.Linear):
        return True
    if infer_nvfp4_tensor_layout(module) is not None:
        return True
    if callable(getattr(module, "tilelang_dense_linear_args", None)):
        return True
    if callable(getattr(module, "tilelang_packed_nvfp4_dequant_gemm_args", None)):
        return True
    if callable(getattr(module, "tilelang_packed_dequant_gemm_args", None)):
        return True
    return all(hasattr(module, name) for name in ("qweight", "scale"))


def build_reference_guarded_linear_candidate_model(
    target_model: nn.Module,
    target: OperatorOptimizationTargetPlan,
    *,
    engine: str,
) -> nn.Module:
    """Wrap one compatible target or child with a guarded linear executor."""

    settings = dict(target.cutile if engine == "cutile" else target.cute_dsl)
    settings["preferred_patterns"] = list(target.patterns or [])
    if supports_reference_guarded_linear_engine(target_model):
        return _ReferenceGuardedLinearWrapper(
            target_model,
            engine=engine,
            fallback=target.fallback,
            settings=settings,
        )
    for child_name, child in target_model.named_children():
        if supports_reference_guarded_linear_engine(child):
            candidate = copy.deepcopy(target_model)
            wrapped_child = candidate.get_submodule(child_name)
            setattr(
                candidate,
                child_name,
                _ReferenceGuardedLinearWrapper(
                    wrapped_child,
                    engine=engine,
                    fallback=target.fallback,
                    settings=settings,
                ),
            )
            return candidate
    raise XQTBackendError(
        f"{engine} inference target requires nn.Linear, qweight/scale tensors, or an NVFP4 bridge"
    )


__all__ = [
    "_ReferenceGuardedLinearWrapper",
    "build_reference_guarded_linear_candidate_model",
    "supports_reference_guarded_linear_engine",
]
