"""FLUX.2 klein NVFP4 compile, CUDA Graph, warmup, and forward helpers."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import torch
from torch import nn

from xqt.core.errors import XQTBackendError
from xqt.kernels.wrappers import OperatorOptimizationTargetPlan
from xqt.kernels.wrappers.compile_backend import compile_with_torch
from xqt.kernels.wrappers.runtime import (
    capture_cuda_graph_with_static_state,
    cuda_graph_tensor_signature,
    replay_cuda_graph_tensor_callable,
)

from .types import (
    Flux2KleinNVFP4CompiledTransformerResult,
    Flux2KleinNVFP4CudaGraphTransformerResult,
    normalize_flux2_klein_nvfp4_engine,
)


def _flux2_forward_kwargs(
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
    joint_attention_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "guidance": guidance,
        "joint_attention_kwargs": (
            None if joint_attention_kwargs is None else dict(joint_attention_kwargs)
        ),
        "return_dict": False,
    }


def _flux2_dynamic_inputs(
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    dynamic_inputs = [
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
    ]
    if guidance is not None:
        dynamic_inputs.append(guidance)
    return tuple(dynamic_inputs)


def _split_flux2_dynamic_inputs(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if len(runtime_args) not in {5, 6}:
        raise XQTBackendError(
            "FLUX.2 transformer CUDA Graph replay expects 5 or 6 tensor inputs"
        )
    guidance = runtime_args[5] if len(runtime_args) == 6 else None
    return (
        runtime_args[0],
        runtime_args[1],
        runtime_args[2],
        runtime_args[3],
        runtime_args[4],
        guidance,
    )


def _forward_flux2_klein_nvfp4_transformer_once(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        output = transformer(
            **_flux2_forward_kwargs(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        )
    if not isinstance(output, tuple) or not output:
        raise XQTBackendError("FLUX.2 transformer forward must return a non-empty tuple")
    first = output[0]
    if not isinstance(first, torch.Tensor):
        raise XQTBackendError("FLUX.2 transformer forward[0] must be a tensor")
    return first


def _flux2_cuda_graph_signature(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(cuda_graph_tensor_signature(tensor) for tensor in runtime_args)


class _Flux2KleinNVFP4CudaGraphModule(nn.Module):
    """Callable whole-transformer CUDA Graph wrapper with strict signature checks."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        graph_state: Mapping[str, Any],
        input_signature: tuple[tuple[Any, ...], ...],
        joint_attention_kwargs: Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self._graph_state = graph_state
        self._input_signature = input_signature
        self._joint_attention_kwargs = (
            None if joint_attention_kwargs is None else dict(joint_attention_kwargs)
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: Mapping[str, Any] | None = None,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        if return_dict:
            raise XQTBackendError("FLUX.2 CUDA Graph helper only supports return_dict=False")
        if joint_attention_kwargs is not None and dict(joint_attention_kwargs) != dict(
            self._joint_attention_kwargs or {}
        ):
            raise XQTBackendError(
                "FLUX.2 CUDA Graph replay requires the same joint_attention_kwargs used at capture time"
            )
        runtime_args = _flux2_dynamic_inputs(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )
        runtime_signature = _flux2_cuda_graph_signature(runtime_args)
        if runtime_signature != self._input_signature:
            raise XQTBackendError(
                "FLUX.2 CUDA Graph replay requires matching shape/stride/dtype/device inputs"
            )
        with torch.no_grad():
            output = replay_cuda_graph_tensor_callable(self._graph_state, runtime_args)
        return (output,)


def compile_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    engine_name: str | None = None,
    materialized_target_count: int = 0,
    compile_engine: str = "inductor",
    mode: str | None = None,
    fullgraph: bool = False,
    dynamic: bool = False,
    options: Mapping[str, Any] | None = None,
) -> Flux2KleinNVFP4CompiledTransformerResult:
    """Compile a FLUX.2 klein transformer for steady-state inference."""

    if mode not in {None, "default"} and options:
        raise XQTBackendError(
            "torch.compile in PyTorch 2.12 does not allow mode and options at the same time"
        )
    compile_plan = OperatorOptimizationTargetPlan(
        name="flux2_klein_nvfp4_transformer_compile",
        engine="torch_compile",
        options={
            "engine": compile_engine,
            **(dict(options) if options is not None else {}),
        },
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    compiled_model, compile_time_ms = compile_with_torch(transformer, compile_plan)
    return Flux2KleinNVFP4CompiledTransformerResult(
        model=compiled_model,
        engine=None if engine_name is None else normalize_flux2_klein_nvfp4_engine(engine_name),
        materialized_target_count=int(materialized_target_count),
        compile_engine=str(compile_engine),
        compile_mode=None if mode in {None, "default"} else str(mode),
        compile_time_ms=float(compile_time_ms),
        warmup_iterations=0,
        warmup_time_ms=0.0,
    )


def capture_flux2_klein_nvfp4_transformer_cuda_graph(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: Mapping[str, Any] | None = None,
    engine_name: str | None = None,
    materialized_target_count: int = 0,
    warmup_iterations: int = 6,
) -> Flux2KleinNVFP4CudaGraphTransformerResult:
    """Capture a fixed-shape whole-transformer CUDA Graph replay path."""

    runtime_args = _flux2_dynamic_inputs(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
    )
    if not runtime_args:
        raise XQTBackendError("CUDA Graph capture requires tensor inputs")
    if not all(tensor.is_cuda for tensor in runtime_args):
        raise XQTBackendError("FLUX.2 CUDA Graph capture requires CUDA tensor inputs")
    signature = _flux2_cuda_graph_signature(runtime_args)
    capture_start = perf_counter()

    def _capture_body(*dynamic_runtime_args: torch.Tensor) -> torch.Tensor:
        (
            capture_hidden_states,
            capture_encoder_hidden_states,
            capture_timestep,
            capture_img_ids,
            capture_txt_ids,
            capture_guidance,
        ) = _split_flux2_dynamic_inputs(dynamic_runtime_args)
        return _forward_flux2_klein_nvfp4_transformer_once(
            transformer,
            hidden_states=capture_hidden_states,
            encoder_hidden_states=capture_encoder_hidden_states,
            timestep=capture_timestep,
            img_ids=capture_img_ids,
            txt_ids=capture_txt_ids,
            guidance=capture_guidance,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    graph_state = capture_cuda_graph_with_static_state(
        runtime_args,
        body=_capture_body,
        warmup=warmup_iterations,
    )
    capture_time_ms = float((perf_counter() - capture_start) * 1000.0)
    wrapped = _Flux2KleinNVFP4CudaGraphModule(
        transformer=transformer,
        graph_state=graph_state,
        input_signature=signature,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    return Flux2KleinNVFP4CudaGraphTransformerResult(
        model=wrapped,
        engine=None if engine_name is None else normalize_flux2_klein_nvfp4_engine(engine_name),
        materialized_target_count=int(materialized_target_count),
        graph_state=graph_state,
        input_signature=signature,
        warmup_iterations=int(warmup_iterations),
        capture_time_ms=capture_time_ms,
    )


def warmup_flux2_klein_nvfp4_transformer(
    transformer: nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    guidance: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    warmup_iterations: int = 6,
    sync_cuda: bool = True,
) -> float:
    """Run explicit warmup for a FLUX.2 klein transformer forward path."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if warmup_iterations == 0:
        return 0.0

    def _forward_once() -> object:
        return transformer(
            **_flux2_forward_kwargs(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        )

    start = perf_counter()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _forward_once()
        if sync_cuda and any(tensor.is_cuda for tensor in (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)):
            torch.cuda.synchronize(hidden_states.device)
    return float((perf_counter() - start) * 1000.0)
