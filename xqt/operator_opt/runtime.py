"""CUDA Graph runtime helpers shared by XQT operator paths."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from xqt.core.errors import XQTBackendError


DEFAULT_CUDA_GRAPH_WARMUP = 2


def cuda_graph_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return the fixed tensor contract required for CUDA Graph replay."""

    return (
        tuple(int(dim) for dim in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _make_static_cuda_graph_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_strided(
        size=tuple(int(dim) for dim in tensor.shape),
        stride=tuple(int(stride) for stride in tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def capture_cuda_graph_with_static_state(
    dynamic_args: tuple[torch.Tensor, ...],
    *,
    body: Callable[..., torch.Tensor],
    warmup: int,
) -> dict[str, Any]:
    """Capture a fixed-shape CUDA Graph with static input storage."""

    if not dynamic_args:
        raise XQTBackendError("CUDA Graph capture requires at least one dynamic tensor")
    if not all(tensor.is_cuda for tensor in dynamic_args):
        raise XQTBackendError("CUDA Graph capture requires CUDA tensor inputs")
    static_dynamic_args = tuple(_make_static_cuda_graph_tensor(tensor) for tensor in dynamic_args)
    for static_arg, runtime_arg in zip(static_dynamic_args, dynamic_args):
        static_arg.copy_(runtime_arg)
    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            body(*static_dynamic_args)
        torch.cuda.synchronize(dynamic_args[0].device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = body(*static_dynamic_args)
    return {
        "graph": graph,
        "static_args": static_dynamic_args,
        "static_output": static_output,
    }


def replay_cuda_graph_tensor_callable(
    state: Mapping[str, Any],
    runtime_args: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Copy runtime inputs into a captured CUDA Graph and replay it."""

    static_args = state.get("static_args")
    graph = state.get("graph")
    static_output = state.get("static_output")
    if (
        not isinstance(static_args, tuple)
        or graph is None
        or not isinstance(static_output, torch.Tensor)
        or len(static_args) != len(runtime_args)
    ):
        raise XQTBackendError("invalid CUDA Graph state")
    for static_arg, runtime_arg in zip(static_args, runtime_args):
        if not isinstance(static_arg, torch.Tensor) or not isinstance(runtime_arg, torch.Tensor):
            raise XQTBackendError("CUDA Graph state contains non-tensor inputs")
        static_arg.copy_(runtime_arg)
    graph.replay()
    return static_output


__all__ = [
    "DEFAULT_CUDA_GRAPH_WARMUP",
    "capture_cuda_graph_with_static_state",
    "cuda_graph_tensor_signature",
    "replay_cuda_graph_tensor_callable",
]
