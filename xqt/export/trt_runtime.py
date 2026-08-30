"""TensorRT runtime session creation, execution, and benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from xqt.core.errors import XQTBackendError

from .trt_diagnostics import (
    _import_tensorrt,
    _load_tensorrt_plugin_libraries,
    inspect_tensorrt_engine,
)
from .trt_types import (
    TensorRTRuntimeBenchmarkResult,
    TensorRTRuntimeExecutionResult,
    TensorRTRuntimeSession,
)


def _torch_dtype_from_trt(dtype: Any, trt: Any) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        getattr(trt, "bfloat16", None): torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    resolved = mapping.get(dtype)
    if resolved is None:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return resolved


def create_tensorrt_runtime_session(
    engine_path: str | Path,
    *,
    device: str = "cuda:0",
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTRuntimeSession:
    """Create a reusable TensorRT runtime session."""

    engine = Path(engine_path)
    if not engine.is_file():
        raise XQTBackendError(f"TensorRT engine file not found: {engine}")

    loaded_plugins = _load_tensorrt_plugin_libraries(plugin_libraries)
    trt = _import_tensorrt()
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
    serialized = runtime.deserialize_cuda_engine(engine.read_bytes())
    if serialized is None:
        raise XQTBackendError(f"Failed to deserialize TensorRT engine: {engine}")
    context = serialized.create_execution_context()
    if context is None:
        raise XQTBackendError("Failed to create TensorRT execution context")
    return TensorRTRuntimeSession(
        engine_path=engine,
        device=device,
        trt=trt,
        runtime=runtime,
        engine=serialized,
        context=context,
        engine_inspector=inspect_tensorrt_engine(
            engine,
            plugin_libraries=plugin_libraries,
        ).to_dict()
        | (
            {"plugin_libraries": loaded_plugins}
            if loaded_plugins
            else {}
        ),
    )


def execute_tensorrt_engine(
    engine_path: str | Path,
    *,
    inputs: Mapping[str, torch.Tensor],
    device: str = "cuda:0",
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTRuntimeExecutionResult:
    """Execute a TensorRT engine once with explicit input tensors."""

    session = create_tensorrt_runtime_session(
        engine_path,
        device=device,
        plugin_libraries=plugin_libraries,
    )
    return execute_tensorrt_session(session, inputs=inputs)


def execute_tensorrt_session(
    session: TensorRTRuntimeSession,
    *,
    inputs: Mapping[str, torch.Tensor],
) -> TensorRTRuntimeExecutionResult:
    """Execute a reusable TensorRT session once with explicit input tensors."""

    torch_device = torch.device(session.device)
    prepared_inputs: dict[str, torch.Tensor] = {}
    for name, tensor in inputs.items():
        tensor_on_device = tensor.to(device=torch_device)
        shape_tuple = tuple(int(dim) for dim in tensor_on_device.shape)
        ok = session.context.set_input_shape(str(name), shape_tuple)
        if ok is False:
            raise XQTBackendError(f"TensorRT rejected input shape for {name}: {shape_tuple}")
        prepared_inputs[str(name)] = tensor_on_device

    unresolved = session.context.infer_shapes()
    if unresolved:
        raise XQTBackendError(
            f"TensorRT shape inference has unresolved tensors: {list(unresolved)}"
        )

    output_tensors: dict[str, torch.Tensor] = {}
    output_shapes: dict[str, list[int]] = {}
    bindings: list[int] = []
    for name in session.engine:
        mode = session.engine.get_tensor_mode(name)
        if mode == session.trt.TensorIOMode.INPUT:
            if name not in prepared_inputs:
                raise XQTBackendError(f"TensorRT input tensor missing: {name}")
            bindings.append(int(prepared_inputs[name].data_ptr()))
            continue
        shape = tuple(int(dim) for dim in session.context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise XQTBackendError(f"TensorRT output shape is unresolved for {name}: {shape}")
        dtype = _torch_dtype_from_trt(session.engine.get_tensor_dtype(name), session.trt)
        output_tensor = torch.empty(shape, dtype=dtype, device=torch_device)
        output_tensors[str(name)] = output_tensor
        output_shapes[str(name)] = [int(dim) for dim in shape]
        bindings.append(int(output_tensor.data_ptr()))

    ok = session.context.execute_v2(bindings)
    if not ok:
        raise XQTBackendError("TensorRT engine execution failed")

    return TensorRTRuntimeExecutionResult(
        engine_path=session.engine_path,
        backend="python_api",
        device=session.device,
        input_shapes={
            str(name): [int(dim) for dim in tensor.shape]
            for name, tensor in prepared_inputs.items()
        },
        output_tensors=output_tensors,
        output_shapes=output_shapes,
        metadata={"engine_inspector": dict(session.engine_inspector)},
    )


def _shape_to_tuple(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def benchmark_tensorrt_engine(
    engine_path: str | Path,
    *,
    input_shapes: Mapping[str, Sequence[int]],
    warmup: int = 10,
    iterations: int = 50,
    device: str = "cuda:0",
    fill_random: bool = True,
    plugin_libraries: Optional[Sequence[str | Path]] = None,
) -> TensorRTRuntimeBenchmarkResult:
    """Benchmark a TensorRT engine using the TensorRT Python runtime."""

    engine = Path(engine_path)
    session = create_tensorrt_runtime_session(
        engine,
        device=device,
        plugin_libraries=plugin_libraries,
    )

    torch_device = torch.device(device)
    inputs: dict[str, torch.Tensor] = {}
    for name, shape in input_shapes.items():
        shape_tuple = _shape_to_tuple(shape)
        if fill_random:
            tensor = torch.rand(shape_tuple, dtype=torch.float32, device=torch_device)
        else:
            tensor = torch.zeros(shape_tuple, dtype=torch.float32, device=torch_device)
        inputs[str(name)] = tensor

    def run_once() -> object:
        return execute_tensorrt_session(session, inputs=inputs).output_tensors

    from xqt.kernels.wrappers.bench import benchmark_callable

    latency = benchmark_callable(
        run_once,
        warmup=warmup,
        iterations=iterations,
        sync_cuda=True,
        device=device,
    ).to_dict()
    execution = execute_tensorrt_session(session, inputs=inputs)
    return TensorRTRuntimeBenchmarkResult(
        engine_path=engine,
        backend="python_api",
        device=device,
        input_shapes={
            str(name): [int(dim) for dim in _shape_to_tuple(shape)]
            for name, shape in input_shapes.items()
        },
        latency=latency,
        output_shapes=execution.output_shapes,
        metadata={
            "fill_random": fill_random,
            **execution.metadata,
        },
    )
