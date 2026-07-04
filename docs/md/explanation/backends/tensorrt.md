# TensorRT 后端

本文介绍 NVIDIA TensorRT 后端. 在 XQT 中, TensorRT 当前是 P0 导出后端, 主要用于 ONNX 到 TensorRT engine 的构建, engine inspection, runtime session, plugin preflight 和 trtexec 性能解析.

## 后端介绍

TensorRT 是 NVIDIA 的深度学习推理优化 SDK. 它面向 NVIDIA GPU, 通常接收 ONNX 或 TensorRT network definition, 经 builder 优化后生成 serialized engine, 再由 runtime 加载执行.

TensorRT 和普通 runtime 最大的区别是: 它会针对目标 GPU, shape profile, precision 和 builder 配置生成专用 engine. 这带来高性能, 也带来更强的环境绑定:

- engine 通常和 TensorRT 版本, CUDA / driver, GPU 架构相关.
- 动态 shape 需要 optimization profile.
- 不支持的 op 通常需要 graph rewrite, plugin 或 fallback 到其他链路.
- INT8 / FP8 等低精度依赖硬件, TensorRT 版本和量化方式.

## 适合的使用场景

- NVIDIA GPU 上追求低延迟或高吞吐推理.
- 模型结构和输入 shape 范围相对稳定.
- 可以提前 build engine, 并缓存 build 产物.
- 需要使用 FP16, BF16, INT8, FP8 等 TensorRT 支持的 precision.
- 能接受为特定 GPU / profile 生成专用部署产物.

## 不适合的场景

- 需要一个跨硬件, 跨版本通用的模型文件.
- 目标机器没有稳定 NVIDIA driver / CUDA / TensorRT 环境.
- 输入 shape 完全不可预期, 且不想维护 profile.
- 模型包含大量 TensorRT 不支持的 op, 又不准备写 plugin.
- 只想做一次格式转换, 不做数值验证和性能验证.

## 核心概念

- builder: 把 network 编译成 engine 的组件.
- engine / plan: build 后的 serialized 可执行产物.
- runtime: 反序列化 engine 并创建 execution context.
- execution context: 执行推理, 设置动态输入 shape, 绑定 tensor.
- optimization profile: 动态 shape 的 min / opt / max 范围.
- tactic: TensorRT 为某个 layer 选择的具体 kernel 策略.
- plugin: TensorRT 原生不支持的 op 或自定义融合实现.
- engine inspector: 用于查看 engine layer, IO tensor 和 fusion 信息.
- trtexec: TensorRT 官方命令行工具, 常用于 build, benchmark, dump profile.

## 核心 API

源码位置: `xqt/export/tensorrt.py`

构建和命令:

- `build_trtexec_command(onnx_path, engine_path, precision=None, profiles=None, trtexec_path="trtexec", extra_args=None, plugin_libraries=None, serialize_plugin_libraries=True)`: 生成 `trtexec` 命令.
- `build_tensorrt_engine(...)`: 构建 TensorRT engine, 支持 `trtexec` 或 Python API 路径.
- `TensorRTBuildResult`: engine build 的产物 metadata.

plugin 和 inspection:

- `validate_tensorrt_plugin_libraries(plugin_libraries)`: 检查 plugin shared library 是否存在, 可加载.
- `inspect_tensorrt_engine(engine_path, plugin_libraries=None)`: 读取 engine inspector 信息.
- `summarize_tensorrt_engine_inspector(...)`: 汇总 layer, IO tensor, quantize / dequantize / fusion 信息.
- `TensorRTPluginValidationResult`, `TensorRTPluginLibraryCheck`, `TensorRTEngineInspectorSummary`.

runtime 和 benchmark:

- `create_tensorrt_runtime_session(engine_path, device="cuda:0", plugin_libraries=None)`: 创建可复用 runtime session.
- `execute_tensorrt_engine(engine_path, inputs, device="cuda:0", plugin_libraries=None)`: 一次性执行 engine.
- `execute_tensorrt_session(session, inputs)`: 复用 session 执行.
- `benchmark_tensorrt_engine(...)`: 对 engine 做 runtime benchmark.
- `TensorRTRuntimeSession`, `TensorRTRuntimeExecutionResult`, `TensorRTRuntimeBenchmarkResult`.

性能解析:

- `parse_trtexec_performance(output)`: 解析 `trtexec` 输出中的 throughput, latency, host latency, GPU compute time.
- `evaluate_tensorrt_performance_thresholds(metrics, thresholds)`: 用 `_min` / `_max` 阈值检查性能.
- `TensorRTPerformanceMetrics`, `TensorRTPerformanceThresholdReport`.

## 简单例子

生成 `trtexec` 命令:

```python
from xqt.export import build_trtexec_command


command = build_trtexec_command(
    "artifacts/model.onnx",
    "artifacts/model.plan",
    precision="fp16",
    profiles={
        "input": {
            "min": [1, 3, 224, 224],
            "opt": [8, 3, 224, 224],
            "max": [16, 3, 224, 224],
        }
    },
)
print(" ".join(command))
```

执行 engine:

```python
import torch

from xqt.export import create_tensorrt_runtime_session, execute_tensorrt_session


session = create_tensorrt_runtime_session("artifacts/model.plan", device="cuda:0")
result = execute_tensorrt_session(
    session,
    inputs={"input": torch.randn(8, 3, 224, 224, device="cuda")},
)
print(result.output_shapes)
```

解析性能输出:

```python
from xqt.export import parse_trtexec_performance, evaluate_tensorrt_performance_thresholds


metrics = parse_trtexec_performance(trtexec_stdout)
report = evaluate_tensorrt_performance_thresholds(
    metrics,
    {"latency_mean_ms_max": 3.0, "throughput_qps_min": 1000.0},
)
```

## 后端实现注意点

- ONNX parse 成功不代表 engine build 成功.
- engine build 成功不代表数值误差可接受.
- 动态 shape 必须记录 profile, 且 benchmark 输入要落在 profile 范围内.
- engine 不要当成跨机器通用格式, report 必须记录 TensorRT, CUDA, driver, GPU 名称和 compute capability.
- plugin library 要记录路径, checksum, 是否序列化到 engine, 以及加载顺序.
- FP8 / FP4 / INT4 等低精度路径必须结合硬件能力和 TensorRT support matrix 验证.
- `trtexec` benchmark 要记录 warmup, duration / iterations, streams, precision, input shape.

## 常见问题

- `Failed to parse ONNX`: ONNX graph 包含 TensorRT parser 不支持的 op 或属性.
- `dynamic network requires explicit profiles`: 输入含动态维度但没有给 profile.
- `engine deserialize failed`: TensorRT 版本, GPU 架构或 plugin 环境不匹配.
- `unresolved tensors`: runtime 设置的输入 shape 不完整或超出 profile.
- 性能不稳定: build tactic, timing cache, GPU clock, warmup, batch size 都会影响结果.
- plugin 找不到: shared library 路径, ABI, TensorRT 版本或符号注册有问题.

## 官方资料

- TensorRT documentation: <https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html>
- TensorRT architecture overview: <https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html>
- TensorRT `trtexec`: <https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html>
- TensorRT support matrix: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html>

