# MNN 后端

本文介绍 MNN 后端. 在 XQT 中, MNN 当前是 P2 adapter, 主要通过 `MNNConvert` 把 ONNX 模型转换成 `.mnn` 文件.

## 后端介绍

MNN 是 Alibaba 开源的轻量深度学习推理框架. 它面向移动端和边缘端, 也覆盖一部分服务端场景. MNN 提供模型转换工具, runtime, 多 backend 执行路径和一些量化 / benchmark 工具.

典型链路:

```text
PyTorch -> ONNX -> MNNConvert -> model.mnn -> MNN Interpreter / Session
```

MNN 的执行 backend 随平台和 build option 变化, 常见包括 CPU, OpenCL, Vulkan, Metal, NNAPI, CoreML, CUDA 等. 实际可用能力要以目标版本和目标平台为准.

## 适合的使用场景

- Android / iOS 端侧部署.
- 希望一个轻量 runtime 覆盖 CPU / GPU / NPU 等路径.
- 已有 ONNX / TensorFlow / TFLite / TorchScript 模型需要转到移动端.
- 使用 Alibaba MNN 生态中的 conversion, quantization, benchmark, demo 工具.
- 对 binary size, thread, precision mode 有明确控制需求.

## 不适合的场景

- NVIDIA GPU 服务端极致性能.
- 模型依赖大量 MNN 不支持的 custom op.
- 不准备固定 MNN 版本和 converter 参数.
- 不准备对齐端侧前处理, layout 和 precision mode.

## 核心概念

- `MNNConvert`: 模型转换命令行工具.
- `.mnn`: MNN 模型文件.
- Interpreter: 加载 MNN 模型.
- Session: 一次 runtime 执行配置和推理上下文.
- backend type: CPU, OpenCL, Vulkan, Metal, NNAPI, CoreML 等.
- precision mode: runtime 选择 FP32 / FP16 / low precision 的策略.
- tensor layout: NCHW, NHWC, NC4HW4 等布局会影响性能和结果对齐.

## 核心 API

源码位置: `xqt/export/mobile.py`

- `build_mnnconvert_command(onnx_path, mnn_path, converter_path="MNNConvert", framework="ONNX", extra_args=None)`: 构造 MNNConvert 命令.
- `export_mnn_from_onnx(onnx_path, mnn_path, converter_path="MNNConvert", framework="ONNX", extra_args=None, timeout=None, dry_run=False)`: 执行 ONNX 到 `.mnn` 转换.
- `CommandExportResult`: 命令型转换结果.

## 简单例子

```python
from xqt.export import build_mnnconvert_command, export_mnn_from_onnx


command = build_mnnconvert_command(
    "artifacts/model.onnx",
    "artifacts/mnn/model.mnn",
)
print(" ".join(command))

result = export_mnn_from_onnx(
    "artifacts/model.onnx",
    "artifacts/mnn/model.mnn",
    dry_run=True,
)
```

带额外转换参数:

```python
result = export_mnn_from_onnx(
    "artifacts/model.onnx",
    "artifacts/mnn/model.mnn",
    extra_args=["--bizCode", "xqt"],
)
```

## 后端实现注意点

- converter 支持和 runtime backend 支持不是同一件事.
- 需要记录 MNN 版本, converter 参数, framework, input / output tensor 名.
- 端侧 session 配置会显著影响性能, 例如 thread, backend type, precision mode.
- `.mnn` 产物通常不包含完整业务前后处理.
- 不同 backend 的 layout, dtype 和 fusion 行为可能不同.
- LLM, diffusion 等新路径可能依赖额外工具或特定 build option.

## 常见问题

- `MNNConvert` 不存在: converter 未安装或不在 PATH.
- 转换成功但 runtime 失败: runtime backend 不支持某个 op 或 layout.
- 数值差异: precision mode, layout conversion, channel order 或前处理不一致.
- 性能差: 线程数, backend type, memory mode 和输入 shape 未调优.
- 版本错位: 示例参数来自新版本, 目标机器安装的是旧版本.

## 官方资料

- MNN GitHub: <https://github.com/alibaba/MNN>
- MNN documentation: <https://mnn-docs.readthedocs.io/>

