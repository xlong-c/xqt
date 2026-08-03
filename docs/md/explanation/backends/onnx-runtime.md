# ONNX Runtime 后端

本文介绍 ONNX Runtime. 在 XQT 中, ONNX Runtime 主要用于 ONNX correctness diff 和通用 ONNX runtime baseline; 它也可以作为实际部署 runtime, 通过 Execution Provider 接入不同硬件.

## 后端介绍

ONNX Runtime 是加载和执行 ONNX graph 的 runtime. 它会读取 `.onnx`, 做图优化, 内存规划和 kernel 调度. 它最重要的扩展点是 Execution Provider, 简称 EP.

EP 决定图中哪些节点跑在哪个硬件或加速库上. 常见 EP 包括:

- `CPUExecutionProvider`
- `CUDAExecutionProvider`
- `TensorrtExecutionProvider`
- `OpenVINOExecutionProvider`
- `DmlExecutionProvider`
- `CoreMLExecutionProvider`
- `QNNExecutionProvider`

ONNX Runtime 可以把整图或子图交给高优先级 EP, 不支持的部分 fallback 到其他 EP. 这既提高了便携性, 也容易让性能问题被隐藏.

## 适合的使用场景

- 用 CPU EP 建立 ONNX correctness baseline.
- 在同一套 ONNX 模型上切换 CPU / CUDA / TensorRT / OpenVINO 等后端.
- 服务端需要成熟 Python / C++ / C# / Java 等语言绑定.
- 端侧或跨平台项目想保留统一 runtime 抽象.
- 调试 ONNX 导出后的输入输出绑定, dtype 和 shape.

## 不适合的场景

- 不检查实际 provider 分配就宣称模型跑在 GPU.
- 需要 TensorRT 极致性能但不想处理 TensorRT profile, plugin, build cache.
- 目标后端并不支持 ONNX Runtime 对应 EP.
- 把 ORT CPU diff 通过当作所有后端数值都通过.

## 核心概念

- `InferenceSession`: ONNX Runtime 的模型执行会话.
- providers: EP 列表, 顺序会影响图分配.
- provider options: 每个 EP 的细节配置.
- graph optimization level: ORT 自身图优化强度.
- fallback: 高优先级 EP 不支持的节点可能回退到低优先级 EP.
- IO binding: 用于减少 CPU / GPU 拷贝和控制 tensor 绑定.

## 核心 API

源码位置: `xqt/export/onnx_exporter.py`

XQT 当前直接暴露的 ONNX Runtime API 主要是:

- `compare_onnxruntime_outputs(onnx_path, reference_output, example_input, input_name="input", input_names=None, atol=1e-5, rtol=1e-5)`: 使用 `CPUExecutionProvider` 创建 `onnxruntime.InferenceSession`, 执行首个输出并和 PyTorch reference 比较.

相关 ONNX API:

- `export_onnx(...)`
- `validate_onnx(...)`
- `convert_onnx_to_fp16(...)`

## 简单例子

```python
import numpy as np
import onnxruntime as ort


session = ort.InferenceSession(
    "artifacts/model.onnx",
    providers=["CPUExecutionProvider"],
)

feeds = {"input": np.random.randn(2, 16).astype("float32")}
outputs = session.run(None, feeds)
print(session.get_providers())
print(outputs[0].shape)
```

XQT diff 例子:

```python
import torch

from xqt.export import compare_onnxruntime_outputs


example = torch.randn(2, 16)
reference = torch.randn(2, 4)

diff = compare_onnxruntime_outputs(
    "artifacts/model.onnx",
    reference,
    example,
    input_names=["input"],
    atol=1e-4,
    rtol=1e-4,
)
```

## 后端实现注意点

- provider 顺序要显式记录, 例如 `["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]`.
- benchmark 时要记录 ORT 版本, provider options, graph optimization level 和 fallback 情况.
- CPU EP diff 适合做 baseline, 不代表 CUDA / TensorRT / OpenVINO EP 数值完全一致.
- ORT TensorRT EP 仍需要 TensorRT 环境, 支持算子和 profile 约束.
- 如果性能异常, 先检查 graph 是否大面积 fallback 到 CPU EP.

## 常见问题

- `No Op registered`: 当前 ORT build 或 EP 不支持某个 ONNX op / opset.
- provider 不生效: 安装的是 CPU-only 包, 或 provider 依赖库缺失.
- 性能比预期差: 图分区没有进入目标 EP, 或 IO 拷贝成本过高.
- 输入名错误: ONNX graph input 名和 feed key 不一致.
- 动态 shape 问题: ORT 可运行, 但目标 EP 可能要求更具体的 shape.

## 官方资料

- ONNX Runtime documentation: <https://onnxruntime.ai/docs/>
- Execution Providers: <https://onnxruntime.ai/docs/execution-providers/>
- Python API summary: <https://onnxruntime.ai/docs/api/python/api_summary.html>

