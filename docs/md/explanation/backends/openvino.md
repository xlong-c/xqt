# OpenVINO 后端

本文介绍 OpenVINO 后端. 在 XQT 中, OpenVINO 当前是 P1 adapter, 主要提供 PyTorch module 或 ONNX 到 OpenVINO IR 的转换, 以及 OpenVINO Runtime 输出对比.

## 后端介绍

OpenVINO 是 Intel 的模型优化和推理工具链. 它提供 conversion pipeline, OpenVINO IR, Runtime API 和设备 plugin, 目标是在 Intel CPU, GPU, NPU 等设备上提供可移植的推理性能.

OpenVINO 的核心不是单个文件格式, 而是统一 runtime + 多设备 plugin:

```text
PyTorch / ONNX / TensorFlow / Paddle
  -> openvino.convert_model
  -> OpenVINO model / IR
  -> ov.Core.compile_model(device)
  -> InferRequest / compiled model inference
```

## 适合的使用场景

- Intel CPU / integrated GPU / NPU 上部署模型.
- 同一套模型需要在 Intel 多设备之间切换.
- CPU latency 或 throughput 优化是重点.
- 希望使用 OpenVINO performance hint, async inference, compiled model cache.
- 需要从 ONNX 或 PyTorch 快速转换到 OpenVINO runtime.

## 不适合的场景

- 目标是 NVIDIA GPU 极致性能, 一般应优先 TensorRT.
- 目标设备不是 OpenVINO 支持或测试过的平台.
- 模型依赖大量 OpenVINO 不支持的自定义 op.
- 不区分 CPU / GPU / NPU plugin 差异就做统一结论.

## 核心概念

- `ov.Core`: OpenVINO Runtime 入口, 用于读取, 编译模型和查询设备.
- `ov.convert_model`: 从 PyTorch, ONNX 等输入转换 OpenVINO model.
- IR: OpenVINO 中间表示, 常见为 `.xml` + `.bin`.
- compiled model: 针对目标 device 编译后的模型.
- infer request: 推理请求, 支持同步和异步执行.
- device plugin: `CPU`, `GPU`, `NPU`, `AUTO`, `HETERO` 等.
- performance hint: `LATENCY`, `THROUGHPUT` 等高层性能意图.

## 核心 API

源码位置: `xqt/export/openvino.py`

- `OpenVINOExportResult`: IR 产物 metadata, 包含 `xml_path`, `bin_path`, `checksum`, `dry_run`, `source_path`, `metadata`.
- `export_openvino_ir(model, output_path, example_input=None, input_shape=None, dry_run=False)`: 将 PyTorch module 或 ONNX path 转为 OpenVINO IR.
- `compare_openvino_outputs(xml_path, reference_output, example_input, device="CPU", atol=1e-5, rtol=1e-5)`: 使用 OpenVINO Runtime 编译模型并比较首个输出.

## 简单例子

从 ONNX 转 OpenVINO IR:

```python
from xqt.export import export_openvino_ir


result = export_openvino_ir(
    "artifacts/model.onnx",
    "artifacts/openvino/model.xml",
    input_shape=[1, 3, 224, 224],
)
print(result.xml_path, result.bin_path)
```

从 PyTorch module 转换并比较:

```python
import torch
from torch import nn

from xqt.export import export_openvino_ir, compare_openvino_outputs


model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
example = torch.randn(2, 16)
reference = model(example)

result = export_openvino_ir(
    model,
    "artifacts/openvino/mlp.xml",
    example_input=example,
)

diff = compare_openvino_outputs(
    result.xml_path,
    reference,
    example,
    device="CPU",
)
```

## 后端实现注意点

- 转换成功不等于所有 device plugin 都能高性能执行.
- `AUTO` / `HETERO` 有助于便携, 但性能分析时要记录实际设备分配.
- CPU, GPU, NPU 的 dtype, layout, supported op 可能不同.
- benchmark 要区分 latency mode 和 throughput mode, 不要混用配置.
- 如果输入是 PyTorch module, `example_input` 必填.
- 如果输入是 ONNX path, `input_shape` 对静态化和编译稳定性很重要.

## 常见问题

- `openvino is required`: 没安装 OpenVINO Python 包.
- conversion 失败: 模型包含不支持的 op, 动态 shape 或前处理逻辑.
- compiled model 失败: 转换出来的模型不能被目标 device plugin 支持.
- 输出差异: dtype lowering, layout, fused op 或前处理不同.
- 性能不稳定: thread, stream, performance hint, batch size 和 NUMA 都会影响 CPU 结果.

## 官方资料

- OpenVINO documentation: <https://docs.openvino.ai/>
- OpenVINO inference guide: <https://docs.openvino.ai/2025/openvino-workflow/running-inference.html>
- OpenVINO model conversion: <https://docs.openvino.ai/2025/openvino-workflow/model-preparation.html>

