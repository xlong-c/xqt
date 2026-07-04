# ONNX 后端

本文介绍 ONNX 模型交换格式和 XQT 的 ONNX exporter. ONNX 本身是模型表示, 不是 runtime; 它通常要交给 ONNX Runtime, TensorRT, OpenVINO, ncnn, MNN 或其他工具继续执行或转换.

## 后端介绍

ONNX 全称 Open Neural Network Exchange. 它定义了模型计算图, tensor 类型, initializer, 标准算子和 opset 版本. 它的价值是把 PyTorch 等训练框架里的模型变成一个更通用的交换格式.

在部署链路里, ONNX 常做中间层:

```text
PyTorch model -> ONNX -> ONNX Runtime / TensorRT / OpenVINO / ncnn / MNN
```

ONNX 是否有用取决于两件事: exporter 是否能正确表达模型, 目标 runtime 是否支持导出的 graph.

## 适合的使用场景

- 把 PyTorch 模型交给非 PyTorch runtime.
- 给 TensorRT, OpenVINO, ncnn, MNN 等工具提供通用输入.
- 用 ONNX Runtime CPU Execution Provider 做部署前 correctness baseline.
- 让模型产物脱离 Python 源码, 便于跨语言加载和检查.
- 对模型做 checker, shape inference, graph inspection 或 FP16 转换.

## 不适合的场景

- 期待自动包含 Python 前处理和后处理.
- 期待所有 PyTorch op 都有标准 ONNX 表达.
- 不关心 opset, dtype, dynamic shape 和 custom op 的情况下直接部署.
- 把 ONNX 文件当成性能优化结果; 性能由后续 runtime 决定.

## 核心概念

- opset: ONNX 算子语义版本, exporter 和 runtime 都要支持.
- initializer: 常用于保存权重 tensor.
- input / output name: 后端绑定输入输出时依赖这些名字.
- dynamic axes / dynamic shapes: 表达动态维度, 但目标后端可能仍要求 profile 或固定 shape.
- custom op: 标准 ONNX 表达不了时使用, 会牺牲可移植性.
- checker: `onnx.checker.check_model` 能检查格式合法性, 不能保证 runtime 支持.

## 核心 API

源码位置: `xqt/export/onnx_exporter.py`

- `ONNXExportResult`: ONNX 产物 metadata, 包含 path, opset, checksum, checked, output_diff, metadata.
- `export_onnx(model, example_input, output_path, opset=None, input_names=None, output_names=None, dynamic_shapes=None, dynamo=True, validate=True, pre_export_fusion=None)`: 使用 PyTorch ONNX exporter 导出模型, 默认启用现代 dynamo exporter.
- `validate_onnx(path)`: 使用 `onnx.checker` 校验 ONNX 文件.
- `convert_onnx_to_fp16(onnx_path, output_path, keep_io_types=False, validate=True)`: 使用 `onnxconverter-common` 做 FP16 转换.
- `compare_onnxruntime_outputs(onnx_path, reference_output, example_input, input_name="input", input_names=None, atol=1e-5, rtol=1e-5)`: 用 ONNX Runtime CPU EP 跑首个输出并和 PyTorch tensor 比较.

## 简单例子

```python
from pathlib import Path

import torch
from torch import nn

from xqt.export import export_onnx, validate_onnx, compare_onnxruntime_outputs


model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
example = torch.randn(2, 16)

with torch.no_grad():
    reference = model(example)

result = export_onnx(
    model,
    example,
    Path("artifacts/model.onnx"),
    input_names=["input"],
    output_names=["logits"],
    opset=18,
    validate=True,
)

validate_onnx(result.path)
diff = compare_onnxruntime_outputs(
    result.path,
    reference,
    example,
    input_names=["input"],
)
```

## 后端实现注意点

- `dynamo=True` 是 XQT 默认路径, 但部分旧模型可能需要 fallback 到传统 exporter.
- `dynamic_shapes` 是 PyTorch exporter 参数, 不是所有后端都能完整消费.
- `input_names` 和 `output_names` 要在导出, runtime feed, report 中保持一致.
- `validate_onnx` 只能证明 ONNX 文件结构合法, 不能证明 TensorRT / OpenVINO 可转换.
- FP16 转换可能改变输入输出 dtype; 如果应用侧仍期望 FP32 IO, 使用 `keep_io_types=True`.
- 导出前后要做 output diff, 因为 ONNX lowering 和 runtime kernel 都可能引入差异.

## 常见问题

- `Unsupported operator`: PyTorch op 没有对应 ONNX symbolic, 或目标 opset 不支持.
- `Checker failed`: graph 结构或类型不合法.
- `Invalid feed input name`: runtime feed 名和 graph input 名不一致.
- 动态 shape 后端失败: ONNX 有符号维度, 但 TensorRT 等后端仍需要 optimization profile.
- 数值差异: 常见于 fused op, layout 转换, dtype lowering, 常量折叠和随机行为.

## 官方资料

- ONNX home: <https://onnx.ai/>
- ONNX concepts: <https://onnx.ai/onnx/intro/concepts.html>
- PyTorch ONNX exporter: <https://docs.pytorch.org/docs/stable/onnx.html>

