# ExecuTorch 后端

本文介绍 ExecuTorch 后端. 在 XQT 中, ExecuTorch 当前是 P2 adapter, 主要通过 PyTorch `torch.export` 和 ExecuTorch EXIR flow 生成 `.pte` 端侧程序.

## 后端介绍

ExecuTorch 是 PyTorch Edge 生态的端侧推理方案. 它面向 mobile, embedded, wearable, microcontroller 等设备, 目标是把 PyTorch 模型通过 export, lowering, edge program 和 runtime 打包成适合设备端执行的程序.

典型链路:

```text
PyTorch nn.Module
  -> torch.export.export
  -> executorch.exir.to_edge
  -> edge_program.to_executorch
  -> .pte
  -> ExecuTorch runtime
```

ExecuTorch 不等于把完整 Python PyTorch 搬到端侧. 它生成的是端侧 runtime 可执行的 program, 并通过 delegate / backend 把子图交给 XNNPACK, Core ML, Vulkan, Qualcomm, Arm 等目标执行路径.

## 适合的使用场景

- PyTorch 模型要部署到移动端或嵌入式设备.
- 希望尽量保持 PyTorch authoring experience.
- 端侧需要离线推理, 控制 binary size 和 runtime 依赖.
- 需要接入不同设备 delegate, 例如 CPU, GPU, NPU, DSP.
- 可以为不同硬件生成不同 `.pte` 产物.

## 不适合的场景

- 服务端 NVIDIA GPU 高吞吐推理, 一般优先 TensorRT / ONNX Runtime / torch.compile.
- 模型 forward 不能被 `torch.export` 捕获.
- 希望一个 `.pte` 在所有 delegate 上都达到最佳性能.
- 不准备处理端侧 runtime, allocator, thread, binary size 和平台 build 问题.

## 核心概念

- `ExportedProgram`: 来自 `torch.export`.
- edge program: ExecuTorch 面向端侧 lowering 的中间程序.
- `.pte`: ExecuTorch program 文件.
- runtime: 设备端加载和执行 `.pte`.
- delegate / backend: 把图或子图交给具体硬件库.
- backend-specific lowering: 针对目标后端生成专门 program.

## 核心 API

源码位置: `xqt/export/mobile.py`

- `ExecuTorchExportResult`: `.pte` 产物 metadata.
- `export_executorch_program(model, example_input, pte_path, dry_run=False, metadata=None)`: 导出 `.pte`.

实现细节:

- 非 dry-run 时导入 `executorch.exir.to_edge`.
- 内部调用 `torch.export.export`.
- 调用 `to_edge(exported).to_executorch()`.
- 最后通过 `write_to_file` 写入 `.pte`.

## 简单例子

```python
from pathlib import Path

import torch
from torch import nn

from xqt.export import export_executorch_program


model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
example = torch.randn(1, 16)

result = export_executorch_program(
    model,
    example,
    Path("artifacts/mobile/model.pte"),
)
print(result.pte_path, result.checksum)
```

dry-run:

```python
result = export_executorch_program(
    model,
    example,
    "artifacts/mobile/model.pte",
    dry_run=True,
    metadata={"target": "android"},
)
```

## 后端实现注意点

- `.pte` 能生成不代表特定 delegate 已经接入.
- 每个 delegate 的 op, dtype, layout, shape 支持不同.
- `torch.export` 阶段失败时, 不要继续定位 ExecuTorch runtime.
- 端侧问题常发生在 binary size, allocator, thread, memory planning 和 platform build.
- report 要区分 generic ExecuTorch export 和 target delegate lowering.

## 常见问题

- `executorch is required`: 没有安装 ExecuTorch Python 包.
- `torch.export` 失败: forward 包含不可捕获 Python 行为或动态 shape 未声明.
- delegate lowering 失败: generic graph 可导出, 但目标硬件不支持其中 op.
- 端侧输出差异: dtype, layout, quantization 或 delegate kernel 与 PyTorch baseline 不一致.

## 官方资料

- ExecuTorch documentation: <https://docs.pytorch.org/executorch/stable/index.html>
- ExecuTorch overview: <https://docs.pytorch.org/executorch/stable/intro-overview.html>
- Exporting to ExecuTorch: <https://docs.pytorch.org/executorch/stable/using-executorch-export.html>

