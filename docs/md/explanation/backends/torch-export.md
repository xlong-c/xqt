# PyTorch torch.export 与 TorchScript 后端

本文介绍 PyTorch 原生导出路径在 XQT 中的角色. 它覆盖 `torch.export` 和 TorchScript, 因为 XQT 当前把二者都放在 `xqt/export/torch_exporter.py`.

## 后端介绍

`torch.export` 是 PyTorch 2 的图捕获和导出机制. 它从 `nn.Module` 和样例输入中捕获一个 `ExportedProgram`, 让模型计算图脱离普通 Python eager 执行, 进入可分析, 可保存, 可交给下游 compiler 或 edge runtime 的形式.

TorchScript 是 PyTorch 较早的序列化部署路线. 它通过 `torch.jit.trace` 或 `torch.jit.script` 生成可保存的 TorchScript module, 常见产物是 `.pt`. 在新项目中, `torch.export` 更适合作为 PyTorch 2 时代的图导出入口; TorchScript 仍适合已有 LibTorch / pnnx / 旧移动端链路.

## 适合的使用场景

- 在 PyTorch 内部先确认模型能否被图捕获.
- 作为 ExecuTorch, ONNX dynamo exporter 或其他 compiler flow 的上游准备.
- 保存一个可检查, 可复现的 PyTorch 原生导出产物.
- 使用 TorchScript 给 pnnx, ncnn 或旧 C++ 部署链路提供输入.
- 对比导出前后 PyTorch 输出, 先排除模型捕获阶段的数值问题.

## 不适合的场景

- 直接替代 TensorRT, OpenVINO, ONNX Runtime 等推理 runtime.
- 把任意 Python forward 完整搬到部署端.
- 不写样例输入就期望自动推断所有动态 shape.
- 把 TorchScript 当成所有新后端的唯一入口.

## 核心概念

- `ExportedProgram`: `torch.export` 的导出结果, 包含图, 参数, buffer 和约束.
- dynamic shapes: `torch.export` 需要显式描述动态维度约束.
- strict export: 更严格时能暴露更多 Python 语义问题, 但导出成功率可能下降.
- trace: TorchScript 的样例路径记录方式, 对 data-dependent control flow 不可靠.
- script: TorchScript 的 Python 子集编译方式, 能保留部分控制流但限制更多.

## 核心 API

源码位置: `xqt/export/torch_exporter.py`

- `TorchExportResult`: `torch.export` 产物 metadata.
- `TorchScriptExportResult`: TorchScript 产物 metadata.
- `export_torch_program(model, example_input, output_path, dynamic_shapes=None, strict=False, validate=True, compare_output=True, atol=1e-5, rtol=1e-5)`: 导出 `ExportedProgram`, 保存文件, 可选加载校验和输出 diff.
- `export_torchscript(model, example_input, output_path, method="trace", check_trace=True, compare_output=True, atol=1e-5, rtol=1e-5)`: 通过 trace 或 script 导出 TorchScript.

## 简单例子

```python
from pathlib import Path

import torch
from torch import nn

from xqt.export import export_torch_program, export_torchscript


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = TinyMLP().eval()
example = torch.randn(2, 16)

exported = export_torch_program(
    model,
    example,
    Path("artifacts/tiny_mlp.pt2"),
    validate=True,
    compare_output=True,
)

scripted = export_torchscript(
    model,
    example,
    Path("artifacts/tiny_mlp.ts"),
    method="trace",
)
```

## 后端实现注意点

- 先用 `model.eval()` 固定推理行为, 否则 dropout / batchnorm 会影响 diff.
- `example_input` 可能是 tensor, tuple, list, dict 或带 kwargs 的结构, XQT 通过 `split_example_input` 统一拆分.
- `torch.export.save` 的产物仍然是 PyTorch 生态内的产物, 不是跨 runtime 通用格式.
- TorchScript trace 要小心输入 shape 和分支覆盖; 对控制流敏感的模型优先尝试 script 或 `torch.export`.
- 任何后续后端失败时, 先回到 `export_torch_program(..., compare_output=True)` 检查捕获阶段是否已经产生差异.

## 常见问题

- 导出失败: forward 中可能包含 Python side effect, data-dependent branch, 不支持的 op 或无法符号化的 shape 逻辑.
- 动态 shape 失败: 需要显式约束动态维度, 不能只靠 `-1` 心智模型.
- 输出 diff 失败: 先确认 `eval()` 模式, random seed, dtype, device 和 tolerance.
- TorchScript trace 正确但部署错误: 可能 trace 只记录了样例路径, 真实输入触发了未记录分支.

## 官方资料

- PyTorch `torch.export` user guide: <https://docs.pytorch.org/docs/stable/export.html>
- PyTorch `torch.export` tutorial: <https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html>
- TorchScript documentation: <https://docs.pytorch.org/docs/stable/jit.html>

