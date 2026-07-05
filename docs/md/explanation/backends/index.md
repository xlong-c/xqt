# XQT 后端文档入口

本文汇总 XQT 当前涉及的导出, runtime 和 operator optimization 后端. 这个目录只做后端背景介绍, 使用场景, 简单例子和核心 API 说明, 不定义新的 XQT workflow schema 或兼容承诺.

## 文档分组

### 导出与 Runtime 后端

- [torch-export.md](torch-export.md): PyTorch `torch.export` 和 TorchScript.
- [onnx.md](onnx.md): ONNX 模型交换格式和 XQT ONNX exporter.
- [onnx-runtime.md](onnx-runtime.md): ONNX Runtime 和 Execution Provider 机制.
- [tensorrt.md](tensorrt.md): NVIDIA TensorRT engine 构建, runtime, plugin 和性能指标.
- [openvino.md](openvino.md): OpenVINO conversion, IR, runtime 和设备 plugin.
- [executorch.md](executorch.md): ExecuTorch `.pte` 端侧部署链路.
- [ncnn.md](ncnn.md): ncnn param/bin, pnnx 和 onnx2ncnn 转换链路.
- [mnn.md](mnn.md): MNNConvert, `.mnn` 产物和移动端 runtime.

### Operator Optimization 后端

- [tilelang.md](tilelang.md): TileLang Python DSL kernel 后端.
- [triton.md](triton.md): Triton Python GPU kernel 后端.
- [cutlass.md](cutlass.md): CUTLASS Python / GEMM 后端.
- [cute-dsl.md](cute-dsl.md): NVIDIA CuTe DSL 后端.
- [cutile.md](cutile.md): CuTile Python DSL 后端.

## 当前 XQT 能力对应

XQT 的导出能力矩阵在 `xqt/export/capability.py` 中声明, 当前包含:

| format | priority | runtimes | 当前状态 |
| --- | --- | --- | --- |
| `torch_export` | P0 | `pytorch` | implemented |
| `onnx` | P0 | `onnxruntime`, `tensorrt`, `openvino` | implemented |
| `tensorrt` | P0 | `tensorrt` | adapter |
| `torchscript` | P1 | `pytorch`, `pnnx` | implemented |
| `openvino` | P1 | `openvino` | adapter |
| `executorch` | P2 | `executorch` | adapter |
| `ncnn` | P2 | `ncnn` | adapter |
| `mnn` | P2 | `mnn` | adapter |

XQT 的 operator backend adapter 在 `xqt/operator_opt/backends/` 中声明, 当前包含:

| backend | 主要 pattern | 当前定位 |
| --- | --- | --- |
| `tilelang` | `attention`, `conv`, `linear`, `linear_marlin`, `norm`, `dequant_gemm_epilogue`, FP4 / NVFP4 packed GEMM | 受限 CUDA kernel + eager fallback / metadata |
| `triton` | fused activation / norm / rope, fp16 / bf16 / int8 / fp8 / int4 GEMM | Python Triton kernel adapter |
| `cutlass` | `gemm_epilogue`, `grouped_gemm` | CUTLASS Python GEMM metadata / fallback adapter |
| `cute_dsl` | `gemm_epilogue`, `grouped_gemm` | CuTe DSL GEMM metadata / reference-guarded dense wrapper |
| `cutile` | `attention`, `conv`, `linear`, `norm`, `dequant_gemm_epilogue`, FP4 / NVFP4 packed GEMM, `bias_silu` | CuTile DSL metadata / reference-guarded wrapper |

## 阅读建议

如果你在写导出后端, 先读:

1. [onnx.md](onnx.md)
2. [onnx-runtime.md](onnx-runtime.md)
3. 目标后端文档, 例如 [tensorrt.md](tensorrt.md) 或 [openvino.md](openvino.md)

如果你在写 operator backend, 先读:

1. [triton.md](triton.md)
2. [tilelang.md](tilelang.md)
3. 目标 DSL 文档, 例如 [cutlass.md](cutlass.md), [cute-dsl.md](cute-dsl.md), [cutile.md](cutile.md)

## 写后端时统一记录

每个后端文档都会重复强调一件事: 后端产物必须能复现. 记录 metadata 时至少包含:

- backend 名称和版本.
- runtime / compiler / SDK 版本.
- target device 和硬件能力.
- 输入模型路径, checksum, input / output 名称和 shape.
- precision, dynamic shape, profile, quantization 配置.
- custom op, plugin, delegate, fallback 信息.
- benchmark 的 warmup, repeat, batch size, thread, provider, device.
- correctness 对比 baseline, tolerance 和误差摘要.
