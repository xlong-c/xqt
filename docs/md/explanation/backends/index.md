# XQT 后端与 Engine 文档入口

本文汇总 XQT 当前涉及的导出 backend, runtime backend 和内部 kernel engine. 这个目录只做背景介绍, 使用场景, 简单例子和核心 API 说明, 不定义新的 XQT workflow schema 或兼容承诺.

术语约定:

- `backend`: 对外导出或部署 runtime, 例如 TensorRT, ONNX Runtime, OpenVINO, ExecuTorch, ncnn, MNN.
- `engine`: XQT 内部 kernel 实现选择, 例如 Triton, TileLang, CUTLASS, CuTe DSL, CuTile, custom CUDA.

XQT 推理优化的对外主体是 `xqt`, 不是一组并列外部 inference backend. `triton` / `tilelang` / `cute_dsl` / `custom_cuda` 只表示 XQT 对某个语义算子的 lowering engine.

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

### Operator Optimization Engine

- [tilelang.md](tilelang.md): TileLang Python DSL kernel engine.
- [triton.md](triton.md): Triton Python GPU kernel engine.
- [cutlass.md](cutlass.md): CUTLASS Python / GEMM engine.
- [cute-dsl.md](cute-dsl.md): NVIDIA CuTe DSL engine.
- [cutile.md](cutile.md): CuTile Python DSL engine.

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

XQT 的 operator engine adapter 在 `xqt/operator_opt/backends/` 中声明. 目录名仍叫 `backends` 是历史工程命名; 文档和新 API 使用 `engine` 描述这些 DSL / custom kernel 选择.

| engine | 主要 pattern | 当前定位 |
| --- | --- | --- |
| `tilelang` | `attention`, `conv`, `linear`, `linear_marlin`, `norm`, `dequant_gemm_epilogue`, FP4 / NVFP4 packed GEMM | 受限 CUDA kernel + eager fallback / metadata |
| `triton` | fused activation / norm / rope, fp16 / bf16 / int8 / fp8 / int4 GEMM | Python Triton kernel adapter |
| `cutlass` | `gemm_epilogue`, `grouped_gemm` | CUTLASS Python GEMM metadata / fallback adapter |
| `cute_dsl` | `gemm_epilogue`, `grouped_gemm` | CuTe DSL GEMM metadata / reference-guarded dense wrapper |
| `cutile` | `attention`, `conv`, `linear`, `norm`, `dequant_gemm_epilogue`, FP4 / NVFP4 packed GEMM, `bias_silu` | CuTile DSL metadata / reference-guarded wrapper |

## 阅读建议

先看代码对齐的总矩阵:

1. [../xqt-engines.md](../xqt-engines.md): engine / quant strategy / pattern / maturity
2. [../xqt-inference.md](../xqt-inference.md): hybrid 推理, 模型包, export/deploy runtime

如果你在写导出后端, 再读:

1. [onnx.md](onnx.md)
2. [onnx-runtime.md](onnx-runtime.md)
3. 目标后端文档, 例如 [tensorrt.md](tensorrt.md) 或 [openvino.md](openvino.md)

如果你在写 operator engine, 再读:

1. [triton.md](triton.md)
2. [tilelang.md](tilelang.md)
3. 目标 DSL 文档, 例如 [cutlass.md](cutlass.md), [cute-dsl.md](cute-dsl.md), [cutile.md](cutile.md)

## 写 backend / engine 时统一记录

每个 backend / engine 文档都会重复强调一件事: 产物必须能复现. 记录 metadata 时至少包含:

- backend 或 engine 名称和版本.
- runtime / compiler / SDK 版本.
- target device 和硬件能力.
- 输入模型路径, checksum, input / output 名称和 shape.
- precision, dynamic shape, profile, quantization 配置.
- custom op, plugin, delegate, fallback 信息.
- benchmark 的 warmup, repeat, batch size, thread, provider, device.
- correctness 对比 baseline, tolerance 和误差摘要.
