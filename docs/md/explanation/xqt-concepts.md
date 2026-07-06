# XQT 概念说明

本文解释 `XQT` 的概念边界和理解地图. 它不单独定义新契约.

## 这是什么

`XQT` 是 `XDL` 仓库中的模型压缩, 图变换和部署实验包. 它只处理模型本身, 不接管训练语义.

## 为什么需要

训练侧和部署侧的关注点不同:

- `XDL` 负责训练生命周期和组件组织
- `XQT` 负责模型压缩, 导出适配, 误差分析和 benchmark

把两者分开, 可以避免在部署工具链里重新发明训练循环, 也能避免在训练框架里堆叠过多后端适配逻辑.

## 核心概念

- `session`: 交互式优化编排入口
- `workflow`: YAML 驱动的阶段式优化流程
- `stage`: 一次模型侧变换, 分析或导出动作
- `manifest`: 产物, 指标和 lineage 的统一记录
- `readiness`: 对某个场景是否可用的能力判断
- `backend`: 对外导出或部署 runtime, 例如 TensorRT, ONNX Runtime, OpenVINO, ExecuTorch, ncnn, MNN
- `engine`: XQT 内部 kernel 实现选择, 例如 Triton, TileLang, CUTLASS, CuTe DSL, CuTile, custom CUDA
- `semantic replacement`: Python 层以 `Linear`, `Conv`, `Norm`, `Attention`, `FeedForward`, `TransformerBlock` 为单位替换模型语义块

## 与相近概念的区别

- 它不是 trainer
- 它不是 task evaluation 平台
- 它不是 dataset / provider 框架
- 它不是某个后端 runtime 的替代品
- 它不是 TensorRT / ONNX Runtime 这类外部 runtime 的薄 wrapper

## Python-first 推理优化

XQT 的推理优化以 Python API 为主入口. `XQTOptimizationSession`, `xqt.convert(...)`, `xqt.nn.*` facade 和 runtime manager 都应能在 Python 层表达. 性能敏感部分再由内部 engine lowering 到 DSL kernel 或 custom CUDA kernel.

这意味着用户看到的主体是:

```text
XQT model transform
  -> semantic block replacement
  -> operator contract
  -> internal engine lowering
  -> benchmark, report, manifest
```

不是:

```text
用户直接选择一组外部 inference backend 来替换 XQT
```

块级替换和 kernel fusion 不冲突. `FeedForward -> XQTFeedForward`, `Attention -> XQTPagedAttention`, `TransformerBlock -> XQTTransformerBlock` 是 Python 层语义替换; 底层可以是一组 kernel, 也可以是 megakernel. 是否真的合成单 kernel 必须由 capability 和 benchmark 证明.

## 当前能力地图

- 已基本可用: PTQ / QDQ / torchao 量化, 常规剪枝, ONNX / torch.export / TorchScript / TensorRT / OpenVINO / ExecuTorch / ncnn / MNN 导出, output diff, layer analysis, latency / memory benchmark, manifest
- 半可用: TensorRT engine / plugin preflight, TileLang 的受限 kernel target, FP4 packed weight 到 TileLang operator stage 的桥接
- 偏实验: 更完整的 AWQ / GPTQ packed megakernel, 以及更广泛的 engine capability 闭环

当前 `TileLang` 的受限 kernel target 主要覆盖:

- `attention`
- `conv`
- direct half `linear`
- direct half `LayerNorm`
- `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体

这些路径并不是同等成熟度. `attention` / direct half `linear` / dequant GEMM 路径已经进入受限的 TileLang operator coverage. `conv` 当前通过 `torch.unfold` / im2col lowered input 加 TileLang half GEMM 执行, 尚不是 fully fused conv kernel, 且只覆盖 fp16, CUDA, `groups=1` 的路径. direct half `LayerNorm` 已接入 TileLang `reduce_sum` kernel, 限制为 CUDA fp16 和 last-dim normalization. CPU 路径只使用 PyTorch eager fallback.

## 常见误区

- 不要把 `XQT` 当成训练恢复或 QAT 工具
- 不要把 profiler 诊断结果和 benchmark 指标混为一谈
- 不要把 `TileLang available` 理解成所有 pattern 都已经是同等成熟的通用 executor
- 不要把 `triton` / `tilelang` / `cute_dsl` / `custom_cuda` 写成和 TensorRT / ONNX Runtime 并列的外部 backend; 它们是 XQT 内部 engine
- 不要在 recipe 里声明 dataset 来源

## 继续阅读

- [../architecture/xqt.md](../architecture/xqt.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- [../XQT_SUMMARY.md](../XQT_SUMMARY.md) - 兼容摘要入口
