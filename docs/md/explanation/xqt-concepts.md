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

## 与相近概念的区别

- 它不是 trainer
- 它不是 task evaluation 平台
- 它不是 dataset / provider 框架
- 它不是某个后端 runtime 的替代品

## 当前能力地图

- 已基本可用: PTQ / QDQ / torchao 量化, 常规剪枝, ONNX / torch.export / TorchScript / TensorRT / OpenVINO / ExecuTorch / ncnn / MNN 导出, output diff, layer analysis, latency / memory benchmark, manifest
- 半可用: TensorRT engine / plugin preflight, TileLang 的受限 kernel target, FP4 packed weight 到 TileLang operator stage 的桥接
- 偏实验: 更完整的 AWQ / GPTQ packed megakernel, 以及更广泛的 backend capability 闭环

## 常见误区

- 不要把 `XQT` 当成训练恢复或 QAT 工具
- 不要把 profiler 诊断结果和 benchmark 指标混为一谈
- 不要在 recipe 里声明 dataset 来源

## 继续阅读

- [../architecture/xqt.md](../architecture/xqt.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- [../XQT_SUMMARY.md](../XQT_SUMMARY.md) - 兼容摘要入口
