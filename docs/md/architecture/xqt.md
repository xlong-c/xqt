# XQT 架构

本文定义 `XQT` 的系统边界, 主链路和工程契约. 它是 `XQT` 架构层的长期事实源之一.

## 负责什么

- 说明 `XQT` 做什么, 不做什么.
- 说明主链路, 配置方式和 Stage 约定.
- 说明 profiling 与 manifest 的工程边界.

## 不负责什么

- 不提供完整训练流程.
- 不承载 task-level validation 设计.
- 不替代具体 backend 官方文档.

## 项目定位

`XQT` 只关注模型本身. 它接收 PyTorch 模型, checkpoint 或导出产物, 执行模型侧压缩, 图变换, 导出适配, 误差分析和 benchmark.

需要梯度更新的流程归 `XDL` 或第三方训练工具, 再把训练后的模型或 checkpoint 交给 `XQT`.

主链路:

```text
PyTorch model / checkpoint / exported artifact
    -> YAML workflow or XQTOptimizationSession
    -> quant / prune / operator / export / deploy / analyze / benchmark stages
    -> optimized model, export target artifact, report and manifest
```

## 核心边界

核心职责:

- 支持 `nn.Module` 和 `state_dict` 作为主要输入.
- 覆盖 PTQ / QDQ, 权重量化, 剪枝, 算子优化, 导出前适配和部署格式转换.
- 能导出 ONNX, TensorRT, OpenVINO, torch.export, TorchScript, ExecuTorch, ncnn, MNN 等产物.
- 记录配置, 源 checkpoint, 指标, 产物校验和执行阶段, 保证结果可复现.

非目标:

- 不替代 `xdl.trainer.Trainer`.
- 不维护通用 model / loss / metric registry.
- 不运行 `zero_grad() -> backward() -> step()` 训练循环.
- 不做 task-level validation 或 accuracy / mAP 评测闭环.
- 不重新实现 TensorRT, OpenVINO, ONNX Runtime, ExecuTorch, ncnn, MNN 等后端.

## 配置方式

`XQT` 只提供两种配置方式:

1. `XQTOptimizationSession`
2. YAML workflow, 通过 `optimize_model()` 运行

原则上不要新增 CLI 参数解析, JSON shaped workflow 或其他配置路径.

YAML workflow 的公开 schema 只有一套: `project`, `model`, `task`, `compression_axes`, `hardware`, `benchmark`, `stages`, `device`. 其中 `stages` 是唯一的优化和导出路径描述; 旧式顶层 `compression`, `export`, `operator_optimization`, `analysis`, `validation` 和 `config_version` 只属于内部 `XQTConfig` 形态,不能出现在 `xqt/recipes`.

## 关键抽象

| 概念 | 说明 |
| --- | --- |
| `OptimizationConfig` | 对外 stage workflow schema |
| `XQTOptimizationSession` | 交互式 stage 编排入口 |
| `ArtifactManifest` / `ArtifactRecord` | 产物追踪 |
| `MetricRecord` | 结构化指标记录 |
| `OptimizationCapability` | 统一 capability 投影 |
| `XQTReadinessReport` / `assess_xqt_readiness()` | readiness 汇总入口 |

## Stage 约定

- 支持: `benchmark`, `prune`, `quant`, `operator`, `export`, `deploy`, `analyze`
- 不支持: `finetune`, `distill`, `eval`, `runtime_eval`, `qat_train`, `recovery`
- 量化 recipe 必须显式写 `backend` 和 `policy`
- `calibration_inputs` 由调用方传入, recipe 不声明数据来源
- planned / capability-only 后端必须在 preflight 和文档中标注
- workflow stage 必须写入 `stage_reports` 和 manifest stage metric

## Profiling 约定

`XQT` 允许理解和记录性能分析工具, 但 profiling 只作为模型侧 `benchmark` / `operator` / `export` / `deploy` 的诊断补充.

- `benchmark` 负责稳定 latency / memory / throughput 基线, profiler 负责解释瓶颈
- 外部 profiler 输出应作为 artifact 挂到 manifest
- report 至少记录 backend, device, target artifact, input shape, warmup, repeat, precision, batch size, profiler 名称和关键参数
- `XQT` 可以提供 profiler preflight 和命令模板, 但不要封装厂商 profiler 的完整 CLI
- profiling 不能引入 dataset / dataloader, evaluation provider, task-level validation 或训练循环

## 修改约束

- 新能力必须服务模型本身: quant, prune, operator optimization, export, diff, benchmark 或 manifest
- 不要在 `XQT` 中新增训练循环, trainer, training provider, evaluation provider, loss wrapper 或任务 registry
- 不要在 `XQT` 中新增 dataset / dataloader 构建或 task-level validation
- 顶层示例脚本保持薄入口, 优先调用 `XQT` 既有 helper

## 继续阅读

- [../explanation/xqt-concepts.md](../explanation/xqt-concepts.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- [../XQT.md](../XQT.md)
