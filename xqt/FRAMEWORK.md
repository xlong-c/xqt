# xqt 包内工程契约

**核心边界: XQT 只关注模型本身.**

XQT 消费训练后的模型/checkpoint/导出产物,做压缩,变换,导出,误差分析和 benchmark. XQT 不做训练,不做 QAT,不做 finetune / distill / recovery,不构建 dataset / dataloader,也不拥有 task provider 语义.

## 关键抽象

| 概念 | 说明 |
|---|---|
| `OptimizationConfig` | 对外 stage workflow schema,包含 `project` / `model` / `task` / `stages`. |
| `XQTConfig` | 旧 pass recipe schema,仅作为内部实现. |
| `load_optimization_config()` | `OptimizationConfig` 加载器. |
| `load_xqt_config()` | `XQTConfig` 加载器,内部保留. |
| `XQTOptimizationSession` | 交互式 stage 编排入口. |
| `ArtifactManifest` / `ArtifactRecord` | 产物追踪. |
| `MetricRecord` | 结构化指标记录. |
| `OptimizationCapability` | 统一 capability 投影,覆盖 quant / prune / operator / export 的 backend,status,runtime,artifact_kind 和硬件/校准/导出要求. |
| `XQTReadinessReport` / `assess_xqt_readiness()` | readiness 汇总入口,输出场景状态,capability matrix 和 reporting schema. |
| `example_inputs` | 导出 / benchmark / layer analysis / operator_opt 所需输入. |
| `calibration_inputs` | PTQ / QDQ calibration 所需输入. |

## 配置方式

XQT 只提供两种配置方式:

1. `XQTOptimizationSession`.
2. YAML workflow,通过 `optimize_model()` 运行.

原则上不要新增 CLI 参数解析,JSON shaped workflow 或其他配置路径.

## Stage 约定

- 支持: `benchmark`, `prune`, `quant`, `operator`, `export`, `deploy`, `analyze`.
- 不支持: `finetune`, `distill`, `eval`, `runtime_eval`, `qat_train`, `recovery`.
- 量化 recipe 必须显式写 `backend` 和 `policy`.
- `calibration_inputs` 由调用方传入,recipe 不声明数据来源.
- planned / capability-only 后端必须在 preflight 和文档中标注.
- workflow stage 必须写入 `stage_reports` 和 manifest stage metric,保留 stage name,backend,target module,artifact 和 lineage.

## Profiling 约定

XQT 允许理解和记录性能分析工具,但 profiling 只作为模型侧 benchmark / operator / export / deploy 的诊断补充.

- `benchmark` 负责稳定 latency / memory / throughput 基线,profiler 负责解释瓶颈,二者在 report 中不能混为一个指标.
- 外部 profiler 输出应作为 artifact 挂到 manifest,例如 NVIDIA `.ncu-rep` / `.nsys-rep`,ROCm trace,VTune result,Ascend `msprof` 输出目录或 TensorBoard profile 目录.
- report 至少记录 backend,device,target artifact,input shape,warmup,repeat,precision,batch size,profiler 名称,profiler 命令关键参数和环境版本.
- XQT 可以提供 profiler preflight 和命令模板,但不要封装厂商 profiler 的完整 CLI,不要隐藏驱动,权限,硬件 counter 和 GUI 依赖.
- profiling 不能引入 dataset / dataloader,evaluation provider,task-level validation 或训练循环.

常见工具映射:

| 生态 | 系统级 timeline | kernel / 算子级 |
|---|---|---|
| NVIDIA CUDA | `nsys` | `ncu` |
| AMD ROCm / HIP | `rocprof-sys` | `rocprof`,ROCm Compute Profiler |
| Intel oneAPI / GPU | VTune Profiler | VTune GPU Hotspots / GPU Offload |
| Apple Metal | Instruments / Metal System Trace | Xcode Metal Debugger / GPU Counters |
| Arm Mali / Immortalis | Arm Streamline | Streamline / Performance Advisor |
| Qualcomm Adreno | Snapdragon Profiler | Snapdragon Profiler GPU counters |
| Google TPU | XProf / TensorBoard Profile | XProf / TensorBoard Profile |
| 华为 Ascend | CANN `msprof`,MindStudio Profiler | `msprof-analyze`,Ascend profiler |

## 修改约束

- 新能力必须服务模型本身: quant, prune, operator optimization, export, diff, benchmark 或 manifest.
- 不要在 XQT 中新增训练循环,trainer,training provider, evaluation provider, loss wrapper 或任务 registry.
- 不要在 XQT 中新增 dataset / dataloader 构建或 task-level validation.
- 公共压缩或部署工具若可复用,再考虑抽到 `tools/` 或 `xdl/`.
- 顶层示例脚本保持薄入口,优先调用 XQT 既有 helper.
