# XQT 摘要

本文保留为 `XQT` 的兼容摘要入口. 新的 Markdown 主导航已经迁到 [index.md](index.md), 并按架构 / 说明 / 使用三层组织.

XQT 只关注模型本身. 它做模型压缩,图变换,导出适配,误差分析和 benchmark,不接管训练,QAT,finetune,distillation,recovery 或 dataset / provider 语义.

XQT 是仓库内唯一推理优化主体. Python API 是主入口; `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `custom_cuda` 是 XQT 内部 engine, 不是外部 inference backend.

## 三层入口

- 架构层: [architecture/xqt.md](architecture/xqt.md)
- 说明层: [explanation/xqt-concepts.md](explanation/xqt-concepts.md)
- 使用层: [usage/xqt-workflows.md](usage/xqt-workflows.md)

## 先看什么

- 先看 [index.md](index.md): 新 Markdown 总入口.
- 再看 [architecture/xqt.md](architecture/xqt.md): `XQT` 架构正文.
- 再看 [explanation/xqt-concepts.md](explanation/xqt-concepts.md): `XQT` 概念说明.
- 再看 [usage/xqt-workflows.md](usage/xqt-workflows.md): `XQT` 工作流入口.
- 需要兼容旧结构时看 [XQT.md](XQT.md): 旧长期事实源.
- 再看 [../../xqt/README.md](../../xqt/README.md): 包内薄入口.
- 需要包内工程约束时看 [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md).

## 现在可用的主链路

- `XQTOptimizationSession`: 交互式 stage 编排入口.
- `optimize_model()`: YAML workflow 入口.
- `assess_xqt_readiness()`: readiness audit 入口.
- `XQTReadinessReport.write_artifacts()`: readiness 产物落盘.

## 能力概览

- 已基本可用: PTQ / QDQ / torchao 量化,常规剪枝,ONNX / torch.export / TorchScript / TensorRT / OpenVINO / ExecuTorch / ncnn / MNN 导出,output diff,layer analysis,latency / memory benchmark,manifest.
- 半可用: TensorRT engine / plugin preflight,TileLang 的受限 kernel target (attention / conv / half linear / half norm / dequant GEMM),FP4 packed weight 到 TileLang operator stage 的桥接.
- 偏实验: 更完整的 AWQ / GPTQ packed megakernel,以及更广泛的 engine capability 闭环.

架构重构状态: 最初诊断报告的 1-8 节不是已完成清单. 当前已完成配置单轨化 Phase A/B/C: workflow 主链已推进到 typed `StageSpec` / stage helper 消费 runtime config; workflow context 为 runtime-only, public `create_context()` 只接受 `OptimizationConfig` 或 workflow 输入; 旧 `load_xqt_config()` / `XQTConfig` / `XQTContext.config` 已删除. 长期指导见 [architecture/xqt-realignment-guide.md](architecture/xqt-realignment-guide.md).

## 性能分析工具

- XQT 记录 profiler 上下文和产物,不接管厂商 profiler 的安装,权限,驱动版本或 GUI 工作流.
- `benchmark` 给出 latency / memory / throughput 基线;profiler 负责瓶颈归因.
- profiler 产物应作为 manifest artifact 保存,例如 `.ncu-rep`,`.nsys-rep`,ROCm trace,VTune result,Ascend `msprof` 目录或 TensorBoard profile 目录.
- 常见映射: NVIDIA `nsys` / `ncu`,AMD `rocprof-sys` / `rocprof`,Intel VTune,Apple Instruments / Metal Debugger,Arm Streamline,Qualcomm Snapdragon Profiler,Google XProf,华为 Ascend `msprof`.

## 场景表

| 场景 | 状态 | 备注 |
| --- | --- | --- |
| FP4 量化 | 半可用 | 有 reference 路径和 TileLang 桥接,仍缺真实 CUDA runtime 数值与性能验证. |
| TileLang megakernel | 半可用 | 已覆盖受限 attention / conv / half linear / half norm / dequant GEMM 路径; conv 当前是 unfold + half GEMM lowering, CPU 只走 PyTorch fallback. |
| TensorRT + `.so` 插件 | 半可用 | 已有 build / inspect / preflight / loadability 检查. |
| 常规剪枝 / 误差分析 | 已基本可用 | layer diff, sensitivity, distribution stats 和 benchmark 已接通. |

## 不属于 XQT

- 训练循环,QAT 训练,finetune,distillation,recovery.
- `training_provider` / `evaluation_provider`.
- dataset / dataloader 构建和 task-level validation.
- 通用 model / loss / metric registry.
