# XQT 工作流

本文汇总 `XQT` 当前推荐的使用入口和最短工作流.

## 负责什么

- 说明 `session` 和 YAML workflow 两条主路径.
- 说明 readiness / report / artifact 的使用落点.
- 指向长期事实源和工程契约.

## 不负责什么

- 不重复完整架构边界.
- 不替代具体 backend 的官方安装文档.
- 不承载训练相关流程.

## 先看什么

1. [../architecture/xqt.md](../architecture/xqt.md)
2. [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
3. [../../../xqt/README.md](../../../xqt/README.md)

## 主路径 1: `XQTOptimizationSession`

适合交互式 stage 编排和实验迭代.

当前公开入口:

- `from xqt import XQTOptimizationSession`

适用场景:

- 需要逐 stage 试验
- 需要在 Python 中动态拼装流程
- 需要更细粒度控制 artifact 和 report

## 主路径 2: YAML workflow

适合标准化工作流和可复现配置.

当前公开入口:

- `optimize_model("path/to/workflow.yaml")`
- `xqt-run-workflow`

适用场景:

- 需要保存可复现实验配置
- 需要把优化流程交给下一个人或 agent
- 需要稳定复跑和归档

## readiness / report / artifact

当前可用落点:

- `assess_xqt_readiness()`: readiness audit 入口
- `XQTReadinessReport.write_artifacts()`: readiness 产物落盘
- `ArtifactManifest` / `ArtifactRecord`: 统一产物追踪

使用原则:

- benchmark 给基线
- profiler 给瓶颈归因
- artifact 和 report 都要进入 manifest

## 修改顺序

1. 先看 [../XQT.md](../XQT.md)
2. 再看 [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
3. 最后看相关 `recipes/*.yaml` 和测试

## 继续阅读

- [../architecture/xqt.md](../architecture/xqt.md)
- [../explanation/xqt-concepts.md](../explanation/xqt-concepts.md)
