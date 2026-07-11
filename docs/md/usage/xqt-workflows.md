# XQT 工作流

本文汇总 `XQT` 当前推荐的使用入口和最短工作流.

## 负责什么

- 说明 `session` 和 YAML workflow 两条主路径.
- 说明 readiness / report / artifact 的使用落点.
- 指向长期事实源和工程契约.

## 不负责什么

- 不重复完整架构边界.
- 不替代具体 runtime / engine 的官方安装文档.
- 不承载训练相关流程.

## 先看什么

1. [../architecture/xqt.md](../architecture/xqt.md)
2. [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
3. [../../../xqt/README.md](../../../xqt/README.md)

## 主路径 1: `XQTOptimizationSession`

适合交互式 stage 编排和实验迭代.

当前公开入口:

- `from xqt import XQTOptimizationSession`
- `xqt.convert(model_or_module, engine=..., policy=...)`
- `xqt.nn.FeedForward` / `xqt.nn.RMSNorm`
- `xqt.nn.Linear` / `xqt.nn.Conv2d` / `xqt.nn.LayerNorm` 是保留 PyTorch module/state_dict 语义并显式记录 runtime intent 的 facade

适用场景:

- 需要逐 stage 试验
- 需要在 Python 中动态拼装流程
- 需要更细粒度控制 artifact 和 report
- 需要以 Python 模块替换表达 `FeedForward` 这类已接入推理语义块, 或为后续 `Linear`, `Conv`, `Norm`, `Attention`, `TransformerBlock` semantic facade 做准备

其中 `engine` 指 XQT 内部实现选择和 report 字段, 例如 `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `custom_cuda`, `torch_compile`. 它不是把 TensorRT / ONNX Runtime 这类外部 runtime 和 XQT 并列. 对推理优化来说, 用户入口仍然是 `xqt`.

## 主路径 2: YAML workflow

适合标准化工作流和可复现配置.

当前公开入口:

- `optimize_model("path/to/workflow.yaml")`
- `xqt-run-workflow`

适用场景:

- 需要保存可复现实验配置
- 需要把优化流程交给下一个人或 agent
- 需要稳定复跑和归档

YAML workflow 和 `xqt-run-workflow` 是 Python 主入口的声明式封装和薄 CLI. 不要在 CLI 参数里重新发明一套并行配置语义.

YAML workflow 只保留一种配置项集合:

- `project`: 项目名和输出目录, `artifact_dir` 是优化模型,导出产物,report 和 manifest 的落点
- `model`: 输入模型,也可以由调用方通过 `optimize_model(..., model=...)` 传入
- `task`: 模型侧 task metadata,不声明数据集或 evaluation provider
- `compression_axes`: 本次优化涉及的模型轴,如 `precision`,`sparsity`,`width`,`depth`
- `hardware`: 运行和产物约束,如 `device` 和 `backends`
- `benchmark`: 默认 benchmark 参数
- `stages`: 唯一的优化和导出路径,包含 `quant`,`prune`,`operator`,`export`,`deploy`,`benchmark`,`analyze`

不要在 `xqt/recipes` 使用旧顶层 `compression`,`export`,`operator_optimization`,`analysis`,`validation` 或 `config_version`. 公开 recipe 必须能直接由 `load_optimization_config()` 加载.

当前 loader 已把 `stages[*].params` 解析为 typed `StageSpec`, workflow 主链已通过 stage helper 消费 runtime config. workflow context 为 runtime-only, public `create_context()` 只接受 `OptimizationConfig` 或 workflow 输入; 旧 `load_xqt_config()` / `XQTConfig` / `XQTContext.config` 已删除, 新 workflow 不应恢复旧 schema.

## readiness / report / artifact

当前可用落点:

- `assess_xqt_readiness()`: readiness audit 入口
- `XQTReadinessReport.write_artifacts()`: readiness 产物落盘
- `ArtifactManifest` / `ArtifactRecord`: 统一产物追踪
- `xqt.runtime.load_model_package()` / `create_inference_runner()`: 推理侧标准文件加载入口. 当前 ONNX export 会额外落一个 `*.xqtpkg/manifest.json` 包, runtime 只消费这个包, 不直接读 quant recipe 或 workflow manifest.

使用原则:

- benchmark 给基线
- profiler 给瓶颈归因
- artifact 和 report 都要进入 manifest

## 修改顺序

1. 先看 [../XQT.md](../XQT.md)
2. 再看 [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
3. 架构重构先看 [../architecture/xqt-realignment-guide.md](../architecture/xqt-realignment-guide.md)
4. 最后看相关 `recipes/*.yaml` 和测试

## 继续阅读

- [../architecture/xqt.md](../architecture/xqt.md)
- [../explanation/xqt-concepts.md](../explanation/xqt-concepts.md)
