# xqt - 模型压缩与部署实验目录

## 开发阶段

**当前处于 v0.x 开发期,未到 v1.0.允许破坏性重构,不需要兼容老接口.怎么方便,怎么清晰,怎么简洁就怎么来.**

- 改 API 时直接改,不需要保留旧入口,不需要 migration guide,不需要 deprecation warning.
- 改 recipe schema 时直接打破兼容,旧的 recipe yaml 跟着一起更新.
- 删模块,改名,合并,拆分都可以,只为最终方案干净服务.
- 取舍时优先顺序: 清晰 > 简洁 > 方便 > 兼容.

## 核心契约

XQT 只关注模型本身.

XQT 负责模型压缩,模型图变换,导出适配,模型误差分析和 benchmark. XQT 不负责训练,QAT 训练,finetune,distillation,KD/prune recovery,dataset/dataloader,training provider 或 evaluation provider. 需要梯度更新或任务验证的流程归 XDL 或第三方工具,再把训练后的模型/checkpoint 或指标交给 XQT.

包内工程契约见 `xqt/FRAMEWORK.md`;长期事实源见 `docs/md/XQT.md`.

## 当前内容

### 包模块

- `core/`: structured config, artifact manifest, checksum, XQT registry.
- `model/`: smoke-only model helper 和模型 forward hook 输出采集工具.
- `pipeline/`: sequential pass manager,preflight 和 YAML runner.
- `workflows/`: stage-based model optimization workflow,支持 `benchmark`,`prune`,`quant`,`operator`,`export`,`deploy`,`analyze`.
- `analysis/`: tensor output diff,layer analysis 和 report helper.
- `benchmark/`: latency 和 memory benchmark helper.
- `quant/`: quantization policy,backend capability matrix,activation calibration,layer sensitivity helper.
- `prune/`: unstructured,structured,N:M 和 block sparse pruning helper.
- `operator_opt/`: `torch.compile`-first operator optimization pass,backend capability matrix and runtime fallback reporting.
- `export/`: torch.export,TorchScript,ONNX,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN 等导出 adapter.
- `xdl_adapter.py`: 从 XDL TrainSetup-like 对象或 checkpoint 提取模型上下文,不接管训练.

### Recipes

Recipe 按技术栈分层组织在 `recipes/` 下:

- `quant/` - 量化.
- `prune/` - 纯模型侧剪枝和 sparsity report.
- `operator/` - 算子优化.
- `detection/` - 检测模型部署和后端产物适配.
- `smoke/` - 综合冒烟测试.

不要新增 XQT 训练 recipe. `finetune`,`distill`,`recovery`,`QAT training` 等 recipe 应放到 XDL 或外部训练工具侧.

### 配置方式

XQT 只提供两种配置方式, 前者为第一选择, 原则上没有其他配置方式:

1. **`XQTOptimizationSession`** (第一配置): Python 交互式 session, 逐步编排 benchmark / prune / quant / operator / export / deploy / analyze. 适合探索, 调试和 notebook.
2. **YAML workflow** (第二配置): 声明式 recipe, 通过 `optimize_model()` 或 CLI 入口运行, 适合可复现批量实验.

不要新增 CLI 参数解析库, JSON-shaped Python dict 硬编码 workflow 或其他配置路径.

### 运行入口

- `from xqt import XQTOptimizationSession` - session 主入口, 即第一配置方式.
- `optimize_model("path/to/workflow.yaml")` - YAML workflow 主入口, 即第二配置方式.
- `xqt-run-workflow`: 运行 stage workflow recipe.
- `run_workflow.py`: workflow 命令入口模块,通过环境变量指定 stage workflow.

早期 `load_xqt_config`,`run_xqt_recipe`,`preflight_xqt_config` 和 `XQTConfig` pass recipe 链路只作为 `xqt.core` / `xqt.pipeline` 内部实现,不要从 `xqt` 顶层重新导出,也不要新增安装后命令入口.

## API 边界

- `xqt` 顶层导出进入 Provisional API: `optimize_model`, `load_optimization_config`, `XQTOptimizationSession`, `OptimizedModelResult`, `OptimizationConfig`, `OptimizationStageConfig`, `OptimizationStageResult`, `StageAcceptanceConfig`, `ArtifactManifest`, `ArtifactRecord`, `MetricRecord`.
- `xqt` 到 XDL 的模型产物适配入口进入 Provisional API: `xdl_setup_to_xqt_context`, `xdl_checkpoint_to_xqt_context`, `load_checkpoint_into_model`.
- 子模块内部实现按 Internal 处理,先服务 recipe 验证.

## 修改约束

- 新能力必须服务模型本身: quant,prune,operator optimization,export,diff,benchmark 或 manifest.
- 不要在 XQT 中新增训练循环,trainer,training provider,evaluation provider,loss wrapper 或任务 registry.
- 不要在 XQT 中新增 dataset/dataloader 构建或 task-level validation.
- 公共压缩或部署工具若可复用,再考虑抽到 `tools/` 或 `xdl/`.
- 外部库依赖,设备要求,模型限制要写清楚.
- 顶层示例脚本要保持薄入口,优先调用 XQT 已有 helper.
- 顶层示例脚本涉及可选依赖时使用 lazy import,失败时抛出清楚的 XQT 异常或错误信息.
- 新增 recipe 必须声明 `compression_axes` 和支持的硬件约束.
- 所有量化/剪枝/导出后都必须能产出模型侧 report 或最小性能基准.

## 注意事项

- 量化,部署和后端 adapter 通常对环境版本敏感,涉及外部 API 时先确认最新文档.
- TensorRT,OpenVINO,ncnn,MNN 和 ExecuTorch 仍以目标机器官方安装方式为准,XQT 只做 adapter 和 preflight 检查.
- 真实 FP8 收益必须依赖支持硬件验证,synthetic smoke 只用于本地闭环验证.
