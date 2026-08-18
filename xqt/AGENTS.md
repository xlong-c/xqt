# xqt - 模型压缩与部署实验目录

## 开发阶段

当前处于 v0.x 开发期,在目标和范围已对齐后,可以按现有方案直接重构,不需要围绕旧接口做兼容层.

## 核心契约

XQT 只关注模型本身.

XQT 负责模型压缩,图变换,导出适配,误差分析和 benchmark. XQT 不负责训练,QAT,finetune,distillation,recovery,dataset / dataloader,training provider 或 evaluation provider. 需要梯度更新或任务验证的流程归 XDL 或第三方工具,再把训练后的模型 / checkpoint 或指标交给 XQT.

包内工程契约见 `xqt/FRAMEWORK.md`;长期事实源见 `docs/md/XQT.md`.

## 当前内容

### 包模块

- `core/`: structured config, workflow/stage schema, artifact manifest 和 checksum.
- `contracts/`: typed payload, quantized storage protocol, reference semantics 和 runtime handoff contract. `contracts` 可以提供 artifact 的 reference forward, 但不依赖 `quant/`, `runtime/` 或 `export/`; backend execution view 归 `runtime/`, packing 实体只保留一份.
- `model/`: smoke-only model helper 和模型 forward hook 输出采集工具.
- `pipeline/`: sequential pass manager,preflight 和 YAML runner.
- `workflows/`: stage-based model optimization workflow,支持 `benchmark`,`prune`,`quant`,`operator`,`export`,`deploy`,`analyze`.
- `analysis/`: tensor output diff,layer analysis 和 report helper.
- `benchmark/`: latency 和 memory benchmark helper.
- `quant/`: 量化子系统. 根目录保留 policy/strategy/capability/plan/types 等 schema 和事实源; `execution/` 负责 plan dispatch 与 report 组装; `quantizers/` 放模型侧量化算法实现,如 FP4 weight-only,MXFP weight-only,ConvRot W4A4,SVD 以及后续 AWQ/GPTQ; quantizer 只产出 contracts storage shell 和 report, 不 import runtime execution view; `backends/` 放 torchao/onnxruntime_qdq 等外部 runtime 或导出适配; `calibration/` 放 activation calibration 和 calibration summary.
- `runtime/`: 混合推理引擎. 只消费已量化 artifact 与 execution policy, 做模块级 / 通道级混合精度调度 (`HybridInferenceEngine`, `apply_execution_policy`, `ChannelHybridSpec`); 不跑 quantizer / calibration / sensitivity.
- `nn/`: 转换向 facade (`Linear` / `Attention` / `FeedForward` 等), 承载 engine 与 precision intent; 与 quant artifact 解耦.
- `prune/`: unstructured,structured,N:M 和 block sparse pruning helper.
- `operator_opt/`: `torch.compile`-first operator optimization pass,backend capability matrix and runtime fallback reporting.
- `export/`: torch.export,TorchScript,ONNX,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN 等导出 adapter.
- `xdl_adapter.py`: 从 XDL TrainSetup-like 对象或 checkpoint 提取模型上下文,不接管训练,只接受 `OptimizationConfig` 或 workflow 输入,不接旧 recipe schema.

### Recipes

Recipe 按技术栈分层组织在 `recipes/` 下:

- `quant/` - 量化.
- `prune/` - 纯模型侧剪枝和 sparsity report.
- `operator/` - 算子优化.
- `detection/` - 检测模型部署和后端产物适配.
- `smoke/` - 综合冒烟测试.

`recipes/` 下所有 YAML 都必须是 stage workflow, 且必须能由 `load_optimization_config()` 加载. 顶层只保留 `project`, `model`, `task`, `compression_axes`, `hardware`, `benchmark`, `stages`, `device`; 优化和导出路径全部写入 `stages`. 不要再新增旧式顶层 `compression`, `export`, `operator_optimization`, `analysis`, `validation` 或 `config_version`.

ONNX target 的已知字段只写在 `stages[*].params.targets[*].onnx`: `input_names`, `output_names`, `dynamo`, `validate`, `runtime_diff`, `pre_export_fusion`, `pre_export_lowering` 与 `optimization`. 模型侧语义推理 contract 写在同一 target 的 `inference`: `schema_version`, `family`, `adapter`, `adapter_version`, `inputs`, `outputs`, `config` 与 `metadata`; 它不属于 backend `params`,也不承载 dataset 或 task-level validation. TensorRT engine-build 已知字段只写在 `stages[*].params.targets[*].tensorrt`: `onnx_path`, `backend`, `trtexec_path`, `extra_args`, `timeout`, `dry_run`, `performance_thresholds`, `workspace_mib`, `builder_optimization_level`, `timing_cache_path`, `log_level`, `plugin_libraries`, `serialize_plugin_libraries`, `validate_plugin_libraries_loadable` 与 `runtime_benchmark`. OpenVINO 已知字段只写在 `stages[*].params.targets[*].openvino`: `onnx_path`, `input_shape`, `dry_run`, `runtime_diff`, `device` 与 `benchmark` (`enabled` / `warmup` / `iterations` / `measure_memory`). TorchExport 已知字段只写在 `stages[*].params.targets[*].torch_export`: `strict`, `validate` 与 `runtime_diff`. TorchScript 已知字段只写在 `stages[*].params.targets[*].torchscript`: `method`, `check_trace` 与 `runtime_diff`. ExecuTorch 已知字段只写在 `stages[*].params.targets[*].executorch`: `dry_run`. ncnn 已知字段只写在 `stages[*].params.targets[*].ncnn`: `source_path`, `converter`, `onnx2ncnn_path`, `pnnx_path`, `bin_path`, `extra_args`, `timeout` 与 `dry_run`; `converter=pnnx` 可显式使用 ONNX 或 TorchScript source. MNN 已知字段只写在 `stages[*].params.targets[*].mnn`: `source_path`, `converter_path`, `framework`, `extra_args`, `timeout` 与 `dry_run`. QNN 已知字段只写在 `stages[*].params.targets[*].qnn`: `source_path`, `converter_path`, `extra_args`, `timeout` 与 `dry_run`; 真实转换依赖目标机 Qualcomm QNN SDK, XQT 只做命令 adapter, preflight 与 artifact metadata. 不要把这些字段放回 target `params`; loader 会明确拒绝所有已知 target 配置的同名旧键.

materialized deploy runtime handle 的已知字段也不能写入 `runtime_handle.params`: ONNX Runtime 的 providers 写在 `runtime_handle.onnxruntime.providers`; TensorRT 的 device 与 runtime plugin libraries 写在 `runtime_handle.tensorrt`. TensorRT runtime handle 若依赖 plugin, 必须在该 runtime config 中显式列出, 不从 engine-build target 隐式继承.

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

早期 `run_xqt_recipe`, `preflight_xqt_config`, `create_manifest(XQTConfig)`, `xqt_config_to_dict()`, `load_xqt_config()` 和 `XQTConfig` 已删除,不要恢复兼容入口. public `create_context()` 只接受 `OptimizationConfig` 或 workflow 输入,不接旧 recipe mapping. 不要从 `xqt` 或 `xqt.core` 聚合入口重新导出旧 schema/loader,也不要新增安装后命令入口.

## API 边界

- `xqt` 顶层导出进入 Provisional API: `optimize_model`, `load_optimization_config`, `XQTOptimizationSession`, `OptimizedModelResult`, `OptimizationConfig`, `OptimizationStageConfig`, `OptimizationStageResult`, `StageAcceptanceConfig`, `ArtifactManifest`, `ArtifactRecord`, `MetricRecord`, `XQTReadinessReport`, `XQTReadinessScenario`, `assess_xqt_readiness`.
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
- 每次落地可执行推理优化 (新 kernel,layout/prepack,fusion,auto route 或已验证的 tile/warp/pipeline 取舍) 时,必须同步更新 [docs/md/explanation/operator-optimization-records.md](../docs/md/explanation/operator-optimization-records.md). 记录目标,公平 baseline,测量方法,数值正确性,适用/回退边界,未采纳方案和可复用规则;没有证据的内容只能标为假设或待验证.

## Profiling 工具约定

- XQT 可以记录和消费 profiler 产物,但不接管厂商 profiler 的安装,权限,驱动版本或 GUI 工作流.
- `benchmark` stage 给出 latency / memory / throughput 基线;`ncu`,`nsys`,`rocprof`,`vtune`,`msprof`,XProf 等 profiler 只用于瓶颈归因和优化线索.
- profiler 输出应作为 artifact 进入 manifest,并在 report 中记录 backend,device,target artifact,input shape,warmup,repeat,precision,batch size,profiler 名称,关键参数和环境版本.
- 允许做 profiler preflight 或命令模板;不要封装厂商 profiler 的完整 CLI,不要新增 dataset / dataloader,evaluation provider,task-level validation 或训练循环.
- 常见对应关系: NVIDIA `nsys` / `ncu`,AMD `rocprof-sys` / `rocprof`,Intel VTune,Apple Instruments / Metal Debugger,Arm Streamline,Qualcomm Snapdragon Profiler,Google XProf / TensorBoard Profile,华为 Ascend `msprof`.

## 注意事项

- 量化,部署和后端 adapter 通常对环境版本敏感,涉及外部 API 时先确认最新文档.
- TensorRT,OpenVINO,ncnn,MNN 和 ExecuTorch 仍以目标机器官方安装方式为准,XQT 只做 adapter 和 preflight 检查.
- 真实 FP8 收益必须依赖支持硬件验证,synthetic smoke 只用于本地闭环验证.
