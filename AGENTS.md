# xqt 仓库 - 模型压缩与部署工具链

本文件是 `xqt` 独立仓库的权威工作区规范. `xqt/AGENTS.md` 保留为包内子目录规范, 内容与本文件一致, 但路径以包内视角书写; 冲突时以本文件为准.

## 开发阶段

当前处于 v0.x 开发期,在目标和范围已对齐后,可以按现有方案直接重构,不需要围绕旧接口做兼容层.

## 核心契约

XQT 只关注模型本身.

XQT 负责模型压缩,图变换,导出适配,误差分析和 benchmark. XQT 不负责训练,QAT,finetune,distillation,recovery,dataset / dataloader,training provider 或 evaluation provider. 需要梯度更新或任务验证的流程归 XDL 或第三方工具,再把训练后的模型 / checkpoint 或指标交给 XQT.

包内工程契约见 `xqt/FRAMEWORK.md`;长期事实源见 `docs/md/XQT.md`.

训练侧兄弟仓库 `XDL` 与本仓库通过 checkpoint / 模型产物衔接. `XDL` 是可选运行时依赖, 由工作区内的本地 editable 安装提供, 本仓库不声明也不固定其版本.

## 当前内容

### 包模块

- `xqt/core/`: structured config, workflow/stage schema, artifact manifest 和 checksum. `core/base/` 是供 `contracts` 依赖的 leaf 层; `core` 高层可以依赖 `contracts`, `contracts` 不反向依赖 `core` 高层.
- `xqt/contracts/`: typed payload, quantized storage protocol, reference semantics 和 runtime handoff contract. `contracts` 可以提供 artifact 的 reference forward, 但不依赖 `compression/quant/`, `runtime/`, `export/` 或 `kernels/`; backend execution view 归 `runtime/`, packing 实体只保留一份. engine resolve 在 `kernels/engine_resolve.py`; NVFP4 unpack 在 `kernels/ops/quantization/nvfp4.py`, bridge 在 `kernels/wrappers/nvfp4.py`; `PrecisionPolicy` / `ModuleContract` 在 `kernels/precision.py`.
- `xqt/model/`: 具体模型适配实现与声明式 profile. 适配器可以负责模型架构组装,checkpoint 映射,特殊 forward 和输入输出包装; 通用 layer,operator,kernel,quant,prune 实现仍归对应模块. profile 只选择 adapter 并记录兼容性元数据.
- `xqt/kernels/`: 计算栈唯一落点. `ops/` 是 tensor kernel + GEMM 合约, `wrappers/` 是 materialize / operator / bench, `nn/` 是 facade / convert / fixtures. `from xqt import nn` 与 `xqt.convert` 仍是公开别名, 实现在 `kernels/nn/`. `gemm/`, `conversion_impl/`, `operator_opt/`, `benchmark/` 与顶层 `nn/` 已删除; GEMM 只在 `kernels/ops/gemm/` 与 `kernels/ops/_impl/gemm_backends/`, convert 实现只在 `kernels/nn/conversion/`, operator 实现只在 `kernels/wrappers/` 与 `kernels/ops/_impl/`, bench 只在 `kernels/wrappers/bench/`, smoke fixture 只在 `kernels/nn/fixtures/`.
- `xqt/pipeline/`: workflow 使用的内部执行层,负责 context 构建,preflight,stage pass 和 export handler;用户编排入口仍是 `workflows/`.
- `xqt/workflows/`: stage-based model optimization workflow,支持 `benchmark`,`prune`,`quant`,`operator`,`export`,`deploy`,`analyze`.
- `xqt/analysis/`: tensor output diff,layer analysis 和 report helper.
- `xqt/compression/`: 模型侧压缩唯一落点. `quant/` 是量化子系统 (policy/strategy/capability/plan, execution, quantizers, backends, calibration); `prune/` 是 unstructured / structured / N:M / block sparse 剪枝. quantizer 只产出 contracts storage shell 和 report, 不 import runtime execution view.
- `xqt/runtime/`: 混合推理引擎. 只消费已量化 artifact 与 execution policy, 做模块级 / 通道级混合精度调度 (`HybridInferenceEngine`, `apply_execution_policy`, `ChannelHybridSpec`); 不跑 quantizer / calibration / sensitivity.
- `xqt/export/`: torch.export,TorchScript,ONNX,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN 等导出 adapter.
- `xqt/xdl_adapter.py`: 从 XDL TrainSetup-like 对象或 checkpoint 提取模型上下文,不接管训练,只接受 `OptimizationConfig` 或 workflow 输入,不接旧 recipe schema.

### Recipes

Recipe 按技术栈分层组织在 `xqt/recipes/` 下:

- `quant/` - 量化.
- `prune/` - 纯模型侧剪枝和 sparsity report.
- `operator/` - 算子优化.
- `detection/` - 检测模型部署和后端产物适配.
- `smoke/` - 综合冒烟测试.

`xqt/recipes/` 下所有 YAML 都必须是 stage workflow, 且必须能由 `load_optimization_config()` 加载. 顶层只保留 `project`, `model`, `task`, `compression_axes`, `hardware`, `benchmark`, `stages`, `device`; 优化和导出路径全部写入 `stages`. 不要再新增旧式顶层 `compression`, `export`, `operator_optimization`, `analysis`, `validation` 或 `config_version`.

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
- `xqt/run_workflow.py`: workflow 命令入口模块,通过环境变量 `XQT_WORKFLOW_CONFIG` 指定 stage workflow.

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
- 每次落地可执行推理优化 (新 kernel,layout/prepack,fusion,auto route 或已验证的 tile/warp/pipeline 取舍) 时,必须同步更新 [docs/md/explanation/operator-optimization-records.md](docs/md/explanation/operator-optimization-records.md). 记录目标,公平 baseline,测量方法,数值正确性,适用/回退边界,未采纳方案和可复用规则;没有证据的内容只能标为假设或待验证.

## 目录规范

- 训练侧入口,数据集和 checkpoint 交接不在本仓库; 数据集与本地数据一律放 `data/` (整体 gitignore).
- 阶段性研究资料放独立的 `../research` 仓库 (本仓库内以软链 `research/` 挂入), 不能替代长期文档.
- 长期事实源与工作文档放 `docs/md/`; HTML 阅读页放 `../research` 下, 样式规范见 `docs/md/architecture/html-style-policy.md`.
- `tests/` 直接对应当前包的测试树, 不再有 monorepo 时期的 `tests/xqt/` 中间层.
- 仓库维护脚本放 `scripts/`, 工程工具放 `tools/`.

## decode 加速链路 (改内核前先看)

端到端链路跨五个文件, 改任何一处都要按整条链路验证, 不能只看单 kernel 微基准:

- `xqt/runtime/graph_decode.py` - `CudaGraphDecodeSession`: 单张 length-agnostic graph, 逐层 decode body, 残差 epilogue 绑定, `decode_batch` 分块回读.
- `xqt/kernels/ops/_impl/triton/decode_kernels.py` - 单遍 GQA decode attention (partial + merge; SIMT 默认, `attention_impl="tc"` 走张量核组变体 R-057), 融合 RoPE/KV scatter, RMSNorm/SwiGLU int8.
- `xqt/kernels/jit/csrc/quantization/awq_w4a16_sm89_kernel.cu` - native W4A16 decode GEMV (interleave-4 打包; `HasBias` 语义就是残差 epilogue).
- `xqt/runtime/modules/awq_w4a16_linear.py` + `xqt/model/minicpm5.py` - 模块入口与 hybrid 视图; `bind_residual` 必须一路转发到 hybrid 视图, 否则运行时绑不上 (曾因此静默退化成 `residual + module(x)`).
- `examples/xqt_models/minicpm5_2b_graph_decode.py` - 五路线 e2e 基准, 产物 `artifacts/xqt/inference/minicpm5-2b/graph_decode_benchmark.json`.

测量纪律 (R-052/053/054 的代价换来的):

- **合成微基准对本链路没有预测力** (多次出现 "合成链快 12%, 真实路线慢 12%"), 内核参数或结构改动必须直接跑真实路线 A/B.
- 单次稳态读数跨运行可波动 (同配置 265-311 tok/s), 加速比结论只看**同一次运行内**的对照.
- 改内核前先读 `docs/md/explanation/operator-optimization-records.md` 里最新的 R-0xx: 那里记着本机带宽上界, 分形状实测和所有已否决方案, 不要重复试已证伪的路.

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
