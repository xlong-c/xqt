# XQT 架构

本文定义 `XQT` 的系统边界, 主链路和工程契约. 它是 `XQT` 架构层的长期事实源之一.

## 负责什么

- 说明 `XQT` 做什么, 不做什么.
- 说明主链路, 配置方式和 Stage 约定.
- 说明 profiling 与 manifest 的工程边界.
- 说明 `xqt.nn`, `wrapper/materialize`, `kernel` 的总体分工入口.


修改指导 (research): [GUIDE](../../research/xqt-quant-inference-architecture/GUIDE.md) (长期原则), [TODO](../../research/xqt-quant-inference-architecture/TODO.md) (当前冲刺), [PLAN](../../research/xqt-quant-inference-architecture/PLAN.md) (接口草案).

## 不负责什么

- 不提供完整训练流程.
- 不承载 task-level validation 设计.
- 不替代具体 runtime / engine 官方文档.
- 不单独展开 `kernel` / `wrapper` / `xqt.nn` 的细边界表.

## 项目定位

`XQT` 只关注模型本身. 它接收 PyTorch 模型, checkpoint 或导出产物, 执行模型侧压缩, 图变换, 导出适配, 误差分析和 benchmark.

需要梯度更新的流程归 `XDL` 或第三方训练工具, 再把训练后的模型或 checkpoint 交给 `XQT`.

在本仓库内, `XQT` 是唯一推理优化主体. `XDL` 不直接接管推理优化, 也不把 TensorRT / ONNX Runtime / OpenVINO 这类外部 runtime 包装成训练侧能力. XQT 自己拥有模型替换, 算子 contract, runtime 状态, benchmark, report 和 manifest.

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
- 以 Python API 为主入口表达模型变换, 例如 `XQTOptimizationSession`, `xqt.convert(...)` 和 `xqt.nn.*` facade.
- 以语义块替换作为推理优化目标: Python 层表达 `Linear`, `Conv`, `Norm`, `Attention`, `FeedForward`, `TransformerBlock`; 中间经 `wrapper/materialize` 做 contract 校验, candidate module 构造和 fallback 记录; engine 层再决定落成一个 kernel, 一组 kernel 或 megakernel.
- `Linear` / `Conv2d` / `LayerNorm` 是保留 PyTorch module/state_dict 语义并记录 runtime intent 的 facade; `FeedForward` / `RMSNorm` / `Attention` / `TransformerBlock` 也是 facade, 但这不表示它们天然等于某个 kernel pattern 或完整 megakernel 已实现.

非目标:

- 不替代 `xdl.trainer.Trainer`.
- 不维护通用 model / loss / metric registry.
- 不运行 `zero_grad() -> backward() -> step()` 训练循环.
- 不做 task-level validation 或 accuracy / mAP 评测闭环.
- 不重新实现 TensorRT, OpenVINO, ONNX Runtime, ExecuTorch, ncnn, MNN 等后端.

## Backend / Engine / Method 术语

XQT 文档和代码必须分词. **完整硬规则** 见 [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md).

| 词 | 含义 | 示例 |
| --- | --- | --- |
| quant `backend` | 量化适配路径 | `pytorch`, `torchao`, `onnxruntime_qdq` |
| quant `method` | 量化算法 | `awq`, `gptq`, `svd` |
| operator `engine` | 内部 kernel / lowering | `tilelang`, `triton`, `cutlass` |
| export/deploy `backend` | 外部 runtime | TensorRT, ONNX Runtime, OpenVINO |

禁止: `quant.params.backend=tilelang` 或 `=svdquant`; 禁止把 awq/gptq/svd 写成 engine methods.

`engine` 字段落点:

- `xqt.convert(..., engine=...)` (materialize preference)
- `operator_optimization.default_engine` / `targets[*].engine`
- `OptimizationCapability.engine`, `StageReport.engine`

用户入口不应把 `triton` / `tilelang` / `cute_dsl` / `custom_cuda` 理解成和 TensorRT / ONNX Runtime 并列的外部后端. 对推理优化来说, 对外主体是 `xqt`; engine 只是 XQT lowering contract 的实现选择.

推荐抽象:

```text
Public API:
  xqt.convert(model, engine=..., policy=...)
  xqt.nn.Linear / Conv2d / LayerNorm / FeedForward / RMSNorm
  xqt.nn.Attention / TransformerBlock

XQT contract:
  precision, layout, packing, fusion, runtime state, target architecture

XQT boundary layer:
  wrapper / materialize / candidate module / fallback metadata

XQT engine:
  triton, tilelang, cutlass, cute_dsl, cutile, custom_cuda
```

`kernel` 与 `xqt.nn` 的明确分割线见 [xqt-kernel-wrapper-nn-boundary.md](xqt-kernel-wrapper-nn-boundary.md). 文档和实现都不应跳过中间的 `wrapper/materialize` 层直接把 facade 等同于 kernel.

`engine="auto"` 这类策略未来应由 XQT capability 和 benchmark/report 决定. 文档不能把 metadata-only engine 写成已验证 executable path.

TileLang CUDA kernel entry 会先检查 runtime compatibility. 已知 packed-tensor ABI 不兼容时, entry 显式报错; operator wrapper 仅在 `fallback_policy` 允许时记录 eager fallback. 因此 package 可导入或静态 capability 为 `executable` 不等于本机完成 kernel correctness 或性能验收.

## XQT Kernels 统一内核命名空间

`xqt.kernels` 是 XQT 唯一的计算栈落点, 复刻 `sglang.kernels` (RFC #29630) 的四层模型, 并按三个子包收口: `ops` (tensor kernel + GEMM 合约), `wrappers` (materialize / operator / bench), `nn` (facade / convert / fixtures). 详细契约见 [xqt-kernels.md](xqt-kernels.md).

分层:

```text
1. Public API        xqt.kernels.ops.<group>.<op>()
2. Dispatch          select_kernel/get_kernel (固定) + BaseFusedOp.forward (择优, 按需)
3. Registry+Metadata registry.py + spec.py (torch-free, 惰性)
4. Backend x Device  KernelBackend(产地) x CapabilityRequirement(设备+SM窗口)
```

调用:

- `xqt.kernels.ops.<group>.<op>()` - 单算子默认路径, 薄封装, 签名与 `FormatSignature` 一致.
- `select_kernel(op, backend).load()` / `get_kernel(op, backend)` - 显式后端, 自研 vs `flashinfer` 等外部库 A/B.
- `BaseFusedOp.forward(backend=)` - 仅 `activation / layernorm` 等可互换多后端 op 的自动择优与 `XQT_FORCE_KERNEL_BACKEND` 一键切回.

实现仍可在 `xqt.kernels.jit` 或 `xqt.kernels.aot` 中开发, 但对外可调用路径必须经 `xqt.kernels.ops.*`. 新增内核或接入 `flashinfer` 仅需在 `xqt/kernels/ops/<group>/__init__.py` 加一行 `register_kernel(KernelSpec(..., backend=FLASHINFER, target="flashinfer:...", capabilities={CUDA}))`.

三层边界关系: `xqt.kernels.nn / torch module -> xqt.kernels.wrappers (contract 校验, candidate 构造) -> xqt.kernels.ops.* -> engine kernel`. `from xqt import nn` 是 `xqt.kernels.nn` 的公开别名, 顶层 `xqt/nn/` 已删除, 与 kernel 不直接耦合.

## 配置方式

`XQT` 只提供两种配置方式:

1. `XQTOptimizationSession`
2. YAML workflow, 通过 `optimize_model()` 运行

原则上不要新增 CLI 参数解析, JSON shaped workflow 或其他配置路径.

YAML workflow 的公开 schema 只有一套: `project`, `model`, `task`, `compression_axes`, `hardware`, `benchmark`, `stages`, `device`. 其中 `stages` 是唯一的优化和导出路径描述; 旧式顶层 `compression`, `export`, `operator_optimization`, `analysis`, `validation` 和 `config_version` 属于已删除的旧 recipe 形态,不能出现在 `xqt/recipes`.

当前 loader 已把 `stages[*].params` 解析为 typed `StageSpec`; workflow 主链也已通过 `run_quant_stage(...)`, `run_prune_stage(...)`, `run_operator_stage(...)`, `run_export_stage(...)`, `run_analyze_stage(...)`, `run_benchmark_stage(...)` 等入口直接消费 typed spec. public `create_context(...)` 已收窄为只接受 `OptimizationConfig`, workflow 映射和带 `stages` 的 workflow 路径, 并直接构造 runtime-only context; `preflight_optimization_config(...)` 是 workflow preflight 入口; `xdl_setup_to_xqt_context(...)` / `xdl_checkpoint_to_xqt_context(...)` 也已收窄为只接受 `OptimizationConfig` 或 workflow 输入.

同时 `XQTContext` 已 workflow 去配置化: `device`, `artifact_dir`, `project_name`, `task_type`, `compression_axes`, `model_target`, `model_params`, `quant_config`, `prune_config`, `analysis_config`, `benchmark_config`, `operator_config`, `output_diff_config`, `export_targets` 已独立成 runtime 字段; `LoadModelPass`, `run_quant_stage(...)`, `run_prune_stage(...)`, `AnalyzePass`, `BenchmarkPass`, `OperatorOptimizationPass`, `ExportPass`, `fake_qdq` 和 workflow CLI 输出已直接走这些字段, 不再用旧 `context.config` 兜底当前 stage runtime 配置. 旧 `QuantPass` / `PrunePass`, `run_xqt_recipe(...)`, `build_pipeline_from_config(...)`, pass order helper, `preflight_xqt_config(...)`, `create_manifest(XQTConfig)`, `xqt_config_to_dict()`, `load_xqt_config()` 和 `XQTConfig` 已删除. 配置单轨化主路径已经完成, 后续重点转向拆大文件, contract 层和能力面收敛.

## 关键抽象

| 概念 | 说明 |
| --- | --- |
| `OptimizationConfig` | 对外 stage workflow schema |
| `ModelAdapter` / `ModelProfile` | `xqt.model` 的模型适配实现与声明记录,当前承载 HunyuanOCR,Unlimited-OCR,Wan 2.1,Flux.2 Klein. adapter 可负责架构组装,checkpoint mapping,特殊 forward 和 IO 包装; profile 通过 `model.profile` 选择 adapter. 通用 layer,operator,kernel 仍由各自模块提供 |
| `StageSpec` | loader 后的 typed stage 参数 |
| `XQTOptimizationSession` | 交互式 stage 编排入口 |
| `ArtifactManifest` / `ArtifactRecord` | 产物追踪 |
| `ModelPackageManifest` / `load_model_package()` | 推理侧文件加载标准, 当前最小闭环为 `manifest.json + runtime/config.json` |
| `InferenceContract` / `create_inference_session()` | 模型侧语义 IO 标准. 以模型族 adapter 复用预处理和输出规范化, 每个模型只声明 contract config |
| `write_quant_pair` / `load_quant_pair` | 扁平 Infer 交付 `model.pt` + `quant.json`; 可挂 `RuntimeQuantContract` |
| `RuntimeQuantContract` / `RuntimeManifest` | 自研 quant 事实源与聚合 manifest (非 HF adapter) |
| `LayoutKernelReport` | load/process/apply 诊断 (含 selected_kernel) |
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
- `ArtifactManifest` 是 workflow / experiment 追踪, 不是推理加载 contract. file-based inference 只消费标准模型包 `manifest.json`, 不直接解析 quant recipe, workflow YAML 或算法私有配置.
- ONNX target 的已知配置收敛为 `targets[*].onnx`: input/output names, `dynamo`, validate, runtime diff, pre-export fusion/lowering 和 ONNX optimization 不再藏在 target `params`. `pre_export_lowering.fp4_weight_only_to_dense_linear` 只在 export copy 上把 `FP4WeightOnlyLinear` materialize 成等价 dense dequantized `nn.Linear`, 并把 source/target storage 与 note 写入 artifact metadata; TensorRT artifact 因此不是 packed-FP4 runtime. `dynamic_shapes` 是唯一的动态 shape 字段: `dynamo=true` 直传现代 exporter, `dynamo=false` 规范化为 legacy ONNX `dynamic_axes`, 必须与 TensorRT profile 一致.
- 当前 runtime 标准包最小闭环由 export 自动产出 `*.xqtpkg/manifest.json`, 并配套 `model/*` 与 `runtime/config.json`. 现阶段只保证 `ONNX + ONNX Runtime` 闭环; inference 入口通过 `xqt.runtime.load_model_package()` 和 `create_inference_runner()` 消费该包.
- semantic inference 通过 `xqt.runtime.create_inference_session()` 进入. `create_inference_runner()` 是 raw tensor runner, `create_inference_session()` 读取 manifest 的 `inference` contract, 选择模型族 adapter, 执行 `preprocess -> runtime -> postprocess`. 旧 manifest 缺少 `inference` 时自动回退到 `tensor` adapter.
- 当前内置 adapter 为 `tensor` 和 `vision.classification`; 可复用的模型族实现通过 `register_inference_adapter()` 注册一次,模型只在 contract 中声明 adapter 名称和 config. `InferenceContract` 只描述模型侧 IO 和 adapter 配置, 不承载 dataset, task-level validation, serving scheduler, tokenizer 训练逻辑或在线 batch 调度.
- TensorRT engine-build 的已知配置收敛为 `targets[*].tensorrt`: `onnx_path`, `backend`, `trtexec_path`, `extra_args`, `timeout`, `dry_run`, `performance_thresholds`, `workspace_mib`, `builder_optimization_level`, `timing_cache_path`, `log_level`, `plugin_libraries`, `serialize_plugin_libraries`, `validate_plugin_libraries_loadable` 与 `runtime_benchmark`. 同名 target `params` 旧键由 loader 明确拒绝; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口也经 `tensorrt` 参数走同一 StageSpec 解析路径. engine-build plugin 配置只作用于构建, 不会隐式成为 runtime handle 配置.
- OpenVINO 的已知配置收敛为 `targets[*].openvino`: `onnx_path`, `input_shape`, `dry_run`, `runtime_diff` 与 `device`. 同名 target `params` 旧键由 loader 明确拒绝; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口也经 `openvino` 参数走同一 StageSpec 解析路径. 未声明 `openvino.onnx_path` 时, export pass 依序使用同 workflow 的先前 ONNX artifact 与当前模型; `runtime_diff` 只在 materialized IR 和 reference output 可用时执行.
- TorchExport 的已知配置收敛为 `targets[*].torch_export`: `strict`, `validate` 与 `runtime_diff`. TorchScript 的已知配置收敛为 `targets[*].torchscript`: `method`, `check_trace` 与 `runtime_diff`, 其中 `method` 只能是 `trace` 或 `script`. 同名 target `params` 旧键由 loader 明确拒绝; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口分别经 `torch_export` 与 `torchscript` 参数走同一 StageSpec 解析路径.
- ExecuTorch 的已知配置收敛为 `targets[*].executorch`: `dry_run`. ncnn 的已知配置收敛为 `targets[*].ncnn`: `source_path`, `converter`, `onnx2ncnn_path`, `pnnx_path`, `bin_path`, `extra_args`, `timeout` 与 `dry_run`; `converter` 只能是 `onnx2ncnn` 或 `pnnx`. pnnx 未显式配置 source 时优先使用同 workflow 的 TorchScript artifact, 再回退 ONNX, 使 preflight 与 ExportPass 选择同一 converter. MNN 的已知配置收敛为 `targets[*].mnn`: `source_path`, `converter_path`, `framework`, `extra_args`, `timeout` 与 `dry_run`. 同名 target `params` 旧键由 loader 明确拒绝; 三者的 Session 单 target 入口也经对应 typed 参数走同一 StageSpec 解析路径. dry-run preflight 不要求可选依赖或 converter executable 已安装.
- materialized deploy runtime handle 的 known config 也已脱离无类型 `params`: ONNX Runtime providers 位于 `runtime_handle.onnxruntime`, TensorRT device 与 runtime plugin libraries 位于 `runtime_handle.tensorrt`. TensorRT runtime session 不再从 engine-build target 隐式继承 plugin libraries.
- `PrecisionPolicy` 是 module conversion, `xqt.nn` facade runtime intent 与 GEMM 的共享精度 contract, 落点是 `xqt.kernels.precision` (不进 `kernels.nn`, 因为 `ops/_impl` 不能 import facade). `MatmulPrecisionSpec` 只在 `gemm_precision` 和 `conversion` 的兼容导入位置作为 `PrecisionPolicy` identity alias 保留; 不再有第二套字段 schema 或双向转换. facade 的 `auto` 仍表示输入 dtype 延迟决策, 但名称与角色字段的规范化同样来自该 contract.

### Session stage 协议

`XQTOptimizationSession` 内部正式维护 `SessionStage` 图, 初始化时会注册 `baseline` 作为第一个 stage. 旧的 `OptimizationStageResult` 仍保留运行结果语义, `SessionStage` 负责表达阶段产物, lineage, parent, persistence 和 capability.

核心约定:

- `SessionStage.payload` 表达当前阶段产物语义.
- `model_snapshots` 表达 session 可恢复的模型态.
- `use(name)` / `revert_to(name)` 优先恢复模型 snapshot, 不把 runtime plan 或 export bundle 误当成模型.
- provider 逻辑收敛在 `xqt/workflows/stage_provider.py`, `optimization.py` 只消费 provider 输出.

当前 payload kind:

- `torch_module`: baseline, benchmark, analyze 等默认模型态 stage.
- `pruned_model`: prune stage, 对应 `xqt.contracts.PrunedModelPayload`; 保留模型,method,目标/实际 sparsity,execution state 和完整 prune report, 不把裸模型误当成剪枝产物语义.
- `quantized_model`: quant stage, 对应 `xqt.contracts.QuantizedModelPayload`; 它继承模型侧通用 `QuantizedModel` contract, 只额外记录 stage lineage, artifacts 和 capability. `xqt.workflows.stage` 仅重导出该 artifact contract.
- `runtime_plan`: operator stage, 对应 `RuntimePlanPayload`.
- `export_bundle`: export stage, 或仅构建 deploy artifact 而未 materialize runtime handle 的 deploy stage, 对应 `ExportBundlePayload`.
- `runtime_handle`: materialized deploy stage 的 executable runtime handle, 对应 `RuntimeHandlePayload`. deploy stage 可创建 ONNX Runtime `InferenceSession`, 或反序列化同 stage 的非 dry-run TensorRT engine 并创建 execution context. runtime session creation 不替代数值或性能验收.

当前 transform-side provider:

- `DefaultStageProvider`
- `ModelQuantizerProvider`
- `OperatorOptimizerProvider`
- `ExportProvider`

session 内比较使用 `XQTOptimizationSession.compare_stages()` 或 `compare_to_baseline()`, 返回 `StageComparison`. 它只做 stage / payload / metric / artifact 层面的结构化摘要, 不引入 task-level validation.

## Profiling 约定

`XQT` 允许理解和记录性能分析工具, 但 profiling 只作为模型侧 `benchmark` / `operator` / `export` / `deploy` 的诊断补充.

- `benchmark` 负责稳定 latency / memory / throughput 基线, profiler 负责解释瓶颈
- 外部 profiler 输出应作为 artifact 挂到 manifest
- report 至少记录 engine, device, target artifact, input shape, warmup, repeat, precision, batch size, profiler 名称和关键参数
- `XQT` 可以提供 profiler preflight 和命令模板, 但不要封装厂商 profiler 的完整 CLI
- profiling 不能引入 dataset / dataloader, evaluation provider, task-level validation 或训练循环

## 修改约束

- 新能力必须服务模型本身: quant, prune, operator optimization, export, diff, benchmark 或 manifest
- 不要在 `XQT` 中新增训练循环, trainer, training provider, evaluation provider, loss wrapper 或任务 registry
- 不要在 `XQT` 中新增 dataset / dataloader 构建或 task-level validation
- 顶层示例脚本保持薄入口, 优先调用 `XQT` 既有 helper

## 继续阅读

- [../explanation/xqt-concepts.md](../explanation/xqt-concepts.md)
- [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md)
- [xqt-kernel-wrapper-nn-boundary.md](xqt-kernel-wrapper-nn-boundary.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- [xqt-realignment-guide.md](xqt-realignment-guide.md)
- [../XQT.md](../XQT.md)
