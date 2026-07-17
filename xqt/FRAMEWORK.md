# xqt 包内工程契约

**核心边界: XQT 只关注模型本身.**

XQT 消费训练后的模型/checkpoint/导出产物,做压缩,变换,导出,误差分析和 benchmark. XQT 不做训练,不做 QAT,不做 finetune / distill / recovery,不构建 dataset / dataloader,也不拥有 task provider 语义.

## 关键抽象

| 概念 | 说明 |
|---|---|
| `OptimizationConfig` | 对外唯一 YAML workflow schema,包含 `project` / `model` / `task` / `compression_axes` / `hardware` / `stages`. |
| `StageSpec` | `stages[*].params` 经 loader 解析后的 typed stage 参数. |
| `load_optimization_config()` | `OptimizationConfig` 加载器. |
| `XQTOptimizationSession` | 交互式 stage 编排入口. |
| `SessionStage` / `StagePayload` | session 内部 stage 图和阶段产物协议. |
| `QuantizedModel` / `QuantizedModelPayload` | `QuantizedModel` 是模型侧量化算法的通用语义结果; Infer 交接面是 `model` + 可选 `compute_config` (`infer_handoff()`). `backend`/`method`/`strategy` 仅 quant lineage. `QuantizedModelPayload` 追加 stage lineage, artifacts 和 capability. |
| `ComputeConfig` / `ModuleComputeSpec` | 可选计算配置: `compute_contract`, `precision`, `required_capabilities`, `preferred_engines` (hint). 禁止 `required_engine` 主键. 见 `docs/md/architecture/xqt-infer-handoff.md`. |
| `PrunedModelPayload` | `xqt.contracts` 定义的 prune stage typed payload, 记录模型,sparsity report,lineage,artifacts 和 runtime capability. |
| `RuntimePlanPayload` | operator stage 的 runtime plan typed payload. |
| `ExportBundlePayload` | export stage, 或未 materialize runtime handle 的 deploy stage 的 export bundle typed payload. |
| `RuntimeHandlePayload` | executable runtime handle typed payload; materialized deploy stage 使用该 payload, 可生产 ONNX Runtime `InferenceSession` 或非 dry-run TensorRT runtime session. |
| `StageComparison` | session 内 stage-to-stage 结构化比较结果. |
| `ArtifactManifest` / `ArtifactRecord` | 产物追踪. |
| `ModelPackageManifest` / `load_model_package()` | 推理侧文件包加载标准, 当前最小闭环为 `manifest.json + runtime/config.json`; 可选 `runtime/compute.json` (compute_config). |
| `write_quant_pair` / `load_quant_pair` | 扁平 Infer 交付: `model.pt` + `quant.json` (`artifact_type=xqt_quant_sidecar`). `quant.json` 是 compute_config + lineage sidecar, 不是 quant recipe; 加载不跑 quantizer. |
| `MetricRecord` | 结构化指标记录. |
| `OptimizationCapability` | 统一 capability 投影,覆盖 quant / prune / operator / export 的 engine,status,maturity,runtime,artifact_kind 和硬件/校准/导出要求. |
| `XQTReadinessReport` / `assess_xqt_readiness()` | readiness 汇总入口,输出场景状态,capability matrix 和 reporting schema. |
| `example_inputs` | 导出 / benchmark / layer analysis / operator_opt 所需输入. |
| `calibration_inputs` | PTQ / QDQ calibration 所需输入. |

## Backend / Engine 术语

完整规则: `docs/md/architecture/xqt-engine-quant-boundary.md` (method / storage / compute / engine 分词与禁止清单).

XQT 是本仓库内唯一推理优化主体. Python API 是主入口, 包括 `XQTOptimizationSession`, `xqt.convert(...)`, `xqt.nn.*` facade 和 `xqt.runtime` hybrid inference engine.

- `xqt.quant` 只负责量化算法与 artifact (packed weight, scale, rotation, execution policy / compute_config metadata, channel hybrid mask). 不把 operator engine 名写成推理必选主键.
- `xqt.runtime.HybridInferenceEngine` 只消费已量化模型与 execution policy / compute_config, 做模块级 / 通道级混合精度推理调度; 不跑 quantizer / calibration / sensitivity.
- `xqt.runtime.engine_resolve` 按 `required_capabilities` (+ 可选 preferred_engines hint) 解析 operator engine; 不是 quant method 选择.
- Operator engine 只管算子实现 / 融合 / MMA lowering. AWQ / GPTQ / SVD 是 quant **method**, 不是 engine methods.
- `ArtifactManifest` 只用于 workflow / experiment 追踪, 不是 file-based inference 的加载契约. 推理侧文件入口二选一: (1) 模型包 `manifest.json` (`load_model_package`); (2) 扁平 `model.pt` + `quant.json` (`load_quant_pair` / `load_quant_pair_into_model`). 二者都只消费已量化存储 + 可选 `compute_config`, 不解析 quant recipe YAML.
- 通道级混合精度: 部分 channel 走 16-bit (或更高), 其余走 4-bit. Quant 侧选 outlier channel 并写入 mask; Runtime 侧 dual-path reference 前向 (`channel_hybrid_linear_reference`), 后续可替换为真实 kernel.

- `backend`: 外部 quant/export/runtime 选择, 例如 `torchao`, `pytorch`, `onnxruntime_qdq`, `tensorrt`, `openvino`. Quant recipe 继续使用 `quant.params.backend`. **`tilelang` / `svdquant` 不是 quant backend**; AWQ/GPTQ/SVD 写 `backend=pytorch` + `method=awq|gptq|svd`.
- `engine`: XQT 内部实现选择和公开 report 字段, 例如 `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `custom_cuda`, `torch_compile`. `xqt.convert(...)`, 单算子 dispatcher, `OptimizationCapability`, `StageReport`, `operator_optimization.default_engine` 和 `targets[*].engine` 统一使用这个字段.
- 不要把 operator engine 写成 quant/export backend alias. 旧 operator 调用点要迁移到 `engine`, 但 quant/backend 语义不能硬改名.
- `maturity`: capability 的实现成熟度分层, 当前统一为 `executable`, `reference_guarded`, `metadata_only`, `planned`. `status` 继续表达接口/适配可用性, 不与 maturity 混用.

不要把 `triton` / `tilelang` / `cute_dsl` / `custom_cuda` 写成和 TensorRT / ONNX Runtime 并列的外部 inference backend. 对推理优化来说, 对外主体是 `xqt`; engine 描述 XQT 如何 lower 某个语义块或算子 contract.

TileLang CUDA kernel entry 会先验证 runtime compatibility. 已知 packed-tensor ABI 不兼容版本 (当前名单含 `0.1.11`) 必须显式失败; operator wrapper 只能依照 `fallback_policy` 记录 eager fallback. `0.1.12` 已在本机 `sm_89` 通过 correctness smoke, 但 package 可导入或静态 capability 为 `executable` 仍不等于完整生产性能验收.

## 配置方式

XQT 只提供两种配置方式:

1. `XQTOptimizationSession`.
2. YAML workflow,通过 `optimize_model()` 运行.

原则上不要新增 CLI 参数解析,JSON shaped workflow 或其他配置路径.

YAML workflow 只保留一种配置形态:

- `project`: 项目名和输出目录,`project.artifact_dir` 是优化模型,导出产物,report 和 manifest 的落点.
- `model`: 输入模型或 checkpoint 信息. 调用方也可以通过 `optimize_model(..., model=...)` 直接传入模型对象.
- `task`: 只描述模型侧 task metadata,不引入 dataset / dataloader 或 evaluation provider.
- `compression_axes`: 本 workflow 涉及的模型优化轴,例如 `precision`,`sparsity`,`width`,`depth`.
- `hardware`: 运行和产物硬件约束,例如 `device` 和 `backends`.
- `benchmark`: workflow 默认 benchmark 参数.
- `stages`: 唯一的优化和导出路径描述. `quant`,`prune`,`operator`,`export`,`deploy`,`benchmark`,`analyze` 都只能作为 stage 出现.

不要在 `xqt/recipes` 新增顶层 `compression`,`export`,`operator_optimization`,`analysis`,`validation` 或 `config_version`. 这些属于已删除的旧 recipe schema,不能再作为公开 recipe 形态.

当前 `load_optimization_config()` 已把 stage 参数解析为 typed `StageSpec`, workflow 主链也已通过 `run_quant_stage(...)`, `run_prune_stage(...)`, `run_operator_stage(...)`, `run_export_stage(...)`, `run_analyze_stage(...)`, `run_benchmark_stage(...)` 等入口直接消费 typed spec. public `create_context(...)` 已收窄为只接受 `OptimizationConfig`, workflow 映射和带 `stages` 的 workflow 路径, 并直接构造 runtime-only context; 传入旧 recipe mapping 会直接报错. `preflight_optimization_config(...)` 是 workflow preflight 入口, 旧 `preflight_xqt_config(...)` 已删除; `xdl_setup_to_xqt_context(...)` / `xdl_checkpoint_to_xqt_context(...)` 也已收窄为只接受 `OptimizationConfig` 或 workflow 输入.

同时 `XQTContext` 已 workflow 去配置化: `device`, `artifact_dir`, `project_name`, `task_type`, `compression_axes`, `model_target`, `model_params`, `quant_config`, `prune_config`, `analysis_config`, `benchmark_config`, `operator_config`, `output_diff_config`, `export_targets` 已独立成 runtime 字段; `LoadModelPass`, `run_quant_stage(...)`, `run_prune_stage(...)`, `AnalyzePass`, `BenchmarkPass`, `OperatorOptimizationPass`, `ExportPass`, `fake_qdq` 和 workflow CLI 输出已直接走这些字段, 不再用旧 `context.config` 兜底当前 stage runtime 配置. 旧 `QuantPass` / `PrunePass`, `run_xqt_recipe(...)`, `build_pipeline_from_config(...)`, pass order helper, `create_manifest(XQTConfig)`, `xqt_config_to_dict()`, `load_xqt_config()` 和 `XQTConfig` 已删除. 新代码不得恢复旧入口或扩大旧 recipe 兼容路径.

## Stage 约定

- 支持: `benchmark`, `prune`, `quant`, `operator`, `export`, `deploy`, `analyze`.
- 不支持: `finetune`, `distill`, `eval`, `runtime_eval`, `qat_train`, `recovery`.
- 量化 recipe 必须显式写 `backend` 和 `policy`.
- `calibration_inputs` 由调用方传入,recipe 不声明数据来源.
- planned / capability-only 后端必须在 preflight 和文档中标注.
- workflow stage 必须写入 `stage_reports` 和 manifest stage metric,保留 stage name,engine,target module,artifact 和 lineage.
- `xqt.nn.Linear` / `Conv2d` / `LayerNorm` 已是保留 PyTorch module/state_dict 语义的 `torch.nn` 子类 facade, 并显式承载 engine 与 precision runtime intent. `FeedForward` / `RMSNorm` / `Attention` / `TransformerBlock` 也是 semantic facade; Attention 的 torch 路径走 SDPA, tilelang 路径经 `materialize_module` 落到 `_TileLangXqtAttentionWrapper`. TransformerBlock 的 tilelang 会 materialize 内部 `attn` 子模块, 但不是完整 block-level 单 kernel fusion. 文档不得把通用 facade 的 block-level 生产性能写成已完成事实.
- `xqt.model.optimize_hunyuan_ocr_svd_int4_blocks(...)` 是模型专用例外: 它以 `w4a16_int4` 储存 SVD residual, 物化 INT8 MMA residual compute view, 然后对外层 `nn.ModuleList` 中每个逻辑 block 分别执行 `torch.compile` 并用真实模型前向 warmup. 该优化是内存中的 block composition, 不是单个 fused block kernel, 且没有可识别 block 时必须显式失败. `HunyuanOcrTileLangDecodeBlock` 则是独立的实验性 `sm_89` 单 token decode API: 它将量化投影和 TileLang norm, GQA attention, SwiGLU, residual 串为固定 KV 长度的多核 CUDA Graph pipeline, 不会自动替换远程代码模型的 `generate`.
- ONNX target 的已知字段统一写在 `targets[*].onnx`: `input_names`, `output_names`, `dynamo`, `validate`, `runtime_diff`, pre-export fusion/lowering 和 ONNX optimization. target `params` 不再承载这些键. `pre_export_lowering` 中当前的 `fp4_weight_only_to_dense_linear` 会复制 export model, 将 `FP4WeightOnlyLinear` materialize 成等价的 dense dequantized `nn.Linear`, 并在 artifact metadata 记录 lowering. 这让通用 ONNX / TensorRT adapter 可消费该模型, 但 resulting artifact 不是 packed-FP4 runtime.
- 当前 export 会为 ONNX target 额外落一个 `*.xqtpkg/manifest.json` 标准模型包, 并附带 `model/*` 与 `runtime/config.json`. file-based inference 通过 `load_model_package()` / `create_inference_runner()` 只消费该包; 现阶段最小闭环只保证 `ONNX + ONNX Runtime`.
- TensorRT engine-build 的已知字段统一写在 `targets[*].tensorrt`: `onnx_path`, `backend`, `trtexec_path`, `extra_args`, `timeout`, `dry_run`, `performance_thresholds`, `workspace_mib`, `builder_optimization_level`, `timing_cache_path`, `log_level`, `plugin_libraries`, `serialize_plugin_libraries`, `validate_plugin_libraries_loadable` 与 `runtime_benchmark`. loader 明确拒绝 target `params` 中的同名旧键; `params` 只保留其他 export backend 的专有选项. `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口也通过 `tensorrt` 参数构造同一个 typed `StageSpec`.
- OpenVINO target 的已知字段统一写在 `targets[*].openvino`: `onnx_path`, `input_shape`, `dry_run`, `runtime_diff` 与 `device`. loader 明确拒绝 target `params` 中的同名旧键; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口通过 `openvino` 参数构造同一个 typed `StageSpec`. `openvino.onnx_path` 缺失时, export pass 使用同 workflow 的先前 ONNX artifact, 再回退到当前模型转换; `runtime_diff` 只在 materialized IR 与可用 reference output 时执行.
- TorchExport target 的已知字段统一写在 `targets[*].torch_export`: `strict`, `validate` 与 `runtime_diff`. TorchScript target 的已知字段统一写在 `targets[*].torchscript`: `method`, `check_trace` 与 `runtime_diff`, 其中 `method` 只能是 `trace` 或 `script`. loader 明确拒绝 target `params` 中的同名旧键; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口分别通过 `torch_export` 与 `torchscript` 参数构造同一个 typed `StageSpec`.
- ExecuTorch target 的已知字段统一写在 `targets[*].executorch`: `dry_run`. ncnn target 的已知字段统一写在 `targets[*].ncnn`: `source_path`, `converter`, `onnx2ncnn_path`, `pnnx_path`, `bin_path`, `extra_args`, `timeout` 与 `dry_run`; `converter` 只能是 `onnx2ncnn` 或 `pnnx`. pnnx 未指定 source 时优先使用同 workflow 的 TorchScript artifact, 再回退 ONNX. MNN target 的已知字段统一写在 `targets[*].mnn`: `source_path`, `converter_path`, `framework`, `extra_args`, `timeout` 与 `dry_run`. 三者的 Session 单 target 入口分别通过 `executorch`, `ncnn` 和 `mnn` 参数构造同一个 typed `StageSpec`; dry-run preflight 不要求本机存在 optional package 或 converter executable.
- materialized deploy runtime handle 的 typed config 位于 `runtime_handle.onnxruntime` 与 `runtime_handle.tensorrt`. 前者声明 ONNX Runtime providers; 后者声明 device 与 runtime plugin libraries. runtime plugin libraries 不从 TensorRT engine-build target 隐式继承, 必须由 deploy runtime config 显式提供.
- `ExportTargetConfig.dynamic_shapes` 是唯一动态 shape 配置. `dynamo=true` 时传给现代 exporter; `dynamo=false` 时必须写 input-name 到 axis-name 的 mapping, 并转换为 legacy ONNX `dynamic_axes`. TensorRT profile 仍必须与实际 ONNX dynamic axes 相容.
- `PrecisionPolicy` 是 module conversion, `xqt.nn` facade runtime intent 与 GEMM 的唯一静态精度 contract. `xqt.operator_opt.backends.gemm_precision.MatmulPrecisionSpec` 与 `xqt.conversion.MatmulPrecisionSpec` 仅保留为同一类的兼容导入名, 不再维护平行 schema 或双向转换. facade 可额外使用 `auto` 延迟到输入 dtype, 但名称和字段 canonicalization 仍复用该 contract.

### Session stage 内部协议

`XQTOptimizationSession` 初始化时注册 `baseline` stage, 后续 accepted stage 会通过 transform-side provider 注册为 `SessionStage`. provider 内部实现位于 `xqt/workflows/stage_provider.py`, 当前包括:

- `DefaultStageProvider`.
- `ModelQuantizerProvider`.
- `OperatorOptimizerProvider`.
- `ExportProvider`.

`SessionStage.payload` 描述阶段产物, `model_snapshots` 只描述可恢复模型态. `use(name)` / `revert_to(name)` 必须保持模型恢复语义, 不能把 `runtime_plan`, `stage_report`, `export_bundle` 或其他非模型 artifact 当成当前模型.

当前 payload kind:

- `torch_module`: 默认模型态.
- `quantized_model`: quant stage.
- `runtime_plan`: operator stage.
- `stage_report`: benchmark / analyze 这类 observation stage 的报告 payload, 不进入可恢复模型快照.
- `export_bundle`: export stage, 或仅构建 deploy artifact 而未 materialize runtime handle 的 deploy stage.
- `runtime_handle`: materialized deploy stage 的 executable runtime handle. ONNX Runtime producer 创建 `InferenceSession`; TensorRT producer 反序列化同 stage 的非 dry-run engine 并创建 execution context. 二者不替代数值或性能验收.

`benchmark` / `analyze` 在 session graph 中注册为 observation-side `SessionStage`, 不再把 `best_stage` 推进到非模型 stage.

`compare_stages()` / `compare_to_baseline()` 返回 `StageComparison`, 只比较 session 内 stage kind, payload kind, capability, metrics 和 artifacts, 不引入 dataset / dataloader 或 task-level validation.

## Profiling 约定

XQT 允许理解和记录性能分析工具,但 profiling 只作为模型侧 benchmark / operator / export / deploy 的诊断补充.

- `benchmark` 负责稳定 latency / memory / throughput 基线,profiler 负责解释瓶颈,二者在 report 中不能混为一个指标.
- 外部 profiler 输出应作为 artifact 挂到 manifest,例如 NVIDIA `.ncu-rep` / `.nsys-rep`,ROCm trace,VTune result,Ascend `msprof` 输出目录或 TensorBoard profile 目录.
- report 至少记录 engine,device,target artifact,input shape,warmup,repeat,precision,batch size,profiler 名称,profiler 命令关键参数和环境版本.
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
