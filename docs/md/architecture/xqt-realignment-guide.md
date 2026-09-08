# XQT 架构矫正长期指导

本文记录 XQT 现状诊断中 1-8 节的落地状态, 并给后续重构提供长期执行准则. 它是指导文件, 不是完成报告; 未完成项不得在其他文档中写成既有事实.

2026-09-05 后续改进的大纲见 [XQT 改进路线图](xqt-improvement-roadmap.md), 任务状态与详细验收只维护在 [改进目标与验收](xqt-improvement-goals.md). 本文保留历史矫正背景, 涉及历史设计债的最新状态以 [设计债台账](xqt-design-debt.md) 为准; 实施前仍需核对源码.

## 结论

最初诊断报告的 1-8 节没有全部实现. 当前已经落地的是配置单轨化 Phase A/B/C: workflow loader 会把 `stages[*].params` 解析成 typed `StageSpec`; `operator`, `export/deploy`, `analyze`, `benchmark`, `quant`, `prune` 都已通过显式参数消费 typed spec 或其投影; 旧 `XQTConfig`, `load_xqt_config()` 和 `XQTContext.config` 已从代码路径删除. 其余多数仍处于规划或局部实现状态.

当前状态:

| 编号 | 原报告主题 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| 1 | 现状总览 | 已形成诊断, 非实现项 | 作为事实基线保留, 不代表结构已经改完. |
| 2 | 优点 | 保留 | Session, capability, readiness, manifest 等方向继续保留. |
| 3 | 核心问题 | 部分处理 | 配置单轨化和两个 Phase 0 monofile 已完成; 术语收敛, transform-wide contract 复用和硬件验收仍待持续推进. |
| 4 | 目标架构 | 局部推进 | `xqt/contracts/` 是转换和 runtime artifact 事实源, ONNX Runtime / TensorRT deploy handle producer 已落地; 其他 transform contract 仍未闭合. |
| 5 | 分阶段架构调整 | 局部推进 | 配置单轨化 Phase A/B/C 与 Phase 0 拆分已完成; 产品面和 target-hardware validation 仍是后续工作. |
| 6 | 具体优化建议 | 部分处理 | capability/readiness maturity 与统一 StageReport execution envelope 已落地; convert/nn/operator 的完全合流仍待做. |
| 7 | 推荐目标形态 | 局部推进 | `contracts/` 已起步, 目录仍未收敛到 `engines/`, `integrations/` 等目标形态. |
| 8 | 结论 | 方向确认 | 仍应按 "收窄主推路径, 拆大文件, 打通 contract 层, 编排单轨" 推进. |

## 已经落地

- `xqt/workflows/stage_specs.py` 已新增 typed stage spec.
- `load_optimization_config()` 会拒绝旧公开顶层字段, 并为每个 stage 挂载 `stage.spec`.
- `XQTOptimizationSession` 动态创建的 stage 会走同一 `StageSpec` 解析.
- `OperatorStageSpec.benchmark` 已改为 typed benchmark override.
- `ExportStageSpec.validate` 已接入 `OutputDiffConfig`.
- `DeployStageSpec` 已与 `ExportStageSpec` 分离, runtime handle 请求单独建模.
- `DeployRuntimeHandleSpec` 已将 ONNX Runtime providers 与 TensorRT device/runtime plugin libraries 收敛为 typed 子配置. materialized TensorRT runtime handle 不再从 engine-build target 隐式继承 plugin libraries.
- `ExportTargetConfig.onnx` 已收敛 ONNX 的 input/output names, dynamo, validate, runtime diff, pre-export fusion/lowering 和 ONNX graph optimization. 同名 target `params` 旧键由 loader 明确拒绝.
- `ExportTargetConfig.tensorrt` 已收敛 TensorRT 的 ONNX source, builder, plugin, performance threshold 和 runtime benchmark 配置. `XQTOptimizationSession.export()` / `.deploy()` 的单 target 入口也经 `tensorrt` 走同一 StageSpec 解析; 同名 target `params` 旧键由 loader 明确拒绝, engine-build plugin 不会隐式成为 runtime handle plugin.
- `ExportTargetConfig.openvino` 已收敛 OpenVINO 的 ONNX source, input shape, dry-run, runtime diff 和 device 配置. `XQTOptimizationSession.export()` / `.deploy()` 的单 target 入口也经 `openvino` 走同一 StageSpec 解析; 同名 target `params` 旧键由 loader 明确拒绝.
- `ExportTargetConfig.torch_export` 与 `.torchscript` 已收敛 PyTorch-native export 配置. `XQTOptimizationSession.export()` / `.deploy()` 的单 target 入口也经对应参数走同一 StageSpec 解析; 同名 target `params` 旧键由 loader 明确拒绝, TorchScript `method` 限制为 `trace` 或 `script`.
- `ExportTargetConfig.executorch`, `.ncnn` 与 `.mnn` 已收敛移动 / 嵌入式 export 配置. `XQTOptimizationSession.export()` / `.deploy()` 的单 target 入口也经对应参数走同一 StageSpec 解析; 同名 target `params` 旧键由 loader 明确拒绝. ncnn 的 `converter` 显式区分 `onnx2ncnn` 与 `pnnx`, ExportPass 和 preflight 已使用同一选择规则.
- workflow 执行层开始从 `StageSpec` 读取参数; `operator`, `export/deploy`, `analyze`, `benchmark`, `quant`, `prune` 的主链都已不再写入对应的 `context.config.*` stage 配置槽位.
- `run_quant_stage(context, spec)`, `run_prune_stage(context, spec)`, `run_operator_stage(context, spec)`, `run_export_stage(context, spec)`, `run_analyze_stage(context, spec)` 和 `run_benchmark_stage(context, spec)` 已落地, workflow 和 `XQTOptimizationSession` 的 stage 主链已通过这些 typed stage 入口调度.
- `xqt.compression.quant.quantizers.fake_qdq.build_fake_qdq_surrogate(...)` 已开始支持显式 `QuantConfig` / `QuantStageSpec`, 并会优先读取 `context.quant_config`, 不再只能从 `context.config.compression.quant` 偷读当前量化设置.
- `xqt/recipes` 公开 YAML 仍保持 stage workflow 形态.
- 旧 `XQTConfig`, `load_xqt_config()` 和 `XQTContext.config` 已从代码路径删除; `xqt.core.config` 只保留共享 `ConfigInput` 类型.
- `xqt.core.reporting.OptimizationCapability` 已新增共享 `maturity` 字段. quant / prune / operator / export capability producer 与 readiness capability matrix 已统一输出 `executable`, `reference_guarded`, `metadata_only`, `planned` 四档成熟度.
- operator target config / plan / report 已新增 `fallback_policy`, 当前支持 `strict` / `prefer_fallback`. `execute_operator_optimization_plan(...)`, operator stage summary 和 manifest metric 已显式记录 `fallback_reason` / `fallback_policy`, 不再把 engine fallback 藏在深层 metadata 中.
- `materialize.py` 已收敛为跨 engine dispatch, component replacement 和 contract 校验. `execution_support.py` 负责运行辅助, `metadata.py` 负责 engine metadata/preflight, `tilelang_wrappers.py` 负责 TileLang candidate construction.
- `xqt/compression/prune/structured.py` 已删除. structured prune 的 public API, discovery wiring 和 toy model 分别位于 `api.py`, `discovery.py`, `toy_models.py`.
- deploy stage 已可 materialize ONNX Runtime `InferenceSession` 与 TensorRT runtime session. TensorRT producer 要求同 stage 的 non-dry-run engine, 并验证 engine deserialization 和 execution-context creation. 设定 `XQT_RUN_TENSORRT_HARDWARE_TESTS=1` 后, `tests/xqt/test_tensorrt_runtime_hardware.py` 会执行真实 ONNX export, TensorRT build, runtime handle, session execution, output diff 和小型 benchmark; 该最小 FP32 Linear -> ReLU -> Linear 路径已在 NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`) 上通过.
- `StageReport.execution` 已固定记录 backend, engine, device, shape, warmup, iterations, fallback 和 artifact kinds.
- `xqt/recipes/smoke/weight_only_tilelang_onnx_tensorrt_golden.yaml` 提供主推 stage 顺序的 target-hardware workflow. 它使用 `dynamo: false` 加 legacy dynamic axes 的 ONNX export 路径, 调用时仍必须显式提供 `example_inputs`; `pre_export_lowering: fp4_weight_only_to_dense_linear` 会只在 export copy 上 materialize dense dequantized 权重, 因而 TensorRT artifact 不是 packed-FP4 runtime. 设定 `XQT_RUN_GOLDEN_HARDWARE_TESTS=1` 且具备 CUDA + ONNX + TileLang 时, `tests/xqt/test_golden_path_recipe.py` 会以临时产物目录运行 dry-run workflow; 同时具备 TensorRT 时还会构建 non-dry-run engine, materialize runtime handle, 做 `1e-3` 数值对比并记录 `[8,64]`, `warmup=2`, `iterations=5` 的 runtime benchmark. 两项均在 NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`) 上通过. 测试继续断言 fallback 不被标记为 applied; 这不是 TileLang kernel 性能验收.
- TileLang CUDA kernel entry 会在已知 packed-tensor ABI 不兼容的 runtime 上显式失败, 而 wrapper 仅在配置允许时记录 eager fallback. TileLang `0.1.11` 仍在不兼容名单; 本机已升级到 `0.1.12` 后, linear / conv / norm / dequant / attention CUDA correctness tests 与 golden dry-run / TensorRT materialize 已在 `sm_89` 通过.

这些表示配置主路径与 Phase 0 拆分已经收敛, 不表示所有 transform 已共用同一 contract, 或所有 engine 已完成目标硬件验收.

## 尚未实现

### 配置与编排

配置单轨化主路径已完成, 但仍有后续治理项:

- `QuantPass` / `PrunePass` 包装层已删除; `run_quant_stage(...)` / `run_prune_stage(...)` 已成为唯一公开主路径.
- `StageSpec -> runtime config` 的投影逻辑已从 workflow 调度层下沉到 `xqt.pipeline` stage helper; 后续需要保持这个单一路径, 不再新增并行 recipe 解析入口.
- `ExportTargetConfig.onnx` 是 ONNX target 的 typed 参数面, 不再让已知 ONNX 配置回流到 target `params`.
- `ExportTargetConfig.openvino` 是 OpenVINO target 的 typed 参数面, 不再让已知 OpenVINO 配置回流到 target `params`.
- `ExportTargetConfig.torch_export` 与 `.torchscript` 是 PyTorch-native target 的 typed 参数面, 不再让已知配置回流到 target `params`.
- `ExportTargetConfig.executorch`, `.ncnn` 与 `.mnn` 是移动 / 嵌入式 target 的 typed 参数面, 不再让已知配置回流到 target `params`.
- `XQTContext` 已承载 `device`, `artifact_dir`, `project_name`, `task_type`, `compression_axes`, `model_target`, `model_params`, `quant_config`, `prune_config`, `analysis_config`, `benchmark_config`, `operator_config`, `output_diff_config`, `export_targets` 等 runtime 字段; `XQTContext.config` 已删除.
- `run_workflow.py`, `export_pass`, `passes.py` 的报告落盘路径, `xqt/kernels/wrappers/execute.py`, `xqt/compression/quant/backends/onnx_qdq.py` 等路径已优先读取 `context.project_name` / `context.device` / `context.artifact_dir`; export targets 已通过 `context.export_targets` 传递.
- `run_xqt_recipe(...)`, `build_pipeline_from_config(...)`, pass order helper, `preflight_xqt_config()`, `create_manifest(XQTConfig)`, `xqt_config_to_dict()`, `load_xqt_config()` 和 `XQTConfig` 已删除. 旧顶层 recipe mapping 由 `load_optimization_config()` 直接拒绝并提示迁移.

目标:

```text
OptimizationConfig
  -> OptimizationStageConfig.spec
  -> run_*_stage(context, spec)
  -> metrics / artifacts / manifest
```

不要再新增任何依赖旧 `context.config.*`, 顶层 `operator_optimization`, 顶层 `export` 或顶层 `analysis` 的生产调用.

### God module

- `xqt/kernels/wrappers/executor.py` 已删除. plan 构建位于 `xqt/kernels/wrappers/plan.py`; cross-engine dispatch / contract validation 位于 `materialize.py`; execution support 位于 `execution_support.py`; engine metadata / preflight 位于 `metadata.py`; CuTile / CuTe DSL reference-guarded materialization 位于 `reference_wrappers.py`; Triton RMSNorm materialization / metadata 位于 `triton_wrappers.py`; TileLang candidate construction 和 wrapper family 位于 `tilelang_wrappers.py`; execution / benchmark / acceptance 位于 `execute.py`; CUDA Graph runtime 位于 `runtime.py`; summary 位于 `reporting.py`.
- structured prune 已不再保留 `structured.py` 汇总模块. dependency graph 位于 `graph.py`, public entry 位于 `api.py`, discovery adapter binding 位于 `discovery.py`, smoke model 位于 `toy_models.py`, concrete candidate collector 位于 `candidates_*.py`, planner 位于 `plan.py`, rewrite 位于 `rewrite.py`, report 位于 `report.py`, N:M / block-sparse 权重稀疏位于 `sparsity.py`.

拆分必须先保持行为不变, 再改接口:

```text
xqt/kernels/wrappers/
  plan.py
  materialize.py
  execution_support.py
  metadata.py
  reference_wrappers.py
  tilelang_wrappers.py
  triton_wrappers.py
  execute.py
  reporting.py
  bench/

xqt/compression/prune/
  graph.py
  candidates.py
  candidates_attention.py
  candidates_conv.py
  candidates_residual.py
  candidates_mbconv.py
  candidates_mlp.py
  candidates_moe.py
  candidates_container.py
  candidates_vit.py
  api.py
  discovery.py
  toy_models.py
  plan.py
  rewrite.py
  report.py
  sparsity.py
```

### Contract 层

`xqt/contracts/` 仍是 stage artifact 与量化存储的事实源. `PrecisionPolicy`, `FeedForwardPrecisionPolicy`, `TensorStorageSpec`, `ModuleContract` 和 `FusionIntent` 已下沉到 `xqt.kernels.precision` (convert / GEMM / facade 共用, `ops/_impl` 不经过 `kernels.nn`). `QuantizedModel` 是模型侧量化算法的通用语义结果; `QuantizedModelPayload`, `PrunedModelPayload`, `RuntimeArtifactPayload`, `RuntimePlanPayload`, `RuntimeHandlePayload` 和 `ExportBundlePayload` 是带 workflow provenance 的 stage artifact, stage 仅保留重导出. `xqt.convert(...)` 通过 lowering schema 构造 contract, 再由 `xqt.kernels.wrappers.materialize_module(module, contract, target)` 统一校验并 materialize operator candidate. `conversion.py` 暂时重导出原名称, 保持其 Provisional API 语义.

contract 层仍未闭合. `OptimizationCapability.maturity` 已先作为共享 capability schema 落地, 后续 contract 目标至少包括:

- `PrecisionPolicy`, `ModuleContract`, `FusionIntent` 已落地. `PrecisionPolicy` 已由 `convert` / `xqt.nn` runtime intent 和 GEMM precision dispatcher 共用, `MatmulPrecisionSpec` 仅为 compatibility alias; facade 的 `auto` 是运行时延迟决策, 不扩张静态 schema. 但其余 transform 子系统尚未全部接入 `ModuleContract` / `FusionIntent`.
- `QuantizedModel` 已收敛模型侧 FP4, MXFP, AWQ/GPTQ, INT8 MMA, W4 storage INT8 MMA, SVD 和 TorchAO 的实际结果; `QuantizedModelPayload` 仅追加 quant stage provenance. analysis-only FakeQDQ surrogate 与 ONNX QDQ graph artifact 不属于模型量化结果, 不使用该 contract.
- `PrunedModelPayload` 已收敛 unstructured,structured,N:M 与 block-sparse prune stage 的模型侧语义, 保留实际模型,method,目标/实际 sparsity,execution state,完整 report,artifacts 和 capability, 不再让 prune stage 回退为无语义的裸模型 payload.
- `RuntimePlan`, `ExportBundle`, `RuntimeHandle` 已落地. deploy stage 可生产 ONNX Runtime 或 TensorRT handle; TensorRT 的 session creation 本身不是数值或性能验收, 但 opt-in hardware smoke 已在 materialized TensorRT handle 上执行 output diff 和 benchmark. 这仍不等同于完整 golden workflow 验收.
- 其他 contract 抽象仍待统一进入 `xqt/contracts/`

### `xqt.nn` 与 `convert`

当前状态必须按事实写:

- `xqt.nn.FeedForward` 和 `RMSNorm` 是真实 XQT facade.
- `xqt.nn.Linear`, `Conv2d`, `LayerNorm` 现为 `torch.nn` 子类 facade. 它们保持 PyTorch 前向和 module/state_dict 语义, 并显式承载 `engine` 与 precision runtime intent; 真正的 engine lowering 仍由 `xqt.convert(...)` 和 `materialize_module(...)` 负责.
- `xqt.convert(...)` 已经有 Linear / Conv2d / LayerNorm / FeedForward 路径. TileLang lowerings 和 Triton FeedForward lowering 会通过 `materialize_module(...)` 复用 contract 校验与 candidate materialization; torch baseline 与尚无对应 operator executor 的 runtime facade 仍不强行套入 materializer.
- `FeedForward` 的 Triton linear / gate fastpath 出错时会保留 reference 计算, 并在 `runtime_config().fallback` / `fallback_count` 写入 engine, stage 和原因, 不再静默吞掉异常.
- `FeedForward.fusion_intent()` 已直接产出 `xqt.kernels.precision.FusionIntent`; `conversion` 构建 FeedForward `ModuleContract` 时消费该 intent, 不再从 runtime report dict 反构 fusion patterns.
- `xqt.nn.Attention` 与 `TransformerBlock` 已落地为 semantic facade. torch 路径用 SDPA / 组合前向; tilelang 路径可配置 runtime intent. `xqt.convert(...)` 已支持 `attention` / `transformer_block` operator kind, 并给所有 convert 路径挂载 `_xqt_module_contract`.
- Stage payload contracts 提供 `from_stage_metrics(...)` 工厂; `stage_provider` 通过这些工厂构造 `QuantizedModelPayload` / `PrunedModelPayload` / `RuntimePlanPayload` / `ExportBundlePayload` / `RuntimeHandlePayload`, 不再在 provider 内手写平行字段映射.
- operator report 在 candidate 带有 `_xqt_module_contract` 时会把 contract 写入 target metadata / `module_contract` 字段.
- `RuntimeHandlePayload` 有 ONNX Runtime 和 TensorRT deploy producer: 同一 deploy stage 的 ONNX artifact 可 materialize 为 `InferenceSession`; 非 dry-run TensorRT engine 可反序列化并创建 execution context. 这两者都不替代显式数值或性能验收.
- Stage report 序列化已能安全处理 live TensorRT session: `TensorRTRuntimeSession.to_dict()` 只保留 metadata, `json_safe_value` 对含 module 的 dataclass 不再因 `asdict` 失败中断 deploy stage report.

`xqt.nn.Attention` 的 tilelang materialize 已通过 `convert(engine='tilelang')` -> `materialize_module` -> `_TileLangXqtAttentionWrapper` 落地, 并与 `nn.MultiheadAttention` operator 路径并存. `TransformerBlock` 的 tilelang 路径会 materialize 内部 `attn` 子模块为 `_TileLangXqtAttentionWrapper` (`converted=True`), 但完整 block-level 单 kernel fusion 仍是后续工作; 不得把 block-level 生产性能写成既有事实. 多 shape 正确性覆盖见 `tests/xqt/test_nn_attention.py`.

### 能力收敛

主推路径应收敛为:

```text
weight-only low-bit
  -> TileLang / Triton operator
  -> benchmark / analyze
  -> ONNX / TensorRT export
```

其他 engine/backend 已开始按共享 capability maturity 宣传, 当前统一词表为:

- `executable`
- `reference_guarded`
- `metadata_only`
- `planned`

没有 hardware validation 的路径不能写成生产可用.

## 术语准则

新文档和新 API 必须区分:

| 词 | 含义 | 例子 |
| --- | --- | --- |
| `backend` | 外部 quant/export/runtime 选择 | `torchao`, `onnxruntime_qdq`, `tensorrt`, `openvino` |
| `engine` | XQT 内部 kernel/lowering 实现 | `torch_compile`, `triton`, `tilelang`, `cutile`, `cute_dsl` |
| `method` | 算法语义 | `awq`, `gptq`, `svd` |
| `strategy` | 量化或优化策略 | `fp4_weight_only`, `static_qdq_int8` |
| `runtime` | 实际执行环境标签 | `pytorch`, `onnxruntime`, `tensorrt` |

不要把 quant 的 `backend` 硬改名为 `engine`. 也不要把 operator 的 `engine` 写成外部 inference backend.

## 文档写作规则

- 长期矫正原则保留在本文, 新问题记入 `xqt-design-debt.md`; 本批改进的大纲与验收分别维护在 `xqt-improvement-roadmap.md` 和 `xqt-improvement-goals.md`, 不复制任务状态.
- 已实现事实写入 `docs/md/architecture/xqt.md`, `docs/md/explanation/xqt-concepts.md`, `docs/md/usage/xqt-workflows.md`, `xqt/FRAMEWORK.md`.
- 规划和现状必须分开. 未完成项必须使用 "目标", "计划", "尚未" 等措辞.
- 修改能力成熟度时, 同步更新 readiness, capability, tests 和文档.
- 修改配置 schema 时, 同步更新 `test_framework_contract.py`, recipe smoke 和 `xqt/AGENTS.md`.

## 下一步顺序

当前执行顺序统一见 [改进路线图的里程碑](xqt-improvement-roadmap.md#3-六大里程碑与阶段准入). 先保证 Session 状态, 校准, artifact 与 acceptance 可信, 再接通模型结构与可执行契约, 最后验收真实模型及其部署产物.

历史硬件 smoke 与局部 kernel 证据仍有价值, 但不能替代新计划要求的真实 checkpoint, block / 整模指标和新进程重载. ONNX / TensorRT dense lowering 与原生低比特执行继续分开验收. 本节不再复制设计债的 open/done 状态, 避免把台账已结案条目重复立项.
