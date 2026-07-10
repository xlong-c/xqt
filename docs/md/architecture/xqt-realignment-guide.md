# XQT 架构矫正长期指导

本文记录 XQT 现状诊断中 1-8 节的落地状态, 并给后续重构提供长期执行准则. 它是指导文件, 不是完成报告; 未完成项不得在其他文档中写成既有事实.

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
- workflow 执行层开始从 `StageSpec` 读取参数; `operator`, `export/deploy`, `analyze`, `benchmark`, `quant`, `prune` 的主链都已不再写入对应的 `context.config.*` stage 配置槽位.
- `run_quant_stage(context, spec)`, `run_prune_stage(context, spec)`, `run_operator_stage(context, spec)`, `run_export_stage(context, spec)`, `run_analyze_stage(context, spec)` 和 `run_benchmark_stage(context, spec)` 已落地, workflow 和 `XQTOptimizationSession` 的 stage 主链已通过这些 typed stage 入口调度.
- `xqt.quant.quantizers.fake_qdq.build_fake_qdq_surrogate(...)` 已开始支持显式 `QuantConfig` / `QuantStageSpec`, 并会优先读取 `context.quant_config`, 不再只能从 `context.config.compression.quant` 偷读当前量化设置.
- `xqt/recipes` 公开 YAML 仍保持 stage workflow 形态.
- 旧 `XQTConfig`, `load_xqt_config()` 和 `XQTContext.config` 已从代码路径删除; `xqt.core.config` 只保留共享 `ConfigInput` 类型.
- `xqt.core.reporting.OptimizationCapability` 已新增共享 `maturity` 字段. quant / prune / operator / export capability producer 与 readiness capability matrix 已统一输出 `executable`, `reference_guarded`, `metadata_only`, `planned` 四档成熟度.
- operator target config / plan / report 已新增 `fallback_policy`, 当前支持 `strict` / `prefer_fallback`. `execute_operator_optimization_plan(...)`, operator stage summary 和 manifest metric 已显式记录 `fallback_reason` / `fallback_policy`, 不再把 engine fallback 藏在深层 metadata 中.
- `materialize.py` 已收敛为跨 engine dispatch, component replacement 和 contract 校验. `execution_support.py` 负责运行辅助, `metadata.py` 负责 engine metadata/preflight, `tilelang_wrappers.py` 负责 TileLang candidate construction.
- `xqt/prune/structured.py` 已删除. structured prune 的 public API, discovery wiring 和 toy model 分别位于 `api.py`, `discovery.py`, `toy_models.py`.
- deploy stage 已可 materialize ONNX Runtime `InferenceSession` 与 TensorRT runtime session. TensorRT producer 要求同 stage 的非 dry-run engine, 并验证 engine deserialization 和 execution-context creation.
- `StageReport.execution` 已固定记录 backend, engine, device, shape, warmup, iterations, fallback 和 artifact kinds.
- `xqt/recipes/smoke/weight_only_tilelang_onnx_tensorrt_golden.yaml` 提供主推 stage 顺序的 target-hardware workflow. 其中 TensorRT target 为 dry-run, 不代表本机完成硬件验收.
- TileLang CUDA kernel entry 会在已知 packed-tensor ABI 不兼容的 runtime 上显式失败, 而 wrapper 仅在配置允许时记录 eager fallback. 本机 TileLang `0.1.11` 属于该限制, 因此本机不提供 TileLang kernel correctness 或性能验收.

这些表示配置主路径与 Phase 0 拆分已经收敛, 不表示所有 transform 已共用同一 contract, 或所有 engine 已完成目标硬件验收.

## 尚未实现

### 配置与编排

配置单轨化主路径已完成, 但仍有后续治理项:

- `QuantPass` / `PrunePass` 包装层已删除; `run_quant_stage(...)` / `run_prune_stage(...)` 已成为唯一公开主路径.
- `StageSpec -> runtime config` 的投影逻辑已从 workflow 调度层下沉到 `xqt.pipeline` stage helper; 后续需要保持这个单一路径, 不再新增并行 recipe 解析入口.
- `XQTContext` 已承载 `device`, `artifact_dir`, `project_name`, `task_type`, `compression_axes`, `model_target`, `model_params`, `quant_config`, `prune_config`, `analysis_config`, `benchmark_config`, `operator_config`, `output_diff_config`, `export_targets` 等 runtime 字段; `XQTContext.config` 已删除.
- `run_workflow.py`, `export_pass`, `passes.py` 的报告落盘路径, `operator_opt/execute.py`, `quant/backends/onnx_qdq.py` 等路径已优先读取 `context.project_name` / `context.device` / `context.artifact_dir`; export targets 已通过 `context.export_targets` 传递.
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

- `xqt/operator_opt/executor.py` 已删除. plan 构建位于 `plan.py`; cross-engine dispatch / contract validation 位于 `materialize.py`; execution support 位于 `execution_support.py`; engine metadata / preflight 位于 `metadata.py`; CuTile / CuTe DSL reference-guarded materialization 位于 `reference_wrappers.py`; Triton RMSNorm materialization / metadata 位于 `triton_wrappers.py`; TileLang candidate construction 和 wrapper family 位于 `tilelang_wrappers.py`; execution / benchmark / acceptance 位于 `execute.py`; CUDA Graph runtime 位于 `runtime.py`; summary 位于 `reporting.py`.
- structured prune 已不再保留 `structured.py` 汇总模块. dependency graph 位于 `graph.py`, public entry 位于 `api.py`, discovery adapter binding 位于 `discovery.py`, smoke model 位于 `toy_models.py`, concrete candidate collector 位于 `candidates_*.py`, planner 位于 `plan.py`, rewrite 位于 `rewrite.py`, report 位于 `report.py`, N:M / block-sparse 权重稀疏位于 `sparsity.py`.

拆分必须先保持行为不变, 再改接口:

```text
xqt/operator_opt/
  plan.py
  materialize.py
  execution_support.py
  metadata.py
  reference_wrappers.py
  tilelang_wrappers.py
  triton_wrappers.py
  execute.py
  reporting.py

xqt/prune/
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

`xqt/contracts/` 已作为转换和 stage artifact 的事实源. `PrecisionPolicy`, `FeedForwardPrecisionPolicy`, `TensorStorageSpec`, `ModuleContract` 和 `FusionIntent` 已从 `conversion.py` 迁入. `QuantizedModelPayload`, `RuntimeArtifactPayload`, `RuntimePlanPayload`, `RuntimeHandlePayload` 和 `ExportBundlePayload` 已从 workflow stage 迁入, stage 仅保留重导出. `xqt.convert(...)` 通过该公开 schema 构造 contract, 再由 `xqt.operator_opt.materialize_module(module, contract, target)` 统一校验并 materialize operator candidate. `conversion.py` 暂时重导出原名称, 保持其 Provisional API 语义.

contract 层仍未闭合. `OptimizationCapability.maturity` 已先作为共享 capability schema 落地, 后续 contract 目标至少包括:

- `PrecisionPolicy`, `ModuleContract`, `FusionIntent` 已落地, 但尚未被所有 transform 子系统共用.
- `QuantizedModelPayload` 已落地; 量化算法侧的通用 `QuantizedModel` semantic contract 仍待统一.
- `RuntimePlan`, `ExportBundle`, `RuntimeHandle` 已落地. deploy stage 可生产 ONNX Runtime 或 TensorRT handle; TensorRT 的 session creation 不是数值或性能验收.
- 其他 contract 抽象仍待统一进入 `xqt/contracts/`

### `xqt.nn` 与 `convert`

当前状态必须按事实写:

- `xqt.nn.FeedForward` 和 `RMSNorm` 是真实 XQT facade.
- `xqt.nn.Linear`, `Conv2d`, `LayerNorm` 现为 `torch.nn` 子类 facade. 它们保持 PyTorch 前向和 module/state_dict 语义, 并显式承载 `engine` 与 precision runtime intent; 真正的 engine lowering 仍由 `xqt.convert(...)` 和 `materialize_module(...)` 负责.
- `xqt.convert(...)` 已经有 Linear / Conv2d / LayerNorm / FeedForward 路径. TileLang lowerings 和 Triton FeedForward lowering 会通过 `materialize_module(...)` 复用 contract 校验与 candidate materialization; torch baseline 与尚无对应 operator executor 的 runtime facade 仍不强行套入 materializer.
- `FeedForward` 的 Triton linear / gate fastpath 出错时会保留 reference 计算, 并在 `runtime_config().fallback` / `fallback_count` 写入 engine, stage 和原因, 不再静默吞掉异常.
- `RuntimeHandlePayload` 有 ONNX Runtime 和 TensorRT deploy producer: 同一 deploy stage 的 ONNX artifact 可 materialize 为 `InferenceSession`; 非 dry-run TensorRT engine 可反序列化并创建 execution context. 这两者都不替代显式数值或性能验收.

因此文档不得把 `Linear`, `Conv2d`, `LayerNorm`, `Attention`, `TransformerBlock` 写成已经完整实现的 XQT semantic modules.

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

- 架构目标写入 `docs/md/architecture/xqt-realignment-guide.md` 或 `xqt-config-unification-todo.md`.
- 已实现事实写入 `docs/md/architecture/xqt.md`, `docs/md/explanation/xqt-concepts.md`, `docs/md/usage/xqt-workflows.md`, `xqt/FRAMEWORK.md`.
- 规划和现状必须分开. 未完成项必须使用 "目标", "计划", "尚未" 等措辞.
- 修改能力成熟度时, 同步更新 readiness, capability, tests 和文档.
- 修改配置 schema 时, 同步更新 `test_framework_contract.py`, recipe smoke 和 `xqt/AGENTS.md`.

## 下一步顺序

1. 在目标 CUDA GPU 的 runtime-compatible TileLang 环境验证 TileLang / Triton 的 correctness 和 benchmark, 依据结果调整 maturity. 已知不兼容 runtime 只能验证 guard 与记录的 fallback, 不能替代 kernel 验收.
2. 在目标 TensorRT 环境运行 non-dry-run golden workflow, 再验证 runtime handle 的数值和性能.
3. Contract 层: 让更多 transform 子系统与 `xqt/contracts/` 合流, 不扩张并行 schema.
4. 持续校正 backend / engine capability 的 maturity 和宣传面.
