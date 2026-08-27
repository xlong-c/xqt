# XQT 架构理解与规范化建议

- 记录日期: 2026-08-23
- 适用范围: `xqt/` 当前 v0.x 实现
- 关联审阅: [xqt-architecture-review.md](xqt-architecture-review.md)
- 关联证据: [xqt-architecture-evidence.md](xqt-architecture-evidence.md)

## 1. 结论摘要

XQT 不是单一的线性量化流水线, 而是两个相互衔接的状态机:

1. **优化工作流状态机**: 负责 stage 编排, 模型变换, 验收, 回滚, lineage 和 report.
2. **量化产物与执行状态机**: 负责 storage artifact, runtime handoff, engine resolve, kernel/fallback 和 package.

XQT 的职责边界仍然是模型本身: 压缩, 图变换, 导出适配, 误差分析和 benchmark. XQT 不负责训练, QAT, finetune, dataset, task-level evaluation 或 serving scheduler.

FlashInfer 暂不纳入 XQT 的架构决策. 先稳定 XQT 自己的 workflow, contract 和 runtime handoff.

## 2. 优化工作流状态机

主链可以简化为:

```text
YAML / XQTOptimizationSession
    -> optimize_model 或 session.run_stage
    -> XQTContext + baseline snapshot
    -> restore from_stage
    -> 执行 quant / prune / operator / export
    -> acceptance gate
       accepted: snapshot + SessionStage + payload + report
       rejected: restore best model
    -> OptimizedModelResult / model package
```

核心事实:

- `optimize_model()` 和 `XQTOptimizationSession` 共享 `run_optimization_stage()` 主链, 不应再新增平行 workflow.
- stage runner 负责真正修改 `context.model`.
- `StageProvider` 负责把 stage 结果包装成 payload, 记录 lineage 和 manifest, 不负责代替 quantizer 或 operator pass.
- 验收失败后恢复的是 model snapshot, 不是简单丢弃一个 payload.

关键代码:

- [xqt/workflows/optimization.py](../xqt/workflows/optimization.py)
- [xqt/workflows/session_runner.py](../xqt/workflows/session_runner.py)
- [xqt/workflows/stage.py](../xqt/workflows/stage.py)
- [xqt/workflows/stage_provider.py](../xqt/workflows/stage_provider.py)

### 2.1 Payload 不是当前模型的同义词

| Payload kind | 语义 | 是否可以直接恢复模型 |
|---|---|---|
| `torch_module` | 普通模型状态 | 是 |
| `pruned_model` | 剪枝后的模型状态 | 是 |
| `quantized_model` | 量化后的模型状态 | 是 |
| `runtime_plan` | 算子候选, benchmark 和 fallback 计划 | 否, 依赖 snapshot |
| `stage_report` | 观察和分析结果 | 否 |
| `export_bundle` | 导出文件和 adapter 结果 | 否 |
| `runtime_handle` | 已物化的本地执行句柄 | 否 |

因此阅读或修改 stage 代码时, 必须同时回答两个问题:

1. `context.model` 当前是什么模型状态?
2. `StagePayload` 记录的是什么能力和产物?

不要把 `runtime_plan` 或 `export_bundle` 当成可继续量化的模型.

## 3. 量化产物与 runtime handoff

规范的 int8 路径是:

```text
nn.Linear
    -> quantizer
    -> contracts storage shell
       + ComputeConfig
       + RuntimeQuantContract
    -> runtime.from_storage(...)
    -> engine resolve
    -> native kernel / reference / fallback
    -> execution metadata
    -> ModelPackage
```

对象职责必须保持分开:

| 对象 | 负责什么 | 不负责什么 |
|---|---|---|
| `XQTContext.model` | 当前内存模型 | 不记录完整历史 |
| `ArtifactManifest` | stage, artifact, metric 和 lineage 历史 | 不决定 kernel |
| `ComputeConfig` | capability 请求, precision 和 engine 偏好 | 不定义 serving scheduler |
| `RuntimeQuantContract` | quant spec, storage layout, required kernels 和 shape | 不实现 kernel |
| `RuntimeManifest` | 汇总 runtime 所需信息 | 不替代模型文件 |
| `ModelPackageManifest` | package 文件索引和校验 | 不启动外部 serving |

普通 int8 的实现落点:

- storage/reference 语义: [xqt/contracts/int8_mma.py](../xqt/contracts/int8_mma.py)
- quantizer: [xqt/quant/quantizers/int8_mma.py](../xqt/quant/quantizers/int8_mma.py)
- execution view: [xqt/runtime/modules/int8_mma_linear.py](../xqt/runtime/modules/int8_mma_linear.py)
- compute handoff: [xqt/contracts/compute.py](../xqt/contracts/compute.py)
- runtime quant contract: [xqt/contracts/runtime_quant.py](../xqt/contracts/runtime_quant.py)
- package: [xqt/runtime/package.py](../xqt/runtime/package.py)

### 3.1 capability 和 engine 的区别

- `required_capabilities` 是硬约束, 例如 `int8_mma`.
- `preferred_engines` 是偏好, 例如 `tilelang` 或 `cuda_sm89`.
- runtime 可以因为 shape, device, build 或 dtype 条件不满足而 fallback.
- `required_engine` / `force_engine` 不应成为 package 或 `ComputeConfig` 的主字段.

因此 `w8a8_int8_mma` 是抽象 kernel capability token, 不是某一个固定 backend 的名字.

## 4. 当前实现的边界例外

文档目标是 `quant -> contracts -> runtime`, 但当前代码仍有部分历史性能特化:

- [xqt/quant/quantizers/convrot_int8.py](../xqt/quant/quantizers/convrot_int8.py) 内含 runtime fastpath, cache 和多 backend fallback 判断.
- [xqt/quant/quantizers/convrot_4bit.py](../xqt/quant/quantizers/convrot_4bit.py) 内含部分 mixed-precision execution 逻辑.
- 部分 quantizer 会 lazy import `operator_opt.kernels`, 所以当前代码的实际依赖图比目标层级更宽.

这不是本轮必须立即重构的功能 bug, 但必须在架构检查中明确标记为例外. 当前的正确理解是:

```text
plain int8:     storage shell -> explicit runtime execution view
ConvRot / FP4:  artifact + reference path + historical fastpath 混合存在
```

另外, contracts 和 runtime execution view 使用了相近的 `Int8MmaLinear` 命名. 阅读时要先看 import 路径, 不能只看类名.

本轮已落地的 handoff 约束:

- `xqt.runtime.materialize_convrot_execution_views()` 通过 `_xqt_convrot_storage_kind` marker 扫描并替换 ConvRot storage, 不反向 import `xqt.quant`.
- `ConvRotInt8Linear.from_linear()` 和 `ConvRotMixedPrecisionLinear.from_linear()` 默认只创建 reference storage; native fastpath 必须经过对应的 `ExecutionView.from_storage()`.
- `inplace=False` 会先复制模型再物化 execution view, 原 quantized model 仍保持可保存的 reference artifact.
- execution view 透传 storage 的 policy 属性和方法, 因此 `apply_execution_policy()` 可以继续按模块路径工作.
- `execution_metadata()["artifact_view"]` 明确区分 `contracts_reference` 和 `runtime_execution`.
- `HybridInferenceEngine.from_quantized_model()` 默认复制量化模型并完成上述 handoff; 需要严格 reference-only 行为时显式传 `materialize_execution_views=False`.
- layer-boundary 检查新增 ConvRot 兼容例外守卫: `operator_opt.kernels` 只能在 quantizer 函数体内 lazy import, 不得在模块导入时加载.

这一步只收敛 handoff 入口, 没有声称已经完成 ConvRot/FP4 的物理拆分. native kernel dispatch, backend selection 和 cache 仍在 quantizer 类中, 按后续 P1 逐模块迁移.

## 5. 规范化建议

以下建议按优先级执行, 暂不要求一次性完成全部重构.

### P0: 先固定理解和验证边界

- 保留两条公开入口: `XQTOptimizationSession` 和 YAML `optimize_model()`.
- 保留 stage snapshot 作为唯一回滚事实源.
- 所有新的 quantizer 至少产出 storage artifact, `ComputeConfig` 或等价 handoff 信息, 以及 report.
- 所有新的 execution view 都必须有 `from_storage()` 或同等显式 materialize 入口.
- 继续禁止 XQT 把 serving scheduler, online batching, cache manager 或训练循环塞进 runtime.

### P1: 收敛 quant 与 execution 边界

- quantizer 的默认 forward 保持 reference semantics.
- kernel dispatch, backend selection, workspace/cache 和 fallback reason 逐步迁移到 `runtime/modules` 或 `operator_opt`.
- ConvRot 和 FP4 优先改成 `StorageLinear` + `ExecutionView` 两层, 保留显式 fastpath 入口.
- 为 layer-boundary test 增加目标约束: quant 不应直接依赖 `xqt.runtime` 或可执行 `operator_opt.kernels`, 除非在明确记录的兼容例外中.

### P1: 降低命名和 handoff 成本

- 后续新类优先使用 `*Storage*` 和 `*Execution*` 命名, 避免 contracts/runtime 同名类继续扩大.
- `ComputeConfig`, `RuntimeQuantContract`, `RuntimeManifest` 的字段不要再通过自由字符串隐式迁移.
- `ModelPackage` 只保存可复现的 model/runtime/config/artifact 信息, 不保存 serving scheduler 状态.

### P2: 再处理大范围重构

- 清理 quantizer 内重复的 policy, replace, calibration 和 report 样板.
- 让 engine registry 成为 capability, pattern, materializer 和 fallback 的单一声明点.
- 让 `xqt.gemm` 成为 GEMM dispatch 的唯一事实源, 逐步减少 runtime/quant 对具体 kernel 文件的直接依赖.
- 对 ConvRot/FP4 这类历史 fastpath 做单独的 numeric, benchmark 和 fallback evidence, 不用静态 import 图替代真实性能验证.

## 6. 阅读和排障检查表

遇到一个新的 quant/operator/runtime 问题时, 按以下顺序检查:

1. 当前 stage 是谁, `from_stage` 和 acceptance 条件是什么?
2. `context.model` 是普通模块, storage shell, execution view 还是 export artifact?
3. 当前模块的 storage layout 和 `RuntimeQuantContract` 是什么?
4. `ComputeConfig` 要求的 capability 是什么, preferred engine 只是偏好还是硬要求?
5. 实际 forward 走了 native, reference 还是 fallback? 是否有 execution metadata?
6. 这个改动属于 quant, contracts, runtime, operator_opt 还是 export? 是否跨越了不该跨的边界?
7. 最终是否能写入 report, manifest 或最小 benchmark evidence?

## 7. 下一步执行顺序

1. 不接入 FlashInfer, 先用普通 int8 路径作为 reference architecture.
2. 走通 `quant -> storage shell -> from_storage -> runtime/package` 的最小闭环.
3. 对比审阅 ConvRot 和 FP4 的边界例外, 建立逐模块清单.
4. 补 layer-boundary 和 handoff contract 检查.
5. 只有在闭环和检查稳定后, 再做 quantizer/runtime 的结构性拆分.

本文件是 `check/` 下的阶段性架构理解和整改建议. 长期行为契约仍以源码, `xqt/FRAMEWORK.md` 和 `docs/md/` 对应架构文档为准.
