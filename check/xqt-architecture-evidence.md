# XQT 架构审阅: 原始证据附录

本文件是 [xqt-architecture-review.md](xqt-architecture-review.md) 的证据附录, 收录依赖图, 重复文件, 文件职责等原始数据表. 数据采集方法: AST 解析全部 461 个 `.py` (排除 `__pycache__` / `.ipynb_checkpoints`), 区分真顶层 import / TYPE_CHECKING import / 函数体内 lazy import, 相对 import 已解析为绝对路径; 重复检测用 md5 全树对拍 + 同名文件 diff; 调用方分析覆盖 xqt/tests/tools/examples/infer/scripts. 关键结论经人工复核与 `importlib` 实测.

## A. 子包级 import 邻接表

格式: src → dst (top = 真顶层 import 次数, tc = TYPE_CHECKING, lazy = 函数体内).

| src → dst | top | tc | lazy | | src → dst | top | tc | lazy |
|---|---|---|---|---|---|---|---|---|
| core → contracts | 1 | 0 | 3 | | runtime → contracts | 41 | 0 | 6 |
| contracts → core | 7 | 0 | 0 | | runtime → core | 20 | 0 | 0 |
| contracts → quant | 0 | 1 | 1 | | runtime → quant | 7 | 0 | 1 |
| contracts → runtime | 0 | 0 | 1 | | runtime → operator_opt | 2 | 0 | 47 |
| quant → core | 40 | 1 | 1 | | runtime → gemm | 2 | 0 | 1 |
| quant → contracts | 19 | 0 | 2 | | runtime → export | 2 | 0 | 0 |
| quant → runtime | 7 | 0 | 1 | | runtime → benchmark | 1 | 0 | 0 |
| quant → operator_opt | 1 | 0 | 19 | | operator_opt → core | 61 | 0 | 0 |
| quant → export | 4 | 0 | 1 | | operator_opt → gemm | 10 | 0 | 0 |
| quant → workflows | 2 | 0 | 0 | | operator_opt → runtime | 5 | 0 | 0 |
| quant → analysis | 2 | 0 | 0 | | operator_opt → export | 2 | 0 | 1 |
| quant → model | 1 | 0 | 0 | | operator_opt → contracts | 2 | 1 | 0 |
| workflows → core | 13 | 0 | 0 | | gemm → core | 42 | 0 | 0 |
| workflows → contracts | 6 | 0 | 0 | | gemm → operator_opt | 1 | 0 | 7 |
| workflows → pipeline | 3 | 0 | 0 | | pipeline → core | 61 | 0 | 1 |
| workflows → quant | 1 | 0 | 1 | | pipeline → workflows | 9 | 1 | 0 |
| workflows → prune | 1 | 0 | 1 | | pipeline → export | 7 | 0 | 0 |
| workflows → readiness | 1 | 0 | 0 | | pipeline → operator_opt | 5 | 0 | 2 |
| workflows → auto | 0 | 0 | 2 | | pipeline → quant | 4 | 0 | 1 |
| export → core | 20 | 0 | 0 | | pipeline → prune/analysis | 3+2 | 0 | 0 |
| export → analysis | 3 | 0 | 0 | | pipeline → runtime/benchmark | 1+1 | 0 | 0 |
| export → quant | 2 | 0 | 0 | | nn → operator_opt | 2 | 0 | 1 |
| export → contracts | 1 | 0 | 0 | | nn → gemm/contracts/core | 1+2+1 | 0 | 0 |
| model → operator_opt | 14 | 0 | 2 | | conversion_impl → nn/operator_opt | 4+5 | 0 | 0 |
| model → core | 12 | 0 | 0 | | conversion_impl → conversion | 0 | 0 | 8 |
| model → quant | 9 | 0 | 0 | | auto → quant | 4 | 0 | 0 |
| model → benchmark | 4 | 0 | 0 | | analysis → model/quant | 0 | 0 | 3 |
| model → runtime | 3 | 0 | 0 | | benchmark → (无) | 0 | 0 | 0 |
| model → analysis | 2 | 0 | 0 | | prune → analysis/export/core | 1+1+1 | 0 | 0 |

顶层模块: `readiness → quant/operator_opt/export/prune/core` (顶层, 8 模块); `conversion → conversion_impl/contracts/core`; `xdl_adapter → workflows/pipeline/core`; `run_workflow → workflows`.

扇入/扇出 (仅真顶层, 去重子包数): pipeline fanout=9 (最高); quant 8/8; operator_opt 7/8; runtime 7/4; model 7/1; workflows 6/6; core fanin=16; contracts fanin=9; analysis 与 benchmark fanout=0 (纯叶子).

## B. 包级循环依赖清单 (真顶层边)

**2-cycle (8 组)**:

1. core ↔ contracts: `core/schema.py:10` → contracts.inference; contracts 7 处 → core.errors (`contracts/module.py:8`, `inference.py:8`, `runtime_quant.py:15` 等).
2. quant ↔ runtime: quant→runtime `quant/bridges/nvfp4.py:5` (整文件 re-export 壳), `quant/layout_apply_report.py:21`, `quantizers/fp4_dynamic.py:21`, `int8_mma.py:15,19`, `w4_storage_int8_mma.py:31,32`; runtime→quant `runtime/serving_config.py:18`, `runtime/bridges/external_materialize.py:12`, `external_weight_only.py:27,28`, `hf_int4_layout.py:11,12`, `hf_int4_pack.py:11`.
3. quant ↔ export: `quant/quantizers/__init__.py:129`, `quant/execution/executor.py:18`, `quant/backends/onnx_qdq.py:18`, `quant/calibration/summary.py:10`; 反向 `export/hf_quant.py:23,24`.
4. quant ↔ workflows: `quant/plan.py:10`, `quantizers/fake_qdq.py:15`; 反向 `workflows/stage_provider.py:16`.
5. quant ↔ model: `quant/sensitivity.py:11` → model.hooks; 反向 `model/flux2_klein/load.py:14,18`, `model/unlimited_ocr.py:20,21,25` 等 9 处.
6. operator_opt ↔ gemm: operator_opt 10 处顶层 → gemm (`kernels/{cutile,cutlass,tilelang,triton}/gemm.py`, `backends/gemm_precision.py:14`); 反向顶层仅死文件 `gemm/backends/tilelang_marlin.py:14`, 真实生效的 7 处全为 lazy.
7. operator_opt ↔ runtime: operator_opt 5 处顶层 → runtime.bridges (`reference_wrappers.py:13`, `triton_dequant_wrappers.py:14`, `kernels/triton/gemm.py:15`, `wrappers/dequant_gemm.py:14`, `backends/gemm_selector.py:20` → engine_resolve); runtime 2 处顶层 (`modules/kv_attention.py:40`, `svd_flux_attention.py:10`) + 47 处 lazy.
8. pipeline ↔ workflows: workflows→pipeline `workflows/optimization.py:27`, `session_runner.py:26,27`; pipeline→workflows `pipeline/preflight.py:11`, `runner.py:21`, `passes.py:68`, `export_pass.py:19`, `pass_helpers/context.py:21`, `prune_stage.py:30`, `quant_stage.py:22`, `preflight_checks/deploy.py:6`.

**3-cycle (按真实方向)**: `readiness → quant → workflows → readiness`; `quant → operator_opt → export → quant`; `quant → runtime → export → quant`; `gemm → operator_opt → runtime → gemm` (`runtime/modules/awq_w4a16_linear.py:12,13`); `quant → model → runtime → quant`; `quant → model → workflows → quant` (`model/hunyuan_ocr.py:15,16`); `pipeline → quant → workflows → pipeline`.

**模块级互 import 对**: `workflows/optimization.py ↔ pipeline/{passes,preflight}.py`; `workflows/session_runner.py ↔ pipeline/runner.py`; `quant/bridges/nvfp4.py ↔ runtime/bridges/nvfp4.py` (单向 re-export 壳).

## C. import 拉起规模实测

- `import xqt`: 拉起 372 个 xqt 子模块 + torch + 85 个 xdl 子模块, 冷 import 6.44 秒 (本机). 链路: `xqt/__init__.py` → workflows → optimization.py → pipeline.passes (拉起 analysis/benchmark/operator_opt/prune/quant) + export_pass (拉起 xqt.export 全家 + runtime.package) + readiness.py + xdl_adapter.py.
- 聚合导出规模: `quant/__init__.py` 299 行 / 28 条聚合 import; `export/__init__.py` 145 行全聚合; `runtime/__init__.py` 257 行; `contracts/__init__.py` 184 行 / 18 条; `gemm/__init__.py` 503 行 + `backends/__init__.py` 290 行 (16 个 backend 模块); `model/__init__.py` 281 行. 良性面: core/workflows/pipeline/nn/benchmark/conversion_impl 的 `__init__` 很小 (4-28 行).

## D. lazy import 热点 (全包 275 处, 分布 104 文件)

| 文件 | 数量 | 主要目标 | 性质 |
|---|---|---|---|
| `runtime/modules/svd_w4a4_legacy.py` | 13 | cute.svdq_w4a4_sm89(7), tilelang.svd_fused | 可选 kernel 按需加载 |
| `runtime/modules/int8_mma_linear.py` | 9 | cute.int8mma_binding(7), tilelang.int8_mma, triton.gemm | 同上 |
| `quant/quantizers/convrot_int8.py` | 8 | cute.convrot_w8a8_sm89, int8mma_binding, tilelang/triton | 同上 |
| `quant/quantizers/convrot_4bit.py` | 7 | cute.convrot_w4a4_rowwise_sm89, svdq_w4a4_sm89 | 同上 |
| `runtime/modules/kv_attention.py` | 6 | tilelang.kv_int8_attention | 同上 |
| `runtime/modules/svd_flux_attention.py` | 5 | cute.svdq_w4a4_sm89 | 同上 |
| `runtime/modules/svd_w8a8_legacy.py` | 5 | cute.svdq_w8a8_sm89 | 同上 |
| `conversion_impl/{attention,block,converter,feedforward,linear}.py` | 8 (合计) | xqt.conversion (顶层模块) | **循环掩盖**: 因 `conversion.py:24,25` 顶层 import conversion_impl |
| `core/reporting.py:634`, `core/schema.py:270,295`, `contracts/runtime_quant.py:69`, `contracts/runtime_manifest.py:224` | 5 | contracts/quant/runtime | **循环掩盖**: contracts docstring 自证 |

## E. 重复 / 分叉文件清单

### E.1 xqt/gemm/ 三份平行基础设施

- `xqt/gemm/common/` (12 模块, 7,108 行): 与平铺版 md5 相同 6 个 (`p4/fp8/benchmark/quantize/grouped_dispatch/tuning_cache`); 已分叉 6 个 (`contracts/dispatch/layout/preflight/reference/registry`, diff 7~236 行, 平铺版更新). 全仓零引用, 实测 import 失败 (`__init__.py:106` 引用不存在的 `xqt.gemm.common.backends`). 由单次 commit 一次性加入后无人维护.
- `xqt/gemm/backends/sm89/` (13 py, 6,897 行): 11 个与平铺版 md5 相同; 已分叉 3 个 - `sm89_build.py` (平铺 973 vs 子包 890, 缺 `Sm89W4A8BuildConfig`/`build_sm89_w4a8_artifact`), `sm90_fp8_wgmma.py` (平铺 792 vs 子包 295, 缺 6 个符号, 且 sm90 文件放 sm89/ 错位), `tilelang_marlin.py` (import 路径机械改写错误, 实测 ModuleNotFoundError). 唯一引用者 `tests/xqt/gemm/test_sm89_backend.py:26`.
- `xqt/gemm/backends/sm89.py` (353 行) 与 `backends/sm89/` 包同名撞车, 包优先解析, 平铺文件不可达.
- `xqt/gemm/backends/` 顶层 9 个 shim (contracts/fp8/grouped_dispatch/layout/preflight/quantize/reference/registry/tuning_cache, 各 ~13 行): `from .. import X as _impl; globals().update(...)`, docstring 自述迁移落地后删除.
- `xqt/gemm/backends/sm90/`, `sm100/`: 空包 (0 字节 `__init__.py`), 零引用.
- 测试分裂: `test_sm89_backend.py:26` 走子包 (旧), `test_sm90_sm120_backends.py` 走平铺 (新).

### E.2 其他重复 / 壳文件

| 文件 | 行数 | 状态 |
|---|---|---|
| `quant/bridges/nvfp4.py` | 23 | 兼容 re-export 壳, 实体在 `runtime/bridges/nvfp4.py`; 仍被 `quant/__init__.py:137` 与 `quantizers/nvfp4_weight_only.py:16` 引用 |
| `quant/execution/{artifacts,component,selection}.py` | - | 纯 compat shim 转发根目录同名模块, 但 27 处真实 import 全走 shim, 实现位置与导入路径颠倒 |
| `runtime/modules/svd_legacy.py` vs `svd_composite.py` | 16 / 26 | 两个几乎相同的 compat re-export shim |
| `contracts/packing_int4.py:23-37` vs `quant/quantizers/fp4_weight_only.py:95-116` | - | `_pack_int4/_unpack_int4` 逐字节相同两份; 引用三处: `runtime/bridges/hf_int4_layout.py:16` 与 `hf_int4_pack.py:14` import quant 版, `runtime/modules/packing_int4.py:3` re-export contracts 版 |
| `workflows/optimization.py:187-200` vs `session_runner.py:67-80` | - | `_OptimizationRunState` 字段完全一致定义两次, 前者是影子定义 |
| 旧 key 黑名单 | - | `workflows/config_compat.py:18-25` 与 `optimization.py:75-82` 内容相同两份 |
| `quant/sensitivity.py:67` vs `quant/calibration/activation.py:160` | - | 同名私有 `_shared_module_names` 两份, 实现微差 |
| `quantizers/awq.py` / `gptq.py` | 各 20 | 显式 re-export, 有测试依赖, 有意保留 |

### E.3 确认死代码 / 占位

`xqt/gemm/common/` 的**现有内容** (整目录过期快照, 已坏; 目录本身保留, 以平铺层内容重建 - 见主报告 2.2); `gemm/backends/tilelang_marlin.py` (122 行零引用); `gemm/backends/sm89/tilelang_marlin.py` (坏+零引用); `gemm/backends/sm90/`, `sm100/` (空包, 迁移时填位或删除); `quant/quantizers/nvfp4_weight_only.py` (392 行孤儿); `quant/backends/bitsandbytes.py`, `transformers.py` (各 3 行空占位); `pipeline/pass_manager.py` `SequentialPipeline` (零调用方); `core/registry.py` 三套 registry 查询侧; `workflows/config_compat.py:is_optimization_workflow_source` (零调用方); `pipeline/preflight.py:preflight_optimization_config` (仅测试调用, 主链不走).

### E.4 生产零调用但活跃的公共 API (由 tests 支撑)

`runtime/engine.py:HybridInferenceEngine`; `runtime/model_runner.py:ModelRunner`; `runtime/composite_inference.py`; `runtime/composite_branch.py:CompositeBranchModule` (无子类, registry 函数被 `svd_legacy_materializers.py:17-117` 使用); `operator_opt/block_kernels.py` (生产零注册).

## F. quant/ 根目录文件职责表

| 文件 | 行数 | 职责 | 被引用 |
|---|---|---|---|
| `types.py` | 410 | QuantScheme 值对象 + Plan/Report/Result 容器 + composite artifact 构建 | 21 (全包枢纽) |
| `capability.py` | 501 | backend 能力矩阵 `_BASE_CAPABILITIES` + nature 推导 | 14 |
| `sensitivity.py` | 550 | 双模型 per-layer 敏感度/误差分析 | 4 |
| `plan.py` | 306 | QuantConfig → QuantizationExecutionPlan | 7 |
| `layout_apply_report.py` | 310 | LayoutKernelReport 构建 + forward 后刷新 | 4 |
| `external.py` | 268 | HF/vLLM 风格外部量化 checkpoint 探测 | 5 |
| `strategy.py` | 259 | WxAy 字符串 → QuantScheme 模板 | 5 |
| `registry.py` | 196 | route 注册表 (C2) | 4 |
| `policy.py` | 174 | QuantizationPolicy + should_quantize_module | 5 |
| `axes.py` | 172 | 三轴公共事实报告 | 1 (仅测试) |
| `selection.py` | 159 | effective selection policy 解释 | 3 |
| `comfy_quant.py` | 149 | ComfyUI int8_tensorwise marker 编解码 | 1 (仅 convrot_int8) |
| `external_methods.py` | 130 | 外部 method alias/override 链 | 1 (仅 external.py) |
| `component.py` | 85 | 模块路径解析/替换 helper | 2 |
| `channel_helpers.py` | 85 | channel 粒度 tensor 工具 | 0 (仅 convrot_4bit 相对 import) |
| `toy_models.py` | 46 | smoke fixture | 0 (仅 3 个 recipe YAML 字符串 target) |
| `external_types.py` | 42 | ExternalQuantInfo 值对象 | 2 |
| `artifacts.py` | 36 | ONNX artifact 命名 helper | 1 |
| `__init__.py` | 299 | ~127 个符号 eager 导出 | - |

复制粘贴计数 (定义处): `_policy_from_mapping` ×11; `_replace_submodule` ×10 (公共版 `component.py:59` 存在); `_ordered_unique`/`_prefix_module_names` 公共版之外 3 份; `_move_batch_to_device`/`_call_model`/`_iter_calibration_batches` ×3; `_module_parameter_count` ×4; `_can_mutate_runtime_cache` ×3; `execute_*_component` 报告组装样板 ~70-100 行 ×13.

事实表多源: strategy 表 3 处 (`core/schema.py:67-83` / `quant/strategy.py:39-123` / `quant/capability.py:28-44`); TRUE-nature compute 2 处 (`capability.py:46` / `axes.py:80`); pytorch methods 8 元组 2 处 (`capability.py:289-298` / `core/schema.py:97-106`); QuantScheme 字段名 2 处 (`types.py:39-43` / `strategy.py:125-134`).

## G. runtime/ 与 contracts/ 文件职责表

### G.1 runtime/ 根目录 (17 文件)

| 文件 | 行数 | 职责 | 被引用 |
|---|---|---|---|
| `engine_resolve.py` | 506 | engine 注册表 + capability→engine 解析 | 17 |
| `package.py` | 496 | model package 读写 + ONNXRuntimeRunner | 4 |
| `inference.py` | 400 | 语义推理 adapter + registry | 3 |
| `engine.py` | 324 | HybridInferenceEngine | 1 (仅 tests) |
| `quant_pair.py` | 316 | quant pair 文件 I/O | 4 |
| `composite_inference.py` | 304 | SVD composite 推理加速 | 1 (仅 tests) |
| `__init__.py` | 257 | 100+ 符号 facade | - |
| `composite_branch.py` | 243 | CompositeBranchModule 基类 (无子类) + materializer registry | 5 |
| `serving_config.py` | 236 | vLLM/SGLang 风格 launch fragment 生成 | 2 |
| `composite_combine.py` | 231 | Add/Concat/Select 合并策略 ABC | 2 |
| `svd_fusion.py` | 230 | SVDQuant fusion 计划+验证报告 | 2 |
| `policy.py` | 198 | execution policy apply/collect | 23 |
| `model_runner.py` | 187 | ModelRunner 薄 runner | 2 (仅 tests) |
| `quant_pair_schema.py` | 176 | QuantPairManifest/LoadedQuantPair dataclass | 2 |
| `channel.py` | 130 | channel 混合精度 helper | 6 |
| `svd_flux_attention.py` | 124 | Flux 专用 materialize 薄壳 | 4 |
| `composite_materialize.py` | 74 | composite 物化薄壳 | 7 |
| `svd_flux_transformer.py` | 33 | 一函数薄壳 | 2 |

`runtime/modules/` (21 文件, 8,142 行): int8_mma_linear 1301, svd_w4a4_legacy 1243, kv_attention 850, svd_flux_attention 657, awq_w4a16_linear 604, svd_w8a8_legacy 588, svd_flux_block 405, svd_gelu_mlp 368, w4_storage_int8_mma_linear 351, fp8_mma_linear 324, composite_add_w4a4 321, svd_flux_transformer 213, composite_add_fp8 194, svd_legacy_materializers 162, composite_add 157, svd_fp8_legacy 156, composite_norm 123, __init__ 60, svd_composite 26, packing_int4 23, svd_legacy 16.

### G.2 contracts/ (15 文件)

| 文件 | 行数 | 职责 | 纯度问题 |
|---|---|---|---|
| `module.py` | 656 | CompositePrecision*Spec, PrecisionPolicy, ModuleContract | 纯 |
| `runtime.py` | 639 | Runtime/StageReport/ExportBundle/ExecutionPolicy Payload | 纯 |
| `compute.py` | 622 | ComputeConfig/ModuleComputeSpec + normalizers | 纯 |
| `runtime_features.py` | 482 | feature spec + metadata 构建 | 含构建行为 |
| `quantized.py` | 311 | QuantizedModel artifact | 纯 |
| `runtime_quant.py` | 292 | RuntimeQuantContract + attach/extract | 含 `first_linear_shapes` 模型遍历 (:264); lazy import quant.types (:69) |
| `layout_kernel_report.py` | 287 | LayoutKernelReport | 纯 |
| `runtime_manifest.py` | 267 | RuntimeManifest + build (:188) | 含构建行为; lazy import runtime.quant_pair (:224) |
| `composite.py` | 234 | CompositeAddModule/CompositeAddLinear | **含 nn.Module 真实 forward (:187)** |
| `inference.py` | 209 | InferenceContract schema | 纯 |
| `channel.py` | 156 | ChannelHybridSpec | 含张量计算 `compute_hybrid_linear` (:98) |
| `pruned.py` | 114 | PrunedModel artifact | 纯 |
| `contract_consume.py` | 104 | 消费校验 ContractConsumeReport | 含校验行为 |
| `packing_int4.py` | 103 | int4/fp4 pack/unpack | **含量化算法本体 `_quantize_grouped_fp4_weight` (:62)** |
| `scale_time.py` | 48 | scale 时间语义 | 纯 |

## H. operator_opt 结构与 backend 分布

### H.1 顶层散文件

| 文件 | 行数 | 职责 | 备注 |
|---|---|---|---|
| `types.py` | 151 | TargetPlan/ExecutionPlan/Report dataclass | 每个 engine 一个专属 dict 字段 |
| `plan.py` | 216 | YAML config → plan 组装 | |
| `execute.py` | 669 | 执行编排 (扫描→能力检查→compile→guard→benchmark→fallback) | engine 集合硬编码 10 处 |
| `execution_support.py` | 156 | 共享 helper | 与 execute.py 命名易混 |
| `metadata.py` | 500 | engine_metadata/quant_guard/numeric validation | |
| `capability.py` | 336 | `_BASE_CAPABILITIES` | 与 engine_resolve 词汇重叠 |
| `advisor.py` | 350 | PrecisionRecommendation/ProfilingPlan | SM helper 与 gemm_selector 记录在案的重复 |
| `toy_models.py` | 370 | 11 个 Toy* fixture | 应归 model/ |
| `patterns.py` | 20 | 纯 facade 转发 pattern_match/ | |
| `reporting.py` | 132 | 验收判定 | |
| `compile_backend.py` | 47 | torch.compile 薄封装 | |
| `cuda_extension.py` | 169 | custom_cuda 脚手架 | |
| `materialize.py` | 164 | candidate 物化 + `_CONTRACT_PATTERNS` | |
| `runtime.py` | 129 | CUDA Graph helper | 与 xqt/runtime/ 命名撞车 |
| `block_kernels.py` | 89 | block-kernel registry | 生产零注册 |
| `_benchmark.py` | 298 | execute.py 私有 helper | 边界正确 |

### H.2 backend 两侧分布对称性

| backend | backends/ adapter | kernels/ 侧 | capability.py | engine_resolve.py |
|---|---|---|---|---|
| triton | triton.py (7.4KB) | triton/ 7 文件 | available | dispatchable |
| tilelang | tilelang.py (11.5KB) + tilelang_validation.py | tilelang/ 12 文件 | available | dispatchable |
| cutile | cutile.py (8KB) | cutile/ 7 文件 | planned | dispatchable=False |
| cutlass | cutlass.py (5KB) | cutlass/ 1 文件 2.6KB | planned | metadata_only |
| cute_dsl | cute_dsl.py (5.3KB) | cute_dsl/ 1 文件 2.7KB | planned | reference_guarded |
| custom_cuda | cuda_extension.py (顶层) | 无 | planned | **未注册** |
| torch_compile | compile_backend.py (顶层) | 无 | available | **未注册** |
| (裸 CUDA) | 无 | cuda/, cute/ | - | cuda_sm89/ptx_sm89 |

### H.3 新增 GEMM-capable backend 改动清单 (实测 13 处)

1. `runtime/engine_resolve.py:_ENGINE_REGISTRY`
2. `operator_opt/capability.py:_BASE_CAPABILITIES`
3. `backends/<name>.py` adapter
4. `backends/__init__.py` 导出块
5. `kernels/<name>/` 子包 + `kernels/__init__.py` + `kernels/<op>.py` 聚合器
6. `types.py` 加 engine 专属字段
7. `execute.py` 约 10 处硬编码 engine 集合
8. `metadata.py` engine_metadata
9. `materialize.py:_CONTRACT_PATTERNS` + candidate builder
10. `pattern_match/candidate.py:_PATTERN_TO_BACKEND`
11. wrapper (子包或平铺, 无规则)
12. `core/schema.py:30` engine 词表
13. 若进 GEMM: `gemm_selector.py` 分支 + `gemm_precision.py` family 表 (+ `gemm/registry.py` 条目 + `gemm/backends/<arch>` adapter)

## I. workflows / pipeline / core 关键事实

- workflows/ 3,311 行: optimization.py 958, stage_specs.py 912, session_runner.py 468, stage.py 461, stage_provider.py 296, session_targets.py 96, config_compat.py 90.
- pipeline/ 2,967 行: passes.py 523, export_pass.py 479, pass_helpers/prune_stage.py 425 (`_run_prune_with_resolved_config` 是 250 行 4 分支大函数), quant_stage.py 246, preflight_checks/ 7 文件, export_handlers/ 9 handler + `_context.py`.
- core/ 1,438 行: reporting.py 769, schema.py 721 (33 个 dataclass, `:463-464` 双 `@dataclass`), inputs.py 169, artifact.py 155, registry.py 109.
- XQTOptimizationSession: 461-946 行 (~486 行), 25 个公开成员; `export()` (764-815) 与 `deploy()` (817-871) 各 20+ 参数主体几乎逐行重复; `readiness()` (576-641) 11 个参数且内联 artifact/manifest 写入.
- StageSpec vs Config 重复: QuantStageSpec/QuantConfig 11 字段同名; PruneStageSpec/PruneConfig 9 字段; AnalyzeStageSpec/AnalysisConfig 13 字段; BenchmarkStageSpec/BenchmarkConfig 5 字段; OperatorStageSpec/OperatorOptimizationConfig 2 字段. spec→config 转换 3 处: `stage_specs.py:373`, `pass_helpers/context.py:91-108` (逐字段手工 11 个), `preflight.py:90,97`.
- `stage_provider.py` (296 行) 是 dispatch table + 6 个 provider 类, 复杂度可控; per-kind 分支链在 `session_runner.run_optimization_stage` (321-341, 6 个 elif).
- `export_pass.py:51-66` 顶部 import 9 个 handler, 每个 handler `from .. import export_pass as _ep` 再调 `_ep.export_onnx(...)` - 循环 import 靠模块对象延迟属性访问规避.

## J. model/ 与 export/ 结构事实

- model/ 26 文件 6,541 行: smoke_* 五件 + families/hooks 符合宣称; flux2_klein/ (load 861 + optimize 445 + runtime 345 + types 294 + targets 220), wan21/ (5 文件 919 行), hunyuan_ocr.py 415 + hunyuan_ocr_tilelang.py 706, unlimited_ocr.py 465. 包内生产代码零 import 这些模型专用子包; import 者全部在 tests/ 与 examples/.
- export/ 19 文件 4,550 行: 无统一接口; result 类型六种; TensorRT 拆 6 文件 1,185 行 vs mobile.py 单文件 444 行装 5 个后端; reporting.py 606 行混三种报告; export_readiness.py 与顶层 readiness.py 命名撞车.
- prune/ 35 文件 7,531 行, 含 rewrite_ops/ (实际改模型的算子级重写), `rewrite_ops/apply.py:9-10` import `xqt.analysis.compare` + `xqt.export.export_readiness` (prune 跨界依赖 export).
- readiness.py 858 行: 自上而下一聚合方向合理, 但每场景私有 readiness 函数与各子系统 capability 模块存在报告逻辑二次编排.
- xdl_adapter.py 108 行: 对 xdl 包零 import (鸭子类型 getattr), 方向零耦合但零类型保障; `:54-66` checkpoint 布局多层 fallback 猜测链.
