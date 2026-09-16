# xqt 包内工程契约

**核心边界: XQT 只关注模型本身.**

XQT 消费训练后的模型/checkpoint/导出产物,做压缩,变换,导出,误差分析和 benchmark. XQT 不做训练,不做 QAT,不做 finetune / distill / recovery,不构建 dataset / dataloader,也不拥有 task provider 语义.

## 关键抽象

| 概念 | 说明 |
|---|---|
| `xqt.kernels` | 计算栈三层: `ops` (tensor kernel + GEMM 合约), `wrappers` (materialize / operator / bench), `nn` (facade / convert / fixtures). 公开 kernel 入口仍是 `xqt.kernels.ops.*`, 详见 `docs/md/architecture/xqt-kernels.md` |
| model adapter/profile | `xqt.model` 提供具体模型适配器协议与 profile registry,当前承载 HunyuanOCR,Unlimited-OCR,Wan 2.1,Flux.2 Klein 实现. adapter 可以实现架构组装,checkpoint mapping,特殊 forward 和 IO 包装; profile 通过 `model.profile` 选择 adapter. 通用 layer,operator,kernel,quant,prune 不归该层. 已加载模型可通过 `model=` 直接注入并跳过 adapter.load. `examples.xqt_models` 保留为兼容 re-export. |
| GEMM 合约 | `xqt/kernels/ops/gemm/contracts.py`: GemmProblem, GemmSpec, QuantSpec 等 (P3/P4 完成, P5 in progress) |
| model structure contract | `xqt.contracts.model_structure.ModelStructureContract` 是模型族结构的声明式事实源: 组件角色分组 (`COMPONENT_ROLES`), merged projection (`MergedProjectionSpec`), 声明式 checkpoint weight mapping (`WeightMappingEntry` + `resolve_weight_mapping` 全覆盖验收, 未覆盖 key 硬错误), `structure_contract_mismatches` 契约-vs-模型一致性纯视图. `build_structure_contract()` (fixtures) 是启发式派生草稿, 真实接入应固化为手写契约. 目标消费方是 quantizer component plan / prune candidates / block materialize; 契约不选 engine, 不拥有 forward 语义, 不改写模型. 对标 SGLang `models/*.py` 声明, 见 `research/xqt-quant-inference-architecture/LAYERING_VS_SGLANG.md`. |
| TileLang Timing Cache | `xqt/kernels/ops/_impl/tilelang/tuning_cache.py`: `TileLangTimingCache`, `TimingCacheKey`, `TimingCacheEntry`, 为 TileLang 算子提供文件级持久化调度缓存, 候选搜索空间生成与自适应 JIT 调度, 对接 `resolve_tilelang_linear_schedule`. |
| TileLang Auto-Tuner | `xqt/kernels/ops/_impl/tilelang/autotuner.py`: `TileLangAutoTuner`, `tune_linear_schedule`, `AutoTuneConfig`. 提供跨硬件维度的 TileLang 算子在线/离线自动寻优, 通过 CUDA Events 测时挑选最优分块并自动写入持久化缓存. |
| SM90 WGMMA 调度 | `xqt/kernels/ops/_impl/tilelang/sm90_wgmma.py`: `SM90WgmmaSchedule`, `resolve_sm90_wgmma_schedule`, `check_sm90_execution_readiness`. 声明并守卫 Hopper 架构下 128-thread Warpgroup 及 3/4 阶段 TMA 异步流水线, 提供严格的硬件对齐与 fallback 保护. |
| MoE Grouped GEMM | `xqt/kernels/ops/gemm/moe_grouped.py`: `MoEGroupedGemmProblem`, `MoEGroupedLayoutReport`, `MoEPersistentScheduler`. 建立稀疏不均衡专家 Token 的持久化切片与任务编排契约, 消除小 Batch 推理下零散 Small Kernel Launch 延迟. |
| Layer-Sequential 量化 | `xqt/compression/quant/sequential.py`: `quantize_layer_sequential`, `discover_sequential_partition`, `LayerSequentialConfig`. 提供逐层激活前向捕获, block 内部局部校准与量化, 激活推进与 CPU 卸载/显存清理机制, 解决 70B+ 大模型在单卡 24G 显存环境下的量化 OOM 瓶颈. |
| 正交旋转变换 | `xqt/compression/quant/transforms/orthogonal.py`: `OrthogonalRotationTransform`, `build_random_orthogonal_matrix`. 提供基于 Haar 分布的离线正交旋转吸收, 削平通道离群值, 抑制 4-bit/3-bit 激活量化损失, 无额外在线推理延迟. |
| 事务图改写引擎 | `xqt/compression/quant/transforms/engine.py`: `speculative_graph_rewrite`, `SpeculativeRewriteConfig`. 提供快照备份, 推演改写, 数值等价性比对与四维门禁自动回滚/提交机制. |
| 自适应优化舍入 | `xqt/compression/quant/quantizers/adaptive_rounding.py`: `optimize_linear_rounding`, `quantize_with_adaptive_rounding`. 基于局部输出重构均方误差微调量化舍入方向, 相比朴素 RTN 显著降低低比特量化误差, 并直接接入 Layer-Sequential 调度器. |
| compressed-tensors 交付 | `xqt/export/hf_quant.py`: `export_compressed_tensors` 原生输出工业级 Safetensors 与完整 `quantization_config` 闭环, 与外部运行时双向互通. |
| 注册表 | `xqt/kernels/ops/gemm/registry.py`: GemmKernelRegistration, maturity levels (executable/metadata_only/planned) |
| Dispatch | `xqt/kernels/ops/gemm/dispatch.py`: dispatch_gemm, fallback chain |
| P5 | TODO-P5.md: persistent grouped scheduler, multi-stream prepack, CUDA Graph |

## 重要类/方法注释

- **GemmKernelRegistration**: 跟踪内核成熟度,能力矩阵,tile 参数和 executor.用于 dispatch 过滤和 fallback.
- **GemmCapability.supports()**: 检查能力是否匹配 problem/quant/epilogue.
- **GemmProblem.__post_init__**: 校验维度,处理 MoE grouped case (M=0 allowed).
| `OptimizationConfig` / `StageSpec` | workflow schema 与 stage typed 视图位于 `xqt.core.workflow_schema` / `xqt.core.stage_specs`; workflow 层只负责 OmegaConf loader 和领域校验. `StageSpec` 不作为 YAML union 字段持久化,由 loader 在运行时附着;runtime Config 统一经 `stage_spec_to_config()` 派生,不手写字段复制. |
| pipeline 执行层 | `xqt.pipeline` 是 `xqt.workflows` 使用的内部执行实现: `runner` 构建 `XQTContext`, `preflight` 做轻量可执行性检查, `passes`/`pass_helpers` 执行 stage, `export_handlers` 处理具体导出目标. 不新增第二套 workflow 编排或 recipe schema; `passes.py` 中的独立 pass 可按职责拆分,但保留旧模块导出以避免内部调用方断裂. |
| `load_optimization_config()` | `OptimizationConfig` 加载器. |
| `XQTOptimizationSession` | 交互式 stage 编排入口. |
| `SessionStage` / `StagePayload` | session 内部 stage 图和阶段产物协议. |
| `QuantizedModel` / `QuantizedModelPayload` | `QuantizedModel` 是模型侧量化算法的通用语义结果; Infer 交接面是 `model` + 可选 `compute_config` (`infer_handoff()`). `backend`/`method`/`strategy` 仅 quant lineage. `QuantizedModelPayload` 追加 stage lineage, artifacts 和 capability. |
| contracts storage shell | `xqt.contracts.int8_mma.Int8MmaLinear` 与 `xqt.contracts.w4_storage.W4StorageInt8MmaLinear` 保存 packed/int8 artifact 并提供 backend-neutral reference forward. `xqt.runtime.modules` 中的同名类是可选执行视图, 通过 `from_storage()` 显式 materialize, quant 不反向 import runtime. |
| contracts boundary | `xqt.contracts` 负责 typed payload,存储协议与 reference 语义; packing/repack helper 由 contracts 提供唯一实体. `contracts` 只依赖 `xqt.core.base` leaf 层的 error/artifact/serialization,`xqt.core` 的 schema/stage/reporting 高层可显式依赖 `contracts`,依赖方向不成环. CUDA/Triton/CuTe 等后端执行视图归 `xqt.runtime` 或 `xqt.kernels.wrappers`,不把 reference forward 误称为 native kernel. engine 解析在 `xqt.kernels.engine_resolve`; NVFP4 unpack 在 `xqt.kernels.ops.quantization.nvfp4`, bridge 在 `xqt.kernels.wrappers.nvfp4`; `PrecisionPolicy` / `ModuleContract` 在 `xqt.kernels.precision`. |
| `RuntimeQuantContract` | 自研 runtime 内部事实源: `QuantScheme` + `storage_layout` + `repack_version` + `required_kernels` + global/local shape + shard/prefill/decode + 可选 `kv_cache_dtype`. 经 `QuantizedModel.with_runtime_quant_contract` / `resolve_runtime_quant_contract` 写入 metadata 键 `runtime_quant_contract`. 也写入 quant pair `quant.json` metadata (见 `write_quant_pair`). 自研模块只读此 contract, 不解析 HF `quantization_config`. HF export / serving_config 是可选 vLLM-style adapter; SGLang 只作为同一输出风格的兼容消费者, 不定义本字段语义. 见 `xqt/contracts/runtime_quant.py` 与 research GUIDE. |
| 三轴公开 API (DEBT-002) | `xqt.contracts.quant_strategy` 是 strategy→`QuantScheme`→nature 以及 canonical method/compute 的唯一事实源. 轴 1 quant method (`quant_method_specs`),轴 2 storage / activation scheme (`quant_storage_specs`,canonical strategy 字符串即 scheme alias),轴 3 compute / MMA contract (`quant_compute_specs`). `quant_axis_report` 汇总三轴;`QuantBackendCapability.methods` 只列轴 1,`storage_strategies` / `compute_contracts` 分列轴 2/3. |
| quantizer 公共骨架 | `xqt.compression.quant.quantizers.base.QuantizerTemplate` 统一 policy 解析,模块替换与 calibration 输入消费;标准 component executor 通过 `component_route_handler()` 生成 registry adapter;模型侧算法通过 `build_component_quantization_report()` 统一公共报告字段. 算法模块只保留 pack/module/nature 等差异. |
| `RuntimeManifest` | 聚合 contract + layout reports + selected/fallback kernels + prefill/decode flags. `build_runtime_manifest(quantized \| pair_path \| mapping)`. 见 `xqt/contracts/runtime_manifest.py`. |
| `RuntimeFeatureMetadata` / `runtime_feature_report` | runtime feature 事实源: `prefix_cache`, `paged_kv`, `kv_cache_quant`, `chunked_prefill`, `speculative_decode`, `continuous_batching` 每项声明 `scope` (model-side metadata vs runtime capability) 与 `xqt_status` / `owner`; report 能解释支持 / 不支持 / 未验证的具体原因. XQT 不实现 cache 管理, batching 调度或在线 decoding. 见 `xqt/contracts/runtime_features.py`. |
| `CudaGraphDecodeSession` | 模型侧 CUDA graph decode 运行时: static KV 缓存 + **单张 length-agnostic 图** (设备标量 `position` / `valid_len` 驱动 replay, 不再按长度桶捕获, 不需要零填充校正) + Triton 单遍 GQA decode attention (`xqt/kernels/ops/_impl/triton/decode_kernels.py`, split-K 两段式, 变长前缀无掩码, K/V 每层只读一遍) + 融合 RoPE/KV-scatter + 融合 RMSNorm/SwiGLU + q/k/v 与 gate/up 投影融合 + o_proj/down_proj **残差 epilogue** (`bind_residual`: `hidden += projection(x)` 单 kernel 原地完成, 与未融合 add 逐位一致) + 可选 W4A16 LM head; `int8_activations=True` 让线性层输入走 per-row INT8 量化/反量化 (QuaRot 式 INT8 激活 + 4-bit 权重); `generate` 以 `readback_chunk=16` 分块回读 token (每 16 token 一次设备同步, EOS 处截断, 超额步如实计入 `decode_steps`). 仅 Llama 家族, 单请求 greedy; MiniCPM5-2B AWQ W4A16 上 e2e `5081 ms` / steady `301.8 tok/s` (w4a8 `5015 ms` / `300.3 tok/s`), 相对 R-052 记录的 bf16 graph 基线 `3.20x`/`3.24x` (同运行时对照 `2.67x`/`2.70x`, 因 bf16 路线同步受益; 单次稳态读数跨运行可波动到 265-311, 只作参考), 相对 BF16 eager `5.1x`. 另有 opt-in prefill 分相特化 `int8_prefill=True`: 只对 MLP gate/up/down 走 per-row INT8 激活 + INT8 权重 (W8A8) Tensor Core GEMM (`_MiniCPM5W4A16HybridLinear.forward_prequant` + `quantize_int8_rowwise_triton` + `gemm_int8_triton`), q/k/v/o 与 attention 保持 bf16; 由 `min_int8_prefill_rows` (默认 `256`) 按行数 gate, 不足阈值或缺少 INT8 视图时逐层回退 dense bf16. MiniCPM5-2B 真实 1715-token prefill 同运行 A/B: `113.89 -> 68.92 ms` (`1.652x`), decode steady `0.997x` (无回归), 首 token 一致, KV `min_cos 0.9985`; 任务级 1536-token char BLEU-4 vs BF16 `0.1803` (dense-prefill `0.1807`, 等价). 边界: INT8 权重由 dense bf16 权重派生 (额外 `N*K` int8 字节/MLP 投影), 小 M (`<~128-192`) 亏损故需阈值. 不承担调度, 批处理, 采样, paged KV 或 serving. 见 R-056 与 `xqt/runtime/graph_decode.py`. |
| `KvCacheMetadata` / `KvScaleAttention` | Attention/KV 模型侧实体 (C9 residual): 消费 `KvScaleArtifact` 与 `RuntimeQuantContract`,保留 torch SDPA reference 路径,并提供 opt-in TileLang KV-int8 fastpath. 模块只保存一个 packed QKV Linear 参数源;fastpath 依次执行 packed QKV GEMM,双输出 packed quantize-layout,直接消费 Q third/INT8 K/V 并产出 BSI 的 attention,以及 output GEMM. `preferred_kernel` 选择 reference/TileLang,独立的 `attention_fastpath="eager"|"graph"` 选择 dispatch;graph capture 覆盖完整 packed forward,按固定 tensor/mode/SM/parameter-storage contract 缓存,`.to()` 清缓存. Report 记录 projection mode,两级 selected kernels,selected fastpath,graph state/reason/cache size 和 fallback. `sm_89` 4-shape paired gate 中 graph 相对 eager 降低 `37.46-69.09%`;输出为 graph-owned storage,后续 replay 会覆盖旧 view. 不包含 page table,cache 管理,并发 serving replay 或调度. 见 `xqt/runtime/modules/kv_attention.py`. |
| `speculative_decode_metadata` / `prefix_paged_kv_metadata` | 只记录模型侧关系与运行时指标: speculative decode 记 draft/target model, acceptance rate, backend support; prefix cache / paged KV 记开关, cache block size, hit rate. 不管理任何缓存. 见 `xqt/contracts/runtime_features.py`. |
| `LayoutKernelReport` | load/process/apply 诊断字段 (bits/group_size/sym/zp/desc_act/g_idx_applied/shapes/padding_ratio/storage_layout/selected_kernel/fallback_reason/sm/min_capability). 键始终出现 (可空). apply 路径经 `xqt/compression/quant/layout_apply_report.py` 写入 quant metadata; 原生 AWQ 默认 `desc_act=False`/`g_idx_applied=False`; 外部加载经 `ExternalLoadReport.layout_kernel` 暴露实测值. |
| `assess_export_readiness` | 量化后 export 阻塞诊断: packed / 特化模块 → `can_export` / `blockers` / `suggested_lowering`. 见 `xqt/export/export_readiness.py`. |
| `benchmark_prefill_decode` | 离线 prefill/decode 分相 latency, 始终 `offline_estimate=True`, 不接 serving. 见 `xqt/kernels/wrappers/bench/phase_latency.py`. |
| `consume_runtime_quant_contract` | 自研模块消费前校验 contract 字段完整性. 见 `xqt/contracts/contract_consume.py`. |
| `ModelRunner` | 模型侧薄封装: 读 RuntimeQuantContract + eager forward + layout selected_kernel/fallback 报告; 可选 offline phase latency. **不是** serving scheduler. 见 `xqt/runtime/model_runner.py`. |
| `ComputeConfig` / `ModuleComputeSpec` | 可选计算配置: `compute_contract`, `precision`, `required_capabilities`, `preferred_engines` (hint). 禁止 `required_engine` 主键. 见 `docs/md/architecture/xqt-infer-handoff.md`. |
| composite 词表 | **additive dual-branch** (`compute_contract=composite_add`, `combine=add`): 低秩高位支路 + 量化 residual 支路并行相加 (SVDQuant). **k-group partition** (`CompositePrecisionGemmSpec`): 沿 K 轴静态分组选支路, 与 additive 分解正交, 不是同一模型. SVD 公开主键是 `method=svd` + storage/compute; `strategy` 仅 WxAy 别名. |
| `PrunedModelPayload` | `xqt.contracts` 定义的 prune stage typed payload, 记录模型,sparsity report,lineage,artifacts 和 runtime capability. |
| `supported_prune_method_specs` | prune method 事实源: `global_l1_unstructured`, `structured`, `nm_structured`, `block_sparse`; 报告 baseline_kind, rewrites_structure, pattern_present, speedup_claimed 和 speedup_verified. 见 `xqt/compression/prune/methods.py`. |
| `structured_prune_support_matrix` | prune granularity 支持矩阵: model family, module type, 是否真实结构改写, exportable, speedup_verified 和 runtime_support. 见 `xqt/compression/prune/support_matrix.py`. |
| `normalize_prune_granularity` / `describe_prune_granularity` | prune granularity 词表事实源: alias 归一 (`attention_head` -> `head`), 每个 granularity 声明 `rewrites_structure` / `exportable` / `runtime_support`; `token` 等 metadata-only 条目在结构化 plan 中显式失败. 见 `xqt/compression/prune/granularity.py`. |
| `assess_prune_safety` | 模型族 safety guard 事实源: Transformer attention head 整除与 MLP pair / gated-MLP shape 契约, Conv2d producer/consumer 与分组卷积契约, Detection detect-head 保护; structured apply 前硬校验, report 写入 `safety`. 见 `xqt/compression/prune/safety.py`. |
| `snapshot_module_dimensions` / `diff_module_dimensions` / `estimate_model_flops` | 剪枝 topology change 事实源: removed modules, changed dimensions, mask-only modules; 形状级 FLOPs 估算与 parameter count / latency benchmark / export status 分字段记录. 见 `xqt/compression/prune/dimensions.py`, `xqt/compression/prune/flops.py`. |
| `RuntimePlanPayload` | operator stage 的 runtime plan typed payload. |
| `scan_operator_candidate_reports` | operator candidate scanner 事实源: 汇总 FX / torch.export 候选模式, shape signature, dtype, device, recommended backend 和 source 失败原因. |
| `operator_pattern_coverage_report` | operator 高频模式覆盖事实源: 汇总 Linear/GEMM,dequant GEMM,attention,norm,RoPE,activation epilogue 是否在候选扫描中出现; 不把 planned backend 冒充 executable. |
| `torch_compile_graph_report` | operator graph break 事实源: 记录 graph count, break count/reasons/details, compile times, mode, dynamic, fullgraph 和 target/benchmark path. |
| `operator_planned_skip_report` | operator planned / skip 事实源: 记录 CUDA-only 或 metadata-only backend 的执行状态, 缺失条件, fallback 和 target/benchmark path. |
| `quant_runtime_guard_report` | operator quant runtime guard 事实源: 记录量化 backend/runtime 来源, 非 PyTorch runtime skip 原因和允许的 PyTorch runtime 集合. |
| `operator_numeric_validation_report` | operator 数值验证事实源: 记录 target/benchmark path, pass/fail/not_run 状态, allclose, 阈值和失败原因; `rejected_numeric` 不替换当前模型. |
| `ExportBundlePayload` | export stage, 或未 materialize runtime handle 的 deploy stage 的 export bundle typed payload. |
| `export_target_capability_report` | export target capability 事实源: 汇总 format,opset,dynamic shape,precision,quant support,sparsity support 和 plugin support; planned/adapter 能力不冒充 runtime 验证. |
| `export_artifact_lineage_report` | export artifact lineage 事实源: 记录 source stage,input/output signature,checksum,size 和 quant/prune/operator 上游 stage lineage. |
| `onnx_graph_diagnostics_report` | ONNX graph diagnostics 事实源: 记录 op/domain 统计,unsupported/custom ops,实际导出的 dynamic axes,input/output signature; `export_onnx` metadata 和 ONNX handler artifact entry 透传该报告. |
| `openvino_runtime_layer_report` | OpenVINO runtime-specific 分层事实源: 区分 conversion, runtime load, output diff 和 runtime benchmark. 当前 export target 尚无 OpenVINO benchmark 配置, benchmark 层显式标为 `not_configured`, 不冒充真实性能验证. |
| `tensorrt_runtime_layer_report` | TensorRT runtime-specific 分层事实源: 区分 dry-run command,engine build,plugin presence,plugin loadability 和 runtime benchmark. plugin presence 只证明文件存在,loadability 才证明当前进程可加载; TensorRT handler artifact entry 和 manifest metadata 透传该报告. |
| `RuntimeHandlePayload` | executable runtime handle typed payload; materialized deploy stage 使用该 payload, 可生产 ONNX Runtime `InferenceSession` 或非 dry-run TensorRT runtime session. |
| `operator_acceptance_record` | operator target 验收事实源: applied / rejected_min_speedup / rejected_numeric / skipped, 记录 min_speedup, numeric diff, fallback reason 和 graph breaks; manifest metric 直接消费该记录. |
| `StageComparison` | session 内 stage-to-stage 结构化比较结果. |
| `ArtifactManifest` / `ArtifactRecord` | 产物追踪. |
| `ModelPackageManifest` / `load_model_package()` | 推理侧文件包加载标准, 当前最小闭环为 `manifest.json + runtime/config.json`; 可选 `runtime/compute.json` (compute_config). |
| `InferenceContract` / `create_inference_session()` | 模型侧语义推理标准. `InferenceContract` 描述模型族, adapter, 语义输入输出和 adapter config; `InferenceSession` 位于低层 runtime runner 之上, 不负责 serving 调度, dataset 或 task-level validation. |
| `write_quant_pair` / `load_quant_pair` | 扁平 Infer 交付: `model.pt` + `quant.json` (`artifact_type=xqt_quant_sidecar`). `quant.json` 含 compute_config + lineage + 可选 `runtime_quant_contract`; 不是 quant recipe; 加载不跑 quantizer. 外部权重: 单文件或 `model.safetensors.index.json` 分片 merge (`weight_io`). |
| `MetricRecord` | 结构化指标记录. |
| `OptimizationCapability` | 统一 capability 投影,覆盖 quant / prune / operator / export 的 engine,status,maturity,runtime,artifact_kind 和硬件/校准/导出要求. |
| export adapter contract | `xqt.export.base.ExportAdapter` 与 `ExportResultBase` 统一 exporter/result 表面;既有函数通过 `FunctionExportAdapter` 复用. TensorRT plugin,engine inspector 与 performance parser 收敛到 `trt_diagnostics.py`,对外只从 `xqt.export.tensorrt` 聚合. |
| `suggest_quant_backends` / `QuantBackendSuggestionReport` | P3 自动策略: 基于 quant capability 单一事实源与设备约束输出候选 backend 建议 (含排除原因), 只建议, 绝不改写用户配置. 见 `xqt/auto/backend_suggestion.py`. |
| `suggest_precision_actions` / `PrecisionSuggestionReport` | P3 自动策略: 基于 layer sensitivity 记录自动建议 `keep_high_precision` / `skip_quantize` / `quantize`; 显式 keep/skip 配置的模块进入 `protected_modules`, 建议增量 (`policy_delta`) 不覆盖显式配置. 见 `xqt/auto/precision_suggestion.py`. |
| `rank_stage_benchmark_history` / `StageBenchmarkLeaderboard` | P3 自动策略: 基于 benchmark 历史 (stage_results / SessionStage / mapping) 输出 best stage 与 rejected stage 排行榜, 支持 speedup / memory / numeric 三种排序; 纯视图, 不改变 session 状态. `XQTOptimizationSession.benchmark_leaderboard()` 是便利入口. 见 `xqt/auto/stage_history.py`. |
| `StageAcceptanceConfig` / `evaluate_stage_acceptance` | P3 acceptance policy 四维: numeric diff (`max_mean_abs` / `max_max_abs` / `max_relative_error`), speedup (`min_speedup`), peak memory (`max_memory_mb`), accuracy drop (`max_accuracy_drop`); 证据缺失时对应检查显式失败并给出原因. 见 `xqt/auto/acceptance.py`. |
| `build_scheme_search_space` / `run_scheme_search` | P3 strategy search 安全边界: 搜索空间定义在 `QuantScheme` 上 (由 canonical strategy 事实源派生), 限制尝试次数, 每次尝试记录 (状态/消息/指标/异常), seed + plan hash 保证可复现; 只输出建议, 不自动改写配置. 见 `xqt/auto/strategy_search.py`. |
| `supported_quant_backends` | quant backend 能力枚举事实源, 从 `_BASE_CAPABILITIES` 派生, 供自动建议与 capability 矩阵共用. 见 `xqt/compression/quant/capability.py`. |
| `classify_model_family` / `component_grouping` / `family_smoke_report` | 模型族通用 helper: 结构启发式 + task 元数据分类 (transformer / vit / detection / llm / diffusion / moe / multimodal / convnet), 组件分组 (attention / ffn / norm / head / expert / router / embedding / backbone), smoke report 显式标记 synthetic 且不声称真实性能收益. 见 `xqt/kernels/nn/fixtures/families.py`. |
| `SmokeViTClassifier` / `build_smoke_vit_classifier` | 小 ViT smoke 模型 (patch embed + TransformerEncoderLayer + norm + head), 供模型族 recipe 做 CPU 链路验证. 见 `xqt/kernels/nn/fixtures/smoke_vit.py`. |
| `transformer_vit_smoke.yaml` | Transformer/ViT 模型族 smoke recipe: prune (unstructured) + quant (torchao w8a8) + ONNX export + benchmark, 与 `SmokeDetectionModule` / detection smoke recipe 并列, 均只验证链路可运行. 见 `xqt/recipes/smoke/` 与 `xqt/recipes/detection/`. |
| `SmokeLLM` / `SmokeMoE` / `SmokeDiffusionDenoiser` / `SmokeMultimodalClassifier` | 各模型族 smoke 模型 (小参数, CPU 可跑): LLM 含独立 q/k/v 投影 (可被 `calibrate_kv_scales` hook), MoE 含 router / experts / shared_expert, Diffusion 含 timestep embedding, Multimodal 含 vision encoder + cross attention. 见 `xqt/kernels/nn/fixtures/smoke_*.py`. |
| `moe_family_report` / `suggest_expert_pruning` | MoE 模型侧 metadata 事实源: expert / shared expert / router 计数, expert parallel readiness 显式 `metadata_only`, load balance 未验证; expert 剪枝只提供纯建议 (按平均 router 分数), 不修改模型. 见 `xqt/kernels/nn/fixtures/smoke_moe.py`. |
| `diffusion_smoke_report` / `visual_token_compression_metadata` / `encoder_cache_metadata` / `multimodal_input_signature` | Diffusion / Multimodal 模型侧 metadata: 采样步数上下文与导出限制, visual token 压缩意图, encoder cache 归属外部 runtime, multimodal 输入签名; 均不实现采样循环 / 压缩 / cache 管理. 见 `xqt/kernels/nn/fixtures/smoke_diffusion.py` 与 `xqt/kernels/nn/fixtures/smoke_multimodal.py`. |
| `llm_smoke.yaml` / `moe_smoke.yaml` / `diffusion_smoke.yaml` / `multimodal_smoke.yaml` | 各模型族 smoke recipe (prune + benchmark, MoE 另含 expert weight-only quant), 显式 synthetic 链路验证. 见 `xqt/recipes/smoke/`. |
| `svd_fusion_report` / `fused_svd_forward` | DEBT-005: SVDQuant FUSE_DOWN / FUSE_UP 融合契约与 CPU reference 数值路径; `cuda_verified=False` 显式标注, 真实 fused CUDA kernel 验证待验证. 见 `xqt/runtime/svd_fusion.py`. |
| `XQTReadinessReport` / `assess_xqt_readiness()` | readiness 汇总入口,输出场景状态,capability matrix 和 reporting schema. |
| `example_inputs` | 导出 / benchmark / layer analysis / operator wrappers 所需输入. |
| `calibration_inputs` | PTQ / QDQ calibration 所需输入. |

## Backend / Engine 术语

完整规则: `docs/md/architecture/xqt-engine-quant-boundary.md` (method / storage / compute / engine 分词与禁止清单).

Engine 词表事实源 (DEBT-001): 实现侧唯一权威是 `xqt.kernels.engine_resolve` 的 `EngineRegistration` 注册表 (`engine_registry_names()`); 配置面词表是 `xqt.core.schema.OPERATOR_OPT_ENGINES`; `xqt.convert` 只接受 materialize preference 子集 `CONVERT_ENGINE_NAMES` (`torch` / `triton` / `tilelang` / `cutile` / `cute_dsl`), 不是 infer 交接主键. 回归测试断言三份词表不漂移.

XQT 是本仓库内唯一推理优化主体. Python API 是主入口, 包括 `XQTOptimizationSession`, `xqt.convert(...)` 和 `xqt.nn.*` facade. `HybridInferenceEngine` 与 `ModelRunner` 仅是 reference/交互式便利封装,不是 deploy runtime handle,serving scheduler 或生产引擎主入口.

- `xqt.compression` 是模型侧压缩唯一落点: `xqt.compression.quant` 负责量化算法与 artifact (packed weight, scale, rotation, execution policy / compute_config metadata, channel hybrid mask); `xqt.compression.prune` 负责 unstructured / structured / N:M / block sparse. quantizer 产出 contracts storage shell, 不把 runtime execution view 当作量化结果, 也不把 operator engine 名写成推理必选主键. 顶层 `xqt.quant` / `xqt.prune` 已删除.
- `xqt.runtime.HybridInferenceEngine` 只消费已量化模型与 execution policy / compute_config,提供 reference/交互式模块级与通道级混合精度检查;`from_quantized_model()` 默认复制并物化 marker-based ConvRot execution view,可用 `materialize_execution_views=False` 保持 reference-only;不跑 quantizer / calibration / sensitivity,也不代表生产 runtime.
- `xqt.kernels.engine_resolve` 按 `required_capabilities` (+ 可选 preferred_engines hint) 解析 operator engine; 不是 quant method 选择.
- Operator engine 只管算子实现 / 融合 / MMA lowering. AWQ / GPTQ / SVD 是 quant **method**, 不是 engine methods.
- `ArtifactManifest` 只用于 workflow / experiment 追踪, 不是 file-based inference 的加载契约. 推理侧文件入口二选一: (1) 模型包 `manifest.json` (`load_model_package`); (2) 扁平 `model.pt` + `quant.json` (`load_quant_pair` / `load_quant_pair_into_model`). 二者都只消费已量化存储 + 可选 `compute_config`, 不解析 quant recipe YAML.
- 模型包的推理调用分为两层: `create_inference_runner()` 只接受已经准备好的 tensor, `create_inference_session()` 通过 `InferenceContract` 和模型族 adapter 完成语义输入预处理与输出规范化. 内置 adapter 目前为 `tensor` 和 `vision.classification`; 模型专用逻辑应优先落成 manifest config, 不复制完整推理入口.
- `InferenceContract` 是模型侧 IO contract, 不是 task registry. tokenizer, 采样循环, dataset, batch scheduler 和 accuracy / mAP 评测仍归调用方或外部 runtime.
- 通道级混合精度: 部分 channel 走 16-bit (或更高), 其余走 4-bit. Quant 侧选 outlier channel 并写入 mask; Runtime 侧 dual-path reference 前向 (`channel_hybrid_linear_reference`), 后续可替换为真实 kernel.

- `backend`: 外部 quant/export/runtime 选择, 例如 `torchao`, `pytorch`, `onnxruntime_qdq`, `tensorrt`, `openvino`. Quant recipe 继续使用 `quant.params.backend`. **`tilelang` / `svdquant` 不是 quant backend**; AWQ/GPTQ/SVD 写 `backend=pytorch` + `method=awq|gptq|svd`.
- `engine`: XQT 内部实现选择和公开 report 字段, 例如 `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `custom_cuda`, `torch_compile`. `xqt.convert(...)`, 单算子 dispatcher, `OptimizationCapability`, `StageReport`, `operator_optimization.default_engine` 和 `targets[*].engine` 统一使用这个字段.
- 不要把 operator engine 写成 quant/export backend alias. 旧 operator 调用点要迁移到 `engine`, 但 quant/backend 语义不能硬改名.
- `maturity`: capability 的实现成熟度分层, 当前统一为 `executable`, `reference_guarded`, `metadata_only`, `planned`. `status` 继续表达接口/适配可用性, 不与 maturity 混用. C10: auto 链头为 tilelang → triton → torch; `cutlass` 为 `metadata_only` 且 `dispatchable=False`; `cute_dsl` / `cutile` 为 `reference_guarded` 且 `dispatchable=False`; 不得把它们标成 auto 链 executable 赢家. `tilelang`/`triton` 的 `provides` 含 `hadamard_groupwise` (C6 online). quant 可执行路由必须声明 `primary_kernel` + `reference_kernel` (`xqt/compression/quant/registry.py`). quant pair 默认可落盘 `runtime_manifest` (V1); forward 后可用 `refresh_layout_kernel_after_forward` 回写实测 `selected_kernel` (V2).

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
- `xqt.nn.Linear` / `Conv2d` / `LayerNorm` 已是保留 PyTorch module/state_dict 语义的 `torch.nn` 子类 facade, 并显式承载 engine 与 precision runtime intent. `FeedForward` / `RMSNorm` / `Attention` / `TransformerBlock` 也是 semantic facade; Attention 的 torch 路径走 SDPA,tilelang 路径经 `materialize_module` 落到 `_TileLangXqtAttentionWrapper`. 受限 TileLang attention 接受匹配的 FP16/BF16 Q/K/V,要求 `dropout_p=0`,`seq_kv >= seq_q`,且 BF16 `head_dim` 必须 16 对齐. Ada `auto` 仍走 native SDPA;显式 TileLang 和 graph 模式才选择自研 kernel. Direct TileLang Linear 接受匹配的 FP16/BF16 activation,weight,bias 和 output,使用 FP32 accumulator,允许 partial M/N,且要求 K 能被 `block_k` 整除. `sm_89 + BF16 + flattened M<=4` 的显式 TileLang 默认 schedule 为 `16x64x32/128 threads/2 stages`;FP16 只在 exact signature `M<=4,K=N=4096,activation=None` 下使用同一 schedule,`N=11008` 或融合 activation 保留 `64x64x64`. Resolver 必须收到 `out_features` 和 `activation` 才能命中 FP16 preset;调用方显式 block 设置优先,Linear `auto` 路由不变. TransformerBlock 的 tilelang 会 materialize 内部 `attn` 子模块,但不是完整 block-level 单 kernel fusion. 文档不得把通用 facade 的 block-level 生产性能写成已完成事实.
- Attention causal 语义统一为方形输入使用 SDPA `is_causal=True`,非方形输入使用 lower-right alignment. Native SDPA,TileLang 和 Triton wrapper/reference 必须对同一 lower-right contract 做 correctness gate,不能让 upper-left native fallback 与 lower-right custom kernel 静默分叉. SM89 exact decode 的 wrapper/runtime profile 只用于拆分 projected kernel,projection/layout 和 module 入口成本;单次 sequential median 不能替代多 seed paired gate. 当前 Triton attention 仅为显式 engine,`attention_fastpath="auto"` 保持既有 native/TileLang 路由.
- Triton dense FP16/BF16 GEMM 的 reduction 由 `accum_dtype` 控制,bias 和 activation epilogue 在 FP32 中执行,output 只在 store 时转换. BF16 no-override 路径在真实或显式 `sm_89` 上为五个受测 `(M,N,K,bias,activation)` exact signature 使用 evidence-backed schedule preset. FP16 resolver 的 exact key 额外包含 `transpose_b`:M1/M4/M8/M64 preset 覆盖 K,N 和 N,K,M256 GELU 只覆盖 K,N. 六个显式调度字段逐项优先,其他 shape/SM 保留 `128x128x32/group_m=8/4w/3s`. N,K weight 通过逻辑 stride 交换直接传入 kernel,无状态 dispatcher 不 materialize transpose,也不保存 tensor 或 hidden weight cache. Stateful `_TritonLinearWrapper` 默认 `weight_layout="transpose_stride"` 和 `linear_fastpath="eager"`;显式 `prepacked_kn` 保存 non-persistent 派生权重,显式 `linear_fastpath="graph"` capture 固定签名的完整 Linear forward. Graph key 必须覆盖 input layout/dtype/device,kernel/layout/SM/schedule 和 weight/bias identity/version;参数更新,prepack refresh 和 module `_apply()` 必须清缓存. Replay 返回 graph-owned output,只验证顺序 inference. SM89 BF16/FP16 三层 gate 都仅 M1 通过 promotion,M4/M64 被拒绝,所以 `gemm_with_precision(engine="auto")` 和全 shape Linear auto route 保持不变.
- `examples.xqt_models.hunyuan_ocr.optimize_hunyuan_ocr_svd_int4_blocks(...)` 是模型专用示例: 它以 `w4a16_int4` 储存 SVD residual,物化 INT8 MMA residual compute view,然后对外层 `nn.ModuleList` 中每个逻辑 block 分别执行 `torch.compile` 并用真实模型前向 warmup. 该优化是内存中的 block composition,不是单个 fused block kernel,且没有可识别 block 时必须显式失败. `HunyuanOcrTileLangDecodeBlock` 则是独立的实验性 `sm_89` 单 token decode API: 它将量化投影和 TileLang norm,GQA attention,SwiGLU,residual 串为固定 KV 长度的多核 CUDA Graph pipeline,不会自动替换远程代码模型的 `generate`.
- ONNX target 的已知字段统一写在 `targets[*].onnx`: `input_names`, `output_names`, `dynamo`, `validate`, `runtime_diff`, pre-export fusion/lowering 和 ONNX optimization. target `params` 不再承载这些键. `pre_export_lowering` 中当前的 `fp4_weight_only_to_dense_linear` 会复制 export model, 将 `FP4WeightOnlyLinear` materialize 成等价的 dense dequantized `nn.Linear`, 并在 artifact metadata 记录 lowering. 这让通用 ONNX / TensorRT adapter 可消费该模型, 但 resulting artifact 不是 packed-FP4 runtime.
- `targets[*].inference` 是模型侧推理 contract, 与 backend export 配置分离. 它声明 `schema_version`, `family`, `adapter`, `adapter_version`, `inputs`, `outputs`, `config` 和 `metadata`; ONNX model package 生成时自动把它写入 `manifest.json.inference`. `InferenceContract` 只需按模型族声明一次, 不要为每个模型复制 runtime runner.
- 当前 export 会为 ONNX target 额外落一个 `*.xqtpkg/manifest.json` 标准模型包, 并附带 `model/*` 与 `runtime/config.json`. file-based inference 通过 `load_model_package()` / `create_inference_runner()` 只消费该包; 现阶段最小闭环只保证 `ONNX + ONNX Runtime`.
- TensorRT engine-build 的已知字段统一写在 `targets[*].tensorrt`: `onnx_path`, `backend`, `trtexec_path`, `extra_args`, `timeout`, `dry_run`, `performance_thresholds`, `workspace_mib`, `builder_optimization_level`, `timing_cache_path`, `log_level`, `plugin_libraries`, `serialize_plugin_libraries`, `validate_plugin_libraries_loadable` 与 `runtime_benchmark`. loader 明确拒绝 target `params` 中的同名旧键; `params` 只保留其他 export backend 的专有选项. `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口也通过 `tensorrt` 参数构造同一个 typed `StageSpec`.
- OpenVINO target 的已知字段统一写在 `targets[*].openvino`: `onnx_path`, `input_shape`, `dry_run`, `runtime_diff`, `device` 与 `benchmark` (`enabled` / `warmup` / `iterations` / `measure_memory`). loader 明确拒绝 target `params` 中的同名旧键; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口通过 `openvino` 参数构造同一个 typed `StageSpec`. `openvino.onnx_path` 缺失时, export pass 使用同 workflow 的先前 ONNX artifact, 再回退到当前模型转换; `runtime_diff` 只在 materialized IR 与可用 reference output 时执行. `openvino_runtime_layer_report` 的 `runtime_benchmark` 层从 `not_configured` 变为配置化状态 (`not_run_dry_run` / `not_run_missing_ir` / `not_run_missing_dependency` / `configured`), 真实 benchmark 执行仍属目标机可选里程碑.
- TorchExport target 的已知字段统一写在 `targets[*].torch_export`: `strict`, `validate` 与 `runtime_diff`. TorchScript target 的已知字段统一写在 `targets[*].torchscript`: `method`, `check_trace` 与 `runtime_diff`, 其中 `method` 只能是 `trace` 或 `script`. loader 明确拒绝 target `params` 中的同名旧键; `XQTOptimizationSession.export()` 与 `.deploy()` 的单 target 入口分别通过 `torch_export` 与 `torchscript` 参数构造同一个 typed `StageSpec`.
- ExecuTorch target 的已知字段统一写在 `targets[*].executorch`: `dry_run`. ncnn target 的已知字段统一写在 `targets[*].ncnn`: `source_path`, `converter`, `onnx2ncnn_path`, `pnnx_path`, `bin_path`, `extra_args`, `timeout` 与 `dry_run`; `converter` 只能是 `onnx2ncnn` 或 `pnnx`. pnnx 未指定 source 时优先使用同 workflow 的 TorchScript artifact, 再回退 ONNX. MNN target 的已知字段统一写在 `targets[*].mnn`: `source_path`, `converter_path`, `framework`, `extra_args`, `timeout` 与 `dry_run`. QNN target 的已知字段统一写在 `targets[*].qnn`: `source_path`, `converter_path`, `extra_args`, `timeout` 与 `dry_run`; 真实转换依赖目标机 Qualcomm QNN SDK. 四者的 Session 单 target 入口分别通过 `executorch`, `ncnn`, `mnn` 和 `qnn` 参数构造同一个 typed `StageSpec`; dry-run preflight 不要求本机存在 optional package 或 converter executable. mobile 导出的失败原因经 `mobile_export_diagnosis` 结构化 (`dry_run_preflight_only` / `source_artifact_missing` / `converter_executable_missing` / `artifact_not_materialized`), 写入 stage entry 的 `export_readiness` 与失败路径的 `context.metrics`, 见 `xqt/export/mobile.py`.
- materialized deploy runtime handle 的 typed config 位于 `runtime_handle.onnxruntime` 与 `runtime_handle.tensorrt`. 前者声明 ONNX Runtime providers; 后者声明 device 与 runtime plugin libraries. runtime plugin libraries 不从 TensorRT engine-build target 隐式继承, 必须由 deploy runtime config 显式提供.
- `ExportTargetConfig.dynamic_shapes` 是唯一动态 shape 配置. `dynamo=true` 时传给现代 exporter; `dynamo=false` 时必须写 input-name 到 axis-name 的 mapping, 并转换为 legacy ONNX `dynamic_axes`. TensorRT profile 仍必须与实际 ONNX dynamic axes 相容.
- `PrecisionPolicy` 是 module conversion, `xqt.nn` facade runtime intent 与 GEMM 的唯一静态精度 contract. `xqt.kernels.ops._impl.gemm_precision.MatmulPrecisionSpec` 与 `xqt.conversion.MatmulPrecisionSpec` 仅保留为同一类的兼容导入名, 不再维护平行 schema 或双向转换. facade 可额外使用 `auto` 延迟到输入 dtype, 但名称和字段 canonicalization 仍复用该 contract.

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
