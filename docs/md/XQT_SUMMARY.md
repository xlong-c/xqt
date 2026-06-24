# XQT 摘要

XQT 只关注模型本身.

它负责模型压缩,模型图变换,导出适配,模型误差分析和 benchmark. 它不负责训练,QAT 训练,finetune,distillation,KD recovery,prune recovery,dataset/dataloader,training provider 或 evaluation provider.

当前能力分层:

- 可运行 readiness audit:
  - `assess_xqt_readiness()` 会聚合 FP4 + TileLang,TensorRT `.so` plugin,剪枝/量化误差分析三个目标场景的 `ready` / `partial` / `blocked` 状态.
  - 报告会包含 `status_counts`,`required_action_count` 和每个场景的 `required_actions` / `distance_to_ready`,用于直接回答"还差多少".
  - `XQTReadinessReport.write_artifacts()` 可同时写出 JSON 和 Markdown readiness 产物;`add_to_manifest()` 可把 readiness 结论和产物记录附加到 `ArtifactManifest`.
  - `XQTOptimizationSession.readiness()` 可在交互式 session 中一行写出 readiness metrics,artifacts 和 manifest.
  - 默认只做快速能力和环境检查;需要 TileLang 编译或 CUDA runtime evidence 时,显式设置 `run_tilelang_probe=True`.
- 已基本可用:
  - 常规量化与分析: PTQ,QDQ,torchao,calibration,activation drift,layer sensitivity,layer-level weight diff 和权重/激活分布统计;`quant_layer_statistics_workflow.yaml` 已提供 `quant -> analyze` workflow.
  - 常规剪枝: unstructured,structured,N:M,block sparse,mask/rewrite.
  - 导出主链路: ONNX,torch.export,TorchScript,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN.
  - 常规分析与 benchmark: output diff,layer analysis,latency,memory,manifest.
- 半可用:
  - TensorRT engine 构建,engine inspector,runtime benchmark.
  - TensorRT 自定义插件 `.so` 加载与 preflight 检查;`validate_tensorrt_plugin_libraries()` 可独立校验 `.so` 是否存在和能否被 `ctypes` 真实加载.
  - TileLang `attention` 和 `dequant_gemm_epilogue` operator target 已可进入 executor;CPU 路径走 reference fallback,CUDA 路径已有最小真实 TileLang attention,dequant GEMM,以及 packed FP4 单 kernel unpack/dequant/GEMM/bias/activation core,但仍只覆盖受限场景.
  - `backend: pytorch` 下的 `strategy: fp4_weight_only` 已有 group-wise reference Linear weight-only 路径,可执行模块替换,分组 scale 量化和误差验证. FP4 在这里是 data format / packed storage,不是执行 backend;执行后端仍是 PyTorch eager,TileLang 或 TensorRT plugin. 当前还能通过最小桥接进入 TileLang `dequant_gemm_epilogue` operator stage,并提供 YAML workflow 示例;operator report 会标记 packed weight 被消费,CUDA 路径的 `unpack_stage=tilelang_fused_gemm_kernel` 和 `epilogue_stage=tilelang_fused_bias_activation`;`validate_tilelang_packed_fp4_fused_gemm()` 已提供 no-CUDA skipped,compile-only 和真 CUDA correctness + latency 的统一结果对象;仍需真实 CUDA runtime 数值和性能验证.
- 偏实验 / planned:
  - CuTile/CUTLASS/custom CUDA 主要还是 capability,adapter,report 边界.
  - TileLang 上 fused FP4/AWQ packed megakernel,以及更完整的 AWQ/GPTQ 路径仍以 capability 声明和研究代码为主,未形成高性能执行闭环.

场景 readiness:

| 场景 | 当前状态 | 已验证证据 | 主要缺口 |
| --- | --- | --- | --- |
| FP4 量化 | 半可用 | `pytorch + fp4_weight_only` 已能做 group-wise reference Linear 替换,并有执行测试;`ReferenceFP4Linear` 已可桥接到 TileLang `dequant_gemm_epilogue` operator stage;`fp4_tilelang_workflow.yaml` 已可通过 `optimize_model()` 跑通;report 明确标记 packed weight 被消费,CUDA 路径的 unpack/dequant/GEMM/bias/activation 已推进到单 TileLang kernel;`validate_tilelang_packed_fp4_fused_gemm()` 已覆盖 no-CUDA skipped 和 `target_arch=sm_80` compile-only 探测 | 还没有真实 CUDA runtime 数值和性能验证,也没有完整 AWQ/GPTQ 执行闭环 |
| TileLang megakernel | 半可用 | `attention` 和 `dequant_gemm_epilogue` target 已进入 executor,能报告 `reference_fallback` / `cuda_tilelang_entry`,并已有对应 CUDA kernel 测试;packed FP4 entry 已能消费 packed weight 后进入 fused unpack/dequant/GEMM/bias/activation core;`tilelang_validation.py` 可在真 CUDA 上产出 correctness + latency + speedup 结构化结果 | 仍只覆盖 attention/fp16/dropout=0/seq_kv>=seq_q 和带最小 block 对齐约束的 dequant GEMM;packed FP4 megakernel 还缺真实 CUDA runtime 验证和性能基准 |
| TensorRT + `.so` 插件 | 半可用,接近工程可用 | build / inspect / runtime benchmark / plugin load / preflight 已接通;`validate_plugin_libraries_loadable=true` 可验证 `.so` loadability | 仍缺真实用户插件 ABI 和部署环境级联验证 |
| 常规剪枝 / 误差分析 | 已基本可用 | activation drift,layer sensitivity,layer weight diff,输出/权重分布统计均已有实现和测试;`quant_layer_statistics_workflow.yaml` 已可通过 `optimize_model()` 跑通并输出 `layer_statistics` | 更高层 task-level 准确率评估仍需外部评测链路 |

当前核心 API:

- `XQTOptimizationSession`
- `optimize_model`
- `load_optimization_config`
- `assess_xqt_readiness`
- `OptimizationConfig`
- `OptimizedModelResult`
- `ArtifactManifest`
- `XQTReadinessReport`

旧的 `XQTConfig` pass recipe,`load_xqt_config`,`run_xqt_recipe` 和 `preflight_xqt_config` 只保留为内部实现,不再作为 `xqt` 顶层 API.

当前包模块:

- `core`,`model`,`integrations`,`pipeline`,`workflows`,`quant`,`prune`,`operator_opt`,`export`,`eval`,`benchmark`

不属于 XQT 的内容:

- `distill/`
- `diffusion_distill/`
- training/evaluation provider
- dataset/dataloader 构建
- `finetune`/`distill` stage
- `eval`/`runtime_eval` stage
- QAT 训练和 recovery training
