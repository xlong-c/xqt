# XQT 摘要

XQT 只关注模型本身.

它负责模型压缩,模型图变换,导出适配,模型误差分析和 benchmark. 它不负责训练,QAT 训练,finetune,distillation,KD recovery,prune recovery,dataset/dataloader,training provider 或 evaluation provider.

当前能力分层:

- 已基本可用:
  - 常规量化与分析: PTQ,QDQ,torchao,calibration,activation drift,layer sensitivity,layer-level weight diff 和权重/激活分布统计.
  - 常规剪枝: unstructured,structured,N:M,block sparse,mask/rewrite.
  - 导出主链路: ONNX,torch.export,TorchScript,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN.
  - 常规分析与 benchmark: output diff,layer analysis,latency,memory,manifest.
- 半可用:
  - TensorRT engine 构建,engine inspector,runtime benchmark.
  - TensorRT 自定义插件 `.so` 加载与 preflight 检查.
  - TileLang attention operator target 已可进入 executor;CPU 路径走 reference fallback,CUDA 路径已有最小真实 TileLang attention kernel,但仍只覆盖受限场景.
  - `backend: pytorch` 下的 `strategy: fp4_weight_only` 已有 group-wise reference Linear weight-only 路径,可执行模块替换,分组 scale 量化和误差验证,但不是高性能 packed kernel 路径.
- 偏实验 / planned:
  - CuTile/CUTLASS/custom CUDA 主要还是 capability,adapter,report 边界.
  - TileLang 上的 FP4/AWQ packed kernel,以及更完整的 AWQ/GPTQ 路径仍以 capability 声明和研究代码为主,未形成高性能执行闭环.

场景 readiness:

| 场景 | 当前状态 | 已验证证据 | 主要缺口 |
| --- | --- | --- | --- |
| FP4 量化 | 半可用 | `pytorch + fp4_weight_only` 已能做 group-wise reference Linear 替换,并有执行测试 | 还没有高性能 packed kernel,也没有完整 AWQ/GPTQ 执行闭环 |
| TileLang megakernel | 半可用 | `attention` target 已进入 executor,能报告 `reference_fallback` / `cuda_tilelang_entry`,并已有 CUDA attention kernel 测试 | 仍只覆盖 attention/fp16/dropout=0/seq_kv>=seq_q,还没有更完整 megakernel 家族 |
| TensorRT + `.so` 插件 | 半可用,接近工程可用 | build / inspect / runtime benchmark / plugin load / preflight 已接通 | 仍缺真实用户插件 ABI 和部署环境级联验证 |
| 常规剪枝 / 误差分析 | 已基本可用 | activation drift,layer sensitivity,layer weight diff,输出/权重分布统计均已有实现和测试 | 更高层 task-level 准确率评估仍需外部评测链路 |

当前核心 API:

- `XQTOptimizationSession`
- `optimize_model`
- `load_optimization_config`
- `OptimizationConfig`
- `OptimizedModelResult`
- `ArtifactManifest`

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
