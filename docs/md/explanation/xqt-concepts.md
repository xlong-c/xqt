# XQT 概念说明

本文解释 `XQT` 的概念边界和理解地图. 它不单独定义新契约.

## 这是什么

`XQT` 是 `XDL` 仓库中的模型压缩, 图变换和部署实验包. 它只处理模型本身, 不接管训练语义.

## 为什么需要

训练侧和部署侧的关注点不同:

- `XDL` 负责训练生命周期和组件组织
- `XQT` 负责模型压缩, 导出适配, 误差分析和 benchmark

把两者分开, 可以避免在部署工具链里重新发明训练循环, 也能避免在训练框架里堆叠过多后端适配逻辑.

## 核心概念

- `session`: 交互式优化编排入口
- `workflow`: YAML 驱动的阶段式优化流程
- `stage`: 一次模型侧变换, 分析或导出动作
- `manifest`: 产物, 指标和 lineage 的统一记录
- `readiness`: 对某个场景是否可用的能力判断
- `backend` (quant): 量化适配路径, 例如 `pytorch`, `torchao`, `onnxruntime_qdq` (**不是** tilelang/svdquant)
- `method` (quant): 量化算法, 例如 `awq`, `gptq`, `svd` (**不是** engine)
- `engine` (operator): XQT 内部 kernel / lowering, 例如 Triton, TileLang, CUTLASS, CuTe DSL
- `backend` (export/deploy): 外部 runtime, 例如 TensorRT, ONNX Runtime, OpenVINO
- `compute_config`: quant→infer 可选计算配置 (精度 + required_capabilities)
- **离线静态权重** (文档优先轴): 权重量化在 quant stage 完成并固化; 写 recipe 时先写这一侧, 再写激活是未量化 / 运行时动态 / 校准静态. 细则见 [xqt-quant.md](xqt-quant.md#2-写量化时的默认表述-离线静态权重优先)
- `semantic replacement`: Python 层以 `Linear`, `Conv`, `Norm`, `Attention`, `FeedForward`, `TransformerBlock` 为单位替换模型语义块
- `wrapper/materialize`: 夹在 facade 和 kernel 之间的边界翻译层, 负责 candidate module, fallback 和 execution metadata

## 与相近概念的区别

- 它不是 trainer
- 它不是 task evaluation 平台
- 它不是 dataset / provider 框架
- 它不是某个后端 runtime 的替代品
- 它不是 TensorRT / ONNX Runtime 这类外部 runtime 的薄 wrapper

## Python-first 推理优化

XQT 的推理优化以 Python API 为主入口. `XQTOptimizationSession`, `xqt.convert(...)`, 当前真实的 `xqt.nn.FeedForward` / `RMSNorm` facade, 以及后续计划中的更多 semantic facade 都应能在 Python 层表达. 性能敏感部分再由内部 engine lowering 到 DSL kernel 或 custom CUDA kernel.

这意味着用户看到的主体是:

```text
XQT model transform
  -> semantic block replacement
  -> operator contract
  -> wrapper / materialize
  -> internal engine lowering
  -> benchmark, report, manifest
```

不是:

```text
用户直接选择一组外部 inference backend 来替换 XQT
```

块级替换和 kernel fusion 不冲突. `FeedForward -> XQTFeedForward`, `Attention -> XQTPagedAttention`, `TransformerBlock -> XQTTransformerBlock` 是 Python 层语义替换; 中间仍需经 wrapper/materialize 把模块语义翻译成 engine 可执行对象; 底层可以是一组 kernel, 也可以是 megakernel. 是否真的合成单 kernel 必须由 capability 和 benchmark 证明.

当前实现状态需要区分:

- `xqt.nn.*` 是 facade, 不是 kernel catalog.
- `wrapper/materialize` 是模块级边界翻译层, 不是临时胶水.
- `kernel` 是 pattern 级执行实现, 不直接理解 `FeedForward` / `TransformerBlock` 这类语义块.

这条边界的正式规则见 [../architecture/xqt-kernel-wrapper-nn-boundary.md](../architecture/xqt-kernel-wrapper-nn-boundary.md).

## 当前能力地图

- 已基本可用: PTQ / QDQ / torchao 量化, 常规剪枝, ONNX / torch.export / TorchScript / TensorRT / OpenVINO / ExecuTorch / ncnn / MNN / QNN 导出 (mobile 目标以 adapter + dry-run preflight + artifact metadata 为主), output diff, layer analysis, latency / memory benchmark, manifest, 自动策略建议 (backend / precision / stage leaderboard / acceptance policy / QuantScheme search), 模型族 smoke 链路 (Transformer / ViT, Detection, LLM, MoE, Diffusion, Multimodal)
- 半可用: TensorRT engine / plugin preflight, TileLang 的受限 kernel target, FP4 packed weight 到 TileLang operator stage 的桥接
- 偏实验: 更完整的 AWQ / GPTQ packed megakernel, 以及更广泛的 engine capability 闭环

当前 `TileLang` 的受限 kernel target 主要覆盖:

- `attention`
- `conv`
- direct FP16/BF16 `linear`
- direct half `LayerNorm`
- `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体

这些路径并不是同等成熟度. `attention` 已进入受限的 CUDA FP16/BF16 TileLang operator coverage,要求 Q/K/V dtype 匹配,`dropout_p=0`,`seq_kv >= seq_q`,且 BF16 `head_dim` 必须 16 对齐. `sm_89` BF16 paired gate 表明收益依赖 shape,所以显式 TileLang 可执行,但 Ada `auto` 仍保留 native SDPA. Direct `linear` 接受匹配的 FP16/BF16 activation,weight,bias 和 output,使用 FP32 accumulator,允许 partial M/N,且 K 必须被 `block_k` 整除;`sm_89 + BF16 + flattened M<=4` 的显式 TileLang 默认使用经 paired gate 验证的 `16x64x32` schedule. FP16 只对 `M<=4,K=N=4096,activation=None` 的 exact signature 使用同一 preset,`N=11008` 或 fused activation 保留 `64x64x64`;后续 `11008 -> 4096` down 与 `4096 -> 11008` up/gate 的 5-seed audit 也因各有一个 seed 未过完整门槛而 `keep_default`. 两种 dtype 都不改变 Linear `auto` 路由. Dequant GEMM 也已进入受限 coverage. `conv` 当前通过 `torch.unfold` / im2col lowered input 加 TileLang half GEMM 执行,尚不是 fully fused conv kernel,且只覆盖 FP16,CUDA,`groups=1` 的路径. Direct half `LayerNorm` 已接入 TileLang `reduce_sum` kernel,限制为 CUDA FP16 和 last-dim normalization. CPU 路径只使用 PyTorch eager fallback.

Triton attention 是独立的 forward-only engine,不是 TileLang attention 的 `auto` 替代. 当前入口要求 contiguous BHSD,匹配 FP16/BF16 Q/K/V,FP32 score/online softmax/output accumulator,`head_dim=16/32/64/128`,`seq_kv >= seq_q`,`dropout_p=0`,并对非方形 causal 使用 lower-right mask;不支持 backward. 在 `sm_89` 上完成 5-shape x 19-candidate sweep 后,只有 `B1,H8,Sq1,Skv1024,D64` 的 FP16/BF16 exact resolver preset 通过 5-seed `9:0` audit,分别为 `16x64/4w/2s` 和 `16x128/4w/2s`. Causal/decode 的初始 paired gate 相对 SDPA 稳定降低 FP16 `22.63%/62.27%` 和 BF16 `21.01%/67.02%`,但 TileLang 仍胜出;small/medium/long prefill 也不支持 Triton route promotion. 后续 exact resolver 对 TileLang/SDPA 的实际 5-seed engine-entry audit 中,两个 dtype 都稳定快于 SDPA,但未在每个 seed 稳定快于 TileLang,因此没有进入 wrapper-stage review. CPU 仍使用 SDPA reference fallback,其他 SM,动态 shape/mask 和 serving 级 KV/cache 不在本覆盖内. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-attention/`.

Attention 的 runtime 结论必须分层阅读. `B1,H8,Sq1,Skv1024,D64,causal` 的 wrapper profile 显示,FP16/BF16 direct Triton kernel 约 `0.018/0.018 ms`,TileLang full MHA 约 `0.104/0.112 ms`,native lower-right full MHA 约 `0.132/0.133 ms`;4 个 projection Linear,2 个 layout copy 和 wrapper 入口占据其余时间. 首次分层 profile 发现 native 非方形 causal 的 `is_causal=True` 是 upper-left 语义,与 TileLang/Triton 的 lower-right contract 不一致;共享 helper 已对非方形调用 `causal_lower_right`,修复后所有路径通过同一 lower-right correctness gate. 这组 sequential wrapper 数据不能取代 5-seed `9 x 31` engine gate,所以 Triton 仍是显式 engine,`attention_fastpath="auto"` 不变.

ConvRot/SVDQuant 的 SM89 native CUDA 路径是专用动态融合,不是把 rotation,quant 和 LoRA 以 Python eager 串起来. ConvRot W8A8 的前端 kernel 直接从 source activation 产生 Hadamard-rotated INT8 activation 和 per-token scale,随后进入 packed W8A8 GEMM. SVDQuant W4A4 使用 Nunchaku 的两阶段 dynamic LoRA 结构:FUSE_DOWN 同时产生 quantized activation 和 LoRA-down activation,FUSE_UP 在 W4A4 GEMM epilogue 中加入 LoRA up 与 bias;W8A8 采用对应的 INT8 两阶段实现. C++ bound runner 预绑定 packed tensor,workspace 和 stream-aware cache,稳态只传 activation,source tensor version 变化时重新 pack. 这些路径没有融合 norm,`norm_fused=false`;旧的显式 `ConvRotNormInt8Linear` 仍是独立 opt-in 能力,不能和动态 native fastpath 混写. 当前证据只覆盖 RTX 4070 Ti SUPER `sm_89` 和已列 dtype/shape,其他 SM 或不支持签名显式回退. 证据见 `research/xqt-gemm/artifacts/2026-08-10-sm89-convrot-svdq-fusion/`.

ConvRot W4A4 的 rowwise native path 是独立的两 kernel contract:第一个 warp 内 regular-Hadamard FHT 直接写 row-major signed INT4 activation 和 FP32 row scale,不物化 rotated FP16/BF16 tensor;第二个 CUTLASS `s4 x s4 -> s32` Tensor Core GEMM 在 epilogue 内依次融合 activation scale,weight scale 和 bias. C++ bound runner 按 `(rows,CUDA stream)` 缓存 activation workspace,支持高维输入和 non-contiguous 输入的显式 contiguous materialization,并按 source weight/scale/bias version 失效重打包. `w4a4_runtime_backend` 的 `auto` 语义只识别 whole-row scale artifact,不能把 grouped scale artifact 静默改成 rowwise;显式 `rowwise` 允许运行时重打包,但相对 grouped dense artifact 的 relative RMSE 约为 `0.188-0.205`,高于 Nunchaku grouped path 的 `0.122-0.123`. 本能力不融合 norm,metadata 保持 `norm_fused=false`;不满足 SM89,dtype,shape,dynamic-scale 或 layout contract 时显式回退. 证据见 `artifacts/xqt/benchmarks/convrot_w4a4_sm89/summary.json` 和 `artifacts/xqt/profiling/convrot_w4a4_sm89/`,逐步案例见 [convrot-w4a4-sm89-optimization.md](convrot-w4a4-sm89-optimization.md).

Triton dense FP16/BF16 GEMM 的 FP32 epilogue 已覆盖 bias,GELU 和 SiLU. BF16
no-override 入口在 `sm_89` 上为五个已 profile 的
`(M,N,K,bias,activation)` exact signature 使用 evidence-backed schedule preset.
FP16 resolver 进一步把 `transpose_b` 纳入 exact key:M1/M4/M8/M64 preset 同时
覆盖 K,N 和 N,K,M256 GELU 只为证据充分的 K,N layout 启用 preset. 六个调度
字段始终逐项显式优先;其他 shape/SM 保留旧默认. N,K weight 由逻辑 stride 交换
直接传入 kernel,不再 per-call materialize transpose;无状态 dispatcher 仍不保存
tensor 或 hidden weight cache,默认使用零额外权重内存的 `transpose_stride`.

Stateful Triton Linear materializer 另提供显式 `prepacked_kn` 和
`linear_fastpath="graph"`. Graph 只复制动态 activation,固定 weight,bias,layout 和
schedule,按 tensor contract 与参数 identity/version 缓存,并在参数更新,prepack
刷新或 `.to()` 后失效. Replay 返回 graph-owned output. SM89 BF16/FP16 三层 gate
都只有 M1 通过 `min_speedup=1.03`;M4/M64 均被 operator-stage speedup 门槛拒绝.
两种 dtype 的 prepack 与 stride graph 都没有稳定差异. 因此这些能力是显式
materialization 选项,`gemm_with_precision(engine="auto")` 和全 shape Linear
auto route 都不改变.

## 硬件与精度速记

XQT 做 quant / operator / export 路由时, 不要先问 "哪种位宽最好", 要先问 "这台机器是哪一代".

- `INT8`: 兼容性最好, 仍然是最稳的通用部署位宽.
- `FP8`: `Ada`, `Hopper`, `CDNA3`, `RDNA4` 开始进入主流高端推理路径.
- `FP4 / FP6`: `Blackwell` 和 `CDNA4` 才是原生主战场. 更老平台上更常见的是 packed storage, bridge format 或实验路径, 不是原生 MMA 主路径.
- `FP16 / BF16`: 是高兼容 fallback 和数值兜底位, 不是通用意义上的 "更高量化优先级".
- 更完整的 `NVIDIA` / `AMD` 代际, `MMA` 指令族和数据类型总表见 [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md).

## 常见误区

- 不要把 `XQT` 当成训练恢复或 QAT 工具
- 不要把 profiler 诊断结果和 benchmark 指标混为一谈
- 不要把 `TileLang available` 理解成所有 pattern 都已经是同等成熟的通用 executor
- 不要把 `triton` / `tilelang` / `cute_dsl` / `custom_cuda` 写成和 TensorRT / ONNX Runtime 并列的外部 backend; 它们是 XQT 内部 **engine**
- 不要把 `awq` / `gptq` / `svd` 当成 engine 或 quant backend; 它们是 quant **method**, 配置写 `backend=pytorch` + `method=...`
- 不要写 `quant.params.backend=tilelang` 或 `=svdquant` (已禁止)
- 不要用一句"动态量化 / 静态量化"概括整条 quant 路径; 先写 **离线静态权重**, 再写激活时机 (见 [xqt-quant.md](xqt-quant.md#2-写量化时的默认表述-离线静态权重优先))
- 不要把 1~8 的长期指导写成全部完成; 当前只是配置单轨化主路径完成, God module 拆分, contract 层和能力面收敛仍未完成
- 不要在 recipe 里声明 dataset 来源

## 继续阅读

- [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md) - engine / quant 边界规则
- [../architecture/xqt-kernel-wrapper-nn-boundary.md](../architecture/xqt-kernel-wrapper-nn-boundary.md) - `kernel` / `wrapper` / `xqt.nn` 分层
- [xqt-engines.md](xqt-engines.md) - operator engine 能力矩阵
- [xqt-quant.md](xqt-quant.md) - quant backend / method / strategy
- [xqt-inference.md](xqt-inference.md) - 推理路径与模型包
- [../architecture/xqt.md](../architecture/xqt.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- [../XQT_SUMMARY.md](../XQT_SUMMARY.md) - 兼容摘要入口
