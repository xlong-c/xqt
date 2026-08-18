# XQT 模型优化工具链

本文保留为 `XQT` 的兼容长期事实源. 新的 Markdown 主导航已经迁到 [index.md](index.md), 并按架构 / 说明 / 使用三层组织.

XQT 只关注模型本身. 它接收 PyTorch 模型,checkpoint 或导出产物,执行模型侧压缩,图变换,导出适配,误差分析和 benchmark. 训练,QAT,finetune,distillation,recovery,dataset / dataloader,training provider 和 evaluation provider 不属于 XQT.

需要梯度更新的流程归 XDL 或第三方训练工具,再把训练后的模型或 checkpoint 交给 XQT.

在本仓库内, XQT 是唯一推理优化主体. Python API 是主入口: `XQTOptimizationSession`, `xqt.convert(...)`, 当前真实的 `xqt.nn.*` semantic facade 用来表达模型替换, 算子 contract, runtime intent 和 benchmark/report. `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `custom_cuda` 是 XQT 内部 engine, 不是和 TensorRT / ONNX Runtime / OpenVINO 并列的外部 backend.

当前 `XQTOptimizationSession` 内部已经维护正式 `SessionStage` 图, `StagePayload` typed payload, transform-side provider 和 `StageComparison` helper. 详细协议以 [architecture/xqt.md](architecture/xqt.md) 和 [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md) 为准.

## 1. 项目定位

本章节保留为兼容锚点. `XQT` 的系统定位和模型侧边界已经迁到:

- [architecture/xqt.md](architecture/xqt.md)
- [explanation/xqt-concepts.md](explanation/xqt-concepts.md)

如果你要更新 `XQT` 做什么, 不做什么, 主链路或模型侧职责, 优先更新这些 canonical 页面.

## 2. 配置方式

本章节保留为兼容锚点. `XQT` 的使用入口和工作流已经迁到:

- [usage/xqt-workflows.md](usage/xqt-workflows.md)
- [architecture/xqt.md](architecture/xqt.md)

如果你要更新 `session` / YAML workflow 的主路径或配置边界, 优先更新这些 canonical 页面.

## 3. 当前可用能力

本章节保留为兼容锚点. `XQT` 的能力地图和概念说明已经迁到:

- [explanation/xqt-concepts.md](explanation/xqt-concepts.md)

如果你要更新当前能力范围, 半可用 / 实验性状态或能力地图, 优先更新上面的说明层页面.

当前 `TileLang` 的受限 kernel target 已覆盖 `attention`, `conv`, direct FP16/BF16 `linear`, `linear_marlin`, direct half `LayerNorm`, `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体. 这些 pattern 不是同等成熟度: `attention` 已有受限 CUDA FP16/BF16 kernel 入口,要求 Q/K/V dtype 匹配,`dropout_p=0`,`seq_kv >= seq_q`,且 BF16 `head_dim` 必须 16 对齐;direct dense `linear` 接受匹配的 FP16/BF16 activation,weight,bias 和 output,使用 FP32 accumulator,允许 partial M/N,但 K 必须能被 `block_k` 整除. `conv` 当前是 `torch.unfold` / im2col 加 TileLang half GEMM 的 lowering 路径而不是 fully fused conv,direct half `LayerNorm` 已接入 TileLang `reduce_sum` kernel 且限制为 last-dim FP16. CPU 路径只使用 PyTorch eager fallback.

BF16 Attention 已在 RTX 4070 Ti SUPER `sm_89` 上完成 5-shape x 9-candidate sweep 和 9 x 31 paired gate. 默认仍为 `64x64/128 threads/2 stages`;相对 BF16 SDPA,medium/causal/decode 稳定降低 `11.85%/38.87%/42.15%`,small 稳定慢 `8.33%`,long 的 `2.07%` 差距低于 promotion 门槛. 因此显式 TileLang BF16 可执行,但 Ada `attention_fastpath="auto"` 仍选择 native SDPA,不增加仅由 synthetic kernel matrix 驱动的 shape heuristic. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-attention-bf16/`.

新增 Triton forward-only attention engine,契约为 contiguous BHSD,匹配 FP16/BF16 Q/K/V,FP32 score/online softmax/output accumulator,`head_dim=16/32/64/128`,`seq_kv >= seq_q`,`dropout_p=0`,非方形 causal 使用 lower-right semantics,不支持 backward. SM89 5-shape x 19-candidate sweep 的 correctness 最大绝对误差为 `0.00390625`;只为 decode 增加两个 exact resolver preset:FP16 `16x64/4 warps/2 stages`,BF16 `16x128/4 warps/2 stages`. 五个独立 seed 的 resolver-vs-default audit 均为 `9:0`;long prefill 的 stages=3 跨 seed 有低于 3% 的 case,因此撤回. 初始 paired gate 中 Triton 相对 SDPA 的 causal/decode latency 分别降低 FP16 `22.63%/62.27%` 和 BF16 `21.01%/67.02%`,但 TileLang 仍快 `48.53%/13.58%` 和 `43.75%/10.00%`;small/medium/long prefill 不进入 Triton 默认路由. 后续 actual exact resolver 的 5-seed engine-entry audit 也只稳定胜过 SDPA:FP16/BF16 对 SDPA gap 为 `35.78-40.40%`/`40.09-45.86%`,而对 TileLang 不满足全 seed promotion 条件. 因此 Triton attention 目前是显式 engine,`attention_fastpath="auto"` 和既有 TileLang/native SDPA 路由保持不变. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-attention/`.

wrapper/runtime 分层 profile 进一步确认 direct attention kernel 不是完整 MHA 的主要 latency 来源:在 `B1,H8,Sq1,Skv1024,D64,causal` 上,FP16/BF16 TileLang full forward 为约 `0.104/0.112 ms`,native lower-right full forward 为约 `0.132/0.133 ms`,而 projection,layout copy 和 wrapper/module 入口占据其余时间. 首次 profile 发现 native 非方形 causal 误用 upper-left SDPA mask,已改为共享 helper 显式调用 `causal_lower_right`;修复后 native,TileLang 和 Triton 全部通过 lower-right correctness. 该 profile 是 sequential 分层证据,不覆盖 5-seed `9 x 31` paired engine gate,所以不改变 `attention_fastpath="auto"`.

ConvRot/SVDQuant 另新增 SM89 专用 native CUDA 动态融合路径,不依赖 Python 算子组合或 `torch.compile`. ConvRot W8A8 在同一前端 kernel 内完成 regular-Hadamard rotation 和 dynamic per-token INT8 quant,再由同一次 C++ dispatch 进入 packed W8A8 GEMM. SVDQuant W4A4 直接采用 Nunchaku 两阶段 FUSE_DOWN/FUSE_UP:第一阶段融合 dynamic W4 activation quant 与 LoRA down,第二阶段融合 W4A4 GEMM,LoRA up 和 bias;W8A8 路径对应融合 dynamic INT8 quant + LoRA down 和 INT8 GEMM + LoRA up/bias. RTX 4070 Ti SUPER `sm_89` 的 3-shape CUDA-event gate 中,ConvRot W8A8 相对同契约 split baseline 为 `1.51x-1.91x`,SVDQuant W4A4 为 `1.44x-1.73x`,SVDQuant W8A8 为 `1.14x-1.19x`;W4A4 的 C++ bound runner 与 direct Nunchaku-fragment floor 的差异为 `-2.52%` 到 `+0.17%`,完整 wrapper 与 strict floor 的差异不超过 `2.22%`. 证据见 `research/xqt-gemm/artifacts/2026-08-10-sm89-convrot-svdq-fusion/`.

2026-08-18 的 quant/runtime 解环后,`Int8MmaLinear` 与 `W4StorageInt8MmaLinear` 的 quant 产物是 `xqt.contracts` reference storage shell;同名 runtime execution view 通过 `from_storage()` 显式 materialize. 上述 ConvRot CUDA 数据仍是已验证 kernel 证据,但 quant artifact 不再直接执行 runtime fastpath;ConvRot 专用 execution view 完成迁移前,通用 quant/workflow 路径只声明 reference forward,不能把历史 kernel gate 写成当前自动路由收益.

FLUX.2 Klein 4B transformer 现在有独立的 ConvRot W8A8 入口:`quantize_flux2_klein_bf16_transformer_to_convrot_int8(...)`,pipeline/load/inference helper 和 paired benchmark helper. 默认策略与 `ComfyUI-INT8-Fast` 对齐,使用 `rot_size=256`,dynamic per-token activation scale,`M < 17` 的 BF16 fallback,并排除 `img_in`,`txt_in`,`time_in`,`guidance_in` 及 modulation projections. Klein helper 默认 `engine="triton"`;通用 `engine="auto"` 仍可显式选择 XQT SM89 native path. XQT Triton W8A8 已支持 rowwise activation scale,并在 BF16/FP16 输入上保持低精度 rotation. 当前只有 synthetic `K=N=3072` operator proxy 对官方 Triton 做过 paired gate;完整 Klein 4B 权重未在本机缓存,因此不能把 proxy 结果写成完整 transformer 或端到端 pipeline parity. 复跑入口见 `research/xqt-gemm/bench_sm89_flux2_klein_convrot_w8a8.py`.

SVDQuant W4A4 现在还提供两个受限的推理融合入口. 第一,`SVDQuantLinear.set_fused_norm(...)` 可把 RMSNorm row scale 接入 main/small-N native runner;在相同 GEMM schedule,相同语义的独立 runner 对比中,fused 相对 standalone Triton RMSNorm + bound runner 的收益为 `0.973x-1.120x`,即依 shape/dtype 从 `-2.7%` 到 `+12.0%`,不能把旧 block benchmark 中手写多 kernel RMSNorm 的成本解释为 norm fusion 单独带来 `1.7x-2.9x`. 该入口目前需要显式接线,尚未证明真实 transformer/model 自动 materialization. 第二,经典 Diffusers `FeedForward` 的 `gelu-approximate`,零 dropout,双 `SVDQuantLinear` 结构可通过 `materialize_svd_for_inference(..., fuse_gelu_mlp=True)` 显式替换为 `SVDQuantGeluMLP`. 固定官方 Nunchaku wheel `1.3.0.dev20260306+cu13.0torch2.12` 下,相同 packed BN128 权重的 `NunchakuFeedForward` module 与 XQT materialized module 在 `M={64,256,1024}`,FP16/BF16 六个 case 的延迟比为 `0.98415x-1.00793x`,全部通过 `<=1.025x` gate. 该结论只覆盖 tanh-approximate GELU MLP,不覆盖 transformer block,完整模型或 FLUX.2 Klein;Klein FFN 是 `linear_in -> SwiGLU -> linear_out`,需要独立 fused SwiGLU 路径. 证据见 R-037/R-038,`research/xqt-gemm/artifacts/2026-08-13-sm89-svdq-norm-runner/result.json` 和 `research/xqt-gemm/artifacts/2026-08-13-sm89-svdq-official-mlp-parity-v1/result.json`.

ConvRot W4A4 另有 SM89 专用 `warp-FHT -> rowwise INT4 -> CUTLASS W4A4` 路径. 通过 `w4a4_runtime_backend=auto|rowwise|nunchaku|reference` 显式选择 contract:`auto` 只有在 artifact 的 `group_size == padded_input_features` 时才解析为 rowwise,grouped artifact 保留 Nunchaku 语义;`rowwise` 显式请求 warp-FHT rowwise path,不支持时按既有 fallback 顺序退回 Nunchaku 或 reference;`nunchaku` 禁止 rowwise;`reference` 禁止所有 native W4A4. rowwise 只覆盖 `sm_89`,FP16/BF16,dynamic activation scale,未旋转输入,`rot_size=256`,输入特征为 `1024` 或 `2048` 的倍数且不超过 `32768`,输出特征为 8 的倍数. RTX 4070 Ti SUPER `sm_89` 上 12 个 `M={64,256,1024},K=N={1024,2048}` case 的完整 wrapper 延迟为 `13.297-26.766 us`;相对 `comfy-kitchen 0.2.28` 官方 CUDA 实现为 `2.360x-3.056x`(中位 `2.910x`),相对 Nunchaku bound 为 `1.175x-1.586x`,相对显式 split rotation 为 `1.875x-2.512x`,wrapper/bound floor 为 `0.995x-1.083x`. 官方对照使用相同 packed signed INT4 weight,row scale 和 bias,不是 bitwise 等价声明. 证据见 `artifacts/xqt/benchmarks/convrot_w4a4_sm89/summary.json` 和 `artifacts/xqt/profiling/convrot_w4a4_sm89/`,逐步实现说明见 [explanation/convrot-w4a4-sm89-optimization.md](explanation/convrot-w4a4-sm89-optimization.md).

BF16 Linear 已在同一 `sm_89` 设备上完成 7-shape x 10-candidate sweep,FP32 Linear/activation 后单次 cast 到 BF16 的数值 reference,9 x 31 paired gate,torch.profiler 和独立进程 Nsight Systems. `M=1/4,N=4096,K=4096` 的 `16x64x32/128 threads/2 stages` 相对原 `64x64x64` 默认均 `9:0` 稳定胜出,paired gap 分别为 `84.85%/81.53%`;相对真实 `gemm_with_precision(engine="torch")` dispatcher 的 paired gap 为 `6.62%/34.43%`. 因此显式 TileLang direct Linear 在 `sm_89 + BF16 + flattened M<=4` 且没有显式 block override 时使用该 preset. Wrapper gate 只有 `M=1` 达到 `3.35%`,`M=4` 仅 `0.93%` 且归为 noise-equivalent,更大 shape 也存在 native 稳定胜出,所以 `gemm_with_precision(engine="auto")` 和 `linear_runtime="auto"` 都保持原路由. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-linear-bf16/`.

FP16 direct TileLang Linear 进一步完成 4-shape x 10-candidate sweep 和 5-seed 独立审计. 只有 `sm_89 + M<=4 + K=N=4096 + activation=None` 使用 `16x64x32/128 threads/2 stages`;主 sweep 中 M1/M4 相对 `64x64x64` 默认均为 `9:0`,paired gap 为 `78.90%/69.23%`,生产 resolver 与显式候选差距均低于 1%. 五个独立 seed 上两个 shape 都是 `9:0`,correctness 最大绝对误差不超过 `0.125`. `N=11008` 的 bias/SiLU shape 保留 `64x64x64`,与 resolver 的 paired gap 只有 `0.40-1.00%`. Nsight Systems 记录每个 measured range 20 次单 kernel launch;M1 默认/候选/Triton kernel median 为 `34.961/15.094/13.759 us`,M4 为 `35.064/15.606/14.785 us`. Memory summary 为空且 NCU 为 `counter_permission_denied`,因此不推断带宽,occupancy 或 stall. 这次晋级只改变显式 TileLang schedule resolver,不改变 Triton route,`gemm_with_precision(engine="auto")` 或 Linear `auto`. 证据见 `research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16/`.

常见 FP16 MLP decode shape 另完成独立负向 gate:`K=11008,N=4096` down projection 和 `K=4096,N=11008` up/gate projection 各覆盖 `M=1/4`. Down M1/M4 的 3-stage candidate 在主 sweep 相对 2-stage default 以 `9:0`,gap `4.09%/3.99%` 胜出,但 5-seed audit 分别有一个 seed 未通过 7/9 round 或 3% gap 门槛;up/gate 的 1-stage candidate gap 只有 `0.69%/1.32%`. 因此最终 `keep_default`,生产 resolver 仍不覆盖这两组 MLP shape. Nsight Systems 只支持 modest kernel-time difference 和相同 launch count;NCU 仍为 `counter_permission_denied`. 证据见 `research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16-mlp/`.

Triton dense BF16 GEMM 的 bias/GELU/SiLU epilogue 已改为 FP32 执行并只在 store 时 cast output,修复 SiLU 的 BF16 `tl.sigmoid` codegen failure,同时把 `M256,N4096,K4096,GELU` 对 FP32-cast reference 的 `mean_abs` 从 `0.0357372` 降到 `0.0001298454`. 后续 SM89 schedule sweep 覆盖 5 shape x 22 candidate,并为五个 exact `(M,N,K,bias,activation)` signature 增加 no-override BF16 preset. Actual resolver 相对旧 `128x128x32/group_m=8/4w/3s` 默认的 9 x 31 paired gap 为 `541.80%/535.56%/39.39%/63.05%/5.31%`,全部 `9:0`;M1/M4/M8 同时稳定胜过 direct TileLang 和 Torch,M64/M256 不支持通用 route promotion. 每个调度字段的显式值逐项优先,其他 BF16 shape/SM 保留旧默认. N,K weight 路径现已通过逻辑 stride 交换直接调用 Triton,不再每次执行 `weight.t().contiguous()`;相对 legacy per-call materialization 的 5-shape paired gap 为 `1707.65%/1573.68%/261.18%/31.42%/222.83%`,均 `9:0`. 无状态 dispatcher 仍不保存 tensor 或 hidden weight cache,默认使用零额外权重内存的 `transpose_stride`. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-epilogue/`,`research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-schedule/` 和 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-weight-layout/`.

SM89 FP16 schedule sweep 另覆盖 5 shape x 28 candidate 和 K,N/N,K 两种 steady-state layout. Resolver exact key 增加 `transpose_b`:M1/M4/M8/M64 两种 layout 都使用各自 preset;M256 GELU 只有 K,N 使用 `64x64x32/group_m=8/4w/3s`,N,K 因候选仅快 `1.90%` 而保留旧默认. K,N resolver 相对旧默认的 paired gap 为 `530.30%/527.80%/40.46%/71.49%/4.71%`;N,K 为 `563.45%/519.08%/8.33%/73.58%/0.14%`,最后一项归为 noise-equivalent. M8/M256 的 K,N 相对 N,K 稳定快 `93.69%/8.40%`,M1/M4/M64 layout 差异低于 3%. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-fp16-schedule/`.

Triton operator materializer 现可把 direct `nn.Linear` 包装为 FP16/BF16 `_TritonLinearWrapper`. 默认 `weight_layout="transpose_stride"` 和 `linear_fastpath="eager"`;显式 `prepacked_kn` 保存非持久化 K,N 派生权重,显式 `linear_fastpath="graph"` capture activation -> Triton Linear -> graph-owned output 的固定签名分支. Graph key 覆盖输入 layout/dtype/device,kernel/layout/SM/schedule 和 weight/bias identity/version;参数更新,prepack refresh 和 module `_apply()` 会清空旧 graph,capture/replay 失败会记录原因并退回 eager Triton. SM89 BF16 三层 gate 中,M1 operator-stage `2.257x` applied,M4/M64 仅 `0.619x/0.357x`;FP16 中 M1 为 `2.166x`,M4/M64 仅 `0.552x/0.392x`. 两种 dtype 都只有 M1 在 wrapper,完整 block 和 `min_speedup=1.03` operator-stage 三层一致通过. Graph prepack 与 graph stride 在两轮三组 shape 中都 noise-equivalent,却额外复制 `32/32/2 MiB` weight,因此 prepack 和 graph 都不成为全 shape 默认,`engine="auto"` 保持不变. Replay output 会被后续 replay 覆盖,保留时必须 clone. 证据见 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-linear-wrapper/` 和 `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-fp16-linear-wrapper/`.

`KvScaleAttention` 另有 opt-in TileLang KV-int8 fastpath: 模块只保存一个 FP16 packed QKV projection 参数源;单次 QKV GEMM 后,双输出 quantize-layout kernel 直接从 packed tensor 的 K/V thirds 写成 INT8 BHSD,K/V per-tensor FP32 scale 保持 device-resident;attention kernel 从 Q third 按 head slice 读取并输出 BSI,在 mainloop 内 dequant K/V,不物化 Q/K/V 或 output layout copy. Report 记录 `projection_mode="packed_qkv"` 和两级 `selected_kernels`. Eager full-module steady-state 为 packed QKV GEMM + quantize-layout + attention + output GEMM 共 4 个 GPU launch.

`attention_fastpath="graph"` 与 `preferred_kernel` 独立,会 capture 完整 packed full forward,每次 replay 只复制动态输入. Graph cache key 覆盖 shape,stride,dtype,device,causal,dropout,head/quant/tile 参数,实际 compute capability 和参数 storage;`.to()` 会清空旧 graph. Report 额外记录 `selected_fastpath` 及 `cuda_graph.state/reason/cache_size/output_storage`. 该路径已在 `sm_89` 的 4 个 seq/causal shape 上通过 bitwise correctness 和 9 x 31 paired gate,相对 eager 降低 `37.46-69.09%`,均 `9:0`;`torch.profiler`/Nsight Systems 证明主机提交从每次 4 个 kernel launch 变为 1 次 D2D input copy + 1 次 graph launch. Replay 输出使用 graph-owned storage,下一次 replay 会覆盖旧 view,需要保留时必须 clone. 这仍是固定签名,顺序调用的模型侧 attention 实体,不表示 XQT 实现了 paged KV cache,cache eviction,continuous batching 或 serving scheduler.

当前 `CuTile` 的 kernel catalog 已对齐上述 `TileLang` pattern, 即 `attention`, `conv`, `linear`, `norm`, `dense_linear_epilogue`, `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体, 并保留原有 `bias_silu` pointwise scaffold. `CuTile` 仍是 metadata-first / reference-guarded engine: XQT 记录 artifact 和 capability, 并通过 `cuda.tile` 或兼容的 `cutile` 模块探测运行时; 内置 executor 可 materialize 线性 / dequant GEMM 的 reference-guarded 推理 wrapper, 但不把它当作完整通用执行器.

当前 `CuTe DSL` engine adapter 覆盖 `gemm_epilogue` 和 `grouped_gemm` metadata. 内置 executor 可把 NVFP4 dense-cache bridge 路由到 `gemm_epilogue` reference-guarded 推理 wrapper; packed NVFP4 权重不会由 CuTe DSL 直接消费.

`PrecisionPolicy` 是 `xqt.convert(...)`, `xqt.nn` facade runtime intent 和 `gemm_with_precision(...)` 的统一静态精度 contract. GEMM 入口中的 `MatmulPrecisionSpec` 只保留为该类的 identity alias; role/mapping canonicalization 也由 `PrecisionPolicy` 提供. facade 的 `auto` 仍仅表示根据输入 dtype 延迟决策, 不构成第二套静态 schema.

新增的轻量算子转换入口使用 `xqt.convert(...)` 和当前真实的 `xqt.nn.*` facade. `Linear` / `Conv2d` / `LayerNorm` 是保留 PyTorch module/state_dict 语义并显式承载 engine 与 precision runtime intent 的 `torch.nn` 子类, 但不应写成已完成的完整 block-level lowering. `FeedForward` / `RMSNorm` / `Attention` / `TransformerBlock` 也是 semantic facade, 但 facade 不直接等于某个 kernel pattern. 这条入口面向单模块或小范围 operator conversion, 对外保持函数形态, 对内通过状态化 converter 完成 contract lowering 和 engine materialization. 其中 `xqt.nn.FeedForward` 采用固定 FFN 骨架 `norm? -> project-in -> activation/gate -> dropout? -> project-out`, 把 `norm`, `activation` 和 `fusion` 显式暴露为顶层配置, 并在模块内部优先复用现有 Triton GEMM epilogue 与 pointwise fastpath. 当前 `xqt.convert(feedforward, engine=..., policy=..., projection_policies=...)` 会把 `PrecisionPolicy` 注入 FFN runtime 配置: `policy` 作为整个 FFN 的默认精度意图, `projection_policies` 可进一步分别覆写 `proj_in` / `proj_gate` / `proj_out` 的 `activation/weight/bias/mma/accum/output` 字段. `engine="triton"` 会经由 shared `materialize_module(...)` 创建独立 FeedForward candidate 并保留 contract metadata; CPU 或 kernel 不可用时会显式记录 eager fallback. 同一套精度契约也接受 `A x B + C = O` 角色写法, 例如 `PrecisionPolicy.from_matmul(A="nvfp4", B="fp16", C="fp32", mma="fp16", accum="fp32", O="fp16")` 或 `policy={"A": "nvfp4", "B": "fp16", "C": "fp32", "MMA": "fp16", "ACCUM": "fp32", "O": "fp16"}`. 若希望避免长期直接传裸字典, 还可以使用结构化 `FeedForwardPrecisionPolicy(default=..., proj_in=..., proj_gate=..., proj_out=...)` 作为 `policy`. 对单次 GEMM / Linear 级别, `xqt.operator_opt.backends.gemm_precision.gemm_with_precision(...)` 现在也支持 `engine=...`, 结构化 `MatmulPrecisionSpec(activation, weight, bias, mma, accum, output)` 及 `MatmulPrecisionSpec.from_roles(...)`, 作为更底层的统一 matmul 精度契约. `runtime_config()` 返回的是 engine, FFN 默认值, fusion report 和每个投影的最终生效配置. 当前 `activation/weight/bias/mma/output` 已进入模块内部 matmul 与输出路径; `accum` 已真实下推到 FFN 所复用的 Triton half/BF16 GEMM kernel, 并可通过统一 `MatmulPrecisionSpec` 进入相同的 GEMM 路径. 更低精度 GEMM 的真实执行仍依赖 packed contract 和 fused dequant GEMM path: `fp4` / `nvfp4` 可作为 Linear/FFN contract 的存储精度意图, dense eager fallback 不会伪装成真实低比特输出. FFN 的 Triton lowering 是复用已有 GEMM epilogue / gated pointwise kernels 的 candidate composition, 不是单个完整 FFN megakernel. `Attention` / `TransformerBlock` 也沿用语义块替换入口, 中间经 `materialize_module(...)` / wrapper 进入 engine, 再决定是一组 kernel 还是 megakernel. `optimize_hunyuan_ocr_svd_int4_blocks(...)` 是模型专用的 block composition: 它以 `w4a16_int4` 保存 SVD residual, 物化 INT8 MMA residual compute view, 对外层 `nn.ModuleList` 的每个逻辑 block 分别执行 `torch.compile`, 并用实际模型输入 warmup. 它不是 single fused block kernel; 若模型没有可识别的量化 block, helper 会报错而不是退化为 Linear 级优化. 它属于 `xqt` 子模块级 Provisional / Internal 能力, 不改变 `xqt.__all__` 顶层工作流入口集合.

`operator_opt` 的运行时验收以 block 为准. `candidate_kind=single_kernel` 只表示替换一个子算子或小 wrapper, `candidate_kind=block_kernel` 表示替换完整 block. 当 `block_kernel` 目标使用 `engine=torch_compile` 并同时声明 `block_kernel` 与 `block_kernel_engine` 时, plan 会先尝试自动 block 图候选, 未应用时再运行手写 block kernel 后备; 两者仍共享同一个 `benchmark_target` block 边界.

配置单轨化已完成 Phase A/B/C: workflow 主链已通过 typed `StageSpec` / stage helper 消费 runtime config; workflow context 为 runtime-only, public `create_context()` 只接受 `OptimizationConfig` 或 workflow 输入; 旧 `load_xqt_config()` / `XQTConfig` / `XQTContext.config` 已删除. 最初诊断报告 1-8 的长期落地状态见 [architecture/xqt-realignment-guide.md](architecture/xqt-realignment-guide.md). 配置单轨化执行清单已完成并清理; 长期状态见 realignment-guide 与 FRAMEWORK.

`OptimizationConfig` 与纯 `StageSpec` schema 已下沉 `xqt.core.workflow_schema` / `xqt.core.stage_specs`;`pipeline`, `quant` 和 `readiness` 不再反向 import workflow 类型. `import xqt` 的公开入口全部按符号 lazy load,冷 import 只加载 `xqt` 自身且不拉起 XDL.

StageSpec 到 runtime Config 的转换统一使用 `xqt.core.stage_specs.stage_spec_to_config()`,不再由 workflow,pipeline 或 preflight 手工维护字段清单. 量化 strategy,`QuantScheme`,nature 与 canonical method/compute 的事实源统一为 `xqt.contracts.quant_strategy`;core schema,quant strategy resolver,capability 和三轴报告只从该契约派生. quantizer 公共 policy,模块替换,calibration 调用与 route adapter 工厂位于 `xqt.quant.quantizers.base`.

## 4. 性能分析工具

本章节保留为兼容锚点. `XQT` 的 profiling 边界和工程约束已经迁到:

- [architecture/xqt.md](architecture/xqt.md)
- [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md)

如果你要更新 profiler 角色边界, artifact 记录方式或厂商工具映射, 优先更新这些 canonical 页面.

## 5. 场景表

本章节保留为兼容锚点. `XQT` 的场景状态, 能力矩阵和 readiness 理解, 优先在下列页面维护:

- [explanation/xqt-concepts.md](explanation/xqt-concepts.md)
- [usage/xqt-workflows.md](usage/xqt-workflows.md)

## 6. 文档分层

- 兼容总页: 当前文件,保留完整正文和旧链接锚点.
- [index.md](index.md): Markdown 总入口.
- [architecture/xqt.md](architecture/xqt.md): 架构层正文.
- [explanation/xqt-concepts.md](explanation/xqt-concepts.md): 说明层正文.
- [usage/xqt-workflows.md](usage/xqt-workflows.md): 使用层正文.
- [XQT_SUMMARY.md](XQT_SUMMARY.md): 兼容摘要入口.
- [../html/xqt.html](../html/xqt.html): 人类阅读页.
- [../../xqt/README.md](../../xqt/README.md): 包内短入口.
- [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md): 包内工程契约.

后续新增长期主题,优先直接落到三层目录,不要继续把新主题收拢回本文件.
