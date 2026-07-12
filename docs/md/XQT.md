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

当前 `TileLang` 的受限 kernel target 已覆盖 `attention`, `conv`, direct half `linear`, `linear_marlin`, direct half `LayerNorm`, `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体. 这些 pattern 不是同等成熟度: `attention` 和 direct half `linear` 已有受限 CUDA fp16 kernel 入口, `conv` 当前是 `torch.unfold` / im2col 加 TileLang half GEMM 的 lowering 路径而不是 fully fused conv, direct half `LayerNorm` 已接入 TileLang `reduce_sum` kernel 且限制为 last-dim fp16. CPU 路径只使用 PyTorch eager fallback.

当前 `CuTile` 的 kernel catalog 已对齐上述 `TileLang` pattern, 即 `attention`, `conv`, `linear`, `norm`, `dense_linear_epilogue`, `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体, 并保留原有 `bias_silu` pointwise scaffold. `CuTile` 仍是 metadata-first / reference-guarded engine: XQT 记录 artifact 和 capability, 并通过 `cuda.tile` 或兼容的 `cutile` 模块探测运行时; 内置 executor 可 materialize 线性 / dequant GEMM 的 reference-guarded 推理 wrapper, 但不把它当作完整通用执行器.

当前 `CuTe DSL` engine adapter 覆盖 `gemm_epilogue` 和 `grouped_gemm` metadata. 内置 executor 可把 NVFP4 dense-cache bridge 路由到 `gemm_epilogue` reference-guarded 推理 wrapper; packed NVFP4 权重不会由 CuTe DSL 直接消费.

`PrecisionPolicy` 是 `xqt.convert(...)`, `xqt.nn` facade runtime intent 和 `gemm_with_precision(...)` 的统一静态精度 contract. GEMM 入口中的 `MatmulPrecisionSpec` 只保留为该类的 identity alias; role/mapping canonicalization 也由 `PrecisionPolicy` 提供. facade 的 `auto` 仍仅表示根据输入 dtype 延迟决策, 不构成第二套静态 schema.

新增的轻量算子转换入口使用 `xqt.convert(...)` 和当前真实的 `xqt.nn.*` facade. `Linear` / `Conv2d` / `LayerNorm` 是保留 PyTorch module/state_dict 语义并显式承载 engine 与 precision runtime intent 的 `torch.nn` 子类, 但不应写成已完成的完整 block-level lowering. `FeedForward` / `RMSNorm` / `Attention` / `TransformerBlock` 也是 semantic facade, 但 facade 不直接等于某个 kernel pattern. 这条入口面向单模块或小范围 operator conversion, 对外保持函数形态, 对内通过状态化 converter 完成 contract lowering 和 engine materialization. 其中 `xqt.nn.FeedForward` 采用固定 FFN 骨架 `norm? -> project-in -> activation/gate -> dropout? -> project-out`, 把 `norm`, `activation` 和 `fusion` 显式暴露为顶层配置, 并在模块内部优先复用现有 Triton GEMM epilogue 与 pointwise fastpath. 当前 `xqt.convert(feedforward, engine=..., policy=..., projection_policies=...)` 会把 `PrecisionPolicy` 注入 FFN runtime 配置: `policy` 作为整个 FFN 的默认精度意图, `projection_policies` 可进一步分别覆写 `proj_in` / `proj_gate` / `proj_out` 的 `activation/weight/bias/mma/accum/output` 字段. `engine="triton"` 会经由 shared `materialize_module(...)` 创建独立 FeedForward candidate 并保留 contract metadata; CPU 或 kernel 不可用时会显式记录 eager fallback. 同一套精度契约也接受 `A x B + C = O` 角色写法, 例如 `PrecisionPolicy.from_matmul(A="nvfp4", B="fp16", C="fp32", mma="fp16", accum="fp32", O="fp16")` 或 `policy={"A": "nvfp4", "B": "fp16", "C": "fp32", "MMA": "fp16", "ACCUM": "fp32", "O": "fp16"}`. 若希望避免长期直接传裸字典, 还可以使用结构化 `FeedForwardPrecisionPolicy(default=..., proj_in=..., proj_gate=..., proj_out=...)` 作为 `policy`. 对单次 GEMM / Linear 级别, `xqt.operator_opt.backends.gemm_precision.gemm_with_precision(...)` 现在也支持 `engine=...`, 结构化 `MatmulPrecisionSpec(activation, weight, bias, mma, accum, output)` 及 `MatmulPrecisionSpec.from_roles(...)`, 作为更底层的统一 matmul 精度契约. `runtime_config()` 返回的是 engine, FFN 默认值, fusion report 和每个投影的最终生效配置. 当前 `activation/weight/bias/mma/output` 已进入模块内部 matmul 与输出路径; `accum` 已真实下推到 FFN 所复用的 Triton half/BF16 GEMM kernel, 并可通过统一 `MatmulPrecisionSpec` 进入相同的 GEMM 路径. 更低精度 GEMM 的真实执行仍依赖 packed contract 和 fused dequant GEMM path: `fp4` / `nvfp4` 可作为 Linear/FFN contract 的存储精度意图, dense eager fallback 不会伪装成真实低比特输出. FFN 的 Triton lowering 是复用已有 GEMM epilogue / gated pointwise kernels 的 candidate composition, 不是单个完整 FFN megakernel. `Attention` / `TransformerBlock` 也沿用语义块替换入口, 中间经 `materialize_module(...)` / wrapper 进入 engine, 再决定是一组 kernel 还是 megakernel. 它属于 `xqt` 子模块级 Provisional / Internal 能力, 不改变 `xqt.__all__` 顶层工作流入口集合.

配置单轨化已完成 Phase A/B/C: workflow 主链已通过 typed `StageSpec` / stage helper 消费 runtime config; workflow context 为 runtime-only, public `create_context()` 只接受 `OptimizationConfig` 或 workflow 输入; 旧 `load_xqt_config()` / `XQTConfig` / `XQTContext.config` 已删除. 最初诊断报告 1-8 的长期落地状态见 [architecture/xqt-realignment-guide.md](architecture/xqt-realignment-guide.md). 配置单轨化执行清单已完成并清理; 长期状态见 realignment-guide 与 FRAMEWORK.

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
