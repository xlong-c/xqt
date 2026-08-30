# XQT Kernel 指导表

本文面向 `YOLO`, `DiT`, `MoE LLM` 三类模型, 给出 **比 layer 更细** 的 kernel 指导表. 它只讨论 `kernel`, `wrapper/materialize`, `todo.nn` / `xqt.nn` facade 的分工, 不替代 engine 能力矩阵或 quant 文档.

## 负责什么

- 按三类模型整理所需的细粒度 kernel family.
- 说明哪些应该是 `todo.nn` / `xqt.nn` 的 layer 抽象, 哪些应该是 kernel pattern.
- 说明 `fusion layer` 和 `fusion kernel` 的边界.
- 给后续 engine / wrapper / facade 设计一个统一 TODO 视图.

## 不负责什么

- 不声明当前所有 kernel 都已实现.
- 不给 quant backend / method 下定义.
- 不把外部 deploy backend 写成 XQT kernel.
- 不把 facade 名字直接当作 kernel 名.

## 总规则

`todo.nn` 或 `xqt.nn` 应只表达 **layer / semantic block 抽象**.  
kernel 选择和组合应由 `wrapper/materialize` 根据 contract, engine, precision, shape 和 fallback policy 决定.

也就是说:

```text
todo.nn.Attention / FeedForward / YoloNeckBlock / MoeExpertFFN
    -> contract + fusion intent
    -> wrapper/materialize
    -> one kernel / several kernels / reference fallback
```

不要写成:

```text
todo.nn.FeedForward == swiglu kernel
todo.nn.Attention == flash attention kernel
```

## 术语

| 词 | 含义 |
| --- | --- |
| `layer` / `semantic facade` | 对外的 `nn.Module` 语义块, 例如 `Linear`, `Attention`, `FeedForward`, `TransformerBlock` |
| `kernel` | pattern 级 tensor 计算实现, 例如 `gemm_fp16`, `rmsnorm`, `rope`, `conv3x3`, `grouped_gemm` |
| `wrapper/materialize` | 在 facade 和 kernel 之间做 shape/layout/cache/fallback 翻译的层 |
| `fusion layer` | 对外暴露为一个语义块, 内部允许由多个 kernel 实现 |
| `fusion kernel` | 单个 kernel 内部直接完成多个子步骤 |
| `FusionIntent` | 不绑定某个 engine 的融合意图 contract |

当前 `FusionIntent` 已是共享 lowering contract, 语义是 "requested operator fusion semantics independent of one kernel engine", 见 [xqt/kernels/precision.py](/root/workspace/xdl/xqt/kernels/precision.py).

## 一条边界线

### `todo.nn` / `xqt.nn` 应负责

- 表达语义块身份
- 暴露 runtime intent
- 暴露 precision / fusion / norm / activation 等结构化配置
- 保持 `nn.Module` / `state_dict` 语义

### kernel 应负责

- tensor 级输入输出
- shape / dtype / layout 限制
- 局部 epilogue
- tile / block / schedule / cache 友好的执行实现

### `wrapper/materialize` 应负责

- 选择 kernel
- 决定一个 layer 用一个 kernel 还是多个 kernel
- 处理 flatten / reshape / pack / dequant / dense-cache / KV cache / fallback
- 暴露 execution metadata

## 三类模型的 kernel 指导表

下面的表故意写成 **kernel family**, 不是 layer 名.

在继续往下看之前, 先记一个总原则:

- `GEMM` 不是一个 kernel.
- 对 XQT 来说, 至少要区分:
  - dense 2D GEMM
  - batched GEMM
  - grouped GEMM
  - dequant GEMM
  - packed-weight GEMM
  - epilogue-fused GEMM
  - attention 专用 matmul
  - router / dispatch 相关 matmul

## GEMM 细分类总表

这一节独立于具体模型, 先把 GEMM 系列拆细.

### 最小实现原则

如果目标是先做 **最小可用实现**, 默认策略应是:

1. 先把大部分热点还原成 `GEMM + bias/activation epilogue`.
2. `conv` 默认先走 `unfold/im2col + GEMM + bias/activation`.
3. 只有当 shape 固定, 数据复用明显, 且 GEMM lowering 已经成为瓶颈时, 再追 fully fused conv kernel.

也就是说, 在最小实现里:

- `conv` 更像一个 **GEMM family 的前端 lowering**
- `conv_bias_silu` 更像一个 **conv lowering + GEMM epilogue**
- 不应一开始就把所有 conv 单独视作独立 kernel 大类

这条原则对 YOLO 尤其重要, 因为很多最小闭环其实先靠 `1x1/3x3 -> im2col GEMM` 就能跑起来.

### 0. 按数学结构分

| GEMM family | 典型 I/O | 说明 | 常见场景 |
| --- | --- | --- | --- |
| `dense_gemm_2d` | `A[M,K] x B[K,N] -> C[M,N]` | 最基础的 2D GEMM | Linear, proj, lm_head |
| `batched_gemm` | `A[B,M,K] x B[B,K,N] -> C[B,M,N]` | batch 维独立 GEMM | 多头 attention 的某些实现 |
| `grouped_gemm` | `group_i: A[Mi,K] x B_i[K,N_i]` | 一次 launch 做多组不同问题 | MoE experts |
| `splitk_gemm` | `A[M,K] x B[K,N]` | 对 K 维切分归约 | 大 K, 吞吐优化 |
| `dequant_gemm` | `A fp16/bf16 x B_q + scale -> C` | GEMM 前或 GEMM 内反量化 | weight-only int4/int8 |
| `packed_weight_gemm` | `A x packed(B)` | 权重按特定 layout/pack 存储 | Marlin, FP4/NVFP4 |
| `epilogue_fused_gemm` | `GEMM -> bias/act/...` | GEMM 后处理直接在 epilogue 做 | GELU, SiLU, bias |
| `conv_lowering_to_gemm` | `im2col/unfold(X) x W -> Y` | conv 先变成 GEMM 问题再求解 | YOLO, patchify conv |
| `quantized_activation_gemm` | `quant(A) x B_q -> C` | activation 也进入低比特 contract | true W8A8, FP8 |
| `attention_score_matmul` | `Q x K^T -> score` | 专用 matmul, shape 与 mask 特殊 | attention |
| `attention_value_matmul` | `P x V -> O` | attention 第二段 matmul | attention |
| `router_gemm` | `X[D] -> logits[E]` | 小 N / 大 E 或相反 | MoE router |
| `expert_gemm` | packed token x expert weight | 本质是 grouped GEMM 的语义子类 | MoE experts |

### 1. 按精度 contract 分

这一层更接近 `PrecisionPolicy` / `gemm_with_precision(...)`.

| family | activation | weight | mma | accum | 说明 |
| --- | --- | --- | --- | --- | --- |
| `gemm_fp16` | fp16 | fp16 | fp16 | fp16/fp32 | 最常见 dense GEMM |
| `gemm_bf16` | bf16 | bf16 | bf16 | fp32 | LLM / DiT 常见 |
| `gemm_fp32` | fp32 | fp32 | fp32 | fp32 | 参考或高精度兜底 |
| `gemm_int8_weight_only` | fp16/bf16 | int8 | int8 或 dequant path | fp32 | 常见 weight-only 路线 |
| `gemm_true_w8a8` | int8 | int8 | int8 | int32/fp32 | 真实 W8A8 |
| `gemm_fp8` | fp8 | fp8 | fp8 | fp16/fp32 | Hopper/Ada 等 |
| `gemm_int4_dequant` | fp16/bf16 | int4 + scale/zero | fp16/int8 path | fp32 | weight-only int4 |
| `gemm_fp4_packed_dequant` | fp16/bf16 | fp4 packed | fp16/专用低比特路径 | fp32 | packed FP4 |
| `gemm_nvfp4_packed_dequant` | fp16/bf16 | nvfp4 packed | fp16/专用低比特路径 | fp32 | Blackwell 向 |
| `gemm_mxfp*` | mxfp* | mxfp* | mxfp* | 实现相关 | MXFP 路线 |

### 1.1 当前仓库里的 GEMM 实现对照表

这一节是"查表找实现"的版本. 优先列出当前仓库已经有名字和入口的 GEMM / GEMM-like pattern.

| family | engine/pattern | 代码入口 | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| `gemm_fp16` | `triton/gemm_fp16` | [xqt/kernels/ops/_impl/triton/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/triton/gemm.py:325) | 可执行 | dense 2D GEMM, 支持 bias/activation |
| `gemm_bf16` | `triton/gemm_bf16` | [xqt/kernels/ops/_impl/triton/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/triton/gemm.py:399) | 可执行 | BF16 dense GEMM |
| `gemm_int8` | `triton/gemm_int8` | [xqt/kernels/ops/_impl/triton/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/triton/gemm.py:443) | reference-guarded | 当前仍回落 reference |
| `gemm_fp8` | `triton/gemm_fp8` | [xqt/kernels/ops/_impl/triton/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/triton/gemm.py:474) | reference-guarded | 当前仍回落 reference |
| `gemm_int4_dequant` | `triton/gemm_int4_dequant` | [xqt/kernels/ops/_impl/triton/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/triton/gemm.py:508) | reference-guarded | INT4 weight-only dequant GEMM |
| `gemm_mxfp*` | `triton/mxfp` | [xqt/kernels/ops/_impl/triton/mxfp_gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/triton/mxfp_gemm.py:302) | 实验 | MXFP 路线 |
| `dense_linear_epilogue` | `tilelang/dense_linear_epilogue` | [xqt/kernels/ops/_impl/tilelang/linear.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/linear.py:16) | 可执行 | 本质是 dense GEMM + epilogue |
| `half_linear` | `tilelang/linear` | [xqt/kernels/ops/_impl/tilelang/linear.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/linear.py:92) | 可执行 | 半精度 linear, 本质 GEMM |
| `dequant_gemm_epilogue` | `tilelang/dequant_gemm_epilogue` | [xqt/kernels/ops/_impl/tilelang/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/gemm.py:73) | 可执行 | dequant + GEMM + epilogue |
| `fp4_packed_dequant_gemm_epilogue` | `tilelang/fp4_packed_dequant_gemm_epilogue` | [xqt/kernels/ops/_impl/tilelang/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/gemm.py:148) | 可执行 | packed FP4 |
| `nvfp4_packed_dequant_gemm_epilogue` | `tilelang/nvfp4_packed_dequant_gemm_epilogue` | [xqt/kernels/ops/_impl/tilelang/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/gemm.py:331) | 可执行 | packed NVFP4 |
| `linear_marlin` | `tilelang/linear_marlin` | [xqt/kernels/ops/_impl/tilelang/linear_marlin.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/linear_marlin.py:515) | 可执行 | Marlin 风格 packed GEMM |
| `int8_mma` | `tilelang/int8_mma` | [xqt/kernels/ops/_impl/tilelang/int8_mma.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/int8_mma.py:146) | 可执行 | true W8A8 |
| `dense_linear_epilogue` | `cutile/dense_linear_epilogue` | [xqt/kernels/ops/_impl/cutile/linear.py](/root/workspace/xdl/xqt/kernels/ops/_impl/cutile/linear.py:13) | reference-guarded | CuTile 对齐 catalog |
| `dequant_gemm_epilogue` | `cutile/dequant_gemm_epilogue` | [xqt/kernels/ops/_impl/cutile/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/cutile/gemm.py:72) | reference-guarded | 参考/桥接 |
| `fp4_packed_dequant_gemm_epilogue` | `cutile/fp4_packed_dequant_gemm_epilogue` | [xqt/kernels/ops/_impl/cutile/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/cutile/gemm.py:177) | reference-guarded | packed FP4 |
| `nvfp4_packed_dequant_gemm_epilogue` | `cutile/nvfp4_packed_dequant_gemm_epilogue` | [xqt/kernels/ops/_impl/cutile/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/cutile/gemm.py:205) | reference-guarded | packed NVFP4 |
| `gemm_epilogue` | `cutlass/gemm_epilogue` | [xqt/kernels/ops/_impl/cutlass/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/cutlass/gemm.py:30) | metadata/reference | 主要 metadata/fallback |
| `grouped_gemm` | `cutlass/grouped_gemm` | [xqt/kernels/ops/_impl/engines/cutlass.py](/root/workspace/xdl/xqt/kernels/ops/_impl/engines/cutlass.py:70) | metadata-only | pattern 已挂, 非主执行路 |
| `gemm_epilogue` | `cute_dsl/gemm_epilogue` | [xqt/kernels/ops/_impl/cute_dsl/gemm.py](/root/workspace/xdl/xqt/kernels/ops/_impl/cute_dsl/gemm.py:30) | metadata/reference | CuTe DSL 方向 |
| `grouped_gemm` | `cute_dsl/grouped_gemm` | [xqt/kernels/ops/_impl/engines/cute_dsl.py](/root/workspace/xdl/xqt/kernels/ops/_impl/engines/cute_dsl.py:70) | metadata-only | MoE 候选方向 |

### 1.2 GEMM 调度入口

如果要从统一入口下手,当前 GEMM 主入口是:

- [`gemm_with_precision(...)`](/root/workspace/xdl/xqt/kernels/ops/_impl/gemm_precision.py:64)

当前已经有的 GEMM composition 入口是:

- `gemm_variant_with_precision(...)`: 按 `KERNEL_GUIDANCE_TABLE` 里的 variant 名做 table-driven 调度.
- `batched_gemm_with_precision(...)`: 共享权重时 flatten 成 2D GEMM, per-batch 权重时逐 batch 复用 2D GEMM.
- `grouped_gemm_with_precision(...)`: 多组 GEMM 的组合实现, 不是 fused grouped kernel.
- `expert_gemm_with_precision(...)`: MoE expert 的 grouped GEMM 语义别名.
- `attention_score_gemm_with_precision(...)`: QK score GEMM, 不包含 mask/softmax/dropout.
- `attention_value_gemm_with_precision(...)`: PV value GEMM, 不包含 attention 融合逻辑.
- `router_gemm_with_precision(...)`: MoE router logits GEMM, 不包含 softmax/topk/dispatch.
- `projection_gemm_with_precision(...)`: Q/K/V/QKV/O projection, FFN up/gate/down, lm_head, detection head 等共享的投影 GEMM.
- `conv1x1_as_gemm_with_precision(...)`: NCHW 1x1 conv 降成 GEMM + bias/activation.
- `conv3x3_im2col_gemm_with_precision(...)`: 3x3 conv 走 unfold/im2col + GEMM + bias/activation.

这些入口是为了先把常见形状收敛到已有 2D GEMM family, 不是在声明已经有 fused grouped GEMM 或 fused attention kernel.
像 `flash_attention_*`, `router_softmax_topk`, `dispatch_pack_tokens`, `combine_scatter_tokens`, `dfl_projection_reduce` 这类名字仍然是独立的 fusion / routing / reduction backlog, 不通过 GEMM dispatcher 伪装执行.

这个入口目前已经能区分:

- `triton`
- `tilelang`
- `torch`
- `auto`

以及按 `PrecisionPolicy` / `MatmulPrecisionSpec` 区分:

- `fp16`
- `bf16`
- `int8`
- `fp8`
- `int4`
- `mxfp*`

如果要做最小实现,很多新 GEMM family 最后都应该优先考虑挂进这个统一入口,而不是先另起一套平行 dispatcher.

### 2. 按 epilogue 分

| epilogue family | 典型形式 | 场景 |
| --- | --- | --- |
| `gemm_bias` | `A@B + bias` | 基础 Linear |
| `gemm_bias_relu` | `A@B + bias -> relu` | CNN |
| `gemm_bias_gelu` | `A@B + bias -> gelu` | DiT / MLP |
| `gemm_bias_silu` | `A@B + bias -> silu` | YOLO / gated 路线 |
| `gemm_swiglu_gate` | `up, gate -> swiglu` | LLM / MoE |
| `gemm_geglu_gate` | `up, gate -> geglu` | DiT / LLM |
| `gemm_dequant_epilogue` | `dequant(B) + gemm + bias/act` | 低比特 weight-only |
| `gemm_residual_add` | `A@B + residual` | block 内融合 |
| `gemm_norm_requested` | `A@B` 后希望接 norm | contract 级意图, 不一定单 kernel |

### 3. 按 shape / runtime 约束分

| family | shape 特征 | 典型问题 |
| --- | --- | --- |
| `tall_skinny_gemm` | `M` 大, `N` 小 | token 数大, hidden 小 |
| `fat_gemm` | `N` 大 | FFN up-proj |
| `small_batch_gemm` | `M` 小 | decode 单 token |
| `large_batch_gemm` | `M` 大 | prefill / DiT full token |
| `aligned_tensorcore_gemm` | `m/n/k` 满足 block 对齐 | TileLang / tensorcore 快路 |
| `misaligned_fallback_gemm` | 不对齐 | 需 fallback 或 reference |
| `prefill_gemm` | full sequence | 吞吐优先 |
| `decode_gemm` | `M` 极小, 高频调用 | launch overhead 敏感 |

### 1. YOLO

| kernel family | 典型 I/O | 常见变体 | 建议归属 |
| --- | --- | --- | --- |
| `conv2d_direct` | `[B,Cin,H,W] -> [B,Cout,H',W']` | `1x1`, `3x3`, stride, padding | kernel |
| `conv2d_im2col_gemm` | 同上 | unfold + GEMM lowering | kernel |
| `depthwise_conv2d` | `[B,C,H,W] -> [B,C,H',W']` | `k=3/5`, stride | kernel |
| `group_conv2d` | `[B,Cin,H,W] -> [B,Cout,H',W']` | groups > 1 | kernel |
| `conv_bn_folded_epilogue` | conv out -> conv out | bias/scale/shift folded | kernel 或 pre-export lowering |
| `conv_bias_silu` | conv out -> conv out | `bias + SiLU` | fusion kernel 候选 |
| `conv_bias_relu` | conv out -> conv out | `bias + ReLU` | fusion kernel 候选 |
| `residual_add` | same -> same | shortcut add | kernel |
| `concat_channel` | 多个 `[B,Ci,H,W]` -> `[B,sum(Ci),H,W]` | neck / PAN concat | wrapper 可直接调 eager, 也可专门 kernel |
| `upsample_nearest` | `[B,C,H,W] -> [B,C,rH,rW]` | `x2` 最常见 | kernel |
| `upsample_bilinear` | 同上 | 对齐 corners 变体 | kernel |
| `spp_pool` | `[B,C,H,W] -> [B,C,H,W]` | 多 kernel size maxpool | kernel |
| `bbox_decode` | head tensor -> boxes | anchor-free / DFL | wrapper / postprocess kernel |
| `topk_score_filter` | logits -> topk | per-class / class-agnostic | kernel |
| `nms` | boxes + scores -> kept idx | batched / class-aware | kernel 或 deploy backend |

YOLO 最低优先级 facade 建议:

- `todo.nn.ConvBlock`
- `todo.nn.CSPBlock`
- `todo.nn.SPPFBlock`
- `todo.nn.YoloHead`

但这些 facade 不应直接绑定某个 `conv_bias_silu` kernel. 它们只是给 wrapper 提供结构化块.

YOLO 里的 GEMM 进一步细分:

| GEMM kernel family | 典型 I/O | 说明 |
| --- | --- | --- |
| `conv1x1_as_gemm` | im2col / reshape 后 `A[M,K] x B[K,N]` | 1x1 conv 常可降成 GEMM |
| `conv3x3_im2col_gemm` | unfold 后 GEMM | 当前很多非 fully fused conv 路线本质在这里 |
| `head_cls_gemm` | `[B,HW,C] x [C,Ncls]` | 检测头分类分支 |
| `head_box_gemm` | `[B,HW,C] x [C,4/4*regmax]` | 检测头回归分支 |
| `dfl_projection_gemm` | distribution -> box scalar | DFL 路径可视为小 GEMM / reduction |
| `conv_bias_silu_epilogue` | conv/GEMM 后接 bias+silu | 高频热点 |

如果要查当前 conv->GEMM lowering 的现有实现锚点:

- TileLang conv pattern 注册: [xqt/kernels/ops/_impl/engines/tilelang.py](/root/workspace/xdl/xqt/kernels/ops/_impl/engines/tilelang.py:148)
- conv 参考 / 入口: [xqt/kernels/ops/_impl/conv.py](/root/workspace/xdl/xqt/kernels/ops/_impl/conv.py:1)

YOLO 的最小实现建议:

- 优先级 1: `conv1x1_as_gemm`
- 优先级 2: `conv3x3_im2col_gemm`
- 优先级 3: `conv_bias_silu_epilogue`
- fully fused `conv2d_direct` 放到后续性能阶段

### 2. DiT

| kernel family | 典型 I/O | 常见变体 | 建议归属 |
| --- | --- | --- | --- |
| `patch_embed_conv` | `[B,C,H,W] -> [B,T,D]` | `Conv2d(stride=patch)` | kernel |
| `patch_embed_linear` | `[B,T,P] -> [B,T,D]` | flatten patch 后 GEMM | kernel |
| `layernorm_lastdim` | `[B,T,D] -> [B,T,D]` | fp16 / bf16 | kernel |
| `adaln_scale_shift` | `[B,T,D] + cond -> [B,T,D]` | scale/shift/gate | kernel 或 wrapper composition |
| `qkv_projection_gemm` | `[B,T,D] -> [B,T,3*H*Dh]` | fused qkv / split qkv | kernel |
| `rope_2d_or_pos_apply` | q/k 或 hidden -> same | 2D sin-cos, learned pos apply | kernel |
| `attention_qk_matmul` | `[B,H,T,Dh] x [B,H,Dh,T] -> [B,H,T,T]` | dense attention | kernel |
| `attention_mask_softmax` | `[B,H,T,T] -> same` | no causal, optional mask | kernel |
| `attention_pv_matmul` | `[B,H,T,T] x [B,H,T,Dh] -> [B,H,T,Dh]` | dense attention | kernel |
| `flash_attention_fwd` | `q,k,v -> o` | fused online softmax path | fusion kernel 候选 |
| `proj_out_gemm` | `[B,T,H*Dh] -> [B,T,D]` | output proj | kernel |
| `gelu_epilogue` | hidden -> hidden | `bias + gelu` | fusion kernel 候选 |
| `swiglu/geglu_gate` | split hidden -> hidden | gated MLP | kernel |
| `ffn_up_proj` | `[B,T,D] -> [B,T,Dff]` | proj_in / proj_gate | kernel |
| `ffn_down_proj` | `[B,T,Dff] -> [B,T,D]` | proj_out | kernel |
| `residual_add` | same -> same | pre/post norm residual | kernel |

DiT 的 facade 建议:

- `todo.nn.PatchEmbed`
- `todo.nn.Attention`
- `todo.nn.FeedForward`
- `todo.nn.DiTBlock`

其中:

- `todo.nn.DiTBlock` 是 **fusion layer**
- `flash_attention_fwd` 才可能是 **fusion kernel**

DiT 里的 GEMM 进一步细分:

| GEMM kernel family | 典型 I/O | 说明 |
| --- | --- | --- |
| `patch_embed_gemm` | `[B,T,P] x [P,D] -> [B,T,D]` | flatten patch 后投影 |
| `q_proj_gemm` | `[B,T,D] x [D,H*Dh]` | 分离 Q |
| `k_proj_gemm` | 同上 | 分离 K |
| `v_proj_gemm` | 同上 | 分离 V |
| `fused_qkv_gemm` | `[B,T,D] x [D,3*H*Dh]` | 更常见的 fused QKV |
| `o_proj_gemm` | `[B,T,H*Dh] x [H*Dh,D]` | attention 输出投影 |
| `ffn_up_gemm` | `[B,T,D] x [D,Dff]` | MLP up |
| `ffn_gate_gemm` | `[B,T,D] x [D,Dff]` | GeGLU / SwiGLU gate |
| `ffn_down_gemm` | `[B,T,Dff] x [Dff,D]` | MLP down |
| `adaln_cond_gemm` | cond -> scale/shift/gate | DiT 条件调制 |
| `bias_gelu_epilogue_gemm` | up proj 后直接 gelu | GELU MLP |
| `bias_geglu_epilogue_gemm` | 双投影后接 GeGLU | gated MLP |

如果要查当前这些路径最接近的实现锚点:

- `fused_qkv_gemm` / `o_proj_gemm` 当前最接近 `gemm_fp16` / `gemm_bf16` 这类 dense GEMM 组合入口
- `flash_attention_fwd` 参考 attention kernel: [xqt/kernels/ops/_impl/attention.py](/root/workspace/xdl/xqt/kernels/ops/_impl/attention.py:1)
- `FeedForward` 融合意图与 runtime 组合: [xqt/kernels/nn/feedforward.py](/root/workspace/xdl/xqt/kernels/nn/feedforward.py:304)

### 3. 普通 MoE LLM

| kernel family | 典型 I/O | 常见变体 | 建议归属 |
| --- | --- | --- | --- |
| `embedding_gather` | token ids -> `[B,S,D]` | token / position | kernel |
| `rmsnorm` | `[B,S,D] -> [B,S,D]` | residual fused / channel-first 罕见 | kernel |
| `rope_apply` | `q,k -> q,k` | prefill / decode | kernel |
| `qkv_projection_gemm` | `[B,S,D] -> [B,S,3*H*Dh]` | fused qkv / separate q,k,v | kernel |
| `qk_matmul_causal` | `[B,H,S,Dh] x [B,H,Dh,S] -> [B,H,S,S]` | prefill causal | kernel |
| `decode_qk_kvcache` | `q x K_cache -> score` | paged / contiguous KV cache | kernel |
| `softmax_masked` | score -> prob | causal, sliding window | kernel |
| `pv_matmul` | prob x V -> attn_out | prefill / decode | kernel |
| `flash_attn_prefill` | `q,k,v -> o` | full-seq fused | fusion kernel 候选 |
| `flash_attn_decode` | `q,kv_cache -> o` | paged KV cache | fusion kernel 候选 |
| `o_proj_gemm` | `[B,S,H*Dh] -> [B,S,D]` | output proj | kernel |
| `router_logits_gemm` | `[B,S,D] -> [B,S,E]` | expert gate logits | kernel |
| `router_softmax_topk` | `[B,S,E] -> indices + weights` | top1 / top2 | fusion kernel 候选 |
| `dispatch_pack_tokens` | tokens -> packed expert batches | sort / segment / pad | wrapper 或 kernel |
| `grouped_gemm_up` | packed tokens -> expert hidden | grouped `up/gate` projection | kernel |
| `grouped_gemm_gate` | packed tokens -> gate hidden | SwiGLU / GeGLU | kernel |
| `swiglu/geglu_gate` | hidden split -> hidden | gate activation | kernel |
| `grouped_gemm_down` | expert hidden -> expert out | proj_out | kernel |
| `combine_scatter_tokens` | expert out -> `[B,S,D]` | weighted combine | wrapper 或 kernel |
| `lm_head_gemm` | `[B,S,D] -> logits` | tied embedding 可复用 weight | kernel |

MoE LLM 的 facade 建议:

- `todo.nn.Attention`
- `todo.nn.FeedForward`
- `todo.nn.MoeRouter`
- `todo.nn.MoeExpertFFN`
- `todo.nn.MoeBlock`

真正新增的 kernel family 主要是:

- `router_softmax_topk`
- `dispatch_pack_tokens`
- `grouped_gemm_*`
- `combine_scatter_tokens`
- `flash_attn_decode` / `decode_qk_kvcache`

MoE LLM 里的 GEMM 进一步细分:

| GEMM kernel family | 典型 I/O | 说明 |
| --- | --- | --- |
| `embed_proj_gemm` | hidden -> hidden | 某些 embedding / adapter 路径 |
| `fused_qkv_gemm_prefill` | `[B,S,D] x [D,3*H*Dh]` | full-seq prefill |
| `fused_qkv_gemm_decode` | `[B,1,D] x [D,3*H*Dh]` | decode 单 token, 小 M |
| `o_proj_gemm_prefill` | `[B,S,H*Dh] x [H*Dh,D]` | prefill |
| `o_proj_gemm_decode` | `[B,1,H*Dh] x [H*Dh,D]` | decode |
| `router_logits_gemm` | `[B,S,D] x [D,E]` | router logits |
| `expert_up_grouped_gemm` | packed token -> expert hidden | expert up proj |
| `expert_gate_grouped_gemm` | packed token -> gate hidden | expert gate proj |
| `expert_down_grouped_gemm` | packed hidden -> out | expert down proj |
| `shared_expert_gemm` | token -> shared expert | shared expert 变体 |
| `lm_head_gemm_prefill` | `[B,S,D] x [D,V]` | prefill logits |
| `lm_head_gemm_decode` | `[B,1,D] x [D,V]` | decode logits |
| `w8a8_true_gemm` | int8 act x int8 weight | 真 W8A8 |
| `int4_weight_only_dequant_gemm` | fp16 act x int4 packed weight | 最常见 weight-only |
| `fp4/nvfp4_packed_gemm` | fp16/bf16 act x fp4/nvfp4 packed weight | 更激进低比特 |
| `marlin_style_packed_gemm` | act x packed int4/int8 layout | packed 权重专用 |

如果要查当前最直接的实现锚点:

- `true W8A8`: [xqt/kernels/ops/_impl/tilelang/int8_mma.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/int8_mma.py:146)
- `packed int4/int8 marlin`: [xqt/kernels/ops/_impl/tilelang/linear_marlin.py](/root/workspace/xdl/xqt/kernels/ops/_impl/tilelang/linear_marlin.py:515)
- `grouped_gemm` 候选方向: [xqt/kernels/ops/_impl/engines/cutlass.py](/root/workspace/xdl/xqt/kernels/ops/_impl/engines/cutlass.py:62), [xqt/kernels/ops/_impl/engines/cute_dsl.py](/root/workspace/xdl/xqt/kernels/ops/_impl/engines/cute_dsl.py:62)

## 跨三类模型的最小 kernel TODO 清单

如果只做最小闭环, 优先级可以这样排:

### P0: 必须先有

| family | 覆盖价值 |
| --- | --- |
| `gemm_fp16/bf16` | DiT / LLM / MoE 主耗时 |
| `conv2d_direct` 或 `conv2d_im2col_gemm` | YOLO 主耗时 |
| `layernorm_lastdim` / `rmsnorm` | DiT / LLM 必需 |
| `swiglu/geglu_gate` | DiT / LLM / MoE 常用 |
| `attention core` (`qk`, `softmax`, `pv`) 或 `flash_attention_fwd` | DiT / LLM 核心 |
| `rope_apply` | LLM 必需 |
| `upsample_nearest` | YOLO neck 必需 |
| `residual_add` / `concat_channel` | 基本结构胶水 |

### P1: 很快会需要

| family | 覆盖价值 |
| --- | --- |
| `router_softmax_topk` | MoE 必需 |
| `dispatch_pack_tokens` / `combine_scatter_tokens` | MoE 必需 |
| `grouped_gemm_up/down` | MoE 性能核心 |
| `conv_bias_silu` | YOLO 高频融合 |
| `bbox_decode` / `nms` | YOLO 端到端 |
| `adaln_scale_shift` | DiT 常见 |
| `flash_attn_decode` / `decode_qk_kvcache` | LLM decode 必需 |

### P2: 后续扩展

| family | 覆盖价值 |
| --- | --- |
| `int8/int4/fp8/fp4 dequant gemm` | 低比特闭环 |
| `depthwise_conv2d` / `group_conv2d` | 更广 YOLO / mobile 变体 |
| `spp_pool` | YOLO 特定结构 |
| `conv3d` / `patch unembed` | 视频 DiT / 扩展模型 |
| `paged KV cache` 细分 kernels | 高端 LLM serving |

## `todo.nn` 应怎么设计

推荐原则:

1. `todo.nn` 只放 **语义稳定** 的 layer / block.
2. 每个 facade 只声明:
   - 结构
   - 精度 intent
   - fusion intent
   - 可选 engine preference
3. facade 不直接写死 kernel 名.
4. 真正的 kernel 组合由 `wrapper/materialize` 决定.

推荐的 `todo.nn` 粒度:

| facade | 说明 |
| --- | --- |
| `Linear`, `Conv2d`, `LayerNorm`, `RMSNorm` | 基础 facade |
| `Attention` | 语义层, 不直接等于 flash attention kernel |
| `FeedForward` | 语义层, 不直接等于 swiglu kernel |
| `TransformerBlock` / `DiTBlock` | fusion layer |
| `MoeRouter`, `MoeExpertFFN`, `MoeBlock` | MoE 语义层 |
| `ConvBlock`, `CSPBlock`, `SPPFBlock`, `YoloHead` | YOLO 语义层 |

同一个 facade 对应的 kernel family 往往不止一个. 例如:

| facade | 可能下沉到的 kernel family |
| --- | --- |
| `todo.nn.Attention` | `fused_qkv_gemm`, `rope_apply`, `flash_attention_fwd`, `flash_attn_decode`, `o_proj_gemm` |
| `todo.nn.FeedForward` | `ffn_up_gemm`, `ffn_gate_gemm`, `swiglu/geglu_gate`, `ffn_down_gemm`, `bias_gelu_epilogue_gemm` |
| `todo.nn.MoeExpertFFN` | `expert_up_grouped_gemm`, `expert_gate_grouped_gemm`, `swiglu/geglu_gate`, `expert_down_grouped_gemm` |
| `todo.nn.ConvBlock` | `conv2d_direct` / `conv2d_im2col_gemm`, `conv_bias_silu`, `residual_add` |

## `fusion layer` 和 `fusion kernel` 怎么分

这是最容易混的地方.

### `fusion layer`

`fusion layer` 是 **对外的语义块抽象**.

例子:

- `todo.nn.FeedForward`
- `todo.nn.TransformerBlock`
- `todo.nn.MoeBlock`
- `todo.nn.ConvBlock`

它的职责是:

- 把多个基础算子组织成一个稳定语义块
- 暴露 `FusionIntent`
- 允许 engine 选择单 kernel 或多 kernel 实现

它 **不要求** 一定存在单个 fused kernel.

### `fusion kernel`

`fusion kernel` 是 **一个 kernel 内部实际完成多个子步骤**.

例子:

- `flash_attention_fwd`
- `router_softmax_topk`
- `conv_bias_silu`
- `proj_in_gelu_epilogue`
- `rmsnorm_residual`
- `dequant_gemm_epilogue`

它的职责是:

- 减少 launch
- 减少中间读写
- 利用 epilogue / schedule / tile reuse

它不需要对外暴露成独立 layer.

### 对 GEMM 来说怎么落

建议把 GEMM 相关融合拆成三层:

1. `fusion intent`
   - 例如 `proj_in_gelu_epilogue`
   - 例如 `swiglu`
   - 例如 `proj_out_epilogue_requested`

2. `fusion composition`
   - `single_fusion_kernel`
   - `two_kernel_pipeline`
   - `multi_kernel_composition`

3. `kernel pattern`
   - `gemm_bias_gelu`
   - `gemm_int4_dequant`
   - `dequant_gemm_epilogue`
   - `flash_attention_fwd`
   - `router_softmax_topk`

不要把这三层混写成一个名字.

## 推荐再细分的 GEMM TODO 表

如果后面真要做 kernel backlog,我建议至少拆到下面这个粒度:

### Dense / 通用 GEMM

- `gemm_fp16_dense_2d`
- `gemm_bf16_dense_2d`
- `gemm_fp32_reference`
- `gemm_fp16_bias`
- `gemm_bf16_bias`
- `gemm_bias`
- `gemm_bias_relu`
- `gemm_bias_gelu`
- `gemm_bias_silu`
- `gemm_residual_add`

### Low-bit / dequant GEMM

- `gemm_int8_weight_only_reference`
- `gemm_true_w8a8`
- `int8_linear`
- `int8_linear_static_activation`
- `gemm_int4_weight_only_dequant`
- `gemm_fp4_packed_dequant`
- `fp4_packed_dequant_gemm_epilogue`
- `gemm_nvfp4_packed_dequant`
- `nvfp4_packed_dequant_gemm_epilogue`
- `gemm_mxfp8`
- `gemm_mxfp6`
- `gemm_mxfp4`
- `marlin_style_packed_gemm`
- `linear_marlin`
- `dequant_gemm_epilogue`

### Attention GEMM / matmul

- `attention_score_matmul`
- `attention_score_gemm_composition`
- `attention_value_matmul`
- `attention_value_gemm_composition`
- `qk_matmul_prefill`
- `pv_matmul_prefill`
- `qk_matmul_decode_kvcache`
- `pv_matmul_decode_kvcache`
- `flash_attention_fwd`
- `flash_attention_decode`
- `q_proj_gemm`
- `k_proj_gemm`
- `v_proj_gemm`
- `fused_qkv_gemm`
- `fused_qkv_gemm_prefill`
- `fused_qkv_gemm_decode`
- `qkv_projection_gemm`
- `o_proj_gemm`
- `o_proj_gemm_prefill`
- `o_proj_gemm_decode`
- `o_projection_gemm`

### Gated MLP / FFN GEMM

- `ffn_up_gemm`
- `ffn_gate_gemm`
- `ffn_down_gemm`
- `swiglu_fusion`
- `geglu_fusion`
- `proj_in_gelu_epilogue`
- `proj_out_epilogue`

### MoE GEMM

- `router_gemm`
- `router_logits_gemm`
- `router_gemm_composition`
- `router_softmax_topk`
- `dispatch_pack_tokens`
- `grouped_gemm_composition`
- `expert_gemm_composition`
- `expert_up_grouped_gemm`
- `expert_gate_grouped_gemm`
- `expert_down_grouped_gemm`
- `shared_expert_gemm`
- `combine_scatter_tokens`

### CNN / detection 里和 GEMM 强相关的 lowering

- `conv1x1_as_gemm`
- `conv3x3_im2col_gemm`
- `depthwise_conv_specialized`
- `conv_bias_silu_epilogue`
- `conv_bias_relu`
- `patch_embed_gemm`
- `patch_embed_linear`
- `head_cls_gemm`
- `head_box_gemm`
- `dfl_projection_gemm`
- `dfl_projection_reduce`

如果按"方便查表实现"的角度,我建议 backlog 名称尽量与现有 pattern 或函数名对齐,例如:

| 建议 backlog 名 | 尽量对齐的现有实现名 |
| --- | --- |
| `gemm_fp16_dense_2d` | `gemm_fp16_triton` |
| `gemm_bf16_dense_2d` | `gemm_bf16_triton` |
| `gemm_int4_weight_only_dequant` | `gemm_int4_dequant_triton` / `dequant_gemm_epilogue_tilelang` |
| `gemm_fp4_packed_dequant` | `fp4_packed_dequant_gemm_epilogue_tilelang` |
| `gemm_nvfp4_packed_dequant` | `nvfp4_packed_dequant_gemm_epilogue_tilelang` |
| `marlin_style_packed_gemm` | `linear_marlin_tilelang` |
| `gemm_true_w8a8` | `int8_mma_tilelang` |
| `grouped_gemm` | `cutlass/cute_dsl grouped_gemm` pattern |

## 对 conv 的建议归类

在最小实现阶段, 推荐这样理解:

| 类别 | 是否先当 GEMM family 处理 | 说明 |
| --- | --- | --- |
| `1x1 conv` | 是 | 最自然的 GEMM 化路径 |
| `3x3 conv` | 是 | 优先 `im2col/unfold + GEMM` |
| `depthwise conv` | 否 | 更像单独特化 family |
| `group conv` | 部分是 | 取决于 groups 和 lowering 成本 |
| `fused conv kernel` | 否 | 这是后续专门优化目标, 不是最小实现前提 |

所以如果现在要做 backlog 排序,应该先写:

- `conv1x1_as_gemm`
- `conv3x3_im2col_gemm`
- `conv_bias_silu_epilogue`

而不是先写:

- `ultimate_fused_conv_kernel`

## 推荐 contract 写法

建议长期保持:

- `FusionIntent` 表达 **希望融合什么**
- `ModuleContract` 表达 **这个 layer 是什么**
- `engine capability + wrapper` 决定 **最终能融合到什么程度**

例如 `FeedForward.fusion_intent()` 当前已经会表达:

- `proj_in_gelu_epilogue`
- `swiglu` / `geglu`
- `norm_requested`
- `proj_out_epilogue_requested`

见 [xqt/kernels/nn/feedforward.py](/root/workspace/xdl/xqt/kernels/nn/feedforward.py:304).

这个 contract 级表达是对的, 因为它说的是:

- `我希望 gate 融合`
- `我希望 norm 融合`

而不是:

- `我必须使用某个 Triton kernel 名`

## 设计建议

### 对 facade

- facade 名称用语义块命名
- 不用 kernel 名命名 facade
- 允许一个 facade 在不同 engine 下 materialize 成不同 kernel 组合

### 对 wrapper/materialize

- 显式记录 `selected kernel set`
- 显式记录 `fallback reason`
- 区分:
  - `single_fusion_kernel`
  - `multi_kernel_composition`
  - `reference_fallback`

### 对 kernel catalog

- kernel 名按 pattern 命名
- 不直接复用 facade 名作为唯一 kernel 名
- 同一个 layer 可以映射到多个 kernel family

## 继续阅读

- [../architecture/xqt-kernel-wrapper-nn-boundary.md](../architecture/xqt-kernel-wrapper-nn-boundary.md)
- [xqt-engines.md](xqt-engines.md)
- [xqt-concepts.md](xqt-concepts.md)
- [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md)
