# 手写算子调优高级指南

本文是一份面向 `XQT` operator optimization 和手写 CUDA / CUTLASS / CuTe DSL 算子的长期指南. 它从算子设计闭环讲到 GEMM, Conv, Linear, Attention, Norm 的专项优化, 并把访存, 通信, 精度, 架构代际特性和 profiling 方法放在同一套判断框架里.

## 负责什么

- 解释如何从 PyTorch 参考实现推进到手写 CUDA / CuTe / CUTLASS 算子.
- 给出 benchmark, profiling, bottleneck hypothesis, 修改和验证的调优闭环.
- 总结 GEMM, Conv, Linear, Attention, Norm 的常见手写路径和优化重点.
- 总结全局内存, shared memory, register, warp, block, cluster, TMA, CUDA Graph 等层次的优化方法.
- 总结 FP32, TF32, FP16, BF16, INT8, FP8, FP4 / NVFP4 等精度路线的实现和验证重点.

## 不负责什么

- 不替代 CUDA, CUTLASS, CuTe DSL 或厂商 profiler 的官方文档.
- 不声明 `XQT` 已经实现本文提到的所有 kernel 形态.
- 不把训练, QAT, finetune, distillation, dataset / dataloader 纳入 XQT 职责.
- 不承诺某个 tile, warp 数或 pipeline stage 对所有 GPU 和 shape 都最优.

## 资料检索记录

本文根据至少 10 轮外部资料检索整理. 优先使用官方文档和论文原文, 不把二手博客当作事实源.

| 轮次 | 主题 | 主要资料 |
| --- | --- | --- |
| 1 | CUDA 执行模型, 内存层次, occupancy, asynchronous copy | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) |
| 2 | coalescing, shared memory bank conflict, 指标化优化 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) |
| 3 | CUTLASS / CuTe GEMM 层次, mainloop, epilogue, scheduler | [CUTLASS documentation](https://docs.nvidia.com/cutlass/) |
| 4 | CuTe DSL tensor, layout, tiled copy, tiled MMA | [CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html) |
| 5 | Hopper TMA, WGMMA, warpgroup, cluster, DSM | [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) |
| 6 | Nsight Compute / Nsight Systems profiling | [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/), [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/) |
| 7 | PTX 指令层: `ldmatrix`, `mma.sync`, `wgmma`, `cp.async` | [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) |
| 8 | Conv implicit GEMM, cuDNN backend, CUTLASS conv | [CUTLASS implicit GEMM convolution](https://docs.nvidia.com/cutlass/media/docs/cpp/implicit_gemm_convolution.html), [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/) |
| 9 | FlashAttention v1 / v2 / v3 | [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691), [FlashAttention-3](https://arxiv.org/abs/2407.08608) |
| 10 | reduction, warp primitive, block primitive | [CUB Documentation](https://nvidia.github.io/cccl/cub/), [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) |
| 11 | FP8 / FP4 / NVFP4, scaling recipe | [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) |
| 12 | Volta, Turing, Ampere, Hopper, Blackwell 代际 tuning | [Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/), [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/), [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/), [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/), [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) |
| 13 | cuBLASLt epilogue, CUTLASS epilogue, CUDA Graph launch overhead | [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/), [CUTLASS epilogue docs](https://docs.nvidia.com/cutlass/), [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) |

## 先建立正确的调优闭环

手写算子不是从 "写 kernel" 开始, 而是从一个可复现的闭环开始:

1. 明确目标: 算子类型, shape, dtype, layout, batch, 动态 shape 范围, GPU 型号和 `sm_*`.
2. 建立参考: PyTorch eager / cuBLAS / cuDNN / FlashAttention / TensorRT plugin 等已知正确实现.
3. 建立 benchmark: warmup, repeat, CUDA event, 同步边界, 输入固定, 输出校验固定.
4. 建立 profiler 证据: `nsys` 看 launch, memcpy, sync 和 runtime gap; `ncu` 看 kernel 内 occupancy, memory, stall, roofline.
5. 提出一个瓶颈假设: 例如 global load 不合并, shared bank conflict, register 压力导致 occupancy 掉, launch-bound, softmax 中间写回太多.
6. 一次只改一个方向: tile shape, `num_warps`, `num_stages`, vector width, swizzle, fusion edge, precision, scheduler.
7. 重新测量: latency, throughput, 带宽, TFLOP/s, 数值误差, 编译时间和架构敏感性.
8. 记录失败: 失败的 tile 和 pipeline stage 也有价值, 因为它们能缩小搜索空间.

没有 benchmark 和 profiler 证据时, 不要直接讨论 "最优 tile". 没有正确性校验时, 不要讨论 "性能提升".

## 一眼判断瓶颈

| 现象 | 常见瓶颈 | 优先检查 |
| --- | --- | --- |
| kernel 时间很短, 端到端仍慢 | launch-bound 或 wrapper-bound | `nsys`, CUDA Graph, kernel fusion, host sync |
| DRAM throughput 高, Tensor Core 低 | memory-bound | coalescing, vector load, tile reuse, shared memory staging |
| Tensor Core 利用低, eligible warps 少 | pipeline 或 occupancy 问题 | `num_stages`, register, shared memory, warp specialization |
| shared memory 指标异常 | bank conflict 或 layout 错 | swizzle, padding, `ldmatrix` layout |
| occupancy 很低 | register / smem / block 资源超限 | 减少 fusion, 缩 tile, 降 unroll, split kernel |
| occupancy 很高但仍慢 | 单 warp 效率差 | Tensor Core mapping, vectorization, memory coalescing |
| 小 shape 不如 PyTorch | launch overhead 或调度开销 | 融合, persistent kernel, CUDA Graph |
| fusion 后变慢 | 融合过度 | register live range, occupancy, epilogue 拆分 |

## 硬件执行层次

写 CUDA 算子时要把问题映射到以下层次:

| 层次 | 关注点 | 常见手段 |
| --- | --- | --- |
| grid / CTA | work partition, tile ownership, block swizzle | CTA tile, split-K, persistent CTA, rasterization |
| warp / warpgroup | warp 分工, MMA mapping, reduction | warp tiling, warp specialization, WGMMA |
| thread / lane | lane 到元素的映射 | vectorized load, coalescing, `shfl`, predication |
| register | accumulator, fragment, scalar 临时变量 | register blocking, live range 控制, unroll 控制 |
| shared memory | 数据复用, staging, producer / consumer | swizzle, padding, double buffer, `cp.async`, TMA |
| global / L2 | 带宽, reuse, cache residency | contiguous layout, vector width, prefetch, CTA swizzle |
| runtime | launch, graph, stream, sync | CUDA Graph, stream capture, 避免隐式同步 |

一个高性能算子通常不是单点优化, 而是让这些层次不互相拖累.

## 从朴素 kernel 到高性能 kernel

### 1. 先写可验证的 reference kernel

最小 kernel 只要求语义清楚:

- 每个输出元素由一个 thread 或一个 warp 负责.
- 边界处理显式写清楚.
- 输出和 PyTorch reference 做 `max_abs`, `max_rel`, `cosine`, `ulp` 或任务相关误差对比.
- 先不追求高性能.

### 2. 调整数据布局和连续访问

第一轮性能提升通常来自访存:

- 全局内存访问按连续地址合并.
- 尽量让每个 warp 访问连续 cache line.
- 让 tensor layout 服务 kernel, 例如 Tensor Core 路径常偏好 NHWC / row-major / column-major 的特定组合.
- 对齐到 16B 或更高的 vector load / store 边界.
- 避免在 hot path 里做 layout transpose, `contiguous()` 或 dtype conversion.

### 3. 引入 tile

tile 是复用的基本单位:

- GEMM: CTA tile 负责 `M x N`, mainloop 沿 `K` 推进.
- Conv: 输出空间 / channel tile 转成 implicit GEMM 的 `M x N x K`.
- Attention: block 内维护 QK, online softmax 和 PV 的局部状态.
- Norm: 一个 row 或多个 row 归到 block / warp reduction.

tile 不只是尺寸, 还包含:

- global 到 shared 的 copy pattern.
- shared 到 register / MMA fragment 的 load pattern.
- warp 到 tile 的 ownership.
- epilogue 写回方式.

### 4. 使用 shared memory 做 staging

shared memory 的价值是用一次 global load 服务多次 compute. 但它有成本:

- shared memory 占用会降低 resident CTA.
- layout 不对会造成 bank conflict.
- 多 stage pipeline 会增加 shared memory 和 register 压力.
- 同步点过多会让 eligible warps 下降.

判断 shared memory 是否值得, 先算数据复用次数. 如果只读一次就写回, shared memory 可能只是额外开销.

### 5. 使用 register blocking

register 里放 accumulator 和小 fragment 可以减少 shared / global 访问. 但 register 是最容易被过度使用的资源:

- accumulator tile 太大, occupancy 会掉.
- fusion epilogue 太复杂, live range 会拉长.
- unroll 太激进, register spilling 会出现.

经验上不要只盯 occupancy. 低 occupancy 但 Tensor Core pipeline 饱满的 GEMM 可能很快; 高 occupancy 但访存和指令发射低效的 kernel 仍然慢.

### 6. 引入 Tensor Core / MMA

GEMM, Linear, Conv 和 Attention 的高性能路径基本绕不开 Tensor Core:

- Volta / Turing / Ampere 上常见 `mma.sync` / `wmma`.
- Hopper 上常见 `wgmma` 和 warpgroup 级协作.
- Blackwell 上会出现新的 Tensor Core 指令族和更复杂的 tensor memory 组织.

MMA 优化关注:

- fragment layout 是否匹配指令.
- `ldmatrix` 或 TMA 到 shared / register 的路径是否高效.
- K tile 是否足够大以 amortize load.
- epilogue 是否破坏 accumulator 的写回效率.

### 7. 引入异步流水线

异步流水线的目标是隐藏 global memory latency:

- Ampere 常见 `cp.async` + double / multi-stage shared memory buffer.
- Hopper 常见 TMA + producer / consumer warp 分工.
- 更复杂场景会使用 warp specialization 或 warpgroup specialization.

流水线调优不是 stage 越多越好. `num_stages` 增加会提高隐藏延迟能力, 也会增加 shared memory 和 register 压力. 如果 profiler 显示 occupancy 掉得太多, 需要减少 stage 或缩 tile.

### 8. 引入 epilogue fusion

epilogue 是融合最自然的位置:

- bias
- activation
- residual add
- clamp
- quantize / dequantize
- scale / zero point
- dropout mask
- norm 的 affine

融合价值来自减少中间张量读写. 但 epilogue 也常导致 register 压力上升. 如果 fusion 后反而慢, 先看 register, spilling, occupancy, instruction mix, 再决定是否拆分.

## CuTe / CUTLASS 视角的写法

CuTe / CUTLASS 的关键不是 "替你调好 kernel", 而是把 layout, tile, copy, MMA, pipeline 这些概念变成可组合对象.

一个 GEMM-like kernel 的思考顺序:

1. 定义 problem shape: `M, N, K`, batch, group, stride.
2. 定义 tensor layout: A, B, C 的 row-major / column-major / interleaved / packed layout.
3. 定义 CTA tile: 例如 `128 x 128 x 64`.
4. 定义 warp / warpgroup MMA tile.
5. 定义 global to shared copy: vector width, alignment, predication.
6. 定义 shared layout: swizzle, padding, `ldmatrix` / WGMMA 友好.
7. 定义 mainloop pipeline: stage 数, producer / consumer, barrier.
8. 定义 epilogue: accumulator 到 output 的 layout, fusion, dtype conversion.
9. 定义 scheduler: split-K, persistent, grouped, stream-K, cluster.
10. 定义校验和 benchmark.

概念伪代码:

```cpp
// 伪代码: 表达结构, 不作为可直接编译实现.
make_problem_shape(M, N, K);
make_gmem_tensors(A, B, C, layout_a, layout_b, layout_c);
make_cta_tile(BLOCK_M, BLOCK_N, BLOCK_K);
make_tiled_copy(vector_width, smem_layout);
make_tiled_mma(instruction_shape, warp_layout);

for each cta_tile:
  initialize accumulators in registers
  for k_tile in K:
    async_copy_gmem_to_smem(A_tile, B_tile)
    wait_for_stage()
    mma(accumulators, A_fragment, B_fragment)
  epilogue(accumulators, bias, activation, scale)
  store(C_tile)
```

手写 CUDA 可以按同一结构写, 只是 layout 和 pipeline 需要自己维护.

## 访存优化

### 全局内存

全局内存优化优先级:

1. 连续访问: warp 内 lane 访问连续地址.
2. 对齐访问: 尽量使用 `float4`, `half2`, `int4`, `uint4` 等 vectorized load / store.
3. 减少访问次数: fusion, register reuse, shared staging.
4. 避免无意义转换: layout transform, dtype cast, temporary tensor.
5. 控制写回: 能在 epilogue 一次完成就不要写中间张量.

典型反模式:

- 每个 thread 访问 stride 很大的地址.
- 每个输出元素重复加载相同权重.
- 先 `im2col` 写大中间矩阵, 再 GEMM, 但 shape 很小导致带宽浪费.
- kernel 内部频繁做 bounds branch, warp divergence 严重.

### L2 和 CTA swizzle

CTA 执行顺序会影响 L2 reuse. 常见手段:

- 对 GEMM 使用 tile rasterization / swizzle, 让相邻 CTA 复用 A 或 B 的 L2 cache.
- 对 attention 按 sequence block 分配, 让 K/V block 被多个 Q block 复用.
- 对 grouped GEMM 按相近 shape 排序, 降低 cache 和调度抖动.

不要把 CTA swizzle 当成默认收益. 它依赖 shape, batch, stride 和 L2 容量.

### Shared memory

shared memory 优化要同时看容量, bank conflict 和同步:

- 为 Tensor Core load 设计 shared layout.
- 使用 swizzle 或 padding 避免 bank conflict.
- 使用 double buffer / multi-stage buffer 隐藏 global load.
- 避免把只用一次的数据搬进 shared.
- 减少 `__syncthreads()` 频率, 能用 warp-level primitive 时不要升级到 block-level sync.

### Register

register 优化关注:

- accumulator tile 大小.
- fusion epilogue 中临时变量数量.
- loop unroll 后的 live range.
- 是否发生 local memory spill.

常见做法:

- 缩小 per-thread output tile.
- 将复杂 epilogue 分阶段计算.
- 使用 `half2` / packed 类型减少指令和寄存器.
- 对不同 shape 做专门 kernel, 避免一个泛化 kernel 保留太多状态.

## 通信优化

这里的通信不是分布式通信, 而是 kernel 内不同执行实体之间的数据交换.

### Thread 内

- 尽量让 thread 持有连续元素.
- 使用 vector type 一次处理多个元素.
- 减少标量临时变量的生命周期.

### Warp 内

warp 内通信优先使用:

- `__shfl_sync` 做 reduction / broadcast.
- `__ballot_sync` 做 mask 聚合.
- warp-level primitive 替代 shared memory 交换.

适合场景:

- Norm 的 row reduction.
- Softmax 的 max / sum reduction.
- 小 channel / 小 hidden size 的 pointwise + reduction.

### CTA 内

CTA 内通信通常通过 shared memory:

- GEMM / Conv staging.
- Attention 的 K/V tile staging.
- 大 hidden size Norm 的 block reduction.

关键是减少同步和 bank conflict. 如果只是 warp 内交换, 不要把数据绕到 shared memory.

### Cluster / DSM / TMA

Hopper 之后可以考虑更重的跨 CTA 协作:

- thread block cluster.
- distributed shared memory.
- TMA bulk copy.
- TMA multicast.

适合大 tile, 高复用, producer / consumer 明确的 kernel. 如果算子很小, 这些机制的同步和调度成本可能超过收益.

### Runtime 通信

小算子端到端慢时, 通信问题可能在 CPU 与 GPU runtime 边界:

- kernel launch 太多.
- host sync 太多.
- stream 使用不当.
- input materialization 发生在测量路径内.

这时优先看 `nsys`, CUDA Graph 和 operator-stage benchmark, 不要急着调 kernel tile.

## 精度优化

### 精度路线速查

| 精度 | 主要价值 | 手写重点 | 风险 |
| --- | --- | --- | --- |
| FP32 | reference, 高精度 | 作为正确性基线 | 慢, 带宽大 |
| TF32 | Ampere+ FP32 输入的 Tensor Core 加速 | 默认 matmul 行为, accumulation | 与严格 FP32 有误差 |
| FP16 | 高吞吐, 低带宽 | Tensor Core, FP32 accumulate | overflow / underflow |
| BF16 | 动态范围优于 FP16 | Tensor Core, FP32 accumulate | 精度位少 |
| INT8 | 推理吞吐和带宽 | scale / zero point, dequant fusion | calibration 和 outlier |
| FP8 | Transformer 推理 / 训练加速 | scaling recipe, amax, accumulation | scale 管理复杂 |
| FP4 / NVFP4 | 极限压缩和带宽 | packed load, unpack, dequant GEMM fusion | 误差和 kernel 复杂度高 |

### Accumulation

低精度输入不等于低精度累加:

- FP16 / BF16 GEMM 通常需要 FP32 accumulate.
- INT8 GEMM 通常累加到 INT32, 再 scale 到 FP16 / BF16 / FP32.
- FP8 / FP4 通常需要更谨慎的 scale 和较高精度 accumulate.

如果要把 accumulate 也降精度, 必须用更严格的数值验证证明可接受.

### Scale 和 dequant fusion

量化算子的核心性能点是避免中间反量化张量:

- 不要先把 weight 全量 dequant 到 FP16 再 GEMM.
- 优先做 `packed load -> unpack -> scale -> MMA / dot -> epilogue`.
- per-tensor scale 简单, per-channel / per-group / block scale 更准但更复杂.
- scale 的加载也要考虑 coalescing 和 cache reuse.

### Packed 低比特

FP4 / INT4 / NVFP4 路线常见成本:

- unpack 指令开销.
- scale 加载开销.
- bit packing alignment.
- epilogue register 压力.

只有当带宽节省和 Tensor Core / 专用路径收益超过 unpack 成本时, 低比特才真正快.

## 架构代际优化

### Volta

关注点:

- Tensor Core 开始成为 GEMM / Conv 的主路径.
- independent thread scheduling 改变了 warp 同步假设.
- 旧代码中依赖隐式 warp-synchronous 行为的写法需要显式同步.

### Turing

关注点:

- Tensor Core 支持更多低精度路径.
- INT8 / INT4 推理更有现实意义.
- 小模型推理要关注 launch overhead 和 fusion.

### Ampere

关注点:

- TF32 让 FP32 GEMM 可以走 Tensor Core 路径.
- BF16 支持更成熟.
- `cp.async` 支持 global 到 shared 的异步 copy.
- 结构化稀疏 Tensor Core 是特定模型和权重格式下的机会.

Ampere 上 GEMM / Conv 手写通常要认真调 `num_stages`, 因为 `cp.async` pipeline 的收益和 shared memory 占用是同一枚硬币的两面.

### Hopper

关注点:

- TMA 适合大块 tensor 搬运.
- WGMMA 把 MMA 协作提升到 warpgroup.
- thread block cluster 和 distributed shared memory 支持更大的协作范围.
- FP8 成为 Transformer kernel 的核心优化方向.
- warp specialization 更常见: producer warp 搬运, consumer warp 计算, epilogue warp 写回.

Hopper 上不要把 Ampere 的 tile 直接照搬. TMA / WGMMA / cluster 会改变最佳 tile 和 pipeline 形态.

### Blackwell

关注点:

- 更强的 Tensor Core 和更细的低精度路线.
- FP4 / NVFP4 的实用性上升.
- CUTLASS 示例中已经能看到面向 `sm100` 的新 MMA / tensor memory 组织.
- pipeline, tensor memory, warpgroup 和 epilogue 之间的资源平衡更重要.

Blackwell 路线应以官方 tuning guide, CUTLASS 示例和目标机器实测为准. 不要仅凭旧架构经验判断 tile.

## GEMM

GEMM 是手写算子的母题. `C = A x B` 的优化经验会迁移到 Linear, Conv, Attention 和低比特 dequant GEMM.

### GEMM 的层次

| 层次 | 例子 | 作用 |
| --- | --- | --- |
| CTA tile | `128 x 128 x 64` | 定义一个 block 负责的 C tile 和 K tile |
| warp / warpgroup tile | `64 x 64 x 64` | 定义 MMA 协作粒度 |
| instruction tile | `mma.sync` / `wgmma` shape | 对应 Tensor Core 指令 |
| thread fragment | 每个 lane 的 A/B/C fragment | 寄存器布局 |
| epilogue tile | C sub-tile | 写回和融合 |

### GEMM 优化顺序

1. 先用 cuBLAS / cuBLASLt 或 CUTLASS 作为性能上界.
2. 确认 layout 和 dtype 能触发 Tensor Core.
3. 选 CTA tile: 大 tile 提高复用, 小 tile 提高并发.
4. 选 K tile: 太小无法 amortize load, 太大增加 smem 和 latency.
5. 选 warp / warpgroup MMA: 匹配架构指令.
6. 设计 shared layout: 支持 `ldmatrix` / WGMMA, 避免 bank conflict.
7. 设计 async pipeline: `cp.async` / TMA stage 数.
8. 设计 epilogue: bias, activation, scale, quantize.
9. 调 scheduler: split-K, persistent, stream-K, grouped GEMM.

### GEMM 常见专项

- Skinny GEMM: `M` 或 `N` 很小, Tensor Core 利用和 launch overhead 都会变差, 常需要专门 kernel.
- Batched GEMM: small batch 时 grouped / persistent 调度很重要.
- Split-K: K 很大时提高并行度, 但需要 reduction, 会增加写回和同步成本.
- Stream-K: 用更细粒度的 K 分片改善负载均衡, 适合不规则 shape.
- Persistent GEMM: CTA 常驻 SM, 从 work queue 取 tile, 适合小 batch 或 grouped shape.
- Dequant GEMM: 权重 packed load, unpack, scale, MMA, epilogue 必须融合.

## Linear

Linear 本质是 GEMM 加 epilogue:

```text
Y = X W^T + bias
```

Transformer 中 Linear 的优化重点:

- batch 和 sequence 维展开后变成大 GEMM.
- QKV projection 可合并成一个更宽的 GEMM.
- MLP gate / up projection 可合并或共享输入 load.
- bias, activation, residual, quantize 可以放进 epilogue.
- 权重量化时优先做 dequant GEMM fusion.

常见策略:

| 场景 | 建议 |
| --- | --- |
| 大 batch / 大 hidden | 用 CUTLASS / cuBLASLt 上界, 手写 epilogue fusion |
| 小 batch / decode | persistent / grouped GEMM, 减少 launch |
| INT8 / FP8 weight-only | packed weight + scale 在 mainloop 或 epilogue 融合 |
| MoE grouped expert | grouped GEMM, 按 expert token 数排序或 bucket |
| gated MLP | `linear -> activation -> multiply` 尽量融合 |

Linear 不要单独优化成 "矩阵乘法" 后又把 bias / activation / quantization 拆出去. 端到端通常输在中间张量.

## Conv

Conv 的手写路径不止一种:

| 路线 | 适合 | 风险 |
| --- | --- | --- |
| direct conv | 小 kernel, 特定 shape | 泛化复杂, Tensor Core 映射难 |
| im2col + GEMM | 简单通用 | 中间矩阵大, 带宽浪费 |
| implicit GEMM | 通用高性能 | index / predication 复杂 |
| Winograd | 3x3 小 kernel | 数值和 transform 开销 |
| FFT | 大 kernel | transform 开销和 workspace |
| depthwise special | depthwise / group conv | FLOPs 少但内存占比高 |

### Conv 优化重点

- Layout: NHWC 常比 NCHW 更适合 Tensor Core conv 路径.
- Implicit GEMM: 不实际 materialize `im2col`, 而是在 tile iterator 中计算 input 坐标.
- Filter reuse: 权重 tile 应尽量复用.
- Output tile: 同时考虑 N, H, W, C 的映射.
- Padding / stride / dilation: 边界 predication 不要造成严重 divergence.
- Group / depthwise: 算术强度低, 更像 memory-bound pointwise + reduction.

### Conv 常见反直觉

- depthwise conv FLOPs 很低, 但不一定快, 因为复用少, 内存和 launch 占比高.
- 大量小 conv 不适合每层一个 kernel, fusion 和 graph 可能比单 kernel 微调更重要.
- `im2col` 简单但可能把输入放大 `R x S` 倍, 对内存带宽不友好.

## Attention

Attention 的瓶颈通常不是单个 matmul, 而是 QK, softmax, PV 和中间矩阵的 IO.

朴素实现:

```text
S = Q K^T
P = softmax(S)
O = P V
```

高性能实现避免把 `S` 和 `P` 全量写回 HBM. FlashAttention 的核心思想是 block 化处理 K/V, 用 online softmax 维护每个 Q block 的 max 和 sum, 只保留必要状态.

### Attention 优化重点

- IO-aware tiling: Q, K, V 按 block 进入 shared / register.
- Online softmax: 分块累积 max, sum 和 output.
- Work partition: head, batch, sequence block 的并行映射.
- Causal mask: 分支和边界处理要避免 warp divergence.
- GQA / MQA: K/V head 复用改变内存访问和 tile 分配.
- Decode attention: `seq_q=1` 时 GEMM 路线不一定适合, 需要专门 kernel.
- FP8 attention: scale, accumulation 和 softmax 稳定性必须一起设计.

### FlashAttention 代际理解

- FlashAttention v1: 重点是 IO-aware exact attention, 避免 `S` / `P` 大中间张量.
- FlashAttention-2: 重点是更好的并行划分和减少非矩阵乘法部分的开销.
- FlashAttention-3: 面向 Hopper 的 asynchrony 和低精度路径, 更强调 TMA / WGMMA / warp specialization.

写 attention kernel 时, 不要只看 QK matmul TFLOP/s. Softmax, scale, mask, dropout, PV, output store 共同决定端到端.

## Norm

Norm 包括 LayerNorm, RMSNorm, GroupNorm 等. 它们通常是 memory-bound + reduction-bound.

### LayerNorm

```text
mean = sum(x) / H
var = sum((x - mean)^2) / H
y = (x - mean) * rsqrt(var + eps) * gamma + beta
```

优化重点:

- 每行一个 block 或多个 warp.
- 小 hidden 用 warp reduction, 大 hidden 用 block reduction.
- vectorized load / store.
- mean / variance 可用两 pass 或 Welford.
- gamma / beta 读访问要合并.
- residual add, bias, dropout 可以融合, 但注意 register.

### RMSNorm

```text
rms = sqrt(sum(x^2) / H + eps)
y = x / rms * weight
```

RMSNorm 比 LayerNorm 少 mean 和 beta, 更适合融合:

- residual + RMSNorm.
- RMSNorm + quantize.
- RMSNorm + matmul 前处理.

### Norm 调优判断

| hidden size | 常见策略 |
| --- | --- |
| 很小 | 一个 warp 处理一行, 减少 shared memory |
| 中等 | 一个 CTA 处理一行, warp reduction + shared reduction |
| 很大 | 多 CTA 分块 reduction, 或拆成两阶段 |
| decode 小 batch | fusion 和 CUDA Graph 优先 |

Norm 的性能上限常由内存带宽决定. 算术指令优化收益有限, 访存合并和融合更重要.

## Warp specialization

Warp specialization 指同一个 CTA / warpgroup 内不同 warp 做不同角色:

- producer warp: 负责 global -> shared 的异步搬运.
- consumer warp: 负责 MMA / compute.
- epilogue warp: 负责转换和写回.

适合场景:

- load 和 compute 阶段都重, 且可以重叠.
- TMA / WGMMA 让 producer / consumer 分离更自然.
- 大 tile, 长 K loop, 数据复用明显.

不适合场景:

- 小 shape, launch-bound.
- 访存只用一次, staging 不划算.
- register / shared memory 已经接近上限.
- kernel 需要高度泛化, shape 变化大.

调优要点:

- producer warp 数不是越多越好, 会挤占 compute warp.
- barrier 和 wait 位置决定 pipeline bubble.
- stage 数增加可能导致 shared memory 占用过大.
- epilogue 如果太重, compute 和 store 会失衡.

## 方寸优化: 局部微调清单

这里把 "方寸优化" 理解为 kernel 内小范围但高频生效的细节优化.

| 细节 | 说明 |
| --- | --- |
| alignment | 输入输出指针对齐到 vector load 要求 |
| vector width | `half2`, `float4`, `uint4` 等向量化访问 |
| predication | 边界用 predicate, 避免复杂 branch |
| loop unroll | 提高 ILP, 但控制 register |
| constexpr shape | 热点 shape 用编译期常量专门化 |
| fast math | 仅在误差允许时使用近似函数 |
| restrict | 指明无 alias, 帮助编译器 |
| launch bounds | 控制寄存器和 occupancy, 需实测 |
| swizzle | 减少 shared bank conflict 或提升 L2 reuse |
| CTA order | 改善 cache locality 和负载均衡 |
| epilogue order | 先 scale 后 activation 或先 activation 后 scale 会影响误差和寄存器 |
| mask packing | dropout / causal / padding mask 尽量压缩和合并访问 |

这类优化每项收益可能很小, 但在热点 kernel 中可以累积. 前提是每一项都要单独测.

## Autotuning 参数

手写算子应把以下参数纳入搜索空间:

- CTA tile: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`.
- warp 数: 4, 8, 16 等.
- pipeline stage: 2, 3, 4 或更多.
- vector width: 1, 2, 4, 8 elements.
- shared layout swizzle.
- MMA instruction shape.
- split-K / stream-K 策略.
- CTA swizzle / rasterization.
- persistent scheduler 开关.
- epilogue fusion 开关.
- precision 和 accumulate dtype.

Autotuning 不能只记录最快 latency. 至少记录:

- GPU 名称和 `sm_*`.
- driver, CUDA, compiler, CUTLASS / CuTe 版本.
- shape, dtype, layout.
- warmup, repeat, sync mode.
- correctness tolerance.
- 编译时间和失败配置.

## Profiling 指标

### `nsys`

用来回答:

- kernel launch 是否太多.
- CPU 是否在等待 GPU.
- 是否有意外 memcpy.
- 是否有 host sync.
- CUDA Graph 是否覆盖正确范围.
- 多 stream 是否真正并发.

### `ncu`

用来回答:

- SM busy 和 Tensor Core 利用如何.
- DRAM / L2 / shared throughput 如何.
- achieved occupancy 和 theoretical occupancy 差异.
- eligible warps 是否不足.
- warp stall 原因是什么.
- register spill 是否出现.
- shared bank conflict 是否严重.
- roofline 位置是 compute-bound 还是 memory-bound.

不要把单个指标当结论. 例如 low occupancy 可能是问题, 也可能是高效 Tensor Core kernel 的正常结果.

## 正确性验证

最低要求:

- 与 PyTorch / cuBLAS / cuDNN reference 对比.
- 检查多组 shape, 包括非整除边界.
- 检查不同 stride / layout.
- 检查不同 dtype.
- 检查 deterministic 或随机路径.
- 检查 NaN / Inf / overflow.

常用指标:

- `max_abs_error`
- `max_relative_error`
- `mean_abs_error`
- `relative_l2`
- cosine similarity
- ULP error
- 下游任务指标或模型输出差异

低精度 kernel 不能只看单个 random tensor. 要覆盖极值, outlier, 近零值和真实模型分布.

## 算子专项速查

| 算子 | 第一优先级 | 第二优先级 | 常见陷阱 |
| --- | --- | --- | --- |
| GEMM | Tensor Core mapping | pipeline + epilogue | tile 泛化过度, epilogue register 爆 |
| Linear | GEMM + epilogue fusion | grouped / persistent | bias / activation 拆分导致中间写回 |
| Conv | implicit GEMM / layout | filter reuse / boundary | im2col 中间矩阵过大 |
| Attention | IO-aware tiling | online softmax / work partition | 只优化 QK, 忽略 softmax 和 PV |
| Norm | vectorized memory + reduction | fusion | reduction 同步过多, hidden size 泛化过度 |

## 什么时候不要手写

以下情况优先用成熟库:

- cuBLAS / cuBLASLt 已经接近峰值, 只是需要普通 GEMM.
- cuDNN 已覆盖目标 conv / norm / attention 且性能足够.
- shape 太多太散, 手写 kernel 维护成本过高.
- 算子不是热点.
- 性能瓶颈在 runtime / graph / data movement, 不是 kernel.
- 没有目标 GPU 实测环境.

手写算子的价值在于明确热点, 明确 shape, 明确融合收益, 明确库无法覆盖的特殊路径.

## XQT 落地建议

在 XQT 中推进手写算子时, 推荐按以下顺序:

1. 在 `operator` stage 建立 PyTorch reference 和 benchmark artifact.
2. 记录 backend, pattern, shape, dtype, device, `sm_*`.
3. 用现有 TileLang / Triton / CUTLASS / CuTe DSL adapter 能力先做 metadata 和 smoke.
4. 对单个热点 pattern 建立 microbenchmark.
5. 用 `nsys` 判定 runtime 层是否已经干净.
6. 用 `ncu` 定位 kernel 内瓶颈.
7. 每次只改一个 kernel 维度.
8. 把 latency, correctness, profiler evidence 写入 report.
9. 如果 microbenchmark 快但 operator-stage 慢, 先查 wrapper, materialization, CUDA Graph 和 capture scope.
10. 只有当证据证明是 kernel 问题时, 才继续 tile / warp / pipeline 调优.

本文提到的高级机制, 例如 WGMMA, TMA, warp specialization, cluster, FP4 / NVFP4 packed GEMM, 在 XQT 中应作为 architecture-sensitive 的实验能力推进, 不应写成默认可用能力.

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [CuTe DSL API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api.html)
- [CUTLASS implicit GEMM convolution](https://docs.nvidia.com/cutlass/media/docs/cpp/implicit_gemm_convolution.html)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUB Documentation](https://nvidia.github.io/cccl/cub/)
- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/)
- [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
