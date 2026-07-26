# 推理优化记录

## 负责内容

本文沉淀 XQT 中已经落地并验证过的推理优化技巧. 每条记录把目标问题,测量证据,实现取舍,适用边界和可复用规则写在一起,使下一次写 kernel,选择 engine 或设计 runtime 路由时可以直接复用判断,而不是只依赖一次性聊天记录或代码注释.

不负责: 任务计划,未验证的研究假设,通用 profiler 教程,或替代 [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md) 的方法论. 阶段性尝试和失效的实验日志放在 `research/`;只有仍能指导实现的结论进入本文.

## 维护规则

在同一项改动中,出现以下任一情况时必须新增或更新记录:

- 新增可执行的推理 kernel,backend 或架构专用路径.
- 改变矩阵布局,weight prepack,cache 生命周期,量化/反量化位置或 fusion 边界.
- 改变 `engine=auto` 的选择条件或 fallback 策略.
- 调整 tile,warp,stage,pipeline 或 epilogue,并以性能数据决定取舍.

每条记录必须包含以下字段:

1. `目标`: operator/module,shape,dtype,device,`sm_*` 和精度契约.
2. `基线`: 被比较实现和公平的测量方法,warmup,repeat,同步边界.
3. `瓶颈与假设`: 只陈述有测量或代码依据的原因.
4. `实现`: kernel/layout/fusion/runtime 的实际改变,以及不在热路径做什么.
5. `结果`: 延迟,吞吐或内存的同条件比较,加上数值误差.
6. `适用边界`: shape,硬件,dtype,依赖和 fallback 行为.
7. `未采纳方案`: 已验证的退化,或明确尚未验证的候选.
8. `可复用规则`: 能迁移到下一个算子的简短判断规则.
9. `验证落点`: 源码,测试,benchmark artifact 或可复跑命令.

不要把没有运行的候选写成失败,不要把 metadata-only backend 写成已执行,也不要把单次 host-wall 时间写成 kernel latency.

---

## R-001: ConvRot W8A8,TileLang front + CUDA/CUTLASS GEMM

### 目标

| 项 | 值 |
| --- | --- |
| module | `ConvRotInt8Linear`,静态 activation scale |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER,`sm_89` |
| shape | `M=256`,`K=4096`,`N=4096`,`rot_size=256` |
| operands | A INT8,W INT8,INT32 accumulation,FP16 output |
| 前端 | TileLang group-wise Hadamard rotation + static activation quant |
| GEMM 候选 | Triton W8A8 对比 CUTLASS `cuda_sm89` W8A8 |

### 基线与方法

每个候选先 warmup 20 次. 之后以 CUDA event 包围 100 个连续 forward,重复 15 轮并取每轮平均值的中位数. 两条路径共享同一输入,相同的静态 activation scale,相同的已量化权重和 bias. 这排除了 Python host-wall 时间以及单次 event 分辨率的干扰.

### 实现与取舍

1. 保留 TileLang 的 `groupwise_hadamard_static_quantize_tilelang`,让旋转和量化对每个 activation 元素只执行一次.
2. 将数学布局的 `qweight_t [K,N]` 一次性预打包为 contiguous `[N,K]`. CUTLASS 以 column-major `KxN` 解释该 buffer,沿 K 连续读取 B fragment,避免运行期转置与跨步 gather.
3. CUTLASS 使用 `64x128x64`,8 warp 的 INT8 Tensor Core mainloop. 对目标 shape,它产生 `4 x 32 = 128` 个 CTA;先前的 `128x256x64` 只产生 `2 x 16 = 32` 个 CTA,并行度不足.
4. 在同一个 CUTLASS epilogue 内计算 `int32_acc * (activation_scale * weight_scale[n]) + bias[n]`,直接写 FP16. `[N,2] float32` 的 scale+bias 向量按层缓存.
5. 不把 rotation/quant 强行融合进每个 GEMM CTA. N 维多 tile 会重复量化同一 activation,增加工作量且破坏前端 kernel 的简单访存模式.

### 结果

| 边界 | `cuda_sm89` | Triton | 加速 |
| --- | ---: | ---: | ---: |
| W8A8 GEMM-only,含 scale+bias epilogue | `0.03892 ms` | `0.05174 ms` | `1.329x` |
| static Linear,TileLang quant + GEMM | `0.04728 ms` | `0.06313 ms` | `1.335x` |
| 完整 ConvRot,rotation + quant + GEMM | `0.10030 ms` | `0.13080 ms` | `1.304x` |

数值验证:

- static Linear 对 `torch._int_mm` 加 scale+bias 参考的最大绝对误差为 `0.000244140625`.
- 完整 ConvRot 的 CUDA 和 Triton 路径最大绝对差为 `0.0009765625`.
- 编译产物 SASS 含 `IMMA.16832.S8.S8.SAT`,确认运行的是 INT8 Tensor Core,不是 FP16 fallback.

### 适用边界与回退

`cuda_sm89` 自动路由仅在以下条件全部成立时启用:

- CUDA `sm_89`.
- FP16 output,静态 activation scale.
- `K % 32 == 0`,`N % 8 == 0`,`M >= 32`.
- `int8mma_sm89.so` 已由 [build_and_test_int8mma.py](../../../xqt/operator_opt/kernels/cute/build_and_test_int8mma.py) 构建.

不满足时,ConvRot 专用静态前端继续使用 Triton GEMM;通用 `Int8MmaLinear` 保留 Triton,TileLang,PTX 和 INT8 reference 后备. `M=1` 保留 DP4A GEMV 路径,不由该 tile 接管.

### 可复用规则

- 当 B fragment 需要沿 K 连续读取,而数学权重是 `[K,N]` row-major 时,先评估一次性 `[N,K]` prepack,不要在每个 forward 或每个 CTA 转置.
- 用目标 shape 的 CTA 网格数量检查 tile. 大 tile 不一定更快;当 M 较小且 GPU SM 数较多时,更小 M/N tile 的并行度可能更重要.
- 对常量 scale/bias/layout 建 layer cache,但 cache key 必须覆盖权重/scale/bias 的版本和 device,防止 `load_state_dict()` 后静默复用旧数据.
- 在决定 megakernel 之前,计算被融合阶段是否会随输出 tile 重复. 只应融合每个输入/输出元素恰好一次的工作.
- 先分别测前端,GEMM-only 和完整 operator. GEMM-only 胜出不等于端到端胜出.

### 验证落点

- [CUTLASS kernel](../../../xqt/operator_opt/kernels/cute/int8mma_kernel.cu)
- [CUDA binding](../../../xqt/operator_opt/kernels/cute/int8mma_binding.py)
- [runtime auto routing](../../../xqt/runtime/modules/int8_mma_linear.py)
- [ConvRot specialized routing](../../../xqt/quant/quantizers/convrot_int8.py)
- [prepack tests](../../../tests/xqt/operator_opt/test_prepack_int8_sm89.py)
- [ConvRot CUDA routing test](../../../tests/xqt/quant/test_convrot_int8_quantizer.py)

