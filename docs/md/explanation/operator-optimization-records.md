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

## R-002: XQT GEMM SM89 dense 与 W8A8 correctness gate

### 目标

把首批自研 `xqt.gemm` CUDA 产物接入统一 contract,并验证 `sm_89` 上的 FP16/BF16 dense 与 INT8 MMA 路径. 逻辑布局固定为 `A[M,K]`,`W[N,K]`,`Y[M,N]`; dense 使用 FP32 accumulator,INT8 使用 `INT8 x INT8 -> INT32` 后乘 scale 写 FP16.

### 基线与方法

设备是 NVIDIA GeForce RTX 4080 SUPER (`sm_89`),CUDA 13.0,CUTLASS 4.6.1. 每个 callable 先 warmup 20 次,再用 CUDA event 测量 15 次,每次 1 个 iteration,并在 event end 后同步. dense baseline 是 Torch `A @ W.T + bias`; W8A8 baseline 是 `torch._int_mm` 加相同的 activation/weight scale. INT8 weight 在计时前一次性 prepack,不把 prepack 成本混进 GEMM latency.

### 实现

1. `dense_sm89.cu` 使用 CUTLASS `128x128x32`,4 warp,3 stage,FP32 accumulator. Python adapter 对 `M/N/K` 按 8 对齐,只在确实需要时 padding,执行后 slice 回 logical output; bias/residual 通过 beta source 传入.
2. `int8mma_kernel.cu` 通过 `sm89.py` 适配 canonical `[N,K]` 与 `sm89_int8_nk_v1` prepack. per-tensor activation scale 走 fused CUTLASS scale+bias epilogue; per-token scale 走 INT32 cuBLASLt 结果后独立 scale.
3. 编译产物先写 `metadata_only` sidecar. 只有 manifest 中有 `correctness_verified` 和具体 cases/evidence,且 artifact 可加载时,`install_*_executors` 才能把 registry entry promotion 为 `executable`.

### 结果

| 路径 | shape | native median | Torch/cuBLAS baseline | max abs error |
| --- | --- | ---: | ---: | ---: |
| dense FP16 | `128x128x64` | `0.074752 ms` | `0.029536 ms` | `0.015625` |
| dense FP16 + padding/slice | `127x129x96` | `0.104096 ms` | `0.028672 ms` | `0.015625` |
| dense BF16 | `128x128x64` | `0.073024 ms` | `0.028768 ms` | `0.125` |
| dense BF16 + padding/slice | `127x129x96` | `0.105152 ms` | `0.028576 ms` | `0.125` |
| W8A8 static per-tensor | `128x128x64` | `0.076800 ms` | `0.047104 ms` | `3.05e-5` |
| W8A8 dynamic per-token | `128x128x96` | `0.109568 ms` | `0.047776 ms` | `6.10e-5` |
| W8A8 static + padding/slice | `127x129x96` | `0.139264 ms` | `0.047936 ms` | `1.53e-5` |
| W8A8 dynamic + padding/slice | `127x129x96` | `0.182048 ms` | `0.047104 ms` | `3.05e-5` |

dynamic activation quantization 的拆分测量为: quantize `0.119040 ms`,预量化 INT8 GEMM `0.113376 ms`,两者端到端 `0.266240 ms`(同样 warmup/repeat). 端到端高于两者简单相加,说明当前 separate launch 的同步和临时 tensor 成本不可忽略; fused dynamic quant + GEMM 仍是后续 kernel 研究项.

decode 对比也覆盖了 `M=1/4/8`. `M=1` 使用 DP4A GEMV,最大误差为 `0`,CUDA event median `0.070144 ms`;同一组已预物化 FP16 权重/activation 的 dense matmul 为 `0.014112 ms`. `M=4/8` 明确回退 `quantized_dequant_reference`,分别为 `0.068608`/`0.068576 ms`,没有把大 MMA tile 强行用于 decode.

这里的 `executable` 只表示 ABI,运行和数值 gate 已通过,不表示当前 shape 已经是默认性能赢家. 本批次的小矩阵上 native 仍慢于 Torch/cuBLAS;默认 dispatch 的 autotune,workspace reuse 和更合适的 small-M tile 仍未完成,不能把这些数字宣传为加速.

### 适用边界与回退

- dense adapter 要求 `sm_89`,输入和权重 dtype 相同,`M >= 8`;不支持的 activation epilogue 走 reference.
- dense 非对齐 shape 会 padding 后 slice,report 记录 `padding_ratio`;对齐 shape 不再执行无意义的 `F.pad` copy.
- W8A8 native 主路径要求对称 INT8, per-channel weight scale,per-tensor 或 per-token activation scale,FP16/BF16 output. zero point,residual 和 fused activation 暂不支持.
- `M=1` 使用 DP4A GEMV;`M=2..31` 暂时保持 reference fallback. 当前 M=1 仍慢于已预物化的 FP16 baseline,因此该分支是 correctness/低比特实验路径,不是已证明的 decode 加速.
- INT8 非对齐 shape 先按 `(M,N,K)=(16,16,32)` padding 后 native 执行并 slice;如果 padding 后仍不满足 cuBLASLt/CUTLASS 条件,dispatch 才捕获 `XQTBackendError`,返回 `quantized_dequant_reference` 并写 `fallback_reason`.

### 未采纳方案

- 没有把 INT8 padding 误写成 kernel 内部自动消除成本. 当前 adapter 已验证 padding + slice 正确,但每次调用仍可能创建临时 padded tensor;prepack/workspace 复用仍待后续优化.
- 没有把 dynamic per-token quantization 声称为 fused native. 目前 quantize 和 GEMM 分离,需要单独 benchmark quant,INT8 GEMM 和端到端成本后再决定是否写 fused kernel.
- 没有因为 shared object 能被 `ctypes.CDLL` 加载就自动升级 registry;manifest correctness gate 是必要条件.

### 可复用规则

- "可运行"与"值得默认选择"是两个 gate. correctness manifest 只负责前者,shape bucket 的 CUDA event benchmark 和 tuning cache 决定后者.
- padding 应在 prepack 或 workspace 生命周期内复用;不能把每次 forward 的 `pad/contiguous` 当作 kernel 性能.
- per-token scale 若无法进入 fused epilogue,先保留 INT32 accumulator,再做独立 scale,并在 report 中拆出动态量化和 GEMM 时间.

### 验证落点

- [SM89 dense source](../../../xqt/gemm/backends/dense_sm89.cu)
- [SM89 dense adapter](../../../xqt/gemm/backends/dense_sm89.py)
- [SM89 INT8 adapter](../../../xqt/gemm/backends/sm89.py)
- [artifact manifest gate](../../../xqt/gemm/preflight.py)
- [CUDA event benchmark](../../../xqt/gemm/benchmark.py)
- [GEMM tests](../../../tests/xqt/gemm)

## R-003: W4A16 GPTQ/AWQ canonical repack

### 目标与实现

为 W4A16 统一 external format 与 kernel format. `repack_gptq_int4` 解码 GPTQ
`int32 [ceil(K/8),N]` words 为 signed codes,`repack_awq_int4` 解码 AWQ
`int32 [K,ceil(N/8)]` 的 reverse nibble order 为 unsigned codes. 两者都输出
`xqt_int4_nk_v1` packed `[N,ceil(padded_K/2)]`,scale/zero point 统一为
`[N,G]`,并把 group size 32/64/128 与 `padded_k` 写入 `PackedWeightMetadata`.

### 结果与边界

CPU synthetic fixture 覆盖 `N=8,K=33,group_size=32`: GPTQ signed 和 AWQ
asymmetric zero point 的 dequant reference 都逐元素一致. sequential `g_idx`
通过;non-sequential act-order 显式抛错,不会静默改变 K 列语义. 当前仅完成
canonical/repack/reference,还没有把它标成 CUDA W4A16 native.

### 验证落点

- [canonical pack](../../../xqt/gemm/layout.py)
- [W4 pack tests](../../../tests/xqt/gemm/test_w4_pack.py)

---

## R-003: TileLang Marlin INT4 dequant GPU smoke (U10)

### 目标

| 项 | 值 |
| --- | --- |
| module | `linear_marlin_tilelang` INT4 packed weight dequant GEMM |
| GPU | 有 CUDA 且 TileLang runtime usable 时运行; 否则 skip |
| shape | `M=32`, `K=64`, `N=64`, `group_size=64` |
| operands | A FP16, W INT4 packed + scale, FP16 out |

### 基线与方法

- 对照: `linear_marlin_reference` 同 shape 同 quant 输入.
- 正确性: `allclose(atol=2e-2, rtol=2e-2)`.
- 延迟: host `perf_counter` 包 10 次 forward (warmup 3), 仅证明路径可测, **不**作加速宣传.

### 结果 (本机抽样, 2026-07-29)

- RTX 4080 SUPER + TileLang 0.1.12: correctness max_abs ~0 (同 seed 下参考对齐), latency smoke ~0.09 ms/iter (host wall, 非 CUDA event 权威).
- 无 CUDA / 无兼容 TileLang: 测试 `pytest.skip`, capability 仍以 registry maturity 为准.

### 适用边界与回退

- 需要 `torch.cuda.is_available()` 且 `tilelang_runtime_usable()`.
- 默认 CI 不强制 GPU; 有硬件才跑 correctness + latency smoke.
- 不把 host-wall 单次数字写入 auto dispatch 默认赢家.

### 可复用规则

- GPU dequant 证据 = 正确性 gate + 可选 latency smoke; 无硬件 skip 不是失败.
- 收益宣传另需 CUDA event 与 fair baseline, 见 R-001 / R-002.

### 验证落点

- [linear_marlin kernel](../../../xqt/operator_opt/kernels/tilelang/linear_marlin.py)
- [GPU smoke test](../../../tests/xqt/test_tilelang_linear_marlin.py) (`test_linear_marlin_int4_gpu_correctness_and_latency_smoke`)
- [TileLang runtime guard](../../../xqt/operator_opt/kernels/tilelang/_common.py)

## R-004: XQT GEMM SM89 canonical W4A16 fallback gate

### 目标与实现

P2 为 canonical `xqt_int4_nk_v1` 增加了一个独立的 CUDA correctness
fallback. `xqt/gemm/backends/w4a16_sm89.cu` 在 K loop 直接从
`[N, ceil(padded_K/2)]` low/high nibble 读取,按 `K // group_size` 选择
`[N,G]` scale,并在需要时减去 `[N,G]` zero point. 它支持 FP16/BF16 A 和同
dtype output,可选 bias,不把完整权重展开成 FP16 buffer.

artifact 内部按 M 选择三个 shape variant: `M=1` 为每个 N 列一个 block
的 reduction GEMV,`M=2..8` 为每线程复用一列权重并累积多行的 small-M GEMM,
其余形状使用 8x16 输出 tile + 32-wide K tile 的 shared-memory kernel.
dispatch report 的
`shape_variant` 会写出实际分支 (`m1_gemv`,`small_m_2_8` 或
`tile_m_8x16x32`),不会把 decode GEMV 和 prefill tile 混为同一
个性能结论.

这个 artifact 的 kernel family 是 `w4a16_dequant_fallback`,而不是
`w4a16`. `sm89_w4a16_cutlass` 仍为 `metadata_only`: CUTLASS 4.6.1 的
SM80 mixed-input warp primitive 能将窄输入 upcast,但标准 `device::Gemm`
epilogue 无法正确表达每个 K group 的 scale,不能据此声称已经有 fused
W4A16 CUTLASS mainloop.

### 真实环境 gate

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER (`sm_89`) |
| CUDA / CUTLASS | CUDA 13.0 / CUTLASS 4.6.1 |
| artifact | `sm89_w4a16_dequant_fallback` |
| correctness | GPTQ signed 与 AWQ unsigned+zero-point, `M=1/2/4/8/32`,含非对齐 `N=257,K=1001,group_size=32` |
| ptxas resources | M=1: 22 registers + 1024 B shared; M=2..8: 40 registers; M>8 tile: 32 registers + 2560 B dynamic shared |
| CUDA event scope | aligned `1024x1024` 与 non-aligned `1003x1001`, group size 32/64/128, `M=1/2/8/32` |
| representative median | aligned `g=128`: M=1 `0.055936 ms`, M=2 `0.166912 ms`, M=8 `0.173568 ms`, M=32 `0.110496 ms` |
| reference median | 同一组: M=1 `0.161344 ms`, M=2 `0.162304 ms`, M=8 `0.164896 ms`, M=32 `0.168944 ms` |

artifact manifest 位于本机 cache 的 sidecar,并记录
`correctness_verified=true` 与 `kernel_role=dequant_fallback`. 这些数字只
证明 ABI,数值和可测性; M=1 decode 有明确收益,小 M 和 prefill 仍需按 shape
选择,不能把 fallback 当作 CUTLASS 性能证据.

### 边界与后续

- `validate_w4a16_packed_weight` 在 dispatch 前检查 storage layout,signedness,group size,packed shape,scale 和 zero point,不允许 GPTQ/AWQ 静默混用.
- fallback promotion 只替换 `sm89_w4a16_dequant_fallback` registry entry;
  不会替换 `sm89_w4a16_cutlass`.
- `mixed_input_probe_sm89.cu` 已编译并运行 canonical decoder. CUTLASS
  4.6.1 的标准 `device::Gemm` 可实例化 `int8 x int4 -> int32` mixed-input
  MMA,但没有 `fp16 x int4` 直接组合;因此这个证据不能升级 W4A16
  `sm89_w4a16_cutlass`.
- 下一步是设计真正的 SM89 fused groupwise mainloop: nibble decode 和 group
  scale 必须在 K tile 内完成,再接入 CUTLASS warp MMA 或自定义 CUTLASS
  primitive. 需要新的 SASS/correctness/event gate 后才能提升 maturity.

## R-005: SM89 W4A16 runtime resource 和 fallback ladder

### 目标

为 `sm89_w4a16_dequant_fallback` 补齐可审计的 runtime resource query 和
fallback 选择证据. 目标设备是 RTX 4080 SUPER (`sm_89`),形状为
`M=1/2/8/32,N=1024,K=1024,group_size=128`,另含 signed/unsigned 和
非对齐 `N=257,K=1001` correctness case.

### 基线与方法

数值基线是同一 `PackedWeight` 的 `reference_w4a16_gemm`. correctness 使用
30 个 case,FP16 输出,`atol=0.125,rtol=2e-2`. latency 使用
`benchmark_cuda_callables`,warmup 20,repeats 15,CUDA event 在同一当前 stream
上同步. resource query 调用 artifact 导出的 `cudaFuncGetAttributes` 和
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`,不发射伪造 kernel.

### 实现

1. `xqt_w4a16_sm89_resource_query` 为 `m1_gemv`,`small_m_2_8` 和
   `tile_m_8x16x32` 导出 registers,static/dynamic shared 和 active blocks/SM.
2. Python adapter 将查询结果序列化为 `Sm89W4A16ResourceReport`,并按
   `max_threads_per_sm` 计算理论线程 occupancy.
3. dispatch 按 registry priority 遍历 executable candidate;失败后继续
   alternate,最后进入匹配 reference. report 新增 `fallback_chain` 和
   `group_size`,并用最高优先级非 reference candidate 计算 padding ratio.
4. SM89 registry 显式保留 CUTLASS 主 tile,alternate tile,已验证 custom CUDA,
   planned Triton 和 reference 五层候选. metadata/planned 不会被当作执行路径.

### 结果

| variant | registers/thread | shared | active blocks/SM | theoretical occupancy |
| --- | ---: | ---: | ---: | ---: |
| `m1_gemv` | 22 | 1024 B static | 6 | 1.0 |
| `small_m_2_8` | 40 | 0 B | 12 | 1.0 |
| `tile_m_8x16x32` | 36 | 2560 B dynamic | 12 | 1.0 |

30 个 correctness case 的最大绝对误差为 `0.0625`,最大相对误差按
`abs(reference).clamp_min(1)` 计算为约 `9.73e-4`. 新一轮代表性 event median
为:

| M | custom CUDA | reference | 结论 |
| ---: | ---: | ---: | --- |
| 1 | `0.065600 ms` | `0.156672 ms` | custom 更快 |
| 2 | `0.195360 ms` | `0.158784 ms` | custom 更慢 |
| 8 | `0.203616 ms` | `0.158656 ms` | custom 更慢 |
| 32 | `0.141312 ms` | `0.160512 ms` | custom 更快 |

同一 `M=32` tile 的 persistent median 为 `0.252928 ms`,明显慢于
non-persistent `0.141312 ms`;因此 persistent 只保留为可复现候选,不进入默认
dispatch.

同一 `M=32` 权重在 256 MiB buffer thrash 前后的 median 为 `0.143232 ms` 和
`0.145600 ms` (`1.0165x`). 这只是 cache-sensitivity 信号,不是 L2 hit rate;
直接 counter 仍受 `ERR_NVGPUCTRPERM` 限制.

`nsys` CUDA timeline 已保存到
`/root/.cache/xqt/gemm/sm89/profiler/w4a16_m32_sm89_nsys.nsys-rep`. `ncu` 因
当前机器的 `ERR_NVGPUCTRPERM` 无法读取 hardware counter,manifest 明确记录
`blocked_permission`;没有把 nsys timeline 或 occupancy 数字推断成 L2/cache hit.

### 适用边界与回退

- 当前 executable 仍是 custom CUDA `dequant_fallback`,不是 CUTLASS fused
  W4A16 mainloop. `sm89_w4a16_cutlass_alt_tile` 仍为 `metadata_only`,Triton
  entry 为 `planned`.
- 任何 canonical nibble order,signedness,zero point 或 non-sequential `g_idx`
  不符合 contract 的输入,在 dispatch/repack 阶段硬失败,不进入 fallback.
- 小 M 不得引用 M=32 的性能结论. M=1 和 M=32 当前有收益,M=2/8 应保持
  reference 选择或等待新的 tile benchmark.

### 未采纳方案

- 不把 `ncu` 缺少权限时的推测 cache 数字写入 manifest.
- 不把 CUTLASS mixed-input probe 或 alternate metadata entry promotion 为
  native W4A16.
- 不在 fallback ladder 中重新 transpose 或展开完整 FP16 权重.

### 可复用规则

- runtime occupancy 必须来自加载后 kernel function 的 CUDA API query;ptxas
  静态资源只能作为编译侧交叉检查.
- fallback report 同时写 candidate chain 和 selected kernel,否则无法区分
  "没有 artifact" 与 "artifact 执行失败".
- profiler counter 不可用时应记录阻塞原因并保留可重复的外部命令,不以间接
  指标替代 cache 证据.

### 验证落点

- [W4A16 CUDA source](../../../xqt/gemm/backends/w4a16_sm89.cu)
- [W4A16 Python adapter](../../../xqt/gemm/backends/w4a16_sm89.py)
- [dispatch and registry tests](../../../tests/xqt/gemm/test_registry.py)
- [resource query test](../../../tests/xqt/gemm/test_sm89_backend.py)
- [reproducible evidence script](../../../research/xqt-gemm/bench_sm89_w4a16.py)

## R-006: SM89 fused groupwise W4A16 CUTLASS warp MMA

### 目标

验证 `[N,G]` group scale 是否能在 K tile 内完成后接入真实 Tensor Core MMA,
而不是把 packed weight 先完整 dequant 成 FP16 再调用 dense GEMM. 目标设备
为 RTX 4080 SUPER (`sm_89`),artifact 名为
`sm89_w4a16_cutlass_fused_mma`.

### 实现

`w4a16_cutlass_fused_sm89.cu` 每个 CTA 使用一个 warp 负责 `16x8` 输出 tile,
K tile 为 16. 线程协作把 activation 和当前 K tile 的 qweight nibble 解码到
shared memory;weight 在写入 shared 前读取 `[N,G]` scale 和可选 zero point.
随后调用 CUTLASS `DefaultMmaTensorOp<16x8x16>` 的 half Tensor Core warp
primitive,累积到 FP32 fragment,再按 CUTLASS row-major accumulator mapping 写
FP16 output 和 bias. 这是一条 `custom_cuda_cutlass_mma` artifact,不是标准
`cutlass::gemm::device::Gemm` INT4 entry;registry 的
`sm89_w4a16_cutlass_fused` 只有在 correctness manifest gate 后才 promotion.

### 真实环境 gate

| 项 | 值 |
| --- | --- |
| correctness | 30-case evidence plus formal GPU odd-K matrix: signed GPTQ canonical + unsigned AWQ zero-point, group size 32/64/128, M/N aligned and logical K includes odd `1001`,含 bias/no-bias |
| max abs error | `0.125` (`atol=0.125,rtol=2e-2`) |
| SASS | `HMMA.16816.F32` (`cuobjdump --dump-sass`) |
| fused median | `0.107200 ms` |
| custom dequant median | `0.147520 ms` |
| dense reference median | `0.167936 ms` |
| tile resource | 40 registers/thread,768 B shared,one warp |

fused sidecar 位于
`/root/.cache/xqt/gemm/sm89/w4a16_cutlass_fused_sm89.so.manifest.json`,其中
`maturity=executable`,`correctness_verified=true`,`cutlass_mainloop=true`,
`scale_application=k_tile_before_mma`. 这些字段只说明 artifact 可以安全实验;
默认 dispatcher 仍需要按 workload tuning 决定是否启用.

### split-K 与 M=1..8 decode 扩展

- split-K: 新增 `xqt_w4a16_cutlass_fused_sm89_fp16_run_splitk` ABI. partial
  kernel 由 `blockIdx.z` 选 K 段(按 16 tile 对齐分段,group scale/zero
  point 仍在 K tile decode 内逐元素应用,group 跨 split 边界语义不变),
  accumulator 以 FP32 写入 `[split_count, M, N]` workspace;第二个
  deterministic reduction kernel 按固定顺序求和,加一次 bias,写 FP16.
  弃用 atomicAdd 以保证 accumulation order 可复现. split-K 为 opt-in
  (`split_k>=2`),`split_k=1` 保持 full-K ABI;`select_fused_split_k` 提供
  形状启发式,选择值写入 manifest evidence. correctness 18 case
  (signed/unsigned x group 32/64/128 x 含 odd logical K=1001,`split_k=4`)
  max abs `0.0625`. prefill `M=32,N=1024,K=1024`: full-K `0.107462 ms`
  vs split-K(heuristic k=4) `0.083539 ms`(约 1.29x);workspace 512 KiB,
  extra write/read 1 MiB,净 latency overhead `-0.023923 ms`.
- decode: 新增 `xqt_w4a16_cutlass_fused_sm89_fp16_run_decode` ABI,替代旧
  `m1_gemv`/`small_m_2_8` decode 分支. SIMT 路径:一 block 一 column,256
  线程按 packed byte 步进(一次 byte 载两个 nibble),每线程持有 M 个
  accumulator 复用 decode 权重,warp shuffle + shared tree 归约,row
  bound 模板化为 1/2/4/8(M=1 不为多余行付 accumulator/shuffle 开销).
  nibble/scale 语义与 prefill mainloop 逐字节一致. correctness 24 case
  (signed/unsigned x group 32/64/128 x M=1/2/4/8,odd logical K=1001,对
  reference 和 padded MMA 双对照): max abs vs reference `0.00048828125`,
  vs padded MMA `0.0625`. decode benchmark `N=4096,K=4096`: M=1 fused
  `0.112486 ms` vs legacy m1_gemv `0.121318 ms`(约 1.08x);M=8 fused
  `0.161252 ms` vs legacy small_m `0.742701 ms`(约 4.6x).
- 已知限制: split-K 启发式未做 per-shape tuning(等 T035 tuning cache);
  decode kernel 拒绝 uint32 向量化(packed row 起点不保证 4 字节对齐);
  M=3/5/6/7 走向上取的模板界有少量空转;ncu 本机 blocked_permission,
  occupancy/带宽 counter 未补.

### 适用边界与回退

- 当前 fused entry 只接受 FP16 A/output,canonical `xqt_int4_nk_v1`,并支持
  signed/unsigned group scale. prefill MMA 路径要求 `M%16=0,N%8=0`;
  `M=1..8` 走 fused decode kernel.logical K 可以不是 16 的倍数,adapter
  会把 activation 零填充到 `ceil(K/16)*16`,且要求该 execution K 不超过
  `PackedWeight.metadata.padded_k`. 不满足对齐或 padded K 条件时回到
  `sm89_w4a16_dequant_fallback` 或 reference,不会静默改变 logical K.
- split-K 为 opt-in,不与 decode 组合(`M=1..8` 且 `split_k>=2` 会被显式
  拒绝). persistent scheduler 仍不在本 artifact 范围.
- `sm89_w4a16_cutlass` 标准 device GEMM entry 仍为 metadata-only,不能把
  这条 custom warp primitive 的证据外推为通用 CUTLASS INT4 support.

### 未采纳方案与规则

- 不把 `[N,G]` scale 放在最终 epilogue;scale 必须在 K tile 的 weight decode
  阶段应用.
- 不以 `int8 x int4 -> int32` mixed-input probe 代替 FP16 Tensor Core
  `HMMA` evidence.
- CUTLASS warp primitive 可以作为 custom mainloop 的执行核心,但 artifact
  名称,manifest metadata 和 registry implementation 必须明确区分
  `custom_cuda_cutlass_mma` 与标准 `device::Gemm`.

### 验证落点

- [fused CUDA mainloop](../../../xqt/gemm/backends/w4a16_cutlass_fused_sm89.cu)
- [fused adapter](../../../xqt/gemm/backends/w4a16_fused_sm89.py)
- [build and manifest gate](../../../xqt/gemm/backends/sm89_build.py)
- [fused evidence script](../../../research/xqt-gemm/bench_sm89_w4a16_fused.py)
- [GPU correctness test](../../../tests/xqt/gemm/test_sm89_backend.py)
- [alignment fallback test](../../../tests/xqt/gemm/test_registry.py)

## R-007: SM89 FP8 tensorwise CUTLASS native contract

### 目标

为 `xqt.gemm` 建立 FP8 tensorwise native 执行路径,并把 storage/value/scale
contract 与 registry maturity 分开记录. 目标设备为 RTX 4080 SUPER
(`sm_89`),CUDA 13.0,CUTLASS 4.6.1. 逻辑布局仍是 `A[M,K]`,
`W[N,K]`,`Y[M,N]`;storage layout 固定为 `xqt_fp8_rowmajor_v1`,每个元素
1 个 `uint8`,row-major.

### 基线与方法

数值基线是 `dequantize_fp8` 后执行 dense reference GEMM. correctness 覆盖
E4M3/E5M2,FP16/BF16 output,bias/no-bias,`M=1/32/256` 和非对齐
`K=33/65`. benchmark 使用 CUDA event,warmup 20,repeats 15,同一
`M=256,N=129,K=65` shape 分别比较 native `sm89_fp8_cutlass` 与
`fp8_dequant_reference`.

### 实现

1. `fp8.py` 固定 E4M3(FP8-E4M3FN) 与 E5M2 format spec,E4M3 max finite
   为 `448`,E5M2 max finite 为 `57344`,scale 统一为 FP32.
2. weight 支持 `weight_offline` 和 `weight_load_time`,activation 支持
   `activation_static` 和 `activation_dynamic`;FP8 禁止 zero point,weight
   必须 signed/symmetric.
3. `fp8_cutlass_probe_sm89.cu` 只作为 capability probe,manifest 保持
   `metadata_only`. 真实 `fp8_cutlass_sm89.cu` 编译为
   `fp8_cutlass_sm89.so`,只有 correctness evidence 写入 sidecar 后才
   promotion 为 `executable`.
4. 当前 native epilogue 只覆盖 `w:per_tensor/a:per_tensor`. per-channel
   weight scale 与 per-token activation scale 需要独立 row/column epilogue,
   不能通过隐式广播或 dequant+dense 伪装成 native FP8.

### 结果

| 路径 | output | native median | dequant reference median | max abs error |
| --- | --- | ---: | ---: | ---: |
| E4M3 | BF16 | `0.269312 ms` | `0.159744 ms` | `0.125` |
| E4M3 | FP16 | `0.278208 ms` | `0.163840 ms` | `0.125` |
| E5M2 | BF16 | `0.266240 ms` | `0.161792 ms` | `0.125` |
| E5M2 | FP16 | `0.283648 ms` | `0.161824 ms` | `0.125` |

12/12 correctness case 通过,全局 max abs error 为 `0.125`,max relative
error 为 `0.0078125`. SASS 同时包含
`QMMA.16832.F32.E4M3.E4M3` 与 `QMMA.16832.F32.E5M2.E5M2`,确认不是
FP8 dequant 后再走 dense BF16/FP16 GEMM. 当前 native 小矩阵仍慢于 dequant
reference,所以 `executable` 只表示可以安全实验和调优,不表示默认性能赢家.

### Profiling 归因

profiling 证据见 `research/xqt-gemm/profile_sm89_fp8.py` 和
`research/xqt-gemm/artifacts/sm89_fp8_profile.json`(shape
`M=256,N=129,K=128`,E4M3,BF16 output,30 iterations).

- torch.profiler: native 路径 CUDA total `731.9 us`,其中 CUTLASS kernel
  本身 `482.6 us / 30 = 16.1 us/launch`,只占约 66%. 其余是 adapter 开销:
  `aten::isfinite` `255.6 us`,`aten::item` 加 `Memcpy DtoH` `152.6 us`
  (`fp8_sm89.py` 每次调用对 scale 做 finite/正数检查后 `.item()`,强制
  DtoH 同步),`aten::abs` `124.5 us`,以及每次调用的 `F.pad`/`contiguous`
  对齐拷贝. reference 路径 CUDA total `775.7 us`,主要为 `aten::copy_`
  `311.9 us` 和 `aten::to`/`_to_copy` `252.8 us`.
- 结论: event benchmark 差距来自 adapter 侧同步与拷贝,不是 FP8 kernel
  本身. 优化方向是把 scale 校验移出热路径(quantize 时校验并 cache)和
  复用 padded buffer,而不是先改 mainloop tile.
- ncu: 本机 `blocked_permission`(ERR_NVGPUCTRPERM). 带
  `--kernel-name regex:.*cutlass.* --set detailed` 的可复跑命令已存进
  artifact,等有 counter 权限的机器再补 occupancy/memory/stall 数据;
  在此之前不猜测 cache 或 occupancy 结论.

### 适用边界与回退

- native 支持 E4M3/E5M2,FP32 accumulator,FP16/BF16 output,bias/no-bias,
  `w:per_tensor/a:per_tensor`.
- `w:per_channel/a:per_tensor`,`w:per_tensor/a:per_token` 和
  `w:per_channel/a:per_token` 保持 reference-only,直到 row/column scale
  epilogue 完成并重新通过 correctness/benchmark gate.
- SM89 entry 不能写成 SM90 WGMMA/TMA. SM90 后续需要独立 backend 和
  artifact manifest.
- 非 native FP8 或 artifact 不存在时回到 FP8 dequant reference,report 必须
  保留 selected kernel 和 fallback reason.

### 未采纳方案与规则

- 不把 probe artifact promotion 为 executable;probe 只证明指令路径可编译.
- 不把 per-channel/per-token scale 静默广播到 tensorwise native path.
- 不把当前 slower-than-reference 的 synthetic benchmark 写成模型加速.
- FP8 blockwise 的 K-block scale 必须进入 mainloop,不能用最终 scalar
  epilogue 替代.

### 验证落点

- [FP8 contract](../../../xqt/gemm/fp8.py)
- [FP8 reference](../../../xqt/gemm/reference.py)
- [SM89 FP8 native source](../../../xqt/gemm/backends/fp8_cutlass_sm89.cu)
- [SM89 FP8 adapter](../../../xqt/gemm/backends/fp8_sm89.py)
- [SM89 FP8 probe](../../../xqt/gemm/backends/fp8_cutlass_probe_sm89.cu)
- [FP8 evidence script](../../../research/xqt-gemm/bench_sm89_fp8.py)
- [FP8 profile script](../../../research/xqt-gemm/profile_sm89_fp8.py)
- [FP8 tests](../../../tests/xqt/gemm/test_fp8.py)
