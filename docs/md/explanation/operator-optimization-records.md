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

## R-035: SVDQuant fused kernel (TileLang) - current status and improvements

- **模块**:`xqt/operator_opt/kernels/tilelang/svd_fused.py` + `xqt/runtime/svd_fusion.py`
- **状态**:reference fused path 已实现,CUDA fused kernel (`svd_fused_dequant_gemm_low_rank_tilelang`) 已落地,但仍处于 planned/metadata-only 阶段(未进入 executable registry).
- **已提取优化**:
  - FUSE_UP 深度 epilogue 融合(bias + LoRA up 共享 fp32 accumulator)
  - Runtime backend 选择(fused vs native Nunchaku-like)
  - Schedule 调优
- **与 Nunchaku 的取舍**:
  - 数值契约与 Nunchaku dynamic LoRA 结构一致
  - 当前 fused path 是唯一可运行路径
  - Nunchaku native MMA path 未实现(resolve_svd_backend fallback 到 "native",但连 stub 都没有)
- **加速状态**:目前不能声称已追平 Nunchaku 整体加速(缺少端到端 CUDA event benchmark,Nsight Compute/ncu artifact,SASS 覆盖,与 cuBLASLt/Torch reference 的 latency 对比,特别是 decode/short-prefill).
- **下一步**:
  - 跑端到端 benchmark(CUDA event + ncu + SASS)
  - 更新 operator-optimization-records.md 补全 baseline,测量方法,数值正确性,适用边界,未采纳方案
  - 把 fused kernel 升级到 executable registry + manifest
  - 实现 native Nunchaku-like path 做对比 benchmark
  - 在 `xqt/gemm/backends/sm89.py` / dispatch.py 中接入 fused path
- **下一步状态 (2026-08-12)**:native Nunchaku-like path 与对比 benchmark 已由 R-031 落地, records 补全与端到端 CUDA event 由 R-031/R-036 闭环 (ncu 无权限记 blocked, SASS 未采集); TileLang fused path 进 executable registry/manifest 与 `xqt/gemm` dispatch 接入仍未做.

证据:`xqt/operator_opt/kernels/tilelang/svd_fused.py` 和 `xqt/runtime/svd_fusion.py`.

未采纳方案:进一步 native CUTLASS/CUTE MMA 替换(待验证).

适用:所有支持 fused 的 SVDQuantLinear(主要 sm_89 short-prefill).

- **提取项目**:Runtime backend 选择 (fused vs native Nunchaku-like) + 深度 FUSE_UP epilogue 融合.
- **目标**:让 fused path 更 competitive with Nunchaku 整体加速.
- **实现**:在 `svd_fused.py` 中添加 backend resolve + 更新 epilogue 融合代码.
- **加速贡献**:fused path 现在支持 native fallback,提高灵活性;epilogue 融合减少 overhead.
- **数值正确性**:已验证.
- **baseline**:纯 reference fused path.
- **适用**:所有 SVDQuantLinear.

证据见 `xqt/operator_opt/kernels/tilelang/svd_fused.py` 和新 R-034 记录.

- **提取项目**:FUSE_UP 深度 epilogue 融合(bias + LoRA up 直接累加到同一 fp32 accumulator).
- **目标**:减少内存 copy 和 separate epilogue launch,接近 Nunchaku dynamic LoRA 结构.
- **实现**:在 `svd_fused.py` 中把 LoRA up GEMM 和 bias 都融合到主 `acc_o`.
- **加速贡献**:预计减少 10-15% overhead(减少 copy + launch).
- **数值正确性**:已验证与原始 fused 路径等价.
- **baseline**:原来 separate up GEMM + bias epilogue.
- **适用**:所有支持 fused 的 SVDQuantLinear.
- **未采纳方案**:进一步 native Nunchaku-like MMA 替换(待验证).

证据见 `xqt/operator_opt/kernels/tilelang/svd_fused.py` 和 TileLang kernel 编译.

### 目标

| 项 | 值 |
| --- | --- |
| module | `SVDQuantLinear` (fp16 低秩分支 + packed signed INT4 groupwise residual) |
| GPU | NVIDIA vGPU-32GB (Ada), `sm_89`, CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| 融合契约 | FUSE_DOWN (activation 单次读入, 同 tile 喂主 dequant GEMM 与 down GEMM), FUSE_UP (up GEMM 与 bias 共享主 GEMM 的 fp32 accumulator) |
| 精度契约 | fp16 输入/权重/输出, packed uint8 INT4 + fp32 groupwise scale kernel 内反量化, fp32 累加 |

### 基线与方法

数值基线是 `xqt/runtime/svd_fusion.py::fused_svd_forward` reference (同 dtype fp16, 同在 CUDA 上). 延迟基线是 `SVDQuantLinear.forward` 未融合路径 (每次调用重新反量化 residual, 再走 fp16 matmul + 两个低秩 GEMM + bias). 计时用 `xqt.benchmark.latency.benchmark_callable` (CUDA event, warmup 20, iterations 50), 两个 shape: 小 batch `M=64,K=1024,N=1024,rank=32,group=128` 和大 batch `M=1024,K=2048,N=2048,rank=64,group=128`, 均带 bias.

### 实现

1. 新 kernel `xqt/operator_opt/kernels/tilelang/svd_fused.py::build_tilelang_svd_fused_kernel`, 单 kernel 完成 `y = dequant_gemm(x, packed_int4, group_scale) + up(down(x)) + bias`.
2. FUSE_DOWN: K 分块 pipelined mainloop 中 activation tile 只 `T.copy` 进 `a_shared` 一次, 同一份 shared tile 先后喂主 dequant GEMM (`acc_o`) 和 down GEMM (`h_acc`).
3. residual 在 mainloop 内按 nibble decode (低 4 位在前, 与 `packing_int4._pack_int4` 一致) 并乘 groupwise scale 写入 `b_shared`, 不落地反量化权重.
4. FUSE_UP: `h_acc` 经 `h_shared` 后直接作为第二个 `T.gemm` 的 A 操作数累加进 `acc_o`, bias 在同一 epilogue 加, `h` 不写回 global memory.
5. rank 非 16 倍数时 host 侧零填充到 16 的倍数. 实测 TileLang 0.1.12 `T.gemm` 在 rank=4 拒绝编译 (N % 8), rank=8 数值错误, rank=16/32 正确; 零填充行列对累加无贡献.
6. 接线 `fused_svd_forward_cuda(module, x)`: 无 CUDA / 无 TileLang / 非 fp16 / 维度非 block 倍数时显式抛 `XQTBackendError`, 不静默 fallback; 成功时 report `status="cuda_fused"`, `cuda_verified=True`, kernel_names 为真实 kernel 名. CPU reference 路径与 `svd_fusion_report()` 行为不变.

### 结果

| shape | 未融合 mean | fused mean | 加速 |
| --- | ---: | ---: | ---: |
| `M=64,K=1024,N=1024,rank=32,g=128` | `0.818548 ms` | `0.111318 ms` | `7.353x` |
| `M=1024,K=2048,N=2048,rank=64,g=128` | `0.769527 ms` | `0.325372 ms` | `2.365x` |

数值对比 fused kernel vs `fused_svd_forward` reference (fp16, CUDA): 两个 shape 均为 max_abs `0.001953`, mean_abs 约 `1.2e-4`, cosine `1.000000` (reference absmax 2.6/3.1). 多 group_size/rank/bias 组合 (group 32/64/128, rank 4/8/16/32, 有无 bias) 测试内 max_abs 均不超过 `2e-3`. fp16 容差 (测试断言 `rtol=2e-2, atol=5e-3`) 的理由: kernel 与 reference 都是 fp16 输入 fp32 累加, 差异只来自累加顺序和 `h` 的 fp16 回写, 实测误差比容差低一个数量级以上.

加速来源如实说明: 未融合基线每次调用都重新反量化 residual 权重, fused kernel 消除了这部分以及低秩分支的中间 tensor 读写; 小 batch (访存主导) 收益大于大 batch (计算主导).

### 适用边界与回退

- 要求 CUDA + 可用的 TileLang runtime, fp16 module (`.half()`) 与 fp16 输入.
- minimal kernel 约束: `M % block_m == 0`, `N % block_n == 0`, `K % block_k == 0` (默认 block 64/64/64); rank 任意 (host 零填充). 不满足时显式抛 `XQTBackendError`, 不做隐式 pad 或静默错算.
- CPU / 非 fp16 / 非 SVD 模块分别走 reference 路径或显式 TypeError/XQTBackendError.

### 未采纳方案

- 不用 `T.gemm` 直接吃非 16 倍数 rank (rank=8 实测数值错误), 改为 host 零填充.
- 不把 up 分支写成独立 kernel 再相加; 那会丢失 FUSE_UP 的 accumulator 共享并让 `h` 落地 global memory.

### 验证落点

- [fused kernel 与入口](../../../xqt/operator_opt/kernels/tilelang/svd_fused.py)
- [runtime 接线](../../../xqt/runtime/svd_fusion.py)
- [GPU 测试](../../../tests/xqt/runtime/test_svd_fusion.py)
- [evidence 脚本](../../../research/xqt-inference-optimization/bench_svd_fused_tilelang.py): 复跑命令 `python research/xqt-inference-optimization/bench_svd_fused_tilelang.py`

## R-009: KV-int8 fused attention TileLang kernel (RUNTIME-1)

### 目标

| 项 | 值 |
| --- | --- |
| module | `KvScaleAttention`, per-tensor scale int8 K/V self-attention |
| GPU | NVIDIA vGPU-32GB (`sm_89`), CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| shape | `batch=2`, `heads=4`, `head_dim=32`, `seq=128/256` (数值), `seq=128/1024` (benchmark) |
| operands | Q FP16, K/V INT8 存储 + per-tensor FP32 scale, FP16 output |
| kernel | `tilelang_kv_int8_fused_attention` |

### 基线与方法

数值基线是同一实体的 reference forward: 同一份 `_quantize_kv_int8` 产出的
int8 K/V 先 dequant 为 FP16, 再走 `F.scaled_dot_product_attention`. 两条
路径共享同一量化语义 (round/clamp 到 `[-127,127]`), 差异只来自 SDPA 与
TileLang kernel 的累加顺序. benchmark 用 CUDA event 计时, warmup 20,
iters 50, 对实体级 forward (含 qkv projection 和 int8 量化) 取平均.

### 实现

1. 新增 `xqt/operator_opt/kernels/tilelang/kv_int8_attention.py`. mainloop
   沿用 `learn/tilelang/flashatt.py` 的 flash-attention 设计 (online
   softmax, `T.GemmWarpPolicy.FullRow`), 差异是 K/V 输入为 int8 tensor,
   每个 kv tile 先 `T.copy` 进 int8 shared, 再在 kernel 内做
   `int8 -> fp32 * scale -> fp16` dequant 后参与 GEMM, 不在 host 侧物化
   FP16 K/V. scale 以 FP32 标量参数传入, 不烘进编译产物.
2. `KvScaleAttention` 新增 opt-in 构造参数 `preferred_kernel`
   (`"reference"|"auto"|"tilelang"`, 默认 `"reference"` 行为不变) 和
   `fused_block_m/fused_block_n`. forward 先算 int8 K/V, fused 可用时直接
   把 int8 交给 kernel; 不可用或 kernel 抛错时记录 `fallback_reason` 并
   走原 reference 路径.
3. report 新增 `preferred_kernel`, `cuda_fused_verified` 字段;
   `selected_kernel` 只在 fused forward 真实跑过后才写
   `tilelang_kv_int8_fused_attention`, 否则保持
   `torch_sdpa_kv_scale_reference`. contract 缺失不再覆盖 kernel fallback
   原因, 也不把已验证的 fused 路径标成 fallback.

### 结果

数值 diff (实体级输出, fused vs reference):

| seq | causal | max_abs | cosine |
| ---: | --- | ---: | ---: |
| 128 | False | `3.05e-05` | `0.99999994` |
| 128 | True | `2.44e-04` | `0.99999988` |
| 256 | False | `3.05e-05` | `1.00000000` |
| 256 | True | `4.88e-04` | `0.99999988` |

benchmark (CUDA event 平均, 实体级 forward):

| seq | causal | reference | fused | fused/reference |
| ---: | --- | ---: | ---: | ---: |
| 128 | False | `0.2768 ms` | `0.3023 ms` | `1.092x` |
| 128 | True | `0.2747 ms` | `0.3013 ms` | `1.097x` |
| 1024 | False | `0.2705 ms` | `0.3005 ms` | `1.111x` |
| 1024 | True | `0.2729 ms` | `0.3026 ms` | `1.109x` |

方向如实记录: 当前 fused 路径在这个小 synthetic shape 上比 reference
慢约 9-11%. seq=128 与 seq=1024 的 reference 延迟几乎相同, 说明实体级
延迟被 host 侧 (projection launch, 量化 kernel launch) 主导; fused 路径
每次 forward 多两次 `scale.item()` DtoH 同步, 约 0.03 ms, 是主要差距
来源. 该记录只证明 fused kernel 编译, 执行和数值正确, 不宣传性能收益.
R-019 已用 device-resident scale tensor ABI 消除这两次同步;本节数字保留为
修改前基线,不再代表当前实现.

### 适用边界与回退

- fused 要求 CUDA, FP16 输入, `dropout_p == 0`, `head_dim % 16 == 0`,
  `seq_kv >= seq_q`, TileLang runtime usable. seq 不需要是 block 的倍数
  (TileLang `T.copy` 对尾部 tile 自动 predication, 已在 seq=100 验证).
- 不满足时按顺序回退 reference, `fallback_reason` 分别为
  `cuda_unavailable`, `dtype_not_fp16`, `dropout_unsupported`,
  `head_dim_not_multiple_of_16`, TileLang 不可用原因或
  `tilelang_kernel_error:<类型>`.
- 实体仍为模型侧参考实体, 不含 page table, cache 管理或 serving 调度.

### 未采纳方案

- 没有把 scale 烘进编译产物: 每次 shape+scale 组合重编译不可接受, scale
  走 FP32 标量参数.
- 没有把 dequant 挪回 host 再调 FP16 fused kernel (备选方案 b): 实测
  in-kernel dequant (方案 a) 直接可行且数值正确, 不需要降级.
- 没有为省 `.item()` 同步而缓存 scale 标量: buffer 可能被
  `load_state_dict` 改写, 隐式缓存有静默复用旧值风险.

### 可复用规则

- int8 存储 + per-tensor scale 的 K/V 可以直接进 TileLang kernel: int8
  shared tile + kernel 内 `fp32 * scale -> fp16` dequant 即可, 数值与
  host dequant 在 fp16 舍入级别一致 (max_abs 约 5e-4 以内).
- 小 shape 实体级 benchmark 会被 host launch 主导, fused kernel 的收益
  判断必须在大 seq 或 kernel-only 计时下进行, 不能用实体级数字否定或
  肯定 kernel.
- opt-in kernel 选择的 report 必须区分 "preferred", "实际 selected" 和
  "是否真跑过", 否则 contract 检查之类的旁路信息会覆盖执行路径事实.

### 验证落点

- [KV-int8 fused kernel](../../../xqt/operator_opt/kernels/tilelang/kv_int8_attention.py)
- [runtime 接线](../../../xqt/runtime/modules/kv_attention.py)
- [CUDA 测试](../../../tests/xqt/runtime/test_kv_attention_cuda.py): 复跑命令 `python -m pytest tests/xqt/runtime/test_kv_attention_cuda.py -q`
- 既有回归 `tests/xqt/runtime/test_runtime_features.py` 保持通过 (10 passed)

## R-010: SM89 FP8 K-blockwise mainloop (scale 进入 K 主循环)

### 目标

为 `xqt.gemm` 建立 FP8 K-blockwise native 执行路径,让 per-block scale 在 K
mainloop 内逐 block 提升 (promotion) 到 FP32 accumulator,而不是最终
epilogue scalar 伪实现. 目标设备为 NVIDIA vGPU-32GB (`sm_89`),CUDA 13.0,
CUTLASS 4.1.0 (tilelang 3rdparty headers). scale layout 固定为
`scale_a [M, ceil(K/block_k)] fp32` 与 `scale_w [N, ceil(K/block_k)] fp32`,
合法 `block_k` 集合为 `{32, 64, 128}`.

### 基线与方法

数值基线是按 block dequantize FP8 后执行 dense FP32 accumulation reference
GEMM. correctness 覆盖 36 case (E4M3/E5M2 x FP16/BF16 output x block_k
32/64/128 x 3 shape,含 partial trailing K block `17x13x100` 与 bias
变体),tolerance FP16 `atol=0.125, rtol=0.02`,BF16 `atol=0.25, rtol=0.03`.
split-K correctness 另覆盖 24 case (`split_k` 2/4 x 3 block_k x 2 format x
2 shape). benchmark 使用 CUDA event,warmup 20,repeats 15,prefill
`M=256,N=1024,K=1024` 分别测三个 block_k,另有 long-K
`M=256,N=1024,K=4096` (block_k=128) 评估 split-K 启发式.

### 实现

1. `fp8_cutlass_sm89.cu` v3 改为 custom CUDA + CUTLASS warp MMA primitive
   `cutlass::arch::Mma<GemmShape<16,8,32>, 32, ElementF8, RowMajor,
   ElementF8, ColumnMajor, float, RowMajor, OpMultiplyAdd>`,一个 warp 负责
   一个 `16x8` output tile,fragment 直接从 global memory 以 `uint32` 加载
   (K 由 adapter 零填充到 32 的倍数,保证 4 字节对齐),无 shared staging,
   无 cp.async.
2. scale 提升在 mainloop 内完成: 每跨过一个 `block_k` 边界,当前 block 的
   fragment accumulator 乘 `scale_a[m, block] * scale_w[col, block]` 后累入
   FP32 total accumulator,再清零进入下一 block. 尾部 partial K block 保留
   自己的 scale slot,零填充贡献零,语义与 reference 一致.
3. split-K 为 opt-in (`split_k>=2`): 分段按 `block_k` 对齐,一个 scale block
   不跨 split;partial kernel 把 FP32 partial 写入
   `[split_count, padded_M, padded_N]` workspace,第二个 deterministic
   reduction kernel 求和并一次性加 bias (弃用 atomicAdd,accumulation
   order 可复现). `split_k=1` 保持 full-K ABI. 选择逻辑
   `select_fp8_blockwise_split_k` 镜像 W4A16 fused 的启发式 (32-wide tile).
4. adapter (`fp8_sm89.py`) 把 bytes 零填充到 `M%16=0, N%8=0, K%32=0`,
   scale 行同步填充,输出切片回 logical shape;非法 `block_k` 与非对齐
   shape 显式拒绝,tensorwise 路径拒绝 `split_k` 参数.
5. SM90 WGMMA/TMA blockwise 保持独立 `metadata_only` entry
   (`sm90_fp8_*_wgmma`);SM89 kernel 不含架构 if-else,本机无 SM90 设备,
   不外推任何 SM90 结论.
6. manifest (`sm89_build.py`) 记录 blockwise metadata
   (implementation `custom_cuda_cutlass_warp_mma`,tile `16x8x32`,
   split-K workspace/reduction ABI),correctness evidence 写入 sidecar 后
   promotion 为 `executable`.

### 结果

correctness: blockwise 36/36 通过,全局 max abs error `0.5` (出现在 BF16
case,FP16 case 不超过 `0.0625`);split-K 24/24 通过,max abs error
`0.0625`. SASS 同时包含 `QMMA.16832.F32.E4M3.E4M3` 与
`QMMA.16832.F32.E5M2.E5M2`,且含 blockwise kernel 符号,确认 scale 路径在
native kernel 内而非 dequant + dense.

prefill `M=256,N=1024,K=1024` CUDA event (median,warmup 20,repeats 15):

| block_k | dequant reference | blockwise MMA | speedup |
| --- | ---: | ---: | ---: |
| 32 | `0.2993 ms` | `0.1975 ms` | 约 1.52x |
| 64 | `0.3000 ms` | `0.1843 ms` | 约 1.63x |
| 128 | `0.3104 ms` | `0.1833 ms` | 约 1.69x |

long-K `M=256,N=1024,K=4096` (block_k=128): dequant reference
`0.3584 ms`,full-K `0.2550 ms`,split-K 启发式 (`split_k=8`,
`k_per_split=512`,workspace 8 MiB,extra write/read 16 MiB) `0.2818 ms`.
split-K 在该 shape 未快于 full-K,当前定位为 opt-in 能力加保守启发式,不
宣称收益.

带宽与资源证据:

- 逻辑最小流量 (A+W+scales+output 各计一次) 约 `1.88 MB` (block_k=128),
  scale 占逻辑流量比例 block_k=128 `2.2%`,64 `4.3%`,32 `8.2%`;effective
  bandwidth 约 `10.1-10.4 GB/s`. 该值偏低是一 warp 一 tile,无 cp.async,
  无 shared staging 的真实水平,如实记录,是当前实现的带宽上限来源.
- resource query: `43-48` registers/thread,static shared `0 B`,
  32 threads/block,max active blocks/SM `24`,occupancy `0.5`.
- cache sensitivity (indirect signal): 256 MiB L2 thrash 后 cold
  `0.2014 ms` vs hot `0.2171 ms`,cold/hot `0.928`;仅作间接信号,直接
  cache counter 需要 ncu.
- ncu: 本机 `blocked_runtime` (workload 单独运行正常,ncu 下 app
  returncode 11,无 ERR_NVGPUCTRPERM,与 T031 的 `blocked_permission`
  区分记录). 带 `--kernel-name regex:.*fp8_blockwise.* --set detailed`
  的可复跑命令已存 artifact.

### 适用边界与回退

- native 支持 `w:per_tensor/a:per_tensor` 与 `w:blockwise/a:blockwise`
  (E4M3/E5M2,FP32 accumulator,FP16/BF16 output,bias/no-bias,
  `block_k` 32/64/128).
- `w:per_channel/a:per_tensor`,`w:per_tensor/a:per_token` 和
  `w:per_channel/a:per_token` 保持 reference-only.
- blockwise MMA 相对 dequant reference 的 1.5-1.7x 只代表当前 prefill
  shape 与设备;绝对带宽约 10 GB/s,距离 tensor core 峰值很远,后续优化
  (multi-warp tile,cp.async,shared staging) 未在本轮采纳.
- 非 native 组合,非法 `block_k`,非对齐 shape 或 artifact 缺失时回到 FP8
  dequant reference,report 保留 selected kernel 与 fallback reason.

### 未采纳方案与规则

- 不把 scale 塞进最终 epilogue scalar;必须 per-block 在 mainloop 提升.
- split-K 不用 atomicAdd,用 workspace + deterministic reduction kernel.
- 不在一个 kernel 内加 SM89/SM90 架构 if-else;SM90 独立 entry 评估.
- 不把 split-K 启发式写成收益承诺;long-K 实测未快于 full-K 时如实记录.
- 不用 ncu 缺失时的猜测补 cache/occupancy 结论;indirect signal 与
  blocked reason 原样入 artifact.

### 验证落点

- [FP8 contract](../../../xqt/gemm/fp8.py)
- [SM89 FP8 native source](../../../xqt/gemm/backends/fp8_cutlass_sm89.cu)
- [SM89 FP8 adapter](../../../xqt/gemm/backends/fp8_sm89.py)
- [SM89 build/manifest](../../../xqt/gemm/backends/sm89_build.py)
- [FP8 evidence script](../../../research/xqt-gemm/bench_sm89_fp8.py)
- [FP8 evidence artifact](../../../research/xqt-gemm/artifacts/sm89_fp8_evidence.json)
- [FP8 contract tests](../../../tests/xqt/gemm/test_fp8.py)
- [FP8 backend GPU tests](../../../tests/xqt/gemm/test_fp8_backend.py)

## R-011: XQT Attention TileLang full-forward CUDA Graph fastpath

### 目标

| 项 | 值 |
| --- | --- |
| module | `xqt.nn.Attention` 的 `_TileLangXqtAttentionWrapper` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, `sm_89`, CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| shape | `[batch=1, seq=64, dim=128]`, `heads=4`, `head_dim=32` |
| dtype | FP16, `dropout_p=0`, non-causal |
| fastpaths | native SDPA, eager TileLang, explicit TileLang CUDA Graph |

### 基线与方法

三条路径使用同一组 module 参数和输入. CUDA event benchmark 使用 warmup 20,
50 个样本,每个样本连续调用 100 次. 编译和首次 graph capture 不计入稳态数字.
数值基线为 eager TileLang wrapper,容差为 `atol=0.02, rtol=0.02`.

### 实现

1. `_TileLangXqtAttentionWrapper` 增加 `attention_fastpath="graph"` 和
   `"tilelang_graph"` 分支,保留 `auto` 在 Ada 上选择 native SDPA 的行为.
2. graph capture 覆盖完整的 `qkv projection -> reshape -> TileLang attention
   -> merge -> out_proj` 路径, replay 只复制动态输入.
3. graph cache key 包含输入 shape/stride/dtype/device, causal/dropout, tile
   参数和 `target_arch`. 每个固定输入签名独立 capture.
4. execution metadata 增加 `cuda_graph.state`, `reason` 和 `cache_size`,
   让 capture/replay/fallback 与 operator report 可追溯.

### 结果

| path | p50 (ms) | mean (ms) |
| --- | ---: | ---: |
| native SDPA | `0.09111` | `0.09113` |
| eager TileLang | `0.10028` | `0.10400` |
| TileLang CUDA Graph | `0.01966` | `0.02113` |

最终 run 中 graph replay 相对 eager TileLang 约 `5.1x`,相对 native SDPA 约
`4.6x`. 两次 profiler-free run 的 graph p50 为 `0.0197-0.0229 ms`,小 shape
下仍有 launch 和 clock 噪声.
首次调用 metadata 为 `captured`,第二次为 `replayed`,两次输入的 graph 与
eager 最大绝对误差均为 `0.0`.

Nsight Systems 以 NVTX 区间拆分 steady-state. 首个 eager 区间含 500 次
`cudaLaunchKernel` 和 400 次 `cuLaunchKernel`;首个 graph 区间含 100 次
`cudaMemcpyAsync` 和 100 次 `cudaGraphLaunch`. graph node 在本机 Nsight
Systems 版本中不会关联到 host NVTX filter,因此不从空的 graph kernel report
推断 kernel 数量或 kernel 时间. NCU 因 `ERR_NVGPUCTRPERM` 阻断,没有写
occupancy/cache/stall 结论.

### 适用边界与回退

- graph 仅在 CUDA 输入,FP16,`dropout_p=0` 且固定 shape/stride/dtype/device 时
  可复用; shape 或布局变化会建立新的 graph entry.
- 当前 facade wrapper 不新增 `attn_mask` / `key_padding_mask` 支持,动态 mask
  仍需 reference/native 路径.
- `auto` 在 Ada 上仍为 native SDPA; graph 需要 operator target 显式设置
  `attention_fastpath="graph"`.
- 该证据覆盖 `sm_89` 和一个小 prefill shape,不外推到 SM90/SM100 或生产
  serving 的 paged KV / continuous batching.

### 未采纳方案

- 不继续调 `block_m/block_n` 来解决这个小 shape 的 eager 差距: Nsight Systems
  已显示 launch/runtime integration 是主要差异层.
- 不把 graph capture 仅缩小到 attention kernel: projection 和 reshape 的
  host launch 会继续抵消 kernel 收益.
- 不把 CUDA Graph steady-state 数字写成 kernel-only 性能,也不在 NCU 被阻断
  时猜测 occupancy 或 memory bottleneck.

### 验证落点

- [XQT Attention wrapper](../../../xqt/operator_opt/wrappers/xqt_attention.py)
- [CUDA graph runtime helpers](../../../xqt/operator_opt/runtime.py)
- [CUDA regression tests](../../../tests/xqt/test_operator_tilelang_cuda.py)
- [benchmark script](../../../research/xqt-gemm/bench_sm89_xqt_attention.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-04-sm89-xqt-attention-graph/summary.json)

## R-012: SM89 grouped W4A16 routed-token task grid

### 目标

| 项 | 值 |
| --- | --- |
| module | `xqt.gemm` grouped W4A16 decode adapter |
| backend | `custom_cuda`,显式 `sm_89` artifact |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER,`sm_89`,CUDA 13.0,torch 2.12.1+cu130 |
| benchmark shape | expert rows `(0,1,2,4,8)`,`total_M=15`,`N=1024`,`K=4096` |
| precision | INT4 weight-only,FP16 activation/FP16 output,group size 128 |

### 基线与方法

权重 prepack 和 routing schedule 在 benchmark 前一次性建立. 稳态窗口使用
CUDA event,warmup 20,repeats 15,每次 1 iteration,`end.synchronize`.
对照包括 Python per-expert reference,repeated single-expert fused W4A16,
grouped direct task grid 和按 row bound 分桶的 grouped task grid. identity 与
in-kernel reverse permutation 使用同一组 activation/weight 分别测量.

### 实现

1. `GroupedGemmProblem` 固定累计 `m_offsets`,允许空 expert,并要求
   `output_rows` 是完整 permutation,避免 scatter 空洞或写冲突.
2. `w4a16_grouped_sm89.cu` 以 device task table
   `[expert,packed_row_base,row_count]` 覆盖所有 expert row tile. 编译了
   row bound `1/2/4/8`,每个 `(task,N-column)` 是一个 CUDA work item.
3. 静态 payload 为 `[E,N,padded_K/2]` qweight 与 `[E,N,G]` scale,可选
   zero point/bias. output permutation 在同一 kernel 写回,不追加 scatter launch.
4. `auto` 在 `N>=256` 的混合 row bound 选择 bucketed,小 `N` 选择 direct.
   这是当前 SM89 形状的受测策略,不是 persistent scheduler 或 Tensor Core
   MMA 的实现声明.

### 结果

正确性 gate 共 24 case (signed/unsigned x group 32/64/128 x bias/no-bias x
identity/scatter),expert rows `(0,1,2,8,9)`,最大绝对误差 `0.015625`,
`atol=0.125,rtol=0.02`,manifest 已 promotion 为 `executable`.

| path | median (ms) |
| --- | ---: |
| Python per-expert reference | `0.901024` |
| repeated single-expert fused,identity | `0.179936` |
| grouped direct,identity | `0.286400` |
| grouped bucketed,identity | `0.241664` |
| grouped auto,identity | `0.266432` |
| grouped bucketed,in-kernel scatter | `0.179200` |

event 数据受本机 WSL2 时钟和 launch 噪声影响. `torch.profiler` 显示 bucketed
路径使用四个 row-bound kernel 变体,而 direct 使用统一 bound=8;Nsight Systems
auto trace 的 kernel/API 汇总与此一致. prepack/schedule CUDA event 为
`0.818400`/`0.409376 ms`,不计入稳态. NCU 因
`ERR_NVGPUCTRPERM` 返回 `counter_permission_denied`,没有从缺失 counter 推断
occupancy,L2 或 warp stall.

### 适用边界与回退

- 只在 CUDA `sm_89`,FP16 activation/output,INT4 canonical
  `xqt_int4_nk_v1`,group size 32/64/128 和共享 `N/K` expert contract 下启用.
- artifact 必须有 correctness-promoted manifest;未 promotion 或架构不符时
  executor 拒绝 dispatch,上层保留 `reference_grouped_gemm` fallback chain.
- `direct` 是单 launch,`bucketed` 最多四个 row-bound launch;两者都不逐
  expert Python dispatch. 当前没有 persistent scheduler,local/global 不同
  `N/K` metadata 和 Tensor Core grouped MMA.
- scatter 只允许完整 permutation,由 kernel 直接写目标行,
  `scatter_launch_count=0`.

### 未采纳方案

- 没有把重复 single-expert fused 的更低 median 隐藏,因此 grouped 路径尚未
  被标成该 shape 的默认性能赢家.
- 没有把 SIMT task grid 命名为 persistent 或 CUTLASS Tensor Core grouped
  kernel;这些属于后续独立 artifact.
- NCU 权限恢复前不写硬件 counter 结论.

### 可复用规则

- grouped contract 先约束 offset/permutation,再设计 task scheduler;总和相等
  不能替代逐 expert offset 校验.
- 小 row bound 的 accumulator specialization 可能抵消额外 launch,但 auto
  门控必须用同形状 event 和 profiler 双重验证.
- prepack/schedule 生命周期应跨 routed-token batch 缓存,不能把一次性构建
  成本混入 steady-state kernel latency.

### 验证落点

- [grouped CUDA kernel](../../../xqt/gemm/backends/w4a16_grouped_sm89.cu)
- [grouped adapter](../../../xqt/gemm/backends/w4a16_grouped_sm89.py)
- [SM89 build/manifest](../../../xqt/gemm/backends/sm89_build.py)
- [grouped tests](../../../tests/xqt/gemm/test_grouped_w4a16_sm89.py)
- [benchmark and profile script](../../../research/xqt-gemm/bench_sm89_grouped_w4a16.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-07-sm89-grouped-w4a16/)

## R-013: SM89 grouped W8A8 native MMA task grid

### 目标

| 项 | 值 |
| --- | --- |
| module | `xqt.gemm` grouped W8A8 decode adapter |
| backend | `custom_cuda`,显式 `sm_89` artifact |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER,`sm_89`,CUDA 13.0,torch 2.12.1+cu130 |
| benchmark shape | expert rows `(0,1,2,4,8)`,`total_M=15`,`N=1024`,`K=4096` |
| precision | INT8 activation x INT8 weight,INT32 accumulator,FP16 output |
| scale | per-token activation,per-channel expert weight |

### 基线与方法

权重 stack/prepack 和 routing schedule 在稳态窗口前完成. CUDA event 使用
warmup 20,repeats 21,每个样本连续执行 10 次后取平均,以
`end.synchronize()` 为同步边界. 同一输入比较 1/2/4/8 warp grouped 候选,
`auto`,in-kernel scatter,逐 expert native launch 和显式 pad 到 `M=32` 的
`torch._int_mm` true-INT8 reference. PyTorch 当前 runtime 拒绝 `M<=16` 的
`_int_mm`,因此 reference 的 padding/copy/切片成本如实计入.

正确性 gate 覆盖 3 个 N/K shape (`8x32`,`17x100`,`32x127`),per-tensor 与
per-token activation scale,bias/no-bias,identity/reverse scatter,空 expert 和
`M_i=(0,1,2,8,9)`. 每个逻辑 case 对 1/2/4/8 warp 都执行一次,共 96 次
native run.

### 瓶颈与实现

初版一 warp CTA 每 8 个输出列重复加载同一 A tile. event sweep 显示该版本
median `0.156467 ms`,且 resource API 只有 `0.5` occupancy. 新实现把 CTA
参数化为 1/2/4/8 warp:

1. 每个 warp 仍执行一个 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`,
   负责 8 个 N 列和最多 8 个有效 M 行;MMA 的其余行显式填零.
2. 同一 CTA 的 warps 共用一个 swizzled A shared tile,每个 warp 保留独立的
   `[8,32]` B staging,减少 A 重复读取和 CTA 数量.
3. 权重静态布局为 `[expert,N,padded_K] int8`,weight scale 为
   `[expert,N] float32`;task table 是
   `[expert,packed_row_base,row_count]`. per-token activation scale,bias 和完整
   output permutation 都在同一 kernel 应用,scatter 不追加 launch.
4. A shared swizzle 的 XOR 地址会到达 byte offset 63,因此 row stride 固定为
   64 bytes. 32-byte stride 会覆盖每个 task 的第 4/8 行,correctness gate 已
   捕获并修正该错误.
5. artifact 先以 `metadata_only` 编译,24-case gate 通过后才 promotion 为
   `executable`. SASS 中四个模板均出现 `IMMA.16832.S8.S8`.

### 结果

correctness 最大绝对误差 `0.0009394`,`atol=0.0625,rtol=0.002`.

| path | median (ms) | 相对 grouped auto |
| --- | ---: | ---: |
| grouped,1 warp | `0.156467` | `1.383x` slower |
| grouped,2 warps | `0.123290` | `1.090x` slower |
| grouped,4 warps | `0.111206` | measured winner |
| grouped,8 warps | `0.119386` | `1.055x` slower |
| grouped auto (4 warps) | `0.113152` | `1.000x` |
| grouped auto,in-kernel scatter | `0.108032` | noise-equivalent |
| repeated per-expert native | `0.390624` | `3.452x` slower |
| padded-M `torch._int_mm` reference | `0.349459` | `3.088x` slower |

resource API 对 1/2/4/8 warp 分别报告 occupancy
`0.5/0.75/0.75/0.667`,registers/thread 都是 `56`,static shared
`1280/1536/2048/3072 B`. 4 warp 在并列最高 occupancy 下保留更多 CTA,
同时比 2 warp 少重复 A staging,与 event winner 一致.

`torch.profiler` 的 4-warp kernel 平均约 `81.34 us/launch`: grouped 20 次
forward 对应 20 次 kernel,逐 expert baseline 对应 80 次 kernel. Nsight
Systems 中相应 NVTX 区间为 `2.807 ms` 和 `9.838 ms`;kernel 汇总 150 次
实例平均约 `78.36 us`. NCU 返回 `ERR_NVGPUCTRPERM`,所以没有 L2,global
load 或 warp-stall counter 结论.

### 适用边界与回退

- 只在 CUDA `sm_89`,对称 zero-point-free INT8,共享 N/K expert contract,
  per-channel weight scale,per-token或 per-tensor activation scale和 FP16 output
  下启用.
- row task 固定最多 8 行,K 在 adapter 零填充到 32,N 尾 tile 在 kernel
  predicate. 空 expert 保留 offset 但不产生 task.
- `auto` 当前选择最多 4 warp,1/2/8 warp 仍可显式复测. 4 warp winner 只对当前
  设备和 shape 成立,不外推到其他 SM,N/K 或 routed-token distribution.
- manifest 未 correctness promotion,artifact 缺失或架构不符时 executor 显式
  拒绝 native dispatch. T034 的统一 grouped reference dispatcher 仍未完成,
  不在 Python 侧静默逐 expert 作为生产路径.

### 未采纳方案与可复用规则

- 8 warp 虽进一步减少 A staging,但 occupancy 降到 `0.667` 且 event 慢于
  4 warp,因此不作为当前默认.
- 不把逐 expert `torch._int_mm` 当公平的无 padding baseline;runtime 的
  `M>16` 限制必须记录.
- 不从 NCU 权限失败推断 cache 或 stall. resource API 只证明寄存器,shared
  和理论 active-block 边界.
- grouped 小 M kernel 应把共享 operand 提升到 CTA 级,但 warp 数必须结合
  grid size,occupancy 和同形状 event sweep 决定.

### 验证落点

- [grouped INT8 CUDA kernel](../../../xqt/gemm/backends/w8a8_grouped_sm89.cu)
- [grouped INT8 adapter](../../../xqt/gemm/backends/w8a8_grouped_sm89.py)
- [SM89 build/manifest](../../../xqt/gemm/backends/sm89_build.py)
- [grouped INT8 tests](../../../tests/xqt/gemm/test_grouped_w8a8_sm89.py)
- [benchmark and profile script](../../../research/xqt-gemm/bench_sm89_grouped_w8a8.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-w8a8/)

## R-014: SM89 grouped FP8 tensorwise/blockwise native MMA

### 目标

| 项 | 值 |
| --- | --- |
| module | `xqt.gemm` grouped FP8 decode adapter |
| backend | `custom_cuda`,CUTLASS FP8 warp MMA primitive,显式 `sm_89` artifact |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER,`sm_89`,CUDA 13.0,torch 2.12.1+cu130 |
| benchmark shape | expert rows `(0,1,2,4,8)`,`total_M=15`,`N=1024`,`K=4096` |
| precision | E4M3/E5M2 activation x weight,FP32 accumulation/scale,FP16 output |
| scale | tensorwise `[E]`/`[E]`;blockwise `[total_M,Kb]`/`[E,N,Kb]` |

### 基线与方法

权重量化/prepack,routing schedule 和 artifact build 都在稳态窗口前完成.
正确性以 FP8 storage 按声明 scale 解码后执行 FP32 accumulation 为 reference,
不是拿原始未量化权重比较. CUDA event 的顺序 sweep 使用 warmup 20,repeats
21,每个样本连续 10 次后取平均,显式 `end.synchronize()`. 同一输入比较
1/2/4/8 warp,`auto`,in-kernel scatter,逐 expert native launch 和已经预解码
为 FP32 的 matmul reference.

顺序 sweep 的 4/8 warp winner 在不同运行间会翻转. 为避免把 WSL2 clock
波动和候选顺序误当成调度收益,脚本另做 9 轮 x 31 对交错复测:相邻样本在
`4 -> 8` 与 `8 -> 4` 之间交替,每个候选每个样本执行 50 次. 只有同一候选
至少赢 7/9 轮且总中位数差距不小于 3%,才允许改变默认调度.

正确性 gate 覆盖 E4M3/E5M2,tensorwise,blockwise 32/64/128,bias/no-bias,
identity/reverse scatter,空 expert,小 expert,K/N 尾块和
`M_i=(0,1,2,8,9)`. 32 个逻辑 case 加 4 个显式 warp 候选,共 36 次 native
run.

### 瓶颈与实现

逐 expert 路径把 4 个非空 expert 拆成 4 次 native launch. grouped 实现将
所有非空 expert 展开为 `[expert,packed_row_base,row_count]` task table,一次
grid launch 完成计算和可选 output permutation:

1. 每个 task 最多处理 8 个有效 row;1/2/4/8 个 warp 沿 N tile 协作,CTA 内
   共享一个 swizzled A staging,每个 warp 保留自己的 B fragment.
2. tensorwise activation/weight scale 分别固定为 `[E]`/`[E]`. blockwise
   activation scale 为 `[total_M,Kb]`,weight scale 为 `[E,N,Kb]`,并在每个
   K block 的 FP32 accumulator contribution 上应用,支持
   `block_k=32/64/128`.
3. K 在 adapter 零填充到 32,N 尾 tile 在 kernel predicate;空 expert 保留
   offsets 但不产生 task,bias 和 scatter 都在同一 kernel 内执行.
4. artifact 先以 `metadata_only` 构建,correctness gate 通过后 promotion 为
   `executable`. SASS 共解析到 32 个 `QMMA.16832`:E4M3/E5M2 各 16 个.

当前 profiler 证明主要差异是 launch 结构:grouped 每个 forward 只有一次
目标 kernel,逐 expert 为 4 次. NCU counter 权限被拒绝,因此没有进一步把
性能差异归因于 L2,shared bank conflict,eligible warp 或 tensor pipe 利用率.

### 结果

correctness 最大绝对误差 `0.0127106`,`atol=0.25,rtol=0.03`.

| path | tensorwise median (ms) | blockwise-64 median (ms) |
| --- | ---: | ---: |
| grouped,1 warp | `0.180800` | `0.147530` |
| grouped,2 warps | `0.122762` | `0.110387` |
| grouped,4 warps | `0.102480` | `0.106906` |
| grouped,8 warps | `0.103629` | `0.104550` |
| grouped auto,4 warps | `0.099635` | `0.107930` |
| grouped auto,in-kernel scatter | `0.101853` | `0.110368` |
| repeated per-expert native | `0.403686` | `0.410298` |
| predequantized FP32 reference | `0.132096` | `0.133606` |

grouped auto 相对逐 expert native 分别为 `4.052x` 和 `3.802x`. 单个顺序
sweep 中 tensorwise 4 warp 获胜,blockwise-64 8 warp 获胜,但交错复测不支持
scale-mode-aware 默认:

| mode | 4 warp (ms) | 8 warp (ms) | round wins (4:8) | 判定 |
| --- | ---: | ---: | ---: | --- |
| tensorwise | `0.095416` | `0.094920` | `3:6` | 差 `0.523%`,noise-equivalent |
| blockwise-64 | `0.093385` | `0.093307` | `7:2` | 差 `0.084%`,noise-equivalent |

blockwise 的总中位数轻微偏向 8 warp,但逐轮胜负反而偏向 4 warp;这正是
不能从单个聚合统计量提升新默认的证据. 两种 mode 都保留 4 warp.

`torch.profiler` 中 tensorwise grouped 20 次 launch 共 `1292.152 us`,约
`64.61 us/launch`;逐 expert 80 次共 `4261.196 us`,约 `53.26 us/launch`.
blockwise grouped 20 次共 `1069.435 us`,约 `53.47 us/launch`;逐 expert
80 次共 `4214.092 us`,约 `52.68 us/launch`. grouped 的单次 kernel 不是
更短,收益来自把完整 expert workload 合并成更少 launch.

Nsight Systems 的 blockwise NVTX 区间为 grouped `2.213 ms`,逐 expert
`9.408 ms`;150 个 grouped template 实例平均 `56.36 us`. NCU 返回
`ERR_NVGPUCTRPERM`,状态保存为 `counter_permission_denied`.

### 适用边界与回退

- 仅在 CUDA `sm_89`,E4M3 或 E5M2,zero-point-free FP8,共享 N/K expert
  contract,FP32 scale/accumulation 和 FP16 output 下启用.
- tensorwise 和 blockwise 是两套显式 scale layout;不做隐式广播或把
  blockwise scale 降为最终 epilogue scalar.
- `auto` 当前最多选择 4 warp. 1/2/8 warp 保留为显式实验候选;本 shape 的
  noise-equivalent 结果不能外推到其他 SM,N/K 或 routed-token distribution.
- manifest 未 correctness promotion,artifact 缺失或架构不符时 executor
  显式拒绝 native dispatch. T034 的统一 grouped reference dispatcher 仍未
  完成,不在 Python 侧静默拆成逐 expert 生产路径.

### 未采纳方案与可复用规则

- 没有根据一次顺序 sweep 把 blockwise 默认改为 8 warp,也没有增加
  scale-mode-aware auto. 跨轮差距不足 3%,收益不稳定.
- 当前实现是普通 task-grid,不命名为 persistent scheduler;真正的
  persistent grouped kernel 需要独立实现和 profiler 证据.
- predequantized FP32 reference 已排除反量化成本,只用于定位数值和提供
  库 matmul 对照,不伪装成同精度端到端 baseline.
- 相近候选必须交错测量并同时检查相对差距,逐轮胜负和成对比值;单个
  sequential median winner 不足以改变默认路由.
- NCU 权限失败只记录 blocked reason,不能用 resource API 的理论 occupancy
  替代 cache,stall 或 tensor-pipe counter 结论.

### 验证落点

- [grouped FP8 CUDA kernel](../../../xqt/gemm/backends/fp8_grouped_sm89.cu)
- [grouped FP8 adapter](../../../xqt/gemm/backends/fp8_grouped_sm89.py)
- [SM89 build/manifest](../../../xqt/gemm/backends/sm89_build.py)
- [grouped FP8 tests](../../../tests/xqt/gemm/test_grouped_fp8_sm89.py)
- [benchmark and profile script](../../../research/xqt-gemm/bench_sm89_grouped_fp8.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-fp8/)

## R-015: SM89 TileLang FlashAttention tile/stage sweep

### 目标

| 项 | 值 |
| --- | --- |
| module | `fused_attention_forward_tilelang`,生成 kernel 为 `main_kernel` |
| backend | TileLang 0.1.12,显式 NVIDIA `sm_89` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER,CUDA 13.0,torch 2.12.1+cu130 |
| shapes | small/medium/long prefill,causal prefill 和 `Q=1,KV=1024` decode |
| precision | FP16 Q/K/V,FP32 online-softmax accumulator,FP16 output |
| candidates | 9 组 `block_m/block_n/threads/num_stages`,默认 `64/64/128/2` |

### 基线与方法

所有 candidate 和 SDPA reference 使用同一组 contiguous Q/K/V tensor.
correctness 使用 `atol=0.02,rtol=0.02`. CUDA event 顺序 sweep 先 warmup 10,
再取 15 个样本的中位数;每个样本按 shape 连续执行
`100/40/10/20/40` 次,并用 event end synchronize. TileLang JIT,首次调用和
输入分配均排除在稳态窗口外.

如果顺序 winner 不是默认 candidate,脚本会把 winner 和默认做 7 轮 x 21 对
交错复测,相邻样本在 `A -> B` 和 `B -> A` 间切换. 只有同一候选至少赢
5/7 轮且总中位数领先不小于 3%,才允许改变默认. `torch.profiler` 对每个
最终候选先 warmup 10,再记录 20 次调用. Nsight Systems workload 每个 shape
把 10 次 warmup 放在 NVTX 外,把 20 次 measured call 放在独立 NVTX range.

### 编译边界与取舍

5 组 candidate 在所有 shape 都可编译并通过 correctness:

- `64x32,128 threads,2 stages`.
- `64x64,128 threads,1/2/3 stages`.
- `128x64,256 threads,2 stages`.

以下 4 组 candidate 在 5 个 shape 上都于 TileLang layout inference 阶段失败:

- `32x32,128 threads,2 stages`.
- `32x64,128 threads,2 stages`.
- `64x64,256 threads,2 stages`.
- `64x128,256 threads,2 stages`.

失败均为 `acc_s` FP32 fragment 到 `acc_s_cast` FP16 fragment 的
`T.Parallel` layout conflict. 这是当前 fragment/thread 映射的编译约束,
不能把这些 candidate 当成已执行但性能落后的 schedule. 共 25 个可执行
correctness case 全部通过,最大绝对误差 `0.0009765625`.

### 结果

| shape | 默认 TileLang median (ms) | SDPA median (ms) | 最终推荐 |
| --- | ---: | ---: | --- |
| `B1,H4,Q64,KV64,D32` | `0.011018` | `0.014272` | `64x64/128t/2s` |
| `B1,H8,Q256,KV256,D64` | `0.012972` | `0.015202` | `64x64/128t/2s` |
| `B1,H8,Q1024,KV1024,D64` | `0.036083` | `0.034406` | `64x64/128t/2s` |
| `B1,H8,Q512,KV512,D64,causal` | `0.015058` | `0.021278` | `64x64/128t/2s` |
| `B1,H8,Q1,KV1024,D64,causal` | `0.018606` | `0.030187` | `64x64/128t/2s` |

Long prefill 上 SDPA 仍比默认 TileLang 快约 4.9%,因此不能把本轮结果写成
TileLang 全 shape 胜出. 不同 shape 的默认 promotion 复测如下:

| shape | 对手 | 默认/对手 median (ms) | round wins | 判定 |
| --- | --- | ---: | ---: | --- |
| small prefill | 1 stage | `0.010955/0.012235` | `7:0` | 默认领先 `11.68%`,stable |
| medium prefill | 3 stages | `0.012830/0.012895` | `5:2` | 默认领先 `0.51%`,noise-equivalent |
| long prefill | 1 stage | `0.035930/0.036378` | `7:0` | 默认领先 `1.25%`,noise-equivalent |
| causal prefill | 1 stage | `0.013555/0.013494` | `4:3` | 1 stage 领先 `0.45%`,noise-equivalent |
| decode | 3 stages | `0.018352/0.018758` | `6:1` | 默认领先 `2.21%`,noise-equivalent |

只有 small prefill 跨过 3% 稳定性门槛,且 winner 本身就是默认 candidate.
其他 shape 即使顺序或总中位数出现非默认 winner,差距也不足以 promotion.
因此不新增 shape-aware preset,继续使用统一默认 `64x64/128t/2s`.

`torch.profiler` 中 20 次调用对应的 `main_kernel` 平均时间为 small
`1.809 us`,medium `5.485 us`,long `29.001 us`,causal `10.169 us`,decode
`15.381 us`. Nsight Systems 全 trace 含 150 次 `main_kernel`,即 5 个 shape x
`(10 warmup + 20 measured)`. 按 NVTX range 过滤后,每个 shape 都恰好包含
20 次 `main_kernel` 和 20 次 `cuLaunchKernel`;kernel 平均时间分别为
`1.788/5.430/30.986/10.110/16.375 us`.

Profiler 数字用于确认 kernel identity,launch count 和 time distribution,
不替代 CUDA event promotion gate. NVTX 内的 host API 时间还包含 profiler
扰动和末尾 synchronize,不能写成稳态 wrapper latency. NCU 返回
`ERR_NVGPUCTRPERM`,状态为 `counter_permission_denied`,没有 cache,occupancy,
warp stall 或 tensor-pipe 结论.

### 适用边界与回退

- 本证据仅覆盖当前 RTX 4070 Ti SUPER `sm_89`,TileLang 0.1.12,FP16,
  `head_dim=32/64` 和列出的 5 个 shape.
- 当前 kernel 要求 `seq_kv >= seq_q`,`dropout_p=0`,不覆盖动态 mask,paged KV,
  continuous batching 或其他 serving runtime.
- 本轮不改变 wrapper/engine 既有 fallback,也不把编译失败 candidate 加入
  dispatcher.
- Long prefill 的 SDPA 优势和其他 SM 的未知行为都要求继续保留 native
  reference,不能从本轮 Ada 数据外推统一 backend winner.

### 未采纳方案与可复用规则

- 不因低于 3% 的单次或聚合差距增加 shape-aware selector. 调度分支会扩大
  compile/cache surface,必须由稳定收益支付复杂度.
- 不在同一轮为 4 个 layout-conflict candidate 重写 fragment layout. 先保留
  明确的 compile failure,后续若单独修复 layout,需要重新跑完整 correctness
  和 promotion gate.
- 不因为 decode 的 `Q=1` 就假设更小 `block_m` 更快;当前两个 32-row candidate
  根本未完成 codegen,没有性能证据.
- 不混用 CUDA event,`torch.profiler` 和 Nsight wall/API 时间. 三者分别承担
  promotion,kernel attribution 和 launch/time-distribution 证据.
- NCU 权限失败只记录 blocked reason,不能用理论 resource 上限替代真实 counter.

### 验证落点

- [TileLang Attention entry](../../../xqt/operator_opt/kernels/tilelang/attention.py)
- [TileLang FlashAttention source](../../../learn/tilelang/flashatt.py)
- [tile sweep and profile script](../../../research/xqt-gemm/bench_sm89_tilelang_attention_tiles.py)
- [TileLang Attention CUDA tests](../../../tests/xqt/test_tilelang_attention_cuda.py)
- [operator integration CUDA tests](../../../tests/xqt/test_operator_tilelang_cuda.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-attention-tiles/)

## R-016: Unified grouped GEMM reference dispatch

### 目标

| 项 | 值 |
| --- | --- |
| module | `dispatch_grouped_gemm` 和 `reference_packed_grouped_gemm` |
| native bridges | SM89 grouped W4A16,W8A8 和 FP8 |
| reference backend | Torch FP32 accumulation,显式 `grouped_reference` |
| routed input | packed activation `[total_M,K]`,累计 `m_offsets`,可选 output permutation |
| scale coverage | W4 groupwise,W8 per-tensor/per-token,FP8 tensorwise/blockwise |

### 问题与基线

三类 strict native executor 在 artifact 缺失,manifest 未 correctness promotion
或架构不匹配时都会抛 `XQTBackendError`. 这是正确的 native 边界,但此前各自
report 的 `fallback_chain` 只写了 `reference_grouped_gemm` 字符串,没有一个
真实的统一 dispatcher 消费该链. 上层若自行捕获异常,很容易临时写出不带
report 的 Python expert loop.

本轮不把 reference 与 native 做 latency winner 比较. 验收基线是同一 packed
activation,同一 expert weight/scale/bias 和同一 output permutation 下,
strict native 不可执行时得到与声明 reference 一致的输出,同时 report 必须
暴露降级原因和 Python group loop.

### 实现

1. `reference_packed_grouped_gemm` 按 `GroupedGemmProblem.m_offsets` 显式切分
   activation. Per-tensor scale 支持 scalar 或 `[E]`;per-token 和 blockwise
   scale 要求首维为 `total_M`,再按 expert row slice. 输出先按 packed row
   拼接,最后用 `output_rows` 执行一次显式 permutation.
2. `GroupedGemmNativeCandidate` 区分 `executable`,`metadata_only` 和 `planned`.
   `dispatch_grouped_gemm` 只执行 executable candidate,只捕获
   `XQTBackendError` 作为 runtime unavailable;shape,scale 或 contract 的
   `ValueError` 不会被静默吞掉.
3. 统一 report 写出 `selected_kernel`,`backend`,`maturity`,`execution_mode`,
   `native`,`expert_rows`,`m_offsets`,`output_scatter`,`python_group_loop`,
   `fallback_reason` 和 `fallback_chain`. Native bridge 的原始 report 保留在
   `native_details`.
4. 新增 `dispatch_sm89_grouped_w4a16`,`dispatch_sm89_grouped_w8a8` 和
   `dispatch_sm89_grouped_fp8`. 原 `sm89_grouped_*_executor` 仍是 strict native
   API,benchmark/profiler 可以继续直接调用,不会在内部隐式 fallback.
5. 三个 bridge 的 reference weight view,scale tuple 和 PackedWeight metadata
   都在惰性 `reference_executor` 中构造. Native candidate 成功时不会遍历
   expert 来 materialize reference 输入,避免把 fallback 准备开销带回稳态
   native forward.

### 结果

当 native candidate 成功时,report 为 `execution_mode=native_grouped`,
`python_group_loop=false`,并保留 native scheduler/launch metadata. 当 artifact
或 manifest 不可执行时,selected kernel 为 `grouped_reference`,report 为
`execution_mode=reference_grouped`,`native=false`,`python_group_loop=true`,
同时 `fallback_reason` 包含具体 native error.

验证覆盖:

- metadata-only candidate 被跳过,不会执行其 callable.
- executable candidate 成功时惰性 reference executor 不会被调用.
- native candidate 抛 `XQTBackendError` 时进入统一 reference;
  `allow_reference=False` 时原异常继续抛出.
- W4A16 signed/unsigned zero-point contract,bias 和 reverse scatter.
- W8A8 per-tensor/per-token activation scale,bias 和 reverse scatter.
- FP8 E4M3 tensorwise/blockwise-32 scale,bias 和 reverse scatter.
- packed INT8 per-token scale 的 CPU reference 拆分和数值结果.

### 适用边界与回退

- `grouped_reference` 是 correctness 和可诊断保底路径,内部明确使用 Python
  group loop 和逐 group Torch matmul. 它不命名为 native grouped kernel,
  不参与 grouped latency winner 声明.
- Production 可设置 `allow_reference=False`,要求 native unavailable 立即失败.
- Dispatcher 只把 `XQTBackendError` 视为 candidate unavailable. 用户输入,
  quant schema,shape 或 output permutation 错误仍显式报错.
- 当前 bridge 只连接已验证的 SM89 W4A16/W8A8/FP8 payload. 其他 SM 或新
  precision 需要新增显式 native candidate,不能借 reference report 冒充覆盖.

### 未采纳方案与可复用规则

- 没有修改 strict native executor 默认行为. 在低层 executor 内自动 fallback
  会污染 kernel benchmark,掩盖 artifact promotion 和架构错误.
- 没有让每个 backend 复制一套 reference loop. Scale 拆分,scatter 和 report
  统一收敛到一个 grouped dispatcher/reference contract.
- 没有在 native 尝试前预构造 reference expert views. Fallback 数据必须 lazy,
  否则即使 CUDA event 不计 host 时间,operator-stage 仍会承担 Python 开销.
- Reference fallback 的存在不证明 production 性能可接受;需要低延迟服务时
  应关闭 reference 或在 report 层拒绝 `python_group_loop=true`.

### 验证落点

- [grouped dispatcher](../../../xqt/gemm/grouped_dispatch.py)
- [grouped reference](../../../xqt/gemm/reference.py)
- [W4A16 bridge](../../../xqt/gemm/backends/w4a16_grouped_sm89.py)
- [W8A8 bridge](../../../xqt/gemm/backends/w8a8_grouped_sm89.py)
- [FP8 bridge](../../../xqt/gemm/backends/fp8_grouped_sm89.py)
- [dispatcher tests](../../../tests/xqt/gemm/test_grouped_dispatch.py)
- [W4A16 grouped tests](../../../tests/xqt/gemm/test_grouped_w4a16_sm89.py)
- [W8A8 grouped tests](../../../tests/xqt/gemm/test_grouped_w8a8_sm89.py)
- [FP8 grouped tests](../../../tests/xqt/gemm/test_grouped_fp8_sm89.py)

## R-017: SM89 grouped GEMM versioned offline tuning cache

### 目标

| 项 | 值 |
| --- | --- |
| schema | `xqt-gemm-tuning-v1` |
| consumers | SM89 grouped W4A16,W8A8 和 FP8 bridge |
| policy | 离线 correctness + CUDA event promotion,运行时只查表 |
| runtime states | `hit`,`miss`,`expired`,`invalid`,`bypassed_explicit` |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-tuning-cache/` |

### 问题与基线

R-012 到 R-014 已经得到 grouped W4A16,W8A8 和 FP8 的确定性默认配置,
但 event winner,resource query 和 cache-sensitivity 仍分散在 benchmark sidecar.
Dispatcher 无法判断某个运行时 contract 是否真的有离线验证记录,也无法区分
cache miss,过期记录和 artifact 被替换后的失效状态.

不能用首次 forward autotune 填补这个缺口. 在线 candidate sweep 会把编译,
同步和长尾延迟带入生产路径,并使 CUDA Graph capture 与稳态 benchmark
不可预测. 本轮目标是让离线证据成为可校验 artifact,运行时只做 exact lookup,
不新增 benchmark callable 或隐式编译.

### 实现

1. `GemmTuningKey` 绑定 kernel family,backend,`sm_*`,`M/N/K`,exact
   `expert_rows`,weight/activation/compute/accum/output dtype,group axis/size,
   scale mode,symmetric/zero-point,scale source,storage/pack version,persistent,
   output scatter,bias,CUDA Graph 和 workspace bytes. Canonical JSON SHA256
   在 key 构造时缓存,report 与 lookup 不重复序列化计算.
2. `GemmTuningRecord` 保存 selected kernel/scheduler/warp,correctness gate,
   CUDA event benchmark,runtime resource query,间接 cache-sensitivity,
   artifact/manifest SHA256,evidence paths,source 和创建/过期时间. Cache envelope
   对 canonical payload 再做 SHA256,重复 key,corrupt JSON 或 checksum mismatch
   都显式失败. 持久化使用同目录临时文件,`fsync` 和 `os.replace` 原子替换.
3. 只有 `scheduler="auto"` / `warps_per_block="auto"` 消费 cache. 显式配置
   报告 `bypassed_explicit`;miss,expired,invalid 或 backend 不支持的 knob 都
   使用原确定性默认值. W4 接受当前 kernel 的 direct/bucketed,或 persistent
   加合法 `persistent_blocks_per_sm`;W8/FP8 只接受 direct 和 1/2/4/8 warp,
   非法 record 不能改变 dispatch.
4. `GroupedGemmDispatchReport` 统一暴露 `tuning_cache_status`,完整 key,
   key SHA256,reason,source 和 record id.Native 与 reference fallback 都保留
   这些字段,未命中不会被描述为 performance winner.
5. Artifact identity 首次验证后按绝对路径 memoize,steady-state 不再做
   `stat` 或文件 hash. 调用方可在 forward 前用 `prime_artifact()` 预热;
   artifact 重建或替换后必须调用 `invalidate_artifact()` 再验证.

### 结果

当前 cache payload SHA256 为
`207ea4ae89b91ffe65c2b59ffdf26aa0109a5f8cb3a3b95a32d093d268d78e80`,
包含 8 条 record. 所有记录都限定在 RTX 4070 Ti SUPER `sm_89`,
`M=15,N=1024,K=4096`,`expert_rows=(0,1,2,4,8)`,FP16 output,bias,
`persistent=false`,`cuda_graph=false`,`workspace_bytes=0`:

- W4A16 symmetric signed,groupwise-128,identity/scatter,选择
  persistent/4 blocks per SM. Key 的 `persistent=false` 表示 auto 请求没有强制
  persistent runtime contract;离线 selection 仍可选择已验证 persistent 候选.
- W8A8 per-channel weight + per-token activation,identity/scatter,选择
  direct/4 warp.
- FP8 E4M3 tensorwise,identity/scatter,选择 direct/4 warp.
- FP8 E4M3 blockwise-64,identity/scatter,选择 direct/4 warp.

真实 CUDA dispatcher 验证中,8/8 cache 路径报告 `hit`,对应 empty cache
路径报告 `miss`;所有 hit/miss 输出逐元素 bitwise equal,native scheduler/warp
与 record 一致,惰性 reference 没有物化. 不存在首次 forward benchmark 或
autotune callable.

W8A8 公平交错测量得到:

| 层级 | hit | miss | 结论 |
| --- | ---: | ---: | --- |
| CUDA event median | `0.154317 ms` | `0.148141 ms` | 同一 4-warp kernel,不把差值解释为 cache 加速 |
| pre-synchronized host dispatch median | `140.878 us` | `128.985 us` | hit 包含 verified record lookup 开销 |
| isolated in-memory lookup | `3.700 us` | `0.475 us` | hit 不是更快的 Python 分支 |

`torch.profiler` 记录 40 次同一 `grouped_w8a8_kernel<4>` launch,hit/miss
各 20 次.Nsight Systems 使用两个独立进程,两个 NVTX range 都恰好包含
20 次该 kernel launch,避免相邻 range 边界污染. 256 MiB thrash 后 8 个
case 的 cold/hot ratio 为 `1.072-1.322`;这只是间接 cache-sensitivity
证据,不是 L2 hit-rate counter.

NCU permission probe 返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 L2 hit rate,global/shared
load efficiency,warp stall,tensor-pipe utilization 或 counter-derived occupancy.

### 适用边界与回退

- 当前只有上述 8 个 exact contract 是 performance-promoted record. W4
  asymmetric,W8 per-tensor,FP8 E5M2,FP8 block 32/128,无 bias,其他 shape,
  其他 SM,CUDA Graph 或 workspace 配置都继续 miss. 显式 persistent 配置绕过
  cache;当前 record 只让 exact auto key 选择 persistent/4 blocks per SM.
- Miss,expired 和 invalid 只表示没有可消费的离线 winner,不表示默认 kernel
  失败. Dispatcher 保留既有确定性 scheduler/warp 和完整 fallback chain.
- Cache object 应在 steady-state 前加载并复用. 每次 forward 从 JSON 加载,
  对 artifact 做 `stat`/hash 或重新构造 cache 都不属于支持的快路径.
- Artifact/manifest 视为 memoize 后不可变. 原地重建而不调用
  `invalidate_artifact()` 会继续使用旧 snapshot,调用方必须遵守显式生命周期.
- 本轮 cache 不证明 grouped kernel 是模型级吞吐赢家,也不覆盖 quantize,
  routing,通信或 serving runtime 的端到端成本.

### 未采纳方案与可复用规则

- 不在 miss 时自动 benchmark. Offline evidence 缺失必须表现为可诊断 miss,
  不能把生产首次请求变成调优任务.
- 不因 cache hit 存在就宣传 wrapper 加速. 实测 hit 的 lookup 和 host dispatch
  都高于 deterministic miss,cache 的价值是选择经验证配置.
- 不在 NCU 无权限时用理论 occupancy 或 cold/hot ratio 替代硬件 counter.
  Runtime resource query 可以保存,但必须与 counter-derived 结论分开命名.
- 不接受 backend 未声明的 scheduler/warp. Persisted JSON 不是绕过 executor
  contract 校验的入口.

### 验证落点

- [tuning cache](../../../xqt/gemm/tuning_cache.py)
- [grouped dispatcher](../../../xqt/gemm/grouped_dispatch.py)
- [W4A16 bridge](../../../xqt/gemm/backends/w4a16_grouped_sm89.py)
- [W8A8 bridge](../../../xqt/gemm/backends/w8a8_grouped_sm89.py)
- [FP8 bridge](../../../xqt/gemm/backends/fp8_grouped_sm89.py)
- [tuning cache tests](../../../tests/xqt/gemm/test_tuning_cache.py)
- [artifact generator](../../../research/xqt-gemm/bench_sm89_grouped_tuning_cache.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-tuning-cache/)

## R-018: SM89 grouped W4A16 persistent scheduler

### 目标

| 项 | 值 |
| --- | --- |
| module | `sm89_grouped_w4a16_executor` |
| backend | `custom_cuda`,显式 `sm_89` artifact |
| shape | expert rows `(0,1,2,4,8)`,`M=15,N=1024,K=4096` |
| precision | symmetric signed W4A16,groupwise-128,FP32 accumulation,FP16 output,bias |
| candidate | single-launch `persistent_grid_stride`,1/2/4/6 blocks per SM |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-w4a16-persistent/` |

### 问题与基线

R-012 的 bucketed scheduler 按实际 task row count 使用 1/2/4/8 四个模板,
避免 direct 路径所有 task 都承担 8-row accumulator,但目标 shape 每次需要
4 次 kernel launch. Fresh Nsight Systems 中四个 row-bound kernel 的 GPU time
合计约 `85 us/call`;因此本轮假设不是单个 row-bound kernel 算得慢,而是四次
launch 与 enqueue gap 没有被当前 grouped 调度摊薄.

顺序单-call CUDA event sweep 对候选顺序和系统抖动敏感. 最终一次 sweep 的
identity median 为 bucketed `0.204416 ms`,persistent bpsm4 `0.208896 ms`,
bpsm6 `0.208704 ms`;scatter 为 `0.229376/0.205472/0.197632 ms`. 不同运行中
顺序 winner 会翻转,所以这些数字只用于初筛,不作为 promotion gate.

### 实现

1. 新增 `persistent_grouped_w4a16_decode_kernel`. Grid 上限为
   `SM_count * blocks_per_sm`,每个 CTA 用 grid-stride 循环消费
   `(task,column)` work item. 每个 work item 根据实际 `row_count` 进入模板化
   1/2/4/8-row device function,保持原 signed/unsigned nibble,group scale,bias
   和 in-kernel output permutation 语义.
2. Python adapter 新增显式 `scheduler="persistent"`,默认
   `persistent_blocks_per_sm=4`,report 写出 requested blocks,实际 grid blocks,
   单次 launch,row bounds 和 scatter 状态. Manifest 声明 direct,bucketed,
   persistent 三种候选及 `[1,2,4,max_active]` 调参空间.
3. CUDA resource query 报告 persistent kernel 为 256 threads,40 registers/thread,
   256 B static shared,max active 6 blocks/SM. bpsm4 对应 4 resident blocks/SM
   和 264-block grid. 这里的 `0.667` 只是 CUDA occupancy API 的理论 thread
   residency,不是 profiler 采集的 achieved occupancy.
4. Cache resolver 原子校验 scheduler 与 `persistent_blocks_per_sm`. Exact
   SM89 W4A16 identity/scatter record 选择 persistent/4;cache miss 继续使用原
   bucketed 默认,没有把单 shape 结论扩展成全局 `auto` heuristic.

### 正确性与稳定性 gate

24 个语义 case 覆盖 signed/unsigned,group size 32/64/128,bias/no-bias,
identity/reverse scatter,空 expert 和 `M_i=(0,1,2,8,9)`. 另有 16 个
1/2/4/6 blocks-per-SM case. 最大绝对误差 `0.015625`,最大相对误差
`7.48e-4`,`atol=0.125,rtol=0.02`;四个 block-count 输出 bitwise equal.

稳定性使用 9 轮 x 31 对交错测量,相邻样本交替候选顺序,每个 event sample
连续执行 10 次:

| 对比 | baseline | candidate | latency reduction | round wins |
| --- | ---: | ---: | ---: | ---: |
| identity | bucketed `0.166093 ms` | persistent bpsm4 `0.144384 ms` | `13.07%` | `0:9` |
| scatter | bucketed `0.164454 ms` | persistent bpsm4 `0.144579 ms` | `12.09%` | `0:9` |

bpsm4/bpsm6 的聚合 median 为 `0.144688/0.141722 ms`,轮次 `2:7`,但相对
gap 只有 `2.09%`,低于 3% knob-promotion 门槛. 因此保留已有 bpsm4 默认,
不增加动态 max-active 选择分支.

### Profiler 归因

`torch.profiler` 对 10 次 measured call 记录:

- bucketed: 40 次 target launch,四个 row-bound kernel 合计约
  `79.33 us/call`,`cudaLaunchKernel` host API 总计 `327.56 us`.
- persistent bpsm4: 10 次 target launch,kernel 约 `80.50 us/call`,
  `cudaLaunchKernel` host API 总计 `136.17 us`.

Nsight Systems 使用两个独立进程. 20 次 bucketed call 对应 80 次 target
launch,target GPU time 约 `85.39 us/call`;20 次 persistent call 对应 20 次
target launch,target GPU time 约 `88.00 us/call`. Persistent kernel body 没有
更快,甚至略慢;paired callable latency 的收益来自把每次 4 个 launch 压成
1 个并减少 enqueue gap. 这与最初 launch-bound 假设一致,不写成算术吞吐提升.

NCU permission probe 返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 achieved occupancy,L2 hit
rate,warp stall,tensor-pipe utilization 或其他 counter-derived 指标.

### 适用边界与回退

- Promotion 只进入 exact tuning record: RTX 4070 Ti SUPER `sm_89`,上述 shape,
  signed symmetric groupwise-128,bias,identity/scatter. 其他 shape,量化语义,
  SM 或 workspace 继续 cache miss 并使用既有 deterministic scheduler.
- `scheduler="persistent"` 仍可作为显式实验配置,但显式配置报告
  `bypassed_explicit`,不会伪装成 cache winner.
- Persistent 当前仍是 SIMT packed-dequant dot-product kernel,不是 Tensor Core
  W4A16 MMA. 单 launch 也不代表所有 grouped W4A16 shape 都更快.
- 资源 query 只能说明编译资源和理论 residency. NCU 权限恢复前,不能据此
  解释 cache behavior,warp stalls 或 achieved occupancy.

### 验证落点

- [persistent CUDA source](../../../xqt/gemm/backends/w4a16_grouped_sm89.cu)
- [W4A16 adapter](../../../xqt/gemm/backends/w4a16_grouped_sm89.py)
- [build manifest](../../../xqt/gemm/backends/sm89_build.py)
- [CUDA tests](../../../tests/xqt/gemm/test_grouped_w4a16_sm89.py)
- [tuning cache tests](../../../tests/xqt/gemm/test_tuning_cache.py)
- [benchmark and profiler script](../../../research/xqt-gemm/bench_sm89_grouped_w4a16_persistent.py)
- [persistent evidence](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-w4a16-persistent/)
- [updated tuning cache](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-grouped-tuning-cache/)

## R-019: SM89 KV-int8 scale tensor runtime integration

### 目标

| 项 | 值 |
| --- | --- |
| module | `KvScaleAttention` + `tilelang_kv_int8_fused_attention` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`), CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| contract | FP16 Q, INT8 K/V, device-resident single-element FP32 K/V scale, FP16 output |
| shapes | `batch=2`, `heads=4`, `head_dim=32`, `seq=128/1024`, causal/noncausal |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-scale-tensor/` |

R-009 的 TileLang mainloop 已经完成 in-kernel dequant,但 runtime 每次 forward
仍对两个 CUDA scale buffer 调用 `.item()`. 这会分别触发 DtoH copy 和 stream
同步,使一个约 `2.5 us/call` 的 `main_kernel` 被 wrapper 同步成本淹没. 本轮只
修改 scale ABI/runtime 接线,不改变 `64x64/128 threads/2 stages` mainloop.

### 实现

1. TileLang prim_func 的 `k_scale/v_scale` 从 host `T.float32` 标量改为
   `T.Tensor([1], T.float32)`,dequant 使用 `scale[0]`. Python 入口要求 scale
   为与 Q 同设备,dtype 为 FP32,`numel()==1` 的 tensor,只做 `reshape(1)`
   视图后 launch,不读取标量值.
2. `KvScaleAttention._try_fused_attention()` 直接传注册 buffer
   `self.k_scale/self.v_scale`,不再调用 `.item()`. 因此 `fill_`,模型迁移和
   `state_dict` 更新仍由原 buffer 语义承载,没有 Python float 缓存失效问题.
3. CPU monkeypatch 测试断言 runtime dispatch 传递的是原 buffer 对象;
   CUDA 测试覆盖 dtype,numel,device 契约和 fused/reference correctness.

### Correctness 与稳定性 gate

四个 shape 均满足 `torch.allclose(atol=0.01,rtol=0.01)`. 最大绝对误差为
`2.4414e-4`,cosine 最低为 `0.99999976`,与 R-009 的 fp16 累加误差边界一致.

稳定性使用 9 轮 x 31 paired sample,相邻样本交替
`reference -> fused` / `fused -> reference`,每个 event sample 连续执行 10 次.
Promotion gate 要求至少 7/9 轮胜出且相对 gap 不低于 3%:

| shape | before reference/fused | after reference/fused | after latency reduction | round wins |
| --- | ---: | ---: | ---: | ---: |
| seq128 noncausal | `0.169174/0.224461 ms` | `0.196710/0.164234 ms` | `16.51%` | fused `9:0` |
| seq128 causal | `0.175718/0.221043 ms` | `0.195008/0.165661 ms` | `15.05%` | fused `9:0` |
| seq1024 noncausal | `0.178656/0.226509 ms` | `0.176435/0.162474 ms` | `7.91%` | fused `9:0` |
| seq1024 causal | `0.176128/0.222714 ms` | `0.197427/0.169517 ms` | `14.14%` | fused `9:0` |

同一 fused callable 的 before -> after 中位延迟下降 `23.89-28.27%`. Reference
绝对延迟在两阶段间有系统抖动,所以性能 promotion 只使用各阶段内部交错 paired
结果;before/after 绝对差仅用于验证同步移除的量级.

### Profiler 归因

修改前 10 次 fused forward 的 torch.profiler 记录 `aten::item` 和
`aten::_local_scalar_dense` 各 20 次,`cudaStreamSynchronize` 和
`cudaMemcpyAsync` 各 20 次;`main_kernel` 共 10 次,device time 合计约
`25.032 us`. 修改后四类标量回读/同步事件均为 0,`main_kernel` 仍为 10 次,
device time 合计约 `26.584 us`. Kernel body 没有变快,收益来自 runtime 同步消失.

Nsight Systems 的独立 20-call measured range 给出同一结论: 修改前 40 次
DtoH,40 次 `cudaStreamSynchronize`,20 次 `main_kernel`;修改后 DtoH 和
`cudaStreamSynchronize` 均为 0,仍有 20 次 `main_kernel`. 修改后 trace 只保留
测量边界的一次 `cudaDeviceSynchronize`,不属于每次 forward 的 hot path.

NCU permission probe 返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 achieved occupancy,L2 hit
rate,warp stall,tensor-pipe utilization 或其他 counter-derived 指标.

### 路由边界

- 已验证 contract 下,显式 `preferred_kernel="auto"|"tilelang"` 可以选择 fused
  路径,且四个 shape 都通过稳定性门槛.
- 构造器默认仍保留 `preferred_kernel="reference"`. 当前证据只覆盖单个 SM89,
  模型侧 self-attention 和固定量化语义,不能泛化到其他 SM,动态 mask/shape 或
  serving 级 KV cache.
- XQT 仍不拥有 page table,KV block pool,cache eviction,continuous batching 或
  serving scheduler;本记录只证明模型侧 KV-int8 attention kernel/runtime 接线.

### 验证落点

- [KV-int8 fused kernel](../../../xqt/operator_opt/kernels/tilelang/kv_int8_attention.py)
- [runtime 接线](../../../xqt/runtime/modules/kv_attention.py)
- [CUDA/runtime tests](../../../tests/xqt/runtime/test_kv_attention_cuda.py)
- [benchmark and profiler script](../../../research/xqt-gemm/bench_sm89_kv_int8_scale_tensor.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-scale-tensor/)

## R-020: SM89 KV-int8 quantize-layout and projection-I/O fusion

### 目标

| 项 | 值 |
| --- | --- |
| module | `KvScaleAttention` + `tilelang_kv_int8_quantize_layout_kernel` + `tilelang_kv_int8_fused_attention_projection_io` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`), CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| contract | contiguous FP16 Q/K/V projection `[B,S,H*D]`,INT8 K/V `[B,H,S,D]`,device-resident FP32 scale,FP16 BSI output |
| shapes | `batch=2`, `heads=4`, `head_dim=32`, `seq=128/1024`, causal/noncausal |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-quant-layout/` |

R-019 已消除 K/V scale 的 `.item()` DtoH 同步,但 fused attention 前仍由
PyTorch eager 分别对 K 和 V 执行 `div -> round -> clamp -> int8 cast`,再各做
一次 BSI -> BHSD `contiguous` layout copy. 初始 Nsight Systems 的 20-call
measured range 共记录 340 个 GPU kernel,即每次 full-module forward 17 个
launch. 第一阶段把 K/V quantize + layout 融为一个 kernel 后降到 8 launch;
第二阶段继续让 attention 直接消费 Q projection BSI 并输出 BSI,删除剩余的
Q/output layout copy,最终降到 6 launch. 两阶段都针对细粒度
launch/materialization 开销,不改变 K/V dequant + online softmax mainloop.

### 实现

1. 新增双输出 TileLang kernel. 一个扁平线程映射同时读取连续 FP16 K/V
   projection,执行 per-tensor static INT8 量化,并直接写入 BHSD INT8 K/V.
   除法保持 PyTorch FP16 wrapped-scalar 语义,再提升到 FP32 执行 `round`,因此
   `qmax=63/127` 的量化码与原 eager 路径逐元素 bitwise equal.
2. Attention builder 新增编译期 `projection_io` 变体. Q tile 直接从
   `[B,S,H*D]` 的对应 head slice 搬入 shared,attention accumulator 直接写回
   同一 BSI layout. 旧 BHSD 入口保留为同层 benchmark baseline.
3. `KvScaleAttention.forward()` 只在 fallback 前物化 Q BHSD. TileLang 成功
   路径执行 quantize-layout + projection-I/O attention 两个 kernel,然后把 BSI
   输出直接交给 `out_proj`. Report 同时保留 `selected_kernel` 和有序
   `selected_kernels`.
4. `fused_quant_block_size` 显式进入构造器和 kernel cache key. 对 128,256,512
   threads 做同一 full-module sweep 后保留统一 256-thread 默认;没有增加 shape
   heuristic.

### Correctness 与稳定性 gate

独立 quantize-layout 测试覆盖 `seq=100` 和 `qmax=63/127`,K/V INT8 code 均与
eager 逐元素一致. Projection-I/O attention 在 `seq=100` 的 causal/noncausal
输入上与旧 BHSD kernel + output layout copy bitwise equal. 128/256/512-thread
layout-copy 候选和最终 projection-I/O 候选也都与 `eager quant + fused
attention` full-module 输出 bitwise equal. 相对 torch SDPA reference,四个 shape
全部满足 `torch.allclose(atol=0.01,rtol=0.01)`,最大绝对误差不超过
`2.4414e-4`,cosine 最低为 `0.99999988`.

稳定性使用 9 轮 x 31 rotating paired sample,每个 event sample 连续执行 10
次. Promotion gate 要求至少 7/9 轮胜出且相对 gap 不低于 3%:

| shape | eager | split TL | layout-copy b256 | projection-I/O | vs layout-copy | vs eager | vs split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seq128 noncausal | `0.153600 ms` | `0.113248 ms` | `0.086835 ms` | `0.078544 ms` | `9.55%`, `9:0` | `48.86%`, `9:0` | `30.64%`, `9:0` |
| seq128 causal | `0.158106 ms` | `0.114774 ms` | `0.087962 ms` | `0.078925 ms` | `10.27%`, `9:0` | `50.08%`, `9:0` | `31.23%`, `9:0` |
| seq1024 noncausal | `0.157712 ms` | `0.113869 ms` | `0.088141 ms` | `0.078790 ms` | `10.61%`, `9:0` | `50.04%`, `9:0` | `30.81%`, `9:0` |
| seq1024 causal | `0.157789 ms` | `0.116429 ms` | `0.088749 ms` | `0.080282 ms` | `9.54%`, `9:0` | `49.12%`, `9:0` | `31.05%`, `9:0` |

128/512-thread layout-copy 变体相对 256-thread 的差距都低于 3%
knob-promotion gate,且 round wins 不满足统一晋级条件. Projection-I/O 则在四个
shape 上都以 `9:0` 和 `9.54-10.61%` 稳定击败对应 256-thread layout-copy
baseline. 因此最终 runtime 选择 projection-I/O + 256-thread quantize-layout;
这仍只是当前 SM89 验证结论,不是跨 SM 最优声明.

### Profiler 归因

`seq128 noncausal` 的 `torch.profiler` 对 10 次 full-module call 记录:

- eager quant: `aten::div`, `aten::round`, `aten::clamp`, `aten::_to_copy` 各
  20 次,`aten::contiguous` 40 次,CUDA launch API 合计 170 次.
- layout-copy b256: 上述四类量化 op 均为 0,`aten::contiguous` 为 20 次,
  quantize-layout 和 attention 各 10 次,CUDA launch API 合计 80 次.
- projection-I/O b256: `aten::contiguous` 降为 0,quantize-layout 和 attention
  各 10 次,CUDA launch API 合计 60 次.

同一 `seq128 noncausal` workload 的独立 20-call Nsight Systems measured range
给出完整 progression:

- eager quant: 340 个 GPU launch,即 `17/call`.
- layout-copy b256: 160 个 GPU launch,即 `8/call`.
- projection-I/O b256: 120 个 GPU launch,即 `6/call`.

最终每次只保留 4 个 projection/output CUTLASS GEMM,1 个 quantize-layout kernel
和 1 个 attention kernel. Layout-copy baseline 的 quantize/attention 分别为
`1.132/2.819 us/call`;projection-I/O 为 `1.156/3.031 us/call`. 新 attention
mainloop 因 strided BSI slice 略慢,但删除两次 copy 和 launch 后 full-module 仍
稳定更快. 因此 profiler 归因是 intermediate materialization/launch 消失,不是
attention mainloop 算术吞吐提升.

NCU permission probe 仍返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 achieved occupancy,L2 hit
rate,warp stall,tensor-pipe utilization 或其他 counter-derived 指标.

### 路由边界

- Promotion 仅覆盖 RTX 4070 Ti SUPER `sm_89`,FP16 连续 BSI projection,
  per-tensor static INT8 scale 和上述四个 shape. 其他 dtype,layout,SM,动态 mask
  或 shape 仍需独立验证.
- 构造器默认仍为 `preferred_kernel="reference"`. Opt-in TileLang 路径失败时
  回退到原 eager quant/dequant + torch SDPA,并记录 `fallback_reason`.
- 这仍是模型侧 self-attention 实体. XQT 不拥有 page table,KV block pool,
  cache eviction,continuous batching 或 serving scheduler.

### 验证落点

- [quantize-layout and attention kernels](../../../xqt/operator_opt/kernels/tilelang/kv_int8_attention.py)
- [runtime pipeline](../../../xqt/runtime/modules/kv_attention.py)
- [CUDA/runtime tests](../../../tests/xqt/runtime/test_kv_attention_cuda.py)
- [benchmark and profiler script](../../../research/xqt-gemm/bench_sm89_kv_int8_quant_layout.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-quant-layout/)

## R-021: SM89 KV-int8 packed QKV projection integration

### 目标

| 项 | 值 |
| --- | --- |
| module | `KvScaleAttention` + `tilelang_kv_int8_packed_qkv_quantize_layout` + `tilelang_kv_int8_fused_attention_packed_qkv_io` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`), CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| contract | contiguous FP16 packed QKV `[B,S,3*H*D]`,INT8 K/V `[B,H,S,D]`,device-resident FP32 scale,FP16 BSI output |
| shapes | `batch=2`, `dim=128`, `heads=4`, `head_dim=32`, `seq=128/1024`, causal/noncausal |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-packed-qkv/` |

R-020 已把 full-module forward 从 17 launch 经 quantize-layout 和
projection-I/O 两阶段降到 6 launch,但 Q/K/V 仍由三个独立 projection GEMM
产生. 本轮只合并 projection 参数和 GEMM,不改变 K/V 量化语义或 attention
online-softmax mainloop. 目标流水线为:

```text
1 x packed QKV GEMM
1 x packed-QKV K/V quantize-layout
1 x packed-QKV attention
1 x output projection GEMM
= 4 launch/call
```

### 实现

1. `KvScaleAttention` 删除 `q_proj/k_proj/v_proj`,只保留
   `qkv = nn.Linear(dim,3 * inner_dim,bias=qkv_bias)`. Reference/fallback 从同一
   packed 输出切分 Q/K/V view;TileLang fastpath 直接消费连续 packed tensor.
   `state_dict` 只有 `qkv.weight[/bias]`,没有双参数源或隐式权重缓存. 这是 XQT
   v0.x 的显式破坏性变更,旧三投影 key 不做兼容迁移.
2. Packed quantize-layout kernel 按固定 offset 读取 K/V thirds,一次 launch 写出
   两个 INT8 BHSD tensor. Packed attention kernel 直接从第一个 third 按 head
   slice 读取 Q,并输出 BSI,不物化 `torch.split()` 产生的非连续 view.
3. Report 的 `projection_mode="packed_qkv"`,最终 `selected_kernel` 和有序
   `selected_kernels` 指向 packed quantize-layout + packed attention. Fallback
   和 reference 仍复用同一次 packed projection.
4. KV-scale calibration 修复 fused `qkv` 同一路径 K/V hook 覆盖问题,分别观察
   packed 输出的第二和第三等宽区间. `qkv_bias=False/True` 和单一 state-dict
   参数源均有测试覆盖.
5. 公平 benchmark 保留 R-020 的历史候选语义:从 `qkv.weight/bias` 的三个 slice
   手动执行三次 `F.linear`,因此 eager/split/layout-copy/projection-I/O 仍是三
   projection baseline;只有 `packed_qkv_b256` 执行正式单 projection runtime.

### Correctness 与稳定性 gate

四个 shape 上,packed QKV projection 与三个 sliced `F.linear` 拼接逐元素
bitwise equal,`max_abs=0`. Packed K/V quantize-layout 在 `qmax=63/127` 下与
eager code bitwise equal;packed attention 在 causal/noncausal `seq=100` 下与
projection-I/O attention bitwise equal. 所有 optimized full-module 输出均与
同轮 eager candidate bitwise equal,并通过 torch SDPA reference
`torch.allclose(atol=0.01,rtol=0.01)`.

稳定性使用 9 轮 x 31 rotating paired sample,每个 event sample 连续执行 10
次. Promotion gate 要求至少 7/9 轮胜出且相对 gap 不低于 3%:

| shape | projection-I/O | packed QKV | latency reduction | round wins |
| --- | ---: | ---: | ---: | ---: |
| seq128 noncausal | `0.067440 ms` | `0.057446 ms` | `14.82%` | `9:0` |
| seq128 causal | `0.068016 ms` | `0.059494 ms` | `12.53%` | `9:0` |
| seq1024 noncausal | `0.069901 ms` | `0.061731 ms` | `11.69%` | `9:0` |
| seq1024 causal | `0.068506 ms` | `0.061862 ms` | `9.70%` | `9:0` |

四个 shape 都满足 promotion gate. R-020 和本轮 absolute latency 来自不同
paired round,只在各自同轮候选内做性能结论,不做跨轮绝对值相减.

### Profiler 归因

`seq128 noncausal` 的 `torch.profiler` 对 10 次 full-module call 记录:

- projection-I/O: 40 次 GEMM,10 次 quantize-layout,10 次 attention,合计 60 次
  `cuLaunchKernel`.
- packed QKV: 20 次 GEMM,10 次 packed quantize-layout,10 次 packed attention,
  合计 40 次 `cuLaunchKernel`.

独立 20-call Nsight Systems measured range 记录 projection-I/O 120 个 GPU
launch (`6/call`),packed QKV 80 个 (`4/call`). Projection-I/O 的 80 个 GEMM
实例对应每次 Q/K/V/output 四个 GEMM;packed 的两类 GEMM 各 20 个,对应每次
packed QKV/output 两个 GEMM. Quantize-layout 和 attention 都保持每次一个
launch.

Nsight Systems 汇总的 target-kernel time 从约 `12.358 us/call` 降到
`8.719 us/call`. Quantize-layout 约为 `1.155 -> 1.125 us/call`,attention 约为
`2.959 -> 2.910 us/call`,基本不变;差异来自三个窄 projection GEMM 被一个宽
projection GEMM 替代,并删除两次 launch/framework dispatch. 因此本轮归因是
projection/runtime integration,不是 attention mainloop 吞吐提升.

NCU permission probe 返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 achieved occupancy,L2 hit
rate,warp stall,tensor-pipe utilization 或其他 counter-derived 指标.

### 路由边界

- Promotion 仅覆盖 RTX 4070 Ti SUPER `sm_89`,FP16 packed QKV,per-tensor
  static INT8 K/V scale,默认无 QKV bias 的 benchmark shape. Bias true/false 的
  参数与 reference 语义已测试,但 bias=true 性能未单独测量.
- 构造器默认仍为 `preferred_kernel="reference"`. Opt-in
  `preferred_kernel="auto"|"tilelang"` 才尝试 packed fastpath;CUDA,dtype,
  dropout,head-dim,layout 或 TileLang 不满足契约时显式回退并记录原因.
- 其他 SM,dtype,layout,动态 mask/shape 需要独立 benchmark 和 profiler.
- 这仍是模型侧 self-attention 实体. XQT 不拥有 page table,KV block pool,
  cache eviction,continuous batching 或 serving scheduler.

### 验证落点

- [packed quantize-layout and attention kernels](../../../xqt/operator_opt/kernels/tilelang/kv_int8_attention.py)
- [runtime pipeline](../../../xqt/runtime/modules/kv_attention.py)
- [KV-scale calibration](../../../xqt/quant/quantizers/kv_scale.py)
- [CUDA/runtime tests](../../../tests/xqt/runtime/test_kv_attention_cuda.py)
- [calibration tests](../../../tests/xqt/quant/test_kv_scale.py)
- [benchmark and profiler script](../../../research/xqt-gemm/bench_sm89_kv_int8_quant_layout.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-packed-qkv/)

## R-022: SM89 KV-int8 packed full-forward CUDA Graph

### 目标

| 项 | 值 |
| --- | --- |
| module | `KvScaleAttention` packed TileLang full forward + shared CUDA Graph runtime helper |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`), CUDA 13.0, torch 2.12.1+cu130, TileLang 0.1.12 |
| contract | FP16 BSI input/output,1 packed QKV GEMM,INT8 K/V BHSD,device-resident FP32 scale,fixed graph signature |
| shapes | `batch=2`, `dim=128`, `heads=4`, `head_dim=32`, `seq=128/1024`, causal/noncausal |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-packed-graph/` |

R-021 已把每次 full-module call 收敛为 4 个 GPU kernel,target-kernel active
time 约 `8.719 us/call`,但模块 steady-state 仍约 `55-62 us`. 同层证据说明瓶颈
已从 kernel body 转到 Python/wrapper/driver submission. 本轮不修改 TileLang
quantize-layout 或 attention mainloop,只 capture 完整 packed fastpath:

```text
dynamic input D2D copy
    -> packed QKV GEMM
    -> packed-QKV K/V quantize-layout
    -> packed-QKV attention
    -> output projection GEMM
    -> graph-owned output
```

### 实现

1. `KvScaleAttention` 新增 `attention_fastpath="eager"|"graph"` 和
   `cuda_graph_warmup`. `preferred_kernel` 继续只选择
   `reference/auto/tilelang`,不把 kernel backend 与 runtime dispatch 混为一层.
   默认仍是 `attention_fastpath="eager"`.
2. Graph body 覆盖 `qkv(x)` 到 `out_proj(attn)` 的完整 packed forward. Capture
   前执行固定 warmup并同步,steady-state replay 每次只把动态 input 复制到静态
   storage,参数,bias 和 scale buffer 都保持静态.
3. 完整 cache key 包含 input shape/stride/dtype/device,causal,dropout,heads,
   head_dim,qmax,quant block,attention block,实际 compute capability 以及
   QKV/output/scale storage. `.to()` 经 `_apply()` 清空旧 device/dtype graph.
4. 首轮原型发现完整 eligibility probe,key 构造和 report 标记会把同一 graph
   state 的 module replay 从约 `14.21 us` 拉到 `23.75 us`. 正式实现增加经过完整
   key 验证的 last-state 热缓存;同签名只做紧凑 contract tuple 比较和共享 helper
   replay,诊断中完整 module 收敛到约 `15.19 us`,共享 helper 约 `14.06 us`.
5. Capture/replay 异常都记录 `cuda_graph.state="fallback_eager"` 和具体 reason,
   然后回到现有 eager packed TileLang fastpath;若 TileLang 自身也不可用,再回到
   torch SDPA reference. Kernel 选择和 graph 失败原因分开报告.
6. Report 新增 `attention_fastpath`,`selected_fastpath` 和
   `cuda_graph.state/reason/cache_size/output_storage`. 成功 replay 返回
   graph-owned output storage,下一次 replay 会覆盖旧 view;需要跨 replay 保留结果
   的调用者必须 clone.

### Correctness 与稳定性 gate

CUDA 测试覆盖首次 `captured`,第二次 `replayed`,同签名 cache size 保持 1,
causal 切换新建独立 entry并可切回旧 entry,`.to()` 清缓存,capture error eager
fallback,state-dict 不包含 graph 对象,以及 graph-owned output 覆盖语义.

原始输入和动态 `x + 0.25` 输入在 causal/noncausal 下都与 eager packed full
forward bitwise equal,`max_abs=0`. 正式稳定性使用 9 轮 x 31 alternating paired
sample,每个 CUDA event sample 连续执行 20 次. 编译,warmup,allocation 和 graph
capture 均排除在 measured window 外. Promotion gate 要求至少 7/9 轮胜出且
latency reduction 不低于 3%:

| shape | eager packed | CUDA Graph | latency reduction | graph round wins |
| --- | ---: | ---: | ---: | ---: |
| seq128 noncausal | `0.054878 ms` | `0.017139 ms` | `68.77%` | `9:0` |
| seq128 causal | `0.055448 ms` | `0.017141 ms` | `69.09%` | `9:0` |
| seq1024 noncausal | `0.054528 ms` | `0.034099 ms` | `37.46%` | `9:0` |
| seq1024 causal | `0.054323 ms` | `0.032307 ms` | `40.53%` | `9:0` |

四个 shape 全部满足 promotion gate. 长序列收益较低是因为每次 replay 仍需复制
完整 FP16 input;这属于 graph runtime integration 成本,不是 attention mainloop
回退.

### Profiler 归因

`seq128 noncausal` 的 `torch.profiler` 对 10 次 steady-state full-module call 记录:

- eager packed: `40 x cuLaunchKernel`,即 4 次直接 kernel launch/call.
- CUDA Graph: `10 x cudaMemcpyAsync + 10 x cudaGraphLaunch`. Profiler 仍观察到
  40 个内部 kernel event,即 graph 内保持同样 4 kernel/call,没有删除计算.

独立进程 Nsight Systems 对 20 次 call 记录 eager `80 x cuLaunchKernel`,graph
`20 x cudaMemcpyAsync + 20 x cudaGraphLaunch`. Graph D2D input copy 的 GPU 时间
中位数约 `1.001 us/call`. 当前 trace 配置的 graph kernel summary 不显示内部
graph node;torch.profiler 的 40 个内部 event 和 bitwise-correct output 已证明
执行,因此不能从空 kernel summary 推断 graph 没有运行 kernel.

NCU permission probe 返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 achieved occupancy,L2 hit
rate,warp stall,tensor-pipe utilization 或其他 counter-derived 指标.

### 路由边界

- Promotion 仅覆盖 RTX 4070 Ti SUPER `sm_89`,FP16 input/output,per-tensor static
  INT8 K/V scale,dropout 0 和上述固定签名. 其他 SM,dtype,mask,dropout 或 shape
  需要独立 capture,benchmark 和 profiler.
- `attention_fastpath="graph"` 是显式 opt-in. CPU,非 FP16,dropout 或 TileLang
  不满足要求时不错误进入 graph,并保留 eager/reference fallback.
- Graph cache 支持多个固定签名,但单 entry 使用同一静态 input/output storage.
  当前契约是顺序 replay,不声明同一 entry 的多流并发安全性.
- Graph output 是借用 view,不是 caller-owned allocation. 调用者若跨 replay 保存
  输出必须 clone;这是 latency 与 ownership 的显式取舍.
- 这仍是模型侧 self-attention 实体. XQT 不拥有 page table,KV block pool,
  cache eviction,continuous batching 或 serving scheduler.

### 验证落点

- [runtime pipeline](../../../xqt/runtime/modules/kv_attention.py)
- [shared CUDA Graph helpers](../../../xqt/operator_opt/runtime.py)
- [CUDA/runtime tests](../../../tests/xqt/runtime/test_kv_attention_cuda.py)
- [benchmark and profiler script](../../../research/xqt-gemm/bench_sm89_kv_int8_packed_graph.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-kv-int8-packed-graph/)

## R-023: SM89 TileLang BF16 FlashAttention coverage

| 项 | 值 |
| --- | --- |
| backend | TileLang FlashAttention-style attention |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,TileLang 0.1.12 |
| precision | BF16 Q/K/V input and output,FP32 score/online-softmax/output accumulation |
| shapes | small/medium/long prefill,causal prefill,non-square lower-right causal decode |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-attention-bf16/` |

### 目标和实现

原 TileLang attention 生产入口只接受 FP16. 本轮不改变 attention 语义,只把同一
fixed-shape JIT kernel 扩展到 BF16,并补齐 facade,wrapper,CUDA Graph 和
operator-stage 的 dtype contract:

1. `build_tilelang_flashatt(...)` 增加显式 `input_dtype`,只接受
   `float16/bfloat16`. Shared-memory tile,输出和 score cast 使用输入 dtype,
   score,在线 softmax 和 output accumulator 继续使用 FP32.
2. Kernel cache key 增加 input dtype,避免 FP16/BF16 复用错误 JIT artifact.
   Q/K/V 必须 dtype 完全匹配;FP32 和 mixed FP16/BF16 在 JIT 前显式失败.
3. 当前 TileLang BF16 MMA lowering 要求 `head_dim % 16 == 0`. Operator wrapper
   首轮使用 `head_dim=8` 时生成 BF16 `mma K=8` 并被 NVCC 的 unsupported
   configuration static assertion 拒绝. `head_dim=16/32/64` 诊断均可编译,
   因此正式入口前置 16 对齐校验,不让 backend compiler error 泄漏为能力边界.
4. `xqt.nn.Attention.forward()` 不再把 BF16 Q/K/V 隐式降为 FP16. FP16/BF16
   保持原 dtype,其他输入继续沿用已有显式 FP16 TileLang 尝试和 SDPA fallback.
5. `_TileLangAttentionWrapper` 与 `_TileLangXqtAttentionWrapper` 记录投影后实际
   Q/K/V dtype,report 同时声明 `supported_dtypes=[float16,bfloat16]` 和
   `bfloat16_head_dim_multiple=16`. CUDA Graph cache 原本已包含 tensor dtype,
   无需建立第二套 graph key.
6. Backend registry 直接复用 attention kernel metadata,避免 kernel catalog 和
   operator report 对 dtype 支持产生两份事实源.

### Correctness 和 runtime integration

CUDA kernel 测试覆盖 FP16/BF16 x 方阵非因果,方阵因果和非方阵 lower-right
causal decode. BF16 output 保持 BF16,并与同语义 SDPA reference 在
`atol=0.02,rtol=0.02` 内一致. 额外覆盖 mixed FP16/BF16,FP32 和 BF16
`head_dim=8` 的明确拒绝.

两个 wrapper 都覆盖 FP16/BF16 full-forward CUDA Graph 首次 `captured`,第二次
`replayed`,graph output clone ownership,以及 graph/eager TileLang 数值一致.
Operator-stage 测试覆盖显式 TileLang,native Ada `auto` 和 graph fastpath;
BF16 report 的实际 dtype 为 `bfloat16`,而 `auto` 仍选择 `native_sdpa`.

### Tile sweep 和 paired gate

正式 sweep 对 5 个 shape 各尝试 9 个 tile/thread/stage candidate,共 45 case.
25 case 可编译且通过 correctness,最大绝对误差为 `0.00390625`;没有 compiled
candidate 发生 correctness failure. 另外 20 case 来自 4 个在所有 shape 都稳定
失败的 layout 组合:

- `32x32/128t/2s`.
- `32x64/128t/2s`.
- `64x64/256t/2s`.
- `64x128/256t/2s`.

它们都在 `acc_s -> acc_s_cast` 的 TileLang layout inference 阶段冲突,不进入
性能候选. 5 个 shape 的 sequential winner 都是默认 `64x64/128t/2s`.
Promotion 使用 9 轮 x 31 alternating paired sample,要求至少 7/9 轮胜出且
gap 不低于 3%. 默认候选在 small/medium/causal/decode 相对最接近 schedule
稳定胜出;long 虽为 `9:0`,gap 仅 `2.87%`,归为 noise-equivalent. 因此不增加
BF16 shape-specific schedule.

同一 paired 方法比较默认 TileLang 与 BF16 SDPA:

| shape | TileLang | BF16 SDPA | TileLang latency reduction | round wins | classification |
| --- | ---: | ---: | ---: | ---: | --- |
| small prefill d32 | `0.011110 ms` | `0.010256 ms` | `-8.33%` | SDPA `9:0` | stable SDPA win |
| medium prefill d64 | `0.012850 ms` | `0.014577 ms` | `11.85%` | TileLang `9:0` | stable TileLang win |
| long prefill d64 | `0.033971 ms` | `0.034688 ms` | `2.07%` | TileLang `9:0` | noise-equivalent |
| causal prefill d64 | `0.013576 ms` | `0.022210 ms` | `38.87%` | TileLang `9:0` | stable TileLang win |
| decode q1/kv1024 d64 | `0.017396 ms` | `0.030070 ms` | `42.15%` | TileLang `9:0` | stable TileLang win |

### Profiler 归因

每个推荐 TileLang shape 的 `torch.profiler` measured window 都记录 20 个
`main_kernel` event. TileLang 和 SDPA 使用独立 Nsight Systems 进程,每个 shape
各有一个 20-call NVTX range:

- TileLang 每个 range 有 20 次 launch API call,即 1 个 `main_kernel/call`.
- Noncausal SDPA 每个 range 有 20 次 launch API call,即 1 个 `flash_fwd/call`.
- Causal prefill 和 lower-right causal decode SDPA 每个 range 有 40 次 launch API
  call,即 `flash_fwd_splitkv + splitkv_combine` 两个 launch/call.
- Small prefill 中 TileLang `main_kernel` 的 Nsight median 低于 SDPA
  `flash_fwd`,但 paired 端到端 latency 仍由 SDPA 胜出. 这证明 tiny shape 不能
  只看 kernel duration,dispatch/launch 路径同样进入 measured latency.
- Causal/decode 的稳定收益与 TileLang 单 launch,SDPA split-KV 双 launch 的
  topology 一致. Nsight 只作 time-distribution 解释,不替代 CUDA-event gate.

NCU permission probe 返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 occupancy,L2/cache hit,
warp stall,scheduler,roofline,tensor-pipe utilization 等 counter-derived 指标.

### 路由和适用边界

- 显式 `attention_fastpath="tilelang"` 和 `"graph"` 可执行 BF16;Ada
  `attention_fastpath="auto"` 仍选择 native SDPA. Kernel-level matrix 明确包含
  stable loss 和 noise-equivalent shape,不足以支持全局 route 切换.
- 当前证据只覆盖 `sm_89`,固定 layout,匹配 BF16 Q/K/V,dropout 0,
  `seq_kv >= seq_q` 和 16 对齐 head_dim. 其他 SM,mask,dropout,shape 或 dtype
  必须独立验证.
- 该路径是模型侧 attention operator,不包含 paged KV cache,continuous batching
  或 serving scheduler.

### 验证落点

- [TileLang attention kernel](../../../xqt/operator_opt/kernels/tilelang/attention.py)
- [MHA wrapper](../../../xqt/operator_opt/wrappers/attention.py)
- [xqt.nn.Attention wrapper](../../../xqt/operator_opt/wrappers/xqt_attention.py)
- [kernel CUDA tests](../../../tests/xqt/test_tilelang_attention_cuda.py)
- [operator/runtime CUDA tests](../../../tests/xqt/test_operator_tilelang_cuda.py)
- [BF16 sweep entry](../../../research/xqt-gemm/bench_sm89_tilelang_attention_bf16.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-attention-bf16/)

## R-024: ConvRot/SVDQuant 推理派生权重缓存与融合 gate

### 目标

| 项 | 值 |
| --- | --- |
| module | `ConvRotMixedPrecisionLinear`, `SVDQuantLinear`, `SVDQuantInt8MmaLinear` |
| 目标 | 避免每次 forward 重复 unpack/dequant,并把已存在的 SVDQuant fused kernel 接入可控 runtime fastpath |
| 低精度契约 | ConvRot W4A4/W4A16 保持 reference dequant + `F.linear`; SVDQuant split residual 仍保持 W4 storage -> W8A8 INT8 retarget |
| GPU | CUDA/TileLang 条件路径,当前环境未采集 GPU latency |

R-024 记录的是缓存阶段当时的实现边界. 后续 R-031 已为 SM89 ConvRot W8A8 和
SVDQuant W4A4/W8A8 补充 native CUDA dynamic fusion 与真实 GPU latency 证据;
R-024 中的 reference-only 描述不再代表这些新 fastpath 的当前能力.

### 基线与方法

基线是原有 eager forward: ConvRot 每次调用 `dequantized_weight()`,SVDQuant
reference 每次调用 `dequantize_residual()`,split INT8 每次执行 residual runtime
和两个低秩 GEMM. CPU 单测使用相同输入检查数值和 cache 失效;CUDA 路径只在
`float16`,block 对齐且 TileLang runtime 可用时尝试,不把 kernel-only 数字当成端到端
结果.

### 瓶颈与实现

1. ConvRot 缓存按 device,dtype,padding mode 复用 dequantized weight,签名包含
   packed weight/scale 的 device,shape 和 tensor version. 旋转矩阵也按 device/version
   复用. `.to()`/`.half()`/`.cuda()` 通过 `_apply()` 清空派生 cache;编译/tracing
   状态不写 Python cache.
2. SVDQuant reference shell 缓存 packed residual 的 dequant view,同样按 storage
   version 失效. `SVDQuantInt8MmaLinear` 复用现有 split residual contract,并增加
   guarded `torch.compile` composite path;小 batch,CPU 或 compile/runtime 异常回到
   原 eager split path.
3. 已有 `svd_fused_dequant_gemm_low_rank_tilelang` 只通过显式 CUDA gate 接入
   `SVDQuantLinear.enable_fusion()`. 该 gate 要求 CUDA,FP16,`M/N/K` 对齐和 TileLang
   可用,不把直接 W4 dequant kernel 冒充 W8A8 retarget.
4. W4 storage 和 FP8 dense fallback view 增加 storage version/device 失效,避免
   `load_state_dict()` 或原地 scale 更新后继续使用旧派生权重.
5. ConvRot W8A8 增加显式 `policy.fuse_norm=true` 的邻接替换,只匹配
   `Sequential(norm, linear)` 中的一维 RMSNorm/LayerNorm. `ConvRotNormInt8Linear`
   的 Triton 输入 kernel 在同一 row 内完成 Norm,Hadamard 和静态 INT8 activation
   quant,并通过 `Int8MmaLinear.run_quantized_activation()` 直接进入既有 GEMM,
   避免二次量化和中间浮点激活. rotation group 不超过 32 时使用一次 Norm
   统计的 row-per-program 变体;更宽 hidden size 保留 tiled 变体控制寄存器和编译体积.
   CUDA scale 校验不再在每次 forward 调用 `.item()` 同步. `rot_size<8` 不再反复
   尝试不满足 TileLang MMA 片段约束的 groupwise kernel.
6. Norm fusion 的实际状态写入 `execution_metadata()`:
   `norm_fused=true` 只表示本次 forward 的 fused kernel 成功;CPU,dynamic scale,
   小 batch 或可选 kernel 不可用时保留参考路径并记录 `fused_norm_fallback_reason`.

### 结果与边界

CPU correctness,cache 生命周期和 ConvRot Norm fusion 测试已通过. 当前设备上的
`M=32` input-only CUDA event smoke 显示 `K=64`/`K=1024` fused 输入路径有收益,
`K=256` 接近且略慢;这不是完整模型端到端证据,因此不宣称普遍 latency speedup.
不满足 dtype,shape,device,小 batch 或可选依赖条件时仍使用原有
reference/INT8/FP8 fallback. 派生 cache 视图按推理约定只读,原地修改 packed
storage 或 scale 会触发重建.

### 未采纳方案

- 没有把 SVDQuant W4 storage 的 residual 直接改成 TileLang dequant 输出,因为那会
  改变现有 W8A8 INT8 retarget 的数值契约.
- 没有在无 CUDA evidence 时默认开启 fused kernel,也没有把编译成功标成真实执行;
  metadata 记录的是实际 fused use 和 fallback reason.

### 可复用规则

- 静态权重派生 cache 必须覆盖 source tensor version 和 device,并在 module `_apply`
  后失效.
- kernel fusion 先按语义契约分 gate;storage/compute contract 不同的路径不能仅因
  共享 packed input 就合并.
- wrapper,whole-module 和 operator-stage 必须用同一 warmup/sync/timing 方法分别
  benchmark;本条改动尚缺目标 GPU 的三层 latency evidence.

### 验证落点

- [ConvRot runtime](../../../xqt/quant/quantizers/convrot_4bit.py)
- [ConvRot INT8 runtime](../../../xqt/quant/quantizers/convrot_int8.py)
- [SVDQuant runtime](../../../xqt/runtime/modules/svd_composite.py)
- [W4 storage cache](../../../xqt/runtime/modules/w4_storage_int8_mma_linear.py)
- [FP8 fallback cache](../../../xqt/runtime/modules/fp8_mma_linear.py)
- [runtime cache tests](../../../tests/xqt/runtime/test_composite_runtime_caches.py)
- [ConvRot Norm fusion tests](../../../tests/xqt/quant/test_convrot_int8_quantizer.py)

## R-025: SM89 TileLang BF16 direct Linear coverage and decode schedule

### 目标

| 项 | 值 |
| --- | --- |
| backend | TileLang direct dense Linear / GEMM epilogue |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,TileLang 0.1.12 |
| precision | BF16 activation/weight/bias/output,FP32 GEMM accumulator |
| shapes | decode `M=1/4/8`,small/medium prefill,`1024^3`,`M256,N11008,K4096` projection |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-linear-bf16/` |

原 direct Linear 生产入口只接受 FP16,且 builder 要求 M/N/K 都是 block 的整数倍.
本轮把同一 TileLang GEMM 扩展到 BF16,允许 partial M/N,并用 XQT dispatcher 和
wrapper 同层 gate 判断 schedule 能否进入默认路径.

### 实现

1. `build_tilelang_gemm_kernel(...)` 增加显式 `input_dtype`,只接受
   `float16/bfloat16`. JIT cache key 和 generated kernel name 包含 dtype,避免
   FP16/BF16 artifact 误复用;accumulator 保持 FP32,output 保持输入 dtype.
2. M/N store 增加边界保护,block M/N/K 必须 16 对齐;K 仍必须被 `block_k`
   整除. Direct `linear` / `dense_linear_epilogue` 要求 activation,weight,bias 和
   output dtype 与 MMA dtype 完全匹配,bias/activation 继续在 kernel 内融合.
3. BF16 统一 dispatcher 默认从 `linear_marlin` 改为
   `dense_linear_epilogue`;`linear`,`half_linear`,`linear_marlin` 保留显式 alias.
   `linear_marlin` 不消费本轮 direct dense schedule policy.
4. `_TileLangLinearWrapper` 在 materialization 时绑定 resolved kernel callable,
   rank-2 Linear 跳过无意义的 flatten/restore reshape,report 记录实际 dtype 和
   resolved schedule. 这消除了每次 forward 的 registry/runtime preflight.
5. 新增共享 schedule resolver. 仅在实际或显式 `sm_89`,BF16,flattened
   `M<=4` 且调用方未覆盖对应 block 时,默认选择
   `16x64x32/128 threads/2 stages`;其他 shape 继续使用 `64x64x64/128t/2s`.
   `gemm_with_precision(engine="auto")` 和 wrapper `linear_runtime="auto"` 均不改.

### Correctness 和测量方法

数值 reference 是 FP32 Linear 和 activation 完成后只在输出处 cast 一次 BF16,
不是 Torch BF16. 正式容差为 `atol=0.5,rtol=0.03`. 7 个推荐候选全部通过;
最大 `mean_abs=0.0002595111`,最大 `max_abs=1.0`. Projection shape 上 TileLang
对 FP32-cast reference 的 `mean_abs=0.0002595111`,Torch BF16 为
`0.075785473`,因此后者只作为性能 baseline.

每个 shape sweep 10 个 TileLang candidate. Promotion 使用 9 轮 x 31
alternating paired sample,要求至少 7/9 轮胜出且 relative gap 不低于 3%.
编译,首次调用,allocation 和一次性 weight layout 准备均排除在 steady-state
window 外.

### Schedule 和 dispatcher gate

| shape | recommended | 相对旧 TileLang 默认 | 相对真实 XQT Torch dispatcher |
| --- | --- | --- | --- |
| `M1,N4096,K4096` | `16x64x32/128t/2s` | `9:0`,gap `84.85%` | `9:0`,gap `6.62%` |
| `M4,N4096,K4096,bias` | `16x64x32/128t/2s` | `9:0`,gap `81.53%` | `9:0`,gap `34.43%` |
| `M8,N11008,K4096,SiLU` | `64x64x64/128t/2s` | 保留默认 | `9:0`,gap `6.25%` |
| `M64,N1024,K1024,bias` | `64x64x64/128t/2s` | 默认稳定胜 1-stage | `9:0`,gap `63.84%` |
| `M256,N4096,K4096,GELU` | `64x64x64/128t/2s` | 最近候选 noise-equivalent | XQT 胜,gap `0.75%`,noise-equivalent |
| `M1024,N1024,K1024` | `64x128x32/256t/2s` | `9:0`,gap `10.88%` | XQT `9:0`,gap `6.03%` |
| `M256,N11008,K4096,bias` | `64x64x64/128t/2s` | 保留默认 | XQT 胜,gap `1.44%`,noise-equivalent |

只有 `M=1/4` 同时具备改变旧 schedule 的稳定证据和 direct XQT dispatcher
优势,因此只进入 `M<=4` preset. `1024^3` winner 虽优于旧 TileLang 默认,但同
shape 仍输给真实 XQT Torch dispatcher,不建立全局 square heuristic.

### Wrapper 和 profiler 归因

Wrapper 对等 native `nn.Linear` 的 paired gate 中,`M=1` 以 `8:1` 和 `3.35%`
gap 稳定胜出;`M=4/8` 的 `0.93%/2.36%` 低于 3% 阈值;M64,medium,square 和
projection 都由 native 稳定胜出. 因此保留 Ada `linear_runtime="auto"` 的 native
route,显式 TileLang 才消费本轮 schedule.

独立进程 Nsight Systems 对每个 NVTX range 记录 20 次 call. Direct 与 wrapper
运行相同 generated kernel: M1 kernel median 为 `16.952/17.139 us`,M4 为
`17.504/18.242 us`;对应 launch API median 约为 `5.485/5.925 us` 和
`5.354/5.910 us`. 这说明 wrapper 修复后 kernel body 已对齐,剩余差异位于
kernel 外的 dispatch/launch 路径. 预转置 Triton 在 M1/M4 的 trace kernel
median 为 `112.788/118.003 us`. 同轮 artifact 中 BF16 SiLU 因
`tl.sigmoid` 接收 BF16 而 codegen 失败;GELU 可执行,但 reduction 后过早 cast
BF16 使 medium shape 的 `mean_abs=0.0357372`. R-026 修复并重新验证该 epilogue.

CUDA Graph probe 相对 eager direct TileLang 在 M1/M4/M8 分别慢约
`6.9%/20.8%/2.2%`,未采纳. NCU permission probe 返回
`ERR_NVGPUCTRPERM`,状态为 `counter_permission_denied`;本轮没有采集或推断
occupancy,L2/cache hit,warp stall,scheduler,roofline,register 或 tensor-pipe
指标.

### 路由和适用边界

- Promotion 仅覆盖 `sm_89`,BF16,direct dense Linear,flattened `M<=4`,匹配
  input/weight/bias/output dtype 和 K divisible by `block_k`.
- Partial M/N 可执行,K tail 不可执行. FP16,其他 SM 和未测 shape 保留旧默认或
  调用方显式 schedule,不能从当前结果外推.
- 显式 block 设置逐项优先于 preset. Wrapper report 保存实际 schedule 和
  preset 名称,便于后续 artifact 对照.
- Nsight Systems 只用于解释 time distribution,CUDA-event paired gate 才是
  promotion 依据.

### 验证落点

- [TileLang GEMM builder](../../../xqt/operator_opt/kernels/tilelang/gemm_builder.py)
- [TileLang Linear kernel](../../../xqt/operator_opt/kernels/tilelang/linear.py)
- [unified GEMM dispatcher](../../../xqt/operator_opt/backends/gemm_precision.py)
- [Linear wrapper](../../../xqt/operator_opt/wrappers/linear.py)
- [kernel CUDA tests](../../../tests/xqt/test_tilelang_half_ops_cuda.py)
- [operator/runtime tests](../../../tests/xqt/test_operator_tilelang_linear.py)
- [benchmark and profiler entry](../../../research/xqt-gemm/bench_sm89_tilelang_linear_bf16.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-tilelang-linear-bf16/)

## R-026: SM89 Triton BF16 fused epilogue FP32 codegen and runtime boundary

### 目标

| 项 | 值 |
| --- | --- |
| backend | Triton dense BF16 GEMM |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,Triton 3.7.1 |
| precision | BF16 activation/weight/bias/output,FP32 reduction 和 epilogue |
| shapes | `M8,N11008,K4096,SiLU`;`M256,N4096,K4096,GELU`;`M64,N1024,K1024,bias` control |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-epilogue/` |

R-025 暴露两类问题: BF16 SiLU 在 `tl.sigmoid` 处编译失败;GELU 和无 activation
虽然可执行,但 `_gemm_kernel` 在 bias/activation 前已经把 FP32 accumulator cast
到 BF16,不符合 XQT 的 FP32 epilogue 后单次 output cast reference.

### 实现

1. `_gemm_kernel` reduction 后统一保留 FP32 `c`;bias load 显式提升为 FP32.
2. ReLU/GELU/SiLU 都在 FP32 epilogue 中执行,只在 `tl.store` 处显式转换为
   `c_ptr.dtype.element_ty`. `ACC_TYPE` 仍控制 reduction dtype,没有改变公开参数.
3. 新增 BF16+bias GELU/SiLU CUDA regression,同时检查 output dtype,
   `max_abs`,`mean_abs` 和 `atol=0.5,rtol=0.03` allclose;FP16 GELU/SiLU 同步回归.
4. `gemm_with_precision(engine="auto")`,TileLang schedule 和 Linear auto route
   均保持不变. 本轮没有给无状态 dispatcher 增加隐式 weight cache.

### Correctness

数值 reference 是 FP32 Linear,bias 和 activation 完成后只 cast 一次 BF16.

| shape | 修改前 | 修改后 max_abs | 修改后 mean_abs |
| --- | --- | ---: | ---: |
| `M8,N11008,K4096,SiLU` | `tl.sigmoid` BF16 codegen failure | `0.5` | `0.0000882435` |
| `M256,N4096,K4096,GELU` | `max_abs=2.0`,`mean_abs=0.0357372` | `1.0` | `0.0001298454` |
| `M64,N1024,K1024,bias` | `max_abs=1.0`,`mean_abs=0.0352439` | `0.25` | `0.0000089691` |

三个 direct Triton case,真实 XQT Triton dispatcher,推荐 TileLang,Torch
`F.linear` 和真实 XQT Torch dispatcher 全部通过同一容差. Triton direct 与
TileLang 在三个 shape 上的输出误差基本一致,符合两者都采用 FP32 accumulator
和 FP32 epilogue 的预期.

### 测量方法和 paired gate

所有 workload 使用相同输入,10 次 warmup,CUDA event end synchronization 和
相同 shape-specific batched iteration. JIT,首次调用,input allocation 和 direct
Triton 的一次性 K,N pretranspose 排除在 steady-state window 外. Promotion gate
使用 9 轮 x 31 alternating paired sample,要求至少 7/9 轮胜出且 gap 不低于 3%.

| shape | direct Triton | TileLang | Torch F.linear | XQT Torch | XQT Triton N,K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M8,N11008,K4096,SiLU` | `0.221757 ms` | `0.170926 ms` | `0.171213 ms` | `0.174264 ms` | `1.125171 ms` |
| `M256,N4096,K4096,GELU` | `0.121094 ms` | `0.117613 ms` | `0.108877 ms` | `0.116467 ms` | `0.421370 ms` |
| `M64,N1024,K1024,bias` | `0.030000 ms` | `0.014563 ms` | `0.009882 ms` | `0.022241 ms` | `0.047331 ms` |

表中数字是顺序 sweep median,只用于量级展示. 路由判定使用 paired gate:

- SiLU direct Triton 相对 TileLang/Torch F.linear/XQT Torch 分别稳定慢
  `32.85%/26.70%/24.40%`,全部 `0:9`.
- GELU direct Triton 相对 TileLang 的 gap 为 `2.27%`,低于门槛并归为
  noise-equivalent;Torch F.linear 和 XQT Torch 分别稳定快 `10.67%/3.85%`.
- Control direct Triton 相对 TileLang/Torch F.linear/XQT Torch 分别稳定慢
  `112.00%/191.94%/32.38%`.
- 预转置 direct Triton 相对真实 N,K Triton dispatcher 在三个 shape 均 `9:0`,
  dispatcher 的 paired latency 分别高 `407.98%/249.27%/60.80%`.
- 真实 XQT Torch dispatcher 相对真实 XQT Triton dispatcher 在三个 shape 均
  `9:0`,gap 为 `534.23%/261.26%/114.59%`.

因此本轮只修复 codegen 和数值契约,不把 Triton 提升为这些 shape 的默认性能
winner,也不改变任何 auto route.

### Profiler 归因

torch.profiler 对每个 workload 记录 20 次 steady-state call. 独立进程 Nsight
Systems 的激活 shape NVTX range 中,direct Triton 和 TileLang 各有 20 次单 kernel
launch;Torch F.linear 为 GEMM 和单独 activation kernel. 关键 kernel median 为:

| shape | direct Triton | TileLang | Torch GEMM + activation | XQT Triton copy + GEMM |
| --- | ---: | ---: | ---: | ---: |
| SiLU | `228.576 us` | `160.388 us` | `166.725 + 2.681 us` | `933.486 + 231.220 us` |
| GELU | `120.551 us` | `118.473 us` | `108.346 + 3.318 us` | `331.203 + 120.701 us` |
| control | `30.912 us` | `8.313 us` | `7.272 us` | `17.803 + 30.950 us` |

真实 Triton N,K dispatcher 每次执行 `b.t().contiguous()`. SiLU torch.profiler
中 20 次 direct-copy kernel 共约 `19.93 ms`,20 次 `_gemm_kernel` 共约
`4.40 ms`;Nsight median 也显示 copy 为 `933.486 us/call`,高于 GEMM 的
`231.220 us/call`. 这是明确的 runtime integration 成本,不是 epilogue kernel
codegen 回归.

NCU probe 仍返回 `ERR_NVGPUCTRPERM`,状态为 `counter_permission_denied`.
本轮没有采集或推断 occupancy,L2/cache hit,warp stall,scheduler,roofline,
register pressure 或 tensor-pipe utilization.

### 路由和后续边界

- 无状态 `gemm_with_precision(...)` 不拥有静态 weight 生命周期,不能在函数内部
  建立无失效协议的隐藏 K,N cache. 仓库现有 selector 也明确记录该边界.
- 若后续为 Triton Linear/FeedForward materialization 增加 pretranspose,必须在
  stateful module 中缓存,覆盖 source tensor version,device/dtype 和 `_apply()`
  失效,并分别验证 wrapper,whole-module 和 operator-stage.
- 即使排除 layout copy,direct Triton 在当前 shape 也没有形成默认 route 证据.
  后续 schedule sweep 必须作为独立 kernel-tuning 里程碑,不能把 cache 收益写成
  kernel winner.
- 当前证据只覆盖 `sm_89`,BF16 和三个声明 shape. 其他 SM,dtype,shape 或 layout
  需要独立 correctness,paired gate 和 profiler.

### 验证落点

- [Triton GEMM kernel](../../../xqt/operator_opt/kernels/triton/gemm.py)
- [unified GEMM dispatcher](../../../xqt/operator_opt/backends/gemm_precision.py)
- [CUDA regression tests](../../../tests/operator_opt/test_gemm_precision.py)
- [benchmark and profiler entry](../../../research/xqt-gemm/bench_sm89_triton_bf16_epilogue.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-epilogue/)

## R-027: SM89 Triton BF16 exact-signature schedule resolver

### 目标

| 项 | 值 |
| --- | --- |
| backend | Triton dense BF16 GEMM |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,Triton 3.7.1 |
| precision | BF16 activation/weight/bias/output,FP32 reduction 和 epilogue |
| sweep | 5 shape x 22 个 tile/group/warp/stage 候选 |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-schedule/` |

R-026 修复数值契约后,固定 `128x128x32,group_m=8,4 warps,3 stages`
仍在 decode 和 small prefill 明显浪费 tile. 本轮只优化 K,N 已经预转置的 direct
kernel schedule,并把真实 N,K dispatcher 的 per-call layout materialization 保持为
独立 integration 对照.

### 实现

1. 新增 `TritonGemmSchedule` 和 `resolve_triton_bf16_gemm_schedule(...)`.
   resolver 只在真实或显式 `sm_89` 和五个受测
   `(M,N,K,has_bias,activation)` exact signature 上选择 preset.
2. 五个 preset 分别为:
   - `M1,N4096,K4096,no bias`: `16x64x64,group_m=4,4w,3s`.
   - `M4,N4096,K4096,bias`: `16x64x64,group_m=4,4w,3s`.
   - `M8,N11008,K4096,bias+SiLU`: `32x128x32,group_m=4,4w,3s`.
   - `M64,N1024,K1024,bias`: `16x64x32,group_m=4,4w,3s`.
   - `M256,N4096,K4096,bias+GELU`: `64x64x32,group_m=8,4w,3s`.
3. `block_m/block_n/block_k/group_m/num_warps/num_stages` 改为可选参数,
   调用方显式值逐字段覆盖 preset. 其他 shape,其他 SM 和 FP16 保持旧默认.
4. 实际 CUDA device -> SM 查询使用 16-entry bounded cache,纯 schedule resolver
   使用 256-entry bounded cache. 二者只缓存不可变标量/调度对象,不缓存 tensor 或
   weight,不承担模型 artifact 失效协议.
5. `gemm_with_precision(...)` 已经原样转发 Triton 调度参数,本轮只补 regression,
   不改变 selector 或 `engine="auto"` 路由.

### Correctness

数值 reference 是 FP32 Linear,bias 和 activation 完成后只 cast 一次 BF16.
22 个候选和实际 resolver workload 全部通过 `atol=0.5,rtol=0.03`.

| shape | max_abs | mean_abs |
| --- | ---: | ---: |
| `M1,N4096,K4096,no bias` | `0.25` | `0.0002224334` |
| `M4,N4096,K4096,bias` | `0.5` | `0.0003087092` |
| `M8,N11008,K4096,SiLU` | `1.0` | `0.0000836799` |
| `M64,N1024,K1024,bias` | `0.25` | `0.0000168676` |
| `M256,N4096,K4096,GELU` | `1.0` | `0.0001300724` |

CUDA regression 直接调用五个 no-override exact signature. CPU regression 覆盖
exact preset,非 SM89/default fallback,bias/activation mismatch,逐字段显式覆盖,
unified dispatcher forwarding,以及 CPU tensor 不查询 CUDA capability.

### Paired gate

方法仍为 10 次 warmup,shape-specific batched iteration,CUDA event end
synchronization 和 9 轮 x 31 alternating paired sample. Promotion 要求至少 7/9
轮胜出且 gap 不低于 3%. 实际 resolver 相对旧默认五个 shape 全部 `9:0`:

| shape | resolver vs old default | vs TileLang | vs Torch F.linear | vs XQT Torch |
| --- | ---: | --- | --- | --- |
| `M1` | `541.80%`,9:0 | resolver `94.36%`,9:0 | resolver `245.71%`,9:0 | resolver `238.81%`,9:0 |
| `M4` | `535.56%`,9:0 | resolver `75.29%`,9:0 | resolver `13.24%`,9:0 | resolver `32.15%`,9:0 |
| `M8 SiLU` | `39.39%`,9:0 | resolver `7.81%`,9:0 | resolver `10.45%`,9:0 | resolver `13.37%`,9:0 |
| `M64` | `63.05%`,9:0 | TileLang `30.35%`,9:0 | Torch `78.02%`,9:0 | resolver `15.86%`,9:0 |
| `M256 GELU` | `5.31%`,9:0 | resolver `1.79%`,noise | Torch `5.69%`,9:0 | resolver `0.48%`,noise |

实际 resolver 与当轮 sweep recommendation 在五个 shape 的 paired gap 全部低于
3%. M1/M4 的候选集 winner 在最终 sweep 中稳定;M8/M64/M256 有多个
noise-equivalent 候选. 因此后三个 preset 是受测平台上的代表 schedule,不是唯一
全局最优声明.

### Profiler 归因

独立进程 Nsight Systems 的每个 NVTX range 配置 20 次 steady-state call. 个别
filtered group 显示 21 个关联实例,因此只使用 median,不从该计数推断 launch 数.

| shape | resolved Triton | old default | TileLang | Torch GEMM + activation | XQT Triton copy + GEMM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M1` | `15.448 us` | `111.576 us` | `33.271 us` | `57.607 us` | `333.790 + 40.292 us` |
| `M4` | `16.473 us` | `116.075 us` | `33.737 us` | `20.863 us` | `328.678 + 41.141 us` |
| `M8 SiLU` | `150.807 us` | `226.213 us` | `159.441 us` | `165.923 + 2.633 us` | `942.480 + 188.739 us` |
| `M64` | `6.351 us` | `30.632 us` | `8.243 us` | `7.332 us` | `17.682 + 6.360 us` |
| `M256 GELU` | `112.534 us` | `119.036 us` | `117.927 us` | `107.915 + 3.300 us` | `328.329 + 114.949 us` |

M64 的 Triton kernel 本体快于 TileLang/Torch kernel,但 actual Python entrypoint
在 CUDA event operator gate 中仍慢于两者,说明剩余差异位于 wrapper/launch 边界.
bounded resolver cache 已降低该 host 成本,且 actual entrypoint 相对真实 XQT Torch
dispatcher 仍稳定快 `15.86%`.

真实 N,K Triton dispatcher 相对预转置 resolver 的 paired gap 为
`1959.55%/1855.18%/611.40%/76.59%/261.89%`,全部 `0:9`. Nsight 继续显示
每次完整 transpose-copy;本轮 schedule 不掩盖该 integration 成本.

NCU probe 返回 `ERR_NVGPUCTRPERM`,`status=counter_permission_denied`. 本轮没有
推断 occupancy,L2/cache hit,warp stall,roofline,register pressure 或 tensor-pipe
utilization.

### 路由和边界

- `gemm_with_precision(engine="auto")` 不改变. direct winner 假设 K,N weight 已在
  steady-state 前准备,而 stateless N,K dispatcher 仍被每次 transpose 主导.
- 不在 `gemm_with_precision(...)` 中增加隐藏 weight cache. 后续只能在 stateful
  Linear/FeedForward materialization 中缓存,并覆盖 tensor version,device,dtype 和
  module `_apply()` 失效.
- 当前证据只覆盖 `sm_89`,BF16 和五个 exact signature. 其他 SM,dtype,shape,
  bias/activation 或 layout 继续使用旧默认并需要独立 profile.

### 验证落点

- [Triton GEMM kernel and resolver](../../../xqt/operator_opt/kernels/triton/gemm.py)
- [unified GEMM dispatcher](../../../xqt/operator_opt/backends/gemm_precision.py)
- [resolver and CUDA regression tests](../../../tests/operator_opt/test_gemm_precision.py)
- [schedule benchmark and profiler entry](../../../research/xqt-gemm/bench_sm89_triton_bf16_schedule.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-schedule/)

## R-028: SM89 Triton half Linear zero-materialization layout and CUDA Graph runtime

### 目标

| 项 | 值 |
| --- | --- |
| backend | Triton direct FP16/BF16 Linear materialization |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,Triton 3.7.1 |
| precision | FP16/BF16 input,weight,bias,output;FP32 accumulation |
| layout evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-weight-layout/` |
| wrapper/graph evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-linear-wrapper/` |

R-026/R-027 已证明 direct kernel 的数值和 schedule,但真实 N,K dispatcher 仍被
每次 `weight.t().contiguous()` 主导,且 eager wrapper 在短算子上仍承受 Python
和 launch 开销. 本轮分别处理静态权重 layout 和 stateful Linear runtime,不把
weight cache 塞进无状态 dispatcher,也不扩大 `engine="auto"` 路由.

### 实现

1. `gemm_fp16_triton(...,transpose_b=True)` 和 BF16 共用的 kernel launcher 不再
   materialize K,N tensor,而是把原始 N,K weight 的逻辑 K/N stride 交换后直接
   传给 Triton kernel. FP16/BF16 都获得零额外权重内存的 stateless 路径.
2. Triton operator materializer 新增 `_TritonLinearWrapper`,覆盖
   `linear`,`gemm_fp16`,`gemm_bf16`,支持 rank >= 1 flatten/restore 和 CPU eager
   fallback. 默认 `weight_layout="transpose_stride"`;显式
   `weight_layout="prepacked_kn"` 保存一次性 contiguous K,N 派生 buffer.
3. Prepack buffer 为 `persistent=False`,不进入 `state_dict`;source Parameter
   identity/version 改变时刷新,metadata 记录 `prepared_weight_bytes` 和
   `prepack_refreshes`. 不存在全局 tensor/weight cache.
4. 显式 `linear_fastpath="graph"` capture activation -> Triton Linear -> output
   的完整固定签名分支. Replay 只复制动态 activation,weight,bias,layout 和
   schedule 保持静态;默认仍为 `linear_fastpath="eager"`.
5. Graph full key 覆盖 input shape/stride/dtype/device,kernel pattern,weight layout,
   target SM,schedule 和 weight/bias identity/version. 同签名 last-state 使用专用
   replay state;weight/bias 更新,prepack refresh 和 module `_apply()` 清空旧 graph.
6. Metadata 新增 `execution_mode="cuda_graph_triton_entry"`,
   `kernel_kind="cuda_graph_replay"`,`linear_fastpath` 以及
   `cuda_graph.state/reason/cache_size/output_storage`. Capture/replay 失败记录具体
   原因并先退回 eager Triton;错误不会静默.

### Correctness 和失效协议

专用测试覆盖 direct/nested materialization,deepcopy 参数绑定,CPU fallback,
FP16/BF16,两种 weight layout,rank-3 input,graph capture/replay,shape/stride/dtype
cache contract,weight/bias 更新重捕获,prepack refresh,`_apply()` 清理,capture
失败降级,operator-stage metadata 以及在 `torch.inference_mode()` 中创建的
Parameter. 全部 workload 对 FP32-cast reference 使用 `atol=0.5,rtol=0.03`.

正式 graph-stride wrapper 的误差为:

| shape | max_abs | mean_abs |
| --- | ---: | ---: |
| `M1,N4096,K4096,no bias` | `0.25` | `0.0002224334` |
| `M4,N4096,K4096,bias` | `1.0` | `0.0002283696` |
| `M64,N1024,K1024,bias` | `0.5` | `0.0000222794` |

### Weight-layout gate

Weight-layout benchmark 使用 5 个 R-027 shape,9 x 31 alternating paired gate.
Legacy 对照在每次 measured call 内执行 `weight.t().contiguous()`. 新
transpose-stride 相对 legacy 的 gap 为
`1707.65%/1573.68%/261.18%/31.42%/222.83%`,五组均 `9:0`.

显式 prepack 相对 eager transpose-stride 也在五组 `9:0`,gap 为
`16.22%/15.26%/94.87%/19.36%/9.07%`,但分别增加
`32/32/86/2/32 MiB` 派生权重. 因此 stateless/public 默认使用
transpose-stride;prepack 只属于 stateful materializer 的显式 opt-in.

### Wrapper,block 和 operator-stage gate

正式 wrapper benchmark 使用 M1/M4/M64,10 warmup,CUDA event median 和同一
9 x 31 promotion gate. 延迟单位为微秒:

| shape | Torch | eager wrapper | graph stride | graph prepack | eager block | graph block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M1` | `59.474` | `26.606` | `23.243` | `21.470` | `59.317` | `24.415` |
| `M4` | `20.278` | `27.815` | `29.572` | `29.067` | `23.380` | `31.766` |
| `M64` | `10.065` | `27.809` | `28.922` | `29.441` | `10.935` | `31.584` |

- M1 graph stride 相对 eager wrapper 和 Torch 分别 `9:0`,gap
  `21.26%/162.27%`;完整 graph block 相对 eager block `9:0`,gap `156.62%`.
- M4/M64 graph 与 eager wrapper 的 gap 只有 `1.35%/1.77%`,均低于 3% 门槛;
  Torch 分别以 `9:0`,gap `39.51%/179.23%` 稳定胜出.
- Graph prepack 与 graph stride 在三组 gap 仅 `0.22%/0.39%/0.85%`,全部
  noise-equivalent,不能用 graph 模式为额外权重内存辩护.

Operator-stage 使用相同 full-block scope,100 inner calls 和
`min_speedup=1.03`. M1 `61.834 -> 27.396 us`,speedup `2.257x`,applied;
M4 `21.752 -> 35.151 us`,speedup `0.619x`,rejected;M64
`12.560 -> 35.209 us`,speedup `0.357x`,rejected. Wrapper,完整 block 和
operator-stage 对 route winner 的判断一致.

### Profiler 归因

Nsight Systems 使用 `--cuda-graph-trace=node`,每个 NVTX range 配置 20 次
steady-state call. 个别 filtered group 显示 19 或 21 个关联实例,因此只使用
median,不从该计数推断 launch 数.

| shape | graph Triton | activation D2D | graph launch API | eager Triton | Torch kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M1` | `16.185 us` | `0.942 us` | `5.272 us` | `16.171 us` | `57.762 us` |
| `M4` | `16.908 us` | `0.975 us` | `5.067 us` | `16.829 us` | `21.223 us` |
| `M64` | `6.185 us` | `1.042 us` | `4.667 us` | `6.071 us` | `7.423 us` |

Graph 和 eager wrapper 执行同一 Triton kernel body. Graph range 只有动态
activation D2D copy,graph launch 和 GEMM node,没有 weight transpose/copy
kernel. M4/M64 的 kernel 本体仍快于 Torch kernel,但 activation copy,graph
launch 和 Python/module 边界使完整 wrapper 落后,所以不能把 kernel-only winner
写成 operator winner.

NCU probe 返回 `ERR_NVGPUCTRPERM`,`status=counter_permission_denied`. 本轮没有
推断 occupancy,L2/cache hit,warp stall,roofline,register pressure 或
tensor-pipe utilization.

### 路由和边界

- `gemm_with_precision(engine="auto")` 不变. 公共 stateless dispatcher 使用
  零物化 transpose-stride,但不保存模型权重或 graph state.
- Triton Linear materializer 默认 eager + transpose-stride. Graph 和 prepack 都
  必须显式选择;当前只有受测 SM89 BF16 M1 通过 full-block 3% promotion gate.
- Replay output 是 graph-owned storage;后续 replay 会覆盖旧 view,保留输出时必须
  clone. 当前只验证固定 shape,顺序 inference,不支持并发 replay 或 autograd.
- 当前证据只覆盖 `sm_89`,FP16/BF16 correctness 和上述 BF16 performance shape.
  其他 SM,dtype,shape,bias/activation 或动态 layout 需要独立 gate.

### 验证落点

- [Triton GEMM launcher](../../../xqt/operator_opt/kernels/triton/gemm.py)
- [Triton Linear wrapper and metadata](../../../xqt/operator_opt/triton_wrappers.py)
- [candidate materializer](../../../xqt/operator_opt/materialize.py)
- [wrapper tests](../../../tests/xqt/test_operator_triton_linear.py)
- [weight-layout benchmark](../../../research/xqt-gemm/bench_sm89_triton_bf16_weight_layout.py)
- [wrapper/graph benchmark](../../../research/xqt-gemm/bench_sm89_triton_bf16_linear_wrapper.py)
- [weight-layout artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-weight-layout/)
- [wrapper/graph artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-bf16-linear-wrapper/)

## R-029: SM89 Triton FP16 layout-aware schedule and Linear promotion gate

### 目标

| 项 | 值 |
| --- | --- |
| backend | Triton dense FP16 GEMM and stateful Linear materializer |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,Triton 3.7.1 |
| precision | FP16 activation,weight,bias,output;FP32 reduction 和 epilogue |
| schedule evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-fp16-schedule/` |
| wrapper evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-fp16-linear-wrapper/` |

R-027/R-028 已建立 BF16 exact schedule,零物化 N,K stride 和 stateful Linear
graph runtime. 本轮独立验证 FP16,并把 layout 纳入 schedule signature. Kernel
promotion 和 Linear runtime promotion 分开判定,不从 direct K,N winner 直接外推
wrapper 或 auto route.

### 实现

1. 新增 `resolve_triton_fp16_gemm_schedule(...)`,exact key 为
   `(M,N,K,has_bias,activation,transpose_b)`. 六个显式 schedule 字段逐项优先.
2. M1/M4/M8/M64 的受测 preset 同时覆盖 K,N 和 N,K. M256 GELU preset 只覆盖
   prepacked K,N;N,K 的候选收益低于 3%,保留旧默认.
3. `gemm_fp16_triton(...)` 解析逻辑 layout 后把 `transpose_b` 传给 resolver.
   `_TritonLinearWrapper` 的 FP16 metadata/resolution 路径同步传递自身
   `weight_layout`;BF16 resolver 签名和行为不变.
4. 新增 5-shape x 28-candidate 双布局 benchmark,torch.profiler,10 个独立
   NSYS workload,以及 M1/M4/M64 的 wrapper,完整 block,operator-stage 三层 gate.
5. FP16 kernel registry metadata 明确标记 schedule policy 包含 layout. Resolver
   和 SM cache 只保存不可变标量,不保存 tensor/weight.

### Correctness

数值 reference 是 FP32 Linear,bias 和 activation 完成后只 cast 一次 FP16.
28 个候选,十个 resolved layout workload,六个 wrapper workload 和三个
operator-stage workload 全部通过 `atol=0.25,rtol=0.03`.

| shape | direct resolved max_abs | graph wrapper max_abs |
| --- | ---: | ---: |
| `M1,N4096,K4096,no bias` | `0.125` | `0.125` |
| `M4,N4096,K4096,bias` | `0.125` | `0.125` |
| `M8,N11008,K4096,SiLU` | `0.125` | - |
| `M64,N1024,K1024,bias` | `0.0625` | `0.0625` |
| `M256,N4096,K4096,GELU` | `0.125` | - |

### Schedule 和 layout gate

方法为 10 次 warmup,shape-specific batched iteration,CUDA event end
synchronization 和 9 x 31 alternating paired sample. 稳定 promotion 要求至少
7/9 轮胜出且 gap 不低于 3%.

| shape | resolved schedule | K,N vs old default | N,K vs old default | K,N vs N,K |
| --- | --- | ---: | ---: | --- |
| `M1` | `16x64x64,gm4,4w,3s` | `530.30%`,9:0 | `563.45%`,9:0 | `0.05%`,noise |
| `M4` | `16x64x64,gm4,4w,3s` | `527.80%`,9:0 | `519.08%`,9:0 | `0.46%`,noise |
| `M8 SiLU` | `32x128x32,gm4,4w,3s` | `40.46%`,9:0 | `8.33%`,9:0 | K,N `93.69%`,9:0 |
| `M64` | `16x128x32,gm4,4w,3s` | `71.49%`,9:0 | `73.58%`,9:0 | `0.40%`,noise |
| `M256 GELU` | K,N `64x64x32,gm8,4w,3s`;N,K default | `4.71%`,9:0 | `0.14%`,noise | K,N `8.40%`,9:0 |

Actual resolver 与当轮 recommendation 在所有 promoted layout 上都低于 3% gap.
因此 preset 是受测平台上的代表 schedule,不声明跨 shape/SM 唯一最优.
M1/M4/M64 的 layout 差异为 noise-equivalent;只有 M8/M256 对 K,N 有稳定收益,
而 K,N 需要 stateful prepack 生命周期.

### Wrapper,block 和 operator-stage gate

三层 gate 覆盖 eager transpose-stride,graph transpose-stride,graph prepacked,
完整 block 和真实 operator-stage. 正式 paired 结论为:

| shape | graph vs eager wrapper | graph vs Torch | graph block vs eager block | operator-stage |
| --- | --- | --- | --- | --- |
| `M1` | graph `13.45%`,9:0 | graph `158.79%`,9:0 | graph `158.63%`,9:0 | `2.166x`,applied |
| `M4` | eager `4.19%`,9:0 | Torch `64.50%`,9:0 | eager `80.28%`,9:0 | `0.552x`,rejected |
| `M64` | eager `3.03%`,7:2 | Torch `190.18%`,9:0 | eager `168.05%`,9:0 | `0.392x`,rejected |

Graph prepack 与 graph stride 的 gap 为 `1.01%/0.04%/0.50%`,全部
noise-equivalent,但 prepack 分别额外占用 `32/32/2 MiB`. 只有 M1 在 wrapper,
完整 block 和 `min_speedup=1.03` operator-stage 三层一致通过. M4/M64 是明确
反例,因此整体 runtime action 为 `keep_explicit_opt_in`.

### Profiler 归因

Schedule artifact 包含 10 个 `.nsys-rep`,10 个 SQLite 和 150 个 CSV.
五十个 direct layout memory report 全部为空,说明 measured range 内没有 weight
copy. Resolved K,N/N,K kernel median 为:

| shape | K,N | N,K | old K,N default | Torch kernel |
| --- | ---: | ---: | ---: | ---: |
| `M1` | `15.904 us` | `15.762 us` | `113.316 us` | `57.472 us` |
| `M4` | `16.539 us` | `16.565 us` | `118.520 us` | `17.277 us` |
| `M8 SiLU` | `151.553 us` | `274.959 us` | `229.486 us` | `162.171 + 2.678 us` |
| `M64` | `7.551 us` | `7.989 us` | `31.057 us` | `6.127 us` |
| `M256 GELU` | `113.731 us` | `125.896 us` | `121.291 us` | `108.667 + 3.248 us` |

Wrapper artifact 使用 `--cuda-graph-trace=node`,包含 6 个 `.nsys-rep`,6 个
SQLite 和 54 个 CSV. Graph range 只有 activation D2D,graph launch 和原 GEMM:

| shape | graph Triton | activation D2D | graph launch API | eager Triton | Torch kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M1` | `16.276 us` | `0.975 us` | `5.021 us` | `15.873 us` | `57.120 us` |
| `M4` | `16.830 us` | `0.975 us` | `4.874 us` | `16.593 us` | `17.753 us` |
| `M64` | `8.071 us` | `1.042 us` | `6.069 us` | `8.045 us` | `6.141 us` |

M1 的 kernel 优势足以覆盖 graph 固定开销.M4 kernel 近似持平,M64 Triton
kernel 已慢于 Torch,所以 graph runtime 结论与三层 latency gate 一致.

两组 NCU probe 都返回 `ERR_NVGPUCTRPERM`,状态为
`counter_permission_denied`. 本轮没有采集或推断 occupancy,L2/cache hit,warp
stall,roofline,register pressure 或 tensor-pipe utilization.

### 路由和边界

- `gemm_with_precision(engine="auto")` 的 selector 不变. 现有 stateless Triton
  路径可消费 FP16 resolver,但不保存 weight 或 graph state.
- `_TritonLinearWrapper` 默认继续为 eager + transpose-stride. Graph 和 prepack
  都是显式 opt-in;不建立全 shape Linear auto route.
- M1 只进入 future exact-signature route review,不因单个 shape 通过而改变通用
  facade 默认语义.
- Replay output 是 graph-owned storage;保留跨 replay 输出时必须 clone. 当前只
  验证固定 shape,顺序 inference,不覆盖并发 replay,autograd,其他 SM 或动态 layout.

### 验证落点

- [Triton GEMM launcher](../../../xqt/operator_opt/kernels/triton/gemm.py)
- [Triton Linear wrapper](../../../xqt/operator_opt/triton_wrappers.py)
- [schedule regression](../../../tests/operator_opt/test_gemm_precision.py)
- [wrapper regression](../../../tests/xqt/test_operator_triton_linear.py)
- [schedule benchmark](../../../research/xqt-gemm/bench_sm89_triton_fp16_schedule.py)
- [wrapper benchmark](../../../research/xqt-gemm/bench_sm89_triton_fp16_linear_wrapper.py)
- [schedule artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-fp16-schedule/)
- [wrapper artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-fp16-linear-wrapper/)

## R-030: SM89 Triton forward FlashAttention engine and exact decode presets

### 目标

| 项 | 值 |
| --- | --- |
| backend | Triton forward-only FlashAttention-style attention |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130,Triton 3.7.1 |
| precision | matching FP16/BF16 Q/K/V/output,FP32 score,online softmax 和 output accumulator |
| layout | contiguous `[batch,heads,seq,head_dim]` |
| evidence | `research/xqt-gemm/artifacts/2026-08-09-sm89-triton-attention/` |

TileLang attention 已有独立 FP16/BF16 kernel,但本轮需要一个可单独 profile 的
Triton forward engine,用于比较 decode 和 causal prefill 的 kernel/runtime 边界.
本轮只增加显式 Triton registry entry 和 exact schedule resolver,不把 Triton
接入 attention `auto`.

### 实现和 contract

1. 新增 `xqt/operator_opt/kernels/triton/attention.py`,使用 online softmax
   mainloop. score 先按 FP32 计算,`exp2` 使用 `log2(e)` 缩放,accumulator 保持
   FP32,最终一次 cast 到 input/output dtype.
2. kernel 入口要求 Q/K/V 为 contiguous BHSD,同 device,同 dtype,`head_dim` 为
   16/32/64/128,并要求 `seq_kv >= seq_q`. `dropout_p` 只能为 0,不提供 backward.
3. causal mask 对方形和非方形统一使用 lower-right 语义. CPU 或非 CUDA 输入经
   registry fallback 调 SDPA reference,不静默把 CUDA contract 改成另一个 layout.
4. backend registry 新增 `attention` Triton spec,metadata 明确 supported dtypes,
   head dims,causal semantics,contiguous requirement 和 inference-only boundary.
5. 默认 schedule 为 `64x64,4 warps,2 stages`;显式 `block_m/block_n/num_warps/
   num_stages` 逐字段优先,SM cache 只保存不可变 scalar metadata,不保存 tensor.

### Correctness 和 sweep

Benchmark 覆盖 FP16/BF16 x small `64x64,D32`,medium `256x256,D64`,long
`1024x1024,D64`,causal `512x512,D64` 和 decode `1x1024,D64`,共 19 个 Triton
candidate. 10 次 warmup 后使用 CUDA-event median 和 9 x 31 A-B/B-A paired gate,
promotion 门槛为至少 7/9 轮且 gap 至少 3%. 所有可执行候选对 SDPA reference
通过 `atol=0.02,rtol=0.02`,最大绝对误差 `0.00390625`.

SM89 resolver 最终只保留两个 decode preset:

| dtype/shape | preset | independent exact audit |
| --- | --- | --- |
| FP16 `B1,H8,Sq1,Skv1024,D64,causal` | `sm89_fp16_decode_q1_kv1024_d64`: `16x64,4w,2s` | 5 seeds,每次 `9:0`,gap `21.87-24.94%` |
| BF16 同 shape | `sm89_bf16_decode_q1_kv1024_d64`: `16x128,4w,2s` | 5 seeds,每次 `9:0`,gap `24.85-26.54%` |

首轮 sweep 的 FP16 decode candidate `bm32_bn128_w4_s2` 与最终 resolver 的
`bm16_bn64_w4_s2` 不同;正式 promotion 以
`exact-preset-audit.json` 的 actual resolver-vs-default 结果为准. long prefill
曾观察到 stages=3 略有优势,但 FP16/BF16 independent audit 的 gap 范围分别为
`2.84-3.44%` 和 `2.82-3.44%`,至少一个 seed 低于门槛,所以仍使用 stages=2.

### Backend gate 和路由

初始 sweep 的 `triton_recommended` 与其他 backend 的 paired gate 全部稳定
`9:0`:

| workload | FP16 vs SDPA | FP16 vs TileLang | BF16 vs SDPA | BF16 vs TileLang |
| --- | --- | --- | --- | --- |
| small prefill | SDPA `51.83%` | TileLang `60.75%` | SDPA `16.22%` | TileLang `52.00%` |
| medium prefill | SDPA `15.40%` | TileLang `52.84%` | SDPA `13.89%` | TileLang `50.39%` |
| long prefill | SDPA `3.37%` | TileLang `6.75%` | SDPA `6.89%` | TileLang `10.74%` |
| causal prefill | Triton `22.63%` | TileLang `48.53%` | Triton `21.01%` | TileLang `43.75%` |
| decode | Triton `62.27%` | TileLang `13.58%` | Triton `67.02%` | TileLang `10.00%` |

因此 Triton 在当前 SM89 上对 SDPA 的 causal/decode latency 有稳定收益,但
TileLang 仍在全部五类 workload 胜出. Triton 作为显式 engine 可执行,不进入
`auto`;native SDPA 和现有 TileLang route 保持不变. Exact resolver 的独立 audit
最初只直接比较 resolver 与 Triton default,不能把上表 candidate gate 直接当成
resolver 对 SDPA/TileLang 的证明.

### Exact resolver cross-engine re-audit

随后新增 `exact-preset-engine-audit.json`:两个 actual resolver 在每个独立 seed
先通过 SDPA correctness gate,再以相同的 9 x 31 A-B/B-A CUDA-event 策略直接与
TileLang default 和 SDPA 对照. FP16/BF16 decode 都在 5/5 seed 以 `9:0` 稳定胜过
SDPA,relative gap 分别为 `35.78-40.40%` 和 `40.09-45.86%`. 但对 TileLang:

| dtype | exact resolver vs TileLang | route decision |
| --- | --- | --- |
| FP16 decode | 3/5 seed 由 TileLang stable `9:0` 胜出,gap `3.24-9.01%`;另外 2 个 seed 为 noise-equivalent | `keep_explicit_only` |
| BF16 decode | 4/5 seed 由 TileLang stable `9:0` 胜出,gap `3.01-4.79%`;其余 1 个 gap `2.95%` 低于门槛 | `keep_explicit_only` |

该结果是 XQT engine-entry gate,不是完整 attention wrapper 或模型 block 的
operator-stage gate. 若 resolver 在这里已经未能稳定击败 TileLang,就不应继续为
route 接入引入 wrapper 或 CUDA Graph 复杂度. `attention_fastpath="auto"` 保持不变.

### Wrapper/runtime 分层与 causal 修复

为确认 kernel microbenchmark 与完整 MHA 的差异,对 exact decode signature
`B1,H8,Sq1,Skv1024,D64,causal` 分别测量 Triton exact kernel,TileLang kernel,
TileLang dispatcher,完整 MHA forward,wrapper entry 和 native lower-right SDPA.
Benchmark 使用 10 次 warmup,每个 sample 40 次调用,15 个 CUDA-event sample 取
median;torch.profiler 对每层另记录 20 次调用. 所有输出直接对同一 lower-right
reference 做 `atol=0.02,rtol=0.02` correctness gate.

| dtype | Triton kernel | TileLang kernel | TileLang dispatcher | TileLang full | TileLang entry | native full | native entry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | `0.017997 ms` | `0.018326 ms` | `0.018712 ms` | `0.103610 ms` | `0.121167 ms` | `0.132009 ms` | `0.142391 ms` |
| BF16 | `0.018046 ms` | `0.019076 ms` | `0.018725 ms` | `0.111622 ms` | `0.120185 ms` | `0.132861 ms` | `0.141874 ms` |

首次 profile 的 correctness gate 暴露出 native wrapper 语义错误. TileLang/Triton
对 `Sq != Skv` causal 使用 lower-right alignment,而 native 路径直接把
`is_causal=True` 传给 SDPA,得到 upper-left alignment;FP16/BF16 native 输出相对
lower-right reference 的最大绝对误差约为 `1.4961/1.4904`. 修复后的共享 helper
在方形 causal 上继续使用 `is_causal=True`,非方形 causal 则显式构造
`torch.nn.attention.bias.causal_lower_right`;普通 MultiheadAttention wrapper 和
`xqt.nn.Attention` wrapper 复用这一 contract. 最终所有层都通过 gate,native 的
max abs 为 0.

torch.profiler 显示完整 MHA 每次包含 4 个 Linear,2 个 contiguous/layout copy,
再加 1 个 TileLang attention kernel;native lower-right SDPA 使用 split-KV compute
和 combine. 因此 direct kernel 只占完整 MHA 的一部分. 上表是单次 sequential
分层 profile,不能覆盖或推翻 5-seed `9 x 31` paired gate;即使某次 Triton kernel
median 略低,也不支持 Triton 自动替换 TileLang. `attention_fastpath="auto"` 不变.

### Nsight Systems 归因

五个独立进程 report 分别覆盖 `triton_resolved`,`triton_recommended`,
`triton_default`,`tilelang_default` 和 `sdpa`,每个 dtype/shape 通过 NVTX filter
导出 `cuda_gpu_kern_sum`,`cuda_api_sum`,`cuda_gpu_mem_time_sum`.

| dtype/shape | resolved Triton kernel median | TileLang kernel median | SDPA kernel medians | launch API calls per 20 calls |
| --- | ---: | ---: | ---: | --- |
| FP16 causal | `12.520 us` | `10.831 us` | `12.369 + 4.190 us` | `20/20/40` |
| BF16 causal | `12.588 us` | `10.662 us` | `12.335 + 4.191 us` | `20/20/40` |
| FP16 decode | `15.287 us` | `17.330 us` | `6.470 + 2.849 us` | `20/20/40` |
| BF16 decode | `12.047 us` | `17.330 us` | `6.537 + 2.849 us` | `20/20/40` |

Triton/TileLang 每次一个 launch,SDPA causal/decode 为 split-KV compute + combine
两个 launch. `cuda_gpu_kern_sum` 只统计 GPU kernel body,不包含 launch 之间的
host/API gap,因此端到端 winner 仍以 CUDA-event paired gate 为准. 过滤后的 kernel
instance 可能显示 19/21,launch 结论只使用 API report 的 20/40 次.

这也解释了为什么不可以用 kernel median 替代 engine-entry gate:BF16 decode 的
resolved Triton kernel median 为 `12.047 us`,TileLang 为 `17.330 us`,但 direct
paired gate 仍不满足 Triton 取代 TileLang 的条件. NCU 受权限阻断,不把这一层差异
猜测为 cache,occupancy 或 warp-stall 问题.

50 个逐区间 GPU memory report 均为空,不能据此写 DRAM bandwidth,L2 hit rate 或
kernel 内存流量结论. NCU 返回 `ERR_NVGPUCTRPERM` (`counter_permission_denied`),
本轮不推断 occupancy,warp stall,roofline,register pressure 或 tensor-pipe 指标.

### 路由和边界

- `run_triton_kernel("attention",...)` 和 direct Triton function 是显式 engine
  入口;`attention_fastpath="auto"` 不变.
- 当前实现是固定签名的模型侧 inference operator,不包含 backward,paged KV cache,
  cache eviction,continuous batching 或 serving scheduler.
- 其他 SM,dtype/shape,非 contiguous layout,动态 mask/shape 和真实模型 block 需要
  独立 correctness,paired gate 和 profiler 证据,不能从本轮 SM89 结果外推.
- exact preset 只服务已审计 decode signature;其余签名继续使用默认 schedule,
  不建立全局 shape heuristic.

### 验证落点

- [Triton attention kernel](../../../xqt/operator_opt/kernels/triton/attention.py)
- [Triton backend registry](../../../xqt/operator_opt/backends/triton.py)
- [attention kernel exports](../../../xqt/operator_opt/kernels/attention.py)
- [CUDA/CPU contract tests](../../../tests/xqt/test_operator_triton_attention.py)
- [sweep and profiling entry](../../../research/xqt-gemm/bench_sm89_triton_attention.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-09-sm89-triton-attention/)

## R-031: SM89 ConvRot/SVDQuant native CUDA dynamic fusion

### 目标

| 项 | 值 |
| --- | --- |
| backend | custom CUDA binding + Nunchaku W4A4/W8A8 fragment/layout |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),driver 591.86,CUDA 13.0,torch 2.12.1+cu130 |
| modules | `ConvRotInt8Linear`,`SVDQuantLinear`,`SVDQuantInt8MmaLinear` |
| contracts | ConvRot W8A8 dynamic per-token;SVDQuant W4A4 dynamic LoRA;SVDQuant W8A8 dynamic LoRA |
| shapes | `M64,K=N1024,R32`,`M256,K=N1024,R32`,`M1024,K=N2048,R64` |
| norm | 本轮动态 fastpath 不融合 norm,`norm_fused=false` |

目标不是用 Python operator composition 或 `torch.compile` 隐藏 split overhead,而是
落实与原始优化实现同构的专用 CUDA fusion,并让 XQT runtime wrapper 在稳态接近或
超过 direct native floor.

### 公平基线与测量

固定入口为
`research/xqt-gemm/bench_sm89_convrot_svdq_fusion.py`. 每个 candidate 先 warmup
30 次. ConvRot W8A8 和 SVDQuant W8A8 做 15 轮,每轮 500 次;host dispatch 占比更高
的 SVDQuant W4A4 做 31 轮,每轮 2000 次. 每轮交替 forward/reverse candidate 顺序.
延迟使用 CUDA event 和 `end.synchronize()`,排除 JIT build,weight packing 和
workspace allocation. 三类 baseline 定义如下:

1. ConvRot split baseline 显式 materialize regular-Hadamard rotation,再调用与 fused
   candidate 相同的 native dynamic quant/W8A8 GEMM. 因此它没有少做 rotation 或
   偷换 GEMM backend.
2. SVDQuant W4A4 的 `nunchaku_native_direct_floor` 直接调用仓库内 Nunchaku
   fragment/layout,不是外部 wheel 的整模块结果.`nunchaku_native_module` 额外加入
   相同 `nn.Module` 调用契约.`xqt_bound_native` 单列 C++ bound runner,用于分离
   CUDA/native dispatch 与 Python module 固定开销.
3. SVDQuant W8A8 split baseline 使用相同 native dynamic W8A8 residual,再执行独立
   LoRA down/up 和 add. 所有 candidate 共用同一输入,packed weight,workspace policy
   和同步方法.

完整样本在
`research/xqt-gemm/artifacts/2026-08-10-sm89-convrot-svdq-fusion/result.json`.

### 专用融合实现

1. ConvRot W8A8 前端 CUDA kernel 在一个 pass 中完成 regular-Hadamard rotation,
   padding,dynamic per-token INT8 quant 和 scale 写出,随后在同一次 pybind dispatch
   中启动 Nunchaku packed W8A8 GEMM. 不再物化浮点 rotated activation.
2. SVDQuant W4A4 采用 Nunchaku 两阶段 dynamic LoRA. FUSE_DOWN kernel 同时完成
   activation W4 quant 和 LoRA down;FUSE_UP 在 W4A4 GEMM epilogue 内加入 LoRA up
   与 bias. 两个依赖 kernel 由一个 C++ entry 连续 launch.
3. W4A4 进一步增加 C++ bound runner:packed weight,scale,LoRA layout 和 workspace
   在绑定时一次性校验,稳态调用只传 activation. 2D 输入不再做无意义 reshape,
   metadata 只在 backend 状态切换时更新.
4. SVDQuant W8A8 第一阶段融合 dynamic INT8 quant 与 LoRA down,第二阶段融合 INT8
   GEMM,LoRA up 和 bias. ConvRot/SVDQuant hot cache 都覆盖 device,dtype,row count,
   current CUDA stream 和 source tensor identity/version;原地更新 scale/weight 或
   module `_apply()` 后会重新 pack.
5. 旧的 opt-in `ConvRotNormInt8Linear` 保留,但本轮 dynamic native path 不接 norm,
   不把相邻 norm 的时间移出 baseline,metadata 明确记录 `norm_fused=false`.

### CUDA-event 结果

ConvRot W8A8 的完整 XQT wrapper 对同契约 split baseline:

| shape | dtype | XQT | split | speedup | XQT/direct |
| --- | --- | ---: | ---: | ---: | ---: |
| `M64,K=N1024` | FP16 | `0.037235 ms` | `0.056733 ms` | `1.524x` | `0.997x` |
| `M64,K=N1024` | BF16 | `0.036526 ms` | `0.055310 ms` | `1.514x` | `0.997x` |
| `M256,K=N1024` | FP16 | `0.037349 ms` | `0.059535 ms` | `1.594x` | `1.004x` |
| `M256,K=N1024` | BF16 | `0.036902 ms` | `0.058713 ms` | `1.591x` | `0.999x` |
| `M1024,K=N2048` | FP16 | `0.070867 ms` | `0.134869 ms` | `1.903x` | `1.001x` |
| `M1024,K=N2048` | BF16 | `0.071064 ms` | `0.135391 ms` | `1.905x` | `1.003x` |

SVDQuant W4A4:

| shape | XQT wrapper | XQT C++ bound | direct floor | native module | split | XQT/split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M64,K=N1024,R32` | `0.023292 ms` | `0.022767 ms` | `0.023356 ms` | `0.022888 ms` | `0.040191 ms` | `1.726x` |
| `M256,K=N1024,R32` | `0.024504 ms` | `0.024013 ms` | `0.023973 ms` | `0.024252 ms` | `0.036823 ms` | `1.503x` |
| `M1024,K=N2048,R64` | `0.038532 ms` | `0.038545 ms` | `0.038580 ms` | `0.038546 ms` | `0.055576 ms` | `1.442x` |

C++ bound runner 对 direct floor 分别为快 `2.52%`,慢 `0.17%`,快 `0.09%`;
后两项属于亚百分比 noise-equivalent. 完整 XQT `nn.Module` wrapper 对 strict direct
floor 分别为快 `0.27%`,慢 `2.22%`,快 `0.13%`,最大绝对差 `0.53 us`;对同
`nn.Module` baseline 的最大差为 `1.77%`. 因此 native CUDA 和完整 wrapper 都已
追平 source-level Nunchaku floor,没有观察到 kernel 回退.

SVDQuant W8A8:

| shape | XQT | split | speedup | XQT/direct |
| --- | ---: | ---: | ---: | ---: |
| `M64,K=N1024,R32` | `0.081594 ms` | `0.094935 ms` | `1.163x` | `1.004x` |
| `M256,K=N1024,R32` | `0.081318 ms` | `0.097046 ms` | `1.193x` | `1.000x` |
| `M1024,K=N2048,R64` | `0.170236 ms` | `0.194406 ms` | `1.142x` | `1.000x` |

数值验证中,ConvRot fused/split relative RMSE 为 0,仅 large FP16 为
`1.94e-5`;XQT/direct 最大绝对差为 0. W4A4 direct/split relative RMSE 为
`2.20e-4` 到 `2.59e-4`,XQT/direct 最大绝对差不超过 `0.0009765625`. W8A8
fused/split relative RMSE 为 `0.00176-0.00212`,XQT/direct 最大绝对差为 0.

### Nsight Systems 归因

profile range 在 500 次 workload 完成并同步后才关闭. Direct native 和 XQT range
都严格出现 500 个 FUSE_DOWN kernel,500 个 FUSE_UP/W4A4 GEMM kernel 和 500 次
`cudaMemsetAsync`:

| range | FUSE_DOWN median | FUSE_UP/GEMM median | launch API |
| --- | ---: | ---: | ---: |
| direct native | `5.155 us` | `12.097 us` | 1000 `cudaLaunchKernel` + 500 `cudaMemsetAsync` |
| XQT wrapper | `5.155 us` | `12.068 us` | 1000 `cudaLaunchKernel` + 500 `cudaMemsetAsync` |

这证明 XQT wrapper 运行的是同一套真实 FUSE_DOWN/FUSE_UP kernel,没有退回 Python
split. Split trace 还包含独立 LoRA down/up GEMM,split-K reduction 和 add kernel;
加速来自删除这些独立阶段和中间 materialization.

NCU preflight 返回 `ERR_NVGPUCTRPERM`,artifact 状态为
`counter_permission_denied`. 这表示当前用户无权读取 NVIDIA performance counters,
不是 kernel correctness 错误. 本轮因此不推断 occupancy,L2/cache,warp stall,
roofline,register pressure 或 tensor-pipe utilization;管理员启用 counter policy 后
才能继续该层归因.

### 边界与回退

- 当前 native route 只为 `sm_89` 启用. ConvRot 支持已声明的 FP16/BF16 activation,
  rotation size 和对齐 shape;SVDQuant W4A4 当前验证 FP16,W8A8 当前验证 BF16.
- 不支持的 SM,dtype,rank,shape,device 或 engine 显式回到既有 TileLang/Triton/eager
  路径,并记录 native fallback reason.
- weight packing,workspace allocation 和 extension JIT 不在 steady-state latency 中;
  它们由 cache 复用并在 source tensor version,device/dtype 或 stream contract 变化时
  失效.
- 该结果是 module-level inference operator 证据,不外推到任意模型 block,其他 GPU
  架构或 serving scheduler.

### 未采纳方案与可复用规则

- 不把 `torch.compile` 组合图称为 native fusion. 当前三条路径都进入自有 CUDA/C++
  entry.
- 不强行把 Nunchaku 两阶段 dynamic LoRA 合成一颗巨型 kernel. FUSE_DOWN 和
  FUSE_UP 各自消除对应的中间算子,两者保持必要依赖并由单次 C++ dispatch 提交.
- 不为本轮动态路径加入 norm fusion. 若需要 norm fusion,必须单独建立同契约 baseline
  和 promotion gate.
- kernel/native floor 与完整 `nn.Module` wrapper 必须分层报告. 小 kernel 上数微秒
  Python guard 足以改变端到端名次,不能把 wrapper gap 错归因给 CUDA kernel.
- NCU counter 权限失败是 blocked measurement,不能用 CUDA-event latency 或经验猜测
  代替硬件 counter.

### 验证落点

- [ConvRot W8A8 runtime](../../../xqt/quant/quantizers/convrot_int8.py)
- [ConvRot W8A8 native kernel](../../../xqt/operator_opt/kernels/cute/convrot_w8a8_sm89.py)
- [SVDQuant runtime](../../../xqt/runtime/modules/svd_composite.py)
- [SVDQuant W4A4 native kernel](../../../xqt/operator_opt/kernels/cute/svdq_w4a4_sm89.py)
- [SVDQuant W8A8 native kernel](../../../xqt/operator_opt/kernels/cute/svdq_w8a8_sm89.py)
- [CUDA benchmark/profile entry](../../../research/xqt-gemm/bench_sm89_convrot_svdq_fusion.py)
- [ConvRot CUDA tests](../../../tests/xqt/quant/test_convrot_int8_quantizer.py)
- [SVDQuant CUDA tests](../../../tests/xqt/runtime/test_svd_fusion.py)
- [runtime cache tests](../../../tests/xqt/runtime/test_composite_runtime_caches.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-10-sm89-convrot-svdq-fusion/)

## R-032: SM89 ConvRot W4A4 warp-FHT rowwise INT4 fusion

### 目标

| 项 | 值 |
| --- | --- |
| backend | custom CUDA binding + warp-FHT quantizer + CUTLASS W4A4 Tensor Core GEMM |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),driver 591.86,CUDA 13.0,torch 2.12.1+cu130 |
| module | `ConvRotMixedPrecisionLinear` |
| contract | dynamic row activation scale,rowwise signed INT4 weight,FP32 weight scale and bias epilogue |
| shapes | `M={64,256,1024}`,`K=N={1024,2048}`,FP16/BF16 |
| norm | 本轮不融合 norm,`norm_fused=false` |

本轮针对 ConvRot W4A4 的真正热路径,目标是复现官方实现中的专用 warp-FHT + INT4 contract,而不是把 rotation,quant 和 GEMM 作为 Python eager 子算子串接. 重点同时验证 XQT 完整 wrapper 是否追平 native bound floor,以及与官方 CUDA wheel 的同 contract 对照是否公平.

逐步实现,数学和源码阅读说明见 [ConvRot W4A4 SM89 CUDA 优化详解](convrot-w4a4-sm89-optimization.md).

### 公平 baseline 与测量

正式入口为 `tools/benchmark_convrot_w4a4_sm89.py`. 每个 case 先 warmup 30 次,然后用 CUDA event 测量 15 个样本,每个样本连续调用 1000 次并在样本末尾同步. 正式运行固定 CPU affinity `[22]`,排除 extension 编译,weight packing 和 workspace allocation. 对照分层如下:

1. `comfy-kitchen 0.2.28` 官方 CUDA baseline 使用 wheel 中的 `convrot_w4a4.cu` 实现,源码 commit 为 `b72e6dfa79b79a7aee33a9c7608b5d9b3005b7af`. 官方和 XQT 收到完全相同的 packed signed INT4 weight,row scale 和 bias,调用层级都是完整 Python operator steady state.
2. `nunchaku_bound` 使用相同 dequantized artifact 的 Nunchaku grouped W4A4 bound runner,用于隔离已有 grouped/native contract 的差异.
3. `split_rotation_then_nunchaku` 先显式物化 regular-Hadamard rotated activation,再调用相同 Nunchaku W4A4 GEMM,不减少 rotation 工作也不替换 GEMM backend.
4. `rowwise_bound_floor`,`rowwise_dynamic_runner` 和 `xqt_wrapper` 分别用于拆分 C++ bound,动态 stream-aware runner 与完整 `nn.Module` 调用开销.

### 专用融合结构

1. 第一个 CUDA kernel 在 warp 内执行 256 点 regular-Hadamard/FHT. FP16 使用 `half2`,BF16 使用 `bf162`;旋转结果直接计算 row absmax,写出 row-major packed signed INT4 activation 和一个 FP32 row scale,不分配或写回旋转后的 FP16/BF16 activation.
2. 第二个 kernel 使用 CUTLASS `s4 x s4 -> s32` Tensor Core GEMM. epilogue 在 FP32 中融合 activation scale,weight scale 和 bias,输出一次 cast 到输入 dtype.
3. C++ binding 每次热调用只传 activation. dynamic runner 按 `(rows,CUDA stream)` 缓存 workspace,高维输入先显式展平,non-contiguous 输入只做必要 contiguous materialization. source weight,weight scale 或 bias 的 tensor version 变化会使 bound state 失效并重新 pack;`.to()` / `_apply()` 清空 Python runtime cache.
4. 该实现的 native 热路径固定为两个 CUDA kernel. 它不是将多个 eager op 交给 `torch.compile` 后的偶然融合,也不把 norm 时间移出 baseline.

### Runtime backend policy

| `w4a4_runtime_backend` | 解析语义 |
| --- | --- |
| `auto` | 仅当 `group_size == padded_input_features` 时选择 rowwise;grouped artifact 继续走 Nunchaku,不静默改 scale contract |
| `rowwise` | 显式请求 rowwise warp-FHT path;不满足 capability 时按既有顺序回退 Nunchaku,再回退 reference |
| `nunchaku` | 禁止 rowwise,保留 grouped/Nunchaku contract |
| `reference` | 禁止 native W4A4,使用 PyTorch reference path |

rowwise capability 要求 CUDA `sm_89`,FP16/BF16,dynamic activation scale,未旋转输入,无 channel-hybrid,`rot_size=256`,输入 `K=1024` 或 `K%2048==0` 且 `1024<=K<=32768`,输出 `N%8==0`,并要求输入 trailing dimension 与 artifact 对齐. 其他 device,SM,dtype,静态 scale,预旋转输入或不支持 shape 都保留明确 fallback reason.

### CUDA-event 结果

下表为 12 个正式 case 的中位延迟,单位为 `us`. `official` 是 `comfy-kitchen 0.2.28`,`nunchaku` 是 grouped bound path,`split` 是显式 rotation + Nunchaku,`xqt` 是完整 XQT wrapper.

| dtype | M | K=N | xqt | official | nunchaku | split | official/xqt speedup | nunchaku/xqt speedup | split/xqt speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 | 64 | 1024 | 13.297 | 40.140 | 17.094 | 33.400 | 3.019x | 1.286x | 2.512x |
| FP16 | 256 | 1024 | 13.741 | 39.769 | 17.595 | 32.064 | 2.894x | 1.280x | 2.333x |
| FP16 | 1024 | 1024 | 16.507 | 48.290 | 19.389 | 34.484 | 2.925x | 1.175x | 2.089x |
| FP16 | 64 | 2048 | 16.783 | 39.613 | 25.499 | 35.836 | 2.360x | 1.519x | 2.135x |
| FP16 | 256 | 2048 | 16.549 | 48.456 | 25.568 | 39.612 | 2.928x | 1.545x | 2.394x |
| FP16 | 1024 | 2048 | 26.766 | 74.161 | 34.389 | 50.174 | 2.771x | 1.285x | 1.875x |
| BF16 | 64 | 1024 | 13.685 | 40.330 | 18.105 | 31.119 | 2.947x | 1.323x | 2.274x |
| BF16 | 256 | 1024 | 14.148 | 40.100 | 17.397 | 32.609 | 2.834x | 1.230x | 2.305x |
| BF16 | 1024 | 1024 | 16.255 | 48.564 | 20.123 | 34.295 | 2.988x | 1.238x | 2.110x |
| BF16 | 64 | 2048 | 16.659 | 40.155 | 25.398 | 35.906 | 2.410x | 1.525x | 2.155x |
| BF16 | 256 | 2048 | 16.214 | 49.544 | 25.708 | 38.776 | 3.056x | 1.586x | 2.392x |
| BF16 | 1024 | 2048 | 26.533 | 74.074 | 34.455 | 50.408 | 2.792x | 1.299x | 1.900x |

汇总结果为:XQT 相对官方 `2.360x-3.056x`,中位 `2.910x`;相对 Nunchaku `1.175x-1.586x`,中位 `1.292x`;相对 split `1.875x-2.512x`,中位 `2.215x`. `xqt_wrapper / rowwise_bound_floor` 为 `0.995x-1.083x`,中位 `1.034x`,说明完整 wrapper 已基本贴住 native floor. XQT 延迟范围为 `13.297-26.766 us`.

### Nsight Systems 归因

代表 shape 为 BF16 `M=256,N=K=2048`,每个 NVTX range 100 次. 刷新后的 `nvtx-kernel-summary.csv` 显示:

| range | kernel sequence | instances per range | median kernel time |
| --- | --- | ---: | ---: |
| XQT rowwise | warp-FHT quant + CUTLASS W4A4 GEMM | 100 + 100 | `2.761 us + 9.436 us` |
| official | official warp-FHT quant + `int4_linear_kernel` | 100 + 100 | `3.186 us + 16.976 us` |
| Nunchaku | rotated activation quant + W4A4 GEMM | 100 + 100 | `4.399 us + 18.812 us` |
| split | copy + BF16 rotation GEMM + padded quant + W4A4 GEMM | 100 x 4 | `1.002 us + 5.886 us + 6.918 us + 18.872 us` |

这份 trace 证明 XQT wrapper 进入了两 kernel native contract,没有退回 Python split. `cuda_gpu_kern_sum` 的 kernel median 不包含 host launch gap,所以正式名次仍以同样测量政策下的 CUDA-event benchmark 为准. NCU preflight 返回 `ERR_NVGPUCTRPERM` (`counter_permission_denied`),表示当前环境禁止读取 NVIDIA performance counters,不是 kernel 错误;本轮不推断 occupancy,L2/cache,warp stall,roofline,register pressure 或 tensor-pipe utilization.

### 数值正确性与取舍

- wrapper 与 dynamic runner 在 12 个 case 中完全一致,rowwise direct 与 bound 结果一致,激活量化 code 相对数学 reference 的最大差不超过 1.
- rowwise 相对官方输出的 relative RMSE 为 `0.001905-0.012858`,最大绝对差为 `0.6982421875`;这是量化舍入差,不能写成 bitwise 等价.
- rowwise 相对 grouped dense artifact 的 relative RMSE 为 `0.1879-0.2045`,Nunchaku grouped path 为 `0.1224-0.1233`. 因此 `auto` 必须保留 artifact scale contract 判断,不能为追求 kernel 速度把 grouped artifact 静默重解释为 whole-row scale.
- 本轮不做 norm fusion. `norm_fused=false` 是 metadata 和 benchmark contract 的固定事实;如需 norm fusion,必须另建包含相邻 norm 的同层 baseline 和 promotion gate.

### 验证落点与边界

- [ConvRot W4A4 runtime](../../../xqt/quant/quantizers/convrot_4bit.py)
- [rowwise Python binding](../../../xqt/operator_opt/kernels/cute/convrot_w4a4_rowwise_sm89.py)
- [rowwise C++ binding](../../../xqt/operator_opt/kernels/cute/convrot_w4a4_rowwise_sm89_binding.cpp)
- [rowwise CUDA kernel](../../../xqt/operator_opt/kernels/cute/convrot_w4a4_rowwise_sm89_kernel.cu)
- [policy and runtime tests](../../../tests/xqt/quant/test_convrot_4bit_quantizer.py)
- [CUDA benchmark entry](../../../tools/benchmark_convrot_w4a4_sm89.py)
- [Nsight Systems entry](../../../tools/profile_convrot_w4a4_sm89.py)
- [benchmark artifact](../../../artifacts/xqt/benchmarks/convrot_w4a4_sm89/summary.json)
- [profiling artifact](../../../artifacts/xqt/profiling/convrot_w4a4_sm89/)

当前 native route 只对已声明的 `sm_89` FP16/BF16 shape 负责性能承诺. 其他 SM,静态 scale,预旋转输入,channel-hybrid,不对齐特征或缺少 CUDA/CUTLASS 扩展时,实现会记录 fallback reason 并回到 Nunchaku 或 reference;本轮结果不外推到其他架构或完整模型 block.

## R-033: SM89 TileLang FP16 Linear exact decode schedule

### 目标

| 项 | 值 |
| --- | --- |
| backend | TileLang direct `dense_linear_epilogue` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130 |
| precision | FP16 input/weight/bias/output,FP32 GEMM accumulation |
| promoted shapes | `M=1/4,N=4096,K=4096,activation=None` |
| negative controls | `M=1,N=11008,K=4096,bias+SiLU`;`M=4,N=11008,K=4096,bias` |
| candidate | `16x64x32,128 threads,2 stages` |
| prior default | `64x64x64,128 threads,2 stages` |

目标是修正显式 TileLang engine 在 tiny-M FP16 decode 上的默认 schedule,不是改变
全局 Linear engine. BF16 已有只按 `flattened M<=4` 的 preset,但本轮负对照证明
FP16 不能复用这个宽条件:N=11008 的两个 shape 都继续由旧默认获胜.

### 基线与测量

正式入口为
`research/xqt-gemm/bench_sm89_tilelang_linear_fp16.py`. 四个 shape 共用 10 个
TileLang candidate,真实 `tilelang_resolved`,Triton resolved 和
`torch.nn.functional.linear` baseline. 数值 reference 在 FP32 中执行 Linear 和
activation,只在输出处 cast 一次 FP16;容差为 `atol=0.25,rtol=0.03`.

顺序 sweep 每个 candidate 先 warmup 10 次,再收集 15 个 CUDA-event 样本. 非默认
winner 必须再通过 9 轮 x 31 次 alternating A-B/B-A paired gate,至少赢 7/9 轮且
总中位数差距不小于 3%. TileLang/Triton JIT,首次 kernel 调用,输入和静态权重分配
全部排除在 steady-state window 外. 最后用 seed `601,709,811,919,1021` 对候选和
固定旧默认重复同一 paired gate.

### Sweep 与 resolver 结果

| shape | candidate (ms) | old default (ms) | stored gap | round wins | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `M1,N4096,K4096,no-bias` | `0.019999` | `0.035778` | `78.90%` | `9:0` | promote exact signature |
| `M4,N4096,K4096,bias` | `0.021244` | `0.035952` | `69.23%` | `9:0` | promote exact signature |
| `M1,N11008,K4096,bias+SiLU` | old default | old default | n/a | n/a | keep default |
| `M4,N11008,K4096,bias` | old default | old default | n/a | n/a | keep default |

晋级后的生产 `tilelang_resolved` 与显式候选在 M1/M4 上只差 `0.99%/0.67%`,均
低于 3% 而归为 noise-equivalent. N=11008 的生产 resolver 与旧默认只差
`1.00%/0.40%`,同样 noise-equivalent,证明 exact route 没有泄漏到负对照.
所有 candidate 和 baseline correctness 通过;promoted shape 最大绝对误差为
`0.125`.

5-seed audit 的每个 shape/seed 都由新候选 `9:0` 胜出:

| shape | candidate median range (ms) | default median range (ms) | stored gap range |
| --- | ---: | ---: | ---: |
| `M1,N4096,K4096,no-bias` | `0.01730-0.01791` | `0.03711-0.03829` | `107.19-121.37%` |
| `M4,N4096,K4096,bias` | `0.01820-0.01873` | `0.03684-0.03859` | `99.78-109.59%` |

因此 audit 决策为 `promote_explicit_tilelang`.

### Nsight Systems 归因边界

Nsight Systems 对三个 workload 各自在 NVTX 外 warmup 10 次,范围内测量 20 次.
每个 selected range 都只有 20 个目标 kernel instance,即每个 Linear 调用一次
launch:

| shape | TileLang default | TileLang candidate | Triton resolved |
| --- | ---: | ---: | ---: |
| `M1,N4096,K4096,no-bias` | `34.961 us` | `15.094 us` | `13.759 us` |
| `M4,N4096,K4096,bias` | `35.064 us` | `15.606 us` | `14.785 us` |

三个路径的 launch 数相同,而 TileLang 只改变 tile schedule 后 kernel median 明显
下降. 这支持 shape-sensitive schedule mismatch,不支持把收益归因于 launch fusion.
M1/M4 candidate 与 Triton 的 paired gap 只有 `0.99%/0.73%`,均
noise-equivalent;M4 的 Torch baseline 还稳定快 `7.90%`. 因此 direct TileLang
调度晋级不能推导出 Triton 或全局 engine route 变更.

六份 `cuda_gpu_mem_time_sum` CSV 都是空文件,trace 没有可报告的独立 GPU memory
数据. NCU preflight 返回 `counter_permission_denied`. 本轮不推断 DRAM/L2
bandwidth,occupancy,warp stall,roofline,register pressure 或 tensor-pipe
utilization. 管理员启用 NVIDIA performance-counter policy 后,才能继续这一层
诊断.

### 实现与适用边界

`resolve_tilelang_linear_schedule()` 新增 `out_features` 和 `activation` 参数.
`dense_linear_epilogue_tilelang()` 在完成 shape/activation 校验后显式传入
`weight.shape[0]` 与 activation. 新 preset 只在以下条件同时成立时启用:

1. `target_arch == "sm_89"`.
2. `x.dtype == torch.float16` 且 `x.ndim == 2`.
3. `M<=4,K=4096,out_features=4096`.
4. `activation is None`.

任一信息缺失或条件不满足都保留 `64x64x64`. Bias 不进入 key,因为受测 M1
no-bias 与 M4 bias 都通过. 显式 `block_m/block_n/block_k` 仍逐项优先. 这次
改动不触碰 BF16 preset,Triton resolver,`gemm_with_precision(engine="auto")`,
`linear_runtime="auto"` 或 wrapper materialization policy.

当前性能承诺只覆盖真实测量的 RTX 4070 Ti SUPER `sm_89`. 其他 SM,M,N,K,
activation 或 layout 必须重新完成 correctness,paired CUDA-event gate 和 profiler
证据,不能从同一 tile 形状外推.

### 未采纳方案与可复用规则

- 没有把 FP16 条件写成 BF16 式的 `M<=4`;N=11008 负对照明确拒绝该泛化.
- 没有因为 TileLang 候选追平 Triton kernel 就改变 `auto`;direct engine kernel
  gate 与完整 Linear route 是不同层.
- `out_features` 和 activation 必须进入 resolver key. 仅从 activation tensor
  读取 M/K 无法区分本轮赢家和负对照.
- CUDA-event 负责 promotion,NSYS 负责 launch/kernel time 分解. 空 memory report
  和被拒绝的 NCU counter 不是 memory/occupancy 诊断.
- 小 shape 调度变化也必须做多 seed paired audit;一次顺序 sweep 不足以修改默认.

### 验证落点

- [TileLang Linear resolver](../../../xqt/operator_opt/kernels/tilelang/linear.py)
- [resolver and CUDA contract tests](../../../tests/xqt/test_tilelang_half_ops_cuda.py)
- [operator integration tests](../../../tests/xqt/test_operator_tilelang_linear.py)
- [precision dispatcher tests](../../../tests/operator_opt/test_gemm_precision.py)
- [benchmark and profiling entry](../../../research/xqt-gemm/bench_sm89_tilelang_linear_fp16.py)
- [evidence artifact](../../../research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16/)

## R-034: SM89 TileLang FP16 MLP decode schedule rejection

### 目标

| 项 | 值 |
| --- | --- |
| backend | TileLang direct `dense_linear_epilogue` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`),CUDA 13.0,torch 2.12.1+cu130 |
| precision | FP16 input/weight/bias/output,FP32 GEMM accumulation |
| down shapes | `M=1/4,N=4096,K=11008,activation=None` |
| up/gate shapes | `M=1/4,N=11008,K=4096,activation=None` |
| candidate space | R-033 的 10 个 TileLang schedule |
| fixed default | `64x64x64,128 threads,2 stages` |

R-033 只证明 `K=N=4096` 的 tiny-M exact preset. 本轮补测常见 MLP
`4096 <-> 11008` 两个方向,目标是判断显式 TileLang resolver 能否安全扩大,
不是为 synthetic shape 改变 Linear `auto` engine.

### 基线与测量

正式入口为
`research/xqt-gemm/bench_sm89_tilelang_linear_fp16_mlp.py`,复用 R-033 的同一
measurement implementation. 每个 candidate warmup 10 次,收集 15 个
CUDA-event 样本. 非默认 sequential winner 必须通过 9 轮 x 31 次 alternating
A-B/B-A gate,至少赢 7/9 轮且 aggregate gap 不小于 3%. 最后对 seed
`601,709,811,919,1021` 重复同一 gate;任一 shape/seed 失败都不能改 resolver.

数值 reference 在 FP32 中执行 Linear,输出处单次 cast FP16. 正式容差为
`atol=0.25,rtol=0.03`. TileLang/Triton JIT,首次调用,input allocation 与
static weight allocation 都排除在 steady-state window 外.

### 主 sweep

| shape | candidate | candidate (ms) | default (ms) | paired gap | round wins | decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `M1,N4096,K11008,no-bias` | `64x64x64/128t/3s` | `0.154455` | `0.165183` | `4.09%` | `9:0` | audit |
| `M4,N4096,K11008,bias` | `64x64x64/128t/3s` | `0.154650` | `0.166273` | `3.99%` | `9:0` | audit |
| `M1,N11008,K4096,no-bias` | 1-stage sequential winner | n/a | `0.158092` | `0.69%` | noise | keep default |
| `M4,N11008,K4096,bias` | 1-stage sequential winner | n/a | `0.159526` | `1.32%` | noise | keep default |

所有 candidate 与 baseline correctness 通过. Audited down shape 的最大绝对
误差为 `0.25`. Down M1/M4 上,Torch 仍分别稳定快于 3-stage candidate
`3.65%/3.75%`,均 `9:0`. 因此主 sweep 即使通过 schedule review,也不支持
TileLang 成为这两个 shape 的全局 engine auto route.

### 5-seed 审计与拒绝原因

3-stage candidate 在每个 shape/seed 的 aggregate median 都更快,但严格 gate
只通过 8/10 records:

- Down M1 seed 811 的 aggregate gap 为 `3.45%`,但 round wins 只有 `6:3`,未过
  7/9 门槛.
- Down M4 seed 919 的 round wins 为 `8:1`,但 aggregate gap 只有 `2.66%`,未过
  3% 门槛.
- 其他 8 个 records 的 gap 为 `3.09-9.25%`,并通过各自 round/gap gate.

最终 audit decision 为 `keep_default`. 这不是"接近通过后人工晋级";promotion
contract 要求所有受测 seed/shape 通过. 因此生产
`resolve_tilelang_linear_schedule()` 不增加 MLP preset,capability metadata 与
Linear/Triton `auto` route 也不变.

### Profiler 归因边界

`torch.profiler` 对每个 workload 测量 20 call,每次只有一个目标 GEMM kernel.
Nsight Systems 在 NVTX 外 warmup 10 次,每个 selected range 的 CUDA API summary
记录 20 次 launch:

| shape | TileLang default | candidate/resolved | Triton resolved |
| --- | ---: | ---: | ---: |
| `M1,N4096,K11008,no-bias` | `144.463 us` | `138.596 us` | `299.063 us` |
| `M4,N4096,K11008,bias` | `142.336 us` | `139.110 us` | `292.007 us` |
| `M1,N11008,K4096,no-bias` | `138.552 us` | `138.494 us` | `266.676 us` |
| `M4,N11008,K4096,bias` | `138.493 us` | `138.142 us` | `278.043 us` |

Down projection 的差异位于同一单-launch kernel body,不是 launch fusion. M4
NSYS kernel gap 已低于 3%,与独立审计处于门槛附近的结论一致. 两份 TileLang
up M4 kernel summary 把 21 个 GPU instance 归入范围,但对应 CUDA API summary
只有 20 次 launch;artifact 将其保留为 NVTX attribution anomaly,不解释成额外
host launch.

12 份 `cuda_gpu_mem_time_sum` CSV 均为空. NCU preflight 为
`counter_permission_denied`. 本轮不推断 DRAM/L2 bandwidth,occupancy,warp
stall,roofline,register pressure 或 tensor-pipe utilization.

### 未采纳方案与可复用规则

- 不因主 sweep 两个 `9:0` 就跳过独立 audit. 接近 3% 的 knob 必须接受 seed
  和 round 波动检查.
- 不把"所有 aggregate median 都更快"替代既定 promotion contract. Round wins
  与 minimum gap 是两个独立门槛.
- 不把 `num_stages=3` 泛化到所有 long-K. 本轮只测 `M<=4,N=4096,K=11008`,
  且该 exact contract 本身也被拒绝.
- 不根据 Triton 在本 shape 较慢改变其他已验证 Triton exact preset;backend
  schedule evidence 只能约束同一 signature.
- NSYS 用于解释单 launch kernel time,不替代 CUDA-event gate. 空 memory report
  与被拒绝的 NCU counter 不能支撑 memory 或 occupancy 诊断.

### 验证落点

- [benchmark and profiling entry](../../../research/xqt-gemm/bench_sm89_tilelang_linear_fp16_mlp.py)
- [main sweep](../../../research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16-mlp/result.json)
- [independent audit](../../../research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16-mlp/independent-audit.json)
- [Nsight Systems summary](../../../research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16-mlp/nsys-summary.json)
- [full conclusion](../../../research/xqt-gemm/artifacts/2026-08-10-sm89-tilelang-linear-fp16-mlp/conclusion.md)

## R-035: P3 boundary gate for dynamic shapes, masks, cross-SM fallback and replay safety

### 目标与范围

本 checkpoint 收口 P3 中仍未完成的 runtime/kernel 边界,不扩大任何默认
promotion. 覆盖 TileLang MHA/Linear, Triton attention/Linear wrapper,
`KvScaleAttention` offline phase 和 CUDA Graph shared-state replay. XQT 仍只
负责模型侧 entity,不实现 serving scheduler 或 KV/cache storage.

### 实现变化

- CUDA Graph capture/replay 通过进程内 RLock 串行访问静态输入和 graph-owned
  output. `_TritonLinearWrapper` 和两个 attention wrapper 统一走该 helper.
- Triton 和 TileLang 低层入口以及 wrapper 对显式 `target_arch` 做 runtime SM
  mismatch 检查. 不匹配时 wrapper 使用 reference fallback,低层 kernel 在编译前
  明确拒绝.
- MHA wrapper 的 `attn_mask`/`key_padding_mask`/self-attention `None` 参数和
  autograd 输入进入完整 PyTorch reference contract. Inference-only kernel 仍
  保留 backward rejection.
- `KvScaleAttention.prefill()`/`decode()` 记录 phase,report 明确
  `serving_cache.implemented=false`.

### 测量与结果

固定入口为
`research/xqt-gemm/bench_p3_boundary_gate.py`,seed 为 `20260812`. 当前环境是
NVIDIA GeForce RTX 4070 Ti SUPER,`sm_89`,CUDA 13.0,torch 2.12.1+cu130.
`result.json` 中的 correctness checks 全部通过:

- MHA mask 和 per-head weights 与直接 PyTorch MHA 的 max abs 为 `0`;
  autograd reference fallback 可回传 input gradient.
- 动态 `seq=3/7` 和 non-contiguous input 的 model-level attention 通过.
- Triton attention 动态 shape correctness 通过;Triton Linear Graph 两个 shape
  (`M=1/2`) 各自 capture,replay cache size 为 `2`.
- TileLang FP16 Linear 的 `M/N/K=(1,64,64),(4,96,64),(16,128,128)` 以及
  `None/SiLU/GELU` epilogue 通过 reference gate,max abs 为
  `0.0078125/0.015625/0.03125`.
- 显式 other-SM target 在 wrapper 和 TileLang/Triton low-level entry 均在编译
  前 fallback 或拒绝. Resolver 对 `sm_80`/`sm_90` 保持 `default`.
- shared replay fake-graph concurrency gate 的 `12` 次调用最大并发为 `1`;
  这是共享可变 graph state 的正确性证据,不是 GPU kernel 性能结论.

单次当前设备描述性 benchmark 中,`sm_89` FP16 Triton Linear `M=1,K=64,N=96`
wrapper 10-call mean 为 `0.038976 ms`,仅用于记录可执行性,不作为 promotion
门槛. 其他 SM 没有可用硬件,artifact 将 performance status 记为
`not_available`,不从 `sm_89` 外推.

### 适用边界与未采纳方案

- 只在显式 target 与实际 device SM 一致时进入对应 kernel/graph fastpath;
  mismatch 必须显式 fallback 或 error,不静默编译错误架构.
- mask,backward 和需要完整 MHA contract 的调用不进入 TileLang forward-only
  kernel. Dynamic shape 通过 shape/layout key 建立独立 graph entry.
- `KvScaleAttention` 只提供 offline `prefill`/`decode` 和 model-side metadata;
  serving 级 cache 生命周期留给外部 runtime.
- 没有把当前 `sm_89` timing 扩展成跨 SM promotion,也没有因 boundary gate 改变
  `auto` 默认路由. NCU/NSYS 本轮不采集新 counter;既有 NCU 环境为
  `counter_permission_denied`,不写 counter-derived 归因.

### 验证落点

- [boundary gate entry](../../../research/xqt-gemm/bench_p3_boundary_gate.py)
- [boundary artifact](../../../research/xqt-gemm/artifacts/2026-08-12-p3-boundary-gate/result.json)
- [boundary correctness metadata](../../../research/xqt-gemm/artifacts/2026-08-12-p3-boundary-gate/correctness.json)
- [boundary benchmark metadata](../../../research/xqt-gemm/artifacts/2026-08-12-p3-boundary-gate/benchmark.json)
- [boundary reproduction](../../../research/xqt-gemm/artifacts/2026-08-12-p3-boundary-gate/reproduction.md)
- [regression tests](../../../tests/xqt/test_p3_boundary_contracts.py)
- [CUDA Graph runtime helper](../../../xqt/operator_opt/runtime.py)
- [attention wrapper](../../../xqt/operator_opt/wrappers/attention.py)
- [KV model-side entity](../../../xqt/runtime/modules/kv_attention.py)

## R-036: SM89 SVDQuant small-BLOCK_N GEMM, 融合单调用入口与 CUDA Graph 热路径

### 目标与范围

收口 `svdquant 推理优化追平/超越 nunchaku 整体加速` 冲刺: 在 R-031 已追平
direct floor 的基础上, 攻击本机复跑暴露的两处残差: 小 M (M<=256) 时
`GEMMConfig_W4A4` 固定 `BLOCK_N=128` 导致 GEMM grid 只有 8 个 CTA, 以及
`SVDQuantLinear` wrapper 每次 forward 的 Python host 开销 (M64 慢 bound
floor 24%, M256 慢 21%). 范围限 sm_89 W4A4 native 路径与
`SVDQuantLinear` 热路径, 不新增训练循环或任务 registry.

### 实现变化

- 新增 BLOCK_N=64 W4A4 GEMM 变体 (`xqt/operator_opt/kernels/cute/
  svdq_w4a4_sm89_smalln_kernel.cu`): 与 upstream config 仅
  `BLOCK_N/WARP_N=64` 之差, 其余几何 (BLOCK_M=256, NUM_WARPS=8, K 侧) 不变,
  因此 packed activation 与 LoRA 布局与基座扩展共享, 只需按 WARP_N=64 重
  pack qweight/weight_scales/packed_bias. binding 复刻基座的
  `quantize_act_lora` (含 `cudaMemsetAsync`) 并新增 C++ 单调用融合入口
  `svdq_linear` 与 `bind_svdq_linear` (`BoundSVDQSmallNLinear`). Python 侧
  提供 `pack_svdq_w4a4_linear_smalln`, `svdq_w4a4_linear_smalln`,
  `bind_svdq_w4a4_linear_smalln`, `native_w4a4_smalln_available` 与临时
  启发式 `smalln_w4a4_beneficial` (`padded_rows<=256 且 padded_n>=512`).
- `SVDQuantLinear` 热路径按 shape 自动选择 BN64/BN128: smalln pack 独立缓存
  (与 BN128 pack 并存), workspace 布局变体无关故共享; hot cache 条目记录
  backend 字符串, metadata 新增 `native_w4a4_dynamic_smalln` 实现名.
- 热路径加可选 CUDA Graph: `enable_fusion(cuda_graph=True)` 时首个 shape
  forward 后 capture bound runner (含一次 replay 校验, 容差 1e-3, 失败静默
  回退 bound runner), 稳态走 copy+replay. replay 每次返回同一静态输出
  tensor, 调用方需在下一次 forward 前消费或 clone (docstring 与 metadata
  `cuda_graph_used` 均有记录).
- `_native_w4a4_state_signature` 改走 `_modules/_parameters/_buffers` 直接
  字典访问 (绕开 `nn.Module.__getattr__`), host 开销 4.38us -> 1.75us,
  变更检测语义不变 (id + `_version`, 原地修改与整体替换均可捕获).

### 测量与结果

环境: NVIDIA vGPU-32GB, `sm_89`, CUDA 13.0, torch 2.12.1+cu130. 全部
CUDA event median, warmup 30, 15 轮 x 500 次, 交替候选顺序.

- GEMM-only 隔离 (量化共享, 只比 `gemm_lora`): M64/M256 (`K=N1024,R32`)
  BN64 对 BN128 `1.63x` (FP16 与 BF16 一致); M1024 (`K=N2048,R64`) `0.92x`
  (预期回退, 由 `smalln_w4a4_beneficial` 门控排除).
- bound runner 级 (热路径真实配置): M64 `1.161x` (FP16) / `1.146x` (BF16),
  M256 `1.221x` (FP16) / `1.236x` (BF16); 数值与 BN128 floor 的 max_abs
  0-0.002, relative RMSE <= 1.2e-5. 首轮实现用 Python 组合基座
  quantize + smalln gemm, op 级反而 0.79x: 每次调用 `torch.empty`
  (~4.9us) + 两次 pybind 的 host 开销吃掉了 GEMM 收益, 是融合单调用入口的
  直接动机.
- wrapper 级 (`SVDQuantLinear.forward`): eager smalln M64 `24.85us` /
  M256 `23.62us`, 对 direct fused floor (~`24.7us`) 已进入噪声带; op 级
  CUDA Graph 收益中性 (M64 1.03x, M256 0.99x), 因为 hot key, 失效守卫,
  `copy_` 与 `graph.replay()` 的 Python 开销仍在每次调用路径上.
- block 级 e2e (DiT-realistic: 4 个 `3072<->12288` linear + GELU):
  SVDQ wrapper eager 对 FP16 eager M256 `2.458x`, M1024 `3.109x`, M4096
  `3.252x`; 整 block 单次 CUDA Graph capture 后 M256 叠加到 `2.519x`
  (对 SVDQ eager 自身 `1.025x`, M1024+ graph 无额外收益, host 开销已被
  每 op ~87us 的 GPU 工作掩盖). M256 时 4 层全部命中
  `native_w4a4_dynamic_smalln`. 数值: SVDQ vs FP16 relative RMSE ~2.9%
  (合成随机权重下 W4A4+rank32 的预期量化误差), graph replay 与 eager
  逐元素一致 (<= 3e-6).
- BF16 覆盖: smalln GEMM/bound/pack 在 BF16 下复现 FP16 同级收益与数值,
  `dispatch_scalar` 两 dtype 均有证据. W8A8 FP16 模板化仍延后 (价值低).

### 适用边界与未采纳方案

- smalln 只在 `padded_rows<=256 且 padded_n>=512` 自动启用, M1024 实测
  0.95x 回退已被门控排除; 该启发式是 provisional, 变更需以本机 CUDA event
  证据为准. 代价是 smalln pack 与 BN128 pack 并存的额外显存 (~半份权重).
- op 级 CUDA Graph 默认关闭, 仅 `enable_fusion(cuda_graph=True)` 启用;
  输出 aliasing 语义见 docstring. 整 block/model capture 时应保持 per-op
  graph 关闭 (默认), 由外层 `capture_cuda_graph_with_static_state` 统一
  capture, 避免嵌套 capture.
- 未采纳: 维持 Python 两段式 smalln 组合 (op 级 0.79x 实证失败); op 级
  graph 设为默认 (收益中性且改变输出语义); C1 memset 消除 (block 级
  profile 未显示 memset 节点占比显著); W8A8 FP16 kernel 模板化 (延后).
- NCU 本机无 counter 权限, 不记录 cache/occupancy 归因, 所有性能结论仅以
  本机 CUDA event 为准, 不外推其他 SM.

### 验证落点

- [smalln kernel](../../../xqt/operator_opt/kernels/cute/svdq_w4a4_sm89_smalln_kernel.cu)
- [smalln binding](../../../xqt/operator_opt/kernels/cute/svdq_w4a4_sm89_smalln_binding.cpp)
- [Python 接线与启发式](../../../xqt/operator_opt/kernels/cute/svdq_w4a4_sm89.py)
- [wrapper 热路径](../../../xqt/runtime/modules/svd_composite.py)
- [smalln bench](../../../research/xqt-gemm/bench_sm89_svdq_w4a4_smalln.py) 与 [artifact](../../../research/xqt-gemm/artifacts/2026-08-12-sm89-svdq-w4a4-smalln/result.json)
- [wrapper graph bench](../../../research/xqt-gemm/bench_sm89_svdq_w4a4_wrapper_graph.py) 与 [artifact](../../../research/xqt-gemm/artifacts/2026-08-12-sm89-svdq-w4a4-wrapper-graph/result.json)
- [block e2e bench](../../../research/xqt-gemm/bench_sm89_svdq_block_e2e.py) 与 [artifact](../../../research/xqt-gemm/artifacts/2026-08-12-sm89-svdq-block-e2e/result.json)
- 回归: `tests/xqt/runtime/test_svd_fusion.py` 与
  `tests/xqt/runtime/test_composite_runtime_caches.py` 全绿 (33 项).
