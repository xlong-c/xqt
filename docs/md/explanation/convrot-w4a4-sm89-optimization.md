# ConvRot W4A4 SM89 CUDA 优化详解

本文以 XQT 的 `ConvRotMixedPrecisionLinear` 为例,逐步解释 `sm_89` 上 ConvRot W4A4 专用 CUDA 路径是怎样从显式 rotation + quant + GEMM,收敛为稳定的两 kernel 热路径的. 重点不是背诵某个 kernel,而是理解每一步消除了什么成本,改变了什么数据契约,以及怎样用公平 benchmark 和 profiler 证据证明结果.

通用 kernel 调优方法见 [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md). XQT 量化术语和 artifact/runtime 边界见 [xqt-quant.md](xqt-quant.md). 本文对应的简明优化记录是 [operator-optimization-records.md 的 R-032](operator-optimization-records.md#r-032-sm89-convrot-w4a4-warp-fht-rowwise-int4-fusion).

## 负责什么

- 解释 ConvRot 的数学等价关系和 W4A4 rowwise 计算契约.
- 按实现顺序拆解 warp-FHT,动态 INT4 量化,nibble packing,CUTLASS GEMM,epilogue fusion 和 runtime cache.
- 说明为什么最终保留两个 CUDA kernel,以及为什么当前没有融合 norm.
- 给出与官方 `comfy-kitchen 0.2.28`,Nunchaku 和显式 split 路径的公平性能对照.
- 说明数值误差来自哪里,以及 rowwise scale 和 grouped scale 为什么不能静默互换.
- 给出源码阅读顺序,复跑命令和下一步可验证的优化方向.

## 不负责什么

- 不把 RTX 4070 Ti SUPER 的结果外推到其他 GPU 或完整模型.
- 不把 W4A4 runtime fastpath 写成量化 artifact 无条件更优.
- 不声称当前实现融合了 RMSNorm 或 LayerNorm.
- 不在 NCU counter 不可用时猜测 occupancy,cache hit rate,warp stall,register pressure 或 roofline 位置.
- 不提供逐提交的历史回放. 本文按最终实现的依赖关系重建优化步骤;没有保存独立 benchmark artifact 的中间步骤,不会虚构逐步加速百分比.

---

## 1. 先看最终结论

最终热路径固定为两个 CUDA kernel:

```text
FP16/BF16 source activation [M,K]
  -> kernel 1:
       256-point regular-Hadamard FHT
       + per-row absmax
       + dynamic signed INT4 quantization
       + row-major nibble packing
  -> packed activation [M,K/2] bytes + FP32 activation scale [M]
  -> kernel 2:
       CUTLASS s4 x s4 -> s32 Tensor Core GEMM
       + FP32 activation-scale multiply
       + FP32 weight-scale multiply
       + bias add
       + FP16/BF16 output store
  -> output [M,N]
```

这里最重要的四个结论是:

1. rotation 和 activation quantization 已融合. 不会物化 rotated FP16/BF16 activation.
2. dequantization 和 bias 已融合进 GEMM epilogue. 不会再启动独立 scale 或 bias kernel.
3. norm 没有融合. `execution_metadata()["norm_fused"]` 固定为 `False`.
4. rowwise native path 会改变 grouped artifact 的 weight scale contract. 因此它是显式性能/精度取舍,不是对所有 ConvRot W4A4 artifact 的透明替换.

在 RTX 4070 Ti SUPER `sm_89` 上,12 个 FP16/BF16 case 的完整 XQT wrapper 实测为:

| 对照 | XQT 加速比范围 | 中位数 |
| --- | ---: | ---: |
| `comfy-kitchen 0.2.28` 官方 CUDA wheel | `2.360x-3.056x` | `2.910x` |
| Nunchaku bound W4A4 | `1.175x-1.586x` | `1.292x` |
| 显式 rotation + Nunchaku | `1.875x-2.512x` | `2.215x` |

XQT 完整 wrapper 延迟为 `13.297-26.766 us`. `xqt_wrapper / rowwise_bound_floor` 为 `0.995x-1.083x`,中位 `1.034x`. 小于 `1.0` 的个别值是微秒级测量噪声,不能解释成 Python wrapper 物理上快于其 native floor;整体结论是 wrapper 已基本贴住 bound runner.

---

## 2. 源码阅读地图

如果第一次读这类优化,可以分四遍看:

1. 第一遍读第 1,3,4,5 节,先建立数学和公平比较口径.
2. 第二遍读第 7-12 节,沿着 activation 从 FP16/BF16 到 packed INT4,再到 CUTLASS output.
3. 第三遍读第 14-17 节,理解为什么 kernel 快不等于完整 module 快.
4. 第四遍读第 18-22 节,学习怎样用 benchmark,profiler 和数值 reference 验收.

建议按以下顺序阅读:

1. [convrot_4bit.py](../../../xqt/compression/quant/quantizers/convrot_4bit.py): 数学 reference,artifact,backend policy,fallback 和 Python cache.
2. [convrot_w4a4_rowwise_sm89.py](../../../xqt/kernels/ops/_impl/cute/convrot_w4a4_rowwise_sm89.py): capability,JIT build,rowwise weight packing 和 Python binding.
3. [convrot_w4a4_rowwise_sm89_binding.cpp](../../../xqt/kernels/jit/csrc/quantization/convrot_w4a4_rowwise_sm89_binding.cpp): 参数验证,dynamic runner,workspace cache 和 tensor version 失效.
4. [convrot_w4a4_rowwise_sm89_kernel.cu](../../../xqt/kernels/jit/csrc/quantization/convrot_w4a4_rowwise_sm89_kernel.cu): warp-FHT,quant/pack,CUTLASS EVT epilogue 和 GEMM dispatch.
5. [test_convrot_4bit_quantizer.py](../../../tests/xqt/quant/test_convrot_4bit_quantizer.py): non-contiguous,高维,multi-stream,mutation 和数值 reference 验证.
6. [benchmark_convrot_w4a4_sm89.py](../../../tools/benchmark_convrot_w4a4_sm89.py): 正式 CUDA-event benchmark 口径.
7. [profile_convrot_w4a4_sm89.py](../../../tools/profile_convrot_w4a4_sm89.py): NVTX range 和 Nsight Systems workload.

两个重要的上游参考是:

- `comfy-kitchen` commit `b72e6dfa79b79a7aee33a9c7608b5d9b3005b7af` 的 `convrot_w4a4.cu`: rowwise signed INT4 contract,CUTLASS EVT epilogue 和官方 wheel baseline.
- `ComfyUI-AnimaTurbo` 的 `warp_fht/convrot_warp_quantize.cu`: warp 内 256 点 FHT 的 register mapping.

XQT CUDA 文件保留 Apache-2.0 来源声明. 当前实现不是直接照搬一条宽泛的官方 dispatch,而是把上游关键结构缩减为 XQT 的 `sm_89 + FP16/BF16 + rot_size=256 + rowwise W4A4` 专用 contract.

---

## 3. 第一步先理解数学等价关系

### 3.1 记号

设:

- 输入 `X` 的形状为 `[M,K]`.
- Linear 权重 `W` 的形状为 `[N,K]`.
- bias `b` 的形状为 `[N]`.
- 每 256 个 feature 使用一个归一化 regular-Hadamard 矩阵 `R_256`.
- 完整 `K` 维旋转 `R` 是多个 `R_256` 的 block diagonal 组合.

PyTorch `F.linear(X,W,b)` 的数学形式是:

```text
Y = X W^T + b
```

ConvRot 同时旋转 activation 和 weight:

```text
X_rot = X R
W_rot = W R
```

因为归一化 Hadamard 矩阵是正交矩阵:

```text
R R^T = I
```

所以:

```text
X_rot W_rot^T
= (X R) (W R)^T
= X R R^T W^T
= X W^T
```

这就是为什么 weight 可以离线旋转,activation 在每次 forward 在线旋转,而未量化时仍保持原始 Linear 的数学结果.

### 3.2 regular-Hadamard 的构造

源码的 4 阶基矩阵是:

```text
H4 = [ 1  1  1 -1
       1  1 -1  1
       1 -1  1  1
      -1  1  1  1 ]
```

高阶矩阵通过 Kronecker product 构造. 对 `rot_size=256=4^4`:

```text
H256 = kron(kron(kron(H4,H4),H4),H4)
R256 = H256 / sqrt(256)
     = H256 / 16
```

CUDA FHT 并不显式存储 `256 x 256` 矩阵. 它把四次 4 点 Hadamard butterfly 展开到 warp shuffle 和 register 运算中. 每一级乘 `0.5`,四级总缩放正好是:

```text
0.5^4 = 1/16 = 1/sqrt(256)
```

### 3.3 W4A4 rowwise 计算公式

对旋转后的 activation row `m`:

```text
a_scale[m] = max(
    min(max_k(abs(X_rot[m,k])), finite_max(input_dtype)) / 7,
    1e-10
)

Q_a[m,k] = clamp(round(X_rot[m,k] / a_scale[m]), -7, 7)
```

对离线旋转后的 weight row `n`:

```text
w_scale[n] = max(max_k(abs(W_rot[n,k])) / 7, 1e-10)

Q_w[n,k] = clamp(round(W_rot[n,k] / w_scale[n]), -7, 7)
```

INT4 GEMM 先得到 INT32 accumulator:

```text
Acc[m,n] = sum_k(Q_a[m,k] * Q_w[n,k])
```

最终 epilogue 是:

```text
Y[m,n] = cast_output(
    float(Acc[m,n]) * a_scale[m] * w_scale[n] + bias[n]
)
```

这组公式是理解所有 layout,epilogue 和数值测试的主线.

---

## 4. 初始路径为什么慢

最直观的 reference/split 实现通常是:

```text
input
  -> pad/reshape
  -> torch.matmul(input_group, R256)
  -> materialized rotated FP16/BF16 tensor
  -> abs/amax/div/round/clamp/cast/pack
  -> packed INT4 activation
  -> W4A4 GEMM
  -> scale/bias/output
```

这个写法数学清楚,但热路径有四类成本.

### 4.1 rotated tensor 的全局内存往返

显式 rotation 必须写出 `[M,K]` 的 FP16/BF16 tensor,后续 quant kernel 再读一遍. 仅这一个中间 tensor 就增加:

```text
write bytes = M * K * 2
read bytes  = M * K * 2
```

以 BF16 `M=256,K=2048` 为例,理论上至少多出:

```text
256 * 2048 * 2 bytes * 2 directions = 2 MiB
```

这是被删除的逻辑 device traffic,不是 NCU 实测 DRAM bytes. 当前环境无权读取 performance counters,因此不能继续推断实际 L2/DRAM 命中情况.

### 4.2 rotation 被当成通用 GEMM

`R256` 是固定的 `+1/-1` 结构,通用 BF16 GEMM 仍要按矩阵乘法装载和调度. FHT 可以用加减和固定缩放完成同一变换,无需加载完整旋转矩阵.

### 4.3 动态量化容易拆成多个 eager kernel

如果用 PyTorch eager 表达 `abs -> amax -> divide -> round -> clamp -> cast -> pack`,会产生多个 launch 和临时 tensor. 即使编译器偶尔融合,也不能把偶然生成的图当作稳定 backend contract.

### 4.4 Python wrapper 和 workspace 分配会污染微秒级算子

当核心 kernel 只有十几微秒时,以下操作都可能变得可见:

- 每次 forward 重新 pack weight.
- 每次 forward 分配 activation workspace.
- Python 逐层调用 rotation,quant 和 GEMM binding.
- host 端反复解析 shape,dtype,device 和 backend.

因此优化不能只写一个更快的 kernel,还必须把它接成稳定的 operator hot path.

---

## 5. 优化步骤 0: 先固定公平 contract

在改 kernel 前,必须先回答"比较的到底是不是同一件事". 本轮固定了以下 contract:

| 项 | 固定值 |
| --- | --- |
| GPU | RTX 4070 Ti SUPER,`sm_89` |
| activation dtype | FP16 或 BF16 |
| rotation | regular-Hadamard,`rot_size=256` |
| activation scale | dynamic,每个 input row 一个 FP32 scale |
| weight | rowwise signed INT4,每个 output row 一个 FP32 scale |
| rounding | deterministic round-to-nearest,不启用 stochastic rounding |
| bias | FP32,在 GEMM epilogue 加入 |
| output | 与 input 相同的 FP16/BF16 |
| norm | 不包含,`norm_fused=false` |

与官方 wheel 比较时,XQT 和 `comfy-kitchen 0.2.28` 收到完全相同的:

- packed signed INT4 weight.
- rowwise weight scale.
- bias.

因此官方对照是在同一个 weight runtime contract 上比较 operator 实现,不是拿 grouped quant 和 rowwise quant 混在一起.

正式 benchmark 排除以下一次性成本:

- JIT extension 编译.
- grouped artifact 解量化和 rowwise weight 重新量化.
- workspace 首次分配.
- CUTLASS tile 首次 autotune.

这些成本在 warmup 前完成或被 30 次 warmup 吸收. 这衡量的是稳态推理,不是冷启动.

---

## 6. 优化步骤 1: 先收窄 capability,再做专用 kernel

当前 rowwise fastpath 只接受:

- CUDA `sm_89`.
- FP16 或 BF16 input/output.
- `activation_scale_mode="dynamic"`.
- `rot_size=256`.
- `K=1024`,或 `K%2048==0`.
- `1024<=K<=32768`.
- `N%8==0`.
- input 尚未旋转.
- `input_features == padded_input_features`.
- 无 channel-hybrid.
- eager runtime cache 可用,即不处于禁止 mutation 的 compile/trace 上下文.

这个 gate 不是功能缺陷的掩饰,而是性能实现的边界. 它允许 CUDA 内核依赖以下不变量:

- FHT 永远是 256 点,循环可以完全 unroll.
- 每个 warp 处理的 256 点 group 数量只有两种编译期形式.
- A/B 都满足 signed INT4 Tensor Core 的对齐要求.
- 不需要在同一个 kernel 里支持 static scale,任意 rotation size,padding 或 mixed channel policy.

不满足条件时,模块记录明确的 fallback reason,再尝试 Nunchaku native path,最后回到 PyTorch reference. fastpath 不会用错误 layout 或错误 scale contract"勉强执行".

---

## 7. 优化步骤 2: 把 256 点 FHT 全放进 warp register

### 7.1 lane 和 slot 怎样覆盖 256 个元素

一个 256 点 group 中的元素位置写成:

```text
e = lane + 32 * slot
lane in [0,31]
slot in [0,7]
```

因此一个 warp 的 32 个 lane,每个 lane 持有 8 个 slot,正好覆盖:

```text
32 * 8 = 256 elements
```

XQT 进一步用 `half2` 或 `bf162` 把两个不同 256 点 group 的同一位置装进一个 paired register:

```text
values[pair][slot].low  = group(2*pair)[lane + 32*slot]
values[pair][slot].high = group(2*pair+1)[lane + 32*slot]
```

两种 kernel 形态是:

| K 范围 | `PairsPerWarp` | 每 warp group 数 | 每 row warp 数 |
| --- | ---: | ---: | ---: |
| `K=1024` | 2 | 4 | 1 |
| `K%2048==0` | 4 | 8 | `K/2048` |

例如 `K=2048` 时,一个 warp 用 `4 x 8` 个 paired register 逻辑上持有 `4 x 8 x 2=64` 个元素/lane,全 warp 正好持有 2048 个元素.

launcher 还会按 `warps_per_row` 选择一个 block 放几行:

| K | warps/row | rows/block | threads/block | shared max buffer |
| --- | ---: | ---: | ---: | --- |
| 1024 | 1 | 1 | 32 | 0 |
| 2048 | 1 | 4 | 128 | 0 |
| 4096 | 2 | 2 | 128 | `2 x 2` floats |
| 6144 | 3 | 2 | 192 | `2 x 3` floats |
| 8192 及以上 | 4 及以上 | 1 | `32 x warps_per_row` | `warps_per_row` floats |

这部分是前端 kernel 的 shape-specific dispatch. 源码只为跨 warp 的 row reduction分配上述少量 shared float,block 数为 `ceil(rows/rows_per_block)`.

### 7.2 四级 butterfly 的映射

256 是 `4^4`,所以需要四级 4 点 butterfly:

| 级 | 对应 base-4 digit | 数据位置 | warp 操作 | 本级归一化 |
| --- | --- | --- | --- | ---: |
| 0 | 最低位 | `lane` bits 0-1 | shuffle xor `1,2,3` | `0.5` |
| 1 | 次低位 | `lane` bits 2-3 | shuffle xor `4,8,12` | `0.5` |
| 2 | 第三位 | `lane` bit 4 + 相邻 slot | shuffle xor `16` + local slot pair | `0.5` |
| 3 | 最高位 | slot bits 1-2 | 纯 register slot 重排 | `0.5` |

前两级只改变 lane 内的 base-4 digit. 第三级跨 lane 16 并配对 `slot` 与 `slot+1`. 第四级只在当前线程的 `slot={0,2,4,6}` 或 `{1,3,5,7}` 间组合,不需要 warp shuffle.

### 7.3 为什么保持 native FP16/BF16 运算顺序

`h4_combine` 对 paired type 使用:

```text
__hadd2
__hsub2
__hmul2
```

FP16 使用 `half2`,BF16 使用 `bf162`. 这样做有两个目的:

1. 一条 paired 指令同时处理两个 group 的同一位置.
2. 保持与上游 native half/bfloat16 rotation 相同的运算次序,避免先转 FP32 再用代数重排造成额外的量化 code 差异.

这里不能仅凭源码声称 register 使用量或 occupancy 一定更优. NCU counter 当前不可用. 能确认的事实是 rotated row 不再写入全局内存,并且 FHT 数据在 kernel 内保持在 paired register 中.

---

## 8. 优化步骤 3: 在同一 kernel 内完成 absmax,quant 和 pack

FHT 结束后,`values` 仍在 register 中. 内核立即执行下面三步.

### 8.1 row absmax reduction

每个 lane 先扫描自己持有的 paired values:

```text
local_max = max(abs(all local low/high values))
```

然后用:

```text
shuffle xor 16,8,4,2,1
```

完成 warp 内 max reduction.

当一个 row 只有一个 warp时,不需要 shared memory. 当 `K>2048` 导致一个 row 使用多个 warp 时,每个 warp 只向 shared memory 写一个 `float local_max`,同步后再合并成整行 absmax. shared memory 的用途被缩小到"每 warp 一个标量",而不是保存整条 rotated row.

### 8.2 生成动态 row scale

scale 公式是:

```text
scale = max(min(absmax, finite_max(dtype)) / 7, 1e-10)
```

- `finite_max(FP16)=65504`.
- `finite_max(BF16)=3.38953139e38`.
- `1e-10` 防止全零 row 出现除零.
- 只有 `lane==0 && warp_in_row==0` 写一次 `scales[row]`.

### 8.3 直接量化并打包 INT4

每个元素执行:

```text
q = clamp(round(value / scale), -7, 7)
```

相邻列由相邻 lane 持有. 偶数 lane 用 `shuffle xor 1` 取到奇数 lane 的 code,然后一次写出两个 nibble:

```text
byte = (q_even & 0x0f) | ((q_odd & 0x0f) << 4)
```

例如:

```text
q_even = -3 -> two's-complement nibble 0xd
q_odd  =  5 -> nibble 0x5
packed byte = 0x5d
```

最终 activation buffer 形状为 `[M,K/2]`,dtype 在 PyTorch 侧表现为 `int8`,但 GEMM 解释的是每个 byte 内的两个 signed INT4 nibble.

### 8.4 这一步实际删除了什么

融合前:

```text
FHT -> write rotated half/bfloat16 -> read rotated tensor -> reduce/quant -> write INT4
```

融合后:

```text
FHT registers -> reduce/quant registers -> write INT4
```

它删除了 rotated tensor 的分配,写回和再次读取,也把 activation quantization 从一串 eager 操作收敛为一个稳定 CUDA kernel.

---

## 9. 优化步骤 4: 让 packed layout 直接匹配 CUTLASS

### 9.1 Activation A

CUTLASS 逻辑上看到:

```text
A: [M,K], signed INT4, RowMajor
```

物理上每两个相邻 K 元素占一个 byte:

```text
PyTorch storage shape: [M,K/2], int8 bytes
logical element order: A[m,0],A[m,1],A[m,2],A[m,3],...
```

### 9.2 Weight B

Linear weight 在 Python 侧保存为:

```text
qweight storage: [N,K/2], row-major bytes
```

GEMM 需要逻辑 `B:[K,N]`. 同一片内存可以直接解释为:

```text
B: [K,N], signed INT4, ColumnMajor
```

因此 hot path 不需要 transpose 或 layout conversion. "row-major `[N,K]` weight storage"和"column-major `[K,N]` GEMM operand"是同一内存的两种逻辑视图.

### 9.3 Output C/D

输出是:

```text
D: [M,N], FP16/BF16, RowMajor
```

完整 layout 表:

| Operand | 逻辑形状 | CUTLASS layout | PyTorch 物理 storage |
| --- | --- | --- | --- |
| A activation | `[M,K]` INT4 | RowMajor | `[M,K/2]` bytes |
| B weight | `[K,N]` INT4 | ColumnMajor | `[N,K/2]` bytes |
| D output | `[M,N]` FP16/BF16 | RowMajor | `[M,N]` |

CUTLASS 配置令 `AlignA=32`,`AlignB=32`,即每次对齐访问 32 个 INT4 元素,等于 16 bytes. output 对齐为 128 bits. `N%8==0` 是当前 output vector store 和 kernel contract 的一部分.

---

## 10. 优化步骤 5: 使用 SM89 signed INT4 Tensor Core GEMM

GEMM 主体使用:

```text
ElementA       = signed int4
ElementB       = signed int4
Accumulator    = int32
Compute        = float
Instruction    = 16 x 8 x 64
Architecture   = Sm89
Operator       = multiply-add-saturate
Pipeline stage = 3
```

当前保留两个候选 tile:

| candidate | threadblock tile | warp tile | instruction tile | stages |
| --- | --- | --- | --- | ---: |
| 0 | `128x256x128` | `64x64x128` | `16x8x64` | 3 |
| 1 | `128x128x256` | `64x64x256` | `16x8x64` | 3 |

第一次遇到一个新的 `(M,N,K)` 时,dispatch 会:

1. 检查每个 candidate 是否能实现当前参数.
2. 每个 candidate warmup 8 次.
3. 用 CUDA event 测量 32 次调用的总时间.
4. 选择最短者并缓存到全局 `(M,N,K) -> runner index` map.
5. 在线程局部保存最近一个 shape 和 runner,让连续相同 shape 避免 map lookup 和 mutex.

正式 benchmark 已经过 warmup,不包含这段 autotune. 因此当前结果描述的是 steady state. 如果关注首 token 冷启动,应把 JIT build,weight pack,workspace allocation 和 tile selection 单独列为另一套指标.

与宽泛的多架构 kernel 集合相比,这里只编译两个经过目标 shape 筛选的 SM89 tile. 这减少了当前 extension 的代码范围,但不能仅凭 candidate 数量断言速度来源. 实测证据是代表 shape 的 XQT CUTLASS GEMM median 为 `9.436 us`,官方 wheel 的 `int4_linear_kernel` median 为 `16.976 us`.

---

## 11. 优化步骤 6: 把反量化和 bias 融进 epilogue

如果 GEMM 只输出 INT32 accumulator,还需要后续 kernel 做:

```text
accumulator * activation_scale * weight_scale + bias
```

XQT 使用 CUTLASS Epilogue Visitor Tree 在 accumulator 写回前完成全部操作:

```text
VisitorAccFetch
  -> multiply activation_scale[m]
  -> multiply weight_scale[n]
  -> add bias[n]
  -> cast FP16/BF16
  -> store output[m,n]
```

广播方向是:

- `activation_scale[m]` 沿 N 广播.
- `weight_scale[n]` 沿 M 广播.
- `bias[n]` 沿 M 广播.

当原始 Linear 没有 bias 时,Python packing 会准备一个 FP32 全零 bias buffer,保持同一 epilogue contract,避免在 hot kernel 中增加 bias/no-bias 分支.

这一步删除了独立 dequantization,bias add 和 output cast kernel. INT32 accumulator 不需要先写到全局内存再读回.

---

## 12. 为什么最终保留两个 kernel,没有强行做 megakernel

直觉上可能会问: rotation,quant 和 GEMM 为什么不全部塞进一个 CUDA kernel?

关键依赖是 activation row scale:

```text
完整 row FHT
  -> 整行 absmax
  -> row scale
  -> 所有 INT4 codes
  -> GEMM
```

对大 K,row 可能跨多个 warp. GEMM 只有在 packed activation 和 scale 完整可见后才能消费. 把两阶段塞进一个普通 kernel 会遇到:

- block 间缺少普通 `__syncthreads()` 等价的全 grid barrier.
- persistent producer/consumer 或 cooperative launch 会显著增加调度复杂度.
- FHT/quant 偏好大量 paired value register,GEMM 又需要 shared-memory pipeline 和 accumulator register,资源需求会叠加.
- quant 和 GEMM 的最优 CTA/warp ownership 不相同.
- 一个阶段的 shape 特化会限制另一个阶段的 tile 搜索空间.

因此当前的工程选择是:

```text
kernel 1: 专注 FHT + row reduction + quant + pack
kernel 2: 专注 Tensor Core GEMM + epilogue
```

两个 kernel 之间的边界恰好是紧凑的 INT4 activation `[M,K/2]` 和 scale `[M]`,中间数据已经比 FP16/BF16 rotated tensor 小 4 倍. Nsight Systems 也确认每次 XQT range 固定是 2 个 kernel,完整 wrapper 已贴近 bound floor. 在没有新证据前,为了少一次 launch 强行合并并不一定更快.

---

## 13. 为什么当前不融合 norm

当前实现明确报告:

```python
module.execution_metadata()["norm_fused"] is False
```

原因不是"norm 永远不能融合",而是本轮优化的 operator contract 从 Linear 输入开始. benchmark 的所有对照路径都不包含相邻 norm. 如果只在 XQT 路径加入 norm,再与不含 norm 的官方 Linear 比较,性能口径会失真.

真正的 norm fusion 至少需要新增这些契约:

- pattern 必须确认是相邻的 RMSNorm 或 LayerNorm,不能把任意 module 猜成 norm.
- 明确 epsilon,weight,bias,residual 和 dtype 语义.
- 明确 norm 的 row reduction 与 FHT 的 row reduction怎样共享数据和同步.
- 明确模型图中原 norm 是否被删除,避免重复执行.
- benchmark baseline 必须是 `norm + ConvRot W4A4`,四条路径都包含相同的 norm 工作.
- metadata 和 fallback 必须区分 `norm_fused=true/false`.

理论上,把 norm 与 FHT/quant 融合到第一个 kernel 有机会删除一次 normalized activation 的写回和再次读取. 但这需要单独实现和验收. 当前 W4A4 rowwise 数据不能支持"已经做了 norm fusion"或"norm fusion 一定再快多少"的结论.

仓库里的旧 `ConvRotNormInt8Linear` 是另一条显式 opt-in 能力,不能当作本 W4A4 dynamic fastpath 已融合 norm 的证据.

---

## 14. 优化步骤 7: 用 C++ dynamic runner 消除稳态管理开销

只有快 kernel 还不够. 完整 `nn.Module` 必须稳定进入它,并避免每次重新分配 workspace.

热调用链是:

```text
ConvRotMixedPrecisionLinear.forward(input)
  -> _rowwise_w4a4_hot_forward(input)
  -> cached DynamicConvRotW4A4RowwiseLinear(input)
  -> input.contiguous() only when needed
  -> flatten leading dimensions into rows
  -> workspace_for(rows,current_cuda_stream)
  -> quantize/FHT kernel
  -> CUTLASS GEMM kernel
  -> restore original leading dimensions
```

### 14.1 按 row 数和 stream 缓存 workspace

C++ dynamic runner 的 workspace key 是:

```text
(rows, cuda_stream_pointer)
```

每个 workspace 包含:

```text
packed activation: [rows,K/2] int8 bytes
activation scales: [rows] float32
```

为什么 key 必须包含 stream:

- 两个 stream 可能并发执行同一个 module.
- 如果共享一个 activation buffer,一个 stream 会覆盖另一个 stream 尚未消费的数据.
- stream-aware workspace 用额外内存换取正确的并发语义.

测试会在默认 stream 和额外 `torch.cuda.Stream()` 上调用同一个 runner,并检查 `workspace_count()==2`.

### 14.2 支持高维和 non-contiguous 输入

dynamic runner 接受 trailing dimension 为 K 的高维输入,例如 `[B,S,K]`. 它通过:

```text
rows = input.numel() / K
```

折叠 leading dimensions,输出再把最后一维替换成 N.

对 non-contiguous input,C++ 显式调用 `input.contiguous()`. 对已经 contiguous 的 tensor,PyTorch 可复用原 storage;对非连续 view,只做必要 materialization. 这项成本属于真实 wrapper contract,没有从 benchmark 中偷偷移除.

### 14.3 output allocation仍然存在

dynamic runner 每次调用仍用 `torch::empty` 创建 output. 被缓存的是 activation workspace 和 scales,不是最终输出. 因此"热路径无任何 allocation"是不准确的表述.

---

## 15. 优化步骤 8: 让 cache 可失效,而不是返回陈旧结果

缓存 packed weight 和 bound runner 会引入一个风险: 用户可能原地修改 module buffer.

当前有五层状态检查:

| 层 | key/检查 | 作用 |
| --- | --- | --- |
| Python packed cache | device,dtype,data pointer,tensor version,shape 等 compute-view signature | weight/scale/bias 或执行 dtype 变化时重新构造 rowwise packed state |
| Python runner cache | device index,input dtype,source tensor object identity | module buffer 被替换时放弃旧 runner |
| C++ bound state | source weight/scale/bias 的 PyTorch tensor `_version()` | 原地 mutation 时抛 `XQT_ROWWISE_W4A4_STALE_STATE` |
| C++ workspace cache | `(rows,stream)` | shape/stream 变化时分配独立 workspace |
| CUDA GEMM dispatch | `(M,N,K)` + thread-local last key | 复用已选择的 CUTLASS tile |

当 C++ 抛出 `XQT_ROWWISE_W4A4_STALE_STATE` 时,Python hot path 清理 runner,回到 materialization path 重新 pack,再绑定新的 dynamic runner.

`.to()` 或其他 `_apply()` 设备/dtype 迁移会清空 Python runtime cache. 测试还覆盖:

- 原地修改 `weight_scale` 后输出变化且 runner 被替换.
- 原地修改 bias 后输出按 bias 变化.
- dtype 迁移后 packed cache 和 runner cache 均为空.

这是性能实现的一部分. 没有正确失效的 cache 只能在 benchmark 里快,不能作为可用 runtime.

---

## 16. 优化步骤 9: 把 backend policy 写成显式 contract

`w4a4_runtime_backend` 支持四个值:

| 配置 | rowwise 行为 | 后续 fallback |
| --- | --- | --- |
| `auto` | 只有 `group_size == padded_input_features` 的 whole-row artifact 才自动选择 rowwise | grouped artifact 优先保留 Nunchaku,再到 reference |
| `rowwise` | 显式请求 rowwise warp-FHT path | capability/执行失败后尝试 Nunchaku,再到 reference |
| `nunchaku` | 禁止 rowwise | 使用 Nunchaku,失败后 reference |
| `reference` | 禁止所有 native W4A4 | 使用 dynamic A4 和 dequantized weight 的 `F.linear` reference |

为什么 `auto` 不能对 grouped artifact 自动启用 rowwise,见第 20 节的数值结果. 简单地说:

```text
grouped scale: 每个 weight row 有多个 group scale
rowwise scale: 每个 weight row 只有一个 scale
```

把前者解量化后再压成后者会增加权重量化误差. 用户显式写 `rowwise` 表示接受这个取舍;`auto` 不能替用户做这个决定.

每次 forward 后应查看:

```python
metadata = module.execution_metadata()

metadata["resolved_w4a4_runtime_backend"]
metadata["native_w4a4_used"]
metadata["rowwise_w4a4_used"]
metadata["runtime_weight_layout"]
metadata["fused_epilogue"]
metadata["native_w4a4_fallback_reason"]
metadata["norm_fused"]
```

配置名或 quantization nature 不是某次 forward 已执行 native MMA 的充分证据.

---

## 17. 编译和部署边界

Python binding 使用 `torch.utils.cpp_extension.load` JIT 编译 extension. 当前编译条件是:

```text
C++:  -O3 -std=c++20
CUDA: -O3 -std=c++20
       --expt-relaxed-constexpr
       --expt-extended-lambda
       --generate-line-info
arch:  TORCH_CUDA_ARCH_LIST=8.9
```

CUTLASS headers 来自 TileLang bundled `3rdparty/cutlass/include`. 运行前需要:

- CUDA 可用.
- 当前 device capability 恰好是 `8.9`.
- TileLang 安装中存在 CUTLASS headers.
- C++/CUDA extension toolchain 可用.

可用以下环境变量强制关闭这条 backend:

```bash
XQT_DISABLE_CONVROT_W4A4_ROWWISE_SM89=1
```

关闭或 capability 不满足时,模块通过已有 fallback 路径继续执行,而不是加载错误架构的 binary.

---

## 18. 优化步骤 10: 用分层 benchmark 判断到底快在哪里

正式证据环境是:

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER,66 SM,17170956288 bytes 显存 |
| compute capability | `sm_89` |
| driver | `591.86` |
| GPU state | P0,SM clock `2670 MHz`,memory clock `10501 MHz`,power limit `320 W` |
| host | Linux/WSL2 x86_64 |
| Python | `3.12.12` |
| PyTorch | `2.12.1+cu130` |
| CUDA runtime | `13.0` |
| official baseline | PyPI `comfy-kitchen 0.2.28`,source commit `b72e6dfa79b79a7aee33a9c7608b5d9b3005b7af` |

正式 benchmark 同时测六个关键层级:

| 名称 | 包含内容 | 用途 |
| --- | --- | --- |
| `rowwise_bound_floor` | 固定 rows 的 C++ bound runner | 近似 native operator floor |
| `rowwise_dynamic_runner` | C++ 按 rows/stream 管 workspace | 检查动态管理成本 |
| `xqt_wrapper` | 完整 `ConvRotMixedPrecisionLinear` | 用户实际调用层 |
| `comfy_kitchen_official` | 官方完整 Python operator | 同 rowwise weight contract 官方对照 |
| `nunchaku_bound` | grouped Nunchaku W4A4 | 已有 native contract 对照 |
| `split_rotation_then_nunchaku` | 显式 rotation + Nunchaku | 判断 rotation/quant fusion 的价值 |

测量政策:

```text
warmup              = 30 calls
calls per sample     = 1000
samples              = 15
latency statistic    = median CUDA-event elapsed time / call
CPU affinity         = [22]
random seed          = 2028
```

CUDA event 包围每个样本的 1000 次连续调用,样本末尾同步. 这种写法降低单次 event 和 host measurement 噪声,同时保留 CUDA stream 上实际的 kernel launch gap.

### 18.1 完整 12-case 结果

下表单位为 `us`. speedup 列统一写成 `baseline / XQT`,所以大于 1 表示 XQT 更快.

| dtype | M | K=N | XQT wrapper | official | Nunchaku | split | official/XQT | Nunchaku/XQT | split/XQT |
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

### 18.2 怎样解释这些数字

- `official/XQT=2.360x-3.056x` 说明当前 XQT 完整 operator 已超过所测官方 wheel,不是只在 isolated kernel microbenchmark 中更快.
- `Nunchaku/XQT=1.175x-1.586x` 说明专用 rowwise CUTLASS path 也超过已有 grouped bound path,但两者数值 contract 不完全相同.
- `split/XQT=1.875x-2.512x` 直接体现删除显式 rotation materialization 和使用专用两 kernel path的收益.
- `wrapper/bound floor` 中位只有 `1.034x`,说明 Python policy/cache 检查没有吞掉 kernel 收益.

这些是最终实现相对各 baseline 的总收益. 因为没有为每个中间开发版本保存正式 artifact,不能把最终 `2.910x` 拆成"FHT贡献多少,epilogue贡献多少,cache贡献多少"的精确加法.

---

## 19. Nsight Systems: 两个 kernel 各自快了多少

代表 workload 是 BF16 `M=256,N=K=2048`,每个 NVTX range 连续执行 100 次.

| 路径 | 每次调用的 kernel 序列 | kernel median |
| --- | --- | ---: |
| XQT rowwise | warp-FHT quant + CUTLASS W4A4 | `2.761 us + 9.436 us` |
| official | official ConvRot quant + `int4_linear_kernel` | `3.186 us + 16.976 us` |
| Nunchaku | rotated activation quant + W4A4 GEMM | `4.399 us + 18.812 us` |
| split | copy + BF16 rotation GEMM + padded quant + W4A4 GEMM | `1.002 us + 5.886 us + 6.918 us + 18.872 us` |

可以做三个不依赖 hardware counter 的时间结论:

1. XQT quant kernel 相对官方 quant kernel 的 kernel-body speedup 约为 `3.186/2.761=1.154x`.
2. XQT GEMM 相对官方 wheel GEMM 的 kernel-body speedup 约为 `16.976/9.436=1.799x`.
3. split 路径确实有 4 个 kernel,XQT 确实只有 2 个 kernel;没有偷偷回到 Python split.

XQT 两个 kernel 的 median 之和约为 `12.197 us`,官方约为 `20.162 us`. 这只是 kernel body 的结构对照,不是完整 operator latency. CUDA-event benchmark 还包含同一 stream 上的 launch gap和 wrapper/runtime 调度,所以最终 operator speedup不能用两个 kernel median 简单相除替代.

### 19.1 NCU 为什么没有数据

Nsight Compute 返回:

```text
ERR_NVGPUCTRPERM
counter_permission_denied
```

它表示当前 driver/container 环境不允许读取 NVIDIA performance counters,不是 kernel 执行错误. 因此本文不推断:

- achieved occupancy.
- L1/L2/DRAM hit rate或带宽.
- warp stall 原因.
- register count和spill.
- Tensor Core pipe utilization.
- roofline 位置.

等权限开放后,这些指标应作为下一轮诊断,而不是用 Nsight Systems 的 duration 猜出来.

---

## 20. 数值正确性: 必须同时看三种 reference

性能优化不能只做一个 `allclose`. 本轮用了三层比较.

### 20.1 wrapper 与同一 native contract

- 完整 wrapper 与 C++ dynamic runner 在 12 个 case 中完全一致.
- rowwise direct binding 与 bound runner 一致.
- non-contiguous FP16/BF16 input 也通过 packed INT4 accumulator reference.

reference 直接解包 A/W codes,执行:

```text
int32_acc = activation_codes @ weight_codes.T
expected = int32_acc * a_scale[:,None] * w_scale[None,:] + bias
```

这验证了 packed layout,CUTLASS accumulator 和 epilogue 广播方向.

### 20.2 XQT 与官方同 rowwise contract

XQT 相对官方输出:

```text
relative RMSE: 0.001905 - 0.012858
max absolute difference: 0.6982421875
```

两边使用完全相同的 packed weight,row scale 和 bias,但 activation FHT/rounding 的 native 运算顺序和 kernel 实现仍可能产生边界 code 差异. 因此这是同 contract 的数值接近,不是 bitwise 等价声明.

Python 数学 reference 的 activation code允许 INT4 的 `[-8,7]` 表示范围,当前 rowwise CUDA kernel 使用对称 `[-7,7]` 和 `absmax/7` contract. 测试中 CUDA activation code 相对数学 reference 的最大差不超过 1.

### 20.3 rowwise runtime 与原 grouped artifact

原始 `ConvRotMixedPrecisionLinear.from_linear(...,group_size=128)` 生成 grouped weight scale. 显式 `w4a4_runtime_backend="rowwise"` 会:

```text
grouped packed artifact
  -> dequantize to rotated dense weight
  -> requantize with one scale per output row
  -> rowwise packed runtime weight
```

相对 grouped dense artifact 的 relative RMSE 是:

| runtime path | relative RMSE |
| --- | ---: |
| rowwise | `0.1879-0.2045` |
| Nunchaku grouped | `0.1224-0.1233` |

这说明 rowwise 的更快 kernel 伴随更粗的 weight scale granularity. 所以:

- 官方公平性能对照必须给两边相同 rowwise packed weight.
- grouped dense artifact 用于衡量精度取舍,不能当作同输出 contract.
- `auto` 不能把 grouped artifact 静默重解释为 rowwise.
- 真正部署前仍需在目标模型/任务上验证质量,operator relative RMSE 不能替代端到端指标.

---

## 21. 每一步删除了什么成本

| 优化步骤 | 删除或压缩的成本 | 当前证据 |
| --- | --- | --- |
| capability 特化 | 通用 rotation size,dtype,SM,padding分支 | 源码 gate和明确 fallback |
| warp register FHT | 通用 rotation matrix load和显式 rotation GEMM | split trace 有 rotation GEMM,XQT 没有 |
| FHT + quant + pack | rotated FP16/BF16 tensor 写回/重读,多个 eager quant op | XQT 只有一个前端 kernel |
| INT4 native layout | hot-path transpose和activation repack | A/B 直接进入 CUTLASS |
| CUTLASS s4 GEMM | dequantized floating-point GEMM | trace 显示 signed INT4 Tensor Core kernel |
| EVT epilogue | 独立 activation scale,weight scale,bias和cast kernel | GEMM 后无额外 epilogue kernel |
| shape tile cache | 每次调用重新搜索 candidate | `(M,N,K)` cache + thread-local fast key |
| C++ dynamic runner | 每次 workspace allocation和Python子算子链 | wrapper/bound floor中位 `1.034x` |
| stream-aware workspace | 并发 stream 共享 scratch的错误和重复分配 | multi-stream测试,`workspace_count()==2` |
| version invalidation | cache 返回陈旧weight/bias结果 | mutation和`.to()`测试 |

这张表是结构归因. 只有带独立时间数据的行才能解释为测得的速度变化;其他行是源码和行为测试证明的成本删除,不能伪装成独立百分比.

---

## 22. 如何复跑

### 22.1 运行 policy 和 CUDA correctness tests

```bash
pytest -q tests/xqt/quant/test_convrot_4bit_quantizer.py -k "rowwise or runtime_backend"
```

CUDA case 需要 `sm_89`,TileLang bundled CUTLASS headers 和可用 extension toolchain. 其他机器会跳过或走 capability fallback.

### 22.2 运行正式 benchmark

官方 baseline wheel 当前通过这个临时路径注入:

```bash
taskset -c 22 env \
  PYTHONPATH=/tmp/xqt-upstream-9kxLGA/comfy-kitchen-wheel \
  python tools/benchmark_convrot_w4a4_sm89.py
```

结果写入:

```text
artifacts/xqt/benchmarks/convrot_w4a4_sm89/summary.json
```

复跑前先确认 `comfy-kitchen` 版本和源码 commit,否则不能把新结果与本文直接拼接.

### 22.3 运行 Nsight Systems

```bash
env PYTHONPATH=/tmp/xqt-upstream-9kxLGA/comfy-kitchen-wheel \
  nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=artifacts/xqt/profiling/convrot_w4a4_sm89/nsys/convrot_w4a4_sm89 \
  python tools/profile_convrot_w4a4_sm89.py
```

现有 artifacts 位于:

- [benchmark summary](../../../artifacts/xqt/benchmarks/convrot_w4a4_sm89/summary.json)
- [profiling conclusion](../../../artifacts/xqt/profiling/convrot_w4a4_sm89/conclusion.md)
- [Nsight Systems kernel summary](../../../artifacts/xqt/profiling/convrot_w4a4_sm89/nsys/kernel-summary.csv)
- [NCU permission record](../../../artifacts/xqt/profiling/convrot_w4a4_sm89/ncu-permission.json)

在 GPU,driver,container 和权限未变化前,不要重复运行 NCU. 当前失败原因已经记录为 `counter_permission_denied`.

---

## 23. 可以迁移到其他低比特算子的规则

### 23.1 先定义 runtime operand contract,再选 kernel

同样写着 W4A4,可能分别表示 grouped scale,rowwise scale,block scale或某种 packed layout. 不先固定 scale/layout,性能数字没有可比性.

### 23.2 融合的第一目标通常是删除大中间 tensor

本例最关键的 fusion 不是把所有东西塞成一个 megakernel,而是让 FHT 结果直接变成 INT4. fused boundary 选在最紧凑,最稳定的中间表示上.

### 23.3 register-only 变换适合固定小结构

256 点 regular-Hadamard 有固定 butterfly 和天然 warp mapping. 对任意矩阵或任意 rotation size,同样策略未必成立.

### 23.4 epilogue 是 scale,bias和残差融合的自然位置

只要广播语义明确,在 accumulator 写回前完成 dequant和bias,通常比先写 INT32/FP32临时 tensor更合理.

### 23.5 operator runtime必须和kernel一起优化

workspace,stream,shape cache,version失效和高维输入处理会决定微秒级 kernel能否在真实 module中保留收益.

### 23.6 benchmark必须同时保留官方,结构和floor baseline

- 官方 baseline回答"是否追平或超过外部实现".
- split baseline回答"fusion是否真的删除了工作".
- bound floor回答"wrapper是否吞掉kernel收益".
- 数值 baseline回答"更快是否换了contract".

---

## 24. 下一步优化方向

下面都是需要新证据才能 promotion 的方向,不是当前已实现能力.

1. 为明确选择 rowwise contract 的 recipe 直接生成 rowwise artifact,避免首次 forward 从 grouped artifact解量化再量化. 必须同时更新质量报告和 schema,不能静默改变默认 artifact.
2. 对更多实际模型 shape 扩展 CUTLASS candidate sweep. 候选增加前要用独立 benchmark证明收益,避免只增加编译时间和binary体积.
3. 在 NCU权限开放后测 register,spill,occupancy,memory throughput和stall,再决定 FHT `PairsPerWarp`,rows-per-block和GEMM tile是否需要调整.
4. 为相邻 RMSNorm/LayerNorm建立独立 fused operator,使用包含 norm的公平 baseline,而不是修改当前 Linear-only结果的口径.
5. 评估 CUDA Graph或预分配output对固定shape服务场景的收益. 当前dynamic runner仍分配output.
6. 为 `sm_90a`/Blackwell单独设计架构路径. 不应把 `Sm89` CUTLASS配置直接当作新架构最优实现.
7. 扩展 K/rotation支持时重新设计warp ownership. 当前 `K=1024`和`K%2048==0`来自paired group mapping,不是任意K通用公式.

---

## 25. 最后用一句话复盘

这次优化真正有效的地方,不是简单地把"rotation,quant,GEMM"写在同一个函数名里,而是先固定 rowwise W4A4 contract,再用 warp register FHT直接产生CUTLASS可消费的packed INT4 activation,把scale和bias放进GEMM epilogue,最后用stream-aware C++ runner把两kernel路径稳定接到完整XQT wrapper中. 性能因此超过所测官方实现,但代价是rowwise weight scale精度边界必须显式暴露,而norm fusion仍需作为另一项独立工作来验证.
