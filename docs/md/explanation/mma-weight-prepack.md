# 离线 MMA 权重预排板 (Weight Prepack)

## 负责内容

说明 XQT 中 **offline weight prepack**: 在 forward 之前把权重从数学布局改写成 Tensor Core / MMA 友好布局, 热路径只做连续访存 + `mma.sync`.

不负责: 训练 / QAT, 原生 FP4 MMA (Blackwell), 或替代 W4 存储 pack (那是另一层).

---

## 为什么需要

`mma.m16n8k32` 等指令吃的是 **warp fragment**, 不是任意 `[K,N]` 矩阵.

对 INT8 B 矩阵, 每线程需要:

```
B[k+0][n], B[k+1][n], B[k+2][n], B[k+3][n]  -> 一个 uint32
```

若权重是 row-major `[K,N]`, 同一列上相邻 K **地址跨步为 N**, 只能 4 次标量读再 pack.

**Runtime 转置进 SMEM** 会在全局内存上做跨步 gather, 往往更慢.

**Offline prepack** 在量化 / retarget / 加载时做一次 ` [K,N] -> [N,K] ` (或更细的 atom layout), 推理时:

- G2S 沿 K 向量化 (`cp.async` 16B)
- B fragment 一次 `uint32` load

---

## 与 W4 存储 / INT8 转义的关系

| 层 | 布局 | 何时做 |
|----|------|--------|
| W4 存储 | packed int4 + group scale | 量化时, 省 HBM |
| INT8 compute view | 数学 `[K,N]` int8 + channel scale | retarget 时 |
| **MMA prepack** | 算力 `[N,K]` (或 atom 序) | **offline**, 可缓存 |

路径:

```
packed W4
  -> dequant / re-encode int8  [K,N]     (w4_storage_int8_mma)
  -> prepack int8:sm_89:b_nk   [N,K]     (本机制)
  -> ptx_sm89 kernel (prepacked_b)
```

W4 仍可不膨胀; prepack 作用在 **int8 热视图** 上, 可与 `release_int8_compute_view` 同类生命周期管理.

---

## API

```python
from xqt.operator_opt.kernels.prepack import (
    list_prepack_specs,
    prepack_weight,
    unpack_weight,
    INT8_SM89_B_NK,
)

list_prepack_specs()
# int8:sm_89:b_nk -> implemented
# fp4:sm_89:b_mma, int4:..., fp8:..., sm_80/sm_90 -> placeholder

result = prepack_weight(qweight_t, INT8_SM89_B_NK)  # qweight_t: [K,N] int8
packed = result.packed  # [N,K] int8 contiguous
math = unpack_weight(packed, INT8_SM89_B_NK, math_shape=result.math_shape)
```

`Int8MmaLinear(engine="ptx_sm89")` 在构造时自动 prepack 并缓存 `_qweight_prepacked_b`.

Static 激活路径 (推荐):

1. `int8mma_quantize_static_half` — half → int8
2. `int8mma_run_prepacked_b` — 吃 prepack 权重

`engine=auto` 在 sm_89 上: **M≥192 用 ptx_sm89, 否则 tilelang** (小 M 融合 quant 更强).

---

## Layout 注册表

| layout key | 状态 | 说明 |
|------------|------|------|
| `int8:sm_89:b_nk` | **implemented** | `[K,N]->[N,K]`, Ada ptx_sm89 |
| `int4:sm_89:b_mma` | placeholder | W4 / int4 路径预留 |
| `fp4:sm_89:b_mma` | placeholder | FP4 存 + INT8 算或未来 FP4 MMA |
| `fp8:sm_89:b_mma` | placeholder | FP8 fragment + scale |
| `int8:sm_80:b_mma` | placeholder | Ampere 专用 tag |
| `int8:sm_90:b_mma` | placeholder | Hopper WGMMA 布局 |

新增精度 / 架构: 在 `xqt/operator_opt/kernels/prepack/` 注册 `PrepackSpec`, 实现 `pack`/`unpack`, 再挂对应 kernel.

---

## 实现文件

| 文件 | 说明 |
|------|------|
| `xqt/operator_opt/kernels/prepack/base.py` | 注册表, `prepack_weight` / `unpack_weight` |
| `xqt/operator_opt/kernels/prepack/int8_sm89.py` | INT8 sm_89 B_NK |
| `xqt/operator_opt/kernels/cute/int8mma_kernel.cu` | `int8mma_run` + `int8mma_run_prepacked_b` |
| `xqt/operator_opt/kernels/cute/int8mma_binding.py` | ctypes + `prepack_qweight_t_for_ptx_sm89` |
| `xqt/quant/quantizers/int8_mma.py` | `engine=ptx_sm89` 自动 prepack |
| `tests/xqt/operator_opt/test_prepack_int8_sm89.py` | 单测 |

---

## 调优注意

- Prepack 是 **layout 变换**, 不改变数值 (同一 int8 码点).
- 换 tile / MMA atom / arch 时 layout 可能失效, 必须用 **layout key** 绑定 kernel, 不要静默 reinterpret.
- 小 batch 上 prepack 收益可能被激活 quant / launch 盖住; 大 N/K 时更明显.
- 更细的 atom 级 swizzle (ldmatrix 序) 可作为 `int8:sm_89:b_atom` 后续 layout, 不必破坏 `b_nk`.

---

## 相关文档

- [w4-int8-mma-retarget.md](w4-int8-mma-retarget.md): W4 存储 + INT8 算力转义
- [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md): 手写算子调优 (shared layout / ldmatrix)
