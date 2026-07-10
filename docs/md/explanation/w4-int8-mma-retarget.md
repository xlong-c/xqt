# W4 存储 + INT8 MMA 计算转义

## 负责内容

本文说明 XQT 中 "W4 存储 + INT8 MMA 计算转义" 的设计动机, 实现路径, 策略对照和基准验收方法.

不负责: 原生 FP4 MMA (Blackwell 专属), QAT 训练, 或稠密 FP16 对比.

---

## 背景

在非 FP4 原生 GPU (SM < 100, 如 Ada SM89) 上部署已量化的 W4A4/FP4 模型时, 若直接走 "dequant → FP16 GEMM" (PSEUDO 量化), 显存有压缩但算力仍落在 FP16 水平, 没有吃到低精度 MMA 加速.

XQT 新增的 `w4_storage_int8_mma` 策略将同一份 packed W4 权重在运行时重定标为 per-channel INT8, 然后通过 INT8 Tensor Core (~2-4x FP16 峰值吞吐) 执行 W8A8 GEMM, 从而在非原生 FP4 设备上获得比退化到 FP16 更好的算力利用.

核心契约:

- **W4 存储不膨胀**: HBM 中仍是 packed signed int4 码 + group scale
- **计算走 INT8 Tensor Core**: 运行时 dequant 后 re-encode 为 per-channel int8
- **不做 QAT**: 这是 offline pack + runtime retarget, 不涉及训练

---

## 策略对照

### 三条量化路径

| 策略 | nature | 存储 | 计算 | 适用场景 |
|------|--------|------|------|---------|
| `fp4_weight_only` | **PSEUDO** | packed int4 | dequant → FP16 GEMM | 只压缩显存, 不算力加速 |
| `dynamic_int8_mma` | **TRUE** | per-channel int8 | W8A8 INT8 Tensor Core | 原生 INT8 模型 |
| **`w4_storage_int8_mma`** | **TRUE** | packed int4 (**同上**) | dequant → int8 → W8A8 INT8 Tensor Core | **非 FP4 设备上的算力转义** |

### 关键区别

- `fp4_weight_only` 和 `w4_storage_int8_mma` 共享同一份 packed W4 存储
- 前者的 forward 是 unpack → dequant → `F.linear(fp16)`
- 后者的 forward 是 unpack → dequant → per-channel int8 re-encode → `Int8MmaLinear(int8)`
- 两者都可以从同一份 FP4WeightOnlyLinear 转换而来, 不需重新量化

---

## 实现结构

```
xqt/quant/quantizers/
  fp4_weight_only.py         ← packed W4 存储 (FP4WeightOnlyLinear)
  int8_mma.py                ← W8A8 INT8 MMA (Int8MmaLinear)
  w4_storage_int8_mma.py     ← W4 存储 + INT8 MMA 转义 (W4StorageInt8MmaLinear)

xqt/operator_opt/kernels/tilelang/
  int8_mma.py                ← TileLang INT8 MMA kernel (int8_mma_tilelang / int8_linear_tilelang)
```

### W4StorageInt8MmaLinear 核心逻辑

```
forward(inputs):
  1. _ensure_compute_view():
     packed W4 → dequant (float) → per-channel int8 (qweight_t + channel_scale)
     → 构建 Int8MmaLinear(engine=tilelang|torch_int_mm)
  2. compute(inputs):
     激活量化 (int8) + INT8 MMA + rescale → output

release_int8_compute_view():
  丢弃 Int8MmaLinear, 只保留 W4 存储 (显存不膨胀)
```

### 支持的引擎

| 引擎 | 加速比 (vs torch._int_mm) | 状态 |
|------|--------------------------|------|
| `torch_int_mm` | 基准 | PyTorch 内置, 但 Ada 上 dispatch 旧 CUTLASS kernel |
| `tilelang` | ~1.35-1.47x | JIT 编译 warp 级 INT8 MMA, 小 batch 融合路径仍强 |
| `ptx_sm89` | 大矩阵 ~60 TOPS 级 | 手写 Ada sm_89 PTX (cp.async + m16n8k32), 见 `xqt/operator_opt/kernels/cute/` |

---

## 使用方式

### Python API

```python
from xqt.quant import quantize_with_w4_storage_int8_mma

# 从 FP4WeightOnlyLinear 转换 (推荐)
result = quantize_with_w4_storage_int8_mma(
    fp4_model,
    engine="tilelang",
    source="fp4_weight_only",
    activation_scale_mode="static",
)

# 或从 nn.Linear 一步到位 (自动 pack W4 + retarget)
result = quantize_with_w4_storage_int8_mma(
    model,
    engine="tilelang",
    source="linear",
    group_size=128,
)
```

### Session API

```python
from xqt import XQTOptimizationSession

session = XQTOptimizationSession(
    project={"name": "w4_int8", "artifact_dir": "artifacts/"},
    model=model,
    example_inputs=x,
)

session.quant(
    name="retarget",
    strategy="w4_storage_int8_mma",
    policy={
        "dtype": "int8",
        "scheme": "w4_storage_int8_mma",
        "engine": "tilelang",
        "group_size": 128,
    },
)
```

### 静态激活标定 (推荐)

```python
for name, module in w4_model.named_modules():
    if isinstance(module, W4StorageInt8MmaLinear):
        scale = float(calib_input.float().abs().amax() / 127.0)
        module.activation_scale_mode = "static"
        module._activation_scale = scale
        module.release_int8_compute_view()
        compute = module._ensure_compute_view()
        compute.activation_scale_mode = "static"
        compute.set_static_activation_scale(scale)
```

---

## 加速基准

### 测试脚本

`examples/mlp_w4_int8_mma_acceptance.py`

运行:

```bash
PYTHONPATH=/root/workspace/xdl python examples/mlp_w4_int8_mma_acceptance.py
```

### 测试设计

- 10 层 MLP (Linear + ReLU), 分类头保持 FP16
- 同一份 packed W4 网络, 对照两条运行时路径:

| 路径 | 含义 |
|------|------|
| **A (参考)** | 同一 packed W4, dequant → FP16 GEMM (无 INT8 转义) |
| **B (被测)** | 同一 packed W4, TileLang INT8 MMA 转义 |

- **不做 QAT**: 权重是随机初始化的, 量化后不训练
- **准确度定义**: B 的 argmax 输出与 A 的一致率 (而非某任务标签)

### 验收门禁 (BenchConfig)

```python
max_accuracy_drop_pp = 1.0     # B vs A top-1 drop <= 1pp
min_speedup_b_vs_a = 1.15      # B median latency / A median latency >= 1.15x
```

### 实测数据 (RTX 4070 Ti SUPER, SM89, 2026-07 重测)

**最优配置 (推荐验收点)**: depth=10 MLP, `engine=tilelang`, tile `64x64x64`, threads=128, stages=2, static activation, `cache_int8_compute_view=True`.

| width | batch | A (W4→FP16) | B (TileLang INT8) | **speedup** |
|-------|-------|-------------|-------------------|-------------|
| 8192 | 64 | 2.622ms | 1.795ms | **1.46x** |
| **10240** | **64** | **4.626ms** | **2.858ms** | **1.62x** (稳定 median) |
| 10240 | 64 | ~5.09ms | ~2.81ms | **~1.81x** (单次扫峰) |
| 12288 | 64 | 6.356ms | 4.004ms | **1.59x** |

门禁仍用 `min_speedup_b_vs_a = 1.15`; 冲高收益优先 **width≥8192 且 batch=64**.

### 引擎对照 (同一 W4 网络, width=10240, batch=64)

| 引擎 | 相对 path A | 备注 |
|------|-------------|------|
| **tilelang** | **~1.6–1.8x** | **batch=64 端到端最优** (融合 static 激活 quant 进 MMA pipeline) |
| torch_int_mm | ~0.4–1.4x | 视 shape, Ada 上常慢 |
| **ptx_sm89** | batch=64 ~0.7x; **M≥256 单层可超 TileLang** | 已接: offline prepack B + CUDA quant + prepacked GEMM (~100 TOPS). `engine=auto` 在 M≥192 选 ptx |

### ptx_sm89 融合路径 (v8)

```
half activations
  -> int8mma_quantize_static_half  (vector CUDA quant)
  -> int8mma_run_prepacked_b        (B offline [N,K], ~100 TOPS GEMM)
  -> half output (+ optional bias)
```

`Int8MmaLinear(engine="ptx_sm89", activation_scale_mode="static")` 走该路径; 构造时自动 prepack.
M 会 pad 到 128 的倍数以匹配 BM=128.

### 多 batch 扫描 (width=8192, tilelang)

| batch | speedup vs A |
|-------|--------------|
| 32 | ~1.47x |
| 48 | ~1.55x |
| **64** | **~1.46–1.72x** |
| 96 | ~0.91x |
| 128 | ~1.01x |

INT8 路径在 **M≈32–64** 时最优; M≥96 时 FP16 cuBLAS 常追上或反超.

### 多宽度扫描 (batch=64, tilelang)

| width | speedup |
|-------|---------|
| 4096 | ≤1.0x (常无收益) |
| 6144 | ~1.1–1.3x |
| 8192 | **~1.5x** |
| **10240** | **~1.6–1.8x (峰)** |
| 12288 | **~1.6x** |

**宽度越大, INT8 收益越显著** (算力量 O(N^2), 固定开销摊薄). 最优甜点: **width=10240, batch=64, tilelang 64³**.

---

## 为何达不到理论 2-4x

Ada 上 INT8 Tensor Core 理论峰值 ~353 TOPS, FP16 ~88 TFLOPS (FP32 accum). 理论比 ~4x.

实测达不到的原因:

| 因素 | 解释 |
|------|------|
| **M=64 太小** | 64 行只能填 1-2 个 SM 的 Tensor Core 流水线, 大量 SM 空闲 |
| **cuBLAS FP16 高度调优** | 手写汇编, 接近 70% 峰值效率 |
| **TileLang JIT kernel** | 自动生成, ~45% 峰值效率 |
| **torch._int_mm 用旧 CUTLASS kernel** | SM80 fallback, 不是 SM89 优化版 |
| **激活量化 + rescale 开销** | 每层 ~0.07ms, 占单层的 ~30% |

### 更优引擎探索

| 方向 | 状态 |
|------|------|
| TileLang INT8 MMA | 端到端 MLP batch=64 仍强: ~1.54x vs FP16 |
| 手写 PTX `ptx_sm89` | **已接入** `Int8MmaLinear` / `w4_storage_int8_mma`; 大 GEMM ~61 TOPS @4096; 小 shape 未必赢融合 TileLang |
| CuTe DSL (Python) | CUTLASS 4.6 无 warp 级 MmaI8Op |
| CUTLASS C++ CuTe | 头文件可用, 需模板化 kernel (开发量大) |
| CUTLASS PR #3302 | 上游 open, 未合入 |

---

## 数值分析

### 误差来源分解

| 环节 | 误差量级 |
|------|---------|
| W4 → per-channel int8 re-encode | ~0.003% 相对 (16 个离散码点, 几乎无损) |
| 激活 FP16 → per-tensor int8 | ~1-3% 相对, 10 层累乘后 argmax 差 ~1-2pp |
| int32 → fp16 cast | 舍入误差, 可忽略 |

### 准确度优化

- **静态激活标定** 优于动态: 用校准数据预先固定 per-layer scale, 避免运行时 max 计算
- **99.99 分位标定** 优于 max: 裁剪离群值减少过饱和
- **混合精度**: 前几层保留 FP16 (误差尚未放大), 中间层转 INT8 (算力最大化)

---

## 离线预排板

W8A8 / `ptx_sm89` 热路径应对 int8 权重做 **offline prepack** (`int8:sm_89:b_nk`), 见 [mma-weight-prepack.md](mma-weight-prepack.md).
`Int8MmaLinear(engine="ptx_sm89")` 已在构造时自动 prepack B 为 `[N,K]`.

## 相关文件

| 文件 | 说明 |
|------|------|
| `xqt/quant/quantizers/w4_storage_int8_mma.py` | W4StorageInt8MmaLinear + quantize API |
| `xqt/quant/quantizers/int8_mma.py` | Int8MmaLinear (W8A8 INT8 MMA) |
| `xqt/quant/quantizers/fp4_weight_only.py` | FP4WeightOnlyLinear (W4 存储) |
| `xqt/operator_opt/kernels/tilelang/int8_mma.py` | TileLang INT8 MMA kernel |
| `xqt/operator_opt/kernels/cute/int8mma_kernel.cu` | Ada sm_89 手写 INT8 MMA 源码 |
| `xqt/operator_opt/kernels/cute/build/int8mma_sm89.so` | 编译产物 |
| `xqt/operator_opt/kernels/cute/int8mma_binding.py` | ctypes 绑定 (`engine=ptx_sm89`) |
| `xqt/operator_opt/kernels/prepack/` | 离线 MMA 预排板注册表 (int8 实现, 其它精度占位) |
| `docs/md/explanation/mma-weight-prepack.md` | 预排板设计说明 |
| `xqt/quant/capability.py` | 策略 → nature 映射 (TRUE/PSEUDO) |
| `xqt/core/schema.py` | strategy 别名注册 |
| `examples/mlp_w4_int8_mma_acceptance.py` | 10 层 MLP 验收脚本 |
| `tests/xqt/quant/test_w4_storage_int8_mma.py` | 单元测试 |
