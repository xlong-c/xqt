# XQT 量化 (Quant) 能力说明

本文对照当前源码, 说明 **quant 侧** backend / method / strategy 怎么写, 实现落在哪, 以及和 operator engine 的边界.

**权威事实源是代码**. 边界硬规则以架构层为准:

- [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md) (词表 + 禁止项)
- [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md) (model + compute_config)

## 负责什么

- quant backend 名单与 capability.
- quant method (awq/gptq/svd/...) 与实现入口.
- strategy 枚举现状 (混轴) 与 TRUE/PSEUDO nature.
- 推荐配置写法与错误写法对照.
- 文档与配置叙述时的 **离线静态权重优先** 写法.

## 不负责什么

- 不列 operator engine pattern 表 (见 [xqt-engines.md](xqt-engines.md)).
- 不展开推理三条路径细节 (见 [xqt-inference.md](xqt-inference.md)).
- 不把 method 写成 engine methods.
- 不把"动态 / 静态"当成与 method 一一对应的唯一标签 (见下节拆轴).

---

## 1. 边界摘要

```text
Quant:  method → storage artifact (+ 可选 compute_config)
Engine: 只认 compute/capabilities, 做 kernel lowering
Infer:  model + compute_config; 不跑 quantizer
```

| 合法 | 非法 |
| --- | --- |
| `backend=pytorch`, `method=awq` | `backend=awq` |
| `backend=pytorch`, `method=svd`, `strategy=svd_int4` | `backend=svdquant` |
| operator stage `engine=tilelang` | `quant.params.backend=tilelang` |

---

## 2. 写量化时的默认表述: 离线静态权重优先

文档, 注释, report 文案和配置说明里, **先写权重侧, 再写激活侧**. 不要用单独一句"动态量化 / 静态量化"概括整条 recipe.

`W?A?` 不是一次量化阶段可独立证明的性能结论. 每次描述必须分开写:

1. 量化时的权重存储.
2. 激活是在量化阶段固化, 还是每次 forward 生成.
3. 运行时真实使用的 operands 和 compute path.
4. 会触发浮点或 reference 路径的条件. 没有该条件时也必须明确写为"未配置"或"不存在自动回退".

`QuantizationReport.nature` 和 result metadata 中的 `quantization_nature` 只分类**已请求的 compute contract 或当前实现路线**. 它们不是某次 forward 已使用原生 MMA 的证据. 需要逐层查看 `execution_metadata()` 中的 engine, operands, `native_mma_executed` 和 fallback 状态.

### 2.1 默认拆轴

| 轴 | 默认事实 (XQT 主路径) | 文档应优先写 |
| --- | --- | --- |
| **权重** | 离线量化并固化 (scale / packed codes / 残差 qweight) | **离线静态权重** |
| **激活** | 多数路径不量化, 或推理时按当前 tensor 现算 scale | **未量化** / **运行时动态激活** / (少数) **校准静态激活** |
| **算法 method** | `awq` / `gptq` / `svd` 等 | method 名 + 上两轴, 不把 method 说成"动态量化算法" |

推荐句式:

```text
离线静态权重量化 (weight-only INT4) + 激活保持 FP16/BF16
离线静态权重残差 (INT4/FP4) + 16-bit 低秩支路 + 运行时动态 4-bit 激活  # SVDQuant / Nunchaku 语义
离线静态权重 + 校准得到的静态激活 scale (QDQ 图)  # onnxruntime_qdq / static_qdq_int8
```

避免的句式:

```text
SVDQuant 是动态量化          # 权重是离线静态的; 动态的是激活
AWQ 是静态量化              # 更准确: 离线静态 weight-only; 激活通常不量化
整条路径叫 dynamic quant     # 除非明确只在讲激活 scale 的生成时机
```

### 2.2 和 strategy 名字的对应

| strategy / 路径 | 权重 | 激活 (文档优先写法) |
| --- | --- | --- |
| `weight_only_int4/8`, `fp4_weight_only`, `mxfp_weight_only` | 离线静态 | 未量化 (FP16/BF16 等) |
| `awq` / `gptq` + weight-only strategy | 离线静态 (校准可参与 scale/packing) | 未量化; 校准数据用于 **权重** 而非写死激活 scale |
| `svd_int4` / `svd_fp4` | 离线静态残差 + 静态低秩支路 | 原论文/Nunchaku: **运行时动态** 激活 quant; XQT 默认 `activation_scale_mode=dynamic` |
| `svd_int4_int8_mma` / `svd_fp4_int8_mma` | 同上 (W4 residual storage) | residual 走 W8A8 INT8 MMA retarget; 低秩支路保持源精度 |
| `dynamic_int8_mma`, `fp8_dynamic`, 动态 FP4 Linear | 离线静态 (或预打包) | **运行时动态** 激活 quant |
| `static_qdq_int8`, `onnxruntime_qdq` | 离线静态 | **校准静态** 激活 scale 进图 |

### 2.3 W/A 的量化时与运行时事实表

| 路径 | 量化时 | 运行时 | 条件和不得省略的说明 |
| --- | --- | --- | --- |
| `weight_only_int4/8`, `fp4_weight_only`, AWQ/GPTQ weight-only | W4/W8 离线静态打包 | A 保持输入浮点 dtype. 当前 XQT 路线是 dequant/reference floating-point compute | 这不是 W4A4 或 W8A8. A 从未承诺低比特, 因而不存在"A 退化"的条件 |
| `dynamic_int8_mma` | W8 离线静态. A 不写入 artifact | 每个 forward 将 A 编码为 INT8. `Int8MmaLinear.execution_metadata()["runtime_precision"]` 报告实际路径 | `quantize_with_int8_mma()` 创建 `min_int8_rows=0`, 因而**未配置**小 batch 浮点回退. 直接构造模块且 `min_int8_rows > 0` 时, `input_rows < min_int8_rows` 才会走浮点回退. engine 不可用时可走 INT8 reference, 不等于已使用原生 MMA |
| `w4_storage_int8_mma` | HBM/state_dict 保持 W4 + group scale | forward 将 W4 解码并重编码为 W8 compute view, A 同时编码为 INT8 | 是 W4 storage -> W8A8 compute retarget, 不是 native W4A4/FP4 MMA. 同样默认**未配置**小 batch 浮点回退; 实际 W8A8/native MMA 状态来自 delegated `runtime_precision` |
| `convrot_w4a4` / `method=convrot` + `strategy=w4a4_int4` | 旋转后 W4 离线静态打包. A 不写入 artifact | dynamic 模式每次 forward 在线旋转并量化 A4. 支持条件下可解析为 rowwise warp-FHT + CUTLASS W4A4,或 Nunchaku native W4A4;不支持时才把 A/W 解码到 `F.linear` reference | artifact/strategy nature 不能证明某次 forward 已执行 native MMA. 真实路径看 `execution_metadata()` 的 `resolved_w4a4_runtime_backend`,`native_w4a4_used`,`runtime_weight_layout` 和 fallback reason. rowwise 会把 grouped artifact 重量化为 whole-row scale,因此 `auto` 不会对 grouped artifact 静默启用 |
| `method=convrot` + `strategy=w8a8_int8` | 规则 Hadamard 分组旋转后 W8 离线静态 per-row INT8. A 不写入 artifact | 每个 forward 在线旋转 A; dynamic 模式在线量化. static 模式可由 calibration_inputs 生成 scale 并走 TileLang static quant, 在 `sm_89` + FP16 + `K % 32 == 0` + `N % 8 == 0` + `M >= 32` 时自动接 CUTLASS `cuda_sm89` true W8A8 GEMM. 热路径将数学 `[K,N]` 权重离线预打包并缓存为 `[N,K]`, 避免每次 GEMM 转置. 设置 `fuse_norm=true` 且模型存在显式 `Sequential(norm, linear)` 邻接时,静态 CUDA 路径用一个 Triton kernel 完成 Norm + Hadamard + INT8 activation quant,再直接进入 INT8 GEMM;真实状态看 `execution_metadata()[\"norm_fused\"]`. 其他形状或设备保留 Triton/TileLang/`torch_int_mm`/`ptx_sm89` 后备 | Comfy 生态主路径对应物. 模块 buffer 含 stock 形 `comfy_quant` marker (`format=int8_tensorwise`). native MMA 只看 per-forward metadata; Norm 融合只对一维 RMSNorm/LayerNorm 生效 |
| `w4a4_nvfp4`, `w4a4_mxfp4` | W4 FP4 离线打包. A 不写入 artifact | 每个 forward 打包 A4 FP4 并消费 packed W4 | 当前实现分类为 PSEUDO, 不宣称 native FP4 MMA. `FP4DynamicLinear.execution_metadata()` 会记录 engine candidates 和实际 taken fallback |
| `static_qdq_int8` / `onnxruntime_qdq` | W/A scale 由校准写入 QDQ graph | provider 决定 Q/DQ 是否融合或 lower 到整数 kernel | QDQ graph 只说明图契约, 不证明某 provider 已跑原生 W8A8. 需要 provider profiling 或 runtime report |
| torchao `w8a8_*` / FP8 dynamic | XQT 将 W/A config 交给 torchao | torchao 和 PyTorch 决定实际 module 和 kernel | XQT 没有该外部模块的逐 forward execution metadata. 只能描述已配置 W/A route, 不得写成已验证的 native MMA 或 speedup |

### 2.4 写作检查

- 先写"离线静态权重 ...", 再补激活.
- 提到校准数据时, 写清校准服务 **权重** 还是 **激活 scale**, 还是两者都有.
- 引用 SVDQuant 时对齐原项目: PTQ 离线权重 + 推理时 `quantize(x)` 出 `ascales` (动态激活), 不是整图静态激活 QDQ.

---

## 3. Quant backend (当前)

源码: `xqt/quant/capability.py` `list_quant_backend_capabilities()`.

| backend | status | 角色 |
| --- | --- | --- |
| `pytorch` | available | **主路径**: 内置 method (awq/gptq/svd) + 多种 strategy |
| `torchao` | available | 外部 torchao 适配 |
| `onnxruntime_qdq` | available | 静态 QDQ 图 (离线静态权重 + 校准静态激活) |
| `bitsandbytes` | planned | 未接入执行 |

**已删除的 quant backend 名**: `tilelang`, `svdquant`.  
访问时 capability / plan 会 `ValueError` 并提示正确写法.

刷新:

```python
from xqt.quant.capability import list_quant_backend_capabilities, describe_quant_backend_capability

print(sorted(list_quant_backend_capabilities()))
print(describe_quant_backend_capability("pytorch", method="awq", strategy="weight_only_int4").to_dict())
```

---

## 4. Quant method (算法)

| method | 含义 (优先写权重) | 实现 |
| --- | --- | --- |
| `awq` | 离线静态 weight-only (activation-aware 校准服务权重) | `quantizers/awq.py`, `awq_gptq_weight_only.py` |
| `gptq` | 离线静态 weight-only (Hessian-aware) | `quantizers/gptq.py`, `awq_gptq_weight_only.py` |
| `svd` / `svdquant` | 离线静态权重残差 + 16-bit 低秩; 激活默认运行时动态 | `quantizers/svd.py` |
| `convrot` | 离线规则 Hadamard 分组旋转 + 静态权重量化; 激活在线旋转 | `quantizers/convrot_4bit.py` (storage 默认 reference;经 `ConvRotW4A4ExecutionView.from_storage()` 才尝试 rowwise 或 Nunchaku native W4A4), `quantizers/convrot_int8.py` (storage 默认 reference;经 `ConvRotInt8ExecutionView.from_storage()` 才尝试 W8A8 INT8 MMA,当前 runtime activation scale 为 per-tensor) |
| (torchao methods) | 由 torchao backend 承载 | `backends/torchao.py` |
| (qdq static) | 离线静态权重 + 校准静态激活 (ONNX QDQ) | `backends/onnx_qdq.py` |

ConvRot 配置 (W8A8, 对齐 Comfy INT8-Fast / stock `int8_tensorwise`):

```yaml
params:
  backend: pytorch
  method: convrot
  strategy: w8a8_int8
  compute: w8a8_int8_mma
  policy:
    dtype: int8
    scheme: convrot_w8a8
    rot_size: 256          # 必须是 4 的幂; 推荐 256
    activation_scale_mode: dynamic
```

Comfy marker 编解码: `xqt.quant.comfy_quant` (`encode_int8_tensorwise_marker` / `decode_comfy_quant_marker`).
调研笔记: `research/convrot-comfy/README.md`.

### ConvRot 的旋转块和形状契约

ConvRot 的 `rot_size` 是实际使用的规则 Hadamard 分组大小 `N0`, 必须为 `1` 或 `4^k` (例如 `16`, `64`, `256`, `1024`). 量化器不会因为某个 Linear 的输入特征更窄就把请求的 `N0` 静默降级; 逻辑输入维不足或不能整除时, 模块内部向上 padding.

- W4A4 的内部输入维对齐到 `lcm(rot_size, group_size)` 的倍数.
- W8A8 的内部输入维对齐到 `lcm(rot_size, 64)` 的倍数, 其中 `64` 是当前 INT8 MMA 的 K 对齐.
- `logical_input_features` 和 `padded_input_features` 会写入 result/module metadata. forward 输出始终回到逻辑输出形状, padding 对调用方透明.
- batch 维和 sequence/token 维只会折叠进旋转的 batch 侧, 没有长度整除限制. W4A4 dynamic 激活 scale 沿最后一维归约, 每个 token 独立生成一个 scale; static 模式才使用校准得到的 per-tensor scale.

`ConvRotMixedPrecisionLinear(compute_precision="w4a4")` 的 artifact/strategy 分类和单次 runtime execution 必须分开看. `from_linear()` 和 quantizer 默认只创建 reference storage; 通过 `ConvRotW4A4ExecutionView.from_storage()` 显式物化后,支持条件下才会尝试两类 native W4A4:rowwise warp-FHT + CUTLASS `s4 x s4 -> s32` 路径,或 Nunchaku grouped W4A4 路径;两者都不可用时回到 dequantized `F.linear` reference. `w4a4_runtime_backend=auto|rowwise|nunchaku|reference` 控制解析语义,真实状态以 `execution_metadata()` 的 `artifact_view`,`resolved_w4a4_runtime_backend`,`native_w4a4_used`,`runtime_weight_layout`,`native_w4a4_fallback_reason` 和 `norm_fused` 为准. SM89 rowwise 实现和性能证据见 [convrot-w4a4-sm89-optimization.md](convrot-w4a4-sm89-optimization.md).

配置 (默认先 weight-only / 离线静态权重):

```yaml
params:
  backend: pytorch
  method: awq          # 或 gptq | svd
  strategy: weight_only_int4
  policy: { bits: 4, group_size: 128 }
```

SVD (权重离线静态; 激活侧见第 2 节, 默认动态):

```yaml
params:
  backend: pytorch
  method: svd
  strategy: svd_int4   # 或 svd_fp4 / svd_int4_int8_mma / svd_fp4_int8_mma
  policy: { rank: 32, group_size: 128, quant_dtype: int4 }
```

---

## 5. Strategy 枚举 (现状, 混轴)

源码: `xqt/core/schema.py` `CANONICAL_QUANT_STRATEGIES`.

同一列表混了 storage / compute / 算法痕迹, **不是** 目标三轴 API, 见 DEBT-002. 使用时心里按三轴拆; 叙述时仍 **优先离线静态权重**, 再标激活:

| 更像 method 痕迹 | 更像 storage (权重形态) | 更像 compute (常含激活时机) |
| --- | --- | --- |
| (method 字段: awq/gptq/svd) | `weight_only_int4/8`, `fp4_weight_only`, `mxfp_weight_only` | `dynamic_int8_mma`, `tilelang_int8_mma`, `w4_storage_int8_mma` |
| | `static_qdq_int8` | `fp8_dynamic` |
| | `svd_fp4` / `svd_int4` (残差存储形态) | |
| | | `svd_fp4_int8_mma` / `svd_int4_int8_mma` (W4 residual + W8A8 MMA) |

TRUE / PSEUDO nature: `xqt/quant/capability.py` `_STRATEGY_NATURE`.  
TRUE = 已请求原生低精度 MMA contract; PSEUDO = 当前 XQT 路线是 dequant/reference floating-point compute; UNKNOWN = strategy/storage 单独不足以判定运行时 compute. 三者都不是单次 forward 的执行证明.

名字里带 `dynamic` 的 strategy, 通常指 **激活或 runtime scale 动态**, 不表示权重是在线量化.

---

## 6. 执行分发

源码: `xqt/quant/execution/executor.py`.

- 按 `backend` + `method` + `strategy` 路由到 quantizer / 外部 backend.
- `backend=tilelang` / `backend=svdquant` **直接失败**.
- 产出 `QuantizedModel` / report; 可选 `compute_config` (如 int8_mma 路径).
- 权重量化结果写入 model / buffer; 动态激活 quant 发生在 runtime module forward, 不在 quant stage 写死激活 scale (QDQ 静态路径除外).
- 量化 report 必须同时保留量化时 `precision_description` 和运行时观察入口. `Int8MmaLinear` 的 `runtime_precision` 会显式报告 `native_mma_executed`, `int8_operands_executed`, `float_fallback_taken` 及小 batch 回退状态.

---

## 7. 与 engine / infer 的衔接

```text
quant stage
  -> model (packed weights / SVD modules / ...)   # 离线静态权重 artifact
  -> optional compute_config (required_capabilities, precision)
operator stage (可选)
  -> engine resolve by capabilities  # tilelang/triton/...
runtime / package / export
  -> HybridInferenceEngine 或 模型包; 不跑 quantizer
  -> 若路径含动态激活: forward 内 quantize(x) / ascales
```

- operator engine 能力: [xqt-engines.md](xqt-engines.md)
- 推理路径: [xqt-inference.md](xqt-inference.md)
- 交接 schema: [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md)

---

## 8. 常见误区

- 把 `tilelang` 当成 quant backend.
- 把 `svdquant` 当成 quant backend (应是 method).
- 在 engine 文档里找 "tilelang methods = awq, gptq".
- Infer 时再传 `method=awq` 当必选.
- 认为 strategy 字符串丛林已经是最终三轴 API.
- 用一句"动态量化"概括 SVDQuant / dynamic MMA (应先写 **离线静态权重**, 再写运行时动态激活).
- 把 AWQ/GPTQ 的校准数据说成"静态激活 scale 已进图" (主路径是 weight-only; 校准服务权重).
- 把 `nature=TRUE` 或 `compute_contract=w8a8_int8_mma` 写成"运行时已使用 native MMA". 必须先看 per-forward metadata.
- 把 `w4a4` 名称或 strategy nature 写成当前必然执行 W4A4 MMA. ConvRot 必须报告 rowwise/Nunchaku/reference 的实际解析和 fallback;动态 FP4 等其他路径也必须以逐 forward 执行证据为准.

---

## 9. 继续阅读

- [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md)
- [xqt-engines.md](xqt-engines.md)
- [xqt-inference.md](xqt-inference.md)
- [xqt-concepts.md](xqt-concepts.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
