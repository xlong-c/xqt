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

### 2.3 写作检查

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
| (torchao methods) | 由 torchao backend 承载 | `backends/torchao.py` |
| (qdq static) | 离线静态权重 + 校准静态激活 (ONNX QDQ) | `backends/onnx_qdq.py` |

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
PSEUDO = 存储压缩, 计算前 dequant; TRUE = 期望原生低精度 MMA 路径.

名字里带 `dynamic` 的 strategy, 通常指 **激活或 runtime scale 动态**, 不表示权重是在线量化.

---

## 6. 执行分发

源码: `xqt/quant/execution/executor.py`.

- 按 `backend` + `method` + `strategy` 路由到 quantizer / 外部 backend.
- `backend=tilelang` / `backend=svdquant` **直接失败**.
- 产出 `QuantizedModel` / report; 可选 `compute_config` (如 int8_mma 路径).
- 权重量化结果写入 model / buffer; 动态激活 quant 发生在 runtime module forward, 不在 quant stage 写死激活 scale (QDQ 静态路径除外).

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

---

## 9. 继续阅读

- [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md)
- [xqt-engines.md](xqt-engines.md)
- [xqt-inference.md](xqt-inference.md)
- [xqt-concepts.md](xqt-concepts.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
