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

## 不负责什么

- 不列 operator engine pattern 表 (见 [xqt-engines.md](xqt-engines.md)).
- 不展开推理三条路径细节 (见 [xqt-inference.md](xqt-inference.md)).
- 不把 method 写成 engine methods.

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

## 2. Quant backend (当前)

源码: `xqt/quant/capability.py` `list_quant_backend_capabilities()`.

| backend | status | 角色 |
| --- | --- | --- |
| `pytorch` | available | **主路径**: 内置 method (awq/gptq/svd) + 多种 strategy |
| `torchao` | available | 外部 torchao 适配 |
| `onnxruntime_qdq` | available | 静态 QDQ 图 |
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

## 3. Quant method (算法)

| method | 含义 | 实现 |
| --- | --- | --- |
| `awq` | activation-aware weight-only | `quantizers/awq.py`, `awq_gptq_weight_only.py` |
| `gptq` | Hessian-aware weight-only | `quantizers/gptq.py`, `awq_gptq_weight_only.py` |
| `svd` / `svdquant` | 低秩分支 + 量化残差 (SVDQuant) | `quantizers/svd.py` |
| (torchao methods) | 由 torchao backend 承载 | `backends/torchao.py` |
| (qdq static) | ONNX QDQ | `backends/onnx_qdq.py` |

配置:

```yaml
params:
  backend: pytorch
  method: awq          # 或 gptq | svd
  strategy: weight_only_int4
  policy: { bits: 4, group_size: 128 }
```

SVD:

```yaml
params:
  backend: pytorch
  method: svd
  strategy: svd_int4   # 或 svd_fp4
  policy: { rank: 32, group_size: 128, quant_dtype: int4 }
```

---

## 4. Strategy 枚举 (现状, 混轴)

源码: `xqt/core/schema.py` `CANONICAL_QUANT_STRATEGIES`.

同一列表混了 storage / compute / 算法痕迹, **不是** 目标三轴 API, 见 DEBT-002. 使用时心里按三轴拆:

| 更像 method 痕迹 | 更像 storage | 更像 compute |
| --- | --- | --- |
| (method 字段: awq/gptq/svd) | `weight_only_int4/8`, `fp4_weight_only`, `mxfp_weight_only` | `dynamic_int8_mma`, `tilelang_int8_mma`, `w4_storage_int8_mma` |
| | `static_qdq_int8` | `fp8_dynamic` |
| | `svd_fp4` / `svd_int4` (残差存储形态) | |

TRUE / PSEUDO nature: `xqt/quant/capability.py` `_STRATEGY_NATURE`.  
PSEUDO = 存储压缩, 计算前 dequant; TRUE = 期望原生低精度 MMA 路径.

---

## 5. 执行分发

源码: `xqt/quant/execution/executor.py`.

- 按 `backend` + `method` + `strategy` 路由到 quantizer / 外部 backend.
- `backend=tilelang` / `backend=svdquant` **直接失败**.
- 产出 `QuantizedModel` / report; 可选 `compute_config` (如 int8_mma 路径).

---

## 6. 与 engine / infer 的衔接

```text
quant stage
  -> model (packed weights / SVD modules / ...)
  -> optional compute_config (required_capabilities, precision)
operator stage (可选)
  -> engine resolve by capabilities  # tilelang/triton/...
runtime / package / export
  -> HybridInferenceEngine 或 模型包; 不跑 quantizer
```

- operator engine 能力: [xqt-engines.md](xqt-engines.md)
- 推理路径: [xqt-inference.md](xqt-inference.md)
- 交接 schema: [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md)

---

## 7. 常见误区

- 把 `tilelang` 当成 quant backend.
- 把 `svdquant` 当成 quant backend (应是 method).
- 在 engine 文档里找 "tilelang methods = awq, gptq".
- Infer 时再传 `method=awq` 当必选.
- 认为 strategy 字符串丛林已经是最终三轴 API.

---

## 8. 继续阅读

- [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md)
- [xqt-engines.md](xqt-engines.md)
- [xqt-inference.md](xqt-inference.md)
- [xqt-concepts.md](xqt-concepts.md)
- [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
