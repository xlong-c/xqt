# XQT Infer 交接面方案 (model + compute_config)

**状态**: planned → 与 DEBT-001/002/003 同批落地  
**用途**: 统一 quant→infer 交接契约; 代码与测试以本页 + 源码为准.  
**不是**: recipe 全量重写说明, 也不替代 [xqt-design-debt.md](xqt-design-debt.md) 台账.
词表与禁止项: [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md).

相关:

| 文档 | 角色 |
| --- | --- |
| 本文 | Infer 交接面 schema 与迁移裁决 |
| [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md) | engine / quant 词表与禁止项 |
| [xqt-design-debt.md](xqt-design-debt.md) | DEBT-001/002/003 现象与锚点 |
| [../explanation/xqt-inference.md](../explanation/xqt-inference.md) | 推理能力说明 (目标 + 现状) |
| [../explanation/xqt-engines.md](../explanation/xqt-engines.md) | operator engine 能力矩阵 |
| [../explanation/xqt-quant.md](../explanation/xqt-quant.md) | quant backend / method / strategy |
| [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md) | 包内工程契约 |

---

## 1. 目标契约 (拍板)

```text
[Quant]
  输入: 浮点/训练后模型 + 算法/校准配置
  输出:
    1. model          # 网络与量化存储在 module / graph 内
    2. compute_config # 可选; 算子级 compute contract + required_capabilities
  禁止: 把 required_engine / 强制 engine 名写成推理必选主键
  允许: preferred_engines 仅作 hint; quant 报告里保留 method/backend 血统

[Infer]
  输入: model + 可选 compute_config (或等价 ExecutionPolicy / 包内 sidecar)
  行为:
    resolve(required_capabilities [, preferred_engines hint]) -> engine candidate
    不满足则 fallback / 报错 (policy), 不回流量化
  禁止: 再跑 quantizer / calibration / sensitivity
```

一句话: **推理侧执行已量化模型 + 满足计算契约; 不是再选一次 quant backend/method.**

---

## 2. 主键与禁止项

| 字段角色 | 合法 | 非法 (作为推理主键) |
| --- | --- | --- |
| `model` | 是 | - |
| `compute_config.modules[*].compute_contract` | 是 (如 `int8_mma`) | - |
| `compute_config.modules[*].required_capabilities` | 是 | - |
| `compute_config.modules[*].precision` | 是 (w8a8/...) | - |
| `preferred_engines` | 仅 hint | 硬失败唯一条件 |
| `required_engine` | **禁止作为 schema 主字段** | 不得写入 compute_config 主键 |
| `backend` / `method` / `strategy` | quant 报告 / lineage | 不得作为 Infer 构造必选 |

---

## 3. Schema (JSON 友好)

### 3.1 `ComputeConfig` (`schema_version: "1.0"`)

```json
{
  "schema_version": "1.0",
  "default_precision": "w8a8",
  "modules": [
    {
      "name": "proj",
      "compute_contract": "int8_mma",
      "precision": "w8a8",
      "required_capabilities": ["int8_mma", "dequant_gemm_epilogue"],
      "storage": {"format": "int8_per_out_channel", "layout": "qweight_t"},
      "preferred_engines": ["tilelang", "ptx_sm89"]
    }
  ],
  "metadata": {}
}
```

规则:

1. `required_capabilities` 可为空列表, 表示只声明 precision / contract, 由默认路径执行.
2. 不得出现顶层或 module 级 **`required_engine`** 主键; 若历史 JSON 误写, loader 忽略并记 warning metadata, 不作为硬约束.
3. `preferred_engines` 只影响 resolve 排序, 不具备 capability 的 engine 仍跳过.
4. `storage` 可选; 权重已在 model buffer 内时可省略.

### 3.2 代码类型 (源码)

- `xqt.contracts.compute.ModuleComputeSpec`
- `xqt.contracts.compute.ComputeConfig`
- `normalize_compute_contract()` / `compute_config_from_mapping()` / `compute_config_to_dict()`

### 3.3 落点

| 载体 | 路径 / 字段 |
| --- | --- |
| 内存 quant 结果 | `QuantizedModel.compute_config` |
| Hybrid policy | `ExecutionPolicyPayload.required_capabilities` + 可选 `compute_config` |
| Operator plan | `RuntimePlanPayload.required_capabilities`; `engine` 为 materialize **结果**, 可 `unresolved` |
| 模型包 | `runtime/compute.json` (可选 entrypoint); `runtime/config.json` 仍可含 ORT providers |
| Module 内嵌 | `_xqt_module_contract` 继续承载 module contract; compute_config 可从 modules 投影 |

---

## 4. 三轴 (DEBT-002) 与交接面关系

```text
1. Quant method     # awq / gptq / svd / minmax / ...  → 只进 quant report
2. Storage          # w4 / w8 / fp4 / ...              → model buffers (+ 可选 storage 字段)
3. Compute / MMA    # int8_mma / fp4_mma / ...         → compute_config.compute_contract
   + Operator engine # triton / tilelang / ...         → resolve 结果, 非 quant 输出主键
```

本交接面 **只消费轴 2+3 的执行视图**. 轴 1 不得成为 Infer 输入主键.

quant capability 表 (`backend → methods` 含 awq/gptq) 仍是债; 已删除 quant backend 名 `tilelang` / `svdquant` (awq/gptq/svd 只在 pytorch method). recipe 字段名 `backend`+`strategy` 仍保留; 文档禁止把 method 写成 engine methods.

---

## 5. 裁决 DEBT-001/003 的 API 问题

### 5.1 `xqt.convert(..., engine=...)` (DEBT-001)

| 裁决 | 说明 |
| --- | --- |
| convert 职责 | **语义 / precision / facade contract**; 可选 **eager materialize** |
| engine 参数 | 保留为 **materialize preference** (hint), 文档不再写成 quant/infer 交接主键 |
| 默认 | `engine=None` 时默认 `"torch"` (只记 intent / 最小 materialize), 不强制 triton/tilelang |
| 唯一权威 resolve | operator stage / runtime capability resolve; convert 不替代 |

### 5.2 `QuantizedModel.backend/method/strategy` (G2)

| 裁决 | 说明 |
| --- | --- |
| 保留字段 | 兼容 quant 报告与 stage metrics |
| Infer 可见性 | `infer_handoff()` / 文档约定: 推理只取 `model` + `compute_config` |
| 新字段 | `compute_config: ComputeConfig | None` |

### 5.3 `RuntimePlanPayload.engine` (G4)

| 裁决 | 说明 |
| --- | --- |
| `required_capabilities` | 一等字段 |
| `engine` | materialize 后的 **结果** (或 `"unresolved"`); 构造时可缺省 |
| `preferred_engines` | 可选 list, 仅 hint |

### 5.4 quantizer 内 engine (G1)

| 裁决 | 说明 |
| --- | --- |
| 默认 | `engine="auto"` |
| 选择逻辑 | 抽到 `xqt.runtime.engine_resolve` (capability + hint) |
| kernel import | forward / resolve 路径 lazy import, quantizer 顶层不绑死 tilelang/cute |
| quant 输出 | metadata 可记 `preferred_engines` / 实际 resolve 统计; 不写 `required_engine` |

### 5.5 export lowering (G6)

| 裁决 | 说明 |
| --- | --- |
| 识别 | duck type: `dequantize_weight` + packed storage 属性, 不硬依赖 quantizer 类名 import 作唯一条件 |
| 过渡 | 仍可 isinstance 兼容, 但主路径认 storage protocol |

### 5.6 模型包 (G7)

| 裁决 | 说明 |
| --- | --- |
| `preferred_backend` | ORT/TRT 等 **文件 runtime 名** 保留 (外部 backend, 非 operator engine) |
| `compute_config` | 可选写入 `runtime/compute.json` + entrypoints |
| 不解析 | quant recipe YAML |

---

## 6. Resolve 伪代码

```text
def resolve_engine(required_capabilities, preferred_engines=None, device=...):
    candidates = engines_that_provide(required_capabilities)
    if preferred_engines:
        candidates = stable_sort_by_preference(candidates, preferred_engines)
    for engine in candidates:
        if probe_available(engine, device):
            return engine
    raise / fallback_policy
```

`auto` = 空 preferred + 默认优先级 (例 int8_mma: ptx_sm89 → tilelang → torch_int_mm).

---

## 7. 迁移与测试范围

1. contracts: `ComputeConfig`, payload 字段, `QuantizedModel.compute_config` / `infer_handoff`.
2. runtime: `engine_resolve`, policy/package 读写 compute_config, Hybrid 可读 compute_config.
3. quant: `int8_mma` 默认 auto + lazy kernel + resolve 调用.
4. convert: 文档 + `engine` 可选默认 torch; 不删参数.
5. export: lowering duck type.
6. tests: contract 序列化, package compute.json, int8 resolve 不顶层 import kernels, convert 无 engine 默认.
7. 文档: design-debt 标 planned/done, inference/engines/FRAMEWORK 同步.

**本批明确不做**:

- 不改公开 recipe `backend`+`strategy` 语义 (DEBT-002 全量 capability 表拆分可后续).
- 不删除 awq/gptq quantizer.
- 不引入训练/QAT/dataset.

---

## 8. 验收清单

- [ ] Infer 构造路径可不传 backend/method/strategy
- [ ] compute_config 无 `required_engine` 主键; 有 `required_capabilities`
- [ ] int8_mma quantizer 模块 import 不强制加载 tilelang/cute
- [ ] RuntimePlan / ExecutionPolicy 可序列化 capabilities
- [ ] 模型包可选 `runtime/compute.json`
- [ ] 文档口径与代码一致, DEBT-003 主线 done 或 planned 带落地范围
