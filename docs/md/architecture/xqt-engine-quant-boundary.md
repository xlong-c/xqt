# XQT Engine / Quant 边界与术语规则

**用途**: 定义 `engine` 与 `quant` 的分工, 词表, 硬规则和配置写法.  
**角色**: 架构层事实源之一; 改 quant/engine API 或文档口径前先读本文.  
**不是**: 完整 capability 矩阵 (见说明层), 也不是 recipe 教程 (见 usage).

| 相关文档 | 角色 |
| --- | --- |
| 本文 | **边界与禁止项** (规则源) |
| [xqt-infer-handoff.md](xqt-infer-handoff.md) | quant → infer 交接面 (`model` + `compute_config`) |
| [xqt-design-debt.md](xqt-design-debt.md) | 未完成债 (DEBT-001/002 等) |
| [../explanation/xqt-engines.md](../explanation/xqt-engines.md) | operator engine 能力矩阵 (pattern / maturity) |
| [../explanation/xqt-quant.md](../explanation/xqt-quant.md) | quant 侧 backend / method / strategy 说明 |
| [../explanation/xqt-inference.md](../explanation/xqt-inference.md) | 推理路径 |
| [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md) | 包内工程契约摘要 |

---

## 1. 一句话

```text
Quant 决定: 用什么算法得到什么存储, 以及需要什么计算契约 (capabilities).
Engine 决定: 在满足 capabilities 的前提下, 用哪条 kernel / lowering 路径执行.
Infer 只消费: 已量化模型 + 可选计算配置; 不回流量化, 不强制 engine 名.
```

---

## 2. 词表 (必须分词)

| 词 | 含义 | 合法落点 | 禁止落点 |
| --- | --- | --- | --- |
| **quant method** | 量化算法 / 如何得到 scale 与 packed weight | `quant.params.method`; quant report; `xqt/quant/quantizers/` | operator engine methods; Infer 必选输入 |
| **storage** | 权重/激活存什么 | model buffers; 可选 `compute_config.modules[].storage` | 与某个 engine 绑死 |
| **compute / MMA contract** | 算什么 (计算契约) | `compute_config.compute_contract`; `required_capabilities` | 写成 quant method 同义词 |
| **operator engine** | 谁来 lowering / 跑 kernel | `operator` stage `engine=`; `xqt.convert` materialize preference; `operator_opt` | quant `backend` 名; AWQ/GPTQ/SVD 的 methods 列表 |
| **quant backend** | quant **适配路径** (外部库或 PyTorch 主路径) | `quant.params.backend` | 与 operator engine 同名混用 |
| **export / deploy backend** | 外部 runtime (ORT/TRT/...) | export/deploy targets; 模型包 `preferred_backend` | 与 triton/tilelang 并列成 "XQT engine" |
| **strategy** (现状) | 历史兼容枚举, 混有 storage/compute 痕迹 | recipe `strategy` (过渡) | 新能力的唯一主键 (应逐步拆到 method×storage×compute) |

### 2.1 合法 quant method 示例

`awq`, `gptq`, `svd` (别名 `svdquant`), 以及外部路径上的 method 名 (如 torchao 的 `dynamic_int8`).

### 2.2 合法 operator engine 示例

`triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `torch_compile`, `custom_cuda`, `deployment_engine`.

### 2.3 合法 quant backend 示例 (当前)

`pytorch`, `torchao`, `onnxruntime_qdq`, `bitsandbytes` (planned).

### 2.4 已删除 / 禁止的 quant backend 名

| 禁止名 | 原因 | 正确写法 |
| --- | --- | --- |
| `tilelang` | 是 operator **engine**, 不是 quant backend | quant: `backend=pytorch` + method/strategy; operator: `engine=tilelang` |
| `svdquant` | 是 quant **method**, 不是 quant backend | `backend=pytorch` + `method=svd` + `strategy=svd_fp4\|svd_int4\|svd_fp4_int8_mma\|svd_int4_int8_mma` |
| `awq` / `gptq` 作 backend | 是 method | `backend=pytorch` + `method=awq\|gptq` |

代码会拒绝上述错误 backend (见 `xqt/quant/capability.py`, `xqt/quant/execution/executor.py`).

---

## 3. 三轴模型 (设计与配置目标)

```text
1. Quant method     # awq | gptq | svd | minmax | torchao 路径 | onnx static qdq | ...
2. Storage          # w4 | w8 | fp4 | nvfp4 | mxfp | qdq graph | low-rank+residual | ...
3. Compute / MMA    # int8_mma | fp4_mma | int4_mma | w4_storage_int8_mma | mix_fp4_int8_mma | ...
   + Operator engine # triton | tilelang | cutlass | ...   (resolve, 非 quant 输出主键)
```

规则:

1. 多种 method 可产出 **同一 storage**.
2. 同一 storage 可适配 **多种 compute contract**.
3. 新 MMA 路径扩展 **contract 名** / `required_capabilities`, 不要再往 "backend.methods" 塞算法字符串.
4. recipe 现状仍用 `backend` + `method` + `strategy` 字段名; **语义** 必须按上表理解, 不得把 method 塞进 engine 表.

---

## 4. Quant 侧规则

### 4.1 负责

- 标定, 打包, 模块替换, quant report / lineage.
- 可选写出 `compute_config` (精度 + `required_capabilities` + 可选 `preferred_engines` hint).
- 算法身份 (`method`) 留在 quant 报告; **不是** Infer 构造必选.

### 4.2 不负责

- 不选择 operator engine 作为硬约束主键 (`required_engine` 禁止).
- 不实现 triton/tilelang kernel materialize (那是 operator / convert).
- 不做训练 / QAT / dataset / task eval.

### 4.3 推荐 recipe 形态

```yaml
stages:
  - name: quant_model
    kind: quant
    params:
      backend: pytorch          # quant 适配路径, 不是 engine
      method: awq               # 或 gptq | svd
      strategy: weight_only_int4  # 或 fp4_weight_only | svd_int4 | dynamic_int8_mma | ...
      policy: { ... }

  - name: op_opt
    kind: operator
    params:
      default_engine: tilelang  # operator engine
      # 或按 capability resolve; 见 engine_resolve
```

### 4.4 算法 → 实现位置

| method | 实现 | 默认 backend |
| --- | --- | --- |
| `awq` / `gptq` | `xqt/quant/quantizers/awq.py`, `gptq.py`, `awq_gptq_weight_only.py` | `pytorch` |
| `svd` | `xqt/quant/quantizers/svd.py` | `pytorch` |
| torchao 系列 | `xqt/quant/backends/torchao.py` | `torchao` |
| static QDQ | `xqt/quant/backends/onnx_qdq.py` | `onnxruntime_qdq` |

---

## 5. Operator Engine 侧规则

### 5.1 负责

- 算子 pattern 实现, 融合, materialize, MMA / dequant-gemm 等 **计算路径**.
- capability 矩阵: `list_operator_engine_capabilities()`.
- 按 `required_capabilities` (+ 可选 preferred hint) **resolve** engine (`xqt/runtime/engine_resolve.py`).

### 5.2 不负责

- 不实现 AWQ / GPTQ / SVD 标定算法.
- 不在 engine methods 表里挂 `awq`/`gptq`/`svd`.
- 不替代 TensorRT / ORT 等外部 deploy backend.

### 5.3 convert 与 engine

- `xqt.convert(..., engine=...)` 中 `engine` 是 **materialize preference**, 不是 quant/infer 交接主键.
- `engine=None` 默认 `"torch"` (最小 intent 路径).
- 权威 resolve 在 operator stage / runtime capability, 不在 quantizer 内绑死.

---

## 6. Infer 交接规则 (摘要)

完整 schema 见 [xqt-infer-handoff.md](xqt-infer-handoff.md).

```text
Quant 输出:  model + 可选 compute_config
Infer 输入:  同上
Infer 行为:  resolve(required_capabilities) -> execute
禁止:        再跑 quantizer; 必选 method/backend; 主键 required_engine
允许:        preferred_engines 仅作 hint
```

`QuantizedModel.infer_handoff()` 只返回 `model` + `compute_config`.  
`backend` / `method` / `strategy` 可保留在 quant lineage, 但不是 Infer 必选.

---

## 7. 禁止清单 (写代码 / 写文档时对照)

1. **禁止** `quant.params.backend=tilelang` 或 `=svdquant`.
2. **禁止** 在 operator engine 文档主表展示 "engine → awq/gptq methods".
3. **禁止** 把 `required_engine` 写成 compute_config 主键.
4. **禁止** 文档把 triton/tilelang 与 TensorRT/ORT 并列成外部 inference backend.
5. **禁止** quantizer 顶层强制 import 某 engine kernel 作为唯一路径 (默认 `auto` + lazy + resolve).
6. **禁止** 把 planned / reference_guarded 写成生产性能已验收.
7. **禁止** 新能力只往 `CANONICAL_QUANT_STRATEGIES` 塞混合字符串而不标明 method/storage/compute 语义.

---

## 8. 源码锚点

| 主题 | 路径 |
| --- | --- |
| quant backend 矩阵 | `xqt/quant/capability.py` |
| quant 执行分发 | `xqt/quant/execution/executor.py` |
| quant methods 实现 | `xqt/quant/quantizers/` |
| operator engine 矩阵 | `xqt/operator_opt/capability.py` |
| engine resolve | `xqt/runtime/engine_resolve.py` |
| compute_config | `xqt/contracts/compute.py` |
| Infer handoff | `xqt/contracts/quantized.py` `infer_handoff` |
| Hybrid runtime | `xqt/runtime/engine.py` |
| convert | `xqt/conversion.py` |
| strategy 枚举 (混轴现状) | `xqt/core/schema.py` `CANONICAL_QUANT_STRATEGIES` |

---

## 9. 文档地图 (谁写什么)

| 文档 | 写 | 不写 |
| --- | --- | --- |
| **本文** | 边界, 词表, 禁止项, 配置语义 | 完整 pattern 列表, 逐步 tutorial |
| `explanation/xqt-engines.md` | engine 矩阵, pattern, maturity | quant method 当 engine methods |
| `explanation/xqt-quant.md` | quant backend/method/strategy 现状与写法 | operator kernel 清单 |
| `explanation/xqt-inference.md` | 三条推理路径 + 交接面对照 | quant 算法细节 |
| `architecture/xqt-infer-handoff.md` | compute_config schema 与迁移 | engine pattern 表 |
| `FRAMEWORK.md` | 包内摘要契约 | 长矩阵 |

更新规则: 行为变化时 **先改代码与本文**, 再同步说明层与 `FRAMEWORK.md`.

---

## 10. 验收自检 (PR / 文档改动)

- [ ] 新词是否落在正确列 (method vs storage vs compute vs engine vs backend)?
- [ ] 是否引入了禁止的 quant backend 名?
- [ ] engine 文档是否避免 "methods=awq/gptq"?
- [ ] Infer 路径是否只依赖 model + compute_config / policy?
- [ ] 示例 YAML 是否 `backend=pytorch` + `method=...` 且 operator 用 `engine=...`?
