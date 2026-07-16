# XQT 设计债台账

**用途**: 累计 XQT 设计 / API / 术语问题, **后面统一制定修改方案再动**.  
**不是**: 完成报告, 执行清单, 或新契约事实源.

相关已有文档分工:

| 文档 | 角色 |
| --- | --- |
| 本文 `xqt-design-debt.md` | 只记账: open 问题, 锚点, 裁决问题 |
| [xqt-realignment-guide.md](xqt-realignment-guide.md) | 长期矫正准则与现状落地状态 |
| [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md) | engine / quant 词表与禁止项 (规则源) |
| [xqt-infer-handoff.md](xqt-infer-handoff.md) | quant→infer 交接面 |
| [xqt.md](xqt.md) / [FRAMEWORK.md](../../../xqt/FRAMEWORK.md) | 已实现事实 |

## 使用规则

1. 发现问题先加本文件, 编号 `DEBT-XXX`, 状态 `open`.
2. **未拍板前**: 不改公开 API, 不零散重构, 不把 planned 写成已实现.
3. 方案成熟后: 从本文件迁到具体 todo / plan, 本条标 `planned` 或 `moved`.
4. 代码落地后: 标 `done`, 写清落地 PR / 提交范围, 或整段归档删除.
5. 每条最少包含:
   - 现象 (用户可见 / API 可见)
   - 源码锚点
   - 为什么不合理
   - 后续方案必须回答的问题
   - 本阶段明确不做

## 状态词

- `open`: 已记录, 等待统一方案
- `planned`: 已写出修改方案 / 迁到执行清单
- `done`: 已落地
- `wontfix`: 明确不改, 需写理由

## 索引

| ID | 标题 | 状态 | 优先级 |
| --- | --- | --- | --- |
| [DEBT-001](#debt-001-xqtconvert-engine-把推理-engine-绑在-convert-参数上) | `xqt.convert(..., engine=...)` 绑定推理 engine | planned | high |
| [DEBT-002](#debt-002-quant-capability-把算法方法与-mma-计算契约-engine-缠在一起) | quant capability 把 AWQ/GPTQ/SVD 与 MMA/engine 缠在一起 | planned | high |
| [DEBT-003](#debt-003-量化与推理未严格解耦-推理应只消费模型--计算配置) | 量化与推理未严格解耦; 推理应只消费模型 + 计算配置 | done | high |
| [DEBT-004](#debt-004-gemm-selector-fp4nvfp4-goal-感知尚未落地) | gemm selector `goal` 对 fp4/nvfp4 尚无差异化 | open | medium |
| [DEBT-005](#debt-005-svdquant-应走-composite-add-混合精度而非特例-runtime) | SVDQuant 应走 composite_add 混合精度而非特例 runtime | planned | high |

---

## DEBT-001: `xqt.convert(..., engine=...)` 把推理 engine 绑在 convert 参数上

**状态**: planned  
**提出**: 2026-07-12  
**优先级**: high (术语与 API 边界)  
**方案**: [xqt-infer-handoff.md](xqt-infer-handoff.md) §5.1 - `engine` 降级为 materialize preference; 默认 `torch`; 不删参数.

### 现象

文档与阅读页写 engine 是 XQT 内部实现:

- 集合示例: `torch_compile`, `deployment_engine`, `triton`, `tilelang`, `cutile`, `cutlass`, `cute_dsl`, `custom_cuda`
- 出现在 `xqt.convert(..., engine=...)`, operator targets, capability / StageReport
- 不是与 TensorRT 并列的外部推理后端

质疑: **engine 本质是推理侧 lowering / 执行选择, 为什么作为 convert 参数?**  
convert 更像语义 / 精度 / contract 变换; 选 kernel 实现更像 operator 或 runtime 的事.

### 源码锚点

| 位置 | 现状 |
| --- | --- |
| `xqt/conversion.py` | docstring: `Module conversion facade for operator-oriented XQT engines`. 公开 `convert(..., engine=..., target=..., policy=..., fallback=...)`. `EngineKind` 仅 `torch / triton / tilelang / cutile / cute_dsl`. |
| `xqt/conversion_impl/converter.py` | convert 时建 `OperatorContract`, 再按 module kind 做 semantic + engine materialize. |
| `xqt/nn/*` | facade 构造也带 `engine` runtime intent. |
| `xqt/operator_opt/*` | operator stage 另有 engine / capability / materialize. |
| `xqt/runtime/*` | `HybridInferenceEngine` 走 execution policy, 不经 convert 选 engine. |
| `xqt/core/schema.py` | `OPERATOR_OPT_ENGINES` 与 convert 的 `EngineKind` 集合不完全同构. |

### 为什么不合理

1. **生命周期混淆**: convert ≈ 语义/contract; engine ≈ 运行时 kernel 路径.
2. **多入口选 engine**: convert, nn facade, operator stage, runtime, 无唯一明显做法.
3. **词表不一致**: convert `EngineKind` vs `OPERATOR_OPT_ENGINES` vs 文档列举.
4. **与 backend/engine 术语准则张力**: 已要求分词, 但 convert 仍把 engine 抬成转换 API 主旋钮.

### 一种可能的现状解释 (非定论)

实现上 convert 可能是 **eager materialize** 捷径: 写 contract 的同时立刻 bind kernel.  
即便如此, 是否应把 engine 放在 convert 顶层参数, 仍要统一裁决.

### 后续统一方案必须回答

1. convert 的 canonical 职责?
   - A. 只写 contract / precision / facade, 不 bind engine
   - B. 允许 bind engine, 但明确是 eager materialize, 与 operator stage 对齐
   - C. 拆成 `to_facade` / `materialize(engine=...)` 或改名
2. engine 的唯一权威选择点?
   - operator stage / `operator_opt.execute`
   - runtime policy
   - nn facade
   - convert
3. 三套 engine 词表如何合一?
4. 文档 / HTML 口径何时改 (方案前是否先弱化误导表述)?
5. 迁移策略: 兼容窗口, 默认是否只记 intent, 相关 tests 范围?

### 本阶段落地范围 (planned 部分完成)

- `engine=None` 默认 `"torch"`; docstring 标明 preference 非交接主键
- 完整拆 `to_facade` / `materialize` 仍可后续

### 相关

- [xqt-infer-handoff.md](xqt-infer-handoff.md)
- [xqt-realignment-guide.md](xqt-realignment-guide.md) 术语准则; convert/nn/operator 合流仍待做
- [xqt.md](xqt.md), [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- [../../html/xqt.html](../../html/xqt.html) (阅读层, 仍反映现状 API)

---

## DEBT-002: quant capability 把算法方法与 MMA 计算契约 / engine 缠在一起

**状态**: planned  
**提出**: 2026-07-12  
**优先级**: high (术语, 能力模型, 与 operator 边界)  
**方案**: [xqt-infer-handoff.md](xqt-infer-handoff.md) §4 - 三轴模型; Infer 只消费 compute/storage; recipe backend+strategy 暂不改.

### 现象

阅读页 / `xqt-engines.md` 中出现一张 **quant backend → methods** 表, 例如:

| backend | methods |
| --- | --- |
| `torchao` | `dynamic_int8`, `fp8_dynamic`, `weight_only_int4`, ... |
| `pytorch` | `awq`, `gptq`, `dynamic_int8_mma`, `tilelang_int8_mma`, `w4_storage_int8_mma` |
| `tilelang` | `awq`, `gptq` |
| `svdquant` | `svd_fp4`, `svd_int4` |

用户质疑 (正确方向):

1. **Operator engine 只应管**: 算子实现, 算子优化, 融合, 以及 **MMA / 计算契约** (如 `int8_mma`, `fp4_mma`, `int4_mma`, 再扩展 `mix_fp4_int8_mma` 等适配).
2. **AWQ / GPTQ / SVDQuant** 是 **量化算法 / 标定与打包方法**, 不是 engine, 也不该作为 engine 的 methods 绑死.
3. 把 `tilelang` 同时做成 quant backend 且 methods=`awq,gptq`, 会让人以为 TileLang engine = AWQ/GPTQ, 语义错误.
4. 目标轴应是 **compute contract (MMA 族)** 可组合扩展, 而不是 backend 枚举绑死算法名.

### 源码锚点 (当前事实)

| 位置 | 现状 |
| --- | --- |
| `xqt/quant/capability.py` `_BASE_CAPABILITIES` | quant **backend** 名含 `torchao`, `onnxruntime_qdq`, `pytorch`, **`tilelang`**, **`svdquant`**, `bitsandbytes`; 每项挂 `methods` 元组, 混入 `awq`/`gptq` 与 `*_mma` strategy 名. |
| `xqt/quant/capability.py` `_STRATEGY_NATURE` | strategy → TRUE/PSEUDO (存储 vs 原生 MMA 意图); 这是更接近 "计算契约" 的轴, 但未成为公开主模型. |
| `xqt/core/schema.py` `CANONICAL_QUANT_STRATEGIES` | 同时包含 `weight_only_*`, `static_qdq_int8`, `svd_*`, `*_mma`, `convrot_w4a4` - **算法 / 存储 / 计算** 挤在同一 strategy 枚举. |
| `xqt/quant/quantizers/` | 算法实现按文件拆: `awq.py`, `gptq.py`, `svd.py`, `int8_mma.py`, `w4_storage_int8_mma.py`, `fp4_weight_only.py` ... |
| `xqt/operator_opt/capability.py` | **真正的 operator engine** 矩阵 (triton/tilelang/cutlass/...), **不** 列 awq/gptq. |
| `xqt/operator_opt/backends/tilelang.py` | kernel patterns: attention/linear/dequant_gemm/... **不是** awq/gptq. |

### 三轴应拆开 (目标模型草案, 未拍板)

```text
1. Quant method (算法 / 如何得到 scale 与 packed weight)
   例: awq, gptq, svdquant, rtq/minmax, torchao_path, onnx_qdq_static, ...

2. Storage / precision contract (存什么)
   例: w4, w8, fp4, nvfp4, mxfp, qdq_int8 graph, ...

3. Compute / MMA contract (算什么)  +  Operator engine (用谁 lowering)
   例: int8_mma, fp4_mma, int4_mma, fp16_mma, mix_fp4_int8_mma, w4_storage_int8_mma, ...
   engine: triton, tilelang, cutlass, cute_dsl, cutile, torch_compile, ...
```

AWQ/GPTQ 只出现在轴 1; SVD 分解是轴 1 (+ 可选 low-rank 存储形态);  
`int8_mma` / `fp4_mma` 出现在轴 3, 可被多种 method 产出的 artifact **适配**, 而不是写死在某个 backend.methods 列表里.

### 为什么当前不合理

1. **engine 文档误读**: 读者在 "engines" 文里看到 backend→awq/gptq 表, 以为 engine 强绑量化方法.
2. **同名冲突**: `tilelang` 既是 operator engine 又是 quant backend, 职责重叠.
3. **扩展性差**: 新 MMA 路径 (`mix_fp4_int8_mma`) 应挂 compute contract, 却被迫塞进某个 backend 的 methods 或 strategy 字符串丛林.
4. **与 FRAMEWORK 术语准则冲突**: method / strategy / engine / backend 已要求分词, 但 capability 表仍把 method 塞进 backend 字段.

### 后续统一方案必须回答

1. quant capability 的公开主键是什么?
   - A. 保留 backend, 但 methods 只列 **存储/计算契约**, 算法另表
   - B. 拆成 `QuantMethodCapability` × `StorageContract` × `ComputeContract` 三张矩阵
   - C. strategy 枚举降级为兼容别名, 新 API 只认三轴
2. `tilelang` / `svdquant` 作为 quant **backend** 名是否删除或改名?
3. `awq`/`gptq` 是否只作为 method, 其产出统一落到 `fp4_weight_only` / `weight_only_int4` 等 storage + 可选 `*_mma` compute?
4. operator engine 文档是否 **禁止** 再展示 quant backend→methods 主表, 只交叉引用 quant 专章?
5. 迁移: recipe 里 `backend`+`strategy` 字段如何兼容一版?

### 本阶段落地范围

- 新增 `ComputeConfig` / `compute_contract` / `required_capabilities` (轴 3 交接面)
- **已删除** quant backend 名 `tilelang` / `svdquant`; `awq`/`gptq`/`svd` 仅作为 `backend=pytorch` 的 quant method
- executor / capability 拒绝 `backend=tilelang` 与 `backend=svdquant`; operator stage 仍用 `engine=tilelang`
- recipe 字段名 `backend`+`strategy` 保留; 全量三轴公开 API 仍后续

### 相关

- [xqt-infer-handoff.md](xqt-infer-handoff.md)
- [../explanation/xqt-engines.md](../explanation/xqt-engines.md) (已标注该表为现状债)
- DEBT-001 (convert 与 engine 绑定)
- DEBT-003 (量化 / 推理严格解耦)
- [xqt-realignment-guide.md](xqt-realignment-guide.md) 术语准则
- quantizers: `awq.py`, `gptq.py`, `svd.py`, `int8_mma.py`, `w4_storage_int8_mma.py`

---

## DEBT-003: 量化与推理未严格解耦; 推理应只消费模型 + 计算配置

**状态**: done  
**提出**: 2026-07-12  
**优先级**: high (架构边界; 与 DEBT-001/002 同批统一方案)  
**落地**: [xqt-infer-handoff.md](xqt-infer-handoff.md); `ComputeConfig`, `engine_resolve`, quantizer lazy kernels, package `runtime/compute.json`, payload capabilities.

### 目标契约 (用户原则, 拍板方向)

量化与推理 **严格解耦**:

```text
[Quant 阶段]
  输入: 浮点/训练后模型 + 算法/校准配置
  输出:
    1. 已量化模型 (权重/缓冲/图结构, 网络描述在 model 内)
    2. 可选: 计算配置文档 (各算子/模块的计算逻辑与精度要求, capability 需求)
  不输出: 强制 engine 名, 不把 triton/tilelang 写进推理必选字段

[Infer 阶段]
  输入: 已量化模型 (+ 可选计算配置)
  行为:
    - 按配置中的 精度 / 算子计算契约 执行
    - 引擎选择: 只要求 engine 具备 capability, 不强行指定 engine
    - 不跑 quantizer / calibration / sensitivity
```

一句话: **推理侧不是 "再选一次 quant backend/method", 而是 "执行已量化模型 + 满足计算契约"**.

### 源码审阅: 已对齐的部分

| 点 | 证据 |
| --- | --- |
| runtime 包声明不跑 quant | `xqt/runtime/__init__.py`, `engine.py` docstring: 不跑 quantizer/calibration/sensitivity |
| runtime → quant 无 import | 目录 import 扫描: `xqt/runtime/*` 不 import `xqt.quant` |
| compute 契约共享且较纯 | `xqt/contracts/compute.py`: `SUPPORTED_COMPUTE_PRECISIONS`, `SupportsComputePrecision`, 无 quantizer 逻辑 |
| Hybrid 默认消费已量化 module | `HybridInferenceEngine(model, policy=...)` / `from_quantized_model` |
| 文件推理与 quant recipe 分离 | `package.py` 只认模型包 manifest, 不解析 quant YAML |

### 源码审阅: 违反 / 模糊严格解耦 的点

#### G1. Quantizer 直接依赖 operator kernel (量化阶段绑死 engine 实现)

- `xqt/quant/quantizers/int8_mma.py` 顶层 import:
  - `xqt.operator_opt.kernels.tilelang.int8_mma`
  - 运行时再 import `xqt.operator_opt.kernels.cute.int8mma_binding`
- 同文件 `_VALID_ENGINES = {auto, tilelang, torch_int_mm, ptx_sm89, ...}`
- **问题**: 量化结果 / 前向路径在 quantizer 内就选 engine, 推理侧无法 "只收模型 + 配置,再按 capability 选 engine".

#### G2. Quant 产物字段携带 backend/method/strategy, 推理仍可读算法身份

- `QuantizedModel`: `backend`, `method`, `strategy` + `metadata`
- `QuantizedModelPayload`: 另有 `execution_policies`, `algorithm_metadata`, `capability`, `composite_quant_artifacts`
- **问题**: 算法身份应留在 quant 报告; 推理交接面应是 **model + compute/capability 配置**, 而不是再暴露 awq/gptq/torchao 作为推理输入主字段.

#### G3. ExecutionPolicy 混有 runtime 名, 缺 "capability 需求" 形态

- `ExecutionPolicyPayload`: `policy_kind`, `runtime`, `precision_overrides`, `metadata`
- `SUPPORTED_COMPUTE_PRECISIONS` 仅 `w4a4/w4a16/w8a8/bf16`, 无完整算子级 "需要 int8_mma / fp4 dequant gemm" 的 capability 声明
- **问题**: 配置更像 per-module precision 开关, 不是 "引擎须具备的能力清单"; 也没有 "禁止硬编码 engine,只写 capability" 的 schema.

#### G4. Operator RuntimePlan 仍以 engine 为主键

- `RuntimePlanPayload.engine` 必填; targets 带 engine
- **问题**: 与 "不强行指定引擎, 只要求 capability" 冲突; plan 应优先记 **compute contract + required capabilities**, engine 仅作为 materialize 结果/候选.

#### G5. convert / nn 在量化与推理之间又插一层 engine 绑定

- `xqt.convert(..., engine=...)` (DEBT-001)
- conversion_impl 直接调 `operator_opt` materialize
- **问题**: 第三条入口再次把 engine 绑进 "变换", 模糊 quant 输出与 infer 输入边界.

#### G6. Export lowering 依赖具体 quant 模块类型

- `xqt/export/lowering.py` import `FP4WeightOnlyLinear`
- **问题**: 部署 lowering 应认 **storage/contract**, 而不是 quantizer 类名 (可接受过渡, 但需收敛).

#### G7. 模型包 runtime_config 仍偏 "指定 backend"

- `runtime/config.json` + `preferred_backend` / `providers`
- 当前闭环 ONNX+ORT 合理, 但缺少与 "计算契约配置" 并列的一等字段; quant 侧如何写入 "各算子精度与逻辑" 未标准化.

### 文档审阅

| 文档 | 相对目标 |
| --- | --- |
| `FRAMEWORK.md` / runtime docstring | 已写 "runtime 不跑 quantizer", 方向对 |
| `xqt-inference.md` | 如实描述三条路径, 但未把 "只收模型+配置,不强制 engine" 立为目标契约 |
| `xqt-engines.md` | 已拆 method/storage/compute, 仍含 quant backend 现状表 (标了债) |
| quant capability 表 / recipes | 仍强化 backend+method 作为主轴 |

### 目标交接面 (方案草案, 未实现)

```text
QuantOutput
  model: nn.Module | graph artifact     # 网络与量化存储在此
  compute_config:                       # 可选独立文件/对象
    modules[]:
      name / pattern
      storage: {format, layout, scales...}   # 事实, 已在权重里也可省略
      compute: {contract: int8_mma|fp4_mma|..., activation, accum, ...}
      required_capabilities: [ "int8_mma", "dequant_gemm_epilogue", ... ]
    # 禁止: required_engine: "tilelang" 作为硬约束主字段
    # 允许: preferred_engines 仅作 hint

InferInput = QuantOutput.model + QuantOutput.compute_config
InferRuntime:
  resolve_engine(required_capabilities) -> engine candidate
  不满足则 fallback / 报错 (按 policy), 不回流量化
```

### 后续统一方案必须回答

1. `QuantizedModel` 公开字段是否去掉或降级 `backend/method/strategy` 对推理的可见性?
2. `compute_config` 文件格式放哪: 模型包 `runtime/compute.json`? 独立 sidecar? module `_xqt_module_contract` 唯一?
3. quantizer 内 `_VALID_ENGINES` 与 kernel import 如何拆到 operator/runtime 选择层?
4. `RuntimePlanPayload` 是否改为 `required_capabilities` 主键, `engine` 仅结果字段?
5. 与 DEBT-001/002 的合并顺序: 先拆 quant 三轴, 还是先定 infer 交接面?

### 落地摘要 (代码)

| 项 | 位置 |
| --- | --- |
| Infer 交接面方案 | `docs/md/architecture/xqt-infer-handoff.md` |
| `ComputeConfig` / contracts | `xqt/contracts/compute.py`, `QuantizedModel.infer_handoff()` |
| capability resolve | `xqt/runtime/engine_resolve.py` |
| ExecutionPolicy / RuntimePlan capabilities | `xqt/contracts/runtime.py` |
| int8_mma 解耦 | `xqt/quant/quantizers/int8_mma.py` lazy kernel + default `auto` |
| 模型包 compute.json | `xqt/runtime/package.py` |
| export duck type | `xqt/export/lowering.py` |

### 相关

- DEBT-001, DEBT-002
- [../explanation/xqt-inference.md](../explanation/xqt-inference.md)
- [../explanation/xqt-engines.md](../explanation/xqt-engines.md)
- `xqt/runtime/engine.py`, `xqt/contracts/quantized.py`, `xqt/contracts/compute.py`
- `xqt/quant/quantizers/int8_mma.py`

---



---

## DEBT-005: SVDQuant 应走 composite_add 混合精度而非特例 runtime

**状态**: planned (部分落地)  
**提出**: 2026-07-15  
**优先级**: high (quant/infer 解耦 + 混合精度主路径)

### 现象

SVDQuant 量化结果是 **低秩高位支路 + 量化 residual 支路**, 语义上是 additive mixed-precision / dual-branch GEMM.  
此前实现把存储, 计算契约, INT8 MMA engine 绑在 `SVDQuantInt8MmaLinear` 特例里, 未走 composite / hybrid handoff.

### 已落地 (方案 C 第一刀)

| 项 | 位置 |
| --- | --- |
| dual-branch `ModuleComputeSpec.branches` + `combine` | `xqt/contracts/compute.py` |
| contract `composite_add` | 同上 `SUPPORTED_COMPUTE_CONTRACTS` |
| quant 默认先写 `SVDQuantLinear` 存储壳 + `compute_config` | `xqt/quant/quantizers/svd.py` |
| `materialize_compute` 绑 residual INT8 | `SVDQuantLinear.materialize_compute` |
| Infer materialize 入口 | `xqt/runtime/composite_materialize.py`, `HybridInferenceEngine.from_quantized_model` |
| 默认仍可 eager materialize (`materialize_compute=True`) | quant policy 可关, 由 Infer 再绑 |

### 已落地 (方案 C 第二刀)

| 项 | 位置 |
| --- | --- |
| `Int8MmaLinear` | `xqt/runtime/modules/int8_mma_linear.py` |
| `W4StorageInt8MmaLinear` | `xqt/runtime/modules/w4_storage_int8_mma_linear.py` |
| `SVDQuantLinear` / `SVDQuantInt8MmaLinear` / `LowRankBranch` | `xqt/runtime/modules/svd_composite.py` |
| packing helpers | `xqt/runtime/modules/packing_int4.py` |
| quantizers 仅 re-export 算法入口 + 兼容类名 | `xqt/quant/quantizers/{int8_mma,w4_storage_int8_mma,svd}.py` |
| `runtime/*` 静态无 `quant.quantizers` import | 扫描通过 |

### 仍待做

1. `strategy=svd_*` 完全降级为兼容别名; 主键只剩 method×storage×compute.
2. composite_add fused 内核 (FUSE_DOWN/UP) 与 `CompositePrecisionGemmSpec` partition 模型统一文档词表 (additive vs k-group).
3. hunyuan helper / recipes 文档改成 composite 口径.
4. ~~切断 `operator_opt` → `xqt.quant.bridges`~~ 已迁到 `xqt.runtime.bridges.nvfp4`; quant.bridges 仅兼容 re-export. `tilelang_validation` 内对 FP4 quantizer 的 import 保持 lazy (仅 validation fixture).

### 相关

- DEBT-002, DEBT-003
- [xqt-infer-handoff.md](xqt-infer-handoff.md)
- [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md)

## 追加模板

复制下面块追加新债:

```markdown
## DEBT-00X: 一句话标题

**状态**: open
**提出**: YYYY-MM-DD
**优先级**: high | medium | low

### 现象
### 源码锚点
### 为什么不合理
### 后续统一方案必须回答
### 本阶段明确不做
### 相关
```
