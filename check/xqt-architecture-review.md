# XQT 架构审阅报告: 低耦合 / 高内聚体检

- 审阅对象: `xqt/` 包 (461 个 Python 文件, 约 137,460 行, 不含 `__pycache__` / `.ipynb_checkpoints`)
- 审阅基线: `xqt/AGENTS.md`, `xqt/FRAMEWORK.md`, `docs/md/XQT.md` 宣称的设计契约
- 审阅方法: 静态结构分析 (AST 级 import 依赖图, md5 全树对拍, 同名文件 diff, 调用方 grep), 关键结论经人工复核与实测验证
- 原始证据表 (依赖邻接表, 重复文件清单, 文件职责表) 见 [xqt-architecture-evidence.md](xqt-architecture-evidence.md)

## 0. 结论摘要 (TL;DR)

XQT 的概念骨架设计意图是清晰的: stage workflow 单一配置形态, session 与 YAML 两条入口共享同一条执行主链, quant 的 C2 route 注册表单一事实源, contracts 不依赖实现层, analysis/benchmark 是纯叶子模块. 这些做对了.

但实际代码与设计契约之间存在系统性偏差, 可归纳为四类:

1. **最底层不干净**: `xqt/core` 直接依赖训练框架 `xdl`, 且 core↔contracts 互相 import. 整个包的基石建立在外部训练框架之上.
2. **流产迁移现场未清理**: `xqt/gemm/` 内存在三份平行基础设施 (平铺层 / `common/` 死快照 7,108 行 / `backends/sm89/` 撞车子包 6,897 行), 其中 `common/` 已坏到无法 import, `sm89/` 子包与平铺层**同时被真实 import**, 且已发生内容分叉.
3. **循环依赖成网, 聚合导出放大**: core↔contracts, quant↔runtime, quant↔export, quant↔workflows, pipeline↔workflows, operator_opt↔runtime 等 8 组包级双向依赖; `import xqt` 实测拉起 372 个 xqt 子模块 + 85 个 xdl 子模块, 冷 import 6.44 秒.
4. **领域逻辑复制粘贴成规模**: 量化 quantizer 层 11 份 `_policy_from_mapping`, 10 份 `_replace_submodule`, 13 份逐行同构的 execute 样板, 而公共基类 Protocol 空转; 同一 strategy 事实表在 3 处平行维护, 靠回归测试防漂移而非单一事实源.

按 v0.x "允许破坏性重构" 的阶段约定, 上述问题当前的修复成本处于历史最低点, 建议尽快处理 P0/P1 项.

## 1. 设计上做对了什么

为公正起见, 先列出经核实健康的设计:

- **两种配置方式共享执行主链**: `optimize_model()` (YAML 路径) 与 `XQTOptimizationSession.run_stage()` (session 路径) 最终都调 `run_optimization_stage(state, stage, runners=_stage_runners())` (`xqt/workflows/session_runner.py:302`). stage 执行, 验收, payload 注册, report 只有一份实现, 没有平行链路.
- **contracts 层的依赖方向干净**: `xqt/contracts/` 不 import quant/runtime/operator_opt 实现层 (唯一跨界是 `xqt.core.errors`), 仅有的两处概念耦合 (`runtime_quant.py:69`, `runtime_manifest.py:224`) 用 lazy import 显式规避并在 docstring 中自证.
- **analysis / benchmark 是纯叶子**: 扇出为 0, 只观察不修改, 基于 Protocol 鸭子类型解耦 quant 记录 (`analysis/layer_analysis.py:22`).
- **quant 的 route 注册表 (C2) 是单一事实源**: capability maturity 从 route 表派生, 没有第二份注册表.
- **export/ 与 pipeline/export_handlers 的分工方向正确**: export/ 是函数库, export_handlers 是编排层, 逻辑没有两处实现.
- **大部分 lazy import 是合理的**: 全包 275 处函数体内 import 中约 2/3 是可选 GPU kernel 的按需加载 (runtime/quant → operator_opt.kernels), 属正确的可选依赖管理.

## 2. P0: 架构红线

### 2.1 `xqt/core` 反向依赖训练框架 xdl, 且与 contracts 互咬

**证据**:

- `xqt/core/errors.py:3` `from xdl.errors import XDLError` - XQT 全部异常层级的基类建立在 XDL 之上.
- `xqt/core/schema.py:8` `from xdl.metric.detection_utils import DetectionPostprocessConfig` - 最底层 schema 直接引用训练侧 metric 工具.
- `xqt/core/schema.py:10` 顶层 `from xqt.contracts.inference import InferenceContractConfig`, 而 contracts 有 7 处顶层 import `xqt.core.errors` - core↔contracts 包级循环.
- 实测 `import xqt` 连带拉起 85 个 xdl 子模块; `import xqt.core.schema` 即拉起整个 contracts 包 + xdl.

**问题**: core 是全包依赖的基石 (fanin=16, 最高). 它依赖 xdl 意味着: (a) XQT 无法脱离训练框架独立安装/部署, 与 "XQT 只关注模型本身, 二者通过 checkpoint 衔接, 不互相接管职责" 的工作区契约直接冲突; (b) 部署侧环境的 import 成本和依赖面被训练侧污染; (c) xdl 的任何变动都可能震碎 xqt 最底层.

**建议 (方向已定)**: `errors.py` 的异常基类下沉为 XQT 自有层级 (不做独立第三方 micro 包); `DetectionPostprocessConfig` 移入 `xqt/core/schema.py` 本地定义 (它只是 detection task 的 schema 字段); `InferenceContractConfig` 的引用反转为 contracts→core 单向 (core 不感知 inference contract).

### 2.2 `xqt/gemm/` 三份平行基础设施: 流产迁移现场

这是全包最具体的低内聚证据, 分三层:

**第一层: `xqt/gemm/common/` 整目录死代码且已损坏 (7,108 行, 12 个模块)**

- 全仓库 (xqt/tests/tools/examples/infer) grep `gemm.common` / `from .common` 零命中.
- 实测 `import xqt.gemm.common` → `ModuleNotFoundError: No module named 'xqt.gemm.common.backends'` - 其 `__init__.py:106` 引用不存在的子包, 说明快照后从未被运行过.
- 与平铺版关系: `p4/fp8/benchmark/quantize/grouped_dispatch/tuning_cache` 6 个文件 md5 完全相同; `contracts/dispatch/layout/preflight/reference/registry` 6 个已分叉 (diff 7~236 行), 方向均为平铺版更新. git 历史显示由单次 commit 一次性加入, 之后只有平铺版在演进.

**第二层: `xqt/gemm/backends/sm89/` 子包与平铺层同时存活 (6,897 行)**

- 13 个 py 中 11 个与平铺 `backends/*.py` md5 完全相同; 平铺版被 `backends/__init__.py` 和多个测试真实使用, 子包版被 `tests/xqt/gemm/test_sm89_backend.py:26` 真实 import - **两份都被引用, 是最糟的重复形态**.
- 3 个文件已分叉 (平铺新, 子包旧): `sm89_build.py` (973 vs 890 行, 子包缺 `Sm89W4A8BuildConfig`), `sm90_fp8_wgmma.py` (792 vs 295 行, 且 sm90 的文件放在 sm89/ 里属错位), `tilelang_marlin.py` (子包版 import 路径机械改写错误, 实测 ModuleNotFoundError, 是潜伏的坏文件).
- `backends/sm89.py` (平铺, 353 行) 与 `backends/sm89/` (包) 同名撞车, 包优先解析使平铺文件成为不可达死代码.

**第三层: 为维持旧副本存在的 9 个 compat shim + 2 个空包**

- `backends/` 顶层 `contracts.py / fp8.py / grouped_dispatch.py / layout.py / preflight.py / quantize.py / reference.py / registry.py / tuning_cache.py` 各约 13 行, 模板为 `from .. import X as _impl; globals().update(...)`, docstring 自述 "remove together with the flat re-export when the layout migration lands".
- `backends/sm90/`, `backends/sm100/` 是只有空 `__init__.py` 的占位包; 而 sm90/sm120 的真实代码 (`sm90_fp8_wgmma.py`, `sm120.py`) 直接加在平铺层.

**问题**: 这是一次按架构子目录重组 gemm/backends 的迁移做到一半的现场. 后果: (a) 任何人改 gemm 都要先考古 "到底哪份是真的"; (b) 已分叉文件意味着修过的 bug 可能只修了一半; (c) shim 自我声明是临时的, 与 v0.x "不需要兼容老接口" 约定冲突; (d) 死代码 7,000+ 行持续污染 grep / 静态分析 / 新人理解.

**建议 (方向已定: 保留 `common/`, 完成迁移)**: 目标布局确认为共享基础设施归 `xqt/gemm/common/`, 架构专用 backend 归 `xqt/gemm/backends/<arch>/` (sm89/sm90/sm100/sm120) - 目录语义更清晰. 执行要点:

1. **内容以平铺层为准, 位置以子包为准**: `common/` 现有副本是过期快照且 `__init__.py:106` 已坏 (引用了不存在的 `common.backends`), 不能保留其内容; 把平铺层的最新版本移入 `common/` 覆盖. 同理 `backends/sm89/` 的 3 个分叉文件 (`sm89_build.py`, `sm90_fp8_wgmma.py`, `tilelang_marlin.py`) 用平铺版新内容对齐, 其中 `sm90_fp8_wgmma.py` 本就错位, 应移入 `backends/sm90/`; `sm120.py` 与 `_sm1xx_runtime.py` 落入对应 arch 子包.
2. **删除平铺层**: 被移走的重复文件, 9 个 shim, 与 `backends/sm89/` 撞车的不可达 `backends/sm89.py` (同名撞车必须只留包), 坏的 `tilelang_marlin.py` 双份.
3. **切换引用**: `gemm/__init__.py` (503 行) 与 `backends/__init__.py` (290 行) 改为从 `common/` 和各 arch 子包聚合; 全仓 (含 tests) 对 `xqt.gemm.X` 平铺路径的 import 改为 `xqt.gemm.common.X` / `xqt.gemm.backends.<arch>.X`. `tests/xqt/gemm/test_sm89_backend.py:26` 已在用子包路径, 无需改.
4. 迁移完成的标准: 全树 md5 对拍无重复, `import xqt.gemm.common` 与每个 arch 子包可独立导入, `tests/xqt/gemm/` 全绿.

### 2.3 循环依赖成网, `import xqt` 拉起全树

AST 级依赖分析 (区分顶层 / TYPE_CHECKING / 函数体内 import) 发现的**包级 2-cycle 共 8 组**:

| 环 | 关键证据 |
|---|---|
| core ↔ contracts | `core/schema.py:10` → contracts.inference; contracts 7 处 → core.errors |
| quant ↔ runtime | 双向各 7 处顶层 import. quant→runtime: `quant/layout_apply_report.py:21`, `quantizers/fp4_dynamic.py:21`, `quantizers/int8_mma.py:15,19`; runtime→quant: `runtime/serving_config.py:18`, `runtime/bridges/external_weight_only.py:27,28` (isinstance 识别 quantizer 模块类) |
| quant ↔ export | `quant/quantizers/__init__.py:129` → `xqt.export`; `export/hf_quant.py:23,24` → quant.quantizers |
| quant ↔ workflows | `quant/plan.py:10`, `quantizers/fake_qdq.py:15` → workflows.stage_specs; `workflows/stage_provider.py:16` → quant.capability |
| quant ↔ model | `quant/sensitivity.py:11` → model.hooks; model 9 处 → quant |
| operator_opt ↔ gemm | operator_opt 10 处 → gemm (复用 GemmSpec); gemm/backends 反向 import operator_opt kernels |
| operator_opt ↔ runtime | operator_opt 5 处顶层 → runtime.bridges; runtime 2 处顶层 + 47 处 lazy → operator_opt |
| pipeline ↔ workflows | workflows 3 处 → pipeline; pipeline 9 处 → workflows (`passes.py:68`, `export_pass.py:19`, `preflight.py:11` 等) |

另有多个 3-cycle: `readiness → quant → workflows → readiness`, `quant → operator_opt → export → quant` 等.

**根因高度集中**: `QuantStageSpec` 等 stage spec 类型和 `OptimizationConfig` 住在 workflows 层, 导致 pipeline 的 pass/preflight, quant 的 plan, readiness 都不得不反咬编排层. **把 stage_specs / config 的 schema 下沉到 contracts 或 core, 可一次性拆掉 pipeline↔workflows, quant↔workflows, readiness 三角三组环.**

**聚合导出放大循环**: `xqt/__init__.py` (65 行, 17 个 eager 导出) → workflows → optimization.py → pipeline.passes + readiness.py → quant/operator_opt/export/prune 全家. 实测 `import xqt` 拉起 372 个 xqt 子模块 + torch + 85 个 xdl 子模块, 冷 import 6.44 秒. `quant/__init__.py` (299 行, 28 条聚合 import) 中的 `quantizers/__init__.py:129` 顶层 import `xqt.export` 是循环放大器 (contracts/runtime_quant.py 的 docstring 已明写这条环存在). 为聚合导出而引入的反向边 (`quant/bridges/nvfp4.py` 整个文件是 runtime.bridges 的 re-export 壳) 同样违反 v0.x 约定.

**建议**: (a) stage spec/config schema 下沉 contracts/core; (b) quantizer 模块类 (被 runtime isinstance 识别的那些) 下沉为协议或移入 runtime, quant 侧注册; (c) `quant/quantizers/__init__.py` 删掉对 export 的顶层 import, route handler 内 lazy; (d) 顶层 `xqt/__init__.py` 改为 `__getattr__` 全 lazy 导出 (现有 6 个符号已是 lazy, 推广即可).

## 3. P1: 显著设计债

### 3.1 两套互不相通的 GEMM 决策系统

- 体系 A (`xqt/gemm/`): registry 驱动的合约级 dispatch - `dispatch_gemm()` 查 `GemmKernelRegistration` maturity + executor 走 fallback chain. **但生产代码里几乎无人调用**: 调用方只有 `tests/xqt/gemm/*` 和 research bench; 包内唯一实质用户是 `runtime/modules/awq_w4a16_linear.py:12-13`.
- 体系 B (`operator_opt/backends/gemm_precision.py` + `gemm_selector.py`): function-call 级 `gemm_with_precision(engine="auto")`, 自带 family 表 + pattern 路由 + 硬编码 if/elif engine 排名. `gemm_precision` 从不查 `xqt.gemm` registry; registry 也看不到 operator_opt kernel (23 个 cutlass 条目全是 metadata_only).
- 桥接只有一条: 体系 B 的各 kernel `from xqt.gemm import dense_gemm_reference` - gemm/ 对体系 B 的实际角色只是 "reference 提供商". 物理依赖反向穿透: `gemm/backends/sm89.py:25-32` 直接加载 `operator_opt/kernels/cute/build/int8mma_sm89.so` 并 import `operator_opt.kernels.cute.int8mma_binding`, docstring 自述 "migrated from the historical operator_opt location" - 迁移未收尾.
- 同一 GEMM 的 schedule 知识散在两处: `kernels/triton/gemm.py:685-843` 硬编码 preset vs `gemm/tuning_cache.py` 离线调优缓存, 机制不同, 覆盖不同.

**问题**: "GEMM 合约 + dispatch" 这个 gemm/ 子包的存在意义, 与实际生产路径 (runtime/quant 直连 operator_opt/kernels, 20+ 处) 脱节. 任何 GEMM 行为改动都要判断改哪套, 新人无法从代码结构推断真实路径.

**建议 (方向已定: 以 `xqt.gemm` 为准)**: `xqt.gemm` (contracts + registry + dispatch + tuning_cache) 是已确认的高性能实现与唯一 GEMM 决策系统, 体系 B 应向它收敛而不是并存:

1. 先完成 2.2 的 `common/` 迁移, 保证 `xqt.gemm` 自身只有一份实现.
2. 把 operator_opt/kernels 中实际可执行的 kernel 注册进 `gemm/registry.py` (替换当前 23 个 metadata_only 的 cutlass 占位条目), 让 registry 能看到真实 kernel.
3. `gemm_with_precision` / `gemm_selector.select_gemm_engine` 的 `engine="auto"` 路由改为查询 registry 的 maturity + fallback chain, 删除 `gemm_selector.py:128` 的硬编码 if/elif 排名; `gemm_precision.py` 收敛为 precision 规范化 + 角色命名 wrapper 的薄层.
4. runtime/quant 直连 operator_opt/kernels 的 20+ 处调用逐步改走 `dispatch_gemm`, 让生产路径真实经过 registry.
5. schedule 知识收敛到 `gemm/tuning_cache.py` (离线版本化调优记录), `kernels/triton/gemm.py:685-843` 的硬编码 preset 改为从 tuning cache 读取 + 默认值兜底.

### 3.2 quant 层概念事实表多处平行维护, 靠测试防漂移

量化领域同一事实存在多份手工平行表:

- strategy 字符串表 **3 处**: `core/schema.py:67-83` `CANONICAL_QUANT_STRATEGIES` (15 项), `quant/strategy.py:39-123` `_STRATEGY_SCHEME_TEMPLATES`, `quant/capability.py:28-44` `_STRATEGY_NATURE`. `strategy.py:141-144` 注释自认 "a regression test asserts the two never drift" - 用测试补偿设计缺陷.
- TRUE-nature compute 集合 **2 处**: `capability.py:46` vs `axes.py:80` (完全相同的 frozenset).
- pytorch backend methods 8 元组 **2 处**: `capability.py:289-298` vs `core/schema.py:97-106`.
- QuantScheme 字段名集合 **2 处**: `types.py:39-43` vs `strategy.py:125-134`.
- nature 推导: `capability._resolve_nature` 是推导入口, 但被 5 个模块跨层引用私有函数, 而 `fp4_weight_only.py:871`, `turboquant.py:662` 又 hardcode `nature=PSEUDO` 绕过推导 - 私有函数公共化与调用方绕开两种坏味道并存.

**问题**: 每新增/修改一个 strategy 需同步 3 张表 + 路由表 + capability storage_strategies 共 5 处, 漏一处就是静默不一致. 这直接违反 "一种事只有一个明显的做法" 的工作区原则.

**建议**: strategy → scheme 模板 → nature → capability 派生为单链: `strategy.py` 一张主表 (strategy 名 → QuantScheme 模板 → nature), `core/schema.py` 的词表与 `capability.py` 的 nature 表都从它派生.

### 3.3 quantizer 层复制粘贴成规模, 公共抽象空转

13 个 quantizer 全同构 (Result dataclass + pack helper + 自定义 nn.Module + `quantize_with_*` + `execute_*_component`), 但同构靠的是复制而不是抽象:

- `_policy_from_mapping` **11 份** (fp4_dynamic:146, fp4_weight_only:64, int8_mma:66, mxfp:55, nvfp4:69, turboquant:503, w4_storage:52, convrot_4bit:269, convrot_int8:1475, svd:275, backends/torchao:74).
- `_replace_submodule` **10 份** - 而 `component.py:59` 已有公共 `replace_component_model`; 更荒诞的是各 `execute_*_component` 用公共版, `quantize_with_*` 内部却用私有复制版.
- `_ordered_unique` / `_prefix_module_names`: 公共版在 `component.py:10,23`, `fake_qdq.py:46,57` 和 `plan.py:23` 又复制.
- `_move_batch_to_device` / `_call_model` / `_iter_calibration_batches`: fp4_weight_only, convrot_4bit, svd 各一份.
- `execute_*_component` 报告组装样板 ~70-100 行 × 13 处逐行同构 (对比 `fp4_weight_only.py:789-887` 与 `turboquant.py:615-677`).
- `quantizers/base.py` 定义了 `Quantizer` Protocol, **没有任何一个 quantizer 实现它** - 抽象存在但空转, 复制粘贴因此没有收敛点.
- `quantizers/__init__.py` (657 行) 中 13 个 route handler 是相同形体的 adapter (~15 行 × 13), 本可由工厂收敛.

**影响**: god file 由此而来 - `convrot_int8.py` 1,955 行 (单类 `ConvRotInt8Linear` 约 1,075 行), `convrot_4bit.py` 1,740 行. 新增一个 quantizer 的正确姿势无法从结构学到, 只能抄一份 700-2,000 行的现有文件再改.

**建议**: 把 `Quantizer` Protocol 升级为有骨架的基类/模板方法 (policy 解析, 模块替换, calibration 消费, report 组装各留 hook), 13 个 quantizer 收敛为只声明算法差异 (pack/unpack, module 类, nature). route handler 用工厂生成.

### 3.4 "quant 只产 artifact" 边界被突破, 量化模块两套平行存在

契约宣称: `xqt.quant` 只负责量化算法与 artifact; `xqt.runtime` 消费 artifact 做调度, 不跑 quantizer. 实际:

- **quantizer 内嵌多引擎 kernel dispatch**: `convrot_int8.py` 的 `ConvRotInt8Linear.forward` (1054-1117) 是 cutlass → native cute → tilelang → reference 的多引擎 dispatcher 含 fallback reason 跟踪; 7 处 import `operator_opt.kernels`. `convrot_4bit.py` docstring (第 6 行) 自称 "Inference / mixed-precision dispatch lives in xqt.runtime" - **与同文件实际代码直接矛盾**. `fp4_dynamic.py:15` 顶层 import kernel 公共层并自带引擎选择逻辑.
- **量化 Linear 模块类两套平行**: quantizers/ 自定义 8 个模块类 (`FP4WeightOnlyLinear`, `AWQGPTQWeightOnlyLinear` 等), 而 `runtime/modules/` 已有另一套 (`Int8MmaLinear`, `W4StorageInt8MmaLinear`, `AWQW4A16Linear`, `Fp8MmaLinear`...). AWQ 甚至两边各一个: runtime 的走 SM89 kernel, quant 的走 dequant reference. 同时 `int8_mma.py:19`, `w4_storage_int8_mma.py:31` 又复用 runtime.modules - 三种风格并存, 新人无法判断新算法该把模块类放哪.
- **私有 packing 函数跨层互引**: `contracts/packing_int4.py:23-37` 的 `_pack_int4/_unpack_int4` 与 `quant/quantizers/fp4_weight_only.py:95-116` **逐字节相同**; contracts 版自称 "shared by quant and runtime" 但 quant 没复用它; runtime/bridges 的两处 import 的是 quant 那份, `runtime/modules/packing_int4.py:3` re-export contracts 那份 - 三处引用两个拷贝.
- **通用 transform 层反向依赖特定 quantizer**: `quant/transforms/rotation.py:22-26` import `convrot_4bit` 的 3 个私有函数.

**建议 (方向已定: 推广 CompositeAdd 模式)**: 量化模块类的归属规则定为 - 存储语义壳 + reference forward 归 contracts (沿用 `CompositeAddLinear` 先例, design-debt 已记录的有意设计), 执行视图归 `runtime/modules/`; 删除双份 (含两个 AWQ 的并存, 并补齐从量化产物到 SM89 native 模块的 materialize 路径); kernel dispatch 一律移到 runtime 侧, quantizer 的 forward 只保留 dequant reference; packing 函数以 contracts 版为唯一实体, 其余改 import.

### 3.5 God module / god class

| 对象 | 规模 | 承担的职责数 |
|---|---|---|
| `workflows/optimization.py` | 958 行 | 4 类: 公开 schema (OptimizationConfig 等) + config 加载校验 + 6 个 stage runner 薄包装 + XQTOptimizationSession |
| `XQTOptimizationSession` | ~486 行 | 25 个公开成员 (7 property + 18 方法); `readiness()` 11 个参数; `export()`/`deploy()` 各 20+ 参数且主体几乎逐行重复 (仅差 runtime_handle 处理) |
| `operator_opt/backends/gemm_precision.py` | 2,170 行 | 6 类: precision 规范化 + family 规格表 (~10 档) + variant 派发表 (75 表项) + engine 路由与 3+5 个 dispatcher + ~25 个角色命名 wrapper + capability 报告 |
| `workflows/stage_specs.py` | 912 行 | 13 个 spec + 构建/转换 + ~25 个 `_validate_*` + 9 个 legacy-keys frozenset; export target 校验按格式裂成 9 对模板化函数 |
| `core/reporting.py` | 769 行 | capability 投影 + metric schema + StageReport 构建 + 6 个 `_find_first_*` 递归启发式 (从任意嵌套 dict 按 key 名扒值, 脆弱的隐式契约) |
| `runtime/modules/int8_mma_linear.py` | 1,301 行 | 单类堆叠 6 条后端路径 (torch_int_mm/tilelang/triton/ptx_sm89/cuda_sm89/GEMV + fused 变体), `_can_use_*`/`_run_*` 方法 20+ |

**建议**: god file 的共同模式是 "词表/表驱动数据 + 每档一个手写函数" 混在一起. 拆法: 表数据 (family/dispatch/preset) 独立成纯数据模块; dispatcher 按 engine 拆到对应 backends 文件; session 的 export/deploy 合并主体, readiness 的 artifact 写入移到 reporting.

### 3.6 StageSpec 与 Config 字段双份定义, 转换靠手工逐字段

`workflows/stage_specs.py` 的 `*StageSpec` 与 `core/schema.py` 的 `*Config` 大面积重复定义:

- `QuantStageSpec` (stage_specs.py:133-146) vs `QuantConfig` (schema.py:252-267): **11 个字段同名**, 仅差 `enabled`/`scheme`.
- `PruneStageSpec` vs `PruneConfig`: 9 字段全同名仅差 `enabled`; `AnalyzeStageSpec` vs `AnalysisConfig`: 13 字段全同名; `BenchmarkStageSpec`, `OperatorStageSpec` 同理.

spec→config 转换逻辑分散在 3 处: `stage_specs.py:373`, `pass_helpers/context.py:91-108` (**逐字段手工列出 11 个字段, 新增字段时静默漏拷**), `preflight.py:90,97`.

另有重复定义: `_OptimizationRunState` 在 `optimization.py:187-200` 与 `session_runner.py:67-80` 字段完全一致地定义了两次, 且前者是从未被实例化的影子定义; context 字段初始化清单在 `optimization.py:_stage_context` (297-320) 与 `pipeline/runner.py:55-106` 重复; 旧 key 黑名单在 `config_compat.py:18-25` 与 `optimization.py:75-82` 重复.

**建议 (方向已定: 自动派生)**: StageSpec 保持为 `stages[*].params` 的 typed 视图, Config 由 spec 自动派生 (dataclass 继承或 `dataclasses.replace` 注入 `enabled=True` 等 runtime 字段), 不再维护两份同名字段定义; 删除影子 `_OptimizationRunState`.

### 3.7 contracts 层混入计算行为

契约宣称 contracts 是 typed payload/contract 定义层, 实际含有真实计算:

- `contracts/composite.py:92` `CompositeAddLinear` 是含真实 `forward()` (:187, dequantize + F.linear + low-rank) 的 nn.Module.
- `contracts/packing_int4.py:62` `_quantize_grouped_fp4_weight` 是量化算法本体.
- `contracts/channel.py:98` `compute_hybrid_linear` 是张量计算 helper; `contracts/runtime_quant.py:264` `first_linear_shapes(model)` 遍历模型.

这些设计是刻意的 "fact source + consume", 但已超出宣称边界, 且是 3.4 所述 packing 双拷贝问题的温床.

**建议 (方向已定: 混合方案)**: 更新契约文档承认 "contracts = payload + 存储协议 + reference 语义, 后端执行归 runtime" (FRAMEWORK.md 补 1-2 条, `xqt/AGENTS.md` 补 contracts 条目); packing 双拷贝消重为 contracts 唯一实体 (6 个文件改 import); `CompositeAddLinear` 等含 forward 的类保留, 但明示其为 artifact 的 reference 语义载体.

### 3.8 engine 词表与能力矩阵三处维护, 新增 backend 需动 13 处

engine 能力描述存在三套词汇: `runtime/engine_resolve.py` (maturity 4 级), `operator_opt/capability.py` (available/hardware_native), `gemm_precision.py:2004-2109` (dispatchable/min_capability) - 同一概念三种表达, 且 `pattern_match/candidate.py:21` 用 `deployment_backend` 而其他地方用 `deployment_engine`, 词汇已漂移.

**新增一个 GEMM-capable backend 需要动的位置 (实测 13 处)**: ① engine_resolve `_ENGINE_REGISTRY`; ② capability `_BASE_CAPABILITIES`; ③ backends/ 新 adapter + ④ 其 `__init__` 导出; ⑤ kernels/ 新子包 (+ kernels/__init__ 聚合); ⑥ `types.py` 加 engine 专属 dict 字段; ⑦ `execute.py` 约 10 处硬编码 engine 集合; ⑧ `metadata.py`; ⑨ `materialize.py:_CONTRACT_PATTERNS`; ⑩ `pattern_match/candidate.py:_PATTERN_TO_BACKEND`; ⑪ wrapper (子包或平铺, 两种形态并存无规则); ⑫ `core/schema.py:30` 词表; ⑬ 若进 GEMM: gemm_selector 分支 + gemm_precision family 表 (+ gemm/registry 条目).

**问题**: 可扩展性的核心指标是 "新增一个 X 要动几处". 13 处且分散在 6 个子包, 意味着扩展成本极高且必然漏改 (execute.py 的硬编码集合就是漏改高发区). 此外 backend 在 backends/ 与 kernels/ 两侧的分布不对称 (cutlass 一侧 5KB 一侧 2.6KB, cutile 一侧 8KB 一侧 7 文件), cute/ (裸 CUDA, 无 `__init__.py`, 含检入的 .so 构建产物) 与 cute_dsl/ (Python DSL 探索线) 命名仅差一个后缀, 极易误放代码.

**建议**: engine 注册收敛为单一声明点 (engine_resolve 的 EngineRegistration 扩展为承载 capability/pattern/materialize 的完整声明), capability.py 与 gemm_precision 的能力描述从它派生; execute.py 的硬编码集合改为查询注册表; `types.py` 的 engine 专属字段改为 `dict[str, Any]` 按 engine key 存取; 重命名 cute/ 或 cute_dsl/ 消除歧义.

### 3.9 模型专用代码住进通用框架层

- `xqt/model/` 宣称 "smoke-only model helper", 实际含 `flux2_klein/` (2,165 行, 真实 FLUX.2 加载+NVFP4 量化+benchmark 全流程), `wan21/` (919 行), `hunyuan_ocr.py` (415 行) + `hunyuan_ocr_tilelang.py` (706 行), `unlimited_ocr.py` (465 行), 共 4,200+ 行真实模型专用代码.
- **包内无任何生产代码 import 这些子包** - 消费者全部在 tests/ 与 examples/; 而它们自身顶层 import `quant.quantizers.*`, `runtime.modules`, `workflows.optimization` (如 `model/hunyuan_ocr.py:15,16`), 是 model↔quant, model→workflows 依赖环的来源.
- runtime 根目录的 `svd_flux_attention.py` / `svd_flux_transformer.py` 是 Flux (特定 diffusers 模型) 专用 helper, 前者直接 import SM89 专用 CUDA kernel; `composite_inference.py:43-56` 硬编码 `module_type.__module__.startswith("diffusers.")` 匹配第三方类.
- toy/fixture 散落: `quant/toy_models.py` (仅 3 个 recipe YAML 字符串 target 引用), `operator_opt/toy_models.py` (370 行, 11 个 Toy* fixture), `prune/toy_models.py` - `xqt/model/smoke_*` 已存在的情况下三处重复安放.

**问题**: 应用/示例级代码住进库包, 让通用层反向依赖模型知识, 模糊了 "框架" 与 "应用" 的边界; `optimize_hunyuan_ocr_svd_int4_blocks` 这类模型专用例外在 FRAMEWORK.md 中需要专门解释, 本身就是边界被破坏的自证.

**建议 (方向已定: 移入 `examples/`)**: 真实模型专用代码 (flux2_klein/wan21/hunyuan/unlimited_ocr, 共 4,200+ 行) 移出 `xqt/` 归入 `examples/`, model/ 只保留 smoke/families/hooks; toy_models 统一归 model/ 或 tests/fixtures; runtime 根目录的 svd_flux_* 并入 `runtime/modules/` (模块本体已在那里). 迁移时注意这些代码当前反向 import `quant.quantizers.*` / `runtime.modules` / `workflows.optimization`, 移到 examples/ 后这些依赖方向自然理顺 (应用依赖框架).

## 4. P2: 局部问题, 命名债与卫生

### 4.1 死代码与空转基础设施 (可立即清理)

| 对象 | 状态 | 证据 |
|---|---|---|
| `pipeline/pass_manager.py` `SequentialPipeline` | 宣称的 pass manager, 实际零调用方 | grep 仅出现于自身定义与 `pipeline/__init__.py:3` 导出 |
| `core/registry.py` 三套 registry 查询侧 | `PASS_REGISTRY.get/build`, `RECIPE_REGISTRY`, `EXPORTER_REGISTRY` 全部零调用; `register_recipe`/`register_exporter` 连注册调用都没有 | 实际查表由 `export_pass.py:269` 的普通 dict `_FORMAT_HANDLERS` 承担 |
| `pipeline/preflight.py` `preflight_optimization_config` | 主链无人调用 (optimize_model 与 session 都不走 preflight), 仅 5 个测试使用; 但 `docs/md/architecture/xqt.md:115` 已把它文档化为公开入口 | 方向已定: 接入主链 (workflow 运行前执行 preflight), 而不是删除 |
| `quant/quantizers/nvfp4_weight_only.py` | 392 行孤儿; 但 `research/xqt-quant-inference-architecture/README.md:40` 的 quantizer 表已列出 | 方向已定: 完成接线 (注册 route + `quantizers/__init__` 导出), 若验证实现不完整再删 |
| `quant/backends/bitsandbytes.py`, `transformers.py` | 各 3 行空占位 | `__all__=[]`, 零 import |
| `gemm/backends/tilelang_marlin.py` | 122 行零引用 | 不在 backends/__init__ 导出; 处置随 2.2 迁移一并收尾 |
| `runtime/engine.py` `HybridInferenceEngine`, `model_runner.py`, `composite_inference.py`, `composite_branch.py` 的 `CompositeBranchModule` | 生产零调用, 仅 tests 实例化; `CompositeBranchModule` 无任何子类 | 公共 API 面由 tests 支撑, 宣称的 "混合推理引擎" 主入口名不副实 |
| `workflows/config_compat.py` 的 `is_optimization_workflow_source` | 零调用方 | 且文件名误导: 内容实际是旧 schema 拒绝器, 方向与 "compat" 相反 |

注: `xqt/gemm/common/` (现内容为 7,108 行过期快照且已损坏) 与 `backends/sm89/` 撞车子包不在本表 - 其处置为保留目录布局, 以平铺层最新内容完成迁移, 见 2.2.

### 4.2 命名撞车与误导性命名

- **engine 两套语义**: `runtime/engine.py` (HybridInferenceEngine, 混合精度前向) 与 `runtime/engine_resolve.py` (engine 注册表) 完全无调用关系, 同名不同义.
- **readiness 三处同名不同层**: `xqt/readiness.py` (环境/场景级) vs `export/export_readiness.py` (模型能否导出) vs `prune/safety.py` (safety guard).
- **`_legacy` 命名债**: `runtime/modules/` 下 5 个 `*_legacy.py` (svd_w4a4/w8a8/fp8/svd/svd_legacy_materializers) 全部被活跃引用, 命名与状态矛盾; `svd_legacy.py` 与 `svd_composite.py` 是两个几乎相同的 compat re-export shim.
- **operator_opt/runtime.py** (CUDA Graph helper, 129 行) 与 `xqt/runtime/` 子系统命名撞车.
- **runtime/ 与 runtime/modules/ 同名文件**: `svd_flux_attention.py` 两处 (124 行薄壳 vs 657 行本体), 同名不同层易误读.
- **wrappers 两种组织形态并存**: TileLang wrapper 拆子包 (`wrappers/`), Triton/reference wrapper 平铺顶层 (`triton_wrappers.py` 1,209 行, `reference_wrappers.py` 563 行), 分界线是 "当时谁大到值得拆", 不是规则.
- **execution/ 三个 compat shim 反成真实导入路径**: `quant/execution/{artifacts,component,selection}.py` 纯转发根目录同名模块, 但 11 个 quantizer + 2 个 backend 共 27 处 import 全走 shim - 实现位置与导入路径颠倒.
- **`quant/bridges/` 名不副实**: 只剩 23 行 nvfp4 re-export shim (实体在 `runtime/bridges/`), 按 v0.x 约定该 shim 无存在理由.

### 4.3 export/ 子系统接口不统一

- 无共同 Protocol/ABC; 各后端自定义 result 类型六种风格 (`ONNXExportResult`, `TorchExportResult`, `OpenVINOExportResult`, `CommandExportResult`, `ExecuTorchExportResult`, `HFQuantExportReport`), 函数命名不一 (`export_onnx` / `export_torch_program` / `export_openvino_ir` / `export_ncnn_from_onnx`).
- 粒度不一致: TensorRT 拆 6 个文件共 1,185 行, 而 `mobile.py` 一个文件 (444 行) 装 ncnn/pnnx/MNN/QNN/ExecuTorch 五个 adapter.
- `export_pass.py` ↔ `export_handlers/` 循环 import 异味: export_pass 顶部 import 9 个 handler, 每个 handler 又 `from .. import export_pass as _ep` 再调 `_ep.export_onnx(...)` - handler 依赖 export_pass 的 import surface 而非直接 import `xqt.export`, 靠模块对象延迟属性访问规避循环.

### 4.4 卫生问题

- 包内混入 `.ipynb_checkpoints/` 6 处 (含过期副本, 污染 grep 结果) 与 `operator_opt/kernels/cute/build/int8mma_sm89.so` 3.2MB 编译产物 - 均已 gitignored 未入库, 但本地工作区持续干扰分析工具; 建议在 `.gitignore` 之外加物理清理脚本或 pre-commit 检查.
- `operator_opt/kernels/__init__.py` (923 行) 内嵌 `KERNEL_GUIDANCE_TABLE` - 自我声明是 `docs/md/explanation/xqt-kernel-guidance.md` 的镜像, 文档以代码字面量形式双份维护.
- `core/schema.py:463-464` `OpenVINOBenchmarkConfig` 的 `@dataclass` 装饰器写了两遍 (小 bug, 顺手可修).
- `xdl_adapter.py:54-66` checkpoint 布局猜测链 (嵌套 state_dict 多层 fallback) 违反本工作区 "不做隐式修正" 原则; 但该文件对 xdl 零 import, 鸭子类型方向是对的.

## 5. 改进路线建议 (按优先级排序)

**第一批 (删除与迁移收尾, 无设计风险)**:

1. 完成 `xqt/gemm/` 向 `common/` + `backends/<arch>/` 布局的迁移 (方向已定, 见 2.2): 平铺层最新内容移入 `common/` 与各 arch 子包 (错位文件归位), 删除平铺层重复文件, 9 个 shim, 撞车的 `backends/sm89.py`, `tilelang_marlin.py` 双份, 修复 `common/__init__.py:106` 坏 import 并切换全仓引用路径.
2. 删 `pipeline/pass_manager.py` + `core/registry.py` 的 PASS/RECIPE/EXPORTER registry (或先删导出再观察); 删 `nvfp4_weight_only.py`, `quant/backends/{bitsandbytes,transformers}.py`, `config_compat.py` 死函数, `optimization.py` 影子 `_OptimizationRunState`.
3. 修 `core/schema.py:463` 双 `@dataclass`.

**第二批 (解环, 恢复层级)**:

4. core 去 xdl 化: errors 基类本地化, `DetectionPostprocessConfig` 本地定义.
5. `stage_specs.py` 的 spec 类型与 OptimizationConfig schema 下沉 contracts/core - 一次性拆掉 pipeline↔workflows, quant↔workflows, readiness 三角三组环.
6. quant↔runtime 解环: 被 runtime isinstance 识别的量化模块类按 CompositeAdd 模式归位 (存储壳归 contracts, 执行视图归 runtime/modules, 方向已定见 6.1); 删 `quant/bridges/` shim; packing 函数以 contracts 版为唯一实体.
7. `xqt/__init__.py` 全量 lazy 导出 (推广现有 `__getattr__` 模式), `quant/quantizers/__init__.py` 删顶层 import export.

**第三批 (收敛复制粘贴, 恢复抽象)**:

8. quantizer 基类落地: `_policy_from_mapping`/`_replace_submodule`/calibration 消费/report 组装各留模板方法, 13 个 quantizer 逐个收敛; route handler 工厂化.
9. strategy 事实表单源化: strategy.py 主表 → schema 词表与 capability nature 派生.
10. StageSpec → Config 自动派生 (方向已定, 见 3.6), 删掉 3 处手工转换.
11. engine 声明单点化 (见 3.8), execute.py 硬编码集合改查表.
12. GEMM 体系收敛 (方向已定: 以 `xqt.gemm` 为准, 见 3.1): operator_opt 的可执行 kernel 注册进 gemm registry, `gemm_selector`/`gemm_precision` 的 auto 路由改查 registry, 生产调用改走 `dispatch_gemm`, schedule 知识收敛到 `tuning_cache`. 属执行工作, 不再是方向讨论.

**第四批 (边界收敛, 方向已全部确认, 决策依据见第 6 节)**:

13. 模型专用代码迁入 `examples/` (方向已定, 见 3.9); toy_models 归一.
14. runtime 生产零调用的公共 API 定位 (方向已定): `HybridInferenceEngine` 与 `ModelRunner` 降级为 reference/交互式便利封装, 修正 FRAMEWORK.md:92 与 `docs/md/explanation/xqt-inference.md` "路径 A" 的主入口宣称; `composite_inference.py` 保留 (有 tests 外真实消费者), 仅修正文档定位; 顺带处理 `fuse_composite_modules` 因 `CompositeBranchModule` 无子类而为 no-op 的问题.
15. export/ 定义统一 adapter Protocol 与 result 基类; 粒度向粗对齐 (方向已定: 合并) - TensorRT 六件 (tensorrt facade + trt_build/trt_types/trt_runtime/trt_perf/trt_inspect/trt_plugins) 收敛回少量文件, `mobile.py` 保持单文件多后端形态.
16. contracts 层边界 (方向已定: 混合方案, 见 3.7): 文档化 "contracts = payload + 存储协议 + reference 语义", packing 双拷贝消重, `xqt/AGENTS.md` 补 contracts 条目.

### 5.1 执行进度 (2026-08-18)

| 项 | 状态 | 说明 |
|---|---|---|
| 1 gemm 迁移 | 已落地 | 平铺层迁入 `common/` + `backends/<arch>/`, shim/撞车/双份全删, 全仓引用与文档已切 |
| 2 死代码删除 | 已落地 | pass_manager/registry/backends 占位/影子 state 已删; 例外按决策 10 留批 4: `preflight_optimization_config` 接入主链, `nvfp4_weight_only.py` 接线 |
| 3 双 @dataclass | 已落地 | `core/schema.py` 修复 |
| 4 core 去 xdl 化 | 已落地 | errors 基类本地化 + `DetectionPostprocessConfig` 本地定义 + analysis 懒加载; 实测 `import xqt` 拉起 xdl 模块数 = 0 |
| 5 stage_specs/Config 下沉 | 已落地 | `StageSpec` 与 workflow schema 下沉 `core/`; `pipeline/` 与 `quant/` 不再反向依赖 `workflows/` 类型定义, workflow 层只保留装配与校验. |
| 6 quant↔runtime 解环 | 已落地 | `Int8MmaLinear`/`W4StorageInt8MmaLinear` 的 reference storage shell 已下沉 `contracts/`,runtime 同名类继承为 execution view. ConvRot quant artifact 默认 reference forward,CUDA fastpath 由 `ConvRot*ExecutionView.from_storage()` 显式开启;quant 不再 import runtime. |
| 7 顶层懒加载与 quant↔export 解环 | 已落地 | `xqt.__init__` 全部公开符号按需加载, 冷 import 不拉起 XDL; quantizer/executor 的 export handler 改为函数内 lazy import. |
| 8 quantizer 公共骨架 | 已落地 | policy 解析,模块替换,calibration helper 与 route handler 工厂已收敛到 `quantizers/base.py`;模型侧算法统一通过 `build_component_quantization_report()` 组装公共报告字段. |
| 9 strategy 事实表单源化 | 已落地 | strategy→`QuantScheme`→nature 与 canonical method/compute 统一由 `contracts/quant_strategy.py` 声明,core schema,capability 与三轴报告单向派生. |
| 10 StageSpec→Config 自动派生 | 已落地 | `stage_spec_to_config()` 统一处理默认值,base,override 与排除字段;workflow,pipeline,preflight 的手工字段复制已删除. |
| 11 engine 单点声明 | 已落地 | `EngineRegistration` 统一承载 capability,pattern,materializer 与可见性;调用侧改为注册表投影. |
| 12 GEMM 体系收敛 | 已落地 | operator precision kernel 进入 `GemmKernelRegistry`;auto route 与 `gemm_with_precision` 统一经过 registry/`dispatch_gemm`,Triton preset 归入 tuning cache. |
| 13 模型代码落点 | 已落地 | FLUX.2 Klein,Wan2.1,Hunyuan OCR,Unlimited OCR 迁入 `examples/xqt_models`;toy fixture 统一到 `xqt/model/toy_models.py`. |
| 14 runtime API 定位 | 已落地 | `HybridInferenceEngine`/`ModelRunner` 明确为 reference/交互式封装;composite fusion 以 `enable_fusion` capability 发现真实实现. |
| 15 export 契约与粒度 | 已落地 | 新增 `ExportAdapter`/`ExportResultBase`;TensorRT plugin/inspect/perf 合并到 `trt_diagnostics.py`. |
| 16 contracts 边界 | 已落地 | typed payload,存储协议,packing 与 reference 语义归 contracts;native execution view 归 runtime/operator_opt. |

层级守卫 `tests/xqt/test_layer_import_boundaries.py` 与 `pytest -q tests/xqt` 已全绿. 16 项架构整改均已落地,`root-workspace-xdl` 知识图谱已刷新并达到 `ready` 状态;最终状态以 TODO 文档的验证清单为准.

## 6. 决策记录

### 6.1 决策总表 (全部已定)

| # | 决策点 | 结论 | 落点 |
|---|---|---|---|
| 1 | 两套 GEMM 体系的方向 | 以 `xqt.gemm` (contracts+registry+dispatch+tuning_cache) 为唯一权威, operator_opt 侧向其收敛 | 3.1, 路线图 12 |
| 2 | gemm/ 内部布局 | 保留 `common/` + `backends/<arch>/` 子包布局, 以平铺层最新内容完成迁移 | 2.2, 路线图 1 |
| 3 | 模型专用代码落点 | 迁入 `examples/` | 3.9, 路线图 13 |
| 4 | export/ 粒度 | 向粗对齐 (合并): TensorRT 六件收敛, mobile.py 保持单文件 | 路线图 15 |
| 5 | core 异常基类 | XQT 自有层级, 不做独立 micro 包 | 2.1, 路线图 4 |
| 6 | StageSpec 与 Config | Config 由 spec 自动派生 | 3.6, 路线图 10 |
| 7 | runtime 零调用 API 定位 | `HybridInferenceEngine`/`ModelRunner` 降级为 reference 封装并修正文档宣称; `composite_inference.py` 保留 | 路线图 14, 依据见 6.2 |
| 8 | contracts 层边界 | 混合方案: 文档化 "payload + 存储协议 + reference 语义", packing 双拷贝消重为 contracts 唯一实体 | 3.7, 路线图 16 |
| 9 | 量化模块类归属 | 推广 CompositeAdd 模式: 存储壳 + reference 归 contracts, 执行视图归 runtime/modules | 3.4, 路线图 6 |
| 10 | 死代码外部引用 | 工作区内零引用确认可删; 例外: `preflight_optimization_config` 接入主链, `nvfp4_weight_only.py` 完成接线 | 4.1 |

### 6.2 决策依据 (原定未决点, 已按推荐定案)

**未决 1 (已定): runtime 零调用公共 API 的定位 (路线图 14)**

三个 API 不是同一处境, 建议拆开决策:

- `HybridInferenceEngine` (engine.py, 324 行): 完整但很薄, 实质能力是 `apply_execution_policy` (policy.py), 且生产已直接调用它 (`model/flux2_klein/load.py:580`, 绕过 engine 类). `from_quantized_model` 是 execution_policies/compute_config 唯一端到端消费示例. 文档 (`docs/md/explanation/xqt-inference.md` "路径 A", FRAMEWORK.md:92) 宣传为主入口, 实际零接线. 接入 deploy 主链需新增第三类 runtime handle, 且无 validate/benchmark 配套, 新增能力约等于零. 倾向: **降级为 reference/交互式便利封装**, 同步修正两处文档宣称.
- `ModelRunner` (model_runner.py, 187 行): 纯组装壳, 无独有 capability; 真实的 contract 消费已由模块自行完成 (`runtime/modules/kv_attention.py:319` 直接调 `consume_runtime_quant_contract`). 2026-08-03 一次性交付后 dormant. 其 report() 可由 `build_runtime_manifest` + `benchmark_prefill_decode` 两行替代. 倾向: **降级或删除均可**, 删除损失仅 tests 与 research 叙事.
- `composite_inference.py` (304 行): 三者中唯一活跃维护 (最近两个提交触及), 且有 tests 之外的真实消费者 (`research/xqt-gemm/bench_sm89_svdq_official_mlp_parity_v1.py:48` Nunchaku parity gate, 收益记录在 docs/md/XQT.md:51). 倾向: **保留**, 仅修正文档定位; 顺带注意 `fuse_composite_modules` 当前因 `CompositeBranchModule` 无子类而是 no-op.

**未决 2 (已定: 混合方案): contracts 层边界 (3.7, 路线图 16)**

- 轻/重二分的实测结论: `runtime_quant`/`runtime_manifest`/`runtime_features`/`contract_consume` 是纯 dataclass 构建与校验, 本就属于 contracts 典型职责, 不应计入 "行为混入". 真正的重行为只有三处: `CompositeAddLinear.forward`/`dequantize_residual` (composite.py:159,187), `_quantize_grouped_fp4_weight` (packing_int4.py:62), `compute_hybrid_linear` (channel.py:98).
- `CompositeAdd*` 下沉 contracts 是**有意设计**而非意外: design-debt 文档 (`docs/md/architecture/xqt-design-debt.md:445,471`) 记录了决策, `tests/xqt/runtime/test_composite_add.py:18-21` 显式锁定 artifact/runtime view 分离, 5 个子类全在 `runtime/modules/`. contracts 恰是 quant 与 runtime 的共同下游中立位, 移到任一侧都制造新的反向依赖.
- packing 双拷贝 (contracts 版与 fp4_weight_only 版 8 个 helper 语义全同) **无论选什么都应消重**为 contracts 唯一实体, 6 个文件改 import 即可, 顺带消除 `quant/quantizers/w4_storage_int8_mma.py:32` 绕道 runtime.modules 的反向依赖.
- 文档面事实: FRAMEWORK.md 并无 "contracts 只放数据" 的明文承诺, 关键抽象表已把行为函数与 contracts 路径写在一起; `xqt/AGENTS.md` 包模块清单缺 contracts 条目. 选 "承认轻行为" 的文档成本很小 (FRAMEWORK.md 补 1-2 条 + AGENTS.md 补条目).
- 倾向: **混合方案** - 文档化 "contracts = payload + 存储协议 + reference 语义, 后端执行归 runtime", packing 消重, 重计算保留但明示为 reference 语义.

**未决 3 (已定: 推广 CompositeAdd 模式): 量化模块类归属 (3.4, 路线图 6)**

- 体系一 (quantizers 自定义 8 个类) 的真实消费者很少: 主要是 `quant/__init__.py` 导出与 readiness 元数据字符串; 跨层 isinstance 集中于一个类 `AWQGPTQWeightOnlyLinear` (`runtime/bridges/` 3 文件 + `quant/layout_apply_report.py:65`).
- 两个 AWQ 服务不同来源且**无转换路径**: `AWQGPTQWeightOnlyLinear` 是算法侧 dequant reference (兼用于识别外部 AWQ checkpoint); `AWQW4A16Linear` 是 SM89 native inference-only, 只被 tests 与 research bench 实例化 - 生产链路中从量化产物到 SM89 native 模块的 materialize 路径当前不存在.
- 既有先例两种, 方向相反: CompositeAdd 模式 (存储壳+reference 归 contracts, 执行视图归 runtime/modules, 有意设计); int8_mma 模式 (`quantizers/int8_mma.py:15,19` 直接用 `runtime.modules` 的类, quant→runtime 反向依赖).
- 选项与改动量: (a) 模块类全下沉 runtime - quant→runtime 依赖全面化, 但顺便拆分 quantizer god file, 统一产物形态; (b) 抽协议 - isinstance 点少改动小, 但不解决 quantizer 类内嵌 kernel dispatch (3.4 主问题依旧在); (c) 推广 CompositeAdd 模式 - 与既有有意设计一致, 与未决 2 联动, contracts 层会变大.
- 倾向: **(c)**, 它已是被测试与文档锁定的既定模式, 且能同时回答未决 2; 若未决 2 选择 "移出计算", 则退而选 (a).

**未决 4 (已定): "生产零调用" 结论的外部引用确认 (审阅局限)**

- 全工作区 (docs/research/learn/config/examples/infer/tools/scripts/pyproject) 排查结果: 死代码嫌疑 (`SequentialPipeline`, 三套 registry 及 register 函数, `CompositeBranchModule`, `is_optimization_workflow_source`, `tilelang_marlin`, `gemm.common`) **零代码引用**, 删除在本工作区范围内安全.
- 两个文档化例外: `preflight_optimization_config` 在 `docs/md/architecture/xqt.md:115` 被写为 "workflow preflight 入口" - 要么接入主链 (optimize_model/session 调用), 要么修正文档; `quantizers/nvfp4_weight_only.py` 在 `research/xqt-quant-inference-architecture/README.md:40` 的 quantizer 表中列为存在 - 要么完成接线, 要么修正文档.
- 唯一无法自查的范围: 本工作区之外的私有脚本若直接 import xqt 内部模块, 需另行确认; 当前按工作区内零引用执行删除 (已与维护者确认).

## 7. 审阅局限

- 本审阅基于静态结构分析与少量 import 实测, 未运行测试套件与 benchmark; "生产零调用" 的结论基于全仓 (xqt/tests/tools/examples/infer/scripts) grep + import 图, 若有仓库外消费者 (例如外部脚本直接 import) 需另行确认.
- 依赖图按 AST 解析, 字符串形式的动态 import (config target, importlib) 已人工补查主要路径, 但不排除遗漏.
- 行数与文件对拍数据以 2026-08-18 工作区状态为准.
