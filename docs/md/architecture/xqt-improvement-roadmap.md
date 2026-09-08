# XQT 改进路线图

本文负责定义 XQT 框架优化的总体方向, 核心里程碑划分, 任务推进顺序, 交付物度量准则以及明确不做的事项. 详细的实施目标, 源码依据, 任务依赖与测试要求维护在 [XQT 改进目标与验收](xqt-improvement-goals.md). 本文不定义已实现 API, 不替代 [XQT 架构](xqt.md) 和 [包内工程契约](../../../xqt/FRAMEWORK.md).

制定日期: 2026-09-05. 最近审阅: 2026-09-06. 状态: 全部里程碑 (M0 至 M5 全部 17 项任务) 已全部达成并通过真实硬件实测验收. 详细验收证据与测量数据见 [改进目标与验收](xqt-improvement-goals.md) 的任务表和第 8 节完成记录.

---

## 1. 总体判断与核心主线

XQT 经过前期迭代, 已经建立了良好的工程基础:
- Session 交互式与 YAML 声明式单一工作流对齐;
- Typed stage 执行链与配置系统;
- 量化存储 (storage) 与执行 (compute) 的分轴抽象;
- 基于 ModelStructureContract 的模型结构契约;
- 算子候选比较, Benchmark 与 Report / Manifest 产物生成;
- 覆盖 1600+ 项测试的多层自动化测试网.

当前下一阶段的重点**不是重构框架核心架构, 也不是简单扩充 backend, kernel 数量或堆砌新 recipe**. 当前的主要瓶颈在于各项能力在端到端串联时的**状态可信性与工程严密性**:
1. **状态被动污染**: 当 stage 执行失败或被验收门槛拒绝 (rejected) 时, 共享 context 与当前模型状态未能安全隔离与回滚, 导致脏状态泄漏给后续阶段;
2. **契约表达失真**: 模型级契约可能仅反映首层 Linear 局部信息, 请求静态量化在缺少 scale 时静默回退到动态 MMA 却仍声称静态, 使得报告与实际模型状态脱节;
3. **执行路径含糊**: 引擎解析仅做静态能力匹配, 甚至将尚未实现的 planned/metadata_only 引擎选入候选链, 缺乏针对当前硬件与环境的严格可执行性检查 (fail-closed);
4. **局部收益与整模脱节**: 局部算子微小的 speedup 可能被 runtime wrapper 开销, layout copy 或图中断 (graph break) 完全抵消, 无法延续到真实大模型与重新加载的部署产物中.

### 核心主线: 真实模型结构契约驱动的端到端优化闭环

下一阶段的核心演进逻辑是: **以真实模型为载体, 以显式契约为纲领, 保证每一步变换的事务隔离与真实收益, 交付可新进程重载运行的部署产物**.

```text
[真实 Checkpoint + ModelProfile]
       │
       ▼
[ModelStructureContract 校验与角色绑定] (XQT-010)
       │
       ▼
[Baseline 测量与门槛冻结] (XQT-014, XQT-009)
       │  (测定 Eager Latency, 峰值显存, 容差与提升阈值)
       │
       ├── (主分支: Quant 事务执行) ──► [模块合同 + MMA/FP4/SVD/ConvRot] (XQT-006, 007)
       │                                     │
       └── (独立分支: 拓扑 Prune) ────────────┤
                                             ▼
                               [Typed Graph Rewrite] (Dequant+GEMM / Norm+Quant) (XQT-012)
                                             │
                                             ▼
                               [Block Materialize / Compile] (CUDA Graph / Wrapper) (XQT-013)
                                             │
                                             ▼
                               [整模级 Acceptance 判定] (XQT-002, 009, 015)
                                (通过 ──► Commit 推进 Current; 拒绝 ──► Rollback 原状态)
                                             │
                                             ▼
                               [Quant Pair 安全发布] (两阶段 Atomic Commit) (XQT-005)
                                             │
                                             ▼
                               [独立新进程重载 + 实际推理 + 外部质量验收] (XQT-016)
```

### XDL 与 XQT 的职责边界

根据工作区规范, XDL 与 XQT 保持明确的单向衔接, 互不侵入:

| 维度 | XDL (训练侧) | XQT (部署侧) |
| --- | --- | --- |
| **核心职责** | 负责参数训练, SFT/LoRA 微调, RL 偏好优化 (GRPO), 知识蒸馏, 训练期 Checkpoint 保存 | 负责模型结构契约解析, 离线压缩 (量化/剪枝), 计算图重写, 算子/Block 编译, 部署导出与 Benchmark |
| **执行模型** | CoreModel + Trainer 驱动生命周期, 支持分布式 (DDP/Accelerate/DeepSpeed), 拥有 Optimizer/Loss/Scheduler | Session 或 YAML workflow 驱动 typed stages, 纯无状态推理或前向执行, 零参数梯度更新 |
| **数据与评测** | 拥有 Dataset, DataLoader, 负责训练 batch 构造与任务评测提供者 (Evaluation Provider) | 只消费调用方传入的固定 Tensor 或校准 Iterable (用于激活统计), 不构造数据集, 不实现任务评测引擎 |
| **质量证据衔接** | 产出真实模型权重 (Checkpoint) 与下游任务评测打分 (如生成质量, 困惑度, 准确率) | 将外部评测打分作为元数据绑定到优化报告 (Report/Manifest), 不在包内实现 recovery/finetune 循环 |

---

## 2. 保留与调整

| 既有基础 (保留) | 现状不足 | 调整重点 (目标) |
| --- | --- | --- |
| **Session 优先, YAML 次之, 共用执行链** | 失败 stage 会原地修改 context.model; `from_stage` 与 `use()` 导致 lineage 错乱 | 为共享执行链引入严格的 Prepare-Execute-Validate-Commit 事务机制与独立的 Current 模型身份 (XQT-002, XQT-003) |
| **quant method / storage / compute 分轴** | 请求 static MMA 缺 scale 时静默退化为 dynamic 但报告仍标 static; 模块级合同缺失 | 建立模块级量化合同 (Module Quant Contract), 缺 scale 显式失败, 报告与模块真实状态严格一致 (XQT-006, XQT-007) |
| **model structure / inference / compute 契约** | 直接传入 model 实例时跳过 adapter 导致契约丢失; 仅按首层 Linear 统计模型 shape | 统一模型直接注入与 adapter 加载的契约管道, 逐模块记录全局/局部维度与保护层 (XQT-007, XQT-010) |
| **kernel / wrapper / block / model 分层** | wrapper 存在冗余开销; CUDA Graph 缺乏边界与缓存淘汰管理; 算子加速未能转化为整模提速 | 以 Block 准入契约与整模性能/显存双门槛为准, 消除 wrapper 与 layout 转换开销, 规范 Graph 缓存生命周期 (XQT-013, XQT-015) |
| **capability, readiness, report, manifest** | 静态能力目录与本机执行解析混淆, 未实现/非调度引擎被意外选出; 指标递归取 max 掩盖退化 | 严格拆分候选能力查询与真实可执行性解析 (Fail-closed); 指标支持显式 key path 与严格聚合判定 (XQT-008, XQT-009) |
| **产物发布 (Quant Pair)** | 写入路径未防范目录逃逸; 先写权重后写 sidecar 非原子操作, 异常留下半损坏产物 | 统一读写路径校验, 实施 Staging -> Checksum -> Atomic Replace 两阶段发布 (XQT-005) |
| **既有 Registry 与后端适配** | `kernels/registry` 与 `ops/gemm/registry` 职责存在重叠与镜像混淆 | 收敛为单一权威源, 保持单向派生, GEMM dispatch 专注算子实现, 全局 registry 专注元数据视图 (XQT-011) |

遵循当前开发阶段原则: XQT 目前处于 v0.x 开发期, 取舍顺序为: **清晰 > 简洁 > 方便 > 兼容**. 允许按对齐的范围直接重构接口, 但必须保持各入口语义对称, 拒绝隐式兼容与宽松字符串修正.

---

## 3. 六大里程碑与阶段准入

优先级用于衡量缺陷与风险等级 (`P0` 为状态/产物可信性缺陷; `P1` 为核心闭环链路; `P2` 为架构收敛与复用扩展); 里程碑表示前后推进的依赖阶段, 严禁混用.

```text
[M0. 可复核基线] (XQT-001, 014资源前置)
       │
       ▼
[M1. 状态与结果可信] (XQT-002, 003, 004, 005, 006, 007, 009)
       │
       ▼
[M2. 模型与执行契约接通] (XQT-008, 010, 011)
       │
       ▼
[M3. Block 优化路径打通] (XQT-012, 013, 014)
       │
       ▼
[M4. 真实模型端到端交付] (XQT-015, 016)
       │
       ▼
[M5. 第二同族模型复用验证] (XQT-017)
```

### M0. 建立可复核基线

- **核心目标**: 消除既有工程质量告警, 固化多层测试基线与软硬件环境记录, 尽早完成真实模型资源可用性探测.
- **对应任务**: [XQT-001](xqt-improvement-goals.md#xqt-001); 提前执行 [XQT-014](xqt-improvement-goals.md#xqt-014) 的模型权重与本地资源前置检查.
- **准入条件 (Entry Criteria)**:
  - 记录当前 worktree 状态与基线 commit. 工作区可以有用户已有改动, 但基线报告必须说明它们是否参与测量, 不得擅自清理;
  - 记录实际 Python, PyTorch, CUDA, driver 和 GPU 环境. 不把某个开发机版本写成项目的未经验证硬性要求.
- **核心交付物 (Key Deliverables)**:
  - 修复 `prune/discovery.py` 中 2 处 `_StructuredCandidate` 未定义导致的 `ruff` F821 错误;
  - 明确 CPU 必跑, SM89 硬件必跑, Optional backend 三层测试运行规约与环境标记;
  - 产出基线测试运行日志, 记录 passed / skipped / failed 明细及每个 skip 的具体环境原因;
  - 产出首选模型 FLUX.2 Klein 4B 的资源可用性记录: source/revision, 许可状态, 文件清单与校验和, 本地可用磁盘, 加载峰值和运行峰值的预检结果.
- **退出条件 (Exit Criteria)**:
  - `ruff check xqt` 清零;
  - pytest 基线回归完全可复现, skip 原因清晰透明, 无因依赖缺失导致的意外伪绿.
- **回退与阻断预案**: 若 FLUX.2 Klein 4B 权重不可获得或下载受限, 记录 XQT-014 为 `blocked`, 同时推进 M1/M2 基础正确性任务. 是否切换验收模型需要单独决策, 重新冻结完整 baseline, 不能以 Qwen 等异构模型悄悄替代.

### M1. 保证状态与结果可信

- **核心目标**: 彻底解决 Stage 事务回滚, 模型身份与 Lineage, 校准状态隔离, 产物写入安全, 量化请求/实现一致性以及指标读取的作用域污染问题.
- **对应任务**: [XQT-002](xqt-improvement-goals.md#xqt-002), [XQT-003](xqt-improvement-goals.md#xqt-003), [XQT-004](xqt-improvement-goals.md#xqt-004), [XQT-005](xqt-improvement-goals.md#xqt-005), [XQT-006](xqt-improvement-goals.md#xqt-006), [XQT-007](xqt-improvement-goals.md#xqt-007), [XQT-009](xqt-improvement-goals.md#xqt-009).
- **准入条件 (Entry Criteria)**: M0 顺利退出, 测试基线日志与环境已归档.
- **核心交付物 (Key Deliverables)**:
  - `Prepare-Execute-Validate-Commit` 事务机制, 保证 rejected 或 exception 时原始模型与配置完全不受污染;
  - 明确的 Current, Baseline, Best 模型身份流转与 Lineage DAG, 杜绝自指与来源丢失;
  - 带精确 `finally` 恢复的临时推理校准上下文管理器, 隔离 BatchNorm 与 Dropout 副作用;
  - 路径严格防逃逸与 Staging -> Checksum -> Atomic Commit 的 Quant Pair 发布机制;
  - Static INT8 MMA 缺 scale 时 fail-closed 抛出异常, 拒绝静默 fallback 为 dynamic;
  - 模块级量化合同 (Module Quant Contract) 与显式 no-op 状态;
  - 具有显式路径定位与聚合规则的指标读取契约, 根除同名 key 盲目取 max 问题.
- **退出条件 (Exit Criteria)**:
  - 编写专门的失败注入回归测试集 (包含 rejected pruning, exception in report, invalid scale, path escape, duplicate metric key 等极端情况), 全部断言通过;
  - 共享 context 在任何失败后保持与执行前严格等价.

### M2. 接通模型与执行契约

- **核心目标**: 将模型结构契约 (ModelStructureContract) 贯通至执行主链, 严格分离引擎的静态候选能力与本机实际可执行性, 收敛 Registry 权威源.
- **对应任务**: [XQT-008](xqt-improvement-goals.md#xqt-008), [XQT-010](xqt-improvement-goals.md#xqt-010), [XQT-011](xqt-improvement-goals.md#xqt-011).
- **准入条件 (Entry Criteria)**: M1 中 XQT-003, XQT-007 等契约基础任务完成并通过回归.
- **核心交付物 (Key Deliverables)**:
  - 拆分 `query_engine_capabilities` (静态目录) 与 `resolve_executable_engine` (真实执行), 强化 SM 算力与成熟度校验 (fail-closed);
  - 统一直接传入模型 (direct model injection) 与 Adapter checkpoint 加载的结构契约注入与校验管道;
  - 明确拓扑变更 (prune / rewrite) 触发契约失效与重建的机制;
  - 明确 `kernels/registry` 与 `ops/gemm/registry` 的职责划分, 消除双向镜像与冗余维护.
- **退出条件 (Exit Criteria)**:
  - 受支持的模型 profile 可自洽生成并校验执行目标, 映射未覆盖权重或模块不匹配时显式报错;
  - 无可用本机执行路径时显式拒绝, 不将 planned / metadata_only 伪装成可执行引擎.

### M3. 建立 Block 优化路径

- **核心目标**: 落地强类型计算图重写 (Typed Graph Rewrite), 实现从模型结构契约驱动的 Block 级提取, 编译与 Runtime Materialization, 并冻结真实模型基线与准入门槛.
- **对应任务**: [XQT-012](xqt-improvement-goals.md#xqt-012), [XQT-013](xqt-improvement-goals.md#xqt-013), [XQT-014](xqt-improvement-goals.md#xqt-014).
- **准入条件 (Entry Criteria)**: M2 契约打通, XQT-009 指标作用域与 XQT-002 事务机制已就绪.
- **核心交付物 (Key Deliverables)**:
  - 强类型 GraphTransform 规范及首批三大 pattern (`dequant_gemm`, `norm_quant`, `activation_quant`) 契约与 reference 实现;
  - 通用 Block Materialize 机制, 消除冗余 wrapper 与 layout copy, 规范 CUDA Graph 缓存与生命周期;
  - 在真实模型权重上测定未优化的 Eager 延迟 (p50/p95), 稳态吞吐, 峰值显存基线, 并正式冻结 Block 准入与整模验收门槛 (数值容差, 性能提升阈值).
- **退出条件 (Exit Criteria)**:
  - 从真实模型提取的 Block 在 SM89 目标硬件上通过数值一致性与性能门槛;
  - 冻结门槛正式成文并归档, 不得在后续优化中随意降低标准.

### M4. 交付真实模型闭环

- **核心目标**: 在首选真实模型 (FLUX.2 Klein 4B) 上贯穿端到端优化, 结合外部质量证据, 产出部署产物并在全新独立 Python 进程中重载运行, 证明低比特加速与显存收益.
- **对应任务**: [XQT-015](xqt-improvement-goals.md#xqt-015), [XQT-016](xqt-improvement-goals.md#xqt-016).
- **准入条件 (Entry Criteria)**: M3 中 Block 优化验证通过, 真实模型 Baseline 已冻结.
- **核心交付物 (Key Deliverables)**:
  - 完整模型优化工作流 (Profile -> Quant -> Rewrite -> Materialize -> Benchmark), 产出完备消融实验报告 (Eager vs Quant vs Quant+Rewrite vs Quant+Compile);
  - 外部任务评测证据 (如图像生成特征 diff 或评测指标) 正式绑定模型产物;
  - 独立 Python 进程中基于 `quant.json` + 权重产物的完整加载与推理脚本, 验证数值对齐与真实吞吐/显存收益;
  - 独立呈现 Native PyTorch 低比特路径与 ONNX/TensorRT 部署路径的对照报告.
- **退出条件 (Exit Criteria)**:
  - 优化后模型在冻结容差内通过数值验收, 达成预定整模加速与显存下降目标;
  - 产物脱离原始 Session/Recipe, 可在全新进程中直接重载并执行稳定推理.

### M5. 验证可复用性

- **核心目标**: 引入第二个同族真实模型, 检验当前沉淀的结构契约, 事务执行与优化主链是否具备真正的声明式复用能力. 异构模型属于后续架构扩展, 不混入本里程碑的结论.
- **对应任务**: [XQT-017](xqt-improvement-goals.md#xqt-017).
- **准入条件 (Entry Criteria)**: M4 首个模型闭环完全通过并交付.
- **核心交付物 (Key Deliverables)**:
  - 第二真实模型的结构契约定义与 Adapter 接入; 它必须与首个模型共享已声明的算子语义和 profile 族边界;
  - 该模型在 XQT 主链上的优化, 产物导出与独立重载验证报告;
  - 评估新增的 profile / mapping / adapter 代码与通用主链改动. 如确需扩展通用语义, 明确说明理由和对首个模型的回归影响.
- **退出条件 (Exit Criteria)**: 第二模型成功产出达标部署产物, 未对通用核心造成破坏性回退.

---

## 3.1 跨里程碑闸门

里程碑描述工作范围, 以下闸门描述是否可以把结果推广到下一阶段. 任务可以并行推进, 但主交付不能跳过闸门.

```text
G0 基线可复核
  -> G1 失败不污染状态
  -> G2 契约与执行可解释
  -> G3 真实模型 baseline 冻结
  -> G4 block / 整模候选通过
  -> G5 新进程重载与实际部署
```

| 闸门 | 必须证据 | 不足以通过 |
| --- | --- | --- |
| G0 | lint,测试分层,commit 和环境记录, 每个 skip 有原因 | 一次本地运行或"多数测试通过" |
| G1 | reject / exception / retry 后 current model,accepted graph 和旧 artifact 不变量测试 | 只看到 `accepted=False`, 或只恢复参数而未恢复拓扑/元数据 |
| G2 | request,resolved contract,materialized executor,observed runtime 和 fallback 分字段报告. 字段定义与最低溯源要求见 [四层证据与产物溯源](xqt-improvement-goals.md#四层证据与产物溯源) | 静态 capability,package import 或 planned entry |
| G3 | 真实 checkpoint/revision,输入 profile,baseline raw samples,数值/质量/性能门槛冻结记录 | synthetic proxy,历史截图或未绑定 artifact 的指标 |
| G4 | 同一 scope 内通过数值,性能,显存,fallback 和质量门槛的候选证据 | 最快 kernel,单一 target speedup 或 compile 成功 |
| G5 | 新进程加载同一发布 bundle, 实际 forward,diff 和 steady-state benchmark | dry-run,文件存在或 runtime session 仅创建成功 |

未通过闸门的实验可保留为 `rejected` / `not_decided` 诊断, 但不能更新 accepted artifact,能力 maturity 或对外性能表述.

## 3.2 代码边界与交付责任

实施前在任务记录中标明要改的责任面. 下面是默认 ownership, 不是要求新建同名抽象或跨任务重构全部目录.

| 责任面 | 默认代码范围 | 必须说明 |
| --- | --- | --- |
| Session / workflow 状态 | `xqt/workflows/`, `xqt/pipeline/` | candidate 生命周期,commit 时机和失败后的可见状态 |
| 模型结构 | `xqt/model/`, `xqt/contracts/model_structure.py` | profile,mapping,角色,contract version 与拓扑失效 |
| 量化 / 校准 | `xqt/compression/quant/`, `xqt/contracts/` | request -> module result -> storage/compute contract 的映射 |
| engine / runtime | `xqt/kernels/`, `xqt/runtime/` | 静态能力,执行探测,materialize,fallback 与 observed evidence |
| 产物 / 导出 | `xqt/contracts/quant_pair.py`, `xqt/export/`, `xqt/runtime/package.py` | path ownership,发布协议,schema 和新进程加载条件 |
| 验收 / 证据 | `tests/xqt/`, `research/` artifacts | scope,raw samples,环境,阈值和结论状态 |

一个改动触及多个责任面时, 先写出 data flow,状态所有权和失败边界. 单纯把逻辑移动到另一目录不能算完成.

## 3.3 推进停止与回退

- 缺少权重,许可证,可选依赖或目标硬件时, 记录 `blocked` 条件和恢复路径, 不以 toy/synthetic 输入替代真实模型 gate.
- block 层通过而模型层失败时, 保留 block 诊断但回退到已接受模型. 整模门槛不能自动向局部结果靠拢.
- 性能噪声,数值差异或质量证据来源不完整时, 结论为 `not_decided`; 不以"基本通过"更新阶段状态.
- 需要破坏 Provisional API 时, 先列明调用方,测试,recipe 和文档迁移范围. v0.x 允许断开旧接口, 但不增加隐式兼容层.
- 同一方案连续两轮未达到冻结门槛时, 回到 profiler / runtime integration / contract 假设进行诊断, 而不是继续调参并选择性报告最优值.

---

## 3.4 变更包与状态治理

路线图的最小推进单位是一个 `XQT-NNN` 变更包, 而不是一次笼统的"优化 XQT"提交. 每个变更包在开始前必须写明: 任务 ID, 源码基线 commit, worktree 是否有无关改动, 责任面, 不变量, 允许的外部副作用, 计划测试层级和预期证据路径. 这份记录可以放在任务完成记录的草稿或对应 `research/` 实验说明中, 但不得替代 goals 的任务状态.

| 状态转换 | 允许条件 | 必须留下的记录 | 禁止行为 |
| --- | --- | --- | --- |
| `pending -> in_progress` | 已复核源码, 依赖具备或被明确豁免 | 基线 commit, 范围, 已知风险和首个失败复现 | 用代码改动或聊天结论替代状态记录 |
| `in_progress -> blocked` | 缺少权重, 许可证, 硬件, 外部 runtime 或必要决策 | 缺失条件, 已做检查, 恢复条件和可并行任务 | 用 synthetic 结果伪装真实模型 gate |
| `in_progress -> pending` | 主动放弃尚未开始的实现 | 放弃原因, 保留或删除的诊断产物 | 将半成品实现描述为可用接口 |
| `in_progress -> done` | 满足 goals 的全部完成标准 | 测试命令和结果, 事实文档, 决策记录, 证据索引 | 仅因 lint 或局部单测通过而标记完成 |
| `in_progress -> deferred` | 有明确的范围或优先级决策 | 迁移目标, 未完成范围和重新启动条件 | 无说明地删除任务或验收条件 |

单次 stage attempt 的 `accepted`, `rejected`, `failed`, `not_decided` 和 `interrupted` 是运行结果, 不改变任务状态. `interrupted` 只描述无法确认内存事务结果的进程中断, 不得推断为 accepted. attempt report 必须带稳定的 `attempt_id`, `source_stage`, 输入 profile 和源 artifact/模型摘要; 相同 stage 名的重试必须产生新的 attempt 记录, 不能覆盖已接受 stage 的 report 或 manifest 项.

任务之间共享代码时, 后执行的变更包必须重新验证先前任务的不变量. 例如, XQT-013 改写 runtime cache 后, 必须复跑 XQT-002 的拒绝回滚和 XQT-003 的 lineage 回归, 而不是只验证新的 CUDA Graph 路径. 所有性能数字必须先是可复核 artifact, 再能进入 README, 能力矩阵或对外说明.

---

## 4. 首个真实模型候选与冻结规范

**FLUX.2 Klein 4B 是首选候选, 不是已经冻结的唯一验收载体.** 仓库在 `xqt/model/flux2_klein/` 已有基础适配代码, 适合检验模型结构契约, 混合精度, Block 组合与产物交付. XQT-014 完成 source/revision/许可/资源/基线核验前, 不能把它写成已可复跑的 golden model.

### 已知代码标识与待核验项

- **Base Checkpoint**: `black-forest-labs/FLUX.2-klein-4b`
- **NVFP4 Checkpoint**: `black-forest-labs/FLUX.2-klein-4b-nvfp4` (单权重文件: `flux-2-klein-4b-nvfp4.safetensors`)
- **当前代码入口**: 相关常量位于 `xqt/model/flux2_klein/types.py`; 加载与运行 helper 位于同目录的 `load.py`, `optimize.py`, `runtime.py`.
- **当前开发机**: RTX 4070 Ti SUPER (`sm_89`) 可作为首个硬件平台, 但正式报告仍须记录实际 GPU,driver,runtime 和依赖版本.
- **必须在 XQT-014 核验**: checkpoint revision/commit,许可接受状态,文件清单/每文件 checksum,Diffusers 兼容版本,模型 config,实际 block 数和维度,加载方法与所需内存. 这些不从模型名或局部 helper 推断.

### 显存与计算预算

16GB `sm_89` 是本轮目标资源约束, 不是性能结果. XQT-014 先测量以下四项, 才能冻结候选的显存门槛:

1. checkpoint 加载和模型 `.to(device)` 的峰值;
2. 给定输入 profile 下 BF16 eager 的 allocated/reserved/设备进程总占用;
3. quant storage, prepack, graph capture 和 runtime workspace 分别新增的峰值;
4. 目标平台预留的安全余量及 OOM 重试策略.

不得将估计的权重体积或参数 bit 数当作运行时峰值显存. 量化后发生 dense materialization / cache expansion 时必须单列记录. 可接受峰值和安全余量只能在 baseline 测定后, 结合实际可用显存与 CUDA runtime 开销冻结, 不能在规划阶段预设固定数值.

### 输入输出协议与优化边界

- **当前 helper 的 tensor 边界**: NVFP4 runtime helper 使用 `hidden_states`, `encoder_hidden_states`, `timestep`, `img_ids`, `txt_ids`, 可选 `guidance` 和 `joint_attention_kwargs`. 完整模型实际 forward signature 仍以 XQT-014 加载出的目标 revision 为准.
- **冻结输入 profile**: 至少命名 `primary` 和 `guardrail` 两组 profile. 每组保存 batch, 每个 tensor 的 shape/stride/dtype/device, optional kwargs, 固定生成 seed/来源以及是否允许 CUDA Graph. 不在计划中提前猜测 sequence length 或 tensor layout.
- **XQT 模型侧优化边界**:
  - XQT **只负责** `Flux2KleinTransformer.forward(...)` 的 Tensor 计算, 评估其时延 (ms), 吞吐 (steps/s), 峰值显存 (MB) 与输出 Tensor 数值一致性 (Cosine Similarity, Max Abs Diff);
  - 端到端文生图 Pipeline (包含 T5/CLIP 文本编码器分词, 50 步扩散迭代循环调度, VAE Decoder 图像重建, 以及最终生成图像的审美/语义打分) **属于外部调用方职责 (XDL 或评测工具)**;
  - 外部评测负责执行完整生成并向 XQT 回传指标证据, XQT 不在内部实现文生图 Pipeline 调度器. 每份证据必须绑定输入 prompts/seeds 或相当的样本清单,评测工具 revision 和对应 artifact checksum.

### 备选模型切换准则

若因网络环境, HuggingFace 访问权限或存储限制无法获得 FLUX.2 Klein 4B 权重:
1. 任务在 XQT-014 标记为 `blocked`, 并明确记录缺失条件;
2. 由项目维护者明确决策是否切换验收模型, 并说明它对模型族,质量指标和部署路径的影响;
3. 切换必须更新文档记录, 重新测定基线并冻结门槛, 严禁私自更换或使用随机权重的 toy model 替代.

### 冻结包与变更规则

XQT-014 通过时必须生成一个版本化的 "target freeze" 记录. 它至少包含: model/profile/contract schema version, repo ID 和不可变 revision, 每个权重文件的相对路径/大小/SHA256, adapter 与加载配置版本, `primary`/`guardrail` profile 的完整规范, reference artifact ID, 质量和性能阈值, warmup/采样/聚合/离群值规则, 环境指纹以及原始样本文件的校验和. 该记录是 XQT-015/016 的唯一对照来源.

下列任意变化都必须新建 freeze version 并重新测量受影响的 baseline, 不得沿用旧门槛: checkpoint revision 或权重, adapter/结构契约, 输入 profile 或 seed 集, runtime/driver/PyTorch 版本, 编译/Graph 模式, 计时方法, 误差定义或质量评测版本. 只改变报告排版或新增不参与判定的诊断字段可复用原 freeze, 但必须在报告中明确说明.

冻结记录应作为只追加的机器可读文件保存于对应 `research/` 或 artifact 目录, 文本报告只引用其 `freeze_id` 与 SHA256. 以下结构是 XQT-014 的设计目标, 用于约束字段完整性, 并不表示当前仓库已经提供同名 loader 或 schema:

```text
freeze_id: flux2-klein-4b/<revision>/<profile-set>/<sequence>
created_at: UTC timestamp
source:
  git_commit, worktree_summary, model_repo, model_revision
  weights: [{relative_path, size_bytes, sha256}]
contract:
  profile_id, adapter_id, adapter_version, structure_schema_version, topology_fingerprint
profiles:
  primary / guardrail: {input_spec, kwargs_spec, sample_set_id, sample_set_sha256}
reference:
  artifact_id, artifact_sha256, runtime_mode, executed_engine
measurement:
  timing_method, synchronization, warmup, iterations, outlier_rule, quantile_rule
  raw_samples_path, raw_samples_sha256, AB_BA_order
acceptance:
  metric_path, scope, unit, direction, aggregation, threshold, rationale
environment:
  gpu, compute_capability, driver, CUDA, PyTorch, optional_dependencies
```

freeze 文件不得包含访问令牌,私有 prompt 原文或未获授权的权重内容. 对无法公开的输入,保存具备访问控制的样本集标识与不可逆摘要,并在记录中写明重放所需的授权路径. 修订 freeze 只能创建新文件并通过 `supersedes_freeze_id` 建立关系, 不得原地修改已经被 candidate report 引用的阈值或 raw sample checksum.

---

## 5. 交付物与度量准则

各维度验收必须提供确凿的客观证据, 不得以主观声明或局部中间结果作为完成依据:

| 维度 | 必须交付的证据 | 不足以算完成的结果 |
| --- | --- | --- |
| **状态正确性** | 失败注入回归 (reject, exception, revert), 证明 context.model, 配置与历史完全恢复原状 | accepted 字段为 false, 但当前模型已被原地就地篡改 (in-place mutation) |
| **语义一致性** | 逐模块存储与计算合同 (Module Quant Contract), 覆盖各层 dtype/layout/kernel, 明确 no-op 原因 | 仅有一份模型级 summary 文本或首层 Linear shape 冒充整模 |
| **数值正确性** | 同输入下的 Output Diff 报告 (包含 mean_abs, max_abs, cosine_similarity), 明确容差标准 | 单个算子单元测试通过, 但整模 forward 输出出现 NaN 或严重精度坍塌 |
| **性能收益** | 成对 (Paired) 采样基准测试, 给出明确配置 (Batch, Shape), p50/p95 延迟, 吞吐与稳态显存 | 仅引用最快子算子的微基准加速比, 或挑选单次最好成绩作为结论 |
| **质量证据** | 外部评测系统对当前特定 Checkpoint / Artifact 的实测质量分数 (如 PPL, Task Accuracy) | 宣称"理论无损", 或直接用 Tensor 误差估算下游生成质量 |
| **可交付性** | 独立 Python 进程仅依赖导出的产物 (quant.json + weights) 成功加载并复现推理结果 | 产物虽然写出, 但加载必须依赖原 Session 上下文或未经持久化的内存变量 |
| **可维护性** | 单一权威源注册表, 清晰的模块分层, 完备的单元测试覆盖, 随行事实文档同步更新 | 代码行数静态减少, 但各模块之间仍然存在隐式循环依赖与反向引用 |

### 性能测量与报告规约

1. **时钟分离**: CUDA Event 测得的 Pure GPU Kernel 时间与同步后的 Wall-clock 时间必须分列, Profiler 挂载下的开销不得混入正式 Benchmark 数据;
2. **冷热启动分离**: 必须明确区分 Checkpoint 加载, JIT 编译, 权重 Prepack, CUDA Graph Capture 阶段的开销与 Steady-state 稳态推理耗时;
3. **架构对称性**: 比较加速比时, Baseline 与 Candidate 必须在同等硬件环境, 相同输入 Tensor 与相同编译/图优化模式下比对 (禁止用 Eager Baseline 对照 Graph Candidate 并宣称全归功于量化算子);
4. **路径区分**: PyTorch 原生低比特推理路径与 ONNX / TensorRT dense lowering 路径必须独立测试与报告, 前者证明 low-bit runtime 加速, 后者证明部署导出兼容性.
5. **配对样本不可混算**: 每个 profile 的 baseline 与 candidate 必须共享输入和样本编号, 原始延迟序列要能关联到同一轮次. 首选报告每 profile 的 p50/p95 和 `speedup = baseline_p50_ms / candidate_p50_ms`; 不得把不同 profile 的 speedup 算术平均后冒充主结论. 多 profile 的总通过规则默认取最差 profile, 只有在 freeze 中预先声明权重和聚合公式时才可另行汇总;
6. **测量顺序防偏差**: 同一进程内比较时, 要预先固定或随机化 AB/BA 测量顺序并记录顺序. 任一侧重新编译,触发缓存失效,发生 OOM 或改变时钟频率时, 本轮成对样本失效, 必须单列原因并重新测量;
7. **结论分级**: `materialized` 只能说明实现已落入模型或 artifact. 只有同时提供有效 `observed` 原始样本, 数值差异和实际 fallback 记录, 才能声称指定 profile 上的性能结论. `not_decided` 必须保留, 不可合并进 accepted 平均值.

---

## 6. 明确不做的事项 (Non-goals)

为防止架构膨胀并确保力量集中于关键主线, 本轮优化严禁跨越以下红线:

1. **不引入第二套编排机制**: 严格收敛于 Session 与 YAML workflow 既有执行链, 严禁新增第三种编排器或 DSL;
2. **不接管训练与微调职责**: XQT 不做 QAT, 不做 LoRA/Finetune 微调, 不做知识蒸馏 (KD), 不做训练期 Recovery, 不引入 Optimizer 与 Loss 计算;
3. **不构建数据服务与 Serving 系统**: XQT 不做通用的 Dataset / DataLoader 构建, 不做多并发 Batching 调度, 不做 HTTP/gRPC Serving 服务框架;
4. **不搞无约束的全库大搬家**: 不单纯为了追求目录视觉整齐而做跨包大重构, 不做无实质收益的文件拆分与重命名;
5. **不建设宽泛的通用自动搜索平台**: 暂不开发跨几千种超参的大规模 NAS 或自动策略搜索, 聚焦于已有算子候选 (Acceptance) 的确定性度量与择优;
6. **不预设单 Kernel 极致融合**: 不要求将整个 TransformerBlock 强行写为单个 Megakernel, 优先以已有成熟算子组合和 Block-level 编译解决瓶颈, 由实测 Profiler 证据驱动深度优化.

---

## 7. 执行机制与文档维护

1. **原子推进**: 每次实施必须精准锁定一个任务 ID, 先固化失败回归测试, 编写小范围实施方案, 再执行代码与文档改动;
2. **唯一状态源**: [XQT 改进目标与验收](xqt-improvement-goals.md) 是任务状态, 依赖关系与验收记录的**唯一法定状态源**, 本路线图仅维护宏观方向与里程碑定义;
3. **闭环标准**: 任务从 `pending` 变更为 `done` 之前, 必须同时具备: 缺陷回归测试通过, 必选硬件测试执行, 事实文档 (`xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md`) 同步更新, 知识图谱刷新与标点检查通过;
4. **决策公开**: 涉及关键架构分歧与参数门槛冻结时, 必须在 goals.md 的 [待冻结决策表](xqt-improvement-goals.md#6-待冻结决策) 中记录决策日期, 依据与权衡考量.

---

### 文档关联索引

- [改进目标与验收](xqt-improvement-goals.md): 17 项任务的详细目标, 源码依据, 依赖与完成标准.
- [包内工程契约](../../../xqt/FRAMEWORK.md): XQT 内部核心抽象, 模块边界与架构事实.
- [XQT 架构总览](xqt.md): XQT 面向开发者的宏观架构说明.
- [设计债台账](xqt-design-debt.md): 既有历史技术债务与处置记录.
- [架构矫正长期指导](xqt-realignment-guide.md): 架构矫正的历史背景与长期设计原则.
- [Block Runtime Optimization](xqt-operator-block-optimization.md): 算子与 Block 优化的准入边界规范.
