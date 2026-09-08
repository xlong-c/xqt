# XQT 改进目标与验收

本文承接 [XQT 改进路线图](xqt-improvement-roadmap.md), 定义可逐项落地实施的详细目标, 源码依据, 任务依赖, 回归测试方案与完成验收标准. 本文不描述当前全部能力, 不定义已实现的新 API, 不替代源码和既有工程契约.

制定日期: 2026-09-05. 最近审阅: 2026-09-06. 初始状态: 所有 17 项任务均为 `pending`. 本文将规划基线和后续实施状态严格分离: 未附完整测试, 硬件和 artifact 证据的 worktree 改动不改变任务状态. 本文是这批改进的唯一法定任务状态源; 长期保留目标与验收理由, 临时执行日志与大文件结果留在 `research/` 或既有 artifact 目录.

---

## 1. 阅读与使用准则

1. 先阅读 [路线图](xqt-improvement-roadmap.md), 理解 6 个里程碑的演进逻辑, 再从任务总表选择已满足全部前置依赖的任务开展实施;
2. 实施前必须重新核对当前源码与实测证据, 把缺陷复现固化为确定性失败测试. 不把历史行号或静态图谱边当成当前事实;
3. 严格按任务定义的边界形成实施方案. 文中出现的架构概念与数据结构名称是设计约束, 不是要求机械添加同名冗余字段或过度包装;
4. 完成代码实现, 回归测试通过, 并在对应事实文档 (`xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md` 等) 同步更新后, 记录完整证据, 方可将状态更新为 `done`;
5. 所有任务默认要求函数类型注解, optional dependency 采用 lazy import 机制, 抛出明确命名的 XQT 异常类型, 保证 Session 与 YAML 入口语义严格对称. 严禁新增平行配置链, 训练/微调基础设施或独立 Serving 调度器.

---

### 1.1 状态, attempt 与证据的更新规则

任务表只记录长期实施状态: `pending`, `in_progress`, `blocked`, `done`, `deferred`. `accepted`, `rejected`, `failed`, `not_decided`, `interrupted` 是一次 stage attempt 或性能实验的结果, 绝不是任务状态. 一个任务可以拥有多个 `rejected` attempt, 仍保持 `in_progress`; 任何一个 accepted attempt 也不能绕过尚未满足的依赖和完成标准.

| 记录对象 | 最小标识 | 可以修改什么 | 不可以修改什么 |
| --- | --- | --- | --- |
| 任务 | `XQT-NNN` | 本表状态, 依赖, 正式完成记录 | 已冻结的 baseline, 其他任务状态 |
| stage attempt | `attempt_id + stage_name + source_stage` | 临时 metrics, 诊断 report, 拒绝原因 | accepted DAG, current artifact, 正式 recipe 历史 |
| 发布 artifact | artifact ID + manifest/sidecar checksum | 在校验完成后成为可引用产物 | 既有已发布目录中的文件内容 |
| benchmark run | freeze version + profile ID + raw-samples checksum | 对该 profile 的 observed 证据 | 其他 profile 的阈值或全模型结论 |

状态从 `pending` 进入 `in_progress` 前, 记录源码基线 commit, worktree 摘要和预期回归. 进入 `blocked` 时, 写明缺失的资源或决策, 已验证的替代路径和恢复条件. 进入 `done` 时, 必须在本文件第 8 节追加完整记录, 同一次改动原子更新任务表和事实文档. 不允许根据口头结论, 未提交 patch, 单条 benchmark 或无法定位输入的截图变更状态.

### 1.2 Attempt 记录, 密封与中断边界

每次运行 stage 前都必须分配新的,不可复用的 `attempt_id`. 推荐使用 UUID 或等价的全局唯一标识, 不能用易重复的 stage 名或时间戳作为唯一键. 一个 attempt 的内部生命周期为 `created -> running -> sealed`; `sealed` 后的 `result` 只能是 `accepted`, `rejected`, `failed`, `not_decided` 或 `interrupted`, 不能被下一次同名重试覆盖. 这套生命周期是实施目标, 不表示当前所有 report 均已具备这些字段.

每个密封 attempt 最少应保存如下机器可读记录. 字段可拆在 report, manifest 和 research artifact 中, 但必须能由 `attempt_id` 相互定位:

| 字段组 | 必填字段 | 目的 |
| --- | --- | --- |
| 身份 | `attempt_id`, `stage_name`, `stage_kind`, `source_stage`, `source_model_fingerprint` | 证明候选从哪个已接受模型派生, 防止同名 stage 覆盖历史 |
| 时间与环境 | `started_at`, `ended_at`, process / host 标识, device 和依赖摘要 | 区分不同环境或中断后的重试 |
| 请求与计划 | request 摘要, resolved contract / plan 摘要, fallback policy | 区分调用方意图与实际选择 |
| 结果 | `result`, `reason_code`, 人类可读 reason, exception type / 摘要 | 让配置错误,执行失败,证据不足和验收拒绝可区分 |
| 影响范围 | candidate artifact 临时路径,正式 artifact 引用, metrics / contract 摘要 | 证明哪些内容可提交,哪些内容只能诊断 |
| 证据 | report / raw-samples / profiler 文件相对路径与 SHA256 | 让结论可被后续复核, 而非依赖控制台输出 |

Session 内存状态不承诺在进程被强制终止,主机掉电或解释器崩溃后继续存在. 此类情况必须记录为 `interrupted`, 而不是伪造为 `rejected` 或 `accepted`; 下一次启动可把未密封 attempt 标记为不可恢复并创建新 attempt. XQT-005 的发布协议仍必须保证最终 Quant Pair 对读者只呈现完整旧包或完整新包. 对遗留 staging / backup 目录, 恢复流程只能列出,校验并由显式维护操作处理, 不得在未验证其所有权时自动删除用户文件.

### 1.3 文档同步矩阵

规划文档只定义后续目标, 不能代替当前事实文档. 每个变更包在实现完成时, 应按下表检查需同步的页面; 未涉及的行在完成记录中写明 `not_applicable` 和理由.

| 变化类别 | 必须同步的事实文档 | 需要时同步 | 不应提前更新为已实现 |
| --- | --- | --- | --- |
| Session stage, lineage, workflow schema | `xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md` | `docs/md/usage/xqt-workflows.md`, 公共 API 边界 | `docs/md/XQT.md` 的现状能力段与 HTML 阅读页 |
| Quant / runtime contract, calibration 语义 | `xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md` | `docs/md/explanation/xqt-concepts.md`, `docs/md/XQT.md` | 任何完整模型性能或质量结论 |
| Quant Pair 格式, loader 或 trust boundary | `xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md` | 使用层文档和 package 示例 | 将 `.pt` / pickle 写成可安全读取不可信输入 |
| Benchmark, acceptance, target freeze | 本文和路线图中的规划段 | `research/` 的原始证据与冻结记录 | 在未完成 L3 前把 synthetic / 局部结果写成整模事实 |
| 新模型 profile / adapter / 导出路径 | `xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md` | `docs/md/XQT.md`, 使用层和 HTML 阅读页 | 将 `planned` 或 preflight 结果写成 executable 能力 |

---

## 2. 审阅依据与限制

- **审阅源码基线**: `9d254042f268e3bf5a9f9ab546fcf738c0b494a4`. 下列 `已确认` / `待验证` 结论均指向此初始审阅基线, 不是对之后 worktree 的永久断言. 后续实施必须记录实际测量所用 commit 和 worktree diff 摘要, 不清理或覆盖用户已有修改.
- **图谱与知识库状态**: 审阅时项目 `root-workspace-xdl` 状态为 `ready`, coverage generation 为 `2026-09-05T17:51:04Z`. 图谱用于定向发现, 不替代当前源码. 每个变更包都要记录其实际查询 generation,候选路径和 `check_index_coverage` 结果; 若 worktree 在生成后有改动, 必须阅读当前源码, 不能用旧图谱证明最新实现.
- **静态检查初始现状**: 初始审阅执行 `ruff check xqt` 时确认存在 2 项 F821 静态错误:
  - `xqt/compression/prune/discovery.py:270`: `_StructuredCandidate` 未定义;
  - `xqt/compression/prune/discovery.py:299`: `_StructuredCandidate` 未定义.
- **测试套件初始现状**: 初始审阅执行 `pytest -q tests/xqt --collect-only` 收集到 1625 项测试用例. 历史全量测试结果记录为 1620 passed, 5 skipped (涉及未构建的 W4A16 artifact, optional HF checkpoint fixture 和未开启或未满足条件的 golden CUDA / TensorRT 测试). 环境具备 NVIDIA GeForce RTX 4070 Ti SUPER (`sm_89`). XQT-001 负责在实际实施基线上重跑并生成正式的测试分层报告; 历史数量不构成当前通过结论.

### 证据分类标准

为了杜绝主观臆断并区分已确证事实与潜在风险, 全文证据严格划分为以下三类:
1. `已确认`: 审阅时已编写最小用例复现, 或当前源码逻辑能够直接确认的行为与缺陷;
2. `待验证`: 依赖特定失败注入, 别名冲突或特定硬件支持矩阵的潜在风险, 在未构造出确定性复现前不能直接当作已证明的运行缺陷;
3. `目标`: 本轮计划要求建立的规范行为或新增契约, 不暗示当前完全没有相关代码原型.

---

### 四层证据与产物溯源

任何会影响模型,执行引擎,性能结论或发布产物的 stage, 必须将下列四层事实分别记录. 它们不可互相替代, 也不得只以顶层字符串总结替代下层实证.

| 层级 | 含义 | 最低记录内容 | 不能证明什么 |
| --- | --- | --- | --- |
| `request` | 调用方要求的策略和约束 | recipe / Session 参数, 目标硬件, profile 名称, allow/deny fallback 策略 | 不证明引擎可用, 也不证明任何变换已经发生 |
| `resolved` | 经过契约,能力和环境检查后选择的计划 | model / module contract, 选择理由, 被排除候选与原因, 版本化的 dispatch plan | 不证明计划已经 materialize 或实际执行 |
| `materialized` | 已经写入模型,图或 artifact 的具体实现 | 逐模块替换清单, 实际 storage/layout, artifact checksum, graph cache key 或 export bundle | 不证明 forward 走过该路径, 也不证明有性能收益 |
| `observed` | 在冻结环境和输入上实际测得的运行结果 | executed engine / kernel, fallback 次数与原因, raw timing samples, 输出差异, 显存与质量证据 | 不得外推到其他模型,shape,硬件或外部 Pipeline |

若任一层缺失, stage 的结论只能是 `not_decided` 或诊断信息. `planned`, `metadata_only`, package import 成功,artifact 文件存在,CUDA Graph capture 成功均最多属于前 3 层, 不属于 `observed` 性能或执行证据.

每个可发布 artifact 或可比较 benchmark 最少携带以下 provenance 字段. 字段可位于 manifest,report 与外部原始数据之间, 但必须可相互索引且稳定定位:

| 范畴 | 最低字段 |
| --- | --- |
| 源码与模型 | `git_commit`, worktree diff 摘要, repo ID, revision, 权重文件清单与 SHA256, model/profile/contract schema version |
| 输入与执行环境 | profile ID, tensor shape/stride/dtype/device, kwargs 规范化摘要, seed 或可复现样本引用, GPU/driver/CUDA/PyTorch/关键可选依赖版本 |
| 路由与结果 | request/resolved/materialized/observed 四层记录, fallback policy 与实际 fallback, stage attempt ID, parent stage / artifact checksum, accepted/rejected/not_decided 原因 |
| 测量与文件 | timing 方法,warmup 和采样数,raw samples 文件 checksum,单位,阈值版本, 所有发布文件的相对路径,大小,SHA256 和 schema version |

输入不能公开保存时, 可以保存受保护的样本索引或不可逆摘要, 但必须足以由拥有权限的评测方定位同一输入集. 不得为省略 provenance 而把外部质量分数写成无法关联模型产物的自由文本.

### 证据失效与结果分类

实现必须区分配置错误, 运行失败和不通过验收, 不能把它们都折叠成 `accepted=False`:

| 情形 | 例子 | 对 stage / artifact 的结果 | 调用方可见行为 |
| --- | --- | --- | --- |
| 配置或契约无效 | 未知 transform, 缺失 static scale, 路径逃逸, 非法单位/聚合名 | 不执行候选变换, 不生成 artifact | 抛出明确的 XQT 配置/量化/产物异常 |
| 执行失败 | CUDA OOM, runner 异常, sidecar 写入失败, report 序列化失败 | 回滚 source, 清理暂存, 留下失败诊断 | 原异常带 attempt/source 上下文后重新抛出 |
| 证据不足 | 缺少 reference benchmark, 输入 profile 不匹配, raw samples 不完整 | `not_decided`, 不允许 commit | 明确列出缺少字段, 不伪造阈值比较 |
| 验收拒绝 | 数值, 显存, fallback 或性能未达冻结门槛 | `rejected`, source 保持 current | 保存只读诊断, 不更新 accepted DAG 或产物 |
| 验收通过 | 四层证据完整且全部门槛通过 | `accepted`, 才可进入事务 commit | 记录 report/manifest/provenance, 允许后续引用 |
| 进程中断 | 进程被杀死,解释器崩溃,主机掉电或无法确认 commit 是否完成 | `interrupted`, 不得推断为 accepted / rejected | 下次启动通过可验证 artifact 判断发布状态, 未密封 attempt 只能作为诊断 |

`observed` 中出现 NaN, Inf, 无法解析的单位, 布尔值伪装的数值, 负时延/显存或无法关联 reference 的 speedup 时, 不得自动转换, 截断或填默认值. 这类数据要么是输入契约错误, 要么使该 attempt 进入 `not_decided`/`rejected`; 实现必须在 report 中明确采用的分支和原始值.

---

## 3. 任务总表与关键路径

优先级说明:
- `P0`: 核心状态正确性, 产物安全性或验收可信性缺陷 (阻断性风险);
- `P1`: 构成核心端到端闭环的关键能力链路;
- `P2`: 架构治理, 注册表收敛与可复用性扩展.

任务状态词表与更新规则见 [1.1 状态, attempt 与证据的更新规则](#11-状态-attempt-与证据的更新规则). 状态仅在本表唯一维护; 第 8 节只追加证据记录, 不维护第二份状态.

| ID | 任务名称 | 优先级 | 里程碑 | 核心关注点 | 额外依赖 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| [XQT-001](#xqt-001) | 工程检查与可复核基线 | P1 | M0 | 消除 ruff F821, 建立三层测试分层与基线记录 | 无 | done |
| [XQT-002](#xqt-002) | Stage 事务与失败隔离 | P0 | M1 | Prepare-Execute-Validate-Commit 隔离回滚机制 | XQT-001 | done |
| [XQT-003](#xqt-003) | 当前模型身份与 lineage | P0 | M1 | Current / Baseline / Best 状态机与防自指 DAG | XQT-001, XQT-002 | done |
| [XQT-004](#xqt-004) | 校准模式与状态隔离 | P0 | M1 | 逐模块 training 状态恢复与 Hook 清理保护 | XQT-001 | done |
| [XQT-005](#xqt-005) | Quant pair 安全与完整发布 | P0 | M1 | 路径防逃逸与 Staging 两阶段原子发布 | XQT-001 | done |
| [XQT-006](#xqt-006) | Quant 请求与实际结果一致 | P0 | M1 | 静态量化缺 scale 显式失败, 杜绝静默降级 | XQT-001, XQT-002, XQT-004 | done |
| [XQT-007](#xqt-007) | 模块级量化合同与 no-op 语义 | P1 | M1 | 模块级合同架构, 解决首层 shape 局部失真 | XQT-001, XQT-006 | done |
| [XQT-008](#xqt-008) | 候选能力与可执行性分离 | P1 | M2 | 拆分静态查询与可执行解析, SM 算力严检 | XQT-001, XQT-007 | done |
| [XQT-009](#xqt-009) | 指标作用域与 acceptance | P0 | M1 | 显式路径定位与聚合规则, 根除盲目取 max | XQT-001 | done |
| [XQT-010](#xqt-010) | 模型结构契约主链接线 | P1 | M2 | 统一直接注入与 adapter 管道, 拓扑失效更新 | XQT-001, XQT-003, XQT-007 | done |
| [XQT-011](#xqt-011) | Registry 权威源与分层收敛 | P2 | M2 | kernels/registry 与 ops/gemm 职责清晰划分 | XQT-001, XQT-008 | done |
| [XQT-012](#xqt-012) | Typed graph rewrite 与 pattern | P1 | M3 | 强类型图变换, dequant_gemm / norm 等初始模式 | XQT-001, XQT-002, XQT-010 | done |
| [XQT-013](#xqt-013) | 通用 block materialize 与 runtime | P1 | M3 | 消除 wrapper 开销, 规范 CUDA Graph 缓存 | XQT-001, XQT-008, XQT-009, XQT-011, XQT-012, XQT-014 | done |
| [XQT-014](#xqt-014) | 真实模型 baseline 与门槛冻结 | P1 | M3 | FLUX.2 Klein 4B 资源核验, 测定并冻结基线 | XQT-001, XQT-009 | done |
| [XQT-015](#xqt-015) | 真实模型优化与质量闭环 | P1 | M4 | 端到端全链路优化, 消融实验与整模达标 | XQT-001, XQT-003, XQT-004, XQT-007, XQT-013, XQT-014 | done |
| [XQT-016](#xqt-016) | Artifact 重载与实际部署验收 | P1 | M4 | 独立 Python 进程重载产物, 实际推理与报告 | XQT-001, XQT-005, XQT-008, XQT-015 | done |
| [XQT-017](#xqt-017) | 第二同族模型复用验证 | P2 | M5 | 第二模型接入, 检验结构契约通用化程度 | XQT-001, XQT-016 | done |

### 关键路径与推进顺序

```text
[XQT-001] (工程基线)
    │
    ├──► [XQT-002] (Stage事务) ──► [XQT-003] (Lineage与身份)
    │                                  │
    ├──► [XQT-004] (校准隔离) ──────────┼──► [XQT-006] (量化一致性) ──► [XQT-007] (模块级合同)
    │                                  │                                    │
    ├──► [XQT-005] (产物安全发布)        │                                    ├──► [XQT-008] (能力分离) ──► [XQT-011] (注册表收敛)
    │                                  │                                    │                                 │
    └──► [XQT-009] (指标作用域) ────────┴────────────────────────────────────┼──► [XQT-010] (结构契约接线)      │
               │                                                            │           │                     │
               │                                                            │           ▼                     │
               │                                                            └──► [XQT-012] (图变换)           │
               │                                                                        │                     │
               ▼                                                                        ▼                     ▼
         [XQT-014] (真实模型基线冻结) ──────────────────────────────────────────► [XQT-013] (Block Materialize / Runtime)
               │                                                                        │
               └──────────────────────────────────────┬─────────────────────────────────┘
                                                      ▼
                                                [XQT-015] (真实模型优化与端到端闭环)
                                                      │
                                                      ▼
                                                [XQT-016] (产物独立重载与部署验收)
                                                      │
                                                      ▼
                                                [XQT-017] (第二同族模型复用验证)
```

除了 XQT-001 外, 所有任务默认必须在 XQT-001 建立的稳定工程基线之上开展. XQT-004, XQT-005, XQT-009 可与主干并行实施. XQT-014 的模型资源核验应尽早推进.

---

### Stage 状态机不变量

XQT-002 与 XQT-003 必须共同实现下面的状态边界. 表中 "恢复 source" 是恢复执行前已接受父 stage 的模型拓扑,参数,契约,正式配置历史和当前 artifact 引用, 不只是将 `context.model` 指针改回去.

| attempt 阶段或结果 | candidate model | `current_stage` / accepted DAG | 正式 recipe / 配置历史 | 已发布 artifact | 允许保留的诊断 |
| --- | --- | --- | --- | --- | --- |
| `prepare` | 从固定的 `source_stage` 建立隔离候选或可回滚快照 | 不变. source 必须已在 accepted DAG 中 | 不变 | 不变 | preflight 计划和资源估计 |
| `execute` | 仅 candidate 可被原地修改 | 不变 | 不变 | 不变 | 临时 metrics,候选文件和执行日志 |
| `accepted` + `commit` | candidate 成为新 stage 的模型快照 | 新 stage 原子加入 DAG 并成为 `current_stage`; `best_stage` 仅按显式规则更新 | stage 配置原子追加 | 仅已校验的 artifact 可引用为当前产物 | 完整 report 与 provenance |
| `rejected` | 丢弃 candidate 并恢复 source | 不变, 不创建 accepted child | 不变 | 不变 | 可保留只读 attempt report, 标明拒绝阈值和原因 |
| `failed` / exception | 清理 candidate 并恢复 source | 不变, 不留下半成品 node | 不变 | 不变 | 异常摘要,已清理临时路径和可重试标识 |
| `observation` | 只读使用 `current_stage` 的模型 | 不变. benchmark / analyze 不产生新的模型版本 | 不变 | 不变, 除非它只发布只读观测报告 | benchmark / analysis report |
| `use(stage)` / `revert_to(stage)` | 从指定已接受 stage 重建或恢复模型 | `current_stage` 原子切换到指定 stage; `best_stage` 不隐式充当 parent | 不修改历史 | 不修改发布内容 | 切换事件与 source checksum |

所有 accepted stage 的 parent 必须在创建前已经存在, 不能为自身, `baseline_stage` 永不被覆盖. 不变量测试必须比较模型拓扑清单,参数/关键 buffer 摘要,结构契约,正式历史和 artifact 引用; 对可变对象, 仅比较 Python 对象身份或 `state_dict` 入口数不足以证明回滚正确.

事务快照的最低边界是: `context.model`, `structure_contract`, `metrics`, `artifacts`, manifest 可变字段, benchmark cache, `current_stage`, accepted DAG 与正式 recipe history. 每个字段要么恢复到 source 的等价值, 要么明确定义为只读的 attempt 诊断. 需要写文件的 stage 只能把临时路径挂在 attempt 范围, 由 XQT-005 的发布协议在 commit 后替换最终产物; stage 回滚不能通过删除一个已引用的旧 artifact 来实现.

当 report 序列化, manifest 更新或 acceptance 计算本身抛错时, 其处理与 runner 失败相同: 不创建 accepted child, 不追加正式配置, 不推进 `current_stage`, 并清理 candidate 及其暂存产物. 重试同名 stage 前必须确认失败 attempt 已结束; 新 attempt 不得复用或覆盖旧 attempt 的 report key.

回滚测试需要使用语义快照, 而不是比较 Python 对象 identity 或只比较 `state_dict` 的 key 数. 快照至少包含: `named_modules()` 的路径/类型/关键 shape, parameter 与 buffer 的 dtype/device/值摘要,序列化后的 structure/runtime contract,正式 metrics 与 artifact 引用, accepted stage DAG, `current_stage` 和正式 recipe history. 可以排除 attempt ID,时间戳,临时目录名称和 profiler handle 等刻意短生命周期字段, 但排除项必须在测试旁明确列出. 任何无法恢复的外部副作用都必须在 stage request 中显式禁止或在执行前失败, 不能通过放宽快照比较来掩盖.

### 测试证据分层

每项任务都要先选择与风险匹配的最小测试层, 再按任务类型追加更高层证据. 测试名称,环境变量与 skip 条件应进入 XQT-001 的执行规约, 不在本文预先假定具体 marker 或环境变量名称.

| 层级 | 适用任务 | 必须证明 | 最低要求 |
| --- | --- | --- | --- |
| L0: 确定性单元测试 | XQT-001 至 XQT-012 | schema,错误分支,回滚,路径安全和聚合规则 | CPU 可运行, 覆盖正常,拒绝和异常三个方向 |
| L1: 进程内集成测试 | XQT-002 至 XQT-012 | Session 与 YAML 的语义对称, report / manifest / artifact 传递正确 | 使用小型真实结构或受控 fixture, 但不得将其当真实模型性能验收 |
| L2: 目标硬件测试 | XQT-008, XQT-013 至 XQT-016 | 引擎确实可 materialize 并在目标 GPU 上执行, CUDA Graph / cache 行为正确 | 明确 opt-in 条件, 不满足时记录 skip 原因; 作为相应里程碑 exit gate 时必须实际执行 |
| L3: 真实模型与独立进程验收 | XQT-014 至 XQT-017 | checkpoint,输入 profile,端到端 forward,外部质量与重载交付 | 只接受冻结模型,输入和环境的成对测量; 无权重或硬件时标记 `blocked` |

对性能结论, L0/L1 只能验证逻辑与测量工具, L2 只能证明指定硬件和 scope 的运行路径, L3 才能支撑首选真实模型的整模结论. 任何 optional backend 的 skipped 测试均不能被计入该 backend 的通过数.

---

## 4. 详细任务定义

<a id="xqt-001"></a>

### XQT-001. 工程检查与可复核基线

- **源码依据**:
  - `已确认`: 运行 `ruff check xqt` 报错 `F821 Undefined name '_StructuredCandidate'`, 位于 [discovery.py](../../../xqt/compression/prune/discovery.py) 第 270 行 (`def _candidate_touches_protected(candidate: _StructuredCandidate, ...)`) 与第 299 行 (`kept: list[_StructuredCandidate] = []`), 属于未定义私有类型的悬空引用;
  - `已确认`: 当前 `pytest tests/xqt` 收集 1625 项测试, 历史记录存在 5 项 skip, 但缺乏统一的环境标记分层与执行脚本, 导致本地硬件测试覆盖范围模糊.
- **目标与范围**:
  - 修复 `discovery.py` 中的类型引用错误, 修正为正确的已定义类型或补充必要的数据类定义, 不使用 `# noqa` 或全局忽略配置掩盖问题;
  - 制定并固化测试套件三层执行规约:
    1. **CPU 必跑基线**: 覆盖 contracts, schema, reporting, configs, 离线算法, 不依赖 CUDA 硬件;
    2. **SM89 必跑基线**: 覆盖 W8A8 MMA, FP8, INT8 Prepack, TileLang/Triton 算子与 Block 运行;
    3. **Optional Backend 门禁**: 覆盖 TensorRT, ONNX Runtime, CuTe DSL, CuTile 等依赖外部库的集成测试;
  - 建立明确的测试环境变量控制机制 (如 `XQT_TEST_CUDA=1`, `XQT_TEST_TRT=1`), 明确区分"因环境无硬件而跳过"与"必选测试通过".
- **设计考量与不变式**:
  - 静态检查零告警: `ruff check xqt` 必须完全通过. `mypy xqt` 只有在本任务明确引入可执行的仓库级配置,依赖桩和已声明检查范围后, 才能作为额外门禁; 不能把尚未建立的全库 mypy 基线伪装成 M0 硬要求;
  - 门禁严肃性: 里程碑验收时, 目标硬件相关的测试用例不得被静默 skip, 任何 skip 项必须在日志中记录明确原因.
- **验收与测试要求**:
  - 运行 `ruff check xqt` 退出码为 0;
  - 运行 CPU 必跑测试集与 SM89 必跑测试集, 产出完整的测试日志 (记录源码 commit, 实际 Python/PyTorch/CUDA/driver 版本, GPU, passed/skipped 统计);
  - 对 5 项历史 skip 逐一复查并分类归档 (标记为缺少权重, 缺少 TRT 环境或预期行为).
- **非目标**: 不做全库代码风格大重构, 不强制在无对应硬件的环境下安装全部第三方专有运行时.

---

<a id="xqt-002"></a>

### XQT-002. Stage 事务与失败隔离

- **源码依据**:
  - `已确认`: 在 [session_runner.py](../../../xqt/workflows/session_runner.py) 的 `run_optimization_stage()` 中, 第 323-345 行直接调用 `runners.prune(...)` 或 `runners.quant(...)`, 这些 runner 对 `state.context.model` 进行了原地 (in-place) 结构或参数修改;
  - `已确认`: 第 360-362 行的回滚逻辑仅当 `not accepted and stage.revert_on_reject and stage.from_stage is not None` 时才调用 `restore_stage_model()`. 若 `revert_on_reject` 为 False (默认) 或 `from_stage` 为 None, 即便 stage 被验收拒绝 (`accepted=False`), 原始模型依然保留了被破坏的结构或参数, 已接受的 stage 列表中虽然只有 baseline, 但内存中模型已被篡改;
  - `已确认`: 在 [optimization.py](../../../xqt/workflows/optimization.py) 的 `run_stage()` 第 524 行, `self._state.config.stages.append(stage)` 在 `_run_optimization_stage` 之前执行, 若执行抛出未捕获异常, 该失败 stage 会残留在已配置列表中, 阻塞后续重试.
- **目标与范围**:
  - 引入 `Prepare -> Isolated Candidate Execution -> Validate/Acceptance -> Commit or Rollback` 的四阶段事务机制;
  - 当 stage 被拒绝 (`accepted=False`) 或抛出异常时, `state.context` 下的 `model`, 结构契约, 配置状态, 指标历史与产物引用必须无条件恢复到该 stage 执行前的纯净状态;
  - 失败或被拒绝的 stage 记录可以追加独立的诊断报告与 attempt 日志, 但严禁将脏模型提交为当前已接受模型;
  - 修正 `run_stage()` 中的配置追踪逻辑, 仅在 stage 成功通过 acceptance 并且完成 commit 时才追加到正式历史中, 失败重试不应受阻.
- **设计考量与不变式**:
  - **大模型快照策略**: 4B+ 级别大模型无法进行盲目无节制的 `deepcopy` (可能导致 16GB 显存 OOM). 需明确定义快照等级:
    - 参数级修改: 暂存 CPU state_dict 或仅备份被量化层;
    - 拓扑级修改 (Prune / Rewrite): 备份子模块引用关系或拓扑图, 支持逆向子模块替换;
    - 不可深拷贝资源 (CUDA runtime contexts, live graph handles): 显式声明生命周期与清理策略;
  - **事务提交原子性**: 仅当 Stage 逻辑执行成功, 产物生成成功, 报告序列化成功, Acceptance 校验通过四者同时满足时, 才执行原子 Commit.
- **验收与测试要求**:
  - 编写失败注入测试: 针对 `prune` 和 `quant` 阶段分别模拟 `accepted=False`, 断言 `context.model` 的结构 (named_modules, shapes) 与权重数值与执行前完全一致;
  - 模拟 runner 中途抛出 `RuntimeError`, 断言 `context.model` 成功恢复, 且可以在同一 session 中立即重试新 stage;
  - 断言失败 attempt 的中间临时产物不被误发布为当前有效模型.
- **非目标**: 不支持恢复由用户自定义函数造成的外部操作系统副作用 (如随意删除文件或外部网络请求).

---

<a id="xqt-003"></a>

### XQT-003. 当前模型身份与 lineage

- **源码依据**:
  - `已确认`: 在 [optimization.py](../../../xqt/workflows/optimization.py) 的 `use(stage_name)` 与 `revert_to(stage_name)` (第 510-519 行) 中, 代码仅执行了 `self._state.context.model = restore_stage_model(...)`, 完全没有更新当前模型身份指针 (`current_stage`), 且保留了历史 `best_stage`;
  - `已确认`: 在 [session_runner.py](../../../xqt/workflows/session_runner.py) 第 267-269 行与第 356-358 行, 来源推断逻辑为:
    `source_stage_name = stage.from_stage or state.best_stage or state.baseline_stage or "baseline"`
    当用户执行 `session.use("baseline")` 后, 下一个 stage 若未显式传入 `from_stage`, 其来源会被错误推断为历史最优的 `best_stage` (例如已被修剪的模型), 导致 Lineage DAG 父子关系彻底错乱;
  - `已确认`: 在 `run_optimization_stage()` 第 360 行将 `state.best_stage = stage.name` 赋值后, 紧接着在第 404 行调用 `_record_stage_report()`, 该函数内部重新推断 `source_stage_name`, 极易将新生成的 `best_stage` 作为自身的父节点, 产生致命的报告"自指 (self-reference)"缺陷.
- **目标与范围**:
  - 显式定义清晰的模型状态机, 严格分离四个核心概念:
    1. `baseline_stage`: 会话初始的模型状态;
    2. `current_stage`: 当前正在操作和前向演进的模型身份 (调用 `use()` 切换此身份);
    3. `best_stage`: 历史中按显式指标准则排序评选出的最优阶段 (纯只读查询视图, 不隐式作为父节点推断源);
    4. `observation_stage`: 纯观察阶段 (如 `benchmark`, `analyze`), 执行后不推进模型版本, 不改变 `current_stage`;
  - 在每个 stage 执行前固定 `source_stage_name = stage.from_stage or state.current_stage`, 贯穿整个执行过程, 杜绝在执行后重新推算;
  - 修复 `_record_stage_report()` 中的自指漏洞, 确保 `report.lineage["from_stage"]` 真实反映其输入源.
- **设计考量与不变式**:
  - **DAG 拓扑一致性**: 任何新生成的 stage 其父节点必已存在于 `stages_by_name` 中;
  - **幂等切换**: `session.use(A)` 之后无论查询多少次, `current_stage` 恒为 `A`; 后续新建 stage 的 parent 必为 `A`.
- **验收与测试要求**:
  - 回归用例: 执行 `baseline -> stage_A -> session.use("baseline") -> stage_B`, 断言 `stage_B` 的 parent 为 `baseline` 而非 `stage_A`;
  - 检查生成的 `manifest.json` 与 stage reports, 断言没有任何 stage 的 `from_stage` 与自身同名;
  - 验证 `benchmark` 与 `analyze` stage 执行后, `current_stage` 保持前一个模型 stage 不变.
- **非目标**: 不在 XQT 内部构建完整的 Git 级版本控制引擎, 仅维持内存与产物中的确定性有向无环图 (DAG).

---

<a id="xqt-004"></a>

### XQT-004. 校准模式与状态隔离

- **源码依据**:
  - `已确认`: 在 [scale_artifact.py](../../../xqt/compression/quant/calibration/scale_artifact.py) 的 `run_calibration_batches()` (第 58-77 行) 中, 仅使用了 `with torch.no_grad():`, 完全缺失 `model.eval()` 状态隔离;
  - `已确认`: 若外部传入处于训练模式 (`model.training == True`) 的模型, 校准 forward 会导致 `BatchNorm2d` 持续更新其 `running_mean` 与 `running_var`, `Dropout` 会随机丢弃特征, 从而使激活统计严重失真并污染模型原始参数;
  - `已确认`: 仅在最外层执行 `model.eval()` 并在退出时执行 `model.train()` 是极其危险的, 因为大模型常包含部分冻结层或混合模式模块, 单一开关会永久破坏各子模块原有的差异化训练标志;
  - `待验证`: 第 137-174 行的 `handles` 列表虽然有 `finally` 块, 但在生成器或自定义异常路径下需确保完全释放, 防止内存 hook 泄漏.
- **目标与范围**:
  - 实现专用的临时推理校准上下文管理器 (如 `preserve_module_training_modes(model)`):
    1. 进入上下文时, 递归遍历并记录所有子模块的原始 `module.training` 布尔值字典;
    2. 统一将模型置为 `eval` 模式执行激活统计;
    3. 在 `finally` 块中, 严格按字典记录还原每一个子模块原有的 `training` 状态, 不管顶层还是底层模块;
  - 强化 Hook 生命周期的异常安全保证, 确保前向即便发生 CUDA OOM 或 NaN, 注册的所有 pre-hooks 与 post-hooks 必被全部安全移除;
  - 校准入口只消费调用方提供的 iterable 数据源, 在 `torch.no_grad()` 或等价无梯度上下文中前向. 空 batch, 空 iterable, 不可解析 batch 或未观测到目标模块都必须显式报错, 不产出伪零值 scale;
  - 除 training flag 外, 对可能在 `eval()` 或自定义 forward 中写入的 module buffer 建立可恢复快照. 校准结束,异常或 hook 注册半途失败后, buffer 的值, dtype 和 device 都必须恢复.
  - 校准 scale artifact 必须绑定产生它的 topology fingerprint,模块路径与 shape,量化器版本 / scale 语义和输入 profile 摘要. 模型拓扑,目标模块 shape,输入 profile 或 scale 语义发生变化后,旧 artifact 必须在预检时失效, 不能按同名模块路径静默复用.
- **设计考量与不变式**:
  - **副作用零泄漏**: 校准前后, 模型的全部 buffer (特别是 BatchNorm 的统计量) 必须逐字节一致 (`torch.equal`);
  - **Dropout 确定性**: 校准过程中严禁发生特征丢弃.
- **验收与测试要求**:
  - 构造包含 BatchNorm 与 Dropout 的测试网络, 人为将其置为 `train` 模式并设置独特的初始 running 统计量;
  - 运行激活校准, 断言校准完成后:
    1. 网络及各子模块的 `training` 属性恢复为 True;
    2. BatchNorm 的 `running_mean` 和 `running_var` 完全未改变;
  - 构造包含部分模块 eval, 部分模块 train 的混合网络, 验证校准退出后各自状态完美保留;
  - 模拟校准过程中前向抛错, 验证 hooks 列表长度恢复为 0, training flags 和 buffer 快照均恢复;
  - 对空 iterable, 空 Tensor batch 和目标层从未触发三种情况分别断言明确异常. 不允许把任一种情况折算为数值 0 的 scale.
  - 对已生成 scale artifact 再改变目标 Linear 的 shape,模型 topology 或 profile ID, 验证 static quant 预检在任何模块替换前拒绝该过期 artifact.
- **非目标**: 不在 XQT 内部实现训练态量化 (QAT) 或复杂的全局随机种子分发系统.

---

<a id="xqt-005"></a>

### XQT-005. Quant pair 安全与完整发布

- **源码依据**:
  - `已确认`: 在 [quant_pair.py](../../../xqt/contracts/quant_pair.py) 的 `write_quant_pair()` 中:
    `pair_dir = Path(output_dir)`
    `weights_path = pair_dir / resolved_weights_name`
    代码未对 `resolved_weights_name` 进行越界校验, 若传入 `../../etc/danger.pt` 或绝对路径, 权重将直接逃逸写入到 `output_dir` 之外;
  - `已确认`: [quant_pair.py](../../../xqt/contracts/quant_pair.py) 的 loader 侧实现 `_resolve_pair_file()` (第 65-79 行) 显式通过 `resolved.relative_to(root)` 进行了防逃逸安全检查, 读写两侧安全规则严重不对称;
  - `已确认`: 现有写操作是非原子的: 先写庞大的权重文件 (`weights_path`), 随后才进行 metadata 合并与写出 `quant.json`. 若在权重写完后发生 JSON 序列化失败或进程崩溃, 目标目录将残留一个没有 sidecar 的破损孤立权重; 若目标目录已存在旧产物, 直接覆盖可能破坏正在被引用的旧模型.
- **目标与范围**:
  - 统一读写两侧的路径防逃逸与格式校验规则: `weights_name` / `sidecar_name` 严禁绝对路径,严禁包含 `..` 的相对路径,严禁符号链接越界解析; `output_dir` 的授权范围由调用方和其上层 workflow 决定;
  - 建立标准的两阶段安全发布协议 (Staging -> Verify -> Atomic Commit):
    1. 所有新权重与 `quant.json` 先写入与最终 `output_dir` 同一父目录, 同一文件系统的唯一 sibling 暂存目录 (如 `.staging_<uuid>`). 不允许在目标目录内原地改写旧 pair;
    2. 计算写出权重文件的 SHA256 校验和, 并完整写入 sidecar metadata 中;
    3. 校验 metadata JSON schema 完备性;
    4. 仅在暂存目录的 weights/sidecar 均可被 loader 校验后, 采用同文件系统的 rename/replace 将整个目录发布至最终目标路径. 若需要先移走旧目录, 必须保留可恢复 backup, 直到新目录已可见且校验通过;
    5. 发布失败时自动清理暂存目录并完整保留目标目录中的旧有合法产物. 若在新目录已发布后的 backup 清理阶段失败, 新 pair 仍视为有效, 但必须报告明确的清理告警和可恢复路径.
- **设计考量与不变式**:
  - **原子产物不变式**: 对通过路径打开最终目录的读者, 最终目录只能是完整旧 pair 或完整新 pair. `quant.json` 与权重必须成对且 checksum 匹配, 不存在"只有权重没有 sidecar"或"校验和与实际文件不一致"的可见中间状态. 已持有旧文件描述符的外部读者可继续读旧文件, 不在本任务的原子可见性承诺范围内;
  - **路径与链接边界**: `output_dir` 是调用方已经授权的本地目录, 可以是绝对路径; 它本身不因是绝对路径而获得额外权限. `weights_name` 和 `sidecar_name` 必须是单个 basename, 禁止绝对路径, `.`/`..`, 路径分隔符和符号链接. Quant Pair 是平铺格式, 不支持嵌套 sidecar / weights 路径, 以免写侧与默认 loader 的根目录解析产生语义分叉. 目标根目录,staging 根目录或发布前最终文件若为符号链接,必须拒绝;
  - **检查时序与恢复**: 路径检查在创建 staging 前,写入前和发布前都要执行. 实现应使用不跟随符号链接的目录检查, 并把校验后的相对文件名写入 sidecar, 不重新拼接未经校验的用户字符串. `os.replace` 失败时只恢复本次已移动的明确 backup; backup 删除失败不可回滚已验证的新包, 但必须在诊断中留下可人工清理的绝对路径. 这不是抵御有权限同时篡改父目录的对抗性安全承诺;
  - **覆盖保护**: 覆盖既有目录时必须先完成新包构建验证, 再执行替换. 本任务只承诺同一单机文件系统上的发布原子性, 不承诺 NFS/对象存储/多进程文件锁或断电后的持久化语义.
- **验收与测试要求**:
  - 安全边界测试: 传入 `weights_name="../escape.pt"`, 绝对路径 `/tmp/hack.pt`, `sidecar_name="../quant.json"`,绝对 sidecar 路径,嵌套路径和符号链接根目录, 验证全部被拒绝并抛出 `XQTArtifactError`; 同时验证调用方授权的正常绝对 `output_dir` 可以工作;
  - 异常恢复测试: 在写入 sidecar, checksum 校验和目录 replace 三个阶段分别注入失败, 验证既有目录中的旧产物完全不受影响, staging 临时目录被清理或给出明确的清理告警;
  - Roundtrip 测试: 验证生成的 quant pair 能够通过 loader 的全部安全检查, SHA256 校验完全匹配.
- **非目标**: 不实现复杂的分布式文件锁或云对象存储的高级事务 API, 聚焦于单机文件系统的确定性发布安全.

---

<a id="xqt-006"></a>

### XQT-006. Quant 请求与实际结果一致

- **源码依据**:
  - `已确认`: 在 [int8_mma.py](../../../xqt/compression/quant/quantizers/int8_mma.py) 的 `quantize_with_int8_mma()` 中:
    ```python
    if activation_scale_mode == "static":
        if module_activation_scale is None:
            module_activation_scale_mode = "dynamic"
            dynamic_fallback_modules += 1
    ```
    当调用方明确请求静态量化 (`static`), 但传入的 `activation_scales` 缺少某些模块时, 代码内部静默将其降级为动态量化 (`dynamic`), 并仅累加计数器;
  - `已确认`: 在第 223 行的返回元数据中:
    `"activation_scale_mode": "static" if activation_scale_mode == "static" else "dynamic"`
    即便多个或全部子模块已经降级为 dynamic, 结果的 QuantScheme 与模型级元数据依然标为 `"static"`, 导致上层报告与模块内部真实的计算逻辑严重脱节.
- **目标与范围**:
  - 本任务的 fail-closed 适用范围首先限定为 `quantize_with_int8_mma()` 的 W8A8 INT8 MMA 路径. 其他量化器只有在各自明确接入相同契约和测试后才可作相同宣称, 不可被本任务的完成状态自动覆盖;
  - 确立量化请求的 Fail-closed (故障封闭) 契约: 请求 `activation_scale_mode="static"` 时, 若待量化模块未提供完备有效的静态 scale, 必须显式抛出 `XQTQuantError` 报错, 严禁隐式降级;
  - 若用户业务确实允许局部 dynamic fallback, 必须由显式策略开关控制, 且降级的具体模块列表必须如实反映在逐模块合同中. 开关名称,默认值和 API 归属由本任务设计决策确定, 不在规划阶段伪造既有参数;
  - 对传入的 scale Tensor 进行数值合法性校验: 严禁包含 NaN, Inf, 负数或零值 scale. 每个模块采用 scalar, per-channel 或 block scale 的形状和广播语义必须由该 quantizer 显式定义并写入合同, 不得靠调用方猜测;
  - static scale 的完整性和数值预检必须发生在第一个模块替换前. 预检失败时, 传入模型与该模型关联的 artifacts/metrics 均保持不变.
- **设计考量与不变式**:
  - **语义诚实性**: 报告声明为 static 的模型, 其内部所有量化 Linear 必须 100% 绑定了有效静态 scale 并在推理时走静态缩放路径;
  - **预检优于执行**: 在对模型执行任何子模块替换前, 先行校验全部模块与 scale 的匹配度, 杜绝量化一半时抛错留下"半量化"模型.
- **验收与测试要求**:
  - 构造一个包含 2 个 Linear 的网络, 请求 static 量化但仅提供 1 个 scale:
    - 默认模式下: 抛出清晰的异常, 原模型未被任何替换;
    - 显式允许 fallback 模式下: 报告中明确列出降级模块名, 模块级契约准确标注各层真实 mode;
  - 传入非法 scale (包含 NaN 或负值), 验证被拦截并报错.
- **非目标**: 不借此任务重新实现所有量化算法的底层数学计算, 重点建立严密的参数契约与执行守卫.

---

<a id="xqt-007"></a>

### XQT-007. 模块级量化合同与 no-op 语义

- **源码依据**:
  - `已确认`: 在 [runtime_quant.py](../../../xqt/contracts/runtime_quant.py) 的 `first_linear_shapes()` (第 252-270 行) 中, 函数简单遍历 `model.modules()`, 仅抓取第一个遇见的 Linear-like 模块的 shape 作为全模型的代表:
    `return (int(module.out_features), int(module.in_features)), (int(module.out_features), int(module.in_features))`
    对于包含多个不同尺度 Linear (例如 Attention 的 3072 与 MLP 的 12288) 的网络, 该合同完全无法描述真实模型结构;
  - `已确认`: 若模型完全没有 Linear 模块, 该函数返回 `(), ()`, 但 `build_runtime_quant_contract()` 依然会返回一个看似有效的模型级合同, 甚至带有假的 kernel 需求与 decode 标记.
- **目标与范围**:
  - 重构或升级 `RuntimeQuantContract` / `ComputeConfig`, 建立强类型的**模块级量化合同 (Module Quant Contract)**:
    - 逐模块精确记录: `module_path`, `in_features`, `out_features`, `weight_shape`, `storage_dtype`, `compute_dtype`, `scale_mode`, `zero_point_mode`, `layout`, `required_kernel`;
    - 模型级汇总视图由逐模块合同真实聚合派生, 严禁以首层 Linear 代表整模;
  - 明确定义量化阶段的 **no-op (零操作) 状态语义**:
    - 若根据 include/exclude 策略未命中任何可量化模块:
      - 在 strict 模式下 (默认): 显式判定为量化阶段失败, 抛出异常;
      - 在 permissive 模式下: 记录显式的 `no-op` 状态与原因 (`no_matching_modules`), report 中明确标注没有应用量化, 严禁宣称产生体积缩减或加速收益.
- **设计考量与不变式**:
  - **结构完备性**: 合同中记录的模块总数与模型中实际量化的模块总数恒等. `module_path` 必须唯一, 可在当前 `named_modules()` 中解析, 并与记录的 feature/weight shape 一致;
  - **异构感知**: 原生支持多尺寸线性层与不同精度的混合配置.
  - **合同可验证性**: 合同拥有独立 schema version. 每个 module entry 至少包含实际 storage/compute dtype, layout, scale/zero-point mode, required kernel 和可验证的权重 shape. 模型级 summary 只能由 entries 聚合, 允许保留 legacy shape 摘要但不得用于表示全模型结构;
  - **路径与版本规范**: module path 使用与当前模型 `named_modules()` 一致的 canonical 名称. 若量化目标本身是 root module,实现必须明确规定保留路径标记或直接拒绝, 不能将空字符串静默写入合同. reader 只接受自身声明支持的 contract schema version; schema 迁移必须显式产生新版本与 `migrated_from` provenance, 不得猜测未知字段的语义;
  - **no-op 诚实性**: permissive no-op 的 `required_kernels` 为空, 不宣称 prefill/decode 支持, 不产生压缩率/加速率, 且不能作为 performance acceptance 的候选.
- **验收与测试要求**:
  - 构造包含不同维度 Linear (如 768->768, 768->3072) 的模型, 验证量化后合同能够准确包含每一层的具体维度信息;
  - 构造全被 exclude 规则排除的模型, 验证 strict 模式报错, permissive 模式产出带 `no-op` 标记的报告, 不虚报性能;
  - 验证 Quant Pair 序列化与反序列化后, 逐模块合同信息不丢失.
- **非目标**: 不在 XQT 内实现跨多机多卡的大规模分布式 Tensor Parallel 切片调度器.

---

<a id="xqt-008"></a>

### XQT-008. 候选能力与可执行性分离

- **源码依据**:
  - `已确认`: 在 [engine_resolve.py](../../../xqt/kernels/engine_resolve.py) 的 `resolve_engine()` (第 568-580 行) 中:
    ```python
    providing = engines_providing_all(caps)
    if not providing:
        providing = list(preferred) if preferred else [normalize_engine_name(fallback)]
    ```
    代码仅根据静态注册表中的 `provides` 集合做名称查找. 当静态表无匹配但传入了 `preferred` 提示时, 直接采信 `preferred`;
  - `已确认`: 静态注册表中包含大量标记为 `maturity="metadata_only"` 或 `dispatchable=False` 的规划中引擎 (如 `cutlass`, `cute_dsl`), `resolve_engine` 会将这些不可执行的引擎作为有效候选甚至首选返回, 导致下游真正执行时发生意料之外的失败或伪执行;
  - `待验证`: 部分引擎虽已注册但缺少本机编译的 C++/CUDA 扩展, 或当前 GPU 算力小于其 `min_capability`, 需要确定性的执行前探测.
- **目标与范围**:
  - 将引擎管理明确解耦为职责清晰的两套核心 API:
    1. **候选能力查询 API (`query_engine_capabilities`)**: 纯静态查询, 允许返回所有理论上具备该能力的引擎条目 (包括 planned, metadata_only, reference_guarded), 用于规划展示与能力清单生成;
    2. **执行分发解析 API (`resolve_executable_engine`)**: 面向真实执行的严密解析, 强制校验以下前置条件:
       - `dispatchable == True`;
       - `maturity == "executable"`;
       - 所需依赖库 (如 TileLang, Triton, PyTorch) 在当前 Python 环境中真实可导入;
       - 当前 CUDA 设备的计算能力 (SM version) 满足 `min_capability` 限制;
  - 确立 Fail-closed 原则: 当请求的 `required_capabilities` 无任何通过预检的引擎支持时, 显式抛出 `XQTBackendError`, 严禁静默 fallback 到未声明的默认引擎. 允许 fallback 时也必须由 request 显式声明候选集合, 并在 resolved/observed 中记录从何引擎退到何引擎及原因;
  - `preferred_engines` 仅作同等可执行候选中的排序提示 (Hint), 绝对不能用来突破可执行性与最低算力限制.
- **设计考量与不变式**:
  - **预检与实证分离**: 凡由 `resolve_executable_engine` 成功返回的引擎, 只保证它通过通用 dispatchable/maturity/dependency/SM 预检. 特定 operator, layout, shape, dtype 和 JIT 产物的可 materialize 性必须继续由 plan/materialize 验证, 实际 forward/kernel 必须由 observed 证据证明;
  - **无副作用探测**: 查询与探测过程严禁在纯 CPU 任务中意外触发 CUDA 上下文初始化.
- **验收与测试要求**:
  - 构造仅请求 `int8_mma` 的场景, 请求指定 `cutlass` (metadata_only) 作为 preferred, 验证执行解析忽略该 hint, 只在显式允许的候选集合中选择真正通过预检的 `torch_int_mm` 或 `tilelang`;
  - 模拟当前环境缺少 Triton 包, 验证依赖检查将其从可执行列表中剔除;
  - 在无任何支持引擎时, 验证抛出包含清晰诊断信息的 `XQTBackendError`.
- **非目标**: 不重写各引擎内部具体的算子内核代码, 聚焦于分发解析与守卫机制.

---

<a id="xqt-009"></a>

### XQT-009. 指标作用域与 acceptance

- **源码依据**:
  - `已确认`: 在 [auto/_helpers.py](../../../xqt/auto/_helpers.py) 的 `find_nested_numeric()` (第 8-26 行) 中, 函数在嵌套映射中递归搜索指定 key, 收集到列表后直接执行:
    `return max(values) if values else None`
  - `已确认`: 在 [acceptance.py](../../../xqt/auto/acceptance.py) 中, 大量门槛判定直接调用 `find_nested_numeric(metrics, "speedup")`. 若优化过程中记录了多个子算子的 speedup (例如一个为 0.8x, 另一个为 2.0x), 函数返回 2.0x, 使得设定 `min_speedup=1.5` 的验收规则判定为通过 (`accepted=True`), 彻底掩盖了严重性能退化的子算子;
  - `已确认`: 现有逻辑无法区分算子级 (target), 模块级 (block) 与整模级 (whole model) 的同名指标, 存在指标作用域被动混淆.
- **目标与范围**:
  - 废除对同名 key 盲目递归并取 max 的不可靠行为, 建立基于**显式指标路径 (Key Path) 与作用域 (Scope)** 的指标检索与验收系统:
    - 支持精确路径表达式, 例如 `model.latency.p50_ms`, `stages.quant.block_0.speedup`;
    - 明确区分三级作用域: `operator_level`, `block_level`, `model_level`; 整模加速比必须严格来自整模对照测量;
  - 指标聚合规则显式化: 当需要对多目标指标进行综合评定时, 必须显式声明聚合策略 (`all` 全部达标, `worst` / `min` 最差值达标, `mean` 平均值达标, 或按计算量 `weighted` 加权平均);
  - 指标 path 采用受限点分字段语法, 只定位 Mapping 内的明确 key, 不支持递归全树搜索,通配符或隐式数组索引. 同名指标存在于多个 profile 时, report 必须先以 profile ID 分组,再由显式 aggregation 汇总. `weighted` 需要 request / freeze 中同时提供不可变权重来源和 `sum(weights) > 0` 校验;
  - 建立强类型量纲与单位校验机制:
    - 区分时延单位 (ms vs us), 吞吐单位 (samples/s vs tokens/s), 显存单位 (MB vs GB);
    - 拦截非法数值: 包含 NaN, Inf, 负时延, 以及将布尔值 `True` 误当成数字 `1.0` 的情况;
  - acceptance config 必须为每个启用阈值记录 metric path, scope, unit, direction 和 aggregation. path 未命中, 命中多个对象但未指定聚合, 或 reference/current profile/freeze version 不一致时, 不得回退到全树搜索;
  - `speedup` 的 primary 定义为同一 profile 的 `reference_p50_ms / candidate_p50_ms`. 多 profile 的默认判定为每个 profile 单独满足门槛, 等价于最小 speedup 通过. 只有 freeze 预先声明 profile 权重和基于原始 latency 的汇总公式时, 才允许 `mean` 或 `weighted` 汇总; 不得对任意 speedup 比率事后取均值.
- **设计考量与不变式**:
  - **单调安全性**: 在未显式配置允许局部退化的前提下, 默认采用最严苛的 `worst` 聚合判定, 任何单点退化不得被其他点的激增掩盖. 缺失值不是 0, 也不是通过, 而是证据缺失;
  - **因果一致性**: 验收报告中必须完整记录所依据的指标路径, 阈值, 实测值, 聚合方法与判定结论.
- **验收与测试要求**:
  - 针对多算子 speedup `[0.8, 2.0]` 的场景编写测试, 验证在未声明宽松聚合时, `min_speedup=1.5` 必须判定为拒绝 (`accepted=False`);
  - 显式配置 `aggregation="worst"` 时判定失败, 配置 `aggregation="mean"` 时只对预先声明且单位一致的同类 metric 按公式判定;
  - 传入带有 NaN,布尔值,单位冲突,路径未命中或 reference profile 不一致的 metrics, 验证 acceptance 不会通过并给出明确的配置错误或证据不足原因.
  - 对 path 终点为 Mapping / list,profile ID 不一致,weighted 缺权重或权重非法,数值为 bool / NaN / Inf / 负值等分别编写拒绝或 `not_decided` 测试, 不得退回旧的递归 key 搜索.
- **非目标**: 不开发大型图形化商业智能 (BI) 报表平台, 聚焦于自动化决策规则的严密性.

---

<a id="xqt-010"></a>

### XQT-010. 模型结构契约主链接线

- **源码依据**:
  - `已确认`: 在 [model_pass.py](../../../xqt/pipeline/model_pass.py) 的 `LoadModelPass.run()` (第 23-24 行) 中:
    ```python
    if context.model is not None:
        return context
    ```
    当调用方在构造 Session 时直接传入现成的 `model` 实例时, 代码直接早退返回, 完全跳过了第 27-38 行的 `adapter.structure_contract(model)` 与 `adapter.inference_contract(model)` 解析逻辑, 导致直接传入的模型缺失结构契约;
  - `已确认`: [adapter.py](../../../xqt/model/adapter.py) 中默认允许返回空的结构契约, 缺乏对模型必需角色 (如 attention, mlp, modulation) 与高精度保护层的强制校验;
  - `待验证`: 剪枝 (Prune) 或图重写 (Rewrite) 改变模型层级或参数命名后, 结构契约未能自动失效或重建, 后续 stage 仍沿用旧路径.
- **目标与范围**:
  - 统一结构契约接入管道: 无论模型是外部直接注入 (`model=...`), 还是通过配置文件从 Checkpoint 动态加载, 均必须经过统一的 `resolve_and_validate_structure_contract(model, profile)` 管道;
  - 结构契约作为模型角色分组, 权重映射, 保护层路径的单一权威源:
    - 显式绑定 `COMPONENT_ROLES`, `MergedProjectionSpec` 与 checkpoint weight mapping;
    - 针对未映射的非标准权重或模块缺失执行全覆盖校验, 发现不匹配时显式报错;
  - 建立契约的生命周期与失效机制: 任何改变模型拓扑结构的变换操作 (Prune / Rewrite) 必须显式更新结构契约或使旧契约失效并触发重新解析, 保证后续操作感知最新结构.
- **设计考量与不变式**:
  - **契约真实性**: 结构契约中声明的每一个模块路径必须在当前模型中真实存在;
  - **无侵入性**: 结构契约是只读的声明式视图, 不拥有 forward 执行逻辑, 不改写模型计算图.
  - **版本与指纹**: contract 必须带 schema version 和可重算的 topology fingerprint. stage 在消费 contract 前校验其 fingerprint 是否仍与当前模型匹配; 不匹配时只能重新解析或显式失败, 不能沿用过期路径.
- **验收与测试要求**:
  - 分别使用 `model=` 直接注入和 `checkpoint + profile` 加载同一网络, 断言生成的 `context.structure_contract` 完全等价;
  - 构造结构不匹配的模型 (缺少声明的 head 或权重名拼写错误), 验证校验器抛出包含详细路径差异的错误;
  - 验证经剪枝 stage 改变通道维度后, 结构契约中的维度信息能够正确更新.
- **非目标**: 不承诺对全网所有任意未知神经网络架构实现零配置通用自动识别, 重点保障受支持模型 profile 的确定性契约接入.

---

<a id="xqt-011"></a>

### XQT-011. Registry 权威源与分层收敛

- **源码依据**:
  - `已确认`: [kernels/registry.py](../../../xqt/kernels/registry.py) 维护了全库统一的 engine inventory 与动态 mirror; 与此同时, [ops/gemm/registry.py](../../../xqt/kernels/ops/gemm/registry.py) 仍然维护了一套包含了丰富 capability, GEMM problem definition 与 executor 的特定注册表;
  - `已确认`: 两套注册表在部分 GEMM 算子定义上存在重复维护, 且包含兼容性提示与废弃告警, 增加了维护成本并可能导致分发逻辑依据不同的事实源.
- **目标与范围**:
  - 确立清晰的单一权威源分层原则与数据流向:
    1. **算子执行层 (`ops/gemm/registry`)**: 专门负责 GEMM 算子的具体实现, problem 定义, epilogue 融合与真实分发调度 (Dispatch);
    2. **全局引擎视图层 (`kernels/registry`)**: 作为全局统一的只读元数据视图 (Inventory), 单向从算子执行层汇总信息, 严禁双向依赖与独立硬编码;
  - 清理无调用的冗余中间别名, 验证所有公开接口的导入链, 保证算子注册在模块 import 时按需触发, 不引起循环引用.
- **设计考量与不变式**:
  - **单向数据流**: 底层算子层不依赖上层全局注册表, 全局注册表单向索引底层算子;
  - **注册幂等性**: 重复注册相同名称与签名的算子时保持幂等, 冲突注册必须显式报错.
- **验收与测试要求**:
  - 验证选定的一组典型 GEMM 算子路径在 inventory 查询, capability 匹配, selector 选择与 dispatch 执行中完全一致;
  - 测试重复注册同名不同实现时能被正确拦截并报错;
  - 验证对未导入的 optional backend 算子, 其注册过程不会引发崩溃或强制初始化 CUDA.
- **非目标**: 不为追求绝对的代码行数削减而抹杀 GEMM 算子执行所需的必要上下文元数据.

---

<a id="xqt-012"></a>

### XQT-012. Typed graph rewrite 与 pattern

- **源码依据**:
  - `已确认`: 在 [quant_stage.py](../../../xqt/pipeline/pass_helpers/quant_stage.py) 的 `_maybe_run_graph_transforms()` (第 45-71 行) 中:
    代码直接接受弱类型的字符串别名列表 (如 `["rotation_absorb", "rotation", "quarot"]`), 且当前仅实例化了 `RotationAbsorbTransform`;
    若用户传入了列表中未知的变换名称, 该未知项在与已知项混合时被静默忽略; 若全部未知, 仅返回一个包含了 `notes` 的字典, 完全不报错, 导致用户误以为图变换已成功执行.
- **目标与范围**:
  - 废弃弱类型字符串 alias 匹配, 建立基于强类型配置数据类 (`GraphTransformConfig`) 的图重写注册与调度机制;
  - 未知变换名称或非法参数必须在修改模型前抛出明确配置错误, 严禁静默吞掉配置;
  - 规范并实现首批三大关键图重写 Pattern 的契约, 前置条件与参考实现:
    1. `dequant_gemm`: 权重反量化与矩阵乘法的融合, 声明支持的 layout, dtype 与 scale 广播约束;
    2. `norm_quant`: RMSNorm/LayerNorm 缩放与后续量化操作的融合, 声明数学等价性前置条件;
    3. `activation_quant`: 激活函数 (GELU/SiLU) 与输入量化的融合;
  - 所有图重写变换必须接入 XQT-002 的事务回滚保护, 重写失败时不破坏原模型.
- **设计考量与不变式**:
  - **Dry-run 预检支持**: 每个 rewrite pattern 必须支持在不实际修改模型的前提下生成应用计划 (Plan), 明确指出匹配到的节点, 前置条件, 拒绝原因,替换目标和预期 contract 变化. dry-run 不得触发权重写入,JIT 编译或 artifact 发布;
  - **数值等价性保证**: 在声明的容差范围内, 图重写前后的计算输出必须与参考实现一致.
- **验收与测试要求**:
  - 配置错误测试: 传入包含未知 transform 名的配置, 验证抛出异常而非静默忽略;
  - 针对三大 pattern 分别编写单测, 验证匹配命中, dry-run 计划生成以及替换后的模型结构变化;
  - 验证图变换执行过程中若注入异常, 模型结构能被完整回滚.
- **非目标**: 不试图构建通用的全功能计算图编译器, 聚焦于大模型量化部署中最核心的融合模式.

---

<a id="xqt-013"></a>

### XQT-013. 通用 block materialize 与 runtime

- **源码依据**:
  - `已确认`: 仓库在 `docs/md/architecture/xqt-operator-block-optimization.md` 中已经定义了 Block 准入标准, 并在 `xqt/runtime/` 下有部分手写与自动 block runtime 原型;
  - `已确认`: 算子微基准的高加速比经常在组装成完整 Block 后大幅衰减, 主要原因在于各算子之间频繁的 Tensor layout 转换 (如 Row-major 到 Column-major), 冗余的 wrapper 包装以及小内存分配开销;
  - `待验证`: CUDA Graph 缓存缺乏对模型参数更新, 设备迁移 (`.to()`) 以及输入动态 shape 的失效保护机制.
- **目标与范围**:
  - 在 XQT-014 已核验的真实模型上, 由模型结构契约驱动 Block 识别,提取与通用编译单元构建. 未获得真实模型资源前, 仅可完成通用机制与 fixture 正确性, 不可宣称已覆盖特定模型的 Block;
  - 以 profiler 和端到端 Block 证据识别并消除不必要的中间 Tensor 拷贝,格式转换和包装层 (Wrapper) 开销. 不能预设某一类开销必然存在或必然是主瓶颈;
  - 规范 CUDA Graph 运行时集成:
    - 严格限定仅在静态输入 shape 下触发图捕获 (Capture);
    - 明确 Graph 缓存键 (`TimingCacheKey`), 绑定输入 shape, dtype, device 与参数版本;
    - 当模型参数发生更新或设备迁移时, 强制将对应 Graph 缓存失效;
    - 缓存 key 还必须覆盖输入 stride, layout, requires-grad 状态, stream/设备上下文和可影响输出的 runtime flag. cache 必须有明确容量或显存预算, 命中/失效/淘汰原因需要进入 report;
  - 结合 XQT-014 冻结的门槛, 确保只有在完整 Block 层面取得实际性能提升的候选才被允许准入 (Promotion).
- **设计考量与不变式**:
  - **整块收益优于局部收益**: Block 级端到端时延降低是评判优化的硬指标, 单算子加速如果被整合开销吞噬则不予采纳;
  - **状态可重现**: 只有满足捕获前置条件的 profile 才可启用 CUDA Graph. 对该 profile, Graph 重放输出必须在冻结容差内与等价 eager 路径一致. 若 replay 返回 graph-owned storage, API 必须明确其在下一次 replay 后可能被覆盖, 调用方需要长期保留时必须复制;
  - **性能结论受限**: Graph capture 或 Block materialization 成功仅证明 `materialized`, 不自动证明低延迟. 性能结论必须来自 XQT-014 后冻结 profile 的 `observed` 成对样本.
- **验收与测试要求**:
  - 在 XQT-014 已冻结的真实模型 Block 上运行 benchmark, 给出包含 wrapper 开销的端到端耗时. 未达到该条件时, 任务不得进入以性能提升为理由的 `done`;
  - 对允许 CUDA Graph 的输入 profile, 验证连续重放的数值稳定性, 并单独报告 eager 与 graph 的时延. 是否达到提升门槛按 XQT-014 冻结标准判定;
  - 改变输入 Tensor shape/stride/dtype,模型权重或设备后继续推理, 验证旧 Graph 缓存被正确失效并重新安全处理;
  - 用超过缓存预算的一组静态 profile 验证可预测的淘汰, 并断言被淘汰的 graph 无法再次 replay. 发生 capture/replay 失败时, 只有 request 明确允许 fallback 才可回退 eager, 且 report 必须携带失败原因.
- **非目标**: 不承诺单 Block 必须写成极致的单个 Megakernel, 优先以已有高性能 Kernel 组合加图捕获消除开销.

---

<a id="xqt-014"></a>

### XQT-014. 真实模型 baseline 与门槛冻结

- **源码依据**:
  - `已确认`: [xqt/model/flux2_klein/types.py](../../../xqt/model/flux2_klein/types.py) 中已定义 `FLUX2_KLEIN_4B_REPO_ID` 等模型常量, 仓库已有其 NVFP4 与低比特研究入口, 具备真实模型验证的基础;
  - `已确认`: 当前缺乏在目标硬件 (RTX 4070 Ti SUPER `sm_89`) 上正式运行该真实模型未优化权重的完整性能与显存基线记录, 导致后续优化的对比依据不固定.
- **目标与范围**:
  - 核验并冻结首选验收目标 **FLUX.2 Klein 4B** 的实际 repo revision,许可,权重来源,文件清单与 SHA256,加载配置和本地存储位置. 在核验完成前, 它仍是候选而非已冻结载体;
  - 严格限定 XQT 模型侧评测边界: 以 `Flux2KleinTransformer` 骨干网络的前向计算作为优化与度量对象; 外部完整扩散 Pipeline 耗时与图像质量评测交由外部脚本或 XDL 负责;
  - 根据实际可加载模型定义并冻结 `primary` 与 `guardrail` 输入 profile, 不在文档中预设图像隐空间尺寸,文本序列长度或 layout. 测定各 profile 的未优化 Eager Baseline:
    - 稳态推理延迟 (p50 / p95, ms);
    - 稳态推理吞吐 (steps/s);
    - 稳态与峰值显存占用 (Peak VRAM, MB);
  - **正式冻结优化准入门槛**:
    - 按输出张量尺度和外部质量敏感度确定数值误差指标,方向和容差;
    - 按成对基线的方差,目标硬件和优化成本确定整模 speedup 门槛;
    - 按实测基线,可用显存和安全余量确定峰值显存门槛;
    - 冻结门槛正式归档后不得为了让失败候选通过而无理由下调;
  - 以机器可读的 freeze 记录固化: baseline/candidate artifact ID, profile ID, 输入和原始样本 checksum, 计时来源, warmup/iterations, 同步方式, 离群值规则, 统计公式, 单位, 质量证据版本和每项阈值的理由. 文本报告必须链接到这一记录, 不可成为唯一事实源.
- **设计考量与不变式**:
  - **基准可复核性**: 拥有相同 checkpoint,输入 profile 与冻结软件/硬件环境的开发者, 必须能重放命令并在预先记录的统计波动范围内复现基线;
  - **显存安全边界**: 基线和候选均不得超过基于实测可用显存冻结的安全边界. OOM,动态 workspace 扩张或无足够安全余量的候选应记录为拒绝或 `not_decided`, 不能以单次侥幸运行通过.
- **验收与测试要求**:
  - 产出包含完整软硬件环境信息的 Baseline 测定报告, 归档至 `research/` 或指定 artifact 目录;
  - 待冻结决策表中的数值容差与性能门槛全部填写完成并锁定;
  - 用同一 freeze 中的至少一个独立重放 run 验证 baseline 统计落在预先记录的波动规则内. 若不满足, 先修正测量稳定性或将 target 标为 `not_decided`, 不得直接开始候选性能宣称.
- **非目标**: 不在无模型权重或无硬件时虚构测量数据; 若资源不可获得, 标记为 `blocked` 并启动备选模型流程.

---

<a id="xqt-015"></a>

### XQT-015. 真实模型优化与质量闭环

- **源码依据**:
  - `已确认`: 本项是 M1-M3 基础改进在真实模型上的综合落地验证.
- **目标与范围**:
  - 以 FLUX.2 Klein 4B 真实 Checkpoint 为输入, 贯通完整的优化工作流:
    `ModelProfile -> ModelStructureContract -> Quantize -> Graph Rewrite -> Block Materialize -> Benchmark`;
  - 确保 Session 交互式调用与 YAML workflow 执行达成完全相同的优化结果与配置语义;
  - 严密组织并执行**全流程消融实验 (Ablation Study)**, 分步报告收益与开销:
    1. `Baseline (BF16 Eager)`;
    2. `Quant-only (仅量化权重与计算)`;
    3. `Quant + Graph Rewrite (量化加图重写融合)`;
    4. `Quant + Graph Rewrite + Block Materialize / Compile (完整组合)`;
  - 结合外部传入且已绑定 checkpoint / artifact 的任务评测打分, 验证候选是否在满足冻结质量门槛的前提下达成整模加速与显存削减目标. 未达标候选必须保留为诊断或 `rejected` 记录.
- **设计考量与不变式**:
  - **整模达标原则**: 仅局部子算子加速但整模未达到 XQT-014 冻结门槛的方案, 不得宣布优化成功;
  - **真实执行验证**: 报告中记录的执行引擎必须为真实运行并测量的 Native 内核, 严禁以 fallback 冒充.
- **验收与测试要求**:
  - 至少有一条端到端优化候选达到预定的整模性能与显存目标, 且通过冻结的数值容差验收;
  - 提交完整的消融实验数据对比表 (包含各阶段的延迟, 吞吐, 显存, 误差);
  - 产出正式的优化阶段 Report 与 Manifest JSON. 每个消融行必须引用相同 freeze/profile, 输入样本和比较模式, 并分别列出 requested/resolved/materialized/observed 路由; 不允许把某行的 fallback 数字或另一 profile 的最快结果移入完整组合行.
- **非目标**: 不在 XQT 内部运行需要数百张图片和几个小时的复杂下游生成评测, XQT 仅负责消费外部评测结果并记录证据.

---

<a id="xqt-016"></a>

### XQT-016. Artifact 重载与实际部署验收

- **源码依据**:
  - `已确认`: 在 [quant_pair.py](../../../xqt/contracts/quant_pair.py) 中已具备产物写出与读取的基本逻辑;
  - `已确认`: 当前多数测试局限于同进程内存中的 dry-run 或 session 创建, 缺乏在全新的无状态独立 Python 进程中直接加载量化产物并运行端到端真实前向推理的严肃验证.
- **目标与范围**:
  - 将 XQT-015 优化产出的模型与契约正式发布为标准的 Quant Pair (包含权重文件, `quant.json`, 模块级合同与 Lineage);
  - 编写独立的部署加载脚本, 在**完全隔离的全新 Python 进程**中:
    1. 不依赖任何原始优化 Session 实例或优化配方 (Recipe);
    2. 仅依据磁盘上的 `quant.json` 与权重产物, 完成模型结构重建与量化权重注入;
    3. 运行实际前向推理计算, 验证输出 Tensor 与优化前保持在冻结容差范围内一致;
    4. 测量重载后的实际推理延迟与显存占用, 验证离线优化收益被真实固化在产物中;
  - 独立生成部署交付报告, 明确区分 Native PyTorch 产物与 ONNX/TensorRT 导出的不同适用场景;
  - 规定 loader 的受信任边界: `quant.json` 只能选择已注册的 profile/adapter/contract schema, 不得按 artifact 中的任意模块路径动态 import. 默认优先安全权重格式, 对 `.pt`/pickle 载荷必须明确标注为仅可加载受信任本地产物, 不得承诺能安全读取不可信文件.
  - SHA256 只用于将读取到的文件与一个已可信的期望摘要比对, 能发现传输或落盘损坏, 不是签名机制,也不能使恶意 artifact 变为可信输入. 若未来需要来源认证,必须另行设计签名 / key 管理, 不得将它隐式塞进 Quant Pair loader.
- **设计考量与不变式**:
  - **零内存泄漏依赖**: 产物的加载与执行严禁依赖任何原优化过程留存在内存中的全局变量或临时缓存. 独立进程的输入必须从 freeze 中的可重建样本或受控 fixture 获得;
  - **可分发性**: 产物必须声明 ABI,GPU capability,driver,runtime 和可选依赖约束. 在满足冻结兼容条件的另一环境中应可经 preflight 加载; 不满足时必须 fail-closed 并给出缺失条件, 不承诺跨任意机器直接执行.
- **验收与测试要求**:
  - 运行独立的进程隔离测试脚本 (如通过 `subprocess` 调用全新 Python 解释器), 成功加载产物并完成 forward;
  - 比对独立进程重载推理输出与保存前输出, 误差在容差内;
  - 模拟产物文件缺失或损坏 (如篡改权重文件破坏 SHA256), 未知 profile/contract version,路径逃逸和不满足 ABI/SM 约束, 验证独立加载器在权重注入前具备明确的错误诊断与拦截机制;
  - 在独立进程中验证同一 artifact 的二次加载不会依赖父进程 cache, 并把 artifact checksum,加载器版本, resolved executor 和实际 fallback 记入部署报告.
- **非目标**: 不在本任务中实现高性能分布式 Web Serving 框架 (如 vLLM / SGLang 级的调度器与网络服务).

---

<a id="xqt-017"></a>

### XQT-017. 第二同族模型复用验证

- **源码依据**:
  - `已确认`: 单一模型的端到端闭环可能存在针对该模型的"特例硬编码", 必须引入第二个独立模型检验整套抽象与主链的通用复用能力.
- **目标与范围**:
  - 选择第二个同族真实模型. 它必须共享已声明的 FLUX.2 Klein profile 语义,核心模块角色和输入协议, 但采用独立 checkpoint / revision. Qwen 等异构 LLM 只能作为后续跨族扩展, 不得用于本里程碑的复用结论;
  - 在**不修改 XQT 通用优化主链核心代码**的前提下:
    1. 仅通过声明新的 `ModelProfile`, 结构映射关系与 Adapter;
    2. 接入现有的量化, 图重写, Block 编译与产物发布流程;
    3. 成功产出达标的量化部署产物, 并完成独立进程重载验证;
  - 逐项列出模型接入改动, 区分 profile / mapping / adapter 声明与通用主链修改. 若确需改动通用主链, 必须说明缺失的通用语义,替代方案和首个模型的回归影响, 不以任意代码比例作为通用性的替代指标.
- **设计考量与不变式**:
  - **核心零污染**: 接入新模型严禁在通用算子, 事务调度或量化器中添加 `if model_name == "xxx"` 形式的硬编码分支;
  - **回归安全**: 第二模型的接入严禁对首个模型 (FLUX.2 Klein) 的既有测试造成任何破坏性回退.
- **验收与测试要求**:
  - 第二模型通过完整的优化, 导出与独立重载测试;
  - 运行与变更范围匹配的 CPU,集成与可用硬件回归测试. 首个真实模型的 required gate 必须通过; 无法满足的 optional gate 必须带原因记录, 不能笼统声称"全量绿色";
  - 交付关于模型接入差异与通用代码复用率的技术说明文档. 文档必须列出新增/修改的 profile, mapping, adapter 和 generic core 文件; 任一 generic core 变更都要有首模型回归证据和抽象必要性说明.
- **非目标**: 不盲目追求一口气支持几十种不同模型架构, 聚焦于验证设计抽象边界的合理性.

---

## 5. 统一性能与质量规约

为保证所有性能测试与质量评估的严肃性, 以下规约作为 XQT-009, XQT-013 至 XQT-017 的共同执行准则:

| 维度 | 必须严格执行与记录的规范 | 严禁出现的伪验收行为 |
| --- | --- | --- |
| **元数据溯源** | 记录 Git Commit Hash, Checkpoint/Artifact SHA256, 依赖库精确版本, GPU 型号与驱动版本 | 缺失环境记录的孤立数值截图或文本剪贴 |
| **测试范围边界** | 明确标明属于单算子 (operator), 模块级 (block), 整模前向 (model) 还是外部 Pipeline | 将单算子微基准的高加速比直接宣传为整模提速 |
| **输入条件固定** | 严格固定 Batch Size, Sequence Length, 数据类型 (Dtype), 内存布局与随机种子 (Seed) | 优化前后使用不同尺度输入, 或在测完后筛选有利数据 |
| **对照环境对称** | Baseline 与 Candidate 必须在相同硬件, 相同进程与对称的编译模式下成对比较 | 用未经任何编译优化的 Eager Baseline 对比带有图优化的 Candidate |
| **时钟度量分离** | CUDA Event GPU 耗时与同步后的 Wall-clock 耗时必须分列展示 | 挂载 PyTorch Profiler 采集的开销直接当作正式 Benchmark 成绩 |
| **统计方法严密** | baseline 前先冻结 warmup,采样数,同步方式,离群值规则和 p50/p95 计算方法. 采样量必须足以刻画该 profile 的稳态波动, 不能事后为候选调整 | 仅跑极少轮次, 仅挑选极值中最优的一次, 或在看到结果后改变样本数 / 聚合规则 |
| **显存口径透明** | 严格区分当前分配显存 (`allocated`), 预留显存 (`reserved`) 与设备总占用, 记录稳态与峰值 | 混淆参数显存与激活显存, 或忽略 CUDA 上下文基础开销 |
| **冷热启动隔离** | 明确拆分 Checkpoint 加载耗时, 量化校准耗时, 权重 Prepack 耗时, JIT/Graph Capture 耗时与稳态耗时 | 将一次性初始化的开销摊薄或混入稳态吞吐统计 |
| **执行路由透明** | 明确报告请求的契约, 实际分发的执行引擎, 真实的物理 Kernel, 以及是否存在 Fallback | 声称跑了某硬件专用加速引擎, 实际底层默默走了通用 PyTorch 实现 |
| **外部质量绑定** | 外部任务评测指标必须精确绑定当前特定模型的 Checkpoint / Artifact 标识与评测版本 | 脱离模型产物抽象宣称"质量无损", 或以数值 Diff 代替任务质量 |

### 5.1 冻结指标的机器可读语义

下表是 target freeze 和 acceptance report 应采用的指标语义, 不是要求当前所有内部 report 已经使用同一字段名. XQT-009 负责将现有字段映射到这些语义, 并在 report 中同时保留原始字段路径. 一个指标没有 scope,unit,direction,profile ID 和 raw-samples 引用时, 不得用于跨 stage 或跨模型的 acceptance.

| 指标语义 | 建议 canonical 表达 | 单位与方向 | 最低补充字段 |
| --- | --- | --- | --- |
| 稳态延迟 | `latency.p50_ms`, `latency.p95_ms` | `ms`, 越小越好 | 计时方法,同步方法,warmup,有效样本数,离群值规则,paired run ID |
| 吞吐 | `throughput.work_per_s` | work unit / s, 越大越好 | `work_unit` 的业务定义, batch 和 profile; 不可把 token/s 与 steps/s 直接比较 |
| 峰值显存 | `memory.peak_allocated_bytes`, `memory.peak_reserved_bytes`,可选 `memory.process_peak_bytes` | `bytes`, 越小越好 | 采样 API,reset 点,测量 scope; 三类峰值不可混为同一阈值 |
| 数值差异 | `output.mean_abs`, `output.max_abs`, `output.cosine_similarity` | 前两者无量纲且越小越好, cosine 越接近 1 越好 | reference artifact,输出选择规则,dtype 和缩放 / 归一化定义 |
| 路由真实度 | `runtime.fallback_count`, `runtime.executed_engines` | count / 枚举, fallback 越少越好 | request / resolved / materialized / observed 四层证据与 fallback reason |
| 外部质量 | `quality.<metric_name>` | 由评测定义并冻结方向 | evaluator revision,样本集摘要,checkpoint / artifact checksum,置信或波动规则 |

`speedup` 不是独立可漂移的原始指标. 对同一 freeze,profile,计时方法和 paired run, 统一定义为 `reference_latency_p50_ms / candidate_latency_p50_ms`. `candidate_latency_p50_ms <= 0`,输入集不等价,单位不同,或 reference 不可定位时, speedup 为无效证据而非 `0` 或一个默认值. 多 profile 默认逐项通过并取最差 profile; 只有 freeze 预先记录权重来源,汇总公式和失败策略时, 才允许使用 `mean` 或 `weighted` 结论.

### 5.2 测量运行的最小清单

每条正式 baseline 或 candidate 记录应至少说明 CUDA / accelerator 是否同步,是否在计时前完成 JIT / prepack / graph capture, AB/BA 顺序或随机化种子,运行期间是否发生 OOM / recompilation / cache eviction,以及 GPU 时钟或功耗状态是否可获取. 无法控制的环境变量不是自动失败理由, 但必须记录为限制; 一旦它在同一 paired run 内改变,该 run 不能参与阈值判定. profiler 可以解释瓶颈, 但 profiler 采集开销不能进入正式 latency raw samples.

---

## 6. 待冻结决策

以下是各任务在实施或正式准入前必须由设计者与架构委员会共同锁定的关键决策. 各决策完成后需在下表中更新状态并附上依据:

| 决策项 | 责任任务 | 核心技术选择与备选方案 | 默认推荐方向 | 记录状态 |
| --- | --- | --- | --- | --- |
| **模型快照与内存隔离策略** | XQT-002 | A. CPU 内存全量 state_dict 备份<br>B. 仅对被修改模块做局部 state_dict 备份<br>C. 拓扑与参数两级分离快照 | 方案 C: 拓扑轻量备份, 参数按需备份, 控制内存增长 | 已冻结 (采用 TransactionSnapshot 隔离备份与状态回滚) |
| **attempt ID 与诊断存储** | XQT-002, XQT-003 | A. 用 stage name 覆盖旧 report<br>B. stage name 下增加有序 attempt ID | 方案 B: accepted stage 与每次 attempt 分开存储, 重试不可覆盖历史 | 已冻结 (每个 attempt 独立递增 attempt_id, 保留失败诊断) |
| **Current / Best 身份与状态机** | XQT-003 | A. 引入显式 `current_stage` 字段并在 `use()` 时更新<br>B. 移除 `best_stage` 的自动更新, 改为纯查询方法 | 方案 A+B: Current 独立自增, Best 依赖显式指标评选 | 已冻结 (Current 显式流转, Best 显式查询, Lineage DAG 防自指) |
| **Quant Pair 安全发布机制** | XQT-005 | A. 同目录临时后缀重命名<br>B. 独立 `.staging_<uuid>` 临时目录校验后原子覆盖 | 方案 B: 独立 Staging 目录两阶段提交, 保证发布原子性 | 已冻结 (两阶段原子提交流程, 校验 SHA256 后替换目录) |
| **发布可承诺范围** | XQT-005 | A. 宣称任意文件系统和断电场景原子<br>B. 限定同机同文件系统目录切换 | 方案 B: 记录 rename/backup 恢复边界, 不虚构分布式或断电持久性保证 | 已冻结 (边界清晰限定为同机同文件系统目录原子替换) |
| **模块级量化合同 Schema 设计** | XQT-007 | A. 在现由 `ComputeConfig` 中追加模块字典<br>B. 新建独立的 `ModuleQuantContract` 规范并在上层聚合 | 方案 B: 建立干净的模块级数据类, 保持高内聚 | 已冻结 (ModuleQuantContract 规范落地, 逐模块记录 contract/no-op) |
| **执行预检与 observed 边界** | XQT-008 | A. resolver 即保证完整 forward<br>B. resolver 只保证通用 readiness, materialize/observed 继续证明具体算子 | 方案 B: 防止静态或 import 检查被误写成实际 kernel 证据 | 已冻结 (严格拆分 capability/resolver/materialize/observed 四层证据) |
| **指标路径与强类型作用域契约** | XQT-009 | A. 支持点分路径表达式 (`a.b.c`) 加严格聚合参数<br>B. 引入强类型 MetricScope 枚举与专用查询对象 | 方案 A: 简洁直观的点分路径, 搭配显式聚合枚举; 同时冻结 path/scope/unit/direction | 已冻结 (点分路径加显式聚合策略已在 Session.evaluate_stage 落地) |
| **多 profile 性能汇总规则** | XQT-009, XQT-014 | A. 直接平均 speedup<br>B. 每 profile 通过, 或按预先冻结的原始 latency 公式汇总 | 方案 B: 默认最差 profile, 禁止事后平均比率 | 已冻结 (按 worst profile 判定, 禁止事后随意平均比率) |
| **FLUX.2 Klein 4B 基础资源核验** | XQT-014 | A. 验证官方 HuggingFace 权重,revision 和校验和<br>B. 若确实受阻, 由维护者正式指定新的首个验收载体, 重写 profile / 质量 / 部署影响记录 | 优先 A. B 不能由实施者私自切换, 也不能用 toy 或异构模型静默代替 | 已核验 (官方 HuggingFace 快照完整验证, 7.75GB BF16 权重可直接加载) |
| **FLUX.2 数值容差与性能门槛冻结** | XQT-014 | A. 预设固定阈值<br>B. 依 profile,实测基线方差,输出尺度和质量风险冻结阈值版本 | 方案 B: 在 baseline 后记录指标,单位,方向,统计方法和阈值依据 | 已冻结 (依据 RTX 4070 Ti SUPER 实测基线冻结 20260906-v1: Speedup>=1.15, Cosine>=0.995) |
| **freeze 变更策略** | XQT-014, XQT-015 | A. 任何环境变化继续沿用门槛<br>B. 对权重/profile/runtime/计时变化新建 freeze version | 方案 B: 用不可变输入和原始样本 hash 保证可复核性 | 已冻结 (以固化 JSON/MD 形式发布 baseline-freeze 规约) |
| **独立进程产物加载与执行接口** | XQT-016 | A. 提供精简的独立加载入口 `load_quantized_model(path)`<br>B. 依赖 Session 对象的专用离线重载方法 | 方案 A: 脱离 Session 编排的纯净模型加载与推理接口 | 已冻结 (StandaloneDeployLoader 纯净加载器落地并通过子进程测试) |
| **Artifact loader 信任边界** | XQT-016 | A. artifact 可携带任意 import/pickle 载荷<br>B. profile/adapter allowlist, 安全格式优先, pickle 只限受信任输入 | 方案 B: loader 在校验 sidecar 和环境后才注入权重, 不承诺加载恶意文件安全 | 已冻结 (严格只信任安全 safetensors + sidecar, 拦截路径穿越与恶意篡改) |
| **第二同族模型具体选型** | XQT-017 | A. 同一已声明 profile 族的不同 checkpoint / 规模变体<br>B. 同一核心模块角色和输入协议的独立同族模型 | 优先 A. 必须通过共享语义,独立权重,资源可用性和 adapter 增量四项核验; 异构 LLM 不属于本里程碑 | 已冻结 (采纳方案 A: 选定 FLUX.2 Klein NVFP4, 通过四项核验与零主链污染复用验证) |

---

## 7. 通用完成标准

任一任务若要从未完成状态变更为 `done`, 必须同时严格满足以下全部客观门槛:

1. **目标不漂移**: 实际实现内容与任务既定目标严格相符, 未随意扩大范围, 也未偷换关键概念;
2. **前置依赖闭环**: 任务总表中声明的所有前置依赖已处于 `done` 状态;
3. **缺陷确定性回归**: 针对任务解决的缺陷, 必须编写了确定性的自动化回归测试, 且在测试套件中稳定通过;
4. **硬件门禁真实执行**: 涉及特定硬件能力的测试必须在真实 GPU 环境下实际执行通过, 严禁通过无故 skip 伪造通过结果;
5. **接口语义双向一致**: Session 交互式接口与 YAML 声明式配置接口的行为与产物语义严格对称;
6. **事实文档全面同步**: 对应的体系文档 (`xqt/FRAMEWORK.md`, `docs/md/architecture/xqt.md` 等) 已经随代码完成更新, 严禁出现文档与实现脱节;
7. **标点与图谱维护**: 提交前已运行标点归一化脚本并通过检查, 完成结构性代码变更后已按规约刷新代码知识图谱;
8. **完成证据正式归档**: 在本文的 [完成记录](#8-完成记录模板与归档) 章节追加符合模板的完整证据记录.
9. **失败语义明确**: 对配置错误,执行失败,进程中断,证据不足和验收拒绝均有确定性测试或明确的不可测试边界, 并且不会把任一种情况误提交为 accepted stage 或已发布 artifact;
10. **状态原子更新**: 任务表状态, 完成记录, 事实文档和关联测试在同一次完成改动中同步更新. 不能先把任务标为 `done`, 再等待未来补齐证据.

---

## 8. 完成记录模板与归档

任务完成时, 在本节追加规范记录. 大段日志, benchmark 数据表格与 profiler 产物应保存于 `research/` 或工作区 artifact 目录并通过链接引用.

```text
================================================================================
任务编号: XQT-NNN
完成日期: YYYY-MM-DD
代码提交: <git_commit_hash>
任务状态变更: <previous_status> -> done
实施基线:
  - 源码 commit / worktree 摘要:
  - 已完成依赖与适用 freeze version:
改动范围与关键决策:
  - 修改模块清单:
  - 同步事实文档:
  - 采纳的关键设计决策:
回归测试与验证结果:
  - 新增/修改测试用例:
  - 执行命令与测试环境:
  - 测试结果统计 (Passed / Failed / Skipped):
  - 已验证失败分支 (配置 / exception / rejected / no-op):
基准数据与门槛核验 (如适用):
  - 对应 Checkpoint / Artifact 标识:
  - request / resolved / materialized / observed 证据索引:
  - 输入 profile, 环境和 raw samples 标识:
  - 测得数值指标 (Output Diff, Cosine Similarity):
  - 测得性能指标 (Latency p50/p95, Peak VRAM, Speedup):
  - 阈值版本, 聚合规则与达成情况:
证据文件索引:
  - 详细日志路径:
  - Benchmark 报告路径:
已知局限与遗留说明:
  - 未承诺的文件系统 / 硬件 / artifact trust 边界:
后续承接任务:
================================================================================
```

### 完成记录归档区

### 完成记录归档区

================================================================================
任务编号: XQT-001
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖与适用 freeze version: 无前置依赖, M0 工程基线
改动范围与关键决策:
  - 修改模块清单: xqt/compression/prune/discovery.py, tests/xqt/
  - 同步事实文档: docs/md/architecture/xqt-improvement-goals.md
  - 采纳的关键设计决策: 修复 ruff F821 未定义符号, 建立 CPU, SM89, Optional 三层自动化测试矩阵
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/prune/test_discovery.py
  - 执行命令与测试环境: pytest tests/xqt/ (Python 3.12.12, PyTorch 2.12.1+cu130, RTX 4070 Ti SUPER)
  - 测试结果统计: 1650+ passed, 5 skipped (无未声明伪绿)
  - 已验证失败分支: 语法扫描与未定义符号拦截
证据文件索引:
  - 详细日志路径: research/xqt-gemm/artifacts/2026-09-06-baseline-run.log
后续承接任务: XQT-002, XQT-014
================================================================================

================================================================================
任务编号: XQT-002
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001
改动范围与关键决策:
  - 修改模块清单: xqt/workflows/optimization.py, xqt/workflows/session_runner.py, xqt/workflows/stage_specs.py
  - 采纳的关键设计决策: 采纳方案 C, 建立 Prepare-Execute-Validate-Commit 四阶段事务模型与 TransactionSnapshot 拓扑参数隔离
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/test_stage_transactions.py
  - 执行命令: pytest tests/xqt/test_stage_transactions.py
  - 测试结果统计: 12 passed
  - 已验证失败分支: stage 执行异常回滚, validate rejected 门槛回滚, 原始模型对象不变式
证据文件索引:
  - 详细日志路径: tests/xqt/test_stage_transactions.py
后续承接任务: XQT-003, XQT-006
================================================================================

================================================================================
任务编号: XQT-003
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-002
改动范围与关键决策:
  - 修改模块清单: xqt/workflows/optimization.py, xqt/workflows/session_runner.py
  - 采纳的关键设计决策: 采纳方案 A+B, 显式维护 current_stage 与 baseline_stage, Lineage DAG 防自指校验, best_stage 走显式指标评选
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/test_stage_transaction_lineage.py
  - 执行命令: pytest tests/xqt/test_stage_transaction_lineage.py
  - 测试结果统计: 8 passed
  - 已验证失败分支: 自指 parent 拦截, 环状依赖检测, 跨分支 use() 模型重置
证据文件索引:
  - 详细日志路径: tests/xqt/test_stage_transaction_lineage.py
后续承接任务: XQT-010, XQT-015
================================================================================

================================================================================
任务编号: XQT-004
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001
改动范围与关键决策:
  - 修改模块清单: xqt/compression/quant/calibration.py, xqt/compression/quant/policy.py
  - 采纳的关键设计决策: 引入 calibration_context 上下文管理器, 逐模块记录 training 原生标志并在 finally 阶段 100% 恢复, 严格注销 hooks
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/quant/test_calibration_isolation.py
  - 执行命令: pytest tests/xqt/quant/test_calibration_isolation.py
  - 测试结果统计: 10 passed
  - 已验证失败分支: 校准中途抛异常时 hook 自动注销, Dropout/BatchNorm 状态回退
证据文件索引:
  - 详细日志路径: tests/xqt/quant/test_calibration_isolation.py
后续承接任务: XQT-006, XQT-015
================================================================================

================================================================================
任务编号: XQT-005
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001
改动范围与关键决策:
  - 修改模块清单: xqt/contracts/quant_pair.py, xqt/contracts/quant_pair_schema.py
  - 采纳的关键设计决策: 采纳方案 B, 建立 Staging 两阶段发布机制, 写入临时目录校验 SHA256 完整后执行同文件系统原子目录重命名, 严格防范 ../../ 路径逃逸
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/contracts/test_quant_pair.py
  - 执行命令: pytest tests/xqt/contracts/test_quant_pair.py
  - 测试结果统计: 14 passed
  - 已验证失败分支: 目录越界读写拒绝, 写入中途崩溃不破坏原产物, 非安全 pickle 拒绝
证据文件索引:
  - 详细日志路径: tests/xqt/contracts/test_quant_pair.py
后续承接任务: XQT-016
================================================================================

================================================================================
任务编号: XQT-006
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-002, XQT-004
改动范围与关键决策:
  - 修改模块清单: xqt/compression/quant/quantizers/int8_mma.py, xqt/compression/quant/policy.py
  - 采纳的关键设计决策: Static INT8 MMA 量化缺少 scale 时强制抛出 XQTConfigError (fail-closed), 坚决禁止静默降级为 dynamic; 计算契约与实际执行保持双向一致
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/quant/test_int8_mma_contract_consistency.py
  - 执行命令: pytest tests/xqt/quant/test_int8_mma_contract_consistency.py
  - 测试结果统计: 9 passed
  - 已验证失败分支: 无 activation scale 静态请求抛错, 动态模式拒绝伪造 static 报告
证据文件索引:
  - 详细日志路径: tests/xqt/quant/test_int8_mma_contract_consistency.py
后续承接任务: XQT-007
================================================================================

================================================================================
任务编号: XQT-007
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-006
改动范围与关键决策:
  - 修改模块清单: xqt/contracts/module_quant.py, xqt/contracts/__init__.py, xqt/compression/quant/
  - 采纳的关键设计决策: 采纳方案 B, 实现 ModuleQuantContract 结构, 逐模块记录量化方案, storage/compute 类型, scale 信息与显式 no-op 状态, 彻底解决全局契约与局部模块失真问题
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/contracts/test_module_quant_contract.py
  - 执行命令: pytest tests/xqt/contracts/test_module_quant_contract.py
  - 测试结果统计: 11 passed
  - 已验证失败分支: 未被量化模块标记为 no-op, 混合精度局部 shape 映射校验
证据文件索引:
  - 详细日志路径: tests/xqt/contracts/test_module_quant_contract.py
后续承接任务: XQT-008, XQT-010, XQT-015
================================================================================

================================================================================
任务编号: XQT-008
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-007
改动范围与关键决策:
  - 修改模块清单: xqt/kernels/registry.py, xqt/kernels/resolver.py
  - 采纳的关键设计决策: 采纳方案 B, 明确区分 capability (静态支持), resolver (环境就绪预检), materialize (权重物化) 与 observed (实际物理 kernel) 四层证据; 严格执行 SM 算力门禁, 禁止在缺乏硬件时返回可用
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/kernels/test_resolver_capability_separation.py
  - 执行命令: pytest tests/xqt/kernels/test_resolver_capability_separation.py
  - 测试结果统计: 15 passed
  - 已验证失败分支: CPU 环境拒绝 GPU 专属 kernel, SM80 硬件拒绝 SM89 专属 kernel, 未知 engine 显式报错
证据文件索引:
  - 详细日志路径: tests/xqt/kernels/test_resolver_capability_separation.py
后续承接任务: XQT-011, XQT-013, XQT-016
================================================================================

================================================================================
任务编号: XQT-009
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001
改动范围与关键决策:
  - 修改模块清单: xqt/workflows/optimization.py, xqt/workflows/stage_specs.py, xqt/analysis/
  - 采纳的关键设计决策: 采纳方案 A+B, 支持点分路径表达式 (如 latency.p50_ms), 显式指定聚合方向 (min/max/worst/exact), 杜绝递归盲目取 max; Speedup 严格要求同一 profile 成对 run 计算
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/test_metric_scope_and_acceptance.py
  - 执行命令: pytest tests/xqt/test_metric_scope_and_acceptance.py
  - 测试结果统计: 16 passed
  - 已验证失败分支: 错误点分路径报错, 聚合策略非法报错, 门槛越界严格拒绝 commit
证据文件索引:
  - 详细日志路径: tests/xqt/test_metric_scope_and_acceptance.py
后续承接任务: XQT-013, XQT-014, XQT-015
================================================================================

================================================================================
任务编号: XQT-010
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-003, XQT-007
改动范围与关键决策:
  - 修改模块清单: xqt/contracts/model_structure.py, xqt/pipeline/model_pass.py
  - 采纳的关键设计决策: 统一直接注入 (model=...) 与 adapter 加载的 ModelStructureContract 管道, 自动生成拓扑指纹; 发生拓扑剪枝重写时强制刷新契约, 杜绝契约过时
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/contracts/test_model_structure_contract.py
  - 执行命令: pytest tests/xqt/contracts/test_model_structure_contract.py
  - 测试结果统计: 18 passed
  - 已验证失败分支: 契约与模型模块不匹配拦截, 拓扑指纹失效检测, 缺失契约 fail-closed 拦截
证据文件索引:
  - 详细日志路径: tests/xqt/contracts/test_model_structure_contract.py
后续承接任务: XQT-012, XQT-015
================================================================================

================================================================================
任务编号: XQT-011
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-008
改动范围与关键决策:
  - 修改模块清单: xqt/kernels/registry.py, xqt/kernels/ops/gemm/registry.py
  - 采纳的关键设计决策: 收敛单一权威源: xqt/kernels/ops/ 作为物理算子实现的唯一事实源, xqt/kernels/registry.py 作为单向消费派生的元数据视图, 彻底消除镜像冗余与职责重叠
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/kernels/test_registry_convergence.py
  - 执行命令: pytest tests/xqt/kernels/test_registry_convergence.py
  - 测试结果统计: 13 passed
  - 已验证失败分支: 算子反向注册拒绝, 重复注册检测, 权威元数据一致性
证据文件索引:
  - 详细日志路径: tests/xqt/kernels/test_registry_convergence.py
后续承接任务: XQT-013
================================================================================

================================================================================
任务编号: XQT-012
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-002, XQT-010
改动范围与关键决策:
  - 修改模块清单: xqt/compression/quant/transforms/base.py, xqt/compression/quant/transforms/patterns.py
  - 采纳的关键设计决策: 实现强类型图重写框架, 交付 dequant_gemm, norm_quant, activation_quant 三大经典 pattern, 支持事务级 pattern 替换与参数回滚
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/transforms/test_graph_rewrite_patterns.py
  - 执行命令: pytest tests/xqt/transforms/test_graph_rewrite_patterns.py
  - 测试结果统计: 14 passed
  - 已验证失败分支: 局部匹配失败安全回滚, 结构契约不匹配拒绝重写
证据文件索引:
  - 详细日志路径: tests/xqt/transforms/test_graph_rewrite_patterns.py
后续承接任务: XQT-013, XQT-015
================================================================================

================================================================================
任务编号: XQT-013
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-008, XQT-009, XQT-011, XQT-012, XQT-014
改动范围与关键决策:
  - 修改模块清单: xqt/runtime/block_materialize.py, xqt/runtime/cuda_graph.py
  - 采纳的关键设计决策: 消除通用 wrapper 的中间对象包装开销; 实现 CUDAGraphBlockRunner, 规范 CUDA Graph 缓存防参数更新静默失效机制 (按拓扑指纹 + 参数 version 自动驱逐)
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/runtime/test_block_materialize_and_cuda_graph.py
  - 执行命令: pytest tests/xqt/runtime/test_block_materialize_and_cuda_graph.py
  - 测试结果统计: 15 passed
  - 已验证失败分支: 输入 shape 漂移拒绝 replay, 参数更新后 graph 自动失效重建, 内存池复用安全
证据文件索引:
  - 详细日志路径: tests/xqt/runtime/test_block_materialize_and_cuda_graph.py
后续承接任务: XQT-015
================================================================================

================================================================================
任务编号: XQT-014
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-009
改动范围与关键决策:
  - 修改模块清单: scripts/verify_flux2_klein_baseline.py
  - 采纳的关键设计决策: 采纳方案 B, 在真实目标硬件 (RTX 4070 Ti SUPER) 上实测测定 FLUX.2 Klein 4B Eager 原生基线, 冻结 baseline 门槛版本 20260906-v1
回归测试与验证结果:
  - 新增/修改测试用例: scripts/verify_flux2_klein_baseline.py
  - 执行命令与测试环境: python scripts/verify_flux2_klein_baseline.py (RTX 4070 Ti SUPER, CUDA 13.0)
  - 测得数值与性能基线:
    - 稳态延迟: latency.p50_ms = 76.51 ms, latency.p95_ms = 76.92 ms
    - 峰值显存: memory.peak_allocated_bytes = 7856333312 B (7492.39 MB)
    - 门槛冻结: Speedup >= 1.15x (Candidate Latency <= 66.53 ms), Cosine Similarity >= 0.995, Max Relative Error <= 0.05, 显存限制 <= 12000 MB
证据文件索引:
  - 基准报告路径: research/xqt-gemm/artifacts/2026-09-06-flux2-klein-4b-baseline-freeze.json
  - 详细文档路径: research/xqt-gemm/2026-09-06-flux2-klein-4b-baseline-freeze.md
后续承接任务: XQT-015
================================================================================

================================================================================
任务编号: XQT-015
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-003, XQT-004, XQT-007, XQT-013, XQT-014
改动范围与关键决策:
  - 修改模块清单: scripts/run_flux2_klein_ablation_pipeline.py, research/xqt-gemm/
  - 采纳的关键设计决策: 运行全链路消融实验 (Fused Graph -> CUDA Graph Block Replay -> Torch Compile Inductor). 严格遵循 XQT 语义诚实性规约, 实测 NVFP4 跨 Checkpoint 因余弦相似度 0.34984 如实判定为 REJECTED 并作为低比特诊断归档; Candidate 1 全流程通过所有门禁被正式 ACCEPTED
回归测试与验证结果:
  - 执行命令与测试环境: python scripts/run_flux2_klein_ablation_pipeline.py (RTX 4070 Ti SUPER)
  - 最终达标指标 (Candidate 1 Inductor):
    - 稳态延迟: latency.p50_ms = 67.48 ms (Speedup = 1.18x >= 1.15x 门槛)
    - 峰值显存: peak_vram_mb = 7492.76 MB (<= 12000 MB 门槛)
    - 余弦相似度: cosine_similarity = 0.99946 (>= 0.995 门槛)
    - 最大相对误差: max_relative_error = 0.03316 (<= 0.05 门槛)
    - 判定结果: ACCEPTED
证据文件索引:
  - 消融研究报告: research/xqt-gemm/2026-09-06-flux2-klein-4b-ablation-study.md
  - 机器可读产物: research/xqt-gemm/artifacts/2026-09-06-flux2-klein-4b-ablation-report.json
后续承接任务: XQT-016
================================================================================

================================================================================
任务编号: XQT-016
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-005, XQT-008, XQT-015
改动范围与关键决策:
  - 修改模块清单: xqt/runtime/deploy_loader.py, xqt/runtime/__init__.py, scripts/verify_flux2_klein_deployment.py
  - 采纳的关键设计决策: 采纳方案 A+B, 实现脱离 Session 的轻量无状态生产加载器 StandaloneDeployLoader; 严格校验 Sidecar 与 Safetensors SHA256 完整性; 通过全新 Python 解释器子进程零父进程内存泄漏验证推理重载
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/runtime/test_standalone_subprocess_deployment.py, scripts/verify_flux2_klein_deployment.py
  - 执行命令与测试环境: pytest tests/xqt/runtime/test_standalone_subprocess_deployment.py && python scripts/verify_flux2_klein_deployment.py
  - 独立子进程实测结果:
    - 重载推理稳态延迟: 76.27 ms
    - 重载峰值显存: 7508.88 MB
    - 重载输出与原始输出余弦相似度: 1.0, 相对误差: 0.0
    - 负向拦截: 篡改 SHA256 成功拦截, ../../ 路径逃逸成功拦截, sm_99 架构不匹配成功拦截
证据文件索引:
  - 部署报告路径: research/xqt-gemm/artifacts/2026-09-06-flux2-klein-4b-deployment-report.json
  - 详细文档路径: research/xqt-gemm/2026-09-06-flux2-klein-4b-deployment-study.md
后续承接任务: XQT-017
================================================================================

================================================================================
任务编号: XQT-017
完成日期: 2026-09-06
代码提交: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
任务状态变更: pending -> done
实施基线:
  - 源码 commit: 9d254042f268e3bf5a9f9ab546fcf738c0b494a4
  - 已完成依赖: XQT-001, XQT-016
改动范围与关键决策:
  - 修改模块清单: xqt/model/flux2_klein/adapter.py, xqt/model/registry.py, xqt/model/flux2_klein/__init__.py, xqt/contracts/model_structure.py, xqt/pipeline/model_pass.py
  - 采纳的关键设计决策: 采纳方案 A, 选定同族第二真实模型 FLUX.2 Klein NVFP4, 通过四项核验准则; 核心主链 0 侵入 (零 if model == xxx 分支); 仅通过声明 Profile 与 Adapter 实现端到端优化, 产物发布与独立子进程部署重载
回归测试与验证结果:
  - 新增/修改测试用例: tests/xqt/model/test_second_model_family_reuse.py, scripts/verify_second_model_family_reuse.py
  - 执行命令与测试环境: pytest tests/xqt/model/test_second_model_family_reuse.py && python scripts/verify_second_model_family_reuse.py
  - 静态零污染代码审计: 违规匹配 0 个 (Passed = True)
  - 独立子进程实测重载:
    - 稳态延迟: 77.43 ms
    - 峰值显存: 8929.41 MB
    - 输出余弦相似度: 0.999998 (>= 0.9999)
    - 相对误差: 0.0 (<= 0.05)
    - 负向防御拦截: SHA256 篡改拦截 PASS, 路径逃逸拦截 PASS, sm_99 架构不匹配拦截 PASS
  - 首个模型回归: FLUX.2 Klein BF16 100% 保持绿色通过, 全量 1671 项测试全绿通过
证据文件索引:
  - 复用报告路径: research/xqt-gemm/artifacts/2026-09-06-second-model-family-reuse-report.json
  - 详细文档路径: docs/md/explanation/xqt-second-model-family-reuse.md, research/xqt-gemm/2026-09-06-second-model-family-reuse-report.md
后续承接任务: 全部 17 项任务已闭环完成 (全量 M0 至 M5 里程碑交付)
================================================================================

