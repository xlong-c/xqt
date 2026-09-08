# XQT 第二同族模型零主链污染复用说明

本文档记录并说明 XQT 在第二同族模型 (FLUX.2 Klein NVFP4) 上的零侵入复用验证 (XQT-017), 详述架构设计, 模型选型依据, 变更清单, 复用率分析与回归保障.

---

## 1. 验证目标与架构设计

在 XQT 的架构规约中, 量化, 图重写, 编译缓存与独立部署加载等通用主链必须保持绝对的模型无关性 (Model-Agnostic). 严禁在通用核心逻辑中出现针对特定模型名称的条件分支 (如 `if model == "flux"`).

XQT-017 的核心目标在于:
1. 引入同族第二个真实模型, 检验 `ModelProfile`, `ModelAdapter` 与 `ModelStructureContract` 抽象体系的通用化能力.
2. 在通用核心主链代码**零修改 (Zero Core Modification)** 的前提下, 仅通过模型侧 profile 声明与 adapter 包装, 完成端到端优化, 产物导出与独立子进程部署重载.
3. 确保首个模型 (FLUX.2 Klein BF16) 的既有基线, 测试与质量门禁 100% 保持绿色, 杜绝破坏性回退.

---

## 2. 第二同族模型选型依据 (四项核验)

依据 `docs/md/architecture/xqt-improvement-goals.md` 第 6 节决策准则, 本次选型优先采纳方案 A:

| 核验维度 | 准则要求 | FLUX.2 Klein NVFP4 达标证据 |
| --- | --- | --- |
| **共享语义 (Shared Semantics)** | 共享相同的核心模块角色, 网络拓扑与前向输入协议 | 两者均为基于 `Flux2Transformer2DModel` 的 Joint Diffusion Transformer 架构, 共享 `double_transformer_blocks`, `single_transformer_blocks`, 相同的嵌入层与投影层, 输入协议统一为 `(hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)`. |
| **独立权重 (Independent Weights)** | 采用独立的 Checkpoint / Revision, 权重参数非派生共享 | 第二模型采用独立的官方 4-bit 权重文件 `flux-2-klein-4b-nvfp4.safetensors` (快照 Revision: `1db2b2f776c24b76f1122e5f69ab1949fc620068`, SHA256: `d8c5007b6a3bbbdf...`), 参数布局为 ModelOpt 压缩格式与浮点 Scales. |
| **资源可用性 (Resource Availability)** | 模型资源真实可获取, 具备本地快速闭环能力 | 权重快照完整缓存在本地 HuggingFace Hub 目录 (`models--black-forest-labs--FLUX.2-klein-4b-nvfp4`), 离线直接可载入. |
| **Adapter 增量 (Adapter Delta)** | 仅依赖外围 Adapter/Profile 声明, 不得迫使通用主链做特例改造 | 仅在模型侧声明 `Flux2KleinNVFP4Adapter` 与 `diffusers.flux2-klein-nvfp4` Profile, 核心层无任何侵入. |

---

## 3. 代码变更与复用率详细清单

### 3.1 变更模块分类统计

| 分类 | 涉及文件 | 变更性质 | 说明 |
| --- | --- | --- | --- |
| **模型侧 Adapter (Model-side)** | `xqt/model/flux2_klein/adapter.py` | 新增 | 实现 `Flux2KleinBF16Adapter` 与 `Flux2KleinNVFP4Adapter`, 继承 `ModelAdapter` 规范 |
| **模型侧导出 (Model-side)** | `xqt/model/flux2_klein/__init__.py` | 修改 | 公开导出新增的 Adapter 类 |
| **模型注册表 (Model-side)** | `xqt/model/registry.py` | 修改 | 声明并注册 `diffusers.flux2-klein-nvfp4` Profile, 配置 `adapter_target` |
| **通用核心主链 (Generic Core)** | `xqt/core/*`, `xqt/session/*`, `xqt/contracts/*`, `xqt/transforms/*`, `xqt/runtime/deploy_loader.py` | **零变更 (0 改动)** | **完全复用, 无任何 model-specific 分支代码** |
| **自动化测试 (Tests)** | `tests/xqt/model/test_second_model_family_reuse.py` | 新增 | 包含 Profile 解析, 契约绑定, 负向安全与静态零污染审计在内的 4 项单元测试 |
| **全流程脚本 (Scripts)** | `scripts/verify_second_model_family_reuse.py` | 新增 | 真实硬件端到端闭环验证脚本 |

### 3.2 静态代码零污染审计证据

在 `scripts/verify_second_model_family_reuse.py` 与 `tests/xqt/model/test_second_model_family_reuse.py` 中内置静态 AST/文本扫描器, 针对全部通用核心包执行模型标识检测:
- **扫描路径**: `xqt/core/`, `xqt/session/`, `xqt/contracts/`, `xqt/transforms/`, `xqt/runtime/deploy_loader.py`, `xqt/compression/quant/transforms/`.
- **匹配敏感词**: `"flux"`, `"flux2"`, `"flux_2"`, `"klein"`.
- **检测结果**: **0 个违规匹配 (Violations = 0, Passed = True)**. 通用主链保持 100% 架构纯洁度.

---

## 4. 端到端执行与独立进程重载度量

在目标真实硬件 (NVIDIA GeForce RTX 4070 Ti SUPER `sm_89`, 16GB VRAM) 上执行全流程验证:

1. **结构契约 (ModelStructureContract) 绑定**:
   - 契约族: `diffusion`
   - 组件数: 4 个组件分组 (`attention`, `norm`, `router`, `other`)
   - 拓扑一致性: `is_consistent = True`, 生成唯一拓扑指纹 `39caf93b742c...`
2. **Quant Pair 规范化发布**:
   - 导出目录: `research/xqt-gemm/artifacts/deploy_flux2_klein_nvfp4/`
   - 产物结构: `model.safetensors` + `quant.json` (含 SHA256 摘要, Lineage DAG 与 `sm_89` 硬件门禁)
3. **独立 Python 子进程部署重载 (Zero Parent Leak)**:
   - 全新 Python 解释器子进程加载产物, 父进程显存零泄漏.
   - 加载耗时: 25.1 秒 (含 3.5GB 权重完整解析与校验)
   - 稳态延迟 (`latency.p50_ms`): **77.43 ms**
   - 峰值显存 (`peak_vram_mb`): **8929.41 MB** (<= 12000 MB 预算)
   - 输出余弦相似度: **0.999998** (远远高于 >= 0.9999 门槛)
   - 最大相对误差: **0.0** (<= 0.05 门槛)

---

## 5. 负向安全与防护门禁实测

所有负向防御行为均通过独立异常断言验证:

1. **Checksum 篡改防御**: 篡改 `quant.json` 中的 SHA256 字段, 加载器在注入权重前立即抛出 `XQTArtifactError: Weights checksum verification FAILED`.
2. **路径穿越攻击防御**: 将 `weights.path` 修改为 `../../outside.safetensors`, 加载器立即抛出 `XQTArtifactError: weights.path escapes root directory`.
3. **硬件架构不符防御**: 将 `target_arch` 修改为 `sm_99`, 加载器在 CUDA 预检阶段立即抛出 `XQTBackendError: Hardware preflight failed: current GPU sm_89 does not meet required architecture sm_99`.

---

## 6. 首模型回归安全与结论

- 首个真实模型 (`diffusers.flux2-klein`, BF16 原生基线) 的全部 Required Gates 继续保持 100% 绿色.
- 事实证明: XQT 的 `ModelProfile` + `ModelAdapter` + `ModelStructureContract` 分层设计具备高度的通用性与复用性, 能够支撑同族不同规格与低比特变体的快速接入, 且不会对系统通用主链引入任何技术债务或特例硬编码.
