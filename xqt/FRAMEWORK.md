# XQT 包内工程契约

**核心边界：XQT 只关注模型本身。**

XQT 消费训练后的模型/checkpoint/导出产物，做压缩、变换、导出、误差分析、benchmark。
XQT 不做训练、不做 QAT、不做 finetune/distill/recovery、不构建 dataset/dataloader、不拥有 task provider 语义。

## 关键抽象

| 概念 | 说明 |
|---|---|
| `XQTConfig` | v1 pass recipe schema，内部实现，包含 `compression` / `operator_optimization` / `export` / `validation` / `benchmark` / `analysis` |
| `OptimizationConfig` | 对外 stage workflow schema，包含 `project` / `model` / `task` / `stages` |
| `load_xqt_config()` | `XQTConfig` 加载器，`OmegaConf.structured(XQTConfig)` 打底 + YAML 覆盖 |
| `load_optimization_config()` | `OptimizationConfig` 加载器，同上模式 |
| Pass | 原子操作单位，通过 `register_pass(name)` 注册。内置 pass 见 `xqt/pipeline/passes.py` |
| Stage | 面向用户的工作流编排层。`stage.kind` → 对应 pass name，`stage.params` → 传给 pass |
| `XQTContext` | 阶段间传递的共享上下文（模型引用、输入、产物记录） |
| `OptimizationStageResult` | 单个 stage 执行结果（模型、diff、benchmark 数据、artifacts） |
| `ArtifactManifest` / `ArtifactRecord` | 产物追踪。`ArtifactRecord` 记录单次导出/变换产物，`ArtifactManifest` 汇总 |
| `MetricRecord` | 结构化指标记录（latency、memory、diff 等），可序列化 |
| Preflight | 配置前置校验。检查 schema、backend 可用性、capability 标注 |
| `example_inputs` | 调用方传入的模型前向输入（导出/benchmark/layer analysis/operator_opt 需要） |
| `calibration_inputs` | 调用方传入的 calibration 输入 iterable（PTQ/QDQ 需要） |

**量化命名约定**：

| 字段 | 含义 | 示例 |
|---|---|---|
| `backend` | 运行时/执行后端 | `torchao` `onnxruntime_qdq` `pytorch` |
| `method` | 量化算法 | `awq` `gptq` |
| `strategy` | 后端内模式/dtype | `dynamic_int8` `static_int8` `fp8_dynamic` |

AWQ/GPTQ 写在 `method`，不写在 `backend`。

**量化语义约定 —— 真量化 vs 伪量化 vs 融合 vs 去量化节点**：

量化策略产出的 speedup 取决于它在硬件层实际减少了什么。XQT 要求所有量化报告和 capability 必须区分以下概念。

### 真量化 (TRUE quantization, `QuantizationNature.TRUE`)

定义：量化后的运算使用**原生低精度 tensor core MMA 指令**，将低精度输入直接送入 ALU，每周期计算量随元素打包密度线性提升。

- 硬件行为：`mma.sync` / `wgmma` / `tcgen05.mma` 的 K 维度翻倍（fp8/int8 吃 K=32，fp16 只能吃 K=16）。
- 速度来源：**算力提升**。每周期吞吐翻倍，速度提升应接近理论值。
- 示例：`fp8_dynamic`（W8A8）走 `mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32`。

### 伪量化 (PSEUDO quantization, `QuantizationNature.PSEUDO`)

定义：量化后的运算需要将**权重/激活反量化回 fp16/bf16** 后再送入 MMA，tensor core 仍跑 fp16 路径。量化只是存储压缩手段。

- 硬件行为：存储是低精度（省显存带宽），但 MMA 走 `m16n8k16.f16`（K=16，没翻倍）。
- 速度来源：**仅内存带宽**。算力不变，速度提升受限于 weight-load 节省，典型收益远低于理论 packing 比。
- 示例：`fp8_weight_only`（W8A16）、`int4_weight_only`（W4A16）、`dynamic_int8`。

判断方法：

| 策略 | A 精度 | B(weight) 精度 | MMA 指令 | nature |
|---|---|---|---|---|
| fp8_dynamic | fp8 | fp8 | m16n8k32 f32.e4m3.e4m3.f32 | TRUE |
| fp8_weight_only | fp16 | fp8 (dequant→fp16) | m16n8k16 f16.f16.f16.f32 | PSEUDO |
| int8_weight_only | fp16 | int8 (dequant→fp16) | m16n8k16 f16.f16.f16.f32 | PSEUDO |
| int4_weight_only | fp16 | int4 (dequant→fp16) | m16n8k16 f16.f16.f16.f32 | PSEUDO |
| dynamic_int8 | int8→fp16 | int8→fp16 | m16n8k16 f16 | PSEUDO |
| ONNX QDQ INT8 | QDQ→int8 | QDQ→int8 | 取决于后端硬件 | PSEUDO (CPU) / TRUE (GPU with dp4a) |

### 算子融合 (operator fusion)

定义：将量化/反量化操作与后继计算（如 matmul、conv）**合并为单一 kernel**，消除中间显存读写。

- 融合前：`dequant weight (load fp8→fp16) → matmul (fp16)`，两个 kernel + 一次中间写入。
- 融合后：`fused_dequant_matmul (load fp8→fp16 in-register → mma)`，一个 kernel，零中间写入。
- 报告中通过 `QuantizationReport.fusion_applied` 字段记录已融合的算子对。

**去量化节点消除 (dequant node elimination)**：

定义：在计算图中，如果反量化节点 (`DequantizeLinear`) 后紧跟着一个支持原生低精度输入的后端算子（如 TensorRT 的 `Myelin` 或 ONNX Runtime 的 `QLinearMatMul`），则反量化节点被图优化器**删除**，由算子直接消费低精度张量。

- 与伪量化的关键区别：伪量化在 PyTorch 层面保留了显式的 `torch.dequantize()` 调用或在 kernel 内做反量化；去量化节点消除是图优化器层面的**编译时消除**。
- 报告中通过 `QuantizationReport.dequant_nodes_eliminated` 记录消除数量。

### 报告字段

`QuantizationReport` 中与语义相关的新增字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `nature` | `QuantizationNature` | TRUE / PSEUDO / UNKNOWN |
| `compute_speedup_expected` | `Optional[float]` | TRUE 时预期算力提速比（基于 packing ratio）；PSEUDO 时为 None |
| `fusion_applied` | `list[str]` | 已应用的算子融合对，如 `["dequant+matmul"]` |
| `dequant_nodes_eliminated` | `int` | 图优化层面消除的去量化节点数 |

## 主链路

```
模型输入 (nn.Module / checkpoint / 导出产物)
  → 压缩或图变换 (quant / prune / operator_opt)
  → 模型侧 diff / layer analysis (TensorDiff / compare_tensors / layer_error_rows)
  → 导出目标产物 (ONNX / TensorRT / OpenVINO / torch.export / mobile)
  → benchmark / manifest
```

## 职责边界

| 负责 | 不负责 |
|---|---|
| PTQ / QDQ / 权重量化 / activation calibration | 训练循环 (`zero_grad → backward → step`) |
| 结构化 / 非结构化 / N:M / block sparse 剪枝 | QAT 训练 / finetune / distill / KD recovery |
| `torch.compile` / Triton / TileLang / CUTLASS 算子优化 | dataset / dataloader 构建 |
| ONNX / TensorRT / OpenVINO / torch.export / TorchScript / ExecuTorch / ncnn / MNN 导出 | task-level validation / 准确率评估 |
| 模型间 output diff / layer error / sensitivity 分析 | training provider / evaluation provider |
| latency / memory benchmark | 通用 trainer / loss registry / metric registry |
| recipe / manifest / report 工程记录 | 后端内部优化器 (TensorRT tactic / OpenVINO graph optimizer 等) |

需要梯度的流程归 XDL 或外部训练工具，XQT 只消费训练后的模型。

## 外部输入

XQT 不拥有 data schema。需要前向输入时由调用方显式传入：

- `example_inputs` — 导出 / benchmark / layer analysis / operator_opt 所需输入
- `calibration_inputs` — PTQ / QDQ calibration 所需输入

`train_split` / `validation_split` / `calibration_split` 不属于 XQT schema。

## 配置方式（仅两种）

| 优先级 | 方式 | 适用场景 |
|---|---|---|
| 第一 | `XQTOptimizationSession` (Python) | 探索、调试、notebook |
| 第二 | stage workflow YAML (`optimize_model()`) | 可复现批量实验 |

原则上没有其他配置方式。不要新增 CLI 参数解析或硬编码 workflow。

## Provisional API

```
XQTOptimizationSession
optimize_model  /  load_optimization_config
OptimizationConfig  /  OptimizationStageConfig  /  OptimizationStageResult
OptimizedModelResult  /  StageAcceptanceConfig
ArtifactManifest  /  ArtifactRecord  /  MetricRecord
xdl_setup_to_xqt_context  /  xdl_checkpoint_to_xqt_context  /  load_checkpoint_into_model
```

子模块 (`xqt.core` / `xqt.pipeline` / `xqt.quant` / `xqt.prune` / `xqt.operator_opt` / `xqt.export` / `xqt.analysis` / `xqt.benchmark` / `xqt.model`) 内部符号按 Internal 处理，不从顶层重新导出。

## Stage 约定

- 支持的 stage：`benchmark` `prune` `quant` `operator` `export` `deploy` `analyze`
- 禁止的 stage：`finetune` `distill` `eval` `runtime_eval`
- 量化 recipe 必须显式写 `backend` 和 `policy`
- QDQ / PTQ 的 `calibration_inputs` 由调用方传入，recipe 不声明数据来源
- planned / capability-only 后端必须在 preflight 和文档中标注

## 文档分层

- `docs/md/XQT.md` — 长期事实源
- `xqt/FRAMEWORK.md` — 包内工程契约（本文件）
- `research/` — 阶段性研究
