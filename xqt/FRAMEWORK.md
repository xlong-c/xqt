# XQT 框架形态与工程契约

本文是 `xqt/` 包内的工程契约,用于约束框架形态,职责边界,使用场景和新增能力规范. `docs/md/XQT.md` 仍是 XQT 的长期事实源;本文负责给直接修改 `xqt/` 的开发者一个更短的边界说明.

本文已吸收两份旧 XQT 草稿中仍然有效的内容. 与本契约冲突的旧设想不再保留.

## 1. 一句话定位

XQT 负责从训练产物到推理产物之间的模型变换链路:

```text
PyTorch model / checkpoint / exported artifact
    -> optional model compression or graph transform
    -> calibration / analysis / runtime diff
    -> export target artifact
    -> benchmark / manifest / report
```

XQT 专注模型本身,图变换,压缩误差,部署格式和运行时验证. 它不拥有训练生命周期,不拥有通用 trainer,不拥有任务级 model/loss/metric registry.

## 2. 框架形态

XQT 的主形态是 `config -> context -> stage/pass -> artifacts/metrics/manifest`.

- `config`: YAML/OmegaConf 或 Python API 只描述模型来源,数据 split,压缩策略,导出目标和验收阈值.
- `context`: `XQTContext` 持有当前模型,reference model,数据 split,产物,指标和 provider.
- `stage/pass`: 执行量化,剪枝,算子优化,导出,分析,benchmark 或 runtime_eval.
- `artifacts`: ONNX,QDQ ONNX,TensorRT engine,OpenVINO IR,torch.export,TorchScript,mobile artifact 等.
- `metrics`: output diff,layer error,sparsity,QDQ graph stats,calibration summary,latency,memory 和 provider 返回的任务指标.
- `manifest`: 记录配置快照,源 checkpoint,压缩维度,产物校验和,指标和执行阶段.

长期用户心智优先收敛到 stage workflow: 每个 stage 显式声明 `name`,`kind`,`split`/`calibration_split`/`validation_split`,`params` 和 `accept`. Pythonic 探索模式可以通过 `XQTOptimizationSession` 逐步调用同一套 stage 执行核心,但不应在脚本里手写整份完整 workflow dict.

## 3. 职责边界

XQT 应该负责:

- 模型压缩和图变换: PTQ,QDQ,权重量化,剪枝 mask/rewrite,算子替换接入,导出前适配.
- 无梯度校准: observer fit,activation range 统计,ONNX Runtime QDQ calibration reader 和 calibration summary.
- 误差诊断: tensor diff,layer diff,SQNR,cosine similarity,推荐高精度模块,剪枝敏感度和 runtime boundary report.
- 部署产物转换: ONNX,TensorRT,OpenVINO,torch.export,TorchScript,ExecuTorch,ncnn,MNN 等 adapter.
- 性能与产物记录: latency,memory,artifact manifest,report,backend capability 和 preflight 检查.

XQT 不应该负责:

- 自己实现通用 `optimizer.zero_grad() -> backward() -> step()` 训练循环.
- 自己维护通用 trainer,model registry,loss registry 或 task metric registry.
- 自己复制 XDL,Ultralytics,HuggingFace,timm,diffusers 等任务框架的训练语义.
- 自己实现推理引擎或后端优化器,例如 TensorRT tactic selection,OpenVINO graph optimizer 或 ONNX Runtime kernel.
- 把 smoke-only model/helper 宣传成通用模型能力.

任何带反向传播的流程都必须委托给 provider. XDL provider 走 `xdl.trainer.Trainer`;第三方 provider 走第三方自己的 train/evaluate API 或用户传入的 callable. XQT 只构造 job,传入模型,数据 split,teacher/reference 和参数,再收集 provider report.

## 4. QAT 和恢复训练边界

QAT,剪枝 recovery,KD recovery 和微调都属于带梯度流程,训练循环由 XDL,纯 PyTorch 脚本或第三方 provider 执行.

XQT 在这些场景里只负责模型侧能力:

- 准备或描述 QAT/QDQ 所需的量化策略,模块选择,observer/fake-quant 配置和导出约束.
- 在训练前后对模型做压缩图变换,转换或导出.
- 对训练后的 QAT/压缩模型做 QDQ/export/runtime diff/benchmark.
- 把 provider 返回的训练指标写入 XQT report,但不解释为 XQT 自己训练出的结果.

如果用户使用原生 PyTorch 做 QAT,推荐形态是:

```text
user torch training loop / XDL Trainer / third-party Trainer
    -> QAT or recovery checkpoint
    -> XQT convert/export/analyze/runtime_eval/benchmark
```

`calibration_split` 只表示 PTQ/observer 无梯度数据源. `train_split` 只表示 provider 训练数据源. `validation_split` 用于 provider metric,runtime diff 或 benchmark 验证.

## 5. 使用场景

### 5.1 PTQ/QDQ 部署

适合 CNN,ViT,detection 或外部 ONNX 资产:

```text
baseline eval
    -> export fp32 ONNX
    -> static INT8 QDQ calibration
    -> QDQ ONNX
    -> runtime_eval
    -> TensorRT/OpenVINO/ONNX Runtime benchmark
```

XQT 在这个场景中拥有主链路,因为没有反向传播训练循环.

### 5.2 剪枝与压缩诊断

适合结构化剪枝,非结构化剪枝,N:M 或 block sparse 报告:

```text
baseline eval/benchmark
    -> prune transform
    -> sparsity/report
    -> optional provider recovery
    -> eval/runtime_eval/benchmark
```

如果剪枝后需要恢复精度,必须显式提供 training provider. 没有 provider 时,XQT 只能做剪枝变换和误差/性能评估.

### 5.3 QAT 后导出

适合需要训练时适应量化噪声的模型:

```text
external QAT training
    -> trained fake-quant or quant-aware checkpoint
    -> XQT convert/QDQ/export
    -> runtime_eval/benchmark/manifest
```

XQT 可以提供 QAT-ready model transform 或转换 adapter,但不承载 QAT optimization loop.

### 5.4 算子接入与部署验证

适合用户自带 Triton,TileLang,CUTLASS,custom CUDA 或 TensorRT plugin:

```text
operator wrapper
    -> PyTorch-side diff and benchmark
    -> export compatibility check
    -> deployment artifact
    -> runtime_eval
```

XQT 负责接入,验证,报告和导出编排. 算子代码,plugin 编译和后端内部调优归用户或目标后端.

## 6. Recipe 和 API 规范

- 固定可复现链路优先写 stage workflow YAML.
- 交互探索优先用 `XQTOptimizationSession`,不要在 Python 脚本里拼完整 workflow dict.
- 示例脚本必须是薄入口,只选择 YAML,构造最少参数,调用 `optimize_model` 或 session API.
- 量化 recipe 必须显式写 `backend`/`strategy` 和 `policy`;不要依赖隐式默认策略.
- QDQ/PTQ recipe 必须显式写 `calibration_split`,不能把 validation 数据隐式当 calibration fallback.
- 带梯度恢复或 QAT 的 recipe 必须显式提供 provider,并在报告中标注 provider 名称.
- 任务级 metric 优先走 `evaluation_provider`;XQT 内部 task metric fallback 只能作为过渡兼容,不能扩张成长期任务评估系统.
- 每条压缩或导出路径都应至少产出 output diff 或 task metric,以及最小 latency benchmark.
- planned/capability-only 后端必须在 preflight 和文档中明确标注,不能写成已经可执行闭环.
- 新增长期文档时遵守仓库文档分层: `docs/md/XQT.md` 是事实源,`xqt/FRAMEWORK.md` 是包内工程契约,阶段性研究继续放 `research/`.

## 7. 不再单独维护的草稿

两份旧 XQT 草稿已经收口到本文:

- 保留定位,场景,stage workflow,校准,误差分析和算子接入原则.
- 保留职责边界,provider 模型,PTQ calibration 与训练 split 的区分,以及 XQT registry/loss/metric 边界.

不再保留的旧内容包括:

- 把 XQT 写成半个训练框架的设想.
- 把 `finetune`/`distill` 写成 XQT 自有训练能力的表述.
- 把未来 planned backend 写成已落地执行能力的表述.
- 把任务 metric,任务模型或训练 loss 纳入 XQT 长期 API 的表述.
