# XQT 模型优化工具链

本文是 `xqt/` 的长期事实源. 当前 XQT 的核心契约是:

```text
XQT 只关注模型本身.
```

XQT 接收 PyTorch 模型,checkpoint 或已导出模型产物,执行模型侧压缩,图变换,导出适配,误差分析和 benchmark. 训练,QAT 训练,finetune,distillation,KD recovery,prune recovery,dataset/dataloader,training provider 和 evaluation provider 不属于 XQT.

需要梯度更新的流程归 XDL 或第三方训练工具. XQT 只消费训练后的模型或 checkpoint.

## 1. 项目定位

主链路:

```text
PyTorch model / checkpoint / exported artifact
    -> model compression or graph transform
    -> model-side diff / layer analysis
    -> export target artifact
    -> benchmark / manifest
```

核心目标:

- 支持 PyTorch `nn.Module` 和 `state_dict` 作为主要输入.
- 覆盖 PTQ/QDQ,权重量化,剪枝,算子优化,导出前适配和部署格式转换.
- 能导出 ONNX,TensorRT engine,OpenVINO IR,torch.export,TorchScript,ExecuTorch,ncnn,MNN 等产物.
- 记录配置,源 checkpoint,指标,产物校验和执行阶段,保证模型转换结果可复现.
- 与 XDL 训练体系保持松耦合. XDL 负责训练/QAT/recovery/dataset/task validation;XQT 负责训练后的模型优化,分析和部署产物适配.

非目标:

- 不替代 `xdl.trainer.Trainer`.
- 不维护通用 model/loss/metric registry.
- 不运行 `optimizer.zero_grad() -> backward() -> step()` 训练循环.
- 不承载 QAT 训练,finetune,distillation 或任何 recovery training.
- 不构建 dataset/dataloader,不拥有 `data`/`data_splits` schema,不做 task-level validation.
- 不重新实现 TensorRT,OpenVINO,ONNX Runtime,ExecuTorch,ncnn,MNN 等后端.
- 不引入命令行参数解析库. 入口脚本读取 YAML 配置或明确的环境变量.

## 2. 当前包边界

`xqt/` 当前作为仓库顶层实验包维护,不并入 `xdl/` 主框架.

Provisional API:

- `optimize_model`
- `load_optimization_config`
- `XQTOptimizationSession`
- `OptimizedModelResult`
- `OptimizationConfig`
- `OptimizationStageConfig`
- `OptimizationStageResult`
- `StageAcceptanceConfig`
- `ArtifactManifest`,`ArtifactRecord`,`MetricRecord`
- `xdl_setup_to_xqt_context`,`xdl_checkpoint_to_xqt_context`,`load_checkpoint_into_model`

Internal implementation:

- `load_xqt_config`,`XQTConfig`,`run_xqt_recipe`,`preflight_xqt_config`
- `xqt.core`
- `xqt.pipeline`
- `xqt.quant`
- `xqt.prune`
- `xqt.operator_opt`
- `xqt.export`
- `xqt.analysis`
- `xqt.benchmark`
- `xqt.model`

## 3. 模块划分

```text
xqt/
├── core/          # config schema, manifest, registry, import helpers
├── model/         # smoke model helpers and forward hook output capture
├── pipeline/      # pass manager, built-in passes, preflight, YAML runner
├── workflows/     # stage workflow/session API for model optimization
├── quant/         # quantization policy, calibration, QDQ, sensitivity
├── prune/         # pruning masks, structured rewrite, sparsity reports
├── operator_opt/  # torch.compile and backend capability/report adapters
├── export/        # torch.export,TorchScript,ONNX,TensorRT,OpenVINO,mobile
├── eval/          # tensor diff, layer analysis, report helpers
├── benchmark/     # latency and memory benchmark helpers
├── recipes/       # smoke, quant, prune, operator, detection recipes
└── xdl_adapter.py # model/checkpoint context bridge, not training bridge
```

Removed from XQT:

- `distill/`
- `diffusion_distill/`
- `integrations/` (→ `xdl/metric/detection_utils.py`)
- `training_provider` / `evaluation_provider`
- `XDLTrainingProvider`
- `data/`
- `finetune` / `distill` stage
- `eval` / `runtime_eval` stage
- HF text KD recipe and prune recovery recipe

## 4. 外部输入契约

XQT 不拥有数据层. 具体约束:

- 不提供 `xqt.data` 包.
- 不构建 dataset,dataloader 或 XDL dataset bridge.
- 不维护 `data`,`data_splits`,`train_split`,`validation_split`,`calibration_split` 配置字段.
- 不做 task-level validation,accuracy,mAP 或 label 解析.

需要模型前向输入时,调用方在 Python API 中显式传入:

- `example_inputs`: 一批模型输入,用于 export,benchmark,layer analysis 或 operator optimization.
- `calibration_inputs`: iterable,用于 PTQ/QDQ/observer calibration.

如果方法需要训练数据,验证数据或反向传播,先在 XDL 或第三方训练工具中完成,再把训练后模型/checkpoint 或评估指标交给 XQT.

## 5. 配置方式与 Stage Workflow

XQT 只提供两种配置方式, 前者为第一选择:

### 第一配置: `XQTOptimizationSession` (Python 交互式 session)

```python
from xqt import XQTOptimizationSession

session = XQTOptimizationSession(
    project={"name": "my_optimization", "artifact_dir": "artifacts/..."},
    model=my_model,
    model_config={...},
    task={"type": "detection", ...},
    example_inputs=example_inputs,
    calibration_inputs=calibration_inputs,
)

# 逐步编排 stage, state 自动在 session 内累积
session.benchmark(name="baseline", warmup=1, iterations=10)
session.prune(name="prune_l1", method="global_l1_unstructured", target_sparsity=0.3, from_stage="initial")
session.quant(name="quant_int8", backend="onnxruntime_qdq", strategy="static_int8", policy={...})
session.export(name="export_onnx", format="onnx", output_path="model.onnx", opset=18)

result = session.result()  # OptimizedModelResult
```

Session 提供 6 种 stage 方法和状态管理: `benchmark()` / `prune()` / `quant()` / `operator()` / `export()` / `analyze()`, 以及 `revert_to()`, `set_example_inputs()`, `result()` 等. 模型快照, 延迟对比和接受度阈值由 session 内部自动管理.

### 第二配置: stage workflow YAML

`OptimizationConfig` 通过 YAML 声明式定义多阶段优化链路, 通过 `optimize_model()` 运行. 原则上除了以上两种配置方式, XQT 不提供其他配置路径 (不要新增 CLI 参数解析, JSON-shaped Python dict 硬编码 workflow 等).

这里的 YAML workflow 指包含 `project`,`model`,`task`,`stages` 的 stage 配置. 早期 `XQTConfig` pass recipe 仍可作为内部测试和实现装配格式存在,但不再是 `xqt` 顶层公共 API,也不再提供 `xqt-run-recipe` 或 `xqt-preflight` 安装后命令入口.

| Stage | 职责 |
| --- | --- |
| `benchmark` | 对当前模型做 latency/memory benchmark. |
| `prune` | 执行模型侧剪枝,mask/rewrite 和 sparsity report. |
| `quant` | 执行 PTQ/QDQ/torchao 等量化路径. |
| `operator` | 执行 `torch.compile` 或记录 planned backend capability/report. |
| `export` | 导出 ONNX,torch.export,TorchScript 等产物. |
| `deploy` | 导出部署后端产物或 dry-run plan. |
| `analyze` | 分析 layer diff,activation drift,prune candidates 和高精度保留建议. |

不支持的 stage:

- `eval`
- `runtime_eval`
- `finetune`
- `distill`
- `qat_train`
- `recovery`

## 6. Recipe 约定

- 量化 recipe 必须显式写 `backend` 和 `policy`;`backend` 表示执行/运行时后端,如 `torchao`,`onnxruntime_qdq`,`pytorch`,`tilelang`;`method` 表示量化算法,如 `awq`,`gptq`;`strategy` 表示后端内的具体模式或 dtype 策略,如 `static_int8`,`dynamic_int8`,`fp8_dynamic`. **新增**: recipe 的 `strategy` 必须通过 `_STRATEGY_NATURE` 映射表获得明确的 `nature` 分类;新策略若未分类默认为 `UNKNOWN`,直到有人在 capability 矩阵中标注.
- 量化报告和 capability 描述必须区分 `nature` (TRUE 真量化 vs PSEUDO 伪量化),`fusion_applied` (算子融合) 和 `dequant_nodes_eliminated` (去量化节点消除). 伪量化方案不得在报告中暗示或写成"已获得算力提升".
- AWQ/GPTQ 这类算法不能写成 `backend: awq` 或 `backend: gptq`,应写成 `backend: pytorch` 或 `backend: tilelang` 加 `method: awq` / `method: gptq`.
- QDQ/PTQ 执行时必须由调用方显式传入 `calibration_inputs`;recipe 只描述 backend/policy/artifact.
- 导出,benchmark,operator 和 analysis 需要前向样例时,由调用方显式传入 `example_inputs`.
- 剪枝 recipe 只能表达模型侧剪枝和 report,不能表达 recovery training.
- planned/capability-only 后端必须在 preflight 和文档中明确标注,不能写成已执行闭环.
- 压缩或导出路径至少产出 artifact/manifest/report;task metric 和验证闭环交给 XDL 或用户项目.

stage workflow 示例:

```yaml
project:
  name: image_resnet_stage_workflow
  artifact_dir: artifacts/xqt/image_resnet_stage_workflow

model:
  target: xdl.model.resnet18
  params:
    num_classes: 10
  device: cpu

task:
  type: classification

stages:
  - name: prune_sparse
    kind: prune
    params:
      method: global_l1_unstructured
      target_sparsity: 0.2
  - name: qdq_quant
    kind: quant
    from_stage: prune_sparse
    params:
      backend: onnxruntime_qdq
      strategy: static_int8
      policy:
        input_names: [input]
        output_names: [output]
  - name: deploy
    kind: deploy
    params:
      targets:
        - format: onnx
          output_path: artifacts/xqt/image_resnet_stage_workflow/model.onnx
          params:
            runtime_diff: true
```

运行上面的 workflow 时,调用方需要用 Python API 传入 `example_inputs` 和,若启用 QDQ/PTQ,`calibration_inputs`.

## 7. 能力分层

量化:

- torchao weight-only / FP8 adapter.
- `backend: pytorch` 下的 `strategy: fp4_weight_only` 已有 group-wise reference Linear weight-only 路径,支持分组 scale 量化和误差分析,不代表高性能 kernel 已完成.
- ONNX Runtime static QDQ INT8 adapter.
- activation statistics 和 calibration summary;校准输入来自外部 `calibration_inputs`.
- layer sensitivity,mixed precision recommendation,以及按层输出/权重分布统计.

**量化语义约定** (真量化 vs 伪量化 vs 融合 vs 去量化节点):

XQT 要求所有量化策略必须标明 `nature` 字段,区分以下几种优化机制,避免把"存储压缩"包装成"算力提升":

| nature | 定义 | speedup 来源 | K 维度 | 示例策略 |
|---|---|---|---|---|
| `TRUE` | 原生低精度 tensor core MMA,K 维度翻倍 | 算力提升 | K=32 (fp8/int8) | `fp8_dynamic` (W8A8) |
| `PSEUDO` | 存储低精度,计算前反量化到 fp16,走 fp16 MMA | 仅内存带宽 | K=16 | `fp8_weight_only`, `int4_weight_only`, `int8_weight_only`, `dynamic_int8` |
| `UNKNOWN` | 未分类 | 未知 | 未知 | 新策略默认值 |

关键判断依据: 只需看一条——**MMA 指令的 K 维度是否翻倍**。
- K=32: 真量化,算力翻倍 (`mma.sync m16n8k32 f32.e4m3.e4m3.f32`)
- K=16: 伪量化,算力不变 (`mma.sync m16n8k16 f32.f16.f16.f32`)

`QuantizationReport` 新增字段:

| 字段 | 类型 | 含义 |
|---|---|---|
| `nature` | `QuantizationNature` | TRUE / PSEUDO / UNKNOWN |
| `compute_speedup_expected` | `Optional[float]` | TRUE 时预期理论 speedup;PSEUDO 时为 None |
| `fusion_applied` | `list[str]` | 已应用的算子融合对 (如 `["dequant+matmul"]`) |
| `dequant_nodes_eliminated` | `int` | 图优化层面消除的去量化节点数 |

**算子融合**: 将 dequant + matmul 合并为单一 kernel,消除中间显存读写. 检测方法: 融合后应只有一个 kernel launch,中间不再有显存中转.

**去量化节点消除**: 计算图中 `DequantizeLinear` 节点被后端算子直接消费低精度张量而消除. 量化 report 中由 `dequant_nodes_eliminated` 记录. 与伪量化的区别: 伪量化在 PyTorch 层保留显式 `dequantize()` 调用,去量化节点消除是图编译器的编译时优化.

新增能力或后端时,必须在 capability 矩阵中显式标注 `nature`,不写默认 `UNKNOWN`.

剪枝:

- global L1 unstructured pruning.
- structured channel/filter/MLP/head/block pruning helpers.
- N:M and block sparse reports.
- pruning schedule 仅表示逐步应用模型剪枝,不包含 recovery training.

算子优化:

- built-in executor 当前以 `torch_compile` 为主要实际执行路径.
- TileLang 当前已有一个最小可执行 target: `attention` pattern 可在 toy attention model 上进入 executor,完成模块替换,numeric diff 和 benchmark;CPU 路径走 reference fallback,CUDA 路径已有最小真实 TileLang attention kernel,但当前仍限定在 float16,`dropout_p=0` 和 `seq_kv >= seq_q`.
- Triton/CuTile/CUTLASS/custom CUDA 当前主要还是 capability/adapter/report 边界.
- TensorRT/OpenVINO 自身 graph fusion,kernel selection 或 engine/IR 优化属于部署后端收益,不写成 XQT operator replacement gain.

导出:

- `torch.export` / TorchScript.
- ONNX export/checker.
- TensorRT `trtexec` / Python API adapter,支持在 export target params 中声明 `plugin_libraries` 加载自定义 `.so` 插件库.
- OpenVINO optional adapter.
- ExecuTorch/ncnn/MNN mobile adapter.

TensorRT 自定义插件约定:

- `export.targets[].params.plugin_libraries`: `.so` 路径列表. `trtexec` 后端会展开为 `--dynamicPlugins` 和 `--setPluginsToSerialize`; `python_api` 后端会在 build / inspect / runtime benchmark 前显式 `ctypes.CDLL(..., RTLD_GLOBAL)` 加载.
- `export.targets[].params.serialize_plugin_libraries`: 是否把插件库附带写入 `trtexec --setPluginsToSerialize`,默认 `true`.
- preflight 会检查每个插件库路径是否存在,但不会尝试编译或校验插件 ABI.

场景 readiness 矩阵:

| 场景 | 当前状态 | 已验证证据 | 主要缺口 |
| --- | --- | --- | --- |
| FP4 量化 | 半可用 | `xqt/quant/fp4_backend.py` 已提供 `pytorch + fp4_weight_only` 的 group-wise reference Linear weight-only 路径,并有执行测试 | 还没有高性能 packed kernel,也没有完整 AWQ/GPTQ 闭环 |
| TileLang megakernel | 半可用 | `xqt/operator_opt/executor.py` 的 attention target 已能进入 executor,并在 report 中区分 `reference_fallback` 与 `cuda_tilelang_entry`;`xqt/operator_opt/kernels/tilelang/attention.py` 已接入最小真实 CUDA attention kernel | 仍只覆盖 attention/fp16/`dropout_p=0`/`seq_kv>=seq_q`,还没有更完整的 TileLang kernel 家族 |
| TensorRT + `.so` 插件 | 半可用,接近工程可用 | `xqt/export/tensorrt.py` 已支持 plugin libraries 的 build / inspect / runtime load,preflight 也会校验路径 | 仍缺真实插件 ABI 和目标部署环境的端到端验证 |
| 常规剪枝 / 误差分析 | 已基本可用 | activation drift,layer sensitivity,layer weight diff,输出/权重分布统计都已在 `xqt/quant/` 和 `xqt/analysis/` 接通 | 更高层任务准确率和业务指标仍需外部评测链路 |

## 8. 当前最小闭环

建议第一轮闭环:

```text
PyTorch image classification model
    -> ONNX export
    -> ONNX Runtime QDQ INT8 quantization
    -> TensorRT INT8 engine or dry-run report
    -> benchmark report
    -> manifest
```

完成这条闭环后,再扩展其他模型族和后端.

## 9. 外部资料

涉及第三方 API 时先查官方文档:

- torchao 文档: <https://docs.pytorch.org/ao/stable/index.html>
- PyTorch ONNX exporter: <https://docs.pytorch.org/docs/2.12/onnx.html>
- PyTorch pruning tutorial: <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>
- ONNX 文档: <https://onnx.ai/onnx/intro/>
- TensorRT quick start: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html>
- OpenVINO model preparation: <https://docs.openvino.ai/2026/openvino-workflow/model-preparation.html>
