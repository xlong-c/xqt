# XQT 模型优化工具链

本文是 `xqt/` 的长期事实源. 当前 XQT 的核心契约是:

```text
XQT 只关注模型本身.
```

XQT 接收 PyTorch 模型,checkpoint 或已导出模型产物,执行模型侧压缩,图变换,导出适配,误差分析和运行时验证. 训练,QAT 训练,finetune,distillation,KD recovery,prune recovery,training provider 和 evaluation provider 不属于 XQT.

需要梯度更新的流程归 XDL 或第三方训练工具. XQT 只消费训练后的模型或 checkpoint.

## 1. 项目定位

主链路:

```text
PyTorch model / checkpoint / exported artifact
    -> model compression or graph transform
    -> model-side diff / layer analysis
    -> export target artifact
    -> runtime diff / benchmark / manifest
```

核心目标:

- 支持 PyTorch `nn.Module` 和 `state_dict` 作为主要输入.
- 覆盖 PTQ/QDQ,权重量化,剪枝,算子优化,导出前适配和部署格式转换.
- 能导出 ONNX,TensorRT engine,OpenVINO IR,torch.export,TorchScript,ExecuTorch,ncnn,MNN 等产物.
- 记录配置,源 checkpoint,指标,产物校验和执行阶段,保证模型转换结果可复现.
- 与 XDL 训练体系保持松耦合. XDL 负责训练/QAT/recovery;XQT 负责训练后的模型优化和部署验证.

非目标:

- 不替代 `xdl.trainer.Trainer`.
- 不维护通用 model/loss/metric registry.
- 不运行 `optimizer.zero_grad() -> backward() -> step()` 训练循环.
- 不承载 QAT 训练,finetune,distillation 或任何 recovery training.
- 不重新实现 TensorRT,OpenVINO,ONNX Runtime,ExecuTorch,ncnn,MNN 等后端.
- 不引入命令行参数解析库. 入口脚本读取 YAML 配置或明确的环境变量.

## 2. 当前包边界

`xqt/` 当前作为仓库顶层实验包维护,不并入 `xdl/` 主框架.

Provisional API:

- `load_xqt_config`
- `run_xqt_recipe`
- `preflight_xqt_config`
- `optimize_model`
- `load_optimization_config`
- `XQTOptimizationSession`
- `OptimizedModelResult`
- `OptimizationConfig`
- `OptimizationStageConfig`
- `OptimizationStageResult`
- `StageAcceptanceConfig`
- `XQTConfig`
- `ArtifactManifest`,`ArtifactRecord`,`MetricRecord`
- `xdl_setup_to_xqt_context`,`xdl_checkpoint_to_xqt_context`,`load_checkpoint_into_model`

Internal implementation:

- `xqt.core`
- `xqt.pipeline`
- `xqt.quant`
- `xqt.prune`
- `xqt.operator_opt`
- `xqt.export`
- `xqt.eval`
- `xqt.benchmark`
- `xqt.data`
- `xqt.model`
- `xqt.integrations`

## 3. 模块划分

```text
xqt/
├── core/          # config schema, manifest, registry, import helpers
├── data/          # calibration/validation sample loaders
├── model/         # smoke model helpers and forward hook output capture
├── integrations/  # detection output decode adapter
├── pipeline/      # pass manager, built-in passes, preflight, YAML runner
├── workflows/     # stage workflow/session API for model optimization
├── quant/         # quantization policy, calibration, QDQ, sensitivity
├── prune/         # pruning masks, structured rewrite, sparsity reports
├── operator_opt/  # torch.compile and backend capability/report adapters
├── export/        # torch.export,TorchScript,ONNX,TensorRT,OpenVINO,mobile
├── eval/          # tensor diff, layer analysis, runtime reports
├── benchmark/     # latency and memory benchmark helpers
├── recipes/       # smoke, quant, prune, operator, detection recipes
└── xdl_adapter.py # model/checkpoint context bridge, not training bridge
```

Removed from XQT:

- `distill/`
- `diffusion_distill/`
- `training_provider` / `evaluation_provider`
- `XDLTrainingProvider`
- `finetune` / `distill` stage
- HF text KD recipe and prune recovery recipe

## 4. 数据角色

XQT 只保留模型优化需要的数据角色:

- `calibration`: PTQ/observer/QDQ calibration 的代表性输入.
- `validation`: output diff,runtime diff,smoke metric 和 benchmark 的输入.

`train_split` 不属于 XQT schema. 如果方法需要训练数据或反向传播,先在 XDL 或第三方训练工具中完成训练,再把训练后模型/checkpoint 交给 XQT.

## 5. Stage Workflow

`OptimizationConfig` 是当前主要用户心智. 支持的 stage:

| Stage | 职责 |
| --- | --- |
| `eval` | 对当前 PyTorch 模型做无梯度 smoke eval 或 detection eval. |
| `benchmark` | 对当前模型做 latency/memory benchmark. |
| `prune` | 执行模型侧剪枝,mask/rewrite 和 sparsity report. |
| `quant` | 执行 PTQ/QDQ/torchao 等量化路径. |
| `operator` | 执行 `torch.compile` 或记录 planned backend capability/report. |
| `export` | 导出 ONNX,torch.export,TorchScript 等产物. |
| `deploy` | 导出部署后端产物或 dry-run plan. |
| `analyze` | 分析 layer diff,activation drift,prune candidates 和高精度保留建议. |
| `runtime_eval` | 对导出产物做 runtime diff/metric/latency 验证. |

不支持的 stage:

- `finetune`
- `distill`
- `qat_train`
- `recovery`

## 6. Recipe 约定

- 量化 recipe 必须显式写 `backend` 和 `policy`;`backend` 表示执行/运行时后端,如 `torchao`,`onnxruntime_qdq`,`pytorch`,`tilelang`;`method` 表示量化算法,如 `awq`,`gptq`;`strategy` 表示后端内的具体模式或 dtype 策略,如 `static_int8`,`dynamic_int8`,`fp8_dynamic`.
- AWQ/GPTQ 这类算法不能写成 `backend: awq` 或 `backend: gptq`,应写成 `backend: pytorch` 或 `backend: tilelang` 加 `method: awq` / `method: gptq`.
- QDQ/PTQ recipe 必须显式写 `calibration_split`.
- `validation_split` 用于 diff,metric 或 benchmark,不能隐式当作 calibration fallback.
- 剪枝 recipe 只能表达模型侧剪枝和 report,不能表达 recovery training.
- planned/capability-only 后端必须在 preflight 和文档中明确标注,不能写成已执行闭环.
- 每条压缩或导出路径至少产出 output diff 或 smoke metric,以及最小 latency benchmark.

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

data_splits:
  calibration:
    target: synthetic_classification
    sample_limit: 4
    batch_size: 1
    params:
      input_shape: [3, 224, 224]
      num_classes: 10
  validation:
    target: synthetic_classification
    sample_limit: 4
    batch_size: 1
    params:
      input_shape: [3, 224, 224]
      num_classes: 10

stages:
  - name: baseline_eval
    kind: eval
    split: validation
  - name: prune_sparse
    kind: prune
    split: validation
    params:
      method: global_l1_unstructured
      target_sparsity: 0.2
  - name: qdq_quant
    kind: quant
    from_stage: prune_sparse
    calibration_split: calibration
    validation_split: validation
    params:
      backend: onnxruntime_qdq
      strategy: static_int8
      policy:
        input_names: [input]
        output_names: [output]
  - name: deploy
    kind: deploy
    split: validation
    params:
      targets:
        - format: onnx
          output_path: artifacts/xqt/image_resnet_stage_workflow/model.onnx
          params:
            runtime_diff: true
```

## 7. 能力分层

量化:

- torchao weight-only / FP8 adapter.
- ONNX Runtime static QDQ INT8 adapter.
- calibration dataloader,activation statistics,calibration summary.
- layer sensitivity and mixed precision recommendation.

剪枝:

- global L1 unstructured pruning.
- structured channel/filter/MLP/head/block pruning helpers.
- N:M and block sparse reports.
- pruning schedule 仅表示逐步应用模型剪枝,不包含 recovery training.

算子优化:

- built-in executor 当前以 `torch_compile` 为主要实际执行路径.
- Triton/TileLang/CuTile/CUTLASS/custom CUDA 当前主要作为 capability/adapter/report 边界.
- TensorRT/OpenVINO 自身 graph fusion,kernel selection 或 engine/IR 优化属于部署后端收益,不写成 XQT operator replacement gain.

导出:

- `torch.export` / TorchScript.
- ONNX export/checker/runtime diff.
- TensorRT `trtexec` / Python API adapter.
- OpenVINO optional adapter.
- ExecuTorch/ncnn/MNN mobile adapter.

## 8. 当前最小闭环

建议第一轮闭环:

```text
PyTorch image classification model
    -> baseline eval
    -> ONNX export
    -> ONNX Runtime QDQ INT8 quantization
    -> output diff
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
