# xqt

`xqt` 是 XDL 仓库中的量化,蒸馏,剪枝,扩散少步蒸馏和部署格式导出实验项目. 当前目录是包化实验工具链, 不代表稳定公共 API.

长期工作文档分两层:

- 摘要入口: [../docs/md/XQT_SUMMARY.md](../docs/md/XQT_SUMMARY.md)
- 详细事实源: [../docs/md/XQT.md](../docs/md/XQT.md)

可选依赖:

- `pip install -e ".[xqt]"`: ONNX/QDQ/torchao 基础路径.
- `pip install -e ".[xqt-hf]"`: HuggingFace 文本 KD/prune 路径.
- `pip install -e ".[xqt-diffusion]"`: diffusion/Flux/SD 类路径.
- `pip install -e ".[xqt-all]"`: XQT Python 侧全量可选依赖.

理解 XQT 时不要从顶层临时脚本推断项目能力. 这些脚本只保留为手动实验或示例入口,不计入长期文档中的模块状态,recipe backlog 或 API 边界. 可复用能力以 `xqt/` 包模块,`xqt/recipes/*.yaml`,测试和 [../docs/md/XQT_SUMMARY.md](../docs/md/XQT_SUMMARY.md), [../docs/md/XQT.md](../docs/md/XQT.md) 为准.

当前包模块:

- `core`: structured config, artifact manifest, checksum, XQT registry.
- `data`: synthetic classification/detection samples, calibration dataloader utilities, torchvision image classification loader, Ultralytics detection loader, `xdl.dataset` bridge loader.
- `model`: 外部模型 adapter, 当前包含 Ultralytics YOLO detection wrapper 和 dataset helper.
- `pipeline`: sequential pass manager 和最小 YAML runner.
- `workflows`: stage-based optimization workflow,把 eval/benchmark/prune/quant/finetune/distill/operator/export/deploy/runtime_eval 等阶段按配置编排.
- `eval`: tensor output diff, detection decode/mAP 和 metric flatten helper.
- `benchmark`: latency 和 memory benchmark helper.
- `quant`: quantization policy, backend capability matrix, activation calibration, layer sensitivity helper.
- `operator_opt`: `torch.compile`-first operator optimization pass, backend capability matrix and runtime fallback reporting.
- `quant.onnx_qdq`: ONNX Runtime static QDQ INT8 quantization adapter;执行器会把 QDQ graph summary,实际命中的 quantized op types 和 calibration summary 写入 report.
- `export`: `torch.export` ExportedProgram, TorchScript fallback, ONNX export/checker/runtime diff, FP16 ONNX conversion, TensorRT `trtexec` / Python API adapter 和性能阈值报告, OpenVINO optional adapter/dry-run.
- `export.mobile`: ExecuTorch `.pte`, `pnnx`/ONNX -> ncnn, ONNX -> MNN 的可选 adapter, 支持 dry-run 命令验证.
- `prune`: PyTorch global L1 pruning, sparsity report, CNN structured channel/filter pruning, ViT/Transformer structured MLP neuron pruning, attention head pruning, block pruning, and N:M structured sparsity reports.
- `prune.schedule`: pruning schedule and prune + KD helper.
- `distill`: logit KD, feature/relation distillation loss, feature hook helper.
- `distill.cache`: teacher logits/features disk cache helper.
- `distill.hf_text`: HuggingFace 文本分类 teacher/student bundle 和 KD recipe 支架.
- `distill.training`: small PyTorch teacher -> student logit distillation helper.
- `diffusion_distill`: few-step timestep schedule helper.
- `diffusion_distill.cache`: prompt, latent and trajectory cache helper.
- `diffusion_distill.losses`: consistency/LCM style latent distillation loss helper.
- `diffusion_distill.report`: fixed-seed image grid and sampling report metadata.
- `export.capability`: deployment format capability matrix.
- `xdl_adapter`: 从 XDL TrainSetup-like 对象或 checkpoint 创建 XQT context.

阅读和扩展顺序:

1. 先看 `core/schema.py` 和 `core/config.py`,确认 recipe schema 和 OmegaConf 加载规则.
2. 再看 `pipeline/runner.py` 和 `pipeline/passes.py`,确认默认 pass 顺序和真实执行行为.
3. 按任务进入 `quant/`,`prune/`,`distill/`,`diffusion_distill/`,`export/`,`operator_opt/`,`eval/`,`benchmark/` 对应 helper.
4. 最后看 `recipes/*.yaml` 和 `tests/xqt/`,确认当前路径是否已经有可运行闭环.

可复用逻辑不要沉到顶层示例脚本里. 新增后端或压缩能力时,优先放进对应包模块,再用 recipe 和测试验证.

当前 recipe:

- `recipes/smoke_cpu.yaml`: CPU-only schema 和 runner smoke recipe, 用于验证配置,PyTorch native export 和 manifest 路径.
- `recipes/image_resnet_onnx_qdq_int8.yaml`: ResNet/CNN ONNX Runtime QDQ INT8 + TensorRT dry-run smoke recipe, 包含真实 `trtexec` 运行时使用的 `performance_thresholds` 示例.
- `recipes/image_resnet_cifar100_qdq_cpu.yaml`: 使用本地 CIFAR-100 的 ResNet ONNX Runtime QDQ CPU recipe.
- `recipes/yolo_detection_smoke.yaml`: synthetic detection + toy detection module smoke recipe, 用于验证 detection schema,baseline eval,prune,ONNX export 和 manifest.
- `recipes/yolo_detection_practice.yaml`: detection stage workflow recipe,显式拆分 baseline eval/latency,ONNX export/runtime eval,QDQ artifact/runtime eval 和 global L1 pruning 阶段. 它属于 `OptimizationConfig + stages[]` 主链,不是 pass recipe;operator/deploy 能力由独立 recipe 和测试覆盖,不塞进默认 practice 主链.
- `recipes/detection/yolo_detection_trt_qdq_practice.yaml`: detection QDQ -> TensorRT INT8 practice recipe,当前默认使用 TensorRT Python API dry-run 固定 build 参数,artifact 和 report 结构,为真实 engine build 预留主链.
- `recipes/detection/external_detection_trt_deploy.yaml`: 外部 detection ONNX -> TensorRT deploy workflow 样例,不要求加载 PyTorch 模型,用于承接 RT-DETR 等仓库外导出的 ONNX 资产.
- `recipes/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval.yaml`: 当前最接近真实部署意义的 detection workflow 样例. 直接消费外部 RT-DETR ONNX,不依赖 `ultralytics`,覆盖 QDQ INT8,TensorRT 真 engine 和 `coco8_detection` 上的 runtime_eval.
- `recipes/multi_component_quant_smoke.yaml`: toy `vision_encoder -> projector -> decoder` 异构量化 smoke recipe, 用于验证 component policy,多 backend metrics 和 manifest 表达.
- `recipes/prune_finetune_cpu.yaml`: 线性 sparsity schedule + teacher KD 微调的 CPU smoke recipe.
- `recipes/cnn_structured_prune.yaml`: chain-like CNN 结构化 channel pruning CPU smoke recipe.
- `recipes/vit_structured_prune.yaml`: ViT structured pruning CPU smoke recipe, 支持通过 override 切换 `mlp_neuron`, `head`, `block`.
- `recipes/hf_text_kd_prune.yaml`: HF 文本分类 KD + 全局 L1 非结构化剪枝 recipe 支架, 依赖 `transformers` 和 `datasets`, 不下载权重到 git.

示例入口:

- `examples/yolo_detection_practice.py`: 单文件 detection practice 入口,默认内嵌一个不依赖 `ultralytics` 的轻量 stage workflow,实际调用 `xqt.workflows.optimize_model` 并打印 stage 摘要,workflow manifest 和 result 路径. 需要切到 YAML 或真实 RT-DETR TensorRT workflow 时,用 `XQT_YOLO_PRACTICE_CONFIG` 指向对应 recipe.

量化 recipe 约定:

- `compression.quant.strategy` 显式写出 backend 目标策略,不要只藏在 `policy`.
- `onnxruntime_qdq` recipe 显式写 `calibration_split`,并提供对应 `data.calibration`; preflight 和 runner 不再把 `validation` 当隐式 calibration fallback.
- `torchao` recipe 显式写 `skip_quantize` / `keep_high_precision`,避免默认规则隐式变化.
- `xqt.quant.capability` 记录当前可用和计划中的 backend 能力边界; preflight 会把 CUDA,calibration 和 ONNX exportable graph 等约束写入检查结果.
- `gptq`,`awq`,`bitsandbytes` 目前只是 planned backend 接口预留,可进入 config/preflight,但 runner 不会执行.
- `operator_optimization` 当前只有 `torch_compile` 会实际执行;report 会记录 `metadata.graph_break_report` 和 `metadata.fallback_detail`. `deployment_backend`,`triton`,`tilelang`,`custom_cuda` 当前进入 config/preflight 和 manifest,但不会在 built-in executor 里执行内核替换.
- YOLO detection practice 当前只把 global L1 unstructured pruning 当作 sparsity/report baseline,report 会显式标记 `speedup_claimed=false`. detection `method=structured` 会被 preflight warning 和 workflow 执行层 guard,并在 report 中写出 `execution_state=skipped`,`skip_reason` 和 `skipped_modules[]`,直到 residual,CSP/C2f,concat,SPPF 和 detect head 的 dependency graph/rewrite 支持补齐.
- `xqt.workflows.optimize_model` 是 v0.x 的破坏性 stage workflow 入口. 新 practice/研究入口优先表达为 `data_splits + stages`,不要再新增一个任务一个 workflow 的专用配置模型.
- TensorRT/OpenVINO dry-run 会构造部署命令或 Python API build 计划并写入 manifest;真实 engine/IR 生成需要目标机器安装对应后端并关闭 dry-run. TensorRT 当前支持 `trtexec` 和 `python_api` 两条 adapter 路径. TensorRT/OpenVINO 自身的 graph fusion, kernel selection 或 engine 优化属于部署后端收益,不要写成 XQT operator replacement gain.

运行入口:

- `xqt-preflight`: 不解析命令行参数,默认检查 `recipes/smoke_cpu.yaml`.
- `XQT_CONFIG=/abs/path/to/recipe.yaml xqt-preflight`: 检查 recipe 的 target,数据 root,可选依赖,后端命令和硬件要求.
- `xqt-run-recipe`: 不解析命令行参数,默认运行 `recipes/smoke_cpu.yaml`.
- `XQT_CONFIG=/abs/path/to/recipe.yaml xqt-run-recipe`: 切换 recipe.
- `XQT_WRITE_MANIFEST=0 xqt-run-recipe`: 跳过 manifest 写入.
- `xqt-run-workflow`: 不解析命令行参数,默认运行 `recipes/detection/yolo_detection_smoke.yaml`.
- `XQT_WORKFLOW_CONFIG=/abs/path/to/workflow.yaml xqt-run-workflow`: 切换 stage workflow recipe.

detection 相关说明:

- detection recipe 现在优先按 stage workflow 理解和执行,不是旧 pass recipe 的变体.
- 对更有部署意义的真实 detection 模型,当前优先路径是外部 ONNX -> QDQ -> TensorRT,而不是把 `ultralytics` 运行时依赖引入 `xqt` 主链.
