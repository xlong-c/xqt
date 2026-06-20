# xqt

`xqt` 是 XDL 仓库中的量化,蒸馏,剪枝,扩散少步蒸馏和部署格式导出实验项目. 当前目录仍是实验脚本集合, 不代表稳定公共 API.

长期架构草案见 [../docs/md/XQT.md](../docs/md/XQT.md).

可选依赖:

- `pip install -e ".[xqt]"`: ONNX/QDQ/torchao 基础路径.
- `pip install -e ".[xqt-hf]"`: HuggingFace 文本 KD/prune 路径.
- `pip install -e ".[xqt-diffusion]"`: diffusion/Flux/SD 类路径.
- `pip install -e ".[xqt-all]"`: XQT Python 侧全量可选依赖.

当前脚本:

- `torchao_vit.py`: ViT + torchao FP8 量化,逐层误差分析和性能测试实验.
- `bf16_clein.py`: FLUX.2 klein BF16 推理实验.
- `sdnq_clein.py`: FLUX.2 klein SDNQ 4bit 动态量化推理实验.

当前包模块:

- `core`: structured config, artifact manifest, checksum, XQT registry.
- `data`: synthetic classification samples, calibration dataloader utilities, torchvision image classification loader, `xdl.dataset` bridge loader.
- `pipeline`: sequential pass manager 和最小 YAML runner.
- `eval`: tensor output diff 和 metric flatten helper.
- `benchmark`: latency 和 memory benchmark helper.
- `quant`: quantization policy, backend capability matrix, activation calibration, layer sensitivity helper.
- `operator_opt`: `torch.compile`-first operator optimization pass, backend capability matrix and runtime fallback reporting.
- `quant.onnx_qdq`: ONNX Runtime static QDQ INT8 quantization adapter.
- `export`: `torch.export` ExportedProgram, TorchScript fallback, ONNX export/checker/runtime diff, TensorRT `trtexec` adapter 和性能阈值报告, OpenVINO optional adapter.
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

当前 recipe:

- `recipes/smoke_cpu.yaml`: CPU-only schema 和 runner smoke recipe, 用于验证配置,PyTorch native export 和 manifest 路径.
- `recipes/image_resnet_onnx_qdq_int8.yaml`: ResNet/CNN ONNX Runtime QDQ INT8 + TensorRT dry-run smoke recipe, 包含真实 `trtexec` 运行时使用的 `performance_thresholds` 示例.
- `recipes/image_resnet_cifar100_qdq_cpu.yaml`: 使用本地 CIFAR-100 的 ResNet ONNX Runtime QDQ CPU recipe.
- `recipes/image_vit_torchao_fp8.yaml`: ViT torchao FP8 CUDA recipe, 当前默认使用 `cuda:0`,可在支持 FP8 的 NVIDIA GPU 上直接跑 quant + benchmark; 真实收益仍应结合 pretrained 权重和目标任务数据复验.
- `recipes/multi_component_quant_smoke.yaml`: toy `vision_encoder -> projector -> decoder` 异构量化 smoke recipe, 用于验证 component policy,多 backend metrics 和 manifest 表达.
- `recipes/prune_finetune_cpu.yaml`: 线性 sparsity schedule + teacher KD 微调的 CPU smoke recipe.
- `recipes/cnn_structured_prune.yaml`: chain-like CNN 结构化 channel pruning CPU smoke recipe.
- `recipes/vit_structured_prune.yaml`: ViT structured pruning CPU smoke recipe, 支持通过 override 切换 `mlp_neuron`, `head`, `block`.
- `recipes/hf_text_kd_prune.yaml`: HF 文本分类 KD + 全局 L1 非结构化剪枝 recipe 支架, 依赖 `transformers` 和 `datasets`, 不下载权重到 git.

量化 recipe 约定:

- `compression.quant.strategy` 显式写出 backend 目标策略,不要只藏在 `policy`.
- `onnxruntime_qdq` recipe 显式写 `calibration_split`.
- `torchao` recipe 显式写 `skip_quantize` / `keep_high_precision`,避免默认规则隐式变化.
- `xqt.quant.capability` 记录当前可用和计划中的 backend 能力边界; preflight 会把 CUDA,calibration 和 ONNX exportable graph 等约束写入检查结果.
- `gptq`,`awq`,`bitsandbytes` 目前只是 planned backend 接口预留,可进入 config/preflight,但 runner 不会执行.
- `operator_optimization` 当前只有 `torch_compile` 会实际执行;`deployment_backend`,`triton`,`tilelang`,`custom_cuda` 当前进入 config/preflight 和 manifest,但不会在 built-in executor 里执行内核替换.

运行入口:

- `xqt-preflight`: 不解析命令行参数,默认检查 `recipes/smoke_cpu.yaml`.
- `XQT_CONFIG=/abs/path/to/recipe.yaml xqt-preflight`: 检查 recipe 的 target,数据 root,可选依赖,后端命令和硬件要求.
- `xqt-run-recipe`: 不解析命令行参数,默认运行 `recipes/smoke_cpu.yaml`.
- `XQT_CONFIG=/abs/path/to/recipe.yaml xqt-run-recipe`: 切换 recipe.
- `XQT_WRITE_MANIFEST=0 xqt-run-recipe`: 跳过 manifest 写入.
