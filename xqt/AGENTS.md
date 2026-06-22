# xqt — 模型压缩与部署实验目录

## 目录职责

- 存放量化,蒸馏,剪枝,扩散少步蒸馏和部署格式导出相关实验脚本与模型尝试
- 作为独立于 `xdl/` 主框架的专题实验区
- 长期工作文档见 `docs/md/XQT.md`

## 当前内容

### 实验脚本

- `torchao_vit.py`: ViT + torchao FP8 量化实验,逐层误差分析和性能测试
- `bf16_clein.py`: FLUX.2 klein BF16 推理实验
- `sdnq_clein.py`: FLUX.2 klein SDNQ 4bit 动态量化推理实验
- `xdl_adapter.py`: XDL TrainSetup/checkpoint 到 XQT context 的桥接

### 包模块

- `core/`: structured config, artifact manifest, checksum, XQT registry
- `data/`: synthetic classification samples, calibration dataloader utilities
- `pipeline/`: sequential pass manager 和最小 YAML runner
- `eval/`: tensor output diff 和 metric flatten helper
- `benchmark/`: latency 和 memory benchmark helper
- `quant/`: quantization policy, activation calibration, layer sensitivity helper
  - `quant.onnx_qdq`: ONNX Runtime static QDQ INT8 quantization adapter
- `prune/`: PyTorch global L1 pruning, sparsity report, basic structured rewrite helper
  - `prune.schedule`: pruning schedule and prune + KD helper
- `distill/`: logit KD, feature/relation distillation loss, feature hook helper
  - `distill.cache`: teacher logits/features disk cache helper
  - `distill.hf_text`: HuggingFace 文本分类 teacher/student bundle 和 KD recipe 支架
  - `distill.training`: small PyTorch teacher -> student logit distillation helper
- `diffusion_distill/`: few-step timestep schedule helper
  - `diffusion_distill.cache`: prompt, latent and trajectory cache helper
  - `diffusion_distill.losses`: consistency/LCM style latent distillation loss helper
  - `diffusion_distill.report`: fixed-seed image grid 和 sampling report metadata
- `export/`: torch.export ExportedProgram, TorchScript fallback, ONNX export/checker/runtime diff, TensorRT trtexec adapter 和性能阈值报告, OpenVINO optional adapter
  - `export.mobile`: ExecuTorch .pte, pnnx/ONNX -> ncnn, ONNX -> MNN 的可选 adapter
  - `export.capability`: deployment format capability matrix

### Recipes

- `recipes/smoke_cpu.yaml`: CPU-only schema 和 runner smoke recipe,验证配置,PyTorch native export 和 manifest 路径
- `recipes/image_resnet_onnx_qdq_int8.yaml`: ResNet/CNN ONNX Runtime QDQ INT8 + TensorRT dry-run smoke recipe
- `recipes/image_resnet_cifar100_qdq_cpu.yaml`: 使用本地 CIFAR-100 的 ResNet ONNX Runtime QDQ CPU recipe
- `recipes/image_vit_torchao_fp8.yaml`: ViT torchao FP8 CUDA recipe,默认使用 cuda:0
- `recipes/prune_finetune_cpu.yaml`: 线性 sparsity schedule + teacher KD 微调的 CPU smoke recipe
- `recipes/hf_text_kd_prune.yaml`: HF 文本分类 KD + 全局 L1 非结构化剪枝 recipe 支架

### 运行入口

- `xqt-preflight`: 检查 recipe 的 target,数据 root,可选依赖,后端命令和硬件要求
- `xqt-run-recipe`: 运行 recipe
- `entrypoints/`: 命令行入口脚本,通过环境变量 `XQT_CONFIG` 指定 recipe

### 可选依赖

- `pip install -e ".[xqt]"`: ONNX/QDQ/torchao 基础路径
- `pip install -e ".[xqt-hf]"`: HuggingFace 文本 KD/prune 路径
- `pip install -e ".[xqt-diffusion]"`: diffusion/Flux/SD 类路径
- `pip install -e ".[xqt-all]"`: XQT Python 侧全量可选依赖

## API 边界

- `xqt` 顶层导出进入 Provisional API: `load_xqt_config`, `run_xqt_recipe`, `preflight_xqt_config`, `XQTConfig`, `ArtifactManifest`, `ArtifactRecord`, `MetricRecord`
- `xqt` 到 XDL 的适配入口进入 Provisional API: `xdl_setup_to_xqt_context`, `xdl_checkpoint_to_xqt_context`, `load_checkpoint_into_model`
- 子模块内部实现(`xqt.core`, `xqt.pipeline`, `xqt.quant`, `xqt.prune`, `xqt.distill`, `xqt.diffusion_distill`, `xqt.export`)按 Internal 处理,先服务 recipe 验证

## 理解方式

- 理解 XQT 项目时,以 `docs/md/XQT.md`,`xqt/README.md`,`xqt/recipes/*.yaml`,`xqt/pipeline/runner.py`,`xqt/pipeline/passes.py` 和对应包模块为事实源.
- 顶层临时脚本只作为手动实验或示例入口保留,不要把它们计入 XQT 模块状态,长期路线,API 边界或 recipe backlog.
- XQT 主链路按 `config -> context -> pass pipeline -> artifacts/metrics/manifest` 理解;不要从某个示例脚本反推主架构.
- 可复用量化,剪枝,蒸馏,导出,评估或 benchmark 逻辑必须优先落在 `quant/`,`prune/`,`distill/`,`diffusion_distill/`,`export/`,`eval/`,`benchmark/` 等包模块中,再用 recipe 和测试验证.

## 修改约束

- 公共压缩或部署工具若可复用,再考虑抽到 `tools/` 或 `xdl/`
- 外部库依赖,设备要求,模型限制要写清楚
- 顶层示例脚本要保持薄入口,优先调用 XQT 已有 helper;不要在脚本里复制后端 adapter,逐层分析,报告导出或 benchmark 逻辑.
- 顶层示例脚本涉及可选依赖时使用 lazy import,失败时抛出清楚的 XQT 异常或错误信息.
- 新增 recipe 必须声明 `compression_axes` 和支持的硬件约束
- 所有量化/剪枝/导出后都必须能跑精度 diff 和最小性能基准

## 注意事项

- 量化,部署和扩散少步蒸馏脚本通常对环境版本敏感,涉及外部 API 时先确认最新文档
- TensorRT, OpenVINO, ncnn, MNN 和 ExecuTorch 仍以目标机器官方安装方式为准,XQT 只做 adapter 和 preflight 检查
- 真实 FP8 收益必须依赖支持硬件验证,synthetic smoke 只用于本地闭环验证
