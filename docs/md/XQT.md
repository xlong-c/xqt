# XQT 压缩与部署工具链详细版

先看摘要入口时,优先打开 [XQT_SUMMARY.md](XQT_SUMMARY.md). 本文保留为详细版事实源,负责完整模块状态,能力矩阵,recipe 现状和任务清单.

本文档是 `docs/md/` 中唯一保留的 XQT 长期工作文档. 它负责定义 `xqt/` 的目标边界, 当前模块状态, 数据角色, 量化, 剪枝, 蒸馏, 导出, 算子优化, recipe 和仍有效任务. 已完成或失效的阶段性 `XQT_*.md` 专项文档不再保留; 后续临时调研优先放到 `research/`, 只有长期有效结论回写到本文.

本文不是稳定 API 承诺. 当前 `xqt/` 仍是实验包, 只有 `xqt` 顶层导出的配置, runner, stage workflow, manifest 和 XDL adapter 入口按 Provisional API 管理; 子模块细节仍按 Internal 处理.

## 1. 项目定位

`xqt` 是基于 PyTorch 的模型压缩与部署工具链, 面向从训练产物到推理产物的工程路径:

```text
PyTorch checkpoint
    -> 可选蒸馏
    -> 可选剪枝
    -> 可选量化
    -> 可选算子优化
    -> 可选扩散少步蒸馏
    -> 导出部署格式
    -> 精度验证和性能基准
```

核心目标:

- 支持 PyTorch `nn.Module` 和 `state_dict` 作为主要输入.
- 覆盖量化, 通用蒸馏, 剪枝, 扩散模型少步蒸馏四类优化技术.
- 能导出主流部署格式, 首期优先 `ONNX`, `TensorRT engine`, `OpenVINO IR`, `torch.export` 产物和 TorchScript 兼容路径.
- 记录每次转换的配置, 数据, 版本, 指标和产物校验信息, 保证结果可复现.
- 与 XDL 训练体系保持松耦合, 先作为 `xqt/` 独立实验项目推进, 成熟后再考虑抽公共能力到 `tools/` 或 `xdl/`.

当前包边界决策:

- `xqt` 继续作为仓库顶层实验包维护,不并入 `xdl/` 主框架子模块.
- 只有 `xqt` 顶层导出的配置,runner,stage workflow,manifest 和 XDL adapter 入口进入 Provisional API.
- `xqt.core`, `xqt.pipeline`, `xqt.quant`, `xqt.prune`, `xqt.distill`, `xqt.diffusion_distill`, `xqt.export` 仍按 Internal 处理,先服务 recipe 验证.

安装建议:

- 基础 ONNX/QDQ/torchao 路径: `pip install -e ".[xqt]"`.
- HuggingFace 文本 KD/prune 路径: `pip install -e ".[xqt-hf]"`.
- diffusion/Flux/SD 类路径: `pip install -e ".[xqt-diffusion]"`.
- XQT 全量可选依赖: `pip install -e ".[xqt-all]"`.
- TensorRT, OpenVINO, ncnn, MNN 和 ExecuTorch 仍以目标机器官方安装方式为准, XQT 只做 adapter 和 preflight 检查.

首个支持场景:

- 主场景: image classification + ResNet/CNN, 优先打通 PyTorch baseline -> ONNX export -> ONNX Runtime QDQ INT8 -> TensorRT/OpenVINO 友好产物 -> benchmark/manifest.
- 并行场景: image classification + ViT/Transformer, 优先打通 torchao weight-only/FP8 dynamic 路径和逐层敏感度报告,真实 FP8 收益必须依赖支持硬件验证.
- 当前本地闭环先使用 synthetic image smoke,避免把 CI 绑定到 ImageNet 或 pretrained 权重下载.真实验收再接 ImageNet/timm 或用户提供的校准集.

非目标:

- 不替代 `xdl.trainer.Trainer` 的训练生命周期.
- 不重新实现 TensorRT, OpenVINO, ONNX Runtime, ExecuTorch, ncnn, MNN 等后端.
- 不承诺一次支持所有模型族和硬件. 每个 recipe 必须声明支持的模型结构, 输入形状, 精度策略和硬件约束.
- 不引入命令行参数解析库. 入口脚本读取 YAML 配置或明确的代码内配置.

## 2. 当前仓库现状

`xqt/` 已经从零散实验脚本收敛成包化实验工具链. 当前源码包含:

- `xqt/core`: structured config schema, OmegaConf 加载, artifact manifest, checksum, registry, dotted target import 和错误类型.
- `xqt/data`: synthetic classification/detection samples, calibration dataloader, prompt 数据, torchvision image classification loader, Ultralytics detection loader, HuggingFace 文本数据和 `xdl.dataset` bridge.
- `xqt/model`: 外部模型 adapter, 当前包含 Ultralytics YOLO detection wrapper, 数据集解析和参考导出 helper.
- `xqt/pipeline`: pass manager, built-in passes, preflight 和 YAML runner.
- `xqt/workflows`: stage-based optimization workflow,把 eval/benchmark/prune/quant/finetune/distill/operator/export/deploy/runtime_eval 等阶段按配置编排.
- `xqt/quant`: torchao adapter, ONNX Runtime QDQ static quantization, calibration, policy, sensitivity, component quantization plan 和 backend capability.
- `xqt/prune`: global L1 pruning, structured pruning, importance ranking, pruning schedule, prune + KD helper, N:M 和 block sparse 报告.
- `xqt/distill`: logit KD, feature/relation loss, feature hook, teacher cache, HuggingFace text bundle 和基础训练 helper.
- `xqt/diffusion_distill`: timestep schedule, prompt/latent/trajectory cache, consistency/LCM style loss 和 sampling report metadata.
- `xqt/export`: `torch.export`, TorchScript fallback, ONNX export/checker/runtime diff, TensorRT `trtexec` / Python API adapter, OpenVINO adapter, ExecuTorch/ncnn/MNN mobile adapter 和导出前融合 helper.
- `xqt/operator_opt`: `torch.compile`, Triton, TileLang, CuTile, CUTLASS 和 custom CUDA 的 capability, plan, executor, pattern 和 fallback report.
- `xqt/eval`, `xqt/benchmark`: detection decode/mAP, metric flatten, output diff, JSON/CSV/Markdown 报告, latency 和 memory benchmark.
- `xqt/xdl_adapter.py`: 从 XDL TrainSetup-like 对象或 checkpoint 创建 XQT context.
- `xqt/recipes`: CPU smoke, ONNX QDQ, CIFAR-100 QDQ, YOLO detection practice, 多组件量化, structured prune, KD prune 和 operator optimization recipes.

顶层独立脚本只作为手动实验或示例入口保留,不计入 XQT 的长期模块状态,recipe backlog 或 API 边界. 理解和修改 XQT 时,以包模块,`xqt/recipes/*.yaml`,runner/preflight,测试和本文为事实源;不要从某个示例脚本反推项目架构.

相关资料:

- `research/cnn-deploy/README.md`: CNN 压缩, ONNX 导出, TensorRT, OpenVINO, 服务化和部署检查清单的研究资料.
- `research/rtdetrv4-m-int8-deploy/`: ONNX Q/DQ, NVIDIA ModelOpt, SmoothQuant 和 TensorRT 部署实验资料.
- `learn/quant/`: Stable Diffusion 3.5 量化实验和量化理论笔记.
- `learn/math/Hessian.md`: GPTQ/OBS 类二阶量化和剪枝的理论背景.
- `learn/tilelang/`: TileLang 学习实验,当前有 FlashAttention forward 示例.
- `learn/rwkv/rwkv8/rwkv8_tilelang.py`: TileLang ROSA suffix-match 教学 kernel.
- `research/diffusion-models-survey-2025/`: 扩散模型少步推理, RL 后训练和蒸馏相关研究资料.
- `pyproject.toml`: XQT extras 当前包含 `xqt`, `xqt-hf`, `xqt-yolo`, `xqt-diffusion`, `xqt-all`; `optimization` 额外保留 `torchao`, `triton`, `tilelang`.

## 3. 设计原则

- PyTorch 优先. 训练侧和变换侧以 `nn.Module`, `state_dict`, `torch.export.ExportedProgram` 和标准 Tensor 输入输出为第一等对象.
- 配置集中. 复杂 pipeline 使用 OmegaConf 加载 YAML, 配置只描述目标, 参数, artifact 位置和验证阈值.
- 后端隔离. TensorRT, OpenVINO, ONNX Runtime, torchao 等外部能力通过 adapter 封装, 不把外部 API 散落在业务流程里.
- 每步可验证. 量化, 蒸馏, 剪枝, 扩散少步蒸馏, 导出后都必须能跑精度 diff, 任务指标和最小性能基准.
- 结构化产物. 每个产物目录包含模型文件, 配置快照, 版本信息, 指标报告和 manifest.
- 渐进稳定. `xqt` 初期全部视为实验 API. 只有经过测试, 文档和真实 recipe 验证后, 才考虑进入稳定边界.
- 实用优先. 优先实现对真实推理延迟, 显存, 吞吐或采样步数有确定收益的方案; 论文指标好但缺少后端支持的方案先放 research.

给 agents 和开发者的阅读顺序:

1. `xqt/core/schema.py` 和 `xqt/core/config.py`: 看 recipe schema,默认值,OmegaConf 加载和校验边界.
2. `xqt/pipeline/runner.py` 和 `xqt/pipeline/passes.py`: 看 `config -> context -> pass pipeline -> artifacts/metrics/manifest` 主链路和真实执行顺序.
3. 按任务进入 `xqt/quant`,`xqt/prune`,`xqt/distill`,`xqt/diffusion_distill`,`xqt/export`,`xqt/operator_opt`,`xqt/eval`,`xqt/benchmark`: 看可复用 helper 和后端 adapter.
4. `xqt/recipes/*.yaml` 和 `tests/xqt/`: 看当前哪些路径已经有可运行闭环,哪些依赖本地数据,CUDA 或外部后端命令.

新增能力时优先落在包模块,再通过 recipe,preflight 和测试暴露. 顶层示例脚本只能是薄入口,不能成为长期事实源.

## 4. 当前模块划分

`xqt/` 当前已经是包化实验工具链,核心目录如下:

```text
xqt/
├── __init__.py
├── core/
│   ├── artifact.py        # artifact manifest, checksum, metadata
│   ├── config.py          # OmegaConf 加载, structured config
│   ├── registry.py        # xqt 内部 recipe/pass/exporter 注册
│   ├── errors.py          # XQT 异常类型
│   ├── imports.py         # dotted target 构建
│   ├── schema.py          # XQTConfig structured schema
│   └── types.py           # XQTContext 等运行态类型
├── data/
│   ├── builders.py        # split target 构建入口
│   ├── calibration.py     # calibration dataloader 和样本抽取
│   ├── hf_text.py         # HuggingFace 文本数据支架
│   ├── prompts.py         # prompt 数据
│   ├── samples.py         # example input, input signature, synthetic data
│   └── torchvision.py     # torchvision image classification loader
├── distill/
│   ├── losses.py          # logit KD, feature KD, relation KD
│   ├── hooks.py           # teacher/student 中间层对齐
│   ├── cache.py           # teacher logits/features 缓存
│   ├── hf_text.py         # HuggingFace 文本分类 KD/prune recipe 支架
│   └── training.py        # 小型 teacher -> student 训练 helper
├── diffusion_distill/
│   ├── trajectory.py      # teacher 轨迹, 噪声, timestep 和 scheduler 采样
│   ├── losses.py          # consistency/LCM 类少步蒸馏损失
│   ├── cache.py           # prompt, latent 和 trajectory cache
│   ├── report.py          # 固定 seed 图片网格和采样报告
│   └── spec.py            # diffusion few-step spec
├── prune/
│   ├── capability.py      # runtime/export/稀疏能力描述
│   ├── importance.py      # L1/L2, BN gamma, Taylor, gradient * weight
│   ├── masks.py           # mask 管理和持久化
│   ├── rewrite.py         # 结构化剪枝后的模块重写
│   ├── schedule.py        # pruning schedule 和 KD helper
│   └── structured.py      # channel/filter/head/block 等结构化剪枝
├── quant/
│   ├── capability.py      # quant backend 能力矩阵
│   ├── torchao_backend.py # torchao PTQ/QAT/weight-only/FP8 adapter
│   ├── onnx_qdq.py        # ONNX Runtime static QDQ quantization adapter
│   ├── sensitivity.py     # 逐层误差和混合精度建议
│   ├── calibration.py     # observer, 校准循环, 校准报告
│   ├── executor.py        # component quantization plan 执行
│   ├── plan.py            # 量化执行计划
│   ├── policy.py          # allowlist, denylist, dtype, granularity
│   └── types.py           # quant report 数据类型
├── operator_opt/
│   ├── capability.py      # torch.compile,Triton,TileLang,CuTile,CUTLASS,custom CUDA 能力矩阵
│   ├── compile_backend.py # torch.compile/Inductor/CUDA Graphs adapter
│   ├── patterns.py        # FX/torch.export graph pattern 发现
│   ├── backends/          # Triton,TileLang,CuTile,CUTLASS 等后端 adapter
│   ├── kernels/           # 算子 reference,guarded entry 和 backend-specific kernel
│   ├── executor.py        # 组件级 operator optimization 执行器
│   └── types.py           # graph break,kernel count,latency 和 manifest 报告
├── export/
│   ├── capability.py      # 部署格式能力矩阵
│   ├── fusion.py          # 导出前前置融合
│   ├── onnx_exporter.py   # torch.onnx.export, dynamo=True 优先
│   ├── torch_exporter.py  # torch.export.ExportedProgram 和 TorchScript 兼容
│   ├── tensorrt.py        # ONNX -> TensorRT engine
│   ├── openvino.py        # PyTorch/ONNX -> OpenVINO IR
│   └── mobile.py          # ExecuTorch/ncnn/MNN 等可选路径
├── eval/
│   ├── accuracy.py        # task metric, topk, mAP, loss, custom metric
│   ├── compare.py         # PyTorch vs exported backend 输出差异
│   └── report.py          # JSON/CSV/Markdown 报告
├── benchmark/
│   ├── latency.py         # warmup, p50/p90/p99, batch sweep
│   ├── memory.py          # 显存/内存统计
│   └── profiler.py        # torch profiler 和后端 profiler 接入
├── pipeline/
│   ├── pass_manager.py    # 串联 quant/distill/prune/diffusion_distill/export/eval
│   ├── passes.py          # 内置 pass 实现
│   ├── preflight.py       # recipe target,可选依赖和后端命令检查
│   └── runner.py          # 从 YAML 执行 recipe
├── xdl_adapter.py         # XDL TrainSetup/checkpoint 到 XQT context 的桥接
├── recipes/               # smoke, quant, prune, distill, operator optimization recipes
└── entrypoints/
    ├── preflight.py       # xqt-preflight
    └── run_recipe.py      # xqt-run-recipe,读取 XQT_CONFIG 或默认 YAML
```

测试放在 `tests/xqt/`, 长期工作文档放在 `docs/md/`, 阶段性研究继续放在 `research/`.

## 5. 压缩维度

XQT 的 recipe 不只按技术名分类, 还要声明压缩目标. 同一个模型可以同时做宽度压缩, 深度压缩和数值精度压缩, 但每一类的验证指标不同.

| 维度 | 压缩对象 | 典型技术 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| 宽度压缩 | channel, hidden size, MLP width, attention heads, filter 数 | 结构化剪枝, 窄 student, head pruning, width multiplier | 参数量, FLOPs, 显存和部分延迟下降 | 结构改写复杂, 残差和归一化层容易不匹配 |
| 深度压缩 | layer/block 数, encoder/decoder 层数, diffusion sampling steps | layer pruning, shallow student, early-exit, 扩散少步蒸馏 | 推理链路变短, 延迟下降更直接 | 表达能力下降, 需要蒸馏或微调恢复 |
| 精度压缩 | weight/activation dtype 和量化粒度 | BF16/FP16, INT8, INT4, FP8, QDQ, GPTQ/AWQ | 显存, 带宽和吞吐改善 | 数值误差, 后端兼容和混合精度策略 |
| 稀疏压缩 | weight mask, N:M pattern, block sparse pattern | 非结构化剪枝, N:M 稀疏, SparseGPT/Wanda | 参数存储下降, 特定硬件可加速 | 普通 GPU/CPU 不一定变快 |
| 步数压缩 | diffusion denoising steps, solver steps | LCM, consistency distillation, progressive distillation, flow distillation | 生成模型延迟大幅下降 | 图像质量, prompt adherence 和多样性下降 |

模块映射:

- `quant` 主要负责精度压缩.
- `prune` 主要负责宽度压缩和稀疏压缩, 也可以负责 layer/block 级深度压缩.
- `distill` 主要负责训练更窄或更浅的 student, 也负责剪枝后的精度恢复.
- `diffusion_distill` 主要负责扩散模型的步数压缩, 必要时叠加 LoRA student 或少量宽度压缩.

Recipe 必须声明 `compression_axes`, 例如 `["precision"]`, `["width", "precision"]`, `["steps"]`. 报告也要按维度输出收益: 参数量, FLOPs, 稀疏率, dtype, 采样步数, latency, memory 和 task metric.

## 6. 核心数据结构

首期需要先定义轻量 dataclass, 让各模块共享同一套对象:

- `ModelSpec`: 模型构建方式, checkpoint 路径, dtype, device, unwrap 规则, eval mode 规则.
- `DataSpec`: 训练数据, 校准数据, 验证数据, batch size, transform, collate, sample limit.
- `ShapeSpec`: input names, output names, static shape, dynamic shape, batch 维, sequence 维.
- `CompressionPlan`: quant, distill, prune, diffusion_distill 的顺序, 启停开关和 `compression_axes`.
- `ExportTarget`: format, opset, backend, precision, dynamic shapes, output path.
- `ValidationSpec`: 精度阈值, 输出 diff 阈值, 任务 metric 阈值, 性能阈值.
- `ArtifactManifest`: 源 checkpoint checksum, 配置快照, 依赖版本, 产物列表, 指标, 生成时间.
- `DiffusionSpec`: prompt/condition 数据, scheduler, timestep 分布, latent shape, teacher steps, student steps, guidance 策略.

## 7. 四类优化能力

### 7.1 量化

量化是 XQT 的第一优先级, 因为它通常不需要完整重训, 对显存, 吞吐和部署格式的收益最直接.

实用技术分层:

| 技术 | 适用对象 | 工程价值 | XQT 优先级 |
| --- | --- | --- | --- |
| BF16/FP16 baseline | CNN, Transformer, Diffusion | 最稳的推理基线, 也是量化对照组 | P0 |
| Weight-only INT8/INT4 | LLM, DiT, Linear-heavy 模型 | 显存下降明显, 接入成本低 | P0 |
| FP8 dynamic activation + weight | Ada/Hopper GPU 上的 Transformer/ViT/DiT | 有机会同时降显存和提吞吐, 依赖硬件和 torchao 支持 | P0 |
| ONNX Q/DQ static INT8 | CNN, 检测模型, 传统视觉模型 | TensorRT/OpenVINO/ONNX Runtime 友好 | P0 |
| SmoothQuant | 激活 outlier 明显的 Transformer | 把激活难量化问题迁移到权重, 对 INT8 部署实用 | P1 |
| AWQ/GPTQ | LLM 和部分大 Transformer | 4-bit weight-only 精度更稳, 但依赖实现和模型结构 | P1 |
| Rotation/equalization quantization | LLM/Transformer | 通过旋转或等价缩放降低 outlier, 代表路线包括 QuaRot/SpinQuant | P1 |
| KV cache quantization | LLM 推理 | 长上下文显存收益明显, 对服务场景实用 | P1 |
| HQQ/SpQR/AQLM 类 weight quantization | LLM | 更激进的低 bit 压缩路线, 需要独立后端评估 | P2 |
| QAT | 低 bit, 精度敏感模型 | 精度最好, 成本高, 应作为后续增强 | P2 |
| FP4/NF4/W4A4 | LLM/DiT 研究和定制 kernel | 潜力大, 但后端强绑定, 先放实验区 | P2 |

首期任务:

- `torchao` adapter: weight-only INT4/INT8, FP8 dynamic activation + weight, allowlist/denylist.
- ONNX Q/DQ adapter: 复用 ONNX Runtime quantization 的 static calibration, 以 TensorRT/OpenVINO 友好图为目标. 当前已有 `quant/onnx_qdq.py` 和 `backend: onnxruntime_qdq` 内置 pass;执行后会读取生成的 ONNX 图,在 quant metadata 中记录 `qdq_graph`,`qdq_node_count` 和实际命中的 `quantized_op_types`.
- FP16 ONNX baseline: `xqt.export.convert_onnx_to_fp16()` 使用 `onnxconverter-common` 把 FP32 ONNX 转为 FP16 ONNX,用于检测和分类部署对照. 该依赖在 `xqt` extra 中提供;检测评估会按 ONNX Runtime 输入 dtype 自动 cast feed.
- NVIDIA ModelOpt adapter: 从 `research/rtdetrv4-m-int8-deploy/` 现有实验收敛 SmoothQuant/INT8 Q/DQ 路径.
- PTQ 校准: 校准 dataloader, calibration sample limit, observer 统计, 校准报告.
- KV cache quantization 预研: 先定义接口和报告字段, 不作为首期强依赖.
- 敏感层分析: 以 `xqt/quant/sensitivity.py` 为通用入口,输出逐层激活 diff,weight diff,summary 和混合精度建议.
- 混合精度策略: 首层, 末层, normalization, embedding, small matmul, attention projection 等可配置保留高精度.
- QAT 预留: fake quant 插入和微调流程先设计接口, 不作为首期阻塞项.

实用判断:

- CNN/检测部署优先 ONNX Q/DQ INT8 + TensorRT/OpenVINO.
- Transformer/ViT/DiT 优先 torchao weight-only 或 FP8, 再评估 SmoothQuant/AWQ/GPTQ.
- 扩散模型先量化 text encoder 和 transformer/UNet 的 Linear, VAE 和首尾层默认保守.
- 量化产物必须标记 runtime: `pytorch`, `onnxruntime`, `tensorrt`, `openvino` 或 `custom_kernel`.

验收标准:

- 同一模型至少跑通 FP32/BF16 baseline 和量化版本的输出 diff.
- 报告包含 layer name, layer type, dtype policy, act diff, weight diff, latency change.
- 量化产物能继续进入至少一个 export target 或明确标记为 PyTorch runtime only.

### 7.2 通用蒸馏

通用蒸馏负责分类, 检测, NLP 等非扩散采样任务. 扩散模型少步蒸馏单独放到 7.4, 因为它的目标不是压小模型, 而是减少采样步数和重构推理轨迹.

实用技术分层:

| 技术 | 适用对象 | 工程价值 | XQT 优先级 |
| --- | --- | --- | --- |
| Logit KD | 分类, 文本分类 | 最简单稳定, 适合作为第一个 recipe | P0 |
| Feature KD | CNN/ViT/检测 | 帮小模型学中间表示, 需要层对齐 | P0 |
| Teacher logits/features cache | 大 teacher | 降训练成本, 工程收益明确 | P0 |
| Distill + prune finetune | 剪枝恢复精度 | 实用组合, 对剪枝后恢复有价值 | P0 |
| Shallow/narrow student | 宽度/深度压缩 | 最容易获得真实延迟收益, 但需要重新训练 | P0 |
| Relation KD/RKD | 表征学习, 检索 | 有用但实现和评估更复杂 | P1 |
| Sequence-level KD | 生成式 NLP, 翻译, 摘要 | 直接蒸馏 teacher 生成序列, 适合生成任务 | P1 |
| Rationale/step-by-step distillation | 推理型任务 | 用 teacher 推理过程减少 student 数据需求 | P1 |
| On-policy/GKD 类蒸馏 | LLM student | student 自己采样, teacher 提供分布或偏好信号, 减少训练推理分布偏移 | P1 |
| Online/mutual distillation | 多模型训练 | 训练成本高, 不做首期 | P2 |
| Task-specific distillation | 检测/分割/生成 | 需要按任务设计 loss, 分阶段加入 | P2 |

首期任务:

- 离线蒸馏: 固定 teacher, 训练 student.
- Logit KD: KL divergence + temperature + CE 组合.
- Feature KD: 中间层 hook, adapter 投影, MSE/Cosine loss.
- 宽度/深度 student recipe: teacher 保持原始结构, student 显式减少 hidden size, heads, MLP width 或 layer 数.
- Teacher 输出缓存: 对大 teacher 减少重复前向成本.
- HuggingFace 文本分类参考 recipe: teacher/student 使用 `AutoModelForSequenceClassification`, 数据使用 `datasets` + `DataCollatorWithPadding`, 损失为 `alpha * KL(student / T, teacher / T) * T^2 + (1 - alpha) * CE`.
- XDL 接入: 提供可被 `CoreModel.training_step()` 调用的 distillation loss helper, 不直接改 `Trainer`.

验收标准:

- 一个分类模型 teacher -> student recipe.
- teacher 不反传梯度, student 可正常保存 checkpoint.
- 报告包含 teacher metric, student baseline metric, distilled student metric.

### 7.3 剪枝

剪枝要分清楚 "稀疏率好看" 和 "真实加速" 两件事. XQT 允许非结构化剪枝作为 baseline, 但部署收益优先看结构化剪枝, N:M 稀疏或后端明确支持的稀疏格式.

实用技术分层:

| 技术 | 适用对象 | 工程价值 | XQT 优先级 |
| --- | --- | --- | --- |
| 全局 L1 非结构化剪枝 | Linear/Conv baseline | 简单, 适合验证 mask 和恢复训练 | P0 |
| 结构化 channel/filter 剪枝 | CNN/Conv-heavy 模型 | 真实减少 FLOPs 和导出图尺寸 | P0 |
| Attention head/MLP neuron pruning | Transformer | 对实际延迟有潜力, 需要结构改写 | P1 |
| Layer/block pruning | Transformer/CNN | 深度压缩, 延迟收益直接 | P1 |
| N:M 稀疏 | NVIDIA Ampere+ | 有硬件路径, 但格式和后端约束强 | P1 |
| SparseGPT/Wanda 类一次性 LLM 剪枝 | LLM | 校准成本低, 适合研究和显存优化 | P1 |
| LLM-Pruner/Sheared/SliceGPT 类结构化 LLM 压缩 | LLM | 同时裁剪宽度和深度, 更可能带来真实延迟收益 | P1 |
| Taylor/OBS/二阶剪枝 | 精度敏感模型 | 效果可能更稳, 实现成本较高 | P2 |
| Dynamic sparsity | 训练期稀疏 | 复杂, 不做首期 | P2 |

首期任务:

- 非结构化剪枝 baseline: 使用 PyTorch pruning 工具验证 mask, sparsity 和恢复训练流程.
- 结构化剪枝 MVP: 已支持 CNN channel/filter, ViT/Transformer 的 MLP neuron, attention head, block pruning, 以及 N:M structured sparsity 报告与 recipe 验证.
- YOLO/Ultralytics 检测模型的结构化剪枝默认在 preflight 阶段被 guard: residual,CSP/C2f,concat,SPPF 和 detect head 拓扑需要 dependency graph + rewrite 支持后才能开放. 当前 YOLO practice 只使用 `global_l1_unstructured` 作为 sparsity/report baseline,不把非结构化剪枝描述成真实部署加速.
- 深度剪枝预研: layer/block dropping, 保留残差和 normalization 结构一致性.
- 重要性评估: L1/L2, BN gamma, gradient * weight. Taylor 和 OBS 放到第二阶段.
- 剪枝计划: 一次性剪枝, 迭代剪枝和线性稀疏率 schedule 都通过 YAML 描述.
- 剪枝后验证: 参数量, FLOPs 估算, 输出差异, 任务 metric, 导出可行性.
- PyTorch pruning 参考 recipe: `torch.nn.utils.prune.global_unstructured` + `prune.L1Unstructured`, 每个 epoch 按线性 schedule 提升累计目标稀疏率, 训练后用 `prune.remove()` 固化 `weight_orig * weight_mask`.

验收标准:

- 结构化剪枝后的模型不依赖 mask 才能运行.
- ONNX 导出后的图真实减少 channel/filter, 而不是只带零权重.
- 剪枝报告列出每层剪枝率和全局压缩率.

### 7.4 扩散模型少步蒸馏

扩散少步蒸馏是第四个独立模块. 它的目标是把 20/30/50 step teacher 的采样能力压到 1/2/4/8 step student, 关注的是采样轨迹, scheduler, noise prediction/velocity prediction 和条件一致性.

实用技术分层:

| 技术 | 适用对象 | 工程价值 | XQT 优先级 |
| --- | --- | --- | --- |
| Progressive distillation | DDPM/latent diffusion | 经典可解释, 逐步把步数减半 | P1 |
| Consistency distillation/LCM | Stable Diffusion/latent diffusion | 2-4 step 实用, LoRA 方式成本较低 | P0 |
| Rectified Flow/Flow Matching distillation | Flow/DiT/TwinFlow 类模型 | 与仓库生成模型方向契合, 可做中期重点 | P1 |
| RCGM | diffusion transformer/多模态生成模型 | N 阶递归一致速度场估计, 面向 any-step generation | P1 |
| Distribution Matching Distillation/DMD/DMD2 | 文生图 diffusion/DiT | 直接匹配生成分布, 常用于 1-4 step generator | P1 |
| Score identity / SiD 类 distillation | diffusion/score model | 使用 score identity 约束少步 generator, 可归入分布匹配路线 | P1 |
| Shortcut/MeanFlow 类 any-step generator | diffusion/flow model | 训练一次支持多步/少步推理, 与 RCGM 同属 any-step 候选 | P1 |
| TCD/Trajectory consistency 类 distillation | SD/latent diffusion | 轨迹一致性约束, 适合和 LCM 对比 | P1 |
| Flash Diffusion/Hyper-SD 类工程 recipe | SDXL/SD3/Flux 类模型 | 多方法组合的实用少步方案, 适合作为对标 recipe | P1 |
| Reward-guided distillation | 对齐审美/偏好/人类反馈的文生图 | 可把 reward model 或偏好信号并入少步蒸馏 | P1 |
| Adversarial diffusion distillation | SDXL/文生图 | 1-4 step 质量好, 训练复杂 | P2 |
| Trajectory matching | DiT/视频/多条件生成 | 泛化好, 但实现和评估重 | P2 |

首期任务:

- 定义 `DiffusionSpec`: teacher pipeline, student pipeline/LoRA, scheduler, prediction type, guidance, teacher steps, student steps.
- 实现 prompt/condition 数据读取, latent 缓存和 teacher trajectory 缓存.
- 实现 4-step LCM/consistency 类 recipe, 优先 LoRA student, 降低训练成本.
- 实现 `epsilon`, `v_prediction`, `x0` 三类 prediction target 的统一 loss 包装.
- 预留 DMD/DMD2 类 distribution matching loss 接口, 支持 fake score/teacher score, generator update 和可选 discriminator/reward signal.
- 预留 RCGM adapter. RCGM 当前应按 any-step / N-th order recursive consistent velocity field estimation 路线归入扩散少步蒸馏, 等官方训练和推理代码发布后再决定是否进入 MVP.
- 实现 1/2/4/8 step sampler 对比和固定 seed 图片网格报告.
- 评估指标首期用 CLIP score, LPIPS/SSIM 可选, 人工网格检查必须保留.
- 对接 `xdl.trainer.CoreModel` 手动优化路径, 避免改 Trainer 生命周期.

验收标准:

- 同一 prompt 集合下, teacher 多步结果和 student 少步结果可复现实验.
- 报告包含 teacher steps, student steps, scheduler, guidance scale, latency, VRAM 和图片网格.
- 少步 student 可单独保存 LoRA 或完整权重, 并能通过推理脚本加载.

### 7.5 分析与误差诊断

`analyze` pass 是 XQT 内用于压缩前后诊断的长期入口. 早期单独维护的通用误差分析文档已经收拢到本文; 后续若要沉淀框架级通用函数,先在源码中形成稳定 API,再同步到对应长期文档.

XQT 中误差分析优先覆盖这些场景:

- 量化,剪枝,蒸馏和少步蒸馏前后的模型对比.
- PyTorch eager,`torch.compile`,`torch.export`,ONNX Runtime,TensorRT 和 OpenVINO 等执行路径对齐.
- 导出前融合,算子替换或 custom kernel 替换后的数值回归.
- 校准数据,验证数据,prompt 数据或预处理链路变化后的输入和中间表示漂移.
- teacher / student feature 对齐,敏感层排序,混合精度豁免层和剪枝候选层筛选.

优先记录的误差类型:

- 有效性: shape,dtype,device,NaN,Inf,empty tensor 和 `requires_grad` 状态.
- 数值误差: max abs,mean abs,median abs,p95/p99 abs,relative error,MSE,RMSE 和 allclose.
- 相似性: cosine similarity,Pearson correlation,Spearman rank correlation.
- 分布漂移: mean,std,min,max,quantile,histogram,zero ratio,saturation ratio,clipping ratio,outlier ratio.
- 结构化误差: per-channel,per-head,per-token,per-time-step,spatial heatmap diff 和 attention map diff.
- 离散决策: argmax mismatch,top-k overlap,sign mismatch,threshold flip 和 token mismatch.
- 任务级指标: accuracy,F1,perplexity,mAP,PSNR,SSIM,BLEU,ROUGE 等 delta. 任务指标只能说明后果,不能替代中间误差定位.

当前源码映射:

- `xqt/eval/compare.py`: 基础 tensor 输出对比.
- `xqt/quant/sensitivity.py`: 逐层输出误差,weight diff 和混合精度建议.
- `xqt/quant/calibration.py`: activation summary 和 activation drift.
- `xqt/distill/hooks.py`: teacher / student feature alignment.
- `xqt/prune/importance.py`: importance + sensitivity 组合排序.
- `xqt/eval/report.py`: analysis records 到 JSON/CSV/Markdown/DataFrame 友好行的转换.
- `xdl/callbacks/layer_monitor.py`: 训练期权重和梯度统计的参考实现,不是 XQT API.

`analyze` pass 的报告应尽量保持这些字段稳定:

- `records`: 逐层 `LayerAnalysisRecord`,包含 module name,type,reference/candidate summary,diff,weight diff 和 recommendation.
- `activation_drift`: reference/candidate activation distribution summary 和 drift delta.
- `importance`: pruning importance 统计.
- `prune_candidates`: 结合 importance 和 sensitivity 的候选层排序.
- `recommended_high_precision_modules`: 建议保留高精度的模块名.
- `teacher_student_alignment`: teacher / student 中间特征对齐报告.
- `pareto_points`: benchmark 后用于精度/延迟/内存折中展示的数据点.

推荐分析顺序:

1. 先做有效性检查,排除 shape,dtype,NaN,Inf 等硬错误.
2. 再看整体输出 diff 和任务指标 delta,判断是否存在真实回归.
3. 再做逐层排序,定位最敏感层,漂移层或候选豁免层.
4. 最后只对需要解释的层做结构化可视化,例如 heatmap,per-token 图,直方图或时序曲线.

## 8. 导出和部署格式

优先级建议:

| 优先级 | 格式 | 用途 | 首期策略 |
| --- | --- | --- | --- |
| P0 | `torch.export.ExportedProgram` | PyTorch 2 AOT 图和后续导出中间层 | 已实现 `torch.export.export` + `torch.export.save/load` adapter |
| P0 | ONNX | 跨框架交换, TensorRT/OpenVINO/ONNX Runtime 入口 | `torch.onnx.export(..., dynamo=True)` 优先 |
| P0 | TensorRT engine | NVIDIA GPU 高性能推理 | ONNX -> TensorRT, 支持 FP16/INT8 profile, `trtexec` 与 Python API build,性能摘要解析和阈值报告 |
| P1 | OpenVINO IR | Intel CPU/GPU/NPU 推理 | `openvino.convert_model` 或 ONNX -> IR,支持 dry-run 命令构造 |
| P1 | TorchScript | 旧 PyTorch 部署兼容 | 已实现 trace/script fallback,主要服务旧部署和 pnnx 输入 |
| P2 | ExecuTorch | PyTorch 移动端/边缘端 | 等核心 pipeline 稳定后接入 |
| P2 | ncnn/MNN | 手机和边缘部署 | 通过 ONNX 转换, 作为可选 exporter |
| P2 | Triton model repo | 服务化目录结构 | 生成目录和 `config.pbtxt`, 不内置服务运行 |

导出前检查:

- `model.eval()` 和 `torch.no_grad()`/`torch.inference_mode()`.
- 移除 DataParallel/DDP wrapper.
- 输入输出名字, shape 和 dtype 明确.
- 动态 shape 使用 `dynamic_shapes`, 旧 `dynamic_axes` 只用于非 dynamo fallback.
- 自定义算子, Python 控制流和 inplace 操作要在导出前发现并报告.

导出后检查:

- ONNX checker 或后端 parser 通过.
- PyTorch 原模型和导出模型输出 diff 达标.
- 目标硬件基准测试达标.
- manifest 写入导出参数, 依赖版本, opset, backend, profile 和校准信息.

当前导出边界:

- TensorRT adapter 当前支持 `trtexec` 和 `python_api` 两条路径. `dry_run: true` 时会只构造命令或 Python API build 计划,记录 precision,profile shape,source ONNX 和性能阈值配置;真实 engine 生成仍依赖目标机器安装 TensorRT 和可用 NVIDIA GPU.
- 对 `python_api` 真实 build, XQT 现在会额外把 TensorRT engine inspector 摘要写入 artifact metadata,包括 layer count,Q/DQ layer 痕迹,`quantized + Conv` 这类融合签名以及 I/O tensor 摘要. 这用于区分"只是生成了 engine"和"engine 内部确实出现了量化/融合迹象".
- 对部分真实 QDQ detection ONNX,TensorRT 11.1 还会卡在两类导入限制: `INT32 bias -> DequantizeLinear` 和非对称 activation zero point. XQT 现在会在 TensorRT build 前自动把这类 `INT32` bias QDQ 折叠成 float bias initializer;对 Hugging Face 社区 RT-DETR R18vd,还需要在 quant recipe 侧显式使用 `QInt8 + ActivationSymmetric + WeightSymmetric`.
- detection eval 当前已经支持 Hugging Face 社区 RT-DETR 风格的 `logits + pred_boxes` 输出,并会把 `pred_boxes` 的归一化 `cxcywh` 按当前模型输入尺寸还原到像素框. 结合本地 `coco8_detection` data split,现在已经能对 `onnx-community-rtdetr_r18vd-direct.onnx` 在真实 detection 样本上产出 task metric.
- detection runtime_eval 当前已经同时支持 `runtime: onnxruntime` 和 `runtime: tensorrt`. 对 TensorRT 路径,运行时会复用同一个 engine/runtime/context session,避免把反复反序列化 engine 的初始化成本混进 latency. 这样 `runtime_eval_trt` 的 latency 可以和 deploy stage 的 TensorRT engine benchmark 放在同一个部署语义下比较.
- OpenVINO adapter 支持从 PyTorch module 或 ONNX path 转 IR;`dry_run: true` 时不要求安装 `openvino`,只记录 `openvino.convert_model` 命令,输入 shape,source path 和预期 `.xml/.bin` 路径.
- `ExportPass` 会优先使用 `params.onnx_path` 或上游 `last_onnx` 作为 TensorRT/OpenVINO 输入. 在 QDQ 场景中,TensorRT/OpenVINO dry-run 会指向量化后的 QDQ ONNX,不是重新导出 FP32 图.
- `xqt.workflows.optimize_model` 现在也会把 stage workflow 结果落盘到 artifact dir,包括 `workflow_manifest.json` 和 `workflow_result.json`. 对真实 detection deploy 路线,这两个文件会保留 stage metrics,导出 artifact,TensorRT runtime benchmark 和 engine inspector 摘要,方便把一次成功 build 收敛成可复现事实源.
- `xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval.yaml` 是当前最接近"有部署意义的真实 detection 模型"闭环的 recipe: 不引入 `ultralytics`,直接消费 Hugging Face 社区 RT-DETR R18vd ONNX,先做 QDQ INT8,再 build 真 TensorRT engine,最后在 `coco8_detection` 上跑 ONNX Runtime 和 TensorRT 两条 task metric. 当前事实源位于 `artifacts/xqt/detection/hf_rtdetr_r18vd_qdq_trt_tensorrt_friendly_eval/runtime_eval_trt/`,其中:
  - `runtime_eval_onnx` 得到 `map50_95 = 0.0717`,`map50 = 0.1328`,`map75 = 0.0824`
  - `build_trt_int8_engine.runtime_benchmark.latency.mean_ms = 3.67`
  - `runtime_eval_trt` 得到 `map50_95 = 0.1018`,`map50 = 0.1524`,`map75 = 0.0941`,`latency.mean_ms = 4.48`
  - 同一 engine inspector 记录到 `quantized_conv_count = 66`,`fused_layer_count = 93`,`quantize_layer_count = 48`,`dequantize_layer_count = 68`

## 9. 统一 pipeline

建议 pipeline 使用 pass manager 串联, 每个 pass 都有输入 artifact, 输出 artifact 和验证报告:

```text
load_model
    -> load_data
    -> baseline_eval
    -> optional_distill
    -> optional_prune
    -> optional_quant
    -> optional_diffusion_distill
    -> export_targets
    -> validate_exports
    -> benchmark_exports
    -> write_manifest
```

Pass 约束:

- 每个 pass 必须声明输入类型和输出类型.
- 每个 pass 必须能被单独关闭.
- 每个 pass 的输出不能覆盖上一步产物.
- 任一验证失败时默认停止, 除非 YAML 明确允许 `continue_on_failure`.

## 10. 模块接口草案

首期不追求复杂抽象, 但每个模块要保持相同的调用形态:

```python
class XQTPass:
    name: str

    def run(self, context: XQTContext) -> XQTContext:
        ...
```

`XQTContext` 至少包含:

- `config`: OmegaConf 解析后的普通容器或 structured config.
- `model`: 当前 PyTorch 模型或 wrapper.
- `teacher`: 可选 teacher 模型.
- `data`: 校准,训练,验证和 example input.
- `artifacts`: 上游产物路径和 manifest 片段.
- `metrics`: baseline, diff, task metric, latency, memory.
- `device`: 当前执行设备.

Pass 输入输出约束:

| Pass | 输入 | 输出 | 最小验证 |
| --- | --- | --- | --- |
| `load_model` | `ModelSpec` | `model`, `example_input` | 模型 `eval()` 前向成功 |
| `load_data` | `DataSpec` | dataloader/sample batch | batch shape 和 dtype 符合 `ShapeSpec` |
| `baseline_eval` | model + validation data | baseline metric/report | 指标可复现 |
| `analyze` | baseline/candidate model + sample batch | sensitivity/drift/recommendation report | 报告字段完整,可选阈值检查通过 |
| `quant` | model + calibration data | quantized model/artifact | output diff + dtype policy report |
| `prune` | model + pruning plan | pruned model/artifact | sparsity/shape report + output diff |
| `distill` | teacher + student + train data | distilled checkpoint | student metric 不低于阈值 |
| `diffusion_distill` | teacher pipeline + prompts + scheduler | LoRA 或 student weights | 固定 seed 图片网格 + latency |
| `export` | model/artifact + `ExportTarget` | ONNX/TRT/IR/exported program | parser/checker 通过 |
| `benchmark` | runtime artifact + input spec | latency/memory report | warmup 后稳定采样 |
| `write_manifest` | all reports | `manifest.json` | checksum 和版本字段完整 |

Manifest 必填字段:

- `source_checkpoint`, `source_checksum`, `created_at`, `xqt_version`.
- `compression_axes`, `passes`, `config_snapshot`.
- `dependencies`: Python, torch, CUDA, torchao, onnx, TensorRT/OpenVINO 等版本.
- `artifacts`: 文件路径, 格式, checksum, runtime.
- `metrics`: baseline, optimized, delta, thresholds, pass/fail.

## 11. 首批 recipe backlog

首批 recipe 用来打通接口, 不追求覆盖所有模型:

| ID | 目标 | 模块 | 价值 | 退出条件 |
| --- | --- | --- | --- | --- |
| `image_resnet_onnx_qdq_int8` | ResNet/CNN 分类 | quant + export | 打通 ONNX Q/DQ 和 ONNX Runtime diff | 已有 YAML smoke recipe, synthetic image runner 测试和 TensorRT 阈值解析测试,真实 TensorRT engine 待目标机器验证 |
| `image_resnet_cifar100_qdq_cpu` | ResNet/CIFAR-100 | quant + eval + benchmark | 在本地真实数据上验证 ONNX Runtime QDQ CPU 路径 | 已用本地 CIFAR-100 生成真实 QDQ ONNX, manifest 和 benchmark |
| `yolo_detection_smoke` | synthetic detection + toy model | detection eval + prune + export | 用最小依赖验证 detection schema,baseline eval,prune,ONNX export 和 manifest | 已有 recipe 和 runner 测试覆盖 |
| `yolo_detection_practice` | synthetic detection + toy model | detection eval + quant + export + prune | 验证 detection task schema,mAP,ONNX QDQ 和 stage workflow 主链 | 当前 recipe 已改为 `data_splits + stages` schema,由 `xqt.workflows.optimize_model` 编排 baseline eval/latency,FP32 ONNX runtime eval,QDQ artifact/runtime eval 和 global L1 unstructured prune. operator/deploy 能力由独立 recipe 和测试覆盖,不塞进默认 practice 主链. 示例入口只打印 stage result,不再维护 YOLO 专用场景矩阵或报告生成器 |
| `yolo_detection_trt_qdq_practice` | synthetic detection + toy model | detection quant + deploy | 验证 detection QDQ ONNX -> TensorRT INT8 build 参数,artifact 和 report 主链,不依赖 ultralytics 运行时 | 当前 recipe 默认使用 TensorRT Python API dry-run,固定 `Conv`-only QDQ + INT8 profile + engine artifact 结构,为目标机器上的真实 build 做准备 |
| `external_detection_trt_deploy` | 外部 detection ONNX | detection deploy | 验证 stage workflow 直接消费仓库外 ONNX,生成 TensorRT engine artifact/report 主链 | 当前 recipe 只依赖 `params.onnx_path`,不要求加载 PyTorch 模型,默认 Python API dry-run,用于承接 RT-DETR 等外部导出资产 |
| `multi_component_quant_smoke` | 异构 toy 模型 | quant + manifest | 验证 component policy,多 backend report 和 manifest 表达 | 已有 CPU smoke recipe 和 runner 测试覆盖 |
| `hf_text_kd_prune` | HF 文本分类 | distill + prune | 复用知乎示例但改为 YAML | 已有 recipe 支架和 fake HF pipeline 测试, 真实 checkpoint 运行待补 |
| `cnn_structured_prune` | Conv2d-heavy 模型 | prune + export | 验证真实宽度压缩 | ONNX 图 channel/filter 真实减少 |
| `sd_lcm_lora_4step` | Stable Diffusion 类模型 | diffusion_distill | 打通少步蒸馏最小闭环 | 4-step sampler, 图片网格和 LoRA 保存 |
| `diffusion_quant_linear` | DiT/UNet/text encoder | quant + diffusion eval | 验证扩散模型量化策略 | 固定 prompt 输出可比, VRAM/latency 有报告 |

Recipe 文件按技术栈分层放在 `xqt/recipes/` 下 (`quant/`, `prune/`, `distill/`, `operator/`, `detection/`, `smoke/`), 示例数据和大模型权重不进入 git. 每个 recipe 必须声明:

- `compression_axes`.
- 支持模型族和已验证 checkpoint.
- 依赖包和最低版本.
- 需要的硬件和显存估计.
- 可运行的最小样本数.
- 预期产物和验收阈值.

实践示例 TODO:

- [../../research/xqt-practice-examples/vit-classification-todo.md](../../research/xqt-practice-examples/vit-classification-todo.md): ViT 分类实践示例,同时检验 XQT 量化,剪枝,算子优化和组合 pipeline.
- [../../research/xqt-practice-examples/yolo-detection-todo.md](../../research/xqt-practice-examples/yolo-detection-todo.md): YOLO 检测实践示例,同时检验 XQT detection data,mAP,postprocess,量化,剪枝,算子优化和部署链路.

这两个 TODO 属于实践计划,不是已实现事实. 示例编写时如果发现 XQT 功能实现错误,优先修复或破坏性重构 `xqt` 内部实现,不要在示例脚本中绕开错误或降低验收标准.

当前已提供 `xqt/recipes/detection/yolo_detection_smoke.yaml` 作为最小 detection smoke,以及 `xqt/recipes/detection/yolo_detection_practice.yaml` + `examples/yolo_detection_practice.py` 作为 detection practice 入口. Practice recipe 使用通用 `OptimizationConfig` 风格,顶层是 `model`,`task`,`data_splits`,`stages`;`examples/yolo_detection_practice.py` 本身也是可直接运行的单文件示例,默认内嵌一条轻量 detection workflow,不依赖 `ultralytics`,并调用 `xqt.workflows.optimize_model` 打印 stage 摘要和 workflow 产物路径. 需要切换到 YAML 或真实 RT-DETR TensorRT workflow 时,用 `XQT_YOLO_PRACTICE_CONFIG` 指向对应 recipe. 每个 stage 都有独立 `kind`,`split`,`from_stage`,`params`,`accept`,`revert_on_reject` 等编排字段,运行后进入 `OptimizedModelResult.stages`,同时返回当前模型,`best_model`,stage metrics 和 artifacts. 这条链路不再维护 YOLO 专用场景矩阵,也不再把底层逻辑复制进示例脚本.

这里要区分现状和方向:

- 当前 CLI 主入口仍然是 `xqt-run-recipe` / `xqt-preflight`, 默认通过 `XQT_CONFIG` 选择 pass recipe;另外已经补了 `xqt-run-workflow`,默认指向 detection smoke workflow,通过 `XQT_WORKFLOW_CONFIG` 切换 stage workflow recipe.
- `xqt/recipes/detection/hf_rtdetr_r18vd_qdq_trt_practice.yaml` 提供了一个不引入 `ultralytics` 运行时依赖的真实 detection 资产样例: 直接消费 Hugging Face 社区版 RT-DETR R18vd 单输入 ONNX(`pixel_values -> logits,pred_boxes`),走 ONNX Runtime QDQ INT8 + TensorRT Python API deploy 主链.
- `OptimizationConfig` + `optimize_model` 已经是 stage workflow 的核心执行路径, 通用 stage workflow CLI 已补齐最小入口, 但 recipe 模板和完整用户心智仍在继续收敛.
- 长期方向是让 stage workflow + 默认模板承接主用户心智, pass recipe 逐步退到兼容层和内部桥接层, 而不是立刻假定 `XQTConfig` 已经废弃.

已验证的 YOLO practice 边界:

- `baseline_eval` 和 `baseline_latency` 阶段先记录 PyTorch mAP 和 latency,作为后续精度/速度验收参照.
- `export_fp32_onnx` 和 `fp32_onnx_runtime` 阶段先转换部署格式,再通过 ONNX Runtime 评估 mAP,raw output diff,decoded detection diff 和 latency.
- `quant_qdq` 使用 explicit calibration split 生成 XQT 自己的 ONNX Runtime QDQ INT8 产物,`qdq_onnx_runtime` 再对该产物跑同一套 runtime eval 和验收阈值.
- `prune_sparse`,`prune_eval`,`prune_latency` 阶段只跑 global L1 unstructured pruning,报告会标记 `baseline_kind=unstructured_sparsity_report` 和 `speedup_claimed=false`,不声称非结构化 mask 会带来真实部署加速.
- detection 场景下 `method=structured` 会在 preflight 给出 warning,并在 workflow 执行层返回 `execution_state=skipped`,`applied=false`,`skip_reason` 和 `skipped_modules[]`;只有 residual/CSP/C2f/concat/SPPF/detect head 的 dependency graph/rewrite 支持补齐后,才允许把它写成真实结构化剪枝.
- stage workflow 支持 `operator` 阶段实际评估 `torch_compile` candidate,并在 target report 的 `metadata.graph_break_report` 和 `metadata.fallback_detail` 中记录 graph break,compiled regions 和 fallback reason. 当前默认 `yolo_detection_practice.yaml` 不再包含 operator stage,但已有 synthetic detection workflow 测试覆盖 detection `torch_compile` 报告契约.
- `deployment_backend` targets 用于记录 backbone/neck/head/postprocess 等部署阶段语义和 fallback reason,不伪装成 PyTorch 内核替换. TensorRT/OpenVINO 自身的 graph fusion,kernel selection 或 engine/IR 优化属于部署后端收益,不要混写成 XQT operator replacement gain.
- `deploy` 阶段可导出 ONNX/TensorRT/OpenVINO 部署目标. TensorRT 和 OpenVINO dry-run 只构造命令或 Python API build 计划;如果关闭 dry-run,目标机器仍需要安装对应后端并自行承担真实 engine/IR 生成和性能验收. 当前默认 `yolo_detection_practice.yaml` 不再包含 deploy dry-run,真实 detection 部署主链优先看 RT-DETR external ONNX -> QDQ -> TensorRT recipe.

上面的 YOLO TODO 仍保留,用于继续推进真实 TensorRT engine,真实 OpenVINO IR,YOLO 结构化剪枝 dependency graph/rewrite 和 postprocess custom kernel 等更完整的部署验收.

## 12. 配置草案

XQT 当前保留两类 YAML, 需要明确区分:

- `XQTConfig` pass recipe: 面向 `xqt-run-recipe`,用于固定 pass pipeline,preflight 和 manifest. 这是当前仍在使用的兼容入口, 不是已经移除的旧系统.
- `OptimizationConfig` stage workflow: 面向 `xqt.workflows.optimize_model`,用于研究/实践入口按阶段组合 eval,benchmark,prune,quant,finetune,distill,operator,export,deploy 和 runtime_eval. 这是当前更接近长期方向的用户编排模型.

两者当前都存在, 但角色不同:

| 维度 | `XQTConfig` pass recipe | `OptimizationConfig` stage workflow |
| --- | --- | --- |
| 入口现状 | `xqt-run-recipe` / `xqt-preflight` + `XQT_CONFIG` | Python API `optimize_model`,局部示例入口 |
| 数据表达 | `data.calibration/train/validation` | `data_splits.<name>` |
| 优化表达 | 顶层 `compression.*` / `export.*` | `stages[].kind + params + accept` |
| 适合场景 | 固定 pass pipeline,preflight,manifest | 研究/实践/多阶段编排 |
| 长期方向 | 兼容层 / 内部桥接配置 | 主用户心智 |

当前不强行合并两套 schema. 更合理的方向是先把文档, 命名和用户心智对齐, 再决定是否做破坏性收敛.

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
  train:
    target: synthetic_classification
    sample_limit: 8
    batch_size: 2
    params:
      input_shape: [3, 224, 224]
      num_classes: 10

stages:
  - name: baseline_eval
    kind: eval
    split: validation
    params:
      baseline: true
  - name: baseline_latency
    kind: benchmark
    split: validation
    params:
      warmup: 10
      iterations: 50
  - name: prune_sparse
    kind: prune
    split: validation
    params:
      method: global_l1_unstructured
      target_sparsity: 0.2
  - name: prune_eval
    kind: eval
    split: validation
    compare_to: baseline_eval
    accept:
      metric: top1
      max_drop: 0.01
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

这个示例代表的是**长期收敛方向**, 不是说当前所有 recipe 都已经迁移完成.

底层 pass recipe 示例:

```yaml
project:
  name: image_resnet_onnx_qdq_int8
  artifact_dir: artifacts/xqt/image_resnet_onnx_qdq_int8

model:
  target: xdl.model.resnet18
  params:
    num_classes: 10
  checkpoint: null
  dtype: float32
  device: cpu

data:
  validation:
    target: synthetic_classification
    sample_limit: 4
    batch_size: 1
    params:
      input_shape: [3, 224, 224]
      num_classes: 10
      seed: 7

compression:
  axes:
    - precision
  distill:
    enabled: false
  prune:
    enabled: false
  quant:
    enabled: true
    backend: onnxruntime_qdq
    strategy: static_int8
    calibration_split: calibration
    keep_high_precision: [head]
    policy:
      source_name: resnet18_source.onnx
      output_path: artifacts/xqt/image_resnet_onnx_qdq_int8/resnet18_qdq.onnx
      input_names: [input]
      output_names: [output]
      sample_limit: 2
      activation_type: QUInt8
      weight_type: QInt8
      op_types_to_quantize: [Conv, MatMul, Gemm]

export:
  targets:
    - format: tensorrt
      output_path: artifacts/xqt/image_resnet_onnx_qdq_int8/resnet18_int8.engine
      precision: int8
      params:
        dry_run: true
        performance_thresholds:
          throughput_qps_min: 1.0
          latency_p99_ms_max: 10.0
          gpu_compute_time_p99_ms_max: 10.0

validation:
  output_diff:
    atol: 1.0e-3
    rtol: 1.0e-3
  metric:
    name: top1
    max_drop: 0.01

benchmark:
  warmup: 50
  iterations: 200
  percentiles: [50, 90, 99]
  measure_memory: true
```

关于 calibration / fit, 当前正式口径应当是:

- `calibration` / `calibration_split` 表示给量化方法使用的一批代表性输入数据.
- `fit` / observer 统计 / sample-based optimize 属于 quant stage 内部实现, 不应作为主用户 API 暴露.
- `validation` / `validation_split` 表示量化后用于验收误差, 任务指标和运行时表现的数据.
- 对用户应当暴露的是 calibration 输入和 calibration summary 输出, 而不是显式 `observer.fit(...)`.

关于 operator backend, 当前正式口径应当是:

- built-in executor 当前以 `torch_compile` 为主要实际执行路径.
- `torch_compile` report 会记录 `metadata.graph_break_report` 和 `metadata.fallback_detail`,用于说明 Dynamo explain 状态,graph break 数量,compiled regions 和因为 skip,compile exception 或 `min_speedup` 未达而回到 eager 的原因.
- `deployment_backend` 当前是 metadata-only, 用于记录部署阶段语义和 fallback reason, 不做 PyTorch 内核替换.
- `triton` / `tilelang` / `cutile` / `cutlass` / `custom_cuda` 当前更多是 capability / planned 表达, 不应写成"配置后就会由 built-in executor 真正编译执行".
- TensorRT/OpenVINO 自身的 graph fusion,kernel selection,engine tactic 或 IR 优化属于部署后端收益,不计入 XQT operator replacement gain;XQT 只记录这些 adapter 的 artifact,latency 和 runtime boundary.
- 手写 CUDA / custom kernel -> TensorRT plugin 部署链路仍是目标方向, 不是当前已经做实的默认闭环.

关于 runtime artifact 引用, 当前仍依赖 `ExportPass` / `QuantPass` 写入的内部 artifact key 约定. 这部分在实现层已存在, 下表汇总了当前最常见的 key, 便于用户理解 `runtime_eval.params.artifact` 应该引用哪些上游产物.

当前常用 artifact key 对照如下:

| Key | 典型写入方 | 值类型 | 何时可用 | 说明 |
| --- | --- | --- | --- | --- |
| `manifest` | `run_xqt_recipe()` | `Path` | runner 完成且开启 manifest 写入后 | 最终 manifest JSON 路径. |
| `analysis_json` | `AnalyzePass` | `Path` | `analysis.export.json=true` 时 | 逐层分析 JSON 报告. |
| `analysis_csv` | `AnalyzePass` | `Path` | `analysis.export.csv=true` 时 | 逐层分析 CSV 报告. |
| `analysis_markdown` | `AnalyzePass` | `Path` | `analysis.export.markdown=true` 时 | 逐层分析 Markdown 报告. |
| `metrics_json` | `WriteMetricsPass` | `Path` | metrics 写盘 pass 执行后 | 汇总 metrics JSON. |
| `metrics_markdown` | `WriteMetricsPass` | `Path` | metrics 写盘 pass 执行后 | 汇总 metrics Markdown. |
| `last_onnx` | `ExportPass`, `QuantPass` | `Path` | 最近一次默认 ONNX 导出或主模型 QDQ 量化后 | `runtime_eval` 默认优先尝试引用的 key. |
| `last_engine` | `ExportPass` | `Path` | 最近一次 TensorRT engine 导出后 | `runtime_eval(runtime=tensorrt)` 默认优先尝试引用的 key. |
| `quant_onnx` | `QuantPass` | `Path` | 主模型 ONNX Runtime QDQ 量化后 | 指向主模型量化产物 ONNX. |
| `last_torchscript` | `ExportPass` | `Path` | 导出 TorchScript 后 | 最近一次 TorchScript 导出产物. |
| `export_<index>` | `ExportPass` | `Path` 或 `list[Path]` | 每个 export target 执行后 | 通用导出槽位. `index` 对应 `export.targets` 中的顺序. `ncnn` 这类多文件导出时可能是路径列表. |
| `operator_optimization_candidates` | `OperatorOptimizationPass` | `dict` | operator optimization 执行后 | FX / torch.export 候选扫描结果. 这是内存态报告, 不是运行时 artifact 路径. |
| `operator_optimization_report` | `OperatorOptimizationPass` | `Path` | operator optimization 执行后 | `operator_optimization.json` 报告文件. |
| `tokenizer` | `LoadModelPass` | 任意对象 | HF bundle 模型装载后 | 主要给 HF 文本 bundle 使用, 不是文件路径. |

`QuantPass` 还有一组按组件名派生的 key, 用于多组件量化:

| Key 模式 | 何时写入 | 说明 |
| --- | --- | --- |
| `quant_onnx_<component>` | 组件级 ONNX Runtime QDQ 量化后 | 组件名不是 `model` 时使用. |
| `last_onnx_<component>` | 组件级 ONNX Runtime QDQ 量化后 | 对应组件最近一次 ONNX 量化产物. |

例如组件名为 `backbone` 时, 会写入 `quant_onnx_backbone` 和 `last_onnx_backbone`.

`runtime_eval` 当前的解析顺序是:

1. `params.path` 或 `params.artifact_path`
2. `params.artifact` 或 `params.artifact_key`
3. 默认回落到 `last_onnx` 或 `last_engine` (`runtime=tensorrt` 时)

因此:

- 跑默认 ONNX runtime 验证时, 不写 `artifact` 通常会落到最近一次 `last_onnx`.
- 验证量化主模型时, 显式写 `artifact: quant_onnx` 更清楚.
- 验证多组件量化产物时, 应显式写组件派生 key, 例如 `artifact: quant_onnx_backbone`.

对 detection `runtime_eval`, 当前还有一层需要明确的 runtime boundary 语义:

- report 会附带 `runtime_boundary` 字段, 当前已用于 ONNX Runtime 和 TensorRT detection runtime_eval.
- `runtime_boundary.quantized_runtime_scope` 表达量化命中的前向图范围. 现阶段 ONNX Runtime QDQ 主链记录为 `onnx_graph_only`.
- `runtime_boundary.quant_backend`, `quantized_op_types`, `qdq_node_count`, `quantize_linear_count`, `dequantize_linear_count` 用于说明量化图内部实际命中的 backend 和 QDQ 图规模.
- `runtime_boundary.decode_stage` 当前固定记为 `runtime_eval_postprocess`.
- `runtime_boundary.decode_execution` 当前固定记为 `outside_quantized_graph`, 表示 detection decode / score activation / box rescale / NMS 等后处理仍在量化图外执行.
- 当 `detection_postprocess.format=auto` 时, `runtime_boundary.raw_model_output_contract` 当前记录为 `logits+pred_boxes`, 对应 Hugging Face 社区 RT-DETR ONNX / TRT 这条真实闭环.
- 因此, detection TRT/ONNX 的量化或融合收益当前只应解释为 runtime 内部前向图收益, 不应把 Python 侧后处理混写成 quant graph 内收益.

对 detection dataset metadata, 当前 resize 语义也需要固定口径:

- `metadata.dataset.resize` 表示 recipe 级目标输入尺寸.
- `metadata.dataset.resize_mode` 当前固定记为 `direct_resize`.
- `metadata.dataset.letterbox` 当前固定记为 `false`.
- `metadata.batches[].orig_sizes` 表示原图尺寸.
- `metadata.batches[].input_image_sizes` 表示真正送入模型的尺寸.
- 这表示当前内置 `synthetic_detection` / `coco8_detection` loader 还没有 letterbox / padding 语义, `ImageBoxesTransform` 只做直接缩放和可选翻转. 因此 detection mAP 和 runtime_eval 的当前事实应按 direct resize 解释,不要误写成 letterbox 预处理链路.

## 13. 任务排期

### 阶段 0: 文档和边界

- [x] 确认 `xqt` 是否长期作为独立包, 或未来并入 `xdl` 子模块. 当前决策是保持仓库顶层实验包,不并入 `xdl/` 主框架子模块.
- [x] 把 `xqt/README.md` 改成入口文档, 链接本文.
- [x] 盘点现有实验脚本, 标注依赖, 硬件和数据路径.
- [x] 定义首个支持场景: image classification + ResNet/CNN ONNX QDQ INT8 为主线, PyTorch runtime torchao 量化为并行后端能力.

### 阶段 1: 基础设施

- [x] 新建 `xqt/core`, `xqt/pipeline`, `xqt/eval`, `xqt/benchmark`, `xqt/recipes` 基础目录.
- [x] 定义 dataclass/structured config 和 OmegaConf 加载.
- [x] 实现 artifact manifest, checksum, 版本记录.
- [x] 实现 `compression_axes` schema, 支持 `width`, `depth`, `precision`, `sparsity`, `steps`.
- [x] 实现最小 pass manager 和 YAML runner. 当前 runner 已能执行内置 CPU smoke pipeline, 真实后端 pass 仍在后续阶段.
- [x] 建立 `tests/xqt/` smoke tests.

### 阶段 2: 基线评估和报告

- [x] 实现 PyTorch baseline eval.
- [x] 实现输出 diff 比较工具.
- [x] 实现 latency benchmark, 包含 warmup, CUDA synchronize, p50/p90/p99.
- [x] 实现 memory benchmark, CPU 使用进程 RSS,CUDA 使用 peak allocated/reserved.
- [x] 生成 JSON/CSV/Markdown 报告.

### 阶段 3: 量化 MVP

- [x] 实现 `quant/torchao_backend.py` 和 `quant/sensitivity.py`,承载 torchao adapter,allowlist/denylist,dtype policy,逐层激活 diff,weight diff 和混合精度建议.
- [x] 支持 allowlist/denylist, dtype policy 和逐层误差报告的基础 helper.
- [x] 支持 calibration dataloader 的基础 activation statistics helper.
- [x] 实现 ONNX Runtime static QDQ INT8 adapter, 支持 PyTorch iterable 到 `CalibrationDataReader` 的桥接和内置 `onnxruntime_qdq` pass.
- [x] 跑通一个 ResNet/CNN 的 PTQ smoke recipe. 当前 `image_resnet_onnx_qdq_int8.yaml` 可在测试中通过 synthetic image + ONNX Runtime QDQ adapter + TensorRT dry-run 路径.
- [x] 跑通真实本地数据的 ONNX Runtime QDQ CPU recipe. 当前 `image_resnet_cifar100_qdq_cpu.yaml` 已使用本地 CIFAR-100 生成真实 QDQ ONNX, manifest 和 benchmark.
- [ ] 跑通 `image_resnet_onnx_qdq_int8` 或 detection QDQ recipe 的 TensorRT 真实 engine 生成. 当前 XQT 的 preflight 已会校验 target,data root,可选依赖,后端命令和 CUDA 需求,导出侧也已同时支持 `trtexec` 和 `python_api` 两条 adapter. 但当前测试环境还没有可验证的真实 TensorRT engine 构建结果,因此主链仍以 dry-run,输出解析,性能阈值判定和 manifest metric 覆盖为主. `hf_text_kd_prune` 的真实 HF 数据运行还需要安装 `datasets`.

### 阶段 4: PyTorch native,ONNX 和 TensorRT 导出

- [x] 实现 `export/torch_exporter.py`, 支持 `torch_export` ExportedProgram 保存/加载校验和 TorchScript trace/script fallback.
- [x] 实现 `export/onnx_exporter.py`, 默认 `dynamo=True`.
- [x] 实现导出前前置融合 helper, 支持 eager/fx 两种模式并在 ONNX export metadata 中留痕.
- [x] 实现 ONNX checker 和 ONNX Runtime diff 验证.
- [x] 实现 `export/tensorrt.py`, 支持 `trtexec` 和 TensorRT Python API 两条 build 路径,包含 dry-run,命令/plan 构造测试,性能摘要解析和 `performance_thresholds` 阈值报告.
- [x] 支持 TensorRT dynamic shape profile 和 FP16/INT8 标记.

### 阶段 5: 剪枝 MVP

- [x] 实现非结构化 pruning baseline 和 sparsity 报告, 并接入内置 `prune` pass.
- [x] 实现 Conv2d/Linear 结构化剪枝和模块改写的基础 helper.
- [x] 增加剪枝后微调 recipe. 当前 `xqt/recipes/prune/unstructured/prune_finetune_cpu.yaml` 使用线性 sparsity schedule, teacher 注入和 KD 微调 smoke 测试.
- [x] 验证剪枝模型可导出 ONNX.
- [x] 增加 HF Transformer 文本分类的全局 L1 非结构化剪枝 recipe, 用 YAML 替代 argparse. 当前已提供 `xqt/recipes/distill/hf_text_kd_prune.yaml`, HF bundle adapter 和 fake HF pipeline 测试, 真实 checkpoint 验证仍作为后续任务.

### 阶段 6: 蒸馏 MVP

- [x] 实现 logit KD loss helper.
- [x] 实现 feature hook 和层对齐配置的基础 helper.
- [x] 实现 teacher logits/features 缓存.
- [x] 跑通 teacher -> student 分类 recipe 的基础 PyTorch helper 和内置 `distill` pass.
- [x] 增加剪枝 + KD 组合 recipe 的基础 helper, 支持 teacher logits 磁盘缓存和线性稀疏率 schedule.

### 阶段 7: 扩散少步蒸馏 MVP

- [x] 实现 `DiffusionSpec` 和 prompt/latent/trajectory 缓存格式.
- [x] 实现 4-step LCM/consistency LoRA recipe 的基础 loss/report helper.
- [x] 实现固定 seed 图片网格和 latency/VRAM 报告的元数据结构.
- [x] 预留 DMD/DMD2/RCGM adapter 接口, 不阻塞首期闭环.

### 阶段 8: OpenVINO 和移动端扩展

- [x] 实现 OpenVINO IR exporter 和 runtime diff 的可选依赖 adapter.
- [x] 评估 ExecuTorch 接入路径. 当前实现为 `export/mobile.py` 中的可选依赖 adapter, 支持 `.pte` dry-run 和真实依赖存在时的 `torch.export` -> ExecuTorch 路径.
- [x] 评估 ncnn/MNN 通过 ONNX 转换的可维护成本. 当前实现为 `pnnx`, ONNX -> `onnx2ncnn` 和 ONNX -> `MNNConvert` 命令 adapter, 支持 dry-run. ncnn 官方当前更推荐 `pnnx`, `onnx2ncnn` 仅作为兼容路径, 真实二进制验证后再升级为 P1 recipe.
- [x] 增加部署格式能力矩阵.

### 阶段 9: XDL 集成和稳定化

- [x] 提供从 XDL `TrainSetup` 或 checkpoint 构建 XQT context 的 adapter. 当前文件为 `xqt/xdl_adapter.py`, 支持 TrainSetup-like 对象和 PyTorch/XDL-style checkpoint.
- [x] 决定哪些 API 可以进入 `docs/md/README.md#xdl-api-稳定边界` 的 Provisional 区. 当前只把 `xqt` 顶层配置,runner,manifest 和 XDL adapter 入口列为 Provisional,子模块仍按 Internal 处理.
- [x] 增加不使用 argparse 的 recipe 运行入口. 当前入口为 `xqt-run-recipe`,读取 `XQT_CONFIG` 和 `XQT_WRITE_MANIFEST`.
- [x] 增加 recipe preflight 检查. 当前 `xqt.preflight_xqt_config` 和 `xqt-preflight` 会检查 target,数据 root,可选依赖,后端命令和 CUDA 硬件需求.
- [x] 增加 CPU smoke tests, 至少覆盖不依赖 GPU 的 PyTorch native export,ONNX export 和 manifest. 当前由 `tests/xqt/test_runner.py` 等覆盖, 是否接入仓库 CI 配置另行决定.
- [x] 为首批 recipe 补用户阅读页或示例文档. 当前阅读页为 `docs/html/xqt.html`,事实源仍以本文和源码为准.

## 14. 首期最小闭环

建议第一轮只做一条闭环, 不同时追所有方向:

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

完成这条闭环后, 再并行扩展剪枝和蒸馏. 原因是量化和导出最能暴露模型 I/O, shape, dtype, artifact, benchmark 和后端 adapter 的公共问题.

另一个适合早期验证的 NLP 闭环:

```text
HuggingFace text classification model
    -> teacher/student baseline eval
    -> logit KD training
    -> optional global L1 unstructured pruning
    -> optional teacher logits cache
    -> sparsity report
    -> ONNX export
    -> ONNX Runtime diff
    -> manifest
```

这个闭环来自简单代码示例, 适合验证 distill/prune/pipeline/report 的公共抽象. 但实现时必须按本仓库约束改成 YAML/OmegaConf 配置, 不引入 `argparse`.

## 15. 外部基线

涉及第三方 API 时先查官方文档, 不要只按旧示例写:

- torchao 文档: <https://docs.pytorch.org/ao/stable/index.html>
- PyTorch ONNX exporter: <https://docs.pytorch.org/docs/2.12/onnx.html>
- PyTorch pruning tutorial: <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>
- PyTorch knowledge distillation tutorial: <https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html>
- ONNX 文档: <https://onnx.ai/onnx/intro/>
- TensorRT quick start: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html>
- OpenVINO model preparation: <https://docs.openvino.ai/2026/openvino-workflow/model-preparation.html>
- 量化方法族: GPTQ, AWQ, SmoothQuant, OmniQuant, HQQ, SpQR, AQLM, QuaRot, SpinQuant, KIVI/KV cache quantization. 实现前必须分别核对论文, 官方仓库和目标后端支持.
- 剪枝方法族: Wanda, SparseGPT, LLM-Pruner, Sheared LLaMA, SliceGPT, ShortGPT, movement pruning, N:M sparsity. 需要区分压缩率, 真实延迟和后端稀疏支持.
- 通用蒸馏方法族: TinyBERT, MiniLM, Distilling step-by-step, MiniLLM, GKD/on-policy distillation, sequence-level KD. 生成任务和分类任务的 loss 不能混用.
- 扩散少步蒸馏方法族: LCM, TCD, Progressive Distillation, DMD/DMD2, SiD, Flash Diffusion, Hyper-SD, Shortcut Models, MeanFlow, RCGM, ADD/SDXL-Turbo. 需要按 prediction target, scheduler 和 reward/discriminator 依赖拆 adapter.
- RCGM: <https://github.com/LINs-lab/RCGM>. 该方向对应 N-th order recursive consistent velocity field estimation, 面向 any-step / few-step 生成. 当前仓库训练和推理代码仍标记为 TODO, 先作为 P1 研究接入路线.
- Zhihu 示例: <https://zhuanlan.zhihu.com/p/1938005398637507760>. 这篇文章演示 HF 文本分类 KD, 全局 L1 非结构化剪枝, 剪枝 + KD 组合和 teacher logits 缓存. 只能作为 recipe 参考, 代码不能直接照搬, 因为它使用 `argparse` 且未按本仓库 YAML 配置规范组织.
