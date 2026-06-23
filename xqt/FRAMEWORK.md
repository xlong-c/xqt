# XQT 框架形态与工程契约

本文是 `xqt/` 包内工程契约. XQT 的核心边界只有一条:

```text
XQT 只关注模型本身.
```

XQT 负责模型压缩,模型图变换,导出适配,模型误差分析和模型运行时验证. XQT 不负责训练,不负责 QAT 训练循环,不负责 finetune,不负责 distillation/recovery,不负责 task provider,也不拥有 XDL 或第三方框架的训练语义.

## 1. 定位

XQT 的主链路是:

```text
PyTorch model / checkpoint / exported artifact
    -> model compression or graph transform
    -> model-side diff / layer analysis
    -> export target artifact
    -> runtime diff / benchmark / manifest
```

允许输入:

- `torch.nn.Module`
- `state_dict` 或 checkpoint 加载后的模型
- ONNX / torch.export / TorchScript 等已导出模型产物

允许输出:

- 变换后的 PyTorch 模型
- ONNX,QDQ ONNX,TensorRT engine,OpenVINO IR,torch.export,TorchScript,mobile artifact 等部署产物
- output diff,layer error,sparsity,QDQ graph stats,calibration summary,latency,memory,artifact manifest 和 report

## 2. 应该负责

- PTQ,QDQ,权重量化,activation calibration,layer sensitivity 和混合精度建议.
- 结构化剪枝,非结构化剪枝,N:M,block sparse,mask/rewrite 和 sparsity report.
- `torch.compile` 这类模型运行图优化,以及 Triton/TileLang/CUTLASS 等后端的 capability/adapter 边界.
- ONNX,TensorRT,OpenVINO,torch.export,TorchScript,ExecuTorch,ncnn,MNN 等导出和部署适配.
- PyTorch 模型与导出产物之间的 output diff,runtime diff,layer analysis 和 benchmark.
- recipe/preflight/manifest/report 这些围绕模型变换链路的工程记录.

## 3. 不应该负责

- 训练循环: `zero_grad -> backward -> step`.
- QAT 训练,finetune,distillation,KD recovery,prune recovery 或 diffusion distillation.
- training provider / evaluation provider 抽象.
- 通用 trainer,task model registry,loss registry 或 metric registry.
- 复制 XDL,Ultralytics,HuggingFace,timm,diffusers 等框架的训练/评估语义.
- 自己实现 TensorRT tactic selection,OpenVINO graph optimizer,ONNX Runtime kernel 等后端内部优化器.

需要梯度更新的流程归 XDL 或第三方训练工具. XQT 只消费训练后的模型或 checkpoint,继续做模型侧压缩,转换,导出和验证.

## 4. 数据角色

XQT 只保留模型优化需要的数据角色:

- `calibration`: PTQ/observer/QDQ calibration 的代表性输入.
- `validation`: output diff,runtime diff,task smoke metric 和 benchmark 的输入.

`train_split` 不属于 XQT 配置语义. 如果某个优化方法需要训练数据或反向传播,它应该在 XDL 或外部训练工具里完成,再把训练后的模型交给 XQT.

## 5. Stage 和 API 规范

- 可复现链路优先写 stage workflow YAML.
- 交互探索优先用 `XQTOptimizationSession`.
- 支持的 stage 限定为 `eval`,`benchmark`,`prune`,`quant`,`operator`,`export`,`deploy`,`analyze`,`runtime_eval`.
- 量化 recipe 必须显式写 `backend` 和 `policy`;`backend` 是执行/运行时后端,`method` 是量化算法,`strategy` 是后端内模式或 dtype 策略.
- AWQ/GPTQ 等算法必须写在 `method`,不能写成 `backend`.
- QDQ/PTQ recipe 必须显式写 `calibration_split`,不能把 validation 数据隐式当 calibration fallback.
- planned/capability-only 后端必须在 preflight 和文档中明确标注,不能写成已经可执行闭环.
- 新增长期文档时遵守仓库文档分层: `docs/md/XQT.md` 是事实源,`xqt/FRAMEWORK.md` 是包内工程契约,阶段性研究继续放 `research/`.

## 6. 删除边界

以下内容不属于 XQT:

- `finetune` / `distill` stage.
- `distill/` 和 `diffusion_distill/` 包.
- `training_provider` / `evaluation_provider` / `XDLTrainingProvider`.
- HF text KD,teacher cache,teacher/student feature alignment 和训练 loss wrapper.
- 剪枝后的 KD recovery 或 QAT recovery.

如果未来需要这些能力,应优先在 XDL 训练体系或第三方 provider 中实现,再以训练后模型/checkpoint 的形式进入 XQT.
