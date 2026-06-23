# XQT 摘要

XQT 只关注模型本身.

它负责模型压缩,模型图变换,导出适配,模型误差分析和 benchmark. 它不负责训练,QAT 训练,finetune,distillation,KD recovery,prune recovery,dataset/dataloader,training provider 或 evaluation provider.

当前可用能力:

- 量化: PTQ,QDQ,torchao,calibration,layer sensitivity.
- 剪枝: unstructured,structured,N:M,block sparse,mask/rewrite.
- 算子优化: `torch.compile` 和 planned backend capability/report.
- 导出: ONNX,torch.export,TorchScript,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN.
- 分析: output diff,layer analysis,latency,memory,manifest.

当前核心 API:

- `load_xqt_config`
- `run_xqt_recipe`
- `preflight_xqt_config`
- `optimize_model`
- `XQTOptimizationSession`

当前包模块:

- `core`,`model`,`integrations`,`pipeline`,`workflows`,`quant`,`prune`,`operator_opt`,`export`,`eval`,`benchmark`

不属于 XQT 的内容:

- `distill/`
- `diffusion_distill/`
- training/evaluation provider
- dataset/dataloader 构建
- `finetune`/`distill` stage
- `eval`/`runtime_eval` stage
- QAT 训练和 recovery training
