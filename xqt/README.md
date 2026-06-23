# xqt

`xqt` 是 XDL 仓库中的模型压缩,模型图变换和部署格式导出实验包. 当前目录是包化实验工具链,不代表稳定公共 API.

核心契约: XQT 只关注模型本身. 它接收 PyTorch 模型,checkpoint 或导出产物,执行量化,剪枝,算子优化,导出,误差分析和 benchmark. 训练,QAT 训练,finetune,distillation,KD/prune recovery,dataset/dataloader 和 provider 编排不属于 XQT.

包内工程契约:

- [FRAMEWORK.md](FRAMEWORK.md): 直接修改 `xqt/` 前先看的边界说明.

长期工作文档:

- 摘要入口: [../docs/md/XQT_SUMMARY.md](../docs/md/XQT_SUMMARY.md)
- 详细事实源: [../docs/md/XQT.md](../docs/md/XQT.md)

可选依赖:

- `pip install -e ".[xqt]"`: ONNX/QDQ/torchao 基础路径.
- `pip install -e ".[xqt-all]"`: XQT Python 侧全量可选依赖.

当前包模块:

- `core`: structured config, artifact manifest, checksum, XQT registry.
- `model`: smoke-only model helper 和模型 forward hook 输出采集工具.
- `integrations`: detection output decode adapter.
- `pipeline`: sequential pass manager,preflight 和 YAML runner.
- `workflows`: stage-based model optimization workflow,支持 `benchmark`,`prune`,`quant`,`operator`,`export`,`deploy`,`analyze`.
- `eval`: tensor output diff,layer analysis 和 report helper.
- `benchmark`: latency 和 memory benchmark helper.
- `quant`: quantization policy,backend/method capability matrix,activation calibration,layer sensitivity helper.
- `operator_opt`: `torch.compile`-first operator optimization pass,backend capability matrix and runtime fallback reporting.
- `export`: `torch.export`,TorchScript,ONNX,TensorRT,OpenVINO,ExecuTorch,ncnn,MNN 等导出 adapter.
- `prune`: unstructured,structured,N:M 和 block sparse pruning helper.
- `xdl_adapter`: 从 XDL TrainSetup-like 对象或 checkpoint 提取模型上下文,不接管训练.

阅读和扩展顺序:

1. 先看 `core/schema.py` 和 `core/config.py`,确认 recipe schema 和 OmegaConf 加载规则.
2. 再看 `pipeline/runner.py` 和 `pipeline/passes.py`,确认默认 pass 顺序和真实执行行为.
3. 按任务进入 `quant/`,`prune/`,`operator_opt/`,`export/`,`eval/`,`benchmark` 等模块.
4. 最后看 `recipes/*.yaml` 和测试,确认当前路径是否已经有可运行闭环.

当前主 recipe 方向:

- smoke CPU: 配置,PyTorch native export 和 manifest 路径.
- ONNX QDQ INT8: PyTorch model 或外部 ONNX -> QDQ ONNX,校准输入由调用方显式传入.
- TensorRT/OpenVINO/mobile: 导出 adapter,dry-run/真实后端产物构建.
- pruning: 纯模型侧 mask/rewrite/sparsity report,不做 recovery training.
- operator optimization: `torch.compile` 和 planned backend capability/report.

运行入口:

- `xqt-preflight`: 不解析命令行参数,默认检查 `recipes/smoke_cpu.yaml`.
- `XQT_CONFIG=/abs/path/to/recipe.yaml xqt-preflight`: 检查 recipe.
- `xqt-run-recipe`: 不解析命令行参数,默认运行 `recipes/smoke_cpu.yaml`.
- `XQT_CONFIG=/abs/path/to/recipe.yaml xqt-run-recipe`: 切换 recipe.
- `XQT_WRITE_MANIFEST=0 xqt-run-recipe`: 跳过 manifest 写入.
- `xqt-run-workflow`: 不解析命令行参数,默认运行模型优化 workflow.
- `XQT_WORKFLOW_CONFIG=/abs/path/to/workflow.yaml xqt-run-workflow`: 切换 stage workflow recipe.

不属于 XQT:

- 训练循环,QAT 训练,finetune,distillation,KD recovery,prune recovery.
- `training_provider`,`evaluation_provider`,`XDLTrainingProvider`.
- dataset/dataloader 构建和 task-level validation.
- 通用 model/loss/metric registry.
- 任务框架的训练/评估语义.
