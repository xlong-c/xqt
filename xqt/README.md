# xqt

`xqt` 是 XDL 仓库中的模型压缩,图变换和部署实验包. 它不是稳定公共 API,只保留薄入口和跳转.

## 先看什么

- Markdown 总入口: [../docs/md/index.md](../docs/md/index.md)
- 架构正文: [../docs/md/architecture/xqt.md](../docs/md/architecture/xqt.md)
- 概念说明: [../docs/md/explanation/xqt-concepts.md](../docs/md/explanation/xqt-concepts.md)
- 工作流入口: [../docs/md/usage/xqt-workflows.md](../docs/md/usage/xqt-workflows.md)
- 兼容事实源: [../docs/md/XQT.md](../docs/md/XQT.md)
- 兼容摘要入口: [../docs/md/XQT_SUMMARY.md](../docs/md/XQT_SUMMARY.md)
- 包内工程契约: [FRAMEWORK.md](FRAMEWORK.md)

## 核心边界

XQT 只关注模型本身. 它接收 PyTorch 模型,checkpoint 或导出产物,执行量化,剪枝,算子优化,导出,误差分析和 benchmark. 训练,QAT 训练,finetune,distillation,recovery,dataset / dataloader 和 provider 编排不属于 XQT.

## 当前可用入口

- `from xqt import XQTOptimizationSession`
- `optimize_model("path/to/workflow.yaml")`
- `xqt-run-workflow`

## 修改顺序

1. 先看 `docs/md/architecture/xqt.md`.
2. 再看 `docs/md/usage/xqt-workflows.md`.
3. 再看 `xqt/FRAMEWORK.md`.
4. 最后看相关 `recipes/*.yaml` 和测试.
