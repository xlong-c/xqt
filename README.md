# xqt

`xqt` 是模型侧优化工具链: 只关注模型本身, 负责模型压缩, 图变换, 导出适配, 误差分析和 benchmark. 训练, QAT, finetune, distillation, recovery, dataset / dataloader 以及 training / evaluation provider 不属于本仓库, 它们属于训练侧兄弟仓库 `XDL`; 训练后的模型或 checkpoint 通过产物衔接交给 `xqt`.

## 配置方式

`xqt` 只提供两种配置方式, 前者为第一选择:

1. **`XQTOptimizationSession`** (第一配置): Python 交互式 session, 逐步编排 benchmark / prune / quant / operator / export / deploy / analyze. 适合探索, 调试和 notebook.
2. **YAML workflow** (第二配置): 声明式 recipe, 通过 `optimize_model()` 或 CLI 入口运行, 适合可复现批量实验.

不新增 CLI 参数解析库, JSON-shaped Python dict 硬编码 workflow 或其他配置路径.

## 运行

```bash
pip install -e .
XQT_WORKFLOW_CONFIG=xqt/recipes/smoke/smoke_workflow.yaml xqt-run-workflow
```

`run_workflow.py` 从环境变量 `XQT_WORKFLOW_CONFIG` 读取 stage workflow recipe, 未设置时回退到 `xqt/recipes/smoke/smoke_workflow.yaml`. 包内 recipe 见 `xqt/recipes/`, 顶层示例脚本见 `examples/`, 示例必须用 `PYTHONPATH=. python examples/...` 运行.

`xqt` 与 `XDL` 通过 checkpoint / 模型产物衔接, 不互相接管职责. 5 个 recipe 使用 `xdl.model.*` 字符串 target, 需要与 `XDL` 同工作区做本地 editable 安装 (`pip install -e ../xdl`); 只做纯模型侧压缩, 导出或 benchmark 时不需要安装 `XDL`.

## 文档入口

- 兼容长期事实源: [docs/md/XQT.md](docs/md/XQT.md)
- 架构正文: [docs/md/architecture/xqt.md](docs/md/architecture/xqt.md)
- 概念说明: [docs/md/explanation/xqt-concepts.md](docs/md/explanation/xqt-concepts.md)
- 工作流入口: [docs/md/usage/xqt-workflows.md](docs/md/usage/xqt-workflows.md)
- 包内工程契约: [xqt/FRAMEWORK.md](xqt/FRAMEWORK.md)
- 算子优化记录: [docs/md/explanation/operator-optimization-records.md](docs/md/explanation/operator-optimization-records.md)

## 开发

```bash
python -m pytest tests/ -q
ruff check xqt/
```

工程约束和 API 边界见 [AGENTS.md](AGENTS.md).
