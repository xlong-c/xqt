# xqt — 模型压缩与部署实验目录

## 目录职责

- 存放量化,蒸馏,剪枝,扩散少步蒸馏和部署格式导出相关实验脚本与模型尝试
- 作为独立于 `xdl/` 主框架的专题实验区
- 长期工作文档见 `docs/md/XQT.md`

## 当前内容

- `core/`, `data/`, `pipeline/`, `eval/`, `benchmark/`, `quant/`, `prune/`, `distill/`, `diffusion_distill/`, `export/`: 当前 XQT 包化基础能力
- `recipes/smoke_cpu.yaml`: CPU smoke recipe
- `recipes/hf_text_kd_prune.yaml`: HF 文本分类 KD + pruning recipe 支架

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

## 注意事项

- 量化,部署和扩散少步蒸馏脚本通常对环境版本敏感,涉及外部 API 时先确认最新文档
