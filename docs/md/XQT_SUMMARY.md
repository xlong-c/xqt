# XQT 摘要入口

本文是 `docs/md/XQT.md` 的摘要版入口,用于先解释 `xqt/` 现在是什么,适合解决什么问题,以及应该从哪里继续深挖. 真实事实边界,模块状态,recipe 现状和任务清单仍以 [XQT.md](XQT.md) 及源码为准.

## 这篇摘要负责什么

- 给出 XQT 的定位,目标链路和非目标.
- 说明当前模块分层和推荐阅读顺序.
- 把常见任务映射到详细版章节.

## 这篇摘要不负责什么

- 不展开每个 backend adapter,recipe 字段和任务 backlog.
- 不单独承诺 API 稳定性.
- 不替代详细版中的模块状态表,能力矩阵和阶段性任务清单.

## 一句话定位

`xqt/` 是 XDL 仓库里的模型压缩与部署实验工具链. 它把量化,剪枝,蒸馏,扩散少步蒸馏,导出和性能验证收拢成可编排的包模块与 recipe,目标是从 PyTorch 产物走到可验证的推理产物.

## 当前边界

- `xqt` 仍是实验包,不是 XDL 主框架稳定子模块.
- 顶层配置,runner,stage workflow,manifest 和 XDL adapter 按 Provisional 管理.
- `core`,`pipeline`,`quant`,`prune`,`distill`,`diffusion_distill`,`export`,`operator_opt` 等子模块细节默认按 Internal 管理.

需要明确 API 边界时,回到 [XQT.md](XQT.md) 和 [README.md#xdl-api-稳定边界](README.md#xdl-api-稳定边界).

## 目标链路摘要

XQT 主要关心这条工程路径:

```text
PyTorch checkpoint
    -> 可选蒸馏 / 剪枝 / 量化 / 算子优化 / 扩散少步蒸馏
    -> 导出部署格式
    -> 精度验证与性能基准
```

它强调:

- PyTorch 优先.
- 配置集中到 YAML / OmegaConf.
- 外部后端通过 adapter 隔离.
- 每步都要能验证指标和产物.
- 产物目录要带 manifest 与配置快照.

## 常见任务先看哪里

| 任务 | 先看 | 详细版 |
| --- | --- | --- |
| 想快速判断 XQT 适不适合当前项目 | [目标链路摘要](#目标链路摘要) | [XQT.md#1-项目定位](XQT.md#1-项目定位) |
| 想知道当前已经有哪些模块 | [模块地图摘要](#模块地图摘要) | [XQT.md#4-当前模块划分](XQT.md#4-当前模块划分) |
| 想做 PTQ / QDQ / torchao 量化 | [能力分层摘要](#能力分层摘要) | [XQT.md#71-量化](XQT.md#71-量化) |
| 想做 structured prune 或 prune + KD | [能力分层摘要](#能力分层摘要) | `XQT.md` 中剪枝与蒸馏相关章节 |
| 想做 diffusion few-step distillation | [能力分层摘要](#能力分层摘要) | `XQT.md` 中扩散蒸馏相关章节 |
| 想导出 ONNX / TensorRT / OpenVINO / mobile | [模块地图摘要](#模块地图摘要) | `XQT.md` 中导出相关章节 |
| 想知道 recipe 和 workflow 怎么串起来 | [阅读顺序摘要](#阅读顺序摘要) | [XQT.md#3-设计原则](XQT.md#3-设计原则), [XQT.md#4-当前模块划分](XQT.md#4-当前模块划分) |
| 想确认当前 backlog 或支持状态 | [当前状态摘要](#当前状态摘要) | [XQT.md](XQT.md) 对应任务与状态章节 |

## 模块地图摘要

当前最值得先建立心智模型的是这几层:

- `core`: schema,OmegaConf 加载,artifact manifest,registry.
- `pipeline` / `workflows`: 把 pass 或 stage 按配置编排起来.
- `quant`,`prune`,`distill`,`diffusion_distill`,`operator_opt`,`export`: 具体优化和部署能力.
- `eval`,`benchmark`: 做精度 diff,任务指标和性能报告.
- `data`,`model`,`xdl_adapter`: 衔接数据,外部模型和 XDL.

如果你只是想找一个改动落点,这一级通常已经够用. 具体文件职责见 [XQT.md#4-当前模块划分](XQT.md#4-当前模块划分).

## 阅读顺序摘要

推荐顺序:

1. `xqt/core/schema.py` 和 `xqt/core/config.py`
2. `xqt/pipeline/runner.py` 和 `xqt/pipeline/passes.py`
3. 按任务进入 `quant/`,`prune/`,`distill/`,`diffusion_distill/`,`export/`,`operator_opt/`
4. 最后看 `xqt/recipes/*.yaml` 和 `tests/xqt/`

这套顺序的原因是: 先看配置和执行骨架,再看能力模块,最后看 recipe 与测试闭环. 详细说明见 [XQT.md#3-设计原则](XQT.md#3-设计原则).

## 能力分层摘要

XQT 现在重点覆盖四类能力:

- 量化: BF16/FP16 baseline, weight-only, FP8, ONNX Q/DQ 等.
- 剪枝: global L1, structured pruning,N:M 和 block sparse 报告.
- 蒸馏: logit KD,feature/relation distillation, teacher cache, prune + KD.
- 部署与算子优化: `torch.export`,ONNX,TensorRT(`trtexec` / Python API),OpenVINO,mobile adapter,`torch.compile` 等.

其中量化和导出是最优先的工程闭环. 更细的优先级和技术分层回到 [XQT.md#71-量化](XQT.md#71-量化) 及后续相关章节.

## 当前状态摘要

可以把 XQT 理解为"已经包化,仍在快速演进"的阶段:

- 已经有 `core/pipeline/recipes/tests` 这套基本骨架.
- 已经有 smoke recipe 和若干真实路径样例.
- 仍然有不少 backend 和高级压缩策略停留在 planned / preflight / dry-run 层,但 TensorRT 已经不只是一条命令拼装路径,导出侧已有 Python API build adapter 可接真实 QDQ ONNX.

因此,判断一条路径能不能直接拿来用,不要只看模块名,还要检查对应 recipe,测试和详细版中的状态说明.

## 何时必须直接看详细版

- 你要新增 recipe 字段,manifest 字段或 workflow stage.
- 你要判断某个 backend 是已执行,仅 preflight,还是只做接口预留.
- 你要更新 API 边界,模块状态或 backlog.
- 你要引用当前支持矩阵做设计决策.

这些场景不要停留在摘要,直接打开 [XQT.md](XQT.md).
