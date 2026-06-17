# xqt — 模型压缩与部署实验目录

## 目录职责

- 存放量化,蒸馏,剪枝,扩散少步蒸馏和部署格式导出相关实验脚本与模型尝试
- 作为独立于 `xdl/` 主框架的专题实验区
- 长期架构草案见 `docs/md/XQT.md`

## 当前内容

- `bf16_clein.py`
- `sdnq_clein.py`
- `torchao_vit.py`
- `core/`, `data/`, `pipeline/`, `eval/`, `benchmark/`, `quant/`, `prune/`, `distill/`, `diffusion_distill/`, `export/`: 当前 XQT 包化基础能力
- `recipes/smoke_cpu.yaml`: CPU smoke recipe
- `recipes/hf_text_kd_prune.yaml`: HF 文本分类 KD + pruning recipe 支架

## 修改约束

- 这里偏实验和专题验证,不默认代表稳定接口
- 公共压缩或部署工具若可复用,再考虑抽到 `tools/` 或 `xdl/`
- 外部库依赖,设备要求,模型限制要写清楚

## 注意事项

- 量化,部署和扩散少步蒸馏脚本通常对环境版本敏感,涉及外部 API 时先确认最新文档
