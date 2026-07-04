# XQT 模型优化工具链

本文保留为 `XQT` 的兼容长期事实源. 新的 Markdown 主导航已经迁到 [index.md](index.md), 并按架构 / 说明 / 使用三层组织.

XQT 只关注模型本身. 它接收 PyTorch 模型,checkpoint 或导出产物,执行模型侧压缩,图变换,导出适配,误差分析和 benchmark. 训练,QAT,finetune,distillation,recovery,dataset / dataloader,training provider 和 evaluation provider 不属于 XQT.

需要梯度更新的流程归 XDL 或第三方训练工具,再把训练后的模型或 checkpoint 交给 XQT.

## 1. 项目定位

本章节保留为兼容锚点. `XQT` 的系统定位和模型侧边界已经迁到:

- [architecture/xqt.md](architecture/xqt.md)
- [explanation/xqt-concepts.md](explanation/xqt-concepts.md)

如果你要更新 `XQT` 做什么, 不做什么, 主链路或模型侧职责, 优先更新这些 canonical 页面.

## 2. 配置方式

本章节保留为兼容锚点. `XQT` 的使用入口和工作流已经迁到:

- [usage/xqt-workflows.md](usage/xqt-workflows.md)
- [architecture/xqt.md](architecture/xqt.md)

如果你要更新 `session` / YAML workflow 的主路径或配置边界, 优先更新这些 canonical 页面.

## 3. 当前可用能力

本章节保留为兼容锚点. `XQT` 的能力地图和概念说明已经迁到:

- [explanation/xqt-concepts.md](explanation/xqt-concepts.md)

如果你要更新当前能力范围, 半可用 / 实验性状态或能力地图, 优先更新上面的说明层页面.

当前 `TileLang` 的受限 kernel target 已覆盖 `attention`, `conv`, direct half `linear`, direct half `LayerNorm`, `dequant_gemm_epilogue` 及 packed FP4 / NVFP4 变体. 这些 pattern 不是同等成熟度: `attention` 和 direct half `linear` 已有受限 CUDA fp16 kernel 入口, `conv` 当前是 `torch.unfold` / im2col 加 TileLang half GEMM 的 lowering 路径而不是 fully fused conv, direct half `LayerNorm` 已接入 TileLang `reduce_sum` kernel 且限制为 last-dim fp16. CPU 路径只使用 PyTorch eager fallback.

## 4. 性能分析工具

本章节保留为兼容锚点. `XQT` 的 profiling 边界和工程约束已经迁到:

- [architecture/xqt.md](architecture/xqt.md)
- [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md)

如果你要更新 profiler 角色边界, artifact 记录方式或厂商工具映射, 优先更新这些 canonical 页面.

## 5. 场景表

本章节保留为兼容锚点. `XQT` 的场景状态, 能力矩阵和 readiness 理解, 优先在下列页面维护:

- [explanation/xqt-concepts.md](explanation/xqt-concepts.md)
- [usage/xqt-workflows.md](usage/xqt-workflows.md)

## 6. 文档分层

- 兼容总页: 当前文件,保留完整正文和旧链接锚点.
- [index.md](index.md): Markdown 总入口.
- [architecture/xqt.md](architecture/xqt.md): 架构层正文.
- [explanation/xqt-concepts.md](explanation/xqt-concepts.md): 说明层正文.
- [usage/xqt-workflows.md](usage/xqt-workflows.md): 使用层正文.
- [XQT_SUMMARY.md](XQT_SUMMARY.md): 兼容摘要入口.
- [../html/xqt.html](../html/xqt.html): 人类阅读页.
- [../../xqt/README.md](../../xqt/README.md): 包内短入口.
- [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md): 包内工程契约.

后续新增长期主题,优先直接落到三层目录,不要继续把新主题收拢回本文件.
