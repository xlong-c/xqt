# XQT 架构整改 TODO

> 来源: [xqt-architecture-review.md](xqt-architecture-review.md)
>
> 更新日期: 2026-08-23
>
> 当前架构理解和后续规范化建议: [xqt-architecture-understanding.md](xqt-architecture-understanding.md)

## 第一批: 删除与迁移收尾

- [x] 1. 完成 `xqt/gemm/` 向 `common/` 与 `backends/<arch>/` 的布局迁移,删除平铺重复实现,shim 与目录撞车.
- [x] 2. 删除无调用基础设施和占位实现,接通保留的 preflight 与 NVFP4 路径.
- [x] 3. 修复 `core/schema.py` 重复 `@dataclass`.

## 第二批: 解环与恢复层级

- [x] 4. core 去 XDL 化,本地化异常基类和检测配置,保证冷启动不加载 XDL.
- [x] 5. 将 StageSpec 与 workflow schema 下沉 `xqt/core/`,解除 pipeline/quant 对 workflows 类型的反向依赖.
- [x] 6. 拆开 quant storage artifact 与 runtime execution view,统一 contracts packing 实体并删除 quant bridge shim.
  - [x] ConvRot quant artifact 默认 reference forward;native fastpath 通过 `ConvRot*ExecutionView.from_storage()` 显式开启.
  - [x] 增加 `xqt.runtime.materialize_convrot_execution_views()` 批量完成 marker-based handoff,不让 runtime 反向 import quantizer.
- [x] 7. 顶层公开 API 全量懒加载,解除 quant 对 export 的导入依赖.

## 第三批: 收敛抽象与事实源

- [x] 8. 落地 quantizer 模板基类,收敛 policy 解析,模块替换,calibration 消费与 report 组装,并将 route handler 工厂化.
  - [x] `QuantizerTemplate` 与共享 policy,模块替换,calibration helper 已落地,原重复定义已删除.
  - [x] 13 个同构 route adapter 已由 `component_route_handler()` 工厂生成.
  - [x] 各模型侧量化算法统一通过 `build_component_quantization_report()` 组装公共报告字段.
- [x] 9. 将 strategy,scheme,nature 与 capability 收敛为 `xqt/contracts/quant_strategy.py` 单向派生的事实源.
- [x] 10. 通过 `stage_spec_to_config()` 自动派生 StageSpec 到 Config,删除手工逐字段转换.
- [x] 11. 将 engine capability,pattern 与 materialize 声明收敛到单一注册点,删除调用侧硬编码集合.
  - [x] `EngineRegistration` 统一承载状态,运行时,能力,pattern 可见性与 materializer.
  - [x] `operator_opt`/`core.schema`/pattern matcher/capability projection 改为查询注册表.
- [x] 12. 以 `xqt.gemm` 为唯一 GEMM 决策系统,统一 registry,dispatch,生产调用与 tuning cache.
  - [x] operator precision kernel 进入 `GemmKernelRegistry`,auto selector 只投影 registry 声明.
  - [x] `gemm_with_precision()` 通过 `dispatch_gemm()` 执行注册候选与 fallback.
  - [x] Triton evidence-backed schedule preset 归入 `gemm.common.tuning_cache`.

## 第四批: 边界与公共接口收敛

- [x] 13. 将真实模型专用代码迁入 `examples/`,统一 toy model/fixture 位置.
- [x] 14. 将零生产调用的 runtime API 明确降级为 reference/交互式封装,同步文档并处理无效 composite fusion.
- [x] 15. 为 export 定义统一 adapter Protocol 与 result 基类,合并过细的 TensorRT 文件粒度.
- [x] 16. 完成 contracts 混合边界收敛: typed payload,存储协议与 reference 语义归 contracts,后端执行归 runtime.

## 已通过检查

- [x] `tests/xqt/test_layer_import_boundaries.py` 层级守卫通过.
- [x] StageSpec/workflow,quant,runtime,export 针对性回归通过.
- [x] `python -m compileall` 通过.
- [x] `ruff check xqt examples/xqt_models tests/xqt` 通过.
- [x] `git diff --check` 通过.
- [x] 冷启动 `import xqt` 不加载 XDL.
- [x] 导入 `xqt.quant.quantizers` 不提前加载 `xqt.export`.

## 收尾要求

- [x] 完成结构性变更后运行 `index_repository`,项目名使用 `root-workspace-xdl` (`status=ready`,24,005 nodes,128,787 edges).
- [x] `pytest -q tests/xqt` 全量通过,文档标点检查通过.
