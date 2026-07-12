# XQT Operator Engine 能力说明

本文对照当前源码, 说明 XQT **内部 operator engine** (kernel / lowering) 支持哪些 pattern, 成熟度如何, 以及 convert / operator stage 如何选 engine.

**词表与硬规则** (method ≠ engine, 禁止 quant backend=tilelang/svdquant 等) 以架构层为准:

- [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md)
- quant 侧写法: [xqt-quant.md](xqt-quant.md)
- Infer 交接: [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md)

**权威事实源是代码**, 不是本文. 环境会变, 请以 capability API 与 registry 为准.

## 负责什么

- 列出 `OPERATOR_OPT_ENGINES` 与 `list_operator_engine_capabilities()` 的对照表.
- 列出各 engine 注册的 kernel pattern (`list_*_kernel_specs()`).
- 说明 built-in materialize 实际覆盖到哪, 与 "registry 里有名字" 的区别.
- 说明 engine 与 quant 的衔接点 (不写 quant method 表).

## 不负责什么

- 不定义 quant method / backend 全表 (见 [xqt-quant.md](xqt-quant.md)).
- 不把 AWQ / GPTQ / SVD 写成 engine methods.
- 不定义新 schema, 不替代 `xqt/FRAMEWORK.md` 与架构边界文.
- 不把 planned / reference_guarded 写成生产性能承诺.
- 不展开厂商 profiler 或手写 CUDA 调优 (见 [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md)).
- 不把 TensorRT / ONNX Runtime 写成 engine (它们是 **export/deploy backend**).

## 术语 (engine 视角)

| 词 | 含义 | 代码锚点 |
| --- | --- | --- |
| `engine` | XQT 内部 kernel / lowering 选择 | `OPERATOR_OPT_ENGINES`; `operator_opt/capability.py` |
| `pattern` | engine registry 中的算子名 | 各 `*_KERNEL_REGISTRY` |
| `maturity` / `status` | 实现成熟度 / 接口可用性 | capability API |
| quant method | 算法身份 (awq/gptq/svd) | **不在本文主表**; 见 quant 文档 |
| quant backend | pytorch/torchao/... | **不是** engine 名 |

**重要**: package 可导入或 `available=True` **不等于** 本机 correctness / 性能验收通过.

## 如何从代码刷新本页信息

```python
from xqt.operator_opt.capability import list_operator_engine_capabilities
from xqt.operator_opt.backends.tilelang import list_tilelang_kernel_specs
from xqt.operator_opt.backends.triton import list_triton_kernel_specs
from xqt.operator_opt.backends.cutile import list_cutile_kernel_specs
from xqt.operator_opt.backends.cutlass import list_cutlass_kernel_specs
from xqt.operator_opt.backends.cute_dsl import list_cute_dsl_kernel_specs
from xqt.core.schema import OPERATOR_OPT_ENGINES

print(list_operator_engine_capabilities())
print(sorted(list_tilelang_kernel_specs()))
print(OPERATOR_OPT_ENGINES)
```

---

## 1. Engine 总表 (capability 静态基线 + 本机探测)

源码: `xqt/operator_opt/capability.py` `_BASE_CAPABILITIES` + `describe_operator_engine_capability()`.

schema 集合: `OPERATOR_OPT_ENGINES = torch_compile, deployment_engine, triton, tilelang, cutile, cutlass, cute_dsl, custom_cuda`.

| engine | status (基线) | maturity | requires_cuda | exportable | built-in 执行定位 (代码 notes/limitations) |
| --- | --- | --- | --- | --- | --- |
| `torch_compile` | available | executable | 否 | 否 | 整模 / 组件 `torch.compile`; 图 break 会降效 |
| `triton` | available | executable | 是 | 否 | **materialize 主路径**: RMSNorm + `xqt.nn.FeedForward` 组合; registry 里其它 pattern 尚无通用 wrapper |
| `tilelang` | available | executable | 是 | 否 | **materialize 主路径**: attention / conv / half linear / half norm / dequant·dense linear 族; 有 shape/dtype 限制 |
| `cutile` | planned | reference_guarded | 是 | 否 | 可 materialize reference-guarded linear/dequant wrapper; 非完整 CuTile codegen 验收 |
| `cute_dsl` | planned | reference_guarded | 是 | 否 | NVFP4 dense-cache 可走 reference-guarded dense GEMM epilogue; **不直接吃 packed NVFP4** |
| `cutlass` | planned | metadata_only | 是 | 否 | 主要记 metadata + reference fallback |
| `deployment_engine` | planned | metadata_only | 否 | 是 | 表示 TRT/ORT/OpenVINO 部署融合意图; **不改写** PyTorch runtime module |
| `custom_cuda` | planned | planned | 是 | 否 | 预留 nvcc 扩展; 当前注册 custom op 示例 `bias_gelu`, 默认不编译加载 |

`available` 字段由环境探测填充 (如 `torch.compile` 是否存在, `triton` / `tilelang` / `cutlass` 包是否可导入等). TileLang 在 package 不可用时仍可能 `available=True`, 但 notes 会写明只剩 reference fallback.

---

## 2. Kernel pattern 注册表 (对照 backends)

### 2.1 TileLang (`xqt/operator_opt/backends/tilelang.py`)

`list_tilelang_kernel_specs()` 当前 keys:

| pattern | 说明 (结合 capability limitations) |
| --- | --- |
| `attention` | fp16 attention; `dropout_p=0`, `seq_kv >= seq_q` 等限制 |
| `conv` | 非 fully fused conv; 常见为 im2col / unfold + GEMM 路径, 覆盖受限 |
| `conv3d_1x1x1` | 1x1x1 特例 |
| `linear` | direct half linear |
| `linear_marlin` | Marlin 风格 packed 路径 (显式 pattern) |
| `norm` | direct half LayerNorm 类, last-dim 等限制 |
| `dense_linear_epilogue` | dense linear + epilogue |
| `dequant_gemm_epilogue` | dequant + GEMM epilogue |
| `fp4_packed_dequant_gemm_epilogue` | packed FP4 dequant GEMM |
| `nvfp4_packed_dequant_gemm_epilogue` | packed NVFP4 dequant GEMM |
| `int8_mma` | INT8 MMA 相关 pattern |

kernel 实现目录: `xqt/operator_opt/kernels/tilelang/` (`attention.py`, `conv.py`, `linear.py`, `linear_marlin.py`, `norm.py`, `gemm.py`, `int8_mma.py`, ...).

wrapper: `xqt/operator_opt/wrappers/` (`attention.py`, `xqt_attention.py`, `linear.py`, `conv.py`, `conv3d.py`, `norm.py`, `dequant_gemm.py`, ...).

### 2.2 Triton (`xqt/operator_opt/backends/triton.py`)

`list_triton_kernel_specs()` 当前 keys:

| pattern | 类别 |
| --- | --- |
| `rmsnorm`, `rmsnorm_channel_first`, `rmsnorm_residual` | Norm |
| `bias_gelu`, `geglu`, `swiglu` | Pointwise / activation |
| `rope` | 位置编码 |
| `gemm_fp16`, `gemm_bf16`, `gemm_fp8`, `gemm_int8`, `gemm_int4_dequant` | GEMM 精度族 |

kernel 目录: `xqt/operator_opt/kernels/triton/` (`gemm.py`, `linear.py`, `pointwise.py`, `mxfp_gemm.py`, ...).

**materialize 缺口** (capability limitations 原文语义): 除 RMSNorm 与 `FeedForward` 组合外, 其它 registered pattern **尚未** 都有通用 operator wrapper.

### 2.3 CuTile (`xqt/operator_opt/backends/cutile.py`)

patterns: `attention`, `bias_silu`, `conv`, `dense_linear_epilogue`, `dequant_gemm_epilogue`, `fp4_packed_dequant_gemm_epilogue`, `linear`, `norm`, `nvfp4_packed_dequant_gemm_epilogue`.

定位: metadata-first / reference-guarded, 与 TileLang pattern 对齐的 catalog, **不是** 同等 executable 深度.

### 2.4 CUTLASS / CuTe DSL

| engine | patterns |
| --- | --- |
| `cutlass` | `gemm_epilogue`, `grouped_gemm` |
| `cute_dsl` | `gemm_epilogue`, `grouped_gemm` |

### 2.5 torch_compile / deployment_engine / custom_cuda

- `torch_compile`: 无 pattern registry; 对当前 module 调 `torch.compile`.
- `deployment_engine`: 无 PyTorch kernel registry; 能力占位.
- `custom_cuda`: `cuda_extension` 描述; 示例 op `bias_gelu`.

---

## 3. Engine 与 Quant 的衔接 (规则入口)

**硬规则不在本文展开**, 见:

1. [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md) - 词表, 三轴, 禁止清单
2. [xqt-quant.md](xqt-quant.md) - quant backend / method / strategy 现状表
3. [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md) - model + compute_config

Operator engine **只** 负责: 算子实现, 融合, pattern materialize, MMA / dequant-gemm 等 **计算路径**.

```text
quant method (awq/gptq/svd/...)  --backend=pytorch-->  storage in model
                                                      (+ optional compute_config)
operator engine (tilelang/triton/...)  --resolve(capabilities)-->  kernels
```

| 不要在 engine 文档写 | 应写在 |
| --- | --- |
| `tilelang methods = awq, gptq` | quant method 表 ([xqt-quant.md](xqt-quant.md)) |
| `backend=tilelang` 做 quant | 禁止; 用 `engine=tilelang` 做 operator |
| `backend=svdquant` | 禁止; 用 `method=svd` |

capability resolve: `xqt.runtime.engine_resolve` (按 `required_capabilities`, 可选 preferred hint).  
convert 的 `engine=` 是 materialize **preference**, 不是 quant 输出主键.

---

## 4. convert / nn facade 与 engine

| 入口 | 源码 | 当前行为 |
| --- | --- | --- |
| `xqt.convert(..., engine=...)` | `xqt/conversion.py`, `conversion_impl/` | 建 `OperatorContract` 并可能 **eager materialize**; `EngineKind` 子集为 `torch/triton/tilelang/cutile/cute_dsl` |
| `xqt.nn.*` | `xqt/nn/` | facade 构造带 engine intent; 导出 `Linear`, `Conv2d`, `LayerNorm`, `FeedForward`, `RMSNorm`, `Attention`, `TransformerBlock` |
| operator stage | `xqt/operator_opt/execute.py`, `materialize.py` | workflow 内正式 plan + capability + fallback report |

`Attention` / `TransformerBlock` 当前 engine 以 `torch` / `tilelang` 为主; **不是** 完整 block-level 生产 megakernel.

---

## 5. 源码地图

```text
xqt/operator_opt/
  capability.py          # engine 矩阵
  execute.py / materialize.py / plan.py
  backends/              # tilelang, triton, cutile, cutlass, cute_dsl, gemm_*
  kernels/               # 各 engine 实现与公共 pattern
  wrappers/              # 模块级 wrapper
  triton_wrappers.py / tilelang_wrappers.py / reference_wrappers.py

xqt/core/schema.py       # OPERATOR_OPT_ENGINES, quant strategies
xqt/quant/capability.py  # quant backend + nature
xqt/quant/quantizers/    # 算法实现
xqt/conversion.py        # convert facade
xqt/nn/                  # semantic facade
```

---

## 6. 继续阅读

- 边界规则: [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md)
- Quant 说明: [xqt-quant.md](xqt-quant.md)
- Infer 交接: [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md)
- 分册背景: [backends/index.md](backends/index.md) (`tilelang.md`, `triton.md`, ...)
- 调优方法论: [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md)
- 推理与模型包: [xqt-inference.md](xqt-inference.md)
- 概念: [xqt-concepts.md](xqt-concepts.md)
- 架构: [../architecture/xqt.md](../architecture/xqt.md)
- 设计债: [../architecture/xqt-design-debt.md](../architecture/xqt-design-debt.md)
