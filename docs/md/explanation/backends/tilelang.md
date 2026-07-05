# TileLang 后端

本文介绍 XQT 中的 TileLang operator optimization 后端. TileLang 在这里不是通用模型部署 runtime, 而是面向特定算子 pattern 的 Python DSL kernel 后端.

## 后端介绍

TileLang 是面向高性能 tensor kernel 编写和编译的 Python DSL. 它的定位接近 "用 Python 描述 tile-level GPU kernel, 再编译到目标设备". 对 XQT 来说, TileLang 适合承载受限但高价值的融合算子, 例如 attention, linear, norm, dequant GEMM epilogue 和 packed FP4 / NVFP4 GEMM.

XQT 当前 TileLang adapter 的重点不是在 import 时真正编译所有 kernel, 而是:

- 建立 pattern -> reference -> kernel 的 registry.
- 在 CUDA tensor 可用时执行 TileLang kernel.
- 在不满足条件时用 eager reference fallback.
- 产出 compile settings 和 artifact metadata.
- 给 dtype 设定默认 validation threshold.

## 适合的使用场景

- 对少数热点算子做 kernel-level 优化.
- 研究 quantized GEMM, dequant epilogue, packed FP4 / NVFP4 路径.
- 对 attention, linear, layer norm 等固定 pattern 做专项 benchmark.
- 在 XQT operator stage 中记录 backend artifact metadata 和 validation 结果.

## 不适合的场景

- 直接部署完整模型.
- 自动覆盖任意 PyTorch graph.
- CPU 推理性能优化.
- 未经 shape, dtype, device 限制检查就替换生产算子.

## 当前支持 pattern

源码位置: `xqt/operator_opt/backends/tilelang.py`

`TILELANG_KERNEL_REGISTRY` 当前包含:

- `attention`
- `conv`
- `dequant_gemm_epilogue`
- `dense_linear_epilogue`
- `linear`
- `linear_marlin`
- `norm`
- `fp4_packed_dequant_gemm_epilogue`
- `nvfp4_packed_dequant_gemm_epilogue`

注意: 这些 pattern 不是同等成熟度. 文档里看到 `pattern` 存在, 只表示 XQT registry 中有 adapter 入口, 不等于任意模型都能自动替换.

## 核心 API

- `TileLangKernelSpec`: 单个 TileLang kernel registry entry.
- `TileLangCompileSettings`: target, target_arch, cache_dir, threads, num_stages, pass_configs.
- `TILELANG_KERNEL_REGISTRY`: pattern 到 spec 的映射.
- `TILELANG_DTYPE_VALIDATION_THRESHOLDS`: dtype 到默认 atol / rtol 的映射.
- `get_tilelang_kernel_spec(pattern)`: 获取单个 pattern 的 spec.
- `list_tilelang_kernel_specs()`: 列出 registry metadata.
- `run_tilelang_kernel(pattern, *args, fallback="eager", **kwargs)`: 执行 TileLang kernel 或 eager fallback.
- `build_tilelang_artifact_metadata(pattern, settings=None)`: 构造 manifest 友好的 metadata.
- `tilelang_version()`: 返回当前安装的 TileLang 版本, 不可导入时返回 `None`.
- `tilelang_validation_thresholds(dtype)`: 获取默认数值阈值.
- `validate_tilelang_packed_fp4_fused_gemm(...)`: packed FP4 fused GEMM 校验入口.

## 简单例子

查看 registry:

```python
from xqt.operator_opt.backends import list_tilelang_kernel_specs


specs = list_tilelang_kernel_specs()
print(sorted(specs))
```

构造 metadata:

```python
from xqt.operator_opt.backends import (
    TileLangCompileSettings,
    build_tilelang_artifact_metadata,
)


metadata = build_tilelang_artifact_metadata(
    "linear",
    TileLangCompileSettings(
        target="cuda",
        target_arch="sm_90",
        cache_dir="artifacts/tilelang",
        threads=128,
        num_stages=2,
    ),
)
```

执行 kernel 或 fallback:

```python
import torch

from xqt.operator_opt.backends import run_tilelang_kernel


x = torch.randn(4, 16)
w = torch.randn(8, 16)
bias = torch.randn(8)

out = run_tilelang_kernel("linear", x, w, bias, fallback="eager")
```

## 后端实现注意点

- 大多数 pattern 要求 CUDA tensor; CPU 路径通常只是 eager reference fallback.
- dtype threshold 要随 dtype 放宽, FP4 / FP8 类路径不能沿用 FP32 阈值.
- `compile_status` 当前多为 `metadata_only`, 不要误写成已经有可复用二进制 artifact.
- 需要记录 target_arch, cache_dir, kernel metadata, validation threshold 和 fallback.
- Conv 当前在 XQT 中是受限 lowering 思路, 不应描述成通用 fully fused conv.
- packed FP4 / NVFP4 路径要单独记录 packing layout, scale, group size 和 dequant 语义.

## 常见问题

- `Unsupported TileLang pattern`: registry 中没有这个 pattern.
- `requires CUDA tensors`: pattern 要求 CUDA, 但输入在 CPU 且 fallback 不允许.
- 输出误差偏大: dtype threshold, scale, layout, packed weight 解码或 accumulation dtype 不一致.
- benchmark 不稳定: kernel launch overhead, warmup, CUDA graph, shape 和 cache 状态都会影响结果.

## 官方资料

- TileLang GitHub: <https://github.com/tile-ai/tilelang>
- TileLang documentation: <https://tilelang.com/>
