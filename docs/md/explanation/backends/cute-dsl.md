# CuTe DSL Engine

本文介绍 XQT 中的 CuTe DSL engine. CuTe DSL 是 NVIDIA CUTLASS 生态中的 Python DSL 路线, 用于表达 tile-level tensor 程序和 GEMM 类 kernel.

## Engine 介绍

CuTe 是 CUTLASS 中的 tensor layout 和 tiling 抽象. CuTe DSL 把这些思想暴露到 Python DSL 中, 让开发者用更高层的 Python 表达 tile, layout, copy, MMA 和 epilogue 等 GPU kernel 结构.

XQT 当前的 CuTe DSL adapter 只覆盖少量 GEMM pattern, 主要用于:

- registry 和 metadata 记录.
- reference-guarded dense GEMM epilogue fallback.
- 为后续真实 CuTe DSL kernel 编译和 benchmark 留出结构.

## 适合的使用场景

- GEMM epilogue 和 grouped GEMM 的 DSL 原型.
- 研究 tile shape, cluster shape 和 target arch 对 kernel 的影响.
- 在 CUTLASS / CuTe DSL 生态里探索更细粒度的 kernel 控制.
- 为后续 custom CUDA engine 记录稳定 metadata.
- 为 FLUX.2 klein NVFP4 这类 packed Linear 模块的一次性 dense-cache bridge 提供可调用推理 wrapper.

## 不适合的场景

- 完整模型部署.
- 非 NVIDIA GPU.
- 期望当前 XQT adapter 自动生成完整生产 CUDA binary.
- 不理解 tile / layout / MMA 的情况下盲目替换 GEMM.

## 当前支持 pattern

源码位置: `xqt/operator_opt/backends/cute_dsl.py`

`CUTE_DSL_KERNEL_REGISTRY` 当前包含:

- `gemm_epilogue`
- `grouped_gemm`

## 核心 API

- `CuteDSLKernelSpec`: 单个 CuTe DSL kernel registry entry.
- `CuteDSLCompileSettings`: target_arch, cache_dir, tile_shape, cluster_shape, pass_configs.
- `CUTE_DSL_KERNEL_REGISTRY`: pattern 到 spec 的映射.
- `cute_dsl_version()`: 尝试从 `cutlass.cute` 或 `cutlass` 获取版本.
- `get_cute_dsl_kernel_spec(pattern)`: 获取 pattern spec.
- `list_cute_dsl_kernel_specs()`: 列出 registry metadata.
- `run_cute_dsl_kernel(pattern, *args, fallback="eager", **kwargs)`: 执行 kernel 或 fallback.
- `build_cute_dsl_artifact_metadata(pattern, settings=None)`: 构造 metadata.

## 简单例子

```python
from xqt.operator_opt.backends import (
    CuteDSLCompileSettings,
    build_cute_dsl_artifact_metadata,
)


metadata = build_cute_dsl_artifact_metadata(
    "gemm_epilogue",
    CuteDSLCompileSettings(
        target_arch="sm_90a",
        cache_dir="artifacts/cute_dsl",
        tile_shape=(128, 128, 64),
        cluster_shape=(1, 2, 1),
    ),
)
```

fallback 执行:

```python
import torch

from xqt.operator_opt.backends import run_cute_dsl_kernel


a = torch.randn(128, 256)
b = torch.randn(256, 128)
bias = torch.randn(128)
out = run_cute_dsl_kernel("gemm_epilogue", a, b, bias, fallback="eager")
```

## Engine 实现注意点

- 当前 adapter 的 artifact 多为 metadata, 不是完整 binary.
- CuTe DSL 和 CUTLASS 概念相近, 但文档中要区分 Python DSL prototype 与 C++ template library.
- target arch 对可用指令和性能影响很大, 尤其是 Hopper / Blackwell 相关 MMA.
- cluster shape, tile shape 和 epilogue 语义要写入 metadata.
- 当前 CuTe DSL wrapper 消费 dense weight cache, 不直接消费 packed NVFP4 权重.
- 如果接入真实编译, 要记录 generated source, build config, compile latency 和 runtime latency.

## 常见问题

- `Unsupported CuTe DSL pattern`: registry 没有 pattern.
- `requires CUDA tensors`: CUDA-only pattern 输入不在 CUDA.
- 版本获取为 `None`: 当前环境不可导入 `cutlass.cute`.
- 性能和 CUTLASS 路径不同: DSL lowering, tile config, epilogue 和 compiler 版本都可能不同.

## 官方资料

- CUTLASS GitHub: <https://github.com/NVIDIA/cutlass>
- CUTLASS documentation: <https://docs.nvidia.com/cutlass/>
- CuTe documentation in CUTLASS: <https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html>
