# CUTLASS Engine

本文介绍 XQT 中的 CUTLASS engine. 这里的 CUTLASS 指 NVIDIA CUDA Templates for Linear Algebra Subroutines 及其 Python 入口在 XQT 中的受限 adapter.

## Engine 介绍

CUTLASS 是 NVIDIA 提供的 CUDA C++ template library, 用于构建高性能 GEMM, convolution 和相关 linear algebra kernel. 它大量覆盖 NVIDIA GPU 上的 tile-level GEMM 实现策略, epilogue fusion, tensor core 指令和 architecture-specific 优化.

XQT 当前的 CUTLASS adapter 不是完整 C++ extension build system. 它把 CUTLASS 当成 operator engine, 为少数 GEMM pattern 建立 registry, reference fallback 和 metadata 记录.

## 适合的使用场景

- GEMM / grouped GEMM / GEMM epilogue 相关优化.
- 研究 tensor core tile shape, epilogue fusion, target arch 对性能的影响.
- 为 TensorRT plugin, custom CUDA extension 或独立 kernel 设计做前期验证.
- 需要比 Triton 更接近 NVIDIA GEMM 模板生态的 kernel 路线.

## 不适合的场景

- 直接加载完整模型并执行.
- 在 XQT 当前 adapter 中期待自动 nvcc 编译完整 CUTLASS 工程.
- CPU 或非 NVIDIA GPU engine.
- 没有明确 GEMM shape 和 dtype 的泛化优化.

## 当前支持 pattern

源码位置: `xqt/kernels/ops/_impl/engines/cutlass.py`

`CUTLASS_KERNEL_REGISTRY` 当前包含:

- `gemm_epilogue`
- `grouped_gemm`

## 核心 API

- `CutlassKernelSpec`: 单个 CUTLASS kernel registry entry.
- `CutlassCompileSettings`: target_arch, cache_dir, tile_shape, cluster_shape, pass_configs.
- `CUTLASS_KERNEL_REGISTRY`: pattern 到 spec 的映射.
- `cutlass_version()`: 返回可导入 CUTLASS Python 包版本.
- `get_cutlass_kernel_spec(pattern)`: 获取 pattern spec.
- `list_cutlass_kernel_specs()`: 列出 registry metadata.
- `run_cutlass_kernel(pattern, *args, fallback="eager", **kwargs)`: 执行 CUTLASS kernel 或 eager fallback.
- `build_cutlass_artifact_metadata(pattern, settings=None)`: 构造 metadata.

## 简单例子

```python
from xqt.kernels.wrappers import (
    CutlassCompileSettings,
    build_cutlass_artifact_metadata,
    list_cutlass_kernel_specs,
)


print(list_cutlass_kernel_specs().keys())

metadata = build_cutlass_artifact_metadata(
    "gemm_epilogue",
    CutlassCompileSettings(
        target_arch="sm_90",
        cache_dir="artifacts/cutlass",
        tile_shape=(128, 128, 64),
    ),
)
```

运行或 fallback:

```python
import torch

from xqt.kernels.wrappers import run_cutlass_kernel


a = torch.randn(128, 256)
b = torch.randn(256, 128)
bias = torch.randn(128)

out = run_cutlass_kernel("gemm_epilogue", a, b, bias, fallback="eager")
```

## Engine 实现注意点

- 当前 metadata 中 `compile_status` 是 `metadata_only`, 不代表已经生成可加载 CUDA artifact.
- `tile_shape` 和 `cluster_shape` 是性能核心参数, 必须进入 report.
- CUTLASS 路径应记录 target_arch, CUDA 版本, CUTLASS 版本和 dtype.
- 对 grouped GEMM 要额外记录 group 数, 每组 shape, stride 和 pointer layout.
- 如果后续接入真实编译, 要把 source, generated code, build command, compiler flags 和 checksum 写入 artifact metadata.

## 常见问题

- `Unsupported CUTLASS pattern`: registry 中没有该 pattern.
- `requires CUDA tensors`: 输入不是 CUDA tensor 且 fallback 不允许.
- 版本导入失败: Python 环境没有 CUTLASS 包或版本不匹配.
- 性能不符合预期: tile shape, arch, alignment, dtype, epilogue 和 memory layout 不合适.

## 官方资料

- CUTLASS GitHub: <https://github.com/NVIDIA/cutlass>
- CUTLASS documentation: <https://docs.nvidia.com/cutlass/>
