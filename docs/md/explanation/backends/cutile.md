# CuTile 后端

本文介绍 XQT 中的 CuTile backend. CuTile 在 XQT 中被当作 Python DSL kernel 后端, 更接近 TileLang / CuTe DSL, 而不是 nvcc 构建的传统 CUDA extension.

## 后端介绍

CuTile 是面向 CUDA kernel 的 Python DSL / tile programming 路线. XQT 当前对 CuTile 的处理很保守: adapter 在 import 时不编译 kernel, 记录 capability, registry 和 artifact metadata, 并提供 reference-guarded eager fallback.

这意味着当前 CuTile 文档要以 XQT adapter 的真实状态为准, 不把它写成完整生产 backend.

注意 `cutile_available()` 只表示当前环境可导入 `cuda.tile` 或 legacy `cutile` Python runtime. 这不等价于某个 XQT pattern 已经有真实 CuTile kernel. Executor 会继续查看 pattern metadata: 如果 `production_status` 仍是 `reference_guarded` 或 `metadata_only`, FLUX.2 klein NVFP4 这类 packed Linear 目标不会直连 packed path, 而会优先使用一次性反量化后的 dense-cache path.

## 适合的使用场景

- 小型 fused operator 原型.
- 用 Python DSL 试验 tile-level CUDA kernel.
- 在 XQT operator stage 中记录 CuTile backend metadata.
- 对 `linear`, `norm`, `attention`, `conv`, dequant GEMM 和 packed FP4 / NVFP4 路径做 reference-guarded smoke.
- 为 FLUX.2 klein NVFP4 这类 packed Linear 模块 materialize 可调用推理 wrapper.

## 不适合的场景

- 完整模型部署.
- 大范围自动 graph lowering.
- CPU 优化.
- 未接入真实编译和 profiling 前做生产承诺.

## 当前支持 pattern

源码位置: `xqt/operator_opt/backends/cutile.py`

`CUTILE_KERNEL_REGISTRY` 当前包含:

- `attention`
- `bias_silu`
- `conv`
- `dense_linear_epilogue`
- `dequant_gemm_epilogue`
- `fp4_packed_dequant_gemm_epilogue`
- `linear`
- `norm`
- `nvfp4_packed_dequant_gemm_epilogue`

## 核心 API

- `CuTileKernelSpec`: 单个 CuTile kernel registry entry.
- `CuTileCompileSettings`: target, target_arch, cache_dir, threads, pass_configs.
- `CUTILE_KERNEL_REGISTRY`: pattern 到 spec 的映射.
- `cutile_version()`: 返回可导入 CuTile 版本.
- `get_cutile_kernel_spec(pattern)`: 获取 pattern spec.
- `list_cutile_kernel_specs()`: 列出 registry metadata.
- `run_cutile_kernel(pattern, *args, fallback="eager", **kwargs)`: 执行 kernel 或 eager fallback.
- `build_cutile_artifact_metadata(pattern, settings=None)`: 构造 metadata.

## 简单例子

```python
from xqt.operator_opt.backends import (
    CuTileCompileSettings,
    build_cutile_artifact_metadata,
    list_cutile_kernel_specs,
)


print(list_cutile_kernel_specs())

metadata = build_cutile_artifact_metadata(
    "nvfp4_packed_dequant_gemm_epilogue",
    CuTileCompileSettings(
        target="cuda",
        target_arch="sm_90",
        cache_dir="artifacts/cutile",
        threads=128,
    ),
)
```

执行或 fallback:

```python
import torch

from xqt.operator_opt.backends import run_cutile_kernel


x = torch.randn(4, 16)
weight = torch.randn(8, 16)
bias = torch.randn(8)
out = run_cutile_kernel("linear", x, weight, bias, fallback="eager")
```

## 后端实现注意点

- 当前 `compile_status` 应视为 `metadata_only`.
- 不要在文档或 report 中暗示已经生成稳定 binary artifact.
- 如果接入真实 CuTile 编译, 要补充 source path, compiler flags, cache key, compile latency 和 checksum.
- `bias_silu` 属于小 fused op, benchmark 要特别注意 launch overhead.
- `linear` / dequant GEMM wrapper 当前用于推理可调用性和 metadata 串联, 不代表真实 CuTile kernel 性能已经验证.
- FLUX.2 klein NVFP4 的 `cutile` plan 可以声明 `nvfp4_packed_dequant_gemm_epilogue`, 但当前内置 packed NVFP4 spec 是 reference-guarded, materialized inference 会执行 `dense_linear_epilogue` + dense cache. 只有未来 spec metadata 明确标记真实 runtime kernel 时, executor 才会选择 packed NVFP4 path.
- Linear wrapper 支持 rank-1 及以上输入. 对 transformer 常见的 rank-3 `[batch, tokens, channels]` 输入, wrapper 会在内部 flatten 为 2D 调用 backend, 再恢复原 shape.
- fallback 结果只能说明 reference 正确, 不能说明 CuTile kernel 性能或数值已经验证.

## 常见问题

- `Unsupported CuTile pattern`: registry 中没有 pattern.
- `requires CUDA tensors`: CUDA-only pattern 输入不在 CUDA 且 fallback 不允许.
- 版本为 `None`: 环境中没有可导入的 `cutile`.
- benchmark 没收益: 小算子容易被 launch overhead 主导.

## 官方资料

CuTile 的公开文档和安装方式可能随上游变化. 当前 XQT 文档只承诺说明本仓库 adapter 的真实使用方式. 实现前应以目标机器安装的 CuTile 包或仓库文档为准.
