# CuTile 后端

本文介绍 XQT 中的 CuTile backend. CuTile 在 XQT 中被当作 Python DSL kernel 后端, 更接近 TileLang / CuTe DSL, 而不是 nvcc 构建的传统 CUDA extension.

## 后端介绍

CuTile 是面向 CUDA kernel 的 Python DSL / tile programming 路线. XQT 当前对 CuTile 的处理很保守: adapter 在 import 时不编译 kernel, 只记录 capability, registry 和 artifact metadata, 并提供 eager fallback.

这意味着当前 CuTile 文档要以 XQT adapter 的真实状态为准, 不把它写成完整生产 backend.

## 适合的使用场景

- 小型 fused operator 原型.
- 用 Python DSL 试验 tile-level CUDA kernel.
- 在 XQT operator stage 中记录 CuTile backend metadata.
- 对 `bias_silu` 这类 activation fusion 做受限验证.

## 不适合的场景

- 完整模型部署.
- 大范围自动 graph lowering.
- CPU 优化.
- 未接入真实编译和 profiling 前做生产承诺.

## 当前支持 pattern

源码位置: `xqt/operator_opt/backends/cutile.py`

`CUTILE_KERNEL_REGISTRY` 当前包含:

- `bias_silu`

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
    "bias_silu",
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


x = torch.randn(1024)
bias = torch.randn(1024)
out = run_cutile_kernel("bias_silu", x, bias, fallback="eager")
```

## 后端实现注意点

- 当前 `compile_status` 应视为 `metadata_only`.
- 不要在文档或 report 中暗示已经生成稳定 binary artifact.
- 如果接入真实 CuTile 编译, 要补充 source path, compiler flags, cache key, compile latency 和 checksum.
- `bias_silu` 属于小 fused op, benchmark 要特别注意 launch overhead.
- fallback 结果只能说明 reference 正确, 不能说明 CuTile kernel 性能或数值已经验证.

## 常见问题

- `Unsupported CuTile pattern`: registry 中没有 pattern.
- `requires CUDA tensors`: CUDA-only pattern 输入不在 CUDA 且 fallback 不允许.
- 版本为 `None`: 环境中没有可导入的 `cutile`.
- benchmark 没收益: 小算子容易被 launch overhead 主导.

## 官方资料

CuTile 的公开文档和安装方式可能随上游变化. 当前 XQT 文档只承诺说明本仓库 adapter 的真实使用方式. 实现前应以目标机器安装的 CuTile 包或仓库文档为准.

