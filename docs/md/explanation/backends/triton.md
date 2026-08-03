# Triton Engine

本文介绍 XQT 中的 Triton operator optimization engine. Triton 在这里指 OpenAI Triton Python GPU kernel DSL, 不是 NVIDIA Triton Inference Server.

## Engine 介绍

Triton 是用 Python 编写自定义 GPU kernel 的 DSL 和 compiler. 它让开发者用接近 Python 的方式表达 block-level 并行程序, 编译成 GPU kernel, 常用于编写 fused activation, normalization, rope, GEMM 和 dequant kernel.

在 XQT 中, Triton engine 的定位是 "特定 pattern 的 kernel adapter", 不是整模型 runtime. 当前 adapter 提供 registry, eager reference fallback 和 pattern 执行入口.

## 适合的使用场景

- 编写和验证小而明确的 fused operator.
- 替代 PyTorch eager 中多个小 op 的组合, 减少 memory traffic 和 launch.
- 对 LLM 中的 activation, RMSNorm, RoPE, GEMM 变体做专项优化.
- 研究 fp8, int8, int4 dequant GEMM 等低精度算子.

## 不适合的场景

- 自动编译整个模型.
- 直接导出成移动端 runtime 产物.
- 不写 CUDA kernel 语义就期待自动优化所有 op.
- CPU 推理优化.

## 当前支持 pattern

源码位置: `xqt/operator_opt/backends/triton.py`

`TRITON_KERNEL_REGISTRY` 当前包含:

- `bias_gelu`
- `swiglu`
- `rmsnorm_residual`
- `rope`
- `gemm_fp16`
- `gemm_bf16`
- `gemm_int8`
- `gemm_fp8`
- `gemm_int4_dequant`

## 核心 API

- `TritonKernelSpec`: 单个 Triton kernel registry entry.
- `TRITON_KERNEL_REGISTRY`: pattern 到 spec 的映射.
- `get_triton_kernel_spec(pattern)`: 获取单个 pattern.
- `list_triton_kernel_specs()`: 返回 registry metadata.
- `run_triton_kernel(pattern, *args, fallback="eager", **kwargs)`: CUDA tensor 上执行 Triton kernel, 否则按策略 fallback.

## 简单例子

查看 pattern:

```python
from xqt.operator_opt.backends import list_triton_kernel_specs


print(list_triton_kernel_specs().keys())
```

运行 fused activation:

```python
import torch

from xqt.operator_opt.backends import run_triton_kernel


x = torch.randn(1024, device="cuda")
bias = torch.randn(1024, device="cuda")
out = run_triton_kernel("bias_gelu", x, bias)
```

CPU fallback:

```python
import torch

from xqt.operator_opt.backends import run_triton_kernel


x = torch.randn(1024)
bias = torch.randn(1024)
out = run_triton_kernel("bias_gelu", x, bias, fallback="eager")
```

## Engine 实现注意点

- Triton kernel 参数必须是 tensor, 当前 adapter 会检查输入类型.
- CUDA-only pattern 在 CPU 输入上只有 eager fallback 或报错.
- 对低精度 GEMM, metadata 需要记录 dtype, scale 语义, group size 和 accumulation dtype.
- benchmark 必须区分单 kernel latency, wrapper latency 和 whole operator stage latency.
- 对小 shape, launch overhead 可能盖过 kernel 优化收益.
- 对大 shape, memory bandwidth, occupancy, register pressure 和 shared memory 才是重点.

## 常见问题

- `Unsupported Triton pattern`: registry 没有对应 pattern.
- `requires CUDA tensors`: 输入 tensor 不在 CUDA.
- Triton 编译失败: block size, dtype, constexpr 或版本兼容问题.
- 输出误差: dtype, scale, mask, boundary condition 或 reference 对齐问题.
- benchmark 快但集成慢: 可能是 wrapper, layout conversion, allocation 或 fallback 造成.

## 官方资料

- Triton documentation: <https://triton-lang.org/main/>
- Triton language API: <https://triton-lang.org/main/python-api/triton.language.html>
- Triton tutorials: <https://triton-lang.org/main/getting-started/tutorials/index.html>
