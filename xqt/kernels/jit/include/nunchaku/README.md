# nunchaku headers (vendored)

这些头文件是 XQT 自有 CUDA kernel 的编译期依赖, 从工作区学习目录
`learn/nunchaku/` (Nunchaku 内核阅读版) 复制而来.

## 为什么 vendor

`xqt/kernels/jit/csrc/quantization/` 下的 SVDQuant / ConvRot kernel 需要
Nunchaku 的 GEMM / epilogue / LoRA 头文件. 这些 kernel 属于 XQT 生产路径,
因此头文件必须随 XQT 一起分发, 不能依赖仓库外的学习目录.

## 使用方

- `xqt/kernels/ops/_impl/cute/svdq_w4a4_sm89.py`
- `xqt/kernels/ops/_impl/cute/svdq_w8a8_sm89.py`
- `xqt/kernels/ops/_impl/cute/convrot_w8a8_sm89.py`

三者通过 `xqt.kernels.jit.utils.compile.include_root() / "nunchaku"` 解析本目录,
并作为 `CompileSpec.include_dirs` 传入.

## 文件范围

只保留编译期 include 闭包 (12 个头文件). `learn/nunchaku/` 下的 `.cu` 源文件,
`setup.py`, 测试脚本和 bindings 不在此目录, 它们仍属于学习目录的独立实验.

## 更新方式

上游学习目录更新后, 手工同步这里的头文件, 并重跑
`xqt/kernels/ops/_impl/cute/` 三个 kernel 的构建验证.