# XQT Kernel / Wrapper / NN 边界

本文定义 `XQT` 中 `kernel`, `wrapper/materialize`, `xqt.nn` 三层的职责边界. 它是 `XQT` 架构层事实源.

## 负责什么

- 定义 `kernel`, `wrapper/materialize`, `xqt.nn` 各自负责什么.
- 定义三层之间允许交换什么信息.
- 给 `xqt.convert(...)`, `operator` stage 和 `xqt.nn.*` facade 提供统一边界.

## 不负责什么

- 不列 quant backend / method 全表.
- 不替代各 engine 的 pattern 能力矩阵.
- 不把当前所有 facade 或 wrapper 的成熟度写成性能承诺.

## 一条总规则

`kernel` 和 `xqt.nn` 之间 **不直接建立语义耦合**.

必须经过中间层:

```text
xqt.nn facade / torch module
    -> contract + materialize decision
    -> wrapper / candidate module
    -> engine kernel or eager/reference fallback
```

也就是说:

- `kernel` 不应理解 `FeedForward`, `TransformerBlock`, `YOLO head`, `MoE router` 这类语义块.
- `xqt.nn` 不应承诺自己一定落到某个具体 kernel pattern.
- `wrapper/materialize` 才负责把模块语义翻译成可执行的 engine 调用.

## 三层定义

### 1. `kernel`

代码落点:

- `xqt/kernels/ops/` (公开 `ops.<group>` 薄封装 + GEMM 合约)
- `xqt/kernels/ops/_impl/` (pattern implementation / engine adapter)
- `xqt/kernels/jit/csrc/` (CUDA/C++ 源码)

`kernel` 只负责:

- 一个 pattern 级计算路径的 reference 和 engine 实现.
- 处理 tensor, shape, dtype, layout, stride, tile, epilogue 这类局部执行问题.
- 暴露 pattern metadata, capability 限制和 fallback 所需的最小信息.

`kernel` 不负责:

- 不持有 `nn.Module` 级语义.
- 不理解 stage, manifest, benchmark acceptance, report.
- 不决定 `engine=auto` 或 capability resolve.
- 不直接改写模型结构.
- 不直接声明自己等于 `Linear`, `Attention`, `FeedForward` 或 `TransformerBlock`.

允许输入:

- tensor 参数
- 少量算子级 flags, 例如 `causal`, `dropout_p`, `eps`, `bias`
- 与 pattern 强绑定的 metadata

不允许输入:

- 任意 Python 语义块配置
- workflow stage 对象
- quant recipe 或 task-level 语义

### 2. `wrapper / materialize`

代码落点:

- `xqt/kernels/wrappers/materialize.py`
- `xqt/kernels/wrappers/` (TileLang / Triton / reference wrappers)
- `xqt/kernels/wrappers/execute.py`
- `xqt/kernels/wrappers/bench/`

这一层是 **边界翻译层**, 也是 `kernel` 和 `xqt.nn` 的明确分割线.

`wrapper / materialize` 负责:

- 接收 `ModuleContract` + target plan, 决定是否 materialize.
- 把 `nn.Module` / facade 的权重, bias, 子模块和 runtime intent 翻译为具体 engine 调用.
- 处理 flatten / reshape / permute / cache / dense-cache / CUDA graph / eager fallback.
- 暴露执行 metadata, fallback reason, compile info.
- 在 operator stage 内把 candidate module 接回模型.

`wrapper / materialize` 不负责:

- 不定义新的 facade 语义.
- 不把 kernel pattern 直接暴露成 public API 主入口.
- 不替代 quantizer, export backend 或 runtime package.

判断标准:

- 只要代码开始关心 `nn.Module`, `state_dict`, 子模块替换, candidate module, fallback report, execution metadata, 它就不再是纯 `kernel`, 应该放在 `wrapper/materialize`.

### 3. `xqt.nn`

代码落点:

- `xqt/kernels/nn/` (`from xqt import nn` 是公开别名; 顶层 `xqt/nn/` 已删除)
- `xqt/kernels/nn/convert.py` (公开别名仍是 `xqt.convert`)
- `xqt/kernels/nn/fixtures/`

`xqt.nn` 是 **语义 facade 层**.

`xqt.nn` 负责:

- 用稳定的 Python 模块语义表达 `Linear`, `Conv2d`, `LayerNorm`, `RMSNorm`, `FeedForward`, `Attention`, `TransformerBlock` 等语义块.
- 尽量保留 `torch.nn.Module` / `state_dict` / eager forward 语义.
- 承载 `engine` 与 precision runtime intent.
- 在少数已经明确实现的路径上提供 facade 自带的 runtime fastpath 或 fallback 记录.

`xqt.nn` 不负责:

- 不直接等同于某个 engine registry pattern.
- 不直接承诺 block-level megakernel.
- 不替代 operator stage 的 materialize / benchmark / acceptance.
- 不把 facade 名字扩写成 quant method 或 export backend.

判断标准:

- 只要对象对外暴露为可复用 `nn.Module`, 并强调语义块身份, 它属于 `xqt.nn`, 即使内部暂时调用某个 engine fastpath.

## 三层之间允许交换的契约

### `xqt.nn` -> `wrapper/materialize`

允许:

- `ModuleContract` / `PrecisionPolicy` / `FusionIntent` (`xqt.kernels.precision`)
- `runtime intent` (`engine`, precision fields)
- 标准 `nn.Module` 参数和子模块

不允许:

- 让 facade 直接写死某个 kernel symbol 作为唯一合法执行路径

### `wrapper/materialize` -> `kernel`

允许:

- tensor
- shape / dtype / layout / stride 变换结果
- pattern 级 flags
- reference fallback 策略

不允许:

- 传递高阶模块语义对象
- 传递 workflow stage / manifest 对象

### `kernel` -> 上层

允许返回:

- tensor 结果
- pattern metadata
- compile / runtime metadata
- 显式异常, 由 wrapper 决定是否 fallback

不允许返回:

- 新的语义块定义
- 新的 public workflow schema

## 目录分工

```text
xqt/kernels/nn/
  semantic facade
  convert API
  smoke fixtures
  runtime intent
  eager / limited runtime path

xqt/kernels/wrappers/
  module-to-kernel adapters
  shape/layout translation
  execution metadata
  eager/reference fallback
  operator benchmark helpers

xqt/kernels/ops/
  public tensor op wrappers
  GEMM contracts / dispatch / reference
  pattern-level compute implementation in _impl/
  tensor-only API
  engine-specific reference/kernel pair
```

## 常见误区

- 不要把 `xqt.nn.FeedForward` 写成 "一个 Triton kernel".
- 不要把 `TileLang attention kernel` 写成 `Attention` facade 本身.
- 不要把 `wrapper` 当成临时胶水层而在文档里省略掉; 它就是 XQT 现在的正式边界层.
- 不要让 `kernel` 直接消费 `ModuleContract` 之外的高层 schema.
- 不要让 `xqt.nn` 直接承诺 "这个 facade 一定 materialize 成某个 pattern".

## 当前落点速记

- `xqt.nn.Linear` / `Conv2d` / `LayerNorm`: facade, 保留 PyTorch 模块语义.
- `xqt.nn.FeedForward` / `RMSNorm` / `Attention` / `TransformerBlock`: facade, 可带 runtime intent, 部分路径已有专用 runtime 或 materialize 接线.
- `materialize_module(...)`: facade / torch module 进入 operator candidate 的统一入口.
- `_TileLang*Wrapper`, `_Triton*Wrapper`, `_ReferenceGuarded*Wrapper`: 边界翻译层.
- `*_KERNEL_REGISTRY` 与 `xqt/kernels/ops/_impl/*`: pattern 级 kernel 实现层. `xqt/kernels/ops/_impl/*` 仍是旧路径兼容 shim; GEMM backend 只在 `xqt/kernels/ops/_impl/gemm_backends/`.

## 继续阅读

- [xqt.md](xqt.md)
- [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md)
- [../explanation/xqt-concepts.md](../explanation/xqt-concepts.md)
- [../explanation/xqt-engines.md](../explanation/xqt-engines.md)
- [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
