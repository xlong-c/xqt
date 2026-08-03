# 推理后端基础介绍

本文是一份部署和推理后端的总览介绍. 它只解释常见后端本身是什么, 适合什么场景, 典型产物是什么, 使用时容易卡在哪里. 本文不定义 `XQT` 的 adapter 设计, recipe schema, workflow 约束或能力承诺.

如果要按 XQT 当前后端和 engine 逐个阅读, 优先看 [backends/index.md](backends/index.md). 该目录把 `torch_export`, ONNX, ONNX Runtime, TensorRT, OpenVINO, ExecuTorch, ncnn, MNN 归为导出 / runtime backend, 把 TileLang, Triton, CUTLASS, CuTe DSL 和 CuTile 归为 XQT 内部 kernel engine.

## 读这份文档前先分清三类东西

部署链路里经常把 "格式", "runtime", "compiler", "backend" 混着说. 写后端代码前先拆开:

- 模型表示: 描述模型计算图和权重的格式, 例如 `torch.export` 的 `ExportedProgram`, TorchScript, ONNX, OpenVINO IR, ncnn param/bin, MNN 模型文件, ExecuTorch `.pte`.
- 推理 runtime: 真正加载模型, 分配内存, 调用 kernel 并返回输出的执行环境, 例如 ONNX Runtime, OpenVINO Runtime, TensorRT Runtime, ncnn, MNN, ExecuTorch Runtime.
- 编译或 lowering 工具: 把上游模型变成目标 runtime 可执行产物的转换器, 例如 PyTorch exporter, ONNX exporter, TensorRT builder, OpenVINO conversion pipeline, pnnx, MNNConvert, TVM compiler.
- 硬件执行后端: runtime 内部面向具体硬件的执行路径, 例如 ONNX Runtime 的 CUDA / TensorRT / OpenVINO Execution Provider, ExecuTorch delegate, OpenVINO CPU / GPU / NPU plugin, ncnn Vulkan backend.

一个实际部署路径通常长这样:

```text
PyTorch model
  -> graph capture / export
  -> intermediate representation
  -> backend-specific conversion or build
  -> runtime artifact
  -> runtime session
  -> benchmark / accuracy check / profiling
```

写后端代码时最重要的问题不是 "能不能导出一个文件", 而是:

- 目标 runtime 要什么输入格式.
- 它是在运行时 JIT 优化, 还是提前 build 成专用 engine.
- 动态 shape 是一等能力, 受限能力, 还是需要显式 profile.
- 不支持的算子是 fallback, 报错, 还是需要 custom op / plugin.
- 精度策略由谁决定: exporter, converter, runtime, builder, 还是硬件 plugin.
- 产物是否和硬件, driver, runtime 版本强绑定.

## 一眼对照

| 后端或格式 | 主要角色 | 常见输入 | 常见产物 | 典型硬件 | 适合场景 | 主要风险 |
| --- | --- | --- | --- | --- | --- | --- |
| `torch.export` | PyTorch 图捕获和 Export IR | PyTorch `nn.Module` | `ExportedProgram` | 不是 runtime 本身 | PyTorch 2 导出, 下游 compiler / edge 流程入口 | 图捕获失败, 动态 shape 约束, Python 语义不能完整保留 |
| TorchScript | PyTorch 可序列化图和 runtime | PyTorch script / trace | `.pt` / `.ts` | CPU / CUDA / mobile 等 | 旧 PyTorch C++ / mobile 部署, 保守兼容 | tracing 控制流问题, 新项目通常更偏向 `torch.export` 生态 |
| ONNX | 跨框架模型交换格式 | PyTorch / TensorFlow 等 exporter | `.onnx` | 由 runtime 决定 | 通用交换, 第三方 runtime 接入 | opset, shape, custom op, 数值差异 |
| ONNX Runtime | ONNX 推理 runtime | `.onnx` | session, optimized graph | CPU / CUDA / TensorRT / OpenVINO / DirectML / CoreML / QNN 等 | 通用服务端和端侧 ONNX 推理 | EP 覆盖不完整, provider 顺序, fallback 难察觉 |
| TensorRT | NVIDIA 推理优化 SDK | ONNX, TensorRT network 等 | serialized engine / plan | NVIDIA GPU | NVIDIA GPU 极致吞吐或低延迟 | engine 与硬件和版本绑定, build 慢, 动态 shape / plugin / precision 配置复杂 |
| OpenVINO | Intel 生态推理 runtime 和优化工具 | ONNX, PyTorch, TensorFlow, Paddle 等 | OpenVINO IR / compiled model | Intel CPU / GPU / NPU, 也支持 AUTO / HETERO 等设备策略 | Intel 平台性能便携, CPU 推理, 边缘设备 | 设备 plugin 差异, conversion 覆盖, latency / throughput 配置 |
| ExecuTorch | PyTorch Edge 端侧部署栈 | `torch.export` / PyTorch 2 export flow | `.pte` | mobile, embedded, microcontroller, DSP / NPU delegate | PyTorch 到移动端和嵌入式的一体化链路 | backend-specific lowering, delegate 覆盖, 每个硬件常要单独产物 |
| ncnn | 轻量端侧推理框架 | PyTorch / ONNX 经 pnnx 或转换工具 | `.param` + `.bin` | mobile CPU, Vulkan GPU, desktop, WebAssembly | Android / iOS / 桌面轻量部署, CV 模型 | 算子覆盖, layout 转换, 自定义 layer, 量化和 Vulkan 差异 |
| MNN | 轻量端侧和服务端推理框架 | ONNX / TensorFlow / TFLite / TorchScript 等 | `.mnn` | mobile CPU / GPU / NPU, server CPU / GPU | 移动端多后端推理, Alibaba 生态工具 | converter 覆盖, backend 差异, 文档和示例版本差异 |
| TVM | 机器学习编译器 | 多框架模型或 IR | target-specific module | CPU / GPU / accelerator | 研究, 定制硬件, 深度编译优化 | 工程复杂度高, tuning 成本, 不是简单文件转换器 |

## `torch.export`

`torch.export` 是 PyTorch 2 体系里的图捕获入口. 它的目标是从 Python `nn.Module` 中导出一个可分析, 可变换的完整图表示, 即 `ExportedProgram`. 它不是独立推理 runtime, 更像是 PyTorch 到下游部署栈的稳定中间层.

它和传统 eager PyTorch 的关键区别是: eager 模型依赖 Python 运行时逐行执行, 而 `torch.export` 尝试提前捕获计算图. 捕获成功后, 下游工具可以看到算子, shape 约束, 参数和缓冲区, 进而做 lowering, partition, AOT 编译或端侧打包.

适合用它理解这些问题:

- PyTorch 模型能否离开 Python 运行时.
- 模型里是否有 data-dependent control flow, Python side effect, 动态容器或未支持算子.
- 动态 shape 是否能用显式约束表达.
- 下游后端希望消费的是 ATen 图, Edge dialect, 还是继续 lowering 到其他 IR.

常见卡点:

- 图捕获失败. 如果 forward 里有无法追踪或无法符号化的 Python 行为, export 会报错.
- 动态 shape 不是随便动态. 需要明确哪些维度动态, 范围是什么, 维度之间有没有关系.
- export 成功不等于某个 runtime 可跑. 后续还需要 backend lowering 和算子覆盖.
- 模型输出结构, dataclass, dict, list 等 Python 容器可能需要规范化, 否则下游难处理.

写后端相关代码时, 可以把 `torch.export` 看成 "PyTorch 语义收敛成图" 的步骤, 不是最终部署后端.

## TorchScript

TorchScript 是 PyTorch 较早的序列化和部署路线. 它通过 `torch.jit.script` 或 `torch.jit.trace` 把 PyTorch 模型变成可保存, 可加载, 可优化的 TorchScript 程序. 典型产物是 `.pt` 文件, 可以在 Python 外的 LibTorch / C++ 环境中加载.

两种入口的差异很关键:

- tracing 记录样例输入跑过的算子路径. 它简单, 但对数据相关控制流不可靠.
- scripting 分析 Python 子集, 能保留一部分控制流, 但对 Python 语法和类型有约束.

TorchScript 适合已有 PyTorch C++ 部署, 老 mobile 链路, 或者需要保持较少外部转换依赖的场景. 它的优势是和 PyTorch 生态贴得近, 不需要先转换成 ONNX. 但在 PyTorch 2 之后, 新的图捕获和部署能力更多围绕 `torch.export`, AOTInductor 和 ExecuTorch 展开. 因此新后端设计通常不应把 TorchScript 当成唯一未来路径.

常见卡点:

- trace 对控制流和 shape 分支可能录错.
- script 对 Python 代码风格有要求.
- 自定义 op 和第三方模块可能需要额外注册.
- 模型能保存不代表跨版本长期稳定.

## ONNX

ONNX 是开放的模型交换格式. 它定义了一套计算图, 标准算子, 类型系统和文件格式, 目标是让模型可以在不同训练框架, runtime 和硬件工具链之间流转.

ONNX 的角色是 "中间表示", 不是单独 runtime. `.onnx` 文件需要交给 ONNX Runtime, TensorRT, OpenVINO, ncnn, MNN 或其他支持 ONNX 的工具执行或转换.

ONNX 适合:

- 把 PyTorch 模型交给非 PyTorch runtime.
- 做跨框架交换.
- 接入硬件厂商已经支持 ONNX 的编译器或推理引擎.
- 给部署系统一个相对通用的输入格式.

核心概念:

- opset: 算子语义版本. exporter 和 runtime 的 opset 支持必须对齐.
- graph input / output: 输入输出名字, dtype, shape 是后端适配的关键.
- initializer: 权重通常以 initializer 形式存在图里.
- custom op: 标准算子表达不了时使用, 但会降低可移植性.
- shape inference: 有助于后端优化, 但不一定能推断所有动态维度.

常见卡点:

- exporter 生成的图合法, 但目标 runtime 不支持某个 op.
- 同一个 op 在不同 opset 或 runtime 中行为细节不同.
- 动态 shape 导出后, 目标后端仍可能要求固定 shape 或 profile.
- Python 前处理和后处理不会自动进入 ONNX 图.
- 数值对齐需要单独验证, 尤其是 fused op, layout 转换, precision lowering 后.

## ONNX Runtime

ONNX Runtime 是生产级 ONNX 推理 runtime. 它加载 `.onnx` 模型, 做图优化, 分配内存, 调度算子 kernel, 并通过 Execution Provider 机制接入不同硬件加速库.

Execution Provider 是理解 ONNX Runtime 的核心. ORT 会根据可用 EP 把图分区, 尽量把子图交给更专用的 EP, 不支持的部分回退到默认 CPU EP 或其他 fallback. 常见 EP 包括 CPU, CUDA, TensorRT, OpenVINO, DirectML, CoreML, QNN 等.

适合 ONNX Runtime 的场景:

- 你已经有 `.onnx` 模型, 想要一个统一 runtime.
- 需要在 CPU, NVIDIA GPU, Windows DirectML, Intel OpenVINO, Apple CoreML, Qualcomm QNN 等环境之间切换.
- 希望先用 CPU EP 建立 correctness baseline, 再逐步启用加速 EP.
- 服务端或边缘设备需要成熟语言绑定和部署生态.

常见卡点:

- EP 顺序会影响图分区和性能.
- 加速 EP 不支持的节点可能 fallback, 导致性能不符合预期.
- fallback 有时是好事, 但 benchmark 时必须记录哪些节点真正跑在目标 EP 上.
- 不同 EP 的 provider options 差异很大.
- ORT 本身版本, CUDA / cuDNN / TensorRT / OpenVINO 版本组合会影响可用性.

写后端代码时, 不要只检查 `onnxruntime` 能 import. 更重要的是检查 session 实际启用了哪些 provider, 模型节点分配到了哪些 provider, 以及 fallback 是否被允许.

## TensorRT

TensorRT 是 NVIDIA 面向深度学习推理优化的 SDK. 它通常接收 ONNX 或 TensorRT network definition, 经过 builder 优化后生成 serialized engine, 再由 runtime 加载执行. 在 NVIDIA GPU 上追求低延迟和高吞吐时, TensorRT 是最常见的专用后端之一.

TensorRT 的思路不是简单解释执行模型, 而是针对目标 GPU, shape profile 和 precision 策略构建 engine. builder 会做 layer fusion, tactic selection, precision selection, memory planning 等优化. 这也是它强和复杂的来源.

典型概念:

- engine / plan: build 后的可执行产物. 通常和 TensorRT 版本, GPU 架构, driver / CUDA 环境强相关.
- builder: 负责把网络编译成 engine.
- runtime / execution context: 负责加载 engine 并执行推理.
- optimization profile: 动态 shape 的 min / opt / max 范围.
- tactic: 某个 layer 的具体实现策略.
- plugin: TensorRT 原生不支持的 op 或自定义融合实现.
- precision: FP32, FP16, BF16, TF32, INT8, FP8, FP4, INT4 等能力取决于 TensorRT 版本和硬件.

适合 TensorRT 的场景:

- 目标硬件明确是 NVIDIA GPU.
- 模型结构比较稳定, 可以提前 build.
- 对吞吐或延迟很敏感.
- 能接受针对具体 GPU 和 shape profile 生成专用产物.
- 有条件处理 plugin, calibration, profiler 和版本矩阵.

常见卡点:

- engine 不应被当成跨机器通用模型格式.
- build 时间可能很长, 尤其是大模型或 tactic 搜索范围大时.
- 动态 shape 需要 profile, profile 范围会影响性能和可用性.
- INT8 / FP8 等低精度收益依赖硬件, calibration / quantization 策略和算子覆盖.
- ONNX parse 成功不代表 engine build 成功.
- engine build 成功不代表数值误差可接受.
- plugin library 的加载顺序, ABI 和符号注册经常是部署问题来源.

写 TensorRT 后端时, 产物记录必须保留 TensorRT 版本, CUDA / driver, GPU 名称和 compute capability, builder 配置, precision, profile, plugin 列表, 输入输出 binding 信息.

## OpenVINO

OpenVINO 是 Intel 生态的模型优化和推理工具链. 它提供 conversion 工具, OpenVINO IR, runtime API 和设备 plugin, 目标是在 Intel CPU, GPU, NPU 等设备上提供性能便携的推理能力. OpenVINO Runtime 也支持直接加载多种模型格式, 但部署时常见做法仍是转换和缓存为 OpenVINO 更易优化的表示.

OpenVINO 的核心不是只服务某个单一设备, 而是通过统一 runtime API 和 plugin 把模型映射到不同设备. CPU 推理, 集成显卡, 独立 GPU, NPU, AUTO / HETERO 等设备选择都是它的重要使用场景.

典型概念:

- IR: OpenVINO 的中间表示, 通常包含 `.xml` 和 `.bin`, 新版本也支持更统一的序列化形式.
- `ov.Core`: runtime 入口, 用于读取模型, 编译模型, 查询设备.
- compiled model: 针对目标设备编译后的模型.
- infer request: 执行推理的请求对象, 支持同步和异步.
- performance hint: 用 `LATENCY` 或 `THROUGHPUT` 等高层 hint 让 runtime 选择配置.
- device plugin: CPU, GPU, NPU, AUTO, HETERO 等后端.

适合 OpenVINO 的场景:

- 目标平台是 Intel CPU / GPU / NPU.
- 需要在不同 Intel 设备之间保持同一套 API.
- CPU latency / throughput 优化是重点.
- 希望用 high-level performance hints, async inference, compiled model cache 等 runtime 能力.
- 服务端和边缘设备都要覆盖.

常见卡点:

- conversion 成功不代表所有设备 plugin 都支持.
- CPU, GPU, NPU 的支持算子, layout, precision 行为不同.
- 低精度和压缩能力可能依赖 NNCF, 模型类型和硬件.
- latency 模式和 throughput 模式的最优配置不同, benchmark 不能混用.
- AUTO / HETERO 能提高便携性, 但定位性能瓶颈时要确认每部分实际跑在哪个设备.

写 OpenVINO 后端时, 应记录 OpenVINO 版本, 目标 device, performance hint, precision, input shape, compiled model cache 状态和实际可用设备列表.

## ExecuTorch

ExecuTorch 是 PyTorch Edge 生态的端侧推理方案. 它面向 mobile, embedded, wearable, microcontroller 等设备, 目标是让 PyTorch 模型通过 PyTorch 2 export flow 准备, lowering, 打包成 `.pte`, 再由轻量 runtime 在设备上执行.

ExecuTorch 的关键特点是靠近 PyTorch, 但面向端侧约束. 它不是把完整 Python PyTorch 搬到手机或微控制器上, 而是把模型转换成适合端侧 runtime 的程序, 并通过 delegate / backend 把部分图交给硬件加速库.

典型概念:

- exported program: 通常来自 `torch.export`.
- edge dialect: 更适合端侧执行和 lowering 的图表示.
- `.pte`: ExecuTorch program 产物.
- runtime: 端侧加载和执行 `.pte` 的轻量运行时.
- delegate / backend: 把子图交给 XNNPACK, Core ML, Vulkan, Qualcomm, Arm 等目标后端.
- backend-specific lowering: 针对某个硬件生成专用 `.pte` 或分区.

适合 ExecuTorch 的场景:

- 模型源头是 PyTorch, 目标是 mobile 或嵌入式.
- 想尽量保持 PyTorch authoring experience.
- 需要在设备上离线推理, 减少云端依赖.
- 需要对接端侧 CPU / GPU / NPU / DSP 加速.
- 可以接受针对不同目标设备分别生成产物.

常见卡点:

- `torch.export` 成功只是第一步, delegate lowering 可能仍失败.
- 每个 delegate 支持的 op, dtype, layout 和 shape 都不同.
- 一个 `.pte` 不一定适合所有硬件后端.
- 端侧内存规划, binary size, thread, allocator, no-OS 环境等问题会比服务端更突出.
- Debug 需要同时理解 PyTorch export graph 和设备端 runtime 日志.

写 ExecuTorch 后端时, 要把 "通用 `.pte` 可执行" 和 "某个 delegate 真正加速" 分开记录.

## ncnn

ncnn 是 Tencent 开源的高性能神经网络推理框架, 重点优化 mobile, embedded 和 desktop 部署. 它强调轻量, 无第三方 runtime 依赖, 跨平台, 支持 CPU 和 Vulkan GPU 后端. 在 Android / iOS / 桌面轻量 CV 模型部署里很常见.

ncnn 的典型产物是:

- `.param`: 网络结构和 layer 参数.
- `.bin`: 权重二进制.

常见转换路径是 PyTorch / ONNX 经过 pnnx 或其他工具转换为 ncnn 格式. ncnn runtime 再加载 param/bin, 设置输入输出 blob, 执行 extractor.

适合 ncnn 的场景:

- 端侧二进制要小, 依赖要少.
- CV 模型, 图像处理模型, 传统 CNN / detector / segmenter 部署.
- Android / iOS / desktop / WebAssembly 等多平台轻量推理.
- Vulkan GPU 加速有价值, 但也需要 CPU fallback.

常见卡点:

- 模型转换过程里 layout, pixel format, mean / norm, resize 等前处理必须对齐.
- PyTorch 里的动态控制流和复杂 op 不一定能表达.
- pnnx / ONNX 转换覆盖取决于模型结构.
- 自定义 layer 需要写 ncnn layer 实现并注册.
- Vulkan 路径和 CPU 路径可能有算子覆盖和数值差异.
- int8 量化需要对应 calibration / quant table / runtime 支持.

写 ncnn 后端时, 不要只保存 param/bin. 还应记录输入 blob 名, 输出 blob 名, pixel format, layout, mean / norm, resize 策略, target platform, Vulkan 是否启用.

## MNN

MNN 是 Alibaba 开源的轻量深度学习框架, 主要面向端侧推理, 也覆盖服务端部分场景. 它提供模型转换工具和 runtime, 支持从 ONNX, TensorFlow, TFLite, TorchScript 等格式转换到 `.mnn` 产物, 再在移动端或其他平台加载执行.

MNN 的特点是覆盖面较广: CPU, GPU, NPU 等端侧执行路径, CV / Transformer / LLM 等模型类型, 以及转换, 量化, benchmark, demo app 等工具链. 但具体能力仍然取决于版本, backend 和模型结构.

典型概念:

- MNNConvert: 模型转换工具.
- `.mnn`: MNN 模型文件.
- Interpreter / Session: runtime 加载模型和创建执行会话的核心对象.
- backend: CPU, OpenCL, Vulkan, Metal, NNAPI, CoreML, CUDA 等不同执行路径, 具体以版本和平台为准.
- tensor layout: NCHW / NHWC / NC4HW4 等布局转换会影响性能和数值对齐.

适合 MNN 的场景:

- 移动端需要轻量推理 runtime.
- 需要在 Android / iOS 等设备上覆盖 CPU / GPU / NPU.
- 希望使用现成 conversion, quantization, benchmark 工具.
- Alibaba 生态或已有 MNN 部署经验的项目.

常见卡点:

- converter 支持和 runtime 支持不是同一件事.
- 不同 backend 的 layout 和 precision 可能不同.
- 文档, demo 和 master 分支能力可能随版本变化, 需要锁定版本验证.
- 模型前处理和后处理通常不在 `.mnn` 内, 需要应用侧对齐.
- LLM / diffusion 等新模型路径可能依赖额外工具或特定 build option.

写 MNN 后端时, 需要记录 MNN 版本, converter 参数, 输入输出 tensor 名, layout, backend type, precision mode, thread 配置和目标平台.

## TVM

TVM 是机器学习编译器框架. 它接收来自不同框架或 IR 的模型, 经过图级优化, tensor program lowering, schedule / tuning, codegen, 最终生成可部署模块. 它更像 "可编程编译系统", 而不是普通推理 runtime.

TVM 的优势是灵活: 可以面向 CPU, GPU, accelerator 和定制硬件生成代码, 可以写 schedule, 做 auto-tuning, 接入新 target. 代价是工程复杂度明显高于 TensorRT / OpenVINO / ncnn 这类更产品化的 runtime.

典型概念:

- Relay / Relax: 高层图 IR, 不同版本和路线有差异.
- TensorIR: 更底层的 tensor program 表示.
- target: 编译目标, 例如 LLVM CPU, CUDA, ROCm, Metal, Vulkan, WebGPU 或自定义 target.
- schedule / tuning: 搜索或指定 kernel 实现策略.
- runtime module: 编译后的可加载模块.

适合 TVM 的场景:

- 研究编译优化或新硬件后端.
- 需要深度控制 lowering 和 kernel schedule.
- 目标硬件不是主流 runtime 已经优化好的路径.
- 想比较不同 compiler strategy 的性能.

常见卡点:

- 从模型导入到可部署模块的链路长, 每层都可能失败.
- tuning 成本高, benchmark 方法必须严谨.
- 版本演进会影响 IR, API 和教程.
- 对一般应用部署来说, TVM 往往不是最省事的选择.

写 TVM 后端时, 要把它当 compiler pipeline 处理, 而不是简单 "convert model to file". 需要记录 target, pass context, tuning log, generated module, runtime 参数和 benchmark 环境.

## 后端选择时的思维方式

如果目标是快速跑通:

- PyTorch 内部验证优先 eager / `torch.compile` / `torch.export`.
- 通用交换优先 ONNX.
- 通用 ONNX 推理优先 ONNX Runtime CPU EP 建 baseline.
- NVIDIA GPU 性能优先 TensorRT 或 ONNX Runtime TensorRT EP.
- Intel CPU / GPU / NPU 优先 OpenVINO.
- PyTorch 到移动端优先 ExecuTorch.
- 轻量移动 CV 优先 ncnn 或 MNN.
- 定制硬件或编译研究再考虑 TVM.

如果目标是写后端 adapter, 先问这几个问题:

- 输入是什么: PyTorch module, ExportedProgram, ONNX, TorchScript, 还是已有后端产物.
- 输出是什么: 可执行 session, engine 文件, IR 文件, param/bin, `.mnn`, `.pte`, 还是 report.
- 后端需要哪些环境依赖: Python package, C++ library, shared library, driver, SDK, plugin.
- 产物是否 portable: 能否跨机器, 跨 GPU, 跨 runtime 版本.
- shape 如何处理: fixed, dynamic, symbolic, profile, batch-only dynamic.
- precision 如何处理: default FP32, AMP, FP16, INT8 calibration, weight-only, FP8.
- fallback 如何处理: 禁止 fallback, 允许 fallback, 记录 fallback, 还是拆分图.
- 怎么验证: smoke inference, output diff, latency, memory, profiler, backend capability report.

## 常见误区

- "支持 ONNX" 不等于支持所有 ONNX 模型. opset, custom op, dynamic shape 和 dtype 都会影响结果.
- "GPU backend 可用" 不等于整图都跑在 GPU. 可能只有部分子图被加速.
- "导出成功" 不等于 "部署成功". runtime 加载, 首次编译, 输入输出绑定和数值验证都是独立步骤.
- "低精度可选" 不等于 "低精度有收益". 硬件, kernel, calibration 和带宽瓶颈都会改变结论.
- "同一个模型文件" 不一定适合所有设备. TensorRT engine, ExecuTorch delegate 产物, OpenVINO compiled cache 都可能强绑定设备或版本.
- "benchmark 数字" 不能脱离 warmup, repeat, batch size, input shape, thread, precision, provider 和 device.
- "后端 runtime" 不负责训练语义. 推理后端一般只关心 forward graph, 权重, 输入输出和执行配置.

## 写后端代码时建议记录的元数据

无论目标后端是哪一个, report 或 manifest 至少应该能回答:

- backend 名称和版本.
- runtime / SDK / driver / compiler 版本.
- target device 名称和硬件能力.
- 输入模型来源和 checksum.
- 输出产物路径和 checksum.
- 输入输出 tensor 名称, dtype, shape.
- precision / quantization 配置.
- dynamic shape / profile 配置.
- fallback / partition 信息.
- custom op / plugin / delegate 信息.
- benchmark 配置: warmup, repeat, batch size, shape, thread, provider, device.
- correctness 配置: 对比 baseline, tolerance, 最大误差, 平均误差, 失败样本.

这些信息不是为了文档好看, 而是为了让后端问题能复现. 部署问题最常见的根因就是 "产物和环境没记录清楚".

## 官方资料入口

- TensorRT documentation: <https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html>
- TensorRT architecture overview: <https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html>
- TensorRT support matrix: <https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html>
- OpenVINO documentation: <https://docs.openvino.ai/>
- OpenVINO Runtime inference guide: <https://docs.openvino.ai/2025/openvino-workflow/running-inference.html>
- ONNX home: <https://onnx.ai/>
- ONNX concepts: <https://onnx.ai/onnx/intro/concepts.html>
- ONNX Runtime documentation: <https://onnxruntime.ai/docs/>
- ONNX Runtime Execution Providers: <https://onnxruntime.ai/docs/execution-providers/>
- ExecuTorch documentation: <https://docs.pytorch.org/executorch/stable/index.html>
- ExecuTorch overview: <https://docs.pytorch.org/executorch/stable/intro-overview.html>
- ExecuTorch export and lowering: <https://docs.pytorch.org/executorch/stable/using-executorch-export.html>
- PyTorch `torch.export` user guide: <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html>
- PyTorch `torch.export` tutorial: <https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html>
- ncnn GitHub: <https://github.com/Tencent/ncnn>
- MNN GitHub: <https://github.com/alibaba/MNN>
- Apache TVM: <https://tvm.apache.org/>
- Apache TVM documentation: <https://tvm.apache.org/docs/>
