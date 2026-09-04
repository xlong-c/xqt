# XQT Kernels - 统一内核命名空间

本文定义 `xqt.kernels` 的系统边界, 分层模型和调用契约. 它是 `XQT` 架构层事实源之一, 承接 `sglang.kernels` (RFC #29630) 的设计, 适配 `XQT` 只关注模型本身的约束.

> 修改指导 (research): [GUIDE](../../research/xqt-quant-inference-architecture/GUIDE.md), [TODO](../../research/xqt-quant-inference-architecture/TODO.md).

## 负责什么

- 定义 `xqt.kernels` 的四层模型, 以及 `ops / wrappers / nn` 三个子包的目录形态.
- 定义 `spec / registry / selector / fused_op / ops/<group> / jit / aot` 各自职责和交互规则.
- 定义上层如何调用 `xqt.kernels`, 以及与 `xqt.kernels.nn / wrappers / Stage` 的分工边界.
- 定义新增内核, 接入外部高性能库 (如 `flashinfer`) 的唯一路径.

## 不负责什么

- 不重复 `xqt-nn / wrapper / nn boundary` 的三层语义划分.
- 不展开具体 `CUDA / Triton / TileLang / CuTe` 内核实现细节.
- 不替代 `xqt.kernels.engine_resolve / GEMM contract / operator capability` 的现有契约.
- 不承诺 `xqt.kernels` 已完成的性能收益, 未验证路径以 `metadata_only / planned` 标注.

## 一,为什么要统一

当前 `XQT` 存在三套并行的内核注册形态:

| 位置 | 注册形态 | 职责 |
| --- | --- | --- |
| `xqt/kernels/ops/gemm/registry.py` | `GemmKernelRegistration / GemmCapability / GemmKernelRegistry` | GEMM 能力矩阵与调度 |
| `xqt/kernels/engine_resolve.py` | `EngineRegistration / _ENGINE_REGISTRY` | 引擎级能力解析与 `default_order` |
| `xqt/kernels/ops/_impl/engines/*_KERNEL_REGISTRY` | `TRITON/TILELANG/CUTLASS/CUTILE/CUTE_DSL_KERNEL_REGISTRY` | 按 engine 维度镜像的 pattern 注册 |

新增一个内核或接入 `flashinfer` 需要改三处, 调用方不知道该从 `xqt.kernels.ops.gemm`, `xqt.kernels.ops._impl.engines.triton` 还是 `xqt.kernels.engine_resolve` 导入.

`sglang.kernels` 用 `spec + registry + selector + ops/<group>` 把 `sgl_kernel(AOT) + jit_kernel(JIT) + triton_ops(分散)` 收敛到单一可盘点的薄入口, `XQT` 照搬该模型, 形成 `xqt.kernels` 作为唯一公开内核入口.

迁移期间, 三套旧表仍由既有执行链读取, 但注册时会镜像到 `xqt.kernels.registry`. `GemmKernelRegistry`,`_ENGINE_REGISTRY` 和五张 `*_KERNEL_REGISTRY` 均标记为 deprecated. 兼容层只保存 target 字符串并按需解析, 不在导入阶段编译或加载可选扩展.

`xqt.kernels` 现在按三层收口: `ops` (tensor kernel + GEMM 合约), `wrappers` (materialize / operator / bench), `nn` (facade / convert / fixtures). Python pattern implementation,engine adapter 与 kernel guidance 位于 `xqt/kernels/ops/_impl/`; CUDA/C++ 源码位于 `xqt/kernels/jit/csrc/{gemm,quantization}`. `xqt/gemm/`, `xqt/conversion_impl/`, `xqt/operator_opt/`, `xqt/benchmark/`, `xqt/model/` 与顶层 `xqt/nn/` 已删除; GEMM 只在 `xqt/kernels/ops/gemm/` 与 `xqt/kernels/ops/_impl/gemm_backends/`, convert 实现只在 `xqt/kernels/nn/conversion/`, operator 实现只在 `xqt/kernels/wrappers/` 与 `xqt/kernels/ops/_impl/`, bench 只在 `xqt/kernels/wrappers/bench/`, smoke fixture 只在 `xqt/kernels/nn/fixtures/`. `from xqt import nn` 与 `xqt.convert` 仍是公开别名.

## 二,上层调用接口 (优先阅读)

### 2.1 导入规则

```text
允许:  from xqt.kernels.ops.gemm import gemm_fp16_triton
允许:  from xqt.kernels.ops.attention import fused_attention
允许:  from xqt.kernels import select_kernel, get_kernel, KernelBackend, KernelSpec
允许:  from xqt.kernels import BaseFusedOp  # 仅有可互换多后端的 op 需要
禁止:  直接 from xqt.kernels.ops._impl.triton.gemm import gemm_fp16_triton  (旧路径, 保留 shim 一版本)
禁止:  直接 from xqt.kernels.ops._impl.gemm_backends.sm89 import sm89_w8a8_executor  (应经 xqt.kernels.ops.gemm)
禁止:  直接 from xqt.kernels.ops._impl.engines.triton import TRITON_KERNEL_REGISTRY  (应经 xqt.kernels.registry)
```

实现仍可在 `xqt.kernels.jit` 或 `xqt.kernels.aot` 中开发, 但对外可调用路径必须经 `xqt.kernels.ops.*`.

### 2.2 调用路径总览

```text
上层 (三条主路径, 互不替代):

1. xqt.nn / xqt.convert()  ->  wrapper/materialize  ->  xqt.kernels.ops.*  ->  engine kernel
   (语义 facade, 推荐给模型替换与 convert 场景)

2. operator Stage            ->  materialize_module() ->  xqt.kernels.ops.*  ->  engine kernel
   (YAML workflow / Session.operator(), 走 benchmark + acceptance)

3. 直接调用                  ->  xqt.kernels.ops.<group>.<op>()              ->  engine kernel
   (单算子验证, benchmark 脚本, 单测, 显式外部库对比)

公共下层:

   xqt.kernels.ops.<group>.<op>()
        |
        +--> get_kernel(op_id, backend?)  /  BaseFusedOp.forward()
                 |
                 +--> registry  (KernelSpec 惰性 target 解析)
                 |
                 +--> spec      (KernelBackend x CapabilityRequirement 判定)
```

三条上层路径都收敛到同一个 `registry + selector`, 因此 `benchmark / report / trace` 可统一盘点.

### 2.3 五种典型调用

#### 1) 通过 `xqt.convert` (推荐, 语义块场景)

```python
import torch.nn as nn
from xqt import convert
from xqt.kernels.precision import PrecisionPolicy

model = nn.Sequential(nn.Linear(4096, 4096), nn.GELU())
result = convert(model, engine="tilelang", policy=PrecisionPolicy.from_roles(A="fp16", B="fp16"))
# 内部: convert -> _ModuleConverter -> materialize -> xqt.kernels.ops.gemm.gemm_fp16_tilelang
```

`engine` 是 materialize preference, 不是 quant backend. `convert` 内部经 `wrapper/materialize` 再调 `xqt.kernels.ops.*`.

#### 2) 通过 `XQTOptimizationSession` / YAML workflow

```python
from xqt import XQTOptimizationSession

sess = XQTOptimizationSession(model, artifact_dir="./artifacts")
sess.operator(engine="auto", targets=["linear"])  # 内部 materialize + benchmark + acceptance
sess.export(targets=[{"onnx": {"dynamo": True}}])
```

`operator` stage 内部经 `materialize_module()` 批量替换, 每个 candidate 最终调 `xqt.kernels.ops.*`.

#### 3) 直接调用 `xqt.kernels.ops.*` (单算子, 默认后端)

```python
from xqt.kernels.ops.gemm import gemm_fp16_triton
from xqt.kernels.ops.attention import fused_attention
from xqt.kernels.ops.layernorm import rmsnorm

out = gemm_fp16_triton(a, b, bias=bias)  # 默认走 registry 中该 op 在当前平台的固定路径
```

每个 `ops.<group>.<op>` 是薄封装, 签名与 `spec.FormatSignature` 一致, 内部 `return get_kernel("gemm.gemm_fp16", KernelBackend.TRITON)(...)`.

#### 4) 显式选择后端 (含外部高性能库)

```python
from xqt.kernels import select_kernel, KernelBackend

# 自研 vs 外部库 A/B
native = select_kernel("gemm.bmm_fp8", backend=KernelBackend.TRITON).load()
flash  = select_kernel("gemm.bmm_fp8", backend=KernelBackend.FLASHINFER).load()

out_native = native(a_fp8, b_fp8, scale_a, scale_b)
out_flash  = flash(a_fp8, b_fp8, scale_a, scale_b)
```

同一 `op_id` 可注册多个 `KernelBackend`, `selector` 按 `CapabilityRequirement` 硬过滤后决定固定路径, 剩余多选时必须显式 `backend=`. 新增 `flashinfer` 仅需在 `xqt/kernels/ops/gemm/__init__.py` 加一行 `register_kernel(KernelSpec(..., backend=FLASHINFER, target="flashinfer:bmm_fp8", capabilities={CUDA}))`.

#### 5) 可互换多后端算子 (`BaseFusedOp`, 仅部分 op)

```python
from xqt.kernels.ops.layernorm import _RMSNORM
from xqt.kernels import KernelBackend

# 自动择优 (priority + capability + shape gate)
out = _RMSNORM.forward(x, weight)

# 显式后端
out = _RMSNORM.forward(x, weight, backend=KernelBackend.JIT)

# 全局一键切回纯 torch 对照 (数值 bug 二分)
import os; os.environ["XQT_FORCE_KERNEL_BACKEND"] = "torch"
```

`BaseFusedOp` 仅用于 `activation / layernorm` 等签名完全一致且需运行时自动择优的 op, 其余 op 用 `select_kernel / get_kernel`.

### 2.4 调用契约摘要

| 调用方式 | 适用场景 | 是否经 wrapper | 是否可 `engine=auto` | 适用 op 数 |
| --- | --- | --- | --- | --- |
| `xqt.convert(...)` | 语义块替换, 保留 `nn.Module` 语义 | 是 | 是 | 语义块级 |
| `Session.operator() / YAML operator stage` | 批量替换 + benchmark + acceptance | 是 | 是 | 全量 |
| `xqt.kernels.ops.<group>.<op>()` | 单算子验证, 脚本, 单测 | 否 (薄封装) | 否 (固定路径) | ~90% |
| `select_kernel(op, backend).load()` | 显式后端, A/B 对比, 外部库 | 否 | 否 | 多后端 op |
| `BaseFusedOp.forward(backend=)` | 可互换多后端自动择优 | 否 | 是 (priority) | 少数 (首批 5+4) |

## 三,分层模型

自上而下四层, 与 `sglang.kernels` 一致:

```
1. Public API        xqt.kernels.ops.<group>.<op>()    # 唯一公开入口
2. Dispatch          select_kernel / get_kernel  (固定路径)
                     BaseFusedOp.forward       (可互换多后端自动择优, 按需)
3. Registry+Metadata registry.py + spec.py      # torch-free, 惰性, 可盘点
4. Backend x Device  KernelBackend(产地) x CapabilityRequirement(设备+SM窗口)  # 解耦
```

### 3.1 `spec.py` - 元数据契约

`spec.py` 必须 `torch-free`, `import xqt.kernels` 不触发内核编译或 `torch` 导入. 运行时和量化器中的新调用统一经过 `xqt.kernels.ops.<group>`; ops 内部可以惰性解析旧实现, 以便在不搬动 CUDA 源码的情况下完成切流.

核心类型:

- `KernelBackend(str, Enum)`: 产地, 只标实现从哪来, 不标设备. `TORCH / TORCH_COMPILE / TRITON / TILELANG / CUTILE / CUTLASS / CUTE_DSL / CUSTOM_CUDA / FLASHINFER / ...`
- `DeviceType(str, Enum)`: `CUDA / HIP / NPU / CPU` (XQT 初期 `CUDA` 为主, 其余占位).
- `PlatformInfo(msgspec.Struct)`: `device_type + cuda_arch_major/minor`, `detect()` 惰性探测, 失败回退 `CPU`.
- `CapabilityRequirement`: 单设备 + 可选 `min/max_cuda_arch` 窗口, `is_satisfied_by(PlatformInfo) -> bool`. 快捷量 `CapabilityRequirement.CUDA / .HIP / .NPU`, 构造器 `CapabilityRequirement.cuda(min_sm=(8,9))`.
- `KernelSpec`: `op("<group>.<name>") + backend + target("module:attr") + capabilities(frozenset, OR语义, 空集=任意) + format_signature + description`. `load()` 时才 `importlib.import_module` 并解析 `target`.
- `FormatSignature`: `supported_dtypes + in_place + description`, 宽松描述, 非强 schema.

关键约束: `KernelBackend` 与 `DeviceType` 解耦. 同一 `AOT / JIT / TRITON` 可同时支持 `CUDA+HIP`, 设备相关库 (`flashinfer->CUDA`) 用 `capabilities` 表达, 不新增 `CUDA_FLASHINFER` 这类烘焙命名.

### 3.2 `registry.py` - 进程级注册表

```python
registry = KernelRegistry()  # 进程单例

def register_kernel(spec: KernelSpec) -> KernelSpec:
    return registry.register(spec)
```

- `register(spec)`: 同一 `(op, backend)` 重复注册且 `spec` 完全相等则幂等, 否则抛 `ValueError` (避免 import 顺序决定实现).
- `get(op) -> List[KernelSpec]`, `get_backend(op, backend) -> KernelSpec`, `has(op)`, `ops() -> sorted`, `all_specs()`.
- 注册只记录 `target` 字符串, 不 `import torch` 也不触发 JIT 编译, 保证 `import xqt.kernels` 可在 CPU 环境用于盘点.

### 3.3 `selector.py` - 固定路径解析

```python
def select_kernel(op: str, backend: KernelBackend | None = None) -> KernelSpec: ...
def get_kernel(op: str, backend: KernelBackend | None = None) -> Callable: ...  # 缓存
```

规则 (无优先级启发式):

- 单后端注册: 直接返回该后端, 无需 `backend` 参数.
- 多后端注册: 按 `capabilities_satisfied(spec.capabilities, PlatformInfo.detect())` 硬过滤.
  - 剩 1 个: 返回该后端 (固定路径).
  - 剩 0 个: 抛 `ValueError` (当前平台无可用后端).
  - 剩多个: 必须显式 `backend=`, 否则抛 `ValueError`.

`get_kernel` 在 `select_kernel` 之上加 `lru_cache` 并调用 `spec.load()` 解析 `target` 为可调用对象. `ops.<group>` 的薄封装应调 `get_kernel`.

### 3.4 `fused_op.py` - 可互换多后端契约 (按需)

`BaseFusedOp` 是 `torch.nn.Module` 子类, 仅用于签名一致且需运行时自动择优的 op (首批 `activation x5 + layernorm x4`).

- 子类声明: `op = "layernorm.rmsnorm"`, `priority = (AOT, JIT, TRITON, TORCH)`, `capabilities: dict[KernelBackend, frozenset[CapabilityRequirement]]`, `format_signature`, `descriptions`.
- 子类实现: `forward_native` 必需 (纯 `torch` 对照), `forward_<backend>` 按需覆写 (`forward_triton / forward_jit / forward_aot / forward_cute_dsl / forward_flashinfer / ...`). 未覆写视为不可用.
- 平台分支 (第二维度, 与 `backend` 正交): `forward_cuda / forward_hip / forward_npu / forward_cpu` 及 `register_oot_forward`, 按 `xqt.srt.utils.is_cuda()` 等判定, 独立于 `KernelBackend`.

Dispatch 优先级 (高到低), 前两步每次调用判定, 后四步首调后缓存到 `self._forward_method`:

1. 显式 `forward(..., backend=KernelBackend.X)`
2. 全局强制 `XQT_FORCE_KERNEL_BACKEND` / `set_kernel_backend()`
3. OOT 平台覆写 (`register_oot_forward`)
4. 已声明且 `backend_eligible()` 通过的 `priority` 首个 `forward_<backend>`
5. 平台 `forward_<device>`
6. `forward_native`

`backend_eligible(backend, *args, **kwargs)` 默认检查 `capabilities[backend]` 是否满足 `PlatformInfo.detect()`, 子类可追加 `shape/dtype` 门控, 覆写后 dispatch 自动转为每次调用动态判定.

`register_fused_op(instance, module, attr)` 同时把该实例的每个 available backend 注册为 `KernelSpec`, 使 `select_kernel(..., backend=)` 仍可用.

配套: `enter_torch_compile() / leave_torch_compile()` (幂等), `enable_kernel_trace() / get_kernel_trace() / clear_kernel_trace()`.

## 四,`ops/<group>` 子结构

### 4.1 目录形态

```
xqt/kernels/
  ops/
    __init__.py            # eager import 全部 group, 填充 registry
    activation/            # BaseFusedOp 型试点
      __init__.py
      activation.py        # _GatedActivationOp 基类 + forward_native
      activation_triton.py # forward_triton 变体 (可选)
      activation_jit.py    # forward_jit 变体 (tilelang/cutile, 可选)
    attention/
      __init__.py          # 起步 inventory 型, 逐步转 registry 型
      attention.py
      kv_int8_attention.py
    gemm/
      __init__.py          # registry 型
      gemm.py              # 通用 gemm 入口
      dequant_gemm.py      # dequant 融合
    layernorm/             # BaseFusedOp 型试点
    norm/                  # 备选别名, 与 layernorm 二选一
    quantization/
    kvcache/
    moe/                   # grouped_gemm, moe_align_block_size
    mamba/ diffusion/ sampling/ communication/ memory/ speculative/ ...  # 空包占位
```

占位组保持 `__init__.py` + `__all__ = []`, 使 `registry.ops()` 形状稳定, 后续 `git mv` 实现文件即可.

### 4.2 `__init__.py` 三形态

| 形态 | 适用 | 写法 |
| --- | --- | --- |
| `BaseFusedOp` 型 | 一个 op 多后端同签名 | 定义 `class RMSNormOp(BaseFusedOp)` + `register_fused_op(RMSNormOp(), __name__, "_RMSNORM")` + 模块级薄函数 `def rmsnorm(...): return _RMSNORM(...)` |
| `registry` 型 | backend 间签名不一致 | `register_kernel(KernelSpec(op, backend, target="xqt.kernels.ops.gemm.gemm:gemm_fp16_triton", capabilities={CUDA}))` + `def gemm_fp16_triton(...): return get_kernel("gemm.gemm_fp16", TRITON)(...)` |
| `inventory` 型 | 纯 triton 族, 先盘点后收敛 | 批量 `register_kernel` + `__all__ = []`, 实现仍在散文件, 后续逐步补薄函数 |

### 4.3 命名约定

- 实现文件: `<op>.py`
- Triton 变体: `<op>_triton.py` (平铺, 不建 `triton/` 子包)
- JIT 薄封装: `<op>_jit.py` (后缀, 非 `_jit_` 前缀), 内部 `try/except` + `cache_once + load_jit()`
- 专用 triton 子包仅当单文件超过 500 行或需独立 `csrc` 时才建, 例如 `attention/nsa_triton_decode/`.

## 五,`jit/` 子结构

```
xqt/kernels/jit/
  __init__.py      # JIT infra package (不在导入时编译)
  __main__.py      # CLI: 生成 .clangd / 打印 KERNEL_PATH
  utils/
    arch.py        # get_jit_cuda_arch / make_jit_cuda_arch
    common.py      # cache_once / lazy_register
    deps.py        # REGISTERED_DEPENDENCIES
    compile/       # CompileSpec / cache / cpp_args / loader / paths / toolchain
  csrc/
    gemm/*.cu          # SM89 GEMM 与 capability probe 源码
    quantization/*.cu/.cuh  # CuTe/Nunchaku 量化源码
  include/
    xqt_kernel/    # 跨 kernel 通用头: tensor.h / tile.cuh / vec.cuh / runtime.cuh / ffi.h
```

- `utils` 与具体内核解耦, 供 `*_jit.py` 的 `load_jit()` 调用. `CompileSpec` 统一 source, include, C++/CUDA flags, target arch 与 cache; `load_extension()` 惰性调用 `torch.utils.cpp_extension.load`, 临时设置 `TORCH_CUDA_ARCH_LIST`/`MAX_JOBS`, 并按 spec 生成稳定 cache 目录. cache key 同时包含额外编译参数和 source SHA256 摘要, 源码变化不会复用旧产物.
- `get_jit_cuda_arch()` 接受 `sm_89`/`8.9`/`(8, 9)` 及 PyTorch 常见的逗号,分号分隔和 `+PTX` 标记; 规范化后 loader 只向当前构建传入一个确定架构.
- `csrc/<group>/` 存放 `__global__` 内核与 `Params` 结构体, 头文件用 `#include <xqt_kernel/...>` 引用 `include/xqt_kernel/`.
- `utils` 从 `xqt/kernels/ops/_impl/engines/{tilelang,cutile,cutlass,cute_dsl}` 的 `CompileSettings` 抽取, 统一 metadata 序列化与 artifact 路径; 旧 backend adapter 仍保留兼容 API.
- 原 `xqt/kernels/ops/_impl/*` 与 `xqt/gemm/backends/*` 中的 Python kernel implementation 已物理迁入 `ops/_impl/` 和 `ops/_impl/gemm_backends/`; 原 CUDA/C++ 源码已物理迁入 `csrc/gemm` 与 `csrc/quantization`.
- canonical binding 统一通过 `csrc_path()` 解析源码. `xqt/gemm/` 已删除, 不再保留旧 CUDA include 入口.

### 5.1 实现归档映射

| 原路径 | canonical 路径 | 兼容策略 |
| --- | --- | --- |
| `xqt/kernels/ops/_impl/<backend>/*.py` | `xqt/kernels/ops/_impl/<backend>/*.py` | 旧模块路径保留 forwarding alias |
| `xqt/gemm/backends/<sm>/*.py` | `xqt/kernels/ops/_impl/gemm_backends/<sm>/*.py` | `xqt/gemm/` 已删除, 直接使用 canonical 路径 |
| `xqt/kernels/ops/_impl/cute/*.{cu,cpp,cuh}` | `xqt/kernels/jit/csrc/quantization/*` | binding 通过 `csrc_path("quantization", ...)` 取 canonical 源码 |
| `xqt/gemm/backends/<sm>/*.cu` | `xqt/kernels/jit/csrc/gemm/*` | `xqt/gemm/` 已删除, 直接使用 canonical 源码 |

迁移完成后, 新增实现必须落在 canonical 路径. `xqt.kernels.wrappers` 仍可被旧调用方导入, 但不得在其中新增 kernel implementation 或第二份源码.

## 六,`aot/` 构建树与 JIT/AOT 方案选型

`aot` 是 CMake/pyproject 构建树, 不是 Python 包 (顶层无 `__init__.py`).

```text
xqt/kernels/aot/
  CMakeLists.txt / pyproject.toml / Makefile / build.sh
  csrc/<group>/*.cu/.cc + common_extension.cc  # REGISTER_EXTENSION(NAME)
  include/xqt_kernel_ops.h                      # 核心声明头, TORCH_LIBRARY 分段
  python/xqt_kernel/__init__.py                 # 真正的 import xqt_kernel 包入口
  python/xqt_kernel/<group>.py                  # 各 group Python 绑定
```

当前可空, 后续 wheel 产物落此. `ops/<group>` 的 `AOT` backend `target` 指向 `xqt_kernel:<group>.<op>` 或 `xqt.kernels.aot.python.xqt_kernel:<op>`.

### 6.1 JIT vs AOT 技术权衡对比

| 评估维度 | JIT (Just-In-Time) 运行时即时编译 | AOT (Ahead-Of-Time) 提前离线编译 |
| :--- | :--- | :--- |
| **首次启动延迟** | **慢 (冷启动开销)**. 首个 token 或初次加载时触发编译器调用, 存在数秒至数分钟的 warmup 抖动. | **极快 (零编译等待)**. 运行时直接 `dlopen` 加载预编译 `.so`, 秒级拉起. |
| **运行环境依赖** | **重**. 宿主机必须具备完整编译工具链 (`nvcc`, `gcc/g++`, CUDA Toolkit 开发包, Cutlass 头文件等). | **轻**. 部署环境仅需标准显卡驱动和精简 CUDA Runtime, 无需任何主机编译工具. |
| **形状特化与调优** | **极强**. 可在运行时获取精确的 $M, N, K$, Batch, SeqLen 进行分支消除, 常量折叠与 autotune 调优. | **受限**. 依赖预设模板 (Templates), 穷举参数组合会导致编译耗时与二进制体积组合爆炸. |
| **算子融合能力** | **天然适配**. 便于将 Norm + Quant + GEMM 等跨算子拼接为单个 kernel (如 TorchInductor / Triton). | **较难穷举**. 通常仅能预先手写实现有限的固定融合模式 (如 FlashAttention, FusedRMSNorm). |
| **分发与维护成本** | **维护轻量**. 仅需分发 Python 源码或 `.cu` 模版, 避免跨环境编译复杂 wheel. | **维护繁重**. 需针对不同 CUDA, Python, GPU 架构 (如 SM80/SM89/SM90) 编译构建巨大 wheel, 易发 ABI 冲突. |
| **迭代调试效率** | **高**. 修改 Python / Triton / CUDA 模板后保存即可直接重跑验证. | **低**. 每次微调代码均需触发重新打包与静态构建. |

### 6.2 工业界实践: 双轨混合分层策略 (Hybrid)

现代高性能推理引擎 (如 vLLM, SGLang, XQT) 均不采用二选一的单一方案, 而是遵循分层混合双轨制:

1. **高频底座通用算子归 AOT**:
   - 基础 Dense GEMM, FlashAttention, KV Cache 管理 (`reshape_and_cache`), 通用量化解包等使用最频繁, 逻辑固定的算子, 提前编译为原生扩展库 (如 `sgl-kernel`, `vllm._C`, `flashinfer`), 保证生产上线时冷启动耗时为 0.
2. **前沿定制, 探索性算子与动态融合归 JIT**:
   - 模型定制化融合 (如 SVD 融合, ConvRot 旋转矩阵融合, SM89 专用低比特算子实验), Triton / TileLang 自动调优算子, 优先走 JIT 路径. 借助统一编译缓存 (`~/.cache/xqt/kernels/`), 首次构建后长期复用.

### 6.3 XQT 演进规划与落地路径

结合 XQT 专注于模型压缩, 图变换, 低比特量化与算子实验的定位, 实施两阶段收敛:

1. **研发与压缩实验期 (当前阶段: JIT 优先)**:
   - 算子源文件集中收拢于 `xqt/kernels/jit/csrc/<group>/` (如 `gemm/`, `quantization/`), 统一经 `csrc_path()` 与 `CompileSpec` 驱动 JIT 编译并持久化缓存到 `~/.cache/xqt/kernels/`.
   - 该方式保障新型量化算法 (AWQ, GPTQ, ConvRot, SVDQ, NVFP4) 和算子融合能够随改随跑, 零打包门槛.
2. **生产固化与交付期 (下一阶段: AOT 收敛)**:
   - 当特定核心算子 (如 INT8 Ada GEMM, SVD 融合算子, 基础 RMSNorm) 逻辑完全冻结且参数稳定后, 将其沉淀收拢至 `xqt/kernels/aot/`.
   - 通过 CMake / pyproject 将其打包构建为独立 wheel 二进制扩展, 消除生产 Serving 容器中的 `nvcc` 工具链依赖与冷启动毛刺.

## 七,与现有三层边界的关系

```
xqt.nn facade / torch module
    |
    +--> xqt.kernels.ops.<group>.<op>()   # 直接调用 (单算子验证)
    |
    +--> xqt.convert() / materialize_module()
              |
              +--> wrapper / materialize  (contract 校验, shape/layout 翻译, candidate 构造, fallback)
                        |
                        +--> xqt.kernels.ops.<group>.<op>()   # 边界翻译层再调内核
                                  |
                                  +--> xqt.kernels.jit / aot / triton / tilelang / flashinfer
```

- `xqt.nn` 与 `kernel` 不直接耦合, 必须经 `wrapper/materialize` (见 `xqt-kernel-wrapper-nn-boundary.md`).
- `xqt.kernels.ops.*` 是 `kernel` 层的唯一公开入口, `wrapper/materialize` 内部也经它调用, 不直接 `import xqt.kernels.jit.csrc` 或 `xqt.kernels.ops._impl.triton.gemm`.
- `xqt.kernels.engine_resolve` 的 `EngineRegistration` 是引擎级能力契约, `xqt.kernels.registry` 负责单算子 `op x backend` 盘点, 二者通过 `capabilities` 对齐, 不重复定义 `CapabilityRequirement`.

## 八,术语与禁止项

| 词 | 含义 | 示例 | 禁止 |
| --- | --- | --- | --- |
| `KernelBackend` | 内核产地 | `triton`, `tilelang`, `flashinfer`, `cutlass`, `custom_cuda` | 禁止把 `awq/gptq/svd` 写成 backend |
| `DeviceType` | 设备族 | `cuda`, `hip`, `npu`, `cpu` | 禁止烘焙为 `cuda_flashinfer` |
| `engine` (XQT) | 内部 lowering 选择 | `xqt.convert(engine=...)`, `StageReport.engine` | 禁止等同于 TensorRT/ONNX Runtime |
| `quant backend` | 量化适配路径 | `torchao`, `onnxruntime_qdq` | 禁止 `backend=tilelang` |
| `maturity` | 实现成熟度 | `executable / reference_guarded / metadata_only / planned` | 禁止把 `metadata_only` 标为 `executable` |

## 九,新增内核与外部库接入

1. 在 `xqt/kernels/ops/<group>/__init__.py` 加一行 `register_kernel(KernelSpec(op="gemm.bmm_fp8", backend=KernelBackend.FLASHINFER, target="flashinfer:bmm_fp8", capabilities={CapabilityRequirement.CUDA}, format_signature=..., description="..."))`.
2. 薄封装按需: `def bmm_fp8(...): return get_kernel("gemm.bmm_fp8", KernelBackend.FLASHINFER)(...)` 或直接 `select_kernel(...).load()` 调用.
3. 目录落点: 自研实现放 `xqt/kernels/jit/csrc/<group>/` 或 `xqt/kernels/ops/<group>/<op>*.py`, 外部库无需落实现, `target` 指向外部包即可.
4. 测试与 benchmark 镜像 `ops` 结构: `tests/kernels/ops/<group>/test_<op>.py` + `benchmark/<group>/bench_<op>.py`, 共享 helper 放 `xqt.test.kernels`.
5. 评审规则: 新增可调用内核必须补 `xqt.kernels.ops.*` 入口, 禁止继续扩大 `xqt.kernels.jit` 作为长期公开命名空间 (复刻 `sglang.kernels` review rule).

## 十,迁移与兼容

- Phase 1-2 仅新增 `xqt/kernels` 骨架与 `register_kernel` 盘点, 不搬实现, 零破现有链路.
- Phase 3 为 `ops.<group>` 补薄封装, 新代码从 `xqt.kernels.ops.*` 导入, 旧 `xqt.kernels.ops._impl.*` 保留 shim 一版本并标 `@deprecated`. `xqt.gemm` 已删除.
- Phase 4 已完成实现物理迁移: Python implementation 归档到 `ops/_impl`, CUDA/C++ 归档到 `jit/csrc`, 编译链统一到 `jit/utils/compile`; 旧 `GemmKernelRegistry / _ENGINE_REGISTRY / *_KERNEL_REGISTRY` 仅作 deprecated mirror, 指向 `xqt.kernels.registry`.
- Phase 5 把剩余计算栈收进三个子包: GEMM 合约进 `ops/gemm`, operator/benchmark 进 `wrappers`, nn/convert/fixtures 进 `nn`. 公开别名保留 `from xqt import nn` 与 `xqt.convert`; 顶层 `xqt/nn/` 已删除.

## 继续阅读

- [xqt.md](xqt.md)
- [xqt-kernel-wrapper-nn-boundary.md](xqt-kernel-wrapper-nn-boundary.md)
- [xqt-engine-quant-boundary.md](xqt-engine-quant-boundary.md)
- [xqt-operator-block-optimization.md](xqt-operator-block-optimization.md)
- [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md)
