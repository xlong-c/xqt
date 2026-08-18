# XQT 推理 (Inference) 能力说明

本文对照当前源码, 说明 XQT **推理侧** 有哪些入口, 支持哪些精度与 runtime, 以及文件包 / 混合精度引擎如何工作.

**权威事实源是代码**. 环境与 optional 依赖会变, 请以 capability API 与运行时检查为准.

## 目标契约 (严格解耦, 已落地边界)

量化与推理应 **严格解耦**. 推理阶段的合法输入只有:

1. **已量化模型** (网络结构与量化存储在 model / 图产物内)
2. **可选计算配置** (各算子/模块的计算逻辑与精度要求, 以及 engine 须具备的 capability)

不应:

- 在推理入口再跑 quantizer / calibration
- 把 AWQ/GPTQ/SVD 等 **算法身份** 当作推理必选输入
- **强行指定** `engine=tilelang/triton/...` 作为唯一合法路径; 配置只声明 **需要什么能力**, 由 runtime 在具备该 capability 的 engine 中解析

```text
Quant 输出:  model + compute_config(capabilities, precisions)
Infer 输入:  同上
Infer 行为:  resolve_engine(required_capabilities) -> execute
```

交接面方案与落地见 [../architecture/xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md).  
Engine / quant 词表与禁止项见 [../architecture/xqt-engine-quant-boundary.md](../architecture/xqt-engine-quant-boundary.md).  
DEBT-001/002/003 的裁决和落地范围见 [../architecture/xqt-design-debt.md](../architecture/xqt-design-debt.md).

文件模型包在上述计算交接面之上,另有一个模型侧语义 contract. `compute_config` 描述模型如何执行, `inference` 描述调用方如何把语义输入映射到物理 IO 以及如何解释输出. 二者都不携带 quant method, dataset 或 serving scheduler.

## 负责什么

- 区分三条推理相关路径: PyTorch 混合精度引擎, 标准模型包文件推理, export/deploy 外部 runtime.
- 列出 export format capability 与声明的 precision 集合.
- 列出 hybrid engine 支持的 compute precision 与 channel hybrid 语义.
- 说明 deploy runtime handle 与 model package 的边界.
- 标明 **现状 vs 目标契约** 的差距.

## 不负责什么

- 不接管 TensorRT / ORT / OpenVINO 等官方安装与完整 CLI.
- 不做 task-level accuracy / mAP 评测闭环.
- 不把 quantizer / calibration 算进 runtime (runtime **只消费** 已量化 artifact; 目标与现状 docstring 一致).
- 不把 operator **engine** 与外部 **backend** 混写.
- 不把 quant **method** (awq/gptq/...) 写成推理配置主轴.

## 术语

| 词 | 含义 | 代码锚点 |
| --- | --- | --- |
| 混合精度推理 | 已量化 PyTorch 模块上按 policy 调度 compute precision / channel hybrid | `xqt/runtime/engine.py` |
| 计算配置 | 目标: 算子级精度 + required capabilities; 现状: 近似 `ExecutionPolicy` / module contract | `contracts/compute.py`, `ExecutionPolicyPayload` |
| 模型包 | 文件侧标准加载契约 `manifest.json` + `runtime/config.json` | `xqt/runtime/package.py` |
| 语义推理 contract | 模型族 adapter 的版本化 IO/schema 描述 | `xqt/contracts/inference.py`, `xqt/runtime/inference.py` |
| runtime handle | deploy stage materialize 的可执行 session | `xqt/contracts/runtime.py`, export/deploy pass |
| export backend | ONNX/TRT/OpenVINO/... 导出与适配 | `xqt/export/` |
| engine | XQT 内部 kernel (见 [xqt-engines.md](xqt-engines.md)); **推理配置不应硬编码必选 engine** | `xqt/operator_opt/` |

---

## 0. 现状 vs 目标 (审阅摘要)

| 项 | 目标 | 现状 (代码) |
| --- | --- | --- |
| 推理输入 | 模型 + 可选计算配置 | `QuantizedModel.infer_handoff()` → model + compute_config; Hybrid 可读 compute_config / ExecutionPolicy |
| 是否跑 quant | 否 | runtime 包 **不** import quant, docstring 正确 |
| 是否强制 engine | 否, 只要求 capability | `engine_resolve` 按 capabilities; int8_mma 默认 `auto` + lazy kernel; `preferred_engines` 仅 hint |
| 计算配置 | 算子逻辑 + 精度 + required_capabilities | `ComputeConfig` / `ExecutionPolicyPayload.required_capabilities`; 包可选 `runtime/compute.json` |
| 算法 method | 仅 quant 报告 | `backend/method/strategy` 保留 lineage; Infer 不要求 |

---

## 1. 三条路径总览 (现状实现)

```text
A. PyTorch in-process hybrid inference
   已量化 module (+ ExecutionPolicy / precision_overrides)
     -> HybridInferenceEngine
     -> forward
   # 目标: policy 升级为 compute_config (capability 需求), 不绑死 engine 名

B. File-based model package (当前最小闭环: ONNX + ORT)
   export 写出 *.xqtpkg/
     -> load_model_package()
     -> create_inference_runner() / ONNXRuntimeRunner
     -> create_inference_session() / InferenceSession
     -> preprocess -> run -> postprocess
   # 包内可附带 compute_config 和 inference contract; 仍不解析 quant recipe YAML

C. Workflow deploy runtime handle
   deploy stage (materialize)
     -> RuntimeHandlePayload
     -> ORT InferenceSession 或 TRT execution context
   (不替代数值/性能验收)
```

workflow 的 `ArtifactManifest` 是 **实验追踪**, 不是文件推理加载契约. 文件推理只认模型包 manifest.

---

## 2. 路径 A: HybridInferenceEngine reference/交互式封装

### 2.1 职责边界

源码: `xqt/runtime/engine.py`

该路径用于 Python 交互式检查与 reference forward. 它不是 deploy runtime handle,serving scheduler 或生产引擎主入口;生产部署由 workflow deploy stage 产出的 `RuntimeHandlePayload` 承担.

- **消费**: 已带量化存储的 `nn.Module`, 可选 `ExecutionPolicyPayload`.
- **不做**: quantizer, calibration, sensitivity.
- 默认 `default_precision="w4a4"`, `runtime="pytorch"`.

构造:

```python
from xqt.runtime import HybridInferenceEngine

engine = HybridInferenceEngine(model, default_precision="w4a4")
# 或
engine = HybridInferenceEngine.from_quantized_model(quantized_model)
result = engine(...)  # HybridInferenceResult: output + precision_map + channel_hybrid_map
```

### 2.2 Execution policy

源码: `xqt/runtime/policy.py`, `xqt/contracts`

- `apply_execution_policy(...)`: 按 override 设置各模块 `compute_precision`.
- `build_execution_policy_payload(...)`: 结构化 policy 产物.
- `set_module_compute_precision` / `collect_module_precision_map`.
- 模块需实现 / 兼容 `SupportsComputePrecision`.

### 2.3 支持的 compute precision

源码: `xqt.contracts` → `SUPPORTED_COMPUTE_PRECISIONS`

| 值 | 语义 (命名) |
| --- | --- |
| `w4a4` | 权重与激活均 4-bit 计算意图 (默认) |
| `w8a8` | 8-bit 计算意图 |
| `w4a16` | 4-bit 权重 + 16-bit 激活意图 |
| `bf16` | BF16 计算意图 |

`normalize_compute_precision()` 负责规范化. **意图 ≠ 硬件一定跑原生 MMA**; 具体模块 (如 ConvRot) 如何解释 precision 以模块实现为准. 对 `Int8MmaLinear`, `execution_metadata()["runtime_precision"]` 明确区分 W8A8 reference, native INT8 MMA, 和小 batch 浮点回退; `min_int8_rows=0` 时小 batch浮点回退状态为 `disabled`, 不应虚构该条件.

### 2.4 Channel hybrid

源码: `xqt/runtime/channel.py`, `SUPPORTED_CHANNEL_AXES = {input, output}`

- 部分 channel 走更高精度, 其余走低 bit.
- quant 侧写 mask; runtime 侧 dual-path reference 前向 (`channel_hybrid_linear_reference` 等).
- 后续可换真实 kernel, 当前不要写成已完成生产 kernel 融合.

### 2.5 相关 quant 产物

常见来源: `xqt/quant/quantizers/convrot_4bit.py` 等写入 execution policy metadata;  
`HybridInferenceEngine.from_quantized_model` 会读 `metadata["execution_policies"]`.

---

## 3. 路径 B: 标准模型包 (file inference)

### 3.1 包布局

源码: `xqt/runtime/package.py`

常量:

- `MODEL_PACKAGE_SCHEMA_VERSION = "1.0"`
- `MODEL_PACKAGE_ARTIFACT_TYPE = "xqt_model_package"`

典型目录:

```text
model.xqtpkg/
  manifest.json
  model/<artifact>          # 例如 model.onnx
  runtime/config.json       # runtime 名, providers 等
```

`ModelPackageManifest` 字段: `schema_version`, `artifact_type`, `package_version`, `entrypoints` (`model`, `runtime_config`, 可选 `compute_config`), `model`, `runtime`, `io`, `inference`, `quantization`, `metadata`.

### 3.2 API

```python
from xqt.runtime import (
    load_model_package,
    create_inference_runner,
    write_model_package,
    ONNXRuntimeRunner,
)

pkg = load_model_package("path/to/model.xqtpkg")  # LoadedModelPackage
runner = create_inference_runner(pkg)             # 当前主路径: ONNX + onnxruntime
outputs = runner.run(inputs)
```

`ONNXRuntimeRunner` 约束 (代码硬检查):

- `model_format == "onnx"`
- `preferred_backend == "onnxruntime"`
- providers 来自参数或 `runtime/config.json`, 默认 `CPUExecutionProvider`

### 3.3 谁写出模型包

export 路径 (ONNX target) 会额外落 `*.xqtpkg/manifest.json` (见 FRAMEWORK / export pass).  
`write_model_package(...)` 可手动打包.

**当前最小闭环保证**: ONNX + ONNX Runtime. 其它 format 的 package runner 未同等承诺.

### 3.4 语义推理 contract

每个模型只声明 contract,不重复编写 runtime runner:

```json
{
  "schema_version": "1.0",
  "family": "vision",
  "adapter": "vision.classification",
  "adapter_version": "1",
  "inputs": [{"name": "input", "semantic": "image"}],
  "outputs": [{"name": "output", "semantic": "logits"}],
  "config": {
    "resize": [224, 224],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

`tensor` adapter 负责通用 tensor IO 映射和命名输出; `vision.classification` 负责图像 layout,dtype,resize,normalize 和 logits 解码. 新模型族实现一次 `InferenceAdapter`,通过 `register_inference_adapter()` 注册并声明 adapter version;后续模型只修改 manifest 的 `family`, `adapter`, `adapter_version` 和 `config`.

contract 不负责 tokenizer,prompt 模板,采样循环,dataset,batch scheduler 或准确率评测. 这些属于调用方或外部 serving runtime.

---

## 4. 路径 C: export / deploy 与 runtime handle

### 4.1 Export capability 矩阵

源码: `xqt/export/capability.py` `DEFAULT_EXPORT_CAPABILITIES`

| format | priority | status | maturity | runtimes | 声明 precisions | dynamic_shapes | quantization |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `torch_export` | P0 | implemented | executable | pytorch | fp32, fp16, bf16 | 是 | 否 |
| `onnx` | P0 | implemented | executable | onnxruntime, tensorrt, openvino | fp32, fp16, bf16, int8 | 是 | 是 |
| `tensorrt` | P0 | adapter | reference_guarded | tensorrt | fp32, fp16, bf16, int8, fp8 | 是 | 是 |
| `torchscript` | P1 | implemented | executable | pytorch, pnnx | fp32, fp16, bf16 | 否 | 否 |
| `openvino` | P1 | adapter | reference_guarded | openvino | fp32, fp16, bf16, int8 | 是 | 是 |
| `executorch` | P2 | adapter | reference_guarded | executorch | fp32, fp16, int8 | 否 | 是 |
| `ncnn` | P2 | adapter | reference_guarded | ncnn | fp32, fp16, int8 | 否 | 是 |
| `mnn` | P2 | adapter | reference_guarded | mnn | fp32, fp16, int8 | 否 | 是 |

说明:

- `status=implemented`: 主链路适配较完整; `adapter`: 命令/可选依赖适配, 常含 dry-run.
- `maturity=reference_guarded` 表示不可按生产默认路径宣传.
- TensorRT 的 sparse_support 注记为 "2:4 on supported NVIDIA GPUs" (capability notes).

刷新:

```python
from xqt.export.capability import deployment_capability_matrix
for row in deployment_capability_matrix():
    print(row.to_dict())
```

### 4.2 Deploy runtime handle

- typed config: `runtime_handle.onnxruntime` (providers), `runtime_handle.tensorrt` (device, runtime plugin libraries).
- TRT **runtime** plugin **不** 从 engine-build target 隐式继承, 必须在 runtime config 显式列出.
- materialize 后 payload kind 为 `runtime_handle` (见 `xqt/workflows/stage.py` `PayloadKind`).
- 创建 session / execution context **不等于** 数值或性能验收通过.

### 4.3 与 operator engine 的边界

| 选择 | 用于 |
| --- | --- |
| `engine=tilelang/triton/...` | XQT 内部算子 lowering (PyTorch 侧) |
| export `format=onnx/tensorrt/...` | 外部图 / engine 构建 |
| `deployment_engine` (operator 矩阵中) | 仅 capability 占位, 不改写 module |

不要把 `triton` 写成和 TensorRT 并列的 "外部推理后端".

---

## 5. 精度维度怎么读

推理相关至少有 **三套** 精度语言, 不要混用:

| 维度 | 出现位置 | 例子 |
| --- | --- | --- |
| Quant strategy / storage | quant stage, `schema` strategies | `fp4_weight_only`, `static_qdq_int8` |
| Compute precision (hybrid) | runtime policy | `w4a4`, `w8a8`, `w4a16`, `bf16` |
| Export declared precision | export capability `precisions` | `fp32/fp16/bf16/int8/fp8` |

另见 quant TRUE/PSEUDO: [xqt-engines.md](xqt-engines.md) 第 3 节.

硬件代际与 MMA 总表 (方法论): [operator-kernel-tuning-guide.md](operator-kernel-tuning-guide.md).

---

## 6. Session / workflow 如何接到推理

| 需求 | 入口 |
| --- | --- |
| 交互式 quant 后 PyTorch 侧跑 | `session.quant(...)` 后取 `session.model` → `HybridInferenceEngine` |
| 导出 ONNX 并落包 | `session.export(format="onnx", ...)` 或 YAML export stage |
| materialize ORT/TRT session | `session.deploy(..., runtime_handle=...)` |
| 只读文件推理 | `load_model_package` + `create_inference_runner` |
| 语义文件推理 | `create_inference_session` + `manifest.inference` |
| 环境是否具备某路径 | `assess_xqt_readiness()` / `session.readiness()` |

校准与 example 输入由调用方传入; recipe 不声明 dataset.

---

## 7. 源码地图

```text
xqt/runtime/
  engine.py      # HybridInferenceEngine, HybridInferenceResult
  policy.py      # execution policy apply / build
  channel.py     # channel hybrid
  package.py     # model package + ONNXRuntimeRunner + InferenceSession factory
  inference.py   # semantic contract adapters

xqt/export/
  capability.py  # export format matrix
  onnx_exporter.py, trt_*.py, openvino.py, torch_exporter.py, mobile.py, ...

xqt/contracts/
  runtime.py     # RuntimeHandlePayload, RuntimePlanPayload, ExportBundlePayload, ...
  compute.py     # compute precision contracts
  inference.py   # model-family semantic IO contract
  channel.py     # channel hybrid contracts
  quantized.py   # QuantizedModel / payload

xqt/pipeline/export_pass.py   # export/deploy 执行
xqt/workflows/optimization.py # Session.export / deploy
```

---

## 8. 已知边界与常见误区

1. **模型包 runtime 最小闭环只有 ONNX+ORT**; 不要假设 TRT engine 已有同等 `create_inference_runner`. `inference` contract 本身是 format-independent,但 runner 仍受 backend capability 限制.
2. **ArtifactManifest ≠ 推理加载契约**.
3. **Hybrid engine 不量化**; 先 quant 再 bind policy.
4. **export precisions 是 capability 声明**, 不是 "本机一定能跑出该精度 TRT engine".
5. **FP4/FP8 真实收益** 必须在支持硬件上验证; synthetic smoke 只做闭环.
6. **deploy handle 创建成功** 不替代 output diff / benchmark.

---

## 9. 继续阅读

- Engine / quant 矩阵: [xqt-engines.md](xqt-engines.md)
- 导出与 engine 分册: [backends/index.md](backends/index.md)
- 工作流入口: [../usage/xqt-workflows.md](../usage/xqt-workflows.md)
- 架构与 Stage: [../architecture/xqt.md](../architecture/xqt.md)
- 包内契约: [../../../xqt/FRAMEWORK.md](../../../xqt/FRAMEWORK.md)
- 阅读页导览: [../../html/xqt.html](../../html/xqt.html)
