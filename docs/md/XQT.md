# XQT 模型优化工具链

本文是 `xqt/` 的唯一长期事实源.

XQT 只关注模型本身. 它接收 PyTorch 模型,checkpoint 或导出产物,执行模型侧压缩,图变换,导出适配,误差分析和 benchmark. 训练,QAT,finetune,distillation,recovery,dataset / dataloader,training provider 和 evaluation provider 不属于 XQT.

需要梯度更新的流程归 XDL 或第三方训练工具,再把训练后的模型或 checkpoint 交给 XQT.

## 1. 项目定位

XQT 的主链路是:

```text
PyTorch model / checkpoint / exported artifact
    -> model compression or graph transform
    -> model-side diff / layer analysis
    -> export target artifact
    -> benchmark / manifest
```

核心职责:

- 支持 `nn.Module` 和 `state_dict` 作为主要输入.
- 覆盖 PTQ / QDQ,权重量化,剪枝,算子优化,导出前适配和部署格式转换.
- 能导出 ONNX,TensorRT,OpenVINO,torch.export,TorchScript,ExecuTorch,ncnn,MNN 等产物.
- 记录配置,源 checkpoint,指标,产物校验和执行阶段,保证结果可复现.

非目标:

- 不替代 `xdl.trainer.Trainer`.
- 不维护通用 model / loss / metric registry.
- 不运行 `zero_grad() -> backward() -> step()` 训练循环.
- 不做 task-level validation 或 accuracy / mAP 评测闭环.
- 不重新实现 TensorRT,OpenVINO,ONNX Runtime,ExecuTorch,ncnn,MNN 等后端.

## 2. 配置方式

XQT 只提供两种配置方式:

1. `XQTOptimizationSession`.
2. YAML workflow,通过 `optimize_model()` 运行.

原则上不要新增 CLI 参数解析,JSON shaped workflow 或其他配置路径.

## 3. 当前可用能力

- 已基本可用: PTQ / QDQ / torchao 量化,常规剪枝,ONNX / torch.export / TorchScript / TensorRT / OpenVINO / ExecuTorch / ncnn / MNN 导出,output diff,layer analysis,latency / memory benchmark,manifest,统一 capability / reporting schema.
- 半可用: TensorRT engine / plugin preflight,TileLang 的受限 kernel target,FP4 packed weight 到 TileLang operator stage 的桥接.
- 偏实验: 更完整的 AWQ / GPTQ packed megakernel,以及更广泛的 backend capability 闭环.

## 4. 性能分析工具

XQT 需要明白 profiling,但不接管 vendor profiler 的安装,权限,驱动版本或 GUI 工作流. XQT 的职责是:

- 在 `benchmark` / `operator` / `export` / `deploy` stage 中记录可复现的 profiling 上下文: backend,device,target artifact,input shape,warmup,repeat,precision,batch size,环境版本和命令建议.
- 把外部 profiler 产物作为 artifact 进入 manifest,例如 `.ncu-rep`,`.nsys-rep`,ROCm trace,VTune result,`msprof` 目录或 TensorBoard profile 目录.
- 在 stage report 中区分 latency / memory benchmark 和 profiler 诊断结果: benchmark 给出可比较指标,profiler 给出瓶颈归因和优化线索.
- 对 planned / capability-only 后端只给 preflight 和命令模板,不要把未实测 profiler 结果标记为 applied.

XQT 不做:

- 不封装 `ncu`,`nsys`,`rocprof`,`vtune`,`msprof` 等工具的完整 CLI 语义.
- 不为 profiling 新增 dataset / dataloader,evaluation provider 或 task-level accuracy 流程.
- 不在 XQT 中实现厂商 kernel counter 解析器,除非该解析能直接服务已有 report schema 或 backend preflight.

工具映射:

| 硬件 / 生态 | 系统级 timeline | kernel / 算子级分析 | XQT 记录方式 |
| --- | --- | --- | --- |
| NVIDIA CUDA | Nsight Systems / `nsys` | Nsight Compute / `ncu` | 保存 `.nsys-rep` / `.ncu-rep`,记录 `--set`,`--kernel-name`,`--launch-skip`,`--launch-count` 等关键参数. |
| AMD ROCm / HIP | ROCm Systems Profiler / `rocprof-sys` | ROCProfiler / `rocprof`,ROCm Compute Profiler | 保存 trace/report 目录,记录 ROCm 版本,GPU 型号和 counter collection 配置. |
| Intel oneAPI / GPU | Intel VTune Profiler | VTune GPU Hotspots / GPU Offload | 保存 VTune result 目录,记录 collect 类型和 oneAPI / driver 版本. |
| Apple Metal | Instruments / Metal System Trace | Xcode Metal Debugger / GPU Counters | 保存 trace 或截图摘要,记录 Xcode,macOS,芯片型号和 capture 范围. |
| Arm Mali / Immortalis | Arm Streamline | Streamline / Performance Advisor | 保存 Streamline capture 和 Performance Advisor 报告,记录设备,driver 和 workload. |
| Qualcomm Adreno / Snapdragon | Snapdragon Profiler | Snapdragon Profiler GPU counters | 保存 profiler session,记录 SoC,driver,thermal 状态和 workload. |
| Google TPU | XProf / TensorBoard Profile | XProf / TensorBoard Profile | 保存 profile 目录,记录 TPU 类型,step 范围和 TensorFlow / JAX / PyTorch XLA 版本. |
| 华为 Ascend | CANN `msprof`,MindStudio Profiler | `msprof-analyze`,Ascend profiler | 保存 `msprof` 输出目录,记录 CANN,driver,NPU 型号和采集模式. |
| 摩尔线程 MUSA | Moore Perf System | Moore Perf Compute | 保存工具报告目录,记录 MUSA,driver,GPU 型号和采集范围. |
| 寒武纪 MLU | cnperf / 框架 profiler | cnperf / MLU-OPS Perf-Analyse | 保存报告目录,记录 Neuware,driver,MLU 型号和算子范围. |

推荐工作流:

1. 先跑 XQT `benchmark` stage,拿到 latency / memory / throughput 的稳定基线.
2. 再用系统级 profiler 找到耗时阶段或 kernel 区间.
3. 最后用 kernel / 算子级 profiler 深挖目标 kernel,并把 profiler report 作为 artifact 挂到同一 manifest.
4. 优化后重复 benchmark,用同一输入形状和同一环境记录前后对比.

## 5. 场景表

| 场景 | 状态 | 备注 |
| --- | --- | --- |
| FP4 量化 | 半可用 | 有 reference 路径和 TileLang 桥接,仍缺真实 CUDA runtime 数值与性能验证. |
| TileLang megakernel | 半可用 | 已覆盖受限 attention 和 dequant GEMM 路径. |
| TensorRT + `.so` 插件 | 半可用 | 已有 build / inspect / preflight / loadability 检查. |
| 常规剪枝 / 误差分析 | 已基本可用 | layer diff, sensitivity, distribution stats 和 benchmark 已接通. |
| Readiness / capability matrix | 已基本可用 | `assess_xqt_readiness()` 输出场景状态,capability matrix 和 reporting schema,planned backend 不标记为 applied. |

## 6. 文档分层

- [XQT_SUMMARY.md](XQT_SUMMARY.md): 摘要入口.
- [../html/xqt.html](../html/xqt.html): 人类阅读页.
- [../../xqt/README.md](../../xqt/README.md): 包内短入口.
- [../../xqt/FRAMEWORK.md](../../xqt/FRAMEWORK.md): 包内工程契约.
