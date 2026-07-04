# ncnn 后端

本文介绍 ncnn 后端. 在 XQT 中, ncnn 当前是 P2 adapter, 支持通过 pnnx 或 onnx2ncnn 命令把 TorchScript / ONNX 转换为 ncnn param/bin 产物.

## 后端介绍

ncnn 是 Tencent 开源的高性能神经网络推理框架. 它强调轻量, 跨平台, 少依赖, 面向 mobile, embedded, desktop 和 WebAssembly 等部署场景. ncnn 支持 CPU 执行和 Vulkan GPU 后端, 在移动端 CV 模型中很常见.

ncnn 的典型产物是:

- `.param`: 网络结构和 layer 参数.
- `.bin`: 权重二进制.

常见转换路径:

```text
PyTorch -> TorchScript -> pnnx -> ncnn param/bin
PyTorch -> ONNX -> onnx2ncnn -> ncnn param/bin
```

## 适合的使用场景

- Android / iOS 上部署轻量 CV 模型.
- 端侧二进制和依赖要尽量小.
- 需要 CPU fallback, 同时可选 Vulkan GPU 加速.
- 模型结构以 CNN, detection, segmentation, image restoration 等端侧模型为主.
- 应用侧可以显式处理 image resize, mean / norm, pixel format.

## 不适合的场景

- 模型包含大量动态控制流或复杂 Transformer custom op.
- 需要服务端大模型高吞吐推理.
- 不准备维护前处理和输出后处理对齐.
- 不准备为 custom layer 写 ncnn layer 实现.

## 核心概念

- pnnx: PyTorch / TorchScript 到 ncnn 的转换工具.
- onnx2ncnn: ONNX 到 ncnn param/bin 的转换工具.
- param/bin: ncnn 的模型结构和权重产物.
- blob name: ncnn 输入输出绑定依赖 blob 名.
- extractor: ncnn runtime 中设置输入并提取输出的对象.
- Vulkan backend: ncnn 的 GPU 加速路径.
- custom layer: ncnn 不支持的 layer 需要手写并注册.

## 核心 API

源码位置: `xqt/export/mobile.py`

命令构建:

- `build_pnnx_command(model_path, pnnx_path="pnnx", extra_args=None)`: 构造 pnnx 命令.
- `build_onnx2ncnn_command(onnx_path, param_path, bin_path, onnx2ncnn_path="onnx2ncnn", extra_args=None)`: 构造 onnx2ncnn 命令.

转换:

- `export_ncnn_with_pnnx(model_path, pnnx_path="pnnx", param_path=None, bin_path=None, extra_args=None, timeout=None, dry_run=False)`: 用 pnnx 转换 TorchScript 或 ONNX.
- `export_ncnn_from_onnx(onnx_path, param_path, bin_path, onnx2ncnn_path="onnx2ncnn", extra_args=None, timeout=None, dry_run=False)`: 用 onnx2ncnn 转换 ONNX.
- `CommandExportResult`: 命令型转换结果, 包含 output_paths, command, returncode, stdout, stderr, checksums, dry_run, metadata.

## 简单例子

ONNX 转 ncnn:

```python
from xqt.export import export_ncnn_from_onnx


result = export_ncnn_from_onnx(
    "artifacts/model.onnx",
    "artifacts/ncnn/model.param",
    "artifacts/ncnn/model.bin",
    dry_run=True,
)
print(" ".join(result.command))
```

pnnx 转换:

```python
from xqt.export import export_ncnn_with_pnnx


result = export_ncnn_with_pnnx(
    "artifacts/model.ts",
    param_path="artifacts/ncnn/model.param",
    bin_path="artifacts/ncnn/model.bin",
    extra_args=["inputshape=[1,3,224,224]"],
)
```

## 后端实现注意点

- 转换产物只包含模型图和权重, 前处理通常仍在应用侧.
- 需要记录 input blob, output blob, pixel format, mean / norm, resize 策略.
- pnnx 和 onnx2ncnn 的支持模型范围不同, 失败时应保留命令和 stderr.
- Vulkan 和 CPU 路径的支持 op 和数值结果可能不同.
- int8 量化需要额外 calibration / quant table / runtime 支持.
- custom layer 要记录 layer 名, 注册方式和动态库 / 源码版本.

## 常见问题

- 转换命令不存在: `pnnx` 或 `onnx2ncnn` 不在 PATH.
- 输出文件未生成: 命令返回 0 也要检查 param/bin 是否存在.
- blob 名不一致: 应用侧输入输出名和 param 中名字不匹配.
- 图像结果错位: resize, channel order, mean / norm 和 layout 没对齐.
- Vulkan 路径失败: 设备不支持, build option 不完整或 layer 没有 Vulkan 实现.

## 官方资料

- ncnn GitHub: <https://github.com/Tencent/ncnn>
- ncnn wiki: <https://github.com/Tencent/ncnn/wiki>
- pnnx documentation: <https://pnnx.readthedocs.io/>

