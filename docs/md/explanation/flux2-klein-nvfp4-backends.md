# FLUX.2 klein NVFP4 Engine 推理

本文说明 XQT 对 `black-forest-labs/FLUX.2-klein-4b-nvfp4` 的模型侧接入方式. 该仓库是 FLUX.2 [klein] 4B 的 NVFP4 single-file 权重, XQT 不接管训练, 数据集或任务评测, 只负责把已加载模型中的 packed NVFP4 Linear 模块路由到 operator engine.

## 入口

源码入口:

- `examples.xqt_models.flux2_klein_nvfp4.load_flux2_klein_nvfp4_transformer`
- `examples.xqt_models.flux2_klein_nvfp4.load_flux2_klein_nvfp4_pipeline`
- `examples.xqt_models.flux2_klein_nvfp4.collect_flux2_klein_nvfp4_targets`
- `examples.xqt_models.flux2_klein_nvfp4.materialize_flux2_klein_nvfp4_engine`
- `examples.xqt_models.flux2_klein_nvfp4.run_flux2_klein_nvfp4_inference`

该 FLUX.2 专用 materialize 接口的新调用统一使用 `engine`, 不保留旧接口里的 `backend` alias. 这里的约束只适用于 operator engine 选择, 不改变 XQT 全局对 quant/export/runtime `backend` 的术语定义. 当前支持三个类别:

- `tilelang`
- `cutile`
- `cutedsl` / `cute_dsl`

## 默认路由

| engine | 默认 pattern | 权重路径 | 当前状态 |
| --- | --- | --- | --- |
| `tilelang` | `dequant_gemm_epilogue` | packed NVFP4 bridge, Ada 上可走一次性 dense cache | XQT 主要可执行路径, CUDA kernel / native fastpath / fallback 视 shape 和硬件选择 |
| `cutile` | `nvfp4_packed_dequant_gemm_epilogue` | packed NVFP4 bridge, 当前实际推理优先 dense cache | reference-guarded 推理 wrapper, 记录 CuTile artifact metadata |
| `cute_dsl` | `gemm_epilogue` | packed NVFP4 先进入 dense cache bridge | reference-guarded dense GEMM epilogue wrapper, 不直接消费 packed NVFP4 |

这三个 engine 都能 materialize 可调用模型. 其中 `cutile` 和 `cute_dsl` 当前不能被写成已验证高性能生产 kernel; 它们用于 engine 类别覆盖, artifact 记录, smoke 推理和后续 kernel 扩展.

`cutile` 的 target plan 默认仍记录 `nvfp4_packed_dequant_gemm_epilogue`, 但 executor 不会只因为 `cuda.tile` 可导入就直连 packed NVFP4 path. 只有对应 pattern 的 metadata 明确不是 `reference_guarded` / `metadata_only` 时, 才会选择 packed runtime kernel. 当前内置 CuTile packed NVFP4 spec 仍是 reference-guarded, 因此 FLUX.2 klein NVFP4 推理 wrapper 会退到一次性反量化的 dense cache, 实际执行 `dense_linear_epilogue`. Transformer 中常见的 rank-3 Linear 输入会在 wrapper 内 flatten 成 2D, 调用 engine 后再恢复原 shape.

## 示例

```python
import torch

from examples.xqt_models.flux2_klein_nvfp4 import (
    load_flux2_klein_nvfp4_pipeline,
    materialize_flux2_klein_nvfp4_engine,
)


pipe = load_flux2_klein_nvfp4_pipeline(
    dtype=torch.float16,
    device="cuda",
)

result = materialize_flux2_klein_nvfp4_engine(
    pipe,
    engine="tilelang",
    target_arch="sm_89",
    inplace=True,
)

image = result.model(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
).images[0]
```

切换 engine:

```python
for engine in ("cutedsl", "cutile", "tilelang"):
    result = materialize_flux2_klein_nvfp4_engine(
        pipe,
        engine=engine,
        target_arch="sm_89",
        max_targets=8,
        inplace=False,
    )
    print(engine, result.target_count)
```

## 注意事项

- FLUX.2 klein NVFP4 权重约 4B 参数, 实际整模推理需要足够显存. 模型卡给出的硬件提示是约 13GB VRAM.
- NVFP4 single-file 当前使用 modelopt 风格字段名, 例如 `weight`, `weight_scale`, `weight_scale_2`, `input_scale`. `load_flux2_klein_nvfp4_transformer()` 会用 BF16 主仓库的 `transformer/config.json` 构建 Diffusers transformer, 再把这些 packed Linear 映射为 XQT NVFP4 Linear shim; `weight_scale_2` 会按 compressed-tensors 约定转换为 `1 / weight_scale_2` 后进入统一 bridge.
- double-stream QKV 权重会从 `double_blocks.*.img_attn.qkv` / `txt_attn.qkv` 拆到 Diffusers 的 `to_q/to_k/to_v` / `add_q_proj/add_k_proj/add_v_proj`; single-stream `linear1/linear2` 会映射到 `single_transformer_blocks.*.attn.to_qkv_mlp_proj` / `to_out`.
- `load_flux2_klein_nvfp4_pipeline()` 会用 NVFP4 single-file transformer 搭配 BF16 主仓库的 tokenizer, scheduler, VAE 和 text encoder.
- `materialize_flux2_klein_nvfp4_engine()` 默认扫描 `pipeline.transformer`; 如果传入的是 `nn.Module`, target path 相对该模块.
- `max_targets` 可用于先替换少量 Linear 做 smoke, 再扩大到整模.
- `cutile` 和 `cute_dsl` 的 fallback 结果只能证明 XQT wrapper 和 NVFP4 bridge 语义闭环, 不能证明真实 DSL kernel 性能收益. 当前 `cutile` dense-cache fallback 是 runtime integration 优化, 不是 packed NVFP4 CuTile kernel 性能结论.
