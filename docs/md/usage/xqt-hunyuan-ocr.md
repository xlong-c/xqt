# HunyuanOCR SVD INT4 block inference

## 负责什么

本页说明 `tencent/HunyuanOCR` 的 XQT 模型侧入口 `optimize_hunyuan_ocr_svd_int4_blocks(...)`:

- 用 quant method `svd` 将每个选中的 `nn.Linear` 拆为双支路 additive 存储.
  - `low_rank`: 源精度低秩支路.
  - `quant_residual`: groupwise packed signed INT4 残差.
- 写出 Infer 交接面 `compute_config` (`compute_contract=composite_add`, `combine=add`).
- materialize residual 支路为 W8A8 INT8 MMA compute view.
- 在量化后识别外层 `nn.ModuleList` 的逻辑 block, 分别以 `torch.compile` materialize, 再通过一次真实模型前向完成 warmup.
- 返回原远程代码模型对象, 保留模型自身的 `generate` / `chat` 等推理 API.
- 支持仓库 `dflash/` 子目录中 Transformers-compatible DFlash 模型包的相同路径.

## 不负责什么

- 不创建 OCR 数据集, dataloader 或任务级准确率评估.
- 不重写 HunyuanOCR 的图像预处理, prompt 或 decode API.
- 不把 packed INT4 残差描述为 native INT4 MMA. residual 计算是 W4 storage 到 W8A8 INT8 MMA 的 retarget.
- 不把 block composition 描述为单个 fused block kernel. 每个逻辑 block 是独立 `torch.compile` target.
- 不在找不到量化 block 时退化为仅 Linear 级优化. helper 会显式报错.

## 三轴语义

| 轴 | 本路径取值 |
| --- | --- |
| quant method | `svd` (report / lineage) |
| strategy / storage | `w4a16_int4`, `svd_low_rank_plus_residual` (low-rank factors + packed_signed_int4_group_scale) |
| compute | `composite_add` -> branches: `fp16_mma` (low_rank) + `w4_storage_int8_mma` (quant_residual) |
| inference level | outer `nn.ModuleList` child block, compiled independently with `torch.compile` |

## Python 入口

`example_inputs` 必须是一次可直接调用 HunyuanOCR `model.forward` 的真实输入. helper 优先用它 warmup; 未提供时可使用 `calibration_inputs[0]`. 这是 block materialization 的必需条件, 因为 XQT 不会猜测远程代码的图像和 prompt 输入结构.

```python
from examples.xqt_models.hunyuan_ocr import (
    load_hunyuan_ocr,
    optimize_hunyuan_ocr_svd_int4_blocks,
)

model = load_hunyuan_ocr(device="cuda")
example_inputs = ...  # Tensor, positional tuple/list, or keyword mapping for model.forward
result = optimize_hunyuan_ocr_svd_int4_blocks(
    model,
    rank=32,
    group_size=128,
    engine="auto",
    block_engine="inductor",
    example_inputs=example_inputs,
    calibration_inputs=[example_inputs],
)
quantized_model = result.model
compute_config = result.compute_config
block_report = result.block_optimization.to_dict()
```

`HunyuanOCRBlockOptimization` 记录 block path, engine, dynamic shape 设置, wrapper 建立时间和 warmup 输入来源. quant stage metrics 也会写入 `hunyuan_ocr_block_optimization` 字段.

`quantized_model` 是原模型对象的模块替换版本. 调用方继续按 HunyuanOCR 仓库定义的 inference API 调用它. `torch.compile` materialization 是进程内状态; 持久化后重新加载量化模型时, 应再次调用此 helper 或 block compile 路径.

薄示例入口位于 [hunyuan_ocr_svd_int4_blocks.py](../../../examples/hunyuan_ocr_svd_int4_blocks.py). 示例使用 `trust_remote_code=True`, 因此应固定 revision 并在受信任环境中加载模型代码.

## TileLang Decode Pipeline

`HunyuanOcrTileLangDecodeBlock` 是面向 `sm_89` 的实验性 decode fastpath. 它执行完整的单 token text decoder block pipeline:

1. TileLang RMSNorm.
2. 已量化 `SVDQuantInt8MmaLinear` / `Int8MmaLinear` 的 Q/K/V W8A8 projection.
3. 调用方提供的 XD-RoPE transform, 再执行 TileLang Q/K RMSNorm.
4. 将当前 K/V 写入静态 cache 的最后一个 slot, 使用 TileLang GQA attention.
5. 已量化 O projection (输入宽度为 `attention_dim`, 不是 `hidden_size`), TileLang residual + RMSNorm.
6. 已量化 gate/up/down projection, TileLang SwiGLU 和最终 residual add.

### 真实 Q/O 维度契约 (tencent/HunyuanOCR)

发布权重的 text decoder 与常见 "hidden == heads * head_dim" 假设不同:

| 字段 | 取值 |
| --- | --- |
| `hidden_size` | 1024 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 |
| `head_dim` | 128 |
| `attention_dim` (`query_heads * head_dim`) | 2048 |
| `key_value_dim` (`key_value_heads * head_dim`) | 1024 |
| `intermediate_size` | 3584 |
| `q_proj` | 1024 -> 2048 |
| `k_proj` / `v_proj` | 1024 -> 1024 |
| `o_proj` | 2048 -> 1024 |
| `query_layernorm` / `key_layernorm` | 按 `head_dim=128` |

`HunyuanOcrTileLangDecodeSpec` 因此**不**要求 `hidden_size == query_heads * head_dim`. 它暴露:

- `attention_dim = query_heads * head_dim` (Q 输出 / O 输入宽度)
- `key_value_dim = key_value_heads * head_dim` (K/V 输出宽度)

reshape 契约:

- `q_proj` 输出 reshape 为 `[B, Hq, 1, head_dim]`
- `k/v_proj` 输出 reshape 为 `[B, Hkv, 1, head_dim]`
- attention 输出 flatten 为 `[B, 1, attention_dim]` 再进 `o_proj`
- residual / MLP 始终在 `hidden_size` 上

### Decode schedule (1 / 2 / 3)

单 token decode 默认走三条路径:

| 项 | 默认 | 作用 |
| --- | --- | --- |
| **1. M=1 INT8 no-pad** | 可选: `decode_min_int8_rows=0` | `rows==1` 时优先 `int8_gemv_m1_*` (`__dp4a`, 见 `xqt/operator_opt/kernels/cute/int8mma_kernel.cu`); 不可用时回退 float 域 int8 products. 相对 float GEMV 约 2-3x, 仍慢于 cuBLAS dense, 默认关闭 |
| **2. GQA tile=1** | `gqa_query_tile_rows=1` | 单 token 精确 GQA, 不 pad query 到 64; `64` 仍为 legacy TileLang MMA |
| **3. Projection fusion** | QKV / gate+up INT8 pack | 仅 `decode_min_int8_rows=0` 且 M=1 时启用; dense 默认路径保持逐 projection (更快) |

默认性能路径 (推荐):

| 字段 | 默认 | 约束 / 作用 |
| --- | --- | --- |
| `decode_min_int8_rows` | 16 | M=1 走 dense `bf16_fallback` (最快) |
| `gqa_query_tile_rows` | 1 | exact single-token GQA, 无 query pad |
| `int8_block_m` | 16 | 多 token / 强制 INT8 时使用 |

设 `decode_min_int8_rows=0` 可启用 M=1 INT8 no-pad GEMV + QKV/gate-up fusion (保留 W8A8 语义).

说明:

- sm_89 TileLang `T.gemm` 不能 `block_m=1`; M=1 真 INT8 用 `int8_gemv_m1_*` (DP4A, 见 `xqt/operator_opt/kernels/cute/int8mma_kernel.cu`).
- GQA `query_tile_rows` 仅支持 `{1, 64}`; 16/32 会 layout infer 失败.
- fusion 仅 `Int8MmaLinear` + 相同 static activation scale; SVD residual 回退逐 projection.

`HunyuanOcrTileLangCudaGraphRunner` 会在固定 cache 长度上捕获上述整段多核 pipeline. 每个 CUDA Graph 只对应一个精确 KV cache 长度, 长度必须是 `64` 的倍数; cache 长度改变时必须创建或选择另一个 graph. 这是为了保持 TileLang MMA 和 GQA softmax 的静态 layout, 不是通用动态 cache graph.

该入口不猜测 XD-RoPE 的 `position_ids` 或 Hugging Face `Cache` 状态. 调用方必须把官方 position transform 显式封装为 `position_transform(query, key)`, 并传入已物化的 exact-length K/V cache. 因此它当前是模型内部实验 API, 不替换 `generate` 的默认路径.

`runner.replay(hidden_states)` 返回 CUDA Graph 持有的输出 buffer, 下一次 replay 后即被覆盖. `benchmark_hunyuan_ocr_tilelang_decode_graph(...)` 测量同一 replay 层级的稳态延迟, 不额外计入输出 clone.

## DFlash 子模型

`tencent/HunyuanOCR/tree/main/dflash` 中的模型包可通过专用入口加载. 该入口将 Hugging Face `subfolder` 明确设为 `dflash`, 不复制或推断目录内的模型代码:

```python
from examples.xqt_models.hunyuan_ocr import (
    load_hunyuan_ocr_dflash,
    optimize_hunyuan_ocr_dflash_svd_int4_blocks,
)

dflash_model = load_hunyuan_ocr_dflash(device="cuda")
result = optimize_hunyuan_ocr_dflash_svd_int4_blocks(
    dflash_model,
    rank=32,
    group_size=128,
    example_inputs=example_inputs,
    calibration_inputs=[example_inputs],
)
```

`dflash/` 必须是可由 `transformers.AutoModel.from_pretrained(..., subfolder="dflash", trust_remote_code=True)` 加载的模型包. 若该目录改为非 Transformers artifact, loader 会返回底层加载错误而不会静默退回根模型.

## INT4 compute contract

量化 stage 的每个替换模块在 `compute_config["modules"]` 中声明类似:

```json
{
  "name": "model.layers.0.mlp.up_proj",
  "compute_contract": "composite_add",
  "combine": "add",
  "preferred_mode": "split",
  "precision": "w8a8",
  "required_capabilities": ["composite_add", "int8_mma", "fp16_mma"],
  "storage": {
    "kind": "svd_low_rank_plus_residual",
    "decomposition": "additive",
    "rank": 32,
    "group_size": 128,
    "quant_dtype": "int4"
  },
  "branches": [
    {
      "name": "low_rank",
      "compute_contract": "fp16_mma",
      "precision": "source_precision"
    },
    {
      "name": "quant_residual",
      "compute_contract": "w4_storage_int8_mma",
      "precision": "w8a8",
      "storage": {"format": "packed_signed_int4_group_scale", "quant_dtype": "int4"}
    }
  ]
}
```

`preferred_engines` (来自 helper 的 `engine=`) 只影响 INT8 residual compute resolve 排序, 不是 hard requirement. `block_engine` 是 `torch.compile` backend, 默认 `inductor`; 它与量化 compute engine 是两类配置.

低秩支路尚未与 residual GEMM 融合 (`preferred_mode=split`). 因此必须对完整 HunyuanOCR 的真实 prefill / decode shape 单独 benchmark, 不应把单层 INT8 kernel latency 当成端到端吞吐量.

## 源码锚点

| 主题 | 路径 |
| --- | --- |
| Hunyuan helper | `examples/xqt_models/hunyuan_ocr.py` |
| SVD quant method | `xqt/quant/quantizers/svd.py` |
| Runtime dual-branch modules | `xqt/runtime/modules/svd_w4a4_legacy.py` |
| Residual INT8 module | `xqt/runtime/modules/w4_storage_int8_mma_linear.py` |
| torch.compile backend | `xqt/operator_opt/compile_backend.py` |
| TileLang Hunyuan decode pipeline | `examples/xqt_models/hunyuan_ocr_tilelang.py` |
| TileLang Hunyuan kernels | `xqt/operator_opt/kernels/tilelang/hunyuan_block.py` |

## 验证

```bash
pytest -q tests/xqt/quant/test_quant_svd_method.py tests/xqt/test_hunyuan_ocr_svd_quant.py tests/xqt/test_operator_tilelang_hunyuan_block.py tests/xqt/test_hunyuan_ocr_tilelang_decode.py
python examples/hunyuan_ocr_svd_int4_blocks.py
```

第二条命令需要先在 `CONFIG["example_inputs"]` 填入真实模型输入, 本地模型缓存或可访问 Hugging Face, 以及足够的 CPU/GPU 内存. 完整 OCR 数值和性能验收应使用调用方提供的真实输入, 并比较量化前后的模型输出和 block-level 端到端 benchmark.
