# HunyuanOCR SVD composite_add (FP4 residual storage)

## 负责什么

本页说明 `tencent/HunyuanOCR` 的 XQT 模型侧量化入口:

- 用 quant method `svd` 将每个选中的 `nn.Linear` 拆为 **双支路 additive 存储**:
  - `low_rank`: 源精度低秩支路
  - `quant_residual`: groupwise packed W4 残差
- 写出 Infer 交接面 `compute_config` (`compute_contract=composite_add`, `combine=add`).
- 可选 materialize residual 支路到 W8A8 INT8 MMA (默认开启, 兼容现有 helper).
- 返回原远程代码模型对象, 保留模型自身的 `generate` / `chat` 等推理 API.
- 支持仓库 `dflash/` 子目录中 Transformers-compatible DFlash 模型包的相同路径.

## 不负责什么

- 不创建 OCR 数据集, dataloader 或任务级准确率评估.
- 不重写 HunyuanOCR 的图像预处理, prompt 或 decode API.
- 不把 packed W4 残差描述为 native FP4 MMA. residual 计算是 W4 storage 到 W8A8 INT8 MMA 的 retarget.
- 不把 `engine` 写成 Infer 必选主键 (仅 preferred_engines hint).

## 三轴语义

| 轴 | 本路径取值 |
| --- | --- |
| quant method | `svd` (report / lineage) |
| storage | `svd_low_rank_plus_residual` (low-rank factors + packed_signed_int4_group_scale) |
| compute | `composite_add` → branches: `fp16_mma` (low_rank) + `w4_storage_int8_mma` (quant_residual) |

`strategy=svd_fp4_int8_mma` 仍是兼容别名, 不是新能力主键.

## Python 入口

在有模型权重和 `transformers` 的环境中使用:

```python
from xqt.model.hunyuan_ocr import (
    load_hunyuan_ocr,
    optimize_hunyuan_ocr_svd_fp4_int8_mma,
)
from xqt.runtime import HybridInferenceEngine, materialize_composite_compute

model = load_hunyuan_ocr(device="cuda")
result = optimize_hunyuan_ocr_svd_fp4_int8_mma(
    model,
    rank=32,
    group_size=128,
    engine="auto",
)
quantized_model = result.model
compute_config = result.compute_config
```

`quantized_model` 是原模型对象的模块替换版本. 调用方继续按 HunyuanOCR 仓库定义的 inference API 调用它.

若需要严格 quant/infer 解耦 (只出存储壳, 再由 Infer materialize):

```python
# policy 侧: materialize_compute=False 时 quant 只写 SVDQuantLinear 壳 + compute_config
# 随后:
# model = materialize_composite_compute(model, compute_config)
# engine = HybridInferenceEngine.from_quantized_model(quant_result)
```

薄示例入口位于 [hunyuan_ocr_svd_fp4_int8_mma.py](../../../examples/hunyuan_ocr_svd_fp4_int8_mma.py). 示例使用 `trust_remote_code=True`, 因此应固定 revision 并在受信任环境中加载模型代码.

## DFlash 子模型

`tencent/HunyuanOCR/tree/main/dflash` 中的模型包可通过专用入口加载. 该入口将 Hugging Face `subfolder` 明确设为 `dflash`, 不复制或推断目录内的模型代码:

```python
from xqt.model.hunyuan_ocr import (
    load_hunyuan_ocr_dflash,
    optimize_hunyuan_ocr_dflash_svd_fp4_int8_mma,
)

dflash_model = load_hunyuan_ocr_dflash(device="cuda")
result = optimize_hunyuan_ocr_dflash_svd_fp4_int8_mma(
    dflash_model,
    rank=32,
    group_size=128,
)
```

`dflash/` 必须是可由 `transformers.AutoModel.from_pretrained(..., subfolder="dflash", trust_remote_code=True)` 加载的模型包. 若该目录改为非 Transformers artifact, loader 会返回底层加载错误而不会静默退回根模型.

## 静态校准

默认残差分支采用动态 activation scale. 若调用方提供实际模型输入的 `calibration_inputs`, helper 会默认改用静态 scale. XQT 在替换模块前对选中的 `nn.Linear` 注册 hook, 以每层输入绝对值最大值除以 `127` 计算对称 INT8 scale. 这能启用已存在的 TileLang static activation fusion.

`calibration_inputs` 必须是可直接调用原 HunyuanOCR model 的输入结构, 例如 Tensor, positional tuple/list 或 keyword mapping. 它不描述数据集来源. 报告仅记录输入 shape 和原始 dtype; BF16 输入不会被强制转换为 NumPy.

## 推理契约 (composite_add)

量化 stage 的 `compute_config` 对每个替换模块声明类似:

```json
{
  "name": "vision_encoder",
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
    "quant_dtype": "fp4"
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
      "storage": {"format": "packed_signed_int4_group_scale"}
    }
  ]
}
```

Infer 侧:

- 消费 `model + compute_config` (见 [xqt-infer-handoff.md](../architecture/xqt-infer-handoff.md)).
- 通过 `materialize_composite_compute` / `HybridInferenceEngine.from_quantized_model` 绑 residual 计算路径.
- `preferred_engines` (来自 helper 的 `engine=`) 只影响 resolve 排序, 不是硬失败主键.

低秩支路尚未和 residual GEMM 融合 (`preferred_mode=split`). 因此必须对完整 HunyuanOCR 的真实 prefill / decode shape 单独 benchmark, 不应把单层 INT8 kernel latency 当成端到端吞吐量.

## 源码锚点

| 主题 | 路径 |
| --- | --- |
| Hunyuan helper | `xqt/model/hunyuan_ocr.py` |
| SVD quant method | `xqt/quant/quantizers/svd.py` |
| Runtime dual-branch modules | `xqt/runtime/modules/svd_composite.py` |
| Residual INT8 module | `xqt/runtime/modules/w4_storage_int8_mma_linear.py` |
| Composite materialize | `xqt/runtime/composite_materialize.py` |

## 验证

```bash
pytest -q tests/xqt/quant/test_quant_svd_method.py tests/xqt/test_hunyuan_ocr_svd_quant.py
python examples/hunyuan_ocr_svd_fp4_int8_mma.py
```

第二条命令需要本地模型缓存或可访问 Hugging Face, 以及足够的 CPU/GPU 内存. 完整 OCR 数值和性能验收应使用调用方提供的真实输入, 并比较量化前后的模型输出.
