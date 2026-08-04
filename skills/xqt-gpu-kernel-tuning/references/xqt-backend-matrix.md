# XQT operator engine matrix

This file is the local fact source for XQT operator optimization engines and registered kernel patterns. Read it before proposing a backend path or a profiling workflow that assumes a path is executable.

Run `python .codex/skills/xqt-gpu-kernel-tuning/scripts/dump_xqt_operator_inventory.py` to resolve the current environment-dependent `available` values and registered patterns from code. Do not infer availability from this summary.

## Capability summary

Current engine status and maturity from `list_operator_engine_capabilities()`:

| engine | status | maturity | runtime | exportable | profiling implication |
|---|---|---|---|---:|---|
| `torch_compile` | `available` | `executable` | `pytorch` | no | Generic baseline and graph-break inspection path. |
| `triton` | `available` | `executable` | `pytorch` | no | Built-in paths cover RMSNorm, `xqt.nn.FeedForward`, true W8A8 INT8 GEMM, and composed low-bit linear/dequant paths. |
| `tilelang` | `available` | `executable` | `pytorch` | no | Primary custom-kernel path for attention, conv, direct half linear/norm, and dense/dequant GEMM patterns. Some low-bit paths still require target-hardware validation. |
| `cutile` | `planned` | `reference_guarded` | `pytorch` | no | Materializes reference-guarded paths; do not call it a real CuTile execution path without target-hardware evidence. |
| `cutlass` | `planned` | `metadata_only` | `pytorch` | no | Metadata and reference fallback only; a design-space target rather than the default profiler target. |
| `cute_dsl` | `planned` | `reference_guarded` | `pytorch` | no | Reference-guarded dense GEMM epilogue bridge; package, CUDA, and architecture sensitive. |
| `custom_cuda` | `planned` | `planned` | `pytorch` | no | `torch.library` registration exists, but the base executor does not build or load a CUDA extension. |
| `deployment_engine` | `planned` | `metadata_only` | `deployment_engine` | yes | Deployment-runtime bridge, not a PyTorch custom-kernel tuning target. |

Rules:

1. Keep `status`, `maturity`, and runtime availability separate. An importable package or a reference fallback is not proof that a custom kernel executed.
2. Profile only an executable path unless the task explicitly asks to build a new backend executor.
3. Quote `execution_mode` and `execution_reason` from the XQT operator report when they exist.

## Registered kernel patterns

### `triton`

Current registered patterns:

- `bias_gelu`
- `geglu`
- `gemm_bf16`
- `gemm_fp16`
- `gemm_fp8`
- `gemm_int4_dequant`
- `gemm_int8`
- `gemm_mxfp4`
- `gemm_mxfp6`
- `gemm_mxfp8`
- `gemm_nvfp4_packed_dequant`
- `rmsnorm`
- `rmsnorm_channel_first`
- `rmsnorm_residual`
- `rope`
- `swiglu`

Implication:

- Triton is a strong current target for GEMM, W8A8, norm, activation epilogues, and transformer-side pointwise fusion.
- Low-bit paths may be a composed unpack/dequant plus dense GEMM runtime, rather than one fused tensor-core kernel. Confirm the active `execution_mode` before attributing time to the Triton kernel body.

### `tilelang`

Current registered patterns:

- `attention`
- `conv`
- `conv3d_1x1x1`
- `dense_linear_epilogue`
- `dequant_gemm_epilogue`
- `fp4_packed_dequant_gemm_epilogue`
- `int8_linear`
- `int8_linear_static_activation`
- `int8_mma`
- `linear`
- `linear_marlin`
- `mxfp4_packed_dequant_gemm_epilogue`
- `norm`
- `nvfp4_packed_dequant_gemm_epilogue`

Implication:

- TileLang is the primary XQT custom-kernel target for attention, conv, linear, norm, and fused dequant GEMM work on NVIDIA.
- Direct half paths and the documented attention/conv shapes are executable today; packed FP4 and related low-bit work still requires real CUDA correctness and performance validation.

### `cutile`

Current registered patterns:

- `attention`
- `bias_silu`
- `conv`
- `dense_linear_epilogue`
- `dequant_gemm_epilogue`
- `fp4_packed_dequant_gemm_epilogue`
- `linear`
- `norm`
- `nvfp4_packed_dequant_gemm_epilogue`

Implication:

- CuTile mirrors much of the TileLang catalog, but its current XQT materialization is reference-guarded.
- Use it for backend-expansion work only after confirming that the target run used real CuTile codegen.

### `cutlass` and `cute_dsl`

Current registered patterns for both engines:

- `gemm_epilogue`
- `grouped_gemm`

Implication:

- They are the right design space for architecture-specific GEMM or grouped-GEMM work.
- Current XQT maturity does not make them the default execution or profiling path.

### `custom_cuda`

The current custom-op catalog contains `bias_gelu`, but the base package does not compile or load an extension. Treat it as an implementation-expansion target, not a benchmark target.

## Existing executable families

- `conv`: TileLang.
- `linear` / GEMM: Triton dense/W8A8 and low-bit composed paths, plus TileLang direct and dequant patterns.
- `attn`: TileLang.
- `norm`: Triton RMSNorm variants and TileLang direct half norm.
- fusion blocks: Triton `bias_gelu`, `geglu`, `swiglu`, `rope`; TileLang dequant GEMM epilogues.

## Skill guidance derived from the matrix

1. Prefer `tilelang` for executable XQT attention, conv, direct half linear/norm, and dequant or packed-GEMM tuning work.
2. Prefer `triton` for executable XQT GEMM, W8A8, RMSNorm, feedforward, and pointwise-fusion work.
3. Treat `cutile`, `cutlass`, `cute_dsl`, and `custom_cuda` as backend-expansion targets unless the report proves a real target-specific executor ran.
4. Keep the distinction between `available`, `planned`, `reference_guarded`, and `metadata_only` visible in every recommendation.
