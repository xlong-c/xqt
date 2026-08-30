# Precision strategy for XQT GPU kernel tuning

Use this file when the task asks which input precision to choose, how to validate low precision, or how to tune different precisions on NVIDIA GPUs.

## First principle

Treat "precision choice" as a three-part decision:

1. hardware support on the target `sm_*`,
2. XQT backend support today,
3. numeric risk budget for the operator family.

Do not choose precision from hardware alone.

## Current XQT support snapshot

### Triton GEMM dispatcher

Current unified GEMM precision support in `xqt.kernels.ops._impl.gemm_precision._gemm_triton`:

- `fp16`
- `bf16`
- `int8`
- `fp8`
- `int4`
- `mxfp8`
- `mxfp6`
- `mxfp4`

Implication:

- Triton is the broadest precision exploration surface in the current repo.
- For `linear` and GEMM-heavy tuning, start here unless a TileLang path already exists for the exact fused operator you care about.

### TileLang GEMM dispatcher

Current generic TileLang GEMM dispatcher support:

- `fp16`

Implication:

- Generic TileLang GEMM is still `fp16`-first.
- TileLang low-precision work in the current repo mainly comes through explicit fused kernels, especially:
  - `dequant_gemm_epilogue`
  - `fp4_packed_dequant_gemm_epilogue`
  - `nvfp4_packed_dequant_gemm_epilogue`

### TileLang numeric thresholds already in repo

Current defaults:

- `float32`: `atol=1e-5`, `rtol=1e-5`
- `float16`: `1e-3`, `1e-3`
- `bfloat16`: `1e-2`, `1e-2`
- `float8_e4m3fn`: `5e-2`, `5e-2`
- `float8_e5m2`: `1e-1`, `1e-1`

Use these as starting points, not universal truth.

## NVIDIA-oriented precision map

Use the official compute-capability table as the architecture anchor.

Practical guidance:

- `fp16`: safe default performance path on modern NVIDIA GPUs.
- `bf16`: preferred over `fp16` when dynamic range matters and hardware supports it well.
- `fp8`: treat Hopper and newer as the main target; Ada support is stack-dependent and should be validated per library path.
- `nvfp4` / microscaling formats: Blackwell-first path.
- `int8`: still strong for inference-style GEMM if quantization calibration and scaling are clean.
- `int4` / `fp4`: use only when memory or bandwidth pressure dominates and you can tolerate more engineering and validation cost.

Relevant external facts:

- NVIDIA's compute capability table is the source for mapping GPUs to `sm_*`.
- NVIDIA Transformer Engine docs state Hopper introduced FP8, and Blackwell added NVFP4 and MXFP8.
- Triton docs show FP8 paths tied to newer compute capabilities and block-scaled low-precision matmul on Blackwell-class hardware.

## Decision matrix

### FP32

Use when:

- building the reference path,
- validating correctness,
- debugging a new kernel,
- checking whether a low-precision regression is numeric or scheduling related.

Do not optimize FP32 first unless the real deployment path is FP32.

Profiling focus:

- instruction throughput,
- memory bandwidth,
- whether FP32 is masking a low-precision-friendly layout issue.

### FP16

Use when:

- you want the first serious custom-kernel performance path,
- the model is inference-heavy and stable in half precision,
- you need a strong baseline before trying FP8 or FP4.

Why it is useful:

- easiest path to stable tensor-core throughput,
- lowest numeric risk among the fast paths in the repo,
- best starting point for `conv`, `linear`, and `attn` custom kernels.

Profiling focus:

- tensor-core utilization,
- shared-memory tiling,
- register pressure after fusion,
- launch overhead for small shapes.

### BF16

Use when:

- the operator is sensitive to range,
- FP16 overflows or produces unstable outliers,
- you still want tensor-core-class throughput.

Default policy:

- prefer BF16 over FP16 for attention, norm-adjacent paths, and large-activation transformers when the GPU supports it well.

Profiling focus:

- whether BF16 materially changes occupancy or register use,
- whether kernel shape needs retuning versus FP16.

### FP8

Use when:

- the deployment target is Hopper or Blackwell first,
- GEMM or attention throughput is the main bottleneck,
- the scaling recipe is under control.

Do not start with FP8 if:

- the FP16 or BF16 path is not yet healthy,
- scales and format choices are still unclear,
- the kernel already fails correctness in higher precision.

Practical policy:

- first make FP16 or BF16 fast,
- then port to FP8,
- keep accumulation and validation conservative,
- compare both latency and accuracy drift.

Profiling focus:

- tensor-core saturation,
- scale loading overhead,
- extra dequant or rescale stages,
- whether the kernel moved from compute-bound to memory-bound.

### INT8

Use when:

- weight or activation quantization is already available,
- inference throughput matters more than minimal engineering complexity,
- calibration artifacts are available and stable.

Key risk:

- performance may look fine while accuracy loss comes from poor scaling or zero-point handling rather than the kernel itself.

Profiling focus:

- dequant overhead,
- memory traffic from scales,
- whether packing format hurts coalescing,
- fused epilogue versus separate dequant+matmul stages.

### INT4 / FP4 / NVFP4 / MXFP*

Use when:

- bandwidth or memory footprint is the dominant bottleneck,
- you are targeting Blackwell-class or explicitly low-bit inference paths,
- you are willing to pay more validation and tuning cost.

XQT-local guidance:

- use TileLang for existing packed FP4 or NVFP4 fused dequant GEMM paths,
- use Triton for current microscaling GEMM exploration,
- treat these paths as architecture-sensitive and shape-sensitive.

Profiling focus:

- unpack overhead,
- scale fetch overhead,
- whether the kernel is truly fused into one hot path,
- whether low-bit packing saves enough memory traffic to offset added logic.

## Operator-family policy

### `conv`

- start with `fp16`
- move to `bf16` if dynamic range or accumulation stability needs help
- try lower precision only after the kernel shape and memory path are already healthy

### `linear`

- start with Triton `fp16` or `bf16`
- move to `fp8` or `int8` when GEMM dominates latency
- move to `int4` / `fp4` / `nvfp4` only when bandwidth is the bottleneck and scales are ready

### `attn`

- start with `bf16` or `fp16`
- only move to `fp8` once the fused attention path is already numerically stable
- keep softmax, scaling, and accumulation paths under tighter validation

### `norm`

- prefer `bf16` over `fp16` when numerics are marginal
- low-bit norm kernels need stricter drift checks than GEMM

### fusion / megakernel

- first make the unfused or lightly fused `fp16` or `bf16` path healthy
- then add low-precision fusion one edge at a time
- watch register explosion and occupancy collapse carefully

## Recommended optimization order

Use this order unless the user explicitly says otherwise:

1. `fp32` reference correctness
2. `fp16` fast path
3. `bf16` if range matters
4. `fp8` for Hopper or Blackwell-first targets
5. `int8` if quantized inference path is ready
6. `int4` / `fp4` / `nvfp4` / `mxfp*` only after scales, packing, and fusion structure are stable

## Validation policy

For every lower-precision step:

1. compare against a higher-precision reference,
2. record `max_abs`, `mean_abs`, and allclose-like status,
3. separate compile-only validation from runtime validation,
4. re-run benchmark after correctness passes.

For `fp8` and below, do not accept "fast but noisy" without an explicit user-approved error budget.

## What to do in practice

If the user asks "what precision should I use?" answer like this:

1. identify operator family,
2. identify target `sm_*`,
3. identify whether the bottleneck is compute, bandwidth, or memory footprint,
4. choose the highest precision that still addresses the bottleneck,
5. only then step down precision.

That usually means:

- first choice: `fp16`
- safer transformer choice: `bf16`
- Hopper/Blackwell throughput push: `fp8`
- quantized inference path: `int8`
- bandwidth-extreme path: `fp4` / `nvfp4` / `mxfp*`
