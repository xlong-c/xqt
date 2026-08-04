---
name: xqt-gpu-kernel-tuning
description: Profile, diagnose, and tune XQT custom GPU kernels and backend kernel implementations on NVIDIA platforms. Use when Codex needs to optimize `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, or `custom_cuda` kernels in XQT; analyze kernel time, launch overhead, occupancy, memory behavior, warp stalls, or roofline position with `nsys` or `ncu`; compare kernel behavior across `sm_*` targets; extend XQT with new `conv`, `linear`, `attn`, `norm`, or fused megakernel operators; or turn kernel-level benchmarks and profiler artifacts into concrete optimization hypotheses. Do not use this skill first for wrapper latency, CUDA Graph replay, or microbenchmark-versus-operator mismatches.
---

# XQT GPU kernel tuning

Use this skill to keep GPU operator tuning in XQT aligned with the repository's current workflow:

- benchmark first,
- profile second,
- change one thing,
- remeasure,
- record evidence into XQT artifacts and reports with a reproducible environment and target record.

Do not turn XQT into a generic profiler wrapper, training loop, or vendor-CLI abstraction layer.

## Quick start

1. Read [references/xqt-backend-matrix.md](references/xqt-backend-matrix.md) to see which XQT backends and operator patterns already exist.
2. If the task mentions CUDA Graph, operator-stage reports, wrapper latency, or a mismatch between microbenchmark and end-to-end operator results, use `$xqt-operator-runtime-integration` first.
3. If the task needs a real NVIDIA trace or counter collection, read [references/nvidia-profiling-playbook.md](references/nvidia-profiling-playbook.md) and [references/profile-evidence-contract.md](references/profile-evidence-contract.md).
4. If the task asks which precision to use or how to tune different precisions, read [references/precision-strategy.md](references/precision-strategy.md).
5. If the task is about how to structure the workflow, skim [references/external-skill-survey.md](references/external-skill-survey.md).
6. Run `python .codex/skills/xqt-gpu-kernel-tuning/scripts/dump_xqt_operator_inventory.py` when you need the current backend and kernel inventory from code, not memory.
7. For an actual profiling round, save `python .codex/skills/xqt-gpu-kernel-tuning/scripts/profile_environment.py` and `python .codex/skills/xqt-gpu-kernel-tuning/scripts/ncu_permission_probe.py` output next to the XQT benchmark and profiler artifacts. If the probe reports `counter_permission_denied`, stop NCU collection and report the environment block.
8. Keep every recommendation grounded in one of these:
   - XQT benchmark output
   - XQT operator report
   - `nsys` evidence
   - `ncu` evidence
   - current backend capability state

## Workflow

### 1. Scope the tuning target

Identify:

- backend: `triton`, `tilelang`, `cutlass`, `cute_dsl`, `cutile`, `custom_cuda`, or `torch_compile`
- operator family: `conv`, `linear`, `attn`, `norm`, or fusion / megakernel
- target module or pattern
- dtype and precision mode
- target GPU and resolved `sm_*`
- semantic profiler scope: a stable NVTX range or a resolved demangled kernel filter

Do not guess which backend is appropriate. Check the local matrix first.

### 2. Establish a benchmark baseline in XQT

Prefer existing XQT flows:

- `XQTOptimizationSession.benchmark(...)`
- `XQTOptimizationSession.operator(...)`
- YAML workflow `benchmark` and `operator` stages

At minimum, retain:

- artifact directory
- input shape
- warmup
- iterations
- sync mode
- latency summary
- backend and pattern

Before collecting a trace or counters, retain the numeric-validation result and tolerance for the same candidate. Separate compile, Triton autotune, CUDA Graph capture, and steady-state timing; profiler overhead is not the latency benchmark of record.

If a task asks for profiling before any stable baseline exists, fix the baseline first.

If the task is about precision choice, establish both:

- a higher-precision reference path,
- the target lower-precision path to compare against it.

Before opening `nsys` or `ncu`, classify which layer is actually failing:

- `microbenchmark slow` and `operator-stage slow`:
  likely kernel or schedule work; proceed to profiling.
- `microbenchmark fast` but `operator-stage slow`:
  likely executor, wrapper, input materialization, graph capture scope, or benchmark methodology; use `$xqt-operator-runtime-integration` first.
- `eager path wins but graph path loses`:
  first inspect CUDA Graph dynamic or static state, replay copies, and capture scope before changing kernel tiles.
- `numeric drift appears only in graph mode`:
  first inspect graph state ownership and replay inputs before loosening thresholds.

### 3. Map the XQT target to profiler evidence

Prefer an existing XQT benchmark or a small profile-only launcher. Name the target with `torch.profiler.record_function` and `torch.cuda.nvtx.range` when practical, then profile that range or its resolved generated kernel. Do not add a permanent generic profiler wrapper to XQT merely to get a trace.

If the target has no stable semantic range or kernel filter, run a short `nsys` trace first. Record the hottest demangled kernel, launch window, and why it corresponds to the XQT target before opening `ncu`.

### 4. Choose the analysis layer

Use this decision tree:

- Need end-to-end time split, launch overhead, memcpy, sync, graph behavior, or CPU gaps:
  use `nsys`.
- Need occupancy, memory hierarchy behavior, warp stalls, scheduler behavior, or roofline:
  use `ncu`.
- Need both:
  run `nsys` first to isolate hot kernels, then narrow `ncu` to the hottest kernels.

Do not jump into `nsys` or `ncu` when the current evidence points to runtime integration. Delegate those cases to `$xqt-operator-runtime-integration`.

### 5. Collect profiler evidence in stages

For time analysis, capture `cuda`, `nvtx`, and host-runtime activity with `nsys`, then save the kernel, CUDA API, and memory summaries beside the XQT stage report.

For kernel analysis, use this order:

1. Query the installed profiler with `ncu --list-sets` and `ncu --list-sections`; metric availability is version and GPU dependent.
2. Collect `ncu --set basic` only for the selected steady-state kernel or NVTX range.
3. Add exactly one evidence-justified section such as `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `Occupancy`, `SchedulerStats`, `WarpStateStats`, or an appropriate roofline section.

Do not default to `ncu --set full`. It can require thousands of metrics and many replay passes. A missing counter, counter-permission denial, replay failure, or ambiguous kernel filter is a blocked measurement, not a kernel diagnosis.

### 6. Form a single bottleneck hypothesis

Good hypothesis examples:

- "`conv_tilelang` on `sm_89` is launch-bound because the kernel is too small and fusion is insufficient."
- "`gemm_fp16` Triton path is memory-bound due to uncoalesced loads and low arithmetic intensity."
- "`rmsnorm_residual` is register-limited and occupancy collapses after adding one more fused epilogue."
- "`fp4_packed_dequant_gemm_epilogue` is paying too much unpack overhead on `sm_90` and should become a single fused megakernel."
- "`attention` microbenchmark is slow on `sm_89` because launch overhead dominates the kernel body and fusion is still insufficient."
- "`norm` TileLang path is slower than native because the kernel is too small to amortize launch cost on this shape."
- "`linear` regression on `sm_90` appears after changing tile shape, suggesting occupancy or memory behavior changed rather than runtime integration."

Bad hypothesis examples:

- "The kernel is slow."
- "Try a lot of tuning."
- "Maybe Triton is bad on this GPU."

### 7. Apply one contained change

Examples:

- change tile shape
- change block size
- change `num_warps` or `num_stages`
- change shared-memory layout or swizzle
- change vectorization width
- split or add one fusion edge
- switch `target_arch`
- move one operator family from one backend to another

Do not batch unrelated tuning changes into one patch unless the task explicitly requests a broad rewrite.

### 8. Re-run benchmark and compare

Judge the change using:

- latency delta
- throughput delta
- memory delta if relevant
- numeric correctness
- profiler evidence if the change was motivated by `nsys` or `ncu`

If the change fails:

- keep the evidence,
- state why it failed,
- propose the next best hypothesis.

For small TileLang operator targets such as `conv`, `linear`, `attention`, and `norm`, prefer steady-state batched measurements over single-call host-wall timing. If you suspect the benchmark methodology itself is the issue, hand off to `$xqt-operator-runtime-integration`.

## Backend guidance

### Triton

Prefer Triton for:

- GEMM and linear paths
- RMSNorm or residual-style norm fusion
- RoPE and activation epilogues
- pointwise fusion blocks

Check:

- kernel registry in the local matrix
- tile size
- `num_warps`
- `num_stages`
- register pressure

### TileLang

Prefer TileLang for:

- `conv`
- `attention`
- dense linear epilogues
- FP4 / NVFP4 dequant GEMM fusion
- larger fused NVIDIA-specific megakernels

Always carry explicit `target_arch` for NVIDIA tuning work.

For TileLang work in XQT, focus here when the problem still looks kernel or profiler related:

- TileLang kernel registry and metadata
- shape-dependent launch behavior
- tile shape, `threads`, and `num_stages`
- occupancy, memory, and launch-cost evidence from `nsys` or `ncu`

### CUTLASS / CuTe DSL

Use these when the task is to expand XQT backend coverage or build architecture-specific GEMM paths.

Current repository reality:

- kernel specs and metadata exist,
- general built-in execution coverage is still limited compared with TileLang,
- these are design-space expansion targets, not the default answer for every tuning request.

### CuTile

Use CuTile for smaller pointwise fusion experiments and backend expansion work.
Do not oversell current scope.

## Multi-SM rules

When the task compares or expands across architectures:

1. Resolve the actual compute capability using the NVIDIA official mapping.
2. Map it to `sm_*` with `python .codex/skills/xqt-gpu-kernel-tuning/scripts/sm_arch_map.py` if needed.
3. Keep benchmark shapes, dtype, and methodology fixed across SM targets.
4. Report architecture-sensitive findings explicitly, for example:
   - "works on `sm_89`, regresses on `sm_90` due to occupancy"
   - "needs a different tile shape on `sm_120`"

## Reporting shape

When answering with results, use this structure:

1. target:
   backend, operator family, module or pattern, shape, dtype, GPU, `sm_*`
2. evidence:
   baseline benchmark, numeric validation, environment snapshot, and `nsys` or `ncu` findings
3. bottleneck:
   one sentence
4. optimization:
   one contained change or one ranked shortlist
5. verification:
   how to remeasure in XQT
6. residual risk:
   numeric drift, portability, compile instability, or architecture sensitivity

## Boundaries

Do not:

- add a training loop
- add dataset or dataloader ownership
- hide vendor profiler requirements behind fake abstraction
- invent unsupported backend maturity
- claim `planned` backends are already full executors

Do:

- attach profiler artifacts to XQT experiment lineage
- keep benchmark and profiler roles separate
- preserve the target, environment, command shape, kernel filter or NVTX range, and profiler-status record for every serious round
- preserve the distinction between executable paths and metadata-only paths
- prefer repository evidence over memory
- hand off runtime integration mismatches to `$xqt-operator-runtime-integration` before opening low-level profiler loops
