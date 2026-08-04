# NVIDIA profiling playbook for XQT

Use this file when the task needs NVIDIA-side time analysis, resource analysis, or execution analysis for XQT kernels.
Do not copy full vendor docs into the answer. Extract only the decision logic needed for the current task.

## Scope

This skill treats profiling as a diagnosis layer around XQT:

- `benchmark` gives stable baseline latency, throughput, and optional memory numbers.
- `nsys` explains end-to-end time distribution and launch behavior.
- `ncu` explains why one kernel is slow on the GPU.

Do not replace XQT benchmark reports with profiler output. Keep them separate and attach profiler artifacts into the same experiment lineage.

## SM / compute capability map

Use the official NVIDIA CUDA GPU compute-capability page as the authority.

Useful mappings for current XQT planning:

| arch family | common CC | typical target arch string |
|---|---:|---|
| Ampere datacenter | 8.0 | `sm_80` |
| Ampere client / workstation | 8.6 | `sm_86` |
| Ada | 8.9 | `sm_89` |
| Hopper | 9.0 | `sm_90` |
| Blackwell datacenter | 10.0, 10.3 | `sm_100`, variant-specific follow-up |
| Blackwell client / workstation | 12.0 | `sm_120` |
| GB10 | 12.1 | `sm_121` |

Rules:

1. Never guess the target arch from model name memory. Verify with `nvidia-smi` and the official compute-capability table.
2. Carry the resolved `sm_*` into TileLang, CUTLASS, and CuTe DSL compile settings.
3. When comparing across architectures, keep shapes, dtypes, and benchmark parameters fixed.

## Workflow

### Step 1: establish a clean baseline

Use XQT benchmark or operator stage first.

Minimum capture:

- model / module name
- backend
- pattern
- input shape
- dtype
- batch size
- warmup
- iterations
- device name
- CUDA version
- resolved `sm_*`
- numeric-validation result and tolerance
- whether compile, autotune, allocation, or graph capture was excluded from the steady-state window

Reject noisy data:

- first-run compilation mixed into steady-state timing
- changing clocks or thermal throttling
- mixed shapes in one report
- asynchronous CUDA timing without explicit synchronization

### Step 2: run `nsys` for time analysis

Use `nsys` when the question is:

- where does wall time go?
- are kernels launch-bound?
- is host overhead dominating?
- are memcpy or sync points killing overlap?
- are CUDA Graph or stream choices the bottleneck?

Useful reports:

- `cuda_gpu_kern_sum`
- `cuda_api_sum`
- `cuda_gpu_mem_time_sum`
- `cuda_gpu_mem_size_sum`
- `cuda_gpu_trace` when per-kernel sequence matters

Typical command shape:

```bash
nsys profile -o artifacts/nsys/<name> --trace=cuda,nvtx,osrt python your_entry.py
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv,column --output .,- artifacts/nsys/<name>.nsys-rep
```

When a workload has several similar kernels, use an NVTX range or first record the demangled kernel name from this trace. A generated Triton or TileLang kernel name is not reliably the Python function name.

What to conclude from `nsys`:

- Too many small launches: kernel launch overhead or missing fusion.
- Large `cudaMemcpy*` time: layout, staging, or host-device traffic problem.
- Long CPU launch gaps: Python dispatch, graph breaks, synchronization, or executor structure issue.
- No overlap between memcpy and kernels: stream or dependency structure issue.

### Step 3: run `ncu` for resource and execution analysis

Use `ncu` when the question is:

- why is this kernel underperforming on one GPU?
- is the kernel memory-bound or compute-bound?
- is occupancy too low?
- which stall reasons dominate?
- are register pressure or shared-memory use blocking throughput?

Before selecting a section, query the installed tool:

```bash
ncu --list-sets
ncu --list-sections
```

Start with the basic set for the already selected kernel or NVTX range:

```bash
ncu --target-processes all --set basic --kernel-name-base demangled --kernel-name regex:<kernel> --launch-skip <N> --launch-count <M> --export artifacts/ncu/<name>-basic python your_entry.py
```

Then collect one follow-up section justified by the basic result:

- Occupancy
- Memory Workload Analysis
- Scheduler Statistics
- Warp State Statistics
- Roofline / Speed-of-Light views

Typical targeted command shape:

```bash
ncu --target-processes all --section MemoryWorkloadAnalysis --kernel-name-base demangled --kernel-name regex:<kernel> --launch-skip <N> --launch-count <M> --export artifacts/ncu/<name>-memory python your_entry.py
```

Use `ncu --query-metrics` before a custom metric list. Do not default to `--set full`: it can collect thousands of metrics through multiple replay passes and is unsuitable for the normal tuning loop. Use it only for an explicit deep-dive after narrowing the target and accepting the collection cost.

When iteration speed matters, collect the smallest evidence-justified section:

```bash
ncu --section Occupancy --kernel-name-base demangled --kernel-name regex:<kernel> --launch-skip <N> --launch-count <M> python your_entry.py
```

## Diagnosis map

### Symptom: occupancy is low

Check:

- register usage
- shared memory usage
- threads per block
- barrier count

Likely actions:

- reduce register pressure
- reduce smem footprint
- retune block size / tile shape
- split over-fused kernels if occupancy collapse outweighs fusion gains

### Symptom: scheduler eligible warps are low

Check:

- scoreboard stalls
- dependency chains
- instruction mix
- latency hiding

Likely actions:

- increase parallel work per SM
- improve pipelining
- reduce long dependency chains
- rebalance tile sizes

### Symptom: memory workload analysis shows poor coalescing or bank conflicts

Check:

- global memory access pattern
- shared memory layout
- vectorization
- stride and alignment

Likely actions:

- rewrite loads and stores for contiguous access
- change shared-memory swizzle / layout
- use vectorized transactions
- reorder tensor layout or fusion boundary

### Symptom: roofline places kernel on bandwidth slope far below roof

Check:

- arithmetic intensity
- cache reuse
- redundant reads and writes

Likely actions:

- fuse producer and consumer stages
- tile for reuse
- stage data in shared memory
- reduce intermediate materialization

### Symptom: roofline places kernel in compute region but far below peak

Check:

- tensor-core usage
- instruction selection
- issue efficiency
- pipeline bubbles

Likely actions:

- move to tensor-core-friendly tile shapes
- align dtype and fragment layout with backend best path
- reduce branchy epilogue logic in the hot loop

## Backend-specific hints

### Triton

- First inspect whether the problem is tile size, `num_warps`, `num_stages`, or layout assumptions.
- Favor autotune-ready designs for GEMM, norm, and pointwise fusion.
- Watch register growth when adding multiple fused epilogues.

### TileLang

- Always carry explicit `target_arch`.
- Use it first for `conv`, `attention`, and FP4/NVFP4 fused dequant GEMM.
- Separate compile-only validation from runtime validation in reports.

### CUTLASS / CuTe DSL

- Use them when architecture-aware tiling, cluster shape, or grouped GEMM structure matters.
- Treat them as design-space exploration paths in the current repo, not yet the default executor path.

### CuTile

- Use it for pointwise fusion experiments and simple fastpaths.
- Do not overstate current executor coverage.

## Required artifact lineage

Every serious tuning round should preserve:

- XQT stage report
- XQT benchmark output
- `nsys` report path
- `ncu` report path
- command shape or key flags
- environment summary
- accepted or rejected hypothesis
- numeric-validation result and tolerance
- resolved generated kernel name or NVTX range
- profile environment snapshot, including profiler and driver versions

The final recommendation should name:

1. the bottleneck,
2. the evidence,
3. the proposed optimization,
4. the expected risk,
5. the next measurement to validate the change.
