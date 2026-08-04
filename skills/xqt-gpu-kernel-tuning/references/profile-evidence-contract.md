# XQT profile evidence contract

Use this contract for a real XQT operator or kernel profiling round. It keeps a profiler run reproducible without adding a generic profiler subsystem to XQT.

## Artifact lineage

Create one artifact directory per profiling round. Preserve these logical records, whether the stage report already owns some of them or they are stored beside it:

```text
<artifact-root>/
  target.json
  environment.json
  baseline.json
  nsys/
    <run>.nsys-rep
    kernel-summary.csv
    api-summary.csv
  ncu/
    basic.ncu-rep
    <follow-up>.ncu-rep
  commands.md
  conclusion.md
```

`target.json` must name the XQT stage report or benchmark artifact, backend, operator pattern, exact input shapes and layouts, dtype, precision mode, GPU index, resolved `sm_*`, warmup, measured iterations, synchronization policy, and kernel filter or NVTX range. Do not include secrets or unrelated environment variables.

Capture the environment before measuring:

```bash
python .codex/skills/xqt-gpu-kernel-tuning/scripts/profile_environment.py > <artifact-root>/environment.json
python .codex/skills/xqt-gpu-kernel-tuning/scripts/ncu_permission_probe.py > <artifact-root>/ncu-permission.json
ncu --list-sets
ncu --list-sections
```

The installed profiler, not a remembered metric name, is the authority for the run. Use `ncu --query-metrics` only when a specific metric is needed and record the exact command in `commands.md`.

If `ncu-permission.json` reports `counter_permission_denied`, do not run another NCU stage or infer a bottleneck from missing counters. Preserve the result, use a time trace if it remains available, and tell the user that an administrator must enable the NVIDIA performance-counter policy before counter collection can continue.

## Measurement gates

Do not profile until all of these are true:

1. A same-shape, same-dtype XQT baseline exists and uses CUDA-event timing or an equivalent synchronized device timer.
2. Numeric validation has passed against the declared reference, with the recorded tolerance.
3. Compilation, Triton autotune, lazy allocation, and CUDA Graph capture have completed before the steady-state window.
4. The target workload is deterministic enough to identify its launches. If timing samples are noisy, report the run as inconclusive instead of using counters to explain noise.

Keep cold-start, compile, graph-capture, and steady-state measurements as distinct rows. Never use a profile trace as the latency benchmark of record.

## Semantic target selection

Use an existing XQT benchmark or a small profile-only launcher. Wrap only the requested operator invocation with both `torch.profiler.record_function("xqt/<target>")` and `torch.cuda.nvtx.range("xqt/<target>")` when the runtime permits it. This maps a framework-level target to generated kernel names without introducing a permanent profiler wrapper into XQT.

If an NVTX range is unavailable, run a short Nsight Systems trace first, record the hottest demangled kernel name and launch count, and use that exact filter for Nsight Compute. Do not run Nsight Compute against every process kernel or rely on a source-level Triton function name when generated names differ.

For CUDA C++ source attribution, compile the profiling build with `-lineinfo`; do not use `-G` for representative performance data. Treat Triton, TileLang, and generated-code source mapping as best effort unless the profiler proves it.

## Staged profiler collection

Use Nsight Systems to answer where time goes, then use Nsight Compute only on the selected kernel or range.

```text
Nsight Systems:
  CUDA APIs, GPU kernels, memcpy, NVTX, and OS runtime
  -> top kernels, launch gaps, copies, synchronization, overlap

Nsight Compute stage 1:
  --set basic on the selected steady-state launch window
  -> workload distribution, occupancy, speed-of-light direction

Nsight Compute stage 2:
  one evidence-justified section
  -> MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, Occupancy,
     SchedulerStats, WarpStateStats, or a matching roofline section
```

Do not use `ncu --set full` by default. It can request thousands of metrics and many replay passes, which makes an iterative loop slow or impractical. Use it only for an explicit deep-dive with a narrowed target and an accepted replay cost. Do not assume that a section or metric exists on every Nsight Compute version or every GPU.

| Basic evidence | Follow-up | Typical question |
|---|---|---|
| High DRAM direction or suspicious access efficiency | `MemoryWorkloadAnalysis` | Are loads, stores, cache reuse, or shared-memory layout the limit? |
| High SM direction but low useful issue rate | `ComputeWorkloadAnalysis` | Is tensor-core use, instruction mix, or pipeline utilization limiting? |
| Low achieved occupancy or a resource cap | `Occupancy` | Are registers, shared memory, block size, or barriers restricting residency? |
| Low eligible warps or clear dependency stalls | `SchedulerStats` or `WarpStateStats` | Is latency hiding, dependency depth, or synchronization the limit? |
| Need an arithmetic-intensity hypothesis | matching `SpeedOfLight*RooflineChart` | Is the kernel near the bandwidth or compute roof? |

Treat a missing counter, counter-permission failure, profiler replay failure, or ambiguous filter as a blocked measurement, not evidence of a kernel defect. Preserve the status and return to the benchmark or runtime-integration layer.

## Comparison and conclusion

Compare before and after only when target metadata, correctness policy, profiler selection, and environment are compatible. A conclusion must state:

1. the selected XQT target and generated kernel or range;
2. the stable benchmark delta and observed variation;
3. the profiler evidence and its limits;
4. one bottleneck hypothesis and one contained code change;
5. the next verification command and residual portability or numeric risk.

Hand off to `$xqt-operator-runtime-integration` when the microbenchmark is fast but the XQT operator stage is not, when graph replay changes the result, or when Nsight Systems shows host, copy, or capture overhead rather than a slow kernel body.
