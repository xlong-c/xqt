---
name: gpu-profiling
description: How to profile CUDA kernels in this workspace with torch.profiler and ncu. Use when asked why a kernel is slow, when benchmark numbers need bottleneck attribution, or when writing profiling evidence for xqt.gemm manifests/docs.
---

# GPU Profiling (torch.profiler + ncu)

Rules learned from xqt.gemm SM89 work. CUDA event medians are a promotion gate, not bottleneck analysis; never write "why it is slow" into docs/manifests without profiler evidence.

## torch.profiler

- Wrap the exact workload (same shape, same tensors as the benchmark gate) in `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])`.
- Summarize with `prof.key_averages().table(sort_by="cuda_time_total", row_limit=N)`. Persist both a text table and a JSON summary into `research/xqt-gemm/artifacts/` so docs can cite a re-runnable artifact.
- Watch for hidden device sync in adapters: `isfinite()`, `.item()`, `.cpu()`, `nonzero()` show up as `Memcpy DtoH` and serialize the stream. Distinguish kernel time from adapter/sync overhead before blaming the kernel.

## ncu (Nsight Compute)

- Never run ncu unfiltered against a script that also does warmup/elementwise work: it will profile the wrong kernel (e.g. an `isfinite` elementwise kernel instead of the CUTLASS GEMM). Filter by name:
  - `ncu --kernel-name "regex:.*cutlass.*" --kernel-name-base demangled --launch-skip-before-match 0 ...`
  - Check exact flags on the target machine first: `ncu --help | grep -A2 -e kernel-name -e launch-skip`.
- Prefer a pure native workload for ncu: gate it behind an env var (e.g. `XQT_FP8_PROFILE_WORKLOAD=native python profile_script.py`) so ncu only sees the target launches.
- Guard against recursive profiling: if the script itself shells out to `ncu python <itself>`, use an env var sentinel to skip the ncu invocation when already running under ncu.
- Do not run ncu and torch.profiler in the same process/phase; they contend for CUPTI.
- `ERR_NVGPUCTRPERM` means counter permission is blocked (driver/perf-paranoid). Record `blocked_permission` as a fact in the manifest/docs; do not substitute event timing or occupancy API guesses for L2/cache/stall conclusions.
- What to capture per kernel: duration, occupancy, registers, memory throughput, L2 hit rate, eligible warps, stall reasons, tensor pipe utilization.

## Evidence discipline

- Profiling evidence goes to `research/xqt-gemm/artifacts/`; reference the artifact path from `TODO-P3.md` / `GUIDE.md` / `docs/md/explanation/operator-optimization-records.md`.
- If profiler output contradicts handoff notes, trust the fresh artifact and record the discrepancy; do not write numbers you did not reproduce.
