---
name: xqt-operator-runtime-integration
description: Diagnose and fix XQT operator runtime integration issues around wrappers, CUDA Graph capture or replay, benchmark methodology, candidate materialization, and operator-stage reporting. Use when Codex sees a mismatch between kernel microbenchmarks and operator-stage results, when eager and graph fastpaths disagree, when wrapper-level and whole-module timings diverge, or when a TileLang or Triton kernel looks fast in isolation but slow after XQT integration.
---

# XQT operator runtime integration

## Overview

Use this skill to diagnose the layer between a fast kernel and a disappointing XQT operator report. Focus on executor behavior, wrapper forward paths, CUDA Graph capture scope, candidate replacement, and benchmark fairness before assuming the kernel body is the problem.

## Quick start

1. Read [references/operator-runtime-integration.md](references/operator-runtime-integration.md).
2. Confirm the mismatch layer:
   - kernel microbenchmark vs operator-stage report
   - wrapper benchmark vs whole-module benchmark
   - eager fastpath vs CUDA Graph fastpath
3. Inspect the hot XQT files first:
   - `xqt/operator_opt/executor.py`
   - `xqt/operator_opt/backends/tilelang.py`
   - `xqt/operator_opt/kernels/tilelang/attention.py`
   - `tools/benchmark_tilelang_half_ops.py`
   - `tests/xqt/test_operator_tilelang_*`
4. Keep every conclusion grounded in a same-layer comparison. Do not compare kernel-only microbenchmarks against whole-module operator reports and call the difference a kernel regression.
5. If wrapper, whole-module, and operator-stage measurements all agree that the kernel body itself is slow, switch to `$xqt-gpu-kernel-tuning`.

## Workflow

### 1. Classify the failure layer

Use these first cuts:

- microbenchmark slow and operator-stage slow:
  likely kernel or schedule issue
- microbenchmark fast but operator-stage slow:
  likely runtime integration issue
- wrapper fast but whole target slow:
  likely projection, epilogue, input normalization, or candidate wiring issue
- eager good but graph bad:
  likely graph capture scope or replay-state issue

### 2. Audit benchmark fairness

Check:

- baseline and candidate use the same timing methodology
- both sides use the same warmup policy
- both sides use the same sync policy
- tiny TileLang targets use steady-state batched timing when appropriate

If benchmark policy differs between baseline and candidate, fix that first. The result is otherwise not trustworthy.

### 3. Audit CUDA Graph integration

Check:

- whether the graph captures the whole fastpath branch or only the inner kernel
- whether static parameters are copied every replay
- whether self-attention aliases `query`, `key`, and `value` efficiently
- whether cache keys include all fastpath-relevant layout and mode fields

For attention, prefer capturing:

- `qkv projection -> reshape -> tilelang attention -> merge -> out_proj`

For norm, prefer fixing weight and bias inside captured state and replaying only dynamic activations.

### 4. Audit candidate materialization

Check:

- whether wrappers bind to the copied module rather than a pre-copy module
- whether execution metadata comes from the active wrapped module
- whether whole-model replacement changes the benchmark behavior unexpectedly

### 5. Verify at three layers

Re-measure in this order:

1. wrapper benchmark
2. full target module benchmark
3. operator-stage report

Only call the issue resolved when these layers tell a consistent story.

## Reporting shape

When answering with an integration diagnosis, use this structure:

1. target:
   backend, operator family, target module, shape, dtype, GPU, `sm_*`
2. mismatch:
   which two layers disagreed
3. cause:
   one sentence about capture scope, benchmark policy, wrapper binding, or replay state
4. fix:
   one contained runtime-integration change
5. verification:
   wrapper benchmark, whole-module benchmark, and operator-stage report
6. residual risk:
   numeric drift, cache invalidation, or architecture sensitivity
