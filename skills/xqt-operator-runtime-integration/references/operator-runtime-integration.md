# XQT operator runtime integration playbook

Use this file when the problem is no longer "the kernel is slow" but "the operator-stage result does not match the kernel-level result".

Typical triggers:

- microbenchmark is good but `execute_operator_optimization_plan` report is bad
- eager fastpath and CUDA Graph fastpath disagree sharply
- wrapper-only benchmarks look healthy but whole-module benchmarks regress
- graph replay is slower or less accurate than eager
- numeric drift appears only after wrapping or candidate materialization

## Scope

This file is about XQT runtime integration around the kernel:

- wrapper forward path
- candidate model materialization
- benchmark methodology
- CUDA Graph capture and replay
- operator report metadata

It is not a replacement for `nsys` or `ncu`. Use it first when the evidence points above the kernel body.

## Triage order

### 1. Check whether the mismatch is kernel-level or integration-level

Classify the current evidence:

- kernel microbenchmark slow and operator-stage slow:
  likely kernel issue
- kernel microbenchmark fast but operator-stage slow:
  likely integration issue
- wrapper benchmark fast but operator-stage slow:
  likely candidate materialization, call path, or report benchmark issue
- eager and graph both slow:
  likely kernel or layout issue
- eager healthy but graph bad:
  likely graph capture scope or replay-state issue

Do not start with profiler work if the mismatch is already isolated to integration.

### 2. Compare the same thing at the same layer

Keep comparisons aligned:

- kernel vs kernel
- wrapper vs wrapper
- whole target module vs whole target module
- operator-stage report vs operator-stage report

Do not compare a kernel-only microbenchmark to a whole-module report and call the difference a kernel regression.

## Benchmark fairness rules

### Rule 1: baseline and candidate must use the same timing methodology

Invalid comparison examples:

- baseline single-call timing vs candidate batched steady-state timing
- baseline host wall-clock vs candidate CUDA events
- baseline warmed state vs candidate first-run compile path

Valid comparison examples:

- both baseline and candidate measured with the same batched callable count
- both measured after the same warmup policy
- both measured with the same sync behavior

### Rule 2: small TileLang targets should prefer steady-state batched timing

For tiny `conv`, `linear`, `attention`, and `norm` targets, single-call timing often measures launch and host overhead more than the steady-state fastpath.

Use batched repeated calls when:

- latency is sub-millisecond
- graph replay is under evaluation
- the operator report is intended to represent steady-state behavior

### Rule 3: if you change the benchmark policy, re-check conclusions

A previous "winner" under single-call timing may lose under steady-state timing, and vice versa.

## CUDA Graph checklist

### Capture scope

Ask first: what should the graph actually include?

For an operator fastpath, capture the whole fastpath branch when possible, not only the innermost kernel.

Examples:

- `attention`: prefer `qkv projection -> reshape -> tilelang attention -> merge -> out_proj`
- `norm`: prefer input-to-output path with static parameters fixed in captured state

If the graph only captures the innermost kernel, host-side work around it can erase the gain.

### Dynamic vs static state

Dynamic inputs should usually be:

- user input tensors
- per-call activations

Static captured state should usually be:

- module weights
- module bias
- constant layout metadata encoded by the graph body

Red flag:

- replay copies weight or bias every invocation even though they are constant

### Self-attention alias handling

In self-attention, `query`, `key`, and `value` may be the same tensor object.

Check whether the graph path:

- copies the same runtime tensor into three static buffers unnecessarily
- treats alias-equal inputs as independent dynamic inputs

If yes, de-duplicate runtime args and reconstruct aliases inside the graph body.

### Cache key discipline

Graph cache keys should include:

- shape
- stride
- dtype
- device
- causality or mask mode
- fastpath-relevant schedule knobs
- target architecture when it affects codegen

Do not rebuild graphs for semantically identical inputs, and do not replay a graph under incompatible layout or dtype.

## Candidate materialization checklist

When wrapping nested modules:

1. verify whether the wrapper binds to the copied module, not the pre-copy module
2. verify that `deepcopy` does not leave the wrapper pointing at stale parameters
3. verify that execution metadata is read from the active wrapped module

Red flags:

- wrapper references a module object that is no longer part of the returned candidate model
- wrapper benchmark is good in isolation but bad inside the materialized candidate

## Whole-module vs wrapper diagnosis

When a wrapper benchmark and operator-stage report disagree:

1. benchmark the wrapper directly
2. benchmark the full target module directly
3. benchmark the same full target module through the operator executor call path
4. compare metadata and benchmark strategy at each layer

This usually isolates whether the gap comes from:

- surrounding projection or epilogue work
- input normalization and calling convention
- executor benchmark policy
- stale wrapper attachment

## What to inspect first in XQT

Usually inspect these files in this order:

1. `xqt/kernels/wrappers/executor.py`
2. `xqt/kernels/ops/_impl/engines/tilelang.py`
3. `xqt/kernels/ops/_impl/tilelang/attention.py`
4. `tools/benchmark_tilelang_half_ops.py`
5. `tests/xqt/test_operator_tilelang_*`

Look for:

- capture scope mismatches
- replay copy of static tensors
- unfair baseline/candidate timing
- wrapper binding to the wrong module object
- report metadata that hides the real benchmark strategy

## Reporting shape for integration fixes

When the fix is above the kernel, report it that way:

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
   remaining numeric drift, graph cache invalidation, or architecture sensitivity
