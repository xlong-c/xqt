# External profiling survey

Use this file when deciding whether to borrow a profiling workflow, terminology, or artifact layout. These sources were checked against their default branches on 2026-08-03; do not assume they are installed locally.

## Relevant current projects

### [Triton Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton)

Proton records Python scopes, user annotations, Triton launch metadata, and GPU metrics. It can emit a hierarchical profile or a Chrome/Perfetto trace, and exposes both explicit scopes and a function decorator.

Borrow the semantic range principle: a profile must identify the model or operator intent as well as the generated kernel. Do not make Proton a required XQT dependency or confuse its trace with Nsight Compute counter evidence.

### [vLLM profiler](https://github.com/vllm-project/vllm/tree/main/vllm/profiler)

vLLM bounds collection with delayed start, warmup, active iterations, and a maximum active window. Its profiler wrapper also emits trace files and exposes named context managers for Torch and NVTX annotations.

Borrow fixed sampling windows, explicit trace export, and semantic annotations. Keep them in an XQT profile-only launcher or existing benchmark path, not a new general XQT profiler service.

### [fmh66/kernel-opt-agent](https://github.com/fmh66/kernel-opt-agent)

Its `kernel-profile` skill performs environment checks, correctness validation, CUDA-event timing, NCU collection, and writes separate environment, correctness, summary, and detail artifacts. Its `kernel-loop` preserves a one-change hypothesis history.

Borrow validation-before-counters, one-change iterations, and artifact lineage. Do not copy its standalone kernel ABI because XQT profiles integrated operator candidates.

### [ZJtoast/kernel-profiler-skill](https://github.com/ZJtoast/kernel-profiler-skill)

This skill uses a canonical target record, starts NCU with a compact stage, selects one evidence-driven follow-up section, and keeps raw CSV, command, environment, source-hotspot, and conclusion records together. It also treats counter permission failures as a hard profiling boundary.

Borrow staged NCU collection, exact target filtering, source-attribution requirements, and confidence or limitation reporting. Do not copy its `sudo` setup or make its filesystem layout an XQT public API.

### [Hugging Face kernels](https://github.com/huggingface/kernels)

This project is a kernel distribution and loading system, not a profiling workflow. It is useful as a reminder that kernel work must integrate with real model call sites, but it is not evidence for a profiler design.

## Resulting XQT rules

1. Keep the main skill short and put volatile profiler commands and metric guidance in references.
2. Require a same-layer XQT benchmark and numeric-validation result before profiler interpretation.
3. Profile a stable semantic range or verified generated kernel, never every process kernel by default.
4. Use `nsys` to select the time bottleneck, `ncu --set basic` to classify it, then one targeted NCU section to test the hypothesis.
5. Preserve the environment, command shape, target metadata, raw profiler artifacts, conclusion, and rejected hypotheses in the same experiment lineage.
6. Keep `tilelang`, `triton`, `cutlass`, `cute_dsl`, and `cutile` maturity distinctions explicit, and hand runtime-layer mismatches to `$xqt-operator-runtime-integration`.
