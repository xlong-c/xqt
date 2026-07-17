# XQT Block Runtime Optimization

## Scope

`xqt.operator_opt` optimizes the runtime of an already compressed model. It does not run quantization, pruning, training, or task evaluation.

The acceptance objective is the steady-state latency of a model block on the target hardware. A fast kernel microbenchmark is diagnostic evidence only. It is never sufficient to apply a candidate to the model.

## Candidate Layers

Each operator target has two independent paths:

```text
replacement target -> candidate implementation -> benchmark target block -> model
```

- `candidate_kind=single_kernel`: replaces one operator or a small wrapper. `target` identifies that replacement point. `benchmark_target` identifies the enclosing block that decides acceptance.
- `candidate_kind=block_kernel`: replaces the complete block. `target` and `benchmark_target` must be the same path. This is the manual graph optimization path for a hand-written fused block implementation.

`benchmark_target` defaults to `target` only for narrow targets and existing recipes. New runtime work should name the enclosing block explicitly for `single_kernel` candidates.

## Materialization

`torch_compile` materializes an automatic whole-block graph candidate when the target is a `block_kernel`.

Hand-written block candidates use a named builder:

```python
from torch import nn

from xqt.operator_opt import (
    OperatorOptimizationTargetPlan,
    register_block_kernel_builder,
)


@register_block_kernel_builder("my_decoder_block_tilelang")
def build_decoder_block(
    block: nn.Module,
    target: OperatorOptimizationTargetPlan,
) -> nn.Module:
    return MyFusedDecoderBlock(block, target)
```

The target uses `candidate_kind=block_kernel`, `engine=tilelang` or another XQT engine, and `block_kernel=my_decoder_block_tilelang`. The builder must return a replacement for the full block. XQT refuses an unregistered block builder and does not silently substitute a single-operator wrapper.

Automatic graph optimization can name a manual block fallback in the same target:

```yaml
candidate_kind: block_kernel
target: decoder.layers.0
engine: torch_compile
block_kernel: my_decoder_block_tilelang
block_kernel_engine: tilelang
```

Plan assembly expands this into two candidates for the same block. The first candidate is `candidate_layer=automatic_block_graph` and uses `torch_compile`; the second candidate is `candidate_layer=manual_block_kernel`, uses `block_kernel_engine`, and records `fallback_for` with the automatic target name. The manual candidate is evaluated only when the automatic block candidate is not applied.

## Acceptance

For every candidate, XQT materializes a complete candidate root model. It compares the baseline and candidate versions of the same `benchmark_target` block with identical inputs, warmup policy, synchronization policy, and timing strategy.

XQT applies the candidate only when all conditions hold:

- block outputs satisfy numeric validation.
- block latency meets `min_speedup`.
- the candidate did not execute a configured reference fallback.

Reports record `candidate_kind`, `candidate_layer`, `replacement_target_path`, `benchmark_target_path`, `benchmark_scope=block`, `optimization_basis=block`, the benchmark strategy, and the root-candidate materialization mode. Kernel, wrapper, block, and model end-to-end measurements must remain separate in analysis. Final deployment selection uses the model end-to-end result.

## Graph Fusion Rule

A custom kernel that creates a graph break, layout conversion, extra materialization, or host dispatch overhead must not be retained merely because its standalone timing is fast. Prefer a `block_kernel` that owns the complete hot path and preserves its natural input and output layouts. Examples include `QKV projection -> attention -> out projection`, or residual add together with its following normalization.
