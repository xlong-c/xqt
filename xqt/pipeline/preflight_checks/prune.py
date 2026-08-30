"""Preflight checks for pruning stages."""

from __future__ import annotations

from xqt.core.schema import PruneConfig
from xqt.compression.prune import describe_prune_runtime_capability

from ._base import PreflightReport


def _check_prune_config(
    report: PreflightReport,
    prune: PruneConfig,
    *,
    device: str | None,
    task_type: str,
    prefix: str,
) -> None:
    if not prune.enabled:
        return
    if prune.method == "nm_structured":
        pattern_raw = prune.selection.get("pattern") or prune.params.get("pattern")
        if isinstance(pattern_raw, (list, tuple)) and len(pattern_raw) == 2:
            pattern = (int(pattern_raw[0]), int(pattern_raw[1]))
            capability = describe_prune_runtime_capability(
                method="nm_structured",
                device=device,
                pattern=pattern,
            ).to_dict()
            report.add(
                f"{prefix}.nm_backend",
                bool(capability["supported"]),
                str(capability["reason"]),
                pattern=list(pattern),
                runtime=capability["runtime"],
                speedup_verified=capability["speedup_verified"],
                pattern_present=capability["pattern_present"],
                level="info" if capability["supported"] else "warning",
            )
        else:
            report.add(
                f"{prefix}.nm_backend",
                False,
                "N:M structured pruning requires selection.pattern=[N, M]",
                level="error",
            )
    if prune.method == "block_sparse":
        block_shape_raw = prune.selection.get("block_shape") or prune.params.get(
            "block_shape"
        )
        if isinstance(block_shape_raw, (list, tuple)) and len(block_shape_raw) == 2:
            block_shape = (int(block_shape_raw[0]), int(block_shape_raw[1]))
            capability = describe_prune_runtime_capability(
                method="block_sparse",
                device=device,
                block_shape=block_shape,
            ).to_dict()
            report.add(
                f"{prefix}.block_sparse_backend",
                bool(capability["supported"]),
                str(capability["reason"]),
                block_shape=list(block_shape),
                runtime=capability["runtime"],
                speedup_verified=capability["speedup_verified"],
                pattern_present=capability["pattern_present"],
                level="info" if capability["supported"] else "warning",
            )
        else:
            report.add(
                f"{prefix}.block_sparse_backend",
                False,
                "block_sparse pruning requires selection.block_shape=[rows, cols]",
                level="error",
            )
    metadata = {
        "method": prune.method,
        "target_sparsity": prune.target_sparsity,
        "granularity": prune.granularity,
        "scope": prune.scope,
    }
    if task_type != "detection":
        return
    if prune.method == "global_l1_unstructured":
        report.add(
            f"{prefix}.detection_safety",
            True,
            "unstructured detection pruning records sparsity only; speedup is not claimed",
            **metadata,
        )
        return
    if prune.method != "structured":
        report.add(
            f"{prefix}.detection_safety",
            True,
            "detection pruning method does not rewrite detection head topology",
            **metadata,
        )
        return
    report.add(
        f"{prefix}.detection_safety",
        True,
        (
            "structured detection pruning is guarded; residual/CSP/C2f/SPPF/detect head "
            "dependency rewrite is not implemented"
        ),
        level="warning",
        support="unsupported_in_builtin_executor",
        **metadata,
    )
