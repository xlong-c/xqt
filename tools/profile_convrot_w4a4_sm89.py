"""Emit NVTX ranges for representative ConvRot W4A4 CUDA paths."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xqt.operator_opt.kernels.cute.svdq_w4a4_sm89 import (
    allocate_w4a4_workspace,
    bind_convrot_w4a4_linear,
    pack_w4a4_linear,
    w4a4_linear,
)
from xqt.quant.quantizers.convrot_4bit import (
    ConvRotMixedPrecisionLinear,
    _apply_groupwise_rotation,
    _normalized_regular_hadamard,
)

ROWS = 256
FEATURES = 2048
WARMUP = 30
ITERATIONS = 100
DTYPE = torch.bfloat16


def _load_comfy_kitchen_cuda() -> tuple[Any, str] | None:
    try:
        import comfy_kitchen.backends.cuda as cuda_backend
    except Exception:
        return None
    if not bool(getattr(cuda_backend, "_EXT_AVAILABLE", False)):
        return None
    try:
        package_version = version("comfy-kitchen")
    except PackageNotFoundError:
        package_version = "unknown"
    return cuda_backend, package_version


def _run_range(
    name: str,
    fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            output = fn()
            for _ in range(ITERATIONS - 1):
                output = fn()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return output


def profile_convrot_w4a4_sm89() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ConvRot W4A4 profile launcher")
    if torch.cuda.get_device_capability() != (8, 9):
        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(f"ConvRot W4A4 profile targets sm_89, got sm_{major}{minor}")

    torch.manual_seed(2030)
    device = torch.device("cuda")
    source = torch.nn.Linear(
        FEATURES,
        FEATURES,
        bias=True,
        device=device,
        dtype=DTYPE,
    ).eval()
    module = ConvRotMixedPrecisionLinear.from_linear(
        source,
        rot_size=256,
        group_size=128,
        compute_precision="w4a4",
        activation_scale_mode="dynamic",
        w4a4_runtime_backend="rowwise",
    ).eval()
    inputs = torch.randn(ROWS, FEATURES, device=device, dtype=DTYPE)
    artifact_weight = module.dequantized_weight(
        dtype=DTYPE,
        device=device,
        include_padding=True,
    )
    artifact_bias = module._bias_for(dtype=DTYPE, device=device)
    nunchaku_packed = pack_w4a4_linear(artifact_weight, artifact_bias)
    nunchaku_workspace = allocate_w4a4_workspace(ROWS, nunchaku_packed)
    nunchaku_bound = bind_convrot_w4a4_linear(
        nunchaku_packed,
        nunchaku_workspace,
        rows=ROWS,
        logical_input_features=FEATURES,
        rotated_input_features=FEATURES,
        rot_size=256,
    )
    split_workspace = allocate_w4a4_workspace(ROWS, nunchaku_packed)
    rotation = _normalized_regular_hadamard(256, device=device)
    official_baseline = _load_comfy_kitchen_cuda()
    official_cuda = None if official_baseline is None else official_baseline[0]
    official_version = None if official_baseline is None else official_baseline[1]

    module(inputs)
    runner_cache = module._rowwise_w4a4_runner_cache
    if runner_cache is None:
        raise RuntimeError("rowwise ConvRot wrapper did not materialize its runner")
    rowwise_packed = runner_cache[-2]

    def rowwise_call() -> torch.Tensor:
        return module(inputs)

    def nunchaku_call() -> torch.Tensor:
        return nunchaku_bound(inputs)

    def comfy_kitchen_call() -> torch.Tensor:
        if official_cuda is None:
            raise RuntimeError("comfy-kitchen CUDA baseline is unavailable")
        return official_cuda.convrot_w4a4_linear(
            inputs,
            rowwise_packed.qweight,
            rowwise_packed.weight_scales,
            rowwise_packed.bias,
            convrot_groupsize=256,
            quant_group_size=64,
            linear_dtype="int4",
        )

    def split_call() -> torch.Tensor:
        rotated = _apply_groupwise_rotation(
            inputs,
            rot_size=256,
            rotation_matrix=rotation,
            return_padded=True,
        )
        return w4a4_linear(
            rotated,
            nunchaku_packed,
            workspace=split_workspace,
        )

    for _ in range(WARMUP):
        rowwise_call()
        nunchaku_call()
        split_call()
        if official_cuda is not None:
            comfy_kitchen_call()
    torch.cuda.synchronize()

    rowwise_output = _run_range("xqt/convrot_w4a4/rowwise_wrapper", rowwise_call)
    official_output = (
        None
        if official_cuda is None
        else _run_range(
            "xqt/convrot_w4a4/comfy_kitchen_official",
            comfy_kitchen_call,
        )
    )
    nunchaku_output = _run_range("xqt/convrot_w4a4/nunchaku_bound", nunchaku_call)
    split_output = _run_range("xqt/convrot_w4a4/split_rotation", split_call)
    return {
        "gpu": torch.cuda.get_device_name(),
        "target_arch": "sm_89",
        "dtype": str(DTYPE).removeprefix("torch."),
        "shape": {"m": ROWS, "n": FEATURES, "k": FEATURES},
        "iterations_per_range": ITERATIONS,
        "rowwise_checksum": float(rowwise_output.float().sum().item()),
        "comfy_kitchen_checksum": (
            None
            if official_output is None
            else float(official_output.float().sum().item())
        ),
        "comfy_kitchen_version": official_version,
        "official_comparison": {
            "status": "measured" if official_cuda is not None else "unavailable",
            "comparison_layer": "full_python_operator_nvtx_range",
            "weight_contract": "identical packed signed INT4 rowwise buffers",
        },
        "nunchaku_checksum": float(nunchaku_output.float().sum().item()),
        "split_checksum": float(split_output.float().sum().item()),
        "metadata": module.execution_metadata(),
    }


def main() -> None:
    print(json.dumps(profile_convrot_w4a4_sm89(), indent=2))


if __name__ == "__main__":
    main()
