"""Triton kernels for ConvRot input-side fusion."""

from __future__ import annotations

import torch

from xqt.core.errors import XQTBackendError

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None  # type: ignore[assignment]


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


if triton is not None and tl is not None:

    @triton.jit
    def _norm_hadamard_static_quant_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        rotation_ptr,
        scale_ptr,
        out_ptr,
        eps,
        logical_features: tl.constexpr,
        padded_features: tl.constexpr,
        rot_size: tl.constexpr,
        IS_LAYERNORM: tl.constexpr,
        INPUT_DTYPE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        NUM_TILES: tl.constexpr,
        BLOCK_FEATURE: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fuse last-dimension norm, group rotation, and static INT8 quantization.

        A program owns one input row/group/tile.  It computes that row's
        normalization statistics once for the tile and streams the selected
        rotation block without materializing normalized or rotated tensors.
        """

        program = tl.program_id(0)
        tiles_per_row = NUM_GROUPS * NUM_TILES
        row = program // tiles_per_row
        tile = program - row * tiles_per_row
        group = tile // NUM_TILES
        n_tile = tile - group * NUM_TILES
        feature_offsets = tl.arange(0, BLOCK_FEATURE)
        logical_mask = feature_offsets < logical_features
        row_base = row * padded_features
        values = tl.load(
            x_ptr + row_base + feature_offsets,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(values * values, axis=0)
        if IS_LAYERNORM:
            mean = tl.sum(values, axis=0) / logical_features
            centered = tl.where(logical_mask, values - mean, 0.0)
            variance = tl.sum(centered * centered, axis=0) / logical_features
            inverse = tl.rsqrt(variance + eps)
        else:
            mean = 0.0
            inverse = tl.rsqrt(square_sum / logical_features + eps)

        scale = tl.load(scale_ptr).to(tl.float32)
        n_offsets = tl.arange(0, BLOCK_N)
        n_mask = n_offsets < rot_size
        group_base = group * rot_size
        n_index = n_tile * BLOCK_N + n_offsets
        n_valid = n_mask & (n_index < rot_size)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_tile in tl.static_range(0, rot_size, BLOCK_K):
            k_offsets = k_tile + tl.arange(0, BLOCK_K)
            k_valid = k_offsets < rot_size
            feature_index = group_base + k_offsets
            feature_valid = k_valid & (feature_index < padded_features)
            raw = tl.load(
                x_ptr + row_base + feature_index,
                mask=feature_valid,
                other=0.0,
            ).to(tl.float32)
            affine_mask = feature_index < logical_features
            normed = tl.where(
                affine_mask,
                (raw - mean) * inverse,
                raw * inverse,
            )
            weight = tl.load(
                weight_ptr + feature_index,
                mask=affine_mask,
                other=1.0,
            ).to(tl.float32)
            normed = normed * weight
            if IS_LAYERNORM:
                bias = tl.load(
                    bias_ptr + feature_index,
                    mask=affine_mask,
                    other=0.0,
                ).to(tl.float32)
                normed += tl.where(affine_mask, bias, 0.0)
            rotation = tl.load(
                rotation_ptr
                + k_offsets[:, None] * rot_size
                + n_index[None, :],
                mask=k_valid[:, None] & n_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            if INPUT_DTYPE == 0 and rot_size >= 16:
                accumulator += tl.sum(
                    tl.dot(
                        normed.to(tl.float16)[None, :],
                        rotation.to(tl.float16),
                    ),
                    axis=0,
                )
            elif INPUT_DTYPE == 1 and rot_size >= 16:
                accumulator += tl.sum(
                    tl.dot(
                        normed.to(tl.bfloat16)[None, :],
                        rotation.to(tl.bfloat16),
                    ),
                    axis=0,
                )
            else:
                accumulator += tl.sum(
                    normed[:, None] * rotation,
                    axis=0,
                )
        scaled = accumulator / scale
        rounded = tl.where(
            scaled >= 0.0,
            tl.floor(scaled + 0.5),
            tl.ceil(scaled - 0.5),
        )
        clipped = tl.maximum(tl.minimum(rounded, 127.0), -127.0)
        tl.store(
            out_ptr + row_base + group_base + n_index,
            clipped.to(tl.int8),
            mask=n_valid,
        )

else:  # pragma: no cover - exercised only when Triton is unavailable
    _norm_hadamard_static_quant_kernel = None


if triton is not None and tl is not None:

    @triton.jit
    def _norm_hadamard_static_quant_row_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        rotation_ptr,
        scale_ptr,
        out_ptr,
        eps,
        logical_features: tl.constexpr,
        padded_features: tl.constexpr,
        rot_size: tl.constexpr,
        IS_LAYERNORM: tl.constexpr,
        INPUT_DTYPE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_FEATURE: tl.constexpr,
    ):
        """Fuse one row's Norm statistics with all group rotations.

        The original tiled kernel maps one program to a ``(row, group, tile)``
        and consequently recomputes the row statistics once per rotation
        group.  This variant owns the complete row, computes the inverse norm
        exactly once, then reloads only each group's values while reusing that
        statistic.  It is restricted by the host launcher to moderate group
        counts where the extra reduction work is worthwhile.
        """

        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_FEATURE)
        logical_mask = offsets < logical_features
        row_base = row * padded_features
        values = tl.load(
            x_ptr + row_base + offsets,
            mask=logical_mask,
            other=0.0,
        ).to(tl.float32)
        square_sum = tl.sum(values * values, axis=0)
        if IS_LAYERNORM:
            mean = tl.sum(values, axis=0) / logical_features
            centered = tl.where(logical_mask, values - mean, 0.0)
            variance = tl.sum(centered * centered, axis=0) / logical_features
            inverse = tl.rsqrt(variance + eps)
        else:
            mean = 0.0
            inverse = tl.rsqrt(square_sum / logical_features + eps)

        scale = tl.load(scale_ptr).to(tl.float32)
        group_offsets = tl.arange(0, rot_size)
        group_mask = group_offsets < rot_size
        for group in tl.static_range(0, NUM_GROUPS):
            group_base = group * rot_size
            feature_index = group_base + group_offsets
            feature_mask = feature_index < logical_features
            raw = tl.load(
                x_ptr + row_base + feature_index,
                mask=feature_mask,
                other=0.0,
            ).to(tl.float32)
            group_values = tl.where(
                feature_mask,
                (raw - mean) * inverse,
                raw * inverse,
            )
            group_values = group_values * tl.load(
                weight_ptr + feature_index,
                mask=feature_mask,
                other=1.0,
            ).to(tl.float32)
            if IS_LAYERNORM:
                group_values += tl.where(
                    feature_mask,
                    tl.load(
                        bias_ptr + feature_index,
                        mask=feature_mask,
                        other=0.0,
                    ).to(tl.float32),
                    0.0,
                )
            rotation = tl.load(
                rotation_ptr
                + group_offsets[:, None] * rot_size
                + group_offsets[None, :],
                mask=group_mask[:, None] & group_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if INPUT_DTYPE == 0 and rot_size >= 16:
                rotated = tl.sum(
                    tl.dot(
                        group_values.to(tl.float16)[None, :],
                        rotation.to(tl.float16),
                    ),
                    axis=0,
                )
            elif INPUT_DTYPE == 1 and rot_size >= 16:
                rotated = tl.sum(
                    tl.dot(
                        group_values.to(tl.bfloat16)[None, :],
                        rotation.to(tl.bfloat16),
                    ),
                    axis=0,
                )
            else:
                rotated = tl.sum(group_values[:, None] * rotation, axis=0)
            scaled = rotated / scale
            rounded = tl.where(
                scaled >= 0.0,
                tl.floor(scaled + 0.5),
                tl.ceil(scaled - 0.5),
            )
            clipped = tl.maximum(tl.minimum(rounded, 127.0), -127.0)
            tl.store(
                out_ptr + row_base + group_base + group_offsets,
                clipped.to(tl.int8),
                mask=group_mask,
            )

else:  # pragma: no cover - exercised only when Triton is unavailable
    _norm_hadamard_static_quant_row_kernel = None


def fused_norm_hadamard_static_quantize_triton(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    rotation: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    logical_features: int,
    padded_features: int,
    rot_size: int,
    norm_kind: str = "rmsnorm",
    eps: float = 1e-6,
    block_feature: int = 4096,
    block_k: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Run fused Norm+ConvRot static activation quantization on CUDA."""

    if triton is None or tl is None or _norm_hadamard_static_quant_kernel is None:
        raise XQTBackendError("Triton is required for fused ConvRot Norm quantization")
    tensors = (inputs, weight, rotation)
    if bias is not None:
        tensors = (*tensors, bias)
    if not all(tensor.is_cuda for tensor in tensors):
        raise XQTBackendError("fused ConvRot Norm quantization requires CUDA tensors")
    if inputs.ndim != 2:
        raise XQTBackendError("fused ConvRot Norm quantization expects a 2D input")
    logical = int(logical_features)
    padded = int(padded_features)
    rotation_size = int(rot_size)
    if logical < 1 or padded < logical or rotation_size < 1:
        raise ValueError("invalid ConvRot fused feature dimensions")
    if int(inputs.shape[1]) != padded:
        raise XQTBackendError(
            "fused ConvRot Norm input must use the padded feature extent"
        )
    if weight.ndim != 1 or int(weight.numel()) != logical:
        raise XQTBackendError("Norm weight must match logical input features")
    if bias is not None and (bias.ndim != 1 or int(bias.numel()) != logical):
        raise XQTBackendError("LayerNorm bias must match logical input features")
    if rotation.shape != (rotation_size, rotation_size):
        raise XQTBackendError("rotation shape must match rot_size")
    if padded % rotation_size != 0:
        raise XQTBackendError("padded features must be divisible by rot_size")
    launch_block_k = min(int(block_k), rotation_size)
    launch_block_n = min(int(block_n), rotation_size)
    if rotation_size % launch_block_k != 0 or rotation_size % launch_block_n != 0:
        raise XQTBackendError("rot_size must align with fused Triton tile sizes")
    kind = str(norm_kind).strip().lower()
    if kind not in {"rmsnorm", "layernorm"}:
        raise ValueError("norm_kind must be rmsnorm or layernorm")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError("fused ConvRot Norm input dtype must be fp16, bf16, or fp32")
    if rotation.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise XQTBackendError("fused ConvRot rotation dtype must be fp16, bf16, or fp32")
    scale_tensor = torch.as_tensor(
        scale,
        device=inputs.device,
        dtype=torch.float32,
    ).reshape(())
    # The fused path is called for every inference step.  Do not turn a CUDA
    # scalar into a Python number here: that would synchronize the stream and
    # hide the launch savings this kernel is meant to provide.  CPU callers
    # still get the eager validation.
    if scale_tensor.device.type != "cuda" and float(scale_tensor.item()) <= 0.0:
        raise ValueError("activation scale must be positive")
    bias_tensor = (
        torch.zeros_like(weight, device=inputs.device)
        if bias is None
        else bias.to(device=inputs.device, dtype=inputs.dtype).contiguous()
    )
    configured_feature_block = int(block_feature)
    if configured_feature_block < 1:
        raise ValueError("block_feature must be positive")
    feature_block = _next_power_of_two(padded)
    if feature_block > configured_feature_block:
        raise XQTBackendError(
            "fused Triton ConvRot Norm hidden size exceeds block_feature"
        )
    out = torch.empty_like(inputs, dtype=torch.int8)
    num_groups = padded // rotation_size
    # One-row programs avoid recomputing Norm statistics for every group. Keep
    # the original tiled mapping for very wide rows, where the row variant's
    # register footprint and compile size outweigh that saving.
    use_row_kernel = (
        num_groups <= 32 and _norm_hadamard_static_quant_row_kernel is not None
    )
    input_dtype = (
        0
        if inputs.dtype == torch.float16
        else 1
        if inputs.dtype == torch.bfloat16
        else 2
    )
    input_tensor = inputs.contiguous()
    weight_tensor = weight.to(device=inputs.device, dtype=inputs.dtype).contiguous()
    rotation_tensor = rotation.to(
        device=inputs.device,
        dtype=inputs.dtype,
    ).contiguous()
    if use_row_kernel:
        _norm_hadamard_static_quant_row_kernel[int(inputs.shape[0]),](
            input_tensor,
            weight_tensor,
            bias_tensor,
            rotation_tensor,
            scale_tensor,
            out,
            float(eps),
            logical_features=logical,
            padded_features=padded,
            rot_size=rotation_size,
            IS_LAYERNORM=kind == "layernorm",
            INPUT_DTYPE=input_dtype,
            NUM_GROUPS=num_groups,
            BLOCK_FEATURE=feature_block,
            num_warps=int(num_warps),
            num_stages=int(num_stages),
        )
        return out
    tile_count = (rotation_size + launch_block_n - 1) // launch_block_n
    grid = (int(inputs.shape[0]) * num_groups * tile_count,)
    _norm_hadamard_static_quant_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        rotation_tensor,
        scale_tensor,
        out,
        float(eps),
        logical_features=logical,
        padded_features=padded,
        rot_size=rotation_size,
        IS_LAYERNORM=kind == "layernorm",
        INPUT_DTYPE=input_dtype,
        NUM_GROUPS=num_groups,
        NUM_TILES=tile_count,
        BLOCK_FEATURE=feature_block,
        BLOCK_K=launch_block_k,
        BLOCK_N=launch_block_n,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


__all__ = ["fused_norm_hadamard_static_quantize_triton"]
