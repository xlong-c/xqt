"""Demo script for XQT multi-precision GEMM kernels."""

import torch

from xqt.kernels.ops._impl.gemm_precision import (
    MatmulPrecisionSpec,
    describe_gemm_precision_capability,
    gemm_with_precision,
    list_available_precisions,
)
from xqt.kernels.ops._impl.triton.mxfp_gemm import pack_mxfp


def main() -> None:
    print("=" * 80)
    print("XQT Multi-Precision GEMM Demo")
    print("=" * 80)

    # Check available precisions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    available = list_available_precisions(device)
    print(f"\nAvailable precisions: {', '.join(available)}")

    # Show capability for each precision
    print("\n" + "-" * 80)
    print("Precision Capabilities:")
    print("-" * 80)
    for precision in ["fp16", "bf16", "int8", "fp8", "int4", "fp4", "nvfp4", "mxfp8"]:
        cap = describe_gemm_precision_capability(precision, device)
        status = "✓" if cap["available"] else "✗"
        native = "native" if cap.get("hardware_native") else "emulated"
        print(f"{status} {precision:8s} - {cap['engine']:8s} ({native})")
        if cap.get("notes"):
            for note in cap["notes"]:
                print(f"           {note}")

    if device.type != "cuda":
        print("\n⚠ CUDA not available. Skipping GEMM execution demo.")
        return

    # Run GEMM with different precisions
    print("\n" + "-" * 80)
    print("GEMM Execution Demo")
    print("-" * 80)

    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(N, K, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16)

    # Reference
    ref = torch.matmul(a, b.t()) + bias

    precisions_to_test = ["fp16", "bf16"]
    if torch.cuda.get_device_capability(device)[0] * 10 >= 89:
        precisions_to_test.append("fp8")

    for precision in precisions_to_test:
        if precision not in available:
            continue

        # Prepare inputs
        if precision == "bf16":
            a_p = a.to(torch.bfloat16)
            b_p = b.to(torch.bfloat16)
            bias_p = bias.to(torch.bfloat16)
        elif precision == "fp8":
            a_p = a.to(torch.float8_e4m3fn)
            b_p = b.to(torch.float8_e4m3fn)
            bias_p = bias
        else:
            a_p, b_p, bias_p = a, b, bias

        # Run GEMM
        output = gemm_with_precision(
            a_p,
            b_p,
            bias_p,
            precision=MatmulPrecisionSpec(
                activation=precision,
                weight=precision,
                bias=precision,
                mma=precision,
                accum="fp32",
                output=precision,
            ),
            engine="triton",
            transpose_b=True,
        )

        # Check error
        error = (output.float() - ref.float()).abs().max().item()
        rel_error = error / ref.abs().max().item()

        print(f"{precision:8s}: max_error={error:.6f}, rel_error={rel_error:.4f}")

    # MXFP demo
    if "mxfp8" in available:
        print("\n" + "-" * 80)
        print("MXFP (Microscaling) Demo")
        print("-" * 80)

        tensor = torch.randn(128, 128, device=device)
        print(f"Original tensor: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"  Size: {tensor.numel() * tensor.element_size()} bytes")

        for precision in [8, 6, 4]:
            packed, scales = pack_mxfp(tensor, precision=precision, block_size=32)
            packed_size = (
                packed.numel() * packed.element_size()
                + scales.numel() * scales.element_size()
            )
            compression = tensor.numel() * tensor.element_size() / packed_size

            print(f"\nMXFP{precision}:")
            print(f"  Packed size: {packed_size} bytes")
            print(f"  Compression: {compression:.2f}x")
            print(f"  Mantissa bits: {precision}, Scales: {scales.numel()}")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
