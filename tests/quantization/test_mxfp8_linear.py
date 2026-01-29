# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Mxfp8LinearOp comparing torch and flashinfer backends.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    Mxfp8LinearOp,
    mxfp8_e4m3_quantize,
)

# Import flashinfer wrapper to register torch.ops.vllm.bmm_mxfp8
try:
    import vllm.utils.flashinfer  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


@pytest.mark.parametrize(
    "batch_size,in_features,out_features",
    [
        (1, 128, 256),
        (4, 256, 512),
        (16, 512, 1024),
    ],
)
def test_mxfp8_linear_torch_backend(batch_size, in_features, out_features):
    """Test that torch backend works correctly (no flashinfer required)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    # Create random bf16 weight and input
    weight_bf16 = torch.randn(
        out_features, in_features, device=device, dtype=torch.bfloat16
    )
    input_tensor = torch.randn(
        batch_size, in_features, device=device, dtype=torch.bfloat16
    )

    # Quantize weight to MXFP8
    weight_fp8, weight_scale = mxfp8_e4m3_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    # Run torch backend
    torch_backend = Mxfp8LinearOp(backend="torch")
    output = torch_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=None,
    )

    # Compare with reference (direct bf16 linear with original weights)
    # Note: there will be quantization error, so we use loose tolerance
    reference = torch.nn.functional.linear(input_tensor, weight_bf16)

    print(
        f"\n--- Torch backend test: ({batch_size}, {in_features}) "
        f"x ({out_features}, {in_features}) ---"
    )
    print(f"Output shape: {output.shape}")
    print(f"Reference shape: {reference.shape}")

    abs_diff = (output - reference).abs()
    print(f"Max abs diff: {abs_diff.max().item():.6f}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6f}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(
        f"Reference range: [{reference.min().item():.4f}, {reference.max().item():.4f}]"
    )

    # Check output shape is correct
    assert output.shape == (batch_size, out_features), (
        f"Shape mismatch: got {output.shape}, expected ({batch_size}, {out_features})"
    )

    # Check output is not all zeros or NaN
    assert not output.isnan().any(), "Output contains NaN!"
    assert not output.isinf().any(), "Output contains Inf!"
    assert output.abs().max() > 1e-6, "Output is near-zero!"

    # Verify output magnitude is similar to reference (within 2x)
    # This catches gross errors while allowing quantization error
    output_mag = output.abs().mean().item()
    ref_mag = reference.abs().mean().item()
    assert 0.5 < output_mag / (ref_mag + 1e-6) < 2.0, (
        f"Output magnitude differs too much: output={output_mag:.4f}, ref={ref_mag:.4f}"
    )


def create_mxfp8_weight(out_features: int, in_features: int, device: str = "cuda"):
    """Create a random MXFP8 quantized weight tensor with scales."""
    # Create random bf16 weight
    weight_bf16 = torch.randn(
        out_features, in_features, device=device, dtype=torch.bfloat16
    )

    # Quantize to MXFP8
    weight_fp8, weight_scale = mxfp8_e4m3_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    return weight_fp8, weight_scale, weight_bf16


def create_input(batch_size: int, in_features: int, device: str = "cuda"):
    """Create a random bf16 input tensor."""
    return torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "batch_size,in_features,out_features",
    [
        (1, 128, 256),
        (4, 256, 512),
        (16, 512, 1024),
        (32, 1024, 2048),
        (64, 4096, 4096),
    ],
)
@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_backends_match(batch_size, in_features, out_features):
    """Test that torch and flashinfer backends produce similar results."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    # Create MXFP8 weight
    weight_fp8, weight_scale, weight_bf16 = create_mxfp8_weight(
        out_features, in_features, device
    )

    # Create input
    input_tensor = create_input(batch_size, in_features, device)

    # Create both backends
    torch_backend = Mxfp8LinearOp(backend="torch")
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")

    # Run torch backend (baseline)
    output_torch = torch_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=None,
    )

    # Run flashinfer backend
    output_flashinfer = flashinfer_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=None,
    )

    # Compare results
    # Allow some tolerance due to different computation paths
    rtol = 0.1  # 10% relative tolerance
    atol = 0.1  # absolute tolerance

    # Check shapes match
    assert output_torch.shape == output_flashinfer.shape, (
        f"Shape mismatch: torch={output_torch.shape}, "
        f"flashinfer={output_flashinfer.shape}"
    )

    # Compute difference metrics
    abs_diff = (output_torch - output_flashinfer).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (output_torch.abs() + 1e-6)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(
        f"\n--- Shape: ({batch_size}, {in_features}) "
        f"x ({out_features}, {in_features}) ---"
    )
    print(f"Output shape: {output_torch.shape}")
    print(f"Max absolute diff: {max_abs_diff:.6f}")
    print(f"Mean absolute diff: {mean_abs_diff:.6f}")
    print(f"Max relative diff: {max_rel_diff:.6f}")
    print(f"Mean relative diff: {mean_rel_diff:.6f}")
    t_min, t_max = output_torch.min().item(), output_torch.max().item()
    print(f"Torch output range: [{t_min:.4f}, {t_max:.4f}]")
    f_min, f_max = output_flashinfer.min().item(), output_flashinfer.max().item()
    print(f"Flashinfer output range: [{f_min:.4f}, {f_max:.4f}]")

    # Check if outputs are close
    is_close = torch.allclose(output_torch, output_flashinfer, rtol=rtol, atol=atol)

    if not is_close:
        # Print more debug info
        print("\n--- DEBUG INFO ---")
        print(f"Torch output (first 5 elements): {output_torch.flatten()[:5]}")
        print(
            f"Flashinfer output (first 5 elements): {output_flashinfer.flatten()[:5]}"
        )

        # Check if flashinfer output is all zeros or garbage
        if output_flashinfer.abs().max() < 1e-6:
            print("WARNING: Flashinfer output is near-zero!")
        if output_flashinfer.isnan().any():
            print("WARNING: Flashinfer output contains NaN!")
        if output_flashinfer.isinf().any():
            print("WARNING: Flashinfer output contains Inf!")

    assert is_close, (
        f"Outputs don't match! Max abs diff: {max_abs_diff:.6f}, "
        f"Max rel diff: {max_rel_diff:.6f}"
    )


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_with_bias(batch_size):
    """Test that bias is correctly applied in both backends."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    in_features = 256
    out_features = 512

    # Create MXFP8 weight
    weight_fp8, weight_scale, _ = create_mxfp8_weight(out_features, in_features, device)

    # Create input and bias
    input_tensor = create_input(batch_size, in_features, device)
    bias = torch.randn(out_features, device=device, dtype=torch.bfloat16)

    # Create both backends
    torch_backend = Mxfp8LinearOp(backend="torch")
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")

    # Run both backends with bias
    output_torch = torch_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=bias,
    )

    output_flashinfer = flashinfer_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=bias,
    )

    # Compare
    rtol = 0.1
    atol = 0.1
    assert torch.allclose(output_torch, output_flashinfer, rtol=rtol, atol=atol), (
        f"Bias test failed! torch={output_torch.shape}, "
        f"flashinfer={output_flashinfer.shape}"
    )


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_debug_single():
    """
    Detailed debug test with a single small case.
    Run with: pytest test_mxfp8_linear.py::test_mxfp8_linear_debug_single -v -s
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    # Use dimensions that match flashinfer's tested sizes
    # flashinfer tests use m,n,k >= 128
    batch_size = 128  # M dimension - must be >= 128 for cuDNN
    in_features = 256  # K dimension
    out_features = 128  # N dimension

    print(f"\n{'=' * 60}")
    print(f"Debug test: batch={batch_size}, in={in_features}, out={out_features}")
    print(f"{'=' * 60}")

    # Create simple known weight for debugging
    weight_bf16 = (
        torch.ones(out_features, in_features, device=device, dtype=torch.bfloat16) * 0.1
    )

    # Quantize to MXFP8
    weight_fp8, weight_scale = mxfp8_e4m3_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    print(f"\nWeight fp8 shape: {weight_fp8.shape}, dtype: {weight_fp8.dtype}")
    print(f"Weight scale shape: {weight_scale.shape}, dtype: {weight_scale.dtype}")
    print(f"Weight scale (first row): {weight_scale[0]}")

    # Create simple input
    input_tensor = torch.ones(
        batch_size, in_features, device=device, dtype=torch.bfloat16
    )

    print(f"\nInput shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # Run torch backend
    torch_backend = Mxfp8LinearOp(backend="torch")
    output_torch = torch_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=None,
    )

    print(f"\nTorch output shape: {output_torch.shape}")
    print(f"Torch output (first row): {output_torch[0, :8]}")
    t_min, t_max = output_torch.min().item(), output_torch.max().item()
    print(f"Torch output range: [{t_min:.4f}, {t_max:.4f}]")

    # Run flashinfer backend
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")

    # Also test flashinfer directly to compare
    print("\n--- Direct flashinfer bmm_mxfp8 test ---")
    from flashinfer import bmm_mxfp8 as fi_bmm_mxfp8
    from flashinfer import mxfp8_quantize as fi_mxfp8_quantize

    # Prepare input as 3D [b, m, k]
    input_3d = input_tensor.unsqueeze(0)  # [1, 2, 128]
    print(f"Input 3D shape: {input_3d.shape}")

    # Quantize input
    input_fi_mxfp8, input_fi_scale = fi_mxfp8_quantize(
        input_3d, is_sf_swizzled_layout=False
    )
    print(
        f"Input MXFP8 shape: {input_fi_mxfp8.shape}, "
        f"scale shape: {input_fi_scale.shape}"
    )

    # Prepare weight: dequant -> transpose -> make 3D -> requant
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    weight_bf16_dequant = dequant_mxfp8_to_bf16(weight_fp8, weight_scale)
    weight_t_bf16 = weight_bf16_dequant.t().contiguous()  # [K, N] = [128, 64]
    weight_3d = weight_t_bf16.unsqueeze(0)  # [1, 128, 64]
    print(f"Weight 3D shape: {weight_3d.shape}")
    print(f"Weight 3D (first block): {weight_3d[0, 0, :4]}")  # Should be ~0.1

    # Quantize weight
    weight_fi_mxfp8, weight_fi_scale = fi_mxfp8_quantize(
        weight_3d, is_sf_swizzled_layout=False
    )
    print(
        f"Weight MXFP8 shape: {weight_fi_mxfp8.shape}, "
        f"scale shape: {weight_fi_scale.shape}"
    )

    # Run bmm_mxfp8 directly
    output_fi_direct = fi_bmm_mxfp8(
        input_fi_mxfp8,
        weight_fi_mxfp8,
        input_fi_scale,
        weight_fi_scale,
        torch.bfloat16,
        backend="cudnn",
    )
    print(f"Direct flashinfer output shape: {output_fi_direct.shape}")
    print(f"Direct flashinfer output (first row): {output_fi_direct[0, 0, :8]}")
    d_min, d_max = output_fi_direct.min().item(), output_fi_direct.max().item()
    print(f"Direct flashinfer output range: [{d_min:.4f}, {d_max:.4f}]")

    # Now run through our wrapper
    output_flashinfer = flashinfer_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=None,
    )

    print(f"\nFlashinfer output shape: {output_flashinfer.shape}")
    print(f"Flashinfer output (first row): {output_flashinfer[0, :8]}")
    f_min, f_max = output_flashinfer.min().item(), output_flashinfer.max().item()
    print(f"Flashinfer output range: [{f_min:.4f}, {f_max:.4f}]")

    # Compute expected output (simple case: all ones input, all 0.1 weight)
    # Expected: each output element = sum of 0.1 * 1 * in_features = 0.1 * in_features
    expected_value = 0.1 * in_features
    print(f"\nExpected output value (approx): {expected_value}")

    # Compare
    abs_diff = (output_torch - output_flashinfer).abs()
    print(f"\nAbsolute difference (first row): {abs_diff[0, :8]}")
    print(f"Max absolute diff: {abs_diff.max().item():.6f}")
    print(f"Mean absolute diff: {abs_diff.mean().item():.6f}")

    # Check for issues
    if output_flashinfer.abs().max() < 1e-6:
        print("\n*** ERROR: Flashinfer output is near-zero! ***")
    if output_flashinfer.isnan().any():
        print("\n*** ERROR: Flashinfer output contains NaN! ***")
    if output_flashinfer.isinf().any():
        print("\n*** ERROR: Flashinfer output contains Inf! ***")

    # Final assertion
    assert torch.allclose(output_torch, output_flashinfer, rtol=0.1, atol=0.1), (
        "Outputs don't match!"
    )


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
@pytest.mark.parametrize(
    "m,k,n",
    [
        # Small M dimensions (decode-like scenarios)
        (1, 4096, 4096),
        (2, 4096, 4096),
        (4, 4096, 4096),
        (8, 4096, 4096),
        (16, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        # Medium M dimensions
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        # Large M dimensions (prefill-like scenarios)
        (935, 4096, 4096),  # This was the batch size that crashed in lm_eval
        (1024, 4096, 4096),
        # Test with different K/N (Llama hidden_size=4096, intermediate=14336)
        (128, 4096, 14336),
        (128, 14336, 4096),
        # Edge cases
        (1, 128, 128),
        (1, 256, 256),
        (7, 4096, 4096),  # Odd number
        (17, 4096, 4096),  # Prime-ish
        (63, 4096, 4096),  # Not power of 2
        (127, 4096, 4096),  # Just under 128
        (129, 4096, 4096),  # Just over 128
    ],
)
def test_mxfp8_linear_varying_dimensions(m, k, n):
    """
    Test flashinfer bmm_mxfp8 with varying dimensions to find failure cases.
    M = batch/sequence dimension (varies during inference)
    K = input features
    N = output features

    Run with:
        pytest tests/quantization/test_mxfp8_linear.py::\
test_mxfp8_linear_varying_dimensions -v -s
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    print(f"\n{'=' * 60}")
    print(f"Testing M={m}, K={k}, N={n}")
    print(f"{'=' * 60}")

    # Create weight [N, K] and input [M, K]
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.1
    input_tensor = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    # Quantize weight
    weight_fp8, weight_scale = mxfp8_e4m3_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    print(f"Weight shape: {weight_fp8.shape}, scale shape: {weight_scale.shape}")
    print(f"Input shape: {input_tensor.shape}")

    # Test torch backend first (baseline)
    torch_backend = Mxfp8LinearOp(backend="torch")
    try:
        output_torch = torch_backend.apply(
            input=input_tensor,
            weight=weight_fp8,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            bias=None,
        )
        print(f"Torch output shape: {output_torch.shape}")
        print(
            f"Torch output range: [{output_torch.min().item():.4f}, "
            f"{output_torch.max().item():.4f}]"
        )
    except Exception as e:
        pytest.fail(f"Torch backend failed: {e}")

    # Test flashinfer backend
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")
    try:
        # Synchronize before to catch any pending errors
        torch.cuda.synchronize()

        output_flashinfer = flashinfer_backend.apply(
            input=input_tensor,
            weight=weight_fp8,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            bias=None,
        )

        # Synchronize after to catch async errors
        torch.cuda.synchronize()

        print(f"Flashinfer output shape: {output_flashinfer.shape}")
        print(
            f"Flashinfer output range: [{output_flashinfer.min().item():.4f}, "
            f"{output_flashinfer.max().item():.4f}]"
        )

    except Exception as e:
        pytest.fail(f"Flashinfer backend failed with M={m}, K={k}, N={n}: {e}")

    # Check for NaN/Inf
    assert not output_flashinfer.isnan().any(), "Flashinfer output contains NaN!"
    assert not output_flashinfer.isinf().any(), "Flashinfer output contains Inf!"

    # Compare outputs
    abs_diff = (output_torch - output_flashinfer).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"Max absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")

    # Use relative tolerance based on output magnitude
    rtol = 0.15  # 15% relative tolerance for MXFP8
    atol = 0.5  # Absolute tolerance

    assert torch.allclose(output_torch, output_flashinfer, rtol=rtol, atol=atol), (
        f"Outputs don't match for M={m}, K={k}, N={n}! "
        f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
    )

    print("PASSED")


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_stress_varying_m():
    """
    Stress test: run many different M values in sequence to find memory issues.
    This simulates the varying batch sizes during lm_eval.

    Run with:
        pytest tests/quantization/test_mxfp8_linear.py::\
test_mxfp8_linear_stress_varying_m -v -s
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    k = 4096  # hidden_size
    n = 4096  # hidden_size

    # Varying M values to simulate real inference
    m_values = [
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        935,
        1024,
        2048,
    ]

    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")
    torch_backend = Mxfp8LinearOp(backend="torch")

    # Pre-create weight (stays constant)
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.1
    weight_fp8, weight_scale = mxfp8_e4m3_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    print(f"\nStress testing with K={k}, N={n}")
    print(f"Weight shape: {weight_fp8.shape}")

    failed_m_values = []

    for i, m in enumerate(m_values):
        try:
            input_tensor = torch.randn(m, k, device=device, dtype=torch.bfloat16)

            # Run torch baseline
            output_torch = torch_backend.apply(
                input=input_tensor,
                weight=weight_fp8,
                weight_scale=weight_scale,
                out_dtype=torch.bfloat16,
                bias=None,
            )

            # Run flashinfer
            torch.cuda.synchronize()
            output_flashinfer = flashinfer_backend.apply(
                input=input_tensor,
                weight=weight_fp8,
                weight_scale=weight_scale,
                out_dtype=torch.bfloat16,
                bias=None,
            )
            torch.cuda.synchronize()

            # Check for issues
            if output_flashinfer.isnan().any() or output_flashinfer.isinf().any():
                failed_m_values.append((m, "NaN/Inf in output"))
                print(f"  M={m}: FAILED (NaN/Inf)")
                continue

            max_diff = (output_torch - output_flashinfer).abs().max().item()
            if max_diff > 1.0:  # Very loose check
                failed_m_values.append((m, f"Large diff: {max_diff}"))
                print(f"  M={m}: FAILED (diff={max_diff:.4f})")
            else:
                print(f"  M={m}: OK (diff={max_diff:.4f})")

        except Exception as e:
            failed_m_values.append((m, str(e)))
            print(f"  M={m}: CRASHED - {e}")

    if failed_m_values:
        print(f"\n{'=' * 60}")
        print(f"FAILED M values: {failed_m_values}")
        print(f"{'=' * 60}")
        pytest.fail(f"Some M values failed: {failed_m_values}")
    else:
        print("\nAll M values passed!")


if __name__ == "__main__":
    # Run the debug test directly
    test_mxfp8_linear_debug_single()
