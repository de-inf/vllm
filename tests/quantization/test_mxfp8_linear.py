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


def preprocess_weight_for_flashinfer(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
):
    """Pre-process an existing MXFP8 weight for flashinfer backend.

    This simulates what process_weights_after_loading does for flashinfer:
    - Keep weight in [N, K] layout (no transpose at load time)
    - Ensure scales are in 2D [N, K/32] non-swizzled format

    Runtime uses mm_mxfp8 with .t() pattern (like mm_fp4):
    - mm_mxfp8(input, weight.t(), input_scale, weight_scale.t(), dtype)
    - No data copy, just transposed views!

    Args:
        weight_fp8: MXFP8 weight in [N, K] layout
        weight_scale: Scale tensor in [N, K/32] layout (non-swizzled)

    Returns:
        (weight_fp8 [N, K], weight_scale [N, K/32])
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
    )

    N, K = weight_fp8.shape
    scale_k = K // MXFP8_BLOCK_SIZE

    # Ensure weight_scale is 2D non-swizzled [N, K/32]
    if weight_scale.ndim == 1:
        # Assume it's non-swizzled 1D, reshape to 2D
        weight_scale_2d = weight_scale.view(N, scale_k)
    else:
        weight_scale_2d = weight_scale[:N, :scale_k]

    # Ensure uint8 dtype
    if weight_scale_2d.dtype != torch.uint8:
        weight_scale_2d = weight_scale_2d.view(torch.uint8)

    # Return weight in original [N, K] layout with 2D scales
    return (
        weight_fp8,  # [N, K]
        weight_scale_2d.contiguous(),  # [N, K/32]
    )


def create_input(batch_size: int, in_features: int, device: str = "cuda"):
    """Create a random bf16 input tensor."""
    return torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "batch_size,in_features,out_features",
    [
        # Note: M (batch_size) will be padded to 128 if < 128
        (1, 128, 256),
        (4, 256, 512),
        (16, 512, 1024),
        (32, 1024, 2048),
        (64, 4096, 4096),
        (128, 4096, 4096),  # M >= 128, no padding needed
    ],
)
@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_backends_match(batch_size, in_features, out_features):
    """Test that both backends produce results close to bf16 reference.

    Uses cosine similarity as the metric (like flashinfer's own tests).
    MXFP8 has inherent quantization error, so we check that outputs are
    directionally correct rather than exactly matching.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    # Create SAME bf16 weight for both backends
    weight_fp8, weight_scale, weight_bf16 = create_mxfp8_weight(
        out_features, in_features, device
    )

    # Pre-process the SAME weight for flashinfer backend
    weight_fp8_fi, weight_scale_fi = preprocess_weight_for_flashinfer(
        weight_fp8, weight_scale
    )

    # Create input
    input_tensor = create_input(batch_size, in_features, device)

    # Compute bf16 reference
    reference = torch.nn.functional.linear(input_tensor, weight_bf16)

    # Create both backends
    torch_backend = Mxfp8LinearOp(backend="torch")
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")

    # Run torch backend
    output_torch = torch_backend.apply(
        input=input_tensor,
        weight=weight_fp8,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        bias=None,
    )

    # Run flashinfer backend with pre-processed weights
    output_flashinfer = flashinfer_backend.apply(
        input=input_tensor,
        weight=weight_fp8_fi,
        weight_scale=weight_scale_fi,
        out_dtype=torch.bfloat16,
        bias=None,
        out_features=out_features,
        in_features=in_features,
    )

    # Check shapes match
    assert output_torch.shape == output_flashinfer.shape == reference.shape, (
        f"Shape mismatch: torch={output_torch.shape}, "
        f"flashinfer={output_flashinfer.shape}, ref={reference.shape}"
    )

    # Check for issues
    assert not output_flashinfer.isnan().any(), "Flashinfer output contains NaN!"
    assert not output_flashinfer.isinf().any(), "Flashinfer output contains Inf!"
    assert output_flashinfer.abs().max() > 1e-6, "Flashinfer output is near-zero!"

    # Use cosine similarity as metric (like flashinfer's own tests)
    # This measures directional correctness, allowing for magnitude differences
    # that are inherent in MXFP8 quantization.
    # We use 0.9 threshold (same as flashinfer's tests) because our runtime
    # transpose approach avoids double quantization.
    min_cos_sim = 0.9

    ref_flat = reference.reshape(-1).float()
    torch_flat = output_torch.reshape(-1).float()
    fi_flat = output_flashinfer.reshape(-1).float()

    torch_cos_sim = torch.nn.functional.cosine_similarity(
        ref_flat, torch_flat, dim=0
    ).item()
    fi_cos_sim = torch.nn.functional.cosine_similarity(ref_flat, fi_flat, dim=0).item()

    print(
        f"\n--- Shape: ({batch_size}, {in_features}) "
        f"x ({out_features}, {in_features}) ---"
    )
    print(f"Output shape: {output_torch.shape}")
    print("\nCosine similarity vs bf16 reference:")
    print(f"  Torch: {torch_cos_sim:.4f}")
    print(f"  Flashinfer: {fi_cos_sim:.4f}")

    assert torch_cos_sim > min_cos_sim, (
        f"Torch cosine similarity too low: {torch_cos_sim:.4f} (need > {min_cos_sim})"
    )
    assert fi_cos_sim > min_cos_sim, (
        f"Flashinfer cosine similarity too low: {fi_cos_sim:.4f} (need > {min_cos_sim})"
    )


@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_with_bias(batch_size):
    """Test that bias is correctly applied in both backends."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    in_features = 256
    out_features = 512

    # Create weight
    weight_fp8, weight_scale, weight_bf16 = create_mxfp8_weight(
        out_features, in_features, device
    )

    # Pre-process weight for flashinfer backend
    weight_fp8_fi, weight_scale_fi = preprocess_weight_for_flashinfer(
        weight_fp8, weight_scale
    )

    # Create input and bias
    input_tensor = create_input(batch_size, in_features, device)
    bias = torch.randn(out_features, device=device, dtype=torch.bfloat16)

    # Compute bf16 reference with bias
    reference = torch.nn.functional.linear(input_tensor, weight_bf16, bias)

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
        weight=weight_fp8_fi,
        weight_scale=weight_scale_fi,
        out_dtype=torch.bfloat16,
        bias=bias,
        out_features=out_features,
        in_features=in_features,
    )

    # Check no NaN/Inf
    assert not output_flashinfer.isnan().any(), "Flashinfer output contains NaN!"
    assert not output_flashinfer.isinf().any(), "Flashinfer output contains Inf!"

    # Use cosine similarity (0.85 threshold due to double quantization in preprocessing)
    min_cos_sim = 0.85
    ref_flat = reference.reshape(-1).float()
    torch_flat = output_torch.reshape(-1).float()
    fi_flat = output_flashinfer.reshape(-1).float()

    torch_cos = torch.nn.functional.cosine_similarity(
        ref_flat, torch_flat, dim=0
    ).item()
    fi_cos = torch.nn.functional.cosine_similarity(ref_flat, fi_flat, dim=0).item()

    print(f"\nbatch_size={batch_size}: torch_cos={torch_cos:.4f}, fi_cos={fi_cos:.4f}")

    assert torch_cos > min_cos_sim, f"Torch cosine sim too low: {torch_cos:.4f}"
    assert fi_cos > min_cos_sim, f"Flashinfer cosine sim too low: {fi_cos:.4f}"


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

    # Compute expected output (simple case: all ones input, all 0.1 weight)
    # Expected: each output element = sum of 0.1 * 1 * in_features = 0.1 * in_features
    expected_value = 0.1 * in_features
    print(f"Expected output value (approx): {expected_value}")

    # Pre-process weight for flashinfer using the standard function
    weight_fp8_fi, weight_scale_fi = preprocess_weight_for_flashinfer(
        weight_fp8, weight_scale
    )

    print("\n--- Flashinfer pre-processed weight ---")
    print(f"Weight fp8 shape: {weight_fp8_fi.shape}")
    print(f"Weight scale shape: {weight_scale_fi.shape}")

    # Run flashinfer backend
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")
    output_flashinfer = flashinfer_backend.apply(
        input=input_tensor,
        weight=weight_fp8_fi,
        weight_scale=weight_scale_fi,
        out_dtype=torch.bfloat16,
        bias=None,
        out_features=out_features,
        in_features=in_features,
    )

    print(f"\nFlashinfer output shape: {output_flashinfer.shape}")
    print(f"Flashinfer output (first row): {output_flashinfer[0, :8]}")
    f_min, f_max = output_flashinfer.min().item(), output_flashinfer.max().item()
    print(f"Flashinfer output range: [{f_min:.4f}, {f_max:.4f}]")

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
    Test flashinfer bmm_mxfp8 with varying dimensions.
    Uses cosine similarity as metric (like flashinfer's own tests).
    
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

    # Compute bf16 reference
    reference = torch.nn.functional.linear(input_tensor, weight_bf16)

    # Quantize weight for torch backend
    weight_fp8, weight_scale = mxfp8_e4m3_quantize(
        weight_bf16, is_sf_swizzled_layout=False
    )

    print(f"Weight shape: {weight_fp8.shape}, scale shape: {weight_scale.shape}")
    print(f"Input shape: {input_tensor.shape}")

    # Test torch backend
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

    # Pre-process weight for flashinfer backend
    weight_fp8_fi, weight_scale_fi = preprocess_weight_for_flashinfer(
        weight_fp8, weight_scale
    )

    # Test flashinfer backend
    flashinfer_backend = Mxfp8LinearOp(backend="flashinfer")
    try:
        torch.cuda.synchronize()

        output_flashinfer = flashinfer_backend.apply(
            input=input_tensor,
            weight=weight_fp8_fi,
            weight_scale=weight_scale_fi,
            out_dtype=torch.bfloat16,
            bias=None,
            out_features=n,
            in_features=k,
        )

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

    # Use cosine similarity as metric
    # 0.85 threshold due to double quantization in our preprocessing path
    min_cos_sim = 0.85
    ref_flat = reference.reshape(-1).float()
    torch_flat = output_torch.reshape(-1).float()
    fi_flat = output_flashinfer.reshape(-1).float()

    torch_cos = torch.nn.functional.cosine_similarity(
        ref_flat, torch_flat, dim=0
    ).item()
    fi_cos = torch.nn.functional.cosine_similarity(ref_flat, fi_flat, dim=0).item()

    print("\nCosine similarity vs bf16 reference:")
    print(f"  Torch: {torch_cos:.4f}")
    print(f"  Flashinfer: {fi_cos:.4f}")

    assert torch_cos > min_cos_sim, (
        f"Torch cosine sim too low for M={m}, K={k}, N={n}: {torch_cos:.4f}"
    )
    assert fi_cos > min_cos_sim, (
        f"Flashinfer cosine sim too low for M={m}, K={k}, N={n}: {fi_cos:.4f}"
    )

    print("PASSED")


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
def test_mxfp8_linear_stress_varying_m():
    """
    Stress test: run many different M values in sequence to find memory issues.
    This simulates the varying batch sizes during lm_eval.
    Uses cosine similarity as metric (like flashinfer's own tests).

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

    # Pre-process weight for flashinfer
    weight_fp8_fi, weight_scale_fi = preprocess_weight_for_flashinfer(
        weight_fp8, weight_scale
    )

    print(f"\nStress testing with K={k}, N={n}")
    print(f"Weight shape: {weight_fp8.shape}")

    failed_m_values = []
    min_cos_sim = 0.85  # 0.85 threshold due to double quantization in preprocessing

    for i, m in enumerate(m_values):
        try:
            input_tensor = torch.randn(m, k, device=device, dtype=torch.bfloat16)

            # Compute bf16 reference
            reference = torch.nn.functional.linear(input_tensor, weight_bf16)

            # Run torch
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
                weight=weight_fp8_fi,
                weight_scale=weight_scale_fi,
                out_dtype=torch.bfloat16,
                bias=None,
                out_features=n,
                in_features=k,
            )
            torch.cuda.synchronize()

            # Check for issues
            if output_flashinfer.isnan().any() or output_flashinfer.isinf().any():
                failed_m_values.append((m, "NaN/Inf in output"))
                print(f"  M={m}: FAILED (NaN/Inf)")
                continue

            # Use cosine similarity
            ref_flat = reference.reshape(-1).float()
            torch_flat = output_torch.reshape(-1).float()
            fi_flat = output_flashinfer.reshape(-1).float()

            torch_cos = torch.nn.functional.cosine_similarity(
                ref_flat, torch_flat, dim=0
            ).item()
            fi_cos = torch.nn.functional.cosine_similarity(
                ref_flat, fi_flat, dim=0
            ).item()

            if fi_cos < min_cos_sim:
                failed_m_values.append((m, f"cos_sim={fi_cos:.4f}"))
                print(
                    f"  M={m}: FAILED (fi_cos={fi_cos:.4f}, torch_cos={torch_cos:.4f})"
                )
            else:
                print(f"  M={m}: OK (fi_cos={fi_cos:.4f}, torch_cos={torch_cos:.4f})")

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
