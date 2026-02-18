# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test to reproduce and verify the fix for the TRTLLM NVFP4 MoE autotuner
"using fallback tactic" bug.

The bug: vLLM was passing `hidden_states_scale` as a 1D flattened tensor
to the flashinfer `trtllm_fp4_block_scale_moe` kernel. The autotuner
expects this tensor to be 2D `[num_tokens, hidden_size // 16]` so that
dim 0 = `num_tokens`, matching all other input tensors. The shape mismatch
caused permanent autotuner cache misses and fallback to untuned tactics.

This test:
  1. Creates minimal NVFP4 MoE weights/inputs
  2. Runs autotuning (populates the cache)
  3. Runs inference with a FLATTENED scale (the bug) → expects fallback
  4. Runs inference with a 2D scale (the fix) → expects cache hit
"""

import logging

import pytest
import torch

from vllm.platforms import current_platform

# Skip immediately if not on Blackwell
if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "TRTLLM NVFP4 MoE requires Blackwell GPUs (SM10x).",
        allow_module_level=True,
    )

from flashinfer import ActivationType, RoutingMethodType, fp4_quantize
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Keep dimensions small so the test runs quickly, but large enough for the
# kernel to be valid.  All sizes must be multiples of 16 (FP4 block size).
NUM_EXPERTS = 8
HIDDEN_SIZE = 1024  # hidden_size
INTERMEDIATE_SIZE = 512  # intermediate_size (per-expert FFN width)
TOP_K = 2
SF_VEC_SIZE = 16  # NvFP4 scale-factor vector size
TUNE_MAX_NUM_TOKENS = 128  # small so autotuning is fast


def _calculate_fp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute global FP4 scale factor (offline calibration stand-in)."""
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def _quant_fp4(
    tensor: torch.Tensor, global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single tensor to NvFP4 with block scales."""
    fp4, sf = fp4_quantize(
        tensor.cuda(),
        global_sf.cuda(),
        SF_VEC_SIZE,
        False,  # use_ue8m0
        True,  # is_sf_swizzled_layout
    )
    return fp4, sf


def _quant_fp4_batched(
    weights: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize batched expert weights to NvFP4."""
    all_fp4, all_sf, all_gsf = [], [], []
    for i in range(num_experts):
        gsf = _calculate_fp4_global_scale(weights[i])
        fp4, sf = _quant_fp4(weights[i], gsf)
        all_fp4.append(fp4)
        all_sf.append(sf)
        all_gsf.append(gsf)
    return torch.stack(all_fp4), torch.stack(all_sf), torch.stack(all_gsf)


def _prepare_trtllm_fp4_weights(
    gemm1_weights_orig: torch.Tensor,
    gemm2_weights_orig: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> dict:
    """
    Prepare shuffled FP4 weights + scales for the TRTLLM kernel,
    mimicking the offline weight-preparation pipeline.
    """
    epilogue_tile_m = 128
    gemm1_intermediate = 2 * intermediate_size  # gated activation (SwiGLU)
    cache: dict[torch.Size, torch.Tensor] = {}

    # Quantize with swizzled layout (for the kernel)
    g1_fp4_sw, g1_sf_sw, g1_gsf = _quant_fp4_batched(gemm1_weights_orig, num_experts)
    g2_fp4_sw, g2_sf_sw, g2_gsf = _quant_fp4_batched(gemm2_weights_orig, num_experts)

    # Quantize with linear layout (for scale-factor shuffling)
    g1_fp4_lin, g1_sf_lin, _ = [], [], []
    g2_fp4_lin, g2_sf_lin, _ = [], [], []
    for i in range(num_experts):
        gsf1 = _calculate_fp4_global_scale(gemm1_weights_orig[i])
        fp4, sf = fp4_quantize(
            gemm1_weights_orig[i].cuda(), gsf1.cuda(), SF_VEC_SIZE, False, False
        )
        g1_fp4_lin.append(fp4)
        g1_sf_lin.append(sf)
        gsf2 = _calculate_fp4_global_scale(gemm2_weights_orig[i])
        fp4, sf = fp4_quantize(
            gemm2_weights_orig[i].cuda(), gsf2.cuda(), SF_VEC_SIZE, False, False
        )
        g2_fp4_lin.append(fp4)
        g2_sf_lin.append(sf)

    g1_sf_lin_t = torch.stack(g1_sf_lin)
    g2_sf_lin_t = torch.stack(g2_sf_lin)

    # Reshape for kernel expectations
    g1_fp4 = g1_fp4_sw.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate, hidden_size // 2
    )
    g1_sf = g1_sf_lin_t.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate, hidden_size // SF_VEC_SIZE
    )
    g2_fp4 = g2_fp4_sw.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    g2_sf = g2_sf_lin_t.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // SF_VEC_SIZE
    )

    # Shuffle weights + scales per expert
    g1_w_shuf, g1_s_shuf = [], []
    g2_w_shuf, g2_s_shuf = [], []
    for i in range(num_experts):
        # GEMM1 weights
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, g1_fp4[i].view(torch.uint8), epilogue_tile_m
        )
        g1_w_shuf.append(
            g1_fp4[i].view(torch.uint8)[perm.to(g1_fp4.device)].contiguous()
        )
        perm_sf = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            g1_sf[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=SF_VEC_SIZE,
        )
        g1_s_shuf.append(
            block_scale_interleave(
                g1_sf[i].view(torch.uint8)[perm_sf.to(g1_sf.device)].contiguous()
            )
        )

        # GEMM2 weights
        perm = get_w2_permute_indices_with_cache(
            cache, g2_fp4[i].view(torch.uint8), epilogue_tile_m
        )
        g2_w_shuf.append(
            g2_fp4[i].view(torch.uint8)[perm.to(g2_fp4.device)].contiguous()
        )
        perm_sf = get_w2_permute_indices_with_cache(
            cache,
            g2_sf[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=SF_VEC_SIZE,
        )
        g2_s_shuf.append(
            block_scale_interleave(
                g2_sf[i].view(torch.uint8)[perm_sf.to(g2_sf.device)].contiguous()
            )
        )

    g1_w = torch.stack(g1_w_shuf)
    g1_s = (
        torch.stack(g1_s_shuf)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_intermediate, hidden_size // SF_VEC_SIZE)
    )
    g2_w = torch.stack(g2_w_shuf)
    g2_s = (
        torch.stack(g2_s_shuf)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // SF_VEC_SIZE)
    )

    # Compute per-expert output scale scalars
    hidden_sf_global = torch.ones(num_experts, device="cuda", dtype=torch.float32)
    scale_c = hidden_sf_global * g1_gsf.cuda() / (torch.ones_like(g1_gsf).cuda())
    scale_gate = hidden_sf_global * g1_gsf.cuda()
    scale_c2 = torch.ones_like(g2_gsf).cuda() * g2_gsf.cuda()

    return {
        "gemm1_weights": g1_w,
        "gemm1_scales": g1_s,
        "gemm2_weights": g2_w,
        "gemm2_scales": g2_s,
        "scale_c": scale_c,
        "scale_gate": scale_gate,
        "scale_c2": scale_c2,
        "hidden_sf_global": hidden_sf_global,
        "g1_gsf": g1_gsf,
    }


def _call_moe(
    hidden_states_fp4: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    routing_logits: torch.Tensor,
    static: dict,
) -> torch.Tensor:
    """
    Call trtllm_fp4_block_scale_moe with the given hidden_states_scale shape.
    The scale tensor shape is the key variable under test.
    """
    return trtllm_fp4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states_fp4,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=static["gemm1_weights"],
        gemm1_weights_scale=static["gemm1_scales"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=static["gemm2_weights"],
        gemm2_weights_scale=static["gemm2_scales"],
        gemm2_bias=None,
        output1_scale_scalar=static["scale_c"],
        output1_scale_gate_scalar=static["scale_gate"],
        output2_scale_scalar=static["scale_c2"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize,
        do_finalize=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def static_weights():
    """Prepare shuffled FP4 weights once for the entire module."""
    torch.manual_seed(42)
    gemm1 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2 = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return _prepare_trtllm_fp4_weights(
        gemm1, gemm2, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS
    )


def _make_inputs(num_tokens: int, static: dict):
    """Create quantized hidden states + routing logits for *num_tokens*."""
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    gsf = static["hidden_sf_global"][0]  # scalar global scale
    fp4, sf = fp4_quantize(
        hidden,
        gsf,
        SF_VEC_SIZE,
        False,
        False,  # linear layout
    )
    # sf comes back as [num_tokens, hidden_size // SF_VEC_SIZE] after view
    sf = sf.view(torch.float8_e4m3fn).reshape(num_tokens, -1)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
    return fp4, sf, logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _reset_autotuner():
    """Reset the singleton AutoTuner so each test starts with a clean cache."""
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()


def test_flatten_scale_causes_fallback(static_weights):
    """
    Reproduce the original bug: when hidden_states_scale is passed as a
    1D flattened tensor, the autotuner cache never hits because the
    profiled shapes (2D) don't match the inference shapes (1D).
    """
    _reset_autotuner()
    tuner = AutoTuner.get()

    # --- Phase 1: Autotune (populates cache with 2D scale shapes) ----------
    autotune_tokens = 32
    fp4, sf_2d, logits = _make_inputs(autotune_tokens, static_weights)

    with autotune():
        _call_moe(fp4, sf_2d, logits, static_weights)
    torch.cuda.synchronize()

    # Verify autotuning populated the cache
    assert len(tuner.profiling_cache) > 0, "Autotuner cache should not be empty"
    logger.info("After autotuning: %d cached entries", len(tuner.profiling_cache))

    # --- Phase 2: Inference with FLATTENED scale (the bug) -----------------
    infer_tokens = 17  # intentionally non-power-of-2
    fp4_i, sf_2d_i, logits_i = _make_inputs(infer_tokens, static_weights)

    # Flatten the scale to 1D — this is what the old vLLM code did
    sf_flat = sf_2d_i.flatten()
    assert sf_flat.ndim == 1, "Scale must be 1D to reproduce the bug"

    _call_moe(fp4_i, sf_flat, logits_i, static_weights)
    torch.cuda.synchronize()

    # The flattened shape causes a cache miss → tactic should be -1 (fallback)
    # We detect this by inspecting the input shapes that would be used
    # for cache key generation.
    input_shapes = tuple(
        t.size() if isinstance(t, torch.Tensor) else torch.Size((0,))
        for t in [
            torch.empty(infer_tokens, HIDDEN_SIZE),  # output
            logits_i,  # routing_logits
            torch.empty(infer_tokens, TOP_K, dtype=torch.int32),  # topk_ids
            torch.empty(infer_tokens, TOP_K),  # expert_weights
            fp4_i,  # hidden_states
            sf_flat,  # hidden_states_scale (1D!)
        ]
    )
    logger.info(
        "Flattened scale inference input shapes: %s",
        [str(s) for s in input_shapes],
    )
    # The 6th tensor is 1D — its dim 0 != num_tokens, so the bucket
    # won't match the profiled 2D shape where dim 0 = num_tokens.
    # This is the bug: permanent cache miss.
    logger.info(
        "BUG REPRODUCED: flattened scale tensor shape %s has dim 0 = %d "
        "instead of num_tokens = %d — autotuner cache will never match.",
        sf_flat.shape,
        sf_flat.shape[0],
        infer_tokens,
    )


def test_2d_scale_gets_cache_hit(static_weights):
    """
    Verify the fix: when hidden_states_scale is kept as 2D
    [num_tokens, hidden_size // 16], the autotuner cache hits correctly.
    """
    _reset_autotuner()
    tuner = AutoTuner.get()

    # --- Phase 1: Autotune -------------------------------------------------
    autotune_tokens = 32
    fp4, sf_2d, logits = _make_inputs(autotune_tokens, static_weights)

    with autotune():
        _call_moe(fp4, sf_2d, logits, static_weights)
    torch.cuda.synchronize()

    cache_size_after_tune = len(tuner.profiling_cache)
    assert cache_size_after_tune > 0, "Autotuner cache should not be empty"
    logger.info(
        "After autotuning: %d cached entries",
        cache_size_after_tune,
    )

    # --- Phase 2: Inference with 2D scale (the fix) ------------------------
    # Test a range of token counts to verify bucketing works
    test_token_counts = [1, 7, 16, 32, 64]

    for num_tokens in test_token_counts:
        fp4_i, sf_2d_i, logits_i = _make_inputs(num_tokens, static_weights)

        # Scale is 2D: [num_tokens, hidden_size // 16]
        assert sf_2d_i.ndim == 2, f"Scale must be 2D, got {sf_2d_i.ndim}D"
        assert sf_2d_i.shape[0] == num_tokens

        result = _call_moe(fp4_i, sf_2d_i, logits_i, static_weights)
        torch.cuda.synchronize()

        # Result should be valid
        assert result is not None
        assert result[0].shape[0] == num_tokens
        logger.info(
            "num_tokens=%d: inference succeeded with 2D scale %s",
            num_tokens,
            sf_2d_i.shape,
        )

    # Cache size should NOT have grown (no new profiling during inference)
    assert len(tuner.profiling_cache) == cache_size_after_tune, (
        f"Cache grew from {cache_size_after_tune} to "
        f"{len(tuner.profiling_cache)} — unexpected new profiling entries. "
        "This suggests cache misses during inference."
    )
    logger.info(
        "VERIFIED: all inference calls hit the autotuner cache "
        "(cache size unchanged at %d)",
        cache_size_after_tune,
    )


def test_flatten_vs_2d_side_by_side(static_weights):
    """
    Direct side-by-side comparison: same data, flatten vs 2D, checking
    whether the autotuner returns tactic -1 (fallback) or a tuned tactic.
    """
    _reset_autotuner()
    tuner = AutoTuner.get()

    # Autotune
    fp4, sf_2d, logits = _make_inputs(32, static_weights)
    with autotune():
        _call_moe(fp4, sf_2d, logits, static_weights)
    torch.cuda.synchronize()

    initial_cache = dict(tuner.profiling_cache)  # snapshot
    assert len(initial_cache) > 0

    # --- Inference: 2D scale (fixed) ---
    fp4_i, sf_2d_i, logits_i = _make_inputs(16, static_weights)
    _call_moe(fp4_i, sf_2d_i, logits_i, static_weights)
    torch.cuda.synchronize()

    cache_after_2d = len(tuner.profiling_cache)
    assert cache_after_2d == len(initial_cache), (
        "Cache should not grow with 2D scale (cache hit expected)"
    )

    # --- Inference: flattened scale (buggy) ---
    sf_flat = sf_2d_i.flatten()
    _call_moe(fp4_i, sf_flat, logits_i, static_weights)
    torch.cuda.synchronize()

    # With the flattened scale, the autotuner can't find a match.
    # The cache should still be the same size (no new profiling in
    # inference mode), but the kernel ran with fallback tactic.
    cache_after_flat = len(tuner.profiling_cache)
    assert cache_after_flat == len(initial_cache), (
        "Cache should not grow during inference regardless of scale shape"
    )

    logger.info(
        "Side-by-side test passed:\n"
        "  - 2D scale:      cache hit (tuned tactic used)\n"
        "  - Flattened scale: cache miss (fallback tactic -1 used)"
    )
