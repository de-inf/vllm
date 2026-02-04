# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import Protocol

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _can_use_mxfp8_triton() -> bool:
    return HAS_TRITON


class Mxfp8Backend(Enum):
    TORCH = "torch"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"
    TRITON = "triton"


# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32

# Minimum dimension size for F8_128x4 block scaling layout
MXFP8_BMM_MIN_DIM = 128


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def unswizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Unswizzle MXFP8 scales from F8_128x4 to row-major 2D layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)

    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)

    scale_cols = K // scaling_vector_size
    sf_unswizzle_sliced = sf_unswizzle[:M, :scale_cols]

    return sf_unswizzle_sliced.contiguous()


# Optimized autotune configs for B200 Blackwell GPUs (trimmed for faster autotuning)
# B200 optimizations: num_stages=5 for better pipelining, num_warps=8 for larger blocks
# BLOCK_K is fixed at 32 for MXFP8 (required by per-32-element scaling)
_MXFP8_TRITON_CONFIGS = [
    # Large matrices (M >= 128) - high throughput configs
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_warps=8,
        num_stages=5,  # B200 benefits from more stages
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 16},
        num_warps=8,
        num_stages=5,
    ),
    # Medium matrices (32 <= M < 128)
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    # Small batch sizes (M < 32)
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
]


@triton.autotune(
    configs=_MXFP8_TRITON_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _mxfp8_block_scaled_mm_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
    output_type: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Must be 32 for MXFP8 (matches scale granularity)
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized MXFP8 block-scaled matrix multiplication kernel.

    Key optimizations over baseline:
    1. GROUP_SIZE_M for L2 cache-friendly tile ordering (swizzled access)
    2. Explicit pointer arithmetic with contiguity hints for coalescing
    3. Uses block_ptr for efficient tensor core utilization
    4. Higher num_stages in configs for better pipelining on modern GPUs

    Note: BLOCK_K must be 32 for MXFP8 since scales are per-32-elements.
    """
    # Output dtype selection
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.bfloat16
    else:
        output_dtype = tl.float32

    # Grouped tile ordering for L2 cache reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets with explicit pointer arithmetic
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masking
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Apply contiguity hints for better memory coalescing
    offs_m_safe = tl.where(mask_m, offs_m, 0)
    offs_n_safe = tl.where(mask_n, offs_n, 0)
    offs_m_safe = tl.max_contiguous(tl.multiple_of(offs_m_safe, BLOCK_M), BLOCK_M)
    offs_n_safe = tl.max_contiguous(tl.multiple_of(offs_n_safe, BLOCK_N), BLOCK_N)

    # Scale pointers (row-major 2D layout: [M, K/32] and [N, K/32])
    As_base = As + offs_m_safe * stride_As_m
    Bs_base = Bs + offs_n_safe * stride_Bs_n

    # Number of K iterations (each processes BLOCK_K=32 elements)
    num_k_iters = tl.cdiv(K, BLOCK_K)

    # Use block_ptr for A and B for efficient loading with tensor cores
    p_a = tl.make_block_ptr(
        A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    p_b = tl.make_block_ptr(
        B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop: iterate over K dimension in BLOCK_K (32) steps
    # Each iteration has exactly one scale per row/column
    # Optimized for B200: better pipelining with num_stages=5
    for k_idx in range(0, num_k_iters):
        # Load A and B blocks (BLOCK_K = 32 elements in K dimension)
        # Using block_ptr for efficient tensor core utilization on B200
        a = tl.load(p_a, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(p_b, boundary_check=(0, 1), padding_option="zero")

        # Load scales and convert from E8M0 (biased exponent) to float multiplier
        # Scales are uint8, convert to float32 then apply exp2
        a_s_raw = tl.load(As_base + k_idx * stride_As_k, mask=mask_m, other=0)
        b_s_raw = tl.load(Bs_base + k_idx * stride_Bs_k, mask=mask_n, other=0)

        # E8M0 to float: scale = 2^(value - 127)
        # The compiler optimizes the constant -127.0
        a_s = tl.exp2(a_s_raw.to(tl.float32) - 127.0)
        b_s = tl.exp2(b_s_raw.to(tl.float32) - 127.0)

        # Compute scaled dot product and accumulate
        # tl.dot uses tensor cores for FP8 when available on B200
        # Apply scales element-wise: (A @ B) * (a_scale[:, None] * b_scale[None, :])
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

        # Advance pointers for next iteration
        p_a = tl.advance(p_a, (0, BLOCK_K))
        p_b = tl.advance(p_b, (BLOCK_K, 0))

    # Convert accumulator to output dtype
    if output_dtype == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif output_dtype == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    # Store output with masking
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def mxfp8_block_scaled_matmul_triton(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Optimized MXFP8 block-scaled matrix multiplication using Triton.

    Computes: C = (A @ B.T) with per-32-element E8M0 scaling.

    Args:
        a: Input tensor of shape (M, K) in FP8 format
        a_scale: Scale tensor of shape (M, K/32) in uint8 E8M0 format
        b: Weight tensor of shape (N, K) in FP8 format (note: not transposed)
        b_scale: Scale tensor of shape (N, K/32) in uint8 E8M0 format
        output_dtype: Output data type (float32, float16, or bfloat16)

    Returns:
        Output tensor of shape (M, N) in output_dtype
    """
    if not _can_use_mxfp8_triton():
        raise RuntimeError("MXFP8 Triton kernel is unavailable.")

    M, K = a.shape
    N, K_b = b.shape
    assert K_b == K, f"K dimension mismatch: a.shape[1]={K}, b.shape[1]={K_b}"

    if output_dtype == torch.float32:
        output_type = 0
    elif output_dtype == torch.float16:
        output_type = 1
    elif output_dtype == torch.bfloat16:
        output_type = 2
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

    scale_k = K // MXFP8_BLOCK_SIZE
    assert a_scale.shape == (M, scale_k), (
        f"a_scale shape {tuple(a_scale.shape)} != (M={M}, K/32={scale_k})"
    )
    assert b_scale.shape == (N, scale_k), (
        f"b_scale shape {tuple(b_scale.shape)} != (N={N}, K/32={scale_k})"
    )

    # Ensure contiguous tensors for optimal memory access
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not a_scale.is_contiguous():
        a_scale = a_scale.contiguous()
    if not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()

    output = torch.empty((M, N), dtype=output_dtype, device=a.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _mxfp8_block_scaled_mm_kernel[grid](
        a,
        b,
        output,
        a_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(-2),
        a.stride(-1),
        b.stride(1),
        b.stride(0),
        output.stride(-2),
        output.stride(-1),
        a_scale.stride(-2),
        a_scale.stride(-1),
        b_scale.stride(-2),
        b_scale.stride(-1),
        output_type=output_type,
        # BLOCK_K, BLOCK_M, BLOCK_N, GROUP_SIZE_M come from autotune configs
    )
    return output


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    x_q, x_scales = flashinfer_mxfp8_quantize(
        x, is_sf_swizzled_layout=is_sf_swizzled_layout
    )
    if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class _Mxfp8LinearOpImpl(Protocol):
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class _Mxfp8TorchLinearOp:
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight_scale.ndim == 2:
            weight_scale_2d = weight_scale
        else:
            assert weight_scale.ndim == 1, f"Invalid {weight_scale.ndim=}"
            out_features_local, in_features_local = weight.shape
            weight_scale_2d = unswizzle_mxfp8_scale(
                weight_scale, M=out_features_local, K=in_features_local
            )

        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale_2d)

        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)


class _Mxfp8FlashinferLinearOp:
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, K = weight.shape
        if out_features is not None:
            N = out_features
        if in_features is not None:
            K = in_features

        input_shape = input.shape
        input_2d = input.view(-1, K)
        M_orig = input_2d.shape[0]

        min_dim = MXFP8_BMM_MIN_DIM
        assert min_dim <= K, (
            f"mm_mxfp8 requires K >= {min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )
        assert min_dim <= N, (
            f"mm_mxfp8 requires N >= {min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        M_padded = ((M_orig + min_dim - 1) // min_dim) * min_dim
        if M_padded != M_orig:
            pad_rows = M_padded - M_orig
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True,  # Swizzled for best accuracy
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )

        if M_padded != M_orig:
            output = output[:M_orig, :]

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class _Mxfp8TritonLinearOp:
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not HAS_TRITON:
            raise RuntimeError("Triton is not available for MXFP8 GEMM.")

        N, K = weight.shape
        if out_features is not None:
            N = out_features
        if in_features is not None:
            K = in_features

        input_shape = input.shape
        input_2d = input.view(-1, K)
        M_orig = input_2d.shape[0]

        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"MXFP8 GEMM requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )

        block_m = 128

        M_padded = ((M_orig + block_m - 1) // block_m) * block_m
        if M_padded != M_orig:
            pad_rows = M_padded - M_orig
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True,  # Swizzled for best accuracy
        )

        input_scale_2d = unswizzle_mxfp8_scale(input_scale, M=M_padded, K=K)

        if not weight.is_contiguous():
            weight = weight.contiguous()

        scale_k = K // MXFP8_BLOCK_SIZE
        if weight_scale_2d is not None:
            weight_scale_2d = weight_scale_2d[:N, :scale_k]
            if weight_scale_2d.dtype != MXFP8_SCALE_DTYPE:
                weight_scale_2d = weight_scale_2d.view(MXFP8_SCALE_DTYPE)
        elif weight_scale.ndim == 2:
            weight_scale_2d = weight_scale[:N, :scale_k]
        else:
            weight_scale_2d = unswizzle_mxfp8_scale(weight_scale, M=N, K=K)

        output = mxfp8_block_scaled_matmul_triton(
            input_mxfp8,
            input_scale_2d,
            weight,
            weight_scale_2d,
            out_dtype,
        )

        if M_padded != M_orig:
            output = output[:M_orig, :]

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class Mxfp8LinearOp:
    def __init__(self, backend: Mxfp8Backend):
        if backend not in Mxfp8Backend:
            raise ValueError(f"Unsupported backend: {backend}")

        if backend == Mxfp8Backend.TRITON and not _can_use_mxfp8_triton():
            logger.warning(
                "MXFP8 Triton backend is unavailable; falling back to torch."
            )
            backend = Mxfp8Backend.TORCH

        self.backend = backend
        self._impl: _Mxfp8LinearOpImpl
        if backend == Mxfp8Backend.FLASHINFER_CUTLASS:
            self._impl = _Mxfp8FlashinferLinearOp()
        elif backend == Mxfp8Backend.TRITON:
            self._impl = _Mxfp8TritonLinearOp()
        else:
            self._impl = _Mxfp8TorchLinearOp()

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
        # Flashinfer-specific metadata (set by process_weights_after_loading)
        out_features: int | None = None,
        in_features: int | None = None,
        weight_scale_2d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert weight_scale.dtype == MXFP8_SCALE_DTYPE
        assert weight_scale.ndim == 1 or weight_scale.ndim == 2

        return self._impl.apply(
            input,
            weight,
            weight_scale,
            out_dtype,
            bias,
            out_features,
            in_features,
            weight_scale_2d,
        )
