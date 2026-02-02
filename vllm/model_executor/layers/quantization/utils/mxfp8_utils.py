# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """
    Swizzle MXFP8 scale factors from row-major 2D layout to F8_128x4 layout.

    The F8_128x4 swizzle pattern uses 128x4 tiles where scales are arranged as:
    [m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]

    This is the inverse of unswizzle_mxfp8_scale.

    Args:
        sf: 2D scale tensor (uint8) of shape [M, K/32]
        M: Number of rows in the original tensor
        K: Number of columns in the original tensor (not K/32)

    Returns:
        1D swizzled scale tensor for FlashInfer mm_mxfp8 CUTLASS kernel
    """
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    # Calculate tile counts with padding
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    # Padded dimensions
    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    # Pad scale tensor to tile-aligned dimensions
    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    # Reshape to tile structure for swizzling
    # From: [m_padded, k_scale_padded] = [num_m_tiles * 128, num_k_tiles * 4]
    # The 2D unswizzled layout after reshape is [num_m_tiles, 4, 32, num_k_tiles, 4]
    # where 4*32 = 128 (inner_m, outer_m) and num_k_tiles*4 = k_scale_padded
    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    # Transpose dims 1 and 3 to get swizzled layout
    # [num_m_tiles, 4, 32, num_k_tiles, 4] -> [num_m_tiles, num_k_tiles, 32, 4, 4]
    sf_swizzled = sf_reshaped.transpose(1, 3)

    # Flatten to 1D
    return sf_swizzled.contiguous().view(-1)


def unswizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """
    Unswizzle MXFP8 scale factors from F8_128x4 layout to row-major 2D layout.

    The F8_128x4 swizzle pattern uses 128x4 tiles where scales are arranged as:
    [m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]

    Args:
        sf: 1D swizzled scale tensor (uint8)
        M: Number of rows in the original tensor
        K: Number of columns in the original tensor (not K/32)

    Returns:
        2D unswizzled scale tensor of shape [M, K/32]
    """
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    # Calculate tile counts with padding
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    # Reshape to tile structure
    # Layout: [num_m_tiles, num_k_tiles, 32, 4, 4]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)

    # Transpose to unswizzle
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)

    # Slice to original dimensions
    scale_cols = K // scaling_vector_size
    sf_unswizzle_sliced = sf_unswizzle[:M, :scale_cols]

    return sf_unswizzle_sliced.contiguous()


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from flashinfer import mxfp8_quantize as mxfp8_e4m3_quantize
    except ImportError as err:
        raise ImportError(
            "The package `flashinfer` is required to do "
            "MX-FP8 quantization. Please install it with"
            "`pip install flashinfer`"
        ) from err

    x_q, x_scales = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=is_sf_swizzled_layout)
    # For 3D tensors (bmm), keep scales as 1D - bmm_mxfp8 expects 1D scales
    # For 2D tensors, keep swizzled scales as 1D to match flashinfer behavior,
    # otherwise reshape to 2D for compatibility with existing code.
    if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def _cast_mxfp8_scales_to_bf16(scales: torch.Tensor) -> torch.Tensor:
    """
    Cast MXFP8 scales from uint8 to BF16.
    The scales are stored in uint8 format and need to be converted to BF16
    by left-shifting by 7 bits (to form the exponent) and reinterpreting
    as bfloat16.
    Args:
        scales: uint8 tensor containing MXFP8 scales
    Returns:
        BF16 tensor with the converted scales
    """
    return (scales.to(torch.int16) << 7).view(torch.bfloat16)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize MXFP8 tensor to BF16.

    Uses the same approach as ModelOpt's MXFP8QTensor.dequantize():
    - Convert E8M0 biased exponent to scale: descale = 2^(exponent - 127)
    - Reshape data to blocked form for broadcasting
    - Multiply and reshape back

    Args:
        x: FP8 E4M3 tensor to dequantize, shape [..., K]
        scales: uint8 E8M0 scale tensor, shape [..., K/32]

    Returns:
        BF16 dequantized tensor, shape [..., K]
    """
    # Convert FP8 to float32 for precision (matches ModelOpt)
    x_float = x.to(torch.float32)

    # Reshape to blocked form: [..., K] -> [..., K/32, 32]
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    # Convert E8M0 biased exponent to scale factor: descale = 2^(exponent - 127)
    # This matches ModelOpt: torch.exp2(e8m0_scale.float() - 127)
    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    # Broadcast multiply: [..., K/32] -> [..., K/32, 1] for broadcasting
    dequantized = x_blocked * descale.unsqueeze(-1)

    # Reshape back: [..., K/32, 32] -> [..., K]
    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fake implementation for torch.compile tracing.
    Returns empty tensors with the correct shapes and dtypes.
    """
    # FP8 quantized data has same shape as input
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    # Compute scale shape: one scale per block of 32 elements along last dim
    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            # When swizzled, scales are padded: M to multiple of 128, K to multiple of 4
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
        # Fallback for other dimensions
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=mxfp8_e4m3_quantize,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class Mxfp8LinearOp:
    """
    This class executes a MXFP8 linear layer.

    Supports two backends:
    - "torch": Dequantizes weights to bf16 and uses torch.nn.functional.linear
    - "flashinfer": Uses flashinfer's bmm_mxfp8 for native MXFP8 computation
    """

    def __init__(self, backend: str = "torch"):
        if backend not in ("torch", "flashinfer"):
            raise ValueError(
                f"Unsupported backend: {backend}. Must be 'torch' or 'flashinfer'."
            )
        self.backend = backend

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
        if self.backend == "flashinfer":
            return self._apply_flashinfer(
                input,
                weight,
                weight_scale,
                out_dtype,
                bias,
                out_features,
                in_features,
                weight_scale_2d,
            )
        return self._apply_torch(input, weight, weight_scale, out_dtype, bias)

    def _apply_torch(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize weights to bf16 and use torch.nn.functional.linear."""
        out_features, in_features = weight.shape
        scale_k = in_features // MXFP8_BLOCK_SIZE

        # Handle different weight_scale formats:
        # 1. Raw 2D uint8 from quantization: shape (out_features, scale_k)
        # 2. Processed 1D float8_e8m0fnu from process_weights_after_loading:
        #    shape (out_features_padded * scale_k,)
        if weight_scale.ndim == 2:
            # Raw 2D scale - use directly
            weight_scale_2d = weight_scale
            if weight_scale_2d.dtype != MXFP8_SCALE_DTYPE:
                weight_scale_2d = weight_scale_2d.view(MXFP8_SCALE_DTYPE)
        else:
            # Processed 1D scale - need to reshape
            weight_scale_uint8 = weight_scale.view(MXFP8_SCALE_DTYPE)
            out_features_padded = (out_features + 127) // 128 * 128
            weight_scale_2d_padded = weight_scale_uint8.view(
                out_features_padded, scale_k
            )
            weight_scale_2d = weight_scale_2d_padded[:out_features, :]

        # Dequantize weight to bf16
        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale_2d)

        # Standard linear operation
        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    # Minimum dimension size for cuDNN's F8_128x4 block scaling layout
    _BMM_MXFP8_MIN_DIM = 128

    def _apply_flashinfer(
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
        """Use flashinfer's mm_mxfp8 for native MXFP8 computation.

        This mirrors the mm_fp4 pattern:
        - Pass weight.t() and weight_scale.t() (views, no data copy)
        - Uses swizzled scales (1D) for optimal memory access performance
        - No dequant→transpose→requant needed!

        Expects pre-processed weights from process_weights_after_loading:
        - weight: (N, K) - original layout from checkpoint
        - weight_scale: 1D swizzled tensor for mm_mxfp8 CUTLASS kernel
        - weight_scale_2d: 2D non-swizzled [N, K/32] for torch fallback
        """
        # Weight is in original [N, K] layout
        N, K = weight.shape
        if out_features is not None:
            N = out_features
        if in_features is not None:
            K = in_features

        input_shape = input.shape
        input_2d = input.view(-1, K)
        M_orig = input_2d.shape[0]

        # Validate K and N dimensions
        min_dim = self._BMM_MXFP8_MIN_DIM
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

        # CUTLASS MXFP8 GEMM requires minimum M dimension for efficiency.
        # Fall back to torch (dequant to BF16) for small M.
        # if M_orig < min_dim:
        #    fallback_scale = (
        #        weight_scale_2d if weight_scale_2d is not None else weight_scale
        #    )
        #    return self._apply_torch(input, weight, fallback_scale, out_dtype, bias)

        # Pad M to a multiple of the minimum dimension (128) for CUTLASS.
        M_padded = ((M_orig + min_dim - 1) // min_dim) * min_dim
        if M_padded != M_orig:
            pad_rows = M_padded - M_orig
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        # Quantize input to MXFP8 with SWIZZLED scales.
        # FlashInfer mm_mxfp8 accuracy is MUCH better with swizzled scales:
        # - Swizzled: cos_sim ~0.999 (excellent)
        # - Non-swizzled: cos_sim ~0.83 (poor)
        # Weight scales are pre-swizzled in process_weights_after_loading.
        from flashinfer import mm_mxfp8, mxfp8_quantize

        # Quantize input with swizzled scale format
        input_mxfp8, input_scale = mxfp8_quantize(
            input_2d,
            is_sf_swizzled_layout=True,  # Swizzled for best accuracy
        )

        # Ensure the underlying tensor is contiguous
        if not weight.is_contiguous():
            weight = weight.contiguous()

        # Debug: Log tensor shapes before mm_mxfp8
        if not hasattr(self, "_logged_shapes"):
            self._logged_shapes = True
            logger.info(
                "mm_mxfp8 call: input=%s, weight=%s, input_scale=%s, weight_scale=%s",
                tuple(input_mxfp8.shape),
                tuple(weight.shape),
                tuple(input_scale.shape),
                tuple(weight_scale.shape),
            )

        # Both input_scale and weight_scale are 1D swizzled format
        # weight is [N, K], mm_mxfp8 expects b as [K, N]
        output = mm_mxfp8(
            input_mxfp8,  # [M, K] row-major
            weight.t(),  # [N, K] -> [K, N]
            input_scale,  # 1D swizzled
            weight_scale,  # 1D swizzled (from process_weights_after_loading)
            out_dtype=out_dtype,
            backend="cutlass",
        )

        # Slice output to remove padding if we padded M
        if M_padded != M_orig:
            output = output[:M_orig, :]

        if bias is not None:
            output = output + bias

        # Reshape back to original input shape
        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)
