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
    if x_scales.ndim == 1:
        if is_sf_swizzled_layout:
            # TODO: check this, maybe not required?
            # When swizzled, scales are padded: M to multiple of 128, K to multiple of 4
            # We must use the padded dimensions, not the original input dimensions
            def _round_up(val: int, mult: int) -> int:
                return (val + mult - 1) // mult * mult

            M = x.size(0)
            K = x.size(-1) // MXFP8_BLOCK_SIZE
            M_padded = _round_up(M, 128)
            K_padded = _round_up(K, 4)
            x_scales = x_scales.view(M_padded, K_padded)
        else:
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
    Args:
        x: FP8 E4M3 tensor to dequantize
        scales: uint8 tensor containing MXFP8 scales
    Returns:
        BF16 dequantized tensor
    """
    scales_bf16 = _cast_mxfp8_scales_to_bf16(scales)
    # Repeat scales along the last dimension to match the block size
    scales_expanded = scales_bf16.reshape(*x.shape[:-1], -1).repeat_interleave(
        MXFP8_BLOCK_SIZE, dim=-1
    )
    return x.to(torch.bfloat16) * scales_expanded


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
                (M_padded, K_padded), dtype=MXFP8_SCALE_DTYPE, device=x.device
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
                (B, M_padded, K_padded), dtype=MXFP8_SCALE_DTYPE, device=x.device
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
    ) -> torch.Tensor:
        if self.backend == "flashinfer":
            return self._apply_flashinfer(input, weight, weight_scale, out_dtype, bias)
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

    def _apply_flashinfer(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use flashinfer's bmm_mxfp8 for native MXFP8 computation.

        flashinfer's bmm_mxfp8 expects:
        - A: (batch, M, K) - input, quantized as 3D tensor
        - B: (batch, K, N) - weight, quantized as 3D tensor
        - A_scale: 1D tensor from mxfp8_quantize (passed directly)
        - B_scale: 1D tensor from mxfp8_quantize (passed directly)

        IMPORTANT: Both A and B must be quantized as 3D tensors from the start.
        The scales are 1D tensors that match the flattened structure expected by cuDNN.
        """
        out_features, in_features = weight.shape
        input_shape = input.shape
        input_2d = input.view(-1, in_features)
        K = in_features
        N = out_features

        # Use swizzled=False to match the simpler non-swizzled scale layout
        # Both swizzled and non-swizzled work with cuDNN, but must be consistent
        is_swizzled = False

        # Reshape input to 3D [1, M, K] BEFORE quantizing
        # This ensures the scale structure matches what bmm_mxfp8 expects
        input_3d = input_2d.unsqueeze(0)  # [1, M, K]
        input_mxfp8, input_scale = torch.ops.vllm.mxfp8_quantize(input_3d, is_swizzled)

        # For B matrix, we need the weight in [1, K, N] layout
        # The checkpoint stores weight as [N, K] with scales for [N, K] layout
        # We need to dequantize, transpose, and re-quantize in [1, K, N] layout

        # Ensure weight_scale is in uint8 format
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            weight_scale = weight_scale.view(MXFP8_SCALE_DTYPE)

        # weight_scale should be 2D (N, K/32)
        if weight_scale.ndim == 1:
            scale_k = K // MXFP8_BLOCK_SIZE
            out_features_padded = (N + 127) // 128 * 128
            weight_scale = weight_scale.view(out_features_padded, scale_k)
        # Slice to actual size if padded
        weight_scale_2d = weight_scale[:N, : K // MXFP8_BLOCK_SIZE]

        # Dequantize weight to bf16, transpose, then re-quantize for flashinfer
        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale_2d)
        # Transpose to [K, N] and make contiguous, then add batch dim
        weight_t_bf16 = weight_bf16.t().contiguous()
        weight_3d = weight_t_bf16.unsqueeze(0)  # [1, K, N]

        # Re-quantize in [1, K, N] layout with same swizzled setting as input
        weight_t_mxfp8, weight_t_scale = torch.ops.vllm.mxfp8_quantize(
            weight_3d, is_swizzled
        )

        # Pass tensors and scales directly to bmm_mxfp8
        # A: [1, M, K], B: [1, K, N], scales are 1D tensors
        output = torch.ops.vllm.bmm_mxfp8(
            input_mxfp8, weight_t_mxfp8, input_scale, weight_t_scale, out_dtype, "cudnn"
        )

        # Remove batch dimension: (1, M, N) -> (M, N)
        output = output.squeeze(0)

        if bias is not None:
            output = output + bias

        # Reshape back to original input shape (except last dim is now out_features)
        output_shape = (*input_shape[:-1], out_features)
        return output.view(output_shape)
