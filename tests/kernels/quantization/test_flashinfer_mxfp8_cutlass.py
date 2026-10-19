# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CUTLASS MXFP8 linear, including N % 32 != 0, which the kernel
pads (CUTLASS mm_mxfp8 requires N % 32 == 0).

Run `pytest tests/kernels/quantization/test_flashinfer_mxfp8_cutlass.py`.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    FlashInferCutlassMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)

if not FlashInferCutlassMxfp8LinearKernel.is_supported()[0]:
    pytest.skip(
        reason="FlashInfer CUTLASS MXFP8 requires SM100+ and FlashInfer",
        allow_module_level=True,
    )


def _make_layer(weight: torch.Tensor) -> tuple[torch.nn.Module, torch.Tensor]:
    n, k = weight.shape
    weight_mxfp8, weight_scale = mxfp8_e4m3_quantize(
        weight, is_sf_swizzled_layout=False
    )
    weight_scale = weight_scale.view(n, k // MXFP8_BLOCK_SIZE)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    return layer, dequant_mxfp8_to_bf16(weight_mxfp8, weight_scale)


# 130 and 4632 are not multiples of 32; 4608 is.
@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("n", [130, 4632, 4608])
@torch.inference_mode()
def test_flashinfer_cutlass_mxfp8_linear(m: int, n: int) -> None:
    torch.manual_seed(0)
    k = 256
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.05
    layer, weight_dequant = _make_layer(weight)
    kernel = FlashInferCutlassMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(n, k))
    )
    kernel.process_weights_after_loading(layer)

    output = kernel.apply_weights(layer, x)
    assert output.shape == (m, n)

    x_mxfp8, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    x_dequant = dequant_mxfp8_to_bf16(x_mxfp8, x_scale.view(m, k // MXFP8_BLOCK_SIZE))
    reference = x_dequant.float() @ weight_dequant.float().t()
    error = (output.float() - reference).pow(2).mean().sqrt()
    assert error / reference.pow(2).mean().sqrt() < 1e-2
