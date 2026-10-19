# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE backend selection for the FlashInfer CuTe-DSL kernel (Blackwell).

GPU-free: patches the device gate, then exercises the oracle so the CuTe-DSL
backend is reachable with ``--moe-backend flashinfer_cutedsl``, is never picked
automatically, and declines configs the kernel does not implement.
"""

import dataclasses
from unittest.mock import patch

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_mxfp8_moe import (  # noqa: E501
    FlashInferCuteDSLMxfp8Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    _BACKEND_NAME_MAP,
    _SUPPORTED_BACKENDS,
    _mxfp8_backend_to_kernel_cls,
    _select_kernel_cls,
    select_mxfp8_moe_backend,
)

_BACKEND = Fp8MoeBackend.FLASHINFER_CUTEDSL_MXFP8


def _config(
    activation: MoEActivation = MoEActivation.RELU2_NO_MUL,
    hidden_dim: int = 1024,
    intermediate_size: int = 2688,
    in_dtype: torch.dtype = torch.bfloat16,
):
    cfg = make_dummy_moe_config(
        num_experts=512,
        experts_per_token=22,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        in_dtype=in_dtype,
        activation=activation,
    )
    return dataclasses.replace(cfg, moe_backend="flashinfer_cutedsl")


def _blackwell():
    """Patch the device gate so selection can run without a Blackwell GPU."""
    return patch.object(
        FlashInferCuteDSLMxfp8Experts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )


def test_cutedsl_mxfp8_registered_opt_in_only():
    assert _BACKEND_NAME_MAP["flashinfer_cutedsl"] is _BACKEND
    assert _BACKEND not in _SUPPORTED_BACKENDS
    assert _mxfp8_backend_to_kernel_cls(_BACKEND) == [FlashInferCuteDSLMxfp8Experts]


def test_cutedsl_mxfp8_selected_by_name():
    with _blackwell():
        backend, experts_cls = select_mxfp8_moe_backend(_config())
    assert backend is _BACKEND
    assert experts_cls is FlashInferCuteDSLMxfp8Experts


def test_cutedsl_mxfp8_selected_with_dp_ep():
    cfg = _config()
    cfg = dataclasses.replace(
        cfg,
        moe_parallel_config=dataclasses.replace(
            cfg.moe_parallel_config, dp_size=4, ep_size=4, use_ep=True
        ),
    )
    with _blackwell():
        assert _select_kernel_cls(_BACKEND, cfg) is FlashInferCuteDSLMxfp8Experts


@pytest.mark.parametrize(
    "config",
    [
        _config(activation=MoEActivation.SILU),
        _config(in_dtype=torch.float16),
        _config(hidden_dim=1000),
        _config(intermediate_size=2600),
    ],
    ids=["gated", "fp16", "hidden", "intermediate"],
)
def test_cutedsl_mxfp8_declines_unsupported(config):
    with _blackwell(), pytest.raises(ValueError):
        _select_kernel_cls(_BACKEND, config)
