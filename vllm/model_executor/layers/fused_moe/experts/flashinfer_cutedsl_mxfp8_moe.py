# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_row_major_sf_to_mma_layout,
    flashinfer_cute_dsl_fused_moe,
    has_flashinfer_cutedsl_moe_mxfp8,
)


def _to_mma_layout(scale: torch.Tensor) -> torch.Tensor:
    """Re-lay out row-major [E, rows, K // 32] E8M0 scales for the kernel."""
    num_experts, rows, k_blocks = scale.shape
    return flashinfer_convert_row_major_sf_to_mma_layout(
        scale.view(torch.uint8),
        m=rows,
        k=k_blocks * MXFP8_BLOCK_SIZE,
        num_groups=num_experts,
        sf_vec_size=MXFP8_BLOCK_SIZE,
    )


class FlashInferCuteDSLMxfp8Experts(mk.FusedMoEExpertsModular):
    """CuteDSL MXFP8 x MXFP8 MoE experts using the FlashInfer functional API.

    Weights stay MXFP8 (E4M3 + E8M0 block-32 scales); the scales are re-laid
    out for the kernel once, when the experts are built. Activations are
    quantized to MXFP8 here, once per layer call. The kernel routes, runs the
    experts and reduces over top-k in one pass. Only non-gated ReLU² experts
    are claimed for now; other models keep their existing MXFP8 backends.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "mxfp8", (
            "Only mxfp8 quantization is supported."
        )
        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None
        self.hidden_dim = moe_config.hidden_dim
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.local_expert_offset = self.ep_rank * self.local_num_experts
        # The experts are built after the weights are final, so the kernel's
        # scale layout is computed once here; the layer keeps the row-major
        # originals.
        self.w1_sf = _to_mma_layout(quant_config.w1_scale)
        self.w2_sf = _to_mma_layout(quant_config.w2_scale)
        # MXFP8 has no global scales; the kernel takes a per-expert alpha of one.
        self.alpha = torch.ones(
            self.local_num_experts,
            dtype=torch.float32,
            device=quant_config.w1_scale.device,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype != torch.bfloat16:
            return False, "kernel supports only bfloat16 activations and output"
        # The grouped GEMMs tile K and N in blocks of 128 (MXFP8 block scales are
        # swizzled in 128 x 4 tiles).
        for name, dim in (
            ("hidden", moe_config.hidden_dim),
            ("intermediate per partition", moe_config.intermediate_size_per_partition),
        ):
            if dim % 128 != 0:
                return False, (
                    f"kernel does not support a {name} size of {dim} "
                    "(must be a multiple of 128)"
                )
        return True, None

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_cutedsl_moe_mxfp8()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.RELU2_NO_MUL

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # Every rank sees all tokens with global expert ids (replicated under TP, or
        # all-gathered by the Standard-format DP dispatchers); EP is handled by
        # local_expert_offset / num_local_experts in the kernel. Batched-format
        # dispatchers are excluded by activation_format().
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return ((0,), (0,), (M, self.hidden_dim))

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert not apply_router_weight_on_input
        num_tokens = hidden_states.shape[0]
        x_q, x_sf = mxfp8_e4m3_quantize(
            hidden_states.contiguous(), is_sf_swizzled_layout=False
        )
        x_sf = x_sf.view(torch.uint8).reshape(num_tokens, -1)

        flashinfer_cute_dsl_fused_moe(
            x=x_q,
            x_sf=x_sf,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights.float(),
            w1_weight=w1,
            w1_weight_sf=self.w1_sf,
            w1_alpha=self.alpha,
            fc2_input_scale=None,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf,
            w2_alpha=self.alpha,
            num_experts=self.global_num_experts,
            top_k=self.topk,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=output,
            activation_type=activation_to_flashinfer_int(activation),
            quant_mode="mxfp8",
        )
