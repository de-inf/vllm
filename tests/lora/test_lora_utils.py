# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import time

import pytest

from vllm.lora.utils import is_in_target_modules, is_supported_lora_module


class TestIsSupportedLoraModule:
    """Tests for is_supported_lora_module (model-definition check)."""

    def test_suffix_match(self):
        assert is_supported_lora_module(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_no_match(self):
        assert not is_supported_lora_module(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_exact_match(self):
        assert is_supported_lora_module("o_proj", ["o_proj"])

    def test_regex_suffix_matching(self):
        """Regex anchors to end — partial suffix should not match."""
        assert not is_supported_lora_module("model.layers.0.self_attn.o_proj", ["proj"])

    def test_empty_supported_modules(self):
        assert not is_supported_lora_module("model.layers.0.self_attn.o_proj", [])

    def test_multiple_supported_modules(self):
        supported = ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert is_supported_lora_module("model.layers.0.self_attn.v_proj", supported)
        assert not is_supported_lora_module("model.layers.0.mlp.gate_proj", supported)


class TestIsInTargetModules:
    """Tests for is_in_target_modules (deployment-time filter)."""

    def test_none_allows_all(self):
        assert is_in_target_modules("model.layers.0.self_attn.o_proj", None)

    def test_suffix_in_target(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["o_proj", "q_proj"]
        )

    def test_suffix_not_in_target(self):
        assert not is_in_target_modules(
            "model.layers.0.self_attn.o_proj", ["q_proj", "k_proj"]
        )

    def test_empty_target_modules(self):
        assert not is_in_target_modules("model.layers.0.self_attn.o_proj", [])

    def test_exact_name_match(self):
        assert is_in_target_modules("dense1", ["dense1", "dense2"])

    def test_exact_name_no_match(self):
        assert not is_in_target_modules("dense3", ["dense1", "dense2"])

    def test_packed_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_up_proj",
            ["gate_proj", "up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_packed_child_matches_parent_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.mlp.gate_proj",
            ["gate_up_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )

    def test_fused_parent_matches_child_target_modules(self):
        assert is_in_target_modules(
            "model.layers.0.self_attn.fused_qkv_a_proj",
            ["q_a_proj", "kv_a_proj_with_mqa"],
            {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]},
        )


def _build_supported_lora_modules(num_experts: int) -> list[str]:
    """Mirror what ``LRUCacheWorkerLoRAManager._load_adapter`` builds for an
    MoE model: every "experts" entry is expanded into per-expert
    ``experts.{i}.up_proj`` / ``experts.{i}.down_proj`` (see
    vllm/lora/worker_manager.py:117-123).
    """
    base = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
        "embed_tokens",
        "lm_head",
        "in_proj",
        "out_proj",
        "conv1d",
        "fc1_latent_proj",
        "fc2_latent_proj",
        "gate",
        "experts",
    ]
    out = list(base)
    for i in range(num_experts):
        out.append(f"experts.{i}.down_proj")
        out.append(f"experts.{i}.up_proj")
    return out


def _build_module_names(
    num_attn_layers: int,
    num_mamba_layers: int,
    num_moe_layers: int,
    num_experts: int,
) -> list[str]:
    """Synthesize the list of LoRA module names produced by
    ``LoRAModel.from_local_checkpoint`` for a Nemotron-H-style hybrid MoE
    model. Layer types come from the model's ``hybrid_override_pattern``:
    ``M`` = mamba, ``E`` = MoE block, ``*`` = attention.

    Sizes per layer mirror what vLLM's LoRA loader produces:
      - attention (``*``): q_proj, k_proj, v_proj, o_proj
      - mamba    (``M``): in_proj, out_proj, fc1_latent_proj, fc2_latent_proj
      - MoE      (``E``): gate + shared_experts.{up,down}_proj +
                          experts.{i}.{up,down}_proj for i in [0, n_experts)

    For Super (8 *, 40 M, 40 E, 512 experts) the total is ~41 K, matching
    the number that the real worker logs via ``num_lora_modules`` after
    ``from_local_checkpoint``.
    """
    out: list[str] = []
    layer_idx = 0

    for _ in range(num_attn_layers):
        prefix = f"model.layers.{layer_idx}.self_attn"
        for sub in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out.append(f"{prefix}.{sub}")
        layer_idx += 1

    for _ in range(num_mamba_layers):
        prefix = f"model.layers.{layer_idx}.mixer"
        for sub in ("in_proj", "out_proj", "fc1_latent_proj", "fc2_latent_proj"):
            out.append(f"{prefix}.{sub}")
        layer_idx += 1

    for _ in range(num_moe_layers):
        prefix = f"model.layers.{layer_idx}.mixer"
        out.append(f"{prefix}.gate")
        out.append(f"{prefix}.shared_experts.up_proj")
        out.append(f"{prefix}.shared_experts.down_proj")
        for i in range(num_experts):
            out.append(f"{prefix}.experts.{i}.up_proj")
            out.append(f"{prefix}.experts.{i}.down_proj")
        layer_idx += 1

    return out


# Slow scaling tests: skipped in normal CI runs. Enable explicitly when
# investigating LoRA-loading performance, e.g.:
#   RUN_LORA_PERF_TESTS=1 pytest \
#       tests/lora/test_lora_utils.py::TestIsSupportedLoraModulePerf -s -v
_RUN_PERF = os.environ.get("RUN_LORA_PERF_TESTS", "0") == "1"


# (label, num_attn_layers, num_mamba_layers, num_moe_layers, n_routed_experts).
# Counts come from each model's ``hybrid_override_pattern`` in the published
# HF config.json (M=mamba, E=MoE, *=attention) and ``n_routed_experts``:
#   - Nemotron-3-Nano-30B-A3B-BF16   :  6 *, 23 M, 23 E, 128 experts (works)
#   - Nemotron-3-Super-120B-A12B-BF16:  8 *, 40 M, 40 E, 512 experts (hangs)
_MODEL_SHAPES: list[tuple[str, int, int, int, int]] = [
    ("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 6, 23, 23, 128),
    ("NVIDIA-Nemotron-3-Super-120B-A12B-BF16", 8, 40, 40, 512),
]


@pytest.mark.skipif(not _RUN_PERF, reason="set RUN_LORA_PERF_TESTS=1 to enable")
class TestIsSupportedLoraModulePerf:
    """Scaling tests for ``is_supported_lora_module`` on MoE LoRA loads.

    ``_load_adapter`` (vllm/lora/worker_manager.py) iterates this predicate
    once per LoRA module name in the checkpoint (~``num_layers *
    num_experts * 2`` for a per-expert MoE adapter), against the model's
    full ``supported_lora_modules`` list (which is itself ~O(num_experts)).
    The predicate uses an uncompiled regex per element, so the loop is
    O(N_modules * N_supported * regex).

    Calibration note: the wall-clock cost measured by this microbenchmark
    is a **lower bound**. In a real vLLM worker subprocess the same loop
    competes with NCCL/CUDA init, NFS-prefetch flushes, the Python logger
    lock, and NUMA-pinned core sets, so observed times can be much higher.
    Empirically on a Slurm B200 node loading a Nemotron-Super LoRA the
    same 88x512 shape took ~41 s, vs ~1.5 s in this microbenchmark on a
    quiet box -- ~25x. Once the loop crosses 60 s of wall time, vLLM's
    ``shm_broadcast`` worker watchdog fires and the engine deadlocks::

        shm_broadcast.py:681 No available shared memory broadcast block
        found in 60 seconds.

    Tests are intentionally lenient on wall-clock asserts (machine-
    dependent); they print measured per-call cost and end-to-end loop
    time so the scaling cliff is obvious in CI logs.
    """

    @pytest.mark.parametrize(
        "label,n_attn,n_mamba,n_moe,num_experts",
        _MODEL_SHAPES,
        ids=[shape[0] for shape in _MODEL_SHAPES],
    )
    def test_warning_loop_per_model_shape(
        self,
        label: str,
        n_attn: int,
        n_mamba: int,
        n_moe: int,
        num_experts: int,
    ) -> None:
        supported = _build_supported_lora_modules(num_experts)
        module_names = _build_module_names(n_attn, n_mamba, n_moe, num_experts)

        t0 = time.perf_counter()
        n_supported = 0
        for name in module_names:
            if is_supported_lora_module(name, supported):
                n_supported += 1
        dt = time.perf_counter() - t0

        per_call_us = (dt / len(module_names)) * 1e6
        print(
            f"\n[perf] {label:<7}  "
            f"attn={n_attn:>2} mamba={n_mamba:>2} moe={n_moe:>2}  "
            f"experts={num_experts:>4}  "
            f"len(supported)={len(supported):>5}  "
            f"len(modules)={len(module_names):>6}  "
            f"total={dt:7.3f}s  per_call={per_call_us:8.1f}us  "
            f"matched={n_supported}/{len(module_names)}"
        )
        # Sanity: the predicate must accept the realistic inputs we built.
        assert n_supported == len(module_names), (
            "Synthetic module names should all be supported"
        )

    def test_warning_loop_scaling_summary(self) -> None:
        """Run all model shapes back-to-back to expose the cliff.

        Output table is the artifact intended for the GitHub issue.
        """
        results: list[tuple[str, int, int, int, int, int, int, float, float]] = []
        for label, n_attn, n_mamba, n_moe, num_experts in _MODEL_SHAPES:
            supported = _build_supported_lora_modules(num_experts)
            module_names = _build_module_names(n_attn, n_mamba, n_moe, num_experts)
            t0 = time.perf_counter()
            for name in module_names:
                is_supported_lora_module(name, supported)
            dt = time.perf_counter() - t0
            results.append(
                (
                    label,
                    n_attn,
                    n_mamba,
                    n_moe,
                    num_experts,
                    len(supported),
                    len(module_names),
                    dt,
                    dt / len(module_names) * 1e6,
                )
            )

        print(
            "\n[perf summary] is_supported_lora_module on real Nemotron-H "
            "MoE shapes (microbenchmark, lower bound):"
        )
        header = (
            "  model    | attn | mamba | moe | experts | "
            "len(supported) | len(modules) | total(s) | per_call(us)"
        )
        sep = "  " + "-" * (len(header) - 2)
        print(header)
        print(sep)
        for (
            label,
            n_attn,
            n_mamba,
            n_moe,
            n_exp,
            n_sup,
            n_mod,
            dt,
            per,
        ) in results:
            print(
                f"  {label:<7}  | {n_attn:>4} | {n_mamba:>5} | {n_moe:>3} | "
                f"{n_exp:>7} | {n_sup:>14} | {n_mod:>12} | "
                f"{dt:>8.3f} | {per:>11.1f}"
            )

        # Expose the cliff: super must be markedly slower than nano. We pick
        # a conservative >=2x to avoid flakiness on shared-CI machines while
        # still catching a regression that flattens the predicate cost.
        dt_nano = next(r[7] for r in results if "Nano" in r[0])
        dt_super = next(r[7] for r in results if "Super" in r[0])
        ratio = dt_super / max(dt_nano, 1e-6)
        print(f"  scaling super/nano = {ratio:.1f}x")
        assert ratio > 2.0, f"Expected super/nano scaling >=2x, got {ratio:.1f}x"
