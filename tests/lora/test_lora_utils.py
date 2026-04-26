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


def _build_module_names(num_layers: int, num_experts: int) -> list[str]:
    """Synthesize the list of LoRA module names that ``from_local_checkpoint``
    yields for a typical MoE-with-attn model. The sizes match what
    ``_load_adapter`` iterates over in its post-checkpoint warning loop.
    """
    layer_lang_leaves = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mixer.in_proj",
        "mixer.out_proj",
        "mixer.fc1_latent_proj",
        "mixer.fc2_latent_proj",
        "mixer.gate",
    ]
    out: list[str] = []
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        for leaf in layer_lang_leaves:
            out.append(f"{prefix}.{leaf}")
        for i in range(num_experts):
            out.append(f"{prefix}.mixer.experts.{i}.up_proj")
            out.append(f"{prefix}.mixer.experts.{i}.down_proj")
    return out


# Slow scaling tests: skipped in normal CI runs. Enable explicitly when
# investigating LoRA-loading performance, e.g.:
#   RUN_LORA_PERF_TESTS=1 pytest \
#       tests/lora/test_lora_utils.py::TestIsSupportedLoraModulePerf -s -v
_RUN_PERF = os.environ.get("RUN_LORA_PERF_TESTS", "0") == "1"


@pytest.mark.skipif(not _RUN_PERF, reason="set RUN_LORA_PERF_TESTS=1 to enable")
class TestIsSupportedLoraModulePerf:
    """Scaling tests for ``is_supported_lora_module`` on MoE LoRA loads.

    ``_load_adapter`` (vllm/lora/worker_manager.py) iterates this predicate
    once per LoRA module name in the checkpoint (~``num_layers *
    num_experts * 2`` for a per-expert MoE adapter), against the model's
    full ``supported_lora_modules`` list (which is itself ~O(num_experts)).
    The predicate uses an uncompiled regex per element, so the loop is
    O(N_modules * N_supported * regex). On a 88-layer / 512-expert model
    this currently takes 30-40 s of pure-Python time, single-threaded,
    GIL-held -- long enough to trip vLLM's ``shm_broadcast`` worker
    watchdog (60 s) and cause the engine to deadlock with::

        shm_broadcast.py:681 No available shared memory broadcast block
        found in 60 seconds.

    These tests are intentionally not asserting wall-clock thresholds
    (machine-dependent); instead, they print measured per-call cost and
    end-to-end loop time so the scaling cliff is obvious in CI logs.
    """

    @pytest.mark.parametrize("num_experts", [32, 128, 512])
    def test_warning_loop_scaling(self, num_experts: int) -> None:
        # 88 layers matches Nemotron-Super-120B's transformer depth.
        num_layers = 88
        supported = _build_supported_lora_modules(num_experts)
        module_names = _build_module_names(num_layers, num_experts)

        t0 = time.perf_counter()
        n_supported = 0
        for name in module_names:
            if is_supported_lora_module(name, supported):
                n_supported += 1
        dt = time.perf_counter() - t0

        per_call_us = (dt / len(module_names)) * 1e6
        print(
            f"\n[perf] num_experts={num_experts:>3}  "
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
        """Run all expert sizes back-to-back to expose the linear cliff."""
        num_layers = 88
        results: list[tuple[int, float, float]] = []
        for num_experts in (32, 128, 512):
            supported = _build_supported_lora_modules(num_experts)
            module_names = _build_module_names(num_layers, num_experts)
            t0 = time.perf_counter()
            for name in module_names:
                is_supported_lora_module(name, supported)
            dt = time.perf_counter() - t0
            results.append((num_experts, dt, dt / len(module_names) * 1e6))

        print("\n[perf summary] is_supported_lora_module on Nemotron-shaped MoE:")
        print("  num_experts |  total_dt(s) |  per_call(us)")
        print("  ------------+--------------+---------------")
        for n, dt, per in results:
            print(f"  {n:>10}   |  {dt:9.3f}   |  {per:9.1f}")

        # Expose the scaling: 512-expert run should be markedly slower than
        # the 32-expert run. We pick a conservative >=4x to avoid flakiness.
        dt_32 = results[0][1]
        dt_512 = results[-1][1]
        ratio = dt_512 / max(dt_32, 1e-6)
        print(f"  scaling 512/32 = {ratio:.1f}x")
        assert ratio > 4.0, (
            f"Expected >=4x slowdown from 32 to 512 experts, got {ratio:.1f}x"
        )
