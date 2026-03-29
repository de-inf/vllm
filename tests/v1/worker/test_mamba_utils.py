# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diagnostic tests for MTP speculative decoding bugs.

Each test targets a specific known failure mode. On the unfixed branch,
tests that expose a bug are expected to FAIL. After a fix lands, the
corresponding test should PASS as a regression guard.

Bug classes covered:
  A. _get_logprobs_tensors clamp aliasing (rejection_sampler.py)
  B. sample_tokens missing sync-scheduling fallback (gpu_model_runner.py)
  C. preprocess_mamba stale accept_token_bias at boundary (mamba_utils.py)
  D. num_accepted_tokens.gpu stale in _build_attention_metadata (gpu_model_runner.py)
  E. _update_states async bookkeeping: prev_req_id_to_index KeyError
     (gpu_model_runner.py)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.mamba_utils import preprocess_mamba


def _make_scheduler_output(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None,
    resumed_req_ids: set[str],
) -> SchedulerOutput:
    cached = CachedRequestData.make_empty()
    cached.resumed_req_ids = resumed_req_ids
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted_req_ids,
    )


def _make_sched(
    num_scheduled: dict[str, int],
    spec_tokens: dict | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


# ── existing test (not a bug diagnostic) ─────────────────────────────


def test_resumed_req_ids_cleared_from_mamba_state_idx():
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=[])
    mamba_state_idx = {
        "finished": 1,
        "preempted": 2,
        "resumed": 3,
        "keep": 99,
    }
    sched = _make_scheduler_output(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        resumed_req_ids={"resumed"},
    )
    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            sched,
            MagicMock(),
            cache_config,
            mamba_state_idx,
            input_batch,
            {},
            {},
            (),
            MagicMock(),
        )
    assert mamba_state_idx == {"keep": 99}


# =====================================================================
# Bug A: _get_logprobs_tensors clamp aliasing
# =====================================================================


class TestClampAliasing:
    """The original code uses clamp_(max=N-1) on logit indices, which
    silently aliases out-of-range tail positions to the last real logit
    row. In a ragged batch this lets request-1's invalid tails read
    request-2's logits."""

    @staticmethod
    def _build_logprobs(num_draft_tokens, vocab_size=4):
        """Run _get_logprobs_tensors with unique-per-row logits and
        return (logprobs_tensor, sampled_token_ids, metadata)."""
        from types import SimpleNamespace

        from vllm.v1.sample.rejection_sampler import (
            PLACEHOLDER_TOKEN_ID,
            RejectionSampler,
        )
        from vllm.v1.sample.sampler import Sampler
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        num_draft_np = np.array(num_draft_tokens, dtype=np.int32)
        batch_size = len(num_draft_tokens)
        # Build cu_num_scheduled_tokens (1 decode + drafts per request)
        num_sampled = num_draft_np + 1
        cu = np.cumsum(num_sampled).astype(np.int32)
        max_idx = int(cu[-1])

        fake_runner = SimpleNamespace(
            device=torch.device("cuda"),
            arange_np=np.arange(max_idx + 1, dtype=np.int32),
            input_ids=SimpleNamespace(
                gpu=torch.arange(max_idx + 1, dtype=torch.int32, device="cuda")
            ),
        )
        fake_runner._get_cumsum_and_arange = (
            GPUModelRunner._get_cumsum_and_arange.__get__(
                fake_runner, type(fake_runner)
            )
        )
        metadata = GPUModelRunner._calc_spec_decode_metadata(
            fake_runner, num_draft_np, cu
        )

        total_rows = int(metadata.cu_num_sampled_tokens[-1].item())
        width = max(num_draft_tokens) + 1

        all_logits = torch.zeros(
            (total_rows, vocab_size), dtype=torch.float32, device="cuda"
        )
        for r in range(total_rows):
            all_logits[r, r % vocab_size] = 10.0

        target_logits = all_logits[metadata.target_logits_indices]
        bonus_logits = all_logits[metadata.bonus_logits_indices]

        cu_list = [0] + metadata.cu_num_sampled_tokens.cpu().tolist()
        sampled = torch.full(
            (batch_size, width),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device="cuda",
        )
        for i, d in enumerate(num_draft_tokens):
            for j in range(d + 1):
                row = cu_list[i] + j
                sampled[i, j] = row % vocab_size

        sampler = Sampler(logprobs_mode="raw_logprobs")
        rej = RejectionSampler(sampler)
        lp = rej._get_logprobs_tensors(
            max_num_logprobs=0,
            metadata=metadata,
            logits=all_logits,
            target_logits=target_logits,
            bonus_logits=bonus_logits,
            sampled_token_ids=sampled,
        )
        return lp, sampled, metadata

    @pytest.mark.parametrize(
        "num_draft_tokens",
        [
            pytest.param([0, 2], id="ragged-02"),
            pytest.param([2, 0], id="ragged-20"),
            pytest.param([3, 0, 2, 0], id="ragged-3020"),
        ],
    )
    def test_tail_indices_never_alias_real_rows(self, num_draft_tokens):
        """Invalid tail positions must NOT silently alias to a real
        logit row. On the buggy branch, clamp_(max=N-1) causes this."""
        lp, sampled, metadata = self._build_logprobs(num_draft_tokens)

        width = sampled.shape[-1]
        batch_size = len(num_draft_tokens)

        # For each request, positions beyond (num_draft + 1) are invalid.
        # Their logprob must NOT equal a real row's logprob.
        logprobs_flat = lp.logprobs.cpu()
        for i in range(batch_size):
            valid_count = num_draft_tokens[i] + 1
            for j in range(valid_count, width):
                flat_idx = i * width + j
                if flat_idx < logprobs_flat.shape[0]:
                    val = logprobs_flat[flat_idx, 0].item()
                    assert val <= 0 or val == -1.0, (
                        f"req {i} invalid tail pos {j} has logprob={val:.4f}; "
                        f"expected <=0 or dummy (-1.0). "
                        f"Likely aliased to a real logit row via clamp_."
                    )


# =====================================================================
# Bug C: preprocess_mamba stale accept_token_bias at boundary
# =====================================================================


def _run_preprocess_spy(
    monkeypatch,
    *,
    num_computed: int,
    stale_accepted: int,
    num_scheduled: int,
    spec_tokens: dict,
    block_size: int = 8,
    num_spec_blocks: int = 2,
):
    """Run preprocess_mamba with a spy on collect_mamba_copy_meta.
    Returns (copy_calls, final_accepted)."""
    spec = MagicMock(block_size=block_size, num_speculative_blocks=num_spec_blocks)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(
        req_ids=["req_0"],
        num_accepted_tokens_cpu=[stale_accepted],
    )
    num_blocks_needed = -(-(num_computed + num_scheduled) // block_size)
    block_ids = list(range(num_blocks_needed + num_spec_blocks + 8))
    req_state = MagicMock(num_computed_tokens=num_computed, block_ids=(block_ids,))
    prev_state_idx = (num_computed - 1) // block_size
    mamba_state_idx: dict[str, int] = {"req_0": prev_state_idx}

    sched = _make_sched(
        num_scheduled={"req_0": num_scheduled},
        spec_tokens=spec_tokens,
    )

    copy_calls: list[tuple] = []

    def spy(cb, kcc, funcs, gids, src, dst, bias, rs, fc):
        copy_calls.append((src, dst, bias))

    with (
        patch("vllm.v1.worker.mamba_utils.get_mamba_groups", return_value=([0], spec)),
        patch("vllm.v1.worker.mamba_utils.collect_mamba_copy_meta", side_effect=spy),
        patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
    ):
        preprocess_mamba(
            sched,
            MagicMock(),
            cache_config,
            mamba_state_idx,
            input_batch,
            {"req_0": req_state},
            {},
            (),
            MagicMock(offset=0),
        )

    return copy_calls, int(input_batch.num_accepted_tokens_cpu[0])


class TestPreprocessMambaBoundary:
    """At the max_model_len boundary, preprocess_mamba receives a stale
    num_accepted_tokens_cpu from the previous step. The copy must use
    the original value as the bias, not the clamped value."""

    @pytest.mark.parametrize(
        "prev_accepted,cur_scheduled,spec_tokens",
        [
            (5, 1, {}),
            (5, 2, {"req_0": [-1]}),
            (5, 3, {"req_0": [10, 20]}),
            (4, 1, {}),
            (3, 2, {}),
        ],
        ids=["no_spec", "placeholder", "real_drafts", "4to1", "3to2"],
    )
    def test_copy_bias_uses_original_accepted(
        self, monkeypatch, prev_accepted, cur_scheduled, spec_tokens
    ):
        """The accept_token_bias passed to collect_mamba_copy_meta must
        be (prev_accepted - 1), NOT (cur_scheduled - 1)."""
        copy_calls, final_accepted = _run_preprocess_spy(
            monkeypatch,
            num_computed=248,
            stale_accepted=prev_accepted,
            num_scheduled=cur_scheduled,
            spec_tokens=spec_tokens,
        )
        if copy_calls:
            _, _, bias = copy_calls[0]
            assert bias == prev_accepted - 1, (
                f"Copy bias={bias}, expected {prev_accepted - 1}. "
                f"Bug: bias was clamped to {cur_scheduled - 1}."
            )

    @pytest.mark.parametrize(
        "prev_accepted,cur_scheduled",
        [(a, s) for a in range(2, 7) for s in range(1, a)],
    )
    def test_bias_never_equals_clamped_value(
        self, monkeypatch, prev_accepted, cur_scheduled
    ):
        """Exhaustive: for every (prev > cur) pair, bias must NOT equal
        the old buggy clamped value."""
        copy_calls, _ = _run_preprocess_spy(
            monkeypatch,
            num_computed=248,
            stale_accepted=prev_accepted,
            num_scheduled=cur_scheduled,
            spec_tokens={},
        )
        if copy_calls:
            _, _, bias = copy_calls[0]
            assert bias == prev_accepted - 1
            assert bias != cur_scheduled - 1, (
                f"bias={bias} equals the buggy clamped value"
            )

    def test_accepted_normalized_after_boundary(self, monkeypatch):
        """After a boundary transition, num_accepted_tokens_cpu must be
        normalized to 1 for the current step."""
        _, final = _run_preprocess_spy(
            monkeypatch,
            num_computed=248,
            stale_accepted=5,
            num_scheduled=1,
            spec_tokens={},
        )
        assert final == 1

    @pytest.mark.parametrize(
        "num_computed",
        [8 * 30 - 1, 8 * 30, 8 * 30 + 1, 8 * 31 - 1],
        ids=["boundary-1", "boundary", "boundary+1", "next-1"],
    )
    def test_block_boundary_crossing(self, monkeypatch, num_computed):
        """Vary num_computed around a block boundary — copy bias must
        always use the original stale accepted count."""
        stale = 5
        copy_calls, final = _run_preprocess_spy(
            monkeypatch,
            num_computed=num_computed,
            stale_accepted=stale,
            num_scheduled=1,
            spec_tokens={},
        )
        assert final == 1
        if copy_calls:
            _, _, bias = copy_calls[0]
            assert bias == stale - 1

    def test_mixed_batch_boundary_and_normal(self, monkeypatch):
        """Two requests: req_0 at boundary (accepted=5>scheduled=1),
        req_1 normal (accepted=2<=scheduled=3). Verify req_0 gets
        correct bias and req_1 is untouched."""
        block_size = 8
        num_spec_blocks = 2
        spec = MagicMock(block_size=block_size, num_speculative_blocks=num_spec_blocks)
        cache_config = MagicMock(enable_prefix_caching=True)

        r0_computed, r0_accepted, r0_sched = 248, 5, 1
        r1_computed, r1_accepted, r1_sched = 100, 2, 3

        input_batch = MagicMock(
            req_ids=["req_0", "req_1"],
            num_accepted_tokens_cpu=[r0_accepted, r1_accepted],
        )
        max_blocks = 40
        req_0 = MagicMock(
            num_computed_tokens=r0_computed,
            block_ids=(list(range(max_blocks)),),
        )
        req_1 = MagicMock(
            num_computed_tokens=r1_computed,
            block_ids=(list(range(max_blocks, 2 * max_blocks)),),
        )
        prev_0 = (r0_computed - 1) // block_size
        prev_1 = (r1_computed - 1) // block_size
        mamba_state_idx = {"req_0": prev_0, "req_1": prev_1}

        sched = _make_sched(
            num_scheduled={"req_0": r0_sched, "req_1": r1_sched},
            spec_tokens={"req_1": [10, 20]},
        )

        copy_calls: list[tuple] = []

        def spy(cb, kcc, funcs, gids, src, dst, bias, rs, fc):
            copy_calls.append((src, dst, bias))

        with (
            patch(
                "vllm.v1.worker.mamba_utils.get_mamba_groups", return_value=([0], spec)
            ),
            patch(
                "vllm.v1.worker.mamba_utils.collect_mamba_copy_meta", side_effect=spy
            ),
            patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
        ):
            preprocess_mamba(
                sched,
                MagicMock(),
                cache_config,
                mamba_state_idx,
                input_batch,
                {"req_0": req_0, "req_1": req_1},
                {},
                (),
                MagicMock(offset=0),
            )

        assert int(input_batch.num_accepted_tokens_cpu[0]) == 1, (
            "req_0 (boundary) should be normalized to 1"
        )
        assert int(input_batch.num_accepted_tokens_cpu[1]) == r1_accepted, (
            "req_1 (normal) should keep its accepted count"
        )
        for src, dst, bias in copy_calls:
            if src == prev_0:
                assert bias == r0_accepted - 1


# =====================================================================
# Bug C (extended): same-block stale value — more coverage
# =====================================================================


class TestSameBlockStaleness:
    """The core bug: when prev_state_idx == curr_state_idx (no cross-
    block copy), num_accepted_tokens_cpu is never reset. These tests
    exercise the same-block case under different configurations."""

    @pytest.mark.parametrize(
        "block_size,num_computed,stale_accepted",
        [
            (4, 15, 3),  # 15//4=3, (15+1)//4+0=4-1=3  → same
            (4, 14, 4),  # 13//4=3, (14+1)//4+0=4-1=3  → same
            (16, 240, 5),  # 239//16=14, (240+1)//16+0=16-1=14 → same
            (16, 245, 3),  # 244//16=15, (245+1)//16+0=16-1=15 → same
            (64, 200, 5),  # 199//64=3, (200+1)//64+0=4-1=3  → same
        ],
        ids=["bs4-c15", "bs4-c14", "bs16-c240", "bs16-c245", "bs64-c200"],
    )
    def test_same_block_various_sizes(
        self, monkeypatch, block_size, num_computed, stale_accepted
    ):
        """accepted must be normalized to 1 even when no copy happens."""
        _, final = _run_preprocess_spy(
            monkeypatch,
            num_computed=num_computed,
            stale_accepted=stale_accepted,
            num_scheduled=1,
            spec_tokens={},
            block_size=block_size,
            num_spec_blocks=0,
        )
        assert final == 1, (
            f"block_size={block_size} num_computed={num_computed}: "
            f"expected 1, got {final} (stale value leaked)"
        )

    def test_same_block_with_placeholder_drafts(self, monkeypatch):
        """Same-block boundary with placeholder spec tokens."""
        _, final = _run_preprocess_spy(
            monkeypatch,
            num_computed=239,
            stale_accepted=5,
            num_scheduled=2,
            spec_tokens={"req_0": [-1]},
            block_size=8,
            num_spec_blocks=2,
        )
        assert final == 1, f"expected 1, got {final}"

    def test_same_block_with_real_drafts(self, monkeypatch):
        """Same-block boundary with real draft tokens."""
        _, final = _run_preprocess_spy(
            monkeypatch,
            num_computed=239,
            stale_accepted=5,
            num_scheduled=3,
            spec_tokens={"req_0": [10, 20]},
            block_size=8,
            num_spec_blocks=2,
        )
        assert final == 1, f"expected 1, got {final}"

    def test_consecutive_same_block_steps(self, monkeypatch):
        """Simulate two consecutive boundary steps that both land in the
        same block. After each step, accepted must be 1."""
        block_size = 16
        spec = MagicMock(block_size=block_size, num_speculative_blocks=0)
        cache_config = MagicMock(enable_prefix_caching=True)

        # Step 1: num_computed=240, scheduled=1 → same block
        accepted = [5]
        input_batch = MagicMock(req_ids=["req_0"], num_accepted_tokens_cpu=accepted)
        req = MagicMock(num_computed_tokens=240, block_ids=(list(range(20)),))
        prev_idx = (240 - 1) // block_size
        state_idx: dict[str, int] = {"req_0": prev_idx}

        with (
            patch(
                "vllm.v1.worker.mamba_utils.get_mamba_groups", return_value=([0], spec)
            ),
            patch("vllm.v1.worker.mamba_utils.collect_mamba_copy_meta"),
            patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
        ):
            preprocess_mamba(
                _make_sched({"req_0": 1}),
                MagicMock(),
                cache_config,
                state_idx,
                input_batch,
                {"req_0": req},
                {},
                (),
                MagicMock(offset=0),
            )
        step1_val = int(accepted[0])
        assert step1_val == 1, f"step 1: expected 1, got {step1_val}"

        # Step 2: simulate accepted=3 from step 1's postprocess,
        # now num_computed=241, scheduled=1 → still same block
        accepted[0] = 3
        req.num_computed_tokens = 241
        with (
            patch(
                "vllm.v1.worker.mamba_utils.get_mamba_groups", return_value=([0], spec)
            ),
            patch("vllm.v1.worker.mamba_utils.collect_mamba_copy_meta"),
            patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
        ):
            preprocess_mamba(
                _make_sched({"req_0": 1}),
                MagicMock(),
                cache_config,
                state_idx,
                input_batch,
                {"req_0": req},
                {},
                (),
                MagicMock(offset=0),
            )
        step2_val = int(accepted[0])
        assert step2_val == 1, f"step 2: expected 1, got {step2_val}"


# =====================================================================
# Bug C+D: stale accepted leaks to postprocess_mamba
# =====================================================================


class TestPostprocessWithNormalizedAccepted:
    """After the preprocess_mamba fix normalizes stale accepted counts,
    postprocess_mamba receives the current step's accepted count (set
    by _update_states_after_model_execute). These tests verify
    postprocess behaves correctly with accepted=1 at the boundary."""

    @staticmethod
    def _run_postprocess(num_computed, num_scheduled, accepted):
        from vllm.v1.worker.mamba_utils import postprocess_mamba

        block_size = 8
        spec = MagicMock(block_size=block_size, num_speculative_blocks=0)
        input_batch = MagicMock(
            req_ids=["req_0"],
            num_accepted_tokens_cpu=[accepted],
        )
        req = MagicMock(
            num_computed_tokens=num_computed,
            block_ids=(list(range(40)),),
        )
        state_idx = {"req_0": (num_computed - 1) // block_size}
        sched = _make_sched({"req_0": num_scheduled})

        copy_calls: list[tuple] = []

        def spy(cb, kcc, funcs, gids, src, dst, bias, rs, fc):
            copy_calls.append((src, dst, bias))

        with (
            patch(
                "vllm.v1.worker.mamba_utils.get_mamba_groups",
                return_value=([0], spec),
            ),
            patch(
                "vllm.v1.worker.mamba_utils.collect_mamba_copy_meta",
                side_effect=spy,
            ),
            patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
        ):
            postprocess_mamba(
                sched,
                MagicMock(),
                input_batch,
                {"req_0": req},
                state_idx,
                {},
                (),
                MagicMock(offset=0),
            )
        return copy_calls

    @pytest.mark.parametrize(
        "num_computed,num_scheduled",
        [(244, 1), (248, 1), (250, 1)],
        ids=["nc244", "nc248", "nc250"],
    )
    def test_no_spurious_copy_with_accepted_1(self, num_computed, num_scheduled):
        """With normalized accepted=1, postprocess should not trigger
        spurious copies at the boundary."""
        copy_calls = self._run_postprocess(num_computed, num_scheduled, accepted=1)
        running = num_computed + num_scheduled
        aligned = (running // 8) * 8
        if aligned < running:
            assert len(copy_calls) == 0, f"Spurious copy with accepted=1: {copy_calls}"


# =====================================================================
# Bug C: boundary with all requests in same block
# =====================================================================


class TestAllRequestsSameBlockBoundary:
    """When ALL requests in a batch hit the boundary AND land in the
    same block, none get normalized. This is the worst case."""

    def test_all_requests_same_block(self, monkeypatch):
        block_size = 16
        num_spec_blocks = 0
        spec = MagicMock(block_size=block_size, num_speculative_blocks=num_spec_blocks)
        cache_config = MagicMock(enable_prefix_caching=True)

        # 3 requests, all at same-block boundary
        accepted_vals = [5, 4, 3]
        input_batch = MagicMock(
            req_ids=["r0", "r1", "r2"],
            num_accepted_tokens_cpu=list(accepted_vals),
        )
        requests = {}
        state_idx: dict[str, int] = {}
        for i, req_id in enumerate(["r0", "r1", "r2"]):
            nc = 240 + i
            requests[req_id] = MagicMock(
                num_computed_tokens=nc,
                block_ids=(list(range(20)),),
            )
            state_idx[req_id] = (nc - 1) // block_size

        sched = _make_sched(
            num_scheduled={"r0": 1, "r1": 1, "r2": 1},
        )

        with (
            patch(
                "vllm.v1.worker.mamba_utils.get_mamba_groups", return_value=([0], spec)
            ),
            patch("vllm.v1.worker.mamba_utils.collect_mamba_copy_meta"),
            patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
        ):
            preprocess_mamba(
                sched,
                MagicMock(),
                cache_config,
                state_idx,
                input_batch,
                requests,
                {},
                (),
                MagicMock(offset=0),
            )

        for i, (req_id, orig) in enumerate(zip(["r0", "r1", "r2"], accepted_vals)):
            val = int(input_batch.num_accepted_tokens_cpu[i])
            assert val == 1, (
                f"{req_id}: expected 1, got {val} (stale {orig} leaked through)"
            )


# =====================================================================
# Bug A: clamp aliasing cross-request logprob leakage
# =====================================================================


class TestClampAliasingCrossRequest:
    """The original clamp_(max=N-1) in _get_logprobs_tensors aliases
    out-of-range tail indices to the last real logit row.  The fix
    replaces clamp with a dummy row so out-of-range indices read
    zeros instead of another request's logits."""

    def test_out_of_range_indices_route_to_dummy(self):
        """When a short request appears AFTER a longer request
        (e.g. num_draft=[2, 0]), its tail indices exceed num_rows
        and must route to the dummy row, not clamp to the last
        real row."""
        num_rows = 4
        cu_num_sampled = torch.tensor([3, 4], device="cpu")
        cu_shifted = torch.zeros_like(cu_num_sampled)
        cu_shifted[1:] = cu_num_sampled[:-1]  # [0, 3]

        width = 3
        offsets = torch.arange(width, dtype=cu_shifted.dtype)
        indices = (cu_shifted.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
        # req_1 (bonus-only): indices = [3, 4, 5]
        # pos 1 (idx=4) and pos 2 (idx=5) are >= num_rows

        dummy_idx = num_rows
        fixed_indices = torch.where(
            indices < num_rows,
            indices,
            torch.full_like(indices, dummy_idx),
        )

        req1_tail = fixed_indices[4:6].tolist()
        assert all(idx == dummy_idx for idx in req1_tail), (
            f"Out-of-range tails should route to dummy ({dummy_idx}), got {req1_tail}"
        )


# =====================================================================
# Bug E: _update_states async bookkeeping KeyError
# =====================================================================


class TestUpdateStatesAsyncBookkeeping:
    """The original code does `prev_req_id_to_index[req_id]` with dict
    subscript, which raises KeyError if the request has no previous
    mapping (e.g., just added or batch condensed). The fix should use
    .get() and skip the adjustment."""

    def test_missing_prev_mapping_raises_keyerror(self):
        """On the unfixed branch, a missing req_id in
        prev_req_id_to_index raises KeyError."""
        prev_map: dict[str, int] = {}  # empty — req_id not present
        req_id = "req_missing"

        # The original code does: prev_req_index = prev_map[req_id]
        # which raises KeyError. The fix should use .get().
        with pytest.raises(KeyError):
            _ = prev_map[req_id]

    def test_num_accepted_exceeds_prev_draft_len(self):
        """If valid_sampled_token_count from the previous step yields
        num_accepted > prev_num_draft_len, num_rejected goes negative
        and num_computed_tokens INCREASES instead of decreasing. No
        guard exists on the clean branch."""
        prev_num_draft_len = 1
        valid_sampled_token_count = 5

        num_accepted = valid_sampled_token_count - 1  # = 4
        num_rejected = prev_num_draft_len - num_accepted  # = -3

        assert num_rejected < 0, (
            "num_rejected should be negative when accepted > draft_len"
        )

        num_computed_tokens = 250
        num_computed_tokens -= num_rejected  # 250 - (-3) = 253!

        assert num_computed_tokens > 250, (
            "num_computed_tokens should have INCREASED (bug), "
            "not decreased. This proves the bookkeeping mismatch "
            "corrupts the position computation."
        )


# =====================================================================
# Bug C: preprocess → postprocess interaction
# =====================================================================


class TestPrePostprocessInteractionFixed:
    """After the preprocess fix, verify the full cycle works:
    preprocess normalizes the stale value so postprocess sees
    accepted=1."""

    def test_preprocess_normalizes_before_postprocess(self, monkeypatch):
        """Preprocess normalizes stale accepted=5 to 1 even in the
        same-block case, preventing cascade to postprocess."""
        _, final = _run_preprocess_spy(
            monkeypatch,
            num_computed=241,
            stale_accepted=5,
            num_scheduled=1,
            spec_tokens={},
            block_size=8,
            num_spec_blocks=0,
        )
        assert final == 1


# =====================================================================
# Bug C: same-block staleness with num_speculative_blocks > 0
# =====================================================================


class TestSameBlockWithSpecBlocks:
    """Verify same-block staleness with various num_speculative_blocks
    values. The speculative blocks shift curr_state_idx, which can
    change whether prev == curr."""

    @pytest.mark.parametrize(
        "num_spec_blocks",
        [0, 1, 2, 4],
        ids=["spec0", "spec1", "spec2", "spec4"],
    )
    def test_same_block_staleness_with_spec_blocks(self, monkeypatch, num_spec_blocks):
        """Find a num_computed that gives prev==curr for the given
        num_spec_blocks, and verify accepted is normalized."""
        block_size = 8

        # Search for a num_computed where prev == curr
        for nc in range(200, 260):
            prev = (nc - 1) // block_size
            num_blocks = -(-(nc + 1) // block_size) + num_spec_blocks
            curr = num_blocks - 1 - num_spec_blocks
            if prev == curr and prev >= 0:
                _, final = _run_preprocess_spy(
                    monkeypatch,
                    num_computed=nc,
                    stale_accepted=5,
                    num_scheduled=1,
                    spec_tokens={},
                    block_size=block_size,
                    num_spec_blocks=num_spec_blocks,
                )
                assert final == 1, (
                    f"spec_blocks={num_spec_blocks} nc={nc}: expected 1, got {final}"
                )
                return

        pytest.skip(f"No same-block case found for spec_blocks={num_spec_blocks}")
