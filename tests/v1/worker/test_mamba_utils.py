# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.mamba_utils import postprocess_mamba, preprocess_mamba


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


def test_resumed_req_ids_cleared_from_mamba_state_idx():
    """When a request is force-preempted (e.g. reset_prefix_cache),
    it appears in resumed_req_ids but NOT in preempted_req_ids.
    preprocess_mamba must still clear its mamba_state_idx entry,
    otherwise stale indices can point beyond the new block allocation.
    """
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=[])

    mamba_state_idx = {
        "finished": 1,
        "preempted": 2,
        "resumed": 3,  # only in resumed_req_ids, NOT in preempted
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


def test_preprocess_mamba_boundary_with_real_drafts(monkeypatch):
    """accepted=5 > num_scheduled=2 with real draft tokens is a valid
    boundary transition (scheduler clamped drafts at max_model_len).
    Should normalize accepted to 1, not raise."""
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=["req_0"], num_accepted_tokens_cpu=[5])
    req_state = MagicMock(num_computed_tokens=0, block_ids=([0],))
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 2},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={"req_0": [17]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            scheduler_output,
            MagicMock(),
            cache_config,
            {},
            input_batch,
            {"req_0": req_state},
            {},
            (),
            MagicMock(offset=0),
        )
    assert int(input_batch.num_accepted_tokens_cpu[0]) == 1


def test_preprocess_mamba_normalizes_boundary_non_spec_step(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=["req_0"], num_accepted_tokens_cpu=[3])
    req_state = MagicMock(num_computed_tokens=0, block_ids=([0],))
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},  # No draft tokens this step.
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            scheduler_output,
            MagicMock(),
            cache_config,
            {},
            input_batch,
            {"req_0": req_state},
            {},
            (),
            MagicMock(offset=0),
        )
    assert int(input_batch.num_accepted_tokens_cpu[0]) == 1


def test_preprocess_mamba_normalizes_placeholder_only_spec_step(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=["req_0"], num_accepted_tokens_cpu=[3])
    req_state = MagicMock(num_computed_tokens=0, block_ids=([0],))
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 2},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={"req_0": [-1]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            scheduler_output,
            MagicMock(),
            cache_config,
            {},
            input_batch,
            {"req_0": req_state},
            {},
            (),
            MagicMock(offset=0),
        )
    # Boundary transition: accepted=3 > scheduled=2 with placeholder-only
    # drafts.  Normalized to 1 so downstream state is correct for the
    # current non-spec step; the copy uses the original bias.
    assert int(input_batch.num_accepted_tokens_cpu[0]) == 1


def test_preprocess_mamba_boundary_uses_original_bias_for_copy(monkeypatch):
    """At the max_model_len boundary, the previous step may have accepted
    more tokens than the current step schedules (e.g. accepted=5 from a
    full-spec step, now scheduled=1 with no drafts).

    The Mamba state copy must use the *original* accepted count (5) as
    the accept_token_bias so it reads from the correct state slot, while
    normalizing the downstream value to 1.

    Regression test for the MTP logprob corruption bug: the old code
    clamped the copy bias to num_scheduled, reading from the wrong slot
    and producing corrupt logits/logprobs.
    """
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    block_size = 8
    num_spec_blocks = 2
    spec = MagicMock(block_size=block_size, num_speculative_blocks=num_spec_blocks)
    cache_config = MagicMock(enable_prefix_caching=True)

    stale_accepted = 5
    num_scheduled = 1
    num_computed = 248

    input_batch = MagicMock(
        req_ids=["req_0"],
        num_accepted_tokens_cpu=[stale_accepted],
    )
    num_blocks_needed = -(-(num_computed + num_scheduled) // block_size)
    block_ids_list = list(range(num_blocks_needed + num_spec_blocks + 4))
    req_state = MagicMock(
        num_computed_tokens=num_computed,
        block_ids=(block_ids_list,),
    )
    prev_state_idx = (num_computed - 1) // block_size
    mamba_state_idx: dict[str, int] = {"req_0": prev_state_idx}

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": num_scheduled},
        total_num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    copy_calls: list[tuple] = []

    def spy_collect(
        copy_bufs,
        kv_cache_config,
        funcs,
        group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        fwd_ctx,
    ):
        copy_calls.append((src_block_idx, dest_block_idx, accept_token_bias))

    with (
        patch(
            "vllm.v1.worker.mamba_utils.get_mamba_groups",
            return_value=([0], spec),
        ),
        patch(
            "vllm.v1.worker.mamba_utils.collect_mamba_copy_meta",
            side_effect=spy_collect,
        ),
        patch("vllm.v1.worker.mamba_utils.do_mamba_copy_block"),
    ):
        preprocess_mamba(
            scheduler_output,
            MagicMock(),
            cache_config,
            mamba_state_idx,
            input_batch,
            {"req_0": req_state},
            {},
            (),
            MagicMock(offset=0),
        )

    assert int(input_batch.num_accepted_tokens_cpu[0]) == 1

    curr_num_blocks = -(-(num_computed + num_scheduled) // block_size) + num_spec_blocks
    expected_curr_idx = curr_num_blocks - 1 - num_spec_blocks

    if prev_state_idx != expected_curr_idx:
        assert len(copy_calls) == 1, (
            f"Expected exactly one copy call, got {len(copy_calls)}"
        )
        src, dst, bias = copy_calls[0]
        assert src == prev_state_idx
        assert dst == expected_curr_idx
        assert bias == stale_accepted - 1, (
            f"Copy used bias={bias} but expected {stale_accepted - 1} "
            f"(original accepted={stale_accepted}); "
            f"old buggy code would have used bias="
            f"{num_scheduled - 1}"
        )


@pytest.mark.parametrize(
    "accepted,num_scheduled,spec_tokens,expect_boundary",
    [
        (5, 1, {}, True),
        (5, 2, {"req_0": [-1]}, True),
        (5, 3, {"req_0": [10, 20]}, True),
        (3, 3, {"req_0": [10, 20]}, False),
        (1, 1, {}, False),
        (1, 3, {"req_0": [10, 20]}, False),
    ],
    ids=[
        "no_spec_boundary",
        "placeholder_boundary",
        "real_drafts_boundary",
        "real_drafts_no_boundary",
        "normal_single_token",
        "normal_with_drafts",
    ],
)
def test_preprocess_mamba_boundary_detection(
    monkeypatch, accepted, num_scheduled, spec_tokens, expect_boundary
):
    """Verify boundary transition is detected correctly for various
    combinations of accepted count, scheduled tokens, and spec tokens."""
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(
        req_ids=["req_0"],
        num_accepted_tokens_cpu=[accepted],
    )
    req_state = MagicMock(num_computed_tokens=0, block_ids=([0],))

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": num_scheduled},
        total_num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=spec_tokens,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            scheduler_output,
            MagicMock(),
            cache_config,
            {},
            input_batch,
            {"req_0": req_state},
            {},
            (),
            MagicMock(offset=0),
        )

    if expect_boundary:
        assert int(input_batch.num_accepted_tokens_cpu[0]) == 1, (
            f"Boundary transition should normalize accepted to 1, "
            f"got {int(input_batch.num_accepted_tokens_cpu[0])}"
        )
    else:
        assert int(input_batch.num_accepted_tokens_cpu[0]) == accepted, (
            f"Non-boundary case should keep accepted={accepted}, "
            f"got {int(input_batch.num_accepted_tokens_cpu[0])}"
        )


# ---------------------------------------------------------------------------
# Cross-step simulation tests
# ---------------------------------------------------------------------------


def _run_preprocess_spy(
    monkeypatch,
    *,
    num_computed: int,
    stale_accepted: int,
    num_scheduled: int,
    spec_tokens: dict,
    block_size: int = 8,
    num_spec_blocks: int = 2,
    prev_state_idx: int | None = None,
):
    """Helper: run preprocess_mamba with a spy on collect_mamba_copy_meta.

    Returns (copy_calls, final_accepted) where copy_calls is a list of
    (src_block_idx, dst_block_idx, accept_token_bias) tuples.
    """
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    spec = MagicMock(block_size=block_size, num_speculative_blocks=num_spec_blocks)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(
        req_ids=["req_0"],
        num_accepted_tokens_cpu=[stale_accepted],
    )
    num_blocks_needed = -(-(num_computed + num_scheduled) // block_size)
    block_ids = list(range(num_blocks_needed + num_spec_blocks + 8))
    req_state = MagicMock(num_computed_tokens=num_computed, block_ids=(block_ids,))
    if prev_state_idx is None:
        prev_state_idx = (num_computed - 1) // block_size
    mamba_state_idx: dict[str, int] = {"req_0": prev_state_idx}

    sched = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": num_scheduled},
        total_num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=spec_tokens,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

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

    final_accepted = int(input_batch.num_accepted_tokens_cpu[0])
    return copy_calls, final_accepted


@pytest.mark.parametrize(
    "prev_accepted,cur_scheduled",
    [
        (5, 1),
        (5, 2),
        (5, 3),
        (4, 1),
        (3, 2),
        (2, 1),
    ],
    ids=[
        "5→1",
        "5→2",
        "5→3",
        "4→1",
        "3→2",
        "2→1",
    ],
)
def test_cross_step_bias_always_uses_original_accepted(
    monkeypatch, prev_accepted, cur_scheduled
):
    """For any (prev_accepted, cur_scheduled) where prev > cur, the copy
    bias must equal prev_accepted - 1, never cur_scheduled - 1."""
    copy_calls, final_accepted = _run_preprocess_spy(
        monkeypatch,
        num_computed=248,
        stale_accepted=prev_accepted,
        num_scheduled=cur_scheduled,
        spec_tokens={},
    )
    assert final_accepted == 1
    if copy_calls:
        _, _, bias = copy_calls[0]
        assert bias == prev_accepted - 1, f"bias={bias}, expected {prev_accepted - 1}"


# ---------------------------------------------------------------------------
# Block-boundary-crossing tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_computed",
    [
        8 * 30 - 1,  # one token before block boundary (block_size=8)
        8 * 30,  # exactly on block boundary
        8 * 30 + 1,  # one token after block boundary
        8 * 31 - 1,  # next boundary - 1
    ],
    ids=["boundary-1", "boundary", "boundary+1", "next_boundary-1"],
)
def test_block_boundary_crossing_correct_bias(monkeypatch, num_computed):
    """Vary num_computed around a block boundary and confirm copy bias
    always matches the original stale accepted count."""
    stale_accepted = 5
    copy_calls, final_accepted = _run_preprocess_spy(
        monkeypatch,
        num_computed=num_computed,
        stale_accepted=stale_accepted,
        num_scheduled=1,
        spec_tokens={},
        block_size=8,
        num_spec_blocks=2,
    )
    assert final_accepted == 1
    if copy_calls:
        _, _, bias = copy_calls[0]
        assert bias == stale_accepted - 1


# ---------------------------------------------------------------------------
# Multi-request batch tests (boundary + non-boundary in same batch)
# ---------------------------------------------------------------------------


def test_mixed_batch_boundary_and_normal(monkeypatch):
    """Two requests in the same batch: req_0 is at the boundary
    (accepted=5 > scheduled=1), req_1 is normal (accepted=2 <=
    scheduled=3).  Verify req_0 gets the correct bias and req_1 is
    untouched."""
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    block_size = 8
    num_spec_blocks = 2
    spec = MagicMock(block_size=block_size, num_speculative_blocks=num_spec_blocks)
    cache_config = MagicMock(enable_prefix_caching=True)

    r0_computed, r0_accepted, r0_scheduled = 248, 5, 1
    r1_computed, r1_accepted, r1_scheduled = 100, 2, 3

    input_batch = MagicMock(
        req_ids=["req_0", "req_1"],
        num_accepted_tokens_cpu=[r0_accepted, r1_accepted],
    )
    max_blocks = 40
    block_ids_0 = list(range(max_blocks))
    block_ids_1 = list(range(max_blocks, 2 * max_blocks))
    req_0 = MagicMock(num_computed_tokens=r0_computed, block_ids=(block_ids_0,))
    req_1 = MagicMock(num_computed_tokens=r1_computed, block_ids=(block_ids_1,))

    prev_idx_0 = (r0_computed - 1) // block_size
    prev_idx_1 = (r1_computed - 1) // block_size
    mamba_state_idx = {"req_0": prev_idx_0, "req_1": prev_idx_1}

    sched = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": r0_scheduled, "req_1": r1_scheduled},
        total_num_scheduled_tokens=r0_scheduled + r1_scheduled,
        scheduled_spec_decode_tokens={"req_1": [10, 20]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

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
        if src == prev_idx_0:
            assert bias == r0_accepted - 1, (
                f"req_0 copy bias={bias}, expected {r0_accepted - 1}"
            )


# ---------------------------------------------------------------------------
# Property-based: bias is NEVER clamped to scheduled - 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prev_accepted,cur_scheduled",
    [(a, s) for a in range(2, 7) for s in range(1, a)],
)
def test_bias_never_equals_clamped_value(monkeypatch, prev_accepted, cur_scheduled):
    """For every (prev_accepted > cur_scheduled) pair, the copy bias must
    be prev_accepted - 1, never cur_scheduled - 1 (the old buggy value)."""
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
        if prev_accepted != cur_scheduled:
            assert bias != cur_scheduled - 1, (
                f"bias={bias} equals the old buggy clamped value "
                f"(scheduled-1={cur_scheduled - 1})"
            )


def test_postprocess_mamba_rejects_accepted_gt_scheduled(monkeypatch):
    monkeypatch.setenv("VLLM_MTP_FAIL_FAST", "1")
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    input_batch = MagicMock(req_ids=["req_0"], num_accepted_tokens_cpu=[5])
    req_state = MagicMock(num_computed_tokens=16, block_ids=([0],))
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req_0": 2},
        total_num_scheduled_tokens=2,
        scheduled_spec_decode_tokens={"req_0": [7]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    with (
        patch(
            "vllm.v1.worker.mamba_utils.get_mamba_groups",
            return_value=([0], spec),
        ),
        pytest.raises(
            AssertionError,
            match="Invalid num_accepted_tokens before mamba postprocess",
        ),
    ):
        postprocess_mamba(
            scheduler_output,
            MagicMock(),
            input_batch,
            {"req_0": req_state},
            {"req_0": 0},
            {},
            (),
            MagicMock(offset=0),
        )
