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
