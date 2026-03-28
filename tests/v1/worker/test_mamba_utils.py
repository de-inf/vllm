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


def test_preprocess_mamba_rejects_accepted_gt_scheduled(monkeypatch):
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
        scheduled_spec_decode_tokens={},
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
            match="Invalid num_accepted_tokens before mamba preprocess",
        ),
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
