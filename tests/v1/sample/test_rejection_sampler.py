# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.v1.sample.utils import create_allowed_token_ids
from vllm.platforms import current_platform
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    sample_recovered_tokens,
)
from vllm.v1.sample.sampler import Sampler, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = current_platform.device_type


@pytest.fixture
def rejection_sampler():
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    return RejectionSampler(mock_sampler)


def mock_sampler_output(
    rejection_sampler: RejectionSampler, bonus_token_ids: torch.Tensor
):
    rejection_sampler.sampler.return_value = SamplerOutput(
        sampled_token_ids=bonus_token_ids, logprobs_tensors=None
    )


def create_spec_decode_metadata(
    spec_tokens: list[list[int]], logits: torch.Tensor
) -> SpecDecodeMetadata:
    metadata = SpecDecodeMetadata.make_dummy(spec_tokens, device=logits.device)
    metadata.target_logits_indices = torch.arange(logits.shape[0])
    # Output bonus token ids are mocked, so the bonus logit indices should
    # be empty.
    metadata.bonus_logits_indices = torch.empty(0, dtype=torch.int32)
    return metadata


def create_logits_tensor(
    output_token_ids: list[list[int]],
    vocab_size: int = 100,
    token_idx_to_override: int | None = None,
) -> torch.Tensor:
    """Helper function to create logits tensor that
    will produce desired token ids on argmax"""
    token_ids = [tokens[:-1] for tokens in output_token_ids]
    num_total_tokens = sum(len(tokens) for tokens in token_ids)
    logits = torch.full((num_total_tokens, vocab_size), -100.0, device=DEVICE)
    start_loc = 0
    for tokens in token_ids:
        for j, token_id in enumerate(tokens):
            logits[start_loc + j, token_id] = 100.0
        start_loc += len(tokens)
    if token_idx_to_override:
        logits[:, token_idx_to_override] = 99.0
    return logits


def create_sampling_metadata(
    all_greedy: bool,
    output_token_ids: list[list[int]] | None = None,
    prompt_token_ids: torch.Tensor | None = None,
    spec_token_ids: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    generators: dict[int, Any] | None = None,
    frequency_penalties: list[float] | None = None,
    presence_penalties: list[float] | None = None,
    repetition_penalties: list[float] | None = None,
    bad_words_token_ids: dict[int, list[list[int]]] | None = None,
    allowed_token_ids_mask: torch.Tensor | None = None,
) -> SamplingMetadata:
    """Create a v1 sampling metadata object with all_greedy set
    to the given value. Either all greedy or all random sampling
    is used.
    """
    generators = generators or {}
    if all_greedy:
        temperature = None
    else:
        assert temperature is not None

    if any([frequency_penalties, presence_penalties, repetition_penalties]):
        no_penalties = False

        assert output_token_ids
        assert len(output_token_ids) > 0

        frequency_penalties = torch.tensor(frequency_penalties, device=DEVICE)
        presence_penalties = torch.tensor(presence_penalties, device=DEVICE)
        repetition_penalties = torch.tensor(repetition_penalties, device=DEVICE)
    else:
        no_penalties = True
        frequency_penalties = torch.tensor([])
        presence_penalties = torch.tensor([])
        repetition_penalties = torch.tensor([])

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=[] if output_token_ids is None else output_token_ids,
        spec_token_ids=[] if spec_token_ids is None else spec_token_ids,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids={} if bad_words_token_ids is None else bad_words_token_ids,
        logitsprocs=LogitsProcessors(),
    )


########################### Tests for Greedy Sampling ###################
def test_perfect_match(rejection_sampler):
    """Test when output tokens perfectly match speculated tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 2, 3, 4]]  # 4 is the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2, 3, 4]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_early_mismatch(rejection_sampler):
    """Test when there's an early mismatch in tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 5, 3, 4]]  # Mismatch at position 1

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_sequences(rejection_sampler):
    """Test handling multiple sequences of speculated tokens"""
    spec_tokens = [[1, 2], [3]]
    output_tokens = [[1, 2, 5], [3, 4]]  # Two sequences with bonus tokens 5 and 4

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[0][-1], output_tokens[1][-1]], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 2, 5], [3, 4, PLACEHOLDER_TOKEN_ID]], dtype=torch.int, device=logits.device
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_single_token_sequence(rejection_sampler):
    """Test handling sequences with single token"""
    spec_tokens = [[1]]
    output_tokens = [[1, 2]]  # Single token with bonus token 2

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_empty_sequence(rejection_sampler):
    """Test handling empty sequence of speculated tokens"""
    spec_tokens: list[list[int]] = [[]]
    output_tokens = [[5]]  # Just the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[5]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_mismatches(rejection_sampler):
    """Test handling multiple sequences with mismatches"""
    spec_tokens = [[1, 2, 3], [4, 5, 6]]
    output_tokens = [[1, 2, 7, 6], [4, 8, 6, 9]]  # Mismatches in both sequences

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[0][-1], output_tokens[1][-1]], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [
            [1, 2, 7, PLACEHOLDER_TOKEN_ID],
            [4, 8, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2]], [[1, 2, 3]], [[1, 2, 3]]),  # Perfect match with bonus
        ([[1]], [[2, 3]], [[2, PLACEHOLDER_TOKEN_ID]]),  # First mismatch
        (
            [[1, 2], [3, 4]],
            [[1, 5, 6], [3, 4, 7]],
            [[1, 5, PLACEHOLDER_TOKEN_ID], [3, 4, 7]],
        ),  # Mixed matches
    ],
)
def test_parametrized_cases(rejection_sampler, spec_tokens, output_tokens, expected):
    """Parametrized test for various matching scenarios"""
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [tokens[-1] for tokens in output_tokens], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected_tensor = torch.tensor(expected, dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected_tensor)


########################### Tests for Random Sampling ###################
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("frac_seeded", [0.0, 0.5])
@pytest.mark.parametrize("n_rep", [20])
def test_deterministic_when_seeded(
    rejection_sampler,
    k: int,
    vocab_size: int,
    batch_size: int,
    frac_seeded: float,
    n_rep: int,
):
    num_tokens = batch_size * k
    draft_probs = torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_probs = F.softmax(draft_probs, dim=-1)
    target_logits = torch.rand_like(draft_probs)
    bonus_token_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, 1), dtype=torch.int64, device=DEVICE
    )
    draft_token_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, k), dtype=torch.int64, device=DEVICE
    )

    seeded_mask = torch.rand(batch_size, dtype=torch.float32) <= frac_seeded

    results = []
    for _ in range(n_rep):
        seeded_seqs = {
            i: torch.Generator(device=DEVICE).manual_seed(i)
            for i in range(batch_size)
            if seeded_mask[i]
        }

        temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
        sampling_metadata = create_sampling_metadata(
            all_greedy=False, temperature=temperature, generators=seeded_seqs
        )
        spec_decode_metadata = create_spec_decode_metadata(
            draft_token_ids.tolist(), target_logits
        )

        mock_sampler_output(rejection_sampler, bonus_token_ids)
        rep_result = rejection_sampler(
            spec_decode_metadata,
            draft_probs=None,
            logits=target_logits,
            sampling_metadata=sampling_metadata,
        )

        results.append(rep_result.sampled_token_ids)

    for i in range(batch_size):
        if seeded_mask[i]:
            for j in range(1, n_rep):
                assert torch.equal(results[j][i], results[0][i])


def test_rejection_sampling_approximates_target_distribution():
    """Verify rejection sampling approximates target distribution,
    despite sampling from a potentially distinct draft distribution.

    This is done by first creating a random target probability
    distribution and a random draft probability distribution. We then
    sample token ids from the rejection sampler using these draft
    and target distributions. The samples are used to estimate
    the output probability distribution, which we expect to approximate
    the target distribution.

    A basic distance metric is used to determine similarity between
    distributions.

    We expect that as we increase the number of samples,
    the distance between the observed distribution and the target
    distribution decreases. To measure this, we compare the distance
    of the observed distribution against both the target distribution
    and a uniform random distribution. We expect the distance between
    the observed distribution and the target distribution to improve
    much more than the distance improvement between the observed
    distribution and the random distribution.
    """
    torch.set_default_device(DEVICE)
    vocab_size = 10
    k = 2
    num_reference_probs = 100

    # Prepare draft, target, and reference probability distributions
    draft_probs = F.softmax(torch.rand(vocab_size, dtype=torch.float32), dim=-1)
    target_logits = torch.rand(vocab_size, dtype=torch.float32)
    target_probs = F.softmax(target_logits, dim=-1)
    reference_probs = F.softmax(
        torch.rand(num_reference_probs, vocab_size, dtype=torch.float32),
        dim=-1,
    )

    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    distance_wrt_reference: list[float] = []
    distance_wrt_target: list[float] = []

    for num_samples in sample_sizes:
        # Sample using rejection sampling.
        rej_sample_probs = estimate_rejection_sampling_pdf(
            draft_probs, target_logits, k, vocab_size, num_samples
        )
        rej_sample_probs = rej_sample_probs.to(DEVICE)

        # Average distance from reference probs.
        reference_vs_rejsample_dist = (
            torch.dist(reference_probs, rej_sample_probs).item()
            / reference_probs.shape[0]
        )
        target_vs_rejsample_dist = torch.dist(target_probs, rej_sample_probs).item()

        distance_wrt_reference.append(reference_vs_rejsample_dist)
        distance_wrt_target.append(target_vs_rejsample_dist)

        relative_change_in_distance_wrt_target = get_ratio_first_to_last(
            distance_wrt_target
        )
        relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
            distance_wrt_reference
        )

        print(
            f"{num_samples=} {target_vs_rejsample_dist=:.05f} "
            f"{reference_vs_rejsample_dist=:.05f}"
        )
        print(
            f"{num_samples=} {relative_change_in_distance_wrt_target=:.02f} "
            f"{relative_change_in_distance_wrt_reference=:.02f}"
        )

    relative_change_in_distance_wrt_target = get_ratio_first_to_last(
        distance_wrt_target
    )
    relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
        distance_wrt_reference
    )

    expected_improvement_multiplier = 20
    assert (
        relative_change_in_distance_wrt_target
        > relative_change_in_distance_wrt_reference * expected_improvement_multiplier
    )


def get_ratio_first_to_last(elements: list[float]) -> float:
    return elements[0] / elements[-1]


def estimate_rejection_sampling_pdf(
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    k: int,
    vocab_size: int,
    num_samples: int,
) -> torch.Tensor:
    """Estimate the probability distribution of the output tokens
    using rejection sampling.

    Args:
        draft_probs: Draft probability distribution.
        target_logits: Target logits.
        num_samples: Number of samples to draw.

    Returns:
        Estimated probability distribution of the output tokens.
    """
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    rejection_sampler = RejectionSampler(mock_sampler)
    num_tokens = num_samples * k
    # Repeat draft probs num_samples * k times.
    draft_probs = draft_probs.reshape(1, 1, vocab_size).repeat(num_samples, k, 1)

    # Repeat target probs num_tokens times.
    target_logits = target_logits.reshape(1, vocab_size).repeat(num_tokens, 1)

    # Randomly sample draft token ids from draft probs.
    draft_token_ids = torch.multinomial(
        draft_probs[:, 0, :], num_samples=k, replacement=True
    ).reshape(num_samples, k)
    draft_probs = draft_probs.view(num_tokens, vocab_size)

    # Bonus tokens not used but required.
    bonus_token_ids = torch.zeros((1, 1), dtype=torch.int64, device=DEVICE).repeat(
        num_samples, 1
    )

    temperature = torch.ones(num_samples, dtype=torch.float32, device=DEVICE)
    sampling_metadata = create_sampling_metadata(
        all_greedy=False, temperature=temperature
    )
    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.tolist(), target_logits
    )

    mock_sampler_output(rejection_sampler, bonus_token_ids)
    sampler_output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=draft_probs,
        logits=target_logits,
        sampling_metadata=sampling_metadata,
    )
    output_token_ids = sampler_output.sampled_token_ids[:, :-1].flatten()

    hist = torch.histogram(
        output_token_ids.to(dtype=torch.float, device="cpu"),
        bins=vocab_size,
        range=(0, vocab_size),
        density=True,
    )

    return hist.hist


def native_sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    draft_token_ids: torch.Tensor,  # [num_tokens]
    draft_probs: torch.Tensor | None,  # [num_tokens, vocab_size]
    target_probs: torch.Tensor,  # [num_tokens, vocab_size]
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]

    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()

    states = {
        i: generator.get_state()
        for i, generator in sampling_metadata.generators.items()
    }
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

        # In order to generate the same exponential later, reset the CUDA RNG
        # state because RNG state advances after each call.
        generator.set_state(states[i])

    inv_q = q.reciprocal()

    out = torch.empty_like(draft_token_ids)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
        end_idx = int(cu_num_draft_tokens[req_idx].item())
        num_tokens = end_idx - start_idx

        for pos in range(max_spec_len):
            if pos >= num_tokens:
                continue
            token_idx = start_idx + pos

            if draft_probs is None:
                # prob is target_probs[token_idx] except draft_token_id is zeroed
                prob = target_probs[token_idx].clone()
                draft_token_id = draft_token_ids[token_idx]
                prob[draft_token_id] = 0.0
            else:
                prob = (target_probs[token_idx] - draft_probs[token_idx]).clamp_min_(
                    0.0
                )

            score = prob * inv_q[req_idx]
            recovered_id = torch.argmax(score, dim=-1)
            out[token_idx] = recovered_id
    return out


def _test_masked_logits(
    rejection_sampler,
    batch_size: int,
    num_draft_tokens: int,
    vocab_size: int,
    target_logits: torch.Tensor,
    unmasked_indices: torch.Tensor,
    sampling_metadata: SamplingMetadata,
):
    # Set up test parameters
    num_tokens = batch_size * num_draft_tokens

    # Create random draft probabilities.
    draft_probs = torch.rand(
        (num_tokens, vocab_size), dtype=torch.float32, device=DEVICE
    )
    draft_probs = F.softmax(draft_probs, dim=-1)

    # Randomly sample draft token ids from draft probs
    draft_token_ids = torch.multinomial(draft_probs, num_samples=1)
    draft_token_ids = draft_token_ids.reshape(batch_size, num_draft_tokens)
    draft_token_ids = draft_token_ids.tolist()

    # Bonus tokens not used but required
    bonus_token_ids = torch.zeros((batch_size, 1), dtype=torch.int64, device=DEVICE)

    # Create spec decode metadata
    spec_decode_metadata = create_spec_decode_metadata(draft_token_ids, target_logits)

    # Run rejection sampling
    mock_sampler_output(rejection_sampler, bonus_token_ids)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=draft_probs,
        logits=target_logits,
        sampling_metadata=sampling_metadata,
    )

    # Remove bonus tokens and reshape
    output_token_ids = output.sampled_token_ids[:, :-1].flatten().tolist()

    # Check that all sampled tokens are within the unmasked indices.
    for i in range(num_tokens):
        token_id = output_token_ids[i]
        if token_id == PLACEHOLDER_TOKEN_ID:
            continue
        assert token_id in unmasked_indices[i]


@pytest.mark.parametrize("top_k", [1, 5, 99])
def test_top_k(rejection_sampler, top_k):
    """Test rejection sampling with top-k sampling"""
    vocab_size = 100
    batch_size = 100
    num_draft_tokens = 3
    num_tokens = batch_size * num_draft_tokens

    # Randomly create top-k indices.
    top_k_indices = [
        torch.randperm(vocab_size, device=DEVICE)[:top_k] for _ in range(num_tokens)
    ]
    top_k_indices = torch.stack(top_k_indices)

    # Create logits with the uniform distribution.
    target_logits = torch.zeros((num_tokens, vocab_size), device=DEVICE)

    # Increment the logits for top-k indices, a little bit more than the other
    # ones. If the masking is effective, the non-topk indices will never be
    # sampled despite the small difference in logits.
    for i in range(num_tokens):
        target_logits[i, top_k_indices[i]] += 0.1

    # Create sampling metadata
    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        top_k=torch.tensor([top_k] * batch_size, device=DEVICE, dtype=torch.int64),
    )

    _test_masked_logits(
        rejection_sampler,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        target_logits=target_logits,
        unmasked_indices=top_k_indices,
        sampling_metadata=sampling_metadata,
    )


@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.99])
def test_top_p(rejection_sampler, top_p):
    """Test rejection sampling with top-p sampling"""
    vocab_size = 100
    batch_size = 100
    num_draft_tokens = 3
    num_tokens = batch_size * num_draft_tokens

    # Create logits with the uniform distribution.
    target_logits = torch.randn((num_tokens, vocab_size), device=DEVICE)
    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
    rescaled_logits = target_logits / temperature

    logits_sort, logits_idx = rescaled_logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - top_p
    # at least one
    top_p_mask[:, -1] = False

    # Get the top-p indices.
    top_p_indices = []
    for i in range(num_tokens):
        top_p_indices.append(logits_idx[i][~top_p_mask[i]].tolist())

    # Create sampling metadata
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        top_p=torch.tensor([top_p] * batch_size, device=DEVICE, dtype=torch.float32),
    )

    _test_masked_logits(
        rejection_sampler,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        target_logits=target_logits,
        unmasked_indices=top_p_indices,
        sampling_metadata=sampling_metadata,
    )


########################### Tests for Logit Processors ###################
def test_frequency_penalties(rejection_sampler):
    """Test rejection sampling with frequency penalties"""
    spec_tokens = [[1, 1, 1], [], [1, 1, 1]]
    output_tokens = [[1, 1, 1, 1], [7], [1, 1, 1, 1]]  # 1, 7 and 1 are the bonus tokens

    num_requsts = len(spec_tokens)
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[2], [3], [4]],
        spec_token_ids=spec_tokens,
        prompt_token_ids=torch.tensor([[5, 6, 7], [6, 7, 8], [7, 8, 9]], device=DEVICE),
        frequency_penalties=[1.5, 1.5, 0.7],
        presence_penalties=[0.0] * num_requsts,
        repetition_penalties=[1.0] * num_requsts,
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        spec_tokens, device=logits.device
    )
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 15, -1, -1], [7, -1, -1, -1], [1, 1, 15, -1]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_bad_words(rejection_sampler):
    """Test rejection sampling with bad words constraints.

    This test applies bad words to non-consecutive requests (0 and 2, but not 1)
    to verify correct logit indexing when iterating over requests with bad words.
    """
    spec_tokens = [[1, 2, 3], [1, 15, 3], [1, 2, 3]]
    output_tokens = [[1, 2, 3, 4], [1, 15, 3, 4], [1, 2, 3, 4]]

    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[2], [3], [4]],
        spec_token_ids=spec_tokens,
        bad_words_token_ids={
            0: [[2]],
            # Request 1 has no bad words (to test non-consecutive request handling)
            2: [[2]],
        },
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    # Request 0: bad word [2] matches prefix, so token 2 is rejected -> 15
    # Request 1: no bad words, all tokens match -> [1, 15, 3, 4]
    # Request 2: bad word [2] matches prefix, so token 2 is rejected -> 15
    expected = torch.tensor(
        [[1, 15, -1, -1], [1, 15, 3, 4], [1, 15, -1, -1]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_allowed_token_ids(rejection_sampler):
    """Test rejection sampling with allowed token ids"""
    spec_tokens = [[1, 2, 10], [10, 5, 3], [7, 10, 12]]
    output_tokens = [[1, 2, 10, 5], [10, 5, 10, 5], [7, 10, 12, 5]]
    # Not allowed tokens:
    # 0: 0-4
    # 1: 1-5
    # 2: 2-6
    num_allowed_token_ids = 5

    # Use the token 15 as the sampler choose if a token rejected
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)

    batch_size = len(output_tokens)
    _, vocab_size = logits.size()
    mask = create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_allowed_token_ids=num_allowed_token_ids,
        device=logits.device,
    )
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[], [], []],
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=mask,
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    expected = torch.tensor(
        [[15, -1, -1, -1], [10, 5, 10, -1], [7, 10, 12, 5]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize("batch_size", [1, 100])
@pytest.mark.parametrize("vocab_size", [100, 8192, 10000])
@pytest.mark.parametrize("max_spec_len", [1, 3])
@pytest.mark.parametrize("no_draft_probs", [True, False])
def test_sample_recovered_tokens(
    batch_size: int, vocab_size: int, max_spec_len: int, no_draft_probs: bool
):
    num_tokens = batch_size * max_spec_len

    # Create random draft probabilities.
    draft_probs = torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_probs = F.softmax(draft_probs, dim=-1)

    # Create random target probabilities.
    target_logits = torch.rand(
        num_tokens, vocab_size, dtype=torch.float32, device=DEVICE
    )
    target_probs = F.softmax(target_logits, dim=-1)

    # Randomly sample draft token ids from draft probs
    draft_token_ids = torch.multinomial(draft_probs, num_samples=1).to(torch.int32)

    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
    generators = {
        i: torch.Generator(device=DEVICE).manual_seed(i) for i in range(batch_size)
    }
    sampling_metadata = create_sampling_metadata(
        all_greedy=False, temperature=temperature, generators=generators
    )

    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.reshape(batch_size, max_spec_len).tolist(), target_logits
    )

    ref_recovered_token_ids = native_sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids,
        None if no_draft_probs else draft_probs,
        target_probs,
        sampling_metadata,
        device=DEVICE,
    )
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        spec_decode_metadata.num_draft_tokens,
        spec_decode_metadata.cu_num_draft_tokens,
        draft_token_ids,
        None if no_draft_probs else draft_probs,
        target_probs,
        sampling_metadata,
        device=DEVICE,
    )
    assert torch.equal(recovered_token_ids, ref_recovered_token_ids)


########################### Tests for Ragged Batches #####################
@pytest.mark.parametrize(
    "spec_tokens,output_tokens",
    [
        ([[1, 2, 3], [4], [5, 6]], [[1, 2, 3, 10], [4, 11], [5, 6, 12]]),
        ([[1, 2, 3], [], [5, 6]], [[1, 2, 3, 10], [11], [5, 6, 12]]),
        ([[1], [2, 3], [4, 5, 6]], [[1, 10], [2, 3, 11], [4, 5, 6, 12]]),
    ],
    ids=["ragged-3-1-2", "ragged-3-0-2", "ragged-1-2-3"],
)
def test_ragged_all_accept(rejection_sampler, spec_tokens, output_tokens):
    """Ragged draft lengths with all tokens accepted + bonus."""
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus = torch.tensor([t[-1] for t in output_tokens], device=logits.device)
    sd_meta = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus)
    output = rejection_sampler(
        sd_meta, draft_probs=None, logits=logits, sampling_metadata=metadata
    )
    parsed, _ = RejectionSampler.parse_output(output.sampled_token_ids, vocab_size=100)
    assert parsed == output_tokens


def test_ragged_mixed_accept_reject(rejection_sampler):
    """Ragged batch: req0 all-accept, req1 reject-at-0, req2 accept-2."""
    spec_tokens = [[1, 2, 3], [4], [5, 6]]
    output_tokens = [
        # All spec tokens accepted + bonus appended
        [1, 2, 3, 99],
        # First spec token rejected immediately
        # Bonus token 99 will not be used
        [9, 99],
        # First spec token 5 accepted, second spec token 6 rejected
        # Recovery token 8 replaces rejected position, bonus not appended
        [5, 8, 99],
    ]
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus = torch.tensor([99, 99, 99], device=logits.device)
    sd_meta = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus)
    output = rejection_sampler(
        sd_meta, draft_probs=None, logits=logits, sampling_metadata=metadata
    )
    parsed, _ = RejectionSampler.parse_output(output.sampled_token_ids, vocab_size=100)
    # All spec tokens accepted + bonus appended
    assert parsed[0] == [1, 2, 3, 99]
    # Spec token 4 rejected, only recovery token 9 is used
    assert parsed[1] == [9]
    # First spec token 5 accepted, second spec token 6 rejected
    # Recovery token 8 replaces rejected position, bonus not appended
    assert parsed[2] == [5, 8]


######################## Tests for Tail-Alias Bug Fix ####################
def _build_logprobs_metadata(num_draft_tokens):
    """Build SpecDecodeMetadata with correct flattened index layout."""
    num_sampled = [d + 1 for d in num_draft_tokens]
    cu_sampled = np.cumsum(num_sampled, dtype=np.int32)
    cu_draft = np.cumsum(num_draft_tokens, dtype=np.int32)
    target_indices: list[int] = []
    bonus_indices: list[int] = []
    draft_ids: list[int] = []
    start = 0
    for d in num_draft_tokens:
        target_indices.extend(range(start, start + d))
        bonus_indices.append(start + d)
        draft_ids.extend([42] * d)
        start += d + 1
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(draft_ids, dtype=torch.int32),
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=torch.tensor(cu_draft, dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor(cu_sampled, dtype=torch.int32),
        target_logits_indices=torch.tensor(target_indices, dtype=torch.int32),
        bonus_logits_indices=torch.tensor(bonus_indices, dtype=torch.int32),
        logits_indices=torch.arange(int(cu_sampled[-1]), dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "num_draft_tokens,width",
    [
        pytest.param([3, 0, 0], 4, id="ragged-300"),
        pytest.param([2, 1, 0], 4, id="ragged-210"),
        pytest.param([1, 0, 2, 0], 4, id="ragged-1020"),
        pytest.param([0], 4, id="bs1-d0"),
    ],
)
def test_stale_tails_must_not_alias_real_logit_rows(num_draft_tokens, width):
    """_get_logprobs_tensors must not alias invalid tail positions
    to real logit rows when sampled_token_ids has non-PLACEHOLDER tails.

    This tests logprobs-only corruption: when logprobs are requested
    and invalid tail slots contain non-PLACEHOLDER token IDs (e.g. 0),
    the clamp-based indexing silently reads logprobs from another
    request's logit rows.

    This can happen when GPUModelRunner fails to zero _draft_token_ids
    (e.g. sync scheduling with input_fits_in_drafter=False), causing
    leftover draft proposals from a previous scheduler step to
    propagate into SpecDecodeMetadata.

    Example with num_draft_tokens=[0, 2] (two requests) and width=3:

      sampled_token_ids (tails are 0, not PLACEHOLDER):
        req0: [0, 0, 0]    # 0 draft + 1 bonus = 1 valid, 2 invalid tails
        req1: [0, 0, 0]    # 2 draft + 1 bonus = 3 valid, 0 tail

      The size of the first dimension of final_logits
      is the total number of sampled tokens (1 + 3 = 4).

      final_logits (flattened sampled rows, shape [4, vocab_size]):
        row 0: req0 bonus logits           ← req0's data
        row 1: req1 draft-0 target logits  ← req1's data
        row 2: req1 draft-1 target logits  ← req1's data
        row 3: req1 bonus logits           ← req1's data

      _get_logprobs_tensors computes flat indices per [batch, width]:
        req0: [0, 1, 2]      # only [0] is valid; [1, 2] are invalid tails
        req1: [1, 2, 3]      # all valid

      _get_logprobs_tensors will use:
      clamp_(max=final_logits.shape[0] - 1) = clamp_(max=3)

      That means:
      Indices [1, 2] for req0 are already <= 3, so clamp_ does nothing.
      They silently read rows 1 and 2 which belong to req1:
        --> req0 gets logprobs computed from req1's target logits
        --> cross-request information leakage (privacy + correctness)
    """
    sampler = Sampler(logprobs_mode="processed_logits")
    rej = RejectionSampler(sampler)
    metadata = _build_logprobs_metadata(num_draft_tokens)
    total_rows = int(metadata.cu_num_sampled_tokens[-1].item())

    target_ids = metadata.target_logits_indices.cpu().tolist()
    target_logits = (
        torch.empty((0, 2), dtype=torch.float32)
        if not target_ids
        else torch.stack([torch.full((2,), float(i)) for i in target_ids])
    )
    bonus_ids = metadata.bonus_logits_indices.cpu().tolist()
    bonus_logits = torch.stack([torch.full((2,), float(i)) for i in bonus_ids])
    logits = torch.zeros((total_rows, 2), dtype=torch.float32)

    # Stale tails: 0 instead of PLACEHOLDER (bug simulation)
    sampled_token_ids = torch.zeros((len(num_draft_tokens), width), dtype=torch.int32)

    captured = {}

    def _capture(logprobs, num_logprobs, token_ids):
        captured["rows"] = logprobs[:, 0].clone()
        n = token_ids.shape[0]
        return LogprobsTensors(
            logprob_token_ids=torch.zeros((n, 1), dtype=torch.int64),
            logprobs=torch.zeros((n, 1), dtype=torch.float32),
            selected_token_ranks=torch.zeros((n,), dtype=torch.int32),
        )

    rej.sampler.gather_logprobs = _capture
    rej._get_logprobs_tensors(
        max_num_logprobs=1,
        metadata=metadata,
        logits=logits,
        target_logits=target_logits,
        bonus_logits=bonus_logits,
        sampled_token_ids=sampled_token_ids,
    )

    rows = captured["rows"]
    # Build mask for invalid tail positions (j >= num_draft + 1)
    invalid_mask = torch.tensor(
        [j >= d + 1 for d in num_draft_tokens for j in range(width)], dtype=torch.bool
    )
    invalid_rows = rows[invalid_mask]
    assert torch.all(invalid_rows < 0), (
        f"Invalid tails must not alias real logit rows, got: {invalid_rows.tolist()}"
    )


@pytest.mark.parametrize(
    "num_draft_tokens,width",
    [
        pytest.param([3, 0, 2], 4, id="ragged-302"),
        pytest.param([0, 0, 0], 1, id="all-zero"),
        pytest.param([5], 6, id="bs1-k5"),
    ],
)
def test_logprobs_ragged_parse_output_alignment(num_draft_tokens, width):
    """Logprobs pipeline: _get_logprobs_tensors + parse_output must produce
    correct per-request token counts and aligned logprobs for ragged batches."""
    batch_size = len(num_draft_tokens)
    vocab_size = 4
    sampler = Sampler(logprobs_mode="processed_logits")
    rej = RejectionSampler(sampler)
    metadata = _build_logprobs_metadata(num_draft_tokens)
    total_rows = int(metadata.cu_num_sampled_tokens[-1].item())

    target_ids = metadata.target_logits_indices
    target_logits = (
        torch.empty((0, vocab_size), dtype=torch.float32)
        if len(target_ids) == 0
        else torch.randn(len(target_ids), vocab_size)
    )
    bonus_logits = torch.randn(len(metadata.bonus_logits_indices), vocab_size)
    logits = torch.randn(total_rows, vocab_size)
    sampled = torch.full((batch_size, width), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    for i, d in enumerate(num_draft_tokens):
        sampled[i, : d + 1] = torch.randint(0, vocab_size, (d + 1,))

    lp = rej._get_logprobs_tensors(
        1, metadata, logits, target_logits, bonus_logits, sampled
    )
    outputs, olp = RejectionSampler.parse_output(
        sampled, vocab_size=vocab_size, logprobs_tensors=lp
    )
    expected_lens = [d + 1 for d in num_draft_tokens]
    assert [len(r) for r in outputs] == expected_lens
    assert olp is not None
    assert olp.logprobs.shape[0] == sum(expected_lens)
    expected_cu = [0] + np.cumsum(expected_lens, dtype=np.int32).tolist()
    assert olp.cu_num_generated_tokens == expected_cu


def test_zero_drafts_accept_when_target_matches(rejection_sampler):
    """Draft tokens all 0 (drafter skip) and target also favors 0:
    all drafts accepted + bonus appended."""
    spec_tokens = [[0, 0, 0]]
    output_tokens = [[0, 0, 0, 99]]
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus = torch.tensor([99], device=logits.device)
    sd_meta = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus)
    output = rejection_sampler(
        sd_meta, draft_probs=None, logits=logits, sampling_metadata=metadata
    )
    row = output.sampled_token_ids[0].tolist()
    real = [t for t in row if t != PLACEHOLDER_TOKEN_ID]
    assert real == [0, 0, 0, 99]


def test_zero_drafts_reject_when_target_differs(rejection_sampler):
    """Draft tokens all 0 (drafter skip) but target favors 5:
    first draft rejected immediately."""
    spec_tokens = [[0, 0, 0]]
    output_tokens = [[5, 0, 0, 99]]
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus = torch.tensor([99], device=logits.device)
    sd_meta = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus)
    output = rejection_sampler(
        sd_meta, draft_probs=None, logits=logits, sampling_metadata=metadata
    )
    row = output.sampled_token_ids[0].tolist()
    real = [t for t in row if t != PLACEHOLDER_TOKEN_ID]
    assert real == [5]


def test_parse_output_filters_oov_and_placeholder():
    """parse_output filters element-wise: each position is independently
    checked against (token != PLACEHOLDER) and (token < vocab_size).
    Both PLACEHOLDER (-1) and OOV (>= vocab_size) are removed."""
    ids = torch.tensor(
        [
            # 5: valid, -1: placeholder, 120: OOV (>=100), 7: valid --> [5, 7]
            [5, PLACEHOLDER_TOKEN_ID, 120, 7],
            # 9: valid, 11: valid, -1: placeholder, 200: OOV --> [9, 11]
            [9, 11, PLACEHOLDER_TOKEN_ID, 200],
            [159, 57, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int32,
    )
    out, _ = RejectionSampler.parse_output(ids, vocab_size=100)
    assert out == [[5, 7], [9, 11], [57]]
