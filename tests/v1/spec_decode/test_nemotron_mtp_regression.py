# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

PROMPTS = [
    "Solve briefly: 2x + 3 = 11.",
    "Write one sentence about matrix multiplication.",
    "Compute: 19 * 23.",
    "Name three prime numbers greater than 10.",
]


def _extract_gen_logprobs(output):
    # Generated sequence-level outputs:
    # - `token_ids`: completion tokens generated after the prompt
    # - `lps`: per-generated-token logprob from the generation pass
    gen = output.outputs[0]
    token_ids = list(gen.token_ids)
    lps = []
    for j, lp_dict in enumerate(gen.logprobs or []):
        if lp_dict and j < len(token_ids):
            tid = token_ids[j]
            if tid in lp_dict:
                lps.append(lp_dict[tid].logprob)
            else:
                lps.append(next(iter(lp_dict.values())).logprob)
        else:
            lps.append(0.0)
    return token_ids, lps


def _extract_score_logprobs(output, prompt_len, gen_ids):
    # Scoring pass outputs prompt logprobs over the full prompt+completion.
    # We slice back to completion positions (`prompt_len + j`) so we can
    # compare generation-time logprobs vs score-time logprobs token-by-token.
    lps = []
    full_ids = list(output.prompt_token_ids)
    for j in range(len(gen_ids)):
        pos = prompt_len + j
        if pos < len(output.prompt_logprobs) and output.prompt_logprobs[pos]:
            lp_dict = output.prompt_logprobs[pos]
            tid = full_ids[pos]
            if tid in lp_dict:
                lps.append(lp_dict[tid].logprob)
            else:
                lps.append(float("nan"))
        else:
            lps.append(float("nan"))
    return lps


def _compute_tme(lps_a, lps_b):
    total, count = 0.0, 0
    for a, b in zip(lps_a, lps_b):
        if math.isnan(a) or math.isnan(b):
            continue
        total += math.exp(abs(a - b))
        count += 1
    return total / max(count, 1)


def _run_mtp_vs_score(
    *, model: str, tp: int, use_ray: bool, max_num_seqs: int
) -> tuple[float, int]:
    common_kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
        max_model_len=256,
        gpu_memory_utilization=0.7,
        logprobs_mode="processed_logprobs",
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype="float32",
    )
    if use_ray:
        common_kwargs["distributed_executor_backend"] = "ray"

    llm_gen = LLM(
        **common_kwargs,
        # MTP speculation width: each step can propose up to 2 draft tokens
        # before target-model verification.
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    )
    gen_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=96,
        logprobs=0,
    )
    # Duplicate each prompt so the scheduler sees enough requests to form
    # batches. Setting max_num_seqs=1 in common_kwargs effectively forces
    # single-request execution even with multiple queued requests.
    requests = [{"prompt": p} for p in PROMPTS for _ in range(4)]
    gen_outputs = llm_gen.generate(requests, gen_params)

    sequences = []
    for out in gen_outputs:
        gen_ids, gen_lps = _extract_gen_logprobs(out)
        # `prompt_ids`: tokenizer output for prompt text only.
        # `gen_ids`: generated completion token IDs.
        # `full_ids`: prompt_ids + gen_ids, used as input in scoring pass.
        prompt_ids = list(out.prompt_token_ids)
        sequences.append(
            {
                "prompt_ids": prompt_ids,
                "gen_ids": gen_ids,
                "full_ids": prompt_ids + gen_ids,
                "gen_lps": gen_lps,
            }
        )
    del llm_gen

    score_model_len = max(len(s["full_ids"]) for s in sequences) + 1
    llm_score = LLM(**common_kwargs, max_model_len=score_model_len)
    score_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    # Score exactly the same token stream to obtain reference logprobs.
    score_prompts = [{"prompt_token_ids": s["full_ids"]} for s in sequences]
    score_outputs = llm_score.generate(score_prompts, sampling_params=score_params)
    del llm_score

    seq_tmes = []
    for seq, sout in zip(sequences, score_outputs):
        plen = len(seq["prompt_ids"])
        score_lps = _extract_score_logprobs(sout, plen, seq["gen_ids"])
        seq_tmes.append(_compute_tme(seq["gen_lps"], score_lps))

    avg_tme = sum(seq_tmes) / len(seq_tmes)
    outliers = sum(1 for t in seq_tmes if t > 2.0)
    return avg_tme, outliers


_RAY_XFAIL_REASON = (
    "Known Ray+MTP regression: logprob divergence and outliers "
    "on Nemotron MTP workloads with ragged speculative batches."
)


@pytest.mark.parametrize(
    "use_ray,tp,max_num_seqs",
    [
        pytest.param(False, 1, 1, id="mp-tp1-bs1"),
        pytest.param(False, 1, 16, id="mp-tp1-bs16"),
        pytest.param(False, 2, 1, id="mp-tp2-bs1"),
        pytest.param(False, 2, 16, id="mp-tp2-bs16"),
        pytest.param(False, 4, 1, id="mp-tp4-bs1"),
        pytest.param(False, 4, 16, id="mp-tp4-bs16"),
        # Ray + bs=1 is intentionally NOT xfail to verify whether serial
        # execution avoids the regression.
        pytest.param(True, 1, 1, id="ray-tp1-bs1"),
        pytest.param(
            True,
            1,
            16,
            id="ray-tp1-bs16",
            marks=pytest.mark.xfail(
                strict=True,
                reason=_RAY_XFAIL_REASON,
            ),
        ),
        pytest.param(True, 2, 1, id="ray-tp2-bs1"),
        pytest.param(
            True,
            2,
            16,
            id="ray-tp2-bs16",
            marks=pytest.mark.xfail(
                strict=True,
                reason=_RAY_XFAIL_REASON,
            ),
        ),
        pytest.param(True, 4, 1, id="ray-tp4-bs1"),
        pytest.param(
            True,
            4,
            16,
            id="ray-tp4-bs16",
            marks=pytest.mark.xfail(
                strict=True,
                reason=_RAY_XFAIL_REASON,
            ),
        ),
    ],
)
def test_nemotron_mtp_logprob_regression_matrix(
    use_ray: bool, tp: int, max_num_seqs: int
):
    """Optional Nemotron MTP regression matrix for backend and TP size."""
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    # Skip matrix entries that cannot run on the current machine.
    available_gpus = torch.cuda.device_count()
    if available_gpus < tp:
        pytest.skip(
            f"Need at least {tp} GPUs for this case, but found {available_gpus}."
        )

    try:
        avg_tme, outliers = _run_mtp_vs_score(
            model=model,
            tp=tp,
            use_ray=use_ray,
            max_num_seqs=max_num_seqs,
        )
    finally:
        cleanup_dist_env_and_memory()

    # This expectation is intentionally set to the desired post-fix behavior.
    # MP cases should pass today. Ray+bs16 cases are strict-xfail and will
    # turn XPASS once fixed. Ray+bs1 cases are unmarked so we can observe
    # whether batch-size-1 avoids the issue.
    backend = "ray" if use_ray else "mp"
    assert avg_tme < 1.2, (
        f"Expected stable {backend} path at max_num_seqs={max_num_seqs}, "
        f"got avg_tme={avg_tme:.3f}"
    )
    assert outliers == 0, (
        f"Expected no {backend} outliers at max_num_seqs={max_num_seqs}, "
        f"got outliers={outliers}"
    )
