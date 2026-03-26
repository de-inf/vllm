# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
python3 -m pytest tests/v1/spec_decode/test_nemotron_mtp_regression.py -k tp2
"""

import math
from dataclasses import dataclass

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

NEMOTRON_BF16_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
NEMOTRON_FP8_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"

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


@dataclass
class _SeqDiag:
    seq_idx: int
    num_gen_tokens: int
    tme: float
    mean_abs_lp_diff: float
    max_abs_lp_diff: float
    max_abs_lp_diff_pos: int
    last_abs_lp_diff: float
    has_last_token_cliff: bool


def _compute_seq_diag(
    seq_idx: int, gen_lps: list[float], score_lps: list[float]
) -> _SeqDiag:
    valid_abs_diffs: list[float] = []
    valid_positions: list[int] = []
    for j, (a, b) in enumerate(zip(gen_lps, score_lps)):
        if math.isnan(a) or math.isnan(b):
            continue
        valid_abs_diffs.append(abs(a - b))
        valid_positions.append(j)

    if not valid_abs_diffs:
        return _SeqDiag(
            seq_idx=seq_idx,
            num_gen_tokens=len(gen_lps),
            tme=1.0,
            mean_abs_lp_diff=0.0,
            max_abs_lp_diff=0.0,
            max_abs_lp_diff_pos=-1,
            last_abs_lp_diff=0.0,
            has_last_token_cliff=False,
        )

    max_idx = max(range(len(valid_abs_diffs)), key=lambda i: valid_abs_diffs[i])
    max_pos = valid_positions[max_idx]
    max_abs = valid_abs_diffs[max_idx]
    mean_abs = sum(valid_abs_diffs) / len(valid_abs_diffs)
    last_abs = valid_abs_diffs[-1]

    # Heuristic: a "last-token cliff" is when the largest divergence is at the
    # final valid generated token and clearly above baseline jitter.
    has_last_token_cliff = (max_pos == valid_positions[-1]) and (max_abs > 1.0)

    return _SeqDiag(
        seq_idx=seq_idx,
        num_gen_tokens=len(gen_lps),
        tme=math.exp(mean_abs),
        mean_abs_lp_diff=mean_abs,
        max_abs_lp_diff=max_abs,
        max_abs_lp_diff_pos=max_pos,
        last_abs_lp_diff=last_abs,
        has_last_token_cliff=has_last_token_cliff,
    )


def _format_diag_summary(diags: list[_SeqDiag], top_k: int = 4) -> str:
    if not diags:
        return "no sequence diagnostics available"

    sorted_diags = sorted(diags, key=lambda d: d.max_abs_lp_diff, reverse=True)
    top = sorted_diags[:top_k]
    last_cliff_count = sum(1 for d in diags if d.has_last_token_cliff)
    avg_max_abs = sum(d.max_abs_lp_diff for d in diags) / len(diags)

    lines = [
        (
            "diag_overview: "
            f"seqs={len(diags)} "
            f"last_token_cliffs={last_cliff_count} "
            f"avg_max_abs_lp_diff={avg_max_abs:.3f}"
        )
    ]
    for d in top:
        lines.append(
            "top_seq: "
            f"idx={d.seq_idx} tokens={d.num_gen_tokens} "
            f"tme={d.tme:.3f} mean_abs={d.mean_abs_lp_diff:.3f} "
            f"max_abs={d.max_abs_lp_diff:.3f}@{d.max_abs_lp_diff_pos} "
            f"last_abs={d.last_abs_lp_diff:.3f} "
            f"last_cliff={d.has_last_token_cliff}"
        )
    return " | ".join(lines)


def _run_mtp_vs_score(
    *, model: str, tp: int, use_ray: bool, max_num_seqs: int
) -> tuple[float, int, str]:
    common_kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        # Allow BF16 and FP8 checkpoints to choose their intended runtime dtype.
        dtype="auto",
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
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
        max_model_len=8192,
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
    seq_diags: list[_SeqDiag] = []
    for i, (seq, sout) in enumerate(zip(sequences, score_outputs)):
        plen = len(seq["prompt_ids"])
        score_lps = _extract_score_logprobs(sout, plen, seq["gen_ids"])
        seq_tmes.append(_compute_tme(seq["gen_lps"], score_lps))
        seq_diags.append(_compute_seq_diag(i, seq["gen_lps"], score_lps))

    avg_tme = sum(seq_tmes) / len(seq_tmes)
    outliers = sum(1 for t in seq_tmes if t > 2.0)
    return avg_tme, outliers, _format_diag_summary(seq_diags)


_RAY_XFAIL_REASON = (
    "Known Ray+MTP regression: logprob divergence and outliers "
    "on Nemotron MTP workloads with ragged speculative batches."
)


def _select_model_for_tp(tp: int) -> str:
    # TP1 uses the FP8 checkpoint to avoid OOM with the BF16 checkpoint.
    return NEMOTRON_FP8_MODEL if tp == 1 else NEMOTRON_BF16_MODEL


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
    model = _select_model_for_tp(tp)

    # Skip matrix entries that cannot run on the current machine.
    available_gpus = torch.cuda.device_count()
    if available_gpus < tp:
        pytest.skip(
            f"Need at least {tp} GPUs for this case, but found {available_gpus}."
        )

    try:
        avg_tme, outliers, diag_summary = _run_mtp_vs_score(
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
        f"got avg_tme={avg_tme:.3f}. {diag_summary}"
    )
    assert outliers == 0, (
        f"Expected no {backend} outliers at max_num_seqs={max_num_seqs}, "
        f"got outliers={outliers}. {diag_summary}"
    )
