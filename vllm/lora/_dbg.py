# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight tracing helper for LoRA pipeline debugging.

Toggle with ``VLLM_LORA_DBG=1``. All output goes to stderr with
``flush=True`` so messages survive even when the worker freezes.
"""

import os
import sys
import time

_LORA_DBG = os.environ.get("VLLM_LORA_DBG", "0") == "1"
_LORA_DBG_T0 = time.monotonic()


def lora_dbg_enabled() -> bool:
    return _LORA_DBG


def lora_dbg(msg: str) -> None:
    if not _LORA_DBG:
        return
    pid = os.getpid()
    t = time.monotonic() - _LORA_DBG_T0
    print(f"[LORA-DBG t={t:8.3f}s pid={pid}] {msg}", file=sys.stderr, flush=True)
