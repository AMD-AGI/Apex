#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
reward_trace.py — JSONL reward trace emission for debugging and analysis.

Appends one JSON line per reward computation to:
    outputs/<YYYY-MM-DD>/reward_debug/reward_trace.jsonl

Must never raise — logging failures must not break scoring.

Ported from keystone-rl-training/reward_fn.py trace subsystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Route labels ─────────────────────────────────────────────────────────────

ROUTE_NO_ANSWER = "no_answer"
ROUTE_AST_SYNTAX_ERROR = "ast_syntax_error"
ROUTE_AST_BLOCKED = "ast_blocked"
ROUTE_AST_NO_TRITON_JIT = "ast_no_triton_jit"
ROUTE_HIP_STATIC_BLOCKED = "hip_static_blocked"
ROUTE_HIP_NO_BINDING = "hip_no_binding"
ROUTE_HIP_NO_KERNEL = "hip_no_kernel"
ROUTE_EVAL_TIMEOUT = "eval_timeout"
ROUTE_EVAL_COMPILE_FAIL = "eval_compile_fail"
ROUTE_EVAL_HACKING = "eval_hacking_detected"
ROUTE_COMPILED_LOW_CORRECTNESS = "compiled_low_correctness"
ROUTE_CORRECT_BUT_NO_PERF = "correct_but_no_perf"
ROUTE_PERF_REWARDED = "perf_rewarded"

# ── Internal constants ───────────────────────────────────────────────────────

_REWARD_TRACE_ROOT = Path(__file__).parent.parent / "outputs"
_REWARD_TRACE_PREVIEW_CHARS = 200
_PST = timezone(timedelta(hours=-8), name="PST")
_RUN_ROOT_ENV_VAR = "APEX_RUN_ROOT"

_reward_trace_warned = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _current_pst_datetime() -> datetime:
    """Return the current timestamp in fixed PST (UTC-08:00)."""
    return datetime.now(_PST)


def _reward_trace_enabled() -> bool:
    """Enable traces by default outside tests, with env override support."""
    override = os.environ.get("REWARD_TRACE_ENABLE")
    if override is not None:
        return override.lower() not in {"0", "false", "no"}
    return "PYTEST_CURRENT_TEST" not in os.environ


def _resolve_run_root(now: datetime | None = None) -> Path:
    """Return the active run root, honoring the shared run-root env var."""
    configured = os.environ.get(_RUN_ROOT_ENV_VAR)
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path

    ts = now or _current_pst_datetime()
    return _REWARD_TRACE_ROOT / ts.strftime("%Y-%m-%d")


def _reward_trace_file_path(now: datetime | None = None) -> Path:
    """Return the shared JSONL trace file for the active run."""
    trace_dir = _resolve_run_root(now) / "reward_debug"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir / "reward_trace.jsonl"


def _append_jsonl_line(path: Path, payload: dict) -> None:
    """Append one JSON object as a single UTF-8 line."""
    line = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def hash_text(text: str | None) -> str | None:
    """SHA-1 hash of text for trace deduplication. Returns None for None input."""
    if text is None:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def preview_text(text: str | None, limit: int = _REWARD_TRACE_PREVIEW_CHARS) -> str | None:
    """Truncated preview of text for trace readability."""
    if text is None:
        return None
    preview = text[:limit]
    if len(text) > limit:
        preview += "...<truncated>"
    return preview


def extract_syntax_error_location(ast_reason: str | None) -> tuple[int | None, int | None]:
    """Best-effort line/offset extraction from the SyntaxError string."""
    if not ast_reason or not ast_reason.startswith("SyntaxError"):
        return None, None
    line_match = re.search(r"line (\d+)", ast_reason)
    offset_match = re.search(r"offset (\d+)", ast_reason)
    line = int(line_match.group(1)) if line_match else None
    offset = int(offset_match.group(1)) if offset_match else None
    return line, offset


# ── Public API ───────────────────────────────────────────────────────────────

def emit_reward_trace(trace: dict) -> None:
    """
    Append one JSONL reward trace without affecting scoring on failure.

    Parameters
    ----------
    trace : dict
        Trace payload. Must be JSON-serializable. Should include at minimum:
        ``trace_version``, ``timestamp_pst``, ``route``, ``reward_total``,
        and all ``r_*`` component fields.
    """
    global _reward_trace_warned
    if not _reward_trace_enabled():
        return
    try:
        _append_jsonl_line(_reward_trace_file_path(), trace)
    except Exception as exc:  # pragma: no cover - logging must never break scoring
        if not _reward_trace_warned:
            print(f"[reward] failed to write reward trace: {exc}", flush=True)
            _reward_trace_warned = True
