#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
reward_fn.py — Multi-stage gated reward pipeline for GPU kernel RL training.

Ported from keystone-rl-training/reward_fn.py and adapted to consume
Apex's Magpie-based evaluation results instead of calling an internal
evaluator agent.

Reward decomposition (all components always computed; early returns include penalty):
    R_total = R_format + R_ast_cheat + R_compile + R_correct + R_perf + R_penalty

    R_penalty  (-0.001 * max(0, N_think - 500)) — always; discourages token bloat
    R_format   (+0.1)        — <answer> tag present (R_NO_ANSWER=-1.0 if absent)
    R_ast_cheat(-2.0)        — deliberate bypass detected (blocked torch attr,
                               import escape, no @triton.jit); 0.0 for SyntaxError
    R_compile  (+0.2)        — kernel compiled AND passed GPU smoke test
    R_correct  (+0.7 * s)    — s = continuous correctness score in [0, 1]
    R_perf     (gated, [-1, 3] * s) — log2 speedup; only if s > 0.95 and
                                       baseline_ms > MIN_MEASURABLE_MS (10us)

Reward ranges by outcome (approximate):
    -3.1 : deliberate hack (R_format + R_ast_cheat + R_penalty floor)
    -1.0 : no <answer> tag (clean rollout with no think bloat)
     0.1 : formatted, AST clean, failed to compile
     0.3 : compiled, incorrect (correctness ~ 0)
     1.0 : correct (s = 1.0), no performance bonus
     4.1 : correct and 8x faster than baseline (log2(8) = 3.0 * 1.0)

Usage:
    from graders.reward_fn import compute_gated_reward
"""

from __future__ import annotations

import math
import os
import re
import socket
from datetime import datetime

from reward_backends import (
    KERNEL_BACKEND_HIP,
    KERNEL_BACKEND_TRITON,
    normalize_answer_for_backend,
    run_backend_static_check,
    run_triton_static_check,
)
from reward_trace import (
    ROUTE_AST_BLOCKED,
    ROUTE_AST_NO_TRITON_JIT,
    ROUTE_AST_SYNTAX_ERROR,
    ROUTE_COMPILED_LOW_CORRECTNESS,
    ROUTE_CORRECT_BUT_NO_PERF,
    ROUTE_EVAL_COMPILE_FAIL,
    ROUTE_EVAL_HACKING,
    ROUTE_EVAL_TIMEOUT,
    ROUTE_HIP_NO_BINDING,
    ROUTE_HIP_NO_KERNEL,
    ROUTE_HIP_STATIC_BLOCKED,
    ROUTE_NO_ANSWER,
    ROUTE_PERF_REWARDED,
    _current_pst_datetime,
    emit_reward_trace,
    extract_syntax_error_location,
    hash_text,
    preview_text,
)


# ── Reward constants ──────────────────────────────────────────────────────────

R_FORMAT_PASS     =  0.1   # <answer> tag found
R_COMPILE_PASS    =  0.2   # kernel compiled
R_CORRECT_WEIGHT  =  0.7   # multiplied by continuous correctness score
R_PERF_MIN        = -1.0   # clip floor for log2(speedup)
R_PERF_MAX        =  3.0   # clip ceiling for log2(speedup)
R_THINK_WEIGHT    = -0.001 # per-token penalty beyond the free budget
R_THINK_FREE_TOKS =  500   # tokens allowed before penalty kicks in
CORRECTNESS_GATE  =  0.95  # minimum correctness score to unlock R_perf
R_NO_ANSWER       = -1.0   # returned when <answer> tag is missing entirely
R_FORMAT_FAIL     = R_NO_ANSWER  # alias for tests and external callers
R_AST_FAIL_PENALTY = -2.0  # deliberate reward-hacking detected
MIN_MEASURABLE_MS  =  0.01 # baselines below 10us are dominated by clock noise


# ── Tag parsing ───────────────────────────────────────────────────────────────

def parse_solution_tags(solution_str: str) -> tuple[str | None, str | None]:
    """
    Extract <think> and <answer> block contents from the model's output.

    Parameters
    ----------
    solution_str : str
        Raw model output string, expected to contain ``<think>...</think>``
        and ``<answer>...</answer>`` delimiters.

    Returns
    -------
    tuple[str | None, str | None]
        ``(think_content, answer_content)``. Either element may be ``None``
        if the corresponding tags were not found.
    """
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

    think_content = think_match.group(1) if think_match else None
    answer_content = answer_match.group(1) if answer_match else None

    return think_content, answer_content


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count by splitting on whitespace.

    Parameters
    ----------
    text : str
        Text to measure.

    Returns
    -------
    int
        Number of whitespace-delimited words.
    """
    return len(text.split())


# ── Individual reward components ──────────────────────────────────────────────

def compute_format_reward(answer_content: str | None) -> float:
    """
    Return R_FORMAT_PASS if <answer> tag was found, else R_NO_ANSWER.

    Parameters
    ----------
    answer_content : str | None
        Extracted answer block, or ``None`` if the tag was missing.

    Returns
    -------
    float
        ``R_FORMAT_PASS`` (0.1) on success, ``R_NO_ANSWER`` (-1.0) on failure.
    """
    return R_FORMAT_PASS if answer_content is not None else R_NO_ANSWER


def compute_length_penalty(think_content: str | None) -> float:
    """
    Compute the reasoning-length penalty term R_penalty.

    Formula::

        R_penalty = R_THINK_WEIGHT * max(0, N_think_tokens - R_THINK_FREE_TOKS)

    Parameters
    ----------
    think_content : str | None
        Content of the ``<think>`` block, or ``None`` if absent.

    Returns
    -------
    float
        A non-positive penalty value (0.0 if within the free budget).
    """
    if think_content is None:
        return 0.0
    n_tokens = count_tokens_approx(think_content)
    return R_THINK_WEIGHT * max(0, n_tokens - R_THINK_FREE_TOKS)


def run_ast_whitelist_check(
    code: str,
    allowed_imports: list[str] | None = None,
) -> tuple[bool, str]:
    """Backward-compatible wrapper for Triton static checks."""
    return run_triton_static_check(code, allowed_imports=allowed_imports)


def compute_performance_reward(
    baseline_ms: float,
    optimized_ms: float,
    correctness_score: float,
) -> float:
    """
    Compute the gated performance reward R_perf.

    Only awarded when the correctness score exceeds ``CORRECTNESS_GATE``
    (0.95), preventing fast-but-wrong kernels from earning positive
    performance reward.

    Formula::

        speedup  = baseline_ms / optimized_ms
        r_perf   = clip(log2(speedup), R_PERF_MIN, R_PERF_MAX)
        R_perf   = correctness_score * r_perf   (if correctness_score > 0.95)
                 = 0.0                           (otherwise)

    Parameters
    ----------
    baseline_ms : float
        Baseline kernel execution time in milliseconds.
    optimized_ms : float
        Optimized kernel execution time in milliseconds.
    correctness_score : float
        Continuous correctness score in [0, 1].

    Returns
    -------
    float
        Performance reward, or 0.0 if gating conditions are not met.
    """
    if correctness_score <= CORRECTNESS_GATE:
        return 0.0
    if optimized_ms <= 0 or baseline_ms <= 0:
        return 0.0
    if baseline_ms <= MIN_MEASURABLE_MS:
        return 0.0

    speedup = baseline_ms / optimized_ms
    r_perf = max(R_PERF_MIN, min(R_PERF_MAX, math.log2(speedup)))
    return correctness_score * r_perf


def detect_runtime_hacking(
    baseline_output_stats: dict,
    optimized_output_stats: dict,
    optimized_ms: float,
    baseline_ms: float = 0.0,
) -> tuple[bool, str]:
    """
    Post-evaluation hacking detection.

    Mirrors Keystone's ``detect_runtime_hacking()`` checks but operates
    on pre-computed statistics dicts rather than raw torch tensors, so
    this module stays torch-free.

    Parameters
    ----------
    baseline_output_stats : dict
        Statistics of the baseline kernel output. Expected keys:
        ``"std"`` (float), ``"mean"`` (float).
        Optional: ``"has_nan"`` (bool), ``"has_inf"`` (bool),
        ``"matches_input"`` (bool).
    optimized_output_stats : dict
        Statistics of the optimized kernel output. Same keys.
    optimized_ms : float
        Optimized kernel execution time in milliseconds.
    baseline_ms : float
        Baseline kernel execution time in milliseconds.

    Returns
    -------
    tuple[bool, str]
        ``(True, reason)`` if hacking detected, ``(False, "")`` otherwise.
    """
    # 1. NaN / Inf in optimized output
    if optimized_output_stats.get("has_nan", False):
        return True, "nan_inf_output: kernel produces NaN values"
    if optimized_output_stats.get("has_inf", False):
        return True, "nan_inf_output: kernel produces Inf values"

    # 2. Constant output when reference is non-trivial
    opt_std = optimized_output_stats.get("std", float("inf"))
    base_std = baseline_output_stats.get("std", 0.0)
    if opt_std < 1e-6 and base_std > 1e-3:
        return True, "constant_output: kernel returns near-constant tensor"

    # 3. Identity / pass-through: output matches an input but golden doesn't
    if (
        optimized_output_stats.get("matches_input", False)
        and not baseline_output_stats.get("matches_input", False)
    ):
        return True, "identity_kernel: output matches input unchanged"

    # 4. Suspiciously fast (relative threshold, matching Keystone)
    if baseline_ms > 0.1 and optimized_ms > 0 and optimized_ms < baseline_ms * 0.01:
        return (
            True,
            f"suspiciously_fast: {optimized_ms:.4f}ms vs baseline {baseline_ms:.4f}ms",
        )

    return False, ""


# ── Main reward function ──────────────────────────────────────────────────────

def compute_gated_reward(
    solution_str: str,
    ground_truth: dict,
    eval_result: dict,
    kernel_backend: str = KERNEL_BACKEND_TRITON,
    require_tags: bool = True,
    trace_context: dict | None = None,
) -> float:
    """
    Compute the final scalar reward for a single model rollout.

    Executes the multi-stage gated reward pipeline:

    1. Parse ``<think>``/``<answer>`` tags.
    2. Compute length penalty (always applied, even on early return).
    3. Format gate — return early if no ``<answer>`` tag.
    4. Normalize answer, run backend static check (AST gate).
    5. Read pre-computed eval_result from Magpie.
    6. Runtime hacking override.
    7. Sum: R_format + R_compile + R_correct + R_perf + R_penalty.

    Parameters
    ----------
    solution_str : str
        Full raw output string from the LLM, possibly including
        ``<think>``/``<answer>`` tags.
    ground_truth : dict
        Ground truth metadata. May contain ``allowed_imports``, ``op_type``,
        ``base_gpu_kernel_type``.
    eval_result : dict
        Pre-computed evaluation results from Magpie. Required keys:
        ``compiled`` (bool), ``correctness_score`` (float),
        ``baseline_ms`` (float), ``optimized_ms`` (float).
        Optional: ``hacking_detected`` (bool), ``timed_out`` (bool).
    kernel_backend : str
        Backend identifier: ``"triton"`` or ``"hip"``.
    require_tags : bool
        If ``True``, missing ``<answer>`` tags earn ``R_NO_ANSWER``.
        If ``False``, the entire ``solution_str`` is treated as the answer
        with ``r_format = 0.0`` when tags are absent (for Apex agents that
        write bare solution files without tags).
    trace_context : dict | None
        Optional metadata for JSONL trace (task_id, difficulty_level, etc.).

    Returns
    -------
    float
        Scalar reward in approximately [-3.1, 4.1].
    """
    trace_context = trace_context or {}

    # 1. Parse tags
    think_content, answer_content = parse_solution_tags(solution_str)
    think_tokens = count_tokens_approx(think_content) if think_content is not None else 0

    # Handle require_tags=False: treat entire solution as answer
    if not require_tags and answer_content is None:
        answer_content = solution_str.strip() if solution_str.strip() else None

    trace = {
        "trace_version": 1,
        "timestamp_pst": _current_pst_datetime().isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "data_source": trace_context.get("data_source"),
        "task_id": trace_context.get("task_id"),
        "difficulty_level": trace_context.get("difficulty_level"),
        "op_type": trace_context.get("op_type", ground_truth.get("op_type")),
        "kernel_backend": kernel_backend,
        "route": None,
        "reward_total": None,
        "r_format": None,
        "r_ast": 0.0,
        "r_compile": 0.0,
        "r_correct": 0.0,
        "r_perf": 0.0,
        "r_penalty": None,
        "think_tokens": think_tokens,
        "solution_len_chars": len(solution_str),
        "has_think": think_content is not None,
        "has_answer": answer_content is not None,
        "answer_len_chars": len(answer_content) if answer_content is not None else 0,
        "answer_sha1": hash_text(answer_content),
        "answer_preview": preview_text(answer_content),
        "ast_ok": None,
        "ast_reason": None,
        "syntax_error_line": None,
        "syntax_error_offset": None,
        "compiled": None,
        "correctness_score": None,
        "baseline_ms": None,
        "optimized_ms": None,
        "speedup": None,
        "timed_out": None,
        "hacking_detected": None,
        "perf_gate_opened": None,
    }

    # 2. Length penalty — always applied, even on early return
    r_penalty = compute_length_penalty(think_content)
    trace["r_penalty"] = r_penalty

    # 3. Format gate
    if require_tags:
        r_format = compute_format_reward(answer_content)
    else:
        # When tags are not required, give format pass if answer exists, else 0.0
        r_format = R_FORMAT_PASS if answer_content is not None else 0.0
    trace["r_format"] = r_format

    if answer_content is None:
        total_reward = r_format + r_penalty
        trace["route"] = ROUTE_NO_ANSWER
        trace["reward_total"] = total_reward
        emit_reward_trace(trace)
        return total_reward

    normalized_answer = normalize_answer_for_backend(answer_content, kernel_backend)
    trace["answer_len_chars"] = len(normalized_answer)
    trace["answer_sha1"] = hash_text(normalized_answer)
    trace["answer_preview"] = preview_text(normalized_answer)

    # 4. Backend-specific static gate
    allowed_imports = ground_truth.get("allowed_imports")
    ast_ok, ast_reason = run_backend_static_check(
        normalized_answer,
        kernel_backend,
        allowed_imports=allowed_imports,
    )
    trace["ast_ok"] = ast_ok
    trace["ast_reason"] = ast_reason or None
    if not ast_ok:
        is_hack = not ast_reason.startswith("SyntaxError")
        r_ast = R_AST_FAIL_PENALTY if is_hack else 0.0
        trace["r_ast"] = r_ast
        syntax_error_line, syntax_error_offset = extract_syntax_error_location(ast_reason)
        trace["syntax_error_line"] = syntax_error_line
        trace["syntax_error_offset"] = syntax_error_offset
        if ast_reason.startswith("SyntaxError"):
            trace["route"] = ROUTE_AST_SYNTAX_ERROR
        elif ast_reason.startswith("no_triton_jit:") or ast_reason.startswith("no_jit:"):
            trace["route"] = ROUTE_AST_NO_TRITON_JIT
        elif ast_reason.startswith("no_hip_binding:"):
            trace["route"] = ROUTE_HIP_NO_BINDING
        elif ast_reason.startswith("no_hip_kernel:") or ast_reason.startswith("no_hip_code:"):
            trace["route"] = ROUTE_HIP_NO_KERNEL
        elif kernel_backend == KERNEL_BACKEND_HIP:
            trace["route"] = ROUTE_HIP_STATIC_BLOCKED
        else:
            trace["route"] = ROUTE_AST_BLOCKED
        total_reward = r_format + r_ast + r_penalty
        trace["reward_total"] = total_reward
        emit_reward_trace(trace)
        return total_reward

    # 5. Read pre-computed eval result (from Magpie, not an evaluator agent)
    compiled = eval_result.get("compiled", False)
    correctness_score = eval_result.get("correctness_score", 0.0)
    baseline_ms = eval_result.get("baseline_ms", 0.0)
    optimized_ms = eval_result.get("optimized_ms", 0.0)
    timed_out = eval_result.get("timed_out", False)
    hacking_detected = eval_result.get("hacking_detected", False)

    trace["compiled"] = compiled
    trace["correctness_score"] = correctness_score
    trace["baseline_ms"] = baseline_ms
    trace["optimized_ms"] = optimized_ms
    trace["timed_out"] = timed_out
    trace["hacking_detected"] = hacking_detected
    if optimized_ms > 0:
        trace["speedup"] = baseline_ms / optimized_ms

    # 6. Runtime hacking override
    if hacking_detected:
        total_reward = r_format + R_AST_FAIL_PENALTY + r_penalty
        trace["route"] = ROUTE_EVAL_HACKING
        trace["r_ast"] = R_AST_FAIL_PENALTY
        trace["reward_total"] = total_reward
        emit_reward_trace(trace)
        return total_reward

    # 7. Compile reward
    r_compile = R_COMPILE_PASS if compiled else 0.0
    trace["r_compile"] = r_compile

    # 8. Correctness reward
    # TODO: When Magpie supports continuous correctness (tolerance-based),
    # use compute_continuous_correctness() parameterized by alpha per op_type:
    #   compute_bound (GEMM, attention): alpha=5.0
    #   memory_bound (elementwise, reduction): alpha=8.0
    r_correct = R_CORRECT_WEIGHT * correctness_score
    trace["r_correct"] = r_correct

    # 9. Performance reward (gated)
    r_perf = compute_performance_reward(baseline_ms, optimized_ms, correctness_score)
    trace["r_perf"] = r_perf
    trace["perf_gate_opened"] = correctness_score > CORRECTNESS_GATE

    total_reward = r_format + r_compile + r_correct + r_perf + r_penalty
    if timed_out:
        trace["route"] = ROUTE_EVAL_TIMEOUT
    elif not compiled:
        trace["route"] = ROUTE_EVAL_COMPILE_FAIL
    elif correctness_score <= CORRECTNESS_GATE:
        trace["route"] = ROUTE_COMPILED_LOW_CORRECTNESS
    elif r_perf == 0.0:
        trace["route"] = ROUTE_CORRECT_BUT_NO_PERF
    else:
        trace["route"] = ROUTE_PERF_REWARDED
    trace["reward_total"] = total_reward
    emit_reward_trace(trace)
    return total_reward
