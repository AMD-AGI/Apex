#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_reward_fn.py — Comprehensive tests for the multi-stage gated reward pipeline.

All tests use mock eval_result dicts — no GPU, no Magpie, no API keys required.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure graders/ is importable
GRADERS_DIR = Path(__file__).parent.parent / "graders"
sys.path.insert(0, str(GRADERS_DIR))

from reward_fn import (
    CORRECTNESS_GATE,
    MIN_MEASURABLE_MS,
    R_AST_FAIL_PENALTY,
    R_COMPILE_PASS,
    R_CORRECT_WEIGHT,
    R_FORMAT_FAIL,
    R_FORMAT_PASS,
    R_NO_ANSWER,
    R_PERF_MAX,
    R_PERF_MIN,
    R_THINK_FREE_TOKS,
    R_THINK_WEIGHT,
    compute_format_reward,
    compute_gated_reward,
    compute_length_penalty,
    compute_performance_reward,
    count_tokens_approx,
    detect_runtime_hacking,
    parse_solution_tags,
    run_ast_whitelist_check,
)
from reward_backends import (
    KERNEL_BACKEND_HIP,
    KERNEL_BACKEND_TRITON,
    resolve_kernel_backend,
    run_backend_static_check,
    run_hip_static_check,
    run_triton_static_check,
)
from curriculum import get_difficulty_weights
from score import compute_reward_gated as score_compute_reward_gated, total_score


# ── Test fixtures ─────────────────────────────────────────────────────────────

VALID_TRITON_KERNEL = """\
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2, mask=mask)
"""

PASSING_EVAL_RESULT = {
    "compiled": True,
    "correctness_score": 0.98,
    "baseline_ms": 1.5,
    "optimized_ms": 0.5,
    "hacking_detected": False,
    "timed_out": False,
}

FAILING_EVAL_RESULT = {
    "compiled": True,
    "correctness_score": 0.1,
    "baseline_ms": 1.0,
    "optimized_ms": 0.8,
    "hacking_detected": False,
    "timed_out": False,
}

COMPILE_FAIL_EVAL_RESULT = {
    "compiled": False,
    "correctness_score": 0.0,
    "baseline_ms": 0.0,
    "optimized_ms": 0.0,
    "hacking_detected": False,
    "timed_out": False,
}


def _wrap_solution(kernel_code: str, think: str = "optimize block sizes") -> str:
    return f"<think>{think}</think><answer>{kernel_code}</answer>"


# ── Reward constants ──────────────────────────────────────────────────────────

class TestRewardConstants:
    def test_format_pass(self):
        assert R_FORMAT_PASS == 0.1

    def test_compile_pass(self):
        assert R_COMPILE_PASS == 0.2

    def test_correct_weight(self):
        assert R_CORRECT_WEIGHT == 0.7

    def test_perf_min(self):
        assert R_PERF_MIN == -1.0

    def test_perf_max(self):
        assert R_PERF_MAX == 3.0

    def test_think_weight(self):
        assert R_THINK_WEIGHT == -0.001

    def test_think_free_toks(self):
        assert R_THINK_FREE_TOKS == 500

    def test_correctness_gate(self):
        assert CORRECTNESS_GATE == 0.95

    def test_no_answer(self):
        assert R_NO_ANSWER == -1.0

    def test_format_fail_alias(self):
        assert R_FORMAT_FAIL == R_NO_ANSWER

    def test_ast_fail_penalty(self):
        assert R_AST_FAIL_PENALTY == -2.0

    def test_min_measurable_ms(self):
        assert MIN_MEASURABLE_MS == 0.01


# ── Tag parsing ───────────────────────────────────────────────────────────────

class TestParseSolutionTags:
    def test_both_tags(self):
        think, answer = parse_solution_tags("<think>reasoning</think><answer>code</answer>")
        assert think == "reasoning"
        assert answer == "code"

    def test_missing_answer(self):
        think, answer = parse_solution_tags("<think>reasoning</think>no tags")
        assert think == "reasoning"
        assert answer is None

    def test_missing_think(self):
        think, answer = parse_solution_tags("<answer>code</answer>")
        assert think is None
        assert answer == "code"

    def test_no_tags(self):
        think, answer = parse_solution_tags("bare code")
        assert think is None
        assert answer is None

    def test_multiline_answer(self):
        code = "import triton\n\n@triton.jit\ndef k(): pass"
        _, answer = parse_solution_tags(f"<answer>{code}</answer>")
        assert answer == code


# ── Format reward ─────────────────────────────────────────────────────────────

class TestFormatReward:
    def test_with_answer(self):
        assert compute_format_reward("code") == R_FORMAT_PASS

    def test_without_answer(self):
        assert compute_format_reward(None) == R_NO_ANSWER


# ── Length penalty ────────────────────────────────────────────────────────────

class TestLengthPenalty:
    def test_no_think(self):
        assert compute_length_penalty(None) == 0.0

    def test_short_think_no_penalty(self):
        short_text = " ".join(["word"] * 400)
        assert compute_length_penalty(short_text) == 0.0

    def test_long_think_penalized(self):
        long_text = " ".join(["word"] * 1000)
        expected = R_THINK_WEIGHT * (1000 - R_THINK_FREE_TOKS)  # -0.001 * 500 = -0.5
        assert compute_length_penalty(long_text) == pytest.approx(expected)
        assert compute_length_penalty(long_text) == pytest.approx(-0.5)

    def test_exactly_at_budget(self):
        text = " ".join(["word"] * 500)
        assert compute_length_penalty(text) == 0.0

    def test_penalty_always_nonpositive(self):
        for n in [0, 100, 500, 1000, 5000]:
            text = " ".join(["w"] * n) if n > 0 else ""
            assert compute_length_penalty(text) <= 0.0


# ── AST whitelist (Triton) ───────────────────────────────────────────────────

class TestASTWhitelistTriton:
    def test_valid_triton_kernel_passes(self):
        ok, reason = run_triton_static_check(VALID_TRITON_KERNEL)
        assert ok is True
        assert reason == ""

    def test_torch_matmul_detected(self):
        code = VALID_TRITON_KERNEL + "\nx = torch.matmul(a, b)\n"
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert "blocked_torch_attr" in reason
        assert "matmul" in reason

    def test_subprocess_import_detected(self):
        code = "import subprocess\n" + VALID_TRITON_KERNEL
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert "blocked_import" in reason

    def test_exec_call_detected(self):
        code = VALID_TRITON_KERNEL + '\nexec("print(1)")\n'
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert "blocked_builtin" in reason
        assert "exec" in reason

    def test_no_triton_jit_decorator(self):
        code = "import triton\nimport triton.language as tl\ndef kernel(): pass\n"
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert "no_triton_jit" in reason

    def test_syntax_error_not_treated_as_hack(self):
        code = "def kernel(\n"  # Invalid syntax
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert reason.startswith("SyntaxError")

    def test_hardcoded_output_detected(self):
        code = (
            "import triton\nimport triton.language as tl\nimport torch\n"
            "@triton.jit\ndef kernel(): pass\n"
            "def wrapper(): return torch.tensor([1.0, 2.0, 3.0])\n"
        )
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert "hardcoded_output" in reason

    def test_custom_allowed_imports(self):
        code = "import numpy\nimport triton\nimport triton.language as tl\n@triton.jit\ndef k(): pass\n"
        # Default: numpy not allowed
        ok, reason = run_triton_static_check(code)
        assert ok is False
        # Custom: allow numpy
        ok, reason = run_triton_static_check(code, allowed_imports=["triton", "triton.language", "numpy"])
        assert ok is True

    def test_os_module_access_blocked(self):
        code = VALID_TRITON_KERNEL + "\nimport os\nos.system('echo hack')\n"
        ok, reason = run_triton_static_check(code)
        assert ok is False
        assert "blocked" in reason.lower()


# ── Performance reward ────────────────────────────────────────────────────────

class TestPerformanceReward:
    def test_below_correctness_gate(self):
        """correctness_score=0.90 -> R_perf = 0.0 regardless of speedup."""
        assert compute_performance_reward(1.0, 0.5, 0.90) == 0.0

    def test_at_correctness_gate(self):
        """correctness_score=0.95 (exactly at gate, uses strict >) -> R_perf = 0.0."""
        assert compute_performance_reward(1.0, 0.5, 0.95) == 0.0

    def test_2x_speedup(self):
        """2x speedup with correctness=0.99 -> ~0.99 * log2(2) = ~0.99."""
        result = compute_performance_reward(1.0, 0.5, 0.99)
        expected = 0.99 * math.log2(2.0)
        assert result == pytest.approx(expected)

    def test_clipped_at_ceiling(self):
        """16x speedup -> log2(16)=4.0 but clipped to 3.0."""
        result = compute_performance_reward(16.0, 1.0, 1.0)
        assert result == pytest.approx(1.0 * R_PERF_MAX)

    def test_clipped_at_floor(self):
        """0.25x (regression) -> log2(0.25)=-2.0 but clipped to -1.0."""
        result = compute_performance_reward(1.0, 4.0, 1.0)
        assert result == pytest.approx(1.0 * R_PERF_MIN)

    def test_baseline_below_noise_floor(self):
        """baseline_ms=0.005 (5us) -> R_perf = 0.0."""
        assert compute_performance_reward(0.005, 0.001, 1.0) == 0.0

    def test_zero_optimized_ms(self):
        assert compute_performance_reward(1.0, 0.0, 1.0) == 0.0

    def test_zero_baseline_ms(self):
        assert compute_performance_reward(0.0, 1.0, 1.0) == 0.0


# ── Runtime hacking detection ────────────────────────────────────────────────

class TestRuntimeHacking:
    def test_constant_output_detected(self):
        baseline_stats = {"std": 1.0, "mean": 5.0}
        optimized_stats = {"std": 0.0, "mean": 5.0}
        assert detect_runtime_hacking(baseline_stats, optimized_stats, 0.5) is True

    def test_normal_output_passes(self):
        baseline_stats = {"std": 1.0, "mean": 5.0}
        optimized_stats = {"std": 0.8, "mean": 4.9}
        assert detect_runtime_hacking(baseline_stats, optimized_stats, 0.5) is False

    def test_suspiciously_fast_detected(self):
        baseline_stats = {"std": 1.0, "mean": 5.0}
        optimized_stats = {"std": 0.8, "mean": 4.9}
        assert detect_runtime_hacking(baseline_stats, optimized_stats, 0.0005) is True


# ── End-to-end reward (compute_gated_reward) ─────────────────────────────────

class TestComputeGatedReward:
    def test_no_answer_tag_returns_negative(self):
        """solution_str with no <answer> -> R_NO_ANSWER + R_penalty."""
        reward = compute_gated_reward(
            solution_str="no tags here",
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        assert reward == pytest.approx(R_NO_ANSWER)

    def test_answer_tag_gives_format_bonus(self):
        """solution_str with <answer> -> includes R_FORMAT_PASS."""
        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        assert reward > R_FORMAT_PASS  # at least format + compile + correctness

    def test_perfect_kernel_reward_range(self):
        """Correct + 3x fast -> reward in expected range."""
        eval_result = {
            "compiled": True,
            "correctness_score": 1.0,
            "baseline_ms": 3.0,
            "optimized_ms": 1.0,
            "hacking_detected": False,
            "timed_out": False,
        }
        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=eval_result,
        )
        # R_format(0.1) + R_compile(0.2) + R_correct(0.7*1.0) + R_perf(1.0*log2(3)~1.585)
        assert 1.5 < reward < 3.5

    def test_compiled_but_wrong_reward(self):
        """Compiled, correctness=0.1 -> low reward."""
        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=FAILING_EVAL_RESULT,
        )
        # R_format(0.1) + R_compile(0.2) + R_correct(0.7*0.1=0.07) + R_perf(0, gated)
        expected = R_FORMAT_PASS + R_COMPILE_PASS + R_CORRECT_WEIGHT * 0.1
        assert reward == pytest.approx(expected)

    def test_full_hack_reward_floor(self):
        """torch.matmul smuggling -> R_format + R_ast_cheat + R_penalty."""
        hack_code = VALID_TRITON_KERNEL + "\nx = torch.matmul(a, b)\n"
        reward = compute_gated_reward(
            solution_str=_wrap_solution(hack_code),
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        expected = R_FORMAT_PASS + R_AST_FAIL_PENALTY  # 0.1 + (-2.0) = -1.9
        assert reward == pytest.approx(expected)

    def test_syntax_error_no_hack_penalty(self):
        """SyntaxError gets R_format + 0.0 (no hack penalty) + R_penalty."""
        bad_code = "def kernel(\n"
        reward = compute_gated_reward(
            solution_str=_wrap_solution(bad_code),
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        # SyntaxError: r_ast = 0.0, not R_AST_FAIL_PENALTY
        assert reward == pytest.approx(R_FORMAT_PASS)  # 0.1 + 0.0 + 0.0

    def test_compile_fail_reward(self):
        """Compile failure gives format + 0 compile + 0 correct."""
        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=COMPILE_FAIL_EVAL_RESULT,
        )
        expected = R_FORMAT_PASS  # only format, nothing else
        assert reward == pytest.approx(expected)

    def test_hacking_detected_in_eval_result(self):
        """Runtime hacking detection triggers R_AST_FAIL_PENALTY."""
        hacked_result = dict(PASSING_EVAL_RESULT, hacking_detected=True)
        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=hacked_result,
        )
        expected = R_FORMAT_PASS + R_AST_FAIL_PENALTY  # 0.1 + (-2.0) = -1.9
        assert reward == pytest.approx(expected)

    def test_require_tags_false_no_tags(self):
        """require_tags=False with bare code -> r_format=0.1, not -1.0."""
        reward = compute_gated_reward(
            solution_str=VALID_TRITON_KERNEL,
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
            require_tags=False,
        )
        # Should process the code normally since entire content is treated as answer
        assert reward > 0.0

    def test_require_tags_false_with_tags(self):
        """require_tags=False with tags present works normally."""
        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
            require_tags=False,
        )
        assert reward > 0.0

    def test_penalty_applied_on_no_answer(self):
        """R_penalty is applied even when <answer> is missing."""
        long_think = " ".join(["word"] * 1000)
        reward = compute_gated_reward(
            solution_str=f"<think>{long_think}</think>",
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        r_penalty = R_THINK_WEIGHT * (1000 - R_THINK_FREE_TOKS)
        assert reward == pytest.approx(R_NO_ANSWER + r_penalty)

    def test_penalty_applied_on_ast_hack(self):
        """R_penalty is applied even on AST hack path."""
        long_think = " ".join(["word"] * 1000)
        hack_code = "import subprocess\n" + VALID_TRITON_KERNEL
        reward = compute_gated_reward(
            solution_str=f"<think>{long_think}</think><answer>{hack_code}</answer>",
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        r_penalty = R_THINK_WEIGHT * (1000 - R_THINK_FREE_TOKS)
        expected = R_FORMAT_PASS + R_AST_FAIL_PENALTY + r_penalty
        assert reward == pytest.approx(expected)


# ── HIP backend ──────────────────────────────────────────────────────────────

class TestHIPBackend:
    VALID_HIP_CODE = '''
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void launch_add(float* a, float* b, float* c, int n) {
    hipLaunchKernelGGL(add_kernel, dim3((n+255)/256), dim3(256), 0, 0, a, b, c, n);
}

namespace py = pybind11;
PYBIND11_MODULE(hip_add, m) {
    m.def("launch_add", &launch_add);
}
'''

    def test_valid_hip_passes(self):
        ok, reason = run_hip_static_check(self.VALID_HIP_CODE)
        assert ok is True
        assert reason == ""

    def test_missing_global_function(self):
        code = '''
#include <pybind11/pybind11.h>
void add(float* a, float* b) {}
PYBIND11_MODULE(m, m) { m.def("add", &add); }
'''
        ok, reason = run_hip_static_check(code)
        assert ok is False
        assert "no_hip_kernel" in reason

    def test_blocked_api_detected(self):
        code = self.VALID_HIP_CODE + "\nsystem(\"rm -rf /\");\n"
        ok, reason = run_hip_static_check(code)
        assert ok is False
        assert "blocked_hip_api" in reason

    def test_backend_dispatch_hip(self):
        ok, reason = run_backend_static_check(self.VALID_HIP_CODE, "hip")
        assert ok is True

    def test_backend_dispatch_triton(self):
        ok, reason = run_backend_static_check(VALID_TRITON_KERNEL, "triton")
        assert ok is True


# ── Backend resolution ────────────────────────────────────────────────────────

class TestResolveKernelBackend:
    def test_triton_direct(self):
        assert resolve_kernel_backend("triton") == "triton"

    def test_hip_direct(self):
        assert resolve_kernel_backend("hip") == "hip"

    def test_case_insensitive(self):
        assert resolve_kernel_backend("TRITON") == "triton"
        assert resolve_kernel_backend("HIP") == "hip"

    def test_spec_name_triton(self):
        assert resolve_kernel_backend("fused_moe") == "triton"
        assert resolve_kernel_backend("flash_attn_prefill") == "triton"

    def test_spec_name_hip(self):
        assert resolve_kernel_backend("gemm_bf16") == "hip"
        assert resolve_kernel_backend("all_reduce") == "hip"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            resolve_kernel_backend("unknown_kernel_type")


# ── Curriculum ────────────────────────────────────────────────────────────────

class TestCurriculum:
    def test_early_step_weights(self):
        w = get_difficulty_weights(0)
        assert w == {1: 0.8, 2: 0.2, 3: 0.0}

    def test_mid_step_weights(self):
        w = get_difficulty_weights(500)
        assert w == {1: 0.4, 2: 0.4, 3: 0.2}

    def test_late_step_weights(self):
        w = get_difficulty_weights(1000)
        assert w == {1: 0.2, 2: 0.4, 3: 0.4}

    def test_custom_warmup(self):
        w = get_difficulty_weights(50, warmup_steps=100)
        assert w == {1: 0.8, 2: 0.2, 3: 0.0}
        w = get_difficulty_weights(150, warmup_steps=100)
        assert w == {1: 0.4, 2: 0.4, 3: 0.2}
        w = get_difficulty_weights(200, warmup_steps=100)
        assert w == {1: 0.2, 2: 0.4, 3: 0.4}

    def test_weights_sum_to_one(self):
        for step in [0, 100, 500, 750, 1000, 2000]:
            w = get_difficulty_weights(step)
            assert sum(w.values()) == pytest.approx(1.0)


# ── Reward trace ──────────────────────────────────────────────────────────────

class TestRewardTrace:
    def test_trace_emitted_as_jsonl(self, tmp_path, monkeypatch):
        """Trace file is created and contains valid JSONL."""
        monkeypatch.setenv("APEX_RUN_ROOT", str(tmp_path))
        monkeypatch.setenv("REWARD_TRACE_ENABLE", "1")
        # Clear the PYTEST_CURRENT_TEST to allow tracing
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        # Reset the warned flag
        import reward_trace
        reward_trace._reward_trace_warned = False

        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )

        trace_file = tmp_path / "reward_debug" / "reward_trace.jsonl"
        assert trace_file.exists()
        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) >= 1
        trace = json.loads(lines[0])
        assert trace["trace_version"] == 1
        assert trace["reward_total"] is not None
        assert trace["route"] is not None

    def test_trace_failure_does_not_crash_reward(self, monkeypatch):
        """If tracing fails, reward computation still succeeds."""
        monkeypatch.setenv("REWARD_TRACE_ENABLE", "1")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        # Point to a non-writable path
        monkeypatch.setenv("APEX_RUN_ROOT", "/nonexistent/path/that/should/fail")

        import reward_trace
        reward_trace._reward_trace_warned = False

        reward = compute_gated_reward(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            eval_result=PASSING_EVAL_RESULT,
        )
        # Reward should still be computed correctly
        assert reward > 0.0


# ── Score wrapper ─────────────────────────────────────────────────────────────

class TestScoreWrapper:
    def test_gated_delegates(self):
        """compute_reward_gated with use_gated=True delegates to reward_fn."""
        reward = score_compute_reward_gated(
            solution_str=_wrap_solution(VALID_TRITON_KERNEL),
            ground_truth={},
            magpie_result=PASSING_EVAL_RESULT,
            use_gated=True,
        )
        assert reward > 0.0

    def test_legacy_fallback(self):
        """compute_reward_gated with use_gated=False uses legacy formula."""
        magpie_result = {
            "compiled": True,
            "correctness_score": 1.0,
            "baseline_ms": 2.0,
            "optimized_ms": 1.0,
        }
        reward = score_compute_reward_gated(
            solution_str="unused when use_gated=False",
            ground_truth={},
            magpie_result=magpie_result,
            use_gated=False,
        )
        # Legacy: compiled(20) + correct(100) + speedup_score(2.0)
        expected = total_score(True, True, 2.0)
        assert reward == pytest.approx(expected)


# ── Verification criteria from CLAUDE.md ──────────────────────────────────────

class TestVerificationCriteria:
    def test_smoke_test_from_spec(self):
        """The exact smoke test from CLAUDE.md verification criteria."""
        result = compute_gated_reward(
            solution_str=(
                "<think>optimize block sizes</think>"
                "<answer>import triton\n"
                "import triton.language as tl\n"
                "@triton.jit\n"
                "def kernel(x_ptr, out_ptr, N: tl.constexpr):\n"
                "    pid = tl.program_id(0)\n"
                "    offs = pid * 128 + tl.arange(0, 128)\n"
                "    mask = offs < N\n"
                "    x = tl.load(x_ptr + offs, mask=mask)\n"
                "    tl.store(out_ptr + offs, x * 2, mask=mask)\n"
                "</answer>"
            ),
            ground_truth={
                "pytorch_reference_code": "def baseline_fn(x): return x*2",
                "test_shapes_code": "...",
                "base_gpu_kernel_type": "triton",
            },
            eval_result={
                "compiled": True,
                "correctness_score": 0.98,
                "baseline_ms": 1.5,
                "optimized_ms": 0.5,
                "timed_out": False,
                "hacking_detected": False,
            },
        )
        assert 1.0 < result < 4.0, f"Expected reward in (1.0, 4.0), got {result}"
