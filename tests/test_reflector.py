# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for reflector.py — reflection prompt generation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))
from reflector import reflect, should_continue, _get_hints, _read_solution
from score import KernelResult


class TestReflect:
    def test_compile_failure(self, tmp_path):
        (tmp_path / "solution.py").write_text("def kernel(): pass")
        kr = KernelResult(task_id="test", compiled=False, error="SyntaxError: unexpected EOF")
        prompt = reflect(kr, tmp_path, iteration=1, kernel_type="rms_norm")
        assert "Compilation Failure" in prompt
        assert "SyntaxError" in prompt
        assert "def kernel()" in prompt

    def test_correctness_failure(self, tmp_path):
        (tmp_path / "solution.py").write_text("def kernel(): return 42")
        kr = KernelResult(task_id="test", compiled=True, correct=False, error="mismatch at index 0")
        prompt = reflect(kr, tmp_path, iteration=2, kernel_type="gemm_bf16")
        assert "Correctness Failure" in prompt
        assert "mismatch" in prompt

    def test_perf_regression(self, tmp_path):
        (tmp_path / "solution.py").write_text("import numpy as np")
        kr = KernelResult(
            task_id="test", compiled=True, correct=True, speedup=0.7,
            raw={"baseline_ms": 1.0, "optimized_ms": 1.43},
        )
        prompt = reflect(kr, tmp_path, iteration=1, kernel_type="flash_attn_prefill")
        assert "Performance Regression" in prompt
        assert "0.70x" in prompt

    def test_improvement_opportunity(self, tmp_path):
        (tmp_path / "solution.py").write_text("# good solution")
        kr = KernelResult(
            task_id="test", compiled=True, correct=True, speedup=1.5,
        )
        prompt = reflect(kr, tmp_path, iteration=2, kernel_type="fused_moe")
        assert "Improvement Opportunity" in prompt
        assert "1.50x" in prompt

    def test_no_solution_file(self, tmp_path):
        kr = KernelResult(task_id="test", compiled=False, error="no file")
        prompt = reflect(kr, tmp_path, iteration=1)
        assert "solution file not found" in prompt

    def test_hip_solution(self, tmp_path):
        (tmp_path / "solution.hip").write_text("__global__ void kern() {}")
        kr = KernelResult(task_id="test", compiled=False, error="compile error")
        prompt = reflect(kr, tmp_path, iteration=1)
        assert "__global__" in prompt

    def test_iteration_number_in_prompt(self, tmp_path):
        (tmp_path / "solution.py").write_text("pass")
        kr = KernelResult(task_id="test", compiled=False, error="err")
        p1 = reflect(kr, tmp_path, iteration=1)
        p3 = reflect(kr, tmp_path, iteration=3)
        assert "Iteration 1" in p1
        assert "Iteration 3" in p3


class TestShouldContinue:
    def test_stop_at_max_iterations(self):
        kr = KernelResult(task_id="t", compiled=True, correct=True, speedup=1.0)
        assert not should_continue(kr, iteration=3, max_iterations=3)

    def test_stop_at_threshold(self):
        kr = KernelResult(task_id="t", compiled=True, correct=True, speedup=2.5)
        assert not should_continue(kr, iteration=1, max_iterations=5, score_threshold=300.0)

    def test_continue_below_threshold(self):
        kr = KernelResult(task_id="t", compiled=True, correct=True, speedup=0.5)
        assert should_continue(kr, iteration=1, max_iterations=5, score_threshold=300.0)

    def test_continue_on_compile_fail(self):
        kr = KernelResult(task_id="t", compiled=False)
        assert should_continue(kr, iteration=1, max_iterations=3)

    def test_continue_on_correctness_fail(self):
        kr = KernelResult(task_id="t", compiled=True, correct=False)
        assert should_continue(kr, iteration=2, max_iterations=5)


class TestGetHints:
    def test_known_kernel(self):
        hints = _get_hints("flash_attn_prefill")
        assert "Flash attention" in hints

    def test_fused_moe_hints(self):
        hints = _get_hints("fused_moe")
        assert "MoE" in hints

    def test_unknown_kernel(self):
        hints = _get_hints("unknown_kernel")
        assert "source-finder" in hints


class TestReadSolution:
    def test_reads_py(self, tmp_path):
        (tmp_path / "solution.py").write_text("import torch")
        assert "import torch" in _read_solution(tmp_path)

    def test_reads_hip(self, tmp_path):
        (tmp_path / "solution.hip").write_text("__global__ void k() {}")
        assert "__global__" in _read_solution(tmp_path)

    def test_missing(self, tmp_path):
        result = _read_solution(tmp_path)
        assert "not found" in result
