# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for reflector.py — reflection prompt generation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))
from pipeline.reflector import (
    reflect, should_continue, _get_hints, _read_solution,
    _parse_rocprof_metrics, _format_performance_scorecard,
)
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


class TestParseRocprofMetrics:
    SAMPLE_PROFILE = (
        "Memory Bandwidth utilization: 72.3 %\n"
        "Compute utilization: 15.8 %\n"
        "Occupancy: 8 waves/CU\n"
        "Instruction mix: VMEM 45%, VALU 30%, MFMA 10%, LDS 15%\n"
    )

    def test_extracts_bandwidth(self):
        metrics = _parse_rocprof_metrics(self.SAMPLE_PROFILE)
        assert abs(metrics["bandwidth_pct"] - 72.3) < 0.1

    def test_extracts_compute(self):
        metrics = _parse_rocprof_metrics(self.SAMPLE_PROFILE)
        assert abs(metrics["compute_pct"] - 15.8) < 0.1

    def test_extracts_occupancy(self):
        metrics = _parse_rocprof_metrics(self.SAMPLE_PROFILE)
        assert metrics["occupancy"] == 8.0

    def test_extracts_instructions(self):
        metrics = _parse_rocprof_metrics(self.SAMPLE_PROFILE)
        assert "VMEM" in metrics["top_instructions"]
        assert "MFMA" in metrics["top_instructions"]
        assert "LDS" in metrics["top_instructions"]

    def test_memory_bound_recommendation(self):
        metrics = _parse_rocprof_metrics(self.SAMPLE_PROFILE)
        assert "memory-bound" in metrics["recommendation"]

    def test_empty_input(self):
        assert _parse_rocprof_metrics("") == {}


class TestFormatPerformanceScorecard:
    def test_produces_scorecard(self):
        profile = "HBM BW: 65.0 %\nMFMA utilization: 30.0 %\nOccupancy: 12\n"
        scorecard = _format_performance_scorecard(profile)
        assert "Performance Scorecard" in scorecard
        assert "65%" in scorecard
        assert "30%" in scorecard

    def test_empty_returns_empty(self):
        assert _format_performance_scorecard("") == ""
