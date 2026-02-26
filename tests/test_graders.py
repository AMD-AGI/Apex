"""
test_graders.py — Unit tests for graders/score.py, kernel_grader.py, model_grader.py.

Tests are fully offline: Magpie subprocess calls are mocked so no GPU or
Magpie installation is required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))
from score import (
    total_score, speedup_score,
    KernelResult, ModelResult,
    parse_compare_result, parse_benchmark_result,
    PTS_COMPILED, PTS_CORRECT,
)
import kernel_grader
import model_grader


# ── score.py ──────────────────────────────────────────────────────────────────

class TestScoringFormula:
    def test_all_zeros(self):
        assert total_score(False, False, 0.0) == 0.0

    def test_compiled_only(self):
        assert total_score(True, False, 0.0) == PTS_COMPILED

    def test_compiled_and_correct(self):
        assert total_score(True, True, 0.0) == PTS_COMPILED + PTS_CORRECT

    def test_full_score_1x_speedup(self):
        s = total_score(True, True, 1.0)
        assert s == PTS_COMPILED + PTS_CORRECT + 100.0

    def test_speedup_1_5x(self):
        s = total_score(True, True, 1.5)
        assert abs(s - (PTS_COMPILED + PTS_CORRECT + 150.0)) < 1e-6

    def test_speedup_not_awarded_without_correct(self):
        # speedup should not count if kernel fails correctness
        s = total_score(True, False, 3.0)
        assert s == PTS_COMPILED

    def test_speedup_not_awarded_without_compile(self):
        s = total_score(False, False, 5.0)
        assert s == 0.0

    def test_speedup_score_clamped_at_zero(self):
        assert speedup_score(-1.0) == 0.0
        assert speedup_score(0.0)  == 0.0

    def test_speedup_score_positive(self):
        assert speedup_score(2.0) == 200.0


class TestKernelResultDataclass:
    def test_score_computed_on_init(self):
        r = KernelResult(task_id="t1", compiled=True, correct=True, speedup=1.5)
        assert r.score == total_score(True, True, 1.5)

    def test_to_dict_shape(self):
        r = KernelResult(task_id="t1", compiled=True, correct=False, speedup=0.0)
        d = r.to_dict()
        assert set(d) == {"task_id", "compiled", "correct", "speedup", "score", "error"}
        assert d["task_id"] == "t1"

    def test_error_field_none_by_default(self):
        r = KernelResult(task_id="t1")
        assert r.error is None
        assert r.compiled is False


class TestModelResultDataclass:
    def test_score_above_zero_for_good_result(self):
        r = ModelResult(model_id="m1", kernel_score=220.0, e2e_throughput_ratio=1.5)
        assert r.score > 0

    def test_score_zero_for_no_result(self):
        r = ModelResult(model_id="m1", kernel_score=0.0, e2e_throughput_ratio=0.0)
        assert r.score == 0.0

    def test_to_dict_shape(self):
        r = ModelResult(model_id="m1", kernel_score=100.0, e2e_throughput_ratio=1.2)
        d = r.to_dict()
        assert "model_id" in d and "kernel_score" in d and "e2e_throughput_ratio" in d


# ── score.py parse helpers ─────────────────────────────────────────────────────

class TestParseCompareResult:
    def test_full_result(self, magpie_compare_json):
        compiled, correct, speedup = parse_compare_result(magpie_compare_json)
        assert compiled is True
        assert correct  is True
        assert abs(speedup - 10.0 / 6.25) < 1e-4

    def test_compile_fail(self):
        raw = {"optimized": {"compilation": {"success": False}, "correctness": {"passed": False}}}
        compiled, correct, speedup = parse_compare_result(raw)
        assert compiled is False
        assert correct  is False
        assert speedup  == 0.0

    def test_empty_dict_is_graceful(self):
        compiled, correct, speedup = parse_compare_result({})
        assert isinstance(compiled, bool)
        assert speedup >= 0.0

    def test_zero_optimized_time_safe(self):
        raw = {
            "optimized": {
                "compilation": {"success": True},
                "correctness": {"passed": True},
                "performance": {"baseline_ms": 10.0, "optimized_ms": 0.0},
            }
        }
        _, _, speedup = parse_compare_result(raw)
        assert speedup == 0.0


class TestParseBenchmarkResult:
    def test_good_result(self, magpie_benchmark_json):
        ratio = parse_benchmark_result(magpie_benchmark_json)
        assert abs(ratio - 1.5) < 1e-6

    def test_missing_baseline(self):
        ratio = parse_benchmark_result({})
        assert ratio == 0.0

    def test_zero_baseline(self):
        ratio = parse_benchmark_result({"benchmark": {"baseline_tps": 0.0, "optimized_tps": 100.0}})
        assert ratio == 0.0


# ── kernel_grader.py ──────────────────────────────────────────────────────────

class TestFindTasks:
    def test_discovers_solution_dirs(self, mock_output_dir):
        tasks = kernel_grader.find_tasks(mock_output_dir)
        task_names = {t.name for t in tasks}
        assert "task_pass" in task_names
        assert "task_compile" in task_names

    def test_excludes_dirs_without_solution(self, mock_output_dir):
        tasks = kernel_grader.find_tasks(mock_output_dir)
        task_names = {t.name for t in tasks}
        assert "task_fail" not in task_names

    def test_empty_output_dir(self, tmp_path):
        assert kernel_grader.find_tasks(tmp_path) == []


class TestGradeTask:
    def test_missing_solution_returns_error(self, mock_output_dir):
        task_dir = mock_output_dir / "task_fail"
        result = kernel_grader.grade_task(task_dir)
        assert result.error is not None
        assert result.compiled is False

    def test_missing_config_returns_error(self, mock_output_dir):
        task_dir = mock_output_dir / "task_compile"
        result = kernel_grader.grade_task(task_dir)
        assert result.error is not None

    def test_good_task_calls_magpie(self, mock_output_dir, magpie_compare_json):
        task_dir = mock_output_dir / "task_pass"
        with patch("kernel_grader.run_magpie", return_value=magpie_compare_json):
            result = kernel_grader.grade_task(task_dir)
        assert result.compiled is True
        assert result.correct  is True
        assert result.speedup  > 1.0
        assert result.score    > PTS_COMPILED + PTS_CORRECT

    def test_magpie_error_propagated(self, mock_output_dir):
        task_dir = mock_output_dir / "task_pass"
        with patch("kernel_grader.run_magpie", return_value={"error": "timeout"}):
            result = kernel_grader.grade_task(task_dir)
        assert result.error == "timeout"


class TestKernelGraderSummarise:
    def test_empty_results(self):
        s = kernel_grader.summarise([])
        assert s["total_score"] == 0
        assert s["tasks"] == 0

    def test_aggregation(self):
        results = [
            KernelResult("t1", compiled=True, correct=True,  speedup=2.0),
            KernelResult("t2", compiled=True, correct=False, speedup=0.0),
        ]
        s = kernel_grader.summarise(results)
        assert s["tasks"]    == 2
        assert s["compiled"] == 2
        assert s["correct"]  == 1
        assert s["total_score"] == pytest.approx(
            results[0].score + results[1].score
        )


# ── model_grader.py ───────────────────────────────────────────────────────────

class TestModelGraderGradeTask:
    def test_missing_benchmark_falls_back_to_kernel_score(
        self, mock_output_dir, magpie_compare_json
    ):
        task_dir = mock_output_dir / "task_pass"
        with patch("model_grader.run_magpie", return_value=magpie_compare_json):
            # grade_task calls grade_task (kernel) internally via kernel_grader
            with patch("kernel_grader.run_magpie", return_value=magpie_compare_json):
                result = model_grader.grade_task_model(task_dir)
        # No benchmark.yaml → e2e_throughput_ratio == 0
        assert result.e2e_throughput_ratio == 0.0
        assert result.kernel_score > 0

    def test_full_model_grade(
        self, mock_output_dir, magpie_compare_json, magpie_benchmark_json, tmp_path
    ):
        task_dir = mock_output_dir / "task_pass"
        (task_dir / "benchmark.yaml").write_text("framework: sglang\n")
        with patch("kernel_grader.run_magpie", return_value=magpie_compare_json), \
             patch("model_grader.run_magpie",  return_value=magpie_benchmark_json):
            result = model_grader.grade_task_model(task_dir)
        assert result.e2e_throughput_ratio == pytest.approx(1.5)
        assert result.score > 0
