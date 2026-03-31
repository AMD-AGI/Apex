# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_graders.py — Unit tests for graders/score.py, kernel_grader.py, model_grader.py.

Tests are fully offline: Magpie subprocess calls are mocked so no GPU or
Magpie installation is required.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))
from score import (
    total_score, speedup_score,
    KernelResult, ModelResult,
    parse_compare_result, parse_benchmark_result,
    parse_task_config, parse_benchmark_config,
    _extract_compiled, _extract_correct, _extract_time_ms,
    _magpie_bin, extract_tps,
    PTS_COMPILED, PTS_CORRECT,
)
import kernel_grader
import model_grader


# ── score.py — scoring formula ────────────────────────────────────────────────

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
        assert abs(s - (PTS_COMPILED + PTS_CORRECT + speedup_score(1.5))) < 1e-6

    def test_speedup_3x(self):
        s = total_score(True, True, 3.0)
        assert abs(s - (PTS_COMPILED + PTS_CORRECT + speedup_score(3.0))) < 1e-6

    def test_speedup_not_awarded_without_correct(self):
        s = total_score(True, False, 3.0)
        assert s == PTS_COMPILED

    def test_speedup_not_awarded_without_compile(self):
        s = total_score(False, False, 5.0)
        assert s == 0.0

    def test_speedup_score_clamped_at_zero(self):
        assert speedup_score(-1.0) == 0.0
        assert speedup_score(0.0)  == 0.0

    def test_speedup_score_positive(self):
        assert speedup_score(2.0) == 300.0

    def test_correct_without_compiled_still_counts(self):
        s = total_score(False, True, 0.0)
        assert s == PTS_CORRECT


# ── score.py — KernelResult dataclass ─────────────────────────────────────────

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

    def test_zero_score_for_error(self):
        r = KernelResult(task_id="t1", error="magpie failed")
        assert r.score == 0.0

    def test_to_dict_rounds_speedup(self):
        r = KernelResult(task_id="t1", compiled=True, correct=True, speedup=1.12345678)
        d = r.to_dict()
        assert d["speedup"] == 1.1235


# ── score.py — ModelResult dataclass ──────────────────────────────────────────

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

    def test_kernel_score_normalized_to_420(self):
        r = ModelResult(model_id="m1", kernel_score=420.0, e2e_throughput_ratio=1.0)
        assert r.score == pytest.approx(70.0)

    def test_e2e_ratio_below_1_is_zero_improvement(self):
        r = ModelResult(model_id="m1", kernel_score=0.0, e2e_throughput_ratio=0.5)
        assert r.score == 0.0


# ── score.py — parse_compare_result ───────────────────────────────────────────

class TestParseCompareResult:
    def test_legacy_format(self, magpie_compare_json):
        compiled, correct, speedup = parse_compare_result(magpie_compare_json)
        assert compiled is True
        assert correct  is True
        assert abs(speedup - 10.0 / 6.25) < 1e-4

    def test_native_format(self, magpie_compare_native_json):
        compiled, correct, speedup = parse_compare_result(magpie_compare_native_json)
        assert compiled is True
        assert correct  is True
        assert abs(speedup - 10.0 / 6.25) < 1e-4

    def test_compile_fail_legacy(self):
        raw = {"optimized": {"compilation": {"success": False}, "correctness": {"passed": False}}}
        compiled, correct, speedup = parse_compare_result(raw)
        assert compiled is False
        assert correct  is False
        assert speedup  == 0.0

    def test_compile_fail_native(self):
        raw = {
            "results": {
                "kernel_results": [
                    {"compile": {"success": True}, "correctness": {"passed": True}, "performance": {"avg_time_ms": 10}},
                    {"compile": {"success": False}, "correctness": {"passed": False}, "performance": {}},
                ]
            }
        }
        compiled, correct, speedup = parse_compare_result(raw)
        assert compiled is False
        assert correct  is False

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

    def test_native_zero_time_safe(self):
        raw = {
            "results": {
                "kernel_results": [
                    {"compile": {"success": True}, "correctness": {"passed": True}, "performance": {"avg_time_ms": 10}},
                    {"compile": {"success": True}, "correctness": {"passed": True}, "performance": {"avg_time_ms": 0}},
                ]
            }
        }
        _, _, speedup = parse_compare_result(raw)
        assert speedup == 0.0

    def test_flat_compiled_field(self):
        raw = {"compiled": True, "correct": True}
        compiled, correct, _ = parse_compare_result(raw)
        assert compiled is True
        assert correct is True


# ── score.py — _extract helpers ───────────────────────────────────────────────

class TestExtractHelpers:
    def test_extract_compiled_dict(self):
        assert _extract_compiled({"compile": {"success": True}}) is True
        assert _extract_compiled({"compile": {"success": False}}) is False

    def test_extract_compiled_flat(self):
        assert _extract_compiled({"compilation": {"passed": True}}) is True

    def test_extract_compiled_empty(self):
        assert _extract_compiled({}) is False

    def test_extract_correct_dict(self):
        assert _extract_correct({"correctness": {"passed": True}}) is True

    def test_extract_correct_empty(self):
        assert _extract_correct({}) is False

    def test_extract_time_avg(self):
        assert _extract_time_ms({"performance": {"avg_time_ms": 5.5}}) == 5.5

    def test_extract_time_mean(self):
        assert _extract_time_ms({"performance": {"mean_ms": 3.2}}) == 3.2

    def test_extract_time_metrics_nested(self):
        assert _extract_time_ms({"performance": {"metrics": {"avg_time_ms": 7.1}}}) == 7.1

    def test_extract_time_empty(self):
        assert _extract_time_ms({}) == 0.0

    def test_extract_time_zero_skipped(self):
        assert _extract_time_ms({"performance": {"avg_time_ms": 0, "mean_ms": 4.0}}) == 4.0


# ── score.py — extract_tps ────────────────────────────────────────────────────

class TestExtractTps:
    def test_magpie_native_format(self, magpie_benchmark_native_json):
        tps = extract_tps(magpie_benchmark_native_json)
        assert tps == 2500.0

    def test_total_token_throughput(self):
        raw = {"throughput": {"total_token_throughput": 3500.0}}
        assert extract_tps(raw) == 3500.0

    def test_output_throughput_preferred(self):
        raw = {"throughput": {"output_throughput": 2500.0, "total_token_throughput": 3500.0}}
        assert extract_tps(raw) == 2500.0

    def test_flat_tokens_per_sec(self):
        raw = {"tokens_per_sec": 1200.0}
        assert extract_tps(raw) == 1200.0

    def test_flat_tps(self):
        raw = {"tps": 800.0}
        assert extract_tps(raw) == 800.0

    def test_empty_dict_returns_zero(self):
        assert extract_tps({}) == 0.0

    def test_zero_throughput_returns_zero(self):
        raw = {"throughput": {"output_throughput": 0.0}}
        assert extract_tps(raw) == 0.0

    def test_no_throughput_key(self):
        raw = {"success": True, "framework": "sglang"}
        assert extract_tps(raw) == 0.0


# ── score.py — parse_benchmark_result ─────────────────────────────────────────

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

    def test_nested_format(self):
        raw = {"benchmark": {"baseline": {"tokens_per_sec": 800}, "optimized": {"tokens_per_sec": 1200}}}
        ratio = parse_benchmark_result(raw)
        assert abs(ratio - 1.5) < 1e-6

    def test_results_key_fallback(self):
        raw = {"results": {"baseline_tps": 500, "optimized_tps": 1000}}
        ratio = parse_benchmark_result(raw)
        assert abs(ratio - 2.0) < 1e-6

    def test_single_run_returns_zero(self, magpie_benchmark_native_json):
        ratio = parse_benchmark_result(magpie_benchmark_native_json)
        assert ratio == 0.0


# ── score.py — config parsing ─────────────────────────────────────────────────

class TestConfigParsing:
    def test_parse_task_config(self, mock_output_dir):
        cfg = parse_task_config(mock_output_dir / "task_pass" / "config.yaml")
        assert cfg["gpu"]["arch"] == "gfx950"
        assert cfg["baseline"]["path"] == "./baseline.py"
        assert cfg["optimized"]["path"] == "./solution.py"
        assert "command" in cfg["correctness"]
        assert "command" in cfg["performance"]

    def test_parse_benchmark_config(self, tmp_path):
        bm = tmp_path / "benchmark.yaml"
        bm.write_text(
            "framework: sglang\nmodel: meta-llama/Llama-3.1-8B-Instruct\n"
            "precision: fp8\nworkload:\n  input_len: 512\n  output_len: 128\n  concurrency: 32\n"
        )
        cfg = parse_benchmark_config(bm)
        assert cfg["framework"] == "sglang"
        assert cfg["workload"]["input_len"] == 512
        assert cfg["precision"] == "fp8"


# ── score.py — _magpie_bin ────────────────────────────────────────────────────

class TestMagpieBin:
    def test_returns_list(self):
        result = _magpie_bin()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_magpie_on_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAGPIE_ROOT", None)
            with patch.object(shutil, "which", return_value="/usr/bin/magpie"):
                assert _magpie_bin() == ["magpie"]

    def test_magpie_not_on_path_uses_local_or_module(self):
        with patch.object(shutil, "which", return_value=None):
            result = _magpie_bin()
            assert isinstance(result, list)
            assert len(result) >= 1


# ── kernel_grader.py — find_tasks ─────────────────────────────────────────────

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


# ── kernel_grader.py — find_solution ──────────────────────────────────────────

class TestFindSolution:
    def test_finds_solution_py(self, mock_output_dir):
        sol = kernel_grader.find_solution(mock_output_dir / "task_pass")
        assert sol is not None
        assert sol.name == "solution.py"

    def test_finds_solution_hip(self, tmp_path):
        d = tmp_path / "task"
        d.mkdir()
        (d / "solution.hip").write_text("// hip kernel\n")
        sol = kernel_grader.find_solution(d)
        assert sol.name == "solution.hip"

    def test_returns_none_if_missing(self, mock_output_dir):
        assert kernel_grader.find_solution(mock_output_dir / "task_fail") is None


# ── kernel_grader.py — _parse_config ──────────────────────────────────────────

class TestParseConfig:
    def test_parses_yaml(self, mock_output_dir):
        cfg = kernel_grader._parse_config(mock_output_dir / "task_pass" / "config.yaml")
        assert "baseline" in cfg
        assert "optimized" in cfg
        assert cfg["baseline"]["path"] == "./baseline.py"

    def test_returns_dict(self, mock_output_dir):
        cfg = kernel_grader._parse_config(mock_output_dir / "task_pass" / "config.yaml")
        assert isinstance(cfg, dict)


# ── kernel_grader.py — _detect_kernel_type ────────────────────────────────────

class TestDetectKernelType:
    def test_hip_extension(self, tmp_path):
        f = tmp_path / "kernel.hip"
        f.touch()
        assert kernel_grader._detect_kernel_type(f) == "hip"

    def test_cu_extension(self, tmp_path):
        f = tmp_path / "kernel.cu"
        f.touch()
        assert kernel_grader._detect_kernel_type(f) == "hip"

    def test_py_extension(self, tmp_path):
        f = tmp_path / "solution.py"
        f.touch()
        assert kernel_grader._detect_kernel_type(f) == "triton"


# ── kernel_grader.py — grade_task ─────────────────────────────────────────────

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

    def test_good_task_legacy_format(self, mock_output_dir, magpie_compare_json):
        task_dir = mock_output_dir / "task_pass"
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            result = kernel_grader.grade_task(task_dir)
        assert result.compiled is True
        assert result.correct  is True
        assert result.speedup  > 1.0
        assert result.score    > PTS_COMPILED + PTS_CORRECT

    def test_good_task_native_format(self, mock_output_dir, magpie_compare_native_json):
        task_dir = mock_output_dir / "task_pass"
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_native_json):
            result = kernel_grader.grade_task(task_dir)
        assert result.compiled is True
        assert result.correct  is True
        assert abs(result.speedup - 10.0 / 6.25) < 1e-4
        assert result.score > PTS_COMPILED + PTS_CORRECT

    def test_magpie_error_propagated(self, mock_output_dir):
        task_dir = mock_output_dir / "task_pass"
        with patch.object(kernel_grader, "run_magpie_compare", return_value={"error": "timeout"}):
            result = kernel_grader.grade_task(task_dir)
        assert result.error == "timeout"

    def test_missing_baseline_path_in_config(self, tmp_path):
        d = tmp_path / "bad_config"
        d.mkdir()
        (d / "solution.py").write_text("# kernel\n")
        (d / "config.yaml").write_text("gpu:\n  device: 0\n")
        result = kernel_grader.grade_task(d)
        assert "baseline.path" in result.error

    def test_trust_agent_config_param_accepted(self, mock_output_dir):
        task_dir = mock_output_dir / "task_pass"
        with patch.object(kernel_grader, "run_magpie_compare", return_value={"error": "mock"}):
            result = kernel_grader.grade_task(task_dir, trust_agent_config=True)
        assert result is not None

    def test_relative_baseline_resolved_to_repo_root(self, mock_output_dir, magpie_compare_json):
        task_dir = mock_output_dir / "task_pass"
        calls = []

        def capture_call(**kwargs):
            calls.append(kwargs)
            return magpie_compare_json

        with patch.object(kernel_grader, "run_magpie_compare", side_effect=capture_call):
            kernel_grader.grade_task(task_dir)

        assert len(calls) == 1
        bp = calls[0]["baseline_path"]
        assert Path(bp).is_absolute() or bp.startswith("./")


# ── kernel_grader.py — summarise ──────────────────────────────────────────────

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

    def test_avg_speedup_only_correct(self):
        results = [
            KernelResult("t1", compiled=True, correct=True,  speedup=3.0),
            KernelResult("t2", compiled=True, correct=True,  speedup=1.0),
            KernelResult("t3", compiled=True, correct=False, speedup=0.0),
        ]
        s = kernel_grader.summarise(results)
        assert s["avg_speedup"] == pytest.approx(2.0)

    def test_results_list_in_summary(self):
        results = [KernelResult("t1", compiled=True, correct=True, speedup=1.5)]
        s = kernel_grader.summarise(results)
        assert len(s["results"]) == 1
        assert s["results"][0]["task_id"] == "t1"

    def test_scoring_notes_present(self):
        s = kernel_grader.summarise([KernelResult("t1")])
        assert "compiled" in s["scoring_notes"]
        assert "correct" in s["scoring_notes"]


# ── model_grader.py — DEFAULT_MODELS from models.py ──────────────────────────

class TestModelGraderModels:
    def test_default_models_not_empty(self):
        assert len(model_grader.DEFAULT_MODELS) >= 6

    def test_default_models_contains_key_models(self):
        assert "meta-llama/Llama-3.1-8B-Instruct" in model_grader.DEFAULT_MODELS


# ── model_grader.py — grade_task_model ────────────────────────────────────────

class TestModelGraderGradeTask:
    def test_missing_benchmark_falls_back_to_kernel_score(
        self, mock_output_dir, magpie_compare_json
    ):
        task_dir = mock_output_dir / "task_pass"
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            result = model_grader.grade_task_model(task_dir)
        assert result.e2e_throughput_ratio == 0.0
        assert result.kernel_score > 0

    def test_legacy_precomputed_ratio(
        self, mock_output_dir, magpie_compare_json, magpie_benchmark_json,
    ):
        """Legacy format with baseline_tps/optimized_tps uses pre-computed ratio."""
        task_dir = mock_output_dir / "task_pass"
        (task_dir / "benchmark.yaml").write_text(
            "framework: sglang\nmodel: meta-llama/Llama-3.1-8B-Instruct\n"
            "precision: fp8\nworkload:\n  input_len: 512\n  output_len: 128\n  concurrency: 32\n"
        )
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json), \
             patch.object(model_grader, "run_magpie_benchmark", return_value=magpie_benchmark_json):
            result = model_grader.grade_task_model(task_dir)
        assert result.e2e_throughput_ratio == pytest.approx(1.5)
        assert result.score > 0

    def test_native_tps_two_benchmarks(
        self, mock_output_dir, magpie_compare_json, magpie_benchmark_native_json,
    ):
        """Magpie native format: grader runs two benchmarks, computes ratio."""
        task_dir = mock_output_dir / "task_pass"
        (task_dir / "benchmark.yaml").write_text(
            "framework: sglang\nmodel: meta-llama/Llama-3.1-8B-Instruct\n"
            "precision: fp8\nworkload:\n  input_len: 512\n  output_len: 128\n  concurrency: 32\n"
        )
        baseline_result = {
            "success": True, "framework": "sglang",
            "throughput": {"output_throughput": 2000.0, "total_token_throughput": 3000.0},
        }
        optimized_result = {
            "success": True, "framework": "sglang",
            "throughput": {"output_throughput": 3000.0, "total_token_throughput": 4500.0},
        }
        call_count = {"n": 0}
        def mock_benchmark(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return baseline_result
            return optimized_result

        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json), \
             patch.object(model_grader, "run_magpie_benchmark", side_effect=mock_benchmark):
            result = model_grader.grade_task_model(task_dir)

        assert call_count["n"] == 2
        assert result.e2e_throughput_ratio == pytest.approx(3000.0 / 2000.0)
        assert result.score > 0
        assert result.raw["baseline_tps"] == 2000.0
        assert result.raw["optimized_tps"] == 3000.0

    def test_benchmark_error_still_returns_kernel_score(
        self, mock_output_dir, magpie_compare_json,
    ):
        task_dir = mock_output_dir / "task_pass"
        (task_dir / "benchmark.yaml").write_text("framework: sglang\nmodel: test\n")
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json), \
             patch.object(model_grader, "run_magpie_benchmark", return_value={"error": "GPU not found"}):
            result = model_grader.grade_task_model(task_dir)
        assert "GPU not found" in result.error
        assert result.kernel_score > 0

    def test_zero_tps_baseline_error(
        self, mock_output_dir, magpie_compare_json,
    ):
        """When baseline benchmark returns 0 TPS, report error."""
        task_dir = mock_output_dir / "task_pass"
        (task_dir / "benchmark.yaml").write_text("framework: sglang\nmodel: test\n")
        zero_tps = {"success": True, "throughput": {"output_throughput": 0.0}}
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json), \
             patch.object(model_grader, "run_magpie_benchmark", return_value=zero_tps):
            result = model_grader.grade_task_model(task_dir)
        assert "0 TPS" in result.error
        assert result.kernel_score > 0

    def test_kernel_fail_short_circuits(self, mock_output_dir):
        task_dir = mock_output_dir / "task_fail"
        result = model_grader.grade_task_model(task_dir)
        assert result.error is not None


# ── model_grader.py — summarise ───────────────────────────────────────────────

class TestModelGraderSummarise:
    def test_empty_results(self):
        s = model_grader.summarise([])
        assert s["total_score"] == 0
        assert s["tasks"] == 0

    def test_aggregation(self):
        results = [
            ModelResult("m1", kernel_score=220, e2e_throughput_ratio=1.5),
            ModelResult("m2", kernel_score=120, e2e_throughput_ratio=0.0),
        ]
        s = model_grader.summarise(results)
        assert s["tasks"] == 2
        assert s["total_score"] == pytest.approx(results[0].score + results[1].score)

    def test_models_list_in_notes(self):
        s = model_grader.summarise([ModelResult("m1")])
        assert "models" in s["scoring_notes"]
        assert len(s["scoring_notes"]["models"]) >= 6


# ── Integration: full grading pipeline (mocked Magpie) ────────────────────────

class TestFullGradingPipeline:
    def test_kernel_grade_all_with_mock_output(
        self, mock_output_dir, magpie_compare_json
    ):
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            results = kernel_grader.grade_all(mock_output_dir)

        assert len(results) == 2
        pass_result = next(r for r in results if r.task_id == "task_pass")
        assert pass_result.compiled is True
        assert pass_result.correct is True

        compile_result = next(r for r in results if r.task_id == "task_compile")
        assert compile_result.error is not None

    def test_kernel_summarise_pipeline(self, mock_output_dir, magpie_compare_json):
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            results = kernel_grader.grade_all(mock_output_dir)
        summary = kernel_grader.summarise(results)
        assert summary["tasks"] == 2
        assert "total_score" in summary
        assert "results" in summary
        assert len(summary["results"]) == 2

    def test_model_grade_all_with_mock_output(
        self, mock_output_dir, magpie_compare_json
    ):
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json), \
             patch.object(model_grader, "run_magpie_benchmark", return_value={"error": "no GPU"}):
            results = model_grader.grade_all(mock_output_dir)
        assert len(results) == 2

    def test_json_serializable(self, mock_output_dir, magpie_compare_json):
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            results = kernel_grader.grade_all(mock_output_dir)
        summary = kernel_grader.summarise(results)
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)
        assert parsed["tasks"] == 2

    def test_task_filter(self, mock_output_dir, magpie_compare_json):
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            results = kernel_grader.grade_all(mock_output_dir, task_filter="pass")
        assert len(results) == 1
        assert results[0].task_id == "task_pass"

    def test_nonexistent_output_dir(self, tmp_path):
        results = kernel_grader.grade_all(tmp_path / "nonexistent")
        assert results == []

    def test_score_consistency(self, magpie_compare_json):
        compiled, correct, speedup = parse_compare_result(magpie_compare_json)
        result = KernelResult(
            task_id="test",
            compiled=compiled,
            correct=correct,
            speedup=speedup,
        )
        expected = total_score(compiled, correct, speedup)
        assert result.score == pytest.approx(expected)


# ── kernel_grader.py — _try_magpie_perf_measurement ──────────────────────────

class TestTryMagpiePerfMeasurement:
    """Tests for _try_magpie_perf_measurement helper used by library_test and accordo."""

    def test_injects_speedup_on_success(self, tmp_path, magpie_compare_json):
        sol = tmp_path / "solution.py"
        sol.write_text("# kernel\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")

        raw = {"compiled": True, "correct": True}
        with patch.object(kernel_grader, "run_magpie_compare", return_value=magpie_compare_json):
            kernel_grader._try_magpie_perf_measurement(
                raw, str(baseline), str(sol), tmp_path, 300, sol,
            )
        assert "_magpie_speedup" in raw
        assert raw["_magpie_speedup"] > 1.0
        assert raw["_magpie_perf_source"] == "magpie_compare"

    def test_no_inject_on_magpie_error(self, tmp_path):
        sol = tmp_path / "solution.py"
        sol.write_text("# kernel\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")

        raw = {"compiled": True, "correct": True}
        with patch.object(kernel_grader, "run_magpie_compare", return_value={"error": "timeout"}):
            kernel_grader._try_magpie_perf_measurement(
                raw, str(baseline), str(sol), tmp_path, 300, sol,
            )
        assert "_magpie_speedup" not in raw

    def test_no_inject_on_magpie_exception(self, tmp_path):
        sol = tmp_path / "solution.py"
        sol.write_text("# kernel\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")

        raw = {"compiled": True, "correct": True}
        with patch.object(kernel_grader, "run_magpie_compare", side_effect=RuntimeError("boom")):
            kernel_grader._try_magpie_perf_measurement(
                raw, str(baseline), str(sol), tmp_path, 300, sol,
            )
        assert "_magpie_speedup" not in raw

    def test_skips_when_baseline_missing(self, tmp_path):
        sol = tmp_path / "solution.py"
        sol.write_text("# kernel\n")

        raw = {"compiled": True, "correct": True}
        kernel_grader._try_magpie_perf_measurement(
            raw, "/nonexistent/baseline.py", str(sol), tmp_path, 300, sol,
        )
        assert "_magpie_speedup" not in raw

    def test_skips_when_baseline_path_empty(self, tmp_path):
        sol = tmp_path / "solution.py"
        sol.write_text("# kernel\n")

        raw = {"compiled": True, "correct": True}
        kernel_grader._try_magpie_perf_measurement(
            raw, "", str(sol), tmp_path, 300, sol,
        )
        assert "_magpie_speedup" not in raw

    def test_no_inject_when_magpie_returns_zero_speedup(self, tmp_path):
        sol = tmp_path / "solution.py"
        sol.write_text("# kernel\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")

        raw = {"compiled": True, "correct": True}
        zero_result = {"results": {"kernel_results": []}}
        with patch.object(kernel_grader, "run_magpie_compare", return_value=zero_result):
            kernel_grader._try_magpie_perf_measurement(
                raw, str(baseline), str(sol), tmp_path, 300, sol,
            )
        assert "_magpie_speedup" not in raw


# ── kernel_grader.py — library_test mode with Magpie perf ────────────────────

class TestLibraryTestMagpieIntegration:
    """Tests that library_test mode calls Magpie for perf measurement after correctness passes."""

    def _make_library_test_task(self, tmp_path):
        d = tmp_path / "lib_task"
        d.mkdir()
        (d / "solution.py").write_text("# optimized\n")
        (d / "baseline.py").write_text("# baseline\n")
        (d / "config.yaml").write_text(
            "gpu:\n  device: 0\n"
            "baseline:\n  path: ./baseline.py\n"
            "optimized:\n  path: ./solution.py\n"
            "correctness:\n  mode: library_test\n"
            "  unit_test_command: python -m pytest tests/\n"
            "  working_directory: .\n"
            "performance:\n  command: python bench.py\n"
        )
        return d

    def test_magpie_called_when_library_test_passes(self, tmp_path, magpie_compare_json):
        task_dir = self._make_library_test_task(tmp_path)
        magpie_calls = []

        def mock_magpie(**kwargs):
            magpie_calls.append(kwargs)
            return magpie_compare_json

        with patch.object(kernel_grader, "_run_library_test", return_value={
            "compiled": True, "correct": True, "_correctness_mode": "library_test",
        }), patch.object(kernel_grader, "run_magpie_compare", side_effect=mock_magpie), \
            patch.object(kernel_grader, "_measure_speedup", return_value=1.0):
            result = kernel_grader.grade_task(task_dir, isolate_caches=False, trust_agent_config=True)

        assert len(magpie_calls) == 1
        assert result.compiled is True
        assert result.correct is True

    def test_magpie_not_called_when_library_test_fails(self, tmp_path):
        task_dir = self._make_library_test_task(tmp_path)

        with patch.object(kernel_grader, "_run_library_test", return_value={
            "compiled": True, "correct": False, "_correctness_mode": "library_test",
            "error": "2 tests failed",
        }), patch.object(kernel_grader, "run_magpie_compare") as mock_magpie, \
            patch.object(kernel_grader, "_measure_speedup", return_value=0.0):
            result = kernel_grader.grade_task(task_dir, isolate_caches=False, trust_agent_config=True)

        mock_magpie.assert_not_called()
        assert result.correct is False


# ── kernel_grader.py — accordo mode with Magpie perf ─────────────────────────

class TestAccordoMagpieIntegration:
    """Tests that accordo mode calls Magpie for perf measurement after correctness passes."""

    def _make_accordo_task(self, tmp_path):
        d = tmp_path / "acc_task"
        d.mkdir()
        (d / "solution.hip").write_text("// optimized\n")
        (d / "baseline.hip").write_text("// baseline\n")
        (d / "config.yaml").write_text(
            "gpu:\n  device: 0\n"
            "baseline:\n  path: ./baseline.hip\n"
            "optimized:\n  path: ./solution.hip\n"
            "correctness:\n  mode: accordo\n"
            "  accordo:\n"
            "    kernel_name: my_kernel\n"
            "    reference_binary: /path/to/ref\n"
            "    optimized_binary: /path/to/opt\n"
            "    tolerance: 0.001\n"
            "    timeout_seconds: 60\n"
            "performance:\n  command: python bench.py\n"
        )
        return d

    def test_magpie_called_when_accordo_passes(self, tmp_path, magpie_compare_json):
        task_dir = self._make_accordo_task(tmp_path)
        magpie_calls = []

        def mock_magpie(**kwargs):
            magpie_calls.append(kwargs)
            return magpie_compare_json

        with patch.object(kernel_grader, "_run_accordo_check", return_value={
            "compiled": True, "correct": True, "_correctness_mode": "accordo",
        }), patch.object(kernel_grader, "run_magpie_compare", side_effect=mock_magpie), \
            patch.object(kernel_grader, "_measure_speedup", return_value=1.0):
            result = kernel_grader.grade_task(task_dir, isolate_caches=False, trust_agent_config=True)

        assert len(magpie_calls) == 1
        assert result.compiled is True
        assert result.correct is True

    def test_magpie_not_called_when_accordo_fails(self, tmp_path):
        task_dir = self._make_accordo_task(tmp_path)

        with patch.object(kernel_grader, "_run_accordo_check", return_value={
            "compiled": True, "correct": False, "_correctness_mode": "accordo",
            "error": "3 mismatches",
        }), patch.object(kernel_grader, "run_magpie_compare") as mock_magpie, \
            patch.object(kernel_grader, "_measure_speedup", return_value=0.0):
            result = kernel_grader.grade_task(task_dir, isolate_caches=False, trust_agent_config=True)

        mock_magpie.assert_not_called()
        assert result.correct is False

    def test_accordo_timeout_cap_is_900(self):
        """Verify the _MAX_ACCORDO_TIMEOUT constant is 900 (15 min)."""
        raw = {}
        with patch("shutil.which", return_value=None):
            result = kernel_grader._run_accordo_check(
                {"kernel_name": "k", "reference_binary": "r", "optimized_binary": "o",
                 "timeout_seconds": 9999},
                "/tmp",
            )
        assert "accordo" in result.get("error", "") or result.get("compiled") is False
