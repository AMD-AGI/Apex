"""
conftest.py — Shared pytest fixtures for the RL kernel-optimization env tests.
"""

import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


# ── Filesystem fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def mock_output_dir(tmp_path):
    """
    A temporary output/ directory with three mock tasks:
      task_pass    — solution + config.yaml + baseline (should score full points)
      task_compile — solution exists, no config.yaml (error)
      task_fail    — no solution file (should fail gracefully)
    """
    for task_id in ("task_pass", "task_compile", "task_fail"):
        d = tmp_path / task_id
        d.mkdir()

    # task_pass: has solution, baseline, and config
    (tmp_path / "task_pass" / "solution.py").write_text("# optimized kernel\n")
    (tmp_path / "task_pass" / "baseline.py").write_text("# baseline kernel\n")
    (tmp_path / "task_pass" / "config.yaml").write_text(
        "gpu:\n  device: 0\n  arch: gfx950\n"
        "baseline:\n  path: ./baseline.py\n"
        "optimized:\n  path: ./solution.py\n"
        "correctness:\n  command: pytest tests/ -x\n"
        "performance:\n  command: python bench.py\n  iterations: 100\n"
    )

    # task_compile: solution only, no config
    (tmp_path / "task_compile" / "solution.py").write_text("# optimized kernel\n")

    # task_fail: no solution
    (tmp_path / "task_fail" / "notes.txt").write_text("no solution written\n")

    return tmp_path


@pytest.fixture()
def magpie_compare_json():
    """Canonical Magpie compare JSON output — legacy format."""
    return {
        "optimized": {
            "compilation": {"success": True},
            "correctness": {"passed": True},
            "performance": {
                "baseline_ms":  10.0,
                "optimized_ms":  6.25,
            },
        }
    }


@pytest.fixture()
def magpie_compare_native_json():
    """Magpie compare JSON output — native kernel_results format."""
    return {
        "mode": "compare",
        "timestamp": "20260225_120000",
        "results": {
            "kernel_results": [
                {
                    "kernel_id": "baseline",
                    "compile": {"success": True},
                    "correctness": {"passed": True},
                    "performance": {"avg_time_ms": 10.0},
                },
                {
                    "kernel_id": "optimized",
                    "compile": {"success": True},
                    "correctness": {"passed": True},
                    "performance": {"avg_time_ms": 6.25},
                },
            ],
            "comparison_metrics": {},
            "winner": "optimized",
            "rankings": ["optimized", "baseline"],
            "summary": "optimized is 1.6x faster",
        },
    }


@pytest.fixture()
def magpie_benchmark_json():
    """Legacy pre-computed comparison benchmark result."""
    return {
        "benchmark": {
            "baseline_tps":  1000.0,
            "optimized_tps": 1500.0,
        }
    }


@pytest.fixture()
def magpie_benchmark_native_json():
    """Magpie BenchmarkResult.to_dict() — single-run format (actual output)."""
    return {
        "success": True,
        "framework": "sglang",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "throughput": {
            "request_throughput": 50.0,
            "output_throughput": 2500.0,
            "total_token_throughput": 3500.0,
            "completed_requests": 200,
            "total_input_tokens": 102400,
            "total_output_tokens": 25600,
            "duration_seconds": 4.0,
        },
        "latency": {
            "ttft": {"mean_ms": 25.0, "median_ms": 22.0, "p99_ms": 80.0},
            "tpot": {"mean_ms": 5.0, "median_ms": 4.5, "p99_ms": 12.0},
        },
        "execution_time": 120.5,
    }
