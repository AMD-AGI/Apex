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
      task_pass    — solution + config.yaml (should score full points)
      task_compile — solution exists, no config.yaml (error, but compiled)
      task_fail    — no solution file (should fail gracefully)
    """
    for task_id in ("task_pass", "task_compile", "task_fail"):
        d = tmp_path / task_id
        d.mkdir()

    # task_pass: has both solution and config
    (tmp_path / "task_pass" / "solution.py").write_text("# optimized kernel\n")
    (tmp_path / "task_pass" / "config.yaml").write_text(
        "gpu:\n  device: 0\nbaseline:\n  path: ./baseline.py\n"
        "optimized:\n  path: ./solution.py\n"
    )

    # task_compile: solution only, no config
    (tmp_path / "task_compile" / "solution.py").write_text("# optimized kernel\n")

    # task_fail: no solution
    (tmp_path / "task_fail" / "notes.txt").write_text("no solution written\n")

    return tmp_path


@pytest.fixture()
def magpie_compare_json():
    """Canonical Magpie compare JSON output (good result)."""
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
def magpie_benchmark_json():
    """Canonical Magpie benchmark JSON output."""
    return {
        "benchmark": {
            "baseline_tps":  1000.0,
            "optimized_tps": 1500.0,
        }
    }
