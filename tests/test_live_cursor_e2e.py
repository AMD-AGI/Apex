#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Live E2E test: Cursor agent optimizes rms_norm kernel.

Requires:
  - GPU available
  - cursor-agent CLI installed and authenticated
  - Magpie available
  - MAGPIE_ROOT set

Run with: pytest tests/test_live_cursor_e2e.py -m live -v -s
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.live


def _find_rms_norm_baseline() -> Path:
    candidates = [
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "triton" / "normalization" / "rmsnorm.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "triton" / "norm" / "rms_norm.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "rmsnorm.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "op_tests" / "triton" / "test_rmsnorm.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("rms_norm baseline not found")


@pytest.fixture(scope="module")
def results_dir():
    d = Path(tempfile.mkdtemp(prefix="apex_live_cursor_"))
    yield d


class TestCursorOptimizeRmsNorm:
    def test_cursor_optimize_rms_norm(self, results_dir):
        baseline = _find_rms_norm_baseline()

        cmd = [
            sys.executable, str(REPO_ROOT / "workload_optimizer.py"),
            "optimize-kernel",
            "-r", str(results_dir),
            "--kernel", str(baseline),
            "--kernel-name", "rms_norm",
            "--kernel-type", "triton",
            "--max-iterations", "1",
            "--max-turns", "10",
            "--agent-backend", "cursor",
        ]

        env = os.environ.copy()
        env["MAGPIE_ROOT"] = os.environ.get("MAGPIE_ROOT", str(REPO_ROOT.parent / "Magpie"))

        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=600,
        )
        print(f"  stdout (last 2000 chars): {result.stdout[-2000:]}")
        if result.stderr:
            print(f"  stderr (last 1000 chars): {result.stderr[-1000:]}")

        task_dirs = list(results_dir.glob("output/*/solution.py"))
        assert len(task_dirs) >= 1, (
            f"No solution.py found under {results_dir}/output/. "
            f"Return code: {result.returncode}"
        )
        task_dir = task_dirs[0].parent

        solution = task_dir / "solution.py"
        assert solution.exists(), "solution.py not created"

        pipeline_state = results_dir / "pipeline_state.json"
        if pipeline_state.exists():
            state = json.loads(pipeline_state.read_text())
            opt_results = state.get("optimization_results", [])
            if opt_results:
                r = opt_results[0]
                assert r.get("compiled") is not None, "compiled field missing"
                print(f"  Grade: compiled={r.get('compiled')} correct={r.get('correct')} "
                      f"speedup={r.get('speedup')} rl_reward={r.get('rl_reward')}")

        print(f"  Live Cursor E2E test completed. Results in: {results_dir}")
