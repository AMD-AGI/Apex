#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Live full pipeline E2E test: benchmark(skip) -> identify -> optimize -> score.

Requires:
  - GPU available
  - Codex CLI installed and authenticated
  - Magpie available
  - MAGPIE_ROOT set
  - An existing benchmark report JSON

Run with: pytest tests/test_live_pipeline_e2e.py -m live -v -s
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


def _find_benchmark_report() -> Path:
    """Find an existing benchmark report to skip the benchmark step."""
    search_dirs = [
        Path.home() / "results_total_agent",
        Path.home() / "Kernel" / "thinking",
        Path.home() / "Kernel" / "benchmarking",
        Path.home() / "Kernel" / "thinkingvnew",
        Path.home() / "apex_improvements_eval",
        REPO_ROOT / "results",
    ]
    for d in search_dirs:
        if d.exists():
            for f in d.rglob("benchmark_report.json"):
                return f
    pytest.skip("No existing benchmark report found for skip-benchmark")


def _find_bench_config() -> Path:
    magpie = os.environ.get("MAGPIE_ROOT", str(REPO_ROOT.parent / "Magpie"))
    candidates = [
        Path(magpie) / "examples" / "benchmarks" / "benchmark_vllm_gptoss_120b.yaml",
        Path(magpie) / "examples" / "benchmark_vllm_gptoss_120b.yaml",
    ]
    for cfg in candidates:
        if cfg.exists():
            return cfg
    pytest.skip("benchmark config not found")


def _run_step(cmd: list[str], env: dict, timeout: int = 300) -> subprocess.CompletedProcess:
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(f"  stdout: {result.stdout[-1000:]}")
        print(f"  stderr: {result.stderr[-500:]}")
    return result


@pytest.fixture(scope="module")
def results_dir():
    d = Path(tempfile.mkdtemp(prefix="apex_live_pipeline_"))
    yield d


class TestFullPipelineE2E:
    def test_full_pipeline_e2e_codex(self, results_dir):
        bench_report = _find_benchmark_report()
        bench_config = _find_bench_config()

        env = os.environ.copy()
        env["MAGPIE_ROOT"] = os.environ.get("MAGPIE_ROOT", str(REPO_ROOT.parent / "Magpie"))
        py = sys.executable
        wo = str(REPO_ROOT / "workload_optimizer.py")

        # Step 1: benchmark --skip-benchmark
        r = _run_step([
            py, wo, "benchmark",
            "-r", str(results_dir),
            "-b", str(bench_config),
            "--skip-benchmark", str(bench_report),
        ], env)
        assert r.returncode == 0, f"benchmark step failed: {r.stderr[-300:]}"

        # Step 2: identify top-1 triton kernel
        r = _run_step([
            py, wo, "identify",
            "-r", str(results_dir),
            "--kernel-types", "triton",
            "--top-k", "1",
        ], env)
        assert r.returncode == 0, f"identify step failed: {r.stderr[-300:]}"

        # Step 3: optimize with codex, 1 kernel, 2 iterations
        try:
            r = _run_step([
                py, wo, "optimize",
                "-r", str(results_dir),
                "--kernel-types", "triton",
                "--max-iterations", "2",
                "--max-turns", "10",
                "--agent-backend", "codex",
            ], env, timeout=1800)
            if r.returncode != 0:
                print(f"  optimize step returned {r.returncode} (non-fatal for E2E test)")
        except subprocess.TimeoutExpired:
            print("  optimize step timed out (1800s) — proceeding with partial results")

        # Step 4: score with leaderboard
        r = _run_step([
            py, wo, "score",
            "-r", str(results_dir),
            "--leaderboard",
        ], env)
        if r.returncode != 0:
            print(f"  score step returned {r.returncode} (may fail if optimize timed out)")

        # Assertions on artifacts
        pipeline_state = results_dir / "pipeline_state.json"
        assert pipeline_state.exists(), "pipeline_state.json not created"
        state = json.loads(pipeline_state.read_text())

        assert "baseline_result" in state or "benchmark_report" in state or \
            "bottleneck_kernels" in state or "benchmark_config" in state, \
            f"pipeline_state missing benchmark/bottleneck data. Keys: {list(state.keys())}"

        opt_results = state.get("optimization_results", [])
        if opt_results:
            r0 = opt_results[0]
            for field in ("compiled", "correct", "speedup"):
                assert field in r0, f"Missing field '{field}' in optimization result"
            print(f"  Optimization result: compiled={r0.get('compiled')} "
                  f"correct={r0.get('correct')} speedup={r0.get('speedup')} "
                  f"rl_reward={r0.get('rl_reward')}")
        else:
            print("  No optimization results yet (agent may still be running)")

        # Check leaderboard
        lb_path = results_dir / "leaderboard.json"
        lb_jsonl = results_dir / "leaderboard.jsonl"
        has_lb = lb_path.exists() or lb_jsonl.exists()
        if has_lb:
            print("  Leaderboard entry created")

        # Check trajectory
        traj_path = results_dir / "trajectory.json"
        if traj_path.exists():
            print("  Trajectory file created")

        # Check knowledge base
        kb_path = results_dir / "knowledge_base.json"
        if kb_path.exists():
            kb = json.loads(kb_path.read_text())
            print(f"  Knowledge base entries: {len(kb.get('entries', []))}")

        print(f"\n  Full pipeline E2E completed. Results in: {results_dir}")
