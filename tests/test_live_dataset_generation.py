#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Live dataset generation validation: real agent optimization + dataset schema checks.

Runs actual kernel optimizations via agent backends (cursor/codex/claude) and
validates that all dataset artifacts are generated with correct schemas.

Phases:
  1. Standalone optimize-kernel per correctness mode (pytorch, library_test, accordo)
  2. Full pipeline run producing trajectory.json + FileStore + leaderboard
  3. export-rl producing tasks.json + sft_warmstart.jsonl + export_metadata.json
  4. Cross-validation: trajectory-derived vs standalone discovery, 3-mode coverage

Requirements:
  - GPU available (MI355X / CDNA4)
  - At least one agent CLI installed (cursor-agent, codex, or claude)
  - MAGPIE_ROOT set
  - Existing benchmark report for Phase 2 (auto-discovered)

Run with: pytest tests/test_live_dataset_generation.py -m live -v -s
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

pytestmark = pytest.mark.live

AGENT_BACKEND = os.environ.get("APEX_TEST_AGENT_BACKEND", "cursor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_rms_norm_baseline() -> Path:
    candidates = [
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "triton" / "normalization" / "rmsnorm.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "triton" / "norm" / "rms_norm.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "rmsnorm.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("rms_norm baseline not found in tools/rocm/aiter")


def _find_activation_baseline() -> Path:
    candidates = [
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "triton" / "activation.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "triton" / "_triton_kernels" / "activation.py",
        REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "activation.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("activation baseline not found in tools/rocm/aiter")


def _find_ck_gemm_kernel() -> Path:
    candidates = list(
        (REPO_ROOT / "tools" / "rocm" / "composable_kernel").rglob("example_gemm*.cpp")
    ) if (REPO_ROOT / "tools" / "rocm" / "composable_kernel").exists() else []
    if candidates:
        return candidates[0]
    for hip_file in (REPO_ROOT / "tools" / "rocm").rglob("*.hip"):
        return hip_file
    pytest.skip("No HIP/CK kernel found for accordo mode test")


def _find_benchmark_report() -> Path:
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
    pytest.skip("No existing benchmark report found for Phase 2")


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


def _run_cmd(cmd: list[str], env: dict, timeout: int = 600) -> subprocess.CompletedProcess:
    print(f"\n  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if result.stdout:
        print(f"  stdout (last 3000 chars):\n{result.stdout[-3000:]}")
    if result.stderr:
        print(f"  stderr (last 1000 chars):\n{result.stderr[-1000:]}")
    return result


def _make_env() -> dict:
    env = os.environ.copy()
    env["MAGPIE_ROOT"] = os.environ.get("MAGPIE_ROOT", str(REPO_ROOT.parent / "Magpie"))
    return env


STANDALONE_RESULT_REQUIRED_FIELDS = {
    "task_id", "kernel_path", "kernel_type", "kernel_name",
    "correctness_mode", "agent_backend", "agent_model",
    "compiled", "correct", "speedup", "score", "elapsed_seconds",
}


# ---------------------------------------------------------------------------
# Session-scoped fixtures (shared across phases)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def session_results_dir():
    d = Path(tempfile.mkdtemp(prefix="apex_live_dataset_"))
    print(f"\n  Session results dir: {d}")
    yield d


@pytest.fixture(scope="session")
def phase1_pytorch_dir(session_results_dir):
    d = session_results_dir / "phase1_pytorch"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def phase1_libtest_dir(session_results_dir):
    d = session_results_dir / "phase1_libtest"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def phase1_accordo_dir(session_results_dir):
    d = session_results_dir / "phase1_accordo"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def phase2_dir(session_results_dir):
    d = session_results_dir / "phase2_pipeline"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def phase3_dir(session_results_dir):
    d = session_results_dir / "phase3_export"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1a: Standalone optimize-kernel — pytorch mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase1Pytorch:

    @pytest.fixture(scope="class", autouse=True)
    def run_optimization(self, phase1_pytorch_dir):
        marker = phase1_pytorch_dir / ".done"
        if marker.exists():
            return
        baseline = _find_rms_norm_baseline()
        cmd = [
            sys.executable, str(REPO_ROOT / "workload_optimizer.py"),
            "optimize-kernel",
            "-r", str(phase1_pytorch_dir),
            "--kernel", str(baseline),
            "--kernel-name", "rms_norm",
            "--kernel-type", "triton",
            "--correctness-mode", "pytorch",
            "--max-iterations", "1",
            "--max-turns", "10",
            "--agent-backend", AGENT_BACKEND,
        ]
        result = _run_cmd(cmd, _make_env(), timeout=600)
        marker.write_text(str(result.returncode))

    def test_standalone_result_exists(self, phase1_pytorch_dir):
        sr = phase1_pytorch_dir / "standalone_result.json"
        assert sr.exists(), f"standalone_result.json not found in {phase1_pytorch_dir}"

    def test_standalone_result_schema(self, phase1_pytorch_dir):
        sr = json.loads((phase1_pytorch_dir / "standalone_result.json").read_text())
        missing = STANDALONE_RESULT_REQUIRED_FIELDS - sr.keys()
        assert not missing, f"Missing fields: {missing}"

    def test_correctness_mode_is_pytorch(self, phase1_pytorch_dir):
        sr = json.loads((phase1_pytorch_dir / "standalone_result.json").read_text())
        assert sr["correctness_mode"] == "pytorch"

    def test_task_dir_has_baseline(self, phase1_pytorch_dir):
        task_dirs = list((phase1_pytorch_dir / "output").glob("*/baseline.py"))
        assert len(task_dirs) >= 1, "No baseline.py in any task dir"

    def test_task_dir_has_config(self, phase1_pytorch_dir):
        task_dirs = list((phase1_pytorch_dir / "output").glob("*/config.yaml"))
        assert len(task_dirs) >= 1, "No config.yaml in any task dir"

    def test_solution_written(self, phase1_pytorch_dir):
        solutions = list((phase1_pytorch_dir / "output").glob("*/solution*.py"))
        if not solutions:
            sr = json.loads((phase1_pytorch_dir / "standalone_result.json").read_text())
            if sr.get("compiled"):
                pytest.fail("compiled=True but no solution.py found")
            else:
                pytest.skip("Agent did not produce a compilable solution")

    def test_compiled_field_is_bool(self, phase1_pytorch_dir):
        sr = json.loads((phase1_pytorch_dir / "standalone_result.json").read_text())
        assert isinstance(sr["compiled"], bool)

    def test_speedup_is_numeric(self, phase1_pytorch_dir):
        sr = json.loads((phase1_pytorch_dir / "standalone_result.json").read_text())
        assert isinstance(sr["speedup"], (int, float))

    def test_knowledge_base_updated(self, phase1_pytorch_dir):
        kb = phase1_pytorch_dir / "knowledge_base.json"
        sr = json.loads((phase1_pytorch_dir / "standalone_result.json").read_text())
        if sr.get("correct") and sr.get("speedup", 0) > 0:
            assert kb.exists(), "correct+speedup>0 but knowledge_base.json missing"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1b: Standalone optimize-kernel — library_test mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase1LibraryTest:

    @pytest.fixture(scope="class", autouse=True)
    def run_optimization(self, phase1_libtest_dir):
        marker = phase1_libtest_dir / ".done"
        if marker.exists():
            return
        baseline = _find_activation_baseline()
        cmd = [
            sys.executable, str(REPO_ROOT / "workload_optimizer.py"),
            "optimize-kernel",
            "-r", str(phase1_libtest_dir),
            "--kernel", str(baseline),
            "--kernel-name", "silu_mul",
            "--kernel-type", "triton",
            "--correctness-mode", "library_test",
            "--test-cmd", "python -m pytest aiter/op_tests/triton_tests/test_activation.py",
            "--repo-url", "https://github.com/ROCm/aiter",
            "--max-iterations", "1",
            "--max-turns", "10",
            "--agent-backend", AGENT_BACKEND,
        ]
        result = _run_cmd(cmd, _make_env(), timeout=600)
        marker.write_text(str(result.returncode))

    def test_standalone_result_exists(self, phase1_libtest_dir):
        sr = phase1_libtest_dir / "standalone_result.json"
        assert sr.exists(), f"standalone_result.json not found in {phase1_libtest_dir}"

    def test_standalone_result_schema(self, phase1_libtest_dir):
        sr = json.loads((phase1_libtest_dir / "standalone_result.json").read_text())
        missing = STANDALONE_RESULT_REQUIRED_FIELDS - sr.keys()
        assert not missing, f"Missing fields: {missing}"

    def test_correctness_mode_is_library_test(self, phase1_libtest_dir):
        sr = json.loads((phase1_libtest_dir / "standalone_result.json").read_text())
        assert sr["correctness_mode"] == "library_test"

    def test_task_dir_has_baseline(self, phase1_libtest_dir):
        task_dirs = list((phase1_libtest_dir / "output").glob("*/baseline.py"))
        assert len(task_dirs) >= 1, "No baseline.py in any task dir"

    def test_compiled_field_is_bool(self, phase1_libtest_dir):
        sr = json.loads((phase1_libtest_dir / "standalone_result.json").read_text())
        assert isinstance(sr["compiled"], bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1c: Standalone optimize-kernel — accordo mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase1Accordo:

    @pytest.fixture(scope="class", autouse=True)
    def run_optimization(self, phase1_accordo_dir):
        marker = phase1_accordo_dir / ".done"
        if marker.exists():
            return
        kernel = _find_ck_gemm_kernel()
        k_type = "hip" if kernel.suffix in (".hip", ".cpp", ".cu") else "triton"
        cmd = [
            sys.executable, str(REPO_ROOT / "workload_optimizer.py"),
            "optimize-kernel",
            "-r", str(phase1_accordo_dir),
            "--kernel", str(kernel),
            "--kernel-name", "ck_gemm",
            "--kernel-type", k_type,
            "--correctness-mode", "accordo",
            "--max-iterations", "1",
            "--max-turns", "10",
            "--agent-backend", AGENT_BACKEND,
        ]
        result = _run_cmd(cmd, _make_env(), timeout=600)
        marker.write_text(str(result.returncode))

    def test_standalone_result_exists(self, phase1_accordo_dir):
        sr = phase1_accordo_dir / "standalone_result.json"
        assert sr.exists(), f"standalone_result.json not found in {phase1_accordo_dir}"

    def test_standalone_result_schema(self, phase1_accordo_dir):
        sr = json.loads((phase1_accordo_dir / "standalone_result.json").read_text())
        missing = STANDALONE_RESULT_REQUIRED_FIELDS - sr.keys()
        assert not missing, f"Missing fields: {missing}"

    def test_correctness_mode_is_accordo(self, phase1_accordo_dir):
        sr = json.loads((phase1_accordo_dir / "standalone_result.json").read_text())
        assert sr["correctness_mode"] == "accordo"

    def test_compiled_field_is_bool(self, phase1_accordo_dir):
        sr = json.loads((phase1_accordo_dir / "standalone_result.json").read_text())
        assert isinstance(sr["compiled"], bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Full pipeline run → trajectory.json
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase2FullPipeline:

    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(self, phase2_dir):
        marker = phase2_dir / ".done"
        if marker.exists():
            return
        bench_report = _find_benchmark_report()
        bench_config = _find_bench_config()
        cmd = [
            sys.executable, str(REPO_ROOT / "workload_optimizer.py"),
            "run",
            "-r", str(phase2_dir),
            "-b", str(bench_config),
            "--skip-benchmark", str(bench_report),
            "--kernel-types", "triton",
            "--top-k", "1",
            "--max-iterations", "1",
            "--max-turns", "10",
            "--leaderboard",
            "--agent-backend", AGENT_BACKEND,
        ]
        result = _run_cmd(cmd, _make_env(), timeout=1800)
        marker.write_text(str(result.returncode))

    def test_trajectory_json_exists(self, phase2_dir):
        traj = phase2_dir / "trajectory.json"
        assert traj.exists(), "trajectory.json not created"

    def test_trajectory_valid_json(self, phase2_dir):
        traj = json.loads((phase2_dir / "trajectory.json").read_text())
        assert isinstance(traj, dict)

    def test_trajectory_has_required_fields(self, phase2_dir):
        traj = json.loads((phase2_dir / "trajectory.json").read_text())
        required = {
            "trajectory_id", "workload_id", "timestamp",
            "agent_model", "framework", "model_id", "gpu_arch",
            "baseline_tps", "selected_kernels",
            "kernel_optimizations", "trajectory_quality",
        }
        missing = required - traj.keys()
        assert not missing, f"trajectory.json missing fields: {missing}"

    def test_trajectory_quality_valid(self, phase2_dir):
        traj = json.loads((phase2_dir / "trajectory.json").read_text())
        assert traj["trajectory_quality"] in ("good", "mediocre", "bad", "unknown")

    def test_kernel_optimizations_schema(self, phase2_dir):
        traj = json.loads((phase2_dir / "trajectory.json").read_text())
        opts = traj.get("kernel_optimizations", [])
        if not opts:
            pytest.skip("No kernel_optimizations in trajectory (agent may have failed)")
        for opt in opts:
            for field in ("kernel_name", "compiled", "correct", "speedup", "score"):
                assert field in opt, f"kernel_optimization missing '{field}'"
            assert isinstance(opt["compiled"], bool)
            assert isinstance(opt["correct"], bool)
            assert isinstance(opt["speedup"], (int, float))

    def test_pipeline_state_exists(self, phase2_dir):
        ps = phase2_dir / "pipeline_state.json"
        assert ps.exists(), "pipeline_state.json not created"

    def test_pipeline_state_has_optimization_results(self, phase2_dir):
        ps = json.loads((phase2_dir / "pipeline_state.json").read_text())
        assert "optimization_results" in ps or "bottleneck_kernels" in ps, \
            f"pipeline_state.json missing expected keys. Keys: {list(ps.keys())}"

    def test_filestore_trajectory_written(self, phase2_dir):
        traj_dir = REPO_ROOT / "trajectories"
        if traj_dir.exists():
            files = list(traj_dir.glob("*.json"))
            assert len(files) > 0, "No trajectory JSON files in trajectories/"
        else:
            pytest.skip("trajectories/ directory does not exist")

    def test_leaderboard_written(self, phase2_dir):
        lb_json = phase2_dir / "leaderboard.json"
        lb_jsonl = phase2_dir / "leaderboard.jsonl"
        assert lb_json.exists() or lb_jsonl.exists(), \
            "Neither leaderboard.json nor leaderboard.jsonl created"

    def test_trajectory_metadata_has_env_snapshot(self, phase2_dir):
        traj = json.loads((phase2_dir / "trajectory.json").read_text())
        meta = traj.get("metadata", {})
        if meta:
            assert "environment_snapshot" in meta, \
                "metadata exists but missing environment_snapshot"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: export-rl → tasks.json + sft_warmstart.jsonl
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase3ExportRL:

    @pytest.fixture(scope="class", autouse=True)
    def run_export(self, phase2_dir, phase3_dir):
        marker = phase3_dir / ".done"
        if marker.exists():
            return

        traj_dir = REPO_ROOT / "trajectories"
        if not traj_dir.exists():
            traj_dir = phase2_dir

        cmd = [
            sys.executable, str(REPO_ROOT / "workload_optimizer.py"),
            "export-rl",
            "-r", str(phase2_dir),
            "--trajectories-dir", str(traj_dir),
            "--export-output-dir", str(phase3_dir),
            "--sft",
        ]
        result = _run_cmd(cmd, _make_env(), timeout=120)
        marker.write_text(str(result.returncode))

    def test_tasks_json_exists(self, phase3_dir):
        assert (phase3_dir / "tasks.json").exists(), "tasks.json not created"

    def test_tasks_json_valid_json_list(self, phase3_dir):
        tasks = json.loads((phase3_dir / "tasks.json").read_text())
        assert isinstance(tasks, list)
        assert len(tasks) > 0, "tasks.json is empty"

    def test_task_required_fields(self, phase3_dir):
        tasks = json.loads((phase3_dir / "tasks.json").read_text())
        required = {"task_id", "instruction", "base_gpu_kernel_code",
                     "difficulty_level", "op_type", "ground_truth"}
        for task in tasks:
            missing = required - task.keys()
            assert not missing, f"Task {task.get('task_id')} missing fields: {missing}"

    def test_ground_truth_has_all_five_fields(self, phase3_dir):
        tasks = json.loads((phase3_dir / "tasks.json").read_text())
        gt_fields = {"pytorch_reference_code", "test_shapes_code",
                      "repo_url", "unit_test_command", "accordo_config"}
        for task in tasks:
            gt = task["ground_truth"]
            missing = gt_fields - gt.keys()
            assert not missing, f"Task {task['task_id']} ground_truth missing: {missing}"

    def test_modes_mutually_exclusive(self, phase3_dir):
        tasks = json.loads((phase3_dir / "tasks.json").read_text())
        for task in tasks:
            gt = task["ground_truth"]
            has_pytorch = bool(gt.get("pytorch_reference_code"))
            has_lib_test = bool(gt.get("unit_test_command"))
            has_accordo = bool(gt.get("accordo_config"))
            active = sum([has_pytorch, has_lib_test, has_accordo])
            assert active <= 1, (
                f"Task {task['task_id']} has {active} active modes "
                f"(pytorch={has_pytorch}, lib={has_lib_test}, accordo={has_accordo})"
            )

    def test_difficulty_valid_range(self, phase3_dir):
        tasks = json.loads((phase3_dir / "tasks.json").read_text())
        for task in tasks:
            assert task["difficulty_level"] in (1, 2, 3), \
                f"Task {task['task_id']} difficulty={task['difficulty_level']}"

    def test_op_type_valid(self, phase3_dir):
        tasks = json.loads((phase3_dir / "tasks.json").read_text())
        valid = {"memory_bound", "compute_bound"}
        for task in tasks:
            assert task["op_type"] in valid, \
                f"Task {task['task_id']} op_type={task['op_type']}"

    def test_sft_warmstart_exists(self, phase3_dir):
        sft = phase3_dir / "sft_warmstart.jsonl"
        if not sft.exists():
            pytest.skip("sft_warmstart.jsonl not created (may need good trajectories)")

    def test_sft_warmstart_schema(self, phase3_dir):
        sft = phase3_dir / "sft_warmstart.jsonl"
        if not sft.exists():
            pytest.skip("sft_warmstart.jsonl not created")
        lines = sft.read_text().strip().splitlines()
        for line in lines:
            pair = json.loads(line)
            assert "prompt" in pair, "SFT pair missing 'prompt'"
            assert "response" in pair, "SFT pair missing 'response'"
            assert "score" in pair, "SFT pair missing 'score'"

    def test_export_metadata_exists(self, phase3_dir):
        meta = phase3_dir / "export_metadata.json"
        assert meta.exists(), "export_metadata.json not created"

    def test_export_metadata_schema(self, phase3_dir):
        meta = json.loads((phase3_dir / "export_metadata.json").read_text())
        assert "exported_at" in meta
        assert "tasks_exported" in meta
        assert isinstance(meta["tasks_exported"], int)
        assert meta["tasks_exported"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase4CrossValidation:

    def test_standalone_discovery_covers_3_modes(self):
        from export_rl_dataset import generate_standalone_tasks
        tasks = generate_standalone_tasks(max_specs=200)
        modes_found = set()
        for t in tasks:
            gt = t.get("ground_truth", {})
            if gt.get("pytorch_reference_code"):
                modes_found.add("pytorch")
            if gt.get("unit_test_command"):
                modes_found.add("library_test")
            if gt.get("accordo_config"):
                modes_found.add("accordo")
        assert "pytorch" in modes_found, "No pytorch tasks in standalone discovery"
        assert "library_test" in modes_found, "No library_test tasks in standalone discovery"
        assert "accordo" in modes_found, "No accordo tasks in standalone discovery"

    def test_export_deduplicates(self):
        from export_rl_dataset import export
        with tempfile.TemporaryDirectory() as td:
            out1 = Path(td) / "run1"
            out2 = Path(td) / "run2"

            export(
                trajectories_dir=Path(td) / "empty",
                results_dirs=[],
                output_dir=out1,
                standalone=True,
            )
            export(
                trajectories_dir=Path(td) / "empty",
                results_dirs=[],
                output_dir=out2,
                standalone=True,
            )

            tasks1 = json.loads((out1 / "tasks.json").read_text())
            tasks2 = json.loads((out2 / "tasks.json").read_text())

            ids1 = sorted(t["task_id"] for t in tasks1)
            ids2 = sorted(t["task_id"] for t in tasks2)
            assert ids1 == ids2, "Two standalone exports produced different task_ids"

            assert len(ids1) == len(set(ids1)), "Duplicate task_ids in single export"

    def test_trajectory_to_tasks_consistency(self, phase2_dir):
        traj_path = phase2_dir / "trajectory.json"
        if not traj_path.exists():
            pytest.skip("Phase 2 trajectory not available")

        from trajectory import WorkloadTrajectoryRecord
        from export_rl_dataset import _trajectory_to_tasks, _record_from_dict

        data = json.loads(traj_path.read_text())
        record = _record_from_dict(data)
        tasks = _trajectory_to_tasks(record)

        if not tasks:
            pytest.skip("No tasks derived from trajectory (optimization may have failed)")

        for task in tasks:
            assert "task_id" in task
            assert "ground_truth" in task
            assert "instruction" in task

    def test_all_phase1_results_have_consistent_schemas(
        self, phase1_pytorch_dir, phase1_libtest_dir, phase1_accordo_dir
    ):
        results = []
        for d in [phase1_pytorch_dir, phase1_libtest_dir, phase1_accordo_dir]:
            sr = d / "standalone_result.json"
            if sr.exists():
                results.append(json.loads(sr.read_text()))

        if len(results) < 2:
            pytest.skip("Need at least 2 Phase 1 results for cross-validation")

        ref_keys = set(results[0].keys())
        for r in results[1:]:
            common = ref_keys & set(r.keys())
            assert STANDALONE_RESULT_REQUIRED_FIELDS <= common, (
                f"Phase 1 results have inconsistent schemas. "
                f"Missing from one: {STANDALONE_RESULT_REQUIRED_FIELDS - common}"
            )

    def test_correctness_modes_differ_across_phase1(
        self, phase1_pytorch_dir, phase1_libtest_dir, phase1_accordo_dir
    ):
        modes = set()
        for d in [phase1_pytorch_dir, phase1_libtest_dir, phase1_accordo_dir]:
            sr = d / "standalone_result.json"
            if sr.exists():
                data = json.loads(sr.read_text())
                modes.add(data.get("correctness_mode", ""))
        assert len(modes) >= 2, f"Expected multiple correctness modes, got {modes}"
