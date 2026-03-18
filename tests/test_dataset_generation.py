"""
test_dataset_generation.py — Tests for trajectory persistence and dataset generation.

Covers:
  1. Ground truth auto-discovery across tools/rocm/
  2. GroundTruthSpec schema validation (3 modes: pytorch, library_test, accordo)
  3. Accordo config generation for HIP/C++ kernels
  4. Export roundtrip: Apex -> tasks.json -> keystone load_apex_export()
  5. Trajectory persistence (FileStore)
  6. SFT warm-start extraction
  7. Standalone mode (no trajectories)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from ground_truth import (
    GroundTruthSpec,
    MANUAL_REGISTRY,
    REPO_URLS,
    ROCM_DIR,
    discover_all,
    discover_by_library,
    generate_accordo_config,
    get_spec,
    scan_hip_kernels_for_accordo,
    scan_rocm_ground_truth,
    scan_test_commands,
)
from trajectory import (
    FileStore,
    TrajectoryRecord,
    WorkloadTrajectoryRecord,
    _record_from_dict,
    export_for_keystone_rl,
)
from export_rl_dataset import (
    _get_instruction,
    _read_baseline_code,
    export,
    generate_standalone_tasks,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_workload_trajectory() -> dict:
    return {
        "trajectory_id": "test-wl-001",
        "workload_id": "workload__vllm__test_model",
        "timestamp": "2026-03-01T00:00:00+00:00",
        "agent_model": "test-model",
        "agent_version": "v1.0",
        "framework": "vllm",
        "model_id": "test/model-1b",
        "gpu_arch": "gfx950",
        "baseline_tps": 1000.0,
        "final_tps": 1200.0,
        "selected_kernels": ["rms_norm", "fused_moe"],
        "kernel_optimizations": [
            {
                "kernel_name": "rms_norm",
                "compiled": True,
                "correct": True,
                "speedup": 1.15,
                "kernel_score": 215.0,
                "baseline_ms": 0.03,
                "optimized_ms": 0.026,
            },
            {
                "kernel_name": "fused_moe",
                "compiled": True,
                "correct": True,
                "speedup": 1.05,
                "kernel_score": 205.0,
                "baseline_ms": 0.5,
                "optimized_ms": 0.476,
            },
        ],
        "total_reward": 0.75,
        "trajectory_quality": "good",
    }


@pytest.fixture
def sample_kernel_trajectory() -> dict:
    return {
        "trajectory_id": "test-k-001",
        "task_id": "test_model__rms_norm",
        "agent_model": "test-model",
        "agent_version": "v1.0",
        "timestamp": "2026-03-01T00:00:00+00:00",
        "prompt": "Optimize the RMS norm kernel...",
        "baseline_tps": 1000.0,
        "gpu_arch": "gfx950",
        "model_id": "test/model-1b",
        "kernel_type": "rms_norm",
        "framework": "vllm",
        "iterations": [
            {
                "iteration": 1,
                "kernel_result": {"compiled": True, "correct": True, "speedup": 1.1, "score": 210},
                "agent_messages": [{"role": "assistant", "content": "optimized kernel code..."}],
                "solution_code": "import triton\ndef optimized_rmsnorm(): pass",
                "score": 210,
            },
        ],
        "final_score": 210,
        "final_speedup": 1.1,
        "final_tps": 1100.0,
        "reward": 0.7,
        "trajectory_quality": "good",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Ground Truth Auto-Discovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthDiscovery:

    def test_manual_registry_populated(self):
        assert len(MANUAL_REGISTRY) >= 10
        for kt, spec in MANUAL_REGISTRY.items():
            assert spec.kernel_type == kt
            assert spec.mode in ("pytorch", "library_test", "accordo")

    def test_manual_registry_pytorch_has_code(self):
        for spec in MANUAL_REGISTRY.values():
            if spec.mode == "pytorch":
                assert len(spec.pytorch_reference_code) > 10
                assert "def " in spec.pytorch_reference_code

    def test_manual_registry_library_test_has_command(self):
        for spec in MANUAL_REGISTRY.values():
            if spec.mode == "library_test":
                assert spec.unit_test_command
                assert spec.repo_url

    def test_discover_all_returns_specs(self):
        specs = discover_all(max_files=500)
        assert len(specs) > len(MANUAL_REGISTRY)

    def test_discover_all_has_multiple_libraries(self):
        by_lib = discover_by_library(max_files=500)
        assert len(by_lib) >= 2, f"Only found libs: {list(by_lib.keys())}"

    @pytest.mark.skipif(not ROCM_DIR.exists(), reason="tools/rocm not available")
    def test_scan_rocm_finds_pytorch_refs(self):
        specs = scan_rocm_ground_truth(max_files=500)
        pytorch_specs = [s for s in specs if s.mode == "pytorch"]
        assert len(pytorch_specs) > 0

    @pytest.mark.skipif(not ROCM_DIR.exists(), reason="tools/rocm not available")
    def test_scan_rocm_has_valid_source_files(self):
        specs = scan_rocm_ground_truth(max_files=200)
        for s in specs:
            assert s.source_file, f"Missing source_file for {s.kernel_type}"
            assert s.source_library, f"Missing source_library for {s.kernel_type}"

    @pytest.mark.skipif(not ROCM_DIR.exists(), reason="tools/rocm not available")
    def test_scan_rocm_ref_code_is_parseable(self):
        specs = scan_rocm_ground_truth(max_files=200)
        for s in specs:
            if s.mode == "pytorch" and s.pytorch_reference_code:
                try:
                    import ast
                    ast.parse(s.pytorch_reference_code)
                except SyntaxError:
                    pytest.fail(
                        f"Unparseable pytorch_reference_code for {s.kernel_type} "
                        f"from {s.source_file}"
                    )

    def test_get_spec_returns_manual(self):
        spec = get_spec("rms_norm")
        assert spec is not None
        assert spec.mode == "pytorch"
        assert "baseline_fn" in spec.pytorch_reference_code

    def test_get_spec_returns_none_for_unknown(self):
        spec = get_spec("nonexistent_kernel_type_xyz")
        assert spec is None

    def test_repo_urls_mapping(self):
        assert "aiter" in REPO_URLS
        assert "vllm" in REPO_URLS
        assert REPO_URLS["aiter"] == "https://github.com/ROCm/aiter"

    def test_discover_all_no_duplicate_kernel_types(self):
        specs = discover_all(max_files=500)
        kernel_types = [s.kernel_type for s in specs]
        assert len(kernel_types) == len(set(kernel_types)), (
            f"Duplicate kernel types: "
            f"{[kt for kt in kernel_types if kernel_types.count(kt) > 1]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GroundTruthSpec Schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundTruthSpec:

    def test_pytorch_mode_dict(self):
        spec = GroundTruthSpec(
            kernel_type="test_kernel",
            mode="pytorch",
            pytorch_reference_code="def baseline_fn(): pass",
            test_shapes_code="def get_test_inputs(): pass",
        )
        gt = spec.to_ground_truth_dict()
        assert gt["pytorch_reference_code"] == "def baseline_fn(): pass"
        assert gt["test_shapes_code"] == "def get_test_inputs(): pass"
        assert gt["repo_url"] == ""
        assert gt["unit_test_command"] == ""
        assert gt["accordo_config"] == {}

    def test_library_test_mode_dict(self):
        spec = GroundTruthSpec(
            kernel_type="test_kernel",
            mode="library_test",
            repo_url="https://github.com/ROCm/aiter",
            unit_test_command="python -m pytest test_x.py",
        )
        gt = spec.to_ground_truth_dict()
        assert gt["pytorch_reference_code"] == ""
        assert gt["test_shapes_code"] == ""
        assert gt["repo_url"] == "https://github.com/ROCm/aiter"
        assert gt["unit_test_command"] == "python -m pytest test_x.py"
        assert gt["accordo_config"] == {}

    def test_accordo_mode_dict(self):
        config = {"correctness": {"backend": "accordo", "accordo": {"kernel_name": "gemm"}}}
        spec = GroundTruthSpec(
            kernel_type="ck_gemm",
            mode="accordo",
            accordo_config=config,
        )
        gt = spec.to_ground_truth_dict()
        assert gt["pytorch_reference_code"] == ""
        assert gt["test_shapes_code"] == ""
        assert gt["repo_url"] == ""
        assert gt["unit_test_command"] == ""
        assert gt["accordo_config"]["correctness"]["backend"] == "accordo"

    def test_to_dict_roundtrip(self):
        spec = GroundTruthSpec(
            kernel_type="test",
            mode="pytorch",
            pytorch_reference_code="code",
            source_file="test.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="memory_bound",
        )
        d = spec.to_dict()
        assert d["kernel_type"] == "test"
        assert d["mode"] == "pytorch"
        assert d["difficulty_level"] == 2

    def test_difficulty_levels(self):
        for spec in MANUAL_REGISTRY.values():
            assert spec.difficulty_level in (1, 2, 3)

    def test_op_types(self):
        for spec in MANUAL_REGISTRY.values():
            assert spec.op_type in ("memory_bound", "compute_bound")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Accordo Config Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccordoIntegration:

    def test_generate_accordo_config_structure(self):
        config = generate_accordo_config(
            kernel_name="gemm",
            reference_binary="build/bin/example_gemm",
            optimized_binary="build/bin/example_gemm_opt",
            tolerance=0.001,
            working_directory="${CK_HOME}",
        )
        assert config["correctness"]["backend"] == "accordo"
        assert config["correctness"]["accordo"]["kernel_name"] == "gemm"
        assert config["correctness"]["accordo"]["tolerance"] == 0.001
        assert config["correctness"]["accordo"]["optimized_binary"] == "build/bin/example_gemm_opt"

    def test_generate_accordo_config_no_optional(self):
        config = generate_accordo_config(
            kernel_name="conv",
            reference_binary="bin/conv_ref",
        )
        assert "optimized_binary" not in config["correctness"]["accordo"]
        assert "working_directory" not in config["correctness"]["accordo"]

    @pytest.mark.skipif(not ROCM_DIR.exists(), reason="tools/rocm not available")
    def test_scan_hip_kernels_finds_ck(self):
        specs = scan_hip_kernels_for_accordo()
        ck_specs = [s for s in specs if s.source_library == "composable_kernel"]
        assert len(ck_specs) > 0

    @pytest.mark.skipif(not ROCM_DIR.exists(), reason="tools/rocm not available")
    def test_scan_hip_kernels_accordo_mode(self):
        specs = scan_hip_kernels_for_accordo()
        for s in specs:
            assert s.mode == "accordo"
            assert s.accordo_config
            assert s.accordo_config["correctness"]["backend"] == "accordo"

    @pytest.mark.skipif(not ROCM_DIR.exists(), reason="tools/rocm not available")
    def test_accordo_specs_in_discover_all(self):
        specs = discover_all(max_files=500)
        accordo_specs = [s for s in specs if s.mode == "accordo"]
        assert len(accordo_specs) > 0

    def test_accordo_config_json_serializable(self):
        config = generate_accordo_config(
            kernel_name="test",
            reference_binary="bin/test",
        )
        serialized = json.dumps(config)
        deserialized = json.loads(serialized)
        assert deserialized == config


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Trajectory Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryPersistence:

    def test_filestore_save_load_workload(self, tmp_dir, sample_workload_trajectory):
        store = FileStore(base_dir=tmp_dir)
        record = WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory)
        tid = store.save(record)
        assert (tmp_dir / f"{tid}.json").exists()

        loaded = store.load(tid)
        assert isinstance(loaded, WorkloadTrajectoryRecord)
        assert loaded.trajectory_id == tid
        assert loaded.workload_id == "workload__vllm__test_model"
        assert len(loaded.kernel_optimizations) == 2

    def test_filestore_save_load_kernel(self, tmp_dir, sample_kernel_trajectory):
        store = FileStore(base_dir=tmp_dir)
        record = TrajectoryRecord.from_dict(sample_kernel_trajectory)
        tid = store.save(record)
        assert (tmp_dir / f"{tid}.json").exists()

        loaded = store.load(tid)
        assert isinstance(loaded, TrajectoryRecord)
        assert loaded.kernel_type == "rms_norm"
        assert loaded.final_score == 210

    def test_filestore_list_ids(self, tmp_dir, sample_workload_trajectory, sample_kernel_trajectory):
        store = FileStore(base_dir=tmp_dir)
        store.save(WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory))
        store.save(TrajectoryRecord.from_dict(sample_kernel_trajectory))
        ids = store.list_ids()
        assert len(ids) == 2

    def test_filestore_load_all(self, tmp_dir, sample_workload_trajectory, sample_kernel_trajectory):
        store = FileStore(base_dir=tmp_dir)
        store.save(WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory))
        store.save(TrajectoryRecord.from_dict(sample_kernel_trajectory))
        all_records = store.load_all()
        assert len(all_records) == 2
        types = {type(r).__name__ for r in all_records}
        assert "WorkloadTrajectoryRecord" in types
        assert "TrajectoryRecord" in types

    def test_filestore_load_nonexistent(self, tmp_dir):
        store = FileStore(base_dir=tmp_dir)
        assert store.load("nonexistent") is None

    def test_record_from_dict_detects_workload(self, sample_workload_trajectory):
        record = _record_from_dict(sample_workload_trajectory)
        assert isinstance(record, WorkloadTrajectoryRecord)

    def test_record_from_dict_detects_kernel(self, sample_kernel_trajectory):
        record = _record_from_dict(sample_kernel_trajectory)
        assert isinstance(record, TrajectoryRecord)

    def test_real_trajectories_load(self):
        """Test loading real trajectory files from the repo."""
        traj_dir = REPO_ROOT / "trajectories"
        if not traj_dir.exists() or not list(traj_dir.glob("*.json")):
            pytest.skip("No real trajectory files available")
        store = FileStore(base_dir=traj_dir)
        ids = store.list_ids()
        assert len(ids) > 0
        for tid in ids[:3]:
            rec = store.load(tid)
            assert rec is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Export Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportPipeline:

    def test_export_from_trajectories(self, tmp_dir, sample_workload_trajectory, sample_kernel_trajectory):
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir()
        out_dir = tmp_dir / "datasets"

        store = FileStore(base_dir=traj_dir)
        store.save(WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory))
        store.save(TrajectoryRecord.from_dict(sample_kernel_trajectory))

        result = export(
            trajectories_dir=traj_dir,
            results_dirs=[],
            output_dir=out_dir,
            include_sft=True,
            standalone=False,
        )
        assert result["tasks_exported"] > 0
        assert (out_dir / "tasks.json").exists()
        assert (out_dir / "export_metadata.json").exists()

    def test_export_standalone_mode(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        result = export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        assert result["tasks_exported"] > 0
        tasks = json.loads((out_dir / "tasks.json").read_text())
        assert len(tasks) == result["tasks_exported"]

    def test_export_sft_warmstart(self, tmp_dir, sample_kernel_trajectory):
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir()
        out_dir = tmp_dir / "datasets"

        store = FileStore(base_dir=traj_dir)
        store.save(TrajectoryRecord.from_dict(sample_kernel_trajectory))

        result = export(
            trajectories_dir=traj_dir,
            results_dirs=[],
            output_dir=out_dir,
            include_sft=True,
            standalone=False,
        )
        if result["sft_pairs_exported"] > 0:
            sft_path = out_dir / "sft_warmstart.jsonl"
            assert sft_path.exists()
            lines = sft_path.read_text().strip().splitlines()
            for line in lines:
                pair = json.loads(line)
                assert "prompt" in pair
                assert "response" in pair
                assert "score" in pair

    def test_export_metadata(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        meta = json.loads((out_dir / "export_metadata.json").read_text())
        assert "exported_at" in meta
        assert "tasks_exported" in meta
        assert meta["standalone_mode"] is True

    def test_export_deduplicates_tasks(self, tmp_dir, sample_workload_trajectory):
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir()
        out_dir = tmp_dir / "datasets"

        store = FileStore(base_dir=traj_dir)
        store.save(WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory))
        store.save(WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory))

        result = export(
            trajectories_dir=traj_dir,
            results_dirs=[],
            output_dir=out_dir,
            standalone=False,
        )
        tasks = json.loads((out_dir / "tasks.json").read_text())
        task_ids = [t["task_id"] for t in tasks]
        assert len(task_ids) == len(set(task_ids))

    def test_export_quality_filter(self, tmp_dir, sample_workload_trajectory):
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir()
        out_dir = tmp_dir / "datasets"

        store = FileStore(base_dir=traj_dir)
        store.save(WorkloadTrajectoryRecord.from_dict(sample_workload_trajectory))

        result_all = export(
            trajectories_dir=traj_dir,
            results_dirs=[],
            output_dir=out_dir,
            quality_filter=None,
            standalone=False,
        )

        result_good = export(
            trajectories_dir=traj_dir,
            results_dirs=[],
            output_dir=out_dir,
            quality_filter="good",
            standalone=False,
        )

        # When quality="bad" matches nothing, standalone fallback adds tasks.
        # The important check: no "bad" trajectory tasks leak through.
        result_bad = export(
            trajectories_dir=traj_dir,
            results_dirs=[],
            output_dir=out_dir,
            quality_filter="bad",
            standalone=False,
        )
        bad_tasks = json.loads((out_dir / "tasks.json").read_text())
        traj_derived = [t for t in bad_tasks if "test_model" in t["task_id"]]
        assert len(traj_derived) == 0, "Quality filter should exclude non-matching trajectories"

        assert result_all["tasks_exported"] >= result_good["tasks_exported"]

    def test_export_results_dir(self, tmp_dir, sample_workload_trajectory):
        results_dir = tmp_dir / "results"
        results_dir.mkdir()
        with open(results_dir / "trajectory.json", "w") as f:
            json.dump(sample_workload_trajectory, f)

        out_dir = tmp_dir / "datasets"
        result = export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[results_dir],
            output_dir=out_dir,
            standalone=False,
        )
        assert result["tasks_exported"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Task Schema Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskSchema:

    def test_task_has_required_fields(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        tasks = json.loads((out_dir / "tasks.json").read_text())
        required = {"task_id", "instruction", "base_gpu_kernel_code",
                     "difficulty_level", "op_type", "ground_truth"}
        for task in tasks:
            missing = required - task.keys()
            assert not missing, f"Task {task.get('task_id')} missing: {missing}"

    def test_ground_truth_has_all_fields(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        tasks = json.loads((out_dir / "tasks.json").read_text())
        gt_fields = {"pytorch_reference_code", "test_shapes_code",
                      "repo_url", "unit_test_command", "accordo_config"}
        for task in tasks:
            gt = task["ground_truth"]
            missing = gt_fields - gt.keys()
            assert not missing, f"Task {task['task_id']} ground_truth missing: {missing}"

    def test_modes_mutually_exclusive(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        tasks = json.loads((out_dir / "tasks.json").read_text())
        for task in tasks:
            gt = task["ground_truth"]
            has_pytorch = bool(gt.get("pytorch_reference_code"))
            has_lib_test = bool(gt.get("unit_test_command"))
            has_accordo = bool(gt.get("accordo_config"))
            active = sum([has_pytorch, has_lib_test, has_accordo])
            assert active <= 1, (
                f"Task {task['task_id']} has {active} active modes "
                f"(pytorch={has_pytorch}, lib_test={has_lib_test}, accordo={has_accordo})"
            )

    def test_difficulty_valid_range(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        tasks = json.loads((out_dir / "tasks.json").read_text())
        for task in tasks:
            assert task["difficulty_level"] in (1, 2, 3)

    def test_op_type_valid(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        tasks = json.loads((out_dir / "tasks.json").read_text())
        for task in tasks:
            assert task["op_type"] in ("memory_bound", "compute_bound")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Keystone Roundtrip
# ═══════════════════════════════════════════════════════════════════════════════

class TestKeystoneRoundtrip:

    @pytest.fixture
    def exported_tasks_path(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        export(
            trajectories_dir=tmp_dir / "empty",
            results_dirs=[],
            output_dir=out_dir,
            standalone=True,
        )
        return out_dir / "tasks.json"

    def test_keystone_load_apex_export(self, exported_tasks_path):
        keystone_dir = REPO_ROOT.parent / "keystone-rl-training"
        if not (keystone_dir / "prepare_data.py").exists():
            pytest.skip("keystone-rl-training not available")

        sys.path.insert(0, str(keystone_dir))
        from prepare_data import load_apex_export

        tasks = load_apex_export(str(exported_tasks_path))
        assert len(tasks) > 0
        for t in tasks:
            assert t.task_id
            assert t.ground_truth is not None
            assert t.ground_truth.mode in ("pytorch", "library_test", "accordo", "unknown")

    def test_keystone_parquet_conversion(self, exported_tasks_path, tmp_dir):
        keystone_dir = REPO_ROOT.parent / "keystone-rl-training"
        if not (keystone_dir / "prepare_data.py").exists():
            pytest.skip("keystone-rl-training not available")

        try:
            import pandas  # noqa: F401
        except ImportError:
            pytest.skip("pandas not installed")

        sys.path.insert(0, str(keystone_dir))
        from prepare_data import convert_to_verl_parquet

        parquet_path = str(tmp_dir / "test.parquet")
        convert_to_verl_parquet(str(exported_tasks_path), parquet_path)
        assert Path(parquet_path).exists()

        import pandas as pd
        df = pd.read_parquet(parquet_path)
        assert len(df) > 0
        assert "prompt" in df.columns
        assert "reward_model" in df.columns

    def test_keystone_ground_truth_modes_preserved(self, exported_tasks_path):
        keystone_dir = REPO_ROOT.parent / "keystone-rl-training"
        if not (keystone_dir / "prepare_data.py").exists():
            pytest.skip("keystone-rl-training not available")

        sys.path.insert(0, str(keystone_dir))
        from prepare_data import load_apex_export

        tasks = load_apex_export(str(exported_tasks_path))
        modes = {t.ground_truth.mode for t in tasks}
        assert len(modes) >= 2, f"Expected multiple modes, got: {modes}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Instruction Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstructionGeneration:

    def test_known_kernel_type_instruction(self):
        instr = _get_instruction("rms_norm")
        assert "RMSNorm" in instr
        assert "MI355X" in instr

    def test_unknown_kernel_type_instruction(self):
        instr = _get_instruction("my_custom_kernel")
        assert "my_custom_kernel" in instr
        assert "MI355X" in instr

    def test_all_manual_types_have_instructions(self):
        for kt in MANUAL_REGISTRY:
            instr = _get_instruction(kt)
            assert len(instr) > 20


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Standalone Task Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestStandaloneGeneration:

    def test_generate_standalone_tasks(self):
        tasks = generate_standalone_tasks(max_specs=50)
        assert len(tasks) > 0
        for task in tasks:
            assert "task_id" in task
            assert "ground_truth" in task

    def test_standalone_no_duplicate_kernel_types(self):
        tasks = generate_standalone_tasks(max_specs=50)
        task_ids = [t["task_id"] for t in tasks]
        assert len(task_ids) == len(set(task_ids))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. trajectory.py export_for_keystone_rl wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportWiring:

    def test_export_for_keystone_rl_callable(self, tmp_dir):
        out_dir = tmp_dir / "datasets"
        result = export_for_keystone_rl(
            trajectories_dir=tmp_dir / "empty",
            output_dir=out_dir,
            standalone=True,
        )
        assert "tasks_exported" in result
        assert result["tasks_exported"] > 0
