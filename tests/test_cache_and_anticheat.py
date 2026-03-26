"""
test_cache_and_anticheat.py — Tests for cache_manager.py and config_generator.py

Tests cache isolation context managers and anti-cheat config validation
without requiring GPU or Magpie.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))

import cache_manager
import config_generator


# ============================================================================
# cache_manager.py tests
# ============================================================================


class TestIsolatedTritonCache:
    def test_sets_env_var_to_temp_dir(self):
        with cache_manager.isolated_triton_cache() as tmpdir:
            assert os.environ["TRITON_CACHE_DIR"] == tmpdir
            assert Path(tmpdir).is_dir()

    def test_restores_original_env(self):
        os.environ["TRITON_CACHE_DIR"] = "/original/path"
        with cache_manager.isolated_triton_cache():
            pass
        assert os.environ["TRITON_CACHE_DIR"] == "/original/path"
        del os.environ["TRITON_CACHE_DIR"]

    def test_unsets_env_if_not_originally_set(self):
        os.environ.pop("TRITON_CACHE_DIR", None)
        with cache_manager.isolated_triton_cache():
            pass
        assert "TRITON_CACHE_DIR" not in os.environ

    def test_temp_dir_cleaned_up(self):
        with cache_manager.isolated_triton_cache() as tmpdir:
            tmppath = Path(tmpdir)
        assert not tmppath.exists()

    def test_temp_dir_cleaned_up_even_on_exception(self):
        tmppath = None
        with pytest.raises(RuntimeError):
            with cache_manager.isolated_triton_cache() as tmpdir:
                tmppath = Path(tmpdir)
                raise RuntimeError("test")
        assert tmppath is not None and not tmppath.exists()


class TestClearPycache:
    def test_removes_pycache_dir(self, tmp_path):
        module_file = tmp_path / "kernel.py"
        module_file.write_text("x = 1")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "kernel.cpython-310.pyc").write_bytes(b"compiled")

        cache_manager.clear_pycache(module_file)
        assert not pycache.exists()

    def test_no_error_when_pycache_missing(self, tmp_path):
        module_file = tmp_path / "kernel.py"
        module_file.write_text("x = 1")
        cache_manager.clear_pycache(module_file)  # should not raise


class TestClearPycacheTree:
    def test_removes_all_nested_pycache(self, tmp_path):
        for sub in ["a", "a/b", "a/b/c"]:
            (tmp_path / sub / "__pycache__").mkdir(parents=True)
        count = cache_manager.clear_pycache_tree(tmp_path)
        assert count == 3
        assert list(tmp_path.rglob("__pycache__")) == []

    def test_returns_zero_when_none_found(self, tmp_path):
        assert cache_manager.clear_pycache_tree(tmp_path) == 0


class TestClearTorchCaches:
    def test_does_not_crash_without_torch(self):
        cache_manager.clear_torch_caches()  # should not raise


class TestIsolatedTorchCache:
    def test_sets_torchinductor_cache_dir(self):
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        with cache_manager.isolated_torch_cache():
            assert "TORCHINDUCTOR_CACHE_DIR" in os.environ
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


class TestClearComgrCache:
    def test_does_not_crash(self):
        cache_manager.clear_comgr_cache()  # should not raise


class TestIsolatedComgrCache:
    def test_sets_amd_comgr_cache_dir(self):
        os.environ.pop("AMD_COMGR_CACHE_DIR", None)
        with cache_manager.isolated_comgr_cache() as tmpdir:
            assert os.environ["AMD_COMGR_CACHE_DIR"] == tmpdir
            assert Path(tmpdir).is_dir()
        assert "AMD_COMGR_CACHE_DIR" not in os.environ

    def test_restores_original_env(self):
        os.environ["AMD_COMGR_CACHE_DIR"] = "/original"
        with cache_manager.isolated_comgr_cache():
            pass
        assert os.environ["AMD_COMGR_CACHE_DIR"] == "/original"
        del os.environ["AMD_COMGR_CACHE_DIR"]


class TestPurgeModules:
    def test_purges_matching_modules(self):
        sys.modules["_test_fake_module_abc"] = type(sys)("_test_fake_module_abc")
        sys.modules["_test_fake_module_def"] = type(sys)("_test_fake_module_def")
        removed = cache_manager.purge_modules(["_test_fake_module_"])
        assert "_test_fake_module_abc" in removed
        assert "_test_fake_module_def" in removed
        assert "_test_fake_module_abc" not in sys.modules

    def test_returns_empty_when_no_match(self):
        removed = cache_manager.purge_modules(["_nonexistent_prefix_xyz_"])
        assert removed == []


class TestPurgeKernelModules:
    def test_runs_without_error(self):
        cache_manager.purge_kernel_modules()


class TestGpuSyncAndFlush:
    def test_does_not_crash_without_gpu(self):
        cache_manager.gpu_sync_and_flush()


class TestGpuWarmup:
    def test_does_not_crash_without_gpu(self):
        cache_manager.gpu_warmup()


class TestIsolatedGradingEnv:
    def test_sets_all_env_vars(self):
        os.environ.pop("TRITON_CACHE_DIR", None)
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        os.environ.pop("AMD_COMGR_CACHE_DIR", None)

        with cache_manager.isolated_grading_env(warmup_gpu=False) as dirs:
            assert "TRITON_CACHE_DIR" in os.environ
            assert "TORCHINDUCTOR_CACHE_DIR" in os.environ
            assert "AMD_COMGR_CACHE_DIR" in os.environ
            assert "triton_cache" in dirs

        assert "TRITON_CACHE_DIR" not in os.environ
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
        assert "AMD_COMGR_CACHE_DIR" not in os.environ

    def test_yields_dict_with_cache_paths(self):
        with cache_manager.isolated_grading_env(warmup_gpu=False) as dirs:
            assert isinstance(dirs, dict)
            assert "triton_cache" in dirs

    def test_optional_layers(self):
        with cache_manager.isolated_grading_env(
            clear_torch=False, clear_comgr=False, warmup_gpu=False
        ) as dirs:
            assert "triton_cache" in dirs


class TestIsolatedBenchmarkEnv:
    def test_sets_all_env_vars(self):
        os.environ.pop("TRITON_CACHE_DIR", None)
        with cache_manager.isolated_benchmark_env() as dirs:
            assert "TRITON_CACHE_DIR" in os.environ
            assert isinstance(dirs, dict)
        assert "TRITON_CACHE_DIR" not in os.environ


class TestNoopCtx:
    def test_yields_none(self):
        with cache_manager._noop_ctx() as val:
            assert val is None


# ============================================================================
# config_generator.py tests
# ============================================================================


class TestDetectKernelType:
    def test_detects_from_dir_name(self, tmp_path):
        task_dir = tmp_path / "flash_attn_prefill_vllm_gptoss"
        task_dir.mkdir()
        result = config_generator._detect_kernel_type(task_dir)
        if config_generator.KERNEL_MAP:
            assert result == "flash_attn_prefill"
        else:
            assert result is None or result == "flash_attn_prefill"

    def test_returns_none_for_unknown(self, tmp_path):
        task_dir = tmp_path / "unknown_xyz_task"
        task_dir.mkdir()
        assert config_generator._detect_kernel_type(task_dir) is None


class TestDetectFramework:
    def test_detects_vllm_from_name(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_model"
        task_dir.mkdir()
        assert config_generator._detect_framework(task_dir) == "vllm"

    def test_detects_sglang_from_name(self, tmp_path):
        task_dir = tmp_path / "attn_sglang_deepseek"
        task_dir.mkdir()
        assert config_generator._detect_framework(task_dir) == "sglang"

    def test_defaults_to_vllm(self, tmp_path):
        task_dir = tmp_path / "some_task"
        task_dir.mkdir()
        assert config_generator._detect_framework(task_dir) == "vllm"


class TestFindSolution:
    def test_finds_solution_py(self, tmp_path):
        (tmp_path / "solution.py").write_text("# sol")
        assert config_generator._find_solution(tmp_path).name == "solution.py"

    def test_finds_solution_hip(self, tmp_path):
        (tmp_path / "solution.hip").write_text("// sol")
        assert config_generator._find_solution(tmp_path).name == "solution.hip"

    def test_returns_none_when_missing(self, tmp_path):
        assert config_generator._find_solution(tmp_path) is None


class TestSolutionHash:
    def test_returns_hex_string(self, tmp_path):
        sol = tmp_path / "solution.py"
        sol.write_text("import torch\n")
        h = config_generator._solution_hash(sol)
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self, tmp_path):
        s1 = tmp_path / "s1.py"
        s2 = tmp_path / "s2.py"
        s1.write_text("version_a")
        s2.write_text("version_b")
        assert config_generator._solution_hash(s1) != config_generator._solution_hash(s2)


class TestGenerateConfig:
    def test_generates_valid_config(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_gptoss"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# optimized moe kernel")

        config = config_generator.generate_config(task_dir, kernel_type="fused_moe")
        assert config["gpu"]["device"] == 0
        assert config["correctness"]["mode"] in ("pytorch", "library_test", "accordo")
        assert config["performance"]["mode"] == "magpie_builtin"
        assert config["_pipeline_metadata"]["kernel_type"] == "fused_moe"
        assert config["_pipeline_metadata"]["tamper_protected"] is True
        assert config["_pipeline_metadata"]["solution_hash"] != ""

    def test_raises_when_kernel_type_unknown(self, tmp_path):
        task_dir = tmp_path / "unknown_xyz"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        with pytest.raises(ValueError, match="Cannot determine kernel_type"):
            config_generator.generate_config(task_dir)

    def test_uses_baseline_path_override(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_test"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = config_generator.generate_config(
            task_dir, kernel_type="fused_moe", baseline_path="/custom/baseline.py"
        )
        assert config["baseline"]["path"] == "/custom/baseline.py"

    def test_framework_detection(self, tmp_path):
        task_dir = tmp_path / "fused_moe_sglang_model"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = config_generator.generate_config(task_dir, kernel_type="fused_moe")
        assert config["_pipeline_metadata"]["framework"] == "sglang"


class TestWriteConfig:
    def test_writes_config_file(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_test"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")

        path = config_generator.write_config(task_dir, kernel_type="fused_moe")
        assert path.exists()
        assert path.name == "config.yaml"

        content = path.read_text()
        assert "magpie_builtin" in content
        assert "tamper_protected" in content

    def test_overwrites_existing_config(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_test"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        (task_dir / "config.yaml").write_text("old: config\n")

        config_generator.write_config(task_dir, kernel_type="fused_moe")
        content = (task_dir / "config.yaml").read_text()
        assert "old: config" not in content
        assert "magpie_builtin" in content


class TestValidateConfig:
    def test_missing_config_is_invalid(self, tmp_path):
        result = config_generator.validate_config(tmp_path)
        assert result.valid is False
        assert any("missing" in e for e in result.errors)

    def test_pipeline_generated_config_is_valid(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_test"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config_generator.write_config(task_dir, kernel_type="fused_moe")

        result = config_generator.validate_config(task_dir)
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_agent_config_without_metadata_gets_warning(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        (task_dir / "config.yaml").write_text(
            "gpu:\n  device: 0\nbaseline:\n  path: ./baseline.py\n"
        )

        result = config_generator.validate_config(task_dir)
        assert any("not generated by the pipeline" in w for w in result.warnings)

    def test_suspicious_correctness_command_flagged(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = {
            "correctness": {"command": "echo PASS"},
            "performance": {"command": "python bench.py"},
        }
        (task_dir / "config.yaml").write_text(json.dumps(config))

        result = config_generator.validate_config(task_dir)
        assert result.valid is False
        assert any("echo" in e for e in result.errors)

    def test_suspicious_exit0_correctness_flagged(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = {
            "correctness": {"command": "exit 0"},
            "performance": {"mode": "magpie_builtin"},
        }
        (task_dir / "config.yaml").write_text(json.dumps(config))

        result = config_generator.validate_config(task_dir)
        assert result.valid is False
        assert any("exit 0" in e for e in result.errors)

    def test_suspicious_performance_sleep_flagged(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = {
            "correctness": {"mode": "magpie_builtin"},
            "performance": {"command": "sleep 10 && echo done"},
        }
        (task_dir / "config.yaml").write_text(json.dumps(config))

        result = config_generator.validate_config(task_dir)
        assert result.valid is False
        assert any("sleep" in e for e in result.errors)

    def test_suspicious_print_pass_flagged(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = {
            "correctness": {"command": "python -c \"print('PASS')\""},
            "performance": {"mode": "magpie_builtin"},
        }
        (task_dir / "config.yaml").write_text(json.dumps(config))

        result = config_generator.validate_config(task_dir)
        assert result.valid is False

    def test_solution_hash_mismatch_warned(self, tmp_path):
        task_dir = tmp_path / "fused_moe_vllm_test"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# original")
        config_generator.write_config(task_dir, kernel_type="fused_moe")

        (task_dir / "solution.py").write_text("# modified after config was written")
        result = config_generator.validate_config(task_dir)
        assert any("hash mismatch" in w for w in result.warnings)

    def test_devnull_correctness_flagged(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = {
            "correctness": {"command": "python test.py > /dev/null"},
            "performance": {"mode": "magpie_builtin"},
        }
        (task_dir / "config.yaml").write_text(json.dumps(config))

        result = config_generator.validate_config(task_dir)
        assert result.valid is False

    def test_clean_commands_not_flagged(self, tmp_path):
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# sol")
        config = {
            "correctness": {"command": "python -m pytest test_kernel.py"},
            "performance": {"command": "python benchmark.py --warmup 10 --iters 100"},
            "_pipeline_metadata": {"tamper_protected": True},
        }
        (task_dir / "config.yaml").write_text(json.dumps(config))

        result = config_generator.validate_config(task_dir)
        assert result.valid is True
        assert len(result.errors) == 0


class TestBaselinePaths:
    def test_all_kernel_types_have_vllm_baseline(self):
        for kt, paths in config_generator._BASELINE_PATHS.items():
            assert "vllm" in paths, f"{kt} missing vllm baseline path"
            assert paths["vllm"], f"{kt} has empty vllm baseline path"

    def test_baseline_paths_are_relative(self):
        for kt, paths in config_generator._BASELINE_PATHS.items():
            for fw, path in paths.items():
                assert not path.startswith("/"), \
                    f"{kt}/{fw} has absolute baseline path: {path}"


# ============================================================================
# Integration: kernel_grader with config validation
# ============================================================================


class TestGraderConfigIntegration:
    def test_agent_config_gets_regenerated(self, tmp_path, magpie_compare_json):
        """When agent writes a suspicious config, grader should regenerate it."""
        task_dir = tmp_path / "fused_moe_vllm_gptoss"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# optimized kernel\n")
        (task_dir / "config.yaml").write_text(
            json.dumps({
                "correctness": {"command": "echo PASS"},
                "performance": {"command": "echo 1.0"},
            })
        )

        import kernel_grader
        with patch("kernel_grader.run_magpie_compare", return_value=magpie_compare_json):
            result = kernel_grader.grade_task(
                task_dir, isolate_caches=False, trust_agent_config=False
            )

        regenerated_config = (task_dir / "config.yaml").read_text()
        assert "magpie_builtin" in regenerated_config
        assert "echo PASS" not in regenerated_config

    def test_trusted_mode_skips_validation(self, tmp_path, magpie_compare_json):
        """When trust_agent_config=True, config is used as-is."""
        task_dir = tmp_path / "test_task"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# optimized kernel\n")
        agent_config = "gpu:\n  device: 0\nbaseline:\n  path: ./baseline.py\n"
        (task_dir / "config.yaml").write_text(agent_config)

        import kernel_grader
        with patch("kernel_grader.run_magpie_compare", return_value=magpie_compare_json):
            result = kernel_grader.grade_task(
                task_dir, isolate_caches=False, trust_agent_config=True
            )

        assert (task_dir / "config.yaml").read_text() == agent_config

    def test_cache_isolation_flag_respected(self, tmp_path, magpie_compare_json):
        """When isolate_caches=True, TRITON_CACHE_DIR should be changed during grading."""
        task_dir = tmp_path / "fused_moe_vllm_test"
        task_dir.mkdir()
        (task_dir / "solution.py").write_text("# optimized kernel\n")
        config_generator.write_config(task_dir, kernel_type="fused_moe")

        import kernel_grader
        original_triton = os.environ.get("TRITON_CACHE_DIR")

        captured_triton = {}

        def mock_magpie(*args, **kwargs):
            captured_triton["during"] = os.environ.get("TRITON_CACHE_DIR")
            return magpie_compare_json

        with patch("kernel_grader.run_magpie_compare", side_effect=mock_magpie):
            kernel_grader.grade_task(
                task_dir, isolate_caches=True, trust_agent_config=True
            )

        assert captured_triton.get("during") is not None
        if original_triton:
            assert captured_triton["during"] != original_triton
        current_triton = os.environ.get("TRITON_CACHE_DIR")
        assert current_triton == original_triton
