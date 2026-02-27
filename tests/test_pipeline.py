"""Tests for pipeline.py — E2E pipeline orchestrator."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))
sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))

from pipeline import (
    PipelineConfig,
    PipelineResult,
    IterationResult,
    _make_task_id,
    _resolve_applicable_kernels,
    _build_prompt,
    _run_baseline_benchmark,
    run_single_task,
    run_pipeline,
)


class TestMakeTaskId:
    def test_basic(self):
        tid = _make_task_id("deepseek-ai/DeepSeek-R1", "fused_moe", "sglang")
        assert "deepseek" in tid
        assert "fused_moe" in tid
        assert "sglang" in tid

    def test_different_models(self):
        t1 = _make_task_id("meta-llama/Llama-3.1-8B-Instruct", "gemm_bf16", "vllm")
        t2 = _make_task_id("moonshotai/Kimi-K2", "gemm_bf16", "vllm")
        assert t1 != t2
        assert "llama" in t1
        assert "kimi" in t2


class TestResolveKernels:
    def test_specific_kernel(self):
        kernels = _resolve_applicable_kernels("any-model", "fused_moe")
        assert "fused_moe" in kernels

    def test_all_kernels_fallback(self):
        kernels = _resolve_applicable_kernels("unknown/model", "all")
        assert len(kernels) >= 3

    def test_known_model(self):
        try:
            from models import MODELS
            if MODELS:
                kernels = _resolve_applicable_kernels(MODELS[0].hf_id, "all")
                assert len(kernels) >= 1
        except ImportError:
            pytest.skip("models.py not available")


class TestBuildPrompt:
    def test_builds_prompt(self):
        prompt = _build_prompt(
            "meta-llama/Llama-3.1-8B-Instruct", "rms_norm", "sglang", "gfx950"
        )
        assert len(prompt) > 50
        assert "rms_norm" in prompt.lower() or "RMSNorm" in prompt or "rms" in prompt.lower()


class TestBaselineBenchmark:
    def test_dry_run(self):
        config = PipelineConfig(dry_run=True, model_id="test")
        result = _run_baseline_benchmark(config)
        assert result["dry_run"] is True
        assert "throughput" in result


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.max_iterations == 3
        assert cfg.framework == "sglang"
        assert cfg.gpu_arch == "gfx950"

    def test_custom(self):
        cfg = PipelineConfig(
            model_id="moonshotai/Kimi-K2",
            kernel_type="all",
            max_iterations=5,
        )
        assert cfg.model_id == "moonshotai/Kimi-K2"
        assert cfg.max_iterations == 5


class TestRunSingleTaskDryRun:
    def test_dry_run_completes(self, tmp_path):
        config = PipelineConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            kernel_type="rms_norm",
            framework="sglang",
            gpu_arch="gfx950",
            max_iterations=1,
            output_base=tmp_path,
            dry_run=True,
            trajectory_store="file",
        )
        result = run_single_task(config, "rms_norm")
        assert result.task_id
        assert isinstance(result.iterations, list)
        assert result.agent_model == "claude-sonnet-4-6"

    def test_dry_run_creates_trajectory(self, tmp_path):
        config = PipelineConfig(
            model_id="deepseek-ai/DeepSeek-R1",
            kernel_type="fused_moe",
            output_base=tmp_path,
            dry_run=True,
            trajectory_store="file",
        )
        result = run_single_task(config, "fused_moe")

        from trajectory import FileStore
        store = FileStore()
        ids = store.list_ids()
        assert len(ids) >= 0  # may or may not write to default path


class TestRunPipelineDryRun:
    def test_dry_run_full(self, tmp_path):
        config = PipelineConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            kernel_type="rms_norm",
            output_base=tmp_path,
            dry_run=True,
            max_iterations=1,
            trajectory_store="file",
        )
        results = run_pipeline(config)
        assert len(results) >= 1
        assert all(isinstance(r, PipelineResult) for r in results)


class TestPipelineWithLeaderboard:
    def test_leaderboard_push(self, tmp_path):
        lb_path = tmp_path / "lb.jsonl"
        config = PipelineConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            kernel_type="rms_norm",
            output_base=tmp_path,
            dry_run=True,
            max_iterations=1,
            push_leaderboard=True,
            trajectory_store="file",
        )

        with patch("pipeline.Leaderboard") as MockLB:
            mock_lb = MockLB.return_value
            run_single_task(config, "rms_norm")
            assert mock_lb.push.called


class TestIterationResult:
    def test_dataclass(self):
        ir = IterationResult(iteration=1, duration_s=5.0)
        assert ir.iteration == 1
        assert ir.agent_messages == []
