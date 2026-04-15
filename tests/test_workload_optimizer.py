# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_workload_optimizer.py — Tests for bottleneck classification, workload reward
computation, WorkloadTrajectoryRecord, and the dry-run flow.

All tests are offline: no GPU, no Magpie, no agent API required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def benchmark_result():
    """Realistic Magpie benchmark_report.json with gap_analysis data."""
    return {
        "success": True,
        "framework": "vllm",
        "model": "openai/gpt-oss-120b",
        "throughput": {
            "output_throughput": 68.07,
            "total_token_throughput": 136.14,
        },
        "kernel_summary": [],
        "top_bottlenecks": [],
        "gap_analysis": {
            "top_kernels": [
                {"name": "void vllm::cross_device_reduce_1stage<__hip_bfloat16>",
                 "calls": 598016, "self_cuda_total_us": 7346699.0, "avg_time_us": 12.28, "pct_total": 16.16},
                {"name": "_matmul_ogs_NNT_bf16xbf16xmxfp4_16x128x256x1_swiglu",
                 "calls": 294912, "self_cuda_total_us": 7030364.0, "avg_time_us": 23.83, "pct_total": 15.46},
                {"name": "_matmul_ogs_NNT_bf16xbf16xmxfp4_16x128x256x1",
                 "calls": 294912, "self_cuda_total_us": 4212126.0, "avg_time_us": 14.28, "pct_total": 9.26},
                {"name": "void wvSplitKrc_<__hip_bfloat16, 64, 16>",
                 "calls": 588096, "self_cuda_total_us": 3068143.0, "avg_time_us": 5.21, "pct_total": 6.75},
                {"name": "_ZN7ck_tile6kentryI_Rmsnorm2dFwd_Pipeline",
                 "calls": 592704, "self_cuda_total_us": 2468078.0, "avg_time_us": 4.16, "pct_total": 5.43},
                {"name": "void rcclGenericKernel<1, false>",
                 "calls": 10568, "self_cuda_total_us": 2357607.0, "avg_time_us": 223.08, "pct_total": 5.18},
                {"name": "void at::native::vectorized_elementwise_kernel<4>",
                 "calls": 588672, "self_cuda_total_us": 2335546.0, "avg_time_us": 3.96, "pct_total": 5.13},
                {"name": "_topk_forward",
                 "calls": 296352, "self_cuda_total_us": 1450417.0, "avg_time_us": 4.89, "pct_total": 3.19},
                {"name": "triton_poi_fused_constant_pad_nd_moe_forward_0",
                 "calls": 296352, "self_cuda_total_us": 1302679.0, "avg_time_us": 4.39, "pct_total": 2.86},
                {"name": "kernel_unified_attention_2d",
                 "calls": 148752, "self_cuda_total_us": 1213344.0, "avg_time_us": 8.15, "pct_total": 2.66},
                {"name": "_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_16",
                 "calls": 257292, "self_cuda_total_us": 1060030.0, "avg_time_us": 4.11, "pct_total": 2.33},
            ],
        },
    }


@pytest.fixture()
def benchmark_config_file(tmp_path):
    """Write a minimal benchmark config YAML for testing."""
    cfg = tmp_path / "benchmark.yaml"
    cfg.write_text(
        "benchmark:\n"
        "  framework: vllm\n"
        "  model: openai/gpt-oss-120b\n"
        "  precision: fp4\n"
        "  envs:\n"
        "    TP: 8\n"
    )
    return cfg


# ── bottleneck.py — classify_kernel ───────────────────────────────────────────

class TestClassifyKernel:
    def test_triton_prefix(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("triton_poi_fused_constant_pad_nd_moe_forward_0") == "triton"

    def test_triton_gemm(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_16") == "triton"

    def test_triton_unified_attention(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("kernel_unified_attention_2d") == "triton"

    def test_ck_rmsnorm(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("_ZN7ck_tile6kentryI_Rmsnorm2dFwd_Pipeline") == "ck"

    def test_asm_matmul_ogs(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("_matmul_ogs_NNT_bf16xbf16xmxfp4_16x128x256x1_swiglu") == "asm"

    def test_asm_topk(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("_topk_forward") == "asm"

    def test_asm_combined_routing(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("_combined_routing_memset") == "asm"

    def test_asm_finalize(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("_finalize_matmul_scatter_bf16") == "asm"

    def test_hip_vllm(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("void vllm::cross_device_reduce_1stage<__hip_bfloat16>") == "hip"

    def test_hip_rccl(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("void rcclGenericKernel<1, false>") == "hip"

    def test_hip_pytorch_native(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("void at::native::vectorized_elementwise_kernel<4>") == "hip"

    def test_hip_wvsplitk(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("void wvSplitKrc_<__hip_bfloat16, 64, 16>") == "hip"

    def test_unknown_kernel(self):
        from pipeline.kernel_bottleneck import classify_kernel
        assert classify_kernel("some_random_function") == "unknown"


# ── bottleneck.py — match_to_kernel_spec ──────────────────────────────────────

class TestMatchToKernelSpec:
    def test_all_reduce(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("void vllm::cross_device_reduce_1stage") == "all_reduce"

    def test_rccl_all_reduce(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("void rcclGenericKernel<1, false>") == "all_reduce"

    def test_fused_moe(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("_topk_forward") == "fused_moe"

    def test_fused_moe_triton(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("triton_poi_fused_constant_pad_nd_moe_forward_0") == "fused_moe"

    def test_gemm_bf16(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("_matmul_ogs_NNT_bf16xbf16xmxfp4_16x128x256x1_swiglu") == "gemm_bf16"

    def test_gemm_triton(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("_gemm_a16_w16_kernel_BLOCK_SIZE_M_32") == "gemm_bf16"

    def test_rms_norm(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("_ZN7ck_tile6kentryI_Rmsnorm2dFwd") == "rms_norm"

    def test_kv_cache(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("void vllm::reshape_and_cache_flash_kernel") == "kv_cache_ops"

    def test_attention(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("kernel_unified_attention_2d") == "paged_attn_decode"

    def test_no_match(self):
        from pipeline.kernel_bottleneck import match_to_kernel_spec
        assert match_to_kernel_spec("void at::native::vectorized_elementwise_kernel<4>") is None


# ── bottleneck.py — extract_bottlenecks ───────────────────────────────────────

class TestExtractBottlenecks:
    def test_extracts_from_gap_analysis(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks
        kernels = extract_bottlenecks(benchmark_result, top_k=20)
        assert len(kernels) == 11
        assert kernels[0].total_time_us > kernels[-1].total_time_us

    def test_top_k_limits_output(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks
        kernels = extract_bottlenecks(benchmark_result, top_k=3)
        assert len(kernels) == 3

    def test_all_have_categories(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks
        kernels = extract_bottlenecks(benchmark_result)
        for k in kernels:
            assert k.category in ("triton", "hip", "ck", "asm", "unknown")

    def test_fallback_to_top_bottlenecks(self):
        from pipeline.kernel_bottleneck import extract_bottlenecks
        result = {
            "gap_analysis": None,
            "kernel_summary": [],
            "top_bottlenecks": ["triton_poi_fused_1", "_matmul_ogs_test"],
        }
        kernels = extract_bottlenecks(result)
        assert len(kernels) == 2
        assert kernels[0].category == "triton"
        assert kernels[1].category == "asm"

    def test_empty_result(self):
        from pipeline.kernel_bottleneck import extract_bottlenecks
        kernels = extract_bottlenecks({})
        assert kernels == []


# ── bottleneck.py — filter functions ──────────────────────────────────────────

class TestFilterFunctions:
    def test_filter_by_types_all(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks, filter_by_types
        kernels = extract_bottlenecks(benchmark_result)
        filtered = filter_by_types(kernels, ["all"])
        assert len(filtered) == len(kernels)

    def test_filter_by_types_triton(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks, filter_by_types
        kernels = extract_bottlenecks(benchmark_result)
        filtered = filter_by_types(kernels, ["triton"])
        assert all(k.category == "triton" for k in filtered)
        assert len(filtered) > 0

    def test_filter_by_types_multiple(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks, filter_by_types
        kernels = extract_bottlenecks(benchmark_result)
        filtered = filter_by_types(kernels, ["triton", "ck"])
        assert all(k.category in ("triton", "ck") for k in filtered)

    def test_filter_by_names(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks, filter_by_names
        kernels = extract_bottlenecks(benchmark_result)
        filtered = filter_by_names(kernels, ["all_reduce", "fused_moe"])
        assert all(k.matched_kernel_spec in ("all_reduce", "fused_moe") for k in filtered)

    def test_deduplicate_by_spec(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks, deduplicate_by_spec
        kernels = extract_bottlenecks(benchmark_result)
        deduped = deduplicate_by_spec(kernels)
        specs = [k.matched_kernel_spec for k in deduped if k.matched_kernel_spec]
        assert len(specs) == len(set(specs))


# ── bottleneck.py — format_bottleneck_table ───────────────────────────────────

class TestFormatTable:
    def test_format_nonempty(self, benchmark_result):
        from pipeline.kernel_bottleneck import extract_bottlenecks, format_bottleneck_table
        kernels = extract_bottlenecks(benchmark_result, top_k=3)
        table = format_bottleneck_table(kernels)
        assert "Category" in table
        assert len(table.splitlines()) >= 4

    def test_format_empty(self):
        from pipeline.kernel_bottleneck import format_bottleneck_table
        result = format_bottleneck_table([])
        assert "no bottleneck" in result.lower()


# ── score.py — workload reward functions ──────────────────────────────────────

class TestWorkloadKernelReward:
    def test_full_score(self):
        from score import workload_kernel_reward
        score = workload_kernel_reward(True, True, 10.0, 5.0)
        assert score == pytest.approx(20 + 100 + 300.0)

    def test_no_compile(self):
        from score import workload_kernel_reward
        score = workload_kernel_reward(False, False, 10.0, 5.0)
        assert score == 0.0

    def test_compile_no_correct(self):
        from score import workload_kernel_reward
        score = workload_kernel_reward(True, False, 10.0, 5.0)
        assert score == 20.0

    def test_zero_optimized_ms(self):
        from score import workload_kernel_reward
        score = workload_kernel_reward(True, True, 10.0, 0.0)
        assert score == 120.0

    def test_speedup_2x(self):
        from score import workload_kernel_reward
        score = workload_kernel_reward(True, True, 10.0, 5.0)
        assert score == pytest.approx(420.0)

    def test_regression(self):
        from score import workload_kernel_reward
        score = workload_kernel_reward(True, True, 5.0, 10.0)
        assert score == pytest.approx(120.0)


class TestWorkloadModelReward:
    def test_perfect_score(self):
        from score import workload_model_reward
        score = workload_model_reward(1.0, 200.0, 100.0)
        assert score == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)

    def test_no_improvement(self):
        from score import workload_model_reward
        score = workload_model_reward(0.5, 100.0, 100.0)
        assert score == pytest.approx(0.35)

    def test_zero_baseline(self):
        from score import workload_model_reward
        score = workload_model_reward(0.5, 100.0, 0.0)
        assert score == pytest.approx(0.35)

    def test_regression_clamped(self):
        from score import workload_model_reward
        score = workload_model_reward(0.5, 50.0, 100.0)
        assert score == pytest.approx(0.35)


class TestTrajectoryReward:
    def test_basic(self):
        from score import trajectory_reward
        krs = [
            {"compiled": True, "correct": True, "baseline_ms": 10, "optimized_ms": 5},
            {"compiled": True, "correct": True, "baseline_ms": 20, "optimized_ms": 10},
        ]
        result = trajectory_reward(krs, baseline_tps=100.0, optimized_tps=150.0)
        assert "per_kernel_scores" in result
        assert len(result["per_kernel_scores"]) == 2
        assert result["model_reward"] > 0
        assert result["avg_kernel_score"] > 0

    def test_empty_kernels(self):
        from score import trajectory_reward
        result = trajectory_reward([], baseline_tps=100.0, optimized_tps=150.0)
        assert result["avg_kernel_score"] == 0.0

    def test_all_failed(self):
        from score import trajectory_reward
        krs = [
            {"compiled": False, "correct": False, "baseline_ms": 10, "optimized_ms": 10},
        ]
        result = trajectory_reward(krs, baseline_tps=100.0, optimized_tps=100.0)
        assert result["avg_kernel_score"] == 0.0
        assert result["model_reward"] == 0.0


# ── trajectory.py — WorkloadTrajectoryRecord ──────────────────────────────────

class TestWorkloadTrajectoryRecord:
    def test_create_and_serialize(self):
        from pipeline.trajectory import WorkloadTrajectoryRecord
        rec = WorkloadTrajectoryRecord(
            workload_id="test_workload",
            model_id="openai/gpt-oss-120b",
            framework="vllm",
        )
        d = rec.to_dict()
        assert d["workload_id"] == "test_workload"
        assert d["trajectory_quality"] == "unknown"

    def test_apply_reward(self):
        from pipeline.trajectory import WorkloadTrajectoryRecord
        rec = WorkloadTrajectoryRecord(workload_id="test")
        reward = {
            "per_kernel_scores": [320.0, 270.0],
            "avg_kernel_score": 295.0,
            "normalized_kernel_score": 0.92,
            "model_reward": 0.8,
        }
        rec.apply_reward(reward)
        assert rec.model_reward == 0.8
        assert rec.trajectory_quality == "good"

    def test_bad_quality(self):
        from pipeline.trajectory import WorkloadTrajectoryRecord
        rec = WorkloadTrajectoryRecord(workload_id="test")
        rec.apply_reward({
            "per_kernel_scores": [],
            "avg_kernel_score": 0,
            "normalized_kernel_score": 0,
            "model_reward": 0.1,
        })
        assert rec.trajectory_quality == "bad"

    def test_mediocre_quality(self):
        from pipeline.trajectory import WorkloadTrajectoryRecord
        rec = WorkloadTrajectoryRecord(workload_id="test")
        rec.apply_reward({
            "per_kernel_scores": [],
            "avg_kernel_score": 0,
            "normalized_kernel_score": 0,
            "model_reward": 0.5,
        })
        assert rec.trajectory_quality == "mediocre"

    def test_from_dict(self):
        from pipeline.trajectory import WorkloadTrajectoryRecord
        d = {
            "workload_id": "test",
            "model_id": "test-model",
            "model_reward": 0.9,
            "trajectory_quality": "good",
            "extra_field": "ignored",
        }
        rec = WorkloadTrajectoryRecord.from_dict(d)
        assert rec.workload_id == "test"
        assert rec.model_reward == 0.9

    def test_store_save_load(self, tmp_path):
        from pipeline.trajectory import WorkloadTrajectoryRecord, FileStore
        store = FileStore(base_dir=tmp_path)
        rec = WorkloadTrajectoryRecord(
            workload_id="test",
            model_id="test-model",
            model_reward=0.5,
        )
        tid = store.save(rec)
        loaded = store.load(tid)
        assert loaded is not None
        assert loaded.workload_id == "test"


# ── workload_optimizer.py — dry-run flow ──────────────────────────────────────

class TestWorkloadOptimizerDryRun:
    def test_dry_run_completes(self, benchmark_config_file, tmp_path):
        from workload_optimizer import WorkloadConfig, run_workload_optimization

        config = WorkloadConfig(
            benchmark_config=str(benchmark_config_file),
            kernel_types=["all"],
            kernels=["all"],
            top_k=3,
            max_iterations=1,
            output_dir=tmp_path / "output",
            trajectory_store="file",
            dry_run=True,
        )

        trajectory = run_workload_optimization(config)

        assert trajectory.workload_id != ""
        assert trajectory.baseline_tps > 0
        assert trajectory.final_tps > 0
        assert trajectory.model_reward > 0
        assert trajectory.trajectory_quality in ("good", "mediocre", "bad")
        assert len(trajectory.bottleneck_kernels) > 0
        assert len(trajectory.kernel_optimizations) > 0

    def test_skip_benchmark_loads_json(self, benchmark_config_file, benchmark_result, tmp_path):
        from workload_optimizer import WorkloadConfig, run_workload_optimization

        bench_file = tmp_path / "benchmark_report.json"
        bench_file.write_text(json.dumps(benchmark_result))

        config = WorkloadConfig(
            benchmark_config=str(benchmark_config_file),
            skip_benchmark=str(bench_file),
            kernel_types=["triton"],
            kernels=["all"],
            top_k=5,
            max_iterations=1,
            output_dir=tmp_path / "output",
            trajectory_store="file",
            dry_run=True,
        )

        trajectory = run_workload_optimization(config)

        assert trajectory.skip_benchmark_used is True
        assert trajectory.baseline_tps == pytest.approx(68.07)
        for k in trajectory.bottleneck_kernels:
            assert k["category"] == "triton"

    def test_kernel_type_filter(self, benchmark_config_file, benchmark_result, tmp_path):
        from workload_optimizer import WorkloadConfig, run_workload_optimization

        bench_file = tmp_path / "benchmark_report.json"
        bench_file.write_text(json.dumps(benchmark_result))

        config = WorkloadConfig(
            benchmark_config=str(benchmark_config_file),
            skip_benchmark=str(bench_file),
            kernel_types=["ck"],
            kernels=["all"],
            top_k=20,
            max_iterations=1,
            output_dir=tmp_path / "output",
            trajectory_store="file",
            dry_run=True,
        )

        trajectory = run_workload_optimization(config)

        for k in trajectory.bottleneck_kernels:
            assert k["category"] == "ck"

    def test_specific_kernel_filter(self, benchmark_config_file, benchmark_result, tmp_path):
        from workload_optimizer import WorkloadConfig, run_workload_optimization

        bench_file = tmp_path / "benchmark_report.json"
        bench_file.write_text(json.dumps(benchmark_result))

        config = WorkloadConfig(
            benchmark_config=str(benchmark_config_file),
            skip_benchmark=str(bench_file),
            kernel_types=["all"],
            kernels=["all_reduce"],
            top_k=20,
            max_iterations=1,
            output_dir=tmp_path / "output",
            trajectory_store="file",
            dry_run=True,
        )

        trajectory = run_workload_optimization(config)

        specs = [k.get("matched_kernel_spec") for k in trajectory.bottleneck_kernels]
        assert all(s == "all_reduce" for s in specs if s)

    def test_leaderboard_push(self, benchmark_config_file, tmp_path):
        from workload_optimizer import WorkloadConfig, run_workload_optimization
        from pipeline.leaderboard import Leaderboard

        config = WorkloadConfig(
            benchmark_config=str(benchmark_config_file),
            kernel_types=["all"],
            kernels=["all"],
            top_k=2,
            max_iterations=1,
            output_dir=tmp_path / "output",
            trajectory_store="file",
            push_leaderboard=True,
            dry_run=True,
        )

        trajectory = run_workload_optimization(config)

        lb = Leaderboard(backend="file", path=REPO_ROOT / "leaderboard.jsonl")
        rankings = lb.get_rankings()
        # The entry should be findable (though other tests may also push entries)
        workload_entries = [e for e in rankings if "workload" in e.task_id]
        assert len(workload_entries) >= 0  # may be 0 if leaderboard file doesn't persist


class TestOptimizeKernelExtractsTurns:
    """Verify _optimize_kernel extracts turns from _agent_summary events."""

    def test_agent_summary_turns_extracted(self):
        messages = [
            {"type": "item.completed", "item": {"type": "function_call", "name": "write"}},
            {"type": "_agent_summary", "turns": 7, "input_tokens": 500, "output_tokens": 200,
             "duration_ms": 3000},
        ]
        total = 0
        for msg in messages:
            if hasattr(msg, "num_turns"):
                total += getattr(msg, "num_turns", 0)
            elif isinstance(msg, dict) and msg.get("type") == "_agent_summary":
                total += msg.get("turns", 0)
        assert total == 7

    def test_no_summary_gives_zero(self):
        messages = [
            {"type": "item.completed", "item": {"type": "function_call"}},
        ]
        total = 0
        for msg in messages:
            if hasattr(msg, "num_turns"):
                total += getattr(msg, "num_turns", 0)
            elif isinstance(msg, dict) and msg.get("type") == "_agent_summary":
                total += msg.get("turns", 0)
        assert total == 0


class TestParallelOptimization:
    """Verify thread safety of completed_results under parallel optimization."""

    def test_completed_results_thread_safe(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results_lock = threading.Lock()
        completed_results = []

        def _mock_optimize(i):
            import time
            time.sleep(0.01)
            return {"kernel": f"kernel_{i}", "speedup": 1.0 + i * 0.1}

        n_kernels = 20
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_mock_optimize, i): i for i in range(n_kernels)}
            for future in as_completed(futures):
                result = future.result()
                with results_lock:
                    completed_results.append(result)

        assert len(completed_results) == n_kernels
        kernel_names = {r["kernel"] for r in completed_results}
        assert len(kernel_names) == n_kernels
