"""Synthetic tests for _reinject_kernel() gate logic.

Validates the reinjection decision path without needing GPU or Docker:
- speedup threshold gate (>= MIN_SPEEDUP_FOR_REINJECTION)
- compiled/correct prerequisites
- HIP monolithic SO guard
- successful reinjection file copy and metadata
"""
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from workload_optimizer import (
    KernelOptResult,
    WorkloadConfig,
    _reinject_kernel,
    MIN_SPEEDUP_FOR_REINJECTION,
)


def _make_task_dir(tmp: Path, sol_content: str = "# optimized\npass\n") -> Path:
    task_dir = tmp / "output" / "test_kernel"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "solution.py").write_text(sol_content)
    (task_dir / "baseline.py").write_text("# baseline\npass\n")
    return task_dir


def _make_config(tmp: Path) -> WorkloadConfig:
    output_dir = tmp / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return WorkloadConfig(
        output_dir=output_dir,
        results_dir=tmp,
    )


def test_reinjection_success():
    """Compiled, correct, speedup above threshold -> reinjected."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_dir = _make_task_dir(tmp)
        config = _make_config(tmp)
        opt = KernelOptResult(
            kernel_spec="rms_norm",
            category="triton",
            compiled=True,
            correct=True,
            speedup=1.10,
        )
        result = _reinject_kernel(opt, task_dir, config, origin_library="aiter")
        assert result is True, f"Expected reinjection to succeed, got {result}"
        assert opt.reinjected is True
        reinjected_dir = config.output_dir / "reinjected"
        assert reinjected_dir.exists()
        reinjected_files = [f for f in reinjected_dir.glob("rms_norm_*") if not f.name.endswith(".library")]
        assert len(reinjected_files) == 1, f"Expected 1 reinjected file, got {reinjected_files}"
        meta = reinjected_dir / f"{reinjected_files[0].name}.library"
        assert meta.exists()
        assert meta.read_text() == "aiter"
        print("PASS: test_reinjection_success")


def test_reinjection_below_threshold():
    """Speedup below MIN_SPEEDUP_FOR_REINJECTION -> NOT reinjected."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_dir = _make_task_dir(tmp)
        config = _make_config(tmp)
        opt = KernelOptResult(
            kernel_spec="rms_norm",
            category="triton",
            compiled=True,
            correct=True,
            speedup=1.03,
        )
        result = _reinject_kernel(opt, task_dir, config, origin_library="aiter")
        assert result is False, f"Speedup 1.03x should be below threshold {MIN_SPEEDUP_FOR_REINJECTION}x"
        assert opt.reinjected is False
        print("PASS: test_reinjection_below_threshold")


def test_reinjection_not_compiled():
    """compiled=False -> NOT reinjected, regardless of speedup."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_dir = _make_task_dir(tmp)
        config = _make_config(tmp)
        opt = KernelOptResult(
            kernel_spec="rms_norm",
            category="triton",
            compiled=False,
            correct=True,
            speedup=2.0,
        )
        result = _reinject_kernel(opt, task_dir, config, origin_library="aiter")
        assert result is False, "compiled=False should block reinjection"
        print("PASS: test_reinjection_not_compiled")


def test_reinjection_not_correct():
    """correct=False -> NOT reinjected, regardless of speedup."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_dir = _make_task_dir(tmp)
        config = _make_config(tmp)
        opt = KernelOptResult(
            kernel_spec="rms_norm",
            category="triton",
            compiled=True,
            correct=False,
            speedup=2.0,
        )
        result = _reinject_kernel(opt, task_dir, config, origin_library="aiter")
        assert result is False, "correct=False should block reinjection"
        print("PASS: test_reinjection_not_correct")


def test_reinjection_hip_monolithic_so():
    """HIP kernel in monolithic SO library -> NOT reinjected."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_dir = _make_task_dir(tmp)
        (task_dir / "solution.hip").write_text("// hip solution\n")
        config = _make_config(tmp)
        opt = KernelOptResult(
            kernel_spec="flash_attn_prefill",
            category="hip",
            compiled=True,
            correct=True,
            speedup=1.20,
        )
        with patch("workload_optimizer._is_hip_patchable", return_value=False):
            result = _reinject_kernel(opt, task_dir, config, origin_library="aiter")
        assert result is False, "HIP kernel in monolithic SO should block reinjection"
        print("PASS: test_reinjection_hip_monolithic_so")


def test_reinjection_no_solution_file():
    """No solution file in task_dir -> NOT reinjected."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_dir = tmp / "output" / "test_kernel"
        task_dir.mkdir(parents=True, exist_ok=True)
        config = _make_config(tmp)
        opt = KernelOptResult(
            kernel_spec="rms_norm",
            category="triton",
            compiled=True,
            correct=True,
            speedup=1.20,
        )
        result = _reinject_kernel(opt, task_dir, config, origin_library="aiter")
        assert result is False, "Missing solution file should block reinjection"
        print("PASS: test_reinjection_no_solution_file")


if __name__ == "__main__":
    test_reinjection_success()
    test_reinjection_below_threshold()
    test_reinjection_not_compiled()
    test_reinjection_not_correct()
    test_reinjection_hip_monolithic_so()
    test_reinjection_no_solution_file()
    print(f"\nAll 6 reinjection tests PASSED (threshold={MIN_SPEEDUP_FOR_REINJECTION}x)")
