#!/usr/bin/env python3
"""
test_gpu_kernel_grader.py - End-to-end GPU test for kernel_grader.py.

Runs the kernel grader against a standalone Triton fused MoE task,
executing inside the rocm/pytorch Docker container with GPU access.

This tests the real grading pipeline:
  kernel_grader.grade_task() → reads config.yaml → runs correctness/bench
  commands via Docker → parses JSON → returns KernelResult with score.

Usage:
    python3 tests/test_gpu_kernel_grader.py
    python3 tests/test_gpu_kernel_grader.py --docker rocm/pytorch:latest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TASK_DIR = REPO_ROOT / "output" / "test_fused_moe_gpu"
TASK_ID = "test_fused_moe_gpu"
DEFAULT_DOCKER_IMAGE = "rocm/pytorch:latest"

# Import the actual grader under test
sys.path.insert(0, str(REPO_ROOT / "graders"))
from kernel_grader import grade_task, find_solution, summarise, _parse_config
from score import PTS_COMPILED, PTS_CORRECT


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GPU test for kernel_grader.py"
    )
    parser.add_argument(
        "--docker", default=DEFAULT_DOCKER_IMAGE, metavar="IMAGE",
        help=f"Docker image for GPU execution (default: {DEFAULT_DOCKER_IMAGE})",
    )
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("  GPU Kernel Grader — End-to-End Test")
    print("=" * 60)
    print(f"  Task:    {TASK_ID}")
    print(f"  Docker:  {args.docker}")
    print(f"  Files:   {TASK_DIR}")
    print("")

    # ── Pre-flight: verify task files exist ────────────────────────────────────
    required = ["baseline.py", "solution.py", "test_solution.py",
                 "bench.py", "config.yaml"]
    for f in required:
        p = TASK_DIR / f
        if not p.exists():
            print(f"  FAIL: missing {p}")
            sys.exit(1)
        print(f"  [ok] {f} ({p.stat().st_size} bytes)")

    # ── Verify grader helpers work ────────────────────────────────────────────
    sol = find_solution(TASK_DIR)
    assert sol is not None, "find_solution should find solution.py"
    assert sol.name == "solution.py"
    print(f"  [ok] find_solution -> {sol.name}")

    config = _parse_config(TASK_DIR / "config.yaml")
    assert "correctness" in config, "config should have correctness section"
    assert "performance" in config, "config should have performance section"
    assert config["correctness"]["command"] == "python3 test_solution.py"
    assert config["performance"]["command"] == "python3 bench.py"
    print(f"  [ok] _parse_config -> correctness={config['correctness']['command']!r}")
    print("")

    # ── Run the grader (the function under test) ──────────────────────────────
    print("  Running kernel_grader.grade_task() ...")
    result = grade_task(TASK_DIR, docker_image=args.docker)

    # ── Run summarise to test batch reporting ─────────────────────────────────
    summary = summarise([result])

    # ── Report ────────────────────────────────────────────────────────────────
    print("")
    print("=" * 60)
    print("  RESULTS (from kernel_grader.grade_task)")
    print("=" * 60)
    print(f"  Task ID:    {result.task_id}")
    print(f"  Compiled:   {result.compiled}  "
          f"(+{PTS_COMPILED if result.compiled else 0} pts)")
    print(f"  Correct:    {result.correct}   "
          f"(+{PTS_CORRECT if result.correct else 0} pts)")
    if result.correct and result.raw:
        print(f"  Baseline:   {result.raw.get('baseline_ms', '?')} ms")
        print(f"  Optimized:  {result.raw.get('optimized_ms', '?')} ms")
    print(f"  Speedup:    {result.speedup:.2f}x  "
          f"(+{result.speedup * 100:.1f} pts)")
    print(f"  TOTAL:      {result.score:.1f} pts")
    if result.error:
        print(f"  Error:      {result.error}")
    print("=" * 60)

    # ── Summarise output ──────────────────────────────────────────────────────
    print(f"\n  summarise(): total_score={summary['total_score']}, "
          f"compiled={summary['compiled']}/{summary['tasks']}, "
          f"correct={summary['correct']}/{summary['tasks']}, "
          f"avg_speedup={summary['avg_speedup']:.3f}x")

    # ── Assertions ────────────────────────────────────────────────────────────
    print("\n  Assertions:")
    assert result.task_id == TASK_ID, f"task_id mismatch: {result.task_id}"
    print(f"    [ok] task_id == {TASK_ID!r}")

    assert result.compiled, "solution should compile"
    print(f"    [ok] compiled == True")

    assert result.correct, f"solution should be correct, got error: {result.error}"
    print(f"    [ok] correct == True")

    assert result.speedup > 1.0, f"Triton should beat naive PyTorch, got {result.speedup:.2f}x"
    print(f"    [ok] speedup={result.speedup:.2f}x > 1.0")

    assert result.score > PTS_COMPILED + PTS_CORRECT, \
        f"score should exceed compile+correct baseline, got {result.score:.1f}"
    print(f"    [ok] score={result.score:.1f} > {PTS_COMPILED + PTS_CORRECT}")

    assert result.error is None, f"unexpected error: {result.error}"
    print(f"    [ok] error is None")

    assert summary["total_score"] == result.score
    print(f"    [ok] summarise total_score matches")

    # ── JSON output ───────────────────────────────────────────────────────────
    print("\nJSON:")
    print(json.dumps(result.to_dict(), indent=2))

    print("\nAll assertions passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
