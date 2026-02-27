#!/usr/bin/env python3
"""
kernel_grader.py — Kernel-level grader for the RL kernel-optimization sandbox.

Scoring (AgentKernelArena):
  compiled  → +20 pts
  correct   → +100 pts
  speedup S → +S×100 pts   (S = baseline_time / optimized_time)

Usage:
  python3 kernel_grader.py [--output-dir PATH] [--task TASK_ID] [--json]

Expected output/ layout:
  output/
    <task_id>/
      solution.hip | solution.py | solution.cu   ← optimized kernel
      config.yaml                                 ← task config
                                                    (baseline path, commands, etc.)

Each config.yaml follows the prompt template schema:

  gpu:
    device: 0
    arch: gfx950
  baseline:
    path: tools/rocm/aiter/aiter/fused_moe.py
  optimized:
    path: ./solution.py
  correctness:
    command: "pytest tests/ -k fused_moe -x"
  performance:
    command: "python bench_fused_moe.py --arch gfx950"
    warmup_iterations: 10
    iterations: 100

The grader reads this config, locates baseline + solution files, and runs
Magpie compare to evaluate compilation, correctness, and performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import (
    KernelResult,
    parse_task_config,
    run_magpie_compare,
    parse_compare_result,
    PTS_COMPILED, PTS_CORRECT,
)

REPO_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

SOLUTION_NAMES = ["solution.hip", "solution.py", "solution.cu", "kernel.hip", "kernel.py"]


def find_tasks(output_dir: Path) -> list[Path]:
    """Return task directories that contain a solution file."""
    tasks = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if any((d / s).exists() for s in SOLUTION_NAMES):
            tasks.append(d)
    return tasks


def find_solution(task_dir: Path) -> Path | None:
    for name in SOLUTION_NAMES:
        p = task_dir / name
        if p.exists():
            return p
    return None


def _parse_config(config_path: Path) -> dict:
    """Parse a task config.yaml and return the dict."""
    return parse_task_config(config_path)


def _detect_kernel_type(solution: Path) -> str:
    """Infer Magpie kernel type from the solution file extension."""
    ext = solution.suffix.lower()
    if ext in (".hip", ".cu"):
        return "hip"
    return "pytorch"


def grade_task(task_dir: Path, docker_image: str | None = None) -> KernelResult:
    """
    Grade a single task directory.

    Reads config.yaml, finds baseline + solution, runs Magpie compare.
    """
    task_id  = task_dir.name
    solution = find_solution(task_dir)
    config   = task_dir / "config.yaml"

    if solution is None:
        return KernelResult(task_id=task_id, error="no solution file found")

    if not config.exists():
        return KernelResult(
            task_id=task_id,
            compiled=True,
            error=f"config.yaml missing in {task_dir}; solution exists but cannot run Magpie without config",
        )

    cfg = _parse_config(config)

    baseline_path = cfg.get("baseline", {}).get("path", "")
    if not baseline_path:
        return KernelResult(
            task_id=task_id,
            error="config.yaml missing baseline.path",
        )

    if not Path(baseline_path).is_absolute():
        baseline_path = str(REPO_ROOT / baseline_path)

    optimized_path = cfg.get("optimized", {}).get("path", str(solution))
    if not Path(optimized_path).is_absolute():
        optimized_path = str(task_dir / optimized_path)

    testcase_cmd = cfg.get("correctness", {}).get("command")
    kernel_type = _detect_kernel_type(solution)

    raw = run_magpie_compare(
        baseline_path=baseline_path,
        optimized_path=optimized_path,
        testcase_cmd=testcase_cmd,
        kernel_type=kernel_type,
        working_dir=str(task_dir),
    )

    if "error" in raw:
        return KernelResult(task_id=task_id, raw=raw, error=raw["error"])

    compiled, correct, speedup = parse_compare_result(raw)
    return KernelResult(
        task_id=task_id,
        compiled=compiled,
        correct=correct,
        speedup=speedup,
        raw=raw,
    )


def grade_all(output_dir: Path, task_filter: str | None = None) -> list[KernelResult]:
    if not output_dir.exists():
        print(f"[kernel_grader] output dir not found: {output_dir}", file=sys.stderr)
        return []

    tasks = find_tasks(output_dir)
    if task_filter:
        tasks = [t for t in tasks if task_filter in t.name]

    if not tasks:
        print(f"[kernel_grader] no tasks found in {output_dir}", file=sys.stderr)
        return []

    results = []
    for task_dir in tasks:
        print(f"  grading {task_dir.name} ...", file=sys.stderr)
        r = grade_task(task_dir)
        results.append(r)
        status = "✓" if r.correct else ("?" if r.compiled else "✗")
        print(
            f"    [{status}] compiled={r.compiled} correct={r.correct} "
            f"speedup={r.speedup:.2f}× score={r.score:.0f}",
            file=sys.stderr,
        )

    return results


def summarise(results: list[KernelResult]) -> dict:
    if not results:
        return {"total_score": 0, "tasks": 0, "results": []}

    total     = sum(r.score    for r in results)
    compiled  = sum(r.compiled for r in results)
    correct   = sum(r.correct  for r in results)
    avg_sp    = sum(r.speedup  for r in results if r.correct) / max(1, correct)

    return {
        "total_score":   round(total, 2),
        "tasks":         len(results),
        "compiled":      compiled,
        "correct":       correct,
        "avg_speedup":   round(avg_sp, 4),
        "scoring_notes": {
            "compiled":  f"+{PTS_COMPILED} pts",
            "correct":   f"+{PTS_CORRECT} pts",
            "speedup":   "+speedup×100 pts",
        },
        "results": [r.to_dict() for r in results],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Kernel-level grader — scores optimized kernels via Magpie."
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Path to the output/ directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--task", default=None,
        help="Grade only tasks whose ID contains this string.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print full JSON summary to stdout.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"[kernel_grader] scanning {output_dir}", file=sys.stderr)

    results = grade_all(output_dir, task_filter=args.task)
    summary = summarise(results)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"  Kernel grader results")
        print(f"{'='*50}")
        print(f"  Tasks:        {summary['tasks']}")
        print(f"  Compiled:     {summary['compiled']} / {summary['tasks']}")
        print(f"  Correct:      {summary['correct']} / {summary['tasks']}")
        print(f"  Avg speedup:  {summary['avg_speedup']:.3f}×")
        print(f"  TOTAL SCORE:  {summary['total_score']:.1f} pts")
        print(f"{'='*50}")
        for r in results:
            d = r.to_dict()
            flag = "PASS" if d["correct"] else ("COMPILE" if d["compiled"] else "FAIL")
            print(f"  {flag:7s}  {d['task_id']:30s}  "
                  f"{d['speedup']:.2f}×  {d['score']:.0f} pts"
                  + (f"  [{d['error']}]" if d["error"] else ""))


if __name__ == "__main__":
    main()
