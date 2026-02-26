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
      config.yaml                                 ← Magpie compare config
                                                    (see Magpie docs / examples)

Each config.yaml should follow the Magpie kernel config schema and point to
the baseline kernel. Example minimal config.yaml:

  gpu:
    device: 0
  baseline:
    path: /path/to/original/kernel.hip
  optimized:
    path: ./solution.hip
  correctness:
    command: "make test"
  performance:
    command: "make bench"
    iterations: 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))
from score import (
    KernelResult,
    run_magpie,
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


def grade_task(task_dir: Path) -> KernelResult:
    task_id  = task_dir.name
    solution = find_solution(task_dir)
    config   = task_dir / "config.yaml"

    if solution is None:
        return KernelResult(task_id=task_id, error="no solution file found")

    if not config.exists():
        return KernelResult(
            task_id=task_id,
            error=f"config.yaml missing in {task_dir}; cannot run Magpie",
        )

    # Run: magpie compare --config <config.yaml>
    raw = run_magpie(["compare", "--config", str(config)])

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
