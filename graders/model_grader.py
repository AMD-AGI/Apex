#!/usr/bin/env python3
"""
model_grader.py — Model-level grader for the RL kernel-optimization sandbox.

Scoring:
  kernel score  (from kernel_grader, normalised to 0–1, weight 50%)
  + e2e improvement over baseline (tokens/sec ratio − 1, weight 50%)
  × 100  →  final score

Usage:
  python3 model_grader.py [--output-dir PATH] [--model MODEL_ID] [--json]

Expected output/ layout (same as kernel_grader, plus a benchmark config):
  output/
    <task_id>/
      solution.hip | solution.py     ← optimized kernel
      config.yaml                    ← Magpie compare config (kernel eval)
      benchmark.yaml                 ← Magpie benchmark config (e2e eval)

benchmark.yaml schema (Magpie benchmark mode):
  framework: sglang | vllm
  model: meta-llama/Llama-3.1-8B-Instruct   # HuggingFace model ID
  gpu: { device: 0 }
  baseline:
    framework_config: {}               # stock framework, no patch
  optimized:
    patch: ./solution.hip              # kernel patch to apply
  workload:
    input_len:  512
    output_len: 128
    num_prompts: 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import (
    KernelResult,
    ModelResult,
    run_magpie,
    parse_compare_result,
    parse_benchmark_result,
)
from kernel_grader import find_tasks, find_solution, grade_task

REPO_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

# Open-source models iterated over (SGLang/vLLM compatible)
DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "google/gemma-2-9b-it",
]


def grade_task_model(task_dir: Path) -> ModelResult:
    """Run kernel grader + e2e benchmark for one task directory."""
    task_id       = task_dir.name
    benchmark_cfg = task_dir / "benchmark.yaml"

    # ── 1. Kernel-level score ─────────────────────────────────────────────────
    kernel_result = grade_task(task_dir)          # reuse kernel_grader logic
    k_score       = kernel_result.score

    if kernel_result.error and not kernel_result.compiled:
        return ModelResult(
            model_id=task_id,
            error=kernel_result.error,
        )

    # ── 2. End-to-end model benchmark ────────────────────────────────────────
    if not benchmark_cfg.exists():
        # No benchmark config — fall back to kernel score only
        result = ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            e2e_throughput_ratio=0.0,
            raw={"note": "no benchmark.yaml; e2e score skipped"},
        )
        return result

    raw_bench = run_magpie(
        ["benchmark", "--config", str(benchmark_cfg)],
        timeout=600,
    )

    if "error" in raw_bench:
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            raw=raw_bench,
            error=raw_bench["error"],
        )

    e2e_ratio = parse_benchmark_result(raw_bench)
    return ModelResult(
        model_id=task_id,
        kernel_score=k_score,
        e2e_throughput_ratio=e2e_ratio,
        raw=raw_bench,
    )


def grade_all(output_dir: Path, model_filter: str | None = None) -> list[ModelResult]:
    if not output_dir.exists():
        print(f"[model_grader] output dir not found: {output_dir}", file=sys.stderr)
        return []

    tasks = find_tasks(output_dir)
    if model_filter:
        tasks = [t for t in tasks if model_filter in t.name]

    if not tasks:
        print(f"[model_grader] no tasks found in {output_dir}", file=sys.stderr)
        return []

    results = []
    for task_dir in tasks:
        print(f"  grading {task_dir.name} ...", file=sys.stderr)
        r = grade_task_model(task_dir)
        results.append(r)
        print(
            f"    kernel_score={r.kernel_score:.0f}  "
            f"e2e_ratio={r.e2e_throughput_ratio:.3f}  "
            f"score={r.score:.1f}"
            + (f"  [{r.error}]" if r.error else ""),
            file=sys.stderr,
        )

    return results


def summarise(results: list[ModelResult]) -> dict:
    if not results:
        return {"total_score": 0, "tasks": 0, "results": []}

    total         = sum(r.score for r in results)
    avg_k         = sum(r.kernel_score for r in results) / len(results)
    avg_e2e       = sum(r.e2e_throughput_ratio for r in results) / len(results)

    return {
        "total_score":        round(total, 2),
        "tasks":              len(results),
        "avg_kernel_score":   round(avg_k,   2),
        "avg_e2e_ratio":      round(avg_e2e, 4),
        "scoring_notes": {
            "formula": "score = (kernel_score/320 + max(0, e2e_ratio-1)) × 100",
            "models":  DEFAULT_MODELS,
        },
        "results": [r.to_dict() for r in results],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Model-level grader — scores kernels on end-to-end model performance."
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Path to the output/ directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", default=None,
        help="Grade only tasks whose ID contains this string.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print full JSON summary to stdout.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"[model_grader] scanning {output_dir}", file=sys.stderr)

    results = grade_all(output_dir, model_filter=args.model)
    summary = summarise(results)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n{'='*55}")
        print(f"  Model grader results")
        print(f"{'='*55}")
        print(f"  Tasks:           {summary['tasks']}")
        print(f"  Avg kernel score:{summary['avg_kernel_score']:.1f} pts")
        print(f"  Avg e2e ratio:   {summary['avg_e2e_ratio']:.3f}×")
        print(f"  TOTAL SCORE:     {summary['total_score']:.1f} pts")
        print(f"{'='*55}")
        for r in results:
            d = r.to_dict()
            print(
                f"  {d['model_id']:40s}  "
                f"k={d['kernel_score']:.0f}  "
                f"e2e={d['e2e_throughput_ratio']:.3f}×  "
                f"score={d['score']:.1f}"
                + (f"  [{d['error']}]" if d["error"] else "")
            )


if __name__ == "__main__":
    main()
