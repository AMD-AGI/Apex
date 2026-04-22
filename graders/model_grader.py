#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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

benchmark.yaml schema (from model_prompt.py template):
  framework: sglang | vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  gpu: { device: 0, arch: gfx950 }
  baseline:
    framework_config: {}
  optimized:
    patch: ./solution.hip
  workload:
    input_len:  512
    output_len: 128
    num_prompts: 200
    concurrency: 32
  precision: fp8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))
from score import (
    KernelResult,
    ModelResult,
    parse_benchmark_config,
    run_magpie_benchmark,
    run_command_benchmark,
    benchmark_ratio,
    parse_benchmark_result,
    extract_tps,
)
from kernel_grader import find_tasks, find_solution, grade_task

try:
    from models import MODELS
    DEFAULT_MODELS = [m.hf_id for m in MODELS]
except ImportError:
    DEFAULT_MODELS = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "google/gemma-2-9b-it",
    ]

REPO_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "output"


def baseline_benchmark(
    framework: str = "sglang",
    model: str = "",
    precision: str = "fp8",
    concurrency: int = 32,
    input_len: int = 1024,
    output_len: int = 512,
    timeout: int = 600,
) -> dict:
    """
    Run a baseline Magpie benchmark (no optimisations applied).

    Returns the raw benchmark result dict with throughput, latency, and
    top_bottlenecks for bottleneck identification.
    """
    result = run_magpie_benchmark(
        framework=framework,
        model=model,
        precision=precision,
        concurrency=concurrency,
        input_len=input_len,
        output_len=output_len,
        timeout=timeout,
    )
    tps = extract_tps(result)
    result["_baseline_tps"] = tps
    return result


def _run_configured_model_benchmark(cfg: dict, variant: str, timeout: int = 1800) -> dict:
    section = cfg.get(variant, {})
    if not isinstance(section, dict):
        return {"error": f"{variant} benchmark section must be a mapping"}
    command = section.get("command", "")
    if not command:
        return {"error": f"{variant} benchmark command missing"}

    bench_cfg = cfg.get("benchmark", {})
    env = {
        "APEX_BENCHMARK_VARIANT": variant,
        "APEX_BENCHMARK_FRAMEWORK": cfg.get("framework", ""),
        "APEX_BENCHMARK_MODEL": cfg.get("model", ""),
    }
    cwd = section.get("cwd") or bench_cfg.get("cwd")
    result_json = section.get("result_json") or bench_cfg.get("result_json")
    return run_command_benchmark(
        command=command,
        cwd=cwd,
        timeout=timeout,
        env_overrides=env,
        result_json=result_json,
    )


def grade_task_model(task_dir: Path) -> ModelResult:
    """
    Run kernel grader + e2e benchmark for one task directory.

    The e2e benchmark runs TWICE via Magpie:
      1. Baseline run  → baseline TPS (tokens/sec)
      2. Optimized run → optimized TPS
    The ratio (optimized / baseline) feeds into the model score.

    If the benchmark result already contains a pre-computed ratio
    (baseline_tps + optimized_tps), that is used directly.
    """
    task_id       = task_dir.name
    benchmark_cfg = task_dir / "benchmark.yaml"

    # ── 1. Kernel-level score ─────────────────────────────────────────────────
    kernel_result = grade_task(task_dir)
    k_score       = kernel_result.score

    if kernel_result.error and not kernel_result.compiled:
        return ModelResult(
            model_id=task_id,
            error=kernel_result.error,
        )

    # ── 2. End-to-end model benchmark ────────────────────────────────────────
    if not benchmark_cfg.exists():
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            e2e_throughput_ratio=0.0,
            raw={"note": "no benchmark.yaml; e2e score skipped"},
        )

    cfg = parse_benchmark_config(benchmark_cfg)
    bench_cfg = cfg.get("benchmark", {})
    if isinstance(bench_cfg, dict) and bench_cfg.get("kind", "").lower() == "command":
        baseline_bench = _run_configured_model_benchmark(cfg, "baseline")
        if "error" in baseline_bench:
            return ModelResult(
                model_id=task_id,
                kernel_score=k_score,
                raw=baseline_bench,
                error=f"baseline benchmark: {baseline_bench['error']}",
            )

        optimized_bench = _run_configured_model_benchmark(cfg, "optimized")
        if "error" in optimized_bench:
            return ModelResult(
                model_id=task_id,
                kernel_score=k_score,
                raw={"baseline": baseline_bench, "optimized": optimized_bench},
                error=f"optimized benchmark: {optimized_bench['error']}",
            )

        metric = bench_cfg.get("metric", "")
        goal = bench_cfg.get("goal", "maximize")
        ratio = benchmark_ratio(baseline_bench, optimized_bench, metric=metric, goal=goal)
        if ratio <= 0:
            metric_name = metric or ("throughput" if goal != "minimize" else "latency")
            return ModelResult(
                model_id=task_id,
                kernel_score=k_score,
                raw={"baseline": baseline_bench, "optimized": optimized_bench},
                error=f"configured benchmark produced no valid {metric_name} ratio",
            )

        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            e2e_throughput_ratio=ratio,
            raw={
                "benchmark_kind": "command",
                "metric": metric,
                "goal": goal,
                "baseline": baseline_bench,
                "optimized": optimized_bench,
            },
        )

    framework   = cfg.get("framework", "sglang")
    model       = cfg.get("model", "")
    precision   = cfg.get("precision", "fp8")
    workload    = cfg.get("workload", {})
    concurrency = workload.get("concurrency", 32)
    input_len   = workload.get("input_len", 1024)
    output_len  = workload.get("output_len", 512)

    common_kwargs = dict(
        framework=framework, model=model, precision=precision,
        concurrency=concurrency, input_len=input_len,
        output_len=output_len, timeout=600,
    )

    # ── 2a. Baseline benchmark ───────────────────────────────────────────────
    baseline_bench = run_magpie_benchmark(**common_kwargs)
    if "error" in baseline_bench:
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            raw=baseline_bench,
            error=f"baseline benchmark: {baseline_bench['error']}",
        )

    # Try pre-computed ratio first (legacy / comparison format)
    e2e_ratio = parse_benchmark_result(baseline_bench)
    if e2e_ratio > 0:
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            e2e_throughput_ratio=e2e_ratio,
            raw=baseline_bench,
        )

    # Single-run result: extract baseline TPS
    baseline_tps = extract_tps(baseline_bench)
    if baseline_tps <= 0:
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            raw=baseline_bench,
            error="baseline benchmark produced no throughput (0 TPS)",
        )

    # ── 2b. Optimized benchmark ──────────────────────────────────────────────
    optimized_bench = run_magpie_benchmark(**common_kwargs)
    if "error" in optimized_bench:
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            raw={"baseline": baseline_bench, "optimized": optimized_bench},
            error=f"optimized benchmark: {optimized_bench['error']}",
        )

    optimized_tps = extract_tps(optimized_bench)
    if optimized_tps <= 0:
        return ModelResult(
            model_id=task_id,
            kernel_score=k_score,
            raw={"baseline": baseline_bench, "optimized": optimized_bench},
            error="optimized benchmark produced no throughput (0 TPS)",
        )

    e2e_ratio = optimized_tps / baseline_tps
    return ModelResult(
        model_id=task_id,
        kernel_score=k_score,
        e2e_throughput_ratio=e2e_ratio,
        raw={
            "baseline_tps": baseline_tps,
            "optimized_tps": optimized_tps,
            "baseline": baseline_bench,
            "optimized": optimized_bench,
        },
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
