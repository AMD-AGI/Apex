#!/usr/bin/env python3
"""
workload_optimizer.py — Modular workload optimization trajectory pipeline.

Each step can be run independently via subcommands, or all at once via 'run'.
State is persisted in pipeline_state.json so steps can be resumed/rerun.

Subcommands:
    benchmark       Run initial E2E benchmark (or load existing results)
    identify        Identify & classify bottleneck kernels from benchmark
    list-kernels    Show identified kernels (for interactive selection)
    optimize        Optimize selected kernels (agent + grading loop)
    grade           Re-grade existing solutions without re-running agent
    integrate       Re-inject optimized kernels for final benchmark
    benchmark-final Run final E2E benchmark with optimized kernels
    score           Compute trajectory reward and push to leaderboard
    report          Generate markdown report and replication guide
    run             Full pipeline (all steps sequentially)

Usage:
    # Step-by-step (each step resumes from previous state):
    python workload_optimizer.py benchmark   -b config.yaml -r /results --skip-benchmark report.json
    python workload_optimizer.py identify    -r /results --kernel-types triton --top-k 20
    python workload_optimizer.py list-kernels -r /results
    python workload_optimizer.py optimize    -r /results --kernels fused_moe,gemm_bf16
    python workload_optimizer.py integrate   -r /results --kernels fused_moe,gemm_bf16
    python workload_optimizer.py benchmark-final -r /results
    python workload_optimizer.py score       -r /results --leaderboard
    python workload_optimizer.py report      -r /results

    # Full pipeline in one command:
    python workload_optimizer.py run -b config.yaml -r /results --kernel-types triton --leaderboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from score import (
    KernelResult,
    run_magpie_benchmark,
    run_magpie_compare,
    parse_compare_result,
    extract_tps,
    workload_kernel_reward,
    workload_model_reward,
    trajectory_reward,
)
from kernel_grader import grade_task, find_solution
from reflector import reflect, should_continue
from trajectory import WorkloadTrajectoryRecord, get_store
from leaderboard import Leaderboard, LeaderboardEntry
from bottleneck import (
    BottleneckKernel,
    extract_bottlenecks,
    filter_by_types,
    filter_by_names,
    deduplicate_by_spec,
    format_bottleneck_table,
)
from kernel_prompt import (
    KERNEL_SPECS,
    KERNEL_MAP,
    KernelSpec,
    applicable_kernels as _applicable_kernels,
    build_kernel_prompt as _build_rich_kernel_prompt,
    _format_sources_block,
    ARCH_MAP,
    DEFAULT_TARGET,
)
from models import MODELS, ModelConfig


MAGPIE_ROOT = Path(os.environ.get(
    "MAGPIE_ROOT",
    str(REPO_ROOT.parent / "Magpie"),
))
os.environ.setdefault("MAGPIE_ROOT", str(MAGPIE_ROOT))

# ---------------------------------------------------------------------------
# Pipeline state management
# ---------------------------------------------------------------------------

STATE_FILE = "pipeline_state.json"


class PipelineState:
    """Persistent state across pipeline steps, stored as pipeline_state.json."""

    def __init__(self, results_dir: Path):
        self.path = results_dir / STATE_FILE
        self._data: dict = {}
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value
        self.save()

    def update(self, d: dict):
        self._data.update(d)
        self.save()

    def require(self, key: str, step_name: str):
        """Raise if a prerequisite step hasn't been run."""
        v = self._data.get(key)
        if v is None:
            raise SystemExit(
                f"[error] '{key}' not found in state. Run the '{step_name}' step first."
            )
        return v

    @property
    def data(self) -> dict:
        return dict(self._data)

    @property
    def completed_steps(self) -> list[str]:
        return self._data.get("completed_steps", [])

    def mark_step(self, step: str):
        steps = self._data.setdefault("completed_steps", [])
        if step not in steps:
            steps.append(step)
        self.save()

    def record_step_time(self, step: str, elapsed_s: float):
        """Record wall-clock time for a pipeline step."""
        timings = self._data.setdefault("step_timings", {})
        timings[step] = round(elapsed_s, 2)
        self.save()

    @property
    def step_timings(self) -> dict[str, float]:
        return self._data.get("step_timings", {})


def _detect_kernel_python() -> str:
    """Auto-detect python with torch+triton for kernel execution."""
    candidates = [
        Path("/home/sirafati/Kernel/.venv/bin/python3"),
        Path.home() / "Kernel" / ".venv" / "bin" / "python3",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "python3"


SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.
Your task is to optimize GPU kernels for maximum performance on AMD Instinct GPUs.

You have access to the filesystem AND the following MCP tools:

MAGPIE (kernel evaluation framework):
- mcp__magpie__analyze: Analyze a kernel for correctness and performance profiling
- mcp__magpie__compare: Compare baseline vs optimized kernel (correctness + speedup)
- mcp__magpie__hardware_spec: Get GPU hardware specifications
- mcp__magpie__suggest_optimizations: Get optimization suggestions from analysis results
- mcp__magpie__benchmark: Run E2E LLM inference benchmarks

GPU INFO:
- mcp__gpu-info__get_gpu_info: Detect GPU and get hardware specs
- mcp__gpu-info__get_arch_optimization_hints: Architecture-specific optimization hints

SOURCE FINDER:
- mcp__source-finder__find_kernel_source: Find source code for a kernel type
- mcp__source-finder__classify_kernel: Classify kernel by name
- mcp__source-finder__find_ck_template: Find CK templates for an operation
- mcp__source-finder__identify_kernel_origin: Trace which library a kernel comes from

RAG SERVER:
- mcp__rag-server__search_kernel_optimization: Search optimization patterns
- mcp__rag-server__search_gpu_documentation: Search AMD GPU docs
- mcp__rag-server__get_optimization_snippet: Get code snippets for a pattern
- mcp__rag-server__analyze_kernel_for_optimization: Analyze kernel and suggest optimizations
- mcp__rag-server__get_optimization_playbook: Get complete optimization playbook

KERNEL PERF:
- mcp__kernel-perf__profile_kernel: Profile kernel with rocprof
- mcp__kernel-perf__roofline_analysis: Roofline model analysis
- mcp__kernel-perf__statistical_test: Statistical comparison of measurements

FUSION ADVISOR:
- mcp__fusion-advisor__detect_fusion_opportunities: Find kernel fusion opportunities
- mcp__fusion-advisor__generate_fused_kernel: Generate fused kernel implementations
- mcp__fusion-advisor__estimate_fusion_benefit: Estimate fusion benefit

ASM TOOLS:
- mcp__asm-tools__disassemble_kernel: Disassemble kernel to ISA
- mcp__asm-tools__analyze_isa: Analyze instruction mix and register usage
- mcp__asm-tools__count_instructions: Count instruction types

SKILLS (read these files BEFORE starting optimization):
- For Triton kernels: tools/skills/triton-kernel-optimization/SKILL.md
- For HIP/C++ kernels: tools/skills/hip-kernel-optimization/SKILL.md
- Architecture context: tools/skills/gpu-architecture-fundamentals/SKILL.md
- MI300/MI355 specifics: tools/skills/mi300-cdna3-architecture/SKILL.md
- AMD aiter patterns: tools/skills/aiter-reflection/SKILL.md
- Prior experiments: tools/skills/kernel-exp-history/SKILL.md
- Profiling guide: tools/skills/rocprof-compute/SKILL.md

WORKFLOW:
1. Read relevant skill files from tools/skills/ for domain knowledge
2. Read the baseline kernel code, understand it thoroughly
3. Use mcp__gpu-info__get_gpu_info to understand the target GPU
4. Use mcp__source-finder__find_kernel_source to find all implementations
5. Use mcp__rag-server__search_kernel_optimization for relevant patterns
6. Use mcp__fusion-advisor__detect_fusion_opportunities for fusion chances
7. Write an optimized version to solution.py
8. Use mcp__magpie__analyze to profile your solution
9. Use mcp__magpie__compare to compare baseline vs solution
10. Use mcp__asm-tools__analyze_isa for ISA-level analysis if needed
11. Iterate until speedup is substantial

Focus on: memory coalescing, LDS usage, MFMA utilization, register pressure,
bank conflicts, optimal block/tile sizes for the target architecture.

IMPORTANT:
- Write your optimized kernel to solution.py in the task directory
- The solution must be a self-contained Python file with a __main__ block that
  runs the kernel and prints PASS/FAIL
- Do NOT modify files outside the task directory
- Do NOT create new scripts — all evaluation uses Magpie MCP (analyze, compare)
- Do NOT hardcode kernel names — they are provided dynamically by the pipeline
- Only solutions with >5% speedup will be integrated into the final benchmark
- Use Magpie compare to verify correctness AND measure speedup every iteration
"""


@dataclass
class WorkloadConfig:
    benchmark_config: str = ""
    skip_benchmark: Optional[str] = None
    kernel_types: list[str] = field(default_factory=lambda: ["all"])
    kernels: list[str] = field(default_factory=lambda: ["all"])
    top_k: int = 10
    max_iterations: int = 5
    max_turns_per_iter: int = 25
    score_threshold: float = 300.0
    agent_model: str = "claude-sonnet-4-6"
    agent_version: str = "v1.0"
    agent_backend: str = "claude"
    framework: str = ""
    gpu_arch: str = "gfx950"
    docker_image: str = ""
    kernel_python: str = ""
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "output")
    results_dir: Optional[Path] = None
    trajectory_store: str = "file"
    push_leaderboard: bool = False
    dry_run: bool = False
    benchmark_timeout: int = 5400

    @property
    def effective_results_dir(self) -> Path:
        return self.results_dir or self.output_dir


@dataclass
class KernelOptResult:
    kernel_name: str = ""
    kernel_spec: str = ""
    category: str = ""
    compiled: bool = False
    correct: bool = False
    baseline_ms: float = 0.0
    optimized_ms: float = 0.0
    speedup: float = 0.0
    score: float = 0.0
    iterations_used: int = 0
    reinjected: bool = False
    agent_turns: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Step 1: E2E Benchmark
# ---------------------------------------------------------------------------

def _run_initial_benchmark(config: WorkloadConfig) -> dict:
    if config.skip_benchmark:
        skip_path = Path(config.skip_benchmark)
        if not skip_path.exists():
            print(f"  [error] --skip-benchmark path does not exist: {skip_path}")
            return {"error": f"skip-benchmark file not found: {skip_path}"}
        print(f"  Loading existing benchmark: {skip_path}")
        with open(skip_path) as f:
            return json.load(f)

    if config.dry_run:
        return _dry_run_benchmark_result()

    print(f"  Running Magpie benchmark with config: {config.benchmark_config}")
    print(f"  This may take 10-90 minutes for large models...")
    result = run_magpie_benchmark(
        framework=config.framework or "vllm",
        model="",
        benchmark_config_path=config.benchmark_config,
        timeout=config.benchmark_timeout,
    )
    return result


def _dry_run_benchmark_result() -> dict:
    return {
        "success": True, "dry_run": True, "framework": "vllm",
        "model": "dry-run-model",
        "throughput": {"output_throughput": 100.0, "total_token_throughput": 200.0},
        "kernel_summary": [], "top_bottlenecks": [],
        "gap_analysis": {
            "top_kernels": [
                {"name": "triton_poi_fused_constant_pad_nd_moe_forward_0",
                 "calls": 1000, "self_cuda_total_us": 5000000, "avg_time_us": 5000, "pct_total": 25.0},
                {"name": "kernel_unified_attention_2d",
                 "calls": 500, "self_cuda_total_us": 3000000, "avg_time_us": 6000, "pct_total": 15.0},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Step 2-4: Bottleneck extraction, classification, and selection
# ---------------------------------------------------------------------------

def _select_kernels(benchmark_result: dict, config: WorkloadConfig) -> list[BottleneckKernel]:
    all_bottlenecks = extract_bottlenecks(benchmark_result, top_k=config.top_k)

    print(f"\n  All bottleneck kernels ({len(all_bottlenecks)}):")
    print(format_bottleneck_table(all_bottlenecks))

    filtered = filter_by_types(all_bottlenecks, config.kernel_types)
    if len(filtered) != len(all_bottlenecks):
        type_str = ",".join(config.kernel_types)
        print(f"\n  After type filter ({type_str}): {len(filtered)} kernels")

    if config.kernels and "all" not in config.kernels:
        filtered = filter_by_names(filtered, config.kernels)
        print(f"  After name filter: {len(filtered)} kernels")

    deduped = deduplicate_by_spec(filtered)
    if len(deduped) != len(filtered):
        print(f"  After dedup by spec: {len(deduped)} unique kernel specs")

    # Only keep kernels with a matched spec (we need source paths to optimize)
    with_spec = [k for k in deduped if k.matched_kernel_spec]
    if len(with_spec) != len(deduped):
        skipped = len(deduped) - len(with_spec)
        print(f"  Removed {skipped} kernel(s) without known spec mapping")

    if with_spec:
        print(f"\n  Selected kernels for optimization:")
        print(format_bottleneck_table(with_spec))

    return with_spec


# ---------------------------------------------------------------------------
# Kernel source resolution
# ---------------------------------------------------------------------------

def _find_baseline_sources(kernel_spec: str) -> list[str]:
    """Find actual source file paths for a kernel spec from KERNEL_SPECS."""
    try:
        from kernel_prompt import KERNEL_SPECS
        for ks in KERNEL_SPECS:
            if ks.kernel_type == kernel_spec:
                paths = []
                for source in ks.sources:
                    if source.role != "impl":
                        continue
                    for p in source.paths:
                        full = REPO_ROOT / "tools" / "rocm" / p
                        if full.exists():
                            paths.append(str(full))
                if not paths:
                    for source in ks.sources:
                        for p in source.paths:
                            full = REPO_ROOT / "tools" / "rocm" / p
                            if full.exists():
                                paths.append(str(full))
                return paths
    except Exception:
        pass
    return []


def _read_source_code(path: str, max_lines: int = 500) -> str:
    """Read source file content, truncating if very long."""
    try:
        text = Path(path).read_text()
        lines = text.splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n... ({len(lines) - max_lines} more lines truncated)"
        return text
    except Exception as e:
        return f"(could not read: {e})"


def _extract_timing_from_raw(raw: dict) -> tuple[float, float]:
    """Extract (baseline_ms, optimized_ms) from a Magpie compare raw result.

    Checks multiple places in the result dict:
    1. Top-level baseline_ms / optimized_ms (set by _measure_speedup fallback)
    2. kernel_results list from Magpie compare (performance_result.summary.Duration)
    3. _benchmark_speedup if present, combined with optimized time
    """
    if not raw:
        return 0.0, 0.0

    b_ms = float(raw.get("baseline_ms", 0) or 0)
    o_ms = float(raw.get("optimized_ms", 0) or 0)
    if b_ms > 0 and o_ms > 0:
        return b_ms, o_ms

    results = raw.get("results", raw)
    kernel_results = results.get("kernel_results", [])
    if len(kernel_results) >= 2:
        b_ms = _time_from_kr(kernel_results[0])
        o_ms = _time_from_kr(kernel_results[-1])
        if b_ms > 0 and o_ms > 0:
            return b_ms, o_ms

    return b_ms, o_ms


def _time_from_kr(kr: dict) -> float:
    """Extract kernel duration in ms from a Magpie kernel_result entry."""
    perf = kr.get("performance_result") or kr.get("performance", {})
    if isinstance(perf, dict):
        for key in ("avg_time_ms", "mean_ms", "avg_ms", "time_ms", "median_ms"):
            v = perf.get(key)
            if v is not None and float(v) > 0:
                return float(v)
        summary = perf.get("summary", {})
        if isinstance(summary, dict):
            dur = summary.get("Duration")
            if dur is not None and float(dur) > 0:
                return float(dur)
        metrics = perf.get("metrics", {})
        for key in ("avg_time_ms", "mean_ms", "avg_ms"):
            v = metrics.get(key)
            if v is not None and float(v) > 0:
                return float(v)
        kernels = perf.get("kernels", [])
        if kernels and isinstance(kernels, list):
            total_dur = sum(float(k.get("duration", 0) or 0) for k in kernels)
            if total_dur > 0:
                return total_dur
    return 0.0


def _create_task_config(
    task_dir: Path,
    kernel: BottleneckKernel,
    config: WorkloadConfig,
    baseline_paths: list[str],
) -> Path:
    """Pre-create config.yaml in the task directory with baseline paths.

    If no local baseline.py exists, creates a minimal one that copies the
    original source and adds a __main__ block for standalone execution.
    """
    spec = kernel.matched_kernel_spec or "unknown"
    ext = ".py" if kernel.category == "triton" else ".hip"

    local_baseline = task_dir / f"baseline{ext}"
    if local_baseline.exists():
        baseline_path = f"./baseline{ext}"
    elif baseline_paths:
        # Copy the first baseline source into the task dir as baseline.py
        import shutil
        src = Path(baseline_paths[0])
        if src.exists():
            shutil.copy2(src, local_baseline)
            baseline_path = f"./baseline{ext}"
        else:
            baseline_path = baseline_paths[0]
    else:
        baseline_path = ""

    kernel_python = getattr(config, "kernel_python", "") or _detect_kernel_python()

    cfg = {
        "gpu": {"device": 0, "arch": config.gpu_arch},
        "baseline": {"path": baseline_path},
        "optimized": {"path": f"./solution{ext}"},
        "correctness": {"command": f"{kernel_python} solution{ext}"},
        "performance": {
            "command": kernel_python,
            "warmup_iterations": 10,
            "iterations": 100,
        },
    }

    config_path = task_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


# ---------------------------------------------------------------------------
# Step 5: Per-kernel optimization loop
# ---------------------------------------------------------------------------

def _build_kernel_prompt(
    kernel: BottleneckKernel,
    config: WorkloadConfig,
    benchmark_config: dict,
    task_dir: Path,
) -> str:
    """Build a prompt using the rich kernel_prompt.py template + actual source code."""
    spec = kernel.matched_kernel_spec or "unknown"
    model_id = benchmark_config.get("benchmark", {}).get("model", config.framework)
    framework = config.framework or "vllm"
    gpu_arch = config.gpu_arch or DEFAULT_TARGET
    baseline_sources = _find_baseline_sources(spec)

    # Try to build the rich prompt from kernel_prompt.py templates
    rich_prompt = _try_build_rich_prompt(spec, model_id, framework, gpu_arch)

    source_sections = []
    for src_path in baseline_sources[:3]:
        code = _read_source_code(src_path)
        source_sections.append(f"### Source: {src_path}\n```python\n{code}\n```")

    sources_text = "\n\n".join(source_sections) if source_sections else (
        "No source files found on disk. Use source-finder MCP to search for kernel "
        "implementations under tools/rocm/aiter/ and tools/rocm/composable_kernel/."
    )

    profiling_context = textwrap.dedent(f"""\
## Profiling Context

**Profiler kernel name:** `{kernel.name}`
**Category:** {kernel.category}
**Current GPU time:** {kernel.percent_total:.1f}% of total ({kernel.total_time_us/1000:.1f} ms over {kernel.calls} calls)
**Task directory:** `{task_dir}`

## Baseline Source Code (from disk)

{sources_text}

## Your Task

1. Read the skill files listed above for domain-specific optimization knowledge.
2. Read and understand the baseline kernel implementation above.
3. Use MCP tools: source-finder to find all implementations, rag-server for patterns,
   gpu-info for arch hints, fusion-advisor for fusion opportunities.
4. Identify performance bottlenecks (memory access patterns, compute utilization, occupancy).
5. Write an optimized version to: `{task_dir}/solution.py`
6. Use mcp__magpie__compare to validate correctness and measure speedup.
7. The config.yaml at `{task_dir}/config.yaml` already has the baseline path set.

## IMPORTANT Constraints
- Your solution must be functionally equivalent to the baseline (same inputs → same outputs).
- Do NOT modify files outside `{task_dir}/`.
- Focus on real performance improvements, not just code style changes.
- Include the kernel function with the same signature as the baseline.
""")

    if rich_prompt:
        return rich_prompt + "\n\n" + profiling_context
    return profiling_context


def _try_build_rich_prompt(
    kernel_spec: str, model_id: str, framework: str, gpu_arch: str,
) -> str:
    """Try to build a rich prompt using kernel_prompt.py templates.

    Returns the rich prompt text, or empty string if the model/kernel isn't found.
    """
    try:
        kernel = KERNEL_MAP.get(kernel_spec)
        if not kernel:
            return ""

        model = None
        for m in MODELS:
            if m.hf_id == model_id:
                model = m
                break
        if not model:
            for m in MODELS:
                if m.hf_id.split("/")[-1].lower() in model_id.lower():
                    model = m
                    break
        if not model:
            model = MODELS[0]

        result = _build_rich_kernel_prompt(
            model=model,
            kernel=kernel,
            framework=framework,
            gpu_arch=gpu_arch,
        )
        return result.get("prompt", "")
    except Exception as e:
        print(f"    [warn] Could not build rich prompt: {e}")
        return ""


def _make_kernel_task_id(kernel: BottleneckKernel, config: WorkloadConfig) -> str:
    spec = kernel.matched_kernel_spec or kernel.name[:30].replace(" ", "_")
    spec = spec.replace("::", "_").replace("<", "").replace(">", "")
    framework = config.framework or "vllm"
    return f"workload__{framework}__{spec}"


def _run_agent_iteration(
    task_dir: Path,
    prompt: str,
    config: WorkloadConfig,
    iteration: int,
    previous_reflection: str = "",
) -> tuple[list[dict], bool]:
    full_prompt = prompt
    if previous_reflection:
        full_prompt = previous_reflection + "\n\n---\n\n" + prompt

    if config.dry_run:
        solution = task_dir / "solution.py"
        if not solution.exists():
            solution.write_text(textwrap.dedent("""\
                import numpy as np

                def kernel_fn(*args, **kwargs):
                    return args[0] if args else None
            """))
        return [{"role": "assistant", "content": f"[dry-run] iteration {iteration}"}], True

    try:
        from agents.backends import run_agent_task
        messages, solution_written = run_agent_task(
            prompt=full_prompt,
            cwd=task_dir,
            model=config.agent_model,
            max_turns=config.max_turns_per_iter,
            agent=config.agent_backend,
            system_prompt=SYSTEM_PROMPT,
            solution_path=task_dir / "solution.py",
        )
        return messages, solution_written
    except Exception as e:
        print(f"    [agent] Error: {e}")
        import traceback
        traceback.print_exc()
        return [{"type": "error", "error": str(e)[:500]}], False


def _optimize_kernel(
    kernel: BottleneckKernel,
    config: WorkloadConfig,
    benchmark_config: dict,
) -> KernelOptResult:
    task_id = _make_kernel_task_id(kernel, config)
    task_dir = config.output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    spec = kernel.matched_kernel_spec or "unknown"
    print(f"\n    {'='*55}")
    print(f"    Optimizing: {spec} ({kernel.category})")
    print(f"    Task dir:   {task_dir}")
    print(f"    Profiler:   {kernel.name[:70]}")
    print(f"    GPU time:   {kernel.percent_total:.1f}%")
    print(f"    {'='*55}")

    opt_result = KernelOptResult(
        kernel_name=kernel.name,
        kernel_spec=spec,
        category=kernel.category,
    )

    baseline_sources = _find_baseline_sources(spec)
    if baseline_sources:
        print(f"    Baseline sources: {[Path(p).name for p in baseline_sources]}")
    else:
        print(f"    [warn] No baseline sources found for {spec}")

    _create_task_config(task_dir, kernel, config, baseline_sources)
    prompt = _build_kernel_prompt(kernel, config, benchmark_config, task_dir)

    best_kr: Optional[KernelResult] = None
    reflection_prompt = ""
    total_agent_turns = 0

    pre_existing_solution = find_solution(task_dir)
    if pre_existing_solution:
        print(f"    Pre-existing solution found: {pre_existing_solution.name}")
        print(f"    Skipping agent, going straight to grading.")

    for iteration in range(1, config.max_iterations + 1):
        print(f"\n    --- Iteration {iteration}/{config.max_iterations} ---")

        if pre_existing_solution and iteration == 1:
            solution_written = True
        else:
            print("    Running agent...")
            t0 = time.monotonic()
            messages, solution_written = _run_agent_iteration(
                task_dir, prompt, config, iteration, reflection_prompt,
            )
            agent_time = time.monotonic() - t0
            # Count agent turns from messages (look for ResultMessage.num_turns)
            for msg in messages:
                if hasattr(msg, "num_turns"):
                    total_agent_turns += getattr(msg, "num_turns", 0)
            print(f"    Agent completed in {agent_time:.1f}s")

            # Agent may have written a solution before crashing
            if not solution_written and find_solution(task_dir):
                print("    Agent crashed but solution file exists — grading it.")
                solution_written = True

        if not solution_written:
            print("    Agent did not write a solution.")
            opt_result.error = "Agent did not produce solution.py"
            if iteration < config.max_iterations:
                delay = min(5 * iteration, 15)
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
            continue

        print("    Grading with Magpie...")
        kr = grade_task(task_dir, docker_image=config.docker_image or None)
        print(f"      compiled={kr.compiled} correct={kr.correct} "
              f"speedup={kr.speedup:.2f}x score={kr.score:.0f}")
        if kr.error:
            print(f"      error: {kr.error[:200]}")

        if best_kr is None or kr.score > best_kr.score:
            best_kr = kr

        opt_result.iterations_used = iteration

        if kr.score >= config.score_threshold:
            print(f"    Target reached: score={kr.score:.0f} >= {config.score_threshold}")
            break

        reflection_prompt = reflect(
            kr, task_dir, iteration,
            kernel_type=spec,
            target_speedup=config.score_threshold / 100.0,
        )

        if not should_continue(kr, iteration, config.max_iterations, config.score_threshold):
            print(f"    Stopping: score={kr.score:.0f}")
            break

    if best_kr:
        raw = best_kr.raw or {}
        opt_result.compiled = best_kr.compiled
        opt_result.correct = best_kr.correct
        opt_result.speedup = best_kr.speedup
        opt_result.score = best_kr.score
        b_ms, o_ms = _extract_timing_from_raw(raw)
        opt_result.baseline_ms = b_ms
        opt_result.optimized_ms = o_ms
        opt_result.agent_turns = total_agent_turns
        opt_result.error = best_kr.error
    elif config.dry_run:
        opt_result.compiled = True
        opt_result.correct = True
        opt_result.speedup = 1.5
        opt_result.score = 270.0
        opt_result.baseline_ms = 10.0
        opt_result.optimized_ms = 6.67
        opt_result.iterations_used = 1

    return opt_result


# ---------------------------------------------------------------------------
# Step 6: Kernel re-injection
# ---------------------------------------------------------------------------

def _reinject_kernel(
    opt_result: KernelOptResult,
    task_dir: Path,
    config: WorkloadConfig,
) -> bool:
    if not opt_result.compiled or not opt_result.correct:
        print(f"    Skipping re-injection for {opt_result.kernel_spec}: "
              f"compiled={opt_result.compiled} correct={opt_result.correct}")
        return False

    solution = find_solution(task_dir)
    if solution is None:
        print(f"    No solution file found in {task_dir}")
        return False

    inject_dir = config.output_dir / "reinjected"
    inject_dir.mkdir(parents=True, exist_ok=True)

    dest = inject_dir / f"{opt_result.kernel_spec}_{solution.name}"
    shutil.copy2(solution, dest)
    print(f"    Re-injected: {solution.name} -> {dest}")
    opt_result.reinjected = True
    return True


# ---------------------------------------------------------------------------
# Step 7: Final E2E benchmark
# ---------------------------------------------------------------------------

def _run_final_benchmark(config: WorkloadConfig) -> dict:
    if config.dry_run:
        return {
            "success": True, "dry_run": True,
            "throughput": {"output_throughput": 150.0, "total_token_throughput": 300.0},
        }

    if config.skip_benchmark:
        print(f"  Loading existing benchmark: {config.skip_benchmark}")
        return json.loads(Path(config.skip_benchmark).read_text())

    print(f"  Running final E2E benchmark with optimized kernels...")
    result = run_magpie_benchmark(
        framework=config.framework or "vllm",
        model="",
        benchmark_config_path=config.benchmark_config,
        timeout=config.benchmark_timeout,
    )
    return result


# ---------------------------------------------------------------------------
# Leaderboard management
# ---------------------------------------------------------------------------

def _save_leaderboard(
    entry: LeaderboardEntry,
    results_dir: Path,
) -> None:
    """Append entry to leaderboard.jsonl and regenerate leaderboard.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "leaderboard.jsonl"
    json_path = results_dir / "leaderboard.json"

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    entries = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LeaderboardEntry.from_dict(json.loads(line)))

    agents: dict[str, list[LeaderboardEntry]] = {}
    for e in entries:
        agents.setdefault(e.agent_version, []).append(e)

    agent_summaries = []
    for version, runs in sorted(agents.items()):
        avg_arena = sum(r.arena_score for r in runs) / len(runs)
        avg_sp = sum(r.speedup for r in runs) / len(runs)
        agent_summaries.append({
            "agent_version": version,
            "runs": len(runs),
            "avg_arena_score": round(avg_arena, 2),
            "avg_speedup": round(avg_sp, 4),
            "best_arena_score": round(max(r.arena_score for r in runs), 2),
        })

    top = sorted(entries, key=lambda e: e.arena_score, reverse=True)[:20]

    summary = {
        "total_runs": len(entries),
        "agents": agent_summaries,
        "top_scores": [e.to_dict() for e in top],
        "latest_entry": entry.to_dict(),
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Leaderboard saved: {json_path}")
    print(f"  Total runs: {len(entries)}, arena_score: {entry.arena_score:.2f}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    trajectory: WorkloadTrajectoryRecord,
    kernel_results: list[KernelOptResult],
    config: WorkloadConfig,
    baseline_result: dict,
    final_result: dict,
    reward: dict,
    results_dir: Path,
    step_timings: dict[str, float] | None = None,
) -> Path:
    """Generate a comprehensive markdown report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    baseline_tps = trajectory.baseline_tps
    final_tps = trajectory.final_tps
    tps_ratio = final_tps / baseline_tps if baseline_tps > 0 else 0.0

    latency = baseline_result.get("latency", {})
    ttft = latency.get("ttft", {})
    tpot = latency.get("tpot", {})
    e2el = latency.get("e2el", {})

    lines = [
        f"# Workload Optimization Report",
        f"",
        f"**Generated:** {ts}",
        f"**Trajectory ID:** `{trajectory.trajectory_id}`",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Workload | `{trajectory.workload_id}` |",
        f"| Model | `{trajectory.model_id}` |",
        f"| Framework | `{trajectory.framework}` |",
        f"| GPU | `{config.gpu_arch}` |",
        f"| Agent | `{config.agent_model}` ({config.agent_version}) |",
        f"| Benchmark config | `{config.benchmark_config}` |",
        f"| Max iterations/kernel | {config.max_iterations} |",
        f"| Max turns/iteration | {config.max_turns_per_iter} |",
        f"| Kernel type filter | {','.join(config.kernel_types)} |",
        f"| Skip benchmark | {'Yes' if config.skip_benchmark else 'No'} |",
        f"",
        f"## Baseline E2E Performance",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Output throughput | {baseline_tps:.2f} tok/s |",
    ]

    if ttft:
        lines += [
            f"| TTFT mean | {ttft.get('mean_ms', 0):.2f} ms |",
            f"| TTFT p99 | {ttft.get('p99_ms', 0):.2f} ms |",
        ]
    if tpot:
        lines += [
            f"| TPOT mean | {tpot.get('mean_ms', 0):.2f} ms |",
            f"| TPOT p99 | {tpot.get('p99_ms', 0):.2f} ms |",
        ]
    if e2el:
        lines += [
            f"| E2E latency mean | {e2el.get('mean_ms', 0):.2f} ms |",
            f"| E2E latency p99 | {e2el.get('p99_ms', 0):.2f} ms |",
        ]

    lines += [
        f"",
        f"## Bottleneck Kernels Identified",
        f"",
        f"| # | Category | Spec | Time% | Calls | Name |",
        f"|---|----------|------|-------|-------|------|",
    ]

    for i, bk in enumerate(trajectory.bottleneck_kernels or [], 1):
        name = bk.get("name", "")[:60]
        lines.append(
            f"| {i} | {bk.get('category', '')} | {bk.get('matched_kernel_spec', '-')} | "
            f"{bk.get('percent_total', 0):.2f}% | {bk.get('calls', 0)} | `{name}` |"
        )

    lines += [
        f"",
        f"## Kernel Optimization Results",
        f"",
    ]

    for kr in kernel_results:
        status = "PASS" if kr.correct else ("COMPILE" if kr.compiled else "FAIL")
        lines += [
            f"### {kr.kernel_spec} ({kr.category})",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Status | **{status}** |",
            f"| Compiled | {'Yes' if kr.compiled else 'No'} |",
            f"| Correct | {'Yes' if kr.correct else 'No'} |",
            f"| Speedup | {kr.speedup:.3f}x |",
            f"| Score | {kr.score:.1f} |",
            f"| Baseline ms | {kr.baseline_ms:.3f} |",
            f"| Optimized ms | {kr.optimized_ms:.3f} |",
            f"| Iterations | {kr.iterations_used} |",
            f"| Re-injected | {'Yes' if kr.reinjected else 'No'} |",
        ]
        if kr.error:
            lines.append(f"| Error | {kr.error[:100]} |")
        lines.append("")

    lines += [
        f"## Final E2E Performance",
        f"",
        f"| Metric | Baseline | Optimized | Change |",
        f"|--------|----------|-----------|--------|",
        f"| Output throughput (tok/s) | {baseline_tps:.2f} | {final_tps:.2f} | {(tps_ratio-1)*100:+.2f}% |",
        f"| Throughput ratio | 1.00x | {tps_ratio:.4f}x | |",
        f"",
        f"## Reward Scores",
        f"",
        f"| Component | Value |",
        f"|-----------|-------|",
        f"| Per-kernel scores | {reward.get('per_kernel_scores', [])} |",
        f"| Avg kernel score | {reward.get('avg_kernel_score', 0):.4f} |",
        f"| Normalized kernel score | {reward.get('normalized_kernel_score', 0):.4f} |",
        f"| Model reward | {reward.get('model_reward', 0):.4f} |",
        f"| **Total reward** | **{reward.get('total_reward', 0):.4f}** |",
        f"| Quality | {trajectory.trajectory_quality} |",
        f"",
        f"## Scoring Formulas",
        f"",
        f"**Kernel level:**",
        f"```",
        f"score = compiled x 20 + correct x 100 + (baseline_ms / optimized_ms) x 100",
        f"```",
        f"",
        f"**Model level:**",
        f"```",
        f"score = 0.5 x normalized_kernel_score + 0.5 x (optimized_tps / baseline_tps - 1)",
        f"```",
        f"",
        f"## Run Duration",
        f"",
    ]

    step_labels = {
        "benchmark": "Initial E2E benchmark",
        "identify": "Identify bottleneck kernels",
        "optimize": "Kernel optimization loop",
        "grade": "Re-grade solutions",
        "integrate": "Re-inject kernels",
        "benchmark_final": "Final E2E benchmark",
        "score": "Compute scores",
        "report": "Generate reports",
    }

    if step_timings:
        lines += [
            f"| Step | Duration |",
            f"|------|----------|",
        ]
        total_s = 0.0
        for step, label in step_labels.items():
            t = step_timings.get(step)
            if t is not None:
                total_s += t
                if t >= 60:
                    lines.append(f"| {label} | {t:.1f}s ({t/60:.1f} min) |")
                else:
                    lines.append(f"| {label} | {t:.1f}s |")
        lines.append(f"| **Total** | **{total_s:.1f}s ({total_s/60:.1f} min)** |")
    else:
        total_s = trajectory.total_duration_s
        lines.append(f"Total: {total_s:.1f}s ({total_s/60:.1f} min)")

    lines.append("")

    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _generate_replication_guide(
    config: WorkloadConfig,
    results_dir: Path,
) -> Path:
    """Generate a step-by-step CLI replication guide."""
    benchmark_path = config.skip_benchmark or "<path-to-benchmark_report.json>"
    magpie_cfg = Path(config.benchmark_config).absolute()

    lines = [
        f"# Replication Guide: Workload Optimization Trajectory",
        f"",
        f"Step-by-step instructions to reproduce this workload optimization run.",
        f"",
        f"## Prerequisites",
        f"",
        f"1. AMD GPU with ROCm installed (target: {config.gpu_arch})",
        f"2. Python 3.10+ with claude-code-sdk and anthropic packages:",
        f"   ```bash",
        f"   pip install claude-code-sdk anthropic pyyaml",
        f"   ```",
        f"3. Magpie installed at `{MAGPIE_ROOT}`:",
        f"   ```bash",
        f"   cd {MAGPIE_ROOT} && pip install -e .",
        f"   ```",
        f"4. Anthropic API key:",
        f"   ```bash",
        f"   export ANTHROPIC_API_KEY=<your-key>",
        f"   ```",
        f"",
        f"## Step 1: Run Initial E2E Benchmark (or skip if available)",
        f"",
        f"Run a fresh benchmark:",
        f"```bash",
        f"cd {MAGPIE_ROOT}",
        f"source .venv/bin/activate",
        f"python -m Magpie benchmark --benchmark-config {magpie_cfg}",
        f"```",
        f"",
        f"Or reuse an existing result:",
        f"```bash",
        f"# Available results:",
        f"ls {MAGPIE_ROOT}/results/benchmark_vllm_*/benchmark_report.json",
        f"```",
        f"",
        f"## Step 2: Run Full Optimization Trajectory",
        f"",
        f"### Option A: Skip initial benchmark (reuse existing)",
        f"```bash",
        f"cd {REPO_ROOT}",
        f"export ANTHROPIC_API_KEY=<your-key>",
        f"export MAGPIE_ROOT={MAGPIE_ROOT}",
        f"",
        f"/usr/bin/python3 workload_optimizer.py \\",
        f"  --benchmark-config {magpie_cfg} \\",
        f"  --skip-benchmark {benchmark_path} \\",
        f"  --kernel-types {','.join(config.kernel_types)} \\",
        f"  --top-k {config.top_k} \\",
        f"  --max-iterations {config.max_iterations} \\",
        f"  --max-turns {config.max_turns_per_iter} \\",
        f"  --agent-model {config.agent_model} \\",
        f"  --agent-version {config.agent_version} \\",
        f"  --output-dir {config.output_dir} \\",
        f"  --results-dir {results_dir} \\",
        f"  --leaderboard",
        f"```",
        f"",
        f"### Option B: Full run (includes benchmark)",
        f"```bash",
        f"cd {REPO_ROOT}",
        f"export ANTHROPIC_API_KEY=<your-key>",
        f"export MAGPIE_ROOT={MAGPIE_ROOT}",
        f"",
        f"/usr/bin/python3 workload_optimizer.py \\",
        f"  --benchmark-config {magpie_cfg} \\",
        f"  --kernel-types {','.join(config.kernel_types)} \\",
        f"  --top-k {config.top_k} \\",
        f"  --max-iterations {config.max_iterations} \\",
        f"  --leaderboard \\",
        f"  --results-dir {results_dir}",
        f"```",
        f"",
        f"## Step 3: Check Results",
        f"",
        f"```bash",
        f"# View the report",
        f"cat {results_dir}/report.md",
        f"",
        f"# View leaderboard",
        f"cat {results_dir}/leaderboard.json | python3 -m json.tool",
        f"",
        f"# View trajectory",
        f"cat {results_dir}/trajectory.json | python3 -m json.tool",
        f"",
        f"# View kernel optimization outputs",
        f"ls {config.output_dir}/workload__*/",
        f"```",
        f"",
        f"## Step 4: Re-run to Add Another Leaderboard Entry",
        f"",
        f"Each run appends a new entry to `leaderboard.json`. Just re-run Step 2.",
        f"The leaderboard accumulates runs and tracks the best scores.",
        f"",
        f"## Understanding the Scores",
        f"",
        f"- **Kernel score**: `compiled(20) + correct(100) + speedup_ratio(x100)`",
        f"  - Max theoretical per kernel: 320+ (with 2x speedup)",
        f"- **Model score**: `0.5 * norm_kernel + 0.5 * (optimized_tps/baseline_tps - 1)`",
        f"- **Arena score**: Combined total * 100 (for leaderboard ranking)",
        f"",
        f"## Troubleshooting",
        f"",
        f"- If agent fails: check `{config.output_dir}/workload__*/` for partial outputs",
        f"- If Magpie fails: ensure `MAGPIE_ROOT` is set and Magpie venv is active",
        f"- If scores are 0: check that config.yaml has correct baseline paths",
        f"- For API errors: verify ANTHROPIC_API_KEY is valid",
        f"",
    ]

    guide_path = results_dir / "replication_guide.md"
    guide_path.write_text("\n".join(lines))
    return guide_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_workload_optimization(config: WorkloadConfig) -> WorkloadTrajectoryRecord:
    t_start = time.monotonic()
    results_dir = config.effective_results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "run.log"
    log_file = open(log_path, "w")

    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, s):
            for w in self.writers:
                w.write(s)
                w.flush()
        def flush(self):
            for w in self.writers:
                w.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, log_file)

    try:
        return _run_workload_optimization_inner(config, results_dir, t_start)
    finally:
        sys.stdout = original_stdout
        log_file.close()


def _run_workload_optimization_inner(
    config: WorkloadConfig,
    results_dir: Path,
    t_start: float,
) -> WorkloadTrajectoryRecord:

    with open(config.benchmark_config) as f:
        benchmark_cfg = yaml.safe_load(f)

    bench_section = benchmark_cfg.get("benchmark", {})
    model_id = bench_section.get("model", "unknown")
    framework = bench_section.get("framework", config.framework or "vllm")
    if not config.framework:
        config.framework = framework

    workload_id = f"workload__{framework}__{model_id.replace('/', '_')}"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.agent_version = f"v1.0-{run_ts}"

    trajectory = WorkloadTrajectoryRecord(
        workload_id=workload_id,
        agent_model=config.agent_model,
        agent_version=config.agent_version,
        benchmark_config_path=config.benchmark_config,
        benchmark_config=benchmark_cfg,
        framework=framework,
        model_id=model_id,
        gpu_arch=config.gpu_arch,
        skip_benchmark_used=config.skip_benchmark is not None,
    )

    print(f"\n{'='*65}")
    print(f"  WORKLOAD OPTIMIZATION TRAJECTORY")
    print(f"{'='*65}")
    print(f"  Workload:    {workload_id}")
    print(f"  Model:       {model_id}")
    print(f"  Framework:   {framework}")
    print(f"  GPU:         {config.gpu_arch}")
    print(f"  Agent:       {config.agent_model} ({config.agent_version})")
    print(f"  Config:      {config.benchmark_config}")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {config.output_dir}")
    print(f"{'='*65}")

    step_timings: dict[str, float] = {}

    # Step 1: Initial E2E Benchmark
    print(f"\n{'─'*65}")
    print(f"  Step 1: Initial E2E Benchmark")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    baseline_result = _run_initial_benchmark(config)
    trajectory.baseline_benchmark = baseline_result
    baseline_tps = extract_tps(baseline_result)
    trajectory.baseline_tps = baseline_tps
    step_timings["benchmark"] = time.monotonic() - t_step

    if baseline_result.get("error"):
        err = baseline_result["error"]
        print(f"  [error] Benchmark failed: {err[:200]}")
        trajectory.errors.append(f"Baseline benchmark failed: {err[:200]}")
    elif baseline_tps > 0:
        print(f"  Baseline TPS: {baseline_tps:.1f} tok/s")
    else:
        print(f"  [warn] Could not extract TPS from benchmark result")

    if config.skip_benchmark:
        print(f"  (Using cached benchmark)")
    print(f"  Duration: {step_timings['benchmark']:.1f}s")

    # Step 2-4: Bottleneck -> classify -> filter -> select
    print(f"\n{'─'*65}")
    print(f"  Step 2-4: Identify & Select Bottleneck Kernels")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    selected = _select_kernels(baseline_result, config)
    trajectory.bottleneck_kernels = [k.to_dict() for k in selected]
    trajectory.kernel_type_filter = list(config.kernel_types)
    trajectory.selected_kernels = [k.matched_kernel_spec or k.name for k in selected]
    step_timings["identify"] = time.monotonic() - t_step

    if not selected:
        print(f"\n  No kernels selected for optimization. Exiting.")
        trajectory.errors.append("No kernels selected after filtering")
        trajectory.total_duration_s = time.monotonic() - t_start
        return trajectory
    print(f"  Duration: {step_timings['identify']:.1f}s")

    # Step 5: Per-kernel optimization loop
    print(f"\n{'─'*65}")
    print(f"  Step 5: Kernel Optimization Loop ({len(selected)} kernels)")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    kernel_opt_results: list[KernelOptResult] = []

    for i, kernel in enumerate(selected, 1):
        print(f"\n  [{i}/{len(selected)}] Optimizing: "
              f"{kernel.matched_kernel_spec or kernel.name[:50]}")
        opt_result = _optimize_kernel(kernel, config, benchmark_cfg)
        kernel_opt_results.append(opt_result)
        trajectory.kernel_optimizations.append(opt_result.to_dict())
        print(f"    Result: compiled={opt_result.compiled} correct={opt_result.correct} "
              f"speedup={opt_result.speedup:.2f}x score={opt_result.score:.0f}")
    step_timings["optimize"] = time.monotonic() - t_step
    print(f"\n  Optimization duration: {step_timings['optimize']:.1f}s")

    # Step 6: Re-inject
    print(f"\n{'─'*65}")
    print(f"  Step 6: Re-inject Optimized Kernels")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    reinjected = []
    for opt_result in kernel_opt_results:
        bk = BottleneckKernel(name=opt_result.kernel_name, matched_kernel_spec=opt_result.kernel_spec)
        task_id = _make_kernel_task_id(bk, config)
        task_dir = config.output_dir / task_id
        if _reinject_kernel(opt_result, task_dir, config):
            reinjected.append(opt_result.kernel_spec)

    trajectory.reinjected_kernels = reinjected
    step_timings["integrate"] = time.monotonic() - t_step
    print(f"  Re-injected {len(reinjected)} kernel(s): {reinjected}")
    print(f"  Duration: {step_timings['integrate']:.1f}s")

    # Step 7: Final E2E benchmark
    print(f"\n{'─'*65}")
    print(f"  Step 7: Final E2E Benchmark")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    any_success = any(o.compiled and o.correct for o in kernel_opt_results)
    if not any_success and not config.dry_run:
        print(f"  Skipping final benchmark (no kernels were successfully optimized)")
        final_result = baseline_result
    else:
        final_result = _run_final_benchmark(config)
    trajectory.final_benchmark = final_result
    final_tps = extract_tps(final_result)
    trajectory.final_tps = final_tps
    step_timings["benchmark_final"] = time.monotonic() - t_step

    if final_tps > 0:
        print(f"  Final TPS: {final_tps:.1f} tok/s")
    if baseline_tps > 0 and final_tps > 0:
        ratio = final_tps / baseline_tps
        print(f"  Throughput improvement: {ratio:.4f}x ({(ratio-1)*100:.2f}%)")
    print(f"  Duration: {step_timings['benchmark_final']:.1f}s")

    # Step 8: Compute trajectory reward
    print(f"\n{'─'*65}")
    print(f"  Step 8: Compute Trajectory Reward")
    print(f"{'─'*65}")
    t_step = time.monotonic()

    kr_dicts = [
        {"compiled": o.compiled, "correct": o.correct,
         "baseline_ms": o.baseline_ms, "optimized_ms": o.optimized_ms}
        for o in kernel_opt_results
    ]

    reward = trajectory_reward(
        kernel_results=kr_dicts,
        baseline_tps=baseline_tps,
        optimized_tps=final_tps,
    )
    trajectory.apply_reward(reward)

    print(f"  Kernel scores: {reward['per_kernel_scores']}")
    print(f"  Avg kernel score: {reward['avg_kernel_score']:.2f}")
    print(f"  Normalized kernel: {reward['normalized_kernel_score']:.4f}")
    print(f"  Model reward: {reward['model_reward']:.4f}")
    print(f"  Total reward: {reward['total_reward']:.4f}")
    print(f"  Quality: {trajectory.trajectory_quality}")
    step_timings["score"] = time.monotonic() - t_step
    print(f"  Duration: {step_timings['score']:.1f}s")

    # Save trajectory
    trajectory.total_duration_s = time.monotonic() - t_start

    traj_dict = trajectory.to_dict() if hasattr(trajectory, "to_dict") else asdict(trajectory)
    traj_path = results_dir / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(traj_dict, f, indent=2, default=str)
    print(f"\n  Trajectory saved: {traj_path}")

    store = get_store(config.trajectory_store)
    tid = store.save(trajectory)
    print(f"  Trajectory ID: {tid}")

    # Save results summary
    summary = {
        "trajectory_id": tid,
        "workload_id": workload_id,
        "model_id": model_id,
        "framework": framework,
        "gpu_arch": config.gpu_arch,
        "agent_model": config.agent_model,
        "agent_version": config.agent_version,
        "baseline_tps": baseline_tps,
        "final_tps": final_tps,
        "throughput_ratio": final_tps / baseline_tps if baseline_tps > 0 else 0.0,
        "kernel_results": [o.to_dict() for o in kernel_opt_results],
        "reward": reward,
        "quality": trajectory.trajectory_quality,
        "duration_s": trajectory.total_duration_s,
        "step_timings": step_timings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = results_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Leaderboard
    if config.push_leaderboard:
        print(f"\n{'─'*65}")
        print(f"  Leaderboard")
        print(f"{'─'*65}")

        entry = LeaderboardEntry(
            agent_model=config.agent_model,
            agent_version=config.agent_version,
            task_id=workload_id,
            kernel_type="workload",
            model_id=model_id,
            gpu_arch=config.gpu_arch,
            kernel_score=reward["avg_kernel_score"],
            model_score=reward.get("model_reward", 0.0),
            arena_score=reward["total_reward"] * 100,
            baseline_tps=baseline_tps,
            optimized_tps=final_tps,
            throughput_ratio=final_tps / baseline_tps if baseline_tps > 0 else 0.0,
            speedup=sum(o.speedup for o in kernel_opt_results) / max(len(kernel_opt_results), 1),
            iterations_used=sum(o.iterations_used for o in kernel_opt_results),
            trajectory_id=tid,
        )
        _save_leaderboard(entry, results_dir)

    # Generate reports
    print(f"\n{'─'*65}")
    print(f"  Generating Reports")
    print(f"{'─'*65}")

    report_path = _generate_report(
        trajectory, kernel_opt_results, config,
        baseline_result, final_result, reward, results_dir,
        step_timings=step_timings,
    )
    print(f"  Report: {report_path}")

    guide_path = _generate_replication_guide(config, results_dir)
    print(f"  Replication guide: {guide_path}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  WORKLOAD OPTIMIZATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Workload:       {workload_id}")
    print(f"  Kernels:        {len(kernel_opt_results)} optimized, {len(reinjected)} re-injected")
    print(f"  Baseline TPS:   {baseline_tps:.1f}")
    print(f"  Final TPS:      {final_tps:.1f}")
    if baseline_tps > 0 and final_tps > 0:
        print(f"  Improvement:    {final_tps/baseline_tps:.4f}x")
    print(f"  Total reward:   {trajectory.total_reward:.4f} ({trajectory.trajectory_quality})")
    print(f"  Duration:       {trajectory.total_duration_s:.1f}s")
    print(f"  Trajectory ID:  {tid}")
    print(f"  Results dir:    {results_dir}")
    print(f"{'='*65}")

    return trajectory


# ---------------------------------------------------------------------------
# Subcommand handlers — each step can be run independently
# ---------------------------------------------------------------------------

def _load_benchmark_config(args) -> dict:
    """Load benchmark YAML config and set defaults in state."""
    with open(args.benchmark_config) as f:
        return yaml.safe_load(f)


def _init_config_from_args(args) -> WorkloadConfig:
    """Build WorkloadConfig from parsed CLI args."""
    results_dir = Path(args.results_dir)
    output_dir = Path(getattr(args, "output_dir", None) or str(results_dir / "output"))

    return WorkloadConfig(
        benchmark_config=getattr(args, "benchmark_config", ""),
        skip_benchmark=getattr(args, "skip_benchmark", None),
        kernel_types=[t.strip() for t in getattr(args, "kernel_types", "all").split(",")],
        kernels=[k.strip() for k in getattr(args, "kernels", "all").split(",")],
        top_k=getattr(args, "top_k", 10),
        max_iterations=getattr(args, "max_iterations", 5),
        max_turns_per_iter=getattr(args, "max_turns", 25),
        score_threshold=getattr(args, "score_threshold", 300.0),
        agent_model=getattr(args, "agent_model", "claude-sonnet-4-6"),
        agent_version=getattr(args, "agent_version", "v1.0"),
        agent_backend=getattr(args, "agent_backend", "claude"),
        framework=getattr(args, "framework", ""),
        gpu_arch=getattr(args, "gpu", "gfx950"),
        docker_image=getattr(args, "docker_image", ""),
        kernel_python=getattr(args, "kernel_python", ""),
        output_dir=output_dir,
        results_dir=results_dir,
        trajectory_store=getattr(args, "trajectory_store", "file"),
        push_leaderboard=getattr(args, "leaderboard", False),
        dry_run=getattr(args, "dry_run", False),
        benchmark_timeout=getattr(args, "benchmark_timeout", 5400),
    )


def cmd_benchmark(args):
    """Step 1: Run or load E2E benchmark."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)

    benchmark_cfg = _load_benchmark_config(args)
    bench_section = benchmark_cfg.get("benchmark", {})
    model_id = bench_section.get("model", "unknown")
    framework = bench_section.get("framework", config.framework or "vllm")
    if not config.framework:
        config.framework = framework

    state.update({
        "benchmark_config_path": str(args.benchmark_config),
        "benchmark_config": benchmark_cfg,
        "model_id": model_id,
        "framework": framework,
        "gpu_arch": config.gpu_arch,
        "output_dir": str(config.output_dir),
    })

    print(f"\n  Step 1: Initial E2E Benchmark")
    print(f"  {'─'*55}")

    baseline_result = _run_initial_benchmark(config)
    baseline_tps = extract_tps(baseline_result)

    elapsed = time.monotonic() - t0
    state.update({
        "baseline_result_path": config.skip_benchmark or "",
        "baseline_result": baseline_result,
        "baseline_tps": baseline_tps,
    })
    state.mark_step("benchmark")
    state.record_step_time("benchmark", elapsed)

    if baseline_result.get("error"):
        print(f"  [error] Benchmark failed: {baseline_result['error'][:200]}")
    else:
        print(f"  Baseline TPS: {baseline_tps:.1f} tok/s")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_identify(args):
    """Step 2-4: Identify, classify, filter bottleneck kernels."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    baseline_result = state.require("baseline_result", "benchmark")

    print(f"\n  Step 2-4: Identify & Select Bottleneck Kernels")
    print(f"  {'─'*55}")

    selected = _select_kernels(baseline_result, config)

    elapsed = time.monotonic() - t0
    state.update({
        "identified_kernels": [k.to_dict() for k in selected],
        "kernel_type_filter": list(config.kernel_types),
    })
    state.mark_step("identify")
    state.record_step_time("identify", elapsed)

    if not selected:
        print(f"\n  No kernels matched filters.")
    else:
        print(f"\n  {len(selected)} kernels identified. Use 'list-kernels' to view.")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_list_kernels(args):
    """Show identified kernels for interactive selection."""
    results_dir = Path(args.results_dir)
    state = PipelineState(results_dir)
    kernels_data = state.require("identified_kernels", "identify")

    kernels = [BottleneckKernel(**k) for k in kernels_data]

    print(f"\n  Available kernels for optimization ({len(kernels)}):")
    print(f"  {'─'*55}")
    print(format_bottleneck_table(kernels))
    print(f"\n  Kernel specs (use with --kernels flag):")
    for k in kernels:
        spec = k.matched_kernel_spec or k.name[:40]
        has_solution = ""
        output_dir = Path(state.get("output_dir", results_dir / "output"))
        framework = state.get("framework", "vllm")
        task_id = f"workload__{framework}__{spec}"
        task_dir = output_dir / task_id
        if (task_dir / "solution.py").exists() or (task_dir / "solution.hip").exists():
            has_solution = " [has solution]"
        print(f"    - {spec}{has_solution}")

    print(f"\n  Example: workload_optimizer.py optimize -r {results_dir} --kernels {kernels[0].matched_kernel_spec or 'name'}")


def cmd_optimize(args):
    """Step 5: Optimize selected kernels."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    kernels_data = state.require("identified_kernels", "identify")
    benchmark_cfg = state.require("benchmark_config", "benchmark")

    all_kernels = [BottleneckKernel(**k) for k in kernels_data]

    if config.kernels and "all" not in config.kernels:
        selected = [k for k in all_kernels
                     if (k.matched_kernel_spec or "") in config.kernels
                     or k.name in config.kernels]
        if not selected:
            print(f"  [error] No kernels matched: {config.kernels}")
            print(f"  Available: {[k.matched_kernel_spec for k in all_kernels]}")
            return
    else:
        selected = all_kernels

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.agent_version = f"v1.0-{run_ts}"

    print(f"\n  Step 5: Kernel Optimization Loop ({len(selected)} kernels)")
    print(f"  {'─'*55}")

    existing_results = state.get("optimization_results", {})

    for i, kernel in enumerate(selected, 1):
        spec = kernel.matched_kernel_spec or kernel.name[:50]
        print(f"\n  [{i}/{len(selected)}] Optimizing: {spec}")
        opt_result = _optimize_kernel(kernel, config, benchmark_cfg)
        existing_results[spec] = opt_result.to_dict()
        print(f"    Result: compiled={opt_result.compiled} correct={opt_result.correct} "
              f"speedup={opt_result.speedup:.2f}x score={opt_result.score:.0f}")

    elapsed = time.monotonic() - t0
    state.update({
        "optimization_results": existing_results,
        "agent_model": config.agent_model,
        "agent_version": config.agent_version,
    })
    state.mark_step("optimize")
    state.record_step_time("optimize", elapsed)
    print(f"\n  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_grade(args):
    """Re-grade existing solutions without re-running agent."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    kernels_data = state.require("identified_kernels", "identify")

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    all_kernels = [BottleneckKernel(**k) for k in kernels_data]
    if config.kernels and "all" not in config.kernels:
        selected = [k for k in all_kernels
                     if (k.matched_kernel_spec or "") in config.kernels]
    else:
        selected = all_kernels

    print(f"\n  Grading {len(selected)} kernel solutions")
    print(f"  {'─'*55}")

    existing_results = state.get("optimization_results", {})

    for kernel in selected:
        spec = kernel.matched_kernel_spec or "unknown"
        task_id = _make_kernel_task_id(kernel, config)
        task_dir = config.output_dir / task_id

        solution = find_solution(task_dir)
        if not solution:
            print(f"  {spec}: no solution found, skipping")
            continue

        baseline_sources = _find_baseline_sources(spec)
        _create_task_config(task_dir, kernel, config, baseline_sources)

        print(f"  Grading {spec}...")
        kr = grade_task(task_dir)

        baseline_ms, optimized_ms = _extract_timing_from_raw(kr.raw)
        print(f"    compiled={kr.compiled} correct={kr.correct} "
              f"speedup={kr.speedup:.2f}x score={kr.score:.0f}"
              f" baseline={baseline_ms:.3f}ms optimized={optimized_ms:.3f}ms")

        prev = existing_results.get(spec, {})
        existing_results[spec] = {
            "kernel_name": kernel.name, "kernel_spec": spec,
            "category": kernel.category,
            "compiled": kr.compiled, "correct": kr.correct,
            "speedup": kr.speedup, "score": kr.score,
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "iterations_used": prev.get("iterations_used", 0),
            "reinjected": prev.get("reinjected", False),
            "agent_turns": prev.get("agent_turns", 0),
            "error": kr.error,
        }

    elapsed = time.monotonic() - t0
    state.set("optimization_results", existing_results)
    state.mark_step("grade")
    state.record_step_time("grade", elapsed)
    print(f"\n  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_integrate(args):
    """Step 6: Re-inject optimized kernels."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    opt_results = state.require("optimization_results", "optimize")

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    if config.kernels and "all" not in config.kernels:
        specs_to_inject = config.kernels
    else:
        specs_to_inject = list(opt_results.keys())

    print(f"\n  Step 6: Re-inject Optimized Kernels")
    print(f"  {'─'*55}")

    reinjected = []
    for spec in specs_to_inject:
        if spec not in opt_results:
            print(f"  {spec}: not found in optimization results, skipping")
            continue
        result = opt_results[spec]
        opt = KernelOptResult(**{k: v for k, v in result.items() if k in KernelOptResult.__dataclass_fields__})

        bk = BottleneckKernel(name=opt.kernel_name, matched_kernel_spec=spec)
        task_id = _make_kernel_task_id(bk, config)
        task_dir = config.output_dir / task_id

        if _reinject_kernel(opt, task_dir, config):
            reinjected.append(spec)
            opt_results[spec]["reinjected"] = True

    elapsed = time.monotonic() - t0
    state.update({
        "optimization_results": opt_results,
        "reinjected_kernels": reinjected,
    })
    state.mark_step("integrate")
    state.record_step_time("integrate", elapsed)

    print(f"  Re-injected {len(reinjected)} kernel(s): {reinjected}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_benchmark_final(args):
    """Step 7: Run final E2E benchmark."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    state.require("optimization_results", "optimize")

    if not config.benchmark_config:
        config.benchmark_config = state.require("benchmark_config_path", "benchmark")
    if not config.framework:
        config.framework = state.get("framework", "vllm")

    print(f"\n  Step 7: Final E2E Benchmark")
    print(f"  {'─'*55}")

    final_result = _run_final_benchmark(config)
    final_tps = extract_tps(final_result)
    baseline_tps = state.get("baseline_tps", 0)

    elapsed = time.monotonic() - t0
    state.update({
        "final_result": final_result,
        "final_tps": final_tps,
    })
    state.mark_step("benchmark_final")
    state.record_step_time("benchmark_final", elapsed)

    print(f"  Final TPS: {final_tps:.1f} tok/s")
    if baseline_tps > 0 and final_tps > 0:
        ratio = final_tps / baseline_tps
        print(f"  Throughput improvement: {ratio:.4f}x ({(ratio-1)*100:.2f}%)")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_score(args):
    """Step 8: Compute trajectory reward and update leaderboard."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    results_dir = config.effective_results_dir
    opt_results = state.require("optimization_results", "optimize")
    baseline_tps = state.require("baseline_tps", "benchmark")
    final_tps = state.get("final_tps", baseline_tps)

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.agent_version = state.get("agent_version", f"v1.0-{run_ts}")

    print(f"\n  Step 8: Compute Trajectory Reward")
    print(f"  {'─'*55}")

    kr_dicts = []
    kernel_opt_results = []
    for spec, data in opt_results.items():
        kr_dicts.append({
            "compiled": data.get("compiled", False),
            "correct": data.get("correct", False),
            "baseline_ms": float(data.get("baseline_ms", 0)),
            "optimized_ms": float(data.get("optimized_ms", 0)),
        })
        kernel_opt_results.append(KernelOptResult(
            **{k: v for k, v in data.items() if k in KernelOptResult.__dataclass_fields__}
        ))

    reward = trajectory_reward(
        kernel_results=kr_dicts,
        baseline_tps=baseline_tps,
        optimized_tps=final_tps,
    )

    model_id = state.get("model_id", "unknown")
    workload_id = f"workload__{config.framework}__{model_id.replace('/', '_')}"
    tid = str(uuid.uuid4())

    print(f"  Kernel scores: {reward['per_kernel_scores']}")
    print(f"  Avg kernel score: {reward['avg_kernel_score']:.2f}")
    print(f"  Normalized kernel: {reward['normalized_kernel_score']:.4f}")
    print(f"  Model reward: {reward['model_reward']:.4f}")
    print(f"  Total reward: {reward['total_reward']:.4f}")

    state.update({
        "reward": reward,
        "trajectory_id": tid,
        "workload_id": workload_id,
    })

    summary = {
        "trajectory_id": tid,
        "workload_id": workload_id,
        "model_id": model_id,
        "framework": config.framework,
        "gpu_arch": config.gpu_arch,
        "agent_model": config.agent_model,
        "agent_version": config.agent_version,
        "baseline_tps": baseline_tps,
        "final_tps": final_tps,
        "throughput_ratio": final_tps / baseline_tps if baseline_tps > 0 else 0.0,
        "kernel_results": [o.to_dict() for o in kernel_opt_results],
        "reward": reward,
        "step_timings": state.step_timings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if config.push_leaderboard:
        entry = LeaderboardEntry(
            agent_model=config.agent_model,
            agent_version=config.agent_version,
            task_id=workload_id,
            kernel_type="workload",
            model_id=model_id,
            gpu_arch=config.gpu_arch,
            kernel_score=reward["avg_kernel_score"],
            model_score=reward.get("model_reward", 0.0),
            arena_score=reward["total_reward"] * 100,
            baseline_tps=baseline_tps,
            optimized_tps=final_tps,
            throughput_ratio=final_tps / baseline_tps if baseline_tps > 0 else 0.0,
            speedup=sum(o.speedup for o in kernel_opt_results) / max(len(kernel_opt_results), 1),
            iterations_used=sum(o.iterations_used for o in kernel_opt_results),
            trajectory_id=tid,
        )
        _save_leaderboard(entry, results_dir)

    elapsed = time.monotonic() - t0
    state.mark_step("score")
    state.record_step_time("score", elapsed)
    print(f"\n  Trajectory ID: {tid}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_report(args):
    """Generate markdown report and replication guide."""
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    results_dir = config.effective_results_dir

    opt_results_raw = state.require("optimization_results", "optimize")
    baseline_tps = state.require("baseline_tps", "benchmark")
    final_tps = state.get("final_tps", baseline_tps)
    reward = state.get("reward", {})
    model_id = state.get("model_id", "unknown")
    if not config.framework:
        config.framework = state.get("framework", "vllm")
    if not config.benchmark_config:
        config.benchmark_config = state.get("benchmark_config_path", "")

    workload_id = state.get("workload_id",
                             f"workload__{config.framework}__{model_id.replace('/', '_')}")
    tid = state.get("trajectory_id", "unknown")
    config.agent_version = state.get("agent_version", config.agent_version)

    kernel_opt_results = []
    for spec, data in opt_results_raw.items():
        kernel_opt_results.append(KernelOptResult(
            **{k: v for k, v in data.items() if k in KernelOptResult.__dataclass_fields__}
        ))

    trajectory = WorkloadTrajectoryRecord(
        trajectory_id=tid,
        workload_id=workload_id,
        agent_model=config.agent_model,
        agent_version=config.agent_version,
        benchmark_config_path=config.benchmark_config,
        framework=config.framework,
        model_id=model_id,
        gpu_arch=config.gpu_arch,
        baseline_tps=baseline_tps,
        final_tps=final_tps,
        bottleneck_kernels=state.get("identified_kernels", []),
        total_duration_s=0,
    )
    trajectory.apply_reward(reward)

    baseline_result = state.get("baseline_result", {})

    report_path = _generate_report(
        trajectory, kernel_opt_results, config,
        baseline_result, state.get("final_result", {}), reward, results_dir,
        step_timings=state.step_timings,
    )
    print(f"  Report: {report_path}")

    guide_path = _generate_replication_guide(config, results_dir)
    print(f"  Replication guide: {guide_path}")

    state.mark_step("report")


def cmd_run(args):
    """Full pipeline (all steps sequentially) — backward compatible."""
    config = _init_config_from_args(args)
    run_workload_optimization(config)


# ---------------------------------------------------------------------------
# CLI with subcommands
# ---------------------------------------------------------------------------

def _add_common_args(parser):
    """Add arguments shared across all subcommands."""
    parser.add_argument("-r", "--results-dir", required=True,
                        help="Directory for state, reports, leaderboard, trajectory")
    parser.add_argument("--output-dir",
                        help="Directory for kernel task outputs (default: results-dir/output)")
    parser.add_argument("--gpu", default="gfx950",
                        help="Target GPU architecture (default: gfx950)")
    parser.add_argument("--kernel-python", default="",
                        help="Python with torch+triton for kernel execution (auto-detected)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without GPU or API calls")


def _add_benchmark_args(parser):
    parser.add_argument("-b", "--benchmark-config", required=True,
                        help="Path to Magpie benchmark YAML config")
    parser.add_argument("--skip-benchmark",
                        help="Path to existing benchmark_report.json (skip benchmark)")
    parser.add_argument("--framework", default="",
                        help="Inference framework (default: auto-detect from config)")
    parser.add_argument("--benchmark-timeout", type=int, default=5400)


def _add_kernel_filter_args(parser):
    parser.add_argument("--kernel-types", default="all",
                        help="Comma-separated: triton,hip,ck,asm,all (default: all)")
    parser.add_argument("--kernels", default="all",
                        help="Comma-separated kernel spec names, or 'all'")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top bottleneck kernels to consider (default: 10)")


def _add_agent_args(parser):
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max optimization iterations per kernel (default: 5)")
    parser.add_argument("--max-turns", type=int, default=25,
                        help="Max agent turns per iteration (default: 25)")
    parser.add_argument("--score-threshold", type=float, default=300.0,
                        help="Stop early if score exceeds this (default: 300)")
    parser.add_argument("--agent-model", default="claude-sonnet-4-6")
    parser.add_argument("--agent-version", default="v1.0")
    parser.add_argument("--agent-backend", default="claude", choices=["claude", "codex"])


def main():
    parser = argparse.ArgumentParser(
        description="Modular workload optimization trajectory pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # -- benchmark --
    p = subparsers.add_parser("benchmark", help="Step 1: Run or load E2E benchmark")
    _add_common_args(p)
    _add_benchmark_args(p)

    # -- identify --
    p = subparsers.add_parser("identify",
                              help="Step 2-4: Identify & filter bottleneck kernels")
    _add_common_args(p)
    _add_kernel_filter_args(p)

    # -- list-kernels --
    p = subparsers.add_parser("list-kernels",
                              help="Show identified kernels for selection")
    p.add_argument("-r", "--results-dir", required=True)

    # -- optimize --
    p = subparsers.add_parser("optimize",
                              help="Step 5: Optimize selected kernels (agent + grading)")
    _add_common_args(p)
    _add_kernel_filter_args(p)
    _add_agent_args(p)

    # -- grade --
    p = subparsers.add_parser("grade",
                              help="Re-grade existing solutions (no agent)")
    _add_common_args(p)
    _add_kernel_filter_args(p)

    # -- integrate --
    p = subparsers.add_parser("integrate",
                              help="Step 6: Re-inject optimized kernels")
    _add_common_args(p)
    p.add_argument("--kernels", default="all",
                    help="Comma-separated kernel specs to integrate, or 'all'")

    # -- benchmark-final --
    p = subparsers.add_parser("benchmark-final",
                              help="Step 7: Run final E2E benchmark")
    _add_common_args(p)
    _add_benchmark_args(p)

    # -- score --
    p = subparsers.add_parser("score",
                              help="Step 8: Compute reward and update leaderboard")
    _add_common_args(p)
    p.add_argument("--leaderboard", action="store_true",
                    help="Push result to leaderboard")
    p.add_argument("--agent-model", default="claude-sonnet-4-6")
    p.add_argument("--agent-version", default="v1.0")
    p.add_argument("--trajectory-store", default="file")

    # -- report --
    p = subparsers.add_parser("report",
                              help="Generate markdown report and replication guide")
    _add_common_args(p)
    p.add_argument("-b", "--benchmark-config", default="")
    p.add_argument("--agent-model", default="claude-sonnet-4-6")
    p.add_argument("--agent-version", default="v1.0")

    # -- run --
    p = subparsers.add_parser("run",
                              help="Full pipeline (all steps sequentially)")
    _add_common_args(p)
    _add_benchmark_args(p)
    _add_kernel_filter_args(p)
    _add_agent_args(p)
    p.add_argument("--docker-image", default="")
    p.add_argument("--trajectory-store", default="file")
    p.add_argument("--leaderboard", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handlers = {
        "benchmark": cmd_benchmark,
        "identify": cmd_identify,
        "list-kernels": cmd_list_kernels,
        "optimize": cmd_optimize,
        "grade": cmd_grade,
        "integrate": cmd_integrate,
        "benchmark-final": cmd_benchmark_final,
        "score": cmd_score,
        "report": cmd_report,
        "run": cmd_run,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    # Ensure all print output is immediately flushed (important for piped/background runs)
    import functools
    print = functools.partial(print, flush=True)  # noqa: A001
    main()
