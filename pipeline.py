#!/usr/bin/env python3
"""
pipeline.py — Full E2E orchestrator for the RL kernel-optimization pipeline.

Chains: baseline benchmark -> task selection -> agent loop (optimize -> grade ->
reflect -> iterate) -> final benchmark -> trajectory log -> leaderboard push.

Usage:
    # DeepSeek R1 — optimise fused_moe kernel
    python pipeline.py \\
      --model deepseek-ai/DeepSeek-R1 \\
      --kernel fused_moe \\
      --framework sglang \\
      --gpu gfx950 \\
      --max-iterations 3 \\
      --leaderboard

    # Kimi K2 — all applicable kernels
    python pipeline.py \\
      --model moonshotai/Kimi-K2 \\
      --kernel all \\
      --framework sglang

    # Batch mode
    python pipeline.py --batch --config batch_config.yaml --parallel 4

    # Export trajectories for RL training
    python pipeline.py --export-trajectories --quality good --output rl_data.jsonl

    # Dry-run (no GPU, no agent API — validates pipeline structure)
    python pipeline.py --dry-run --model meta-llama/Llama-3.1-8B-Instruct --kernel rms_norm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from score import KernelResult, ModelResult, run_magpie_benchmark, extract_tps
from kernel_grader import grade_task, find_solution
from model_grader import grade_task_model
from reflector import reflect, should_continue
from trajectory import TrajectoryRecord, get_store
from leaderboard import Leaderboard, LeaderboardEntry


@dataclass
class PipelineConfig:
    model_id: str = ""
    kernel_type: str = "all"
    framework: str = "sglang"
    gpu_arch: str = "gfx950"
    agent_model: str = "claude-sonnet-4-6"
    max_iterations: int = 3
    max_turns_per_iter: int = 25
    score_threshold: float = 300.0
    docker_image: str = ""
    output_base: Path = field(default_factory=lambda: REPO_ROOT / "output")
    trajectory_store: str = "file"
    agent_version: str = "v1.0"
    push_leaderboard: bool = False
    dry_run: bool = False
    tp: int = 0
    benchmark_timeout: int = 1800


@dataclass
class IterationResult:
    iteration: int = 0
    kernel_result: dict = field(default_factory=dict)
    model_result: dict | None = None
    agent_messages: list[dict] = field(default_factory=list)
    reflection: str = ""
    duration_s: float = 0.0


@dataclass
class PipelineResult:
    task_id: str = ""
    config: dict = field(default_factory=dict)
    iterations: list[dict] = field(default_factory=list)
    baseline_benchmark: dict = field(default_factory=dict)
    final_benchmark: dict = field(default_factory=dict)
    improvement: dict = field(default_factory=dict)
    total_score: float = 0.0
    agent_model: str = ""
    agent_version: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str | None = None


SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.
You have access to MCP tools for kernel analysis:
  - source-finder: search ROCm repos for kernel implementations
  - kernel-rag: query documentation on HIP, Triton, ROCm kernels
  - gpu-info: query AMD GPU specifications (MI300X, MI355X)
  - fusion-advisor: get advice on kernel fusion opportunities
  - magpie: GPU kernel evaluation framework (analyze, compare, benchmark)
"""


def _resolve_applicable_kernels(model_id: str, kernel_type: str) -> list[str]:
    """Determine which kernels apply to a given model."""
    try:
        from kernel_prompt import KERNEL_SPECS, applicable_kernels
        from models import MODELS

        model = next((m for m in MODELS if m.hf_id == model_id), None)
        if model is None:
            model = next((m for m in MODELS if model_id.split("/")[-1].lower() in m.hf_id.lower()), None)

        if kernel_type != "all" and kernel_type:
            matching = [ks for ks in KERNEL_SPECS if ks.kernel_type == kernel_type]
            return [ks.kernel_type for ks in matching] if matching else [kernel_type]

        if model is not None:
            return [ks.kernel_type for ks in applicable_kernels(model)]

        return [ks.kernel_type for ks in KERNEL_SPECS]
    except ImportError:
        if kernel_type and kernel_type != "all":
            return [kernel_type]
        return ["rms_norm", "gemm_bf16", "flash_attn_prefill"]


def _build_prompt(model_id: str, kernel_type: str, framework: str, gpu_arch: str) -> str:
    """Build a kernel-optimisation prompt for the agent."""
    try:
        from kernel_prompt import build_kernel_prompt
        from models import MODELS

        model = next((m for m in MODELS if m.hf_id == model_id), None)
        if model is None:
            model = next((m for m in MODELS if model_id.split("/")[-1].lower() in m.hf_id.lower()), None)

        if model is not None:
            from kernel_prompt import KERNEL_MAP
            kernel_spec = KERNEL_MAP.get(kernel_type)
            if kernel_spec:
                result = build_kernel_prompt(model, kernel_spec, framework, gpu_arch)
                return result["prompt"]
    except Exception:
        pass

    return textwrap.dedent(f"""\
        ## Task: Optimise {kernel_type} kernel for {model_id}

        Framework: {framework}
        Target GPU: {gpu_arch}

        1. Use source-finder MCP to locate the {kernel_type} implementation
        2. Analyse the kernel using kernel-rag and gpu-info MCPs
        3. Write an optimised version to output/<task_id>/solution.py
        4. Ensure correctness by testing against the baseline
    """)


def _make_task_id(model_id: str, kernel_type: str, framework: str) -> str:
    """Generate a task ID matching the format used in kernel_prompt.py."""
    try:
        from kernel_prompt import make_task_id as _prompt_make_id
        from models import MODELS
        model = next((m for m in MODELS if m.hf_id == model_id), None)
        if model is not None:
            from kernel_prompt import KERNEL_MAP
            ks = KERNEL_MAP.get(kernel_type)
            if ks:
                return _prompt_make_id(model, ks)
    except Exception:
        pass
    model_short = model_id.split("/")[-1].replace(".", "-").lower()
    return f"{model_short}__{kernel_type}"


def _run_baseline_benchmark(
    config: PipelineConfig,
) -> dict:
    """Run Magpie benchmark on the unmodified model to get baseline TPS."""
    if config.dry_run:
        return {
            "dry_run": True,
            "throughput": {"output_throughput": 1000.0},
            "top_bottlenecks": ["flash_attn_prefill", "gemm_bf16", "rms_norm"],
        }

    print(f"  Loading {config.model_id} on {config.framework} "
          f"(tp={config.tp}, this may take 5-30 min for large models)...")
    result = run_magpie_benchmark(
        framework=config.framework,
        model=config.model_id,
        precision="fp8",
        tp=config.tp,
        timeout=config.benchmark_timeout,
    )
    return result


async def _run_agent_iteration(
    task_dir: Path,
    prompt: str,
    config: PipelineConfig,
    iteration: int,
    previous_reflection: str = "",
) -> tuple[list[dict], bool]:
    """Run one agent iteration. Returns (messages, solution_written)."""
    full_prompt = prompt
    if previous_reflection:
        full_prompt = previous_reflection + "\n\n" + prompt

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
        from claude_code_sdk import query as cc_query, ClaudeCodeOptions
        from claude_code_sdk.types import AssistantMessage, ResultMessage

        options = ClaudeCodeOptions(
            cwd=str(REPO_ROOT),
            model=config.agent_model,
            max_turns=config.max_turns_per_iter,
            permission_mode="bypassPermissions",
            system_prompt=SYSTEM_PROMPT,
        )

        messages: list[dict] = []
        msg_count = 0
        try:
            async for message in cc_query(prompt=full_prompt, options=options):
                msg_count += 1
                msg_dict = {"type": type(message).__name__}
                if isinstance(message, AssistantMessage):
                    content_parts = []
                    tool_uses = []
                    for block in message.content:
                        if hasattr(block, "text"):
                            content_parts.append(block.text)
                        if hasattr(block, "name"):
                            tool_uses.append(block.name)
                    text = "\n".join(content_parts)
                    msg_dict["content"] = text[:2000]
                    msg_dict["role"] = "assistant"
                    preview = text[:120].replace("\n", " ").strip()
                    tools_str = f" [tools: {', '.join(tool_uses)}]" if tool_uses else ""
                    print(f"    [agent] msg {msg_count}: {preview}...{tools_str}", flush=True)
                elif isinstance(message, ResultMessage):
                    msg_dict["num_turns"] = message.num_turns
                    msg_dict["cost_usd"] = message.total_cost_usd or 0.0
                    msg_dict["result"] = (message.result or "")[:2000]
                    msg_dict["is_error"] = message.is_error
                    cost = message.total_cost_usd or 0.0
                    print(f"    [agent] Done: {message.num_turns} turns, "
                          f"${cost:.3f}, error={message.is_error}", flush=True)
                messages.append(msg_dict)
        except Exception as stream_err:
            error_msg = str(stream_err)[:300]
            print(f"    [agent] Stream error after {msg_count} msgs: {error_msg}", flush=True)
            messages.append({"type": "error", "error": error_msg})

        solution_written = find_solution(task_dir) is not None
        print(f"    [agent] Solution on disk: {solution_written}", flush=True)
        return messages, solution_written

    except ImportError:
        print("  [warn] claude-code-sdk not installed, using dry-run mode")
        config.dry_run = True
        return await _run_agent_iteration(task_dir, prompt, config, iteration, previous_reflection)


def run_single_task(
    config: PipelineConfig,
    kernel_type: str,
) -> PipelineResult:
    """Run the full pipeline for a single (model, kernel) task."""
    task_id = _make_task_id(config.model_id, kernel_type, config.framework)
    task_dir = config.output_base / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Pipeline: {task_id}")
    print(f"  Model: {config.model_id}")
    print(f"  Kernel: {kernel_type}")
    print(f"  Framework: {config.framework} | GPU: {config.gpu_arch}")
    print(f"{'='*65}")

    result = PipelineResult(
        task_id=task_id,
        config=asdict(config),
        agent_model=config.agent_model,
        agent_version=config.agent_version,
    )

    # Step 1: Baseline benchmark
    print("\n--- Step 1: Baseline benchmark ---")
    baseline = _run_baseline_benchmark(config)
    result.baseline_benchmark = baseline
    baseline_tps = extract_tps(baseline)
    if baseline_tps > 0:
        print(f"  Baseline TPS: {baseline_tps:.1f}")
    elif "error" in baseline:
        print(f"  [warn] Baseline benchmark failed: {baseline.get('error', 'unknown')[:200]}")
        print(f"  Continuing without baseline — kernel optimization still runs.")
    elif config.dry_run:
        baseline_tps = 1000.0
        print(f"  [dry-run] Baseline TPS: {baseline_tps:.1f}")

    # Step 2: Build prompt
    print("\n--- Step 2: Building prompt ---")
    prompt = _build_prompt(config.model_id, kernel_type, config.framework, config.gpu_arch)
    print(f"  Prompt length: {len(prompt)} chars")

    # Step 3: Optimisation loop
    best_result: KernelResult | None = None
    reflection_prompt = ""

    for iteration in range(1, config.max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{config.max_iterations} ---")
        t0 = time.monotonic()

        # 3a. Run agent
        print("  Running agent...")
        messages, solution_written = asyncio.run(
            _run_agent_iteration(task_dir, prompt, config, iteration, reflection_prompt)
        )
        if not solution_written:
            print("  Agent did not write a solution.")
            iter_result = IterationResult(
                iteration=iteration,
                agent_messages=messages,
                duration_s=time.monotonic() - t0,
            )
            result.iterations.append(asdict(iter_result))
            continue

        # 3b. Grade
        print("  Grading...")
        kr = grade_task(task_dir, docker_image=config.docker_image or None)
        print(f"    compiled={kr.compiled} correct={kr.correct} "
              f"speedup={kr.speedup:.2f}x score={kr.score:.0f}")

        if best_result is None or kr.score > best_result.score:
            best_result = kr

        # 3c. Reflect
        reflection_prompt = reflect(
            kr, task_dir, iteration,
            kernel_type=kernel_type,
            target_speedup=config.score_threshold / 100.0,
        )

        iter_result = IterationResult(
            iteration=iteration,
            kernel_result=kr.to_dict(),
            agent_messages=messages,
            reflection=reflection_prompt[:500],
            duration_s=time.monotonic() - t0,
        )
        result.iterations.append(asdict(iter_result))

        # 3d. Check stop condition
        if not should_continue(kr, iteration, config.max_iterations, config.score_threshold):
            print(f"  Stopping: score={kr.score:.0f} (threshold={config.score_threshold})")
            break

    # Step 4: Final benchmark
    print("\n--- Step 4: Final benchmark ---")
    if config.dry_run:
        final_tps = baseline_tps * 1.5 if baseline_tps > 0 else 0
        result.final_benchmark = {"dry_run": True, "throughput": {"output_throughput": final_tps}}
        print(f"  [dry-run] Final TPS: {final_tps:.1f}")
    else:
        print(f"  Running final benchmark (tp={config.tp or 'auto'}, "
              f"timeout={config.benchmark_timeout}s)...")
        final = run_magpie_benchmark(
            framework=config.framework, model=config.model_id, precision="fp8",
            tp=config.tp, timeout=config.benchmark_timeout,
        )
        result.final_benchmark = final
        final_tps = extract_tps(final)
        if final_tps > 0:
            print(f"  Final TPS: {final_tps:.1f}")

    # Compute improvement
    if baseline_tps > 0 and final_tps > 0:
        ratio = final_tps / baseline_tps
        result.improvement = {
            "baseline_tps": baseline_tps,
            "final_tps": final_tps,
            "throughput_ratio": round(ratio, 4),
            "improvement_pct": round((ratio - 1) * 100, 2),
        }
        print(f"  Throughput improvement: {ratio:.2f}x ({(ratio-1)*100:.1f}%)")

    if best_result:
        result.total_score = best_result.score

    # Step 5: Log trajectory
    print("\n--- Step 5: Logging trajectory ---")
    store = get_store(config.trajectory_store)
    record = TrajectoryRecord(
        task_id=task_id,
        agent_model=config.agent_model,
        agent_version=config.agent_version,
        prompt=prompt[:5000],
        baseline_tps=baseline_tps,
        gpu_arch=config.gpu_arch,
        model_id=config.model_id,
        kernel_type=kernel_type,
        framework=config.framework,
        iterations=[asdict(IterationResult(**d)) if isinstance(d, dict) else d
                     for d in result.iterations],
        final_score=result.total_score,
        final_speedup=best_result.speedup if best_result else 0.0,
        final_tps=final_tps,
        throughput_improvement=result.improvement.get("throughput_ratio", 0.0),
        baseline_benchmark=result.baseline_benchmark,
        final_benchmark=result.final_benchmark,
    )
    record.compute_reward()
    tid = store.save(record)
    print(f"  Trajectory saved: {tid} (quality={record.trajectory_quality})")

    # Step 6: Push to leaderboard
    if config.push_leaderboard:
        print("\n--- Step 6: Leaderboard push ---")
        lb = Leaderboard(backend=config.trajectory_store)
        entry = LeaderboardEntry(
            agent_model=config.agent_model,
            agent_version=config.agent_version,
            task_id=task_id,
            kernel_type=kernel_type,
            model_id=config.model_id,
            gpu_arch=config.gpu_arch,
            kernel_score=best_result.score if best_result else 0.0,
            arena_score=result.total_score,
            baseline_tps=baseline_tps,
            optimized_tps=final_tps,
            throughput_ratio=result.improvement.get("throughput_ratio", 0.0),
            speedup=best_result.speedup if best_result else 0.0,
            iterations_used=len(result.iterations),
            total_agent_turns=sum(
                len(it.get("agent_messages", [])) for it in result.iterations
            ),
            trajectory_id=tid,
        )
        lb.push(entry)
        print(f"  Pushed to leaderboard: arena_score={entry.arena_score:.1f}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  PIPELINE RESULT: {task_id}")
    print(f"  Score: {result.total_score:.1f} pts")
    if best_result:
        print(f"  Best: compiled={best_result.compiled} correct={best_result.correct} "
              f"speedup={best_result.speedup:.2f}x")
    print(f"  Iterations: {len(result.iterations)}")
    print(f"  Trajectory: {tid} ({record.trajectory_quality})")
    if result.improvement:
        print(f"  Throughput: {result.improvement.get('throughput_ratio', 0):.2f}x")
    print(f"{'='*65}")

    return result


def run_pipeline(config: PipelineConfig) -> list[PipelineResult]:
    """Run the pipeline for all applicable kernels."""
    kernels = _resolve_applicable_kernels(config.model_id, config.kernel_type)
    tp_label = f"tp={config.tp}" if config.tp > 0 else "tp=auto"
    print(f"\nPipeline: {config.model_id}")
    print(f"  Kernels to optimise: {kernels}")
    print(f"  Framework: {config.framework} | GPU: {config.gpu_arch} | {tp_label}")
    print(f"  Agent: {config.agent_model} ({config.agent_version})")
    print(f"  Max iterations: {config.max_iterations}")

    results = []
    for kernel in kernels:
        result = run_single_task(config, kernel)
        results.append(result)

    # Overall summary
    total = sum(r.total_score for r in results)
    print(f"\n{'='*65}")
    print(f"  OVERALL PIPELINE SUMMARY")
    print(f"{'='*65}")
    print(f"  Model: {config.model_id}")
    print(f"  Tasks: {len(results)}")
    print(f"  Total score: {total:.1f} pts")
    for r in results:
        print(f"    {r.task_id}: {r.total_score:.1f} pts")
    print(f"{'='*65}")

    return results


def export_trajectories(args: argparse.Namespace) -> None:
    """Export trajectories for RL training."""
    store = get_store(args.trajectory_store)
    output = Path(args.output)
    count = store.export_for_rl(output, quality=args.quality, fmt=args.format)
    print(f"Exported {count} trajectories to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="E2E RL kernel-optimization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python pipeline.py --model deepseek-ai/DeepSeek-R1 --kernel fused_moe --leaderboard
              python pipeline.py --model moonshotai/Kimi-K2 --kernel all --framework sglang
              python pipeline.py --model meta-llama/Llama-3.1-70B-Instruct --kernel flash_attn_prefill --framework vllm
              python pipeline.py --dry-run --model meta-llama/Llama-3.1-8B-Instruct --kernel rms_norm
              python pipeline.py --export-trajectories --quality good --output rl_data.jsonl
        """),
    )

    # Pipeline config
    parser.add_argument("--model", default="", help="HuggingFace model ID")
    parser.add_argument("--kernel", default="all", help="Kernel type or 'all'")
    parser.add_argument("--framework", default="sglang", choices=["sglang", "vllm"])
    parser.add_argument("--gpu", default="gfx950", help="GPU architecture target")
    parser.add_argument("--agent-model", default="claude-sonnet-4-6")
    parser.add_argument("--agent-version", default="v1.0")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--score-threshold", type=float, default=300.0)
    parser.add_argument("--docker-image", default="")
    parser.add_argument("--tp", type=int, default=0,
                        help="Tensor parallelism (0=auto-detect from model size and GPU count)")
    parser.add_argument("--benchmark-timeout", type=int, default=1800,
                        help="Magpie benchmark timeout in seconds (default 1800=30min)")
    parser.add_argument("--trajectory-store", default="file", choices=["file", "couchdb", "s3"])
    parser.add_argument("--leaderboard", action="store_true", help="Push results to leaderboard")
    parser.add_argument("--dry-run", action="store_true", help="No GPU/API calls")

    # Batch mode
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--config", default=None, help="Batch config YAML")
    parser.add_argument("--parallel", type=int, default=1)

    # Export mode
    parser.add_argument("--export-trajectories", action="store_true")
    parser.add_argument("--quality", default=None, choices=["good", "mediocre", "bad"])
    parser.add_argument("--format", default="jsonl", choices=["jsonl"])
    parser.add_argument("--output", default="rl_training_data.jsonl")

    args = parser.parse_args()

    if args.export_trajectories:
        export_trajectories(args)
        return

    if not args.model:
        parser.error("--model is required (e.g. --model deepseek-ai/DeepSeek-R1)")

    config = PipelineConfig(
        model_id=args.model,
        kernel_type=args.kernel,
        framework=args.framework,
        gpu_arch=args.gpu,
        agent_model=args.agent_model,
        max_iterations=args.max_iterations,
        max_turns_per_iter=args.max_turns,
        score_threshold=args.score_threshold,
        docker_image=args.docker_image,
        trajectory_store=args.trajectory_store,
        agent_version=args.agent_version,
        push_leaderboard=args.leaderboard,
        dry_run=args.dry_run,
        tp=args.tp,
        benchmark_timeout=args.benchmark_timeout,
    )

    if args.batch and args.config:
        import yaml
        with open(args.config) as f:
            batch_cfg = yaml.safe_load(f)
        models = batch_cfg.get("models", [args.model])
        kernels = batch_cfg.get("kernels", [args.kernel])
        for model in models:
            for kernel in kernels:
                cfg = PipelineConfig(**{**asdict(config), "model_id": model, "kernel_type": kernel})
                run_pipeline(cfg)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
