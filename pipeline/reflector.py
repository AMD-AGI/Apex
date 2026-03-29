# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
reflector.py — Reflection / iteration logic for the RL kernel-optimization pipeline.

After each grading round, analyses the KernelResult and generates a structured
reflection prompt that feeds back into the agent for the next attempt.

Includes profiler data, Magpie compare details, and dual speedup thresholds
(integration minimum vs stretch goal).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))
from score import KernelResult


COMPILE_REFLECTION = dedent("""\
    ## Reflection — Iteration {iteration}: Compilation Failure

    Your previous solution **failed to compile**.

    ### Error output
    ```
    {error}
    ```

    ### Your previous solution
    ```python
    {solution}
    ```

    ### What to fix
    - Check for syntax errors, missing imports, or undefined symbols.
    - Ensure you are using AMD ROCm / HIP compatible APIs (not CUDA-only).
    - If using Triton, verify `tl.*` APIs match the AMD Triton fork.
    {hints}

    ### Instructions
    Write a corrected `solution.py` that compiles successfully.
    Keep the same function signature. Do NOT modify baseline.py or test files.
""")


CORRECTNESS_REFLECTION = dedent("""\
    ## Reflection — Iteration {iteration}: Correctness Failure

    Your solution compiled but **produced incorrect results**.

    ### Error output
    ```
    {error}
    ```

    ### Your previous solution
    ```python
    {solution}
    ```

    ### What to fix
    - Compare your output against the baseline numerically.
    - Check for off-by-one errors in indexing, tiling, or reduction.
    - Verify data types (fp32 vs fp16 vs bf16) and accumulation precision.
    - Ensure edge cases (odd dimensions, non-power-of-2) are handled.
    {hints}

    ### Instructions
    Write a corrected `solution.py` that passes all correctness tests.
    Keep the same function signature.
""")


PERF_REGRESSION_REFLECTION = dedent("""\
    ## Reflection — Iteration {iteration}: Performance Regression

    Your solution is correct but **slower than baseline** ({speedup:.2f}x).

    ### Speedup thresholds
    - **Integration minimum:** {min_speedup:.2f}x (needed for re-injection into E2E)
    - **Stretch goal:** {target_speedup:.1f}x

    ### Your previous solution
    ```python
    {solution}
    ```

    ### Performance analysis
    - Baseline: {baseline_ms:.4f} ms
    - Your solution: {optimized_ms:.4f} ms
    - Speedup: {speedup:.2f}x (need >= {min_speedup:.2f}x)
    {compare_details}
    {profile_section}

    ### What to fix
    - Check for unnecessary memory copies or synchronisation barriers.
    - Verify coalesced memory access patterns (128-byte aligned for AMD).
    - Use MFMA instructions for matrix operations where applicable.
    - Reduce register pressure — check occupancy with `rocprof`.
    - Consider using shared memory (LDS) for data reuse.
    - Try a completely different strategy (library dispatch vs custom Triton).
    {hints}

    ### Instructions
    Write an optimized `solution.py` that is faster than the baseline.
    Correctness must still pass.
""")


IMPROVEMENT_REFLECTION = dedent("""\
    ## Reflection — Iteration {iteration}: Improvement Opportunity

    Your solution is correct and achieves **{speedup:.2f}x speedup** — good progress.
    Current score: {score:.0f} pts.

    ### Speedup thresholds
    - **Integration minimum:** {min_speedup:.2f}x (ACHIEVED — your kernel qualifies for re-injection)
    - **Stretch goal:** {target_speedup:.1f}x — push for more!

    ### Your previous solution
    ```python
    {solution}
    ```

    ### Performance analysis
    - Baseline: {baseline_ms:.4f} ms
    - Your solution: {optimized_ms:.4f} ms
    - Speedup: {speedup:.2f}x
    {compare_details}
    {profile_section}

    ### Optimization suggestions
    - Current speedup: {speedup:.2f}x → target: {target_speedup:.1f}x+
    - Consider fusing adjacent operations (check fusion-advisor MCP).
    - Tune tile sizes and block dimensions for the target GPU.
    - Explore FP8/FP4 quantisation if precision allows.
    - Use `source-finder` MCP to find alternative implementations in CK or rocBLAS.
    - Try a hybrid approach: library dispatch for large shapes, custom Triton for small.
    {hints}

    ### Available MCP tools for deeper analysis
    - `gpu-info get_arch_optimization_hints` for gfx950-specific tips
    - `kernel-rag analyze_kernel_for_optimization` with your solution code
    - `magpie analyze` to profile your kernel

    ### Instructions
    Write an improved `solution.py` with better performance.
    Correctness must still pass. Target: {target_speedup:.1f}x+ speedup.
""")


BELOW_THRESHOLD_REFLECTION = dedent("""\
    ## Reflection — Iteration {iteration}: Below Integration Threshold

    Your solution is correct and slightly faster ({speedup:.2f}x) but **below the
    integration threshold** of {min_speedup:.2f}x. It won't be re-injected for E2E testing.

    ### Your previous solution
    ```python
    {solution}
    ```

    ### Performance analysis
    - Baseline: {baseline_ms:.4f} ms
    - Your solution: {optimized_ms:.4f} ms
    - Speedup: {speedup:.2f}x (need >= {min_speedup:.2f}x for integration)
    {compare_details}
    {profile_section}

    ### What to fix
    - You need at least {min_speedup:.2f}x speedup for the optimization to be used.
    - Consider a fundamentally different approach:
      * Library dispatch (hipBLASLt, rocBLAS) instead of custom kernel, or vice versa
      * Kernel fusion with adjacent operations
      * Different tile sizes / block dimensions
    {hints}

    ### Instructions
    Write a significantly improved `solution.py` that achieves >= {min_speedup:.2f}x speedup.
    Correctness must still pass.
""")


def _read_solution(task_dir: Path) -> str:
    """Read the solution file contents, or return a placeholder."""
    for name in ("solution.py", "solution.hip", "solution.cu"):
        path = task_dir / name
        if path.exists():
            return path.read_text()
    return "# (solution file not found)"


def _read_baseline_head(task_dir: Path, max_lines: int = 100) -> str:
    """Read the first N lines of the baseline for reference in reflections."""
    baseline = task_dir / "baseline.py"
    if not baseline.exists():
        return ""
    try:
        lines = baseline.read_text().splitlines()[:max_lines]
        code = "\n".join(lines)
        if len(lines) == max_lines:
            code += "\n# ... (truncated) ..."
        return code
    except OSError:
        return ""


def _parse_rocprof_metrics(profile_data: str) -> dict:
    """Extract structured metrics from rocprof-compute output.

    Returns a dict with keys: bandwidth_pct, compute_pct, occupancy,
    top_instructions, recommendation.
    """
    if not profile_data:
        return {}

    metrics: dict = {}

    bw_match = re.search(
        r"(?:HBM|Memory)\s*(?:BW|Bandwidth)[^:]*:\s*([\d.]+)\s*%", profile_data, re.IGNORECASE
    )
    if bw_match:
        metrics["bandwidth_pct"] = float(bw_match.group(1))

    compute_match = re.search(
        r"(?:Compute|MFMA|VALU)[^:]*(?:utilization|util)[^:]*:\s*([\d.]+)\s*%",
        profile_data, re.IGNORECASE,
    )
    if compute_match:
        metrics["compute_pct"] = float(compute_match.group(1))

    occ_match = re.search(
        r"(?:Occupancy|Waves?\s*/\s*CU)[^:]*:\s*([\d.]+)", profile_data, re.IGNORECASE
    )
    if occ_match:
        metrics["occupancy"] = float(occ_match.group(1))

    instr_types = []
    for label in ("VMEM", "SMEM", "VALU", "MFMA", "SOP", "LDS"):
        if re.search(rf"\b{label}\b", profile_data, re.IGNORECASE):
            instr_types.append(label)
    if instr_types:
        metrics["top_instructions"] = instr_types

    bw = metrics.get("bandwidth_pct", 0)
    comp = metrics.get("compute_pct", 0)
    if bw > 0 or comp > 0:
        if bw > comp:
            metrics["recommendation"] = (
                "This kernel is memory-bound. Focus on coalescing, "
                "vectorized loads, and reducing global memory traffic."
            )
        else:
            metrics["recommendation"] = (
                "This kernel is compute-bound. Focus on MFMA utilization, "
                "loop unrolling, and reducing instruction count."
            )

    return metrics


def _format_performance_scorecard(profile_data: str) -> str:
    """Format rocprof metrics into a structured Performance Scorecard."""
    metrics = _parse_rocprof_metrics(profile_data)
    if not metrics:
        return ""

    lines = ["\n### Performance Scorecard"]

    bw = metrics.get("bandwidth_pct")
    if bw is not None:
        tag = "BOTTLENECK" if bw > 60 else "room to improve" if bw > 30 else "underutilized"
        lines.append(f"- Memory bandwidth: {bw:.0f}% of peak (~6.5 TB/s) — {tag}")

    comp = metrics.get("compute_pct")
    if comp is not None:
        tag = "BOTTLENECK" if comp > 60 else "room to improve" if comp > 30 else "underutilized"
        lines.append(f"- Compute (MFMA): {comp:.0f}% of peak — {tag}")

    occ = metrics.get("occupancy")
    if occ is not None:
        lines.append(f"- Occupancy: {occ:.0f} waves/CU (max ~16)")

    instrs = metrics.get("top_instructions")
    if instrs:
        lines.append(f"- Key instruction categories: {', '.join(instrs)}")

    rec = metrics.get("recommendation")
    if rec:
        lines.append(f"- **Recommendation:** {rec}")

    return "\n".join(lines) + "\n"


def _get_hints(kernel_type: str) -> str:
    """Return kernel-type-specific optimisation hints."""
    hints_map = {
        "flash_attn_prefill": (
            "- Flash attention: check softmax numerical stability (online softmax).\n"
            "- Use CK tile templates for fused FMHA on AMD.\n"
            "- Profile with `rocprof --stats` to find the hot loop."
        ),
        "fused_moe": (
            "- MoE gate + topk + expert GEMM can be fused into one kernel.\n"
            "- Use aiter's fused_moe_bf16_asm.py as reference for AMD.\n"
            "- Watch expert load balance — uneven distribution kills perf."
        ),
        "gemm_bf16": (
            "- Use MFMA (v_mfma_f32_32x32x8_bf16) for BF16 matrix multiply.\n"
            "- Tile sizes: try 128x128 or 256x128 blocks.\n"
            "- Check rocBLAS and hipBLASLt for reference implementations.\n"
            "- For decode (small M): try torch.addmm to leverage hipBLASLt tuning."
        ),
        "gemm_w8a8": (
            "- INT8/FP8 GEMM: ensure proper quantisation scaling.\n"
            "- Use MFMA fp8 instructions (v_mfma_f32_32x32x16_fp8) on gfx950.\n"
            "- Check aiter/ops/gemm_op_a8w8.py for AMD-optimised path."
        ),
        "rms_norm": (
            "- RMSNorm is memory-bound: maximise bandwidth utilisation.\n"
            "- Use vectorised loads (float4) for 128-byte coalescing.\n"
            "- Consider fusing with subsequent operations."
        ),
        "all_reduce": (
            "- All-reduce: check RCCL configuration and custom_all_reduce.\n"
            "- For small messages, consider tree or ring reduction.\n"
            "- Profile inter-GPU bandwidth with rccl-tests."
        ),
        "paged_attn_decode": (
            "- Paged attention decode is bandwidth-bound on MI355X.\n"
            "- Do NOT change SEQ_PARTITION_SIZE (causes full Triton re-JIT).\n"
            "- Focus on memory access patterns, not compute.\n"
            "- Buffer pooling for exp_sums/max_logits has minimal impact."
        ),
    }
    return hints_map.get(kernel_type, "- Use source-finder MCP to explore alternative implementations.")


def _extract_compare_details(raw: dict) -> str:
    """Extract structured Magpie compare details from raw grading result."""
    if not raw:
        return ""

    lines = []
    kernel_results = raw.get("results", raw).get("kernel_results", [])
    if len(kernel_results) >= 2:
        baseline_kr = kernel_results[0]
        optimized_kr = kernel_results[-1]

        b_perf = baseline_kr.get("performance_result", {})
        o_perf = optimized_kr.get("performance_result", {})

        b_corr = baseline_kr.get("correctness_result", {})
        o_corr = optimized_kr.get("correctness_result", {})

        if isinstance(b_perf, dict) and isinstance(o_perf, dict):
            b_summary = b_perf.get("summary", {})
            o_summary = o_perf.get("summary", {})
            if b_summary or o_summary:
                lines.append("\n### Magpie Compare Details")
                if b_summary:
                    lines.append(f"  - Baseline perf summary: {b_summary}")
                if o_summary:
                    lines.append(f"  - Solution perf summary: {o_summary}")

        if isinstance(o_corr, dict) and o_corr.get("errors"):
            lines.append(f"\n### Correctness errors\n  {o_corr['errors']}")

        if isinstance(o_corr, dict):
            for key in ("max_abs_error", "mean_abs_error", "max_rel_error"):
                val = o_corr.get(key)
                if val is not None:
                    lines.append(f"  - {key}: {val}")

    max_err = raw.get("max_absolute_error") or raw.get("max_abs_error")
    mean_err = raw.get("mean_absolute_error") or raw.get("mean_abs_error")
    if max_err is not None or mean_err is not None:
        lines.append("\n### Numerical Diff")
        if max_err is not None:
            lines.append(f"  - max_abs_error: {max_err}")
        if mean_err is not None:
            lines.append(f"  - mean_abs_error: {mean_err}")

    return "\n".join(lines)


def reflect(
    kernel_result: KernelResult,
    task_dir: Path,
    iteration: int,
    kernel_type: str = "",
    target_speedup: float = 3.0,
    min_speedup: float = 1.05,
    profile_data: str = "",
    previous_speedup: float = 0.0,
) -> str:
    """Generate a reflection prompt based on the grading result.

    Args:
        kernel_result: Grading outcome from Magpie compare.
        task_dir: Directory containing solution and baseline.
        iteration: Current iteration number.
        kernel_type: Kernel spec name for type-specific hints.
        target_speedup: Stretch goal speedup.
        min_speedup: Minimum speedup for re-injection (integration threshold).
        profile_data: Optional rocprof profiling output for the solution.
        previous_speedup: Speedup from prior iteration (for delta tracking).
    """
    solution_code = _read_solution(task_dir)
    hints = _get_hints(kernel_type)
    error = (kernel_result.error or "")[:1000]
    compare_details = _extract_compare_details(kernel_result.raw or {})

    profile_section = ""
    if profile_data:
        scorecard = _format_performance_scorecard(profile_data)
        profile_section = (
            f"\n### Profiling of Your Solution (Iteration {iteration})\n"
            f"{profile_data}\n{scorecard}"
        )

    baseline_ref = _read_baseline_head(task_dir)
    baseline_section = ""
    if baseline_ref:
        baseline_section = (
            f"\n### Baseline for Reference\n```python\n{baseline_ref}\n```\n"
        )

    if not kernel_result.compiled:
        return COMPILE_REFLECTION.format(
            iteration=iteration,
            error=error,
            solution=solution_code,
            hints=hints,
        ) + baseline_section

    if not kernel_result.correct:
        return CORRECTNESS_REFLECTION.format(
            iteration=iteration,
            error=error,
            solution=solution_code,
            hints=hints,
        ) + baseline_section

    raw = kernel_result.raw or {}
    baseline_ms = float(raw.get("baseline_ms", 0) or 0)
    optimized_ms = float(raw.get("optimized_ms", 0) or 0)

    if kernel_result.speedup < 1.0:
        return PERF_REGRESSION_REFLECTION.format(
            iteration=iteration,
            solution=solution_code,
            speedup=kernel_result.speedup,
            baseline_ms=baseline_ms,
            optimized_ms=optimized_ms,
            min_speedup=min_speedup,
            target_speedup=target_speedup,
            compare_details=compare_details,
            profile_section=profile_section,
            hints=hints,
        ) + baseline_section

    if kernel_result.speedup < min_speedup:
        return BELOW_THRESHOLD_REFLECTION.format(
            iteration=iteration,
            solution=solution_code,
            speedup=kernel_result.speedup,
            baseline_ms=baseline_ms,
            optimized_ms=optimized_ms,
            min_speedup=min_speedup,
            compare_details=compare_details,
            profile_section=profile_section,
            hints=hints,
        ) + baseline_section

    return IMPROVEMENT_REFLECTION.format(
        iteration=iteration,
        solution=solution_code,
        speedup=kernel_result.speedup,
        score=kernel_result.score,
        target_speedup=target_speedup,
        min_speedup=min_speedup,
        baseline_ms=baseline_ms,
        optimized_ms=optimized_ms,
        compare_details=compare_details,
        profile_section=profile_section,
        hints=hints,
    ) + baseline_section


def should_continue(
    kernel_result: KernelResult,
    iteration: int,
    max_iterations: int,
    score_threshold: float = 300.0,
    previous_speedup: float = 0.0,
    stall_count: int = 0,
) -> bool:
    """Decide whether to run another optimisation iteration.

    Returns False early if the kernel has stalled (delta < 5% for 2+
    consecutive iterations) to save agent budget.
    """
    if iteration >= max_iterations:
        return False
    if kernel_result.score >= score_threshold:
        return False
    if stall_count >= 2:
        return False
    return True
