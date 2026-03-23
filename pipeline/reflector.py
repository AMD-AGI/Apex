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

    return "\n".join(lines)


def reflect(
    kernel_result: KernelResult,
    task_dir: Path,
    iteration: int,
    kernel_type: str = "",
    target_speedup: float = 3.0,
    min_speedup: float = 1.05,
    profile_data: str = "",
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
    """
    solution_code = _read_solution(task_dir)
    hints = _get_hints(kernel_type)
    error = kernel_result.error or ""
    compare_details = _extract_compare_details(kernel_result.raw or {})
    profile_section = ""
    if profile_data:
        profile_section = f"\n### Profiling of Your Solution (Iteration {iteration})\n{profile_data}"

    if not kernel_result.compiled:
        return COMPILE_REFLECTION.format(
            iteration=iteration,
            error=error,
            solution=solution_code,
            hints=hints,
        )

    if not kernel_result.correct:
        return CORRECTNESS_REFLECTION.format(
            iteration=iteration,
            error=error,
            solution=solution_code,
            hints=hints,
        )

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
        )

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
        )

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
    )


def should_continue(
    kernel_result: KernelResult,
    iteration: int,
    max_iterations: int,
    score_threshold: float = 300.0,
) -> bool:
    """Decide whether to run another optimisation iteration."""
    if iteration >= max_iterations:
        return False
    if kernel_result.score >= score_threshold:
        return False
    return True
