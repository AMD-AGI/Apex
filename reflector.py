"""
reflector.py — Reflection / iteration logic for the RL kernel-optimization pipeline.

After each grading round, analyses the KernelResult and generates a structured
reflection prompt that feeds back into the agent for the next attempt.

Integrates the existing triton-kernel-reflection-prompts skill for AMD/ROCm-specific
reflection templates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent / "graders"))
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

    ### Your previous solution
    ```python
    {solution}
    ```

    ### Performance analysis
    - Baseline: {baseline_ms:.4f} ms
    - Your solution: {optimized_ms:.4f} ms
    - Speedup: {speedup:.2f}x (need > 1.0x)

    ### What to fix
    - Check for unnecessary memory copies or synchronisation barriers.
    - Verify coalesced memory access patterns (128-byte aligned for AMD).
    - Use MFMA instructions for matrix operations where applicable.
    - Reduce register pressure — check occupancy with `rocprof`.
    - Consider using shared memory (LDS) for data reuse.
    {hints}

    ### Instructions
    Write an optimized `solution.py` that is faster than the baseline.
    Correctness must still pass.
""")


IMPROVEMENT_REFLECTION = dedent("""\
    ## Reflection — Iteration {iteration}: Improvement Opportunity

    Your solution is correct and achieves **{speedup:.2f}x speedup** — good progress.
    Current score: {score:.0f} pts. Let's push for more.

    ### Your previous solution
    ```python
    {solution}
    ```

    ### Optimization suggestions
    - Current speedup: {speedup:.2f}x → target: {target_speedup:.1f}x+
    - Consider fusing adjacent operations (check fusion-advisor MCP).
    - Tune tile sizes and block dimensions for the target GPU.
    - Explore FP8/FP4 quantisation if precision allows.
    - Use `source-finder` MCP to find alternative implementations in CK or rocBLAS.
    {hints}

    ### Available MCP tools for deeper analysis
    - `gpu-info get_arch_optimization_hints` for gfx950-specific tips
    - `kernel-rag analyze_kernel_for_optimization` with your solution code
    - `magpie analyze` to profile your kernel

    ### Instructions
    Write an improved `solution.py` with better performance.
    Correctness must still pass. Target: {target_speedup:.1f}x+ speedup.
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
            "- Check rocBLAS and hipBLASLt for reference implementations."
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
    }
    return hints_map.get(kernel_type, "- Use source-finder MCP to explore alternative implementations.")


def reflect(
    kernel_result: KernelResult,
    task_dir: Path,
    iteration: int,
    kernel_type: str = "",
    target_speedup: float = 3.0,
) -> str:
    """
    Generate a reflection prompt based on the grading result.

    Returns a structured prompt string for the agent's next iteration.
    """
    solution_code = _read_solution(task_dir)
    hints = _get_hints(kernel_type)
    error = kernel_result.error or ""

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
            hints=hints,
        )

    return IMPROVEMENT_REFLECTION.format(
        iteration=iteration,
        solution=solution_code,
        speedup=kernel_result.speedup,
        score=kernel_result.score,
        target_speedup=target_speedup,
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
