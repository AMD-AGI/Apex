#!/usr/bin/env python3
"""
eval_fused_moe.py — Focused eval: optimize the fused MoE Triton kernel on MI355X.

This test:
  1. Downloads only the necessary vLLM fused_moe files (sparse checkout)
  2. Runs the Claude Code agent with all MCP tools to optimize the kernel
  3. Grades the result (compile, correctness, speedup) via docker on GPU

Usage:
    python3 tests/fused_moe/eval_fused_moe.py                   # full run
    python3 tests/fused_moe/eval_fused_moe.py --dry-run          # skip agent, use trivial solution
    python3 tests/fused_moe/eval_fused_moe.py --model opus       # use opus model
    python3 tests/fused_moe/eval_fused_moe.py --max-turns 15     # more agent turns

Requirements:
    - pip install claude-agent-sdk
    - docker with rocm/pytorch:latest image
    - GPU access (--device=/dev/kfd --device=/dev/dri)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = Path(__file__).resolve().parent
TASK_ID = "mixtral-8x7b__fused_moe"
OUTPUT_DIR = REPO_ROOT / "output" / TASK_ID

DOCKER_IMAGE = "rocm/pytorch:latest"

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TURNS = 12

# ── Score constants (from graders/score.py) ──────────────────────────────────
PTS_COMPILED = 20
PTS_CORRECT = 100


# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP — ensure vLLM fused_moe files are available
# ══════════════════════════════════════════════════════════════════════════════

def ensure_vllm_files() -> Path:
    """Sparse-checkout vLLM fused_moe directory if not already present."""
    vllm_dir = REPO_ROOT / "files" / "code" / "vllm"
    fused_moe_dir = vllm_dir / "vllm" / "model_executor" / "layers" / "fused_moe"

    if fused_moe_dir.exists() and len(list(fused_moe_dir.glob("*.py"))) > 5:
        print(f"  vLLM fused_moe files already present ({len(list(fused_moe_dir.glob('*.py')))} files)")
        return fused_moe_dir

    print("  Downloading vLLM fused_moe files (sparse checkout)...")
    code_dir = REPO_ROOT / "files" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    if not vllm_dir.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse",
             "https://github.com/vllm-project/vllm.git", str(vllm_dir)],
            check=True, capture_output=True, timeout=120,
        )

    subprocess.run(
        ["git", "sparse-checkout", "set",
         "vllm/model_executor/layers/fused_moe"],
        cwd=str(vllm_dir), check=True, capture_output=True, timeout=60,
    )
    subprocess.run(
        ["git", "checkout"],
        cwd=str(vllm_dir), check=True, capture_output=True, timeout=60,
    )

    count = len(list(fused_moe_dir.glob("*.py")))
    print(f"  Downloaded {count} files to {fused_moe_dir}")
    return fused_moe_dir


def setup_output_dir() -> Path:
    """Create output directory with baseline + test harness symlinks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy baseline and test harness into output dir so agent can find them
    import shutil
    for fname in ["baseline_fused_moe.py", "test_and_bench.py"]:
        src = TEST_DIR / fname
        dst = OUTPUT_DIR / fname
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"  Output dir: {OUTPUT_DIR}")
    return OUTPUT_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT — task description for the agent
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(vllm_fused_moe_dir: Path) -> str:
    return textwrap.dedent(f"""\
        You are an expert GPU kernel engineer specializing in AMD ROCm/Triton optimization.

        ## Target Hardware
        AMD Instinct MI355X (gfx950, CDNA4)
        - Wavefront size: 64
        - Matrix units: MFMA instructions
        - LDS: 64 KB per CU
        - HBM bandwidth: ~6.5 TB/s aggregate

        ## Task
        Optimize the **fused MoE (Mixture of Experts)** Triton kernel for
        **Mixtral-8x7B** (8 experts, top-2 routing).

        The baseline kernel is at:
          `{OUTPUT_DIR}/baseline_fused_moe.py`

        The vLLM reference implementation (for inspiration) is at:
          `{vllm_fused_moe_dir}/fused_moe.py`

        ## What to optimize
        The core Triton kernel `fused_moe_kernel` in baseline_fused_moe.py performs
        the expert GEMM: each token is routed to top-2 experts, and the kernel
        computes token × expert_weight for each assignment.

        Key dimensions (Mixtral-8x7B):
        - hidden_dim (K) = 4096
        - ffn_dim (N) = 14336
        - num_experts = 8, top_k = 2
        - batch sizes: 32 to 512 tokens

        ## Instructions

        1. **Read** the baseline kernel at `{OUTPUT_DIR}/baseline_fused_moe.py`
        2. **Read** vLLM's implementation at `{vllm_fused_moe_dir}/fused_moe.py`
           for optimization ideas
        3. **Write** your optimized solution to:
           `{OUTPUT_DIR}/solution.py`

           Your solution MUST export `fused_moe_forward(hidden_states, expert_weights, topk_ids, topk_weights)` with
           the same signature as the baseline.

        4. **Test** correctness by running (inside docker):
           ```
           docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
             -v {REPO_ROOT}:/workspace -w /workspace/output/{TASK_ID} \\
             {DOCKER_IMAGE} \\
             python3 test_and_bench.py --solution solution.py
           ```

        ## Optimization ideas
        - Tune BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K for MI355X MFMA units
          (try 128×128×64 or 128×256×32)
        - Use `triton.autotune` with multiple configs
        - Sort tokens by expert ID before dispatch to coalesce GEMMs
        - Improve the `moe_align_block_size` function (currently pure Python loops)
        - Use persistent kernels to reduce kernel launch overhead
        - Vectorize memory loads (ensure coalesced access patterns)
        - Use software pipelining (`tl.advance` patterns)
        - Consider GROUP_SIZE_M tuning for L2 cache reuse

        ## Files you should NOT modify
        - baseline_fused_moe.py
        - test_and_bench.py

        ## Grading
        - +20 pts if your solution.py imports without error
        - +100 pts if it passes correctness tests (< 2% relative error vs reference)
        - +speedup × 100 pts for performance improvement over baseline
    """)


# ══════════════════════════════════════════════════════════════════════════════
# 3. AGENT — Claude Code via Agent SDK
# ══════════════════════════════════════════════════════════════════════════════

async def _run_agent_async(prompt: str, model: str, max_turns: int) -> tuple[list, bool]:
    from claude_agent_sdk import query, ClaudeAgentOptions

    options = ClaudeAgentOptions(
        cwd=str(REPO_ROOT),
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        system_prompt=(
            "You are an expert GPU kernel engineer specializing in AMD ROCm "
            "and Triton optimization for MI355X (gfx950, CDNA4). "
            "You have access to MCP tools: fusion-advisor, asm-tools, gpu-info, "
            "rag-server, arxiv, source-finder, and magpie for profiling and analysis. "
            "Use these tools to analyze and optimize the kernel."
        ),
    )

    trajectory = []
    async for message in query(prompt=prompt, options=options):
        trajectory.append(message)
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "name"):
                    args = list(block.input.keys()) if hasattr(block, "input") else []
                    print(f"    tool: {block.name}({args})")
                elif hasattr(block, "text"):
                    # Print first line of text responses
                    first_line = block.text.strip().split("\n")[0][:100]
                    if first_line:
                        print(f"    text: {first_line}...")
        if hasattr(message, "num_turns"):
            cost = getattr(message, "total_cost_usd", 0.0) or 0.0
            print(f"  result: turns={message.num_turns}, cost=${cost:.4f}")

    return trajectory, (OUTPUT_DIR / "solution.py").exists()


def run_agent(prompt: str, model: str, max_turns: int, dry_run: bool) -> bool:
    if dry_run:
        print("  [dry-run] writing trivial autotuned solution...")
        solution = textwrap.dedent("""\
            # Trivial solution: just re-export the baseline with autotuning hint
            from baseline_fused_moe import fused_moe_forward, moe_align_block_size, moe_reference
        """)
        (OUTPUT_DIR / "solution.py").write_text(solution)
        return True

    try:
        from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
    except ImportError:
        print("ERROR: claude-agent-sdk not installed. Run: pip install claude-agent-sdk")
        return False

    _cc = os.environ.pop("CLAUDECODE", None)
    try:
        _, solution_written = asyncio.run(_run_agent_async(prompt, model, max_turns))
    finally:
        if _cc is not None:
            os.environ["CLAUDECODE"] = _cc

    return solution_written


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRADING — run test_and_bench.py in docker
# ══════════════════════════════════════════════════════════════════════════════

def grade_in_docker(solution_path: Path) -> dict:
    """Run correctness + benchmark in docker, return results dict."""
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
        "-v", f"{REPO_ROOT}:/workspace",
        "-w", f"/workspace/output/{TASK_ID}",
        DOCKER_IMAGE,
        "python3", "test_and_bench.py",
        "--solution", f"/workspace/output/{TASK_ID}/{solution_path.name}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = r.stdout.strip()
        # Find the JSON line
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"compiled": False, "correct": False, "error": f"no JSON output: {output[:200]}"}
    except subprocess.TimeoutExpired:
        return {"compiled": False, "correct": False, "error": "docker timeout (300s)"}
    except Exception as e:
        return {"compiled": False, "correct": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fused MoE kernel optimization eval")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip agent, write trivial solution")
    parser.add_argument("--grade-only", action="store_true",
                        help="Skip agent, just grade existing solution.py")
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("  Fused MoE Kernel Optimization — MI355X Eval")
    print("=" * 60)
    print(f"  Task:    {TASK_ID}")
    print(f"  Model:   {args.model}")
    print(f"  Docker:  {DOCKER_IMAGE}")
    print("")

    # Step 1: Ensure files
    print("--- Step 1: Ensuring vLLM fused_moe files ---")
    vllm_dir = ensure_vllm_files()

    # Step 2: Setup output dir
    print("\n--- Step 2: Setting up output directory ---")
    setup_output_dir()

    if not args.grade_only:
        # Step 3: Build prompt
        print("\n--- Step 3: Building prompt ---")
        prompt = build_prompt(vllm_dir)
        print(f"  Prompt length: {len(prompt)} chars")

        # Step 4: Run agent
        print("\n--- Step 4: Running agent ---")
        solution_written = run_agent(prompt, args.model, args.max_turns, args.dry_run)

        if not solution_written:
            print("  Agent did not write solution.py")
            print_results({"compiled": False, "correct": False, "error": "no solution written"})
            sys.exit(1)

    solution = OUTPUT_DIR / "solution.py"
    if not solution.exists():
        print(f"  ERROR: {solution} not found")
        sys.exit(1)
    print(f"  solution.py: {solution.stat().st_size} bytes")

    # Step 5: Grade
    print("\n--- Step 5: Grading (docker + GPU) ---")
    results = grade_in_docker(solution)
    print_results(results)

    sys.exit(0 if results.get("correct") else 1)


def print_results(results: dict):
    compiled = results.get("compiled", False)
    correct = results.get("correct", False)
    speedup = results.get("speedup", 0.0)
    baseline_ms = results.get("baseline_ms", 0.0)
    optimized_ms = results.get("optimized_ms", 0.0)

    score = 0
    if compiled:
        score += PTS_COMPILED
    if correct:
        score += PTS_CORRECT
    score += speedup * 100

    print("")
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Task:       {TASK_ID}")
    print(f"  Compiled:   {compiled}  (+{PTS_COMPILED if compiled else 0} pts)")
    print(f"  Correct:    {correct}   (+{PTS_CORRECT if correct else 0} pts)")
    if baseline_ms > 0:
        print(f"  Baseline:   {baseline_ms:.4f} ms")
        print(f"  Optimized:  {optimized_ms:.4f} ms")
    print(f"  Speedup:    {speedup:.2f}x  (+{speedup * 100:.1f} pts)")
    print(f"  TOTAL:      {score:.1f} pts")
    if results.get("error"):
        print(f"  Error:      {results['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
