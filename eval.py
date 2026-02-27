#!/usr/bin/env python3
"""
eval.py — End-to-end mini evaluation of the RL kernel-optimization environment.

Runs the full pipeline on a single, self-contained task that executes on CPU
(no GPU required) to prove functional correctness of each component:

  1. Prompt constructor  → generates kernel & model optimization prompts
  2. Claude Code agent   → receives the prompt, writes a solution to output/
  3. Local grader        → grades compilation, correctness, and speedup
  4. Score report        → prints results
  5. Component check     → validates prompts, graders, MCPs, and skills

The mini task: optimize a naive Python RMSNorm into a faster NumPy version.
This is intentionally simple so the eval completes quickly and works without
AMD hardware. Real RL tasks use HIP/Triton kernels on MI355X.

Agent: Claude Code (claude-agent-sdk). Auth is handled by the Claude Code CLI
itself — no ANTHROPIC_API_KEY required.

Usage:
    # Uses the Claude API to write an optimized solution
    python3 eval.py

    # Skip the API call; write a trivial numpy solution and grade it
    python3 eval.py --dry-run

    # Use a different model or increase turn budget
    python3 eval.py --model claude-opus-4-6 --max-turns 12

    # Run model-level prompt validation instead of kernel-level
    python3 eval.py --eval-type model --dry-run
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

REPO_ROOT = Path(__file__).parent

# ── Add local modules to path ─────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from score import KernelResult, ModelResult, total_score, PTS_COMPILED, PTS_CORRECT

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TURNS     = 8
TASK_ID       = "eval-mini__rms_norm__cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 1. TASK SETUP — create the mini eval task in output/
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_KERNEL = '''\
"""
baseline.py — Naive Python RMSNorm (intentionally slow).
Do not modify this file.
"""
import math

def rms_norm_baseline(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """Row-wise RMSNorm: out[i] = x[i] / rms(x) * weight[i]"""
    n = len(x)
    # Compute RMS
    mean_sq = sum(xi * xi for xi in x) / n
    rms = math.sqrt(mean_sq + eps)
    return [x[i] / rms * weight[i] for i in range(n)]
'''

TEST_SCRIPT = '''\
"""
test_solution.py — Correctness test for RMSNorm solution.
Compares solution output against the baseline for multiple random inputs.
Run with: python test_solution.py
"""
import sys, math, importlib.util, os, random

# Load baseline
spec = importlib.util.spec_from_file_location("baseline", os.path.join(os.path.dirname(__file__), "baseline.py"))
baseline_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline_mod)

# Load solution
solution_path = os.path.join(os.path.dirname(__file__), "solution.py")
if not os.path.exists(solution_path):
    print("FAIL: solution.py not found"); sys.exit(1)

spec2 = importlib.util.spec_from_file_location("solution", solution_path)
solution_mod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(solution_mod)

if not hasattr(solution_mod, "rms_norm"):
    print("FAIL: solution.py must define rms_norm(x, weight, eps=1e-6)"); sys.exit(1)

random.seed(42)
CASES = 20
TOL   = 1e-4

for _ in range(CASES):
    n = random.choice([64, 128, 256, 512, 1024, 4096])
    x = [random.gauss(0, 1) for _ in range(n)]
    w = [random.uniform(0.5, 1.5) for _ in range(n)]
    ref = baseline_mod.rms_norm_baseline(x, w)
    got = solution_mod.rms_norm(x, w)
    if len(got) != n:
        print(f"FAIL: output length mismatch {len(got)} vs {n}"); sys.exit(1)
    for i in range(n):
        if abs(got[i] - ref[i]) > TOL * max(abs(ref[i]), 1e-8):
            print(f"FAIL: mismatch at index {i}: got {got[i]:.6f}, expected {ref[i]:.6f}")
            sys.exit(1)

print(f"PASS: all {CASES} test cases correct"); sys.exit(0)
'''

BENCH_SCRIPT = '''\
"""
bench.py — Timing benchmark for baseline vs solution RMSNorm.
Outputs JSON: {"baseline_ms": ..., "optimized_ms": ...}
Run with: python bench.py
"""
import sys, os, time, importlib.util, json, random

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

d = os.path.dirname(__file__)
baseline  = load("baseline",  os.path.join(d, "baseline.py"))
solution  = load("solution",  os.path.join(d, "solution.py"))

random.seed(0)
N = 4096
x = [random.gauss(0, 1) for _ in range(N)]
w = [1.0] * N
ITERS = 500

# Warm-up
for _ in range(10):
    baseline.rms_norm_baseline(x, w)
    solution.rms_norm(x, w)

t0 = time.perf_counter()
for _ in range(ITERS):
    baseline.rms_norm_baseline(x, w)
baseline_ms = (time.perf_counter() - t0) * 1000 / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    solution.rms_norm(x, w)
optimized_ms = (time.perf_counter() - t0) * 1000 / ITERS

result = {"baseline_ms": round(baseline_ms, 4), "optimized_ms": round(optimized_ms, 4)}
print(json.dumps(result))
'''

CONFIG_YAML = f"""\
# Magpie-compatible config for the mini eval task.
gpu:
  device: 0
  arch: cpu   # This mini eval runs on CPU
baseline:
  path: ./baseline.py
optimized:
  path: ./solution.py
correctness:
  command: "python test_solution.py"
performance:
  command: "python bench.py"
  iterations: 500
"""


def setup_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "baseline.py").write_text(BASELINE_KERNEL)
    (task_dir / "test_solution.py").write_text(TEST_SCRIPT)
    (task_dir / "bench.py").write_text(BENCH_SCRIPT)
    (task_dir / "config.yaml").write_text(CONFIG_YAML)
    print(f"  task files written to {task_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT — use the real kernel_prompt / model_prompt constructors
# ══════════════════════════════════════════════════════════════════════════════

TASK_PROMPT = textwrap.dedent(f"""\
    ## Task: Optimize RMSNorm (CPU mini eval)

    Baseline kernel:   output/{TASK_ID}/baseline.py
    Write solution to: output/{TASK_ID}/solution.py

    The function signature must be:
        def rms_norm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]

    Optimization ideas:
    - Use numpy for vectorized operations (replace all Python loops)
    - Use np.linalg.norm or manual vectorized RMS

    Steps:
    1. Read baseline.py to understand the function
    2. Write your optimized solution.py
    3. Run `python output/{TASK_ID}/test_solution.py` to verify correctness
    4. Done — do not modify baseline.py, test_solution.py, or bench.py
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3. AGENT — Claude Code via Python Agent SDK
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.
You have access to MCP tools for kernel analysis:
  - source-finder: search ROCm repos for kernel implementations
  - kernel-rag: query documentation on HIP, Triton, ROCm kernels
  - gpu-info: query AMD GPU specifications (MI300X, MI355X)
  - fusion-advisor: get advice on kernel fusion opportunities
  - magpie: GPU kernel evaluation framework (analyze, compare, benchmark)
"""


async def _run_agent_async(task_dir: Path, model: str, max_turns: int) -> tuple[list, bool]:
    """Drive Claude Code via the Agent SDK. Returns (trajectory, solution_written)."""
    from claude_agent_sdk import query, ClaudeAgentOptions

    options = ClaudeAgentOptions(
        cwd=str(REPO_ROOT),
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        system_prompt=SYSTEM_PROMPT,
    )

    trajectory = []
    async for message in query(prompt=TASK_PROMPT, options=options):
        trajectory.append(message)
        if hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "name"):
                    print(f"    tool: {block.name}({list(block.input.keys())})")
        if hasattr(message, "num_turns"):
            cost = getattr(message, "total_cost_usd", 0.0) or 0.0
            print(f"  result: turns={message.num_turns}, cost=${cost:.4f}")

    return trajectory, (task_dir / "solution.py").exists()


def run_agent(task_dir: Path, model: str, max_turns: int, dry_run: bool) -> bool:
    """Run the Claude Code agent. Returns True if solution.py was written."""
    if dry_run:
        print("  [dry-run] writing trivial numpy solution...")
        solution = textwrap.dedent("""\
            import numpy as np

            def rms_norm(x, weight, eps=1e-6):
                a = np.asarray(x, dtype=np.float64)
                w = np.asarray(weight, dtype=np.float64)
                rms = np.sqrt(np.mean(a * a) + eps)
                return (a / rms * w).tolist()
        """)
        (task_dir / "solution.py").write_text(solution)
        return True

    try:
        from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
    except ImportError:
        print("ERROR: claude-agent-sdk not installed. Run: pip install claude-agent-sdk")
        return False

    _cc = os.environ.pop("CLAUDECODE", None)
    try:
        _, solution_written = asyncio.run(_run_agent_async(task_dir, model, max_turns))
    finally:
        if _cc is not None:
            os.environ["CLAUDECODE"] = _cc

    return solution_written


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOCAL GRADER — runs compile/test/bench locally (no Magpie needed)
# ══════════════════════════════════════════════════════════════════════════════

def grade_locally(task_dir: Path) -> KernelResult:
    task_id = TASK_ID

    # ── 1. Compilation check: can we import solution.py? ─────────────────────
    solution = task_dir / "solution.py"
    if not solution.exists():
        return KernelResult(task_id=task_id, error="solution.py not found")

    compile_code = (
        f"import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('sol', '{solution}')\n"
        f"m = importlib.util.module_from_spec(spec)\n"
        f"spec.loader.exec_module(m)\n"
        f"assert hasattr(m, 'rms_norm'), 'rms_norm not defined'\n"
        f"print('OK')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", compile_code],
            capture_output=True, text=True, timeout=10,
        )
        compiled = r.returncode == 0
        if not compiled:
            return KernelResult(task_id=task_id, compiled=False,
                                error=r.stderr.strip()[:200])
    except Exception as e:
        return KernelResult(task_id=task_id, error=str(e))

    # ── 2. Correctness: run test_solution.py ──────────────────────────────────
    try:
        r = subprocess.run(
            [sys.executable, str(task_dir / "test_solution.py")],
            capture_output=True, text=True, timeout=30,
        )
        correct = r.returncode == 0
        test_out = (r.stdout + r.stderr).strip()
    except Exception as e:
        return KernelResult(task_id=task_id, compiled=True, error=str(e))

    if not correct:
        return KernelResult(task_id=task_id, compiled=True, correct=False,
                            error=test_out[:200])

    # ── 3. Performance: run bench.py ──────────────────────────────────────────
    try:
        r = subprocess.run(
            [sys.executable, str(task_dir / "bench.py")],
            capture_output=True, text=True, timeout=60,
        )
        bench_out = r.stdout.strip()
        bench_data = json.loads(bench_out)
        baseline_ms  = bench_data["baseline_ms"]
        optimized_ms = bench_data["optimized_ms"]
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
    except Exception as e:
        return KernelResult(task_id=task_id, compiled=True, correct=True,
                            speedup=1.0, error=f"bench failed: {e}")

    return KernelResult(
        task_id=task_id,
        compiled=True, correct=True, speedup=speedup,
        raw={"baseline_ms": baseline_ms, "optimized_ms": optimized_ms},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPONENT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_prompts(framework: str = "sglang") -> dict:
    """Validate prompt constructors and return stats."""
    from kernel_prompt import all_prompts as kp_all, KERNEL_SPECS, DEFAULT_TARGET
    from model_prompt import all_prompts as mp_all
    from models import MODELS
    from configs import CONFIGS

    kernel_prompts = list(kp_all(framework=framework, gpu_arch=DEFAULT_TARGET))
    model_prompts  = list(mp_all(framework=framework, gpu_arch=DEFAULT_TARGET))

    return {
        "models":         len(MODELS),
        "kernel_specs":   len(KERNEL_SPECS),
        "configs":        len(CONFIGS),
        "kernel_prompts": len(kernel_prompts),
        "model_prompts":  len(model_prompts),
        "framework":      framework,
        "gpu_target":     DEFAULT_TARGET,
    }


def validate_tools() -> dict:
    """Check MCP server directories and skills."""
    mcps_dir   = REPO_ROOT / "tools" / "mcps"
    skills_dir = REPO_ROOT / "tools" / "skills"

    expected_mcps = ["fusion_advisor", "gpu_info", "rag_tool", "source_finder", "magpie"]
    found_mcps = []
    missing_mcps = []
    for name in expected_mcps:
        mcp_path = mcps_dir / name
        if mcp_path.is_dir():
            has_setup = (mcp_path / "setup.sh").exists()
            found_mcps.append(f"{name} ({'+ setup.sh' if has_setup else 'no setup.sh'})")
        else:
            missing_mcps.append(name)

    found_skills = []
    if skills_dir.is_dir():
        for d in sorted(skills_dir.iterdir()):
            if d.is_dir() and (d / "SKILL.md").exists():
                found_skills.append(d.name)

    return {
        "mcps_found":    found_mcps,
        "mcps_missing":  missing_mcps,
        "skills":        found_skills,
        "skills_count":  len(found_skills),
    }


def validate_graders() -> dict:
    """Quick validation that grader imports and scoring work."""
    from score import total_score, KernelResult, ModelResult, extract_tps
    import kernel_grader
    import model_grader

    r = KernelResult(task_id="test", compiled=True, correct=True, speedup=2.0)
    expected = total_score(True, True, 2.0)
    scoring_ok = abs(r.score - expected) < 1e-6

    m = ModelResult(model_id="test", kernel_score=320, e2e_throughput_ratio=1.5)
    model_scoring_ok = m.score > 0

    tps_ok = extract_tps({"throughput": {"output_throughput": 2500}}) == 2500.0

    return {
        "kernel_scoring": scoring_ok,
        "model_scoring":  model_scoring_ok,
        "tps_extraction": tps_ok,
        "kernel_grader":  hasattr(kernel_grader, "grade_task"),
        "model_grader":   hasattr(model_grader, "grade_task_model"),
        "default_models": len(model_grader.DEFAULT_MODELS),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RL kernel-optimization environment evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 eval.py --dry-run              # Quick validation, no API call
              python3 eval.py                        # Full eval with Claude Code
              python3 eval.py --model claude-opus-4-6 --max-turns 12
              python3 eval.py --eval-type model      # Model-level prompt validation
        """),
    )
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS,
                        help=f"Agent turn budget (default: {MAX_TURNS})")
    parser.add_argument("--task-dir",  default=None,
                        help="Override output task directory")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Skip Claude Code call; write a trivial solution and grade it")
    parser.add_argument("--eval-type", choices=["kernel", "model"], default="kernel",
                        help="Type of eval to run (default: kernel)")
    parser.add_argument("--framework", choices=["sglang", "vllm", "both"], default="sglang",
                        help="Framework for prompt validation (default: sglang)")
    args = parser.parse_args()

    task_dir = Path(args.task_dir) if args.task_dir else REPO_ROOT / "output" / TASK_ID

    print("")
    print("=" * 65)
    print("  RL Kernel Optimization — End-to-End Eval")
    print("=" * 65)
    print(f"  Task:      {TASK_ID}")
    print(f"  Model:     {args.model}")
    print(f"  Agent:     Claude Code (claude-agent-sdk)")
    print(f"  Eval type: {args.eval_type}")
    print(f"  Framework: {args.framework}")
    print(f"  Output:    {task_dir}")
    print("")

    # ── Step 1: Setup task ────────────────────────────────────────────────────
    print("--- Step 1: Setting up task ---")
    setup_task(task_dir)

    # ── Step 2: Validate prompt constructors ──────────────────────────────────
    print("\n--- Step 2: Prompt constructors ---")
    try:
        pstats = validate_prompts(args.framework)
        print(f"  models.py:        {pstats['models']} models")
        print(f"  kernel_specs:     {pstats['kernel_specs']} kernel types")
        print(f"  configs.py:       {pstats['configs']} inference configs")
        print(f"  kernel_prompts:   {pstats['kernel_prompts']} tasks ({pstats['framework']}, {pstats['gpu_target']})")
        print(f"  model_prompts:    {pstats['model_prompts']} tasks ({pstats['framework']}, {pstats['gpu_target']})")
    except Exception as e:
        print(f"  ERROR: {e}")
        pstats = {}

    print(f"  [using hand-crafted prompt for CPU mini eval]")

    # ── Step 3: Run agent ─────────────────────────────────────────────────────
    print("\n--- Step 3: Running agent ---")
    solution_written = run_agent(task_dir, args.model, args.max_turns, args.dry_run)

    if not solution_written:
        print("  Agent did not write a solution.")
        result = KernelResult(task_id=TASK_ID, error="no solution written")
    else:
        print(f"  solution.py written ({(task_dir / 'solution.py').stat().st_size} bytes)")

        # ── Step 4: Grade ──────────────────────────────────────────────────────
        print("\n--- Step 4: Grading ---")
        result = grade_locally(task_dir)

    # ── Step 5: Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Task ID:    {result.task_id}")
    print(f"  Compiled:   {result.compiled}  (+{PTS_COMPILED if result.compiled else 0} pts)")
    print(f"  Correct:    {result.correct}   (+{PTS_CORRECT if result.correct else 0} pts)")
    if result.correct and result.raw:
        bms = result.raw.get("baseline_ms")
        oms = result.raw.get("optimized_ms")
        if isinstance(bms, (int, float)):
            print(f"  Baseline:   {bms:.4f} ms")
        if isinstance(oms, (int, float)):
            print(f"  Optimized:  {oms:.4f} ms")
    print(f"  Speedup:    {result.speedup:.2f}×  (+{result.speedup * 100:.1f} pts)")
    print(f"  TOTAL:      {result.score:.1f} pts")
    if result.error:
        print(f"  Error:      {result.error}")
    print("=" * 65)

    # ── Step 6: Component validation ──────────────────────────────────────────
    print("\n--- Component validation ---")

    # Prompts
    if pstats:
        ok = pstats.get("kernel_prompts", 0) > 0
        print(f"  {'✓' if ok else '✗'}  prompts/kernel_prompt.py  ({pstats.get('kernel_prompts', 0)} tasks)")
        ok = pstats.get("model_prompts", 0) > 0
        print(f"  {'✓' if ok else '✗'}  prompts/model_prompt.py   ({pstats.get('model_prompts', 0)} tasks)")
        ok = pstats.get("models", 0) >= 19
        print(f"  {'✓' if ok else '✗'}  prompts/models.py         ({pstats.get('models', 0)} models)")
        ok = pstats.get("configs", 0) >= 10
        print(f"  {'✓' if ok else '✗'}  prompts/configs.py        ({pstats.get('configs', 0)} configs)")

    # Graders
    try:
        gstats = validate_graders()
        print(f"  {'✓' if gstats['kernel_scoring'] else '✗'}  graders/score.py           (kernel scoring)")
        print(f"  {'✓' if gstats['model_scoring'] else '✗'}  graders/score.py           (model scoring)")
        print(f"  {'✓' if gstats['tps_extraction'] else '✗'}  graders/score.py           (TPS extraction)")
        print(f"  {'✓' if gstats['kernel_grader'] else '✗'}  graders/kernel_grader.py")
        print(f"  {'✓' if gstats['model_grader'] else '✗'}  graders/model_grader.py    ({gstats['default_models']} models)")
    except Exception as e:
        print(f"  ✗  graders: {e}")

    # Tools: MCPs
    try:
        tstats = validate_tools()
        for mcp in tstats["mcps_found"]:
            print(f"  ✓  tools/mcps/{mcp}")
        for mcp in tstats["mcps_missing"]:
            print(f"  ✗  tools/mcps/{mcp} (missing)")
        print(f"  {'✓' if tstats['skills_count'] >= 10 else '✗'}  tools/skills/              ({tstats['skills_count']} skills)")
    except Exception as e:
        print(f"  ✗  tools: {e}")

    # Output dir
    print(f"  {'✓' if (REPO_ROOT / 'output').exists() else '✗'}  output/ directory")

    print("")
    exit_code = 0 if result.correct else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
