"""
score.py — Shared scoring logic and Magpie helpers for the RL graders.

AgentKernelArena scoring formula (kernel-level):
  compiled    → +20 pts
  correct     → +100 pts
  speedup S   → +S × 100 pts  (S = baseline_time / optimized_time ≥ 1.0)

  Total max (uncapped): 220+ pts per task

Magpie integration:
  - Primary: Python API via `Magpie` package (pip-installed)
  - Fallback: CLI via `python -m Magpie`
  - MCP: available for interactive agent use (not used by graders)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ── scoring constants (AgentKernelArena) ─────────────────────────────────────

PTS_COMPILED  = 20
PTS_CORRECT   = 100


def speedup_score(speedup: float) -> float:
    """Points awarded for performance improvement. speedup = baseline/optimized."""
    return max(0.0, speedup) * 100.0


def total_score(compiled: bool, correct: bool, speedup: float) -> float:
    return (PTS_COMPILED if compiled else 0) + \
           (PTS_CORRECT  if correct  else 0) + \
           (speedup_score(speedup) if (compiled and correct) else 0)


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class KernelResult:
    task_id:   str
    compiled:  bool           = False
    correct:   bool           = False
    speedup:   float          = 0.0     # baseline_time / optimized_time
    score:     float          = field(init=False)
    raw:       dict           = field(default_factory=dict)
    error:     Optional[str]  = None

    def __post_init__(self):
        self.score = total_score(self.compiled, self.correct, self.speedup)

    def to_dict(self) -> dict:
        return {
            "task_id":  self.task_id,
            "compiled": self.compiled,
            "correct":  self.correct,
            "speedup":  round(self.speedup, 4),
            "score":    round(self.score,   2),
            "error":    self.error,
        }

    def to_trajectory_dict(self) -> dict:
        """Extended dict for trajectory logging (includes raw results)."""
        d = self.to_dict()
        d["raw"] = self.raw
        return d


@dataclass
class ModelResult:
    model_id:             str
    kernel_score:         float  = 0.0
    e2e_throughput_ratio: float  = 0.0   # optimized / baseline tokens-per-second
    score:                float  = field(init=False)
    raw:                  dict   = field(default_factory=dict)
    error:                Optional[str] = None

    def __post_init__(self):
        # Weight: 50% kernel score (normalised to 0-1), 50% e2e improvement
        k_norm = min(self.kernel_score / 320.0, 1.0)   # 320 = compile+correct+3× speedup
        e_norm = max(0.0, self.e2e_throughput_ratio - 1.0)  # improvement over baseline
        self.score = round((k_norm + e_norm) * 100.0, 2)

    def to_dict(self) -> dict:
        return {
            "model_id":             self.model_id,
            "kernel_score":         round(self.kernel_score,         2),
            "e2e_throughput_ratio": round(self.e2e_throughput_ratio, 4),
            "score":                self.score,
            "error":                self.error,
        }

    def to_trajectory_dict(self) -> dict:
        """Extended dict for trajectory logging (includes raw results)."""
        d = self.to_dict()
        d["raw"] = self.raw
        return d


# ── Config YAML parsing ─────────────────────────────────────────────────────

def parse_task_config(config_path: Path) -> dict:
    """
    Parse our task config.yaml (written by the agent) into a dict.

    Expected schema (from kernel_prompt.py template):
      gpu:
        device: 0
        arch: gfx950
      baseline:
        path: tools/rocm/aiter/aiter/fused_moe.py
      optimized:
        path: ./solution.py
      correctness:
        command: "pytest tests/ -k fused_moe -x"
      performance:
        command: "python bench_fused_moe.py --arch gfx950"
        warmup_iterations: 10
        iterations: 100
    """
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_benchmark_config(config_path: Path) -> dict:
    """
    Parse our benchmark.yaml (written by the agent) into a dict.

    Expected schema (from model_prompt.py template):
      framework: sglang
      model: meta-llama/Llama-3.1-8B-Instruct
      gpu:
        device: 0
        arch: gfx950
      baseline:
        framework_config: {}
      optimized:
        patch: ./solution.py
      workload:
        input_len: 512
        output_len: 128
        num_prompts: 200
        concurrency: 32
      precision: fp8
    """
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ── Magpie invocation ────────────────────────────────────────────────────────

MAGPIE_MODULE = "Magpie"


def _magpie_bin() -> list[str]:
    """Return command list to invoke Magpie."""
    magpie_root = os.environ.get("MAGPIE_ROOT", "")
    if magpie_root:
        venv_python = Path(magpie_root) / ".venv" / "bin" / "python3"
        if venv_python.exists():
            return [str(venv_python), "-m", MAGPIE_MODULE]
    if shutil.which("magpie"):
        return ["magpie"]
    local = Path(__file__).parent.parent / "tools" / "magpie" / "Magpie" / "main.py"
    if local.exists():
        return ["python3", str(local)]
    return ["python3", "-m", MAGPIE_MODULE]


def run_magpie_compare(
    baseline_path: str,
    optimized_path: str,
    testcase_cmd: str | None = None,
    kernel_type: str = "pytorch",
    working_dir: str | None = None,
    timeout: int = 300,
) -> dict:
    """
    Run Magpie compare on baseline vs optimized kernel.

    Returns parsed JSON result dict.  On error, returns {"error": "..."}.
    """
    cmd = _magpie_bin() + [
        "compare",
        str(baseline_path),
        str(optimized_path),
        "--type", kernel_type,
    ]
    if testcase_cmd:
        cmd += ["--testcase", testcase_cmd]

    with tempfile.TemporaryDirectory(prefix="magpie_") as tmpdir:
        cmd += ["--output-dir", tmpdir]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
            )

            result_files = list(Path(tmpdir).glob("compare_results_*.json"))
            if result_files:
                with open(result_files[0]) as f:
                    return json.load(f)

            if proc.returncode != 0:
                return {
                    "error": f"magpie compare failed (exit {proc.returncode})",
                    "stderr": proc.stderr[:500],
                }
            return {"error": "no result file produced", "stdout": proc.stdout[:500]}

        except subprocess.TimeoutExpired:
            return {"error": f"magpie compare timed out ({timeout}s)"}
        except FileNotFoundError as e:
            return {"error": f"magpie not found: {e}"}


def _detect_gpu_count() -> int:
    """Detect number of ROCm GPUs available, fallback to 1."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        return max(1, result.stdout.count("VRAM Total Memory"))
    except Exception:
        return 1


def _auto_tp(model: str, gpu_count: int) -> int:
    """Pick tensor parallelism based on model size and available GPUs."""
    model_lower = model.lower()
    if any(tag in model_lower for tag in ["671b", "deepseek-r1", "deepseek_r1"]):
        return min(8, gpu_count)
    if any(tag in model_lower for tag in ["70b", "72b", "65b", "kimi-k2", "kimi_k2"]):
        return min(8, gpu_count)
    if any(tag in model_lower for tag in ["27b", "34b"]):
        return min(4, gpu_count)
    if any(tag in model_lower for tag in ["13b", "14b"]):
        return min(2, gpu_count)
    return 1


def run_magpie_benchmark(
    framework: str,
    model: str,
    benchmark_config_path: str | None = None,
    precision: str = "fp8",
    tp: int = 0,
    concurrency: int = 32,
    input_len: int = 1024,
    output_len: int = 512,
    timeout: int = 1800,
) -> dict:
    """
    Run Magpie benchmark mode.

    tp=0 means auto-detect based on model size and GPU count.
    Output is streamed live to stderr for visibility.
    Returns parsed JSON result dict.  On error, returns {"error": "..."}.
    """
    if tp <= 0:
        gpu_count = _detect_gpu_count()
        tp = _auto_tp(model, gpu_count)

    cmd = _magpie_bin()

    if benchmark_config_path:
        cmd += ["benchmark", "--benchmark-config", str(benchmark_config_path)]
    else:
        cmd += [
            "benchmark", framework,
            "--model", model,
            "--precision", precision,
            "--tp", str(tp),
            "--concurrency", str(concurrency),
            "--input-len", str(input_len),
            "--output-len", str(output_len),
        ]

    with tempfile.TemporaryDirectory(prefix="magpie_bench_") as tmpdir:
        cmd += ["--output-dir", tmpdir]
        import sys
        print(f"  [magpie] Running: {' '.join(cmd)}", file=sys.stderr)
        print(f"  [magpie] TP={tp}, timeout={timeout}s", file=sys.stderr)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stdout_lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line)
                print(f"  [magpie] {line}", end="", file=sys.stderr)

            proc.wait(timeout=timeout)
            stdout_text = "".join(stdout_lines)

            result_files = list(Path(tmpdir).rglob("*.json"))
            if result_files:
                with open(result_files[0]) as f:
                    return json.load(f)

            if proc.returncode != 0:
                return {
                    "error": f"magpie benchmark failed (exit {proc.returncode})",
                    "stderr": stdout_text[-500:],
                }
            return {"error": "no result file produced", "stdout": stdout_text[-500:]}

        except subprocess.TimeoutExpired:
            proc.kill()
            return {"error": f"magpie benchmark timed out ({timeout}s)"}
        except FileNotFoundError as e:
            return {"error": f"magpie not found: {e}"}


# ── Result parsing ───────────────────────────────────────────────────────────

def parse_compare_result(raw: dict) -> tuple[bool, bool, float]:
    """
    Extract (compiled, correct, speedup) from Magpie compare JSON output.

    Magpie compare result schema:
      {
        "mode": "compare",
        "results": {
          "kernel_results": [
            {"kernel_id": "...", "compile": {...}, "correctness": {...}, "performance": {...}},
            ...
          ],
          "comparison_metrics": {...},
          "winner": "...",
          "rankings": [...],
        }
      }

    Also handles flat/legacy formats for robustness.
    """
    results = raw.get("results", raw)

    # Magpie native format: kernel_results list
    kernel_results = results.get("kernel_results", [])
    if len(kernel_results) >= 2:
        baseline = kernel_results[0]
        optimized = kernel_results[-1]

        b_compiled = _extract_compiled(baseline)
        o_compiled = _extract_compiled(optimized)
        compiled = b_compiled and o_compiled

        b_correct = _extract_correct(baseline)
        o_correct = _extract_correct(optimized)
        correct = b_correct and o_correct

        b_time = _extract_time_ms(baseline)
        o_time = _extract_time_ms(optimized)
        speedup = (b_time / o_time) if (b_time > 0 and o_time > 0) else 0.0

        return compiled, correct, speedup

    # Legacy/flat format fallback
    opt = raw.get("optimized", raw)
    compiled = bool(
        opt.get("compilation", {}).get("success") or
        opt.get("compiled") or
        raw.get("compiled")
    )
    correct = bool(
        opt.get("correctness", {}).get("passed") or
        opt.get("correct") or
        raw.get("correct")
    )

    perf = opt.get("performance", {})
    baseline_ms  = float(perf.get("baseline_ms",  0) or perf.get("baseline_time_ms",  0))
    optimized_ms = float(perf.get("optimized_ms", 0) or perf.get("optimized_time_ms", 0))
    speedup = (baseline_ms / optimized_ms) if optimized_ms > 0 else 0.0

    return compiled, correct, speedup


def _extract_compiled(kernel_result: dict) -> bool:
    """Extract compilation status from a Magpie kernel result entry."""
    # Magpie native format: compiling_state / compiling_result
    state = kernel_result.get("compiling_state", "")
    if state:
        # SKIPPED means no compilation needed (e.g. PyTorch) → treat as compiled
        return state in ("SUCCESS", "SKIPPED")

    comp = kernel_result.get("compile", kernel_result.get("compilation", {}))
    if isinstance(comp, dict):
        return bool(comp.get("success", comp.get("passed", False)))
    return bool(comp)


def _extract_correct(kernel_result: dict) -> bool:
    """Extract correctness status from a Magpie kernel result entry."""
    # Magpie native format: correctness_state / correctness_result
    state = kernel_result.get("correctness_state", "")
    if state:
        return state == "SUCCESS"

    cr = kernel_result.get("correctness_result")
    if isinstance(cr, dict):
        return bool(cr.get("success", cr.get("passed", False)))

    corr = kernel_result.get("correctness", {})
    if isinstance(corr, dict):
        return bool(corr.get("passed", corr.get("success", False)))
    return bool(corr)


def _extract_time_ms(kernel_result: dict) -> float:
    """Extract average execution time (ms) from a Magpie kernel result entry."""
    # Magpie native format: performance_result
    perf = kernel_result.get("performance_result") or kernel_result.get("performance", {})
    if isinstance(perf, dict):
        for key in ("avg_time_ms", "mean_ms", "avg_ms", "time_ms", "median_ms"):
            v = perf.get(key)
            if v is not None and float(v) > 0:
                return float(v)
        metrics = perf.get("metrics", {})
        for key in ("avg_time_ms", "mean_ms", "avg_ms"):
            v = metrics.get(key)
            if v is not None and float(v) > 0:
                return float(v)
    return 0.0


def extract_tps(raw: dict) -> float:
    """
    Extract tokens-per-second from a single Magpie benchmark result.

    Magpie's BenchmarkResult.to_dict() schema:
      {
        "success": true,
        "throughput": {
          "output_throughput": 2500.0,
          "total_token_throughput": 3500.0,
          "request_throughput": 50.0,
          ...
        },
        ...
      }

    Also handles flat/legacy formats for robustness.
    """
    tp = raw.get("throughput", {})
    if isinstance(tp, dict):
        for key in ("output_throughput", "total_token_throughput",
                     "tokens_per_sec", "output_tokens_per_sec"):
            v = tp.get(key)
            if v is not None and float(v) > 0:
                return float(v)

    for key in ("output_throughput", "tokens_per_sec", "tps",
                "output_tokens_per_sec", "total_token_throughput"):
        v = raw.get(key)
        if v is not None and float(v) > 0:
            return float(v)

    return 0.0


# ── Workload-level reward functions ──────────────────────────────────────────


def workload_kernel_reward(
    compiled: bool, correct: bool,
    baseline_ms: float, optimized_ms: float,
) -> float:
    """
    Kernel-level reward for the workload optimization trajectory.

    score = compiled × 20 + correct × 100 + (baseline_ms / optimized_ms) × 100
    """
    score = (20.0 if compiled else 0.0) + (100.0 if correct else 0.0)
    if compiled and correct and optimized_ms > 0:
        score += (baseline_ms / optimized_ms) * 100.0
    return round(score, 4)


def workload_model_reward(
    normalized_kernel_score: float,
    optimized_tps: float,
    baseline_tps: float,
) -> float:
    """
    Model-level reward for the workload optimization trajectory.

    score = 0.5 × normalized_kernel_score + 0.5 × (optimized_tps / baseline_tps − 1)
    """
    tps_improvement = (optimized_tps / baseline_tps - 1.0) if baseline_tps > 0 else 0.0
    return round(0.5 * normalized_kernel_score + 0.5 * max(0.0, tps_improvement), 4)


def trajectory_reward(
    kernel_results: list[dict],
    baseline_tps: float,
    optimized_tps: float,
    max_kernel_score: float = 320.0,
) -> dict:
    """
    Combine kernel-level and model-level rewards into a full trajectory score.

    Each entry in kernel_results should contain:
      compiled (bool), correct (bool), baseline_ms (float), optimized_ms (float)

    Returns a dict with kernel_reward, model_reward, total_reward, and
    per-kernel scores.
    """
    per_kernel: list[float] = []
    for kr in kernel_results:
        score = workload_kernel_reward(
            compiled=kr.get("compiled", False),
            correct=kr.get("correct", False),
            baseline_ms=float(kr.get("baseline_ms", 0)),
            optimized_ms=float(kr.get("optimized_ms", 0)),
        )
        per_kernel.append(score)

    avg_kernel = sum(per_kernel) / len(per_kernel) if per_kernel else 0.0
    normalized = min(avg_kernel / max_kernel_score, 1.0)

    model_score = workload_model_reward(normalized, optimized_tps, baseline_tps)

    return {
        "per_kernel_scores": per_kernel,
        "avg_kernel_score": round(avg_kernel, 4),
        "normalized_kernel_score": round(normalized, 4),
        "model_reward": model_score,
        "total_reward": round(normalized * 0.5 + model_score * 0.5, 4),
    }


def parse_benchmark_result(raw: dict) -> float:
    """
    Extract throughput ratio (optimized / baseline) from benchmark JSON.

    Handles two scenarios:
      1. Pre-computed comparison result with baseline_tps / optimized_tps
      2. Raw Magpie single-run result (returns TPS via extract_tps)

    For scenario 2, the caller (model_grader) should run two benchmarks
    and compute the ratio itself.
    """
    bench = raw.get("benchmark", raw.get("results", raw))

    baseline_tps  = float(
        bench.get("baseline_tps",  0) or
        bench.get("baseline_tokens_per_sec",  0) or
        bench.get("baseline", {}).get("tokens_per_sec", 0)
    )
    optimized_tps = float(
        bench.get("optimized_tps", 0) or
        bench.get("optimized_tokens_per_sec", 0) or
        bench.get("optimized", {}).get("tokens_per_sec", 0)
    )
    if baseline_tps > 0 and optimized_tps > 0:
        return optimized_tps / baseline_tps

    return 0.0
